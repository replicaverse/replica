"""Custom-unit responses must follow the event stream and source policy."""

from dataclasses import replace

import numpy as np
import pytest

from replica.analysis.data.preparation import prepare_observations
from replica.analysis.specification.config import AnalysisConfig
from replica.analysis.specification.evaluation import EvaluationSpec
from replica.core import AnnotationLayer, Datum, Span
from replica.dataset import fixation_feature
from replica.dataset.base import EyeTrackingDataset
from replica.dataset.contracts import DatasetInfo, Fixation
from replica.feature import UnitLength
from replica.feature.resolution import resolve_features
from replica.segmentation import Segmenter, WhitespaceSegmenter


class PairSegmenter(Segmenter):
    def __init__(self):
        super().__init__("pairs")

    def segment(self, text):
        words = WhitespaceSegmenter()(text)
        return [Span(words[i].start, words[i + 1].end) for i in range(0, len(words), 2)]


class EventDataset(EyeTrackingDataset):
    info = DatasetInfo(name="events")

    def load(self, path):
        pass


@pytest.fixture
def dataset():
    ds = EventDataset()
    stimulus = ds._make_stimulus(list("abcdef"), "s")
    ds._fixations = [
        Fixation(
            participant_id="p",
            stimulus_id="s",
            start_time=100 * i,
            end_time=100 * (i + 1),
            duration=100,
            x=np.nan,
            y=np.nan,
            unit_index=unit,
            char_offset=2 * unit,
            fixation_index=i + 1,
        )
        for i, unit in enumerate([0, 1, 0, 2, 3, 2, 4, 5])
    ]
    for name, values in [
        ("first_fixation", [100] * 6),
        ("first_run", [100] * 6),
        ("first_pass", [1] * 6),
        ("total_time", [200, 100, 200, 100, 100, 100]),
        ("go_past", [100, 200, 100, 200, 100, 100]),
    ]:
        ds.add_datum(Datum(stimulus, name, np.array(values, dtype=float), "p"))
    return ds


@pytest.mark.parametrize(
    "measure,expected",
    [
        ("first_fixation", [100, 100]),
        ("first_run", [300, 200]),
        ("first_run_in_first_pass", [300, 200]),
        ("total_time", [300, 200]),
        ("go_past", [300, 200]),
    ],
)
@pytest.mark.parametrize("has_offsets", [False, True])
def test_prepared_custom_units_recompute_responses(
    dataset, measure, expected, has_offsets
):
    if not has_offsets:
        dataset._fixations = [
            replace(event, char_offset=None) for event in dataset._fixations
        ]
    original = [d.values.copy() for d in dataset._data]
    spec = EvaluationSpec(
        predictors=[UnitLength()],
        targets=[dataset.feature(measure)],
        segmentation=PairSegmenter(),
    )
    rows = prepare_observations(dataset, spec, AnalysisConfig(transform_dv=lambda x: x))
    np.testing.assert_equal(rows.y, expected)
    np.testing.assert_equal(rows.unit_indices, [1, 2])
    for before, datum in zip(original, dataset._data, strict=True):
        np.testing.assert_equal(before, datum.values)


def test_requested_first_pass_restriction_survives_multiple_dvs(dataset):
    sequence = [(4, 50), (2, 100), (3, 100), (0, 200), (1, 200), (5, 70)]
    dataset._fixations = [
        replace(dataset._fixations[i], unit_index=u, char_offset=2 * u, duration=d)
        for i, (u, d) in enumerate(sequence)
    ]
    for datum in dataset._data:
        if datum.measure_name == "first_run":
            datum.values = np.array([200, 200, 100, 100, 50, 70.0])
        elif datum.measure_name == "first_pass":
            datum.values = np.array([0, 0, 0, 0, 1, 1.0])
    dataset.drop_skips = False
    quantities = ("first_run", "first_run_in_first_pass")
    values = resolve_features(
        dataset,
        [dataset.feature(name) for name in quantities],
        segmentation=PairSegmenter(),
    )
    for resolved, expected in zip(values, ([400, 200, 50], [0, 0, 50]), strict=True):
        np.testing.assert_equal(resolved.blocks[0].values, expected)


def test_native_disagreement_prevents_reaggregation(dataset):
    [datum] = dataset.data("first_fixation")
    wrong = replace(datum, values=datum.values + 1)
    with pytest.raises(ValueError, match="do not reproduce the native measure"):
        dataset.reaggregate(wrong, "words")


def test_absent_trial_cannot_become_zero_durations(dataset):
    [datum] = dataset.data("first_fixation")
    with pytest.raises(ValueError, match="absent trial"):
        dataset.reaggregate(datum, "words", fixations=[])


def test_evaluation_consumes_trial_streams_without_materializing_all_events(
    dataset, monkeypatch
):
    events = dataset._fixations

    def no_bulk_access(*args, **kwargs):
        raise AssertionError("Bulk fixation materialization is unavailable")

    monkeypatch.setattr(dataset, "fixations", no_bulk_access)
    monkeypatch.setattr(
        dataset, "fixation_trials", lambda: iter([(("p", "s"), events)])
    )
    first_fixation, first_run = resolve_features(
        dataset,
        [dataset.feature("first_fixation"), dataset.feature("first_run")],
        segmentation=PairSegmenter(),
    )
    np.testing.assert_equal(first_fixation.blocks[0].values, [100, 100, 100])
    np.testing.assert_equal(first_run.blocks[0].values, [300, 300, 200])


def test_streamed_partial_coverage_cannot_fall_back_to_summed_summaries(
    dataset, monkeypatch
):
    monkeypatch.setattr(
        dataset, "fixation_trials", lambda: iter([(("other", "s"), dataset._fixations)])
    )
    with pytest.raises(ValueError, match="absent trial"):
        resolve_features(
            dataset, [dataset.feature("total_time")], segmentation=PairSegmenter()
        )


def test_reaggregation_selects_the_summary_presentation_and_preserves_events(dataset):
    original = [replace(f, metadata={"trial_index": 2}) for f in dataset._fixations]
    repeated = [replace(f, duration=700, metadata={"trial_index": 3}) for f in original]
    dataset._fixations = repeated + original
    for datum in dataset._data:
        datum.metadata["source_trial_index"] = 2
    stimulus = dataset.stimuli()[0]
    stimulus.add_layer(
        AnnotationLayer("pairs", PairSegmenter()(stimulus.text), ["ab", "cd", "ef"])
    )
    cache = {}
    for measure, expected in [
        ("first_fixation", [100] * 3),
        ("first_run", [300, 300, 200]),
    ]:
        [datum] = dataset.data(measure)
        actual = dataset.reaggregate(
            datum, "pairs", fixations=dataset._fixations, _cache=cache
        )
        np.testing.assert_equal(actual.values, expected)
    assert dataset.fixations() == repeated + original


def test_missing_summary_presentation_cannot_use_a_repeat(dataset):
    for datum in dataset._data:
        datum.metadata["source_trial_index"] = 2
    [datum] = dataset.data("first_fixation")
    with pytest.raises(ValueError, match="released presentation"):
        dataset.reaggregate(datum, "words")


@pytest.mark.parametrize("drop_skips", [False, True])
@pytest.mark.parametrize("policy", ["metadata", "measure"])
def test_source_blank_for_unfixated_units_is_preserved_on_custom_units(
    dataset, drop_skips, policy
):
    dataset.drop_skips = drop_skips
    dataset._fixations = [f for f in dataset._fixations if f.unit_index < 4]
    datum = next(d for d in dataset._data if d.measure_name == "first_fixation")
    datum.values[4:] = np.nan
    if policy == "metadata":
        datum.metadata["unfixated_is_missing"] = True
    else:
        dataset._unfixated_nan_measures = frozenset({"first_fixation"})
    datum.stimulus.add_layer(
        AnnotationLayer(
            "pairs", PairSegmenter()(datum.stimulus.text), ["ab", "cd", "ef"]
        )
    )
    [visible] = dataset.data("first_fixation")
    np.testing.assert_equal(
        dataset.reaggregate(visible, "pairs").values, [100, 100, np.nan]
    )


def test_copco_source_exclusions_interrupt_runs_without_changing_raw_events(dataset):
    from replica.dataset.eye_tracking.copco import CopCo

    copco = CopCo()
    copco._data = dataset._data
    copco._stimuli = dataset._stimuli
    copco._fixations = [
        replace(f, metadata={"run_id": 1, "source_ia_label": "." if i == 1 else "a"})
        for i, f in enumerate(dataset._fixations)
    ]
    for datum in copco._data:
        datum.values[1] = np.nan
        datum.metadata["unfixated_is_missing"] = True
    datum = next(d for d in copco._data if d.measure_name == "first_run")
    datum.stimulus.add_layer(
        AnnotationLayer(
            "pairs", PairSegmenter()(datum.stimulus.text), ["ab", "cd", "ef"]
        )
    )
    [visible] = copco.data("first_run")
    np.testing.assert_equal(copco.reaggregate(visible, "pairs").values, [100, 300, 200])
    assert copco.fixations()[1].unit_index == 1
    assert copco.fixations()[1].metadata["run_id"] == 1


@pytest.mark.parametrize("drop_skips", [False, True])
def test_skip_policy_is_applied_after_region_reconstruction(dataset, drop_skips):
    dataset.drop_skips = drop_skips
    dataset._fixations = [f for f in dataset._fixations if f.unit_index != 3]
    next(d for d in dataset._data if d.measure_name == "first_fixation").values[3] = 0
    [datum] = dataset.data("first_fixation")
    stimulus = datum.stimulus
    stimulus.add_layer(
        AnnotationLayer("pairs", PairSegmenter()(stimulus.text), ["ab", "cd", "ef"])
    )
    reconstructed = dataset.reaggregate(datum, "pairs")
    np.testing.assert_equal(reconstructed.values, [100, 100, 100])


def test_custom_sentence_ids_follow_spans(dataset):
    dataset.stimuli()[0].metadata["per_word_sentence_ids"] = [10, 10, 20, 20, 30, 30]
    spec = EvaluationSpec(
        predictors=[UnitLength()],
        targets=[dataset.feature("first_fixation")],
        segmentation=PairSegmenter(),
    )
    rows = prepare_observations(dataset, spec, AnalysisConfig(transform_dv=lambda x: x))
    np.testing.assert_equal(rows.sentence_ids, [20, 30])


def test_source_unknown_is_not_reinterpreted_as_a_skip(dataset):
    dataset._fixations = [f for f in dataset._fixations if f.unit_index != 3]
    next(d for d in dataset._data if d.measure_name == "first_fixation").values[3] = (
        np.nan
    )
    [datum] = dataset.data("first_fixation")
    with pytest.raises(ValueError, match="do not reproduce the native measure"):
        dataset.reaggregate(datum, "words")


def test_first_fixation_cannot_be_declared_additive(dataset):
    dataset.additive_measures = frozenset({"first_fixation"})
    dataset._fixations = []
    with pytest.raises(ValueError, match="not additive"):
        resolve_features(
            dataset, [dataset.feature("first_fixation")], segmentation=PairSegmenter()
        )


def test_character_landings_support_subword_units(dataset):
    from replica.segmentation import CharacterSegmenter, segment_stimulus

    stimulus = dataset._make_stimulus(["ab", "cd"], "characters-test")
    events = [
        replace(
            dataset._fixations[0],
            stimulus_id=stimulus.stimulus_id,
            unit_index=u,
            char_offset=offset,
            char_offset_is_exact=True,
            start_time=i,
            duration=duration,
        )
        for i, (u, offset, duration) in enumerate([(0, 1, 150), (1, 3, 200)])
    ]
    datum = Datum(stimulus, "first_fixation", np.array([150.0, 200.0]), "p")
    dataset.add_datum(datum)
    target = segment_stimulus(stimulus, CharacterSegmenter())
    actual = dataset.reaggregate(datum, target, fixations=events)
    np.testing.assert_equal(actual.values, [np.nan, 150, 200, np.nan])


@pytest.mark.parametrize("drop_skips", [False, True])
def test_unexported_measure_uses_validated_events(dataset, drop_skips):
    dataset.drop_skips = drop_skips
    stimulus = dataset.stimuli()[0]
    stimulus.add_layer(
        AnnotationLayer("pairs", PairSegmenter()(stimulus.text), ["ab", "cd", "ef"])
    )
    [anchor] = dataset.data("first_fixation")
    result = dataset.reaggregate(anchor, "pairs", measure="right_bounded")
    assert result.measure_name == "right_bounded"
    np.testing.assert_equal(result.values, [300, 300, 200])
    assert result.metadata["native_reference_measure"] == "first_fixation"
    assert result.metadata["has_native_reference"] is False
    assert "right_bounded" not in dataset.measures
    assert anchor.measure_name == "first_fixation"


def test_another_exported_measure_must_pass_its_own_round_trip(dataset):
    [anchor] = dataset.data("first_fixation")
    for datum in dataset._data:
        if datum.measure_name == "go_past":
            datum.values[1] = 999
    with pytest.raises(ValueError, match="'go_past'.*do not reproduce"):
        dataset.reaggregate(anchor, "words", measure="go_past")


def test_unexported_measure_cannot_bypass_an_invalid_anchor(dataset):
    [anchor] = dataset.data("first_fixation")
    with pytest.raises(ValueError, match="do not reproduce"):
        dataset.reaggregate(
            replace(anchor, values=anchor.values + 1), "words", measure="right_bounded"
        )


def test_unknown_measure_is_rejected(dataset):
    [anchor] = dataset.data("first_fixation")
    with pytest.raises(NotImplementedError, match="does not implement"):
        dataset.reaggregate(anchor, "words", measure="not_a_measure")


def test_unexported_measure_does_not_inherit_another_measures_availability(dataset):
    # An arrival after a rightward landing has no first pass, but still has a
    # first fixation. UCL's unavailable-outside-first-pass rule applies only to
    # its released go-past.
    dataset._data = [d for d in dataset._data if d.measure_name == "first_fixation"]
    [anchor] = dataset._data
    anchor.metadata["unavailable_outside_first_pass"] = True
    dataset._fixations[0] = replace(dataset._fixations[0], unit_index=1, char_offset=2)
    dataset._fixations[1] = replace(dataset._fixations[1], unit_index=0, char_offset=0)
    dataset.drop_skips = False
    result = dataset.reaggregate(anchor, "words", measure="go_past")
    assert result.values[0] == 200


@pytest.mark.parametrize("drop_skips", [False, True])
def test_unexported_measure_preserves_skip_choice(dataset, drop_skips):
    dataset._fixations = [e for e in dataset._fixations if e.unit_index != 5]
    for datum in dataset._data:
        datum.values[-1] = 0
    dataset.drop_skips = drop_skips
    [anchor] = dataset.data("first_fixation")
    result = dataset.reaggregate(anchor, "words", measure="right_bounded")
    np.testing.assert_equal(result.values[-1], np.nan if drop_skips else 0)


def test_requesting_another_measure_cannot_change_the_presentation(dataset):
    events = dataset._fixations
    dataset._fixations = [
        replace(event, metadata={"trial_index": trial})
        for trial in (1, 2)
        for event in events
    ]
    for datum in dataset._data:
        datum.metadata["source_trial_index"] = (
            2 if datum.measure_name == "go_past" else 1
        )
    [anchor] = dataset.data("first_fixation")
    with pytest.raises(ValueError, match="another presentation"):
        dataset.reaggregate(anchor, "words", measure="go_past")


def test_unexported_first_run_does_not_assume_source_run_labels_are_contiguous(dataset):
    dataset._data = [d for d in dataset._data if d.measure_name == "first_fixation"]
    dataset._fixations = [
        replace(e, metadata={"run_id": 1}) for e in dataset._fixations
    ]
    [anchor] = dataset.data("first_fixation")
    result = dataset.reaggregate(anchor, "words", measure="first_run")
    np.testing.assert_equal(result.values, [100] * 6)
    assert result.metadata["use_source_runs"] is False


def test_explicit_calculation_rebuilds_runs_after_merging_units(dataset):
    from replica.dataset import FixationCalculation

    spec = EvaluationSpec(
        predictors=[UnitLength()],
        targets=[
            fixation_feature(dataset, "first_run", calculation=FixationCalculation())
        ],
        segmentation=PairSegmenter(),
    )
    rows = prepare_observations(dataset, spec, AnalysisConfig(transform_dv=lambda x: x))
    np.testing.assert_equal(rows.y, [300, 200])
    assert rows.response["calculation"] == FixationCalculation().to_dict()
    assert all(
        o["calculation"] == rows.response["calculation"]
        for o in rows.response["origins"]
    )


def test_alternative_selections_require_distinct_target_names(dataset):
    with pytest.raises(ValueError, match="Target names must be unique"):
        EvaluationSpec(
            predictors=[UnitLength()],
            targets=[
                dataset.feature("first_run"),
                dataset.feature("first_run", source="released"),
            ],
        )


@pytest.mark.parametrize("source", ["default", "released"])
def test_recording_analysis_cannot_ignore_a_unit_target(dataset, source):
    from replica.analysis.evaluation.evaluate import evaluate
    from replica.analysis.specification.config import EncodingConfig

    spec = EvaluationSpec(
        predictors=[UnitLength()],
        targets=[dataset.feature("first_run", source=source)],
    )
    with pytest.raises(ValueError, match="Recording|sampled"):
        evaluate(dataset, spec, EncodingConfig())


@pytest.mark.parametrize("segmentation", [None, WhitespaceSegmenter()])
def test_repeated_observations_require_distinct_presentation_coordinates(
    dataset, segmentation
):
    [datum] = dataset.data("first_run")
    dataset.add_datum(replace(datum, values=datum.values.copy()))
    spec = EvaluationSpec(
        predictors=[UnitLength()],
        targets=[dataset.feature("first_run")],
        segmentation=segmentation,
    )
    with pytest.raises(ValueError, match="duplicate|Duplicate|unique"):
        prepare_observations(dataset, spec)
