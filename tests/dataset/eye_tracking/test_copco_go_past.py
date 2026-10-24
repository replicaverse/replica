"""CopCo's standard regression paths and separately retained extractor values."""

import csv
from dataclasses import replace

import numpy as np
import pytest

from replica.core import AnnotationLayer, Span
from replica.dataset import CopCo, FixationCalculation
from replica.dataset.contracts import Fixation
from replica.dataset.loading.fixation_utils import copco_extractor_go_past
from replica.dataset.loading.reaggregation import aggregate_fixations_to_layer


def _events(units, durations):
    return [
        Fixation(
            "p", "s", i * 100, i * 100 + duration, duration, 0, 0, unit, None, i + 1
        )
        for i, (unit, duration) in enumerate(zip(units, durations, strict=True))
    ]


@pytest.mark.parametrize(
    "units,durations,expected",
    [
        ([1, 0, 1, 0, 1, 2], [100, 50, 80, 40, 60, 90], [50, 230, 90]),
        ([1, 0, 2], [100, 50, 90], [50, 100, 90]),
        ([1, 1, 0, 1, 2], [100, 80, 50, 60, 90], [50, 180, 90]),
        ([1, None, 1, 2], [100, 50, 80, 90], [np.nan, 100, 90]),
    ],
)
def test_released_extractor_windows(units, durations, expected):
    np.testing.assert_equal(
        copco_extractor_go_past(_events(units, durations), 3), expected
    )


def _write_release(root, *, with_fixations=True):
    features = root / "ExtractedFeatures"
    features.mkdir()
    names = [
        "word_first_fix_dur",
        "word_first_pass_dur",
        "word_total_fix_dur",
        "word_go_past_time",
    ]
    rows = []
    for i, (word, values) in enumerate(
        zip(
            "ABCD",
            [[150, 150, 400, 150], [100, 100, 600, 450], [400] * 4, [""] * 4],
            strict=True,
        )
    ):
        rows.append(
            dict(
                speechId=1,
                paragraphId=0,
                sentenceId=1,
                wordId=i,
                trialId=1,
                word=word,
                char_IA_ids=str([i + 1]),
                **dict(zip(names, values, strict=True)),
            )
        )
    with (features / "P01.csv").open("w") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if not with_fixations:
        return
    reports = root / "FixationReports"
    reports.mkdir()
    rows = [
        dict(
            RECORDING_SESSION_LABEL="P01",
            paragraphid=0,
            speechid=1,
            TRIAL_INDEX=1,
            CURRENT_FIX_INDEX=i + 1,
            CURRENT_FIX_DURATION=d,
            CURRENT_FIX_INTEREST_AREA_INDEX=u + 1,
            CURRENT_FIX_INTEREST_AREA_LABEL="ABCD"[u],
        )
        for i, (u, d) in enumerate(
            zip([1, 0, 1, 0, 1, 2], [100, 150, 200, 250, 300, 400], strict=True)
        )
    ]
    with (reports / "FIX_report_P01.txt").open("w") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


@pytest.mark.parametrize("drop_skips", [False, True])
def test_standard_and_released_views_have_distinct_native_and_custom_windows(
    tmp_path, drop_skips
):
    _write_release(tmp_path)
    dataset = CopCo()
    dataset.load(str(tmp_path))
    dataset.drop_skips = drop_skips
    [standard] = dataset.data("go_past")
    [released] = dataset.data("go_past", source="released")
    [conditional] = dataset.data("go_past_in_first_pass")
    skipped = np.nan if drop_skips else 0
    np.testing.assert_equal(standard.values, [150, 1000, 400, skipped])
    np.testing.assert_equal(released.values, [150, 450, 400, np.nan])
    np.testing.assert_equal(conditional.values, [skipped, 1000, 400, skipped])
    assert standard.metadata["measurement_source"] == "fixations"
    stimulus = standard.stimulus
    stimulus.add_layer(
        AnnotationLayer("text", [Span(0, len(stimulus.text))], [stimulus.text])
    )
    for datum, expected in [(standard, 1400), (released, 1400), (conditional, 1400)]:
        np.testing.assert_equal(
            dataset.reaggregate(datum, datum.layer).values, datum.values
        )
        np.testing.assert_equal(dataset.reaggregate(datum, "text").values, [expected])


@pytest.mark.parametrize("drop_skips", [False, True])
def test_explicit_calculation_validates_source_and_records_its_own_identity(
    tmp_path, drop_skips
):
    _write_release(tmp_path)
    dataset = CopCo()
    dataset.load(str(tmp_path))
    dataset.drop_skips = drop_skips
    [source] = dataset.data("go_past", source="released")
    events = dataset.fixations(source.participant_id, source.stimulus.stimulus_id)
    cache = {}
    for calculation, expected in [
        (FixationCalculation(), 1000),
        (FixationCalculation(go_past="copco_extractor@1"), 450),
        (FixationCalculation(), 1000),
    ]:
        result = dataset.reaggregate(
            source,
            "words",
            measure="go_past",
            calculation=calculation,
            fixations=events,
            _cache=cache,
        )
        assert result.values[1] == expected
        assert result.measure_name == "go_past"
        assert np.isnan(result.values[3]) if drop_skips else result.values[3] == 0
        assert result.metadata["calculation"] == calculation.to_dict()
        assert result.metadata["native_reference_measure"] == "go_past"
        assert not result.metadata["has_native_reference"]
    broken = [replace(e, duration=e.duration + 1) for e in events]
    with pytest.raises(ValueError, match="do not reproduce"):
        dataset.reaggregate(
            source,
            "words",
            measure="go_past",
            calculation=FixationCalculation(),
            fixations=broken,
        )


def test_standard_go_past_never_falls_back_to_released_values_without_events(tmp_path):
    _write_release(tmp_path, with_fixations=False)
    dataset = CopCo()
    dataset.load(str(tmp_path))
    assert np.isnan(dataset.data("go_past")[0].values).all()
    assert dataset.data("go_past", source="released")[0].values[1] == 450
    assert (
        dataset.data("go_past")[0].metadata["unavailable_reason"]
        == "source_fixations_absent"
    )


def test_corrupt_released_window_fails_source_validation(tmp_path):
    _write_release(tmp_path)
    path = tmp_path / "ExtractedFeatures/P01.csv"
    path.write_text(path.read_text().replace("100,100,600,450", "100,100,600,999"))
    with pytest.raises(ValueError, match="do not reproduce go_past"):
        CopCo().load(str(tmp_path))


def test_released_windows_do_not_join_distinct_presentations():
    events = _events([0, 1], [100, 200])
    events[1] = replace(events[1], metadata={"trial_index": 2})
    with pytest.raises(ValueError, match="one source trial"):
        copco_extractor_go_past(events, 2)


@pytest.mark.needs_data
@pytest.mark.slow
def test_complete_release_and_independent_standard_windows(data_root):
    from collections import defaultdict
    from pathlib import Path

    root = Path(data_root) / "eye-tracking/original/copco"
    if not root.is_dir():
        pytest.skip("CopCo source is unavailable")
    dataset = CopCo()
    dataset.load(str(root))
    dataset.drop_skips = False
    source = {}
    for path in sorted((root / "ExtractedFeatures").glob("P*.csv")):
        with path.open() as file:
            for row in csv.DictReader(file):
                key = (
                    path.stem,
                    *(
                        int(row[name])
                        for name in ("speechId", "paragraphId", "sentenceId", "wordId")
                    ),
                )
                assert key not in source
                source[key] = float(row["word_go_past_time"] or "nan")
    references = defaultdict(dict)
    for datum in dataset.data():
        if datum.measure_name in {"go_past", "go_past_in_first_pass"}:
            references[datum.participant_id, datum.stimulus.stimulus_id][
                datum.measure_name
            ] = datum
    for datum in dataset.data("go_past", source="released"):
        references[datum.participant_id, datum.stimulus.stimulus_id]["released"] = datum
    compared = changed = 0
    for pair, raw in dataset.fixation_trials():
        datums = references.pop(pair)
        released = datums["released"]
        stimulus = released.stimulus
        metadata = stimulus.metadata
        expected_release = [
            source.pop(
                (
                    pair[0],
                    metadata["speech_id"],
                    metadata["paragraph_id"],
                    sentence,
                    word,
                )
            )
            for sentence, word in metadata["source_word_ids"]
        ]
        np.testing.assert_equal(released.values, expected_release)
        events = dataset._reaggregation_fixations(released, raw)
        rebuilt = aggregate_fixations_to_layer(
            events,
            stimulus,
            calculation=FixationCalculation(go_past="copco_extractor@1"),
        )
        rebuilt["_released_go_past"] = rebuilt["go_past"]
        np.testing.assert_equal(
            dataset._reaggregation_values(released, rebuilt), released.values
        )
        assert len({e.metadata.get("trial_index") for e in events}) == 1
        events = sorted(events, key=lambda e: e.fixation_index)
        expected = np.zeros(len(released.values))
        conditional = np.zeros(len(released.values))
        seen = set()
        for start, event in enumerate(events):
            unit = event.unit_index
            if unit is None or unit in seen:
                continue
            seen.add(unit)
            end = next(
                (
                    i
                    for i in range(start + 1, len(events))
                    if events[i].unit_index is not None and events[i].unit_index > unit
                ),
                len(events),
            )
            expected[unit] = sum(
                e.duration for e in events[start:end] if e.unit_index is not None
            )
            if not any(
                e.unit_index is not None and e.unit_index > unit for e in events[:start]
            ):
                conditional[unit] = expected[unit]
        np.testing.assert_equal(datums["go_past"].values, expected)
        np.testing.assert_equal(datums["go_past_in_first_pass"].values, conditional)
        changed += int(
            (np.isfinite(released.values) & (released.values != expected)).sum()
        )
        compared += len(expected)
    assert not source
    assert not references
    assert compared == 335_203
    assert changed == 34_080


@pytest.mark.parametrize("segmented", [False, True])
def test_response_selection_reaches_analysis_and_serializes(tmp_path, segmented):
    from replica.analysis import AnalysisConfig, prepare_observations
    from replica.analysis.specification.evaluation import EvaluationSpec
    from replica.dataset import fixation_feature
    from replica.feature import UnitLength
    from replica.segmentation import WhitespaceSegmenter

    _write_release(tmp_path)
    dataset = CopCo()
    dataset.load(str(tmp_path))
    choices = [
        (dataset.feature("go_past"), None, 1000),
        (dataset.feature("word_go_past_time", source="released"), None, 450),
        (
            fixation_feature(
                dataset, "go_past", source="released", calculation=FixationCalculation()
            ),
            FixationCalculation(),
            1000,
        ),
        (
            fixation_feature(
                dataset,
                "go_past",
                calculation=FixationCalculation(go_past="copco_extractor@1"),
            ),
            FixationCalculation(go_past="copco_extractor@1"),
            450,
        ),
    ]
    identities = []
    for selection, calculation, expected in choices:
        spec = EvaluationSpec(
            predictors=[UnitLength()],
            targets=[selection],
            segmentation=WhitespaceSegmenter() if segmented else None,
        )
        rows = prepare_observations(
            dataset, spec, AnalysisConfig(transform_dv=lambda x: x)
        )
        np.testing.assert_equal(rows.y, [expected, 400])
        assert rows.response["measure"] == "go_past"
        assert rows.response["source"] == selection.source
        assert rows.response["calculation"] == (
            None if calculation is None else calculation.to_dict()
        )
        assert rows.response["origins"]
        identities.append(rows.response)
        [datum] = dataset.data(
            "go_past", source=selection.source, calculation=calculation
        )
        assert datum.values[1] == expected
    assert len({repr(record) for record in identities}) == len(choices)
    assert dataset.data("go_past", source="released")[0].values[1] == 450


def test_explicit_analysis_calculation_requires_events_even_for_additive_measure(
    tmp_path,
):
    from replica.analysis import AnalysisConfig, prepare_observations
    from replica.analysis.specification.evaluation import EvaluationSpec
    from replica.dataset import fixation_feature
    from replica.feature import UnitLength

    _write_release(tmp_path, with_fixations=False)
    dataset = CopCo()
    dataset.load(str(tmp_path))
    spec = EvaluationSpec(
        predictors=[UnitLength()],
        targets=[
            fixation_feature(dataset, "total_time", calculation=FixationCalculation())
        ],
    )
    with pytest.raises(ValueError, match="require raw fixation events"):
        prepare_observations(dataset, spec, AnalysisConfig())
    with pytest.raises(ValueError, match="fixation events"):
        dataset.data("total_time", calculation=FixationCalculation())


def test_source_field_resolution_requires_dataset_context():
    from replica.dataset import MECO, PROVO

    assert MECO().feature("firstrun.dur", source="released").name == "first_run"
    with pytest.raises(ValueError, match=r"matching declarations: \[\]"):
        PROVO().feature("firstrun.dur", source="released")
