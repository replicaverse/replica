"""Guards that turn silent partial failures in ``evaluate()`` into loud errors
or visible log lines; a silently dropped or misaligned row can change a
reported delta-LL with no signal.
"""

from dataclasses import replace

import numpy as np
import pytest

from replica import evaluate
from replica.analysis import OLS, AnalysisConfig
from replica.analysis.specification.evaluation import EvaluationSpec
from replica.core import AnnotationLayer, Datum, Span, Stimulus
from replica.dataset.base import SPRDataset
from replica.dataset.contracts import DatasetInfo
from replica.feature import (
    FeatureModelOutput,
    ModelFeature,
    Surprisal,
    UnitLength,
)
from tests.analysis._fixtures import ConstantLM, make_toy_dataset


class _NaNMidFeature:
    """Data-derived feature with a NaN at a DV-valid position (index 2), so it
    forces row attrition that the accounting must report."""

    name = "nanmid"

    def __init__(self, masked=False):
        self.masked = masked

    def __call__(self, stimulus, outputs, target_layer="words"):
        n = len(stimulus.layers[target_layer])
        arr = np.ones(n)
        if n > 2:
            arr[2] = np.nan
        if self.masked:
            return np.ma.array(np.nan_to_num(arr, nan=999.0), mask=np.isnan(arr))
        return arr


class _ShortOutputLM:
    """FeatureModel that violates the one-output-per-unit contract."""

    name = "short"
    vocab = None

    def __call__(self, stimulus, extract=None, target_layer="words"):
        n = len(stimulus.layers[target_layer])
        nlp = np.log(np.full(3, 1 / 3))
        # Deliberately one short.
        return [
            FeatureModelOutput(values={"log_p_observed": -1.0, "next_log_probs": nlp})
            for _ in range(n - 1)
        ]


class _WrongLengthFeature:
    """Feature that returns a wrong-length array (protocol violation)."""

    name = "bad_feat"

    def __call__(self, stimulus, outputs, target_layer="words"):
        n = len(stimulus.layers[target_layer])
        return np.zeros(n + 2)  # too long


def test_model_wrong_output_length_raises():
    ds = make_toy_dataset(n_stimuli=2, n_words=5)
    spec = EvaluationSpec(
        targets=[ds.feature("spr_rt")],
        predictors=[ModelFeature(_ShortOutputLM(), Surprisal())],
    )
    with pytest.raises(ValueError, match="outputs do not match.*annotation units"):
        list(evaluate(ds, spec))


@pytest.mark.parametrize("masked", [False, True], ids=["nan", "masked"])
def test_prepared_observations_retain_identity_after_feature_exclusion(masked):
    from replica.analysis.data.preparation import prepare_observations
    from replica.analysis.specification.config import AnalysisConfig

    ds = make_toy_dataset(n_stimuli=3, n_words=5)
    spec = EvaluationSpec(
        targets=[ds.feature("spr_rt")],
        predictors=[ModelFeature(ConstantLM(), _NaNMidFeature(masked=masked))],
    )
    rows = prepare_observations(
        ds, spec, AnalysisConfig(transform_dv=lambda values: values)
    )
    assert len(rows.stimulus_ids) == len(rows.y)
    assert set(rows.stimulus_ids) == {s.stimulus_id for s in ds.stimuli()}
    assert 2 not in rows.unit_indices
    for sid, index, value in zip(
        rows.stimulus_ids, rows.unit_indices, rows.y, strict=True
    ):
        assert any(
            d.stimulus.stimulus_id == sid and d.values[index] == value
            for d in ds.data(measure="spr_rt")
        )


@pytest.mark.parametrize(
    "sentence_ids",
    [
        [1, 2],
        [[1], [2], [3], [4], [5]],
        [1, 2, 3.5, 4, 5],
        [1, 2, np.inf, 4, 5],
        [True] * 5,
        np.array([2**63] * 5, dtype=np.uint64),
        ["1", "2", "3", "4", "5"],
        np.ma.array([0, 999, 1, 2, 3], mask=[False, True, False, False, False]),
    ],
)
def test_supplied_sentence_ids_must_align_without_loss(sentence_ids):
    from replica.analysis import prepare_observations

    ds = make_toy_dataset(n_stimuli=2, n_words=5)
    ds.stimuli()[0].metadata["per_word_sentence_ids"] = sentence_ids
    spec = EvaluationSpec(
        targets=[ds.feature(ds.default_measure)], predictors=[UnitLength()]
    )
    config = AnalysisConfig(model=OLS(), design=None)
    for prepare in (prepare_observations, evaluate):
        with pytest.raises(ValueError, match="per_word_sentence_ids"):
            prepare(ds, spec, config)


@pytest.mark.parametrize(
    "sentence_ids",
    [
        np.array([10, 10, 20, 20, 30], dtype=np.int64),
        np.array([10, 10, 20, 20, 30], dtype=np.float64),
        [np.uint64(2**53 + 1), 2**53 + 1, 20, 20, 30],
    ],
)
def test_supplied_sentence_ids_survive_observation_filtering(sentence_ids):
    from replica.analysis import prepare_observations

    ds = make_toy_dataset(n_stimuli=2, n_words=5)
    for stimulus in ds.stimuli():
        stimulus.metadata["per_word_sentence_ids"] = sentence_ids
    spec = EvaluationSpec(
        targets=[ds.feature(ds.default_measure)],
        predictors=[ModelFeature(ConstantLM(), _NaNMidFeature())],
    )
    rows = prepare_observations(ds, spec, AnalysisConfig(model=OLS(), design=None))
    np.testing.assert_array_equal(
        rows.sentence_ids, [sentence_ids[1], sentence_ids[3], sentence_ids[4]] * 2
    )


def test_feature_wrong_array_length_raises():
    ds = make_toy_dataset(n_stimuli=2, n_words=5)
    spec = EvaluationSpec(
        targets=[ds.feature("spr_rt")],
        predictors=[ModelFeature(ConstantLM(), _WrongLengthFeature())],
    )
    with pytest.raises(ValueError, match="one first-axis row per observation"):
        list(evaluate(ds, spec))


def test_feature_nan_attrition_is_recorded():
    ds = make_toy_dataset(n_stimuli=3, n_words=5)
    spec = EvaluationSpec(
        targets=[ds.feature("spr_rt")],
        predictors=[ModelFeature(ConstantLM(), _NaNMidFeature())],
    )
    result = evaluate(ds, spec, AnalysisConfig(model=OLS(), design=None))[0]
    assert result.response["row_exclusions"]["missing_features"] == 3
    assert result.n_observations == 9


def _dataset_with_partial_datum_feature():
    """DV present for two participants; the datum-level feature 'rate' present
    for p0 only, so p1's rows must be dropped (and reported)."""

    class _DS(SPRDataset):
        info = DatasetInfo(name="toy_partial")

        def load(self, path):
            pass

    ds = _DS()
    for i in range(3):
        words = [f"w{i}_{j}" for j in range(5)]
        text = " ".join(words)
        pos, spans = 0, []
        for w in words:
            spans.append(Span(pos, pos + len(w)))
            pos += len(w) + 1
        stim = Stimulus(text=text, stimulus_id=f"s{i}")
        stim.add_layer(AnnotationLayer("words", spans, words))
        ds._stimuli.append(stim)
        for pid in ("p0", "p1"):
            ds._data.append(
                Datum(
                    stimulus=stim,
                    measure_name="spr_rt",
                    values=np.full(5, 300.0),
                    participant_id=pid,
                )
            )
        # 'rate' only for p0
        ds._data.append(
            Datum(
                stimulus=stim,
                measure_name="rate",
                values=np.full(5, 1.0),
                participant_id="p0",
            )
        )
    return ds


def test_unloaded_dataset_raises_clear_error():
    class _DS(SPRDataset):
        info = DatasetInfo(name="never_loaded")

        def load(self, path):
            pass

    ds = _DS()  # never populated
    spec = EvaluationSpec(
        targets=[ds.feature("spr_rt")],
        predictors=[ModelFeature(ConstantLM(), Surprisal())],
    )
    with pytest.raises(ValueError, match=r"no stimuli.*load"):
        list(evaluate(ds, spec))


def test_a_stimulus_listed_twice_is_refused():
    """Rows are built once per listing, so a repeated stimulus would enter the
    regression twice, under two stimulus groups."""
    from replica.analysis.cv import ItemSplitter
    from replica.analysis.data.preparation import prepare_observations
    from replica.analysis.models import OLS
    from replica.analysis.specification.config import AnalysisConfig

    ds = make_toy_dataset(n_stimuli=3, n_words=5)
    ds._stimuli.append(ds._stimuli[0])
    spec = EvaluationSpec(
        targets=[ds.feature("spr_rt")],
        predictors=[ModelFeature(ConstantLM(), Surprisal())],
    )
    with pytest.raises(ValueError, match="duplicate stimulus identities"):
        evaluate(
            ds,
            spec,
            AnalysisConfig(
                design=None, transform_dv=np.log2, model=OLS(), cv=ItemSplitter(2)
            ),
        )
    with pytest.raises(ValueError, match="duplicate stimulus identities"):
        prepare_observations(ds, spec)


def test_missing_dataset_feature_records_and_keeps_present_rows():
    from replica.analysis import prepare_observations

    ds = _dataset_with_partial_datum_feature()
    spec = EvaluationSpec(
        targets=[ds.feature("spr_rt")],
        predictors=[UnitLength()],
        controls=[ds.feature("rate")],
    )
    rows = prepare_observations(ds, spec, AnalysisConfig(model=OLS(), design=None))
    assert rows.response["row_exclusions"]["missing_features"] == 12
    assert len(rows.y) == 12
    assert rows.participant_ids == {"p0": 0}


@pytest.mark.parametrize("transform", [np.log2, np.ma.log2], ids=["plain", "masked"])
def test_nonpositive_target_is_recorded_after_explicit_log_transform(transform):
    from replica.analysis import prepare_observations

    ds = make_toy_dataset(n_stimuli=4, n_words=5)
    ds.exclude_first_unit = False
    ds._data[0].values[1] = 0.0
    spec = EvaluationSpec(targets=[ds.feature("spr_rt")], predictors=[UnitLength()])
    config = AnalysisConfig(model=OLS(), design=None, transform_dv=transform)
    rows = prepare_observations(ds, spec, config)
    raw = np.concatenate([datum.values for datum in ds._data])
    np.testing.assert_allclose(rows.y, np.log2(raw[raw > 0]))
    result = evaluate(ds, spec, config)[0]
    assert np.isfinite(result.delta_ll)
    assert result.response["row_exclusions"]["nonfinite_transformed_response"] == 1
    assert result.n_observations == 19
    for prepare in (prepare_observations, evaluate):
        with pytest.raises(ValueError, match="Target transform produced nonfinite"):
            prepare(ds, spec, replace(config, missing="error"))


@pytest.mark.parametrize(
    "transform,message",
    [
        (lambda values: values + 1j, "DV transform must return real values"),
        (lambda values: values[:, None], "DV transform must preserve shape"),
    ],
    ids=["complex", "shape"],
)
def test_target_transform_rejects_invalid_outputs(transform, message):
    from replica.analysis import prepare_observations

    ds = make_toy_dataset(n_stimuli=2, n_words=5)
    spec = EvaluationSpec(targets=[ds.feature("spr_rt")], predictors=[UnitLength()])
    config = AnalysisConfig(model=OLS(), design=None, transform_dv=transform)
    for prepare in (prepare_observations, evaluate):
        with pytest.raises(ValueError, match=message):
            prepare(ds, spec, config)


@pytest.mark.parametrize(
    "key,measure",
    [
        (key, measure)
        for key in ("celer", "dundee", "dundee_official", "devarda2023")
        for measure in ("first_run", "first_run_in_first_pass")
    ]
    + [("devarda2023", "right_bounded")],
)
def test_loader_rejects_summing_reading_windows_across_words(key, measure):
    from replica.dataset.catalog.registry import REGISTRY
    from replica.feature import resolve
    from replica.segmentation import Segmenter

    class WholeText(Segmenter):
        def segment(self, text):
            return [Span(0, len(text))]

    ds = REGISTRY[key].loader_cls()()
    stimulus = Stimulus(text="one two", stimulus_id="s1")
    stimulus.add_layer(
        AnnotationLayer("words", [Span(0, 3), Span(4, 7)], ["one", "two"])
    )
    ds._stimuli = [stimulus]
    ds._data = [Datum(stimulus, measure, np.array([100.0, 200.0]), "p1")]
    with pytest.raises(ValueError, match="is not additive"):
        resolve(ds, ds.feature(measure), segmentation=WholeText("region"))
