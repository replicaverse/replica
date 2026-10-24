"""Target roles, validation and shared extraction across comparisons."""

import numpy as np
import pytest

from replica.analysis import OLS, EncodingConfig, evaluate, prepare_observations
from replica.analysis.specification.config import AnalysisConfig
from replica.analysis.specification.evaluation import EvaluationSpec
from replica.core import AnnotationLayer, Datum, Span, Stimulus
from replica.feature import (
    FeatureMeasurement,
    ModelFeature,
    Surprisal,
    UnitLength,
    UnitPosition,
    aggregate,
)
from replica.segmentation import CharacterSegmenter
from tests.analysis._fixtures import ConstantLM, VaryingLM, make_toy_dataset


def test_invalid_target_definitions_fail_before_extraction():
    dataset = make_toy_dataset()
    model = ConstantLM()
    predictor = ModelFeature(model, Surprisal())
    invalid = FeatureMeasurement("surprisal", "bits")
    with pytest.raises(ValueError, match="at least one target"):
        EvaluationSpec(targets=[], predictors=[predictor])
    with pytest.raises(TypeError, match="Feature protocol"):
        EvaluationSpec(targets=[invalid], predictors=[predictor])
    spec = EvaluationSpec(targets=[dataset.feature("spr_rt")], predictors=[predictor])
    spec.targets.append(invalid)
    for prepare in (evaluate, prepare_observations):
        with pytest.raises(TypeError, match="Feature protocol"):
            prepare(dataset, spec)
    with pytest.raises(TypeError, match="Feature protocol"):
        evaluate(dataset, spec, EncodingConfig())
    assert model.call_count == 0


@pytest.mark.parametrize("quantity,modality", [("unknown", None), ("spr_rt", "meg")])
def test_unavailable_target_is_not_replaced_with_a_default(quantity, modality):
    dataset = make_toy_dataset()
    model = ConstantLM()
    spec = EvaluationSpec(
        targets=[dataset.feature(quantity, modality=modality)],
        predictors=[ModelFeature(model, Surprisal())],
    )
    for prepare in (evaluate, prepare_observations):
        with pytest.raises(ValueError, match="unavailable"):
            prepare(dataset, spec)
    assert model.call_count == 0


@pytest.mark.parametrize("comparison", ["joint", "individual"])
def test_multiple_targets_preserve_labels_order_and_share_extraction(comparison):
    dataset = make_toy_dataset(n_stimuli=4, measures=["spr_rt", "total_rt"])
    model = ConstantLM()
    results = evaluate(
        dataset,
        EvaluationSpec(
            targets=[dataset.feature("spr_rt"), dataset.feature("total_rt")],
            predictors=[ModelFeature(model, Surprisal()), UnitLength()],
            comparison=comparison,
        ),
        AnalysisConfig(design=None, transform_dv=np.log2, model=OLS(), n_folds=2),
    )
    names = (
        ["joint(surprisal, unit_length)"]
        if comparison == "joint"
        else ["surprisal", "unit_length"]
    )
    assert [(result.feature_name, result.dv_name) for result in results] == [
        (name, target) for name in names for target in ("spr_rt", "total_rt")
    ]
    assert all(result.n_scored == 16 for result in results)
    assert model.call_count == 4


@pytest.mark.parametrize(
    "rotation", range(3), ids=["observed-target", "model-target", "stimulus-target"]
)
def test_observed_model_and_stimulus_quantities_each_fill_every_role(rotation):
    dataset = make_toy_dataset(n_stimuli=6, n_words=6, n_participants=2)
    human = aggregate(
        dataset.feature("spr_rt"), by="unit", statistic="mean", missing="drop"
    )
    model = ModelFeature(VaryingLM(), Surprisal())
    position = UnitPosition()
    features = (human, model, position)
    raw = {
        (datum.stimulus.stimulus_id, datum.participant_id): datum.values
        for datum in dataset._data
    }
    expected = {}
    for stimulus in dataset.stimuli():
        sid = stimulus.stimulus_id
        source_offset = sum(ord(char) for char in sid) % 5
        # The fixture excludes unit zero; its model emits natural-log probabilities.
        for unit in range(1, 6):
            expected[sid, unit] = {
                human.name: (raw[sid, "p0"][unit] + raw[sid, "p1"][unit]) / 2,
                model.name: (1 + ((3 * unit + source_offset) % 7) / 2) / np.log(2),
                position.name: unit,
            }
    target, predictor, control = features[rotation:] + features[:rotation]
    spec = EvaluationSpec(targets=[target], predictors=[predictor], controls=[control])
    config = AnalysisConfig(design=None, model=OLS(), n_folds=3)
    rows = prepare_observations(dataset, spec, config)
    identities = list(zip(rows.stimulus_ids, rows.unit_indices, strict=True))
    assert identities == list(expected)
    expected_target = [expected[key][target.name] for key in identities]
    np.testing.assert_allclose(rows.y, expected_target)
    np.testing.assert_allclose(rows.raw_y, expected_target)
    assert set(rows.columns) == {predictor.name, control.name}
    for feature in (predictor, control):
        np.testing.assert_allclose(
            rows.columns[feature.name],
            [expected[key][feature.name] for key in identities],
        )
    (result,) = evaluate(dataset, spec, config)
    assert result.dv_name == target.name
    assert result.feature_name == predictor.name
    assert result.n_scored == len(expected) == 30
    assert np.isfinite(result.delta_ll)


class TestTargetSegmentation:
    def test_finer_segmentation_raises(self):
        """CharacterSegmenter on word-level data raises (can't split words)."""
        ds = make_toy_dataset()
        lm = ConstantLM()
        with pytest.raises(ValueError, match="incompatible|not fully contained"):
            evaluate(
                ds,
                EvaluationSpec(
                    targets=[ds.feature("spr_rt")],
                    predictors=[ModelFeature(lm, Surprisal())],
                    segmentation=CharacterSegmenter(),
                ),
                analysis=AnalysisConfig(
                    design=None, transform_dv=np.log2, model=OLS(), n_folds=3
                ),
            )

    def test_non_additive_aggregation_raises(self):
        """Coarser segmentation on a non-additive measure raises."""
        # Build a dataset with "first_fixation" (non-additive) at word level,
        # then try to segment into sentences (coarser)
        from replica.dataset import (
            DatasetInfo,
            EyeTrackingDataset,
        )

        class ToyETDataset(EyeTrackingDataset):
            info = DatasetInfo(name="toy_et")

            def load(self, path):
                pass

        ds = ToyETDataset()
        rng = np.random.default_rng(42)
        words = ["the", "cat", "sat"]
        text = " ".join(words)
        spans = []
        pos = 0
        for w in words:
            spans.append(Span(pos, pos + len(w)))
            pos += len(w) + 1
        layer = AnnotationLayer("words", spans, words)
        stim = Stimulus(text=text, stimulus_id="s0")
        stim.add_layer(layer)
        ds._stimuli.append(stim)
        ds._data.append(
            Datum(
                stimulus=stim,
                measure_name="first_fixation",
                values=rng.uniform(100, 500, size=3),
                participant_id="p0",
            )
        )

        # Sentence segmenter: one span covering the whole text
        from replica.segmentation import from_predicate

        sentence_seg = from_predicate(lambda t, i: False, "sentences")

        lm = ConstantLM()
        with pytest.raises(ValueError, match="not additive"):
            evaluate(
                ds,
                EvaluationSpec(
                    targets=[ds.feature("first_fixation")],
                    predictors=[ModelFeature(lm, Surprisal())],
                    segmentation=sentence_seg,
                ),
                analysis=AnalysisConfig(
                    design=None, transform_dv=np.log2, model=OLS(), n_folds=2
                ),
            )
