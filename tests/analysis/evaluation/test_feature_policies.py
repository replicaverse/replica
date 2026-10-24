"""Observation, fold and scoring contracts for role-independent quantities."""

from dataclasses import replace

import numpy as np
import pytest

from replica.analysis import (
    OLS,
    AnalysisConfig,
    Design,
    Factor,
    Residualize,
    evaluate,
    prepare_observations,
)
from replica.analysis.cv import ItemSplitter
from replica.analysis.models import RidgeCVModel
from replica.analysis.specification.evaluation import EvaluationSpec
from replica.contracts.features import FeatureValues, block_origins
from replica.core import AnnotationLayer
from replica.feature import (
    DerivedFeature,
    Lag,
    ModelFeature,
    Surprisal,
    UnitLength,
    UnitPosition,
    aggregate,
    named,
)
from replica.segmentation import CharacterSegmenter
from tests.analysis._fixtures import VaryingLM, make_toy_dataset
from tests.analysis.inference.test_crossed_bootstrap import GaussianBackend


@pytest.mark.parametrize("model_target", [False, True], ids=["stored", "model"])
@pytest.mark.parametrize("explicit_origins", [False, True])
def test_custom_target_dependencies_cannot_omit_lineage(model_target, explicit_origins):
    dataset = make_toy_dataset(n_stimuli=4, n_participants=2)
    target = (
        ModelFeature(VaryingLM(), Surprisal())
        if model_target
        else dataset.feature("spr_rt")
    )

    class CopyTarget:
        name = "copy_target"
        dependencies = (target,)

        def resolve(self, context, inputs):
            values = inputs[0]
            return FeatureValues(
                tuple(
                    replace(
                        block,
                        origins=block_origins(values, block)
                        if explicit_origins
                        else None,
                    )
                    for block in values.blocks
                )
            )

    message = (
        "consumes held-out target"
        if explicit_origins
        else "explicit contributor origins"
    )
    with pytest.raises(ValueError, match=message):
        evaluate(
            dataset,
            EvaluationSpec(targets=[target], predictors=[CopyTarget()]),
            AnalysisConfig(model=OLS(), design=None, n_folds=2),
        )


@pytest.mark.parametrize("derived", [False, True])
def test_scalar_features_follow_the_selected_targets_native_units(derived):
    dataset = make_toy_dataset(
        n_stimuli=2, n_words=3, n_participants=2, measures=["spr_rt", "other"]
    )
    dataset.exclude_first_unit = False
    for stimulus in dataset.stimuli():
        spans = CharacterSegmenter().segment(stimulus.text)
        stimulus.add_layer(
            AnnotationLayer(
                name="characters",
                spans=spans,
                values=[stimulus.text[span.start : span.end] for span in spans],
            )
        )
    dataset._data = [
        replace(
            datum,
            layer="characters",
            values=np.arange(len(datum.stimulus.layers["characters"]), dtype=float)
            + 10,
        )
        if datum.measure_name == "spr_rt"
        else datum
        for datum in dataset._data
    ]
    response, length = dataset.feature("spr_rt"), UnitLength()
    target = (
        DerivedFeature("scaled", lambda size, value: value / size, length, response)
        if derived
        else response
    )
    spec = EvaluationSpec(
        targets=[target],
        predictors=[UnitPosition(), ModelFeature(VaryingLM(), Surprisal())],
        controls=[length],
    )
    config = AnalysisConfig(model=OLS(), design=None, n_folds=2)
    rows = prepare_observations(dataset, spec, config)
    expected_positions = np.tile(np.arange(1, 12), 4)
    np.testing.assert_array_equal(rows.columns["unit_position"], expected_positions)
    np.testing.assert_array_equal(rows.y, expected_positions + 10)
    np.testing.assert_array_equal(rows.columns["unit_length"], np.ones(len(rows.y)))
    with pytest.raises(ValueError, match="multiple annotation layers"):
        evaluate(
            dataset, replace(spec, targets=[target, dataset.feature("other")]), config
        )


def test_observed_pooling_respects_held_out_axis_and_material_context():
    dataset = make_toy_dataset(
        n_stimuli=4, n_participants=4, measures=["spr_rt", "other"]
    )
    response = dataset.feature("spr_rt")
    length = UnitLength()
    scaled = DerivedFeature("rt_per_character", np.divide, response, length)
    # The fixed denominator remains available when the derived response is held out.
    result = evaluate(
        dataset,
        EvaluationSpec(
            targets=[scaled], predictors=[UnitPosition()], controls=[length]
        ),
        AnalysisConfig(model=OLS(), design=None, n_folds=2),
    )[0]
    assert np.isfinite(result.delta_ll)
    with pytest.raises(ValueError, match="consumes held-out target"):
        evaluate(
            dataset,
            EvaluationSpec(targets=[response], predictors=[Lag(response, n=1)]),
            AnalysisConfig(model=OLS(), design=None, n_folds=2),
        )
    pooled = aggregate(response, by="unit", statistic="mean", missing="drop")
    model_target = ModelFeature(VaryingLM(), Surprisal())
    item_design = Design(
        factors=(Factor("item", "stimulus"),),
        random_effects=(),
        cv_unit="item",
        comparison="predictor",
    )
    config = AnalysisConfig(model=GaussianBackend(), design=item_design, n_folds=2)
    result = evaluate(
        dataset, EvaluationSpec(targets=[model_target], predictors=[pooled]), config
    )[0]
    assert result.n_observations == 16
    assert np.isfinite(result.delta_ll)
    participant_design = Design(
        factors=(Factor("reader", "participant"),),
        random_effects=(),
        cv_unit="reader",
        comparison="predictor",
    )
    with pytest.raises(ValueError, match="pools measured contributors"):
        evaluate(
            dataset,
            EvaluationSpec(targets=[dataset.feature("other")], predictors=[pooled]),
            replace(config, design=participant_design),
        )

    for i, stimulus in enumerate(dataset.stimuli()):
        stimulus.metadata["batch"] = i % 2
    custom = Design(
        factors=(Factor("batch", "stimulus:batch"),),
        random_effects=(),
        cv_unit="batch",
        comparison="predictor",
    )
    native_spec = EvaluationSpec(
        targets=[dataset.feature("other")], predictors=[response]
    )
    native = evaluate(dataset, native_spec, replace(config, design=custom))[0]
    assert native.n_observations == 64
    with pytest.raises(ValueError, match="contributor alignment"):
        evaluate(
            dataset,
            replace(native_spec, predictors=[pooled]),
            replace(config, design=custom),
        )


def test_shared_cohort_uses_common_rows_and_keeps_target_identity():
    dataset = make_toy_dataset(n_stimuli=4, n_participants=2)
    predictor = ModelFeature(VaryingLM(), Surprisal())
    lagged = Lag(predictor, n=2)
    spec = EvaluationSpec(
        targets=[dataset.feature("spr_rt")],
        predictors=[predictor, lagged],
        comparison="individual",
    )
    config = AnalysisConfig(model=OLS(), design=None, n_folds=2, per_fold=True)
    separate = evaluate(dataset, spec, config)
    shared = evaluate(dataset, replace(spec, cohort="shared"), config)
    assert separate[0].n_observations > separate[1].n_observations
    assert (
        shared[0].n_observations
        == shared[1].n_observations
        == separate[1].n_observations
    )
    assert (
        shared[0].metadata["cohort_identity"] == shared[1].metadata["cohort_identity"]
    )
    assert (
        shared[0].metadata["target_identity"] == shared[1].metadata["target_identity"]
    )
    assert [fold.n_test for fold in shared[0].fold_results] == [
        fold.n_test for fold in shared[1].fold_results
    ]


class _Log:
    def __call__(self, values):
        return np.log(values)

    def inverse(self, values):
        return np.exp(values)

    def log_abs_det_jacobian(self, values):
        return -np.log(values)


def test_original_scale_uses_density_jacobian_and_inverse_prediction():
    dataset = make_toy_dataset(n_stimuli=4, n_participants=2)
    spec = EvaluationSpec(
        targets=[dataset.feature("spr_rt")], predictors=[UnitPosition()]
    )
    config = AnalysisConfig(
        model=OLS(), design=None, n_folds=2, transform_dv=_Log(), per_fold=True
    )
    rows = prepare_observations(dataset, spec, config)
    transformed = evaluate(dataset, spec, config)[0]
    original = evaluate(dataset, spec, replace(config, score_scale="original"))[0]
    assert original.delta_ll == pytest.approx(transformed.delta_ll, abs=1e-10)
    splitter = ItemSplitter(2)
    for (train, test), left, right in zip(
        splitter.split(rows.y, rows.groups, rows.participants, rows.items),
        transformed.fold_results,
        original.fold_results,
        strict=True,
    ):
        correction = -np.log(rows.raw_y[test]).sum()
        assert right.baseline_ll == pytest.approx(left.baseline_ll + correction)
        assert right.full_ll == pytest.approx(left.full_ll + correction)
        x = rows.columns["unit_position"][:, None]
        predicted = np.exp(OLS().fit_predict(rows.y[train], x[train], x[test]))
        expected_r2 = 1 - np.sum((rows.raw_y[test] - predicted) ** 2) / np.sum(
            (rows.raw_y[test] - rows.raw_y[test].mean()) ** 2
        )
        assert right.full_r2 == pytest.approx(expected_r2)
    with pytest.raises(ValueError, match="inverse.*log_abs_det_jacobian"):
        evaluate(
            dataset, spec, replace(config, transform_dv=np.log, score_scale="original")
        )


def test_projection_policy_rank_and_nested_tuning_are_explicit():
    target, reference = UnitPosition(), UnitLength()
    with pytest.raises(TypeError, match="policy"):
        Residualize(target, against=(reference,))
    projection = Residualize(
        target,
        against=(reference, named(reference, "copy")),
        policy="least_squares",
        intercept=True,
    )
    train, test = (
        np.array([True, True, True, False]),
        np.array([False, False, False, True]),
    )
    x, y = np.arange(4.0), np.array([2.0, 4.0, 8.0, 100.0])
    with pytest.raises(ValueError, match="rank deficient"):
        projection.apply(y, [x, x], train, test)
    left, right, _ = replace(projection, rank="minimum_norm").apply(
        y, [x, x], train, test
    )
    matrix = np.column_stack([np.ones(4), x, x])
    beta = np.linalg.lstsq(matrix[train], y[train], rcond=None)[0]
    np.testing.assert_allclose(left, y[train] - matrix[train] @ beta)
    np.testing.assert_allclose(right, y[test] - matrix[test] @ beta)
    dataset = make_toy_dataset()
    step = Residualize(target, against=(reference,), policy="centered_slope")
    with pytest.raises(ValueError, match="internal folds"):
        evaluate(
            dataset,
            EvaluationSpec(targets=[dataset.feature("spr_rt")], predictors=[target]),
            AnalysisConfig(model=RidgeCVModel(), design=None, preprocessing=(step,)),
        )


def test_target_preprocessing_rejects_a_focal_alias():
    dataset = make_toy_dataset()
    target = dataset.feature("spr_rt")
    predictor = UnitPosition()
    reference = named(predictor, "position_copy")
    spec = EvaluationSpec(targets=[target], predictors=[predictor])
    config = AnalysisConfig(
        model=OLS(),
        design=None,
        preprocessing=(
            Residualize(target, against=(reference,), policy="centered_slope"),
        ),
    )
    with pytest.raises(ValueError, match="including aliases"):
        evaluate(dataset, spec, config)
