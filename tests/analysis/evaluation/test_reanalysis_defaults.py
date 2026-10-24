"""Recommended configuration, shared acceptance, and retained reporting contracts."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from replica.analysis import (
    AnalysisConfig,
    Design,
    Factor,
    RandomEffect,
    evaluate,
    predictive_test,
    prepare_observations,
)
from replica.analysis.inference.crossed_bootstrap import crossed_bootstrap_test
from replica.analysis.specification.evaluation import EvaluationSpec
from replica.feature import ModelFeature, UnitPosition
from tests.analysis._fixtures import make_toy_dataset
from tests.analysis.inference.test_crossed_bootstrap import (
    SMALL,
    GaussianBackend,
    fixture,
)


def declaration():
    return Design(
        factors=(Factor("reader", "participant"), Factor("source", "stimulus")),
        random_effects=(
            RandomEffect("reader", ("unit_position",)),
            RandomEffect("source"),
        ),
        cv_unit=("reader", "source"),
        comparison="predictor",
    )


def configured(backend=None):
    config = AnalysisConfig(design=declaration(), transform_dv=None, n_folds=2)
    return replace(config, model=backend or GaussianBackend())


def inputs():
    dataset = make_toy_dataset(n_participants=4, n_stimuli=4)
    return (
        dataset,
        EvaluationSpec(
            targets=[dataset.feature(dataset.default_measure)],
            predictors=[UnitPosition()],
        ),
    )


def test_explicit_random_structure_preserves_declared_covariance():
    config = AnalysisConfig(design=declaration(), transform_dv=None)
    assert config.model.reml is False
    assert config.design.random_effects == declaration().random_effects
    assert config.design.random_effects[0].correlated is True
    assert config.design.comparison == "predictor"
    assert config.design.unseen == "error"
    assert config.design.condition_on == ()
    assert config.design.scoring == "marginal"
    assert config.resolve_cv().seed == 42
    assert config.resolve_cv().train_fraction == (0.9,)
    assert config.sigma_source == "training"
    np.testing.assert_equal(config.transform_dv(np.array([0, 300])), [0, 300])
    custom = replace(
        declaration(),
        random_effects=(RandomEffect("reader", ("unit_position",), correlated=False),),
    )
    assert (
        not AnalysisConfig(design=custom, transform_dv=np.log)
        .design.random_effects[0]
        .correlated
    )


@pytest.mark.parametrize(
    "change",
    [
        {"cv_unit": "source"},
        {"condition_on": ("reader",), "cv_unit": "source", "scoring": "residual"},
        {"scoring": "residual"},
        {"comparison": "fixed_effect"},
    ],
)
def test_predictive_inference_rejects_unsupported_targets(change):
    ds, features = inputs()
    config = AnalysisConfig(
        design=replace(declaration(), **change), model=GaussianBackend()
    )
    with pytest.raises(ValueError):
        predictive_test(ds, features, config, inference=SMALL, checkpoint_dir=None)


class DiagnosticBackend(GaussianBackend):
    def __init__(self, diagnostics):
        self.diagnostics = diagnostics

    def fit_grouped(self, y, train, test):
        fit = super().fit_grouped(y, train, test)
        if self.diagnostics == "rank":
            return replace(fit, beta=np.full_like(fit.beta, np.nan))
        return replace(fit, metadata=self.diagnostics)


@pytest.mark.parametrize(
    "diagnostics",
    [
        {"optimizer_convergence": [1]},
        {"convergence_messages": ["failed to converge"]},
        "rank",
    ],
)
def test_evaluation_and_inference_agree_on_rejected_fits(diagnostics):
    ds, features = inputs()
    config = configured(DiagnosticBackend(diagnostics))
    with pytest.raises(ValueError, match="converg|dropped fixed"):
        evaluate(ds, features, config)
    result = predictive_test(
        ds, features, config, inference=SMALL, checkpoint_dir=None
    )[0]
    assert not result.available
    assert result.metadata["failures"]


def test_singular_diagnostics_and_correct_score_survive_without_fold_results():
    ds, features = inputs()
    config = configured(
        DiagnosticBackend(
            {
                "singular": True,
                "optimizer_convergence": [0],
                "convergence_messages": ["boundary (singular) fit"],
                "reml": False,
            }
        )
    )
    result = evaluate(ds, features, config)[0]
    assert result.fold_results is None
    assert result.metadata["scoring"] == "marginal_gaussian"
    assert result.metadata["cv_seed"] == 42
    assert result.metadata["cv_cycles"] == 1
    assert len(result.metadata["fit_diagnostics"]) == 4
    assert all(
        f["baseline"]["singular"] and f["full"]["singular"]
        for f in result.metadata["fit_diagnostics"]
    )
    assert result.metadata["configuration"]["response_transform"] == "identity"
    assert result.metadata["configuration"]["predictive_significance"] == "not_assessed"
    assert (
        result.coefficient_mean_within_fit_se
        == result.coefficients["unit_position"].mean_within_fit_se
    )
    assert (
        result.metadata["coefficient_uncertainty"] == "mean_within_fit_standard_error"
    )
    inference = predictive_test(
        ds, features, config, inference=SMALL, checkpoint_dir=None
    )[0]
    assert all(
        r["error"] is None
        for r in inference.metadata["cycles"]
        if r["series"] == "identity"
    )
    assert inference.metadata["singular_fits"] > 0
    assert all(
        "No training rectangle" in r["error"] for r in inference.metadata["failures"]
    )


def test_reporting_units_survive_evaluation_and_observation_preparation():
    from replica.analysis import OLS
    from replica.feature import (
        FeatureModelOutput,
        InUnit,
        PrecomputedFeatureModel,
        Surprisal,
    )

    ds, _ = inputs()
    stored = {
        stimulus.stimulus_id: [
            FeatureModelOutput(values={"log_p_observed": -(i + 1) * np.log(2)})
            for i in range(len(stimulus.layers["words"]))
        ]
        for stimulus in ds.stimuli()
    }
    features = EvaluationSpec(
        targets=[ds.feature(ds.default_measure)],
        predictors=[
            ModelFeature(PrecomputedFeatureModel("stored", stored), Surprisal()),
            InUnit(
                ModelFeature(PrecomputedFeatureModel("stored", stored), Surprisal()),
                "nats",
            ),
        ],
        comparison="individual",
    )
    config = AnalysisConfig(model=OLS(), design=None, n_folds=2)
    bits, nats = evaluate(ds, features, config)
    assert bits.delta_ll == pytest.approx(nats.delta_ll, abs=1e-10)
    for result, feature, unit in zip(
        (bits, nats), features.predictors, ("bits", "nats"), strict=True
    ):
        declared = result.metadata["feature_measurements"][0]
        assert declared["name"] == feature.name
        assert declared["measurement"]["unit"] == unit
        observations = prepare_observations(
            ds, replace(features, predictors=[feature], comparison="joint"), config
        )
        assert observations.response["feature_measurements"] == [declared]


def test_row_exclusions_reconcile_and_explicit_first_unit_policy():
    ds, features = inputs()
    config = configured()
    obs = prepare_observations(ds, features, config)
    attrition = obs.response["row_exclusions"]
    assert attrition == {
        "input_units_after_loader": 80,
        "response_or_dataset_mask": 16,
        "missing_features": 0,
        "outlier_filter": 0,
        "nonfinite_transformed_response": 0,
        "retained": 64,
    }
    ds.exclude_first_unit = False
    assert len(prepare_observations(ds, features, config).y) == 80
    ds._data[0].values[0] = 0
    logged = prepare_observations(ds, features, replace(config, transform_dv=np.log))
    assert len(logged.y) == 79
    assert logged.response["row_exclusions"]["nonfinite_transformed_response"] == 1
    result = evaluate(ds, features, replace(config, transform_dv=np.log))[0]
    assert result.metadata["row_exclusions"] == logged.response["row_exclusions"]


def test_default_checkpoint_is_unique_logged_and_can_resume(
    tmp_path, monkeypatch, caplog
):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    data, comparison, config = fixture()
    with caplog.at_level("INFO", logger="replica"):
        result = crossed_bootstrap_test(data, comparison, config, inference=SMALL)
    directory = Path(result.metadata["checkpoint_dir"])
    assert directory.is_relative_to(tmp_path)
    assert (directory / "manifest.json").is_file()
    assert str(directory) in caplog.text
    GaussianBackend.calls.clear()
    resumed = crossed_bootstrap_test(
        data, comparison, config, inference=SMALL, checkpoint_dir=directory
    )
    assert not GaussianBackend.calls
    assert result.metadata["cycles"] == resumed.metadata["cycles"]
    again = crossed_bootstrap_test(data, comparison, config, inference=SMALL)
    assert again.metadata["checkpoint_dir"] != str(directory)
    assert again.metadata["cycles"] == result.metadata["cycles"]


def test_default_configuration_resolves_focal_slopes_and_raw_responses():
    from replica.analysis import CrossedGroupSplitter, RLmerModel

    config = AnalysisConfig()
    assert isinstance(config.model, RLmerModel)
    assert not config.model.reml
    assert isinstance(config.resolve_cv(), CrossedGroupSplitter)
    assert config.resolve_cv().train_fraction == (0.9,)
    assert config.resolve_cv().seed == 42
    np.testing.assert_equal(config.transform_dv(np.array([0, 300])), [0, 300])
    resolved = config.resolve(("focal", "focal_lag1"))
    assert resolved.design.random_effects == (
        RandomEffect("participant", ("focal", "focal_lag1"), correlated=False),
        RandomEffect("item"),
    )
    with pytest.warns(UserWarning, match="Review baseline and target"):
        baseline = resolved.design.for_model(("control",))
    assert baseline.random_effects == (
        RandomEffect("participant", correlated=False),
        RandomEffect("item"),
    )
    assert config.design.random_effects is None


def test_bare_and_explicit_defaults_share_evaluation_and_inference(monkeypatch):
    import importlib

    from replica.analysis import RLmerModel

    backend = GaussianBackend()
    backend.calls.clear()
    monkeypatch.setattr(
        RLmerModel,
        "fit_grouped",
        lambda self, y, train, test: backend.fit_grouped(y, train, test),
    )
    bootstrap = importlib.import_module("replica.analysis.inference.crossed_bootstrap")
    monkeypatch.setattr(bootstrap, "_runtime", lambda model: {})
    ds, features = inputs()
    implicit = evaluate(ds, features)[0]
    explicit = evaluate(ds, features, AnalysisConfig())[0]
    assert implicit.metrics == explicit.metrics
    assert implicit.metadata["configuration"] == explicit.metadata["configuration"]
    assert implicit.metadata["configuration"]["response_transform"] == "identity"
    assert implicit.metadata["scoring"] == "marginal_gaussian"
    implicit_test = predictive_test(ds, features, inference=SMALL, checkpoint_dir=None)[
        0
    ]
    explicit_test = predictive_test(
        ds, features, AnalysisConfig(), inference=SMALL, checkpoint_dir=None
    )[0]
    assert implicit_test.gain == explicit_test.gain
    assert implicit_test.metadata["cycles"] == explicit_test.metadata["cycles"]
    assert (
        implicit_test.metadata["configuration"]
        == explicit_test.metadata["configuration"]
    )
    for _, train, test in backend.calls:
        for axis in train.design.cv_units:
            assert not set(train.factors[axis]) & set(test.factors[axis])
        expected = ("unit_position",) if "unit_position" in train.fixed else ()
        assert train.design.random_effects[0].slopes == expected


def test_default_occurrences_are_shared_across_readers_and_distinct_across_positions():
    import json

    ds, features = inputs()
    observations = prepare_observations(ds, features)
    expected = [
        json.dumps([str(source), "words", int(index)])
        for source, index in zip(
            observations.stimulus_ids, observations.unit_indices, strict=True
        )
    ]
    np.testing.assert_array_equal(observations.factors["item"], expected)
    _, counts = np.unique(observations.factors["item"], return_counts=True)
    np.testing.assert_array_equal(counts, 4)


def test_automatic_public_checkpoint_root_resumes_every_response(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    ds = make_toy_dataset(n_participants=6, n_stimuli=6, measures=["spr_rt", "other"])
    features = EvaluationSpec(
        targets=[ds.feature("spr_rt"), ds.feature("other")], predictors=[UnitPosition()]
    )
    results = predictive_test(ds, features, configured(), inference=SMALL)
    root = results[0].metadata["checkpoint_root"]
    assert results[1].metadata["checkpoint_root"] == root
    assert len(list(Path(root).glob("*/manifest.json"))) == 2
    GaussianBackend.calls.clear()
    resumed = predictive_test(
        ds, features, configured(), inference=SMALL, checkpoint_dir=root
    )
    assert not GaussianBackend.calls
    assert [r.metadata["cycles"] for r in resumed] == [
        r.metadata["cycles"] for r in results
    ]


def test_configuration_preserves_the_warning_for_an_unreviewed_baseline():
    config = AnalysisConfig(
        design=replace(declaration(), comparison=None), transform_dv=None
    )
    with pytest.warns(UserWarning, match="Review baseline and target"):
        baseline = config.design.for_model(())
    assert not any(effect.slopes for effect in baseline.random_effects)


@pytest.mark.needs_r
def test_default_lmm_scores_both_models_with_unseen_group_variance(monkeypatch):
    from scipy.stats import norm

    from replica.analysis import RLmerModel

    ds = make_toy_dataset(n_participants=10, n_stimuli=10, n_words=8)
    features = EvaluationSpec(
        targets=[ds.feature(ds.default_measure)], predictors=[UnitPosition()]
    )
    config = AnalysisConfig(n_folds=2, per_fold=True, retain_per_obs_delta_ll=True)
    observations = prepare_observations(ds, features, config)
    responses = {
        (str(p), str(s), str(i)): y
        for p, s, i, y in zip(
            observations.factors["participant"],
            observations.stimulus_ids,
            observations.unit_indices,
            observations.y,
            strict=True,
        )
    }
    fits = []
    original = RLmerModel.fit_grouped

    def capture(self, y, train, test):
        fit = original(self, y, train, test)
        fits.append((fit, train, test))
        return fit

    monkeypatch.setattr(RLmerModel, "fit_grouped", capture)
    result = evaluate(ds, features, config)[0]
    assert result.n_scored == len(observations.y)
    assert len(result.fold_results) == 4
    for fold, (baseline, _, _), (full, train, test) in zip(
        result.fold_results, fits[::2], fits[1::2], strict=True
    ):
        assert baseline.metadata["formula"] == "y ~ 1 + (1 | g0) + (1 | g2)"
        assert (
            full.metadata["formula"]
            == "y ~ 1 + x0 + (1 | g0) + (0 + x0 | g0) + (1 | g2)"
        )
        assert not full.metadata["reml"]
        for factor in ("participant", "stimulus", "item"):
            assert not set(train.factors[factor]) & set(test.factors[factor])
        y = np.array([responses[tuple(map(str, row))] for row in fold.row_ids])
        scores = []
        for fit in (baseline, full):
            variance = np.full(len(y), fit.sigma**2)
            for block in fit.metadata["random_covariances"]:
                z = np.column_stack(
                    [
                        np.ones(len(y)) if name == "(Intercept)" else test.columns[name]
                        for name in block["columns"]
                    ]
                )
                variance += np.einsum("ij,jk,ik->i", z, block["covariance"], z)
            np.testing.assert_allclose(fit.predictive_variance, variance)
            scores.append(norm.logpdf(y, fit.predictions, np.sqrt(variance)))
        np.testing.assert_allclose(
            fold.per_obs_delta_ll, scores[1] - scores[0], rtol=1e-10, atol=1e-12
        )
