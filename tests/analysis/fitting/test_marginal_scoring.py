"""Unseen-group predictive variance and the score used by named evaluation."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import norm

from replica.analysis import (
    AnalysisConfig,
    Design,
    Factor,
    RandomEffect,
    RLmerModel,
    evaluate,
)
from replica.analysis.contracts.grouped import GroupedFit, GroupedFrame
from replica.analysis.fitting.scoring import (
    _gaussian_log_likelihood,
    _gaussian_log_likelihood_pointwise,
)
from replica.analysis.models.mixed_effects.grouped import _marginal_variance
from replica.analysis.specification.evaluation import EvaluationSpec
from replica.feature import ModelFeature, Surprisal
from tests.analysis._fixtures import ConstantLM, make_toy_dataset


def test_marginal_variance_matches_gaussian_integration_and_correlations():
    design = Design(
        factors=(Factor("participant", "participant"), Factor("stimulus", "stimulus")),
        comparison="predictor",
        random_effects=(),
    )
    frame = GroupedFrame(
        {"x": np.array([-2.0, 0.0, 3.0])},
        {"participant": np.array(["a", "b", "c"]), "stimulus": np.array(["s"] * 3)},
        ("x",),
        design,
    )
    covariance = np.array([[0.64, -0.12], [-0.12, 0.25]])
    blocks = [
        {
            "factor": "participant",
            "columns": ["(Intercept)", "x"],
            "covariance": covariance.tolist(),
        }
    ]
    variance = _marginal_variance(0.7, blocks, frame)
    x = frame.columns["x"]
    expected = 0.7**2 + 0.64 - 0.24 * x + 0.25 * x**2
    np.testing.assert_allclose(variance, expected)
    for v, y in zip(variance, [-3.0, 0.0, 2.0], strict=True):
        density = quad(
            lambda b, y=y, v=v: (
                norm.pdf(y, 0.4 + b, 0.7) * norm.pdf(b, 0, np.sqrt(v - 0.7**2))
            ),
            -np.inf,
            np.inf,
        )[0]
        assert norm.pdf(y, 0.4, np.sqrt(v)) == pytest.approx(density, abs=1e-10)


@pytest.mark.needs_r
@pytest.mark.parametrize("correlated", [True, False])
def test_lmer_exports_complete_marginal_variance(correlated):
    from replica.analysis.models.support.r import _r_context

    rng = np.random.default_rng(8305)
    reader, source, word = np.indices((18, 10, 8)).reshape(3, -1)
    x = rng.normal(size=len(reader))
    re = rng.multivariate_normal([0, 0], [[0.64, 0.12], [0.12, 0.16]], 18)
    y = (
        4
        + 0.6 * x
        + re[reader, 0]
        + re[reader, 1] * x
        + rng.normal(0, 0.3, 10)[source]
        + rng.normal(0, 0.7, len(x))
    )
    design = Design(
        factors=(Factor("participant", "participant"), Factor("stimulus", "stimulus")),
        comparison="predictor",
        random_effects=(
            RandomEffect("participant", ("x",), correlated=correlated),
            RandomEffect("stimulus"),
        ),
    )
    masks = ((reader < 13) & (source < 7), (reader >= 13) & (source >= 7))
    frames = [
        GroupedFrame(
            {"x": x[m]},
            {"participant": reader[m].astype(str), "stimulus": source[m].astype(str)},
            ("x",),
            design,
        )
        for m in masks
    ]
    model = RLmerModel(nlminb_fallback=False)
    fit = model.fit_grouped(y[masks[0]], *frames)
    with _r_context() as ro:
        # Reconstruct the covariance from fitted block entries in R.
        expected = np.asarray(
            ro.r("""
            local({
              v <- rep(sigma(.lmer_model)^2, nrow(.test_data))
              for (G in lme4::VarCorr(.lmer_model)) {
                cn <- colnames(G)
                if (length(cn)==2L) {
                  x <- .test_data$x0
                  v <- v + G[1,1] + 2*x*G[1,2] + x^2*G[2,2]
                } else {
                  z <- if (cn=='(Intercept)') 1 else .test_data$x0
                  v <- v + G[1,1]*z^2
                }
              }
              v
            })
        """),
            float,
        )
    np.testing.assert_allclose(fit.predictive_variance, expected, rtol=1e-12)
    assert fit.metadata["scoring"] == "marginal_gaussian"
    assert {b["factor"] for b in fit.metadata["random_covariances"]} == {
        "participant",
        "stimulus",
    }
    # Held-out labels are used only to enforce exclusion, never to change a fit.
    with pytest.raises(ValueError, match="entirely unseen"):
        model.fit_grouped(y[masks[0]], frames[0], frames[0])
    residual_frames = [
        replace(f, design=replace(design, scoring="residual")) for f in frames
    ]
    residual_fit = model.fit_grouped(y[masks[0]], *residual_frames)
    np.testing.assert_allclose(residual_fit.predictions, fit.predictions, atol=1e-10)
    np.testing.assert_allclose(residual_fit.beta, fit.beta, atol=1e-10)
    assert residual_fit.sigma == pytest.approx(fit.sigma, abs=1e-10)
    baseline_frames = [GroupedFrame(f.columns, f.factors, (), design) for f in frames]
    baseline = model.fit_grouped(y[masks[0]], *baseline_frames)
    assert all(
        b["columns"] == ["(Intercept)"] for b in baseline.metadata["random_covariances"]
    )
    np.testing.assert_allclose(
        baseline.predictive_variance, baseline.predictive_variance[0]
    )


def test_pointwise_gaussian_scales_are_used_and_validated():
    y, mu, sigma = np.array([1.0, 4.0]), np.array([2.0, 2.0]), np.array([0.5, 2.0])
    expected = norm.logpdf(y, mu, sigma)
    np.testing.assert_allclose(
        _gaussian_log_likelihood_pointwise(y, mu, sigma), expected
    )
    assert _gaussian_log_likelihood(y, mu, sigma) == pytest.approx(expected.sum())
    for invalid in [np.array([1.0]), np.array([0.0, 1.0]), np.array([np.nan, 1.0])]:
        with pytest.raises(ValueError, match="positive.*aligned"):
            _gaussian_log_likelihood(y, mu, invalid)
    for position in range(3):
        for dtype in (complex, object):
            values = [y, mu, sigma]
            values[position] = values[position].astype(dtype)
            values[position][0] = np.complex128(values[position][0] + 2j)
            with pytest.raises(ValueError, match="real-valued"):
                _gaussian_log_likelihood(*values)
        values = [y, mu, sigma]
        values[position] = np.ma.array(values[position], mask=[True, False])
        with pytest.raises(ValueError, match="finite"):
            _gaussian_log_likelihood(*values)
    fit = GroupedFit(mu, 1.0, np.zeros(1), np.ones(1))
    with pytest.raises(ValueError, match="backend predictive variance"):
        fit.scoring_sigma("marginal")
    with pytest.raises(ValueError, match="positive and aligned"):
        replace(fit, predictive_variance=np.ones(3))


def test_marginal_scoring_cannot_estimate_variance_on_test_responses():
    dataset = make_toy_dataset()
    lm = ConstantLM()
    with pytest.raises(ValueError, match="sigma_source='training'"):
        evaluate(
            dataset,
            EvaluationSpec(
                targets=[dataset.feature(dataset.default_measure)],
                predictors=[ModelFeature(lm, Surprisal())],
            ),
            AnalysisConfig(
                model=RLmerModel(),
                design=Design(),
                sigma_source="test",
            ),
        )
    assert lm.call_count == 0


def test_named_evaluation_uses_predictive_variance_in_both_models():
    class Backend:
        def validate_design(self, design):
            pass

        def fit_grouped(self, y_train, train, test):
            n = len(test.factors["participant"])
            # Identical means and residual scales isolate the score variance.
            return GroupedFit(
                np.full(n, 500.0),
                2.0,
                np.zeros(len(train.fixed) + 1),
                np.ones(len(train.fixed) + 1),
                predictive_variance=np.full(n, 400.0 if train.fixed else 100.0),
            )

    ds = make_toy_dataset(n_participants=4)
    feature = EvaluationSpec(
        targets=[ds.feature(ds.default_measure)],
        predictors=[ModelFeature(ConstantLM(), Surprisal())],
    )
    config = AnalysisConfig(
        model=Backend(),
        design=Design(
            factors=(
                Factor("participant", "participant"),
                Factor("stimulus", "stimulus"),
            ),
            comparison="predictor",
            random_effects=(),
        ),
        n_folds=2,
        per_fold=True,
        retain_per_obs_delta_ll=True,
        transform_dv=lambda y: y,
    )
    marginal = evaluate(ds, feature, config)[0]
    residual = evaluate(
        ds, feature, replace(config, design=replace(config.design, scoring="residual"))
    )[0]
    assert residual.delta_ll == pytest.approx(0)
    assert marginal.delta_ll != pytest.approx(0)
    for fold in marginal.fold_results:
        # Score identity for N(500,20²) versus N(500,10²) independently follows
        # from baseline log scores and the variance ratio, without test fitting.
        squared_error_sum = -200 * (
            fold.baseline_ll + 0.5 * fold.n_test * np.log(2 * np.pi * 100)
        )
        expected = (
            -0.5 * fold.n_test * np.log(4)
            + 0.5 * (1 / 100 - 1 / 400) * squared_error_sum
        )
        assert fold.delta_ll == pytest.approx(expected)
        assert fold.per_obs_delta_ll.sum() == pytest.approx(expected)
