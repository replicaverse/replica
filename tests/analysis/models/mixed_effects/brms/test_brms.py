"""Offline contracts for Bayesian scoring, diagnostics and pipeline dispatch."""

import json
from dataclasses import replace
from pathlib import Path
from subprocess import CompletedProcess

import numpy as np
import pandas as pd
import pytest
from scipy.special import logsumexp
from scipy.stats import norm

from replica import BrmsFit, BrmsModel, BrmsPriors
from replica.analysis import (
    AnalysisConfig,
    FullDataSplitter,
    evaluate,
    prepare_observations,
)
from replica.analysis.models.mixed_effects import brms
from replica.analysis.specification.evaluation import EvaluationSpec
from replica.feature import ModelFeature, Surprisal
from tests.analysis._fixtures import VaryingLM, make_toy_dataset


@pytest.fixture
def arrays():
    return (
        np.arange(6.0),
        np.column_stack([np.arange(6.0), [0, 0, 1, 1, 2, 2], np.arange(6)]),
        np.array([[0.0, 0, 0], [1.0, 99, 99]]),
    )


def healthy():
    return dict(
        max_rhat=1.001,
        min_bulk_ess=1000,
        min_tail_ess=1000,
        min_ebfmi=0.9,
        divergences=0,
        treedepth_hits=0,
    )


@pytest.fixture
def transport(monkeypatch):
    calls = []

    def run(script, args, **kwargs):
        assert script.name == "fit_brms.R"
        root = Path(args[0])
        config = json.loads((root / "config.json").read_text())
        train, test = (pd.read_csv(root / f"{name}.csv") for name in ["train", "test"])
        assert "y" in train and "y" not in test
        calls.append((config, train, test, kwargs))
        n = config["chains"] * (config["iterations"] - config["warmup"])
        mu = np.linspace(-1, 1, n)[:, None] + np.zeros((n, len(test)))
        np.savetxt(root / "expected.csv", mu, delimiter=",")
        np.savetxt(root / "sigma.csv", np.ones(n))
        (root / "summary.json").write_text(
            json.dumps(dict(diagnostics=healthy(), coefficients=[]))
        )
        return CompletedProcess(["Rscript"], 0, "stdout", "stderr")

    monkeypatch.setattr(brms, "run_r_script", run)
    return calls


def small_model(**kwargs):
    return BrmsModel(priors=BrmsPriors(), chains=2, iterations=4, warmup=2, **kwargs)


def test_mixture_score_is_log_mean_density_not_mean_log_or_plugin():
    fit = BrmsFit(np.array([[-4.0], [4.0]]), np.ones(2), np.zeros((2, 1)), {}, {}, None)
    expected = np.log((norm.pdf(4, -4, 1) + norm.pdf(4, 4, 1)) / 2)
    assert fit.log_predictive_density([4])[0] == pytest.approx(expected)
    assert abs(expected - norm.logpdf(4, 0, 1)) > 1
    assert abs(expected - np.mean([norm.logpdf(4, -4, 1), norm.logpdf(4, 4, 1)])) > 1
    # Log-domain integration must not underflow to -inf for extreme responses.
    assert np.isfinite(fit.log_predictive_density([1000])).all()


def test_score_alignment_and_predictive_interval():
    fit = BrmsFit(
        np.zeros((100, 2)),
        np.ones(100),
        np.tile(np.arange(100)[:, None], (1, 2)),
        {},
        {},
        None,
    )
    assert fit.interval().shape == (2, 2)
    np.testing.assert_allclose(fit.interval()[:, 0], [2.475, 96.525])
    for y in [
        [1],
        [np.nan, 1],
        [[1, 2]],
        np.array([1 + 2j, 2]),
        np.array([np.complex128(1 + 2j), 2], dtype=object),
        np.ma.array([1, 2], mask=[True, False]),
    ]:
        with pytest.raises(ValueError):
            fit.log_predictive_density(y)
    with pytest.raises(ValueError):
        fit.interval(1)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"chains": 1},
        {"chains": True},
        {"warmup": 2000},
        {"iterations": 1},
        {"cores": 5},
        {"seed": -1},
        {"seed": 2**31},
        {"max_rhat": np.nan},
        {"min_ess": 0},
        {"adapt_delta": 1},
        {"group_effects": "marginal"},
        {"diagnostics_action": "ignore"},
        {"subject_slopes": (0, 0)},
        {"item_slopes": (-1,)},
        {"timeout": 0},
    ],
)
def test_bad_settings_fail_before_r(kwargs):
    with pytest.raises(ValueError):
        BrmsModel(priors=BrmsPriors(), **kwargs)


def test_priors_are_explicit_proper_and_finite():
    with pytest.raises(TypeError):
        BrmsModel()
    with pytest.raises(TypeError):
        BrmsModel(priors={})
    for key in [
        "intercept_scale",
        "coefficient_scale",
        "group_scale",
        "sigma_scale",
        "correlation_shape",
    ]:
        with pytest.raises(ValueError):
            replace(BrmsPriors(), **{key: 0})
    with pytest.raises(ValueError):
        BrmsPriors(intercept_mean=np.inf)


@pytest.mark.parametrize(
    "key,value",
    [
        ("max_rhat", 1.02),
        ("min_bulk_ess", 399),
        ("min_tail_ess", 399),
        ("min_ebfmi", 0.2),
        ("divergences", 1),
        ("treedepth_hits", 1),
        ("max_rhat", None),
        ("min_bulk_ess", np.nan),
    ],
)
def test_diagnostic_gate_detects_failures(key, value):
    assert not brms._diagnostic_issues(healthy(), 1.01, 400)
    assert brms._diagnostic_issues({**healthy(), key: value}, 1.01, 400)


def test_invalid_arrays_and_rank_fail_before_r(arrays, transport):
    y, train, test = arrays
    bad = train.copy()
    bad[0, -1] = 0.5
    for args in [
        (y[:-1], train, test),
        (y, train[:, :1], test),
        (y, bad, test),
        (y, train, test[:, :2]),
        (y * np.nan, train, test),
        (y, train, test[:0]),
    ]:
        with pytest.raises(ValueError):
            small_model().fit_predict(*args)
    for dtype in (complex, object):
        for position in range(3):
            args = list(arrays)
            args[position] = args[position].astype(dtype)
            args[position].flat[0] = np.complex128(args[position].flat[0] + 2j)
            with pytest.raises(ValueError, match="real-valued"):
                small_model().fit_predict(*args)
    with pytest.raises(ValueError, match="rank deficient"):
        small_model().fit_predict(y, np.column_stack([np.ones(6), train[:, -2:]]), test)
    assert not transport


def test_conditional_and_independent_formula_structure():
    model = small_model(subject_slopes=None, item_slopes=(0,), correlated_item=False)
    formula, groups, correlations = model._formula(2)
    assert formula == "y ~ 1 + x0 + x1 + (1 + x0 + x1 | subject) + (1 + x0 || item)"
    assert groups and correlations
    with pytest.raises(ValueError, match="absent"):
        model._formula(0)
    assert small_model(include_subject_re=False, include_item_re=False)._formula(0) == (
        "y ~ 1",
        False,
        False,
    )


def test_held_out_y_never_reaches_r_and_draws_are_reproducible(
    arrays, transport, tmp_path
):
    y, train, test = arrays
    model = small_model(output_dir=tmp_path)
    first = model.fit_score(y, train, test, [1, 2])
    draws = model.last_fit.predictive_draws.copy()
    second = model.fit_score(y, train, test, [8, 9])
    np.testing.assert_array_equal(first.y_pred, second.y_pred)
    np.testing.assert_array_equal(draws, model.last_fit.predictive_draws)
    assert first.loglik != second.loglik
    assert first.loglik == pytest.approx(first.per_obs_loglik.sum())
    assert transport[0][0] == transport[1][0]
    pd.testing.assert_frame_equal(transport[0][1], transport[1][1])
    pd.testing.assert_frame_equal(transport[0][2], transport[1][2])
    assert len(list(tmp_path.glob("fit-*"))) == 2
    assert not hasattr(
        model, "last_fixed_effects"
    )  # posterior SD must not become a Wald SE
    assert not hasattr(model, "fit_loglik")
    for response in [
        [1, np.nan],
        np.array([1 + 2j, 2]),
        np.array([np.complex128(1 + 2j), 2], dtype=object),
        np.ma.array([1, 2], mask=[True, False]),
    ]:
        model.last_fit = object()
        with pytest.raises(ValueError):
            model.fit_score(y, train, test, response)
        assert model.last_fit is None
    assert len(transport) == 2


def test_failure_does_not_return_stale_posterior(arrays, transport, monkeypatch):
    model = small_model()
    model.fit_predict(*arrays)
    assert model.last_fit is not None
    diagnostics = model.last_fit_diagnostics()
    assert diagnostics["diagnostic_checks_passed"] is True
    assert "converged" not in diagnostics
    diagnostics["details"]["sampling"]["passed"] = False
    assert model.last_fit_diagnostics()["details"]["sampling"]["passed"] is True
    monkeypatch.setattr(brms, "_diagnostic_issues", lambda *args: ["divergences=3"])
    with pytest.raises(RuntimeError, match="sampling diagnostics failed"):
        model.fit_predict(*arrays)
    assert model.last_fit is None
    assert model.last_fit_diagnostics() is None
    assert not model.fit_diagnostics[-1]["diagnostics"]["passed"]
    model.diagnostics_action = "warn"
    with pytest.warns(RuntimeWarning, match="divergences=3"):
        model.fit_predict(*arrays)
    assert not model.last_fit.diagnostics["passed"]
    assert model.last_fit_diagnostics()["diagnostic_checks_passed"] is False
    assert model.last_fit_diagnostics()["warnings"] == ["divergences=3"]
    policy = model.fit_diagnostics[-1]["config"]["diagnostic_policy"]
    assert policy["action"] == "warn"
    assert policy["max_rhat"] == model.max_rhat
    assert policy["min_bulk_ess"] == model.min_ess


def test_single_test_row_dimensions(arrays, transport):
    y, train, test = arrays
    model = small_model()
    assert model.fit_predict(y, train, test[:1]).shape == (1,)
    assert model.last_fit.expected_draws.shape == (4, 1)


def test_evaluate_uses_pointwise_posterior_scores(monkeypatch):
    """Each design is scored under its own posterior draws: the fake backend
    shifts every draw by half the number of fixed columns, so the baseline and
    the full model disagree by a known amount per observation."""
    draws = {}

    def run(script, args, **kwargs):
        root = Path(args[0])
        config = json.loads((root / "config.json").read_text())
        test = pd.read_csv(root / "test.csv")
        n_fixed = sum(column.startswith("x") for column in test.columns)
        n = config["chains"] * (config["iterations"] - config["warmup"])
        mu = np.linspace(-1, 1, n)[:, None] + 0.5 * n_fixed + np.zeros((n, len(test)))
        draws[n_fixed] = mu
        np.savetxt(root / "expected.csv", mu, delimiter=",")
        np.savetxt(root / "sigma.csv", np.ones(n))
        (root / "summary.json").write_text(
            json.dumps(dict(diagnostics=healthy(), coefficients=[]))
        )
        return CompletedProcess(["Rscript"], 0, "stdout", "stderr")

    monkeypatch.setattr(brms, "run_r_script", run)
    dataset = make_toy_dataset(n_stimuli=3, n_words=5, n_participants=3)
    spec = EvaluationSpec(
        targets=[dataset.feature("spr_rt")],
        predictors=[ModelFeature(VaryingLM(), Surprisal())],
    )
    model = small_model()
    config = AnalysisConfig(
        design=None,
        transform_dv=np.log2,
        model=model,
        cv=FullDataSplitter(),
        per_fold=True,
        retain_per_obs_delta_ll=True,
    )
    result = evaluate(dataset, spec, config)
    assert len(result) == 1 and sorted(draws) == [0, 1]
    assert [r["is_full"] for r in model.fit_diagnostics] == [False, True]

    y = prepare_observations(dataset, spec, config).y

    def pointwise(mu):
        return logsumexp(norm.logpdf(y[None, :], mu, 1.0), axis=0) - np.log(len(mu))

    fold = result[0].fold_results[0]
    expected = pointwise(draws[1]) - pointwise(draws[0])
    assert fold.baseline_ll == pytest.approx(pointwise(draws[0]).sum(), rel=1e-12)
    assert fold.full_ll == pytest.approx(pointwise(draws[1]).sum(), rel=1e-12)
    np.testing.assert_allclose(fold.per_obs_delta_ll, expected, rtol=1e-12)
    assert abs(fold.delta_ll) > 1.0
    assert np.isnan(result[0].coefficient_se)
