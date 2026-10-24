"""Real Stan validation; run explicitly with -m 'needs_r and slow'."""

import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.special import logsumexp
from scipy.stats import norm

from replica import BrmsModel, BrmsPriors
from replica.analysis import AnalysisConfig, FullDataSplitter, evaluate
from replica.analysis.specification.evaluation import EvaluationSpec
from replica.feature import ModelFeature, Surprisal
from replica.utils.r import run_r_script
from tests.analysis._fixtures import VaryingLM, make_toy_dataset

pytestmark = [pytest.mark.needs_r, pytest.mark.slow, pytest.mark.statistical]


@pytest.fixture(scope="module")
def rscript():
    executable = os.environ.get("REPLICA_BRMS_RSCRIPT") or shutil.which("Rscript")
    if not executable:
        pytest.skip("Rscript is required for live brms validation")
    probe = subprocess.run(
        [
            executable,
            "--vanilla",
            "-e",
            'quit(status=if(all(vapply(c("brms","cmdstanr","posterior","jsonlite"),requireNamespace,logical(1),quietly=TRUE)))0 else 1)',
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    if probe.returncode:
        pytest.skip(
            "brms/cmdstanr/posterior/jsonlite unavailable: " + probe.stderr[-500:]
        )
    return executable


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory):
    root = os.environ.get("REPLICA_BRMS_TEST_OUTPUT")
    if root:
        path = Path(root)
        path.mkdir(parents=True, exist_ok=True)
        return path
    return tmp_path_factory.mktemp("brms-live")


def model(rscript, root, **kwargs):
    return BrmsModel(
        rscript=rscript,
        output_dir=root,
        seed=294,
        chains=4,
        iterations=kwargs.pop("iterations", 2000),
        warmup=kwargs.pop("warmup", 1000),
        **kwargs,
    )


def analytic_reference(y, x, xt, yt, priors):
    """Integrate sigma by quadrature and beta by conjugate normal algebra."""
    design = np.column_stack([np.ones(len(x)), x])
    test = np.column_stack([np.ones(len(xt)), xt])
    prior_mean = np.array([priors.intercept_mean, 0.0])
    prior_var = np.diag([priors.intercept_scale**2, priors.coefficient_scale**2])
    precision = np.linalg.inv(prior_var)
    sigma = np.geomspace(0.015, 8.0, 6000)
    means, covs, logweights = [], [], []
    residual = y - design @ prior_mean
    for s in sigma:
        A = precision + design.T @ design / s**2
        cov = np.linalg.inv(A)
        mean = cov @ (precision @ prior_mean + design.T @ y / s**2)
        v = design.T @ residual / s**2
        logdet = (
            2 * len(y) * np.log(s)
            + np.linalg.slogdet(prior_var)[1]
            + np.linalg.slogdet(A)[1]
        )
        logweights.append(
            -0.5 * (logdet + residual @ residual / s**2 - v @ cov @ v)
            + norm.logpdf(s, scale=priors.sigma_scale)
        )
        means.append(mean)
        covs.append(cov)
    means, covs = np.array(means), np.array(covs)
    widths = np.gradient(sigma)
    widths[[0, -1]] /= 2
    weights = np.asarray(logweights) + np.log(widths)
    weights = np.exp(weights - logsumexp(weights))
    beta = weights @ means
    variance = np.einsum(
        "s,sij->ij", weights, covs + means[:, :, None] * means[:, None, :]
    ) - np.outer(beta, beta)
    mu = means @ test.T
    predvar = sigma[:, None] ** 2 + np.einsum("ni,sij,nj->sn", test, covs, test)
    lp = logsumexp(
        np.log(
            weights[:, None],
            where=weights[:, None] > 0,
            out=np.full((len(weights), 1), -np.inf),
        )
        + norm.logpdf(yt, mu, np.sqrt(predvar)),
        axis=0,
    )
    predictive_mean = weights @ mu
    predictive_sd = np.sqrt(weights @ (predvar + mu**2) - predictive_mean**2)
    return beta, np.sqrt(np.diag(variance)), float(weights @ sigma), lp, predictive_sd


@pytest.fixture(scope="module")
def analytic_fit(rscript, artifacts):
    rng = np.random.default_rng(57)
    x = rng.normal(size=36) * 0.3
    y = 0.8 + 1.1 * x + rng.normal(size=36) * 0.6
    xt = np.array([-0.4, 0.0, 0.2, 0.5])
    yt = np.array([0.5, 0.9, 1.4, 1.0])
    train = np.column_stack([x, np.zeros((len(x), 2))])
    test = np.column_stack([xt, np.zeros((len(xt), 2))])
    priors = BrmsPriors(intercept_scale=1, coefficient_scale=0.3, sigma_scale=1.3)
    fit = model(
        rscript,
        artifacts / "analytic",
        priors=priors,
        include_subject_re=False,
        include_item_re=False,
    )
    score = fit.fit_score(y, train, test, yt)
    return fit, score, analytic_reference(y, x, xt, yt, priors), yt


def test_posterior_matches_independent_quadrature(analytic_fit, artifacts):
    fit, score, reference, _ = analytic_fit
    beta, sd, sigma, lp, predictive_sd = reference
    coefficients = fit.last_fit.coefficients
    observed = np.array([row["mean"] for row in coefficients])
    observed_sd = np.array([row["sd"] for row in coefficients])
    np.testing.assert_allclose(observed, beta, atol=0.025, rtol=0)
    np.testing.assert_allclose(observed_sd, sd, rtol=0.08)
    assert np.mean(fit.last_fit.sigma_draws) == pytest.approx(sigma, abs=0.02)
    np.testing.assert_allclose(score.per_obs_loglik, lp, atol=0.035, rtol=0)
    np.testing.assert_allclose(
        fit.last_fit.predictive_draws.std(axis=0), predictive_sd, rtol=0.08
    )
    assert fit.last_fit.diagnostics["passed"]
    (artifacts / "analytic-comparison.json").write_text(
        json.dumps(
            dict(
                reference_beta=beta.tolist(),
                sampled_beta=observed.tolist(),
                reference_sd=sd.tolist(),
                sampled_sd=observed_sd.tolist(),
                reference_log_density=lp.tolist(),
                sampled_log_density=score.per_obs_loglik.tolist(),
                diagnostics=fit.last_fit.diagnostics,
            ),
            indent=2,
        )
    )


def direct_brms_check(fit, responses, rscript, directory):
    """Use saved brmsfit independently of the Python adapter's prediction/scoring."""
    script = directory / "direct.R"
    script.write_text("""args <- commandArgs(TRUE)
root <- args[1]
suppressMessages(library(brms))
cfg <- jsonlite::read_json(file.path(root,"config.json"),simplifyVector=TRUE)
f <- readRDS(file.path(root,"fit.rds"))
t <- read.csv(file.path(root,"test.csv"),colClasses=c(subject="character",item="character"))
t$y <- scan(args[2],quiet=TRUE)
set.seed(cfg$seed+1L)
mu <- posterior_epred(f,newdata=t,allow_new_levels=cfg$allow_new_levels,sample_new_levels="gaussian",re_formula=if(cfg$group_effects=="population")NA else NULL)
set.seed(cfg$seed+1L)
ll <- log_lik(f,newdata=t,allow_new_levels=cfg$allow_new_levels,sample_new_levels="gaussian",re_formula=if(cfg$group_effects=="population")NA else NULL)
lpd <- apply(ll,2,function(x) {m <- max(x); m+log(mean(exp(x-m)))})
write.table(mu,args[3],sep=",",row.names=FALSE,col.names=FALSE)
write.table(lpd,args[4],row.names=FALSE,col.names=FALSE)
""")
    np.savetxt(directory / "responses.txt", responses)
    run_r_script(
        script,
        [
            fit.last_fit.artifact_dir,
            directory / "responses.txt",
            directory / "direct-means.csv",
            directory / "direct-lpd.txt",
        ],
        executable=rscript,
        timeout=180,
    )
    direct_mean = np.loadtxt(directory / "direct-means.csv", delimiter=",", ndmin=2)
    direct_lpd = np.loadtxt(directory / "direct-lpd.txt", ndmin=1)
    np.testing.assert_allclose(
        fit.last_fit.expected_draws, direct_mean, atol=1e-10, rtol=1e-10
    )
    np.testing.assert_allclose(
        fit.last_fit.log_predictive_density(responses),
        direct_lpd,
        atol=1e-10,
        rtol=1e-10,
    )


def test_gaussian_scores_equal_direct_brms(analytic_fit, rscript, artifacts):
    fit, _, _, responses = analytic_fit
    direct_brms_check(fit, responses, rscript, artifacts)


@pytest.fixture(scope="module", params=[True, False], ids=["correlated", "independent"])
def crossed_fit(request, rscript, artifacts):
    rng = np.random.default_rng(22)
    n, ns, ni = 720, 24, 30
    subjects = np.arange(n) % ns
    items = rng.integers(0, ni, n)
    x = rng.normal(size=n)
    slopes = rng.multivariate_normal([0, 0], [[0.49, 0.06], [0.06, 0.09]], size=ns)
    y = (
        0.5
        + 0.8 * x
        + slopes[subjects, 0]
        + slopes[subjects, 1] * x
        + rng.normal(0, 0.25, ni)[items]
        + rng.normal(0, 0.6, n)
    )
    train = np.column_stack([x, subjects, items])
    test = np.vstack([train[:5], [0.4, 100, 200], [0.4, 100, 200], [0.4, 101, 201]])
    root = artifacts / ("correlated" if request.param else "independent")
    fit = model(
        rscript,
        root,
        priors=BrmsPriors(),
        subject_slopes=None,
        correlated_subject=request.param,
        iterations=6000,
        warmup=2000,
    )
    score = fit.fit_score(y, train, test, np.ones(len(test)))
    return fit, score, root


def test_crossed_slopes_recovery_and_unseen_group_uncertainty(crossed_fit):
    fit, score, _ = crossed_fit
    assert fit.last_fit.diagnostics["passed"]
    coefficients = {row["variable"]: row for row in fit.last_fit.coefficients}
    assert coefficients["b_x0"]["mean"] == pytest.approx(0.8, abs=0.2)
    draws = fit.last_fit.expected_draws
    np.testing.assert_array_equal(draws[:, -3], draws[:, -2])
    assert np.std(draws[:, -3] - draws[:, -1]) > 0.3
    assert draws[:, -1].std() > 0.3
    assert np.isfinite(score.per_obs_loglik).all()
    interval = fit.last_fit.interval()
    assert np.all(interval[0] < interval[1])


def test_crossed_scores_equal_direct_brms(crossed_fit, rscript):
    fit, score, root = crossed_fit
    direct_brms_check(fit, np.ones(len(score.y_pred)), rscript, root)


def test_sleepstudy_real_data(rscript, artifacts):
    root = artifacts / "sleepstudy"
    root.mkdir(exist_ok=True)
    script = root / "data.R"
    script.write_text(
        'data("sleepstudy",package="lme4");write.csv(sleepstudy,commandArgs(TRUE)[1],row.names=FALSE)'
    )
    run_r_script(script, [root / "data.csv"], executable=rscript)
    data = pd.read_csv(root / "data.csv")
    # Hold out two complete subjects. Fixed scaling is explicit in prior units.
    held = data.Subject.isin(data.Subject.unique()[-2:]).to_numpy()
    x = np.column_stack([data.Days.to_numpy() / 10, data.Subject, np.zeros(len(data))])
    y = data.Reaction.to_numpy() / 100
    fit = model(
        rscript,
        root,
        priors=BrmsPriors(intercept_mean=3, intercept_scale=1),
        subject_slopes=None,
        include_item_re=False,
    )
    score = fit.fit_score(y[~held], x[~held], x[held], y[held])
    assert fit.last_fit.diagnostics["passed"]
    assert fit.last_fit.coefficients[1]["mean"] > 0.5
    direct_brms_check(fit, y[held], rscript, root)
    assert score.loglik == pytest.approx(score.per_obs_loglik.sum())


def test_actual_sampler_through_evaluate(rscript, artifacts):
    fit = model(
        rscript,
        artifacts / "pipeline",
        priors=BrmsPriors(intercept_mean=8, intercept_scale=2),
        include_subject_re=False,
        include_item_re=False,
    )
    dataset = make_toy_dataset(n_stimuli=5, n_words=5, n_participants=2)
    result = evaluate(
        dataset,
        EvaluationSpec(
            targets=[dataset.feature("spr_rt")],
            predictors=[ModelFeature(VaryingLM(), Surprisal())],
        ),
        AnalysisConfig(
            design=None,
            transform_dv=np.log2,
            model=fit,
            cv=FullDataSplitter(),
            per_fold=True,
            retain_per_obs_delta_ll=True,
        ),
    )
    assert len(result) == 1 and len(fit.fit_diagnostics) == 2
    assert all(row["diagnostics"]["passed"] for row in fit.fit_diagnostics)
    fold = result[0].fold_results[0]
    assert fold.delta_ll == pytest.approx(fold.per_obs_delta_ll.sum())
    assert np.isnan(result[0].coefficient_se)
