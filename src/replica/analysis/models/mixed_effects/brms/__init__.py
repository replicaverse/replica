"""Optional Bayesian Gaussian mixed models through an isolated brms R process.

Scores are sums of pointwise posterior predictive log densities.
"""

import json
import subprocess
import tempfile
import warnings
from dataclasses import asdict, dataclass
from importlib.resources import files
from pathlib import Path

import numpy as np

from replica.analysis.contracts.models import FoldScore
from replica.utils.arrays import real_array_with_missing
from replica.utils.r import RScriptError, run_r_script


def _positive(name, value):
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


def _integer(name, value, minimum):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


@dataclass(frozen=True)
class BrmsPriors:
    """Proper Gaussian-model priors, in the units of the supplied design.

    Args:
        intercept_mean: Mean of the uncentered intercept's normal prior.
        intercept_scale: Standard deviation of that normal prior.
        coefficient_scale: Standard deviation of zero-mean normal slope priors.
        group_scale: Scale of half-normal group-level standard-deviation priors.
        sigma_scale: Scale of the half-normal residual-standard-deviation prior.
        correlation_shape: LKJ shape for correlated group-level effects.
    """

    intercept_mean: float = 0.0
    intercept_scale: float = 5.0
    coefficient_scale: float = 2.5
    group_scale: float = 1.0
    sigma_scale: float = 1.0
    correlation_shape: float = 2.0

    def __post_init__(self):
        if not np.isfinite(self.intercept_mean):
            raise ValueError("intercept_mean must be finite")
        for name, value in asdict(self).items():
            if name != "intercept_mean":
                _positive(name, value)


@dataclass(frozen=True)
class BrmsFit:
    """Posterior output for the most recent fit; SDs are not frequentist SEs.

    Args:
        expected_draws: Draws of conditional means, shape (draws, test rows).
        sigma_draws: Residual-scale draws paired with expected_draws.
        predictive_draws: Response draws including Gaussian observation noise.
        coefficients: R posterior summaries, including means, SDs and quantiles.
        diagnostics: Sampler diagnostics and the acceptance decision.
        artifact_dir: Retained inputs, R fit and outputs, or None for temporary fits.
    """

    expected_draws: np.ndarray
    sigma_draws: np.ndarray
    predictive_draws: np.ndarray
    coefficients: list[dict]
    diagnostics: dict
    artifact_dir: Path | None

    @property
    def mean(self) -> np.ndarray:
        """Posterior expected response for each test row."""
        return self.expected_draws.mean(axis=0)

    def interval(self, probability: float = 0.95) -> np.ndarray:
        """Equal-tailed posterior predictive interval, shape (2, test rows).

        Args:
            probability: Probability mass to enclose, strictly between zero and one.
        """
        if not 0 < probability < 1:
            raise ValueError("probability must lie strictly between 0 and 1")
        tail = (1 - probability) / 2
        return np.quantile(self.predictive_draws, [tail, 1 - tail], axis=0)

    def log_predictive_density(self, y: np.ndarray) -> np.ndarray:
        """Integrate each row's Gaussian density over posterior draws.

        Args:
            y: Finite held-out responses in prediction-row order.
        """
        from scipy.special import logsumexp

        y = real_array_with_missing(y, name="y").astype(float, copy=False)
        if y.shape != (self.expected_draws.shape[1],) or not np.isfinite(y).all():
            raise ValueError("y must be a finite vector aligned with prediction rows")
        sigma = self.sigma_draws[:, None]
        logp = (
            -0.5 * np.log(2 * np.pi)
            - np.log(sigma)
            - 0.5 * ((y - self.expected_draws) / sigma) ** 2
        )
        return logsumexp(logp, axis=0) - np.log(len(sigma))


def _diagnostic_issues(diagnostics, max_rhat, min_ess):
    issues = []
    for key, limit, lower in [
        ("max_rhat", max_rhat, False),
        ("min_bulk_ess", min_ess, True),
        ("min_tail_ess", min_ess, True),
        ("min_ebfmi", 0.3, True),
    ]:
        value = diagnostics.get(key)
        if value is None or not np.isfinite(value):
            issues.append(f"{key} is unavailable/non-finite")
        elif (value < limit) if lower else (value > limit):
            issues.append(f"{key}={value:.5g} fails threshold {limit}")
    for key in ["divergences", "treedepth_hits"]:
        if diagnostics.get(key) != 0:
            issues.append(f"{key}={diagnostics.get(key)} (requires zero)")
    return issues


class BrmsModel:
    """Bayesian Gaussian regression with optional subject/item random effects.

    Requires R with brms, cmdstanr, posterior, jsonlite and compiled CmdStan.
    No R/Python bridge is imported. Fit and prediction run in a fresh subprocess.
    Input matrices end in two integer grouping columns, as for RLmerModel.
    Priors must be supplied explicitly; no scaling is performed here.

    Args:
        priors: Proper prior specification in the supplied response/predictor units.
        subject_slopes: Fixed-column indices with subject slopes; None selects all.
        item_slopes: Fixed-column indices with item slopes; None selects all.
        include_subject_re: Include the subject random intercept and selected slopes.
        include_item_re: Include the item random intercept and selected slopes.
        correlated_subject: Correlate the subject random intercept and slopes.
        correlated_item: Correlate the item random intercept and slopes.
        group_effects: 'conditional' uses learned effects and draws new levels from
            the fitted Gaussian group distribution; 'population' sets all to zero.
        allow_new_levels: Permit grouping levels absent from training.
        chains: Independent MCMC chains, at least two.
        iterations: Total iterations per chain, including warmup.
        warmup: Discarded adaptation iterations per chain.
        cores: Concurrent chains, default one; bounded by chains.
        seed: Reproducible fitting and prediction seed, below 2**31 - 2.
        adapt_delta: Target NUTS acceptance probability.
        max_treedepth: Maximum NUTS tree depth; any hits fail the diagnostic gate.
        max_rhat: Largest acceptable rank-normalized R-hat.
        min_ess: Minimum bulk and tail effective sample sizes across parameters.
        diagnostics_action: 'raise' (default) or an explicit 'warn' override.
        rscript: Rscript executable, optionally from an isolated environment.
        output_dir: Optional parent directory for unique retained fit artifacts.
        timeout: Subprocess time limit in seconds, including compilation.
    """

    name = "brms"
    needs_grouping = True
    owns_loglik = True

    def configuration(self) -> dict:
        """Return declared options without fitted state."""
        return {
            "priors": asdict(self.priors),
            "subject_slopes": self.subject_slopes,
            "item_slopes": self.item_slopes,
            "include_subject_re": self.include_subject_re,
            "include_item_re": self.include_item_re,
            "correlated_subject": self.correlated_subject,
            "correlated_item": self.correlated_item,
            "max_rhat": self.max_rhat,
            "min_ess": self.min_ess,
            "diagnostics_action": self.diagnostics_action,
            "rscript": self.rscript,
            "timeout": self.timeout,
            "settings": self.settings,
        }

    def __init__(
        self,
        *,
        priors: BrmsPriors,
        subject_slopes: tuple[int, ...] | None = (),
        item_slopes: tuple[int, ...] | None = (),
        include_subject_re: bool = True,
        include_item_re: bool = True,
        correlated_subject: bool = True,
        correlated_item: bool = True,
        group_effects: str = "conditional",
        allow_new_levels: bool = True,
        chains: int = 4,
        iterations: int = 2000,
        warmup: int = 1000,
        cores: int = 1,
        seed: int = 1,
        adapt_delta: float = 0.95,
        max_treedepth: int = 12,
        max_rhat: float = 1.01,
        min_ess: float = 400,
        diagnostics_action: str = "raise",
        rscript: str = "Rscript",
        output_dir: str | Path | None = None,
        timeout: float = 3600,
    ):
        """Configure explicit priors, grouping structure and sampling controls."""
        if not isinstance(priors, BrmsPriors):
            raise TypeError("priors must be a BrmsPriors instance")
        for name, value, minimum in [
            ("chains", chains, 2),
            ("iterations", iterations, 4),
            ("warmup", warmup, 1),
            ("cores", cores, 1),
            ("seed", seed, 0),
            ("max_treedepth", max_treedepth, 1),
        ]:
            _integer(name, value, minimum)
        if warmup >= iterations or cores > chains or seed >= 2**31 - 2:
            raise ValueError(
                "require warmup < iterations, cores <= chains, seed < 2**31-2"
            )
        if not 0 < adapt_delta < 1 or not np.isfinite(max_rhat) or max_rhat <= 1:
            raise ValueError("require 0 < adapt_delta < 1 and finite max_rhat > 1")
        _positive("min_ess", min_ess)
        _positive("timeout", timeout)
        for name, value, allowed in [
            ("group_effects", group_effects, ("conditional", "population")),
            ("diagnostics_action", diagnostics_action, ("raise", "warn")),
        ]:
            if value not in allowed:
                raise ValueError(f"{name} must be one of {allowed}")
        self.priors = priors
        self.subject_slopes = self._slopes(subject_slopes)
        self.item_slopes = self._slopes(item_slopes)
        self.include_subject_re = include_subject_re
        self.include_item_re = include_item_re
        self.correlated_subject = correlated_subject
        self.correlated_item = correlated_item
        self.max_rhat, self.min_ess = max_rhat, min_ess
        self.diagnostics_action = diagnostics_action
        self.rscript, self.timeout = rscript, timeout
        self.output_dir = None if output_dir is None else Path(output_dir)
        self.settings = dict(
            group_effects=group_effects,
            allow_new_levels=allow_new_levels,
            chains=chains,
            iterations=iterations,
            warmup=warmup,
            cores=cores,
            seed=seed,
            backend="cmdstanr",
            adapt_delta=adapt_delta,
            max_treedepth=max_treedepth,
        )
        self.last_fit: BrmsFit | None = None
        self.fit_diagnostics: list[dict] = []

    @staticmethod
    def _slopes(indices):
        if indices is None:
            return None
        result = tuple(indices)
        for index in result:
            _integer("slope index", index, 0)
        if len(set(result)) != len(result):
            raise ValueError("slope indices must be unique")
        return result

    def _formula(self, n_fixed):
        terms = ["1", *(f"x{i}" for i in range(n_fixed))]
        has_group, has_cor = False, False
        for group, enabled, slopes, correlated in [
            (
                "subject",
                self.include_subject_re,
                self.subject_slopes,
                self.correlated_subject,
            ),
            ("item", self.include_item_re, self.item_slopes, self.correlated_item),
        ]:
            if not enabled:
                continue
            indices = tuple(range(n_fixed)) if slopes is None else slopes
            if any(i >= n_fixed for i in indices):
                raise ValueError(f"{group} slope index absent from the fixed design")
            random = " + ".join(["1", *(f"x{i}" for i in indices)])
            terms.append(f"({random} {'|' if correlated else '||'} {group})")
            has_group = True
            has_cor |= bool(indices) and correlated
        return "y ~ " + " + ".join(terms), has_group, has_cor

    @staticmethod
    def _inputs(y, train, test):
        y, train, test = (
            real_array_with_missing(a, name=name).astype(float, copy=False)
            for name, a in (("y_train", y), ("X_train", train), ("X_test", test))
        )
        if train.ndim != 2 or test.ndim != 2 or train.shape[1] < 2:
            raise ValueError("designs must be 2D with two trailing grouping columns")
        if train.shape[1] != test.shape[1] or y.shape != (len(train),):
            raise ValueError("response/design shapes do not align")
        if len(y) < 2 or len(test) == 0:
            raise ValueError("need at least two training rows and one test row")
        if not all(np.isfinite(a).all() for a in (y, train, test)):
            raise ValueError("inputs contain non-finite values")
        for a in (train, test):
            groups = a[:, -2:]
            if np.any(groups != np.floor(groups)) or np.any(np.abs(groups) > 2**53 - 1):
                raise ValueError(
                    "grouping codes must be exactly representable integers"
                )
        fixed = np.column_stack([np.ones(len(y)), train[:, :-2]])
        if np.linalg.matrix_rank(fixed) != fixed.shape[1]:
            raise ValueError("fixed design is rank deficient; remove redundant columns")
        return y, train, test

    def _run(self, directory, y, train, test):
        import pandas as pd

        n_fixed = train.shape[1] - 2
        formula, has_group, has_cor = self._formula(n_fixed)
        config = dict(
            self.settings,
            priors=asdict(self.priors),
            formula=formula,
            n_fixed=n_fixed,
            has_group=has_group,
            has_cor=has_cor,
            diagnostic_policy=dict(
                max_rhat=self.max_rhat,
                min_bulk_ess=self.min_ess,
                min_tail_ess=self.min_ess,
                min_ebfmi=0.3,
                divergences=0,
                treedepth_hits=0,
                action=self.diagnostics_action,
            ),
        )
        (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
        for name, array in [("train", train), ("test", test)]:
            frame = pd.DataFrame(
                array[:, :-2], columns=[f"x{i}" for i in range(n_fixed)]
            )
            for k, group in enumerate(["subject", "item"]):
                frame[group] = [str(int(v)) for v in array[:, n_fixed + k]]
            if name == "train":
                frame["y"] = y
            frame.to_csv(directory / f"{name}.csv", index=False)
        try:
            completed = run_r_script(
                files(__package__).joinpath("fit_brms.R"),
                [directory],
                executable=self.rscript,
                timeout=self.timeout,
                isolate_process_group=True,
            )
        except (RScriptError, subprocess.TimeoutExpired) as error:
            for name in ("stdout", "stderr"):
                content = getattr(error, name, None) or ""
                if isinstance(content, bytes):
                    content = content.decode("utf-8", errors="replace")
                (directory / f"{name}.log").write_text(content, encoding="utf-8")
            raise
        (directory / "stdout.log").write_text(completed.stdout, encoding="utf-8")
        (directory / "stderr.log").write_text(completed.stderr, encoding="utf-8")
        record = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        mu = np.loadtxt(directory / "expected.csv", delimiter=",", ndmin=2)
        sigma = np.loadtxt(directory / "sigma.csv", ndmin=1)
        expected_shape = (
            self.settings["chains"]
            * (self.settings["iterations"] - self.settings["warmup"]),
            len(test),
        )
        if mu.shape != expected_shape or sigma.shape != (len(mu),):
            raise RuntimeError("brms returned unexpected posterior draw dimensions")
        if (
            not np.isfinite(mu).all()
            or not np.isfinite(sigma).all()
            or np.any(sigma <= 0)
        ):
            raise RuntimeError(
                "brms returned non-finite means or invalid residual scales"
            )
        issues = _diagnostic_issues(record["diagnostics"], self.max_rhat, self.min_ess)
        record["diagnostics"].update(passed=not issues, issues=issues)
        record.update(
            config=config, artifact_dir=str(directory) if self.output_dir else None
        )
        self.fit_diagnostics.append(record)
        (directory / "validation.json").write_text(
            json.dumps(record, indent=2), encoding="utf-8"
        )
        rng = np.random.default_rng(self.settings["seed"] + 2)
        fit = BrmsFit(
            mu,
            sigma,
            mu + sigma[:, None] * rng.normal(size=mu.shape),
            record["coefficients"],
            record["diagnostics"],
            directory if self.output_dir else None,
        )
        if issues:
            message = "brms sampling diagnostics failed: " + "; ".join(issues)
            if self.diagnostics_action == "raise":
                raise RuntimeError(message)
            warnings.warn(message, RuntimeWarning, stacklevel=3)
        return fit

    def _fit(self, y, train, test):
        self.last_fit = None
        y, train, test = self._inputs(y, train, test)
        if self.output_dir is None:
            with tempfile.TemporaryDirectory(prefix="replica-brms-") as directory:
                fit = self._run(Path(directory), y, train, test)
        else:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            directory = Path(tempfile.mkdtemp(prefix="fit-", dir=self.output_dir))
            fit = self._run(directory, y, train, test)
        self.last_fit = fit
        return fit

    def last_fit_diagnostics(self) -> dict | None:
        """Snapshot checks from the current successful fit, not the history."""
        from copy import deepcopy

        if self.last_fit is None:
            return None
        diagnostics = deepcopy(self.last_fit.diagnostics)
        return {
            "convergence_applicable": True,
            "warnings": [
                *(self.fit_diagnostics[-1].get("warnings") or ()),
                *(diagnostics.get("issues") or ()),
            ],
            "diagnostic_checks_passed": diagnostics.get("passed"),
            "details": {"sampling": diagnostics},
        }

    def fit_predict(self, y_train, X_train, X_test) -> np.ndarray:
        """Fit once and return posterior expected responses on the test design.

        Args:
            y_train: Training responses in the units of the explicit priors.
            X_train: Fixed predictors followed by subject and item integer codes.
            X_test: Test design with exactly the same column order.
        """
        return self._fit(y_train, X_train, X_test).mean

    def fit_score(
        self, y_train, X_train, X_test, y_test, *, is_full=False
    ) -> FoldScore:
        """Fit on training rows and sum pointwise posterior predictive log densities.

        Held-out responses never enter the R process. This score integrates each
        observation separately, including group-effect uncertainty; it is not the
        joint predictive density of a held-out group. No ML likelihood-ratio or
        Bayes-factor interpretation applies to differences of these scores.

        Args:
            y_train: Training responses.
            X_train: Training design with trailing subject/item grouping codes.
            X_test: Test design with the same columns.
            y_test: Finite held-out responses, used only after fitting.
            is_full: Evaluation pipeline label, retained with fit diagnostics.
        """
        self.last_fit = None
        y_test = real_array_with_missing(y_test, name="y_test").astype(
            float, copy=False
        )
        if y_test.shape != (len(X_test),) or not np.isfinite(y_test).all():
            raise ValueError("y_test must be finite and aligned with X_test")
        fit = self._fit(y_train, X_train, X_test)
        self.fit_diagnostics[-1]["is_full"] = bool(is_full)
        pointwise = fit.log_predictive_density(y_test)
        if not np.isfinite(pointwise).all():
            raise ValueError("posterior predictive log densities are non-finite")
        record = self.fit_diagnostics[-1]
        record.update(
            score_type="pointwise_log_posterior_predictive_density",
            loglik=float(pointwise.sum()),
        )
        if fit.artifact_dir is not None:
            (fit.artifact_dir / "validation.json").write_text(
                json.dumps(record, indent=2), encoding="utf-8"
            )
        return FoldScore(float(pointwise.sum()), fit.mean, pointwise)
