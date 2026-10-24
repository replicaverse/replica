"""Bias, variance, RMSE, and interval coverage for scalar stochastic estimators.

The estimator accepts a sample size and seed and returns a value, optionally
with its standard error. Independent repetitions compare these estimates with
a known target. Finite-grid classifications do not prove consistency.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from scipy import stats as _stats

from replica.feature.diagnostics._statistics import (
    BIAS_T_THRESHOLD,
    VARIANCE_SLOPE_BAND,
    bias_tstat,
    fit_loglog_slope,
    validate_n_grid,
)

__all__ = ["EstimatorReport", "NLevel", "diagnose_estimator"]

# A replication either returns the estimate, or (estimate, standard_error).
Estimator = Callable[[int, int], "float | tuple[float, float]"]


@dataclass
class NLevel:
    """Simulation-study summary at one sample size N (over R replications)."""

    n: int
    n_reps: int
    bias: float
    bias_se: float  # standard error of the bias estimate (across replications)
    bias_t: float  # bias / bias_se; |t| > ~4 is a detectable bias
    variance: float  # across-replication variance of theta_hat
    sd: float
    rmse: float
    mse: float
    coverage: float = float("nan")  # empirical nominal-95 CI coverage
    ci_width: float = float("nan")
    replicates: np.ndarray = field(default_factory=lambda: np.empty(0))


@dataclass
class EstimatorReport:
    """Simulation-study result for one estimator: a level per N, plus the verdict.

    ``classification`` describes bias at the tested sample sizes;
    ``verdict`` summarizes that evidence. Neither establishes asymptotic
    consistency or unbiasedness.
    """

    theta: float
    n_grid: tuple[int, ...]
    levels: list[NLevel]
    variance_slope: float  # d log(variance) / d log(N); -1 for an i.i.d. MC mean
    bias_slope: float  # d log|bias| / d log(N); <= -1 for an O(1/N) bias
    normality_p: float  # Shapiro-Wilk p at the largest N (CLT -> normal)
    classification: str  # no_bias_detected | decreasing_bias | persistent_bias
    verdict: str
    nominal_coverage: float = 0.95

    @property
    def largest(self) -> NLevel:
        """The level at the biggest sample size, where the bias test is decided."""
        return self.levels[-1]

    def to_text(self) -> str:
        """Render the verdict, a per-N table, and the two log-log slopes."""
        lines = [
            f"Estimand theta = {self.theta:.5f}",
            f"Classification: {self.classification}",
            f"Verdict: {self.verdict}",
            "",
            f"{'N':>7} {'bias':>11} {'bias_t':>8} {'sd':>10} {'rmse':>10} "
            f"{'mse':>11} {'cover':>7}",
        ]
        for L in self.levels:
            cov = "n/a" if not np.isfinite(L.coverage) else f"{L.coverage:.2f}"
            lines.append(
                f"{L.n:>7} {L.bias:>11.5f} {L.bias_t:>8.1f} {L.sd:>10.5f} "
                f"{L.rmse:>10.5f} {L.mse:>11.3g} {cov:>7}"
            )
        lines += [
            "",
            f"variance slope d log var / d log N = {self.variance_slope:.3f} "
            f"(i.i.d. MC mean: -1)",
            f"|bias| slope  d log|bias|/ d log N = {self.bias_slope:.3f} "
            f"(O(1/N) bias: -1; persistent: ~0)",
            f"normality of theta_hat at N={self.largest.n}: Shapiro p = "
            f"{self.normality_p:.3f}",
        ]
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.to_text()


# |bias| at the largest N is < this fraction of the largest |bias| seen.
_BIAS_REDUCTION_RATIO = 0.2


def diagnose_estimator(
    estimate: Estimator,
    theta: float,
    n_grid: tuple[int, ...] = (16, 64, 256, 1024, 4096),
    n_reps: int = 400,
    base_seed: int = 0,
    ci_z: float = 1.959963984540054,  # 95 percent normal CI
) -> EstimatorReport:
    """Run the simulation study and classify the estimator.

    Args:
        estimate: ``estimate(n, seed)`` returning the estimate at sample size
            ``n``, or a ``(estimate, standard_error)`` pair to also measure
            confidence-interval coverage. Must be reproducible given ``seed``.
        theta: the true estimand (known exactly, or a high-N gold reference).
        n_grid: increasing sample sizes.
        n_reps: independent replications R per sample size.
        base_seed: seed of the first replication; each later one takes the next
            integer, so no two replications share a seed.
        ci_z: normal quantile defining the interval whose coverage is measured,
            paired with ``EstimatorReport.nominal_coverage``. Used only when
            ``estimate`` also returns a standard error.
    """
    validate_n_grid(n_grid, n_reps)
    theta = float(theta)

    levels: list[NLevel] = []
    seed = base_seed
    for n in n_grid:
        ests = np.empty(n_reps)
        ses = np.full(n_reps, np.nan)
        for r in range(n_reps):
            out = estimate(n, seed)
            seed += 1
            if isinstance(out, tuple):
                ests[r], ses[r] = float(out[0]), float(out[1])
            else:
                ests[r] = float(out)
        if not np.isfinite(ests).all():
            raise ValueError(f"estimator returned non-finite values at N={n}")

        bias = float(ests.mean() - theta)
        sd = float(ests.std(ddof=1))
        bias_se = sd / np.sqrt(n_reps)
        bias_t = bias_tstat(bias, sd, n_reps)
        rmse = float(np.sqrt(np.mean((ests - theta) ** 2)))
        coverage, ci_width = float("nan"), float("nan")
        if np.isfinite(ses).all() and (ses > 0).all():
            covered = np.abs(ests - theta) <= ci_z * ses
            coverage = float(covered.mean())
            ci_width = float(2 * ci_z * ses.mean())
        levels.append(
            NLevel(
                n=n,
                n_reps=n_reps,
                bias=bias,
                bias_se=bias_se,
                bias_t=bias_t,
                variance=sd**2,
                sd=sd,
                rmse=rmse,
                mse=rmse**2,
                coverage=coverage,
                ci_width=ci_width,
                replicates=ests.copy(),
            )
        )

    logn = np.log(np.array(n_grid, dtype=float))
    with np.errstate(divide="ignore"):
        var_slope = fit_loglog_slope(logn, np.log([L.variance for L in levels]))
    abs_bias = np.array([abs(L.bias) for L in levels])
    bias_slope = fit_loglog_slope(
        logn, np.log(np.where(abs_bias > 0, abs_bias, np.nan))
    )
    # Normality of the sampling distribution at the largest N (CLT).
    big = levels[-1].replicates
    normality_p = (
        float(_stats.shapiro(big).pvalue)
        if big.size >= 3 and np.ptp(big) > 0
        else float("nan")
    )

    classification, verdict = _classify(levels, var_slope, bias_slope)
    return EstimatorReport(
        theta=theta,
        n_grid=tuple(n_grid),
        levels=levels,
        variance_slope=var_slope,
        bias_slope=bias_slope,
        normality_p=normality_p,
        classification=classification,
        verdict=verdict,
    )


def _classify(levels, var_slope, bias_slope):
    last = levels[-1]
    # Compare the largest tested N with the largest observed bias magnitude.
    max_bias = max(abs(L.bias) for L in levels)
    sig_any = any(abs(L.bias_t) > BIAS_T_THRESHOLD for L in levels)
    sig_large = abs(last.bias_t) > BIAS_T_THRESHOLD
    decreasing = abs(last.bias) < _BIAS_REDUCTION_RATIO * max_bias
    var_ok = (
        np.isfinite(var_slope)
        and VARIANCE_SLOPE_BAND[0] <= var_slope <= VARIANCE_SLOPE_BAND[1]
    )
    rate = f" Fitted variance slope: {var_slope:.2f}." if np.isfinite(var_slope) else ""

    if sig_large and not decreasing:
        return (
            "persistent_bias",
            f"Bias remains detectable at N={last.n}: {last.bias:+.4g} "
            f"(|t|={abs(last.bias_t):.1f}), without a substantial reduction "
            "over the tested sample sizes. This does not establish an "
            "asymptotic limit or identify the cause.",
        )
    if sig_any and decreasing:
        return (
            "decreasing_bias",
            f"Bias is detectable at a tested sample size; its magnitude at "
            f"N={last.n} ({abs(last.bias):.4g}) is below {_BIAS_REDUCTION_RATIO:.0%} "
            f"of the largest observed magnitude ({max_bias:.4g}). "
            f"Fitted |bias| slope: {bias_slope:.2f}.{rate} "
            "This does not establish consistency.",
        )
    # No detected bias at the largest N and no observed substantial decrease.
    note = (
        ""
        if var_ok
        else (
            f" Variance slope {var_slope:.2f} does not support a 1/N rate "
            "over the tested sample sizes."
        )
    )
    return (
        "no_bias_detected",
        f"No bias detected at N={last.n} "
        f"(bias {last.bias:+.4g}, |t|={abs(last.bias_t):.1f}).{rate} "
        f"This does not establish unbiasedness or consistency.{note}",
    )
