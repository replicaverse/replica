"""Bias and variance scaling over finite Monte Carlo sample budgets.

Independent seeds estimate the mean error and variance at each budget. The
reported variance slope and bias statistic are diagnostics, not proofs of
asymptotic consistency.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from replica.feature.diagnostics._statistics import (
    BIAS_T_THRESHOLD,
    VARIANCE_SLOPE_BAND,
    bias_tstat,
    fit_loglog_slope,
    validate_n_grid,
)

__all__ = ["ConvergenceReport", "mc_convergence"]

Estimator = Callable[[int, int], np.ndarray]  # (n_samples, seed) -> (n_units,)


@dataclass
class ConvergenceReport:
    """Error and spread of an MC estimator at each sample budget in the sweep.

    The list fields run parallel to ``n_grid``, one entry per budget.
    """

    n_grid: tuple[int, ...]
    bias: list[float] = field(default_factory=list)  # mean error per N
    rmse: list[float] = field(default_factory=list)
    variance: list[float] = field(default_factory=list)  # across-seed, per N
    variance_slope: float = np.nan  # d log var / d log N (expect -1)
    bias_tstat: float = np.nan  # t-stat of mean error at largest N

    deterministic: bool = False  # across-seed variance ~0: not an MC estimator

    @property
    def variance_ok(self) -> bool:
        """True when the fitted variance slope sits in the band 1/N scaling gives."""
        return bool(np.isfinite(self.variance_slope)) and (
            VARIANCE_SLOPE_BAND[0] <= self.variance_slope <= VARIANCE_SLOPE_BAND[1]
        )

    @property
    def biased(self) -> bool:
        """True when the mean error at the largest budget is too large to be noise."""
        # NaN = indeterminate; ±inf (deterministic offset) = definite bias.
        return not np.isnan(self.bias_tstat) and abs(self.bias_tstat) > BIAS_T_THRESHOLD

    def to_text(self) -> str:
        """Render the two verdicts and a per-budget table of bias, rmse, variance."""
        var_line = (
            "across-seed variance ≈ 0: estimator is deterministic; "
            "variance scaling not applicable"
            if self.deterministic
            else f"variance slope: {self.variance_slope:.3f} "
            f"(expect ≈ -1; {'OK' if self.variance_ok else 'VIOLATED'})"
        )
        lines = [
            f"MC convergence over N = {list(self.n_grid)}",
            var_line,
            f"bias t-stat at N={self.n_grid[-1]}: {self.bias_tstat:.2f} "
            f"({'BIASED' if self.biased else 'consistent with unbiased'})",
            "",
            f"{'N':>8} {'bias':>12} {'rmse':>12} {'variance':>12}",
        ]
        for i, n in enumerate(self.n_grid):
            lines.append(
                f"{n:>8} {self.bias[i]:>12.5f} {self.rmse[i]:>12.5f} "
                f"{self.variance[i]:>12.3g}"
            )
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.to_text()


def mc_convergence(
    estimate: Estimator,
    exact: np.ndarray,
    n_grid: tuple[int, ...] = (64, 256, 1024),
    n_reps: int = 8,
    base_seed: int = 0,
) -> ConvergenceReport:
    """Sweep the sample budget of an MC estimator against exact values.

    Args:
        estimate: ``estimate(n_samples, seed)`` returning per-unit values
            aligned with ``exact``.
        exact: the spec's exact values, shape ``(n_units,)``.
        n_grid: increasing sample budgets.
        n_reps: independent seeds per budget.
        base_seed: seed handed to the first replication; every later replication
            takes the next integer, so the seeds never repeat across the sweep.
    """
    exact = np.asarray(exact, dtype=np.float64)
    validate_n_grid(n_grid, n_reps)
    mask = np.isfinite(exact)
    if not mask.any():
        raise ValueError("exact has no finite values")
    # The slope/bias statistics are near-uninformative on tiny samples
    # (single-unit slope check is a coin flip).
    if int(mask.sum()) * n_reps < 48:
        raise ValueError(
            f"{int(mask.sum())} finite units x {n_reps} reps is too few for "
            "the variance-slope and bias checks; use ≥ 48 unit-reps"
        )
    exact = exact[mask]
    report = ConvergenceReport(n_grid=tuple(n_grid))

    seed = base_seed
    per_seed_mean_err_at_max: np.ndarray | None = None
    for n in n_grid:
        runs = []
        for _ in range(n_reps):
            est = np.asarray(estimate(n, seed), dtype=np.float64)[mask]
            if not np.isfinite(est).all():
                raise ValueError(
                    f"estimator returned non-finite values at N={n}, "
                    f"seed={seed}, where exact is finite, fix the estimator "
                    "or mask those positions out of `exact`"
                )
            runs.append(est)
            seed += 1
        E = np.stack(runs)  # (n_reps, n_units)
        err = E - exact[None, :]
        report.bias.append(float(err.mean()))
        report.rmse.append(float(np.sqrt((err**2).mean())))
        report.variance.append(float(E.var(axis=0, ddof=1).mean()))
        if n == n_grid[-1]:
            per_seed_mean_err_at_max = err.mean(axis=1)  # (n_reps,)

    if min(report.variance) <= 1e-30:
        report.deterministic = True  # variance_slope stays NaN
    else:
        report.variance_slope = fit_loglog_slope(
            np.log(n_grid), np.log(report.variance)
        )
    m = per_seed_mean_err_at_max
    if m is not None and len(m) > 1:
        report.bias_tstat = bias_tstat(m.mean(), float(m.std(ddof=1)), len(m))
    return report
