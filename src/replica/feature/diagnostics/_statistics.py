"""Statistics shared by the estimator, convergence, and fingerprint diagnostics."""

import numpy as np

# |t| of the mean error above which a bias counts as detected.
BIAS_T_THRESHOLD = 4.0
# Accepted d log(variance) / d log(N) band for an i.i.d. MC estimator (slope -1).
VARIANCE_SLOPE_BAND = (-1.3, -0.7)


def validate_n_grid(n_grid, n_reps: int) -> None:
    """Reject sample-size grids the bias/variance checks cannot use."""
    if len(n_grid) < 2 or any(b <= a for a, b in zip(n_grid, n_grid[1:], strict=False)):
        raise ValueError(
            f"n_grid must be >= 2 strictly increasing values, got {n_grid}"
        )
    # ddof=1 spread and the bias t-stat are undefined on a single replication.
    if n_reps < 2:
        raise ValueError(
            f"n_reps must be >= 2 for the bias/variance checks, got {n_reps}"
        )


def fit_loglog_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Least-squares slope of ``y`` on ``x`` over jointly finite points."""
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 2 or np.ptp(x[m]) == 0:
        return float("nan")
    return float(np.polyfit(x[m], y[m], 1)[0])


def bias_tstat(mean_err: float, sd: float, n_reps: int) -> float:
    """t of a mean error; zero spread with nonzero mean is a definite (signed-inf) bias."""
    se = sd / np.sqrt(n_reps)
    if se > 0:
        return float(mean_err / se)
    return 0.0 if mean_err == 0 else float(np.sign(mean_err) * np.inf)


def affine_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Fit ``y ~ slope*x + intercept``; returns ``(slope, intercept, r2)``.

    ``r2`` is NaN when ``y`` is constant (zero total sum of squares).
    """
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (slope * x + intercept)
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(np.sum(resid**2)) / ss_tot if ss_tot > 0 else float("nan")
    return float(slope), float(intercept), r2
