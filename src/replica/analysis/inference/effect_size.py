"""Standardized effect size and a cross-validation interval for held-out ΔLL.

* :func:`cohens_f2`, f²-style variance-explained effect size
  = ΔR²/(1 - R²_full), computed from *held-out* prediction-R².
* :func:`nadeau_bengio_ci`, Nadeau & Bengio (2003) variance-corrected interval
  for the mean of k cross-validation fold scores (the naive t-interval is
  anti-conservative because CV folds share training data).

The interval works on the per-fold arrays the pipeline already persists, no
refit needed. Both helpers take plain floats / numpy arrays and have no
REPLICA dependencies.
"""

import math
from collections.abc import Sequence

import numpy as np
from scipy import stats


def _as_float(x) -> float:
    """Coerce to ``float``, mapping ``None`` / non-numeric to NaN."""
    if x is None:
        return float("nan")
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


# ---------------------------------------------------------------------------
# Standardized point effect size
# ---------------------------------------------------------------------------


def cohens_f2(delta_r2: float, r2_full: float) -> float:
    """Cohen's f² = ΔR² / (1 - R²_full), from held-out prediction R².

    Scale-free, so comparable across the DV-transform swap. A negative ``ΔR²``
    gives a negative value, read as a direction; a negative held-out
    ``R²_full`` pushes the denominator above 1 and shrinks the value.

    Args:
        delta_r2: Held-out ΔR², the prediction-R² the focal predictors add over
            the baseline model.
        r2_full: Held-out prediction R² of the full model, focal predictors
            included.

    Returns:
        The ratio, or ``NaN`` when ``r2_full`` is not below 1 (no residual
        variance left to explain) or either input is non-finite.
    """
    d = _as_float(delta_r2)
    rf = _as_float(r2_full)
    if not (math.isfinite(d) and math.isfinite(rf)):
        return float("nan")
    denom = 1.0 - rf
    if denom <= 0.0:
        return float("nan")
    return d / denom


# ---------------------------------------------------------------------------
# Confidence intervals
# ---------------------------------------------------------------------------


def _finite(values: Sequence[float]) -> np.ndarray:
    x = np.asarray(list(values), dtype=float)
    return x[np.isfinite(x)]


def nadeau_bengio_ci(
    fold_values: Sequence[float],
    test_ratio: float | None = None,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Nadeau & Bengio (2003) variance-corrected t-interval for CV fold scores.

    The mean of *k* cross-validation fold scores has an anti-conservative naive
    variance because the folds' training sets overlap. The correction inflates
    the variance by ``(1/k + test_ratio)`` where ``test_ratio = n_test/n_train``.

    Args:
        fold_values: One score per cross-validation fold. Non-finite entries
            are dropped, and the surviving count sets *k*.
        test_ratio: Held-out rows over training rows; ``None`` uses the
            standard *k*-fold value ``1/(k - 1)``.
        alpha: Two-sided error rate for nominal confidence level ``1 - alpha``.

    Returns:
        ``(mean, lo, hi)``. Fewer than two finite fold values yields ``nan``
        bounds.
    """
    x = _finite(fold_values)
    k = x.size
    if k == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(x.mean())
    if k < 2:
        return mean, float("nan"), float("nan")
    if test_ratio is None:
        test_ratio = 1.0 / (k - 1)
    var = float(x.var(ddof=1))
    se_corr = math.sqrt(max(0.0, (1.0 / k + test_ratio) * var))
    half = float(stats.t.ppf(1.0 - alpha / 2.0, k - 1)) * se_corr
    return mean, mean - half, mean + half
