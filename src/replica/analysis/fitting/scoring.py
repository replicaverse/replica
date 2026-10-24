"""Pure statistical helpers for the evaluation pipeline.

Self-contained numeric functions (Gaussian log-likelihood, standardization,
orthogonalization, R², sigma estimation) used by `evaluate`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from replica.utils.arrays import real_array_with_missing

if TYPE_CHECKING:
    from replica.analysis.contracts.models import AnalysisModel


def _gaussian_log_likelihood(
    y: np.ndarray, y_pred: np.ndarray, sigma: float | np.ndarray | None = None
) -> float:
    """Summed Gaussian log-likelihood in nats; see the pointwise scale contract."""
    return float(_gaussian_log_likelihood_pointwise(y, y_pred, sigma).sum())


def _gaussian_log_likelihood_pointwise(
    y: np.ndarray, y_pred: np.ndarray, sigma: float | np.ndarray | None = None
) -> np.ndarray:
    """Gaussian log density in nats for aligned, finite real response vectors.

    ``sigma`` is a positive finite scalar or one scale per response. None
    estimates the root mean squared test residual. Zero fitted scale raises:
    a degenerate Gaussian has no density under this scoring contract.
    """
    y = real_array_with_missing(y, name="y").astype(float, copy=False)
    y_pred = real_array_with_missing(y_pred, name="y_pred").astype(float, copy=False)
    if (
        y.ndim != 1
        or y.shape != y_pred.shape
        or not np.isfinite(y).all()
        or not np.isfinite(y_pred).all()
    ):
        raise ValueError(
            "Gaussian responses and predictions must be finite aligned vectors"
        )
    with np.errstate(over="ignore"):
        residuals = y - y_pred
    overflowed = ~np.isfinite(residuals)
    if sigma is None:
        if not y.size:
            return np.empty(0, dtype=float)
        if overflowed.any():
            largest = max(np.max(np.abs(y)), np.max(np.abs(y_pred)))
            normalized = y / largest - y_pred / largest
        else:
            largest = np.max(np.abs(residuals))
            normalized = residuals / largest if largest else residuals
        with np.errstate(over="ignore"):
            sigma = largest * np.sqrt(np.mean(normalized**2))
    scales = real_array_with_missing(sigma, name="sigma").astype(float, copy=False)
    if (
        scales.shape not in ((), y.shape)
        or not np.isfinite(scales).all()
        or np.any(scales <= 0)
    ):
        raise ValueError(
            "Gaussian scales must be positive, finite and aligned with responses"
        )
    with np.errstate(over="ignore", divide="ignore"):
        standardized = residuals / scales
        if overflowed.any():
            selected_scales = np.broadcast_to(scales, y.shape)[overflowed]
            standardized[overflowed] = (
                y[overflowed] / selected_scales - y_pred[overflowed] / selected_scales
            )
        return -0.5 * np.log(2 * np.pi) - np.log(scales) - 0.5 * standardized**2


def _r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination (R²)."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def _fit_predict_sigma(
    model: AnalysisModel,
    y_train: np.ndarray,
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Use the backend's scale, or one fit with an OLS-style residual estimate.

    The fallback counts fixed columns and an intercept, excluding grouping IDs.
    Backends with other residual degrees of freedom should supply their scale.
    """
    if hasattr(model, "fit_predict_with_sigma"):
        return model.fit_predict_with_sigma(y_train, X_train, X_test)
    predictions = model.fit_predict(y_train, X_train, np.concatenate([X_test, X_train]))
    y_pred_test = predictions[: len(X_test)]
    y_pred_train = predictions[len(X_test) :]
    residuals = y_train - y_pred_train
    n_grouping = 2 if getattr(model, "needs_grouping", False) else 0
    p = X_train.shape[1] - n_grouping + 1
    sigma = np.sqrt(np.sum(residuals**2) / max(len(y_train) - p, 1))
    return y_pred_test, sigma
