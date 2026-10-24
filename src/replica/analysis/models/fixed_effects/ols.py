"""Ordinary least squares with an intercept and residual-scale options."""

import numpy as np

from replica.analysis.contracts.models import AnalysisModel
from replica.analysis.models.support.diagnostics import (
    _coefficient_snapshot,
    _FitDiagnostics,
    _residual_scale,
)

from ._least_squares import column_norms, least_squares, residual_variance


class OLS(_FitDiagnostics, AnalysisModel):
    """Unregularized least squares with an intercept.

    Columns are normalized internally. Dependent columns use the minimum-norm
    solution in those coordinates; standard errors describe that solution.

    Args:
        sigma_correction: How to estimate sigma from training residuals.
            ``"df"`` (default): ``sqrt(RSS / (n - rank))``, requiring positive
            residual degrees of freedom. Rank includes the intercept.
            ``"mle"``: ``sqrt(RSS / n)``, the training residual RMS.
    """

    name = "ols"

    def configuration(self) -> dict:
        """Return declared options without fitted state."""
        return {
            "sigma_correction": self._sigma_correction,
        }

    def __init__(self, sigma_correction: str = "df"):
        """Fix which residual-variance denominator the sigma methods use."""
        if sigma_correction not in ("df", "mle"):
            raise ValueError(
                f"sigma_correction must be 'df' or 'mle', got {sigma_correction!r}"
            )
        self._sigma_correction = sigma_correction

    def _fit(
        self, y_train: np.ndarray, X_train: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, int, np.ndarray]:
        """Return coefficients, augmented design, rank and a covariance factor."""
        self._fit_diagnostics = None
        X_tr = self._design(X_train)
        if np.ma.isMaskedArray(y_train) or np.iscomplexobj(y_train):
            raise ValueError("OLS requires unmasked real responses")
        y_train = np.asarray(y_train, dtype=float)
        if y_train.shape != (len(X_tr),) or not len(y_train):
            raise ValueError(
                "OLS requires a nonempty response matching the design rows"
            )
        if not np.isfinite(y_train).all():
            raise ValueError("OLS requires finite responses")
        beta, rank, covariance_factor = least_squares(X_tr, y_train)
        self._fit_diagnostics = {
            "convergence_applicable": False,
            "rank": int(rank),
            "n_columns": X_tr.shape[1],
            "coefficients": _coefficient_snapshot(beta),
        }
        return beta, X_tr, int(rank), covariance_factor

    @staticmethod
    def _design(values: np.ndarray) -> np.ndarray:
        if np.ma.isMaskedArray(values) or np.iscomplexobj(values):
            raise ValueError("OLS requires unmasked real predictors")
        values = np.asarray(values, dtype=float)
        if values.ndim != 2 or not np.isfinite(values).all():
            raise ValueError("OLS requires a finite two-dimensional predictor matrix")
        return np.column_stack([np.ones(len(values)), values])

    def _variance(self, y, design, beta, rank) -> float:
        rss = np.sum((y - self._quiet_matmul(design, beta)) ** 2)
        variance = float(residual_variance(rss, len(y), rank, self._sigma_correction))
        if not np.isfinite(variance):
            raise ValueError(
                "OLS residual variance requires positive residual degrees of freedom"
            )
        return variance

    @staticmethod
    def _quiet_matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            return A @ B

    def fit_predict(
        self,
        y_train: np.ndarray,
        X_train: np.ndarray,
        X_test: np.ndarray,
    ) -> np.ndarray:
        """Predicted y for the test rows, shape (n_test,).

        Args:
            y_train: Dependent variable on the training rows.
            X_train: Training predictor columns; the intercept is prepended here.
            X_test: The same columns on the held-out rows.
        """
        beta, _, _, _ = self._fit(y_train, X_train)
        X_te = self._design(X_test)
        return self._quiet_matmul(X_te, beta)

    def fit_predict_with_stats(
        self,
        y_train: np.ndarray,
        X_train: np.ndarray,
        X_test: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Like fit_predict but also returns (beta, se, sigma).

        Args:
            y_train: Dependent variable on the training rows.
            X_train: Training predictor columns; the intercept is prepended here.
            X_test: The same columns on the held-out rows.

        Returns:
            y_pred: Test-set predictions, shape (n_test,).
            beta: Coefficient vector (including intercept).
            se: Standard errors of coefficients.
            sigma: Training-set residual standard error under ``sigma_correction``.
        """
        beta, X_tr, rank, covariance_factor = self._fit(y_train, X_train)
        X_te = self._design(X_test)
        y_pred = self._quiet_matmul(X_te, beta)

        sigma2 = self._variance(y_train, X_tr, beta, rank)
        se = column_norms(covariance_factor.T) * np.sqrt(sigma2)

        assert self._fit_diagnostics is not None
        self._fit_diagnostics.update(
            coefficients=_coefficient_snapshot(beta, se),
            residual_scale=_residual_scale(
                variance=sigma2, convention=self._sigma_correction
            ),
        )
        return y_pred, beta, se, np.sqrt(sigma2)

    def fit_predict_with_sigma(
        self,
        y_train: np.ndarray,
        X_train: np.ndarray,
        X_test: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """Like fit_predict but also returns training-set sigma.

        Args:
            y_train: Dependent variable on the training rows.
            X_train: Training predictor columns; the intercept is prepended here.
            X_test: The same columns on the held-out rows.

        Returns:
            The test predictions and the residual standard error of the fit
            under ``sigma_correction``.
        """
        beta, X_tr, rank, _ = self._fit(y_train, X_train)
        X_te = self._design(X_test)
        y_pred = self._quiet_matmul(X_te, beta)

        sigma = np.sqrt(self._variance(y_train, X_tr, beta, rank))

        assert self._fit_diagnostics is not None
        self._fit_diagnostics["residual_scale"] = _residual_scale(
            sigma=sigma, convention=self._sigma_correction
        )
        return y_pred, sigma
