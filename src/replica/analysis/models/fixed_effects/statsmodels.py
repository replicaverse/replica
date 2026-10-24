"""Linear regression with statsmodels."""

from __future__ import annotations

import warnings

import numpy as np

from replica.analysis.contracts.models import AnalysisModel
from replica.analysis.models.support.diagnostics import (
    _coefficient_snapshot,
    _FitDiagnostics,
    _residual_scale,
)


class StatsmodelsOLS(_FitDiagnostics, AnalysisModel):
    """Least squares with an intercept and optional elastic-net fitting.

    Zero penalty uses the direct least-squares solver by default. Explicit
    ``regularized=True`` selects the iterative solver even at zero penalty.
    """

    name = "statsmodels_ols"

    def configuration(self) -> dict:
        """Return declared options without fitted state."""
        return {
            "alpha": self.alpha,
            "L1_wt": self.L1_wt,
            "regularized": self.regularized,
        }

    def __init__(
        self,
        alpha: float = 0.0,
        L1_wt: float = 1.0,
        regularized: bool | None = None,
    ):
        """Configure the penalty and the solver behind it.

        Args:
            alpha: Penalty weight handed to ``fit_regularized``; ``0.0`` leaves
                the coordinate-descent solver unpenalised.
            L1_wt: Share of the penalty that is L1, from ``0.0`` (ridge) to
                ``1.0`` (lasso).
            regularized: ``None`` selects ``fit_regularized`` for nonzero
                ``alpha`` only; true always selects it. False selects the
                direct ``fit()`` and ignores ``alpha`` and ``L1_wt``.
        """
        self.alpha = alpha
        self.L1_wt = L1_wt
        self.regularized = alpha != 0.0 if regularized is None else regularized

    def _fit(self, y_train: np.ndarray, X_train: np.ndarray):
        """Fit model and return the results object."""
        self._fit_diagnostics = None
        import statsmodels.api as sm
        from statsmodels.tools.sm_exceptions import ConvergenceWarning

        # has_constant="add" so the intercept column is always present; the default
        # "skip" would drop it for a constant all-nonzero fold, desyncing train/test.
        X = sm.add_constant(X_train, has_constant="add")
        model = sm.OLS(y_train, X)
        if self.regularized:
            result = model.fit_regularized(alpha=self.alpha, L1_wt=self.L1_wt)
            converged = getattr(result, "converged", None)
            if converged is not None and not converged:
                warnings.warn(
                    "StatsmodelsOLS regularized fit did not converge.",
                    ConvergenceWarning,
                    stacklevel=3,
                )
            self._fit_diagnostics = {
                "convergence_applicable": True,
                "converged": None if converged is None else bool(converged),
                "n_columns": X.shape[1],
                "coefficients": _coefficient_snapshot(result.params),
                "selected_parameters": {"alpha": self.alpha, "l1_ratio": self.L1_wt},
            }
            return result
        result = model.fit()
        # Optional uncertainty reads must not introduce new floating-point warnings.
        with np.errstate(divide="ignore", invalid="ignore"):
            variance, standard_errors = result.scale, result.bse
        self._fit_diagnostics = {
            "convergence_applicable": False,
            "rank": int(result.model.rank),
            "residual_scale": _residual_scale(
                variance=variance, convention="residual_degrees_of_freedom"
            ),
            "n_columns": X.shape[1],
            "coefficients": _coefficient_snapshot(result.params, standard_errors),
        }
        return result

    def fit_predict(
        self,
        y_train: np.ndarray,
        X_train: np.ndarray,
        X_test: np.ndarray,
    ) -> np.ndarray:
        """Predicted y for the test rows, shape (n_test,).

        Args:
            y_train: Dependent variable on the training rows.
            X_train: Training predictor columns; the intercept is added here.
            X_test: The same columns on the held-out rows.
        """
        import statsmodels.api as sm

        result = self._fit(y_train, X_train)
        X_te = sm.add_constant(X_test, has_constant="add")
        return np.asarray(result.predict(X_te))

    def fit_predict_with_sigma(
        self,
        y_train: np.ndarray,
        X_train: np.ndarray,
        X_test: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """Predict the test rows and report the training residual scale.

        Args:
            y_train: Dependent variable on the training rows.
            X_train: Training predictor columns; the intercept is added here.
            X_test: The same columns on the held-out rows.

        Returns:
            The test predictions and ``sqrt(RSS / (n - p))`` over the training
            residuals. Least squares uses design rank for ``p``; regularized
            fits use the column count. The denominator is at least one.
        """
        import statsmodels.api as sm

        result = self._fit(y_train, X_train)
        X_te = sm.add_constant(X_test, has_constant="add")
        y_pred = np.asarray(result.predict(X_te))

        # Compute training-set sigma
        X_tr = sm.add_constant(X_train, has_constant="add")
        residuals = y_train - np.asarray(result.predict(X_tr))
        residual_df = (
            len(y_train) - X_tr.shape[1] if self.regularized else result.df_resid
        )
        sigma = np.sqrt(np.sum(residuals**2) / max(residual_df, 1))

        assert self._fit_diagnostics is not None
        self._fit_diagnostics["residual_scale"] = _residual_scale(
            sigma=sigma, convention="rss_over_max_residual_df_1"
        )
        return y_pred, sigma
