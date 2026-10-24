"""Linear regression with R lm."""

import numpy as np

from replica.analysis.models.support.diagnostics import (
    _coefficient_snapshot,
    _FitDiagnostics,
    _r_fit_diagnostics,
    _residual_scale,
)
from replica.analysis.models.support.prediction import _FitPredictWithSigma
from replica.analysis.models.support.r import (
    _coefficients_by_term,
    _r_context,
    _to_r_dataframe,
)


class RLinearModel(_FitDiagnostics, _FitPredictWithSigma):
    """R's lm(), ordinary linear regression."""

    name = "lm"

    def configuration(self) -> dict:
        """Return declared options without fitted state."""
        return {}

    def _fit_and_predict(self, ro, y_train, X_train, X_test) -> np.ndarray:
        """Fit ``.lm_model`` in R globalenv and return test predictions.

        Must be called within an ``_r_context()`` block.
        """
        self._fit_diagnostics = None
        r_train = _to_r_dataframe(ro, X_train, y=y_train)
        r_test = _to_r_dataframe(ro, X_test)

        # Assign to R globalenv and run in R to preserve object classes
        ro.globalenv[".train_data"] = r_train
        ro.globalenv[".test_data"] = r_test
        ro.r(".lm_model <- lm(y ~ ., data=.train_data)")
        predictions = np.array(ro.r("predict(.lm_model, newdata=.test_data)")).flatten()
        self._fit_diagnostics = _r_fit_diagnostics(
            ro, ".lm_model", X_train.shape[1] + 1
        )
        return predictions

    def fit_predict_with_sigma(
        self,
        y_train: np.ndarray,
        X_train: np.ndarray,
        X_test: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """Predict the test rows and report R's own residual scale.

        Args:
            y_train: Dependent variable on the training rows.
            X_train: Training predictor columns, entered as ``x0``, ``x1``, ...
                in ``y ~ .``; R supplies the intercept.
            X_test: The same columns on the held-out rows.

        Returns:
            The test predictions and ``sigma(model)``, R's residual standard
            error based on the fitted model's residual degrees of freedom.
        """
        self._fit_diagnostics = None
        with _r_context() as ro:
            preds = self._fit_and_predict(ro, y_train, X_train, X_test)
            sigma = float(np.array(ro.r("sigma(.lm_model)"))[0])

        return preds, sigma

    def fit_predict_with_stats(
        self,
        y_train: np.ndarray,
        X_train: np.ndarray,
        X_test: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Returns (y_pred, beta, se, sigma).

        ``beta`` and ``se`` come from ``summary(model)$coefficients`` and are
        ordered as ``[intercept, x0, x1, ...]``, NaN at a column lm drops as
        linearly dependent.

        Args:
            y_train: Dependent variable on the training rows.
            X_train: Training predictor columns, entered as ``x0``, ``x1``, ...
                in ``y ~ .``; R supplies the intercept.
            X_test: The same columns on the held-out rows.
        """
        self._fit_diagnostics = None
        with _r_context() as ro:
            preds = self._fit_and_predict(ro, y_train, X_train, X_test)
            beta, se = _coefficients_by_term(ro, ".lm_model", X_train.shape[1])
            sigma = float(np.array(ro.r("sigma(.lm_model)"))[0])

        assert self._fit_diagnostics is not None
        self._fit_diagnostics["coefficients"] = _coefficient_snapshot(beta, se)
        self._fit_diagnostics["residual_scale"] = _residual_scale(
            sigma=sigma, convention="backend_fitted_residual_scale"
        )
        return preds, beta, se, sigma
