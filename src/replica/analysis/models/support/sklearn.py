"""Prediction and residual scale for fitted scikit-learn regression adapters."""

from abc import ABC, abstractmethod
from typing import Protocol

import numpy as np

from replica.analysis.models.support.diagnostics import _FitDiagnostics, _residual_scale


class _FittedRegressor(Protocol):
    def predict(self, X: np.ndarray) -> np.ndarray: ...


class SklearnRegression(_FitDiagnostics, ABC):
    """Fit once and expose predictions with the training-residual scale."""

    @abstractmethod
    def _fit(self, y_train: np.ndarray, X_train: np.ndarray) -> _FittedRegressor:
        """Construct and fit the backend estimator on training rows."""

    def fit_predict(
        self, y_train: np.ndarray, X_train: np.ndarray, X_test: np.ndarray
    ) -> np.ndarray:
        """Return held-out predictions from one fitted estimator.

        Args:
            y_train: Dependent variable on the training rows.
            X_train: Training design matrix without an intercept column.
            X_test: The same columns on held-out rows.
        """
        return self._fit(y_train, X_train).predict(X_test)

    def fit_predict_with_sigma(
        self, y_train: np.ndarray, X_train: np.ndarray, X_test: np.ndarray
    ) -> tuple[np.ndarray, float]:
        """Return predictions and the maximum-likelihood training-residual scale.

        Args:
            y_train: Dependent variable on the training rows.
            X_train: Training design matrix without an intercept column.
            X_test: The same columns on held-out rows.

        Returns:
            Test predictions and ``sqrt(mean(training_residuals**2))``.
        """
        model = self._fit(y_train, X_train)
        resid = y_train - model.predict(X_train)
        sigma = float(np.sqrt(np.mean(resid**2)))
        if self._fit_diagnostics is not None:
            self._fit_diagnostics["residual_scale"] = _residual_scale(
                sigma=sigma, convention="training_residual_mle"
            )
        return model.predict(X_test), sigma
