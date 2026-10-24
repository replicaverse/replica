"""Prediction adapters for models that return a residual scale."""

import numpy as np

from replica.analysis.contracts.models import AnalysisModel


class _FitPredictWithSigma(AnalysisModel):
    """Derive the ordinary prediction interface from a sigma-returning fit."""

    def fit_predict(
        self,
        y_train: np.ndarray,
        X_train: np.ndarray,
        X_test: np.ndarray,
    ) -> np.ndarray:
        """Predicted y for the test rows, shape (n_test,).

        Args:
            y_train: Dependent variable on the training rows.
            X_train: Training design matrix expected by the concrete R model.
            X_test: The corresponding held-out design matrix.
        """
        y_pred, _ = self.fit_predict_with_sigma(y_train, X_train, X_test)
        return y_pred
