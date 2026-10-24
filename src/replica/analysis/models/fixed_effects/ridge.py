"""Ridge regression with the L2 penalty selected by leave-one-out CV."""

from __future__ import annotations

import numpy as np

from replica.analysis.models.support.diagnostics import _coefficient_snapshot
from replica.analysis.models.support.sklearn import SklearnRegression


class RidgeCVModel(SklearnRegression):
    """Select an L2 penalty by leave-one-out CV within each training fold.

    Fits an unpenalised intercept; the design matrix must contain predictor
    columns only. Penalty selection does not guarantee improved test scores.

    Args:
        alphas: L2 penalties to choose among; defaults to a log grid from 1e-2
            to 1e4. The value chosen for each fold is appended to
            :attr:`chosen_alphas`.
    """

    internal_tuning = True

    name = "ridge_cv"

    def configuration(self) -> dict:
        """Return declared options without fitted state."""
        return {
            "alphas": self.alphas.tolist(),
        }

    def __init__(self, alphas: np.ndarray | list[float] | None = None):
        """Set the candidate L2 penalties."""
        self.alphas = (
            np.asarray(alphas, dtype=float)
            if alphas is not None
            else np.logspace(-2, 4, 13)
        )
        #: L2 penalty chosen by CV for each fit, in call order.
        self.chosen_alphas: list[float] = []

    def _fit(self, y_train: np.ndarray, X_train: np.ndarray):
        self._fit_diagnostics = None
        from sklearn.linear_model import RidgeCV

        model = RidgeCV(alphas=self.alphas)
        model.fit(X_train, y_train)
        self.chosen_alphas.append(float(model.alpha_))
        self._fit_diagnostics = {
            "convergence_applicable": False,
            "n_columns": X_train.shape[1] + 1,
            "coefficients": _coefficient_snapshot([model.intercept_, *model.coef_]),
            "selected_parameters": {"alpha": float(model.alpha_)},
            "details": {"gcv_mode_requested": model.gcv_mode},
        }
        return model
