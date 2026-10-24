"""Elastic-net regression with cross-validated penalty strength and L1/L2 mix."""

from __future__ import annotations

import numpy as np

from replica.analysis.models.support.diagnostics import _coefficient_snapshot
from replica.analysis.models.support.sklearn import SklearnRegression


class ElasticNetCVModel(SklearnRegression):
    """Select penalty strength and L1/L2 mix by CV within each training fold.

    Fits an unpenalised intercept; the design matrix must contain predictor
    columns only.

    Args:
        l1_ratios: L1-versus-L2 mixes to search; values near 1 approach lasso
            and values near 0 approach ridge. Defaults span 0.1..1.0.
        n_alphas: Number of penalty-strength values per ``l1_ratio``, auto-scaled
            from the data by scikit-learn.
        cv: Number of inner folds used to score each ``(alpha, l1_ratio)``.
    """

    internal_tuning = True

    name = "elasticnet_cv"

    def configuration(self) -> dict:
        """Return declared options without fitted state."""
        return {
            "l1_ratios": list(self.l1_ratios),
            "n_alphas": self.n_alphas,
            "cv": self.cv,
        }

    def __init__(
        self,
        l1_ratios: list[float] | None = None,
        n_alphas: int = 25,
        cv: int = 5,
    ):
        """Fix the elastic-net search grid."""
        self.l1_ratios = (
            list(l1_ratios)
            if l1_ratios is not None
            else [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0]
        )
        self.n_alphas = int(n_alphas)
        self.cv = int(cv)
        #: ``(alpha, l1_ratio)`` chosen by CV for each fit, in call order.
        self.chosen: list[tuple[float, float]] = []

    def _fit(self, y_train: np.ndarray, X_train: np.ndarray):
        self._fit_diagnostics = None
        from sklearn.linear_model import ElasticNetCV

        model = ElasticNetCV(
            l1_ratio=self.l1_ratios,
            alphas=self.n_alphas,  # an int is the number of alphas along the path
            cv=self.cv,
            max_iter=5000,
            n_jobs=1,
        )
        model.fit(X_train, y_train)
        self.chosen.append((float(model.alpha_), float(model.l1_ratio_)))
        self._fit_diagnostics = {
            "convergence_applicable": True,
            "n_columns": X_train.shape[1] + 1,
            "coefficients": _coefficient_snapshot([model.intercept_, *model.coef_]),
            "selected_parameters": {
                "alpha": float(model.alpha_),
                "l1_ratio": float(model.l1_ratio_),
            },
            "details": {
                "n_iterations": int(model.n_iter_),
                "max_iterations": model.max_iter,
                "dual_gap": float(model.dual_gap_),
                "tolerance": model.tol,
            },
        }
        return model
