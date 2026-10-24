"""Linear mixed models with statsmodels."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd

from replica.analysis.contracts.models import AnalysisModel, require_training_rows
from replica.analysis.models.support.diagnostics import (
    _coefficient_snapshot,
    _FitDiagnostics,
    _residual_scale,
)


class StatsmodelsLME(_FitDiagnostics, AnalysisModel):
    """Linear mixed-effects model via statsmodels.

    Fits ``mixedlm(formula, data, groups)`` with random intercepts per group.
    Supports both direct ``.fit()`` and the ``AnalysisModel`` protocol
    (``fit_predict``) for use inside ``evaluate()``.

    The class sets ``needs_grouping``, so ``evaluate()`` appends the
    participant and item IDs as the last 2 columns of X. The ``group_col``
    parameter controls which grouping column is used for the LME random
    intercept: ``"item"`` (default, last column) or ``"participant"``
    (second-to-last column).

    Args:
        method: Optimisers ``MixedLM.fit`` tries in order; defaults to
            ``["bfgs"]``.
        nan_policy: ``"raise"`` rejects missing predictors. Explicit ``"fill"``
            replaces predictor NaN with 1.0; ``"omit"`` drops training rows with
            predictor NaN and leaves their test predictions undefined. The policy acts on
            direct ``.fit()`` and ``fit_predict`` calls; ``evaluate()`` drops
            non-finite rows before fitting, so it has no effect on that path.
        group_col: ``"item"`` uses the last appended column
            (stimulus/context ID); ``"participant"`` uses the second-to-last.
        integrated_loglik: Score with the fit's own integrated
            log-likelihood (``MixedLMResults.llf``, REML by statsmodels
            default). Sets ``owns_loglik``, so ``evaluate()`` calls
            :meth:`fit_score` per fold. The score is in-sample, so
            :meth:`fit_score` requires :class:`FullDataSplitter`'s identical
            training and scored rows and raises on any other split.
        reml: Use restricted maximum likelihood, as in the released study.
            Set false for likelihood comparisons with different fixed effects.
        fit_kwargs: Optional optimiser tolerances and iteration limits forwarded
            to ``MixedLM.fit``; omitted, statsmodels defaults are unchanged.
    """

    name = "statsmodels_lme"
    needs_grouping = True
    owns_loglik = False

    def configuration(self) -> dict:
        """Return declared options without fitted state."""
        return {
            "method": self._method,
            "nan_policy": self._nan_policy,
            "group_col": self._group_col,
            "reml": self._reml,
            "fit_kwargs": self._fit_kwargs,
            "integrated_loglik": self.owns_loglik,
        }

    def __init__(
        self,
        method: list[str] | None = None,
        nan_policy: str = "raise",
        group_col: str = "item",
        integrated_loglik: bool = False,
        reml: bool = True,
        fit_kwargs: dict | None = None,
    ):
        """Fix the optimiser, the NaN rule, the grouping column, and the score."""
        if nan_policy not in {"raise", "fill", "omit"}:
            raise ValueError("nan_policy must be 'raise', 'fill', or 'omit'")
        if group_col not in {"item", "participant"}:
            raise ValueError("group_col must be 'item' or 'participant'")
        self._method = method or ["bfgs"]
        self._nan_policy = nan_policy
        self._group_col = group_col
        self.owns_loglik = bool(integrated_loglik)
        self._reml = reml
        self._fit_kwargs = dict(fit_kwargs or {})

    def _split_X_and_groups(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Split feature matrix into predictors and grouping column.

        ``evaluate()`` appends participant and item IDs as the last 2 columns
        of X, because the class sets ``needs_grouping``. This method extracts
        the appropriate grouping column and returns the remaining predictors.
        """
        if X.shape[1] < 2:
            raise ValueError(
                "StatsmodelsLME needs grouping columns: X must end with the "
                "participant and item codes, as evaluate() appends them."
            )
        group_idx = -1 if self._group_col == "item" else -2
        groups = X[:, group_idx]
        if not np.isfinite(groups).all() or np.any(groups != np.floor(groups)):
            raise ValueError("Grouping columns must contain finite integer codes")
        bounds = np.iinfo(int)
        if groups.size and (
            groups.min().item() < bounds.min or groups.max().item() > bounds.max
        ):
            raise ValueError("Grouping codes exceed the integer range")
        groups = groups.astype(int)
        # Remove both grouping columns from predictors
        X_pred = X[:, :-2]
        return X_pred, groups

    def _fit_lme(
        self,
        y: np.ndarray,
        X_pred: np.ndarray,
        groups: np.ndarray,
        feature_names: list[str] | None = None,
    ):
        """Fit LME and return the statsmodels result object."""
        self._fit_diagnostics = None
        from statsmodels.regression.mixed_linear_model import MixedLM

        if not np.isfinite(y).all() or pd.isna(groups).any():
            raise ValueError(
                "Mixed-model responses and group identities must be complete"
            )
        if np.isinf(X_pred).any() or (
            self._nan_policy == "raise" and np.isnan(X_pred).any()
        ):
            raise ValueError(
                "Missing or infinite mixed-model predictors; choose a NaN policy explicitly"
            )
        col_names = (
            feature_names
            if feature_names is not None
            else [f"x{i}" for i in range(X_pred.shape[1])]
        )

        df = pd.DataFrame(X_pred, columns=col_names)
        df["_y"] = y
        df["_group"] = groups

        if self._nan_policy == "fill":
            df[col_names] = df[col_names].fillna(1.0)
        elif self._nan_policy == "omit":
            df = df.dropna(subset=col_names)

        formula = "_y ~ " + " + ".join(["1", *col_names])
        model = MixedLM.from_formula(formula, data=df, groups=df["_group"])
        result = model.fit(
            method=self._method,
            reml=self._reml,
            **getattr(self, "_fit_kwargs", {}),
        )
        # Optional uncertainty reads must not introduce new floating-point warnings.
        with np.errstate(divide="ignore", invalid="ignore"):
            standard_errors = result.bse_fe
        self._fit_diagnostics = {
            "convergence_applicable": True,
            "converged": bool(result.converged),
            "n_columns": len(result.fe_params),
            "coefficients": _coefficient_snapshot(
                result.fe_params,
                standard_errors,
                names=result.fe_params.index,
                naming="design_columns" if feature_names is None else "backend_terms",
            ),
            "residual_scale": _residual_scale(
                variance=result.scale, convention="REML" if self._reml else "ML"
            ),
            "random_covariances": [
                {
                    "factor": self._group_col,
                    "names": list(result.cov_re.columns),
                    "covariance": result.cov_re.to_numpy().tolist(),
                }
            ],
            "details": {
                "method_requested": deepcopy(self._method),
                "reml": self._reml,
                "n_observations": int(result.nobs),
            },
        }
        return result

    def _apply_test_nan_policy(self, X_test_pred: np.ndarray) -> np.ndarray:
        """Apply the declared missing-predictor policy without dropping test rows."""
        if np.isinf(X_test_pred).any() or (
            self._nan_policy == "raise" and np.isnan(X_test_pred).any()
        ):
            raise ValueError(
                "Missing or infinite mixed-model predictors; choose a NaN policy explicitly"
            )
        if self._nan_policy == "fill":
            return np.where(np.isnan(X_test_pred), 1.0, X_test_pred)
        return X_test_pred

    def fit_predict(
        self,
        y_train: np.ndarray,
        X_train: np.ndarray,
        X_test: np.ndarray,
    ) -> np.ndarray:
        """Fit LME on training data, predict on test data (fixed effects only).

        Args:
            y_train: Dependent variable on the training rows.
            X_train: Training design matrix with the participant and item codes
                as its last two columns; ``group_col`` picks the one that
                carries the random intercept.
            X_test: The same columns on the held-out rows.
        """
        self._fit_diagnostics = None
        X_train_pred, groups_train = self._split_X_and_groups(X_train)
        X_test_pred, _ = self._split_X_and_groups(X_test)
        X_test_pred = self._apply_test_nan_policy(X_test_pred)

        result = self._fit_lme(y_train, X_train_pred, groups_train)
        return self._fe_predict(result, X_test_pred)

    @staticmethod
    def _fe_predict(result, X_test_pred: np.ndarray) -> np.ndarray:
        """Predict from the fixed effects alone, dropping the random effects."""
        import statsmodels.api as sm

        X_te = sm.add_constant(X_test_pred, has_constant="add")
        return X_te @ np.asarray(result.fe_params).flatten()

    def fit_score(
        self,
        y_train: np.ndarray,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        is_full: bool,
    ) -> tuple[float, np.ndarray]:
        """Fit on train and return ``(integrated_loglik, y_pred_test)``.

        The entry point under ``integrated_loglik=True``. The score is in-sample,
        so the scored rows must be the training rows (:class:`FullDataSplitter`),
        and ``full_ll - base_ll`` is then the paper's mixedlm likelihood
        difference.

        Args:
            y_train: Dependent variable on the training rows.
            X_train: Training design matrix with the participant and item codes
                as its last two columns.
            X_test: The scored rows' design; must equal ``X_train``.
            y_test: The scored rows' response; must equal ``y_train``.
            is_full: Whether this is the full side of the comparison rather
                than the baseline. Unused, the two sides differ by the columns
                ``X_train`` already carries.

        Returns:
            The mixed model's own ``llf``, which integrates over the random
            intercepts, and the fixed-effects predictions on the scored rows.

        Raises:
            ValueError: When the scored rows differ from the training rows.
        """
        self._fit_diagnostics = None
        require_training_rows(y_train, X_train, X_test, y_test)
        X_train_pred, groups_train = self._split_X_and_groups(X_train)
        X_test_pred, _ = self._split_X_and_groups(X_test)
        result = self._fit_lme(y_train, X_train_pred, groups_train)
        y_pred = self._fe_predict(result, self._apply_test_nan_policy(X_test_pred))
        return float(result.llf), y_pred

    def fit_predict_with_sigma(
        self,
        y_train: np.ndarray,
        X_train: np.ndarray,
        X_test: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """Fit LME, predict, and return sigma (residual std from model).

        Args:
            y_train: Dependent variable on the training rows.
            X_train: Training design matrix with the participant and item codes
                as its last two columns.
            X_test: The same columns on the held-out rows.

        Returns:
            The fixed-effects test predictions and the square root of the
            fit's residual variance, which excludes the random-intercept
            variance.
        """
        self._fit_diagnostics = None
        X_train_pred, groups_train = self._split_X_and_groups(X_train)
        X_test_pred, _ = self._split_X_and_groups(X_test)
        X_test_pred = self._apply_test_nan_policy(X_test_pred)

        result = self._fit_lme(y_train, X_train_pred, groups_train)
        y_pred = self._fe_predict(result, X_test_pred)
        return y_pred, np.sqrt(result.scale)

    def fit(
        self,
        y: np.ndarray,
        X: np.ndarray,
        groups: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> StatsmodelsLMEResult:
        """Fit LME model on the full dataset (no train/test split).

        Args:
            y: Response variable, shape (n,).
            X: Predictor matrix, shape (n, p).
            groups: Group labels for random intercepts, shape (n,).
            feature_names: Optional column names for X.

        Returns:
            StatsmodelsLMEResult with log-likelihood and coefficients.
        """
        result = self._fit_lme(y, X, groups, feature_names=feature_names)

        return StatsmodelsLMEResult(
            llf=result.llf,
            converged=result.converged,
            coefficients={k: v for k, v in result.fe_params.items()},
            pvalues={k: v for k, v in result.pvalues.items() if k != "Group Var"},
            scale=result.scale,
            group_var=result.cov_re.iloc[0, 0]
            if hasattr(result.cov_re, "iloc")
            else 0.0,
            # nobs counts the fitted rows, so it reflects the "omit" NaN policy.
            n_obs=int(result.nobs),
            _result=result,
        )


class StatsmodelsLMEResult:
    """Result from a StatsmodelsLME fit."""

    def __init__(
        self,
        llf: float,
        converged: bool,
        coefficients: dict[str, float],
        pvalues: dict[str, float],
        scale: float,
        group_var: float,
        n_obs: int,
        _result=None,
    ):
        """Hold the fields a model comparison reads off one mixedlm fit.

        Args:
            llf: The fit's integrated log-likelihood, which ``compare`` differences.
            converged: Whether the optimiser reported convergence.
            coefficients: Fixed-effect estimates by term name, intercept included.
            pvalues: Wald p-values by term name, with the group-variance row dropped.
            scale: Residual variance, excluding the random-intercept variance.
            group_var: Variance of the random intercept across groups.
            n_obs: Rows the fit used, so rows the NaN policy dropped are excluded.
            _result: The statsmodels ``MixedLMResults`` these fields were read from.
        """
        self.llf = llf
        self.converged = converged
        self.coefficients = coefficients
        self.pvalues = pvalues
        self.scale = scale
        self.group_var = group_var
        self.n_obs = n_obs
        self._result = _result

    def compare(self, baseline: StatsmodelsLMEResult) -> float:
        """Compute delta log-likelihood vs a baseline model.

        Args:
            baseline: Result of the nested fit whose ``llf`` is subtracted,
                fitted on the same rows.
        """
        return self.llf - baseline.llf
