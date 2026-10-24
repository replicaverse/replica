"""Linear mixed models with lme4."""

import numpy as np

from replica.analysis.contracts.design import Design
from replica.analysis.contracts.grouped import GroupedFit, GroupedFrame
from replica.analysis.models.mixed_effects.grouped import (
    fit_lmer_grouped,
    validate_lmer_design,
)
from replica.analysis.models.mixed_effects.options import (
    LmerOptions,
    random_effect_terms,
)
from replica.analysis.models.support.diagnostics import (
    _coefficient_snapshot,
    _r_fit_diagnostics,
    _residual_scale,
)
from replica.analysis.models.support.r import (
    _coefficients_by_term,
    _importr,
    _r_context,
)
from replica.utils.arrays import real_array_with_missing


class RLmerModel(LmerOptions):
    """R's lme4::lmer(), linear mixed-effects regression.

    Args:
        optimizer: lme4 optimizer passed to ``lmerControl``.
        maxfun: Max optimizer function evaluations (``lmerControl``);
            honored on the fit_predict path only.
        nlminb_fallback: On a convergence message, refit with the other
            optimizer (fit_predict path only; ``fit_score``/``fit_loglik``
            do a single fit).
        **options: The random-effects structure and criterion, see
            :class:`LmerOptions`.
    """

    name = "lmer"

    def configuration(self) -> dict:
        """Return declared options without fitted state."""
        return {
            "optimizer": self.optimizer,
            "maxfun": self.maxfun,
            "nlminb_fallback": self.nlminb_fallback,
            **super().configuration(),
        }

    def __init__(
        self,
        optimizer: str = "bobyqa",
        maxfun: int = 50000,
        nlminb_fallback: bool = True,
        **options,
    ):
        """Fix the lme4 fit options on top of the shared random-effects ones."""
        super().__init__(**options)
        # ModelBlocks lmefit recipe defaults: bobyqa with extended maxfun,
        # retry with optimx::nlminb on a convergence warning.
        self.optimizer = optimizer
        self.maxfun = int(maxfun)
        self.nlminb_fallback = bool(nlminb_fallback)

    def _add_re_design_columns(self, ro, data: dict, n_rows: int) -> list[str]:
        """Put the ``re_design`` columns in the frame and name them.

        Must be called within an ``_r_context()`` block.
        """
        re_design = self._re_design_rows(n_rows)
        if re_design is None:
            return []
        names = []
        for j in range(re_design.shape[1]):
            name = f"r{j}"
            data[name] = ro.FloatVector(re_design[:, j])
            names.append(name)
        return names

    def _build_formula_and_data(self, ro, y, X):
        """Build formula string and R data.frame, handling grouping columns.

        Must be called within an ``_r_context()`` block.
        """
        ncols = X.shape[1]
        n_fixed = ncols - 2

        data = {"y": ro.FloatVector(y)}
        fixed_terms = []
        for i in range(n_fixed):
            col_name = f"x{i}"
            data[col_name] = ro.FloatVector(X[:, i])
            fixed_terms.append(col_name)

        data["participant_id"] = ro.FactorVector(ro.IntVector(X[:, -2].astype(int)))
        data["item_id"] = ro.FactorVector(ro.IntVector(X[:, -1].astype(int)))

        re_terms = self._add_re_design_columns(ro, data, X.shape[0])

        r_df = ro.r["data.frame"](**data)

        n_participants = len(set(X[:, -2].astype(int)))
        n_items = len(set(X[:, -1].astype(int)))

        random_terms = random_effect_terms(
            self,
            fixed_terms,
            re_terms,
            n_participants=n_participants,
            n_items=n_items,
        )

        fixed_str = " + ".join(fixed_terms) if fixed_terms else "1"
        if random_terms:
            formula_str = f"y ~ {fixed_str} + {' + '.join(random_terms)}"
            use_lmer = True
        else:
            formula_str = f"y ~ {fixed_str}"
            use_lmer = False

        return formula_str, r_df, use_lmer

    def _build_test_dataframe(self, ro, X_test):
        """Build test R data.frame with grouping factor columns.

        Must be called within an ``_r_context()`` block.
        """
        ncols = X_test.shape[1]
        n_fixed = ncols - 2
        test_data = {}
        for i in range(n_fixed):
            test_data[f"x{i}"] = ro.FloatVector(X_test[:, i])
        test_data["participant_id"] = ro.FactorVector(
            ro.IntVector(X_test[:, -2].astype(int))
        )
        test_data["item_id"] = ro.FactorVector(ro.IntVector(X_test[:, -1].astype(int)))
        return ro.r["data.frame"](**test_data)

    def validate_design(self, design: Design) -> None:
        """Validate an explicit named design before extracting features.

        Args:
            design: Declaration of model and sampling roles.
        """
        validate_lmer_design(self)

    def fit_grouped(
        self, y_train: np.ndarray, train: GroupedFrame, test: GroupedFrame
    ) -> GroupedFit:
        """Fit named factors and return selected conditional predictions.

        Args:
            y_train: Training response vector.
            train: Named training frame.
            test: Named held-out frame.
        """
        self._fit_diagnostics = None
        y_train = real_array_with_missing(y_train, name="y_train")
        with _r_context() as ro:
            return fit_lmer_grouped(self, ro, y_train, train, test)

    def _fit_and_predict(self, ro, y_train, X_train, X_test) -> tuple[np.ndarray, str]:
        """Fit in R globalenv and return ``(preds, model_var)``.

        ``model_var`` names the fitted R object (``.lmer_model`` or, with no
        random-effect terms, ``.lm_model``) for follow-up reads. Must be
        called within an ``_r_context()`` block.
        """
        formula_str, r_train, use_lmer = self._build_formula_and_data(
            ro, y_train, X_train
        )
        r_test = self._build_test_dataframe(ro, X_test)

        ro.globalenv[".train_data"] = r_train
        ro.globalenv[".test_data"] = r_test

        if use_lmer:
            self._fit_lmer(ro, formula_str)
            preds = np.array(
                ro.r(
                    "predict(.lmer_model, newdata=.test_data,"
                    " re.form=NA, allow.new.levels=TRUE)"
                )
            ).flatten()
            return preds, ".lmer_model"
        ro.globalenv[".formula_str"] = formula_str
        ro.r(".lm_model <- lm(as.formula(.formula_str), data=.train_data)")
        preds = np.array(ro.r("predict(.lm_model, newdata=.test_data)")).flatten()
        return preds, ".lm_model"

    def fit_predict_with_sigma(
        self,
        y_train: np.ndarray,
        X_train: np.ndarray,
        X_test: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """Predict the test rows and report the fit's residual scale.

        Args:
            y_train: Dependent variable on the training rows.
            X_train: Training design matrix, fixed-effect columns followed by
                the integer participant and item codes.
            X_test: The same columns on the held-out rows.

        Returns:
            The fixed-effects test predictions and ``sigma(model)``, the
            residual standard error with the random-effect variances excluded.
        """
        self._fit_diagnostics = None
        y_train = real_array_with_missing(y_train, name="y_train")
        X_train = real_array_with_missing(X_train, name="X_train")
        X_test = real_array_with_missing(X_test, name="X_test")
        with _r_context() as ro:
            preds, model_var = self._fit_and_predict(ro, y_train, X_train, X_test)
            sigma = float(np.array(ro.r(f"sigma({model_var})"))[0])
            self._fit_diagnostics = _r_fit_diagnostics(
                ro, model_var, X_train.shape[1] - 1
            )

        return preds, sigma

    def _fit_lmer(self, ro, formula_str: str) -> None:
        """Fit ``lme4::lmer`` per modelblocks' lmefit recipe.

        Primary optimizer is ``self.optimizer``; default ``"bobyqa"``
        matches modelblocks. Set ``optimizer="nlminb"`` to use
        ``optimx::nlminb`` as the primary fit instead, derivative-based.

        On a convergence warning, refits with the other optimizer and keeps
        the alternative when it converges cleanly or when its relative
        gradient is smaller. Relative gradient here is
        ``max(abs(gradient / pmax(abs(val), 1e-12)))``, which differs from
        the Hessian-scaled Newton step ``max(abs(solve(Hessian, gradient)))``
        used by ``minRelGrad`` in ``resource-lmefit/scripts/lmetools.r`` of
        modelblocks/modelblocks-release.
        """
        _importr("lme4")
        ro.globalenv[".formula_str"] = formula_str
        ro.globalenv[".maxfun"] = self.maxfun
        reml_str = "TRUE" if self.reml else "FALSE"
        primary_is_nlminb = self.optimizer == "nlminb"
        if primary_is_nlminb:
            _importr("optimx")
            ro.r(
                ".lmer_model <- lme4::lmer("
                "  as.formula(.formula_str),"
                "  data=.train_data,"
                f"  REML={reml_str},"
                "  control=lme4::lmerControl("
                "    optimizer='optimx',"
                "    optCtrl=list(method=c('nlminb'), maxit=.maxfun)"
                "  ))"
            )
        else:
            ro.globalenv[".optimizer"] = self.optimizer
            ro.r(
                ".lmer_model <- lme4::lmer("
                "  as.formula(.formula_str),"
                "  data=.train_data,"
                f"  REML={reml_str},"
                "  control=lme4::lmerControl("
                "    optimizer=.optimizer,"
                "    optCtrl=list(maxfun=.maxfun)"
                "  ))"
            )
        if not self.nlminb_fallback:
            return
        has_warn = bool(ro.r("!is.null(.lmer_model@optinfo$conv$lme4$messages)")[0])
        if not has_warn:
            return
        if primary_is_nlminb:
            ro.r(
                ".lmer_model_alt <- tryCatch(lme4::lmer("
                "  as.formula(.formula_str),"
                "  data=.train_data,"
                f"  REML={reml_str},"
                "  control=lme4::lmerControl("
                "    optimizer='bobyqa',"
                "    optCtrl=list(maxfun=.maxfun)"
                "  )), error=function(e) NULL)"
            )
        else:
            _importr("optimx")
            ro.r(
                ".lmer_model_alt <- tryCatch(lme4::lmer("
                "  as.formula(.formula_str),"
                "  data=.train_data,"
                f"  REML={reml_str},"
                "  control=lme4::lmerControl("
                "    optimizer='optimx',"
                "    optCtrl=list(method=c('nlminb'), maxit=.maxfun)"
                "  )), error=function(e) NULL)"
            )
        ro.r(
            "if (!is.null(.lmer_model_alt)) {"
            "  rg1 <- tryCatch(max(abs("
            "    .lmer_model@optinfo$derivs$gradient /"
            "    pmax(abs(.lmer_model@optinfo$val), 1e-12))),"
            "    error=function(e) Inf);"
            "  rg2 <- tryCatch(max(abs("
            "    .lmer_model_alt@optinfo$derivs$gradient /"
            "    pmax(abs(.lmer_model_alt@optinfo$val), 1e-12))),"
            "    error=function(e) Inf);"
            "  warn2 <- .lmer_model_alt@optinfo$conv$lme4$messages;"
            "  if (is.null(warn2) || rg2 < rg1) .lmer_model <- .lmer_model_alt"
            "}"
        )

    def fit_predict_with_stats(
        self,
        y_train: np.ndarray,
        X_train: np.ndarray,
        X_test: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Return ``(y_pred, beta, se, sigma)`` for the fixed effects.

        Args:
            y_train: Dependent variable on the training rows.
            X_train: Training design matrix, fixed-effect columns followed by
                the integer participant and item codes.
            X_test: The same columns on the held-out rows.

        Returns:
            Predictions on *X_test*; the estimate and standard-error columns of
            ``summary(model)$coefficients``, ordered like ``X_train``'s fixed
            columns with an intercept first (as in :class:`RLinearModel`) and
            NaN at a column lme4 drops as linearly dependent; and
            ``sigma(model)``.
        """
        self._fit_diagnostics = None
        y_train = real_array_with_missing(y_train, name="y_train")
        X_train = real_array_with_missing(X_train, name="X_train")
        X_test = real_array_with_missing(X_test, name="X_test")
        with _r_context() as ro:
            preds, model_var = self._fit_and_predict(ro, y_train, X_train, X_test)
            beta, se = _coefficients_by_term(ro, model_var, X_train.shape[1] - 2)
            sigma = float(np.array(ro.r(f"sigma({model_var})"))[0])
            self._fit_diagnostics = _r_fit_diagnostics(
                ro, model_var, X_train.shape[1] - 1
            )

        assert self._fit_diagnostics is not None
        self._fit_diagnostics["coefficients"] = _coefficient_snapshot(beta, se)
        self._fit_diagnostics["residual_scale"] = _residual_scale(
            sigma=sigma, convention="backend_fitted_residual_scale"
        )
        return preds, beta, se, sigma

    def _fit_loglik_fitted(
        self,
        y: np.ndarray,
        X: np.ndarray,
        REML: bool = False,
    ) -> tuple[float, np.ndarray]:
        """One fit, returning ``(logLik, in-sample fitted values)``."""
        self._fit_diagnostics = None
        y = real_array_with_missing(y, name="y")
        X = real_array_with_missing(X, name="X")
        with _r_context() as ro:
            formula_str, r_data, use_lmer = self._build_formula_and_data(ro, y, X)
            ro.globalenv[".train_data"] = r_data

            if use_lmer:
                _importr("lme4")
                ro.globalenv[".formula_str"] = formula_str
                reml_str = "TRUE" if REML else "FALSE"
                if self.optimizer == "nlminb":
                    # lme4 has no bare 'nlminb' optimizer; route via optimx,
                    # matching _fit_lmer.
                    _importr("optimx")
                    ro.globalenv[".maxfun"] = self.maxfun
                    control = (
                        "lme4::lmerControl(optimizer='optimx',"
                        " optCtrl=list(method=c('nlminb'), maxit=.maxfun))"
                    )
                else:
                    ro.globalenv[".optimizer"] = self.optimizer
                    control = "lme4::lmerControl(optimizer=.optimizer)"
                ro.r(
                    f".lmer_model <- lme4::lmer("
                    f"  as.formula(.formula_str),"
                    f"  data=.train_data,"
                    f"  REML={reml_str},"
                    f"  control={control}"
                    f")"
                )
                ll = float(np.array(ro.r("as.numeric(logLik(.lmer_model))"))[0])
                fitted = np.asarray(
                    ro.r("as.numeric(fitted(.lmer_model))"), dtype=float
                )
                model_var = ".lmer_model"
            else:
                ro.globalenv[".formula_str"] = formula_str
                ro.r(".lm_model <- lm(as.formula(.formula_str), data=.train_data)")
                ll = float(np.array(ro.r("as.numeric(logLik(.lm_model))"))[0])
                fitted = np.asarray(ro.r("as.numeric(fitted(.lm_model))"), dtype=float)
                model_var = ".lm_model"
            self._last_fixed = _coefficients_by_term(ro, model_var, X.shape[1] - 2)
            self._fit_diagnostics = _r_fit_diagnostics(ro, model_var, X.shape[1] - 1)
            self._fit_diagnostics["coefficients"] = _coefficient_snapshot(
                *self._last_fixed
            )

        return ll, fitted
