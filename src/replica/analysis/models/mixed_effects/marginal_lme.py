"""Linear mixed-effects model scored by its per-subject marginal likelihood.

Fits ``lme4::lmer`` and scores held-out rows with the marginal
multivariate-normal log-likelihood that integrates out the random effects:
mean ``predict(re.form=NA)``, per-subject covariance ``Z Sigma Zt + sigma^2 I``.

Adapted from ModelBlocks: the fit follows
``resource-lmefit/scripts/lmetools.r::regressLinearModel`` and the held-out
score follows ``resource-lmefit/scripts/get_loglik_lmer.r``, which scores an
already-fitted model. The fit keeps upstream's bobyqa budget of 50000 and,
like upstream by default, no nlminb retry (upstream's ``suppress_nlminb`` flag
is inverted, so the retry never runs unless asked for); ``mb_fit`` takes
``maxfun`` and ``nlminb_fallback`` for callers that want the retry, which then
keeps the nlminb fit whenever it fits, in place of upstream's ``minRelGrad``
tie-break. Deviations: subject levels with zero rows are dropped
(``Freq > 0``), and prediction passes ``allow.new.levels=TRUE`` so held-out
subjects unseen during training do not raise.
"""

from typing import Literal

import numpy as np

from replica.analysis.contracts.models import AnalysisModel
from replica.analysis.models.support.diagnostics import (
    _FitDiagnostics,
    _lme4_diagnostics,
    _r_parameter_diagnostics,
)
from replica.analysis.models.support.r import _r_context
from replica.utils.arrays import real_array_with_missing

# lme4 fit + marginal-MVN held-out loglik, adapted from ModelBlocks; the module
# docstring lists the upstream scripts and the deviations from them.
MB_R_DEFINITIONS = r"""
local({
  ulib <- Sys.getenv("R_LIBS_USER")
  if (nzchar(ulib) && dir.exists(ulib)) .libPaths(c(ulib, .libPaths()))
})
suppressMessages({
  library(lme4); library(mvtnorm); library(Matrix); library(optimx)
})
z. <- function(x) as.numeric(scale(x))

# ModelBlocks (lmetools.r regressLinearModel) fits bobyqa at maxfun 50000 and
# keeps that fit whatever lme4 warns; its nlminb retry is behind a flag that is
# off by default. Those are the defaults here; a caller that wants the retry
# asks for it.
mb_fit <- function(df, formula_str, maxfun=50000, nlminb_fallback=FALSE) {
  ctrl <- lmerControl(optimizer="bobyqa", optCtrl=list(maxfun=maxfun))
  m <- lmer(as.formula(formula_str), data=df, REML=FALSE, control=ctrl)
  conv <- m@optinfo$conv$lme4$messages
  if (nlminb_fallback && !is.null(conv)) {
    ctrl2 <- lmerControl(optimizer="optimx",
                         optCtrl=list(method="nlminb", maxit=maxfun))
    m2 <- tryCatch(
      lmer(as.formula(formula_str), data=df, REML=FALSE, control=ctrl2),
      error=function(e) NULL)
    if (!is.null(m2)) m <- m2
  }
  m
}

mb_loglik <- function(model, df, formula_str) {
  f <- as.formula(formula_str)
  df$subject <- factor(as.character(df$subject))
  df <- df[order(df$subject),]
  subj_table <- as.data.frame(table(df$subject))
  subj_table <- subj_table[subj_table$Freq > 0, ]
  yv <- model.frame(paste0(deparse(f[[2]]), " ~ 1"), data=df)[, 1]
  n_re <- ncol(ranef(model)$subject)
  bar.f <- findbars(f)
  mf <- model.frame(subbars(f), data=df)
  rt <- mkReTrms(bar.f, mf)
  full_Zt <- rt$Zt
  lt <- getME(model, "Lambdat")
  var_e <- sigma(model)^2
  if (n_re > 1) {
    var_u <- var_e * crossprod(lt[1:n_re, 1:n_re])
  } else {
    var_u <- as.numeric(VarCorr(model)$subject)
  }
  preds <- predict(model, re.form=NA, newdata=df, allow.new.levels=TRUE)
  start_idx <- 0
  total_ll <- 0
  for (row in seq_len(nrow(subj_table))) {
    cnt <- subj_table[row, 2]
    Zt <- full_Zt[((row-1)*n_re+1):(row*n_re), (start_idx+1):(start_idx+cnt)]
    cy <- yv[(start_idx+1):(start_idx+cnt)]
    if (n_re > 1) {
      vy <- t(Zt) %*% var_u %*% Zt
    } else {
      zc <- as.numeric(Zt)
      vy <- (t(t(zc)) %*% var_u %*% zc)
    }
    vy <- as.matrix(vy) + var_e * diag(cnt)
    xb <- preds[(start_idx+1):(start_idx+cnt)]
    total_ll <- total_ll + dmvnorm(cy, mean=xb, sigma=vy, log=TRUE)
    start_idx <- start_idx + cnt
  }
  total_ll
}
"""


class MarginalLMEModel(_FitDiagnostics, AnalysisModel):
    """lme4 fit scored by the marginal-MVN held-out log-likelihood.

    ``evaluate()`` appends the grouping columns ``(participant, item)`` after
    the fixed-effect columns. The participant column is the by-subject grouping
    factor; the item column is ignored. The default preserves ModelBlocks'
    separate training/test standardization. Corrected experiments can freeze
    the standardization at the training boundary.

    Args:
        random_slope_control_cols: indices into the control block whose columns
            get a by-subject random slope in the baseline and full models.
        focal_random_slope: give the focal predictor (the extra last fixed
            column in the full design) a by-subject random slope.
        uncorrelated_slopes: build ``(1 + ... || subject)`` rather than
            ``(... | subject)``. Only an intercept-only structure accepts it:
            lme4 orders ``||`` effects term by term, and the marginal
            likelihood slices ``Zt`` and ``Lambdat`` subject by subject, so a
            random slope raises ``NotImplementedError``.
        scaling: ``"partition"`` retains the released ``z.`` formulas;
            ``"training"`` applies training means and sample standard deviations
            to both partitions, including random-slope columns.
        unscaled_cols: fixed-effect column indices to retain in their input
            coding, such as a binary prior-fixation indicator.
    """

    name = "marginal_lme"
    owns_loglik = True
    needs_grouping = True

    def configuration(self) -> dict:
        """Return declared options without fitted state."""
        return {
            "random_slope_control_cols": self.random_slope_control_cols,
            "focal_random_slope": self.focal_random_slope,
            "uncorrelated_slopes": self.uncorrelated_slopes,
            "scaling": self.scaling,
            "unscaled_cols": self.unscaled_cols,
        }

    def __init__(
        self,
        random_slope_control_cols: list[int] | None = None,
        focal_random_slope: bool = False,
        uncorrelated_slopes: bool = False,
        scaling: Literal["partition", "training"] = "partition",
        unscaled_cols: tuple[int, ...] = (),
    ):
        """Fix the by-subject random-effects structure of both fits."""
        self.random_slope_control_cols = list(random_slope_control_cols or [])
        self.focal_random_slope = focal_random_slope
        if uncorrelated_slopes and (
            self.random_slope_control_cols or self.focal_random_slope
        ):
            raise NotImplementedError(
                "MarginalLMEModel scores correlated random slopes only: with "
                "uncorrelated slopes (||) lme4 orders the random effects term "
                "by term, which the per-subject marginal likelihood does not "
                "follow. Use uncorrelated_slopes=False."
            )
        self.uncorrelated_slopes = uncorrelated_slopes
        if scaling not in {"partition", "training"}:
            raise ValueError(f"Unknown scaling boundary: {scaling}")
        self.scaling = scaling
        self.unscaled_cols = tuple(unscaled_cols)
        if any(c < 0 for c in self.unscaled_cols):
            raise ValueError("Unscaled column indices must be nonnegative")
        self.fit_diagnostics: list[dict] = []

    def _formula(self, n_fixed: int, is_full: bool) -> str:
        def term(i):
            if self.scaling == "training" or i in self.unscaled_cols:
                return f"x{i}"
            return f"z.(x{i})"

        fixed_terms = [term(i) for i in range(n_fixed)]
        fixed = " + ".join(fixed_terms) if fixed_terms else "1"
        slope_cols = [c for c in self.random_slope_control_cols if c < n_fixed]
        if is_full and self.focal_random_slope and n_fixed > 0:
            slope_cols = slope_cols + [n_fixed - 1]  # focal is the last column
        re_terms = ["1"] + [term(i) for i in slope_cols]
        sep = "||" if self.uncorrelated_slopes else "|"
        re = f"({' + '.join(re_terms)} {sep} subject)"
        return f"y ~ {fixed} + {re}"

    @staticmethod
    def _frame(ro, y, Xf, subject):
        cols = {"y": ro.FloatVector(np.asarray(y, dtype=float))}
        for i in range(Xf.shape[1]):
            cols[f"x{i}"] = ro.FloatVector(np.asarray(Xf[:, i], dtype=float))
        cols["subject"] = ro.IntVector(np.asarray(subject, dtype=int).tolist())
        return ro.r["data.frame"](**cols)

    def fit_score(
        self,
        y_train: np.ndarray,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        is_full: bool,
    ) -> tuple[float, np.ndarray]:
        """Fit on train, return ``(held_out_loglik, y_pred_test)``.

        ``X_*`` are ``[fixed..., participant, item]``; the participant column
        (second-to-last) is the by-subject grouping factor.

        Args:
            y_train: Dependent variable on the training rows.
            X_train: Training design matrix, fixed-effect columns followed by
                the participant and item codes.
            X_test: The same columns on the held-out rows.
            y_test: Held-out dependent variable, which the marginal-MVN score
                is evaluated at.
            is_full: Whether the focal predictor is present, which decides
                whether ``focal_random_slope`` adds a slope on the last fixed
                column.
        """
        self._fit_diagnostics = None
        y_train = real_array_with_missing(y_train, name="y_train")
        X_train = real_array_with_missing(X_train, name="X_train")
        X_test = real_array_with_missing(X_test, name="X_test")
        y_test = real_array_with_missing(y_test, name="y_test")
        n_fixed = X_train.shape[1] - 2  # drop the (participant, item) group cols
        subj_tr = X_train[:, -2]
        subj_te = X_test[:, -2]
        Xf_tr = X_train[:, :n_fixed]
        Xf_te = X_test[:, :n_fixed]
        if self.scaling == "training":
            # The released z.() wrapper discards scale attributes, so R
            # recomputes test means/SDs. Freeze them at the training boundary.
            center = Xf_tr.mean(axis=0)
            spread = Xf_tr.std(axis=0, ddof=1)
            spread[spread == 0] = 1.0
            for column in self.unscaled_cols:
                if column < n_fixed:
                    center[column], spread[column] = 0.0, 1.0
            Xf_tr = (Xf_tr - center) / spread
            Xf_te = (Xf_te - center) / spread
        formula = self._formula(n_fixed, is_full)
        with _r_context() as ro:
            ro.r(MB_R_DEFINITIONS)
            ro.globalenv[".train"] = self._frame(ro, y_train, Xf_tr, subj_tr)
            ro.globalenv[".test"] = self._frame(ro, y_test, Xf_te, subj_te)
            ro.globalenv[".m"] = ro.r(f"mb_fit(.train, {formula!r})")
            messages = list(ro.r("as.character(.m@optinfo$conv$lme4$messages)"))
            self.fit_diagnostics.append(
                {
                    "is_full": bool(is_full),
                    "n_train": len(y_train),
                    "n_test": len(y_test),
                    "messages": messages,
                    "optimizer_code": int(ro.r(".m@optinfo$conv$opt")[0]),
                    "singular": bool(ro.r("isSingular(.m)")[0]),
                }
            )
            loglik = float(np.asarray(ro.r(f"mb_loglik(.m, .test, {formula!r})"))[0])
            y_pred = np.asarray(
                ro.r("predict(.m, re.form=NA, newdata=.test, allow.new.levels=TRUE)")
            )
            try:
                parameters = _r_parameter_diagnostics(ro, ".m")
                parameters["coefficients"]["naming"] = "backend_terms"
            except Exception as error:
                parameters = {"details": {"parameters_unavailable": str(error)}}
        record = self.fit_diagnostics[-1]
        self._fit_diagnostics = _lme4_diagnostics(
            [record["optimizer_code"]],
            record["messages"],
            singular=record["singular"],
            n_train=record["n_train"],
            n_test=record["n_test"],
        )
        self._fit_diagnostics["details"].update(parameters.pop("details", {}))
        self._fit_diagnostics.update(parameters)
        return loglik, np.asarray(y_pred, dtype=float)

    def fit_predict(
        self, y_train: np.ndarray, X_train: np.ndarray, X_test: np.ndarray
    ) -> np.ndarray:
        """Marginal (fixed-effects) prediction, for API compatibility.

        Args:
            y_train: Dependent variable on the training rows.
            X_train: Training design matrix, fixed-effect columns followed by
                the participant and item codes.
            X_test: The same columns on the held-out rows.
        """
        y_dummy = np.zeros(X_test.shape[0])
        _, y_pred = self.fit_score(y_train, X_train, X_test, y_dummy, is_full=True)
        return y_pred
