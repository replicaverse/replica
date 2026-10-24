"""Random-effect configuration and shared mixed-model scoring."""

import numpy as np

from replica.analysis.contracts.configuration import array_identity
from replica.analysis.contracts.models import require_training_rows
from replica.analysis.models.support.diagnostics import _FitDiagnostics
from replica.analysis.models.support.prediction import _FitPredictWithSigma


def _emit_re(group: str, slope_cols: list[str], uncorrelated: bool) -> list[str]:
    """Build the ``(... | group)`` terms for one grouping factor.

    With ``uncorrelated=True`` the random intercept and each slope are written
    as separate terms with no shared correlation matrix: ``(1|g) + (0+x|g) +
    ...``.

    Args:
        group: Name of the grouping column.
        slope_cols: Fixed-effect column names that receive a random slope.
        uncorrelated: Whether slopes and intercept share a covariance matrix.
    """
    if not slope_cols:
        return [f"(1|{group})"]
    if uncorrelated:
        terms = [f"(1|{group})"]
        for c in slope_cols:
            terms.append(f"(0 + {c} | {group})")
        return terms
    slope_str = "1 + " + " + ".join(slope_cols)
    return [f"({slope_str}|{group})"]


def random_effect_terms(
    model,
    fixed_terms: list[str],
    re_terms: list[str],
    *,
    n_participants: int,
    n_items: int,
) -> list[str]:
    """The lme4 random-effect terms a mixed-model configuration asks for.

    Shared by the R and Rust lmer backends so both fit the same formula. A
    grouping factor with a single level gets no term, and slope columns outside
    the fixed design are dropped, which is what lets a baseline design that
    lacks the focal column fall back to random intercepts.

    Args:
        model: A :class:`LmerOptions` instance, read for its random-effects
            attributes.
        fixed_terms: Names of the fixed-effect columns, ``x0`` onwards.
        re_terms: Names of the ``re_design`` columns, which replace the
            by-subject slope columns when present.
        n_participants: Distinct participant codes in the training rows.
        n_items: Distinct item codes in the training rows.
    """
    n_fixed = len(fixed_terms)
    random_terms: list[str] = []
    if n_participants > 1 and model.include_subject_re:
        if re_terms:
            # Slopes come from their own columns, so they survive a model
            # that drops the matching fixed effects.
            cols = list(re_terms)
        elif model.random_slope_cols_subject is not None and fixed_terms:
            cols = [
                f"x{c}" for c in model.random_slope_cols_subject if 0 <= c < n_fixed
            ]
        elif model.random_slopes_subject and fixed_terms:
            cols = list(fixed_terms)
        else:
            cols = []
        random_terms.extend(
            _emit_re("participant_id", cols, model.uncorrelated_slopes_subject)
        )
    if n_items > 1 and model.include_item_re:
        if model.random_slope_cols_item is not None and fixed_terms:
            cols = [f"x{c}" for c in model.random_slope_cols_item if 0 <= c < n_fixed]
        elif model.random_slopes_item and fixed_terms:
            cols = list(fixed_terms)
        else:
            cols = []
        random_terms.extend(_emit_re("item_id", cols, model.uncorrelated_slopes_item))
    return random_terms


class LmerOptions(_FitDiagnostics, _FitPredictWithSigma):
    """Random-effects structure and criterion shared by the lmer backends.

    Expects the last two columns of ``X`` to be integer participant and item
    codes and sets ``needs_grouping`` so ``evaluate()`` appends them.

    Args:
        random_slopes: If True, add by-subject and by-item random slopes
            for every fixed-effect column, the "maximal" structure of Barr
            et al. (2013); the default keeps random intercepts only.
        random_slopes_subject: Overrides ``random_slopes`` for the subject side.
        random_slopes_item: Overrides ``random_slopes`` for the item side.
        random_slope_cols_subject: Restrict by-subject random slopes to these
            X column indices instead of every fixed-effect column.
        random_slope_cols_item: Same, for the item side.
        include_subject_re: Include the by-subject random-effects term; it is
            added only when there is more than one participant.
        include_item_re: Include the by-item random-effects term.
        uncorrelated_slopes_subject: Use uncorrelated (``||``) rather than
            correlated (``|``) random slopes on the subject side.
        uncorrelated_slopes_item: Same, for the item side.
        reml: Fit by REML rather than ML; the default ML matches the
            ModelBlocks lmefit recipe.
        integrated_loglik: Score with the fit's own integrated (marginal)
            likelihood. Sets ``owns_loglik``, so ``evaluate()`` calls
            :meth:`fit_score` per fold. The score is in-sample, so
            :meth:`fit_score` requires :class:`FullDataSplitter`'s identical
            training and scored rows and raises on any other split.
        re_design: Columns carrying the by-subject random slopes, replacing
            ``random_slope_cols_subject``. Use it when a slope belongs on a
            variable that is not a fixed effect of the model being fit, which
            a nested likelihood-ratio test requires: the reduced model must
            keep the full model's random-effects structure, so it carries
            slopes on predictors it does not otherwise contain. Rows must
            line up with the data the model is fit on, so pair it with
            :class:`FullDataSplitter`, which never subsets them.
    """

    needs_grouping = True
    owns_loglik = False

    def configuration(self) -> dict:
        """Return declared options without fitted state."""
        return {
            "reml": self.reml,
            "random_slopes_subject": self.random_slopes_subject,
            "random_slopes_item": self.random_slopes_item,
            "random_slope_cols_subject": self.random_slope_cols_subject,
            "random_slope_cols_item": self.random_slope_cols_item,
            "include_item_re": self.include_item_re,
            "include_subject_re": self.include_subject_re,
            "uncorrelated_slopes_subject": self.uncorrelated_slopes_subject,
            "uncorrelated_slopes_item": self.uncorrelated_slopes_item,
            "integrated_loglik": self.owns_loglik,
            "re_design": array_identity(self.re_design),
        }

    def __init__(
        self,
        *,
        random_slopes: bool = False,
        random_slopes_subject: bool | None = None,
        random_slopes_item: bool | None = None,
        random_slope_cols_subject: list[int] | None = None,
        random_slope_cols_item: list[int] | None = None,
        include_item_re: bool = True,
        include_subject_re: bool = True,
        uncorrelated_slopes_subject: bool = False,
        uncorrelated_slopes_item: bool = False,
        reml: bool = False,
        integrated_loglik: bool = False,
        re_design: np.ndarray | None = None,
    ):
        """Fix the random-effects structure and the estimation criterion."""
        self.owns_loglik = bool(integrated_loglik)
        self.reml = bool(reml)
        self.random_slopes_subject = (
            random_slopes_subject
            if random_slopes_subject is not None
            else random_slopes
        )
        self.random_slopes_item = (
            random_slopes_item if random_slopes_item is not None else random_slopes
        )
        self.random_slope_cols_subject = (
            list(random_slope_cols_subject)
            if random_slope_cols_subject is not None
            else None
        )
        self.random_slope_cols_item = (
            list(random_slope_cols_item) if random_slope_cols_item is not None else None
        )
        self.include_item_re = include_item_re
        self.include_subject_re = include_subject_re
        # Uncorrelated slopes: separate ``(0+xj|g)`` terms, with no
        # correlation matrix to estimate.
        self.uncorrelated_slopes_subject = bool(uncorrelated_slopes_subject)
        self.uncorrelated_slopes_item = bool(uncorrelated_slopes_item)
        self.re_design = None if re_design is None else np.asarray(re_design, float)
        self._last_fixed: tuple[np.ndarray, np.ndarray] | None = None

    def _re_design_rows(self, n_rows: int) -> np.ndarray | None:
        """``re_design`` checked against the rows being fit, or ``None``."""
        if self.re_design is None:
            return None
        if self.re_design.shape[0] != n_rows:
            raise ValueError(
                f"re_design has {self.re_design.shape[0]} rows but the model is "
                f"being fit on {n_rows}. Supply rows for the whole dataset and "
                "pair with FullDataSplitter, which never subsets them."
            )
        return self.re_design

    def _fit_loglik_fitted(
        self, y: np.ndarray, X: np.ndarray, REML: bool = False
    ) -> tuple[float, np.ndarray]:
        """One fit, returning ``(log-likelihood, in-sample fitted values)``."""
        raise NotImplementedError

    def fit_loglik(self, y: np.ndarray, X: np.ndarray, REML: bool = False) -> float:
        """The fit's log-likelihood on all rows, integrating the random effects.

        Args:
            y: Response variable, shape ``(n,)``.
            X: Design matrix with the two grouping columns appended.
            REML: Fit by REML rather than ML.
        """
        return self._fit_loglik_fitted(y, X, REML=REML)[0]

    def fit_score(
        self,
        y_train: np.ndarray,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        is_full: bool,
    ) -> tuple[float, np.ndarray]:
        """Fit on train and return ``(integrated log-likelihood, fitted values)``.

        The score is the training fit's own likelihood, so the scored rows
        must be the training rows (``FullDataSplitter``); the fit's fitted
        values, conditional on the random effects, are the predictions.

        Args:
            y_train: Dependent variable on the training rows.
            X_train: Training design matrix, fixed-effect columns followed by
                the integer participant and item codes.
            X_test: The scored rows' design; must equal ``X_train``.
            y_test: The scored rows' response; must equal ``y_train``.
            is_full: Whether this is the full side of the comparison; unused,
                the two sides differ by the columns ``X_train`` carries.

        Raises:
            ValueError: When the scored rows differ from the training rows.
        """
        self._fit_diagnostics = None
        require_training_rows(y_train, X_train, X_test, y_test)
        return self._fit_loglik_fitted(y_train, X_train, REML=self.reml)

    def last_fixed_effects(self) -> tuple[np.ndarray, np.ndarray] | None:
        """``(beta, se)`` of the most recent fit, or ``None`` before one."""
        return self._last_fixed
