"""Crossed random-effects lmer that degrades its RE structure on a failed fit.

Wraps :class:`RLmerModel` with by-participant and by-item random slopes on a
single focal predictor. lme4 can fail outright on a fold whose maximal
random-effects covariance is not positive definite (an ``.dsy2dpo``/``.M2sym``
error), especially with uncorrelated ``||`` slopes on measures with many tied
values. This wrapper tries the requested structure first and, on any R error,
falls back to correlated slopes and then to random intercepts only, so the
cross-validation run continues past one ill-conditioned fold.
"""

from __future__ import annotations

import numpy as np

from replica.analysis.models.support.diagnostics import _FitDiagnostics


class RobustCrossedLmer(_FitDiagnostics):
    """Focal-slope crossed lmer with automatic random-effects fallback.

    The full design's last fixed column (index ``n_controls``) receives a
    by-subject and by-item random slope; the baseline design, which lacks that
    column, keeps random intercepts only. Held-out predictions for unseen
    participants and items come from the fixed effects either way, so every
    fallback scores the same quantity, fixed-effect predictions, from a
    different estimate.

    Scoring goes through :meth:`fit_predict_with_sigma` rather than the
    coefficient path: ``summary()``/``vcov`` triggers the ``.dsy2dpo`` failure on
    singular folds, and a held-out delta log-likelihood needs only predictions
    and sigma (hence ``provides_stats = False``). Sets ``needs_grouping`` so
    ``evaluate()`` appends the participant and item code columns.

    Args:
        n_controls: Number of control columns in the design; fixes the focal
            predictor's column index (the column after the controls), where the
            random slopes are placed.
        uncorrelated: Whether the primary structure uses uncorrelated ``||``
            slopes. The first fallback uses the opposite correlation setting.
        reml: Fit by REML rather than ML. Forwarded to every variant.
    """

    name = "lmer_robust"
    needs_grouping = True
    owns_loglik = False
    provides_stats = False

    def configuration(self) -> dict:
        """Return declared options without fitted state."""
        return {"variants": [model.configuration() for model in self._variants]}

    def __init__(
        self, n_controls: int, *, uncorrelated: bool = True, reml: bool = False
    ):
        """Build the primary structure and its two fallbacks."""
        from replica.analysis.models.mixed_effects.lmer import RLmerModel

        slopes = [n_controls]
        # The optimizer retry stays off in every variant: only a hard R error
        # moves the fit to the next structure.
        self._variants = [
            RLmerModel(
                random_slope_cols_subject=slopes,
                random_slope_cols_item=slopes,
                uncorrelated_slopes_subject=uncorrelated,
                uncorrelated_slopes_item=uncorrelated,
                reml=reml,
                nlminb_fallback=False,
            ),
            RLmerModel(
                random_slope_cols_subject=slopes,
                random_slope_cols_item=slopes,
                uncorrelated_slopes_subject=not uncorrelated,
                uncorrelated_slopes_item=not uncorrelated,
                reml=reml,
                nlminb_fallback=False,
            ),
            RLmerModel(reml=reml, nlminb_fallback=False),  # (1|part)+(1|item)
        ]
        #: Times each variant (primary, alt-correlation, intercepts) was used.
        self.variant_uses = [0, 0, 0]

    def _first_ok(self, method: str, *args):
        """Call ``method`` on each variant in turn; raise the last R error."""
        self._fit_diagnostics = None
        from rpy2.rinterface_lib.embedded import RRuntimeError

        errors: list[RRuntimeError] = []
        for i, model in enumerate(self._variants):
            try:
                result = getattr(model, method)(*args)
            except RRuntimeError as exc:
                errors.append(exc)
                continue
            self.variant_uses[i] += 1
            diagnostics = model.last_fit_diagnostics()
            self._fit_diagnostics = {
                **(diagnostics or {}),
                "details": {
                    **((diagnostics or {}).get("details") or {}),
                    "selected_variant": i,
                    "failed_variants": [str(error) for error in errors],
                },
            }
            return result
        raise errors[-1]

    def fit_predict_with_sigma(
        self, y_train: np.ndarray, X_train: np.ndarray, X_test: np.ndarray
    ) -> tuple[np.ndarray, float]:
        """Predictions and sigma from the first variant that fits.

        Args:
            y_train: Dependent variable on the training rows.
            X_train: Training design matrix, fixed-effect columns followed by
                the integer participant and item codes.
            X_test: The same columns on the held-out rows.
        """
        return self._first_ok("fit_predict_with_sigma", y_train, X_train, X_test)

    def fit_predict(
        self, y_train: np.ndarray, X_train: np.ndarray, X_test: np.ndarray
    ) -> np.ndarray:
        """Held-out predictions from the first variant that fits.

        Args:
            y_train: Dependent variable on the training rows.
            X_train: Training design matrix, fixed-effect columns followed by
                the integer participant and item codes.
            X_test: The same columns on the held-out rows.
        """
        return self._first_ok("fit_predict", y_train, X_train, X_test)

    def fit_predict_with_stats(
        self, y_train: np.ndarray, X_train: np.ndarray, X_test: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """``(y_pred, beta, se, sigma)`` from the first variant that fits.

        Args:
            y_train: Dependent variable on the training rows.
            X_train: Training design matrix, fixed-effect columns followed by
                the integer participant and item codes.
            X_test: The same columns on the held-out rows.
        """
        return self._first_ok("fit_predict_with_stats", y_train, X_train, X_test)
