"""Linear mixed-effects regression through the Rust port of lme4.

``RustLmerModel`` fits the same formulas as :class:`RLmerModel` with the
``lme_python`` binding of the ``lme-rs`` crate, which needs neither R nor rpy2.
The two backends share the design-matrix contract (fixed-effect columns
followed by integer participant and item codes), the random-effects options
and the formula builder, so a configuration moves between them unchanged.

Requires the ``lmer-rs`` extra. The published package is a source
distribution, so installing it compiles the crate with a Rust toolchain. Its
extension links BLAS dynamically; :func:`import_lme_python` loads a shared
CBLAS library before the import when the module needs one.
"""

from __future__ import annotations

import ctypes
import importlib
import warnings
from importlib.metadata import version
from types import ModuleType

import numpy as np

from replica.analysis.models.fixed_effects._least_squares import normalized_design
from replica.analysis.models.fixed_effects.ols import OLS
from replica.analysis.models.mixed_effects.options import (
    LmerOptions,
    random_effect_terms,
)
from replica.analysis.models.support.diagnostics import (
    _coefficient_snapshot,
    _residual_scale,
)
from replica.utils.arrays import real_array_with_missing

_CBLAS_CANDIDATES = ("libcblas.so.3", "libcblas.so", "libopenblas.so.0", "libblas.so.3")


def import_lme_python() -> ModuleType:
    """Import ``lme_python``, loading a CBLAS library first when it needs one.

    The compiled extension resolves ``cblas_*`` symbols at load time. When a
    plain import fails on such a symbol, each candidate library is loaded with
    ``RTLD_GLOBAL`` and the import is retried.

    Raises:
        ImportError: When the package is absent or no candidate BLAS resolves
            the missing symbol.
    """
    try:
        return importlib.import_module("lme_python")
    except ImportError as exc:
        if "cblas" not in str(exc):
            raise ImportError(
                "lme_python is required for RustLmerModel; install the"
                " 'lmer-rs' extra (needs a Rust toolchain)"
            ) from exc
        first = exc
    for name in _CBLAS_CANDIDATES:
        try:
            ctypes.CDLL(name, mode=ctypes.RTLD_GLOBAL)
        except OSError:
            continue
        try:
            return importlib.import_module("lme_python")
        except ImportError:
            continue
    raise ImportError(
        "lme_python needs a CBLAS shared library; none of"
        f" {_CBLAS_CANDIDATES} could be loaded"
    ) from first


def _independent_columns(X: np.ndarray) -> list[int]:
    """Indices of the fixed columns that add rank beside an intercept.

    Columns are visited in order and a column that leaves the rank of the
    intercept-augmented design unchanged is dropped, which keeps the earlier
    of two dependent columns, as lme4 does.

    Args:
        X: Fixed-effect columns without an intercept.
    """
    n = X.shape[0]
    basis = [np.ones(n)]
    kept: list[int] = []
    rank = 1
    for j in range(X.shape[1]):
        candidate = np.column_stack([*basis, X[:, j]])
        normalized, _ = normalized_design(candidate)
        new_rank = int(np.linalg.matrix_rank(normalized))
        if new_rank > rank:
            basis.append(X[:, j])
            kept.append(j)
            rank = new_rank
    return kept


def _term_shapes(random_terms: list[str]) -> list[tuple[str, int]]:
    """``(grouping column, effects per level)`` for each random-effect term.

    Args:
        random_terms: Terms as :func:`random_effect_terms` writes them.
    """
    shapes = []
    for term in random_terms:
        lhs, group = term.strip("()").split("|")
        k = len([part for part in lhs.split("+") if part.strip() != "0"])
        shapes.append((group.strip(), k))
    return shapes


def _theta_diagonal_indices(random_terms: list[str]) -> list[int]:
    """Positions of the variance (diagonal) entries in the crate's theta.

    Theta holds one block per random-effect term in formula order; a block
    for ``k`` effects stores its lower-triangular factor column-major, so the
    diagonal of column ``j`` sits ``k - j`` entries after the previous one.

    Args:
        random_terms: Terms as :func:`random_effect_terms` writes them.
    """
    indices: list[int] = []
    offset = 0
    for _, k in _term_shapes(random_terms):
        for j in range(k):
            indices.append(offset)
            offset += k - j
    return indices


def _check_design(
    X_fixed: np.ndarray, random_terms: list[str], levels: dict[str, int]
) -> None:
    """The design checks ``lmerControl`` runs before a fit, at its defaults.

    Args:
        X_fixed: Fixed-effect columns without an intercept.
        random_terms: Terms as :func:`random_effect_terms` writes them.
        levels: Distinct levels per grouping column.

    Raises:
        ValueError: When a grouping factor has at least as many levels as
            there are observations, or a term has at least as many random
            effects as observations (``check.nobs.vs.nlev`` and
            ``check.nobs.vs.nRE``).
    """
    n = X_fixed.shape[0]
    for group, k in _term_shapes(random_terms):
        if n <= levels[group]:
            raise ValueError(
                f"number of levels of {group} ({levels[group]}) must be below"
                f" the number of observations ({n})"
            )
        if n <= levels[group] * k:
            raise ValueError(
                f"number of observations ({n}) <= number of random effects"
                f" ({levels[group] * k}) for the {group} term; the random-effects"
                " parameters and the residual variance are unidentifiable"
            )
    continuous = [
        j
        for j in range(X_fixed.shape[1])
        if not np.isin(X_fixed[:, j], (0.0, 1.0)).all()
    ]
    if continuous:
        with np.errstate(divide="ignore"):
            log_sd = np.log(X_fixed[:, continuous].std(axis=0, ddof=1))
        spread = np.abs(log_sd[:, None] - log_sd[None, :])
        if np.any(np.abs(log_sd) > np.log(1000)) or np.any(spread > np.log(1000)):
            warnings.warn(
                "Some predictor variables are on very different scales:"
                " consider rescaling",
                stacklevel=4,
            )


class RustLmerModel(LmerOptions):
    """lme4-style mixed-effects regression fitted by the ``lme-rs`` crate.

    Takes the options of :class:`LmerOptions` and fits the formula
    :class:`RLmerModel` would. Held-out predictions use the fixed effects
    only, as ``predict(re.form=NA)`` does. A design whose grouping factors
    have one level each is fit by ordinary least squares.

    The crate runs none of lme4's design checks, so they are applied here:
    dependent fixed columns are dropped (later ones first) and ``beta`` and
    ``se`` read NaN at their positions; more grouping levels or random effects
    than rows raise;
    predictor scales that differ by more than 1000x warn; a singular fit
    (a relative-covariance diagonal below 1e-4) sets :attr:`last_singular`;
    a fit that has not converged raises. Missing values are rejected by the
    crate. In lme-rs 0.2.5 an uncorrelated slope term together with a second
    grouping factor stalls, and correlated slopes on both factors are slow
    with thousands of levels; those designs stay on :class:`RLmerModel`.
    """

    name = "lmer_rs"
    #: Whether the most recent mixed fit was singular by lme4's rule.
    last_singular: bool | None = False
    _ols = OLS(sigma_correction="df")

    def configuration(self) -> dict:
        """Return declared options without fitted state."""
        return {
            "max_iterations": self.max_iterations,
            "tolerance": self.tolerance,
            "max_inner_iterations": self.max_inner_iterations,
            **super().configuration(),
        }

    def __init__(
        self,
        *,
        max_iterations: int = 1000,
        tolerance: float = 1e-6,
        max_inner_iterations: int | None = None,
        **options,
    ):
        """Configure optimization bounds and mixed-model terms.

        Args:
            max_iterations: Maximum iterations per outer search stage.
            tolerance: Positive optimization tolerance; convergence still needs inspection.
            max_inner_iterations: Optional bound on each inner optimization.
            **options: Structural and likelihood settings from LmerOptions.
        """
        super().__init__(**options)
        for name, value in (
            ("max_iterations", max_iterations),
            ("max_inner_iterations", max_inner_iterations),
        ):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 1
            ):
                raise ValueError(f"{name} must be a positive integer")
        if not np.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("tolerance must be finite and positive")
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.max_inner_iterations = max_inner_iterations
        self.last_diagnostics: dict = {}

    def _control(self, lme):
        """Refuse the known 0.2.5 stall before entering native fitting."""
        installed = version("lme_python")
        groups = [g for g, _ in _term_shapes(self._random_terms)]
        if (
            installed == "0.2.5"
            and len(set(groups)) > 1
            and len(groups) > len(set(groups))
        ):
            raise ValueError(
                "lme-rs 0.2.5 stalls with uncorrelated slopes and multiple grouping factors; use RLmerModel"
            )
        return lme.FitControl(
            max_iterations=self.max_iterations,
            tolerance=self.tolerance,
            max_inner_iterations=self.max_inner_iterations,
            require_convergence=False,
        )

    def _frame(self, y: np.ndarray, X: np.ndarray):
        """The training frame, its formula and whether it has random terms."""
        y = real_array_with_missing(y, name="y_train")
        X = real_array_with_missing(X, name="X_train")

        import pandas as pd

        n_fixed = X.shape[1] - 2
        kept = _independent_columns(X[:, :n_fixed])
        self._kept = kept
        # Every column stays in the frame for the random-slope terms; only the
        # fixed part of the formula loses the dependent ones.
        all_terms = [f"x{i}" for i in range(n_fixed)]
        fixed_terms = [f"x{i}" for i in kept]
        data = {name: np.asarray(X[:, i], float) for i, name in enumerate(all_terms)}
        data["y"] = np.asarray(y, float)
        participants = X[:, -2].astype(int)
        items = X[:, -1].astype(int)
        data["participant_id"] = participants.astype(str)
        data["item_id"] = items.astype(str)
        re_terms: list[str] = []
        re_design = self._re_design_rows(len(y))
        if re_design is not None:
            for j in range(re_design.shape[1]):
                data[f"r{j}"] = re_design[:, j]
                re_terms.append(f"r{j}")
        levels = {
            "participant_id": len(np.unique(participants)),
            "item_id": len(np.unique(items)),
        }
        random_terms = random_effect_terms(
            self,
            all_terms,
            re_terms,
            n_participants=levels["participant_id"],
            n_items=levels["item_id"],
        )
        if random_terms:
            _check_design(X[:, kept], random_terms, levels)
        self._random_terms = random_terms
        fixed_str = " + ".join(fixed_terms) if fixed_terms else "1"
        formula = f"y ~ {fixed_str}"
        if random_terms:
            formula += " + " + " + ".join(random_terms)
        return pd.DataFrame(data), formula, bool(random_terms)

    def _fit(self, y: np.ndarray, X: np.ndarray, reml: bool):
        """Fit the mixed model; ``None`` when the design has no random term."""
        self._fit_diagnostics = None
        self.last_diagnostics = {}
        self.last_singular = None
        frame, formula, mixed = self._frame(y, X)
        if not mixed:
            return None
        lme = import_lme_python()
        fit = lme.lmer(formula, data=frame, reml=reml, control=self._control(lme))
        self.last_diagnostics = dict(fit.diagnostics or {})
        if not fit.converged:
            raise RuntimeError(
                f"lme-rs did not converge after {fit.iterations} iterations on"
                f" {formula!r}; uncorrelated slopes on two grouping factors are"
                " known to stall in lme-rs 0.2.5, use RLmerModel for that design"
            )
        for message in fit.diagnostics.get("warnings") or ():
            warnings.warn(f"lme-rs: {message}", stacklevel=3)
        theta = np.asarray(fit.theta, dtype=float)
        self.last_singular = bool(
            np.any(theta[_theta_diagonal_indices(self._random_terms)] < 1e-4)
        )
        beta = np.asarray(fit.coefficients, dtype=float)
        se = np.asarray(fit.std_errors, dtype=float)
        self._last_fixed = self._in_design_order(beta, se, X.shape[1] - 2)
        self._fit_diagnostics = {
            "convergence_applicable": True,
            "converged": bool(fit.converged),
            "singular": self.last_singular,
            "n_columns": len(self._kept) + 1,
            "coefficients": _coefficient_snapshot(*self._last_fixed),
            "residual_scale": _residual_scale(
                variance=fit.sigma2, convention="REML" if reml else "ML"
            ),
            "warnings": list(self.last_diagnostics.get("warnings") or ()),
            "details": {**self.last_diagnostics, "n_iterations": int(fit.iterations)},
        }
        return fit

    def _in_design_order(
        self, beta: np.ndarray, se: np.ndarray, n_fixed: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Kept-column estimates spread to ``[intercept, x0, ...]``, NaN where dropped.

        Args:
            beta: Intercept then one estimate per kept column.
            se: Standard errors in the same order.
            n_fixed: Fixed-effect columns in the design, kept or not.
        """
        positions = [0, *(1 + column for column in self._kept)]
        full_beta = np.full(1 + n_fixed, np.nan)
        full_se = np.full(1 + n_fixed, np.nan)
        full_beta[positions] = beta
        full_se[positions] = se
        return full_beta, full_se

    def _fixed_predictions(self, beta: np.ndarray, X_test: np.ndarray) -> np.ndarray:
        """Fixed-effects predictions from the design-ordered coefficients."""
        return beta[0] + X_test[:, self._kept] @ beta[1:][self._kept]

    def fit_predict_with_stats(
        self,
        y_train: np.ndarray,
        X_train: np.ndarray,
        X_test: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """``(y_pred, beta, se, sigma)`` for the fixed effects.

        ``beta`` and ``se`` are ordered as ``[intercept, x0, x1, ...]``, NaN
        at a dropped dependent column.

        Args:
            y_train: Dependent variable on the training rows.
            X_train: Training design matrix, fixed-effect columns followed by
                the integer participant and item codes.
            X_test: The same columns on the held-out rows.
        """
        self._fit_diagnostics = None
        X_test = real_array_with_missing(X_test, name="X_test")
        fit = self._fit(y_train, X_train, self.reml)
        if fit is None:
            preds, beta, se, sigma = self._ols.fit_predict_with_stats(
                y_train, X_train[:, self._kept], X_test[:, self._kept]
            )
            beta, se = self._in_design_order(beta, se, X_train.shape[1] - 2)
            self._last_fixed = (beta.copy(), se.copy())
            self._fit_diagnostics = self._ols.last_fit_diagnostics()
            assert self._fit_diagnostics is not None
            self._fit_diagnostics["details"] = {"fallback": "no_random_terms"}
            self._fit_diagnostics["coefficients"] = _coefficient_snapshot(
                *self._last_fixed
            )
            return preds, beta, se, sigma
        beta, se = self._last_fixed
        sigma = float(np.sqrt(fit.sigma2))
        return self._fixed_predictions(beta, X_test), beta.copy(), se.copy(), sigma

    def fit_predict_with_sigma(
        self,
        y_train: np.ndarray,
        X_train: np.ndarray,
        X_test: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """Fixed-effects predictions and the fit's residual standard error.

        Args:
            y_train: Dependent variable on the training rows.
            X_train: Training design matrix, fixed-effect columns followed by
                the integer participant and item codes.
            X_test: The same columns on the held-out rows.
        """
        preds, _, _, sigma = self.fit_predict_with_stats(y_train, X_train, X_test)
        return preds, sigma

    def _fit_loglik_fitted(
        self, y: np.ndarray, X: np.ndarray, REML: bool = False
    ) -> tuple[float, np.ndarray]:
        """One fit, returning ``(log-likelihood, in-sample fitted values)``."""
        fit = self._fit(y, X, REML)
        if fit is None:
            preds, beta, se, _ = self._ols.fit_predict_with_stats(
                y, X[:, self._kept], X[:, self._kept]
            )
            self._last_fixed = self._in_design_order(beta, se, X.shape[1] - 2)
            self._fit_diagnostics = self._ols.last_fit_diagnostics()
            assert self._fit_diagnostics is not None
            self._fit_diagnostics["details"] = {"fallback": "no_random_terms"}
            self._fit_diagnostics["coefficients"] = _coefficient_snapshot(
                *self._last_fixed
            )
            resid = y - preds
            n = len(y)
            sigma2 = float(np.sum(resid**2) / n)
            loglik = -0.5 * n * (np.log(2 * np.pi * sigma2) + 1)
            return loglik, preds
        return float(fit.log_likelihood), np.asarray(fit.fitted, dtype=float)
