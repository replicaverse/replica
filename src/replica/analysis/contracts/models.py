"""Backend protocols and fold-score contracts."""

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class FoldScore:
    """What a model that owns its held-out likelihood reports for one fold.

    ``per_obs_loglik`` is optional: models whose likelihood is a per-group
    block (a marginal mixed-effects likelihood, say) have no per-observation
    decomposition, so the fold carries no per-observation delta.
    """

    loglik: float
    y_pred: np.ndarray
    per_obs_loglik: np.ndarray | None = None


def require_training_rows(
    y_train: np.ndarray,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> None:
    """Refuse held-out rows for a score that is the training fit's own likelihood.

    Summed over the folds of a held-out splitter, such a score counts each row
    once per fold that trains on it; :class:`FullDataSplitter` scores the
    training rows themselves.

    Args:
        y_train: Response the model is fit to.
        X_train: Design the model is fit to.
        X_test: Design the score is reported for; must equal ``X_train``.
        y_test: Response the score is reported for; must equal ``y_train``.

    Raises:
        ValueError: When the scored rows differ from the training rows.
    """
    same = (
        X_test.shape == X_train.shape
        and np.array_equal(X_test, X_train, equal_nan=True)
        and np.shape(y_test) == np.shape(y_train)
        and np.array_equal(y_test, y_train, equal_nan=True)
    )
    if not same:
        raise ValueError(
            "integrated_loglik=True scores the training fit's own likelihood, "
            "so the scored rows must be the training rows; evaluate it with "
            "FullDataSplitter"
        )


@runtime_checkable
class AnalysisModel(Protocol):
    """Matrix backend supplying predictions for Gaussian scoring.

    Used with ``AnalysisConfig(design=None)``. Optional scale and coefficient
    methods are described in the analysis package's backend extension guide.
    Named designs use ``GroupedModel``; backend-owned scores use ``LikelihoodModel``.
    """

    name: str

    def fit_predict(
        self,
        y_train: np.ndarray,
        X_train: np.ndarray,
        X_test: np.ndarray,
    ) -> np.ndarray:
        """Return predicted y for the test set, shape (n_test,).

        Args:
            y_train: Dependent variable on the fold's training rows, already
                transformed and scaled by ``evaluate()``.
            X_train: Training design matrix, one column per predictor, with no
                intercept column; implementations add their own.
            X_test: The same columns on the fold's held-out rows.
        """
        ...


@runtime_checkable
class LikelihoodModel(Protocol):
    """Matrix backend supplying its own fold likelihood with ``owns_loglik=True``.

    ``fit_score`` receives training responses/design, test design/responses, and
    keyword ``is_full``. Event backends additionally receive the response timeline
    and split masks; the variadic signature accommodates those required inputs.
    No ``fit_predict`` method is required.
    """

    name: str
    owns_loglik: bool

    def fit_score(
        self, *args: Any, **kwargs: Any
    ) -> FoldScore | tuple[float, np.ndarray]:
        """Fit and return held-out log likelihood and aligned predictions.

        Args:
            args: Training responses/design followed by test design/responses.
            kwargs: ``is_full`` and any declared event inputs.
        """
        ...


@runtime_checkable
class StatsPredictor(Protocol):
    """Optional model capability returning predictions and coefficient estimates."""

    def fit_predict_with_stats(
        self, y_train: np.ndarray, X_train: np.ndarray, X_test: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Fit once and report predictions, coefficients, standard errors and sigma.

        Args:
            y_train: Training response.
            X_train: Training design.
            X_test: Test design.
        """
        ...


@runtime_checkable
class FitDiagnosticProvider(Protocol):
    """Optional compact diagnostics for the latest successful backend fit.

    Reset the current record before each fit attempt, including fallbacks.
    Return independent metadata, never a cumulative history or live fit object.
    Evaluation snapshots each side immediately after its fit. Missing fields
    and ``None`` mean unknown, not successful convergence.
    The record describes a completed training fit; a later prediction or scoring
    error can leave that fit's record available. Failed training attempts must
    not expose a previous fit's record.

    Common fields are ``converged``, ``convergence_applicable`` (false for
    direct solvers), ``singular``, ``rank`` and ``n_columns`` (both including
    the intercept), ``selected_parameters``, ``warnings``, and backend-specific
    ``details``. Include only recorded facts; no refit is required. Grouped
    fits carry their authoritative diagnostics in ``GroupedFit.metadata``.
    """

    def last_fit_diagnostics(self) -> dict | None:
        """Return the current fit's diagnostic snapshot, or None if unavailable."""
        ...
