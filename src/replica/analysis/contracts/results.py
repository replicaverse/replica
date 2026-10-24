"""Scalar model-comparison and result contracts."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class ComparisonSpec:
    """The named baseline and added terms of one model comparison.

    Args:
        baseline: Covariates shared by both models, in design-column order.
        added: Predictors added together to the full model.
    """

    baseline: tuple[str, ...]
    added: tuple[str, ...]

    @property
    def name(self) -> str:
        """Display name that distinguishes a joint comparison from one term."""
        return (
            self.added[0] if len(self.added) == 1 else f"joint({', '.join(self.added)})"
        )

    @property
    def identity(self) -> str:
        """Unambiguous serialized contrast, including its baseline."""
        import json

        return json.dumps(
            {"baseline": self.baseline, "added": self.added}, separators=(",", ":")
        )


@dataclass(frozen=True)
class Coefficient:
    """Across-fold mean coefficient and mean within-fit standard error.

    ``standard_error`` is a compatibility field, not the sampling uncertainty
    of ``estimate``. Use ``mean_within_fit_se`` for descriptive reporting.
    """

    estimate: float
    standard_error: float

    @property
    def mean_within_fit_se(self) -> float:
        """Arithmetic mean of per-fit coefficient standard errors."""
        return self.standard_error


@dataclass
class AnalysisResult:
    """Result of one cross-validated model comparison.

    Scalar evaluation reports full-minus-baseline held-out log likelihood in
    ``delta_ll``, in nats, with the splitter's repeated-CV normalization.
    ``delta_ll_per_unit`` divides this gain by the effective scored-observation
    count. ``AnalysisConfig.score_scale`` selects the response scale; changing
    the response transform does not change the information unit.

    ``coefficient`` is the across-fold mean slope on the scaled feature, in
    fitted-response units. A joint comparison reports the first added predictor's
    slope here and all slopes under ``coefficients``.
    """

    dataset_name: str
    model_name: str
    feature_name: str
    dv_name: str | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    n_observations: int = 0
    # Effective held-out rows after repeated-CV normalization.
    n_scored: int = 0
    comparison: ComparisonSpec | None = None
    coefficients: dict[str, Coefficient] = field(default_factory=dict)
    coefficient: float = float("nan")
    coefficient_se: float = float("nan")
    language: str | None = None
    fold_results: list[FoldResult] | None = None
    response: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    @property
    def coefficient_mean_within_fit_se(self) -> float:
        """Mean per-fit SE; ``coefficient_se`` retains the compatibility value."""
        return self.coefficient_se

    @property
    def baseline_ll(self) -> float:
        """Held-out baseline-model log-likelihood (nats, summed)."""
        return self.metrics.get("baseline_ll", float("nan"))

    @property
    def full_ll(self) -> float:
        """Held-out full-model log-likelihood (nats, summed)."""
        return self.metrics.get("full_ll", float("nan"))

    @property
    def delta_ll(self) -> float:
        """Full minus baseline held-out log-likelihood (nats, summed)."""
        return self.metrics.get("delta_ll", float("nan"))

    @property
    def delta_ll_per_unit(self) -> float:
        """``delta_ll`` per held-out observation (nats/observation)."""
        return self.metrics.get("delta_ll_per_unit", float("nan"))

    @property
    def delta_r2(self) -> float:
        """Held-out R² gain of the full model over the baseline."""
        return self.metrics.get("delta_r2", float("nan"))


@dataclass
class FoldResult:
    """Per-fold metrics from a cross-validation run.

    The ``_ll`` fields store the sum over test observations.
    Use ``mean_delta_ll`` for the per-observation average
    (matching conventions in e.g. Pimentel et al. (2023)).
    """

    fold_index: int
    n_test: int
    baseline_ll: float
    full_ll: float
    delta_ll: float
    baseline_r2: float
    full_r2: float
    delta_r2: float
    metadata: dict = field(default_factory=dict)
    per_obs_delta_ll: np.ndarray | None = None
    """Per-observation held-out delta log-likelihood (full minus baseline,
    one entry per test observation; sums to ``delta_ll``). Populated only
    when ``AnalysisConfig.retain_per_obs_delta_ll`` is set. Input to the
    REPLICA-native per-datapoint permutation test."""
    factor_ids: dict[str, np.ndarray] = field(default_factory=dict)
    """Named Design labels aligned with retained held-out loss differences."""
    row_ids: np.ndarray | None = None
    """Participant, stimulus and unit identities in loss-vector order."""
    observations: dict[str, np.ndarray] | None = None
    """Optional aligned model-scale responses, predictions and split membership."""
    observation_keys: tuple[tuple, ...] | None = None
    """Exact prepared observation coordinates; names are in observation_capture metadata."""

    @property
    def mean_delta_ll(self) -> float:
        """Per-observation mean delta log-likelihood."""
        return self.delta_ll / self.n_test if self.n_test > 0 else 0.0
