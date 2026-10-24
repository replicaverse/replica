"""Declared training-fold transformations of indexed quantities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from replica.contracts.features import Feature


@dataclass(frozen=True)
class Residualize:
    """Subtract a training-fitted projection from a selected quantity.

    Args:
        feature: Target, predictor or control to transform for fitting.
        against: Reference quantities, resolved without adding model terms.
        policy: centered_slope subtracts a centered covariance/variance slope
            times the reference, without an intercept. least_squares subtracts
            the fitted projection with the declared intercept and rank policy.
        intercept: Include an intercept for least_squares; disallowed for
            centered_slope.
        rank: Reject rank-deficient least-squares references or use their
            minimum-norm projection. centered_slope retains a zero slope for
            a constant reference.
    """

    feature: Feature
    against: tuple[Feature, ...]
    policy: Literal["centered_slope", "least_squares"] = field(kw_only=True)
    intercept: bool = False
    rank: Literal["raise", "minimum_norm"] = "raise"

    def __post_init__(self) -> None:
        object.__setattr__(self, "against", tuple(self.against))
        if not self.against or any(f is self.feature for f in self.against):
            raise ValueError("Residualize needs references distinct from its feature")
        if len({f.name for f in self.against}) != len(self.against):
            raise ValueError("Residualize reference names must be unique")
        if self.policy not in ("centered_slope", "least_squares"):
            raise ValueError("Unknown residualization policy")
        if self.rank not in ("raise", "minimum_norm"):
            raise ValueError("rank must be 'raise' or 'minimum_norm'")
        if not isinstance(self.intercept, bool):
            raise TypeError("intercept must be boolean")
        if self.policy == "centered_slope" and (
            len(self.against) != 1 or self.intercept
        ):
            raise ValueError("centered_slope requires one reference and no intercept")

    def configuration(self) -> dict:
        """Return the projection policy and named inputs."""
        return {
            "operation": "residualize",
            "feature": self.feature.name,
            "against": [f.name for f in self.against],
            "policy": self.policy,
            "intercept": self.intercept,
            "rank": self.rank,
        }

    def apply(self, values, references, train, test):
        """Fit on training rows and transform both partitions.

        Args:
            values: Complete raw column to transform.
            references: Complete reference columns in declared order.
            train: Training-row mask.
            test: Held-out-row mask.
        """
        return _apply_projection(self.configuration(), values, references, train, test)


def _apply_projection(configuration, values, references, train, test):
    """Apply a detached projection declaration to one training/test split."""
    if configuration["policy"] == "centered_slope":
        reference = references[0]
        beta = _orth_beta(reference[train], values[train])
        return (
            values[train] - beta * reference[train],
            values[test] - beta * reference[test],
            {"beta": [float(beta)], "intercept": 0.0},
        )
    x = np.column_stack(references)
    if configuration["intercept"]:
        x = np.column_stack([np.ones(len(values)), x])
    beta, _, rank, _ = np.linalg.lstsq(x[train], values[train], rcond=None)
    if configuration["rank"] == "raise" and rank != x.shape[1]:
        raise ValueError("Residualize references are rank deficient in training")
    return (
        values[train] - x[train] @ beta,
        values[test] - x[test] @ beta,
        {"beta": beta.tolist(), "rank": int(rank)},
    )


def _orth_beta(against: np.ndarray, target: np.ndarray) -> float:
    """OLS slope of *target* on *against*: ``cov(against, target)/var(against)``.

    The projection coefficient for residualizing ``target`` against
    ``against``, following Opedal et al. (2024). Mean-centering cancels the
    ddof, so this equals R's ``cov(a, t)/cov(a, a)``. Returns 0.0 for a
    constant ``against``.
    """
    a = np.asarray(against, dtype=float)
    t = np.asarray(target, dtype=float)
    am = a - a.mean()
    denom = float(np.dot(am, am))
    if denom == 0.0:
        return 0.0
    return float(np.dot(am, t - t.mean()) / denom)
