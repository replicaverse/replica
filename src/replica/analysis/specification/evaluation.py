"""Assign indexed quantities to statistical roles and observation cohorts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from replica.analysis.contracts.results import ComparisonSpec
from replica.contracts.features import Feature

if TYPE_CHECKING:
    from replica.segmentation import Segmenter

__all__ = ["ComparisonSpec", "EvaluationSpec"]


@dataclass
class EvaluationSpec:
    """Select targets, added predictors and baseline controls.

    Args:
        targets: Quantities to predict, each evaluated separately.
        predictors: Features added to the baseline.
        controls: Features included in both statistical models.
        observations: Optional row index for scalar analyses. The feature's
            values and eligibility do not select rows or become model terms.
        segmentation: Requested annotation units; None retains native units.
        comparison: Add all predictors jointly or evaluate each individually.
        cohort: Individual contrasts use separate complete cases by default;
            shared uses one eligible cohort and fold schedule per target.
    """

    targets: list[Feature]
    predictors: list[Feature]
    controls: list[Feature] = field(default_factory=list)
    observations: Feature | None = None
    segmentation: Segmenter | None = None
    comparison: Literal["joint", "individual"] = "joint"
    cohort: Literal["separate", "shared"] = "separate"

    def __post_init__(self) -> None:
        if not self.targets:
            raise ValueError("EvaluationSpec requires at least one target")
        if self.comparison not in ("joint", "individual"):
            raise ValueError("comparison must be 'joint' or 'individual'")
        if self.cohort not in ("separate", "shared"):
            raise ValueError("cohort must be 'separate' or 'shared'")
        selected = [*self.targets, *self.predictors, *self.controls]
        if self.observations is not None:
            selected.append(self.observations)
        for feature in selected:
            if not (
                isinstance(getattr(feature, "name", None), str)
                and feature.name
                and callable(getattr(feature, "resolve", None))
                and hasattr(feature, "dependencies")
            ):
                raise TypeError("Evaluation terms must implement the Feature protocol")
        names = [feature.name for feature in (*self.predictors, *self.controls)]
        if len(names) != len(set(names)):
            raise ValueError("Predictor and control names must be unique")
        target_names = [feature.name for feature in self.targets]
        if len(target_names) != len(set(target_names)):
            raise ValueError("Target names must be unique")
        if set(target_names) & set(names):
            raise ValueError("Targets and input columns need distinct names")
