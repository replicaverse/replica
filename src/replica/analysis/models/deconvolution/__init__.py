"""Continuous-time deconvolutional regression models and data utilities."""

from replica.analysis.models.deconvolution.data import (
    CDRDesign,
    apply_impulse_ops,
    build_design,
    compute_history_intervals,
    expand_history,
)

__all__ = [
    "CDRDesign",
    "apply_impulse_ops",
    "build_design",
    "compute_history_intervals",
    "expand_history",
]
