"""Numerical diagnostics for predictor targets and stochastic estimators.

Fingerprinting compares outputs with reference quantities. Sampling diagnostics
measure bias, variance, and interval coverage over a finite sample-size grid.
Torch-backed reference models and causal-LM adapters are imported separately.
"""

from replica.feature.diagnostics.references.markov import (
    DEFAULT_INITIAL,
    DEFAULT_TRANSITION,
    DEFAULT_VOCAB,
    MarkovReferenceLM,
    make_stimulus,
)
from replica.feature.diagnostics.references.monte_carlo import simulate_mc_entropy
from replica.feature.diagnostics.sampling.convergence import (
    ConvergenceReport,
    mc_convergence,
)
from replica.feature.diagnostics.sampling.estimator import (
    EstimatorReport,
    NLevel,
    diagnose_estimator,
)

from .fingerprint import FingerprintReport, SpecMatch, fingerprint
from .specs import TargetSpec, assert_identifiable, default_specs

__all__ = [
    "DEFAULT_INITIAL",
    "DEFAULT_TRANSITION",
    "DEFAULT_VOCAB",
    "ConvergenceReport",
    "EstimatorReport",
    "FingerprintReport",
    "MarkovReferenceLM",
    "NLevel",
    "SpecMatch",
    "TargetSpec",
    "assert_identifiable",
    "default_specs",
    "diagnose_estimator",
    "fingerprint",
    "make_stimulus",
    "mc_convergence",
    "simulate_mc_entropy",
]
