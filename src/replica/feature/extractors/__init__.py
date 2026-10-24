"""Public per-unit feature extractors and composition operators.

Reproduction-only estimators are excluded from this catalogue. Import them
explicitly from replica.feature; their opt-in and warnings still apply.
"""

from replica.contracts.features import Feature
from replica.feature.core.composition.columns import PrecomputedFeature
from replica.feature.core.composition.controls import (
    PrevUnitLength,
    UnitLength,
    UnitPosition,
)
from replica.feature.core.composition.transforms import (
    DerivedFeature,
    Interaction,
    InUnit,
    Lag,
)

from ..core.protocols import FeatureNotAvailable
from ..distance.extractors import ResponsivityDistance
from ..entropy.extractors import (
    ContinuationEntropy,
    Entropy,
    ExpectedProbability,
    MCWordEntropy,
    RenyiEntropy,
    SampledEntropy,
    TokenEntropy,
    TokenRenyiEntropy,
)
from ..surprisal.extractors import (
    LayerSurprisal,
    Rank,
    Surprisal,
    SurprisalReduction,
)

__all__ = [
    "ContinuationEntropy",
    "DerivedFeature",
    "Entropy",
    "ExpectedProbability",
    "Feature",
    "FeatureNotAvailable",
    "Interaction",
    "InUnit",
    "Lag",
    "LayerSurprisal",
    "MCWordEntropy",
    "PrecomputedFeature",
    "PrevUnitLength",
    "Rank",
    "RenyiEntropy",
    "ResponsivityDistance",
    "SampledEntropy",
    "Surprisal",
    "SurprisalReduction",
    "TokenEntropy",
    "TokenRenyiEntropy",
    "UnitLength",
    "UnitPosition",
]
