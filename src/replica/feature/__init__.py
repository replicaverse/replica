"""Feature models, extractors, alignment, and composition utilities.

Models produce per-unit output records; extractors turn them into predictors.
Core helpers cover annotation, composition, and output quantities. Numerical
diagnostics are available through the lazily imported diagnostics package.
"""

# Re-export the per-concept predictors so they can be imported straight from
# ``replica.feature``.
from replica.contracts.features import (
    Feature,
    FeatureBlock,
    FeatureIndex,
    FeatureValues,
)
from replica.feature.core.annotation.alignment import (
    TokenAlignment,
    TokenBoundaryError,
    aggregate_nested,
    align_by_spans,
    align_from_tokenizer,
    nested_members,
    tokenize_spans,
)
from replica.feature.core.binding import ModelFeature, named
from replica.feature.core.composition.aggregate import aggregate
from replica.feature.core.composition.columns import PrecomputedFeature
from replica.feature.core.composition.context import PrecedingContextModel
from replica.feature.core.composition.controls import (
    PrevUnitLength,
    UnitLength,
    UnitPosition,
)
from replica.feature.core.composition.models import (
    MergedFeatureModel,
    PrecomputedFeatureModel,
)
from replica.feature.core.composition.transforms import (
    DerivedFeature,
    Interaction,
    InUnit,
    Lag,
)
from replica.feature.resolution import resolve, resolve_features

from .backends import HuggingFaceLM
from .core.measurement import FeatureMeasurement, describe_feature
from .core.protocols import (
    FeatureExtractor,
    FeatureModel,
    FeatureModelOutput,
    FeatureNotAvailable,
)
from .distance import (
    InformationValue,
    InformationValueLM,
    InformationValueScorer,
    NextWordInformationValue,
    ResponsivityDistance,
    ResponsivityLM,
    SimilarityAdjustedSurprisal,
    SimilarityAdjustedSurprisalEmbeddingLM,
    SimilarityAdjustedSurprisalWordLM,
)
from .embedding import HiddenState, HuggingFaceEmbedding
from .entropy import (
    ContinuationEntropyLM,
    ContinuationEntropyLMBuggy,
    ForwardMarginalEntropyLM,
    ForwardWordLookaheadEntropyLM,
    MCWordEntropyLM,
)
from .entropy.extractors import (
    ContinuationEntropy,
    ContinuationEntropyBuggy,
    Entropy,
    ExpectedProbability,
    MCWordEntropy,
    RenyiEntropy,
    SampledEntropy,
    TokenEntropy,
    TokenEntropySum,
    TokenRenyiEntropy,
    TokenRenyiEntropySum,
)
from .frequency import (
    FrequencySource,
    KenLMSource,
    LogFrequency,
    OpenWebTextSource,
    SampledUnigramSource,
    SubtlexSource,
    UnigramSurprisal,
    UnigramSurprisalLM,
    UserTable,
    WordfreqSource,
)
from .modelblocks import (
    ItemMeasuresTable,
    ModelBlocksFeature,
    ModelBlocksRunner,
    write_sentitems,
)
from .surprisal import (
    CharBeamLM,
    ClozeLM,
    LogitLensLM,
    TunedLensLM,
    WordsProbabilityLM,
)
from .surprisal.extractors import (
    LayerSurprisal,
    Rank,
    Surprisal,
    SurprisalReduction,
)
from .surprisal.factory import surprisal_model

__all__ = [
    "FeatureBlock",
    "FeatureIndex",
    "FeatureValues",
    "FeatureExtractor",
    "ModelFeature",
    "named",
    "resolve",
    "resolve_features",
    "aggregate",
    "PrecedingContextModel",
    "CharBeamLM",
    "ClozeLM",
    "ContinuationEntropy",
    "ContinuationEntropyBuggy",
    "ContinuationEntropyLM",
    "ContinuationEntropyLMBuggy",
    "ForwardMarginalEntropyLM",
    "ForwardWordLookaheadEntropyLM",
    "DerivedFeature",
    "Entropy",
    "ExpectedProbability",
    "Feature",
    "FeatureModel",
    "FeatureModelOutput",
    "FeatureNotAvailable",
    "FrequencySource",
    "HiddenState",
    "HuggingFaceEmbedding",
    "HuggingFaceLM",
    "InformationValue",
    "InformationValueLM",
    "InformationValueScorer",
    "Interaction",
    "InUnit",
    "FeatureMeasurement",
    "describe_feature",
    "ItemMeasuresTable",
    "KenLMSource",
    "Lag",
    "LayerSurprisal",
    "LogFrequency",
    "LogitLensLM",
    "TunedLensLM",
    "MCWordEntropy",
    "MCWordEntropyLM",
    "ModelBlocksFeature",
    "ModelBlocksRunner",
    "NextWordInformationValue",
    "MergedFeatureModel",
    "OpenWebTextSource",
    "PrecomputedFeature",
    "PrecomputedFeatureModel",
    "PrevUnitLength",
    "Rank",
    "RenyiEntropy",
    "ResponsivityDistance",
    "ResponsivityLM",
    "SampledEntropy",
    "SampledUnigramSource",
    "SimilarityAdjustedSurprisal",
    "SimilarityAdjustedSurprisalEmbeddingLM",
    "SimilarityAdjustedSurprisalWordLM",
    "SubtlexSource",
    "Surprisal",
    "SurprisalReduction",
    "TokenAlignment",
    "TokenBoundaryError",
    "TokenEntropy",
    "TokenEntropySum",
    "TokenRenyiEntropy",
    "TokenRenyiEntropySum",
    "UnigramSurprisal",
    "UnigramSurprisalLM",
    "UnitLength",
    "UnitPosition",
    "UserTable",
    "WordfreqSource",
    "WordsProbabilityLM",
    "aggregate_nested",
    "nested_members",
    "align_by_spans",
    "align_from_tokenizer",
    "tokenize_spans",
    "surprisal_model",
    "write_sentitems",
]


def __getattr__(name: str):
    # Kept out of the eager imports so the numpy-only fingerprint engine
    # never loads unless asked for.
    if name == "diagnostics":
        from importlib import import_module

        value = import_module("replica.feature.diagnostics")
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
