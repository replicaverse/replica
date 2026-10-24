"""Distance predictors: observed-vs-alternatives distances.

Responsivity (embedding cosine), information value (n-gram overlap or
embedding cosine), and similarity-adjusted surprisal + next-word information
value (embedding cosine, POS identity, Levenshtein), each with its sampling
LM and Feature extractor. Each module header carries its paper.
"""

from .extractors import (
    NextWordInformationValue,
    ResponsivityDistance,
    SimilarityAdjustedSurprisal,
)
from .information_value import (
    InformationValue,
    InformationValueLM,
    InformationValueScorer,
)
from .responsivity import ResponsivityLM
from .similarity_surprisal import (
    SimilarityAdjustedSurprisalEmbeddingLM,
    SimilarityAdjustedSurprisalWordLM,
)

__all__ = [
    "InformationValue",
    "InformationValueLM",
    "InformationValueScorer",
    "NextWordInformationValue",
    "ResponsivityDistance",
    "ResponsivityLM",
    "SimilarityAdjustedSurprisal",
    "SimilarityAdjustedSurprisalEmbeddingLM",
    "SimilarityAdjustedSurprisalWordLM",
]
