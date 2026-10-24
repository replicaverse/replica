"""Standardized evaluation of computational sentence-processing models.

The package root exposes the core composition API without importing every
dataset, feature backend, and analysis integration during startup. Attributes
are loaded from their owning subpackage on first access.
"""

import logging
from importlib import import_module
from typing import Any

__version__ = "0.0.1"

_EXPORT_GROUPS = {
    "analysis": (
        "EvaluationSpec",
        "ComparisonSpec",
        "Residualize",
        "OLS",
        "BrmsModel",
        "BrmsPriors",
        "BrmsFit",
        "AnalysisResult",
        "Coefficient",
        "AnalysisConfig",
        "CrossedBootstrapConfig",
        "PredictiveTestResult",
        "predictive_test",
        "crossed_bootstrap_test",
        "Design",
        "Factor",
        "RandomEffect",
        "GroupedPermutationResult",
        "grouped_permutation_test",
        "AnalysisModel",
        "CVSplitter",
        "EncodingResult",
        "EncodingConfig",
        "FixedSplitter",
        "FoldResult",
        "FullDataSplitter",
        "ItemSplitter",
        "CrossedGroupSplitter",
        "MinMaxScaler",
        "MultiSeedSplitter",
        "NoScaler",
        "RowSplitter",
        "RGamModel",
        "RLinearModel",
        "RLmerModel",
        "Scaler",
        "SequentialBlockSplitter",
        "ShuffledKFoldSplitter",
        "StatsmodelsOLS",
        "ZScoreScaler",
        "benchmark",
        "evaluate",
        "prepare_observations",
        "write_analysis_report",
        "ObservationTable",
    ),
    "core": (
        "AnnotationLayer",
        "Datum",
        "Modality",
        "Recording",
        "Span",
        "Stimulus",
    ),
    "dataset": (
        "ChecksumMismatchError",
        "Dataset",
        "DatasetInfo",
        "fixation_feature",
        "EyeTrackingDataset",
        "FileEntry",
        "IndexMapping",
        "Narratives",
        "ReadingTimeDataset",
        "SPRDataset",
        "TabularDataset",
        "TabularDatasetConfig",
    ),
    "feature": (
        "DerivedFeature",
        "Entropy",
        "Feature",
        "FeatureExtractor",
        "FeatureValues",
        "FeatureBlock",
        "FeatureIndex",
        "ModelFeature",
        "resolve",
        "aggregate",
        "named",
        "FeatureModel",
        "FeatureModelOutput",
        "FeatureNotAvailable",
        "FrequencySource",
        "HiddenState",
        "HuggingFaceEmbedding",
        "HuggingFaceLM",
        "Interaction",
        "InUnit",
        "FeatureMeasurement",
        "describe_feature",
        "Lag",
        "LogFrequency",
        "PrecomputedFeature",
        "PrecomputedFeatureModel",
        "PrevUnitLength",
        "Rank",
        "Surprisal",
        "SurprisalReduction",
        "TokenAlignment",
        "TokenEntropy",
        "UnigramSurprisal",
        "UnitLength",
        "UnitPosition",
        "UserTable",
        "WordfreqSource",
        "align_by_spans",
        "align_from_tokenizer",
        "surprisal_model",
    ),
    "segmentation": (
        "CharacterSegmenter",
        "Segmenter",
        "WhitespaceSegmenter",
        "aggregate_mask",
        "aggregate_values",
        "compute_refinement",
        "from_predicate",
        "segment_stimulus",
    ),
}

_EXPORT_MODULE = {
    name: f"{__name__}.{submodule}"
    for submodule, names in _EXPORT_GROUPS.items()
    for name in names
}
__all__ = list(_EXPORT_MODULE)

# Submodules reachable as attributes (``pg.dataset``) without eager imports.
_SUBMODULES = frozenset(
    {
        "analysis",
        "cli",
        "core",
        "dataset",
        "feature",
        "multiverse",
        "predictors",
        "segmentation",
        "studies",
        "utils",
    }
)


def __getattr__(name: str) -> Any:
    """Load one public attribute or submodule from its owning location."""
    module_name = _EXPORT_MODULE.get(name)
    if module_name is None:
        if name in _SUBMODULES:
            value = import_module(f"{__name__}.{name}")
            globals()[name] = value
            return value
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted((*globals(), *__all__, *_SUBMODULES))


logging.getLogger(__name__).addHandler(logging.NullHandler())
