"""Model evaluation, analysis specifications, and regression backends."""

from replica.analysis.contracts.design import Design, Factor, RandomEffect
from replica.analysis.contracts.results import (
    AnalysisResult,
    Coefficient,
    ComparisonSpec,
    FoldResult,
)
from replica.analysis.data.observations import ObservationTable
from replica.analysis.data.preparation import prepare_observations
from replica.analysis.evaluation import benchmark, evaluate
from replica.analysis.evaluation.encoding import EncodingFoldArtifact, EncodingResult
from replica.analysis.reporting import write_analysis_report
from replica.analysis.specification.config import AnalysisConfig, EncodingConfig
from replica.analysis.specification.evaluation import EvaluationSpec

from .contracts.grouped import GroupedFit as GroupedFit
from .contracts.grouped import GroupedFrame as GroupedFrame
from .contracts.grouped import GroupedModel as GroupedModel
from .contracts.models import FoldScore as FoldScore
from .contracts.models import LikelihoodModel as LikelihoodModel
from .contracts.models import StatsPredictor as StatsPredictor
from .cv import (
    CrossedGroupSplitter,
    CVSplitter,
    FixedSplitter,
    FullDataSplitter,
    ItemSplitter,
    MinMaxScaler,
    MultiSeedSplitter,
    NoScaler,
    RowSplitter,
    Scaler,
    SequentialBlockSplitter,
    ShuffledKFoldSplitter,
    SubjectSentenceMod10Splitter,
    ZScoreScaler,
)
from .data.filters import OutlierFilter as OutlierFilter
from .fitting.transforms import FoldTransform as FoldTransform
from .inference.crossed_bootstrap import CrossedBootstrapConfig, crossed_bootstrap_test
from .inference.crossed_statistics import PredictiveTestResult
from .inference.grouped_permutation import (
    GroupedPermutationResult,
    grouped_permutation_test,
)
from .inference.predictive import predictive_test
from .models import OLS, AnalysisModel, BrmsFit, BrmsModel, BrmsPriors
from .models import (
    RGamModel as RGamModel,
)
from .models import (
    RLinearModel as RLinearModel,
)
from .models import (
    RLmerModel as RLmerModel,
)
from .models import (
    RustLmerModel as RustLmerModel,
)
from .models import (
    StatsmodelsLME as StatsmodelsLME,
)
from .models import (
    StatsmodelsLMEResult as StatsmodelsLMEResult,
)
from .models import (
    StatsmodelsOLS as StatsmodelsOLS,
)
from .specification.preprocessing import Residualize as Residualize

__all__ = [
    "ComparisonSpec",
    "EvaluationSpec",
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
    "EncodingConfig",
    "AnalysisResult",
    "Coefficient",
    "FoldResult",
    "evaluate",
    "prepare_observations",
    "write_analysis_report",
    "ObservationTable",
    "CVSplitter",
    "ItemSplitter",
    "CrossedGroupSplitter",
    "RowSplitter",
    "SequentialBlockSplitter",
    "SubjectSentenceMod10Splitter",
    "FixedSplitter",
    "FullDataSplitter",
    "ShuffledKFoldSplitter",
    "MultiSeedSplitter",
    "Scaler",
    "ZScoreScaler",
    "MinMaxScaler",
    "NoScaler",
    "AnalysisModel",
    "LikelihoodModel",
    "BrmsModel",
    "BrmsPriors",
    "BrmsFit",
    "OLS",
    "benchmark",
    "EncodingResult",
    "EncodingFoldArtifact",
    "RustLmerModel",
    "StatsmodelsOLS",
    "StatsmodelsLME",
    "StatsmodelsLMEResult",
    "RLinearModel",
    "RLmerModel",
    "RGamModel",
    "GroupedModel",
    "GroupedFrame",
    "GroupedFit",
    "FoldScore",
    "StatsPredictor",
    "OutlierFilter",
    "FoldTransform",
    "Residualize",
]
