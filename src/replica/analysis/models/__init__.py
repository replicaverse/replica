"""Regression backends and shared fitting support.

Every backend imports without its optional dependency: the R-backed models
load ``rpy2``, and ``RustLmerModel`` loads ``lme_python``, only when they fit.
"""

from replica.analysis.contracts.models import AnalysisModel
from replica.analysis.models.additive.gam import RGamModel
from replica.analysis.models.fixed_effects.elasticnet import ElasticNetCVModel
from replica.analysis.models.fixed_effects.ols import OLS
from replica.analysis.models.fixed_effects.r import RLinearModel
from replica.analysis.models.fixed_effects.ridge import RidgeCVModel
from replica.analysis.models.fixed_effects.statsmodels import StatsmodelsOLS
from replica.analysis.models.mixed_effects.brms import BrmsFit, BrmsModel, BrmsPriors
from replica.analysis.models.mixed_effects.lmer import RLmerModel
from replica.analysis.models.mixed_effects.lmer_rs import RustLmerModel
from replica.analysis.models.mixed_effects.marginal_lme import MarginalLMEModel
from replica.analysis.models.mixed_effects.robust_lmer import RobustCrossedLmer
from replica.analysis.models.mixed_effects.statsmodels import (
    StatsmodelsLME,
    StatsmodelsLMEResult,
)

__all__ = [
    "AnalysisModel",
    "BrmsModel",
    "BrmsPriors",
    "BrmsFit",
    "OLS",
    "RobustCrossedLmer",
    "ElasticNetCVModel",
    "RidgeCVModel",
    "StatsmodelsOLS",
    "StatsmodelsLME",
    "StatsmodelsLMEResult",
    "RLinearModel",
    "RLmerModel",
    "RustLmerModel",
    "RGamModel",
    "MarginalLMEModel",
]
