"""Frequency / unigram-surprisal predictors, in one place.

All the ways to compute a frequency baseline: the :class:`FrequencySource`
implementations (table / count / sampled), the :class:`UnigramSurprisalLM`
FeatureModel, the :class:`LogFrequency` / :class:`UnigramSurprisal` extractors,
and the :data:`SOURCES` registry that names them and their on-disk pickles.

The :class:`FrequencySource` protocol itself is defined in
:mod:`..core.protocols` and re-exported here.
"""

from ..core.protocols import FrequencySource
from .allocation import AllocationSpec
from .extractors import LogFrequency, UnigramSurprisal
from .model import UnigramSurprisalLM
from .pooled import (
    BoundaryIndex,
    PooledUnigramSource,
    PoolSelection,
    SupportThresholds,
    SweepPartial,
    WordRecord,
    boundary_index,
    build_pool_selection,
    exhaustive_estimate_batch,
    merge_index,
    merge_sweep,
    pps_estimate_batch,
    sweep_shard,
)
from .registry import BITS_KEY, NATS_KEY, SOURCES, FreqSource, pickle_prefix
from .sampled import (
    SampledUnigramSource,
    mc_estimate,
    mmc_estimate,
    mmc_estimate_batch,
)
from .sources import (
    KenLMSource,
    OpenWebTextSource,
    SubtlexSource,
    UserTable,
    WordfreqSource,
)
from .strata import Stratification

__all__ = [
    "BITS_KEY",
    "NATS_KEY",
    "AllocationSpec",
    "BoundaryIndex",
    "FreqSource",
    "FrequencySource",
    "KenLMSource",
    "LogFrequency",
    "OpenWebTextSource",
    "PoolSelection",
    "PooledUnigramSource",
    "SOURCES",
    "SampledUnigramSource",
    "Stratification",
    "SubtlexSource",
    "SupportThresholds",
    "SweepPartial",
    "UnigramSurprisal",
    "UnigramSurprisalLM",
    "UserTable",
    "WordRecord",
    "WordfreqSource",
    "boundary_index",
    "build_pool_selection",
    "exhaustive_estimate_batch",
    "mc_estimate",
    "merge_index",
    "merge_sweep",
    "pickle_prefix",
    "mmc_estimate",
    "mmc_estimate_batch",
    "pps_estimate_batch",
    "sweep_shard",
]
