"""Surprisal-specific FeatureModel backends.

These produce per-unit ``log_p_observed`` values that the
:class:`~replica.feature.Surprisal` Feature extractor consumes.
For the canonical standard-surprisal path (``HuggingFaceLM``), see
:mod:`replica.feature.backends`.

Backends here:

- :class:`CharBeamLM`, character-level top-K beam over tokenizations.
- :class:`ClozeLM`, human cloze norms read off the stimulus.
- :class:`LogitLensLM` / :class:`TunedLensLM`, per-layer lens surprisal.
- :class:`ModelBlocksSurprisalLM`, ModelBlocks ``totsurp`` convention.
- :class:`WordsProbabilityLM`, BOW/EOW-corrected word probability.
"""

from .char_beam import CharBeamLM
from .cloze import ClozeLM
from .logit_lens import LogitLensLM
from .modelblocks import ModelBlocksSurprisalLM
from .tuned_lens import TunedLensLM
from .wordsprob import WordsProbabilityLM

__all__ = [
    "CharBeamLM",
    "ClozeLM",
    "LogitLensLM",
    "TunedLensLM",
    "ModelBlocksSurprisalLM",
    "WordsProbabilityLM",
]
