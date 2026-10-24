"""Entropy-specific FeatureModel backends.

Closed-form entropy from a standard LM is computed directly off
:class:`HuggingFaceLM`'s ``next_log_probs`` and ``token_entropy``
outputs by the :class:`~replica.feature.Entropy`,
:class:`~replica.feature.TokenEntropy`, and
:class:`~replica.feature.RenyiEntropy` Feature extractors.

The backends here are the Monte-Carlo estimators that need to sample
from the LM directly:

- :class:`ContinuationEntropyLM`, Giulianelli et al. (2024a)
  sequence-level entropy via MC over fixed-length continuations, and
  :class:`ContinuationEntropyLMBuggy`, the released estimator's verified
  defects reproduced.
- :class:`MCWordEntropyLM`, Clark et al. (2025) word-level
  entropy via MC with word-boundary stopping.
- :class:`ForwardMarginalEntropyLM` and
  :class:`ForwardWordLookaheadEntropyLM`, the forward-looking profiles.

The three reproduction-only entropy variants require ``allow_reproduction=True``
on their model and extractor constructors, or ``--allow-reproduction`` in the
compute CLI. They still emit :class:`ReproductionOnlyWarning` after opt-in,
explaining the incorrect operation and naming the corrected replacement.

Position-based seeded sampling identifies the actual model/tokenizer bundle by
its content digest, independently of the cache directory. Both compute and direct
constructors resolve the same inventory and prefer safetensors over duplicate
PyTorch weights. Hub files are checked against the pinned Git/LFS object hashes.
Local directories and offline cached names retain content identities without
claiming a verified Hub revision. Explicit identity assertions must be verified.
The inventory and digest are checked again after loading to detect concurrent
edits; these before/after checks do not make the filesystem immutable.

Under the ``model-bundle-v1`` seed scheme each draw is keyed by the seed, the
model bundle digest and the draw's position. A compute identity carries the
digest of the package source, so an artifact or partial written by different
code is not reused. The advancing-stream mode keeps the released seed and draw
order. Buggy continuation uses ``transformers.generate`` with the released
global RNG reset per stimulus, recorded separately as ``global-per-stimulus``.
Identical random uniforms do not guarantee identical sampled tokens across
numerical libraries or devices. Direct online Hub construction resolves a
complete configuration and tokenizer inventory, then loads from local files
only.
"""

from ..core.reproduction import ReproductionOnlyWarning
from .estimators import (
    ContinuationEntropyLM,
    ContinuationEntropyLMBuggy,
    ForwardMarginalEntropyLM,
    ForwardWordLookaheadEntropyLM,
    MCWordEntropyLM,
)

__all__ = [
    "ContinuationEntropyLM",
    "ContinuationEntropyLMBuggy",
    "ForwardMarginalEntropyLM",
    "ForwardWordLookaheadEntropyLM",
    "MCWordEntropyLM",
    "ReproductionOnlyWarning",
]
