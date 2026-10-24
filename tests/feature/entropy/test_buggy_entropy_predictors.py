"""Correctness tests for the two 'buggy' entropy predictors used by the
multiverse grid to contrast buggy-vs-corrected features.

* :class:`TokenEntropySum`, the Pimentel et al. (2023) buggy first-token
  entropy: the SUM of a word's per-subtoken Shannon entropies (the paper's
  released ``agg_entropy`` mis-routed through the summing aggregator), vs the
  corrected first-subtoken :class:`TokenEntropy`.
* :class:`ContinuationEntropyLMBuggy`, the Giulianelli et al. (2024a)
  buggy sequence-level entropy reproducing measures.py's inherited
  ``top_k=50`` sampling and reshape-vs-transpose bug, vs the corrected
  :class:`ContinuationEntropyLM`.
"""

import numpy as np
import pytest

from replica.feature import (
    ContinuationEntropy,
    ContinuationEntropyBuggy,
    ContinuationEntropyLM,
    ContinuationEntropyLMBuggy,
    TokenEntropy,
    TokenEntropySum,
    TokenRenyiEntropy,
    TokenRenyiEntropySum,
)
from replica.feature.entropy import ReproductionOnlyWarning
from tests._stimuli import word_stimulus as _stim

pytestmark = pytest.mark.needs_model


@pytest.mark.slow
def test_pimentel_buggy_sum_geq_corrected_first_token(lm_cache):
    """The buggy SUM aggregation is ≥ the corrected first-subtoken entropy,
    and strictly greater for a word split into more than one subtoken.

    Each subtoken's Shannon entropy is non-negative, so summing over a
    multi-subtoken word can only add to the first subtoken's value.
    ``antidisestablishmentarianism`` tokenises into several subwords, so its
    summed entropy must exceed its first-token entropy; single-token words
    are unchanged.
    """
    stim = _stim("The antidisestablishmentarianism debate continued endlessly", "pim")
    lm = lm_cache(
        "distilgpt2",
        add_bos=True,
        provide_word_dist=False,
        token_level_entropy=True,
        token_entropy_sum=True,
        allow_reproduction=True,
    )
    outs = lm(stim, extract=["token_entropy", "token_entropy_sum"])
    first = TokenEntropy()(stim, outs)
    summed = TokenEntropySum(allow_reproduction=True)(stim, outs)

    finite = np.isfinite(first) & np.isfinite(summed)
    assert np.all(summed[finite] + 1e-9 >= first[finite]), (
        "buggy summed entropy must be >= corrected first-token entropy"
    )
    # The long word is multi-subtoken, so its sum must strictly exceed first.
    assert np.any(summed[finite] > first[finite] + 1e-6), (
        "expected at least one multi-subtoken word where sum > first"
    )


@pytest.mark.slow
def test_pimentel_buggy_renyi_sum_geq_corrected_first_token():
    """The buggy SUM aggregation of the Rényi-1/2 entropy is >= the corrected
    first-subtoken Rényi, and strictly greater for a multi-subtoken word.

    Mirror of :func:`test_pimentel_buggy_sum_geq_corrected_first_token` for the
    Rényi-alpha=0.5 entropy. Each subtoken's Rényi entropy is non-negative, so
    summing over a multi-subtoken word can only add to the first subtoken's
    value; single-token words are unchanged.
    """
    from replica.feature import HuggingFaceLM

    stim = _stim("The antidisestablishmentarianism debate continued endlessly", "pimr")
    # The list-valued alphas require a model outside the hash-keyed cache.
    lm = HuggingFaceLM(
        "distilgpt2",
        add_bos=True,
        provide_word_dist=False,
        token_level_entropy=False,
        token_renyi_alphas=[0.5],
        token_renyi_entropy_sum=True,
        allow_reproduction=True,
    )
    outs = lm(stim, extract=["token_renyi_entropy_alpha0.5", "token_renyi_entropy_sum"])
    first = TokenRenyiEntropy(alpha=0.5)(stim, outs)
    summed = TokenRenyiEntropySum(allow_reproduction=True)(stim, outs)

    finite = np.isfinite(first) & np.isfinite(summed)
    assert np.all(summed[finite] + 1e-9 >= first[finite]), (
        "buggy summed Rényi must be >= corrected first-token Rényi"
    )
    # The long word is multi-subtoken, so its sum must strictly exceed first.
    assert np.any(summed[finite] > first[finite] + 1e-6), (
        "expected at least one multi-subtoken word where sum > first"
    )


@pytest.mark.slow
def test_giulianelli_buggy_runs_and_exceeds_corrected():
    """The buggy continuation entropy returns finite per-word values and sits
    far above the corrected estimate.

    The reshape-as-reinterpret bug scores each sampled continuation under a
    mismatched distribution, which raises the estimate well above the
    corrected manual-sampling value; the inherited top_k=50 on its own lowers
    it.
    """
    stim = _stim("The quick brown fox jumps", "giu")
    with pytest.warns(ReproductionOnlyWarning, match="only for reproduction"):
        buggy = ContinuationEntropyLMBuggy(
            "distilgpt2", allow_reproduction=True, n_samples=32, max_tokens=10, seed=0
        )
    corrected = ContinuationEntropyLM("distilgpt2", n_samples=32, max_tokens=10, seed=0)
    with pytest.warns(ReproductionOnlyWarning, match="only for reproduction"):
        buggy_feature = ContinuationEntropyBuggy(allow_reproduction=True)
    h_buggy = buggy_feature(stim, buggy(stim))
    h_corr = ContinuationEntropy()(stim, corrected(stim))

    # Every word has left context (BOS prefix), so all values are finite.
    assert np.all(np.isfinite(h_buggy)), "buggy entropies must be finite"
    assert np.all(h_buggy > 0), "buggy entropies must be positive"
    # Wide margin so the check is insensitive to sampling noise.
    assert np.nanmean(h_buggy) > np.nanmean(h_corr) + 10.0, (
        f"buggy mean {np.nanmean(h_buggy):.1f} should be far above corrected "
        f"mean {np.nanmean(h_corr):.1f} bits"
    )
