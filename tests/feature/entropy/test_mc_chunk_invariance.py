"""Splitting ``n_samples`` into chunks does not change a sample's draw.

Asserted on the token chain, whose forward pass is a function of the current
token. A real LM confounds the comparison: its forward pass is not
batch-invariant, so a uniform can land across a shifted bin edge and flip a
token, moving a word by about a bit.
"""

import numpy as np
import pytest

from replica.feature.diagnostics.references.torch_models import TokenChainReference
from replica.feature.entropy.estimators import (
    ContinuationEntropyLM,
    ForwardMarginalEntropyLM,
    ForwardWordLookaheadEntropyLM,
    MCWordEntropyLM,
)
from tests.feature.entropy._entropy_mocks import build_estimator

WORDS = ["ba", "dax", "gaxy", "ka", "bax", "da"]
N_SAMPLES = 16


@pytest.fixture(scope="module")
def token_ref():
    return TokenChainReference()


def _chain(token_ref, chunk, seed, n_samples) -> dict:
    """Settings every family shares: the token chain, the seed and the chunking."""
    return {
        "model": token_ref.model,
        "tokenizer": token_ref.tokenizer,
        "model_id": "markov_token_chain",
        "n_samples": n_samples,
        "seed": seed,
        "chunk_size": chunk,
    }


def _mcword(token_ref, chunk, seed=0, n_samples=N_SAMPLES):
    shared = _chain(token_ref, chunk, seed, n_samples)
    return build_estimator(
        MCWordEntropyLM, name="mcword_chain", max_iter=4, alpha=1.0, **shared
    )


def _continuation(token_ref, chunk, seed=0, n_samples=N_SAMPLES):
    shared = _chain(token_ref, chunk, seed, n_samples)
    return build_estimator(
        ContinuationEntropyLM, name="cont_chain", max_tokens=3, **shared
    )


def _forward_marginal(token_ref, chunk, seed=0, n_samples=N_SAMPLES):
    shared = _chain(token_ref, chunk, seed, n_samples)
    return build_estimator(
        ForwardMarginalEntropyLM, name="fwd_marg_chain", max_tokens=3, **shared
    )


def _lookahead(token_ref, chunk, seed=0, n_samples=N_SAMPLES):
    shared = _chain(token_ref, chunk, seed, n_samples)
    return build_estimator(
        ForwardWordLookaheadEntropyLM,
        name="fwl_chain",
        max_iter=4,
        max_words=3,
        alpha=1.0,
        **shared,
    )


# Builder, value key, and how exactly a chunked run must reproduce an
# unchunked one. Forward-marginal folds samples into running sums as the chunks
# arrive, and float addition is not associative, so its total shifts with the
# number of pieces.
FAMILIES = {
    "mcword": (_mcword, "mc_word_entropy", 0.0),
    "continuation": (_continuation, "continuation_entropy", 0.0),
    "forward_marginal": (_forward_marginal, "forward_marginal_entropy", 1e-6),
    "word_lookahead": (_lookahead, "joint_word_entropy", 0.0),
}
# Far above either tolerance, so a flipped draw still fails.
_DRAW_FLIP_BITS = 0.1


def _values(lm, token_ref, key):
    stim = token_ref.make_stimulus(WORDS)
    out = lm(stim, extract=None, target_layer="words")
    return np.array(
        [np.ravel(u.values[key]).astype(float).sum() for u in out], dtype=np.float64
    )


@pytest.mark.parametrize("family", sorted(FAMILIES))
@pytest.mark.parametrize("chunk", [1, 2, 5])
def test_chunking_leaves_every_value_untouched(token_ref, family, chunk):
    """The chain cannot hide a changed draw behind batch-shape noise."""
    build, key, rtol = FAMILIES[family]
    whole = _values(build(token_ref, None), token_ref, key)
    split = _values(build(token_ref, chunk), token_ref, key)
    if rtol == 0.0:
        np.testing.assert_array_equal(whole, split)
    else:
        np.testing.assert_allclose(whole, split, rtol=rtol, atol=0)
    finite = np.isfinite(whole)
    assert finite.any() and np.ptp(split[finite]) > 0
    assert np.max(np.abs(whole[finite] - split[finite])) < _DRAW_FLIP_BITS


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_seed_controls_reproducible_draws(token_ref, family):
    build, key, _ = FAMILIES[family]
    first = _values(build(token_ref, 2, seed=7), token_ref, key)
    repeated = _values(build(token_ref, 2, seed=7), token_ref, key)
    changed = _values(build(token_ref, 2, seed=99), token_ref, key)
    np.testing.assert_array_equal(first, repeated)
    both = np.isfinite(first) & np.isfinite(changed)
    assert not np.array_equal(first[both], changed[both])


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_release_mode_restores_the_shared_stream(token_ref, family):
    """Under the released estimators' path the batch size selects the draw."""
    build, key, _ = FAMILIES[family]
    whole, split = build(token_ref, None), build(token_ref, 2)
    whole.release_rng_stream = split.release_rng_stream = True
    a = _values(whole, token_ref, key)
    b = _values(split, token_ref, key)
    assert not np.array_equal(a[np.isfinite(a)], b[np.isfinite(b)])


def test_the_shared_uniform_plane_is_sliced_by_position(token_ref):
    lm = _mcword(token_ref, 2)
    lm._stimulus_seed = 999
    whole = lm._uniform_block(("k", 3), 0, 8, 12)
    tail = lm._uniform_block(("k", 3), 4, 4, 12)
    np.testing.assert_array_equal(whole[4:].cpu().numpy(), tail.cpu().numpy())
