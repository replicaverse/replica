"""Real-GPT-2 tests for the similarity-adjusted surprisal of Meister et al. (2024).

The exact non-contextual path is anchored against a from-scratch
reference computed with plain ``transformers`` calls in this file; the MC
paths are checked for seed reproducibility and internal consistency
(per-alpha aggregates vs stored per-sample similarities, and the
``i = 1 - exp(-s_z)`` identity linking the two measures at the token level).
"""

import numpy as np
import pytest
import torch

from replica.feature import (
    SimilarityAdjustedSurprisalEmbeddingLM,
    SimilarityAdjustedSurprisalWordLM,
)
from tests._stimuli import word_stimulus

pytestmark = pytest.mark.needs_model

WORDS = ("The", "cat", "sat", "on", "the", "mat", "quietly")


def _make_stimulus(words=WORDS):
    return word_stimulus(" ".join(words), "s1")


@pytest.fixture(scope="module")
def gpt2_ref():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    model.eval()
    return tok, model


@pytest.mark.parametrize("include_wpe0", [False, True])
def test_noncontextual_matches_direct_reference(gpt2_ref, include_wpe0):
    tok, model = gpt2_ref
    stim = _make_stimulus()
    alphas = (1.0, 2.0)
    lm = SimilarityAdjustedSurprisalEmbeddingLM(
        "gpt2",
        contextual=False,
        alphas=alphas,
        include_wpe0=include_wpe0,
        add_bos=True,
        device="cpu",
    )
    outs = lm(stim)

    ids = tok(stim.text)["input_ids"]
    ids = [tok.bos_token_id, *ids]
    with torch.no_grad():
        logits = model(torch.tensor([ids])).logits[0].double()
        emb = model.get_input_embeddings().weight.detach()
        if include_wpe0:
            emb = emb + model.transformer.wpe.weight[0:1].detach()
        emb = emb / emb.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    # Direct m_alpha per token position (each test word is one GPT-2 token,
    # so word w sits at token position w + 1 after BOS).
    for w in range(len(WORDS)):
        t = w + 1
        p = torch.softmax(logits[t - 1], dim=-1)
        cos = emb @ emb[ids[t]]
        sims = (0.5 * (cos + 1.0)).double()
        got = outs[w].values["expected_similarity_tokens"]
        assert got.shape == (1, len(alphas))
        for a_i, alpha in enumerate(alphas):
            ref = float((sims.pow(alpha) * p).sum())
            np.testing.assert_allclose(got[0, a_i], ref, rtol=1e-6)

        m1 = got[0, 0]
        np.testing.assert_allclose(
            outs[w].values["similarity_adjusted_surprisal"], -np.log(m1), rtol=1e-12
        )
        np.testing.assert_allclose(
            outs[w].values["next_word_information_value"], 1.0 - m1, rtol=1e-12
        )


def test_information_value_surprisal_identity(gpt2_ref):
    # Theorem 1's link at the token level: i = 1 - exp(-s_z).
    stim = _make_stimulus()
    lm = SimilarityAdjustedSurprisalEmbeddingLM(
        "gpt2", contextual=False, alphas=(1.0,), add_bos=True, device="cpu"
    )
    for out in lm(stim):
        s = out.values["similarity_adjusted_surprisal_first"]
        i = out.values["next_word_information_value_first"]
        np.testing.assert_allclose(i, 1.0 - np.exp(-s), rtol=1e-10)


def test_noncontextual_high_alpha_approaches_surprisal(gpt2_ref):
    # As alpha grows, z^alpha approaches the identity similarity and
    # similarity-adjusted surprisal approaches standard surprisal.
    tok, model = gpt2_ref
    stim = _make_stimulus()
    lm = SimilarityAdjustedSurprisalEmbeddingLM(
        "gpt2", contextual=False, alphas=(1.0, 32.0, 512.0), add_bos=True, device="cpu"
    )
    outs = lm(stim)

    ids = tok(stim.text)["input_ids"]
    ids = [tok.bos_token_id, *ids]
    with torch.no_grad():
        log_probs = torch.log_softmax(model(torch.tensor([ids])).logits[0].double(), -1)

    for w in range(len(WORDS)):
        t = w + 1
        surprisal = float(-log_probs[t - 1, ids[t]])
        m = outs[w].values["expected_similarity_tokens"][0]
        gap_low, gap_high = (
            abs(-np.log(m[1]) - surprisal),
            abs(-np.log(m[2]) - surprisal),
        )
        assert gap_high < gap_low or gap_high < 1e-3


def test_contextual_mc_seeded_and_consistent():
    stim = _make_stimulus()
    kwargs = dict(
        contextual=True, n_samples=8, alphas=(1.0, 2.0), add_bos=True, device="cpu"
    )
    outs1 = SimilarityAdjustedSurprisalEmbeddingLM("gpt2", seed=7, **kwargs)(stim)
    outs2 = SimilarityAdjustedSurprisalEmbeddingLM("gpt2", seed=7, **kwargs)(stim)
    outs3 = SimilarityAdjustedSurprisalEmbeddingLM("gpt2", seed=8, **kwargs)(stim)

    any_diff = False
    for w in range(len(WORDS)):
        s1 = outs1[w].values["similarity_samples_tokens"]
        s2 = outs2[w].values["similarity_samples_tokens"]
        s3 = outs3[w].values["similarity_samples_tokens"]
        np.testing.assert_array_equal(s1, s2)
        any_diff = any_diff or not np.array_equal(s1, s3)
        assert s1.shape == (1, 8)
        assert np.all((s1 >= 0.0) & (s1 <= 1.0))
        # Stored aggregates re-derive from the per-sample similarities.
        m = outs1[w].values["expected_similarity_tokens"]
        np.testing.assert_allclose(m[0, 0], s1[0].mean(), rtol=1e-9)
        np.testing.assert_allclose(m[0, 1], (s1[0] ** 2).mean(), rtol=1e-9)
    assert any_diff, "different seeds should draw different samples"


def test_word_model_orthographic_seeded_and_consistent():
    stim = _make_stimulus()
    kwargs = dict(n_samples=8, max_iter=5, pos=False, add_bos=True, device="cpu")
    outs1 = SimilarityAdjustedSurprisalWordLM("gpt2", seed=3, **kwargs)(stim)
    outs2 = SimilarityAdjustedSurprisalWordLM("gpt2", seed=3, **kwargs)(stim)

    for w in range(len(WORDS)):
        w1, w2 = outs1[w].values["sampled_words"], outs2[w].values["sampled_words"]
        assert list(w1) == list(w2)
        assert len(w1) == 8
        sims = outs1[w].values["orth_similarity_samples"]
        assert sims.shape == (8,)
        assert np.all((sims >= 0.0) & (sims <= 1.0))
        m = float(sims.mean())
        got = outs1[w].values["similarity_adjusted_surprisal_orth"]
        if m == 0.0:
            assert got == float("inf")
        else:
            np.testing.assert_allclose(got, -np.log(m), rtol=1e-12)
        np.testing.assert_allclose(
            outs1[w].values["next_word_information_value_orth"], 1.0 - m, rtol=1e-12
        )
        # POS disabled: keys present but NaN / empty.
        assert np.isnan(outs1[w].values["similarity_adjusted_surprisal_pos"])
        assert outs1[w].values["pos_match_samples"].size == 0


@pytest.mark.slow
def test_word_model_pos_variant_flair():
    flair = pytest.importorskip("flair")
    assert flair is not None
    stim = _make_stimulus()
    lm = SimilarityAdjustedSurprisalWordLM(
        "gpt2", n_samples=8, max_iter=5, pos=True, add_bos=True, device="cpu", seed=0
    )
    outs = lm(stim)
    for w in range(len(WORDS)):
        assert isinstance(outs[w].values["observed_pos"], str)
        match = outs[w].values["pos_match_samples"]
        assert match.shape == (8,)
        assert set(np.unique(match)).issubset({0.0, 1.0})
        m = float(match.mean())
        got = outs[w].values["similarity_adjusted_surprisal_pos"]
        if m == 0.0:
            assert got == float("inf")
        else:
            np.testing.assert_allclose(got, -np.log(m), rtol=1e-12)
