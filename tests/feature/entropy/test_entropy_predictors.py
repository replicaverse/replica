"""Correctness tests for the three entropy predictors.

* :class:`RenyiEntropy`, closed-form Rényi-alpha entropy over the
  next-word distribution (Pimentel et al., 2023, "On the Effect of
  Anticipation on Reading Times").
* :class:`ContinuationEntropyLM`, MC estimate of sequence-level
  entropy (Giulianelli et al., 2024a, "Generalized Measures of
  Anticipation and Responsivity").
* :class:`MCWordEntropyLM`, MC estimate of word-level entropy with
  word-boundary stopping (Clark et al., 2025, "How Well Does First-Token
  Entropy Approximate Word Entropy as a Psycholinguistic Predictor?").

Each test below pins a *mathematical* property of the implementation,
independent of any frozen numerical reference.
"""

import math

import numpy as np
import pytest

from replica.core import AnnotationLayer, Span, Stimulus
from replica.feature import (
    ContinuationEntropy,
    ContinuationEntropyLM,
    Entropy,
    ExpectedProbability,
    HuggingFaceLM,
    MCWordEntropy,
    MCWordEntropyLM,
    RenyiEntropy,
    TokenEntropy,
    TokenRenyiEntropy,
)
from tests._stimuli import word_stimulus as _stim

pytestmark = pytest.mark.needs_model


@pytest.fixture(scope="module")
def stim():
    return _stim("the quick brown fox", "test_stim")


@pytest.fixture(scope="module")
def hf_outputs(stim, lm_cache):
    """Outputs from a HuggingFaceLM with next_log_probs + token_entropy."""
    lm = lm_cache(
        "distilgpt2",
        add_bos=True,
        provide_word_dist=True,
        token_level_entropy=True,
    )
    return lm(stim, extract=["next_log_probs", "token_entropy"])


# ---------------------------------------------------------------------------
# Rényi entropy, closed-form, from Pimentel et al. (2023)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_renyi_alpha_1_equals_shannon(stim, hf_outputs):
    """At alpha = 1 the Rényi entropy reduces to Shannon entropy (L'Hôpital).

    Both compute the same -Σp·log₂p over the same word-type
    distribution, so the values must be bit-for-bit identical.
    """
    shannon = Entropy()(stim, hf_outputs)
    renyi_1 = RenyiEntropy(alpha=1.0)(stim, hf_outputs)
    np.testing.assert_array_equal(shannon, renyi_1)


@pytest.mark.slow
def test_renyi_alpha_2_equals_collision_entropy(stim, hf_outputs):
    """At alpha = 2 the Rényi entropy is the collision entropy:

        H₂(p) = -log₂ Σ_i p_i²

    where Σ p² is exactly :class:`ExpectedProbability`. The
    implementation should match this identity exactly.
    """
    renyi_2 = RenyiEntropy(alpha=2.0)(stim, hf_outputs)
    exp_p = ExpectedProbability()(stim, hf_outputs)
    expected = -np.log2(np.asarray(exp_p))
    np.testing.assert_allclose(renyi_2, expected, atol=1e-9)


@pytest.mark.slow
def test_token_renyi_alpha_1_equals_token_entropy(stim):
    """At alpha = 1, ``TokenRenyiEntropy`` equals :class:`TokenEntropy`, the Shannon
    entropy of the LM's subword distribution (the Pimentel et al. (2023) setup).

    :class:`Entropy` and :class:`RenyiEntropy` marginalise to word types first.
    """
    lm = HuggingFaceLM(
        "distilgpt2",
        add_bos=True,
        provide_word_dist=False,
        token_level_entropy=True,
        token_renyi_alphas=[1.0],
    )
    outs = lm(stim, extract=["token_entropy", "token_renyi_entropy_alpha1.0"])
    shannon = TokenEntropy()(stim, outs)
    renyi_1 = TokenRenyiEntropy(alpha=1.0)(stim, outs)
    np.testing.assert_allclose(shannon, renyi_1, atol=1e-9, equal_nan=True)


@pytest.mark.slow
def test_token_renyi_pimentel_alpha_half_is_subword_level(stim):
    """The Pimentel et al. (2023) best-fit predictor, Rényi alpha = 0.5 on the
    subword first-token distribution. Must differ from the word-type
    Rényi at the same alpha (the word-type marginal collapses some mass)
    and must satisfy the monotone-in-alpha inequality at the subword
    level: H_{1/2} ≥ H_1 ≥ H_2.
    """
    lm = HuggingFaceLM(
        "distilgpt2",
        add_bos=True,
        provide_word_dist=False,
        token_level_entropy=False,
        token_renyi_alphas=[0.5, 1.0, 2.0],
    )
    outs = lm(
        stim,
        extract=[
            "token_renyi_entropy_alpha0.5",
            "token_renyi_entropy_alpha1.0",
            "token_renyi_entropy_alpha2.0",
        ],
    )
    h_half = TokenRenyiEntropy(alpha=0.5)(stim, outs)
    h_one = TokenRenyiEntropy(alpha=1.0)(stim, outs)
    h_two = TokenRenyiEntropy(alpha=2.0)(stim, outs)
    finite = np.isfinite(h_half) & np.isfinite(h_one) & np.isfinite(h_two)
    assert np.all(h_half[finite] + 1e-9 >= h_one[finite])
    assert np.all(h_one[finite] + 1e-9 >= h_two[finite])


@pytest.mark.slow
def test_renyi_monotone_decreasing_in_alpha(stim, hf_outputs):
    """Rényi entropy is monotonically non-increasing in alpha
    (\\citealt{renyi1961measures}), so H_{1/2} ≥ H_1 ≥ H_2 must hold
    pointwise on a non-degenerate distribution.
    """
    h_half = RenyiEntropy(alpha=0.5)(stim, hf_outputs)
    h_one = RenyiEntropy(alpha=1.0)(stim, hf_outputs)
    h_two = RenyiEntropy(alpha=2.0)(stim, hf_outputs)
    finite = np.isfinite(h_half) & np.isfinite(h_one) & np.isfinite(h_two)
    assert np.all(h_half[finite] + 1e-9 >= h_one[finite])
    assert np.all(h_one[finite] + 1e-9 >= h_two[finite])


# ---------------------------------------------------------------------------
# Continuation entropy, from Giulianelli et al. (2024a)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_continuation_entropy_L1_converges_to_first_token(stim, hf_outputs):
    """At L = 1 each MC sample is exactly one token, so

        H_cont ≈ -(1/N) Σᵢ log p(tᵢ | ctx)   →   H_shannon(p(·|ctx))

    as N → ∞. With N = 2000 the standard error is well below 0.5 bits,
    used here as a generous tolerance.
    """
    lm = ContinuationEntropyLM("distilgpt2", n_samples=2000, max_tokens=1)
    ce = ContinuationEntropy()(stim, lm(stim))
    ft = TokenEntropy()(stim, hf_outputs)
    diff = np.nanmax(np.abs(ce - ft))
    assert diff < 0.5, (
        f"L=1 MC entropy should converge to closed-form first-token "
        f"entropy; got max |diff| = {diff:.3f} bits"
    )


@pytest.mark.slow
def test_continuation_entropy_grows_with_L(stim):
    """Sequence-level entropy is monotone in L: longer continuations
    represent more uncertainty (each extra token's surprisal adds in
    expectation). H_cont(L=10) should be substantially larger than
    H_cont(L=1) on a typical text.
    """
    lm_short = ContinuationEntropyLM("distilgpt2", n_samples=128, max_tokens=1, seed=0)
    lm_long = ContinuationEntropyLM("distilgpt2", n_samples=128, max_tokens=10, seed=0)
    h_short = ContinuationEntropy()(stim, lm_short(stim))
    h_long = ContinuationEntropy()(stim, lm_long(stim))
    assert np.nanmean(h_long) > np.nanmean(h_short)


def test_continuation_entropy_L2_matches_chain_rule_on_synthetic_lm():
    """At L = 2 on a hand-coded 4-token Markov LM, the MC estimate converges to
    the closed-form joint entropy given by the chain rule

        H(t1, t2 | prev) = H(t1 | prev) + Σ_{t1} P(t1|prev) · H(t2 | t1)

    The synthetic LM isolates the entropy math from any HuggingFace LM behaviour.
    """

    import torch
    from torch import nn

    from replica.feature.entropy.estimators import (
        ContinuationEntropyLM,
    )
    from tests.feature.entropy._entropy_mocks import build_estimator

    V = 4
    P = np.array(
        [
            [0.40, 0.30, 0.20, 0.10],
            [0.10, 0.50, 0.20, 0.20],
            [0.10, 0.10, 0.70, 0.10],
            [0.05, 0.10, 0.15, 0.70],
        ]
    )
    assert np.allclose(P.sum(axis=1), 1.0)
    LOG_P = np.log(P)

    class _StubTokenizer:
        vocab_size = V
        bos_token_id = 0
        pad_token_id = 0

    class _SyntheticLM(nn.Module):
        class _Out:
            def __init__(self, logits, past=None):
                self.logits = logits
                self.past_key_values = past

        class _Config:
            n_positions = 1024
            max_position_embeddings = 1024
            vocab_size = V

        def __init__(self):
            super().__init__()
            self.config = self._Config()

        def forward(self, input_ids, past_key_values=None, use_cache=False):
            batch, seq = input_ids.shape
            out = torch.zeros(batch, seq, V)
            for b in range(batch):
                for t in range(seq):
                    prev = int(input_ids[b, t].item())
                    out[b, t] = torch.tensor(LOG_P[prev])
            return self._Out(out, input_ids[:, -1:])

    # Ground truth: joint entropy of length-2 continuations given prev.
    prev = 2
    H_t1 = -np.sum(P[prev] * LOG_P[prev]) / math.log(2)
    H_t2_given_t1 = -np.sum(P * LOG_P, axis=1) / math.log(2)
    H_joint_true = H_t1 + np.sum(P[prev] * H_t2_given_t1)

    # Bypass HuggingFace loading by constructing the LM on the synthetic model.
    lm = build_estimator(
        ContinuationEntropyLM,
        _SyntheticLM(),
        tokenizer=_StubTokenizer(),
        name="synthetic",
        n_samples=30_000,
        max_tokens=2,
    )

    H_mc = lm._estimate([prev])
    assert abs(H_mc - H_joint_true) < 0.1, (
        f"MC L=2 entropy {H_mc:.4f} bits diverges from analytic ground "
        f"truth {H_joint_true:.4f} bits by {abs(H_mc - H_joint_true):.4f} "
        f"bits; sampling noise budget at N=30k is roughly 0.05-0.1 bits "
        f"for this distribution, so a larger gap indicates an algorithmic "
        f"bug."
    )


# ---------------------------------------------------------------------------
# MC word entropy, from Clark et al. (2025)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_mc_word_entropy_bounds_first_token_entropy(stim, hf_outputs):
    """Pimentel et al. (2023), App. B, Theorem 1 ("A Subword Bound on the Rényi
    Entropy"): first-token entropy is a lower bound on true word entropy.
    Clark et al. (2025) estimates word entropy via MC; modulo
    sampling noise the estimate should sit at or above the first-token
    entropy.
    """
    lm = MCWordEntropyLM("distilgpt2", n_samples=256, max_iter=10, seed=0)
    mw = MCWordEntropy()(stim, lm(stim))
    ft = TokenEntropy()(stim, hf_outputs)
    finite = np.isfinite(mw) & np.isfinite(ft)
    # Generous noise budget (1.5 bits) for N=256; gap should be
    # solidly positive on average.
    assert np.all(mw[finite] > ft[finite] - 1.5)
    assert np.nanmean(mw - ft) > 0.5


@pytest.mark.slow
def test_mc_word_entropy_refuses_no_whitespace_stimulus():
    """``MCWordEntropyLM`` stops sampling at the next whitespace token,
    which presupposes the stimulus uses whitespace between word units.
    On Chinese/Japanese/Thai text (no whitespace) the estimator would
    silently run to ``max_iter`` and report nonsense. Refuse loudly so
    :class:`ContinuationEntropyLM` or :class:`RenyiEntropy` is used
    instead.
    """
    # Build a Chinese stimulus with multiple character "words" but no
    # whitespace between them.
    text = "我怀念那些日子"
    chars = list(text)
    spans = [Span(i, i + 1) for i in range(len(chars))]
    bad = Stimulus(text=text, stimulus_id="zh")
    bad.add_layer(AnnotationLayer(name="words", spans=spans, values=chars))

    lm = MCWordEntropyLM("distilgpt2", n_samples=4, max_iter=4)
    with pytest.raises(ValueError, match="no whitespace"):
        lm(bad)


@pytest.mark.slow
def test_mc_word_entropy_refuses_character_layer(stim):
    """Per-character units don't match the word-boundary stopping rule."""
    # Re-shape an existing stimulus to expose a 'characters' layer only.
    text = stim.text
    chars_stim = Stimulus(text=text, stimulus_id="chr")
    chars_stim.add_layer(
        AnnotationLayer(
            name="characters",
            spans=[Span(i, i + 1) for i in range(len(text))],
            values=list(text),
        )
    )
    lm = MCWordEntropyLM("distilgpt2", n_samples=4, max_iter=4)
    with pytest.raises(ValueError, match="word-level target_layer"):
        lm(chars_stim, target_layer="characters")
