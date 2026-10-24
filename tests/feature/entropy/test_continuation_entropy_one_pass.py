"""One-pass continuation draws match independent per-word sampling."""

import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from replica.core import AnnotationLayer, Stimulus
from replica.feature.core.annotation.characters import (
    CHARACTER_LAYER,
    attach_characters,
)
from replica.feature.entropy.estimators import ContinuationEntropyLM
from tests._stimuli import word_stimulus as _make_stim
from tests.feature.entropy._entropy_mocks import (
    UniformCausalLM as _UniformLM,
)
from tests.feature.entropy._entropy_mocks import (
    WhitespaceTokenizer as _UniformTokenizer,
)
from tests.feature.entropy._entropy_mocks import build_estimator


class _StubCache(tuple):
    def to_legacy_cache(self):
        return tuple(self)


class _BigramLM(torch.nn.Module):
    """Logits set by the current token alone, so every draw moves the next step."""

    def __init__(self, vocab_size: int = 8):
        super().__init__()
        self.vocab_size = vocab_size
        self.config = SimpleNamespace(max_position_embeddings=1024, n_positions=1024)

    def forward(self, input_ids, past_key_values=None, use_cache=False, **_):
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        vocabulary = torch.arange(self.vocab_size, dtype=torch.float32)
        centre = (input_ids.float() * 3 + 1).remainder(self.vocab_size)
        logits = -(vocabulary - centre.unsqueeze(-1)).square() / 4
        return SimpleNamespace(logits=logits, past_key_values=_StubCache())


class _WordIdTokenizer(_UniformTokenizer):
    """One token per word with a word-specific id, so every prefix ends differently."""

    word_ids = {"foo": 2, "bar": 5, "baz": 7}

    def __call__(self, text, **kwargs):
        encoded = super().__call__(text, **kwargs)
        encoded["input_ids"] = [self.word_ids[word] for word in text.split()]
        return encoded


def _make_lm(n_samples=4, max_tokens=3, chunk_size=None, seed=0, add_bos=True):
    return build_estimator(
        ContinuationEntropyLM,
        _UniformLM(),
        tokenizer=_UniformTokenizer(),
        name=f"cont_entropy_mock_N{n_samples}_L{max_tokens}",
        n_samples=n_samples,
        seed=seed,
        add_bos=add_bos,
        chunk_size=chunk_size,
        max_tokens=max_tokens,
    )


@pytest.mark.parametrize("chunk_size", [None, 4])
def test_uniform_outputs_preserve_samples_and_analytical_entropy(chunk_size):
    lm = _make_lm(n_samples=8, max_tokens=2, chunk_size=chunk_size)
    outputs = lm(_make_stim("foo bar baz"))
    assert len(outputs) == 3
    for output in outputs:
        samples = output.values["log_p_samples_nats"]
        assert isinstance(samples, np.ndarray)
        assert samples.dtype == np.float64
        assert samples.shape == (8,)
        np.testing.assert_allclose(samples, -2 * math.log(4), rtol=1e-12)
        entropy = output.values["continuation_entropy"]
        assert entropy == pytest.approx(4.0, rel=1e-9)
        assert entropy == pytest.approx(-samples.mean() / math.log(2), rel=0, abs=1e-12)


def test_bos_false_yields_nan_for_first_word():
    """add_bos=False + first word with no left context → NaN entropy."""
    lm = _make_lm(n_samples=4, max_tokens=2, add_bos=False)
    outs = lm(_make_stim("foo bar baz"))
    # word 0 has pred_pos = -1 → NaN
    assert math.isnan(outs[0].values["continuation_entropy"])
    # words 1, 2 have context → finite
    assert math.isfinite(outs[1].values["continuation_entropy"])
    assert math.isfinite(outs[2].values["continuation_entropy"])


def _bigram_lm(**kwargs):
    lm = _make_lm(**kwargs)
    lm._model = _BigramLM()
    lm._tokenizer = _WordIdTokenizer()
    return lm


def _reference_log_p(lm, prefix_end_token: int, word_index: int) -> np.ndarray:
    """Joint log-probabilities (nats) of the continuations one word's uniforms select.

    Every step inverts the model's next-token CDF, given the previous token,
    at the uniform the sample and step own.
    """
    uniforms = lm._uniform_block(("c", word_index), 0, lm.n_samples, lm.max_tokens)
    log_p = np.zeros(lm.n_samples)
    for sample in range(lm.n_samples):
        token = prefix_end_token
        for step in range(lm.max_tokens):
            with torch.no_grad():
                logits = lm._model(torch.tensor([[token]])).logits[0, -1].double()
            probs = torch.softmax(logits, dim=-1).numpy()
            cdf = np.cumsum(probs)
            cdf = cdf / cdf[-1]
            u = float(uniforms[sample, step])
            token = min(int(np.searchsorted(cdf, u)), len(probs) - 1)
            log_p[sample] += math.log(max(float(probs[token]), 1e-40))
    return log_p


def test_one_pass_draws_follow_each_word_prefix_and_uniforms():
    """Each word's draws start at the token before it and use its own uniforms.

    Under the bigram model a draw taken from another prefix, with another
    word's or sample's uniform, or scored at another step changes the joint
    log-probability. Chunks of 4 out of 6 samples also check that a chunk
    takes the uniforms of the samples it holds.
    """
    stimulus = "foo bar baz"
    lm = _bigram_lm(n_samples=6, max_tokens=3, chunk_size=4)
    outs = lm(_make_stim(stimulus))
    # The prefix of word i ends at BOS (id 0) or at the preceding word's id.
    prefix_ends = [0, *(_WordIdTokenizer.word_ids[w] for w in stimulus.split()[:-1])]
    for word_index, (fmo, prefix_end) in enumerate(zip(outs, prefix_ends, strict=True)):
        samples = fmo.values["log_p_samples_nats"]
        expected = _reference_log_p(lm, prefix_end, word_index)
        np.testing.assert_allclose(samples, expected, rtol=0, atol=1e-12)
        assert len(np.unique(samples)) > 1

    reseeded = _bigram_lm(n_samples=6, max_tokens=3, chunk_size=4, seed=999)
    moved = reseeded(_make_stim(stimulus))[1].values["log_p_samples_nats"]
    assert not np.array_equal(moved, outs[1].values["log_p_samples_nats"])


def test_no_units_returns_empty_outputs():
    """A stim with an empty word layer should give zero outputs."""
    s = Stimulus(stimulus_id="t", text="")
    s.add_layer(AnnotationLayer(name="words", spans=[], values=[]))
    lm = _make_lm()
    outs = lm(s)
    assert outs == []


def test_character_layer_is_refused():
    """A per-character layer would leave every character but one per token NaN."""
    stim = _make_stim("foo bar")
    attach_characters([stim])
    with pytest.raises(ValueError, match="word-level target_layer.*CharBeamLM"):
        _make_lm()(stim, target_layer=CHARACTER_LAYER)
