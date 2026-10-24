"""The sampled estimators take a unit of several whitespace words as their input.

On a model double whose every draw costs log 2, the expected sampled cost of
a word is closed-form, so a unit of N words is checked against N times it.
"""

import math

import numpy as np
import pytest
import torch

from replica.core import AnnotationLayer, Span, Stimulus
from replica.feature.entropy._forward_word_lookahead import (
    ForwardWordLookaheadEntropyLM,
)
from replica.feature.entropy.estimators import MCWordEntropyLM
from tests._stimuli import word_stimulus
from tests.feature.entropy._entropy_mocks import (
    UniformCausalLM,
    WhitespaceTokenizer,
    build_estimator,
)


class _HalvingLM(UniformCausalLM):
    """Model double with p(b) = 1/2, p(Ġa) = p(Ġc) = 1/4: every draw costs log 2."""

    def forward(self, input_ids, past_key_values=None, use_cache=False, **_):
        out = super().forward(input_ids, past_key_values, use_cache)
        logits = torch.log(torch.tensor([1e-12, 0.25, 0.5, 0.25]))
        out.logits = logits.expand(*out.logits.shape).clone()
        return out


class _AfterBLM(_HalvingLM):
    """Halving double whose starters have mass 0.45 and 0.05 after the subword ``b``."""

    def forward(self, input_ids, past_key_values=None, use_cache=False, **_):
        out = super().forward(input_ids, past_key_values, use_cache)
        after_b = torch.log(torch.tensor([1e-12, 0.45, 0.5, 0.05]))
        last = input_ids[:, -1]
        out.logits = torch.where(
            (last == 2)[:, None, None], after_b.expand(*out.logits.shape), out.logits
        )
        return out


def _word_cost(max_iter: int) -> float:
    """Expected -log p of one sampled word in nats: the starter and a capped geometric count of draws."""
    draws = 2.0 * (1.0 - 2.0 ** -(max_iter - 1))
    return math.log(2.0) * (1.0 + draws)


def _fused_stimulus() -> Stimulus:
    text = "x of the y z w"
    spans = [Span(0, 1), Span(2, 8), Span(9, 10), Span(11, 12), Span(13, 14)]
    stimulus = Stimulus(text=text, stimulus_id="fused")
    stimulus.add_layer(
        AnnotationLayer("words", spans, [text[s.start : s.end] for s in spans])
    )
    return stimulus


def _mcword(n_samples=4096, max_iter=6, seed=0):
    return build_estimator(
        MCWordEntropyLM,
        _HalvingLM(),
        tokenizer=WhitespaceTokenizer(),
        name="mcword_double",
        n_samples=n_samples,
        seed=seed,
        chunk_size=None,
        max_iter=max_iter,
        alpha=1.0,
    )


def _means(outputs):
    return [-float(np.mean(o.values["log_p_samples_nats"])) for o in outputs]


def test_a_fused_unit_is_sampled_as_its_words():
    lm = _mcword()
    means = _means(lm(_fused_stimulus()))
    cost = _word_cost(6)
    # Unit 0 has no left context; the rest are one word, except "of the".
    assert math.isnan(means[0])
    tolerance = 0.15
    assert abs(means[1] - 2 * cost) < tolerance, (means[1], 2 * cost)
    for value in means[2:]:
        assert abs(value - cost) < tolerance, (value, cost)


def test_single_word_units_take_the_single_word_sampler(monkeypatch):
    lm = _mcword(n_samples=16)
    calls = []
    original = MCWordEntropyLM._sample_word_log_p

    def spy(self, *args, **kwargs):
        calls.append(kwargs["word_index"])
        return original(self, *args, **kwargs)

    monkeypatch.setattr(MCWordEntropyLM, "_sample_word_log_p", spy)
    lm(_fused_stimulus())
    assert calls == [2, 3, 4]


def test_the_same_seed_draws_the_same_unit_samples():
    a = _means(_mcword(n_samples=64, seed=3)(_fused_stimulus()))
    b = _means(_mcword(n_samples=64, seed=3)(_fused_stimulus()))
    c = _means(_mcword(n_samples=64, seed=4)(_fused_stimulus()))
    assert a[1:] == b[1:] and a[1:] != c[1:]


def test_a_word_cut_at_its_cap_feeds_its_subword_before_the_next_word():
    # max_iter 2: word 1 ends on an end-of-word draw (next starter at 1/2) or
    # is cut on b (next starter after b at 0.9 or 0.1); every other draw costs log 2.
    lm = build_estimator(
        MCWordEntropyLM,
        _AfterBLM(),
        tokenizer=WhitespaceTokenizer(),
        name="after_b_double",
        n_samples=4096,
        seed=0,
        chunk_size=None,
        max_iter=2,
        alpha=1.0,
    )
    means = _means(lm(_fused_stimulus()))
    after_b = -(0.9 * math.log(0.9) + 0.1 * math.log(0.1))
    expected = 3.5 * math.log(2.0) + 0.5 * after_b
    assert abs(means[1] - expected) < 0.1, (means[1], expected)
    # Drawing the starter from the logits before ``b`` would cost 4 log 2.
    assert abs(means[1] - 4 * math.log(2.0)) > 0.1


def test_the_windowed_driver_samples_a_fused_unit_as_its_words(monkeypatch):
    lm = _mcword()
    monkeypatch.setattr(lm, "_model_max_ctx_size", lambda: 24)
    means = _means(lm(_fused_stimulus()))
    cost = _word_cost(6)
    assert abs(means[1] - 2 * cost) < 0.15 and abs(means[2] - cost) < 0.15


def test_a_fused_unit_needs_a_second_draw():
    lm = _mcword(n_samples=4, max_iter=1)
    assert len(lm(word_stimulus("a b c", "single"))) == 3
    with pytest.raises(ValueError, match="max_iter of at least 2"):
        lm(_fused_stimulus())


def _lookahead(n_samples=4096, max_iter=12, max_words=2, seed=0):
    return build_estimator(
        ForwardWordLookaheadEntropyLM,
        _HalvingLM(),
        tokenizer=WhitespaceTokenizer(),
        name="lookahead_double",
        n_samples=n_samples,
        seed=seed,
        chunk_size=None,
        max_iter=max_iter,
        max_words=max_words,
        alpha=1.0,
    )


def test_the_look_ahead_takes_the_unit_as_its_first_word():
    lm = _lookahead()
    token_ids, _, positions, _ = lm._prepare_word_sampling(_fused_stimulus(), "words")
    by_unit = dict(positions)
    ctx = torch.tensor([token_ids[: by_unit[1] + 1]])
    with torch.no_grad():
        out = lm._model(ctx.repeat(4096, 1), use_cache=True)
    logits = out.logits[:, -1, :].double()
    kv = out.past_key_values.to_legacy_cache()
    logp, _ = lm._sample_k_words(
        logits, kv, 4096, lm._rng, token_ids[by_unit[1]], 0, 1, 0, first_words=2
    )
    cost = _word_cost(12)
    depth1, depth2 = -logp[:, 0].mean(), -logp[:, 1].mean()
    assert abs(depth1 - 2 * cost) < 0.15, (depth1, 2 * cost)
    assert abs(depth2 - cost) < 0.15, (depth2, cost)
    single, _ = lm._sample_k_words(
        logits, kv, 4096, lm._rng, token_ids[by_unit[1]], 0, 1, 0
    )
    assert abs(-single[:, 0].mean() - cost) < 0.15


@pytest.mark.parametrize("ctx_size", [None, 96])
def test_the_look_ahead_scores_a_fused_unit_at_depth_one(monkeypatch, ctx_size):
    lm = _lookahead()
    if ctx_size is not None:
        monkeypatch.setattr(lm, "_model_max_ctx_size", lambda: ctx_size)
    outputs = lm(_fused_stimulus())
    cost = _word_cost(12)
    steps = outputs[1].values["fwd_word_log_p_step_nats"]
    assert steps.shape == (4096, 2)
    assert abs(-steps[:, 0].mean() - 2 * cost) < 0.15
    assert abs(-steps[:, 1].mean() - cost) < 0.15
    single = outputs[2].values["fwd_word_log_p_step_nats"]
    assert abs(-single[:, 0].mean() - cost) < 0.15
