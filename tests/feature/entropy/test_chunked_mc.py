"""Chunked entropy sampling, stored samples and constructor validation."""

import math

import numpy as np
import pytest
import torch

from replica.feature.entropy.estimators import (
    ContinuationEntropyLM,
    MCWordEntropyLM,
)
from tests._stimuli import word_stimulus
from tests.feature.entropy._entropy_mocks import (
    UniformCausalLM as _UniformLM,
)
from tests.feature.entropy._entropy_mocks import (
    WhitespaceTokenizer as _UniformTokenizer,
)
from tests.feature.entropy._entropy_mocks import build_estimator


def _stub_entropy_init(
    self,
    model_id,
    *,
    add_bos,
    seed,
    model_identity=None,
    start_ids=None,
    context_window=None,
):
    """Initialize only the state needed by constructor-validation tests."""
    self.model_id, self.add_bos, self._seed = model_id, add_bos, seed
    self._rng = None


def _make_continuation(n_samples=8, max_tokens=2, chunk_size=None, seed=0):
    return build_estimator(
        ContinuationEntropyLM,
        _UniformLM(),
        tokenizer=_UniformTokenizer(),
        name=f"cont_entropy_mock_N{n_samples}_L{max_tokens}",
        n_samples=n_samples,
        seed=seed,
        chunk_size=chunk_size,
        max_tokens=max_tokens,
    )


def _make_mcword(n_samples=8, max_iter=3, alpha=1.0, chunk_size=None, seed=0):
    return build_estimator(
        MCWordEntropyLM,
        _UniformLM(),
        tokenizer=_UniformTokenizer(),
        name=f"mcword_entropy_mock_N{n_samples}_M{max_iter}_a{alpha}",
        n_samples=n_samples,
        seed=seed,
        chunk_size=chunk_size,
        max_iter=max_iter,
        alpha=float(alpha),
    )


@pytest.mark.parametrize("chunk_size", [None, 1, 2, 4, 8, 100])
def test_uniform_continuation_entropy_matches_closed_form(chunk_size):
    value = _make_continuation(
        n_samples=8, max_tokens=3, chunk_size=chunk_size
    )._estimate([0, 1, 2])
    # Three draws from four equally likely tokens: 3 * log2(4) bits.
    assert value == pytest.approx(6.0, rel=1e-12)


@pytest.mark.parametrize(
    "make,kwargs,chunk_size",
    [
        pytest.param(
            _make_continuation, dict(max_tokens=3), size, id=f"continuation-{size}"
        )
        for size in (1, 3, 4, 8, 100)
    ]
    + [
        pytest.param(
            _make_mcword,
            dict(max_iter=3, alpha=alpha),
            size,
            id=f"mcword-{alpha}-{size}",
        )
        for alpha, size in (
            [(1.0, size) for size in (1, 2, 4, 8, 100)]
            + [(0.5, size) for size in (1, 2, 4, 8)]
            + [(2.0, 4)]
        )
    ],
)
def test_chunking_preserves_every_draw(make, kwargs, chunk_size):
    value, samples = _skewed(make, n_samples=8, **kwargs)._estimate_with_samples([2])
    chunked_value, chunked_samples = _skewed(
        make, n_samples=8, chunk_size=chunk_size, **kwargs
    )._estimate_with_samples([2])
    assert len(np.unique(samples)) > 1
    np.testing.assert_array_equal(chunked_samples, samples)
    assert chunked_value == value


class TestComputeSurprisalChunkPlumbing:
    """The CLI and construction context expose the chunk-size setting."""

    def test_ctx_dataclass_has_chunk_size(self):
        from replica.compute.cli import build_parser
        from replica.compute.methods import ComputeContext

        assert "chunk_size" in ComputeContext.__dataclass_fields__
        args = build_parser().parse_args(
            ["--dataset", "provo", "--lm", "gpt2", "--chunk-size", "8"]
        )
        assert args.chunk_size == 8


@pytest.mark.parametrize("alpha", [1.0, 0.5])
def test_mcword_stored_samples_reproduce_the_entropy(alpha):
    lm = _make_mcword(n_samples=8, max_iter=3, alpha=alpha)
    outputs = lm(word_stimulus("foo bar baz"))
    assert len(outputs) == 3
    for output in outputs[1:]:
        samples = output.values["log_p_samples_nats"]
        assert samples.shape == (8,)
        if alpha == 1.0:
            expected = -samples.mean() / math.log(2)
        else:
            expected = np.log2(np.mean(np.exp((alpha - 1) * samples))) / (1 - alpha)
        assert output.values["mc_word_entropy"] == pytest.approx(expected, rel=1e-12)


class TestConstructorValidation:
    """Exercise chunk-size validation with model loading replaced by a stub."""

    def test_continuation_rejects_zero(self, monkeypatch):
        from replica.feature.entropy import (
            estimators as est_mod,
        )

        # Stub _BaseEntropyEstimatorLM.__init__ so HF is never touched.
        monkeypatch.setattr(
            est_mod._BaseEntropyEstimatorLM, "__init__", _stub_entropy_init
        )
        with pytest.raises(ValueError, match="chunk_size must be ≥ 1"):
            est_mod.ContinuationEntropyLM("mock", chunk_size=0)

    def test_mcword_rejects_negative(self, monkeypatch):
        from replica.feature.entropy import (
            estimators as est_mod,
        )

        monkeypatch.setattr(
            est_mod._BaseEntropyEstimatorLM, "__init__", _stub_entropy_init
        )
        with pytest.raises(ValueError, match="chunk_size must be ≥ 1"):
            est_mod.MCWordEntropyLM("mock", chunk_size=-3)


class _SkewedLM(_UniformLM):
    """Unequal token probabilities make sampled log-probabilities seed-sensitive."""

    def __init__(self, vocab_size: int = 4):
        super().__init__(vocab_size)
        g = torch.Generator().manual_seed(1234)
        self._skew = torch.randn(vocab_size, generator=g)

    def forward(self, input_ids, past_key_values=None, use_cache=False, **kw):
        out = super().forward(input_ids, past_key_values, use_cache, **kw)
        out.logits = out.logits + self._skew
        return out


def _skewed(make, **kwargs):
    lm = make(**kwargs)
    lm._model = _SkewedLM(lm._tokenizer.vocab_size)
    return lm


@pytest.mark.parametrize(
    "make,kwargs",
    [
        (_make_continuation, dict(n_samples=8, max_tokens=3)),
        (_make_mcword, dict(n_samples=8, max_iter=3)),
    ],
)
def test_fresh_instance_same_seed_is_bit_reproducible(make, kwargs):
    """Equal seeds repeat prefix estimates; a changed seed changes the draws."""
    prefix = [0, 1, 2]
    a = _skewed(make, seed=7, **kwargs)._estimate(prefix)
    b = _skewed(make, seed=7, **kwargs)._estimate(prefix)
    c = _skewed(make, seed=8, **kwargs)._estimate(prefix)
    assert a == b, "same seed, fresh instances: estimates must be bit-identical"
    assert a != c, "different seeds must differ under a skewed (non-uniform) LM"
