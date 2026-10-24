"""Invariants that every surprisal-computing :class:`FeatureModel` must
satisfy. Pins:

1. Idempotence, calling the same LM on the same stimulus twice
   produces bit-identical numbers.
2. Order-independence, processing stimulus B after stimulus A on a
   shared LM instance produces exactly the same numbers as processing
   stimulus B on a fresh LM.

Both are required for the per-trial conditioning contract: each
stimulus is its own reading episode and the LM must not carry context
across stimuli. Mutating ``past_key_values`` in place across calls
would violate #2 and eventually overflow GPT-2's position embedding;
this file pins the contract.

Uses ``distilgpt2`` and a tiny synthetic stimulus to keep the tests
fast. The Pimentel & Meister (2024) path is exercised via the ``gpt2``
alias to ``gpt2-small``.
"""

import numpy as np
import pytest

from replica import Surprisal
from replica.core import AnnotationLayer, Span, Stimulus
from replica.feature import surprisal_model
from tests._stimuli import word_stimulus

pytestmark = pytest.mark.needs_model


@pytest.fixture(scope="module")
def stim_a() -> Stimulus:
    # English ASCII, every surprisal method should be able to score it
    # without tokeniser coverage holes.
    return word_stimulus("the cat sat on the mat", "test_a")


@pytest.fixture(scope="module")
def stim_b() -> Stimulus:
    return word_stimulus("a quick brown fox jumps over a lazy dog", "test_b")


def _eval(lm, stim) -> np.ndarray:
    """Run ``lm`` on ``stim`` and return the Surprisal feature values."""
    # A feature is called as (stimulus, outputs, target_layer); the default
    # target layer is the ``words`` layer the outputs are aligned to.
    return Surprisal()(stim, lm(stim))


@pytest.fixture
def _force_cpu(monkeypatch):
    """Run the determinism checks on CPU by reporting CUDA as unavailable.

    The contract under test is statelessness. CharBeamLM's genlm byte beam uses
    CUDA internally (gated on ``torch.cuda.is_available``, independent of the HF
    model's device), and GPU float ops are non-associative, which would break
    the exact equality below. Disabling CUDA makes every backend (incl. genlm)
    compute on CPU, where the same op sequence is bit-deterministic.
    """
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)


# Method builders: each must return a fresh, fully-initialised LM.
_METHOD_BUILDERS = [
    pytest.param(
        lambda: surprisal_model("distilgpt2", method="surprisal", add_bos=True),
        id="standard",
    ),
    pytest.param(
        lambda: surprisal_model("distilgpt2", method="char_beam", K=3),
        id="char_beam",
    ),
    pytest.param(
        lambda: surprisal_model("gpt2", method="wordsprobability_surprisal_buggy"),
        id="wordsprobability_buggy",
    ),
    pytest.param(
        lambda: surprisal_model("gpt2", method="wordsprobability_surprisal"),
        id="pim_corr",
    ),
]


@pytest.mark.parametrize("build_lm", _METHOD_BUILDERS)
def test_idempotent(_force_cpu, build_lm, stim_a):
    """Same LM, same stimulus, two calls → identical numbers (exact)."""
    lm = build_lm()
    v1 = _eval(lm, stim_a)
    v2 = _eval(lm, stim_a)
    assert v1.shape == v2.shape
    # Exact equality: any drift indicates non-deterministic state
    # (cache mutation, RNG, etc.) which violates the contract.
    np.testing.assert_array_equal(v1, v2)


@pytest.mark.parametrize("build_lm", _METHOD_BUILDERS)
def test_order_independent(_force_cpu, build_lm, stim_a, stim_b):
    """Shared LM after A → B must equal a fresh LM on B alone.

    This pins the per-trial conditioning contract: an LM must not carry
    KV-cache / hidden-state context from one stimulus into the next.
    """
    shared = build_lm()
    _ = _eval(shared, stim_a)  # prime with A
    after = _eval(shared, stim_b)

    fresh = build_lm()
    alone = _eval(fresh, stim_b)

    assert after.shape == alone.shape
    np.testing.assert_array_equal(after, alone)


def test_char_beam_handles_multibyte_utf8():
    """``CharBeamLM`` operates on UTF-8 bytes internally, so a stimulus of
    multi-byte chars (Chinese here) yields finite log-probabilities on every
    char.

    Without the byte-level decode, GPT-2's vocabulary tokens (decoded as
    byte-proxy chars like ``'æĪ'``) match no single Chinese character in the
    input and every char comes back as NaN.
    """
    from replica.feature import CharBeamLM

    # 12 Chinese characters → 36 UTF-8 bytes via BPE.
    text = "我怀念那些日子里同事建立"
    chars = list(text)
    spans = [Span(i, i + 1) for i in range(len(chars))]
    stim = Stimulus(text=text, stimulus_id="test_zh")
    stim.add_layer(AnnotationLayer(name="characters", spans=spans, values=chars))

    lm = CharBeamLM("gpt2", K=3)
    out = lm(stim, target_layer="characters")
    assert len(out) == len(chars)

    log_ps = np.array([o.values["log_p_observed"] for o in out])
    assert np.isfinite(log_ps).all(), (
        f"every Chinese character should have a finite log-prob; got {log_ps}"
    )
    # Sanity: per-character surprisals should be in a plausible range
    # (gpt2 isn't trained on much Chinese, so values are typically large).
    assert (-log_ps > 0).all()
    assert (-log_ps < 200).all(), (
        f"per-char surprisal looks pathological; got {-log_ps}"
    )
