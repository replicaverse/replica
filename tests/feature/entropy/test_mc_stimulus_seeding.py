"""MC entropy draws are a function of the stimulus alone.

Values are not bit-exact across batch sizes: the forward pass is not
batch-size invariant, so a uniform can land across a shifted bin edge and flip
a token. Same mechanism as cross-GPU divergence; seeding cannot remove it.
"""

import numpy as np
import pytest

from replica.core import Stimulus
from replica.feature.entropy import MCWordEntropyLM
from tests._stimuli import word_stimulus

pytestmark = [pytest.mark.needs_data]

_MODEL = "gpt2"


def _stim(stimulus_id: str, words: list[str]) -> Stimulus:
    return word_stimulus(" ".join(words), stimulus_id)


A = _stim("stim-A", ["The", "cat", "sat", "on", "the", "mat"])
B = _stim("stim-B", ["Dogs", "bark", "loudly", "at", "night"])


def _lm(seed=0, chunk=4, n_samples=8):
    return MCWordEntropyLM(
        _MODEL,
        n_samples=n_samples,
        max_iter=20,
        alpha=1.0,
        chunk_size=chunk,
        seed=seed,
    )


def _vals(lm, stim) -> np.ndarray:
    return np.array(
        [
            u.values["mc_word_entropy"]
            for u in lm(stim, extract=None, target_layer="words")
        ]
    )


def test_a_stimulus_is_unaffected_by_what_was_computed_before_it():
    """The property that makes a sharded corpus agree with an unsharded one."""
    lm = _lm()
    _vals(lm, A)
    after_a = _vals(lm, B)

    alone = _vals(_lm(), B)

    np.testing.assert_array_equal(after_a, alone)


def test_order_of_stimuli_does_not_change_any_of_them():
    lm_forward = _lm()
    fwd = {"A": _vals(lm_forward, A), "B": _vals(lm_forward, B)}

    lm_reverse = _lm()
    rev = {"B": _vals(lm_reverse, B), "A": _vals(lm_reverse, A)}

    np.testing.assert_array_equal(fwd["A"], rev["A"])
    np.testing.assert_array_equal(fwd["B"], rev["B"])


def test_repeating_one_stimulus_on_the_same_lm_repeats_its_values():
    lm = _lm()
    np.testing.assert_array_equal(_vals(lm, A), _vals(lm, A))


def test_renaming_a_stimulus_does_not_change_its_values():
    """The draw keys on content, so the predictor is a function of its input."""
    same_words = ["The", "cat", "sat", "on", "the", "mat"]
    one = _vals(_lm(), _stim("id-one", same_words))
    two = _vals(_lm(), _stim("id-two", same_words))
    np.testing.assert_array_equal(one, two)


def test_different_text_gets_different_draws():
    a = _vals(_lm(), _stim("x", ["The", "cat", "sat", "on", "the", "mat"]))
    b = _vals(_lm(), _stim("x", ["Dogs", "bark", "loudly", "at", "night"]))
    assert not np.array_equal(a[:5], b)


def test_release_stream_mode_restores_position_dependence():
    """The escape hatch the upstream bit-identity anchor runs under."""
    lm = _lm()
    lm.release_rng_stream = True
    _vals(lm, A)
    after_a = _vals(lm, B)
    lm2 = _lm()
    lm2.release_rng_stream = True
    assert not np.array_equal(after_a, _vals(lm2, B))


def test_the_seed_still_controls_the_draw():
    assert not np.array_equal(_vals(_lm(seed=0), A), _vals(_lm(seed=1), A))


def test_an_unseeded_model_is_left_alone():
    """seed=None keeps the global RNG, so nothing is re-seeded under it."""
    lm = MCWordEntropyLM(
        _MODEL,
        n_samples=8,
        max_iter=20,
        alpha=1.0,
        chunk_size=4,
        seed=None,
    )
    assert lm._rng is None
    lm._seed_for_stimulus(A, "words")  # must not raise
    assert lm._rng is None


def test_chunk_size_no_longer_selects_the_draw():
    """Agrees within float noise, since the forward pass is batch-dependent."""
    a = _vals(_lm(chunk=4), A)
    b = _vals(_lm(chunk=8), A)
    np.testing.assert_allclose(a, b, rtol=0, atol=1e-3)


def test_a_sample_draws_the_same_uniforms_whatever_the_batch():
    """The batch-invariant part, asserted with no model in the way."""
    lm = _lm()
    lm._stimulus_seed = 12345
    full = lm._uniform_block(("w", 7), 0, 8, 20)
    second_half = lm._uniform_block(("w", 7), 4, 4, 20)
    np.testing.assert_array_equal(full[4:].cpu().numpy(), second_half.cpu().numpy())


def test_values_are_finite_and_one_per_word():
    vals = _vals(_lm(), A)
    assert vals.shape == (6,)
    assert np.isfinite(vals).all()
