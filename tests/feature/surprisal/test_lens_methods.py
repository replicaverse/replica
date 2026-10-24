"""The lens MethodSpecs: layer keys from the config, extract filtering,
sentence scope, and the log_p_observed exclusion.

The mirrored ``log_p_observed`` must never reach a stored cell: the lens
convention (leading-space words, BOS) disagrees with the ``surprisal`` cell
of the same model, and the store must not hold two conflicting columns
under one predictor name.
"""

import numpy as np
import pytest

from replica.compute.methods import (
    METHOD_SPECS,
    ComputeContext,
)
from replica.context import SENTENCE_LAYER
from replica.core import AnnotationLayer
from tests._stimuli import s0_word_stimulus as _word_stimulus

_MODEL = "sshleifer/tiny-gpt2"


def _context(model_id: str = _MODEL, **overrides) -> ComputeContext:
    base = dict(
        model_id=model_id,
        beam_width=5,
        n_samples=64,
        max_tokens=10,
        alpha=1.0,
        seed=0,
        add_bos=True,
        dataset_language=None,
        language_override=None,
    )
    return ComputeContext(**{**base, **overrides})


@pytest.mark.needs_model
def test_extract_keys_name_every_layer_and_nothing_else():
    keys = METHOD_SPECS["logit_lens_surprisal"].extract_keys(_context())
    # tiny-gpt2 has 2 blocks -> layers 0..2 (0 = embeddings).
    assert keys == [f"logit_lens_log_p_L{i}" for i in range(3)]
    tuned = METHOD_SPECS["tuned_lens_surprisal"].extract_keys(_context())
    assert tuned == [f"tuned_lens_log_p_L{i}" for i in range(3)]


@pytest.mark.needs_model
def test_stored_values_carry_no_log_p_observed():
    spec = METHOD_SPECS["logit_lens_surprisal"]
    context = _context()
    lm = spec.build_model(context)
    keys = spec.extract_keys(context)
    outputs = lm(_word_stimulus("The cat sat down."), extract=keys)
    assert len(outputs) == 4
    for out in outputs:
        assert set(out.values) == set(keys)


@pytest.mark.needs_model
def test_windowed_scoring_matches_full_context_where_context_agrees(
    monkeypatch,
):
    """Sliding windows change nothing for positions whose context they keep.

    A stimulus longer than the (forced) window must still score every word,
    and every target position filled by the first window sees exactly the
    full-context left context, so those words must match a full-context run.
    """
    from replica.feature.surprisal import logit_lens as lens_mod

    words = ("the quick brown fox jumps over the lazy dog again and " * 4).split()
    stimulus = _word_stimulus(" ".join(words))

    reference = lens_mod.LogitLensLM(_MODEL, device="cpu")(stimulus)

    window = 16
    monkeypatch.setattr(
        lens_mod, "usable_position_limit", lambda _name, _config: window
    )
    windowed = lens_mod.LogitLensLM(_MODEL, device="cpu")(stimulus)

    assert len(windowed) == len(reference) == len(words)
    keys = [k for k in windowed[0].values if k.startswith("logit_lens_log_p_L")]
    assert keys
    got = np.array([[o.values[k] for k in keys] for o in windowed])
    want = np.array([[o.values[k] for k in keys] for o in reference])
    assert np.isfinite(got).all()
    # tiny-gpt2 encodes each of these words as one token, so word i is
    # target position i; the first window fills targets 0..window-1 with
    # untruncated context.
    np.testing.assert_allclose(got[: window - 1], want[: window - 1], rtol=1e-5)
    # Later positions were scored under windowed (shorter) context, so a
    # blanket match would mean the window was silently ignored.
    assert not np.allclose(got, want, rtol=1e-5)


@pytest.mark.needs_model
def test_sentence_scope_matches_independent_sentences():
    spec = METHOD_SPECS["logit_lens_surprisal"]
    lm_sent = spec.build_model(_context(context_scope="sentence"))
    lm_flat = spec.build_model(_context())

    first = "The cat sat on the mat."
    second = "A dog barked loudly outside."
    stimulus = _word_stimulus(f"{first} {second}")
    words = stimulus.layers["words"]
    stimulus.add_layer(
        AnnotationLayer(
            name=SENTENCE_LAYER, spans=list(words.spans), values=[1] * 6 + [2] * 5
        )
    )

    scoped = lm_sent(stimulus)
    expected = lm_flat(_word_stimulus(first)) + lm_flat(_word_stimulus(second))
    assert len(scoped) == len(expected) == 11
    for key in scoped[0].values:
        got = [o.values[key] for o in scoped]
        want = [o.values[key] for o in expected]
        np.testing.assert_allclose(got, want, rtol=1e-5)
