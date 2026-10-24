"""Regression tests for ``CharBeamLM`` (adapter around ``genlm.bytes.ByteBeamState``).

The algorithm is described in Giulianelli et al. (2024b), "On the Proper
Treatment of Tokenization in Psycholinguistics". REPLICA delegates to the
``genlm-bytes`` library.

Tests load the public ``gpt2`` checkpoint. ``slow`` because loading GPT-2 plus
running the beam search takes a few seconds.
"""

import pytest

from tests._stimuli import word_stimulus

# The module fixture constructs ``CharBeamLM("gpt2")``, which loads a real
# checkpoint; keep the whole module out of the model-free fast tier.
pytestmark = [pytest.mark.needs_model, pytest.mark.slow]

# Frozen per-byte log P(b_t | b_<t) golden values: GPT-2 small under
# top-K=5 character-beam marginalisation. ASCII English so per-byte ≡
# per-char. The tolerance |diff| < 1e-4 nats sits above the float32 noise
# floor because per-call AsyncTransformer init varies slightly across
# transformers versions; the K=5 approximation dominates any drift.
GOLDEN = {
    "The quick brown fox.": [
        -2.5024878389715868,
        -0.35540925516704736,
        -0.25318704444282725,
        -0.1710700391245199,
        -6.078322006921471,
        -0.0032302050872203836,
        -2.1657663058810694,
        -0.5392151963017628,
        -0.0013339562737382238,
        -0.6324270125385496,
        -3.956934800926131,
        -1.4855122938061598,
        -2.183570020430988,
        -0.21363079598809165,
        -0.7977217564004242,
        -0.4043220279293571,
        -3.1388246922584564,
        -1.1527312479108645,
        -0.8956643830062561,
        -4.364100322280425,
    ],
    "Hello world": [
        -3.8331539041876512,
        -1.2038145347573832,
        -2.2882250968725977,
        -0.6842063982296818,
        -0.27459553437423523,
        -0.3552641354757533,
        -4.83280554323297,
        -0.895489095229216,
        -0.20360667067049398,
        -0.0768070527812359,
        -0.00019997996238885207,
    ],
}


@pytest.fixture(scope="module")
def gpt2_lm():
    """One ``CharBeamLM`` per test module, since LM loading is expensive."""
    from replica.feature import CharBeamLM

    return CharBeamLM("gpt2", K=5)


@pytest.mark.slow
@pytest.mark.parametrize("sentence", list(GOLDEN.keys()))
def test_charbeam_lm_aggregates_to_unit_layer(gpt2_lm, sentence):
    """``CharBeamLM`` on a per-character layer must reproduce the
    genlm-bytes byte-level marginals (which == per-char marginals for
    ASCII English)."""
    from replica.core import AnnotationLayer, Span, Stimulus

    spans = [Span(i, i + 1) for i in range(len(sentence))]
    stim = Stimulus(text=sentence, stimulus_id="test")
    stim.add_layer(
        AnnotationLayer(name="characters", spans=spans, values=list(sentence))
    )
    out = gpt2_lm(stim, target_layer="characters")
    assert len(out) == len(sentence)

    expected = GOLDEN[sentence]
    actual = [o.values["log_p_observed"] for o in out]
    diffs = [abs(a - e) for a, e in zip(actual, expected, strict=False)]
    assert max(diffs) < 1e-4, (
        f"CharBeamLM drifted from goldens for {sentence!r}\n"
        f"max |diff| = {max(diffs):.2e}"
    )


@pytest.mark.slow
def test_charbeam_lm_word_aggregation(gpt2_lm, monkeypatch):
    """On a word layer each ``log_p_observed`` is the sum of the per-character
    log-probs in its span.

    The genlm byte beam is nondeterministic run to run, so the byte beam is
    stubbed with a fixed, distinct per-byte array and the comparison is exact.
    """
    import numpy as np

    from replica.core import AnnotationLayer, Span, Stimulus

    sentence = "The quick brown fox."  # ASCII: one byte == one char

    async def _fixed_byte_log_p(text_bytes):
        return -(np.arange(len(text_bytes), dtype=float) + 1.0)

    monkeypatch.setattr(gpt2_lm, "_byte_log_p", _fixed_byte_log_p)

    char_stim = Stimulus(text=sentence, stimulus_id="chars")
    char_stim.add_layer(
        AnnotationLayer(
            name="characters",
            spans=[Span(i, i + 1) for i in range(len(sentence))],
            values=list(sentence),
        )
    )
    per_char = [
        o.values["log_p_observed"]
        for o in gpt2_lm(char_stim, target_layer="characters")
    ]

    word_stim = word_stimulus(sentence, "test_words")
    word_spans = word_stim.layers["words"].spans

    out = gpt2_lm(word_stim, target_layer="words")
    assert len(out) == len(word_spans)

    for o, span, w in zip(out, word_spans, sentence.split(" "), strict=False):
        want = sum(per_char[i] for i in range(span.start, span.end))
        got = o.values["log_p_observed"]
        assert abs(got - want) < 1e-12, f"{w!r}: got {got}, want {want}"


@pytest.mark.slow
def test_charbeam_lm_target_layer_autodetect(gpt2_lm):
    """When ``target_layer`` is omitted, ``CharBeamLM`` should pick
    ``words`` if present, else ``characters``. Pin that contract."""
    from replica.core import AnnotationLayer, Span, Stimulus

    sentence = "Hello world"
    # Stimulus has both layers; words should win by default.
    stim_both = Stimulus(text=sentence, stimulus_id="both")
    stim_both.add_layer(
        AnnotationLayer(
            name="characters",
            spans=[Span(i, i + 1) for i in range(len(sentence))],
            values=list(sentence),
        )
    )
    stim_both.add_layer(
        AnnotationLayer(
            name="words",
            spans=[Span(0, 5), Span(6, 11)],
            values=["Hello", "world"],
        )
    )
    out = gpt2_lm(stim_both)
    assert len(out) == 2  # one output per word

    # Stimulus has only characters; that layer should be used.
    stim_chars = Stimulus(text=sentence, stimulus_id="chars_only")
    stim_chars.add_layer(
        AnnotationLayer(
            name="characters",
            spans=[Span(i, i + 1) for i in range(len(sentence))],
            values=list(sentence),
        )
    )
    out = gpt2_lm(stim_chars)
    assert len(out) == len(sentence)
