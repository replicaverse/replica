"""Window-cap and layer-segmentation extensions of WordInfoValueLM.

Both must be inert where the reference computation applies: the cap only
changes behavior on documents the reference errors out on, and layer
segmentation refuses corpora the reference's whitespace split already
handles.
"""

import math

import numpy as np
import pytest

from replica.core import AnnotationLayer, Span, Stimulus
from replica.feature.distance.word_infovalue import WordInfoValueLM
from tests._stimuli import s0_word_stimulus as _spaced_stimulus

pytestmark = [pytest.mark.needs_model]

_MODEL = "sshleifer/tiny-gpt2"


def _unspaced_stimulus() -> Stimulus:
    text = "我怀念那些日子"
    spans = [Span(0, 1), Span(1, 3), Span(3, 5), Span(5, 7)]
    stimulus = Stimulus(text=text, stimulus_id="zh0")
    stimulus.add_layer(
        AnnotationLayer(
            name="words", spans=spans, values=[text[s.start : s.end] for s in spans]
        )
    )
    return stimulus


def _rows(outputs):
    return [[v for k, v in sorted(out.values.items())] for out in outputs]


def test_cap_is_inert_when_the_document_fits():
    stimulus = _spaced_stimulus("the cat sat on the mat")
    reference = WordInfoValueLM(_MODEL, k=3, max_new_tokens=2, seed=0)
    capped = WordInfoValueLM(_MODEL, k=3, max_new_tokens=2, seed=0, window_policy="cap")
    assert np.allclose(
        _rows(reference(stimulus)), _rows(capped(stimulus)), equal_nan=True
    )


def test_cap_trims_to_a_fitting_word_suffix():
    words = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta"]
    stimulus = _spaced_stimulus(" ".join(words))
    lm = WordInfoValueLM(_MODEL, k=2, max_new_tokens=2, seed=0, window_policy="cap")
    lm._ensure_model()
    lm._model.config.n_positions = 12
    i = len(words) - 1
    context = lm._trimmed_context(stimulus, words, "words", i)
    assert context is not None
    full = " ".join(words[:i])
    assert context != full and full.endswith(context)
    assert lm._token_len(context) + lm.max_new_tokens + 2 <= 12
    lm.window_policy = "reference"
    assert lm._trimmed_context(stimulus, words, "words", i) == full


def test_layer_segmentation_iterates_the_corpus_words():
    stimulus = _unspaced_stimulus()
    whitespace = WordInfoValueLM(_MODEL, k=2, max_new_tokens=2, seed=0)
    with pytest.raises(ValueError, match="whitespace-splits into"):
        whitespace(stimulus)
    layered = WordInfoValueLM(
        _MODEL, k=2, max_new_tokens=2, seed=0, segmentation="layer"
    )
    outputs = layered(stimulus)
    assert len(outputs) == 4
    first = outputs[0].values
    assert all(math.isnan(v) for v in first.values())
    words = layered._unit_words(stimulus, "words")
    assert words == ["我", "怀念", "那些", "日子"]
    assert layered._trimmed_context(stimulus, words, "words", 2) == "我怀念"
    assert layered._separator(stimulus, "words") == ""


def test_layer_segmentation_scores_spaced_units_like_the_reference():
    stimulus = _spaced_stimulus("a spaced sentence here")
    reference = WordInfoValueLM(_MODEL, k=2, max_new_tokens=2, seed=0)
    layered = WordInfoValueLM(
        _MODEL, k=2, max_new_tokens=2, seed=0, segmentation="layer"
    )
    assert layered._separator(stimulus, "words") == " "
    assert layered._unit_words(stimulus, "words") == reference._unit_words(
        stimulus, "words"
    )
    assert np.allclose(
        _rows(reference(stimulus)), _rows(layered(stimulus)), equal_nan=True
    )
    # A unit of two words is one unit: its context is the units before it,
    # joined by spaces, and its surface keeps the inner space.
    text = "a spaced sentence here"
    spans = [Span(0, 1), Span(2, 17), Span(18, 22)]
    fused = Stimulus(text=text, stimulus_id="fused")
    fused.add_layer(
        AnnotationLayer("words", spans, [text[s.start : s.end] for s in spans])
    )
    units = layered._unit_words(fused, "words")
    assert units == ["a", "spaced sentence", "here"]
    assert layered._trimmed_context(fused, units, "words", 2) == "a spaced sentence"
    outputs = layered(fused)
    assert len(outputs) == 3
    assert all(not math.isnan(v) for v in outputs[1].values.values())
    with pytest.raises(ValueError, match="whitespace-splits into"):
        reference(fused)
