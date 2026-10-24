"""attach_characters produces the focal-area geometry's index space."""

import pytest

from replica.core import AnnotationLayer, Span
from replica.feature.core.annotation.characters import (
    CHARACTER_LAYER,
    attach_characters,
)
from tests._stimuli import s0_word_stimulus as _stimulus


def test_one_span_per_character_including_whitespace():
    stimulus = _stimulus("The cat sat.")
    attach_characters([stimulus])
    layer = stimulus.layers[CHARACTER_LAYER]
    assert len(layer.spans) == len(stimulus.text)
    assert [stimulus.text[s.start : s.end] for s in layer.spans] == list(stimulus.text)
    assert layer.values == list(stimulus.text)


def test_existing_layer_is_kept():
    stimulus = _stimulus("a b")
    marker = AnnotationLayer(name=CHARACTER_LAYER, spans=[Span(0, 1)], values=["a"])
    stimulus.add_layer(marker)
    attach_characters([stimulus])
    assert len(stimulus.layers[CHARACTER_LAYER].spans) == 1


def test_unnormalized_text_is_refused():
    stimulus = _stimulus("a  b")
    with pytest.raises(ValueError, match="whitespace-normalized"):
        attach_characters([stimulus])
