"""Sequence context is independent of dataset names and annotation unit types."""

from copy import deepcopy

import pytest

from replica.context import (
    CONTEXT_SEQUENCE,
    PRECEDING_CONTEXT,
    attach_preceding_context,
    context_input,
)
from replica.core import AnnotationLayer, Span, Stimulus
from replica.feature.core.composition.context import PrecedingContextModel
from replica.feature.core.protocols import FeatureModelOutput


def _part(text, position, group="document"):
    return Stimulus(
        text=text,
        stimulus_id=f"{group}-{position}",
        metadata={CONTEXT_SEQUENCE: {"group": group, "position": position}},
        layers={
            "characters": AnnotationLayer(
                "characters", [Span(i, i + 1) for i in range(len(text))], list(text)
            ),
            "paragraphs": AnnotationLayer("paragraphs", [Span(0, len(text))], [text]),
        },
    )


class PrefixModel:
    name = "prefix"
    vocab = None

    def __call__(self, stimulus, extract=None, target_layer="paragraphs"):
        return [
            FeatureModelOutput(values={"prefix": stimulus.text[: span.start]})
            for span in stimulus.layers[target_layer].spans
        ]


@pytest.mark.parametrize("layer,offset", [("characters", 3), ("paragraphs", 1)])
def test_context_scores_requested_layer_without_word_or_page_metadata(layer, offset):
    parts = [_part("a b", 1), _part("cd", 2), _part("later", 3)]
    originals = deepcopy(parts)
    attach_preceding_context(list(reversed(parts)))
    combined, start = context_input(parts[1], layer)
    assert combined.text == "a b cd"
    assert start == offset
    assert combined.unit_strings(layer)[start:] == parts[1].unit_strings(layer)
    got = PrecedingContextModel(PrefixModel())(parts[1], target_layer=layer)
    expected = [
        combined.text[: span.start] for span in combined.layers[layer].spans[start:]
    ]
    assert [output.values["prefix"] for output in got] == expected
    assert len(got) == len(parts[1].layers[layer])
    assert got[0].metadata["target_unit_index"] == 0
    assert got[0].metadata["context_unit_index"] == start
    for original, part in zip(originals, parts, strict=True):
        assert part.text == original.text
        assert part.layers == original.layers
        assert part.stimulus_id == original.stimulus_id


def test_missing_preceding_layer_is_not_reconstructed_with_another_segmenter():
    parts = [_part("ab", 1), _part("cd", 2)]
    del parts[0].layers["characters"]
    attach_preceding_context(parts)
    with pytest.raises(ValueError, match="Missing preceding layer 'characters'"):
        context_input(parts[1], "characters")
    assert context_input(parts[1], "paragraphs")[1] == 1


def test_invalid_sequence_does_not_partially_attach_context():
    parts = [_part("first", 1), _part("third", 3, "other")]
    with pytest.raises(ValueError, match="Missing preceding"):
        attach_preceding_context(parts)
    assert all(PRECEDING_CONTEXT not in part.metadata for part in parts)


def test_duplicate_positions_are_rejected_even_with_distinct_ids():
    parts = [_part("first", 1), _part("other first", 1)]
    parts[1].stimulus_id = "different"
    with pytest.raises(ValueError, match="Duplicate context position"):
        attach_preceding_context(parts)


def test_model_output_must_match_context_units():
    class ShortModel(PrefixModel):
        def __call__(self, stimulus, extract=None, target_layer="paragraphs"):
            return super().__call__(stimulus, extract, target_layer)[:-1]

    parts = [_part("ab", 1), _part("cd", 2)]
    attach_preceding_context(parts)
    with pytest.raises(ValueError, match="incorrect unit count"):
        PrecedingContextModel(ShortModel())(parts[1], target_layer="characters")
