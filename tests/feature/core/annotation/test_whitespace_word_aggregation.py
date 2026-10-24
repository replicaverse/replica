"""Combining per-whitespace-word records into the units of a layer.

A unit of several whitespace words takes the sum of its words' log
probabilities; a whitespace word that crosses a unit boundary has no rule and
raises.
"""

import math

import pytest

from replica.core import AnnotationLayer, Span, Stimulus
from replica.feature.core.annotation.alignment import (
    WHITESPACE_UNIT,
    TokenBoundaryError,
    aggregate_whitespace_words,
    whitespace_spans,
)
from replica.feature.core.protocols import FeatureModelOutput


def _rows(values, key="log_p_observed"):
    return [
        FeatureModelOutput(values={key: float(v)}, metadata={"index": i})
        for i, v in enumerate(values)
    ]


def _stimulus(text, spans):
    stimulus = Stimulus(text=text, stimulus_id="s")
    stimulus.add_layer(
        AnnotationLayer("words", spans, [text[s.start : s.end] for s in spans])
    )
    return stimulus


def test_whitespace_spans_follow_every_kind_of_whitespace():
    text = "  of\tthe\ncat  "
    spans = whitespace_spans(text)
    assert [text[s.start : s.end] for s in spans] == ["of", "the", "cat"]
    assert spans == [Span(2, 4), Span(5, 8), Span(9, 12)]
    assert whitespace_spans("") == []
    assert whitespace_spans(" \n ") == []


def test_single_word_units_keep_their_records():
    stimulus = _stimulus("a bb c", [Span(0, 1), Span(2, 4), Span(5, 6)])
    rows = _rows([1.0, 2.0, 3.0])
    assert aggregate_whitespace_words(stimulus, "words", rows, "sum") == rows


def test_sum_adds_every_value_of_a_multiword_unit():
    stimulus = _stimulus(
        "of the cat in a mat", [Span(0, 6), Span(7, 10), Span(11, 15), Span(16, 19)]
    )
    rows = [
        FeatureModelOutput(
            values={"log_p_observed": float(v), "other": 10.0 * v},
            metadata={"model_name": "m", "index": i},
        )
        for i, v in enumerate([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    ]
    out = aggregate_whitespace_words(stimulus, "words", rows, "sum")
    assert [o.values["log_p_observed"] for o in out] == [3.0, 3.0, 9.0, 6.0]
    assert [o.values["other"] for o in out] == [30.0, 30.0, 90.0, 60.0]
    # The combined record carries the first word's metadata.
    assert out[0].metadata == {"model_name": "m", "index": 0}
    assert out[2].metadata["index"] == 3


def test_sum_propagates_nan():
    stimulus = _stimulus("of the", [Span(0, 6)])
    out = aggregate_whitespace_words(stimulus, "words", _rows([1.0, math.nan]), "sum")
    assert math.isnan(out[0].values["log_p_observed"])


def test_record_count_must_match_the_whitespace_words():
    stimulus = _stimulus("of the", [Span(0, 6)])
    with pytest.raises(ValueError, match="1 whitespace-word records for 2"):
        aggregate_whitespace_words(stimulus, "words", _rows([1.0]), "sum")


@pytest.mark.parametrize("rule", ["mean", "first"])
def test_other_rules_are_refused(rule):
    stimulus = _stimulus("of the", [Span(0, 6)])
    with pytest.raises(ValueError, match=f"unknown multi-word rule '{rule}'"):
        aggregate_whitespace_words(stimulus, "words", _rows([1.0, 2.0]), rule)


def test_a_word_crossing_a_unit_boundary_raises():
    stimulus = _stimulus("ofthe cat", [Span(0, 2), Span(2, 5), Span(6, 9)])
    with pytest.raises(TokenBoundaryError):
        aggregate_whitespace_words(stimulus, "words", _rows([1.0, 2.0]), "sum")


def test_a_unit_without_a_whitespace_word_raises():
    # The second unit covers only whitespace, so no record falls in it.
    stimulus = _stimulus("of the", [Span(0, 2), Span(2, 3), Span(3, 6)])
    with pytest.raises(TokenBoundaryError):
        aggregate_whitespace_words(stimulus, "words", _rows([1.0, 2.0]), "sum")


def test_whitespace_unit_name():
    assert WHITESPACE_UNIT == "whitespace_words"
