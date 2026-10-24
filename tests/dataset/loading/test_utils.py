"""Tests for dataset utility functions."""

import math

import numpy as np
import pytest

from replica.core import AnnotationLayer, Span, Stimulus
from replica.dataset.loading.utils import (
    append_measure_datums,
    build_word_spans,
    parse_float,
    parse_reading_measure,
    scatter_first_occurrence,
    scatter_measure_tuples,
)


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("3.14", 3.14),
        ("42", 42.0),
        ("-1.5", -1.5),
        ("0", 0.0),
    ],
)
def test_parse_float_numeric(raw, expected):
    """``parse_float`` converts well-formed numeric strings."""
    assert parse_float(raw) == expected


@pytest.mark.parametrize("raw", [".", "", "NA", "NaN", "not_a_number", None, object()])
def test_parse_float_returns_nan(raw):
    """``parse_float`` returns NaN for the documented missing-value tokens
    (".", "", "NA", "NaN"), for unparseable strings and for non-string values."""
    assert math.isnan(parse_float(raw))


@pytest.mark.parametrize("raw", ["", "NA", "NaN", np.nan, np.inf])
def test_parse_reading_measure_returns_nan_for_unobserved_values(raw):
    assert math.isnan(parse_reading_measure(raw))


def test_parse_reading_measure_preserves_nonzero_finite_values():
    assert parse_reading_measure("12.5") == 12.5


def test_append_measure_datums_constructs_aligned_datums():
    words = ["one", "two"]
    stimulus = Stimulus(
        text=" ".join(words),
        stimulus_id="example",
        layers={"words": AnnotationLayer("words", build_word_spans(words), words)},
    )
    output = []
    arrays = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]

    append_measure_datums(
        output,
        stimulus=stimulus,
        participant_id="p1",
        measure_names=["first_fixation_duration", "first_run"],
        measure_values=arrays,
        metadata={"site": "en"},
    )

    assert [datum.measure_name for datum in output] == [
        "first_fixation_duration",
        "first_run",
    ]
    assert all(datum.stimulus is stimulus for datum in output)
    assert all(datum.participant_id == "p1" for datum in output)
    assert all(datum.metadata == {"site": "en"} for datum in output)
    np.testing.assert_array_equal(output[0].values, arrays[0])
    np.testing.assert_array_equal(output[1].values, arrays[1])


def test_append_measure_datums_rejects_misaligned_names_and_arrays():
    words = ["one"]
    stimulus = Stimulus(
        text="one",
        stimulus_id="example",
        layers={"words": AnnotationLayer("words", build_word_spans(words), words)},
    )
    with pytest.raises(ValueError, match="2 measure names but 1 value arrays"):
        append_measure_datums(
            [],
            stimulus=stimulus,
            participant_id="p1",
            measure_names=["first_fixation_duration", "first_run"],
            measure_values=[np.array([1.0])],
        )


@pytest.mark.parametrize(
    "words, expected_spans",
    [
        (["hello"], [Span(0, 5)]),
        (["the", "cat"], [Span(0, 3), Span(4, 7)]),
        (["a", "big", "dog"], [Span(0, 1), Span(2, 5), Span(6, 9)]),
        ([], []),
    ],
)
def test_build_word_spans_layout(words, expected_spans):
    """``build_word_spans`` produces single-space-gap spans (or [] for no input)."""
    assert build_word_spans(words) == expected_spans


def test_build_word_spans_extract_round_trip():
    """Spans returned by build_word_spans must extract() back to the input
    words when the joining text is ' '.join(words)."""
    words = ["the", "quick", "brown", "fox"]
    text = " ".join(words)
    spans = build_word_spans(words)
    for span, word in zip(spans, words, strict=True):
        assert span.extract(text) == word


def test_scatter_measure_tuples_places_and_nan_fills():
    """Each (key, values) lands at index_of[key] across the per-measure
    arrays; positions with no record stay NaN."""
    items = [("a", (1.0, 10.0)), ("c", (3.0, 30.0))]
    index_of = {"a": 0, "b": 1, "c": 2}
    arrays = scatter_measure_tuples(items, index_of, n_units=3, n_measures=2)
    assert len(arrays) == 2
    np.testing.assert_array_equal(arrays[0], np.array([1.0, np.nan, 3.0]))
    np.testing.assert_array_equal(arrays[1], np.array([10.0, np.nan, 30.0]))


def test_scatter_measure_tuples_skips_unknown_keys():
    """Keys absent from index_of are dropped."""
    items = [("a", (1.0,)), ("missing", (9.0,))]
    arrays = scatter_measure_tuples(items, {"a": 0}, n_units=1, n_measures=1)
    np.testing.assert_array_equal(arrays[0], np.array([1.0]))


def test_scatter_measure_tuples_skips_out_of_range_index():
    """Indices outside [0, n_units) are skipped."""
    items = [("a", (1.0,)), ("b", (2.0,))]
    arrays = scatter_measure_tuples(items, {"a": 0, "b": 5}, n_units=2, n_measures=1)
    np.testing.assert_array_equal(arrays[0], np.array([1.0, np.nan]))


def test_scatter_measure_tuples_rejects_wrong_tuple_length():
    items = [("a", (1.0, 2.0))]
    with pytest.raises(ValueError, match="expected 1"):
        scatter_measure_tuples(items, {"a": 0}, n_units=1, n_measures=1)


def test_scatter_first_occurrence_keeps_first_not_last():
    """Repeated keys keep their first value (contrast with the last-wins
    scatter_measure_tuples)."""
    items = [("a", (1.0,)), ("a", (99.0,)), ("b", (2.0,))]
    first = scatter_first_occurrence(items, {"a": 0, "b": 1}, 2, 1)
    last = scatter_measure_tuples(items, {"a": 0, "b": 1}, 2, 1)
    np.testing.assert_array_equal(first[0], np.array([1.0, 2.0]))  # first wins
    np.testing.assert_array_equal(last[0], np.array([99.0, 2.0]))  # last wins


def test_scatter_first_occurrence_rejects_wrong_tuple_length():
    items = [("a", (1.0, 2.0))]
    with pytest.raises(ValueError, match="expected 1"):
        scatter_first_occurrence(items, {"a": 0}, n_units=1, n_measures=1)


@pytest.mark.parametrize("raw", [0, "0", 0.0])
def test_parse_reading_measure_preserves_skips(raw):
    assert parse_reading_measure(raw) == 0.0


@pytest.mark.parametrize("raw", [".", "NA", "", np.nan])
def test_missing_measure_is_zero_only_with_source_evidence(raw):
    assert np.isnan(parse_reading_measure(raw))
    assert parse_reading_measure(raw, unfixated=True) == 0.0
