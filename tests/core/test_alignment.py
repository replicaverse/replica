"""Tests for alignment: TokenAlignment, align_by_spans."""

import numpy as np
import pytest

from replica.core import AnnotationLayer, Span, Stimulus
from replica.feature import (
    TokenAlignment,
    align_by_spans,
    align_from_tokenizer,
)


def make_stimulus_with_tokens():
    """Create a stimulus with word and subword token layers.

    Text: "unhappy cats"
    Words:  [Span(0,7)="unhappy", Span(8,12)="cats"]
    Tokens: [Span(0,2)="un", Span(2,7)="happy", Span(8,12)="cats"]
    """
    text = "unhappy cats"
    word_spans = [Span(0, 7), Span(8, 12)]
    word_layer = AnnotationLayer("words", word_spans, ["unhappy", "cats"])

    token_spans = [Span(0, 2), Span(2, 7), Span(8, 12)]
    token_layer = AnnotationLayer("tokens", token_spans, [0, 1, 2])

    stim = Stimulus(text=text, stimulus_id="test")
    stim.add_layer(word_layer)
    stim.add_layer(token_layer)
    return stim


class TestTokenAlignment:
    @pytest.mark.parametrize(
        "method,token_vals,expected",
        [
            ("sum", [1.5, 2.5, 3.0], [4.0, 3.0]),
            ("mean", [1.0, 3.0, 5.0], [2.0, 5.0]),
            ("first", [1.0, 3.0, 5.0], [1.0, 5.0]),
            ("last", [1.0, 3.0, 5.0], [3.0, 5.0]),
        ],
        ids=["sum", "mean", "first", "last"],
    )
    def test_aggregate(self, method, token_vals, expected):
        alignment = TokenAlignment("words", "tokens", [[0, 1], [2]])
        result = alignment.aggregate(np.array(token_vals), method=method)
        np.testing.assert_array_almost_equal(result, expected)

    def test_aggregate_empty_mapping(self):
        alignment = TokenAlignment("words", "tokens", [[], [0]])
        token_vals = np.array([5.0])
        result = alignment.aggregate(token_vals, method="sum")
        assert np.isnan(result[0])
        assert result[1] == 5.0

    def test_aggregate_nan_tokens(self):
        alignment = TokenAlignment("words", "tokens", [[0, 1]])
        token_vals = np.array([np.nan, np.nan])
        result = alignment.aggregate(token_vals, method="sum")
        assert np.isnan(result[0])

    def test_aggregate_invalid_method(self):
        alignment = TokenAlignment("words", "tokens", [[0]])
        with pytest.raises(ValueError, match="Unknown"):
            alignment.aggregate(np.array([1.0]), method="median")


class TestAlignBySpans:
    def test_subword_alignment(self):
        stim = make_stimulus_with_tokens()
        alignment = align_by_spans(stim, "words", "tokens")

        # "un"(mid=1) and "happy"(mid=4.5) both fall in word "unhappy"[0,7)
        # "cats"(mid=10) falls in word "cats"[8,12)
        assert alignment.mapping == [[0, 1], [2]]

    def test_single_token_per_word(self):
        text = "a b"
        stim = Stimulus(text=text, stimulus_id="test")
        stim.add_layer(AnnotationLayer("words", [Span(0, 1), Span(2, 3)], ["a", "b"]))
        stim.add_layer(AnnotationLayer("tokens", [Span(0, 1), Span(2, 3)], [0, 1]))

        alignment = align_by_spans(stim, "words", "tokens")
        assert alignment.mapping == [[0], [1]]


class TestAlignFromTokenizer:
    def test_with_mock_tokenizer(self):
        """Test with a mock tokenizer that returns offset_mapping."""
        text = "unhappy cats"
        stim = Stimulus(text=text, stimulus_id="test")
        stim.add_layer(
            AnnotationLayer("words", [Span(0, 7), Span(8, 12)], ["unhappy", "cats"])
        )

        class MockTokenizer:
            name_or_path = "mock_tokenizer"

            def __call__(
                self, text, return_offsets_mapping=False, add_special_tokens=True
            ):
                return {
                    "input_ids": [100, 200, 300],
                    "offset_mapping": [(0, 2), (2, 7), (8, 12)],
                }

        tokenizer = MockTokenizer()
        alignment = align_from_tokenizer(stim, tokenizer)

        assert alignment.source_layer == "words"
        assert alignment.target_layer == "mock_tokenizer"
        assert "mock_tokenizer" in stim.layers
        assert alignment.mapping == [[0, 1], [2]]

    def test_custom_layer_name(self):
        text = "hello"
        stim = Stimulus(text=text, stimulus_id="test")
        stim.add_layer(AnnotationLayer("words", [Span(0, 5)], ["hello"]))

        class MockTokenizer:
            name_or_path = "mock"

            def __call__(self, text, **kwargs):
                return {
                    "input_ids": [42],
                    "offset_mapping": [(0, 5)],
                }

        alignment = align_from_tokenizer(stim, MockTokenizer(), layer_name="custom")
        assert alignment.target_layer == "custom"
        assert "custom" in stim.layers


class _OffsetTokenizer:
    """Byte-level style offsets: a leading space stays inside the token."""

    name_or_path = "offsets"

    def __call__(self, text, return_offsets_mapping=False, add_special_tokens=True):
        ids, offsets = [], []
        for start, end in [(0, 2), (2, 7), (7, 8), (8, 12), (12, 12)]:
            ids.append(len(ids) + 10)
            offsets.append((start, end))
        return {"input_ids": ids, "offset_mapping": offsets}


class TestTokenizeSpans:
    def test_spans_follow_offsets_and_drop_empty_ones(self):
        from replica.feature.core.annotation.alignment import tokenize_spans

        ids, spans = tokenize_spans("unhappy cats", _OffsetTokenizer())
        assert ids == [10, 11, 12, 13]
        assert spans == [Span(0, 2), Span(2, 7), Span(7, 8), Span(8, 12)]
        # Consecutive spans tile the text, whitespace included.
        assert "".join("unhappy cats"[s.start : s.end] for s in spans) == "unhappy cats"

    def test_align_from_tokenizer_uses_the_same_spans(self):
        stim = Stimulus(text="unhappy cats", stimulus_id="t")
        stim.add_layer(
            AnnotationLayer("words", [Span(0, 7), Span(8, 12)], ["unhappy", "cats"])
        )
        alignment = align_from_tokenizer(stim, _OffsetTokenizer())
        # The standalone space token at [7, 8) has its midpoint in no word.
        assert alignment.mapping == [[0, 1], [3]]


class TestContentSpan:
    def test_trims_whitespace_and_reports_blank(self):
        from replica.feature.core.annotation.alignment import content_span

        assert content_span("a  b", Span(1, 4)) == Span(3, 4)
        assert content_span("a  b", Span(1, 3)) is None
        assert content_span("ab", Span(0, 2)) == Span(0, 2)


def _stim(text, words):
    stim = Stimulus(text=text, stimulus_id="s")
    spans, pos = [], 0
    for word in words:
        start = text.index(word, pos)
        spans.append(Span(start, start + len(word)))
        pos = start + len(word)
    stim.add_layer(AnnotationLayer("words", spans, list(words)))
    return stim


class TestAggregateNested:
    def test_nested_tokens_sum_in_order_and_first_takes_the_first(self):
        from replica.feature.core.annotation.alignment import aggregate_nested

        stim = _stim("The cats sat.", ["The", "cats", "sat."])
        spans = [Span(0, 3), Span(3, 6), Span(6, 8), Span(8, 12), Span(12, 13)]
        values = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
        np.testing.assert_array_equal(
            aggregate_nested(stim, "words", spans, values, "sum"), [1.0, 6.0, 24.0]
        )
        np.testing.assert_array_equal(
            aggregate_nested(stim, "words", spans, values, "first"), [1.0, 2.0, 8.0]
        )

    def test_crossing_token_raises_with_the_positions(self):
        from replica.feature.core.annotation.alignment import (
            TokenBoundaryError,
            aggregate_nested,
        )

        stim = _stim("abcd", ["ab", "cd"])
        spans = [Span(0, 1), Span(1, 3), Span(3, 4)]
        with pytest.raises(TokenBoundaryError) as error:
            aggregate_nested(stim, "words", spans, np.ones(3), "sum", ["a", "bc", "d"])
        message = str(error.value)
        assert message.startswith("s: token 1 'bc' at [1, 3) crosses a words boundary")
        assert "unit 0 'ab' at [0, 2)" in message
        assert "unit 1 'cd' at [2, 4)" in message

    def test_token_running_past_its_unit_into_a_gap_raises(self):
        from replica.feature.core.annotation.alignment import (
            TokenBoundaryError,
            aggregate_nested,
        )

        stim = _stim("ab, cd", ["ab", "cd"])
        spans = [Span(0, 3), Span(3, 6)]
        with pytest.raises(TokenBoundaryError, match="token 0 'ab,' at \\[0, 3\\)"):
            aggregate_nested(stim, "words", spans, np.ones(2), "sum")

    def test_non_blank_token_outside_every_unit_raises(self):
        from replica.feature.core.annotation.alignment import (
            TokenBoundaryError,
            aggregate_nested,
        )

        stim = _stim("ab - cd", ["ab", "cd"])
        spans = [Span(0, 2), Span(2, 4), Span(4, 7)]
        with pytest.raises(TokenBoundaryError, match="lies outside every words unit"):
            aggregate_nested(stim, "words", spans, np.ones(3), "sum")

    def test_blanks_between_units_are_dropped(self):
        from replica.feature.core.annotation.alignment import aggregate_nested

        stim = _stim("a  b", ["a", "b"])
        spans = [Span(0, 1), Span(1, 2), Span(2, 4)]
        values = np.array([1.0, 100.0, 2.0])
        np.testing.assert_array_equal(
            aggregate_nested(stim, "words", spans, values, "sum"), [1.0, 2.0]
        )

    def test_blanks_inside_a_unit_span_are_summed(self):
        from replica.feature.core.annotation.alignment import aggregate_nested

        # The loader's unit text carries the space, so the record for it
        # belongs to the unit like any other record inside the span.
        stim = _stim("ab cd ef", ["ab cd", "ef"])
        spans = [Span(0, 2), Span(2, 3), Span(3, 5), Span(5, 6), Span(6, 8)]
        values = np.array([1.0, 100.0, 2.0, 1000.0, 4.0])
        np.testing.assert_array_equal(
            aggregate_nested(stim, "words", spans, values, "sum"), [103.0, 4.0]
        )
        characters = [Span(i, i + 1) for i in range(len(stim.text))]
        per_character = np.arange(1, len(stim.text) + 1, dtype=float)
        np.testing.assert_array_equal(
            aggregate_nested(stim, "words", characters, per_character, "sum"),
            [1 + 2 + 3 + 4 + 5, 7 + 8],
        )

    def test_blank_crossing_a_unit_boundary_raises(self):
        from replica.feature.core.annotation.alignment import (
            TokenBoundaryError,
            aggregate_nested,
        )

        stim = _stim("ab  cd", ["ab ", "cd"])
        spans = [Span(0, 2), Span(2, 4), Span(4, 6)]
        with pytest.raises(
            TokenBoundaryError, match="token 1 '  ' at \\[2, 4\\) crosses"
        ):
            aggregate_nested(stim, "words", spans, np.ones(3), "sum")

    @pytest.mark.parametrize(
        "text,words,spans,match",
        [
            # A byte-level token carries the space before a word, so the
            # whitespace unit between the words is covered by no record.
            (
                "ab  cd",
                ["ab", " ", "cd"],
                [Span(0, 2), Span(2, 6)],
                "unit 1 ' '",
            ),
            ("ab cd", ["ab", "", "cd"], [Span(0, 2), Span(2, 5)], "unit 1 ''"),
            (
                "ab cd",
                ["ab", "cd"],
                [Span(0, 2)],
                "unit 1 'cd' at \\[3, 5\\) has no token",
            ),
        ],
    )
    def test_unit_without_a_token_raises(self, text, words, spans, match):
        from replica.feature.core.annotation.alignment import (
            TokenBoundaryError,
            aggregate_nested,
        )

        stim = Stimulus(text=text, stimulus_id="s")
        unit_spans, pos = [], 0
        for word in words:
            start = text.index(word, pos) if word else pos
            unit_spans.append(Span(start, start + len(word)))
            pos = start + len(word)
        stim.add_layer(AnnotationLayer("words", unit_spans, list(words)))
        with pytest.raises(TokenBoundaryError, match=match):
            aggregate_nested(stim, "words", spans, np.ones(len(spans)), "sum")

    def test_sequence_start_without_bos_is_nan(self):
        from replica.feature.core.annotation.alignment import aggregate_nested

        stim = _stim("ab cd", ["ab", "cd"])
        spans = [Span(0, 1), Span(1, 2), Span(2, 5)]
        values = np.array([np.nan, 1.0, 2.0])
        result = aggregate_nested(stim, "words", spans, values, "sum")
        assert np.isnan(result[0]) and result[1] == 2.0
        result = aggregate_nested(stim, "words", spans, values, "first")
        assert np.isnan(result[0]) and result[1] == 2.0

    def test_unknown_rule_and_unordered_units_are_rejected(self):
        from replica.feature.core.annotation.alignment import aggregate_nested

        stim = _stim("ab cd", ["ab", "cd"])
        with pytest.raises(ValueError, match="rule"):
            aggregate_nested(stim, "words", [Span(0, 2)], np.ones(1), "mean")
        stim.layers["words"].spans.reverse()
        with pytest.raises(ValueError, match="out of order"):
            aggregate_nested(stim, "words", [Span(0, 2)], np.ones(1), "sum")
