"""Tests for core: Span, AnnotationLayer, Stimulus, Datum."""

import numpy as np
import pytest

from replica.core import AnnotationLayer, Datum, Span, Stimulus
from tests._stimuli import word_stimulus

# --- Span ---


class TestSpan:
    def test_basic(self):
        s = Span(0, 5)
        assert s.start == 0
        assert s.end == 5
        assert len(s) == 5
        assert s.midpoint == 2.5

    def test_extract(self):
        text = "hello world"
        assert Span(0, 5).extract(text) == "hello"
        assert Span(6, 11).extract(text) == "world"

    def test_empty_span(self):
        s = Span(3, 3)
        assert len(s) == 0
        assert s.extract("hello") == ""

    def test_contains(self):
        outer = Span(0, 10)
        inner = Span(2, 5)
        assert outer.contains(inner)
        assert not inner.contains(outer)
        assert outer.contains(outer)

    def test_overlaps(self):
        a = Span(0, 5)
        b = Span(3, 8)
        c = Span(5, 10)
        assert a.overlaps(b)
        assert b.overlaps(a)
        assert not a.overlaps(c)  # [0,5) and [5,10) don't overlap
        assert b.overlaps(c)

    def test_frozen(self):
        s = Span(0, 5)
        with pytest.raises(AttributeError):
            s.start = 1

    def test_invalid_negative_start(self):
        with pytest.raises(ValueError, match="non-negative"):
            Span(-1, 5)

    def test_invalid_end_before_start(self):
        with pytest.raises(ValueError, match="end must be >= start"):
            Span(5, 3)


# --- AnnotationLayer ---


class TestAnnotationLayer:
    def test_basic(self):
        spans = [Span(0, 3), Span(4, 7)]
        values = ["The", "cat"]
        layer = AnnotationLayer("words", spans, values)
        assert layer.name == "words"
        assert len(layer) == 2
        assert layer[0] == (Span(0, 3), "The")

    def test_extract_strings(self):
        text = "The cat sat"
        spans = [Span(0, 3), Span(4, 7), Span(8, 11)]
        layer = AnnotationLayer("words", spans, ["The", "cat", "sat"])
        assert layer.extract_strings(text) == ["The", "cat", "sat"]

    def test_length_mismatch(self):
        with pytest.raises(ValueError, match="same length"):
            AnnotationLayer("bad", [Span(0, 1)], [1, 2])


# --- Stimulus ---


class TestStimulus:
    def test_units_property(self):
        stim = word_stimulus("The cat sat on the mat", "test_1")
        assert len(stim.layers["words"]) == 6
        assert stim.unit_strings("words") == ["The", "cat", "sat", "on", "the", "mat"]

    def test_no_units_layer(self):
        stim = Stimulus(text="hello")
        with pytest.raises(KeyError, match="words"):
            stim.layers["words"]

    def test_add_layer(self):
        stim = word_stimulus("The cat sat on the mat", "test_1")
        token_layer = AnnotationLayer("tokens", [Span(0, 2), Span(2, 3)], [1, 2])
        stim.add_layer(token_layer)
        assert "tokens" in stim.layers
        assert len(stim.layers["tokens"]) == 2


# --- Datum ---


class TestDatum:
    def test_basic(self):
        stim = word_stimulus("The cat sat", "test_1")
        values = np.array([300.0, 250.0, 280.0])
        datum = Datum(stim, "spr_rt", values, participant_id="p1")
        assert datum.measure_name == "spr_rt"
        assert datum.participant_id == "p1"
        np.testing.assert_array_equal(datum.values, values)
        unmasked = Datum(stim, "spr_rt", np.ma.array(values, mask=False))
        assert unmasked.values.dtype == values.dtype
        assert np.shares_memory(unmasked.values, values)
        assert datum.values is values

    @pytest.mark.parametrize(
        "values",
        [
            np.array([np.nan, 250.0, 280.0]),
            np.ma.array([999.0, 250.0, 280.0], mask=[True, False, False]),
            np.ma.array([999, 250, 280], mask=[True, False, False]),
            np.ma.array([True, False, True], mask=[True, False, False]),
        ],
        ids=["nan", "masked-float", "masked-int", "masked-bool"],
    )
    def test_nan_values(self, values):
        stim = word_stimulus("The cat sat", "test_1")
        datum = Datum(stim, "spr_rt", values)
        assert np.isnan(datum.values[0])
        np.testing.assert_array_equal(datum.values[1:], np.asarray(values)[1:])
        assert type(datum.values) is np.ndarray

    def test_length_mismatch(self):
        stim = word_stimulus("The cat sat", "test_1")
        with pytest.raises(ValueError, match="values length"):
            Datum(stim, "spr_rt", np.array([1.0, 2.0]))
