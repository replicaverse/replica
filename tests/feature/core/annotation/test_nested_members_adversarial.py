"""Boundary cases of the guarded token-to-unit aggregation.

Every record either lies inside one unit or raises; nothing is split,
dropped or interpolated except a blank record outside every unit. The cases
here are the token shapes a byte-level tokenizer produces on real text:
pieces of one character sharing a span, a leading space carried inside the
first piece, standalone blanks, and tokens that abut or cross a unit edge.
"""

import numpy as np
import pytest

from replica.core import AnnotationLayer, Span, Stimulus
from replica.feature import FeatureModelOutput
from replica.feature.core.annotation.alignment import (
    TokenBoundaryError,
    aggregate_nested,
    nested_members,
)
from replica.feature.core.outputs.information import LN2
from replica.feature.core.outputs.units import unit_for
from tests._stimuli import word_stimulus


def _units(text, spans):
    stimulus = Stimulus(text=text, stimulus_id="u")
    stimulus.add_layer(
        AnnotationLayer(
            "words", [Span(*s) for s in spans], [text[a:b] for a, b in spans]
        )
    )
    return stimulus


def _rows(spans, values=None):
    values = (
        values if values is not None else [-float(i + 1) for i in range(len(spans))]
    )
    return [
        FeatureModelOutput(
            values={"log_p_observed": value},
            metadata={"model_name": "m", "token_index": i, "span": span, "token": "t"},
        )
        for i, (span, value) in enumerate(zip(spans, values, strict=True))
    ]


def _members(stimulus, spans):
    return nested_members(stimulus, "words", [Span(*s) for s in spans])


def test_unit_without_a_token_raises():
    stimulus = word_stimulus("ab cd")
    with pytest.raises(
        TokenBoundaryError, match="unit 1 'cd' at \\[3, 5\\) has no token"
    ):
        _members(stimulus, [(0, 2), (2, 3)])


def test_empty_unit_raises_instead_of_receiving_a_neighbour():
    stimulus = _units("abcd", [(0, 2), (2, 2), (2, 4)])
    with pytest.raises(
        TokenBoundaryError, match="unit 1 '' at \\[2, 2\\) has no token"
    ):
        _members(stimulus, [(0, 2), (2, 4)])


def test_whitespace_only_unit_takes_only_an_exactly_matching_blank():
    stimulus = _units("ab cd", [(0, 2), (2, 3), (3, 5)])
    # The space is carried inside the next word's token, so the blank unit
    # has no record of its own.
    with pytest.raises(
        TokenBoundaryError, match="unit 1 ' ' at \\[2, 3\\) has no token"
    ):
        _members(stimulus, [(0, 2), (2, 5)])
    # A standalone blank token of the same extent belongs to it.
    assert _members(stimulus, [(0, 2), (2, 3), (3, 5)]) == [[0], [1], [2]]


def test_byte_pieces_sharing_a_span_all_join_the_unit_and_sum():
    stimulus = word_stimulus("한국 어")
    spans = [(0, 1), (0, 1), (1, 2), (1, 2), (1, 2), (2, 4), (3, 4), (3, 4)]
    assert _members(stimulus, spans) == [[0, 1, 2, 3, 4], [5, 6, 7]]
    values = np.arange(1, 9, dtype=float)
    summed = aggregate_nested(
        stimulus, "words", [Span(*s) for s in spans], values, "sum"
    )
    np.testing.assert_array_equal(summed, [15.0, 21.0])
    first = aggregate_nested(
        stimulus, "words", [Span(*s) for s in spans], values, "first"
    )
    np.testing.assert_array_equal(first, [1.0, 6.0])


def test_leading_space_inside_the_first_byte_piece_lands_in_the_word():
    # gpt2 gives the first byte of a space-prefixed character the span with
    # the space and the remaining bytes the character alone.
    stimulus = word_stimulus("αβγ δ")
    spans = [(0, 1), (1, 2), (2, 3), (3, 5), (4, 5)]
    assert _members(stimulus, spans) == [[0, 1, 2], [3, 4]]


def test_unit_at_the_end_of_the_text_and_abutting_units():
    stimulus = _units("abcd", [(0, 2), (2, 4)])
    assert _members(stimulus, [(0, 2), (2, 4)]) == [[0], [1]]
    assert _members(stimulus, [(0, 1), (1, 2), (2, 3), (3, 4)]) == [[0, 1], [2, 3]]
    with pytest.raises(
        TokenBoundaryError, match="token 0 'abcd' at \\[0, 4\\) crosses"
    ):
        _members(stimulus, [(0, 4)])
    with pytest.raises(TokenBoundaryError, match="token 1 'bc' at \\[1, 3\\) crosses"):
        _members(stimulus, [(0, 1), (1, 3), (3, 4)])


def test_leading_whitespace_blank_is_dropped_but_content_is_not():
    stimulus = word_stimulus("  ab cd")
    assert _members(stimulus, [(0, 1), (1, 4), (4, 7)]) == [[1], [2]]
    stimulus = _units("x ab", [(2, 4)])
    with pytest.raises(
        TokenBoundaryError, match="token 0 'x' at \\[0, 1\\) lies outside"
    ):
        _members(stimulus, [(0, 1), (1, 4)])


def test_blank_crossing_a_unit_edge_raises():
    stimulus = _units("a  b", [(0, 2), (2, 4)])
    with pytest.raises(TokenBoundaryError, match="token 1 '  ' at \\[1, 3\\) crosses"):
        _members(stimulus, [(0, 1), (1, 3), (3, 4)])


def test_unit_containing_whitespace_sums_its_tokens():
    stimulus = _units("of the cat", [(0, 6), (7, 10)])
    rows = _rows([(0, 2), (2, 6), (6, 10)], [-1.0, -2.0, -4.0])
    from replica.compute.units import derive_units

    derived = derive_units(
        stimulus,
        rows,
        unit="tokens",
        layer="words",
        key="log_p_observed",
        rule="sum",
    )
    assert [d.values["log_p_observed"] for d in derived] == [-3.0, -4.0]
    assert [d.values["surprisal"] for d in derived] == [3.0 / LN2, 4.0 / LN2]
    assert unit_for("log_p_observed") == "nats" and unit_for("surprisal") == "bits"


@pytest.mark.parametrize(
    "values",
    [
        np.array([np.nan, -1.0, -2.0]),
        np.ma.array([999.0, -1.0, -2.0], mask=[True, False, False]),
    ],
    ids=["nan", "masked"],
)
def test_missing_first_position_reaches_only_its_unit(values):
    stimulus = word_stimulus("ab cd")
    spans = [(0, 1), (1, 2), (2, 5)]
    summed = aggregate_nested(
        stimulus, "words", [Span(*s) for s in spans], values, "sum"
    )
    assert np.isnan(summed[0]) and summed[1] == -2.0
    first = aggregate_nested(
        stimulus, "words", [Span(*s) for s in spans], values, "first"
    )
    assert np.isnan(first[0]) and first[1] == -2.0


def test_zero_width_space_is_content_not_blank():
    text = "x\u200by"
    stimulus = _units(text, [(0, 1), (2, 3)])
    with pytest.raises(TokenBoundaryError, match="lies outside every words unit"):
        _members(stimulus, [(0, 1), (1, 2), (2, 3)])
    assert _members(_units(text, [(0, 3)]), [(0, 1), (1, 2), (2, 3)]) == [[0, 1, 2]]


def test_units_out_of_order_or_overlapping_raise():
    stimulus = _units("ab cd", [(3, 5), (0, 2)])
    with pytest.raises(ValueError, match="overlap or are out of order"):
        _members(stimulus, [(0, 2), (2, 5)])
    stimulus = _units("abcd", [(0, 3), (2, 4)])
    with pytest.raises(ValueError, match="overlap or are out of order"):
        _members(stimulus, [(0, 2), (2, 4)])


def test_aggregate_nested_refuses_other_rules_and_ragged_inputs():
    stimulus = word_stimulus("ab cd")
    spans = [Span(0, 2), Span(2, 5)]
    with pytest.raises(ValueError, match="unknown aggregation rule"):
        aggregate_nested(stimulus, "words", spans, [1.0, 2.0], "mean")
    with pytest.raises(ValueError, match="2 token spans for 3 values"):
        aggregate_nested(stimulus, "words", spans, [1.0, 2.0, 3.0], "sum")
    with pytest.raises(ValueError, match="real-valued"):
        aggregate_nested(stimulus, "words", spans, np.array([1 + 2j, 3 + 4j]), "sum")


def test_no_tokens_at_all_means_every_unit_is_missing():
    with pytest.raises(TokenBoundaryError, match="unit 0 'ab'"):
        _members(word_stimulus("ab cd"), [])
    assert (
        _members(
            Stimulus(
                text="",
                stimulus_id="e",
                layers={"words": AnnotationLayer("words", [], [])},
            ),
            [],
        )
        == []
    )


MULTIBYTE = [
    "αβγ δ",
    "한국어 문장",
    "I 😀 you",
    "“quoted” word",
    "ﬁne print",
    " leading space",
    "abut,me now",
    "tabs\tand  gaps",
]


@pytest.mark.needs_model
@pytest.mark.parametrize("bos", [False, True])
def test_tiny_gpt2_multibyte_words_derive_exactly(bos):
    from replica.compute.units import TOKEN_UNIT, derive_units, verify_token_rows
    from replica.feature import HuggingFaceLM

    lm = HuggingFaceLM(
        "sshleifer/tiny-gpt2",
        add_bos=bos,
        provide_word_dist=False,
        token_level_entropy=True,
        token_renyi_alphas=[0.5],
    )
    rules = {
        "log_p_observed": "sum",
        "token_entropy": "first",
        "token_renyi_entropy_alpha0.5": "first",
    }
    for text in MULTIBYTE:
        stimulus = word_stimulus(text)
        rows = lm.score_tokens(stimulus)
        verify_token_rows(stimulus, rows)
        # Byte pieces of one character share a span; none was dropped.
        assert len(rows) >= len(stimulus.layers["words"])
        expected = lm(stimulus)
        for key, rule in rules.items():
            derived = derive_units(
                stimulus, rows, unit=TOKEN_UNIT, layer="words", key=key, rule=rule
            )
            np.testing.assert_array_equal(
                [d.values[key] for d in derived], [e.values[key] for e in expected]
            )
            values = np.array([d.values[key] for d in derived])
            assert np.isnan(values[0]) == (not bos)
            assert np.isfinite(values[1:]).all()
