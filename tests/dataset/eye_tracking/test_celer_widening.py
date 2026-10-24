"""Widening CELER interest-area labels to their display tokens.

A widened span contains its label, stays inside the text, adds no
whitespace, and neither overlaps nor reorders against its neighbours; the
loader keeps one raw label per unit.
"""

import random

import numpy as np
import pytest

from replica.core import Span
from replica.dataset import CELER
from replica.dataset.eye_tracking.celer import (
    _build_text_and_spans,
    _widen_to_display_tokens,
)
from tests.dataset.eye_tracking._celer import write_sent_ia


def _random_case(rng):
    alphabet = "ab-'\".  "
    text = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 24)))
    spans = []
    cursor = 0
    while cursor < len(text) and rng.random() < 0.8:
        start = rng.randint(cursor, len(text))
        end = rng.randint(start, len(text))
        spans.append(Span(start, end))
        cursor = end
    return text, spans


def _check(text, spans, widened):
    assert len(widened) == len(spans)
    previous_end = 0
    for original, wide in zip(spans, widened, strict=True):
        assert 0 <= wide.start <= original.start
        assert original.end <= wide.end <= len(text)
        assert wide.start >= previous_end
        added = text[wide.start : original.start] + text[original.end : wide.end]
        assert not any(character.isspace() for character in added)
        previous_end = wide.end


def test_widening_invariants_on_random_inputs():
    rng = random.Random(7)
    for _ in range(3000):
        text, spans = _random_case(rng)
        _check(text, spans, _widen_to_display_tokens(text, spans))


@pytest.mark.parametrize(
    "text,spans,expected",
    [
        # Labels dropping quotes and a hyphen are widened to the token edge.
        ('"up"', [(1, 3)], [(0, 4)]),
        ("well-known", [(0, 4), (5, 10)], [(0, 5), (5, 10)]),
        ("a-b-c", [(0, 1), (2, 3), (4, 5)], [(0, 2), (2, 4), (4, 5)]),
        # Two labels inside one display token split it without a gap.
        ("ab cd", [(0, 1), (1, 2)], [(0, 1), (1, 2)]),
        # A label at the text edge and one after leading whitespace.
        ("  ab", [(3, 4)], [(2, 4)]),
        ("ab  ", [(0, 1)], [(0, 2)]),
        # A label that already covers its token is unchanged.
        ("ab cd", [(0, 2), (3, 5)], [(0, 2), (3, 5)]),
        ("", [], []),
    ],
)
def test_widening_examples(text, spans, expected):
    widened = _widen_to_display_tokens(text, [Span(*s) for s in spans])
    assert [(w.start, w.end) for w in widened] == expected


def test_build_text_and_spans_keeps_one_unit_per_label():
    text = 'It went "up" (well-known) on "the" news.'
    labels = ["It", "went", '"up"', "well", "-known", "on", "the", "news."]
    built, spans, words = _build_text_and_spans(text, labels)
    assert built == text
    assert len(spans) == len(words) == len(labels)
    assert [text[s.start : s.end] for s in spans] == words
    assert words == ["It", "went", '"up"', "(well", "-known)", "on", '"the"', "news."]
    for label, word in zip(labels, words, strict=True):
        assert label in word


def test_build_text_and_spans_falls_back_when_a_label_is_absent():
    labels = ["It", "went", "up."]
    built, spans, words = _build_text_and_spans("It went down.", labels)
    assert built == "It went up."
    assert words == labels
    assert [built[s.start : s.end] for s in spans] == labels


def test_loader_keeps_raw_labels_beside_widened_units(tmp_path):
    text = "The cat's (very) well-known nap."
    labels = ["The", "cat", "'s", "very", "well", "-known", "nap"]
    write_sent_ia(
        tmp_path,
        [
            ["a", "reader", str(i), word, "1", *([str(100 * i)] * 4), text]
            for i, word in enumerate(labels, 1)
        ],
    )
    dataset = CELER()
    dataset.load(str(tmp_path))
    [stimulus] = dataset.stimuli()
    spans = stimulus.layers["words"].spans
    units = stimulus.unit_strings("words")
    assert stimulus.text == text
    assert stimulus.metadata["interest_area_labels"] == labels
    assert len(units) == len(spans) == len(labels)
    assert units == ["The", "cat", "'s", "(very)", "well", "-known", "nap."]
    assert all(not any(c.isspace() for c in unit) for unit in units)
    for earlier, later in zip(spans, spans[1:], strict=False):
        assert earlier.end <= later.start
    assert spans[-1].end <= len(text)
    np.testing.assert_array_equal(
        dataset.data("total_time")[0].values, [100 * i for i in range(1, 8)]
    )
