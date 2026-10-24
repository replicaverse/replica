"""Token coverage of text with decomposed accents."""

import pytest

from replica.compute.units import verify_token_rows
from replica.feature import FeatureModelOutput

from .._stimuli import word_stimulus


def _rows(spans):
    return [
        FeatureModelOutput(
            values={"log_p_observed": -1.0}, metadata={"token_index": i, "span": span}
        )
        for i, span in enumerate(spans)
    ]


def test_a_combining_mark_is_covered_with_its_base_letter():
    # "dér" as d + e + U+0301 + r: the tokenizer normalises to "dér" and its
    # token span ends at the base letter, leaving the mark uncovered.
    stimulus = word_stimulus("dér her", "s")
    verify_token_rows(stimulus, _rows([(0, 2), (3, 4), (4, 8)]))
    # A mark whose base letter is uncovered is still a gap.
    with pytest.raises(ValueError, match="covered by no token"):
        verify_token_rows(stimulus, _rows([(0, 1), (3, 4), (4, 8)]))
    # A mark at the start of the text has no base letter to attach to.
    with pytest.raises(ValueError, match="covered by no token"):
        verify_token_rows(word_stimulus("́ab", "s"), _rows([(1, 3)]))
