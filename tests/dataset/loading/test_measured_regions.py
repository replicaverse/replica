"""Measured areas retain their source response and exact predictor spans."""

import numpy as np
import pytest

from replica.core import AnnotationLayer, Span
from replica.dataset.eye_tracking.geco import GECO
from replica.dataset.eye_tracking.meco.waves import MECOL2W2, _l2_display_regions
from replica.dataset.loading.word_regions import preserve_interest_areas
from replica.feature.core.annotation.alignment import align_by_spans


def test_multiword_region_aligns_all_constituent_tokens_once():
    labels, ids, mapping = preserve_interest_areas(
        [(12, "5.\xa0Juni"), (13, "weltweit")]
    )
    ds = GECO()
    stimulus = ds._make_stimulus(labels, "region")
    stimulus.add_layer(
        AnnotationLayer(
            "tokens", [Span(0, 1), Span(1, 2), Span(3, 7), Span(8, 16)], [1, 2, 3, 4]
        )
    )
    alignment = align_by_spans(stimulus, "words", "tokens")
    assert stimulus.text == "5. Juni weltweit"
    assert ids == [12, 13] and mapping == {12: 0, 13: 1}
    assert alignment.mapping == [[0, 1, 2], [3]]
    np.testing.assert_equal(
        alignment.aggregate(np.array([1.0, 2.0, 3.0, 4.0])), [6.0, 4.0]
    )


@pytest.mark.parametrize("regions", [[(1, " ")], [(1, "one"), (1, "two")]])
def test_ambiguous_source_region_fails(regions):
    with pytest.raises(ValueError, match="interest-area ID"):
        preserve_interest_areas(regions)


def test_meco_pencil_line_break_has_one_spelling_and_two_measured_spans():
    ds = MECOL2W2()
    s = ds._make_stimulus(
        ["modern,", "wood-encased", "wood-encased", "pencil."],
        "pencil",
        {
            "source_word_numbers": [19, 20, 21, 22],
            "source_item_id": 11,
            "reader_group": "L2",
            "wave": 2,
            "site": "multiple",
        },
    )
    _l2_display_regions(s, 11)
    assert s.text == "modern, wood-encased pencil."
    assert s.unit_strings() == ["modern,", "wood-", "encased", "pencil."]
    assert s.layers["words"].spans[1].end == s.layers["words"].spans[2].start
    assert s.metadata["source_word_numbers"] == [19, 20, 21, 22]


def test_meco_does_not_repair_an_unrelated_repeated_label():
    ds = MECOL2W2()
    s = ds._make_stimulus(
        ["wood-encased", "wood-encased"],
        "other",
        {
            "source_word_numbers": [2, 3],
            "source_item_id": 11,
            "reader_group": "L2",
            "wave": 2,
            "site": "multiple",
        },
    )
    _l2_display_regions(s, 11)
    assert s.text == "wood-encased wood-encased"
