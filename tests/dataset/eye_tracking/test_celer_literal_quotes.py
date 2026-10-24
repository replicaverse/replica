"""Literal source punctuation must not swallow reading-time fields or rows."""

import numpy as np
import pytest

from replica.dataset import CELER
from tests.dataset.eye_tracking._celer import write_sent_ia


def test_literal_quotes_preserve_both_words_and_response_columns(tmp_path):
    words = ["She", "said", '"Wait', 'here."']
    text = " ".join(words)
    rows = [
        ["a", "reader", str(i), word, "1", *([str(100 * i)] * 4), text]
        for i, word in enumerate(words, 1)
    ]
    rows += [["b", "reader", "1", "'Hello.'", "0", *(["250"] * 4), "'Hello.'"]]
    write_sent_ia(tmp_path, rows)
    ds = CELER()
    ds.load(str(tmp_path))
    assert [s.text for s in ds.stimuli()] == [text, "'Hello.'"]
    assert ds.stimuli()[0].unit_strings("words") == words
    datums = ds.data(measure="total_time")
    np.testing.assert_array_equal(datums[0].values, [100, 200, 300, 400])
    np.testing.assert_array_equal(datums[1].values, [250])


def test_malformed_physical_row_is_rejected(tmp_path):
    write_sent_ia(tmp_path, [["a", "reader", "1", '"Wait', "1", "200"]])
    with pytest.raises(ValueError, match="TSV header"):
        CELER().load(str(tmp_path))


def test_sentence_quotes_outside_measured_areas_join_their_display_token(tmp_path):
    text = 'She said "Wait here."'
    words = ["She", "said", "Wait", "here."]
    write_sent_ia(
        tmp_path,
        [
            ["a", "reader", str(i), word, "1", *([str(100 * i)] * 4), text]
            for i, word in enumerate(words, 1)
        ],
    )
    dataset = CELER()
    dataset.load(str(tmp_path))
    [stimulus] = dataset.stimuli()
    assert stimulus.text == text
    assert stimulus.unit_strings("words") == ["She", "said", '"Wait', 'here."']
    assert [text[sp.start : sp.end] for sp in stimulus.layers["words"].spans] == [
        "She",
        "said",
        '"Wait',
        'here."',
    ]
    np.testing.assert_array_equal(
        dataset.data("total_time")[0].values, [100, 200, 300, 400]
    )


def test_wholly_quoted_word_keeps_its_quotes_as_one_unit(tmp_path):
    text = 'It went "up" on "the" news.'
    labels = ["It", "went", '"up"', "on", "the", "news."]
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
    assert stimulus.unit_strings("words") == [
        "It",
        "went",
        '"up"',
        "on",
        '"the"',
        "news.",
    ]
    starts = [sp.start for sp in stimulus.layers["words"].spans]
    ends = [sp.end for sp in stimulus.layers["words"].spans]
    covered = "".join(text[a:b] for a, b in zip(starts, ends, strict=True))
    assert covered == text.replace(" ", "")


def test_labels_sharing_one_display_token_stay_separate(tmp_path):
    text = "A well-known fact."
    labels = ["A", "well", "-known", "fact."]
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
    assert stimulus.unit_strings("words") == labels
