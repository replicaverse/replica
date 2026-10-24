"""The Dundee loader over the Shain et al. (2024) release.

Synthetic fixation rows verify per-word aggregation.
"""

from collections import defaultdict

import numpy as np
import pytest

from replica.core import AnnotationLayer, Span
from replica.dataset import FixationCalculation
from replica.dataset.eye_tracking.dundee.shain2024 import Dundee
from tests.dataset._measures import measure_values as _values

COLUMNS = [
    "word",
    "subject",
    "docid",
    "sentid",
    "sentpos",
    "time",
    "fdurSP",
    "fdurFP",
    "fdurGP",
    "fdurTT",
    "inregression",
]


def _release(tmp_path, rows, words=None, source_events=None):
    previous, deltas = {}, {}
    for i in sorted(
        range(len(rows)), key=lambda i: (rows[i][1], rows[i][2], rows[i][5])
    ):
        row = rows[i]
        key = row[1], row[2]
        deltas[i] = row[4] - previous.get(key, 0)
        previous[key] = row[4]
    lines = [" ".join([*COLUMNS, "wdelta"])]
    lines += [
        " ".join(str(value) for value in [*row, deltas[i]])
        for i, row in enumerate(rows)
    ]
    (tmp_path / "dundee_X.csv").write_text("\n".join(lines) + "\n")
    items = tmp_path / "items"
    items.mkdir(exist_ok=True)
    if words is None:
        by_position = {row[4]: row[0] for row in rows}
        words = [by_position[i] for i in sorted(by_position)]
    docid = rows[0][2]
    source = [
        " ".join(
            [word.replace('"', "'"), str(docid), "1", "1", *(["0"] * 8), str(i), "0"]
        )
        for i, word in enumerate(words, 1)
    ]
    (items / f"tx{docid:02d}wrdp.dat").write_text("\n".join(source), encoding="cp437")
    (items / f"text{docid:02d}.itm").write_text(
        "+" + " ".join(words) + " >\n", encoding="cp437"
    )
    if source_events is None:
        source_events = [(row[1], row[4], row[5], row[6], 1, 1) for row in rows]
    by_subject = defaultdict(list)
    for subject, number, time, duration, screen, line in source_events:
        by_subject[subject].append((time, number, duration, screen, line))
    for subject, events in by_subject.items():
        lines = ["WORD TEXT LINE OLEN WLEN XPOS WNUM FDUR OBLP WDLP LAUN TXFR"]
        elapsed = 0
        for time, number, duration, screen, line in sorted(events):
            gap = round(time * 1000) - elapsed
            assert gap >= 0
            if gap:
                lines.append(f"*Off-screen -99 -99 0 0 0 -99 {gap} 0 0 0 0")
            lines.append(
                f"{words[number - 1]} {screen} {line} 0 0 0 {number} {duration} 0 0 0 0"
            )
            elapsed = round(time * 1000) + duration
        (tmp_path / f"{subject}{docid:02d}ma1p.dat").write_text(
            "\n".join(lines), encoding="cp437"
        )
    dataset = Dundee()
    dataset.load(str(tmp_path))
    return dataset


def test_a_words_value_is_taken_over_its_fixation_rows(tmp_path):
    # Word 1 is fixated three times; only the first row carries the word-level
    # values, and reading the rows in turn would leave the last row's NaN.
    dataset = _release(
        tmp_path,
        [
            ("The", "sa", 1, 1, 1, 0.000, 200, 500, 700, 900, 0),
            ("The", "sa", 1, 1, 1, 0.200, 120, "None", "None", "None", 0),
            ("The", "sa", 1, 1, 1, 0.320, 180, "None", "None", "None", 0),
            ("cat", "sa", 1, 1, 2, 0.500, 150, 150, 150, 150, 0),
        ],
    )
    assert _values(dataset, "first_run")[0] == 500
    assert _values(dataset, "first_pass")[0] == 1
    assert _values(dataset, "go_past")[0] == 700
    assert _values(dataset, "go_past_in_first_pass")[0] == 700
    assert _values(dataset, "total_time")[0] == 900


def test_first_fixation_is_the_earliest_fixation_not_the_last(tmp_path):
    dataset = _release(
        tmp_path,
        [
            ("The", "sa", 1, 1, 1, 0.000, 200, 500, 500, 500, 0),
            ("The", "sa", 1, 1, 1, 0.200, 120, "None", "None", "None", 0),
            ("cat", "sa", 1, 1, 2, 0.320, 150, 150, 150, 150, 0),
        ],
    )
    assert _values(dataset, "first_fixation")[0] == 200


def test_rows_out_of_time_order_do_not_change_the_result(tmp_path):
    dataset = _release(
        tmp_path,
        [
            ("The", "sa", 1, 1, 1, 0.200, 120, "None", "None", "None", 0),
            ("The", "sa", 1, 1, 1, 0.000, 200, 500, 500, 500, 0),
            ("cat", "sa", 1, 1, 2, 0.320, 150, 150, 150, 150, 0),
        ],
    )
    assert _values(dataset, "first_fixation")[0] == 200
    assert _values(dataset, "first_run")[0] == 500
    assert _values(dataset, "first_pass")[0] == 1


def test_the_word_none_is_a_word_not_a_missing_value(tmp_path):
    # The release writes a missing value as `None`, and text 14 contains the
    # word None ("the legend was born. None of which explains how").
    dataset = _release(
        tmp_path,
        [
            ("born.", "sa", 1, 1, 1, 0.000, 200, 200, 200, 200, 0),
            ("None", "sa", 1, 1, 2, 0.200, 150, 150, 150, 150, 0),
            ("of", "sa", 1, 1, 3, 0.350, 150, 150, 150, 150, 0),
        ],
    )
    assert dataset.stimuli()[0].unit_strings() == ["born.", "None", "of"]


def test_every_row_is_published_as_a_fixation(tmp_path):
    dataset = _release(
        tmp_path,
        [
            ("The", "sa", 1, 1, 1, 0.000, 200, 500, 500, 500, 0),
            ("The", "sa", 1, 1, 1, 0.200, 120, "None", "None", "None", 1),
            ("cat", "sa", 1, 1, 2, 0.320, 150, 150, 150, 150, 0),
        ],
    )
    fixations = dataset.fixations()
    assert [f.duration for f in fixations] == [200.0, 120.0, 150.0]
    assert [f.unit_index for f in fixations] == [0, 0, 1]
    # `time` is trial-relative seconds in the release.
    assert [f.start_time for f in fixations] == [0.0, 200.0, 320.0]
    assert fixations[1].metadata["in_regression"] is True
    assert np.isnan(fixations[0].x)


def test_fixations_can_be_filtered(tmp_path):
    dataset = _release(
        tmp_path,
        [
            ("The", "sa", 1, 1, 1, 0.000, 200, 200, 200, 200, 0),
            ("The", "sb", 1, 1, 1, 0.000, 300, 300, 300, 300, 0),
        ],
    )
    assert len(dataset.fixations()) == 2
    assert [f.duration for f in dataset.fixations(participant="sb")] == [300.0]


def test_interruption_intervals_roundtrip_and_rebuild_on_custom_units(tmp_path):
    dataset = _release(
        tmp_path,
        [
            ("The", "sa", 1, 1, 1, 0.000, 100, 120, 120, 120, 0),
            ("cat", "sa", 1, 1, 2, 0.120, 150, 150, 150, 150, 0),
        ],
    )
    events = dataset.fixations()
    assert [e.duration for e in events] == [100, 150]
    assert events[0].metadata["following_interruptions"] == [
        {"duration": 20.0, "kind": "*Off-screen", "closes_windows": False}
    ]
    stimulus = dataset.stimuli()[0]
    stimulus.add_layer(AnnotationLayer("phrase", [Span(0, 7)], ["The cat"]))
    for measure in ("first_run", "go_past", "total_time"):
        datum = dataset.data(measure)[0]
        np.testing.assert_equal(dataset.reaggregate(datum, "words").values, [120, 150])
        np.testing.assert_equal(dataset.reaggregate(datum, "phrase").values, [270])
        standard = dataset.reaggregate(
            datum, "phrase", calculation=FixationCalculation()
        )
        np.testing.assert_equal(standard.values, [250])
    first = dataset.reaggregate(dataset.data("first_fixation")[0], "phrase")
    np.testing.assert_equal(first.values, [100])


def test_a_release_without_docid_is_rejected(tmp_path):
    (tmp_path / "dundee_X.csv").write_text("word subject sentid sentpos\nThe sa 1 1\n")
    with pytest.raises(ValueError, match="docid"):
        Dundee().load(str(tmp_path))


def test_complete_context_keeps_a_word_nobody_fixated(tmp_path):
    dataset = _release(
        tmp_path,
        [
            ("The", "sa", 1, 1, 1, 0.0, 100, 100, 100, 100, 0),
            ("cat", "sa", 1, 1, 3, 0.1, 200, 200, 200, 200, 0),
        ],
        words=["The", "small", "cat"],
    )
    assert dataset.stimuli()[0].unit_strings() == ["The", "small", "cat"]
    np.testing.assert_equal(_values(dataset, "first_run"), [100, np.nan, 200])
    np.testing.assert_equal(_values(dataset, "first_pass"), [1, 0, 1])
    dataset.drop_skips = False
    np.testing.assert_equal(_values(dataset, "first_run"), [100, 0, 200])
    np.testing.assert_equal(_values(dataset, "first_pass"), [1, 0, 1])
    assert [f.unit_index for f in dataset.fixations()] == [0, 2]


def test_complete_context_is_required_and_labels_are_validated(tmp_path):
    rows = [("cat", "sa", 1, 1, 1, 0.0, 100, 100, 100, 100, 0)]
    with pytest.raises(ValueError, match="does not match"):
        _release(tmp_path, rows, words=["dog"])
    (tmp_path / "items/tx01wrdp.dat").unlink()
    with pytest.raises(FileNotFoundError, match="complete stimulus"):
        Dundee().load(str(tmp_path))


def test_original_display_quotes_are_preserved(tmp_path):
    dataset = _release(
        tmp_path,
        [("'Hello'", "sa", 1, 0, 1, 0, 100, 100, 100, 100, 0)],
        words=['"Hello"'],
    )
    assert dataset.stimuli()[0].text == '"Hello"'


def test_merged_quote_keeps_its_entire_measured_region(tmp_path):
    words = [f"w{i}" for i in range(1, 1172)]
    words[1168:1170] = ["years!'", '"']
    rows = [
        ("years!'''", "sa", 18, 0, 1169, 0, 100, 250, 250, 250, 0),
        ("years!'''", "sa", 18, 0, 1169, 0.1, 150, "NA", "NA", "NA", 0),
        ("w1171", "sa", 18, 0, 1170, 0.25, 200, 200, 200, 200, 0),
    ]
    ds = _release(
        tmp_path,
        rows,
        words,
        [
            ("sa", 1169, 0, 100, 1, 1),
            ("sa", 1170, 0.1, 150, 1, 1),
            ("sa", 1171, 0.25, 200, 1, 1),
        ],
    )
    stimulus = ds.stimuli()[0]
    assert stimulus.text == " ".join(words)
    assert stimulus.unit_strings()[1168] == "years!' \""
    assert stimulus.metadata["source_word_groups"][1168] == [1169, 1170]
    assert len(stimulus.layers["source_words"]) == 1171
    assert len(stimulus.layers["words"]) == 1170
    assert _values(ds, "first_run")[1168] == 250
    assert _values(ds, "first_pass")[1168] == 1
    assert [f.unit_index for f in ds.fixations()] == [1168, 1168, 1169]
    assert (
        ds.fixations()[1].char_offset
        == stimulus.layers["source_words"].spans[1169].start
    )
    # Predictor contributions from both source objects belong to this region.
    from replica.feature.core.annotation.alignment import align_by_spans

    alignment = align_by_spans(stimulus, "words", "source_words")
    contributions = np.zeros(1171)
    contributions[1168:1170] = [2.0, 3.0]
    assert alignment.aggregate(contributions)[1168] == 5.0


def test_stale_word_id_cannot_assign_an_event_to_an_earlier_screen(tmp_path):
    rows = [
        ("this", "sa", 1, 0, 1, 0, 100, 100, 100, 300, 0),
        ("this", "sa", 1, 0, 1, 0.1, 200, "NA", "NA", "NA", 1),
        ("next", "sa", 1, 0, 3, 0.3, 150, 150, 150, 150, 0),
    ]
    ds = _release(
        tmp_path,
        rows,
        ["this", "case", "next"],
        [
            ("sa", 1, 0, 100, 1, 1),
            ("sa", 2, 0.1, 200, 2, 1),
            ("sa", 3, 0.3, 150, 1, 1),
        ],
    )
    assert [f.unit_index for f in ds.fixations()] == [0, None, 2]
    assert ds.fixations()[1].metadata["source_word_number"] == 2
    assert np.isnan(_values(ds, "total_time")[0])
    assert _values(ds, "first_fixation")[0] == 100
    assert _values(ds, "first_run")[0] == 100
    assert _values(ds, "first_pass")[0] == 1
    total = ds.data("total_time")[0]
    assert total.metadata["unavailable_source_values"] == [
        {
            "unit_index": 0,
            "released_value": 300.0,
            "reason": "total_includes_unaligned_event",
        }
    ]
    assert np.isnan(_values(ds, "wdelta")[2])


def test_go_past_containing_an_unaligned_event_is_unavailable(tmp_path):
    rows = [
        ("end", "sa", 1, 0, 2, 0, 100, 100, 300, 100, 0),
        ("start", "sa", 1, 0, 1, 0.1, 200, "NA", "NA", 200, 1),
    ]
    ds = _release(
        tmp_path,
        rows,
        ["start", "end"],
        [
            ("sa", 2, 0, 100, 1, 1),
            ("sa", 1, 0.1, 200, 2, 1),
        ],
    )
    assert np.isnan(_values(ds, "go_past")[1])
    assert _values(ds, "first_run")[1] == 100
    assert _values(ds, "first_pass")[1] == 1
    ds.drop_skips = False
    assert np.isnan(_values(ds, "first_fixation")[0])


@pytest.mark.parametrize(
    "change", ["missing_event", "duration", "display", "duplicate_word_id"]
)
def test_independent_source_disagreements_fail_loudly(tmp_path, change):
    _release(tmp_path, [("cat", "sa", 1, 0, 1, 0, 100, 100, 100, 100, 0)])
    if change == "missing_event":
        (tmp_path / "sa01ma1p.dat").write_text("")
    elif change == "duration":
        path = tmp_path / "sa01ma1p.dat"
        path.write_text(path.read_text().replace("100", "101"))
    elif change == "display":
        (tmp_path / "items/text01.itm").write_text("+dog >\n")
    else:
        path = tmp_path / "items/tx01wrdp.dat"
        path.write_text(path.read_text() + "\n" + path.read_text())
    with pytest.raises(ValueError):
        Dundee().load(str(tmp_path))


def test_repeated_labels_cannot_hide_conflicting_original_positions(tmp_path):
    rows = [
        ("the", "sa", 1, 0, 1, 0, 100, 100, 100, 200, 0),
        ("the", "sa", 1, 0, 1, 0.1, 100, "NA", "NA", "NA", 0),
    ]
    with pytest.raises(ValueError, match="conflicting source word positions"):
        _release(
            tmp_path,
            rows,
            ["the", "the"],
            [
                ("sa", 1, 0, 100, 1, 1),
                ("sa", 2, 0.1, 100, 1, 1),
            ],
        )


def test_unknown_nonword_event_time_is_retained_for_alignment(tmp_path):
    _release(tmp_path, [("cat", "sa", 1, 0, 1, 0.053, 100, 100, 100, 100, 0)])
    path = tmp_path / "sa01ma1p.dat"
    contents = path.read_text().replace(
        "*Off-screen -99 -99 0 0 0 -99 53 0 0 0 0", "\x00\x00 1 0 0 0 0 0 53 0 0 -99 0"
    )
    path.write_text(contents)
    ds = Dundee()
    ds.load(str(tmp_path))
    assert ds.fixations()[0].start_time == 53


def test_first_event_regression_settles_the_first_pass_flag(tmp_path):
    dataset = _release(
        tmp_path,
        [
            ("one", "sa", 1, 1, 1, 0.0, 100, 100, 100, 100, 0),
            ("three", "sa", 1, 1, 3, 0.1, 100, 100, 100, 100, 0),
            ("two", "sa", 1, 1, 2, 0.2, 80, "NA", "NA", 80, 1),
        ],
        words=["one", "two", "three"],
    )
    np.testing.assert_equal(_values(dataset, "first_pass"), [1, 0, 1])
    assert np.isnan(_values(dataset, "first_run")[1])
    dataset.drop_skips = False
    np.testing.assert_equal(
        dataset.data("first_run_in_first_pass")[0].values, [100, 0, 100]
    )


def test_malformed_original_event_is_rejected(tmp_path):
    _release(tmp_path, [("one", "sa", 1, 1, 1, 0.0, 100, 100, 100, 100, 0)])
    path = tmp_path / "sa01ma1p.dat"
    path.write_text(path.read_text() + "\ntruncated event\n")
    with pytest.raises(ValueError, match="Malformed Dundee event"):
        Dundee().load(str(tmp_path))


def test_uncertain_forward_landing_does_not_establish_a_skip(tmp_path):
    dataset = _release(
        tmp_path,
        [
            ("one", "sa", 1, 1, 1, 0.0, 100, 100, 100, 100, 0),
            ("four", "sa", 1, 1, 4, 0.1, 100, 100, 100, 100, 0),
            ("two", "sa", 1, 1, 2, 0.2, 100, "NA", "NA", 100, 1),
            ("five", "sa", 1, 1, 5, 0.3, 100, 100, 100, 100, 0),
        ],
        words=["one", "two", "three", "four", "five"],
        source_events=[
            ("sa", 1, 0.0, 100, 1, 1),
            ("sa", 4, 0.1, 100, 2, 1),
            ("sa", 2, 0.2, 100, 1, 1),
            ("sa", 5, 0.3, 100, 1, 1),
        ],
    )
    dataset.drop_skips = False
    np.testing.assert_equal(
        _values(dataset, "first_pass"), [1, np.nan, np.nan, np.nan, 1]
    )
    assert np.isnan(dataset.data("first_run_in_first_pass")[0].values[1])
    provenance = dataset.data("first_pass")[0].metadata["uncertain_first_pass_values"]
    assert {"unit_index": 1, "released_value": 0.0} in provenance
    assert _values(dataset, "total_time")[1] == 100
