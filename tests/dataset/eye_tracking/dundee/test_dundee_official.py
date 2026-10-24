"""Dundee V.1 loader and the ``go_past_duration`` helper.

Builds a miniature release on disk, so the parse, the alignment of events to
words, the four derived durations and the first-pass flag are exercised
without the corpus.
"""

import numpy as np
import pytest

from replica.dataset.eye_tracking.dundee.official import DundeeOfficial
from replica.dataset.loading.fixation_utils import go_past_duration
from tests.dataset._measures import measure_values as _values

# word, text, screen, line, position, serial, letter position, object length,
# word length, punctuation, before, after, index in text, frequency
WORDS = [
    ("The", 1),
    ("cat", 2),
    ("sat", 3),
    ("down", 4),
]


def _write_release(root, events, marker="*Off-screen"):
    items = root / "english" / "items"
    items.mkdir(parents=True)
    lines = [
        f"{word:<28}  1  1 1   {i}   {i}   0   3   3   0   0   0   {i}   10"
        for word, i in WORDS
    ]
    (items / "tx01wrdp.dat").write_text("\n".join(lines) + "\n")
    (items / "text01.itm").write_text("+The cat sat down>\n")
    header = (
        "WORD                  TEXT  LINE  OLEN  WLEN  XPOS  WNUM  FDUR  "
        "OBLP  WDLP  LAUN    TXFR"
    )
    rows = [header]
    for word_number, duration in events:
        word = WORDS[word_number - 1][0] if word_number > 0 else marker
        rows.append(
            f"{word:<24} 1     1     3     3     1  {word_number:>4}  "
            f"{duration:>4}     1     1     0     10"
        )
    (root / "english" / "sa01ma1p.dat").write_text("\n".join(rows) + "\n")


def _set_fields(path, rows, index, value="2"):
    lines = path.read_text().splitlines()
    for row in rows:
        fields = lines[row].split()
        fields[index] = value
        lines[row] = " ".join(fields)
    path.write_text("\n".join(lines) + "\n")


def _reload(root):
    dataset = DundeeOfficial()
    dataset.load(str(root))
    return dataset


def _conflicting_forward_release(root):
    """The disputed event could have progressed to word 4."""
    _write_release(root, [(1, 200), (4, 150), (3, 100), (4, 110)])
    _set_fields(root / "english" / "items" / "tx01wrdp.dat", (2, 3), 3)
    _set_fields(root / "english" / "sa01ma1p.dat", (3, 4), 2)
    return _reload(root)


def _uncertain_rightward_exit_release(root):
    """An uncertain exit to the right precedes a return to word 1."""
    _write_release(root, [(1, 200), (2, 150), (1, 80), (3, 100)])
    _set_fields(root / "english" / "items" / "tx01wrdp.dat", (2, 3), 3)
    _set_fields(root / "english" / "sa01ma1p.dat", (2, 4), 2)
    return _reload(root)


def _skip_before_conflict_release(root):
    """A known first-pass skip of word 1 precedes a later line conflict."""
    _write_release(root, [(3, 100), (1, 200), (4, 150)])
    _set_fields(root / "english" / "items" / "tx01wrdp.dat", (3,), 3)
    return _reload(root)


def _load(tmp_path, events, marker="*Off-screen"):
    _write_release(tmp_path, events, marker)
    dataset = DundeeOfficial()
    dataset.load(str(tmp_path))
    return dataset


def test_reads_one_stimulus_per_text_with_words_in_order(tmp_path):
    dataset = _load(tmp_path, [(1, 200), (2, 180)])
    stimuli = dataset.stimuli()
    assert len(stimuli) == 1
    assert stimuli[0].unit_strings() == ["The", "cat", "sat", "down"]
    assert stimuli[0].stimulus_id == "dundee_text01"
    assert dataset.participants == ["sa"]


def test_measures_follow_the_fixation_stream(tmp_path):
    # Read 1, 2, regress to 1, return to 2, move on to 3.
    dataset = _load(tmp_path, [(1, 200), (2, 180), (1, 90), (2, 70), (3, 150)])
    first = _values(dataset, "first_fixation")
    first_run = _values(dataset, "first_run")
    first_pass = _values(dataset, "first_pass")
    go_past = _values(dataset, "go_past")
    total = _values(dataset, "total_time")

    assert first[0] == 200  # first landing only
    assert first_run[0] == 200  # the run ends when the eye leaves
    assert first_pass[0] == 1  # reached before any word to its right
    assert total[0] == 290  # both visits
    assert go_past[0] == 200  # passed as soon as word 2 is fixated
    # Word 2 is entered at 200, and the eye only passes it at word 3.
    assert go_past[1] == 180 + 90 + 70
    assert np.isnan(first_run[3])  # never fixated
    assert first_pass[3] == 0
    assert np.isnan(first[3])


def test_off_screen_events_are_fixations_without_a_unit(tmp_path):
    dataset = _load(tmp_path, [(1, 200), (-99, 120), (2, 160)])
    fixations = dataset.fixations()
    assert [f.unit_index for f in fixations] == [0, None, 1]
    assert fixations[1].char_offset is None
    assert fixations[1].metadata["event"] == "off-screen"
    assert _values(dataset, "total_time")[0] == 200


def test_a_blink_advances_time_without_adding_word_dwell(tmp_path):
    dataset = _load(tmp_path, [(1, 200), (-99, 120), (2, 160)], marker="*Blink")
    assert [f.unit_index for f in dataset.fixations()] == [0, 1]
    assert _values(dataset, "total_time")[0] == 200
    assert _values(dataset, "first_run")[0] == 200
    assert _values(dataset, "first_pass")[0] == 1


def test_a_word_entered_by_a_regression_has_no_first_pass(tmp_path):
    # Read word 2, then go back to word 1: word 1 was never first-passed.
    dataset = _load(tmp_path, [(2, 180), (1, 200), (3, 150)])
    dataset.drop_skips = False
    assert _values(dataset, "first_pass")[0] == 0  # a first-pass skip
    assert dataset.data(measure="first_run_in_first_pass")[0].values[0] == 0
    assert _values(dataset, "go_past")[0] == 200
    assert dataset.data(measure="go_past_in_first_pass")[0].values[0] == 0
    # The first run and the first fixation happen whenever the eye arrives.
    assert _values(dataset, "first_run")[0] == 200
    assert _values(dataset, "first_fixation")[0] == 200
    # Total time counts every fixation, however the word was reached.
    assert _values(dataset, "total_time")[0] == 200


def test_accented_words_decode_as_the_release_wrote_them(tmp_path):
    items = tmp_path / "english" / "items"
    items.mkdir(parents=True)
    # CP437 0x82 is an e-acute; latin-1 would make it a control character.
    row = "Val\x82ry".encode("latin-1").decode("latin-1")
    (items / "tx01wrdp.dat").write_bytes(
        (f"{row:<28}  1  1 1   1   1   0   6   6   0   0   0   1   10\n").encode(
            "latin-1"
        )
    )
    (items / "text01.itm").write_bytes("+Valéry>\n".encode("cp437"))
    (tmp_path / "english" / "sa01ma1p.dat").write_text("")
    dataset = DundeeOfficial()
    dataset.load(str(tmp_path))
    assert dataset.stimuli()[0].unit_strings() == ["Valéry"]


def test_corrupt_rows_are_not_events(tmp_path):
    _write_release(tmp_path, [(1, 200), (2, 160)])
    path = tmp_path / "english" / "sa01ma1p.dat"
    nul = "\x00" * 14
    path.write_text(
        path.read_text()
        + f"{nul:<24} 0     0     0     0     0     0     0     0     0     0     0\n"
    )
    dataset = DundeeOfficial()
    dataset.load(str(tmp_path))
    assert len(dataset.fixations()) == 2


def test_fixations_carry_synthetic_timing_and_the_character_grid(tmp_path):
    dataset = _load(tmp_path, [(1, 200), (2, 180)])
    first, second = dataset.fixations()
    assert first.metadata["synthetic_start_time"] is True
    assert (first.start_time, first.end_time) == (0.0, 200.0)
    assert second.start_time == 200.0
    assert first.metadata["xpos_chars"] == 1
    assert first.metadata["line"] == 1
    # The release gives a character column without screen coordinates, so x and
    # y are NaN.
    assert np.isnan(first.x) and np.isnan(first.y)


def test_missing_release_directory_is_reported(tmp_path):
    dataset = DundeeOfficial()
    with pytest.raises(FileNotFoundError, match="distributed by its author"):
        dataset.load(str(tmp_path))


def test_word_indices_must_be_contiguous(tmp_path):
    items = tmp_path / "english" / "items"
    items.mkdir(parents=True)
    (items / "tx01wrdp.dat").write_text(
        "The   1  1 1   1   1   0   3   3   0   0   0   1   10\n"
        "cat   1  1 1   2   2   0   3   3   0   0   0   7   10\n"
    )
    (items / "text01.itm").write_text("+The cat>\n")
    (tmp_path / "english" / "sa01ma1p.dat").write_text("")
    with pytest.raises(ValueError, match="incomplete source word indices"):
        DundeeOfficial().load(str(tmp_path))


def test_screen_conflicts_invalidate_affected_measures_and_keep_event(tmp_path):
    _write_release(tmp_path, [(1, 200), (2, 150), (3, 100)])
    path = tmp_path / "english" / "sa01ma1p.dat"
    lines = path.read_text().splitlines()
    fields = lines[2].split()
    fields[2] = "2"  # Claim line 2 for a source object that sits on line 1.
    lines[2] = " ".join(fields)
    path.write_text("\n".join(lines) + "\n")
    ds = DundeeOfficial()
    ds.load(str(tmp_path))
    assert [f.unit_index for f in ds.fixations()] == [0, None, 2]
    assert ds.fixations()[1].metadata["source_word_number"] == 2
    for measure in ("first_fixation", "first_run", "go_past", "total_time"):
        assert np.isnan(_values(ds, measure)[1])
        assert _values(ds, measure)[2] == 100
    assert np.isnan(_values(ds, "first_pass")[1])
    assert _values(ds, "first_pass")[2] == 1
    assert _values(ds, "first_fixation")[0] == 200
    assert _values(ds, "total_time")[0] == 200
    # The conflicting event also defines the previous region's departure.
    assert np.isnan(_values(ds, "first_run")[0])
    assert np.isnan(_values(ds, "go_past")[0])


def test_positive_duration_nul_record_advances_clock(tmp_path):
    _write_release(tmp_path, [(1, 200), (2, 160)])
    path = tmp_path / "english" / "sa01ma1p.dat"
    lines = path.read_text().splitlines()
    lines.insert(2, "\x00\x00 0 0 0 0 0 0 50 0 0 0 0")
    path.write_text("\n".join(lines) + "\n")
    ds = DundeeOfficial()
    ds.load(str(tmp_path))
    assert len(ds.fixations()) == 2
    assert ds.fixations()[1].start_time == 250


def test_go_past_includes_regressions_and_ends_at_the_first_word_beyond():
    go_past = go_past_duration([0, 1, 0, 1, 2], [100, 120, 90, 80, 150], 3)
    assert go_past[0] == 100
    assert go_past[1] == 120 + 90 + 80
    assert go_past[2] == 150  # closed at the end of the trial


def test_go_past_leaves_unfixated_units_nan():
    go_past = go_past_duration([2], [100], 4)
    assert np.isnan(go_past[0])
    assert go_past[2] == 100


@pytest.mark.parametrize("marker", ["*Off-screen", "*Blink"])
def test_interruptions_preserve_fixation_dwell_and_runs(tmp_path, marker):
    dataset = _load(tmp_path, [(1, 200), (-99, 120), (1, 80), (2, 160)], marker)
    assert _values(dataset, "total_time")[0] == 280
    assert _values(dataset, "first_run")[0] == (200 if marker == "*Off-screen" else 280)
    assert _values(dataset, "go_past")[0] == 280
    assert dataset.fixations()[-1].start_time == 400


def test_conflicting_forward_position_does_not_invent_later_first_pass_skips(
    tmp_path,
):
    ds = _conflicting_forward_release(tmp_path)
    for drop in (False, True):
        ds.drop_skips = drop
        # The disputed event could have progressed to word 4; word 3's later
        # fixation therefore cannot establish either a first pass or a skip.
        assert np.isnan(_values(ds, "first_fixation")[2])
        assert np.isnan(_values(ds, "first_run")[2])
        assert np.isnan(_values(ds, "first_pass")[2])
        assert _values(ds, "total_time")[2] == 100


def test_known_first_pass_skip_before_a_later_conflict_keeps_its_first_fixation(
    tmp_path,
):
    dataset = _skip_before_conflict_release(tmp_path)
    assert _values(dataset, "first_fixation")[0] == 200


def _assert_official_fixation_measures(dataset, *, min_compared):
    from replica.dataset.loading.reaggregation import aggregate_fixations_to_layer
    from tests.dataset._fixation_test_base import group_fixations

    events = group_fixations(dataset.fixations())
    stimuli = {s.stimulus_id: s for s in dataset.stimuli()}
    arrays = {
        (participant, stimulus): aggregate_fixations_to_layer(
            fixations, stimuli[stimulus], "words"
        )
        for (participant, stimulus), fixations in events.items()
    }
    dataset.drop_skips = False
    for measure in ("first_fixation", "first_run", "first_pass", "total_time"):
        compared = 0
        for datum in dataset.data(measure):
            key = (datum.participant_id, datum.stimulus.stimulus_id)
            expected = arrays[key][measure].copy()
            if measure == "first_fixation":
                skipped = np.isnan(expected) & (arrays[key]["total_time"] == 0)
                expected[skipped] = 0
            unavailable = [
                entry["unit_index"]
                for entry in datum.metadata["unavailable_source_values"]
                if entry["measure"] == measure
            ]
            # Only the declared source conflicts may invalidate a reconstruction.
            assert np.isnan(datum.values[unavailable]).all()
            expected[unavailable] = np.nan
            np.testing.assert_equal(datum.values, expected, err_msg=str((key, measure)))
            compared += np.isfinite(datum.values).sum()
        assert compared > min_compared


def test_round_trip_rejects_missingness_without_source_conflict_provenance(tmp_path):
    dataset = _load(tmp_path, [(1, 200), (2, 160)])
    next(d for d in dataset._data if d.measure_name == "first_fixation").values[0] = (
        np.nan
    )
    with pytest.raises(AssertionError):
        _assert_official_fixation_measures(dataset, min_compared=0)


@pytest.mark.needs_data
def test_official_release_matches_available_fixation_measures(data_root):
    from pathlib import Path

    root = Path(data_root) / "eye-tracking/original/dundee"
    if not root.is_dir():
        pytest.skip("Dundee official release is unavailable")
    dataset = DundeeOfficial()
    dataset.load(str(root))
    _assert_official_fixation_measures(dataset, min_compared=300_000)


@pytest.mark.parametrize("row", ["truncated 1 2", "word 1 1 1 1 1 1 bad 1 1 1 1"])
def test_malformed_event_rows_raise(tmp_path, row):
    _write_release(tmp_path, [(1, 200)])
    path = tmp_path / "english/sa01ma1p.dat"
    path.write_text(path.read_text() + row + "\n")
    with pytest.raises(ValueError, match="Malformed Dundee event"):
        DundeeOfficial().load(str(tmp_path))


def test_conflicting_line_assignment_resolves_at_text_level(tmp_path):
    from replica.core import AnnotationLayer, Span

    _write_release(tmp_path, [(1, 200), (2, 150), (3, 100)])
    # The event says word 2 but its line contains words 3 and 4.
    path = tmp_path / "english/items/tx01wrdp.dat"
    lines = path.read_text().splitlines()
    for i in (2, 3):
        fields = lines[i].split()
        fields[3] = "2"
        lines[i] = " ".join(fields)
    path.write_text("\n".join(lines) + "\n")
    path = tmp_path / "english/sa01ma1p.dat"
    lines = path.read_text().splitlines()
    for i in (2, 3):
        fields = lines[i].split()
        fields[2] = "2"
        lines[i] = " ".join(fields)
    path.write_text("\n".join(lines) + "\n")
    dataset = DundeeOfficial()
    dataset.load(str(tmp_path))
    assert dataset.fixations()[1].candidate_unit_indices == (1, 2, 3)
    [stimulus] = dataset.stimuli()
    stimulus.add_layer(
        AnnotationLayer("text", [Span(0, len(stimulus.text))], [stimulus.text])
    )
    for drop_skips in (False, True):
        dataset.drop_skips = drop_skips
        for measure in dataset.measures:
            [datum] = dataset.data(measure)
            expected = (
                1
                if measure == "first_pass"
                else 200
                if measure.startswith("first_fixation")
                else 450
            )
            result = dataset.reaggregate(datum, "text")
            np.testing.assert_equal(result.values, [expected])
