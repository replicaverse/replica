"""Complete trial measures preserve source conventions and terminal fixations."""

import csv
import json

import numpy as np
import pytest

from replica.dataset import RSC, EMTeC, PoTeC


def write_table(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture(params=[PoTeC, EMTeC])
def corpus(request, tmp_path):
    cls = request.param
    words = ["one", "two", "three"]
    rows = []
    for index, word in enumerate(words):
        rows.append(
            dict(
                subject_id="1",
                item_id="1",
                model="test",
                decoding_strategy="test",
                word_id=index,
                word=word,
                type="test",
                task="test",
                word_index_in_text=index + 1,
                word_with_punct=word,
                sent_index_in_text=1,
                FD=100 if index == 0 else 0,
                FFD=100 if index == 0 else 0,
                FRT=100 if index == 0 else 0,
                FPRT=100 if index == 0 else 0,
                TFT=100 if index == 0 else 0,
                RPD_inc=100 if index == 0 else 0,
            )
        )
    summary = "reader1_b0_merged.tsv" if cls is PoTeC else "reading_measures.csv"
    write_table(tmp_path / summary, rows)
    events = []
    for index, (unit, duration) in enumerate([(0, 100), (1, 200)], 1):
        events.append(
            dict(
                reader_id="1",
                subject_id="1",
                item_id="1",
                model="test",
                decoding_strategy="test",
                text_id="b0",
                trial="1",
                TRIAL_ID="1",
                fixation_index=index,
                word_id=unit,
                word=words[unit],
                aoi=[1, 5][unit],
                fixation_duration=duration,
            )
        )
    filename = "reader1_b0_fixations.tsv" if cls is PoTeC else "fixations.csv"
    write_table(tmp_path / filename, events)
    (tmp_path / "word_limits.json").write_text(
        json.dumps({"b0": [[1, 5, 9], [3, 7, 13]]})
    )
    return cls, tmp_path, filename, events


def test_complete_measures_include_terminal_word_and_preserve_source(corpus):
    cls, path, _, _ = corpus
    ds = cls()
    ds.load(str(path))
    for drop in [False, True]:
        ds.drop_skips = drop
        for name in [
            "first_fixation",
            "first_run",
            "total_time",
            "go_past",
            "first_run_in_first_pass",
        ]:
            [datum] = ds.data(name)
            np.testing.assert_equal(datum.values, [100, 200, np.nan if drop else 0])
            np.testing.assert_equal(datum.metadata["released_values"], [100, 0, 0])
            [source] = ds.data(name, source="released")
            np.testing.assert_equal(
                source.values, [100, np.nan if drop else 0, np.nan if drop else 0]
            )
            assert source.metadata["measurement_source"] == "released"
    [go_past] = ds.data("go_past")
    assert "unavailable_outside_first_pass" not in go_past.metadata
    assert go_past.metadata["released_metadata"]["unavailable_outside_first_pass"]
    released = cls(measurement_source="released")
    released.load(str(path))
    released.drop_skips = False
    np.testing.assert_equal(released.data("total_time")[0].values, [100, 0, 0])
    stored = next(datum for datum in ds._data if datum.measure_name == "total_time")
    stored.metadata["released_values"] = np.array([100 + 2j, 0, 0])
    with pytest.raises(ValueError, match="real-valued"):
        ds.data("total_time", source="released")


def test_complete_source_never_falls_back_to_summaries(corpus):
    cls, path, filename, _ = corpus
    (path / filename).unlink()
    with pytest.raises(FileNotFoundError):
        cls().load(str(path))
    cls(measurement_source="released").load(str(path))


def test_duplicate_events_are_rejected(corpus):
    cls, path, filename, events = corpus
    write_table(path / filename, events + [events[-1]])
    with pytest.raises(ValueError, match="Duplicate fixation"):
        cls().load(str(path))


@pytest.mark.parametrize("duration", [-1, float("inf"), float("nan")])
def test_invalid_durations_are_rejected(corpus, duration):
    cls, path, filename, events = corpus
    events[-1]["fixation_duration"] = duration
    write_table(path / filename, events)
    with pytest.raises(ValueError, match="duration"):
        cls().load(str(path))


def test_loaders_retain_author_rounding_and_off_word_policy(corpus):
    cls, path, filename, events = corpus
    first = events[0]
    events = [
        dict(first, fixation_index=1, fixation_duration=100.5),
        dict(first, fixation_index=2, fixation_duration=50.5, word_id=".", aoi="NA"),
        dict(first, fixation_index=3, fixation_duration=200.5),
        dict(events[1], fixation_index=4, fixation_duration=90.5),
    ]
    write_table(path / filename, events)
    ds = cls()
    ds.load(str(path))
    ds.drop_skips = False
    for name in ["first_run", "total_time", "go_past", "first_run_in_first_pass"]:
        [datum] = ds.data(name)
        np.testing.assert_equal(datum.values, [300, 90, 0])
        assert datum.metadata["duration_rounding"] == "floor_ms"
        assert datum.metadata["off_text_ends_run"] is False
    assert [f.duration for f in ds.fixations()] == [100.5, 50.5, 200.5, 90.5]


def test_rsc_preserves_values_and_marks_contradictions():
    ds = RSC()
    ds._build(
        {1: "one two three four"},
        {
            1: {
                "20": [
                    (2, (153, 346, 346, 193, 1)),
                    (3, (100, 100, 200, 200, 1)),
                ]
            }
        },
    )
    [datum] = ds.data("go_past")
    assert datum.values[1] == 193
    assert not ds.exclusion_mask(datum)[1]
    assert ds.exclusion_mask(datum)[2]
    [conditional] = ds.data("go_past_in_first_pass")
    assert not ds.exclusion_mask(conditional)[1]
    ds.exclude_inconsistent_go_past = False
    assert ds.exclusion_mask(datum)[1]
