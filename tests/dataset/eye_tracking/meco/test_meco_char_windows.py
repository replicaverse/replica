"""Character reading windows must use only fixations on the target character."""

import csv
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from replica.dataset.eye_tracking.meco.character import LANGUAGES, MECO_char


def test_right_bounded_excludes_time_on_other_characters(tmp_path):
    with (tmp_path / "stimulus_en.csv").open("w") as file:
        file.write("trial_id,letternum,character\n1,1,a\n1,2,b\n1,3,c\n1,4,d\n")
    with (tmp_path / "en.csv").open("w") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "reader",
                "trial_id",
                "letternum",
                "fixid",
                "start",
                "dur",
                "right_bounded_time",
            ]
        )
        for i, (letter, duration) in enumerate(
            [(2, 100), (1, 70), (2, 90), (3, 80), (2, 60)]
        ):
            writer.writerow(["p", 1, letter, i, i, duration, 999])
    dataset = MECO_char()
    dataset.load(str(tmp_path))
    dataset.drop_skips = False
    np.testing.assert_equal(dataset.data("right_bounded")[0].values, [0, 190, 80, 0])
    np.testing.assert_equal(dataset.data("total_time")[0].values, [70, 250, 80, 0])
    assert all(
        f.metadata["source_right_bounded_time"] == 999 for f in dataset.fixations()
    )
    dataset.drop_skips = True
    np.testing.assert_equal(
        dataset.data("right_bounded")[0].values, [np.nan, 190, 80, np.nan]
    )


@pytest.mark.needs_data
@pytest.mark.parametrize("language", LANGUAGES)
def test_right_bounded_matches_source_prefixes_in_every_language(data_root, language):
    path = Path(data_root) / "eye-tracking/processed/meco_char"
    code = "ge" if language == "de" else language
    if not (path / f"{code}.csv").exists():
        pytest.skip(f"MECO character export unavailable for {language}")
    dataset = MECO_char(language)
    dataset.load(str(path))
    dataset.drop_skips = False
    totals = {
        (d.participant_id, d.stimulus.stimulus_id): d.values
        for d in dataset.data("total_time")
    }
    for datum in dataset.data("right_bounded"):
        assert np.all(
            datum.values <= totals[(datum.participant_id, datum.stimulus.stimulus_id)]
        )
    frame = pd.read_csv(path / f"{code}.csv", dtype={"reader": str})
    # Check a full trial independently, including unvisited characters.
    datum = dataset.data("right_bounded")[0]
    trial = datum.stimulus.metadata["trial_id"]
    rows = frame[
        (frame.reader == datum.participant_id) & (frame.trial_id == trial)
    ].sort_values("start", kind="stable")
    positions = rows.letternum.to_numpy()
    durations = rows.dur.to_numpy()
    stimuli = pd.read_csv(path / f"stimulus_{code}.csv", keep_default_na=False)
    letters = stimuli[stimuli.trial_id == trial].sort_values("letternum").letternum
    expected = []
    for letter in letters:
        exits = np.flatnonzero(positions > letter)
        end = exits[0] if len(exits) else len(positions)
        expected.append(durations[:end][positions[:end] == letter].sum())
    np.testing.assert_equal(datum.values, expected)


@pytest.mark.parametrize(
    ("start", "duration"), [("inf", "100"), ("0", "inf"), ("0", "-1")]
)
def test_invalid_fixation_timing_cannot_become_a_skip(tmp_path, start, duration):
    (tmp_path / "stimulus_en.csv").write_text("trial_id,letternum,character\n1,1,a\n")
    (tmp_path / "en.csv").write_text(
        f"reader,trial_id,letternum,fixid,start,dur,right_bounded_time\np,1,1,1,{start},{duration},999\n"
    )
    with pytest.raises(ValueError, match="Invalid MECO character fixation timing"):
        MECO_char().load(str(tmp_path))


def test_missing_duration_is_retained_and_poisons_only_its_character_sum(tmp_path):
    (tmp_path / "stimulus_en.csv").write_text(
        "trial_id,letternum,character\n1,1,a\n1,2,b\n1,3,c\n"
    )
    (tmp_path / "en.csv").write_text(
        "reader,trial_id,letternum,fixid,start,dur\np,1,1,1,0,100\np,1,1,2,1,NA\np,1,1,3,2,50\np,1,3,4,3,200\n"
    )
    dataset = MECO_char()
    dataset.load(str(tmp_path))
    dataset.drop_skips = False
    assert len(dataset.fixations()) == 4
    assert np.isnan(dataset.fixations()[1].duration)
    for measure in ("total_time", "right_bounded"):
        np.testing.assert_equal(dataset.data(measure)[0].values, [np.nan, 0, 200])
