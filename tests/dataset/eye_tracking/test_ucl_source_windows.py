"""Check UCL's reading windows against the released tables and raw events."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from replica.core import Datum
from replica.dataset.contracts import Fixation
from replica.dataset.eye_tracking.ucl import UCL_ET


def test_first_arrival_recovers_a_regression_without_changing_first_pass():
    dataset = UCL_ET()
    stimulus = dataset._make_stimulus(["one", "two", "three", "four"], "s")
    dataset._data = [
        Datum(stimulus, "first_fixation", np.array([100, np.nan, 120, np.nan]), "p"),
        Datum(
            stimulus, "first_fixation_in_first_pass", np.array([100, 0, 120, 0]), "p"
        ),
    ]
    dataset._fixations = [
        Fixation(
            "p",
            "s",
            i * 200,
            i * 200 + duration,
            duration,
            np.nan,
            np.nan,
            unit,
            None,
            i,
        )
        for i, (unit, duration) in enumerate([(0, 100), (2, 120), (1, 80), (1, 90)])
    ]
    dataset._complete_first_fixation("eyetracking.fix.txt")
    np.testing.assert_equal(dataset._data[0].values, [100, 80, 120, 0])
    np.testing.assert_equal(dataset._data[1].values, [100, 0, 120, 0])
    for drop in (True, False, True):
        dataset.drop_skips = drop
        np.testing.assert_equal(
            dataset.data("first_fixation")[0].values,
            [100, 80, 120, np.nan if drop else 0],
        )


@pytest.mark.needs_data
def test_all_ucl_first_fixations_and_bounded_windows_match_sources(data_root):
    path = Path(data_root) / "eye-tracking/original/ucl"
    if not (path / "eyetracking.fix.txt").exists():
        pytest.skip("UCL release unavailable")
    dataset = UCL_ET()
    dataset.load(str(path))
    dataset.drop_skips = False
    table = pd.read_csv(path / "eyetracking.RT.txt", sep="\t")
    events = pd.read_csv(path / "eyetracking.fix.txt", sep="\t")
    trials = {
        (
            int(d.participant_id),
            d.stimulus.metadata["sent_nr"],
            d.measure_name,
        ): d.values
        for d in dataset.data()
    }
    recovered = 0
    for (subject, sentence), rows in table.groupby(["subj_nr", "sent_nr"]):
        stream = events[(events.subj_nr == subject) & (events.sent_nr == sentence)]
        positions = stream.word_pos.to_numpy()
        durations = stream.fix_duration.to_numpy()
        for row in rows.itertuples():
            index = row.word_pos - 1
            visits = np.flatnonzero(positions == row.word_pos)
            expected_first = durations[visits[0]] if len(visits) else 0
            assert (
                trials[(subject, sentence, "first_fixation")][index] == expected_first
            )
            assert (
                trials[(subject, sentence, "first_fixation_in_first_pass")][index]
                == row.RTfirstfix
            )
            exits = np.flatnonzero(positions > row.word_pos)
            before_exit = visits[visits < exits[0]] if len(exits) else visits
            expected_right = durations[before_exit].sum()
            assert expected_right == row.RTrightbound
            assert trials[(subject, sentence, "right_bounded")][index] == expected_right
            expected_path = 0
            if row.RTfirstpass > 0:
                start = visits[0]
                end = exits[0] if len(exits) else len(durations)
                expected_path = durations[start:end].sum()
            assert expected_path == row.RTgopast
            assert (
                trials[(subject, sentence, "go_past_in_first_pass")][index]
                == expected_path
            )
            recovered += row.RTfirstfix == 0 and expected_first > 0
    assert len(table) == 81_109
    assert recovered == 7_979
