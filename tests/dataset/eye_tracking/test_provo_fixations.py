"""Check every released Provo area by source identity, including missing cloze rows.

The reference uses IA_ID and physical CSV rows. It does not copy the loader's
word-label walk, discard duplicate Word_Number metadata, or exclude compounds.
"""

import csv
import os
import string
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest

from replica.dataset import PROVO

pytestmark = [pytest.mark.needs_data, pytest.mark.slow]


@pytest.fixture(scope="module")
def provo_source(data_root):
    root = Path(data_root) / "eye-tracking/original/provo"
    if not (
        root / "Provo_Corpus-Additional_Eyetracking_Data-Fixation_Report.csv"
    ).exists():
        pytest.skip(f"Provo fixation release unavailable: {root}")
    dataset = PROVO()
    dataset.load(os.fspath(root))
    with (root / "Provo_Corpus-Eyetracking_Data.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    return dataset, rows


def test_every_provo_area_has_the_correct_text_and_duration(provo_source):
    dataset, rows = provo_source
    stimuli = {s.metadata["text_id"]: s for s in dataset.stimuli()}
    data = {
        (d.stimulus.metadata["text_id"], d.participant_id, d.measure_name): d.values
        for d in dataset._data
    }
    trials = {
        (d.stimulus.metadata["text_id"], d.participant_id): d.metadata[
            "source_trial_index"
        ]
        for d in dataset._data
    }
    measures = {
        "IA_FIRST_FIXATION_DURATION": "first_fixation",
        "IA_FIRST_RUN_DWELL_TIME": "first_run",
        "IA_REGRESSION_PATH_DURATION": "go_past",
        "IA_DWELL_TIME": "total_time",
    }
    for row in rows:
        text, position = int(row["Text_ID"]), int(row["IA_ID"]) - 1
        assert trials[text, row["RECORDING_SESSION_LABEL"]] == int(row["TRIAL_INDEX"])
        label = row["IA_LABEL"].strip()
        if (text, position, label) == (36, 23, "?"):
            label = "—"

        # Punctuation/encoding equivalences leave the complete lexical label
        # intact; the two double-dash compounds must match as whole regions.
        def lexical(value):
            return (
                value.lower()
                .replace("’", "'")
                .replace("?", "'")
                .strip(string.punctuation)
            )

        assert lexical(label) == lexical(stimuli[text].unit_strings()[position])
        assert stimuli[text].metadata["source_ia_to_unit"][position + 1] == position
        for source, name in measures.items():
            try:
                expected = float(row[source])
            except ValueError:
                expected = 0 if row["IA_DWELL_TIME"] == "0" else np.nan
            observed = data[text, row["RECORDING_SESSION_LABEL"], name][position]
            assert observed == expected or (np.isnan(observed) and np.isnan(expected))
        # The first-pass flag is EyeLink's IA_SKIP inverted.
        flag = data[text, row["RECORDING_SESSION_LABEL"], "first_pass"]
        assert flag[position] == (0.0 if row["IA_SKIP"] == "1" else 1.0)
    assert len(rows) == 230412


def test_every_provo_fixation_preserves_its_area_and_timing(provo_source):
    dataset, _ = provo_source
    assert len(dataset.fixations()) > 100000
    for event in dataset.fixations():
        assert event.duration == event.end_time - event.start_time + 1
        area = event.metadata.get("ia_index")
        assert event.unit_index == (None if area is None else area - 1)


def test_provo_fixation_sums_match_all_source_first_runs_and_totals(provo_source):
    dataset, rows = provo_source
    # The summary retains the first presentation when a trial was repeated.
    canonical = {}
    for row in rows:
        key = row["RECORDING_SESSION_LABEL"], int(row["Text_ID"])
        trial = int(row["TRIAL_INDEX"])
        canonical[key] = min(canonical.get(key, trial), trial)
    total, first_run = defaultdict(float), defaultdict(float)
    for event in dataset.fixations():
        text = int(event.stimulus_id.removeprefix("provo_"))
        key = event.participant_id, text
        if (
            event.unit_index is None
            or canonical.get(key) != event.metadata["trial_index"]
        ):
            continue
        cell = (*key, event.unit_index)
        total[cell] += event.duration
        if event.metadata.get("run_id") == 1:
            first_run[cell] += event.duration
    for row in rows:
        key = row["RECORDING_SESSION_LABEL"], int(row["Text_ID"]), int(row["IA_ID"]) - 1
        for column, reconstructed in [
            ("IA_DWELL_TIME", total),
            ("IA_FIRST_RUN_DWELL_TIME", first_run),
        ]:
            try:
                expected = float(row[column])
            except ValueError:
                continue
            assert abs(reconstructed[key] - expected) <= 1, (
                key,
                column,
                reconstructed[key],
                expected,
            )


def test_provo_zero_views_and_event_filters_are_reversible(provo_source):
    dataset, _ = provo_source
    raw = {
        (d.stimulus.stimulus_id, d.participant_id, d.measure_name): d.values.copy()
        for d in dataset._data
    }
    for drop in (False, True, False):
        dataset.drop_skips = drop
        for d in dataset.data():
            expected = raw[d.stimulus.stimulus_id, d.participant_id, d.measure_name]
            masked = drop and d.measure_name != "first_pass"
            np.testing.assert_equal(
                d.values,
                np.where(expected == 0, np.nan, expected) if masked else expected,
            )
    dataset.drop_skips = True
    first = dataset.fixations()[0]
    assert all(
        f.participant_id == first.participant_id
        for f in dataset.fixations(participant=first.participant_id)
    )
    assert all(
        f.stimulus_id == first.stimulus_id
        for f in dataset.fixations(stimulus_id=first.stimulus_id)
    )
