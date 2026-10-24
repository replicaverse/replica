"""Round-trip + sanity correctness for CopCo's raw fixation API.

CopCo's fixation reports use character-level interest areas (one IA per
character of a paragraph). REPLICA aggregates to word level via
``char_IA_ids`` from ExtractedFeatures (a list of char-IAs per word).
The round-trip test verifies that aggregating raw fixations to the
``words`` layer reproduces the per-word ``data()`` API.

Smoke / invariant tests come from :class:`FixationsTestBase`.
"""

import os
from dataclasses import replace

import pytest

from tests._require import load_or_skip
from tests.dataset._fixation_test_base import (
    FixationsTestBase,
    SyntheticTimingChecks,
    assert_fixation_round_trip,
    group_fixations,
)

pytestmark = pytest.mark.needs_data


@pytest.mark.slow
class TestCopCoFixations(FixationsTestBase, SyntheticTimingChecks):
    min_fixations = 50_000
    # CopCo's per-fixation report ships only CURRENT_FIX_DURATION;
    # CURRENT_FIX_START / _END are absent, so trial-relative timing is
    # synthesised by accumulating durations.

    @pytest.fixture(autouse=True)
    def load(self, data_root):
        from replica.dataset import CopCo

        path = os.path.join(data_root, "eye-tracking", "original", "copco")
        self.ds = load_or_skip(CopCo(), path)

    def test_metadata_carries_paragraph_id(self):
        """Each fixation's metadata records the paragraph_id and
        sentence_id needed for cross-corpus subsetting."""
        f = next(f for f in self.ds.fixations() if f.unit_index is not None)
        assert f.metadata.get("paragraph_id") is not None
        assert f.metadata.get("sentence_id") is not None

    def test_all_nonpractice_events_have_a_preserved_clock(self, data_root):
        from pathlib import Path

        expected = {}
        coordinates = {}
        root = Path(data_root) / "eye-tracking/original/copco/FixationReports"
        for path in sorted(root.glob("FIX_report_P*.txt")):
            with path.open(encoding="utf-8-sig") as file:
                header = next(file).rstrip("\r\n").split("\t")
                header = [column.strip('"') for column in header]
                for line in file:
                    fields = line.rstrip("\r\n").split("\t")
                    assert len(fields) == len(header)
                    row = dict(zip(header, fields, strict=True))
                    if row["speechid"] == row["paragraphid"] == "UNDEFINEDnull":
                        continue
                    speech, paragraph = int(row["speechid"]), int(row["paragraphid"])
                    if speech == 1327 or paragraph < 0:
                        continue
                    key = (
                        row["RECORDING_SESSION_LABEL"].strip('"'),
                        int(row["TRIAL_INDEX"]),
                        int(row["CURRENT_FIX_INDEX"]),
                    )
                    assert key not in expected
                    expected[key] = float(row["CURRENT_FIX_DURATION"])
                    coordinates[key] = tuple(
                        float(row[column].replace(",", "."))
                        for column in ("CURRENT_FIX_X", "CURRENT_FIX_Y")
                    )
        actual = {
            (f.participant_id, f.metadata["trial_index"], f.fixation_index): f.duration
            for f in self.ds.fixations()
        }
        assert len(actual) == len(self.ds.fixations()) == 397_883
        assert actual == expected
        for event in self.ds.fixations():
            key = (
                event.participant_id,
                event.metadata["trial_index"],
                event.fixation_index,
            )
            assert (event.x, event.y) == coordinates[key]
        elapsed = {}
        for event in self.ds.fixations():
            key = (
                event.participant_id,
                event.stimulus_id,
                event.metadata["trial_index"],
            )
            assert event.start_time == elapsed.get(key, 0)
            elapsed[key] = event.end_time
        assert sum(f.unit_index is None for f in self.ds.fixations()) > 30_000

    def _released_feature_fixations(self):
        # Missing character IDs and period labels mark off-word boundaries.
        events = []
        for fixation in self.ds.fixations():
            metadata = {k: v for k, v in fixation.metadata.items() if k != "run_id"}
            if (
                metadata["inferred_character_mapping"]
                or metadata["source_ia_label"] == "."
            ):
                fixation = replace(fixation, unit_index=None, char_offset=None)
            events.append(replace(fixation, metadata=metadata))
        return group_fixations(events)

    def test_missing_character_events_are_recovered(self):
        assert (
            sum(f.metadata["inferred_character_mapping"] for f in self.ds.fixations())
            > 2000
        )

    def test_round_trip_reproduces_total_time(self):
        """Aggregating fixations to words reproduces ``data(total_time)``
        (=word_total_fix_dur in CopCo's ExtractedFeatures)."""
        # The source assignment rules reproduce the released word summaries.
        assert_fixation_round_trip(
            self.ds,
            fixations_by_key=self._released_feature_fixations(),
            source_measure="total_time",
            min_compared=10_000,
            tolerance=0.0,
        )

    def test_round_trip_reproduces_first_run(self):
        # Original report indices determine adjacency in the author extractor.
        assert_fixation_round_trip(
            self.ds,
            fixations_by_key=self._released_feature_fixations(),
            source_measure="first_run",
            tolerance=0.0,
            min_compared=10_000,
        )

    def test_round_trip_reproduces_first_pass(self):
        # The loader derives the flag from the same stream, so it must agree
        # exactly wherever the stream reproduced the released first run; the
        # flag is NaN on the trials where it did not.
        assert_fixation_round_trip(
            self.ds,
            fixations_by_key=self._released_feature_fixations(),
            source_measure="first_pass",
            tolerance=0.0,
            min_compared=10_000,
        )
