"""Round-trip checks for EMTeC's complete measures.

The author's calculator floors each duration to integer milliseconds and
ignores off-word events when defining runs. The event API retains all rows
and their original precision. No mismatch budget is used for complete measures.
"""

import csv
import os
from collections import defaultdict

import numpy as np
import pytest

from tests._require import load_or_skip
from tests.dataset._fixation_test_base import (
    FixationsTestBase,
    SyntheticTimingChecks,
    assert_fixation_round_trip,
)

pytestmark = pytest.mark.needs_data


@pytest.mark.slow
class TestEMTeCFixations(FixationsTestBase, SyntheticTimingChecks):
    min_fixations = 100_000

    @pytest.fixture(autouse=True)
    def load(self, data_root):
        from replica.dataset import EMTeC

        path = os.path.join(data_root, "eye-tracking", "original", "emtec")
        self.ds = load_or_skip(EMTeC(), path)

    def _floor_durations(self) -> dict:
        """Return per-(participant, stim) lists of fixations with durations
        floored to int (matching EMTeC's pipeline rounding)."""
        from replica.dataset.contracts import Fixation

        out: dict = {}
        for f in self.ds.fixations():
            d = f.duration
            f2 = Fixation(
                participant_id=f.participant_id,
                stimulus_id=f.stimulus_id,
                start_time=f.start_time,
                end_time=f.end_time,
                duration=float(int(d)) if np.isfinite(d) else d,
                x=f.x,
                y=f.y,
                unit_index=f.unit_index,
                char_offset=f.char_offset,
                fixation_index=f.fixation_index,
                metadata=f.metadata,
            )
            out.setdefault((f.participant_id, f.stimulus_id), []).append(f2)
        return out

    def test_aggregator_preserves_sub_ms_precision(self):
        """Premise check: source ships 0.5 ms precision. If this fails,
        EMTeC's release format changed and the rest of these tests need
        to be re-thought."""
        sample = self.ds.fixations()[:10_000]
        has_half_ms = any(
            np.isfinite(f.duration) and (f.duration * 2) % 2 == 1 for f in sample
        )
        assert has_half_ms

    @pytest.mark.parametrize(
        "source_measure,missing_reconstruction,floor_durations",
        [
            ("total_time", "zero", True),
            ("first_fixation", "nan", True),
            ("first_run", "zero", True),
            ("first_pass", "zero", False),
        ],
        ids=["total_time_with_flooring", "first_fixation", "first_run", "first_pass"],
    )
    def test_round_trip_reproduces_measure(
        self, source_measure, missing_reconstruction, floor_durations
    ):
        assert_fixation_round_trip(
            self.ds,
            source_measure=source_measure,
            fixations_by_key=self._floor_durations() if floor_durations else None,
            tolerance=0.0,
            missing_reconstruction=missing_reconstruction,
            min_compared=10_000,
        )

    def test_every_released_fixation_contributes_to_the_timeline(self, data_root):
        source = os.path.join(
            data_root, "eye-tracking", "original", "emtec", "fixations.csv"
        )
        elapsed = defaultdict(float)
        off_text = 0
        with open(source, newline="") as stream:
            rows = csv.DictReader(stream, delimiter="\t")
            for row, event in zip(rows, self.ds.fixations(), strict=True):
                trial = (row["subject_id"], int(row["TRIAL_ID"]))
                assert (event.participant_id, event.metadata["trial_index"]) == trial
                assert event.fixation_index == int(row["fixation_index"])
                assert event.duration == float(row["fixation_duration"])
                assert event.start_time == elapsed[trial]
                elapsed[trial] += float(row["fixation_duration"])
                assert event.end_time == elapsed[trial]
                is_off_text = row["word_id"] == "."
                assert (event.unit_index is None) == is_off_text
                if is_off_text:
                    off_text += 1
                    assert event.char_offset is None
        assert off_text > 0
