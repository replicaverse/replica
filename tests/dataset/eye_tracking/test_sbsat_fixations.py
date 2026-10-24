"""Correctness tests for SBSAT's raw fixation API.

Three layers of verification:

1. Independent: per (participant, stim, trial, unit), sum of fixation
   durations equals EyeLink's ``CURRENT_FIX_INTEREST_AREA_DWELL_TIME``.
   Uses only source columns, no loader quirks.
2. Round-trip total_time: aggregating fixations to the ``words`` layer
   reproduces ``data(measure="total_time")``.
3. Round-trip first_fixation: aggregating reproduces
   ``data(measure="first_fixation")``.

Total time covers all page visits. First fixation is the first observed event.
Source dwell metadata remains per original interest area, including where two
line-wrapped interest areas map to one word.
"""

import os
from collections import defaultdict

import numpy as np
import pytest

from tests.dataset._fixation_test_base import (
    FixationsTestBase,
    SyntheticTimingChecks,
    assert_fixation_round_trip,
)

pytestmark = pytest.mark.needs_data


@pytest.mark.slow
class TestSBSATFixations(FixationsTestBase, SyntheticTimingChecks):
    min_fixations = 10_000

    @pytest.fixture(autouse=True)
    def load(self, data_root):
        from replica.dataset import SBSAT

        path = os.path.join(data_root, "eye-tracking", "original", "sbsat")
        if not os.path.isdir(path):
            pytest.skip(f"SBSAT data not present at {path}")
        self.ds = SBSAT()
        self.ds.load(path)

    def test_per_trial_sum_matches_eyelink_dwell(self):
        """Compare each original IA with its own released dwell, before merging."""
        sums = defaultdict(float)
        dwell = {}
        for fixation in self.ds.fixations():
            if fixation.unit_index is None:
                continue
            key = (
                fixation.participant_id,
                fixation.stimulus_id,
                fixation.metadata["trial_index"],
                fixation.metadata["source_ia_id"],
            )
            sums[key] += fixation.duration
            dwell[key] = fixation.metadata["dwell_time"]
        assert len(dwell) > 1000
        np.testing.assert_allclose(
            [sums[key] for key in dwell], list(dwell.values()), atol=1.0
        )

    @pytest.mark.parametrize(
        "source_measure,tolerance,exclusive_tolerance,missing_reconstruction",
        [
            ("total_time", 1.0, False, "zero"),
            ("first_fixation", 1.0, True, "nan"),
            ("first_run", 1e-3, False, "zero"),
            ("first_pass", 1e-3, False, "zero"),
        ],
        ids=["total_time", "first_fixation", "first_run", "first_pass"],
    )
    def test_round_trip_reproduces_measure(
        self, source_measure, tolerance, exclusive_tolerance, missing_reconstruction
    ):
        assert_fixation_round_trip(
            self.ds,
            source_measure=source_measure,
            tolerance=tolerance,
            exclusive_tolerance=exclusive_tolerance,
            missing_reconstruction=missing_reconstruction,
            min_compared=1_000,
        )

    def test_every_reading_event_is_retained(self, data_root):
        import pandas as pd

        frame = pd.read_csv(
            os.path.join(data_root, "eye-tracking/original/sbsat/18sat_fixfinal.csv"),
            keep_default_na=False,
        )
        source = frame[frame.type == "reading"]
        events = self.ds.fixations()
        assert len(events) == len(source) == 263_032
        assert sum(f.duration for f in events) == source.CURRENT_FIX_DURATION.sum()
        missing_ia = source.CURRENT_FIX_INTEREST_AREA_ID == ""
        assert missing_ia.sum() == 11_945
        assert sum(f.unit_index is None for f in events) == 14_251
