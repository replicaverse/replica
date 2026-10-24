"""Round-trip + independent correctness for OneStop's raw fixation API.

Three layers of verification:

1. Independent: per (participant, stim, trial, unit), sum of the
   fixation durations equals EyeLink's
   ``CURRENT_FIX_INTEREST_AREA_DWELL_TIME``. Uses only source columns,
   no loader quirks.
2. Round-trip total: aggregating fixations to the words layer
   reproduces ``data(measure="total_time")`` (= ``IA_DWELL_TIME``).
3. Round-trip first run: aggregating reproduces
   ``data(measure="first_run")`` (= ``IA_FIRST_RUN_DWELL_TIME``) and the
   ``first_pass`` flag (= ``IA_SKIP`` inverted).

Smoke / invariant tests come from :class:`FixationsTestBase`.
"""

import os

import pytest

from tests.dataset._fixation_test_base import (
    FixationsTestBase,
    RecordedTimingChecks,
    assert_fixation_round_trip,
    assert_trial_dwell_times,
)

pytestmark = pytest.mark.needs_data


@pytest.mark.slow
class TestOneStopFixations(FixationsTestBase, RecordedTimingChecks):
    min_fixations = 100_000

    @pytest.fixture(autouse=True)
    def load(self, data_root):
        from replica.dataset import OneStop

        path = os.path.join(data_root, "eye-tracking", "original", "onestop")
        if not os.path.isdir(path):
            pytest.skip(f"OneStop data not present at {path}")
        self.ds = OneStop()
        self.ds.load(path)

    def test_per_trial_sum_matches_eyelink_dwell(self):
        assert_trial_dwell_times(
            self.ds,
            tolerance=1.5,
            min_compared=50_000,
            positive_only=True,
        )

    @pytest.mark.parametrize(
        "source_measure",
        ["total_time", "first_run", "first_pass"],
        ids=["total_time", "first_run", "first_pass_flag"],
    )
    def test_round_trip_reproduces_measure(self, source_measure):
        # Reconstruct IA_DWELL_TIME, IA_FIRST_RUN_DWELL_TIME and inverted IA_SKIP.
        assert_fixation_round_trip(
            self.ds, source_measure=source_measure, tolerance=0.0, min_compared=50_000
        )
