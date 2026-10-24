"""Round-trip correctness for UCL_ET's raw fixation API.

UCL ships separate ``eyetracking.RT.txt`` (per-word) and
``eyetracking.fix.txt`` (per-fixation) files. The per-word file's
RTfirstfix/RTfirstpass are derived from the per-fixation rows by Frank
et al. (2013)'s preprocessing pipeline, including blink-adjacent fixations.
The round-trip tests retain that default fixation set.

Smoke / invariant tests come from :class:`FixationsTestBase`.
"""

import os

import pytest

from tests.dataset._fixation_test_base import (
    FixationsTestBase,
    SyntheticTimingChecks,
    assert_fixation_round_trip,
)

pytestmark = pytest.mark.needs_data


@pytest.mark.slow
class TestUCLETFixations(FixationsTestBase, SyntheticTimingChecks):
    min_fixations = 50_000

    @pytest.fixture(autouse=True)
    def load(self, data_root):
        from replica.dataset.eye_tracking.ucl import UCL_ET

        path = os.path.join(data_root, "eye-tracking", "original", "ucl")
        if not os.path.isdir(path):
            pytest.skip(f"UCL_ET data not present at {path}")
        self.ds = UCL_ET()
        self.ds.load(path)

    def test_off_text_fixations_present(self):
        """Off-text fixations remain available with ``unit_index=None``."""
        n_off_text = sum(1 for f in self.ds.fixations() if f.unit_index is None)
        assert n_off_text > 0

    def test_blink_flag_populated(self):
        """The source's ``blink`` column marks blink-adjacent fixations with
        ``before/after/both``. The loader must surface those
        as ``metadata["blink"] == True``; ``exclude_blink=True`` then
        drops them. ``exclude_blink=False`` (default) keeps them, Frank
        et al. (2013)'s per-word measures include them."""
        default = self.ds.fixations()
        filtered = self.ds.fixations(exclude_blink=True)
        n_blink_in_default = sum(1 for f in default if f.metadata.get("blink", False))
        assert n_blink_in_default > 0, (
            "loader should mark some fixations as blink-adjacent"
        )
        assert len(filtered) == len(default) - n_blink_in_default
        assert not any(f.metadata.get("blink", False) for f in filtered)

    @pytest.mark.parametrize(
        "source_measure,missing_reconstruction",
        [("first_fixation", "nan"), ("first_run", "zero"), ("first_pass", "zero")],
        ids=["first_fixation", "first_run", "first_pass_flag"],
    )
    def test_round_trip_reproduces_measure(
        self, source_measure, missing_reconstruction
    ):
        # Keep blink-adjacent events, as in Frank et al.'s released word measures.
        # First run also includes fixation-stream completions on regression entries.
        assert_fixation_round_trip(
            self.ds,
            source_measure=source_measure,
            tolerance=0.0,
            missing_reconstruction=missing_reconstruction,
            min_compared=5_000,
        )
