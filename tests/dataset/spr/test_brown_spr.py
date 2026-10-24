"""Real-data tests for the BrownSPR loader.

The corpus is read from Shain et al. (2024)'s OSF release (project 6wvqe);
fetched via ``replica fetch brown_spr_shain2024logarithmic``.
"""

import os

import numpy as np
import pytest

from replica.dataset import BrownSPR

pytestmark = pytest.mark.needs_data


class TestBrownSPRRealData:
    @pytest.fixture(autouse=True)
    def load_dataset(self, data_root):
        path = os.path.join(
            data_root,
            "self-paced-reading",
            "processed",
            "brown_spr_shain2024logarithmic",
        )
        csv = os.path.join(path, "brown_X.csv")
        if not os.path.exists(csv):
            pytest.skip(
                f"Brown SPR data not at {csv}. "
                "Run `replica fetch brown_spr_shain2024logarithmic`."
            )
        self.ds = BrownSPR()
        self.ds.load(path)

    def test_participant_count(self):
        """Smith & Levy (2013) and the Shain et al. (2024) release: 35 participants."""
        assert len(self.ds.participants) == 35

    def test_stimuli_nonempty(self):
        """Brown SPR is the Brown corpus split into sentences; the OSF
        release ships 450 sentences."""
        assert len(self.ds.stimuli()) == 450

    def test_default_measure(self):
        """``fdur`` (per-word self-paced reading time, ms) is the only DV."""
        assert self.ds.default_measure == "spr_rt"
        # ``measures`` is a sorted list (computed dynamically from loaded
        # data + alias table).
        assert self.ds.measures == ["spr_rt"]

    def test_fdur_is_nonneg(self):
        """Self-paced reading times are non-negative; NaN is allowed for
        skipped/missing words."""
        for d in list(self.ds.data())[:200]:
            non_nan = d.values[~np.isnan(d.values)]
            assert (non_nan >= 0).all(), (
                f"negative fdur in {d.participant_id}/{d.stimulus.stimulus_id}: "
                f"{non_nan[non_nan < 0]}"
            )
