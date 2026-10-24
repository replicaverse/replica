"""Real-data tests for the PoTeC loader.

Run with: pytest tests/dataset/eye_tracking/test_potec.py -v --data-root <path>
"""

import os

import pytest

from replica.dataset import PoTeC
from replica.dataset.measures import with_derived
from tests.dataset._measures import FLAGGED_MEASURES, assert_eye_tracking_measures

pytestmark = pytest.mark.needs_data

EXPECTED_TEXT_IDS = [
    "b0",
    "b1",
    "b2",
    "b3",
    "b4",
    "b5",
    "p0",
    "p1",
    "p2",
    "p3",
    "p4",
    "p5",
]


class TestPoTeCRealData:
    @pytest.fixture(autouse=True)
    def load_dataset(self, data_root):
        path = os.path.join(data_root, "eye-tracking", "original", "potec")
        if not os.path.isdir(path):
            pytest.skip(f"PoTeC data not found at {path}")
        self.ds = PoTeC()
        try:
            self.ds.load(path)
        except FileNotFoundError as e:
            pytest.skip(f"PoTeC data not loadable: {e}")

    def test_stimulus_count(self):
        """PoTeC has 12 stimuli (6 biology + 6 physics texts)."""
        assert len(self.ds.stimuli()) == 12

    def test_participant_count(self):
        """PoTeC has 75 participants."""
        assert len(self.ds.participants) == 75

    def test_measures(self):
        # FD retains first arrival; FFD/FPRT retain the first-pass windows.
        assert_eye_tracking_measures(
            self.ds, with_derived([*FLAGGED_MEASURES, "go_past_in_first_pass"])
        )

    def test_all_text_ids_present(self):
        """All 12 text IDs (b0-b5, p0-p5) should be present."""
        loaded_ids = sorted(stim.metadata["text_id"] for stim in self.ds.stimuli())
        assert loaded_ids == EXPECTED_TEXT_IDS

    def test_stimulus_id_format(self):
        """Stimulus IDs should follow 'potec_{text_id}' format."""
        for stim in self.ds.stimuli():
            assert stim.stimulus_id.startswith("potec_")

    def test_first_stimulus_words(self):
        """Verify first stimulus (potec_b0) begins with expected German words."""
        stim = next(s for s in self.ds.stimuli() if s.metadata["text_id"] == "b0")
        words = stim.layers["words"].values[:5]
        assert words == [
            "Um",
            "das",
            "Vorhandensein",
            "der",
            "Polymerasekettenreaktion-Produkte",
        ]
