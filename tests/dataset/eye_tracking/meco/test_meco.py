"""Real-data tests for the MECO L1 loader (English + German).

Run with: pytest tests/dataset/eye_tracking/meco/test_meco.py -v --data-root <path>
"""

import os

import pytest

from replica.dataset.eye_tracking.meco.l1 import MECO_L1
from tests._require import load_or_skip
from tests.dataset._measures import FLAGGED_MEASURES, assert_eye_tracking_measures

pytestmark = pytest.mark.needs_data


class TestMECOL1English:
    @pytest.fixture(autouse=True)
    def load_dataset(self, data_root):
        path = os.path.join(data_root, "eye-tracking", "original", "meco")
        self.ds = load_or_skip(MECO_L1(language="en"), path)

    def test_participant_count(self):
        """MECO L1 English has 46 participants."""
        assert len(self.ds.participants) == 46

    def test_measures(self):
        assert_eye_tracking_measures(
            self.ds,
            [
                *FLAGGED_MEASURES,
                "go_past_in_first_pass",
            ],
        )

    def test_first_stimulus_words(self):
        """Verify first stimulus (trial 1) begins with expected words."""
        stim = sorted(self.ds.stimuli(), key=lambda s: s.metadata["text_id"])[0]
        words = stim.layers["words"].values[:5]
        assert words == ["In", "ancient", "Roman", "religion", "and"]

    def test_language_metadata(self):
        """Each stimulus should have language='en' in its metadata."""
        for stim in self.ds.stimuli():
            assert stim.metadata["language"] == "en"


class TestMECOL1German:
    @pytest.fixture(autouse=True)
    def load_dataset(self, data_root):
        path = os.path.join(data_root, "eye-tracking", "original", "meco")
        self.ds = load_or_skip(MECO_L1(language="de"), path)

    def test_language_metadata(self):
        """Each stimulus should have language='de' in its metadata."""
        for stim in self.ds.stimuli():
            assert stim.metadata["language"] == "de"


class TestMECOL1RussianItemIdentity:
    """Russian source item IDs are 2..13 while trial IDs are 1..12.

    trialid is presentation order, so passage identity comes from the source
    item ID.
    """

    def test_source_items_are_normalized_without_mixing_passages(self, data_root):
        path = os.path.join(data_root, "eye-tracking", "original", "meco")
        ds = load_or_skip(MECO_L1(language="ru"), path)
        stimuli = sorted(ds.stimuli(), key=lambda s: s.metadata["text_id"])
        assert len(stimuli) == 12
        assert [s.metadata["source_item_id"] for s in stimuli] == list(range(2, 14))
        assert stimuli[0].layers["words"].values[0] == "Янус"
