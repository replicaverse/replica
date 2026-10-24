"""Real-data tests for the PROVO loader.

Run with: pytest tests/dataset/eye_tracking/test_provo.py -v --data-root <path>
"""

import os

import pytest

from replica.dataset import PROVO
from tests.dataset._measures import assert_eye_tracking_measures

pytestmark = pytest.mark.needs_data


class TestPROVORealData:
    @pytest.fixture(autouse=True)
    def load_dataset(self, data_root):
        path = os.path.join(data_root, "eye-tracking", "original", "provo")
        if not os.path.isdir(path):
            pytest.skip(f"PROVO data not found at {path}")
        self.ds = PROVO()
        try:
            self.ds.load(path)
        except FileNotFoundError as e:
            pytest.skip(f"PROVO data not loadable: {e}")

    def test_stimulus_count(self):
        """PROVO has 55 stimuli (passages)."""
        assert len(self.ds.stimuli()) == 55

    def test_participant_count(self):
        """PROVO has 84 participants."""
        assert len(self.ds.participants) == 84

    def test_measures(self):
        # ``first_run`` is IA_FIRST_RUN_DWELL_TIME, ``first_pass`` inverts IA_SKIP,
        # ``total_time`` is total dwell (IA_DWELL_TIME).
        assert_eye_tracking_measures(self.ds)

    def test_first_stimulus_words(self):
        """Verify first stimulus begins with expected words."""
        stim = sorted(self.ds.stimuli(), key=lambda s: s.metadata["text_id"])[0]
        words = stim.layers["words"].values[:5]
        assert words == ["There", "are", "now", "rumblings", "that"]

    def test_passages_decode_as_mac_roman(self):
        """Decode the corpus's 0xd5 and 0xd1 bytes as punctuation."""
        by_id = {
            s.metadata["text_id"]: s.layers["words"].values for s in self.ds.stimuli()
        }
        assert not [
            w for words in by_id.values() for w in words if "Õ" in w or "Ñ" in w
        ]
        assert "doesn’t" in by_id[27]
        assert "—" in by_id[36]
        assert "women’s" in by_id[45]
        assert "bonds’" in by_id[54]
