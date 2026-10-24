"""Real-data tests for the Natural Stories loader.

Run with: pytest tests/dataset/spr/test_natural_stories.py -v --data-root <path>
"""

import os

import pytest

from replica.dataset import NaturalStories

pytestmark = pytest.mark.needs_data


class TestNaturalStoriesRealData:
    @pytest.fixture(autouse=True)
    def load_dataset(self, data_root):
        path = os.path.join(
            data_root, "self-paced-reading", "original", "natural_stories"
        )
        if not os.path.isdir(path):
            pytest.skip(f"Natural Stories data not found at {path}")
        self.ds = NaturalStories()
        try:
            self.ds.load(path)
        except FileNotFoundError as e:
            pytest.skip(f"Natural Stories data not loadable: {e}")

    def test_stimulus_count(self):
        """Natural Stories has 10 stimuli (stories)."""
        assert len(self.ds.stimuli()) == 10

    def test_participant_count(self):
        """Natural Stories has 180 participants."""
        assert len(self.ds.participants) == 180

    def test_measures(self):
        assert self.ds.measures == ["spr_rt"]

    def test_first_stimulus_words(self):
        """Verify first stimulus (ns_1) begins with expected words."""
        stim = next(s for s in self.ds.stimuli() if s.stimulus_id == "ns_1")
        words = stim.layers["words"].values[:5]
        assert words == ["If", "you", "were", "to", "journey"]

    def test_first_stimulus_word_count(self):
        """First story (ns_1) has 1073 words."""
        stim = next(s for s in self.ds.stimuli() if s.stimulus_id == "ns_1")
        assert len(stim.layers["words"]) == 1073
