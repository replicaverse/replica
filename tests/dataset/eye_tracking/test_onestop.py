"""Real-data tests for the OneStop loader.

Run with: pytest tests/dataset/eye_tracking/test_onestop.py -v --data-root <path>
"""

import csv
import os

import pytest

from replica.dataset import OneStop
from tests.dataset._measures import assert_eye_tracking_measures

pytestmark = pytest.mark.needs_data


class TestOneStopRealData:
    @pytest.fixture(autouse=True)
    def load_dataset(self, data_root):
        path = os.path.join(data_root, "eye-tracking", "original", "onestop")
        if not os.path.isdir(path):
            pytest.skip(f"OneStop data not found at {path}")
        self.ds = OneStop()
        try:
            self.ds.load(path)
        except FileNotFoundError as e:
            pytest.skip(f"OneStop data not loadable: {e}")

    def test_stimulus_count(self):
        """324 experimental paragraphs, with 15 extra IA sequence variants."""
        stimuli = self.ds.stimuli()
        assert len(stimuli) == 339
        keys = {
            (
                s.metadata["article_batch"],
                s.metadata["article_id"],
                s.metadata["difficulty_level"],
                s.metadata["paragraph_id"],
            )
            for s in stimuli
        }
        assert len(keys) == 324
        assert all(s.metadata["article_id"] != 0 for s in stimuli)

    def test_participant_count(self):
        """OneStop has 180 participants."""
        assert len(self.ds.participants) == 180

    def test_measures(self):
        # ``first_run`` is IA_FIRST_RUN_DWELL_TIME, ``first_pass`` inverts IA_SKIP,
        # ``total_time`` is total dwell (IA_DWELL_TIME).
        assert_eye_tracking_measures(self.ds)

    def test_first_stimulus_words(self):
        """Anchor on an experimental paragraph, excluding practice article 0."""
        stim = next(
            s for s in self.ds.stimuli() if s.stimulus_id == "onestop_b1_a1_Adv_p1"
        )
        assert stim.layers["words"].values[:5] == ["Unveiling", "a", "car", "with", "a"]

    def test_stimulus_metadata_has_paragraph_key(self):
        """Each stimulus should carry the full 4-tuple paragraph key."""
        for stim in self.ds.stimuli():
            for k in (
                "article_batch",
                "article_id",
                "difficulty_level",
                "paragraph_id",
            ):
                assert k in stim.metadata, f"Stimulus {stim.stimulus_id} missing {k}"

    def test_word_labels_consistent_across_participants(self, data_root):
        """Every retained source row must align to its reader's exact IA label."""
        lookup = self.ds._fixation_stimulus_lookup()
        path = os.path.join(
            data_root, "eye-tracking", "original", "onestop", "ia_Paragraph.csv"
        )
        compared = 0
        with open(path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if any(
                    row[flag] != "False"
                    for flag in (
                        "question_preview",
                        "repeated_reading_trial",
                        "practice_trial",
                    )
                ):
                    continue
                key = (
                    int(row["article_batch"]),
                    int(row["article_id"]),
                    row["difficulty_level"],
                    int(row["paragraph_id"]),
                )
                stim, ia_to_idx, _ = lookup[(key, row["participant_id"])]
                assert (
                    stim.layers["words"].values[ia_to_idx[int(row["IA_ID"])]]
                    == row["IA_LABEL"]
                )
                compared += 1
        assert compared == 1055203
