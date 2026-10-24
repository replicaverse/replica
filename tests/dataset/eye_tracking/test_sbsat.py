"""Real-data tests for the SB-SAT loader.

Run with: pytest tests/dataset/eye_tracking/test_sbsat.py -v --data-root <path>
"""

import csv
import os
from collections import defaultdict

import numpy as np
import pytest

from replica.context import CONTEXT_SEQUENCE, attach_preceding_context, context_input
from replica.dataset import SBSAT
from replica.dataset.measures import with_derived

pytestmark = pytest.mark.needs_data


class TestSBSATRealData:
    @pytest.fixture(autouse=True)
    def load_dataset(self, data_root):
        path = os.path.join(data_root, "eye-tracking", "original", "sbsat")
        self.path = path
        if not os.path.isdir(path):
            pytest.skip(f"SB-SAT data not found at {path}")
        self.ds = SBSAT()
        try:
            self.ds.load(path)
        except FileNotFoundError as e:
            pytest.skip(f"SB-SAT data not loadable: {e}")

    def test_stimulus_count(self):
        """SB-SAT has 22 stimuli (book pages)."""
        assert len(self.ds.stimuli()) == 22

    def test_participant_count(self):
        """SB-SAT has 95 participants."""
        assert len(self.ds.participants) == 95

    def test_measures(self):
        # The first run and the first-pass flag are computed from the stream.
        assert set(self.ds.measures) == with_derived(
            [
                "first_fixation",
                "first_pass",
                "first_run",
                "total_time",
            ]
        )

    def test_first_stimulus_words(self):
        """Verify first stimulus (sbsat_dickens_1) begins with expected words."""
        stim = next(s for s in self.ds.stimuli() if s.stimulus_id == "sbsat_dickens_1")
        words = stim.layers["words"].values[:5]
        assert words == ["Even", "then", "my", "only", "friends"]

    def test_stimulus_ids_start_with_sbsat(self):
        for stim in self.ds.stimuli():
            assert stim.stimulus_id.startswith("sbsat_"), (
                f"Stimulus ID {stim.stimulus_id} does not start with 'sbsat_'"
            )

    def test_source_first_visits_follow_story_order(self):
        """A page's preceding text was presented before its first visit."""
        visits = defaultdict(dict)
        with open(
            os.path.join(self.path, "18sat_fixfinal.csv"), newline="", encoding="utf-8"
        ) as f:
            for row in csv.DictReader(f):
                if row["type"] != "reading":
                    continue
                key = (row["RECORDING_SESSION_LABEL"], row["book_name"])
                visits[key][int(row["TRIAL_INDEX"])] = int(row["page"])
        assert visits
        for key, trials in visits.items():
            first = list(dict.fromkeys(page for _, page in sorted(trials.items())))
            assert first == list(range(1, max(first) + 1)), key

    def test_story_context_preserves_pages_and_measurements(self):
        pages = self.ds.stimuli()
        texts = {page.stimulus_id: page.text for page in pages}
        data_before = [d.values.copy() for d in self.ds.data()]
        attach_preceding_context(pages)
        for page in pages:
            assert page.metadata[CONTEXT_SEQUENCE] == {
                "group": page.metadata["book_name"],
                "position": page.metadata["page"],
            }
            preceding = sorted(
                [
                    p
                    for p in pages
                    if p.metadata["book_name"] == page.metadata["book_name"]
                    and p.metadata["page"] <= page.metadata["page"]
                ],
                key=lambda p: p.metadata["page"],
            )
            combined, start = context_input(page)
            assert combined.text == " ".join(texts[p.stimulus_id] for p in preceding)
            assert combined.unit_strings()[start:] == page.unit_strings()
            assert page.text == texts[page.stimulus_id]
        for before, datum in zip(data_before, self.ds.data(), strict=True):
            np.testing.assert_array_equal(before, datum.values)
