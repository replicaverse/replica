"""Real-data tests for the CopCo loader.

Run with: pytest tests/dataset/eye_tracking/test_copco.py -v --data-root <path>
"""

import os

import numpy as np
import pytest

from replica.dataset import CopCo
from tests.dataset._measures import FLAGGED_MEASURES, assert_eye_tracking_measures

pytestmark = pytest.mark.needs_data


class TestCopCoRealData:
    @pytest.fixture(autouse=True)
    def load_dataset(self, data_root):
        path = os.path.join(data_root, "eye-tracking", "original", "copco")
        if not os.path.isdir(path):
            pytest.skip(f"CopCo data not found at {path}")
        self.ds = CopCo()
        try:
            self.ds.load(path)
        except FileNotFoundError as e:
            pytest.skip(f"CopCo data not loadable: {e}")

    def test_paragraph_level_stimuli(self):
        """One paragraph per screen is the SPR unit. Each paragraph is
        keyed by ``(speechId, paragraphId)``, ``paragraphId`` is local
        to each speech, so it must be combined with ``speechId``. The
        ExtractedFeatures release contains 452 distinct paragraphs."""
        stims = self.ds.stimuli()
        assert len(stims) == 452, (
            f"Expected 452 paragraph-level stimuli, got {len(stims)}"
        )
        lens = [len(s.layers["words"]) for s in stims]
        assert max(lens) <= 200, f"unexpectedly long paragraph: max={max(lens)}"
        mean_len = sum(lens) / len(lens)
        assert 50 < mean_len < 100, (
            f"mean paragraph length out of range: {mean_len:.1f}"
        )

    def test_participant_count(self):
        """The release carries 57 feature files; only P14 is absent."""
        assert len(self.ds.participants) == 57

    def test_measures(self):
        assert_eye_tracking_measures(self.ds, FLAGGED_MEASURES)

    def test_first_stimulus_words(self):
        """First paragraph (by speech_id then paragraph_id) is
        ``copco_s1125_p0``; verify its opening Danish words."""
        stim = self.ds.stimuli()[0]
        assert stim.stimulus_id == "copco_s1125_p0"
        words = stim.layers["words"].values[:5]
        assert words == ["Tak", "for", "invitationen", "til", "at"]


class TestMultiWordInterestAreas:
    """The release joins some interest areas with a no-break space."""

    @pytest.fixture(autouse=True)
    def load_dataset(self, data_root):
        path = os.path.join(data_root, "eye-tracking", "original", "copco")
        if not os.path.isdir(path):
            pytest.skip(f"CopCo data not found at {path}")
        self.ds = CopCo()
        self.ds.load(path)
        self.stim = next(
            s for s in self.ds.stimuli() if s.stimulus_id == "copco_s202201_p0"
        )
        self.words = self.stim.layers["words"].extract_strings(self.stim.text)

    def test_regions_have_natural_text(self):
        for stim in self.ds.stimuli():
            for word in stim.layers["words"].extract_strings(stim.text):
                assert "\xa0" not in word and "_" not in word, (stim.stimulus_id, word)

    def test_joined_area_is_preserved(self):
        assert self.words[:4] == ["Olof", "Palme (1927-1986)", "var", "en"]

    def test_region_keeps_the_released_measure(self):
        # P42.csv, speech 202201, paragraph 0, sentence 1470, word 1.
        datum = next(
            d
            for d in self.ds.data(measure="total_time", participant="P42")
            if d.stimulus is self.stim
        )
        assert datum.values[self.words.index("Palme (1927-1986)")] == 275

    def test_fixations_reach_the_source_region(self):
        landed = {
            f.unit_index
            for f in self.ds.fixations()
            if f.stimulus_id == self.stim.stimulus_id
        }
        assert self.words.index("Palme (1927-1986)") in landed


class TestDerivedFirstPass:
    """The first-pass flag follows from the fixation order and restricts the
    released first run to the classical gaze duration."""

    @pytest.fixture(autouse=True)
    def load_dataset(self, data_root):
        path = os.path.join(data_root, "eye-tracking", "original", "copco")
        if not os.path.isdir(path):
            pytest.skip(f"CopCo data not found at {path}")
        self.ds = CopCo()
        self.ds.load(path)
        self.ds.drop_skips = False

    def test_flag_places_each_fixated_word_in_or_out_of_first_pass(self):
        first_run = {
            (d.participant_id, d.stimulus.stimulus_id): d.values
            for d in self.ds.data(measure="first_run")
        }
        restricted = {
            (d.participant_id, d.stimulus.stimulus_id): d.values
            for d in self.ds.data(measure="first_run_in_first_pass")
        }
        reproduced = unavailable = skips = kept = 0
        flags = [d for d in self.ds.data() if d.measure_name == "first_pass"]
        assert flags
        for datum in flags:
            key = (datum.participant_id, datum.stimulus.stimulus_id)
            released, gaze = first_run[key], restricted[key]
            if not datum.metadata["first_run_reproduced"]:
                unavailable += 1
                assert np.isnan(datum.values).all()
                assert np.isnan(gaze).all()
                continue
            reproduced += 1
            fixated = np.isfinite(released) & (released > 0)
            flag = datum.values[fixated]
            assert set(np.unique(flag)) <= {0.0, 1.0}
            np.testing.assert_array_equal(
                gaze[fixated], np.where(flag == 1, released[fixated], 0)
            )
            skips += int((flag == 0).sum())
            kept += int((flag == 1).sum())
            # A word nobody fixated is a first-pass skip; an unreleased cell
            # leaves the flag unknown.
            np.testing.assert_array_equal(
                datum.values[~fixated],
                np.where(np.isfinite(released[~fixated]), 0, np.nan),
            )
        # Nearly every trial's rebuilt stream reproduces the released first run,
        # and a substantial share of fixated words were first entered by a
        # regression.
        assert unavailable <= 0.01 * (reproduced + unavailable)
        assert skips > 0.2 * kept
