"""Real-data tests for the RSC loader.

Run with: pytest tests/dataset/eye_tracking/test_rsc.py -v --data-root <path>
"""

import csv
import os
from collections import Counter

import numpy as np
import pytest

from replica.dataset import RSC, describe_dataset
from tests.dataset._measures import assert_eye_tracking_measures


def _duplicated_rows(data_root, filename="data.csv"):
    """(participant, sentence, position) keys the release records twice."""
    path = os.path.join(data_root, "eye-tracking", "original", "rsc", filename)
    with open(path, encoding="utf-8", newline="") as f:
        seen = Counter(
            (row["DATA_FILE"], row["item.id"], row["word.serial.no"])
            for row in csv.DictReader(f, delimiter="\t")
        )
    return [key for key, count in seen.items() if count > 1]


def _load(data_root, release="published"):
    path = os.path.join(data_root, "eye-tracking", "original", "rsc")
    if not os.path.isdir(path):
        pytest.skip(f"RSC data not found at {path}")
    ds = RSC(release=release)
    try:
        ds.load(path)
    except FileNotFoundError as e:
        pytest.skip(f"RSC data not loadable: {e}")
    return ds


@pytest.mark.needs_data
class TestRSCPublished:
    @pytest.fixture(autouse=True)
    def load_dataset(self, data_root):
        self.ds = _load(data_root)

    def test_sentence_level_stimuli(self):
        """144 isolated sentences, one Stimulus each, 5-12 words per the
        corpus design (sentences built around a single target word)."""
        stims = self.ds.stimuli()
        assert len(stims) == 144
        lens = [len(s.layers["words"]) for s in stims]
        assert min(lens) == 5 and max(lens) == 12

    def test_participant_count(self):
        """The published release (``data.csv``) has 96 participants."""
        assert len(self.ds.participants) == 96

    def test_measures(self):
        assert_eye_tracking_measures(self.ds)

    def test_stimulus_text_is_the_released_sentence(self):
        """Stimuli keep the corpus's own text verbatim, including the
        low-9 quotation mark (U+201A) the release uses for commas."""
        stim = next(s for s in self.ds.stimuli() if s.stimulus_id == "rsc_s1")
        assert stim.text.startswith("На болотах оставался еще лед‚ но")
        assert stim.layers["words"].values[:3] == ["На", "болотах", "оставался"]

    def test_sentence_initial_and_final_words_are_nan(self):
        """``data.csv`` drops the first and last word of every sentence, so
        those positions carry no measure for any participant."""
        by_stim = {}
        for datum in self.ds.data(measure="first_run"):
            by_stim.setdefault(datum.stimulus.stimulus_id, []).append(
                np.asarray(datum.values, float)
            )
        for values in by_stim.values():
            stacked = np.vstack(values)
            assert np.isnan(stacked[:, 0]).all()
            assert np.isnan(stacked[:, -1]).all()

    def test_first_run_and_first_pass_run_are_plausible(self):
        first_run = np.concatenate(
            [np.asarray(d.values, float) for d in self.ds.data(measure="first_run")]
        )
        observed = first_run[np.isfinite(first_run)]
        assert observed.size == 72972
        assert 200 < observed.mean() < 350
        # Words the release flags as skipped in first pass leave the
        # first-pass restriction.
        gaze = np.concatenate(
            [
                np.asarray(d.values, float)
                for d in self.ds.data(measure="first_run_in_first_pass")
            ]
        )
        assert np.isfinite(gaze).sum() == 67530
        flags = np.concatenate(
            [np.asarray(d.values, float) for d in self.ds.data(measure="first_pass")]
        )
        assert set(np.unique(flags[np.isfinite(flags)])) <= {0.0, 1.0}

    def test_duplicate_trials_keep_their_first_occurrence(self, data_root):
        """Some (participant, sentence) trials appear twice in ``data.csv``,
        sometimes disagreeing about the same word. One Datum per
        (participant, stimulus, measure) survives regardless."""
        assert _duplicated_rows(data_root), (
            "no duplicated rows in the release, so the dedup assertion "
            "below would pass vacuously"
        )
        keys = [
            (d.participant_id, d.stimulus.stimulus_id)
            for d in self.ds.data(measure="first_run")
        ]
        assert len(keys) == len(set(keys))


@pytest.mark.needs_data
class TestRSCExtended:
    @pytest.fixture(autouse=True)
    def load_dataset(self, data_root):
        self.ds = _load(data_root, release="extended")

    def test_participant_count(self):
        """``data_103.csv`` holds 102 participants despite its filename."""
        assert len(self.ds.participants) == 102

    def test_sentence_initial_word_is_measured(self):
        """Unlike the published release, the extended one keeps the
        sentence-initial word, so position 0 is populated."""
        values = np.vstack(
            [
                np.asarray(d.values, float)
                for d in self.ds.data(measure="first_run")
                if d.stimulus.stimulus_id == "rsc_s1"
            ]
        )
        assert np.isfinite(values[:, 0]).any()

    def test_same_stimuli_as_published_release(self):
        assert len(self.ds.stimuli()) == 144


def test_unknown_release_rejected():
    with pytest.raises(ValueError, match="Unknown RSC release"):
        RSC(release="data_103")


def test_both_releases_resolve_to_the_shared_download():
    """The extended RSC entry resolves the shared download alias."""
    assert describe_dataset("rsc").download_keys == ("rsc",)
    assert describe_dataset("rsc_extended").download_keys == ("rsc",)


@pytest.mark.needs_data
def test_one_release_loads_without_the_other(data_root, tmp_path, monkeypatch):
    """Checksum verification must demand only the file the loader reads,
    even though the YAML declares both releases."""
    monkeypatch.delenv("PSYCHO_SKIP_CHECKSUMS", raising=False)
    source = os.path.join(data_root, "eye-tracking", "original", "rsc", "data.csv")
    if not os.path.isfile(source):
        pytest.skip(f"RSC data.csv not found at {source}")

    solo = tmp_path / "eye-tracking" / "original" / "rsc"
    solo.mkdir(parents=True)
    os.symlink(source, solo / "data.csv")

    published = RSC(release="published")
    published.load(str(solo))
    assert len(published.stimuli()) == 144

    with pytest.raises(FileNotFoundError, match="data_103.csv"):
        RSC(release="extended").load(str(solo))
