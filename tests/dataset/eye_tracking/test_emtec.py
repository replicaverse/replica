"""Unit tests for EMTeC loader.

Synthetic-data unit tests + real-data invariants. The latter require
``--data-root`` and an actual EMTeC download.
"""

import csv
import os
import tempfile

import numpy as np
import pytest

from replica.dataset.eye_tracking.emtec import EMTeC
from replica.dataset.measures import FIRST_PASS_RUN, with_derived


@pytest.fixture
def emtec_dir():
    """Create a temporary directory with synthetic EMTeC data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "reading_measures.csv"), "w") as f:
            # Minimal columns needed (tab-separated like real data)
            cols = [
                "subject_id",
                "item_id",
                "model",
                "decoding_strategy",
                "type",
                "task",
                "word_id",
                "word",
                "FD",
                "FRT",
                "FFD",
                "FPRT",
                "TFT",
                "RPD_inc",
            ]
            f.write("\t".join(cols) + "\n")

            def _row(*vals):
                vals = (*vals[:8], vals[8], vals[9], *vals[8:])
                f.write("\t".join(str(v) for v in vals) + "\n")

            # Item 1, mistral, greedy_search, 4 words
            words = ["The", "cat", "sat", "down"]
            # Subject S01
            for i, w in enumerate(words):
                ffd = [200, 0, 180, 250][i]  # word "cat" not fixated
                fprt = [200, 0, 180, 250][i]
                tft = [300, 0, 250, 400][i]
                rpd = [200, 0, 200, 300][i]
                _row(
                    "S01",
                    1,
                    "mistral",
                    "greedy_search",
                    "constrained",
                    "fiction",
                    i,
                    w,
                    ffd,
                    fprt,
                    tft,
                    rpd,
                )
            # Subject S02
            for i, w in enumerate(words):
                ffd = [180, 150, 0, 220][i]  # word "sat" not fixated
                fprt = [180, 150, 0, 220][i]
                tft = [280, 200, 0, 350][i]
                rpd = [180, 150, 0, 280][i]
                _row(
                    "S02",
                    1,
                    "mistral",
                    "greedy_search",
                    "constrained",
                    "fiction",
                    i,
                    w,
                    ffd,
                    fprt,
                    tft,
                    rpd,
                )

            # Item 1, phi2, sampling, same 4 words (different stimulus)
            for i, w in enumerate(["The", "dog", "ran", "fast"]):
                _row(
                    "S01",
                    1,
                    "phi2",
                    "sampling",
                    "unconstrained",
                    "non-fiction",
                    i,
                    w,
                    200,
                    200,
                    300,
                    200,
                )

        yield tmpdir


class TestEMTeC:
    def test_load(self, emtec_dir):
        ds = EMTeC(measurement_source="released")
        ds.load(emtec_dir)

        # Two distinct stimuli: (1, mistral, greedy_search) and (1, phi2, sampling)
        assert len(ds.stimuli()) == 2
        assert ds.info.material_namespace == "emtec"
        assert [s.metadata["cv_group"] for s in ds.stimuli()] == ["emtec:1", "emtec:1"]

    def test_stimulus_id_format(self, emtec_dir):
        ds = EMTeC(measurement_source="released")
        ds.load(emtec_dir)

        ids = sorted([s.stimulus_id for s in ds.stimuli()])
        assert "emtec_1_mistral_greedy_search" in ids
        assert "emtec_1_phi2_sampling" in ids

    def test_measures(self, emtec_dir):
        ds = EMTeC(measurement_source="released")
        ds.load(emtec_dir)
        # FD retains first arrival; FFD/FPRT retain the first-pass windows.
        # The catalog also advertises the composable restricted variables.
        assert sorted(ds.measures) == sorted(
            with_derived(
                [
                    "first_fixation",
                    "first_pass",
                    "first_run",
                    "go_past",
                    "go_past_in_first_pass",
                    "total_time",
                ]
            )
        )

    def test_zero_to_nan(self, emtec_dir):
        """0 values should become NaN."""
        ds = EMTeC(measurement_source="released")
        ds.load(emtec_dir)

        # S01 on (1, mistral, greedy_search): word "cat" (idx 1) not fixated
        ff_data = ds.data(measure="first_fixation", participant="S01")
        s1_data = [
            d
            for d in ff_data
            if d.stimulus.stimulus_id == "emtec_1_mistral_greedy_search"
        ]
        assert len(s1_data) == 1
        datum = s1_data[0]
        assert np.isnan(datum.values[1]), "Zero should be converted to NaN"
        assert datum.values[0] == 200  # "The" fixated

    def test_no_zero_in_any_measure(self, emtec_dir):
        """Verify no zero values remain in any eye-tracking duration."""
        ds = EMTeC(measurement_source="released")
        ds.load(emtec_dir)

        for datum in ds.data():
            if datum.measure_name == "first_pass":
                continue  # the flag is 0 on every skip
            non_nan = datum.values[~np.isnan(datum.values)]
            assert not np.any(non_nan == 0), (
                f"Found zero value in {datum.measure_name} for "
                f"participant {datum.participant_id}"
            )

    def test_word_alignment(self, emtec_dir):
        ds = EMTeC(measurement_source="released")
        ds.load(emtec_dir)

        # Word order must follow word_index.
        stim = next(
            s for s in ds.stimuli() if s.stimulus_id == "emtec_1_mistral_greedy_search"
        )
        assert stim.unit_strings("words") == ["The", "cat", "sat", "down"]

    def test_participants(self, emtec_dir):
        ds = EMTeC(measurement_source="released")
        ds.load(emtec_dir)
        # S01 appears in both stimuli, S02 only in first
        assert "S01" in ds.participants
        assert "S02" in ds.participants

    def test_stimulus_metadata(self, emtec_dir):
        ds = EMTeC(measurement_source="released")
        ds.load(emtec_dir)

        stim = next(
            s for s in ds.stimuli() if s.stimulus_id == "emtec_1_mistral_greedy_search"
        )
        assert stim.metadata["model"] == "mistral"
        assert stim.metadata["decoding_strategy"] == "greedy_search"
        assert stim.metadata["type"] == "constrained"
        assert stim.metadata["task"] == "fiction"

    def test_first_pass_flag_and_first_run(self, emtec_dir):
        """FPRT>0 gives the first run with the flag at 1; a never-fixated
        word has a first run of 0 (a skip) with the flag at 0."""
        ds = EMTeC(measurement_source="released")
        ds.load(emtec_dir)
        ds.drop_skips = False

        def s01(measure):
            return next(
                d
                for d in ds.data(participant="S01")
                if d.stimulus.stimulus_id == "emtec_1_mistral_greedy_search"
                and d.measure_name == measure
            ).values

        def s01_gaze():
            return next(
                d
                for d in ds.data(measure=FIRST_PASS_RUN, participant="S01")
                if d.stimulus.stimulus_id == "emtec_1_mistral_greedy_search"
            ).values

        np.testing.assert_equal(s01("first_run"), [200, 0, 180, 250])
        np.testing.assert_equal(s01("first_pass"), [1, 0, 1, 1])
        np.testing.assert_equal(s01_gaze(), [200, 0, 180, 250])
        ds.drop_skips = True
        np.testing.assert_equal(s01_gaze(), [200, np.nan, 180, 250])
        np.testing.assert_equal(s01("first_pass"), [1, 0, 1, 1])

    def test_loader_attributes(self):
        ds = EMTeC(measurement_source="released")
        assert ds.default_measure == FIRST_PASS_RUN
        assert ds.info.name == "emtec"

    def test_covariates_compute(self, emtec_dir):
        ds = EMTeC(measurement_source="released")
        ds.load(emtec_dir)
        stim = ds.stimuli()[0]
        covs = ds.covariates(stim)
        assert "unit_length" in covs
        assert "prev_word_fixated" in covs

    def test_word_id_zero_indexed(self, emtec_dir):
        """word_id in EMTeC is 0-indexed; no conversion needed."""
        ds = EMTeC(measurement_source="released")
        ds.load(emtec_dir)

        # First word should be at index 0
        ff_data = ds.data(measure="first_fixation", participant="S01")
        stim_data = [
            d
            for d in ff_data
            if d.stimulus.stimulus_id == "emtec_1_mistral_greedy_search"
        ]
        datum = stim_data[0]
        # "The" at index 0 should have value 200
        assert datum.values[0] == 200


@pytest.mark.needs_data
class TestEMTeCRealData:
    """Invariants asserted against the real EMTeC release.

    Runs only when ``--data-root`` points at a directory containing
    EMTeC under ``eye-tracking/original/emtec``.
    """

    @pytest.fixture(autouse=True)
    def load_dataset(self, data_root):
        path = os.path.join(data_root, "eye-tracking", "original", "emtec")
        if not os.path.isdir(path):
            pytest.skip(f"EMTeC data not found at {path}")
        self.ds = EMTeC(measurement_source="released")
        self.ds.load(path)

    def test_participant_count(self):
        """EMTeC has ~106-107 participants."""
        n = len(self.ds.participants)
        assert 100 <= n <= 120, f"Expected ~107 participants, got {n}"

    def test_stimulus_count(self):
        """42 items x 3 models x 5 strategies = up to 630 stimuli."""
        n = len(self.ds.stimuli())
        assert 500 <= n <= 700, f"Expected ~588-630 stimuli, got {n}"

    def test_no_zero_values(self):
        """Every zero duration reads as missing under the default view."""
        for datum in self.ds.data():
            if datum.measure_name == "first_pass":
                continue  # the flag is 0 on every skip
            non_nan = datum.values[~np.isnan(datum.values)]
            assert not np.any(non_nan == 0), (
                f"Found zero in {datum.measure_name} for {datum.participant_id} "
                f"on stimulus {datum.stimulus.stimulus_id}"
            )

    def test_stimulus_metadata(self):
        """Each stimulus should declare model and decoding_strategy."""
        for stim in self.ds.stimuli():
            assert stim.metadata["model"] in {"mistral", "phi2", "wizardlm"}
            assert stim.metadata["decoding_strategy"] in {
                "beam_search",
                "greedy_search",
                "sampling",
                "topk",
                "topp",
            }


def _write_fixations(directory, word_ids):
    with open(os.path.join(directory, "fixations.csv"), "w", newline="") as stream:
        rows = [
            {
                "subject_id": "S01",
                "item_id": 1,
                "model": "mistral",
                "decoding_strategy": "greedy_search",
                "TRIAL_ID": 1,
                "fixation_index": index,
                "fixation_duration": duration,
                "word_id": word_id,
            }
            for index, (word_id, duration) in enumerate(
                zip(word_ids, [100, 50, 80, 120], strict=True), 1
            )
        ]
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def test_off_text_fixations_preserve_the_timeline(emtec_dir):
    _write_fixations(emtec_dir, [0, ".", 0, 2])
    ds = EMTeC(measurement_source="released")
    ds.load(emtec_dir)
    events = ds.fixations()
    assert [event.unit_index for event in events] == [0, None, 0, 2]
    assert [event.fixation_index for event in events] == [1, 2, 3, 4]
    assert [event.start_time for event in events] == [0, 100, 150, 230]
    assert [event.end_time for event in events] == [100, 150, 230, 350]
    assert events[1].char_offset is None
    assert "run_id" not in events[1].metadata
    assert events[0].metadata["run_id"] == 1
    assert events[2].metadata["run_id"] == 1  # source measures ignore off-word events


def test_unknown_fixation_word_is_not_silently_dropped(emtec_dir):
    _write_fixations(emtec_dir, [0, 999, 0, 2])
    with pytest.raises(ValueError, match="word_id 999 is absent"):
        EMTeC(measurement_source="released").load(emtec_dir)
