"""Unit tests for UCL SPR and ET loaders.

Synthetic-data unit tests + real-data invariants. The latter require
``--data-root`` and a real UCL download (paper-reported counts and
raw-file alignment).
"""

import csv
import os
import tempfile

import numpy as np
import pytest

from replica.core import CONDITIONS
from replica.dataset.eye_tracking.ucl import UCL_ET, _parse_ucl_stimuli
from replica.dataset.measures import FIRST_PASS_RUN
from replica.dataset.spr.ucl import UCL_SPR

_FIRST_STIMULUS_WORDS = ["Anne", "lost", "control", "and", "laughed."]


def _assert_first_stimulus_words(dataset) -> None:
    stimulus = next(stim for stim in dataset.stimuli() if stim.metadata["sent_nr"] == 1)
    assert stimulus.unit_strings("words") == _FIRST_STIMULUS_WORDS


@pytest.mark.parametrize(
    "cls, default_measure, info_name",
    [
        (UCL_SPR, "spr_rt", "ucl_spr"),
        (UCL_ET, FIRST_PASS_RUN, "ucl_et"),
    ],
)
def test_loader_attributes(cls, default_measure, info_name):
    """Hardcoded `.default_measure` and `.info.name` on UCL loaders."""
    ds = cls()
    assert ds.default_measure == default_measure
    assert ds.info.name == info_name


@pytest.fixture
def ucl_dir():
    """Create a temporary directory with synthetic UCL data files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write stimuli.txt (3 sentences)
        # NOTE: sent 2 deliberately has "Billy" while the RT file has "Bily"
        # to test that UCL_SPR uses the RT file's word column (what was displayed)
        with open(os.path.join(tmpdir, "stimuli.txt"), "w") as f:
            f.write("sent_nr\tsentence\tquestion\tanswer\n")
            f.write("1\tAnne lost control and laughed.\t-\t-\n")
            f.write("2\tBilly wrote on the envelope.\tWas it Bob who wrote?\tn\n")
            f.write("3\tHe called over his shoulder.\t-\t-\n")

        # Write selfpacedreading.RT.txt (2 participants, sentences 1 and 2)
        with open(os.path.join(tmpdir, "selfpacedreading.RT.txt"), "w") as f:
            f.write(
                "subj_nr\tsent_nr\tsent_pos\tcorrect\tanswer_time\tword_pos\tword\tRT\n"
            )
            # Participant 1, sentence 1: "Anne lost control and laughed."
            for i, (w, rt) in enumerate(
                [
                    ("Anne", 350),
                    ("lost", 280),
                    ("control", 320),
                    ("and", 200),
                    ("laughed.", 310),
                ],
                start=1,
            ):
                f.write(f"1\t1\t5\t-\tNaN\t{i}\t{w}\t{rt}\n")
            # Participant 1, sentence 2: RT file has "Bily" (typo), stimuli.txt has "Billy"
            # This tests that UCL_SPR takes its words from the RT file
            for i, (w, rt) in enumerate(
                [
                    ("Bily", 376),
                    ("wrote", 364),
                    ("on", 394),
                    ("the", 353),
                    ("envelope.", 354),
                ],
                start=1,
            ):
                f.write(f"1\t2\t12\tc\t3630\t{i}\t{w}\t{rt}\n")
            # Participant 2, sentence 1
            for i, (w, rt) in enumerate(
                [
                    ("Anne", 400),
                    ("lost", 250),
                    ("control", 330),
                    ("and", 210),
                    ("laughed.", 290),
                ],
                start=1,
            ):
                f.write(f"2\t1\t5\t-\tNaN\t{i}\t{w}\t{rt}\n")

        # Write eyetracking.RT.txt (2 participants, sentences 1 and 2 only)
        with open(os.path.join(tmpdir, "eyetracking.RT.txt"), "w") as f:
            f.write(
                "subj_nr\tsent_nr\tsent_pos\tcorrect\tanswer_time\t"
                "word_pos\tword\tRTfirstfix\tRTfirstpass\tRTrightbound\tRTgopast\n"
            )
            # Participant 1, sentence 1
            f.write("1\t1\t127\t-\tNaN\t1\tAnne\t216\t348\t348\t348\n")
            f.write("1\t1\t127\t-\tNaN\t2\tlost\t152\t152\t152\t152\n")
            f.write("1\t1\t127\t-\tNaN\t3\tcontrol\t180\t220\t220\t280\n")
            f.write("1\t1\t127\t-\tNaN\t4\tand\t0\t0\t0\t0\n")  # Not fixated
            f.write("1\t1\t127\t-\tNaN\t5\tlaughed.\t264\t820\t820\t820\n")
            # Participant 1, sentence 2
            f.write("1\t2\t150\tc\t2106\t1\tBilly\t300\t300\t300\t300\n")
            f.write("1\t2\t150\tc\t2106\t2\twrote\t0\t0\t0\t0\n")  # Not fixated
            f.write("1\t2\t150\tc\t2106\t3\ton\t150\t150\t150\t150\n")
            f.write("1\t2\t150\tc\t2106\t4\tthe\t0\t0\t0\t0\n")  # Not fixated
            f.write("1\t2\t150\tc\t2106\t5\tenvelope.\t264\t820\t820\t820\n")

        yield tmpdir


class TestParseUclStimuli:
    def test_parses_all_sentences(self, ucl_dir):
        stimuli = _parse_ucl_stimuli(os.path.join(ucl_dir, "stimuli.txt"))
        assert len(stimuli) == 3
        assert set(stimuli.keys()) == {1, 2, 3}

    def test_stimulus_text(self, ucl_dir):
        stimuli = _parse_ucl_stimuli(os.path.join(ucl_dir, "stimuli.txt"))
        assert stimuli[1].unit_strings("words") == [
            "Anne",
            "lost",
            "control",
            "and",
            "laughed.",
        ]
        assert stimuli[2].unit_strings("words") == [
            "Billy",
            "wrote",
            "on",
            "the",
            "envelope.",
        ]

    def test_stimulus_id(self, ucl_dir):
        stimuli = _parse_ucl_stimuli(os.path.join(ucl_dir, "stimuli.txt"))
        assert stimuli[1].stimulus_id == "ucl_1"

    def test_metadata(self, ucl_dir):
        stimuli = _parse_ucl_stimuli(os.path.join(ucl_dir, "stimuli.txt"))
        assert stimuli[1].metadata["question"] == "-"
        assert stimuli[2].metadata["question"] == "Was it Bob who wrote?"
        assert stimuli[2].metadata["answer"] == "n"


class TestUCLSPR:
    def test_load(self, ucl_dir):
        ds = UCL_SPR()
        ds.load(ucl_dir)

        # Only sentences 1 and 2 are in the RT data
        assert len(ds.stimuli()) == 2
        assert ds.measures == ["spr_rt"]

    def test_participants(self, ucl_dir):
        ds = UCL_SPR()
        ds.load(ucl_dir)
        assert sorted(ds.participants) == ["1", "2"]

    def test_word_alignment(self, ucl_dir):
        ds = UCL_SPR()
        ds.load(ucl_dir)

        # Get participant 1, sentence 1
        data_p1 = ds.data(participant="1")
        # Find datum for sentence 1
        s1_data = [d for d in data_p1 if d.stimulus.metadata["sent_nr"] == 1]
        assert len(s1_data) == 1

        datum = s1_data[0]
        assert datum.values[0] == 350  # "Anne"
        assert datum.values[1] == 280  # "lost"
        assert datum.values[4] == 310  # "laughed."

    def test_words_from_rt_file(self, ucl_dir):
        """SPR stimulus words come from the RT file.

        stimuli.txt has "Billy" but the RT file has "Bily" (simulating the
        real discrepancies in the UCL dataset). The loader should use the
        RT file's version since that's what participants actually read.
        """
        ds = UCL_SPR()
        ds.load(ucl_dir)

        s2 = next(s for s in ds.stimuli() if s.metadata["sent_nr"] == 2)
        assert s2.unit_strings("words")[0] == "Bily"  # from RT file, not "Billy"

    def test_stimuli_txt_metadata_preserved(self, ucl_dir):
        """Question/answer metadata from stimuli.txt is still available."""
        ds = UCL_SPR()
        ds.load(ucl_dir)

        s2 = next(s for s in ds.stimuli() if s.metadata["sent_nr"] == 2)
        assert s2.metadata["question"] == "Was it Bob who wrote?"
        assert s2.metadata["answer"] == "n"

    def test_covariates_compute(self, ucl_dir):
        ds = UCL_SPR()
        ds.load(ucl_dir)
        stim = ds.stimuli()[0]
        covs = ds.covariates(stim)
        assert "unit_length" in covs
        assert "unit_position" in covs
        assert len(covs["unit_length"]) == len(stim.layers["words"])

    def test_exclusion_mask(self, ucl_dir):
        ds = UCL_SPR()
        ds.load(ucl_dir)
        datum = ds.data()[0]
        mask = ds.exclusion_mask(datum)
        # First word excluded
        assert mask[0] is np.bool_(False)


class TestUCLET:
    def test_load(self, ucl_dir):
        ds = UCL_ET()
        ds.load(ucl_dir)

        # Only sentences 1 and 2 are in the ET data
        assert len(ds.stimuli()) == 2

    def test_measures(self, ucl_dir):
        ds = UCL_ET()
        ds.load(ucl_dir)
        # RTfirstpass yields the first run of first-pass words plus the flag.
        # Total time needs the fixation file, which the fixture does not ship,
        # so the loaded data does not store it.
        assert sorted(ds.measures) == [
            "first_fixation",
            "first_fixation_in_first_pass",
            "first_pass",
            "first_run",
            "first_run_in_first_pass",
            "go_past",
            "go_past_in_first_pass",
            "right_bounded",
        ]

    def test_zero_to_nan(self, ucl_dir):
        """0 values in eye-tracking data should become NaN."""
        ds = UCL_ET()
        ds.load(ucl_dir)

        # Participant 1, sentence 1, word "and" (pos 3) was not fixated (all 0s)
        gaze_data = ds.data(measure="first_fixation", participant="1")
        s1_data = [d for d in gaze_data if d.stimulus.metadata["sent_nr"] == 1]
        assert len(s1_data) == 1

        datum = s1_data[0]
        # word_pos 4 in raw data → index 3 (0-indexed)
        assert np.isnan(datum.values[3]), "Zero should be converted to NaN"

    def test_nonzero_values_preserved(self, ucl_dir):
        ds = UCL_ET()
        ds.load(ucl_dir)

        ff_data = ds.data(measure="first_fixation", participant="1")
        s1_data = [d for d in ff_data if d.stimulus.metadata["sent_nr"] == 1]
        datum = s1_data[0]

        # "Anne" first fixation = 216
        assert datum.values[0] == 216
        # "lost" first fixation = 152
        assert datum.values[1] == 152

    def test_no_zero_in_any_measure(self, ucl_dir):
        """No eye-tracking measure holds a zero under the default view.

        A condition flag is exempt: zero is its value for a unit the
        condition rules out.
        """
        ds = UCL_ET()
        ds.load(ucl_dir)

        for datum in ds.data():
            if datum.measure_name in CONDITIONS:
                continue
            non_nan = datum.values[~np.isnan(datum.values)]
            assert not np.any(non_nan == 0), (
                f"Found zero value in {datum.measure_name} for "
                f"participant {datum.participant_id}"
            )

    def test_covariates_compute(self, ucl_dir):
        ds = UCL_ET()
        ds.load(ucl_dir)
        stim = ds.stimuli()[0]
        covs = ds.covariates(stim)
        assert "prev_word_fixated" in covs
        assert "saccade_launch_distance" in covs


# ---------------------------------------------------------------------------
# Real-data tests, paper-reported counts and raw-file alignment
# ---------------------------------------------------------------------------


@pytest.mark.needs_data
class TestUCLSPRRealData:
    @pytest.fixture(autouse=True)
    def load_dataset(self, data_root):
        path = os.path.join(data_root, "self-paced-reading", "original", "ucl")
        if not os.path.isdir(path):
            pytest.skip(f"UCL SPR data not found at {path}")
        self.ds = UCL_SPR()
        self.ds.load(path)

    def test_participant_count(self):
        """Paper reports 117 participants."""
        assert len(self.ds.participants) == 117

    def test_stimulus_count(self):
        """Paper reports 361 sentences."""
        assert len(self.ds.stimuli()) == 361

    def test_first_stimulus_words(self):
        _assert_first_stimulus_words(self.ds)

    def test_word_alignment_against_raw(self, data_root):
        """Every word in every RT row must match the loaded stimulus."""
        sent_map = {s.metadata["sent_nr"]: s for s in self.ds.stimuli()}
        rt_path = os.path.join(
            data_root,
            "self-paced-reading",
            "original",
            "ucl",
            "selfpacedreading.RT.txt",
        )
        mismatches = 0
        with open(rt_path, encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                sent_nr = int(row["sent_nr"])
                word_pos = int(row["word_pos"]) - 1
                raw_word = row["word"].strip()
                stim = sent_map.get(sent_nr)
                if stim is None:
                    continue
                words = stim.unit_strings("words")
                if word_pos < len(words) and words[word_pos] != raw_word:
                    mismatches += 1
        assert mismatches == 0, f"{mismatches} word mismatches against raw RT file"

    def test_total_values_match_raw_rows(self, data_root):
        """Total word-level values must equal raw file row count."""
        rt_path = os.path.join(
            data_root,
            "self-paced-reading",
            "original",
            "ucl",
            "selfpacedreading.RT.txt",
        )
        with open(rt_path) as fh:
            raw_count = sum(1 for _ in fh) - 1  # minus header
        loaded_count = sum(len(d.values) for d in self.ds.data())
        assert loaded_count == raw_count


@pytest.mark.needs_data
class TestUCLETRealData:
    @pytest.fixture(autouse=True)
    def load_dataset(self, data_root):
        path = os.path.join(data_root, "eye-tracking", "original", "ucl")
        if not os.path.isdir(path):
            pytest.skip(f"UCL ET data not found at {path}")
        self.ds = UCL_ET()
        self.ds.load(path)

    def test_participant_count(self):
        """Paper reports 43 participants for eye-tracking."""
        assert len(self.ds.participants) == 43

    def test_stimulus_count(self):
        """Paper reports 205 sentences for eye-tracking subset."""
        assert len(self.ds.stimuli()) == 205

    def test_measures(self):
        # The fixation file adds total time and completes ``first_run`` on
        # words first entered by a regression.
        assert sorted(self.ds.measures) == [
            "first_fixation",
            "first_fixation_in_first_pass",
            "first_pass",
            "first_run",
            "first_run_in_first_pass",
            "go_past",
            "go_past_in_first_pass",
            "right_bounded",
            "total_time",
        ]

    def test_first_run_extends_first_pass_run_to_regression_entries(self):
        self.ds.drop_skips = False
        gaze = {
            (d.participant_id, d.stimulus.stimulus_id): d.values
            for d in self.ds.data(measure="first_run_in_first_pass")
        }
        flags = {
            (d.participant_id, d.stimulus.stimulus_id): d.values
            for d in self.ds.data(measure="first_pass")
        }
        regression_entries = 0
        for datum in self.ds.data(measure="first_run"):
            key = (datum.participant_id, datum.stimulus.stimulus_id)
            released, flag = gaze[key], flags[key]
            assert set(np.unique(flag)) <= {0.0, 1.0}
            np.testing.assert_array_equal((released > 0), flag == 1)
            first_pass = flag == 1
            np.testing.assert_array_equal(
                datum.values[first_pass], released[first_pass]
            )
            assert (datum.values >= released).all()
            regression_entries += int(((flag == 0) & (datum.values > 0)).sum())
        assert regression_entries > 0

    def test_total_time_matches_raw_word_fixation_sums(self, data_root):
        stimuli = {s.metadata["sent_nr"]: s for s in self.ds.stimuli()}
        reference = {}
        path = os.path.join(
            data_root, "eye-tracking", "original", "ucl", "eyetracking.fix.txt"
        )
        with open(path, encoding="utf-8") as source:
            for row in csv.DictReader(source, delimiter="\t"):
                sentence = int(row["sent_nr"])
                key = (row["subj_nr"], stimuli[sentence].stimulus_id)
                values = reference.setdefault(
                    key, np.zeros(len(stimuli[sentence].layers["words"]))
                )
                if row["word_pos"] == "NaN":
                    continue
                position, duration = (
                    int(row["word_pos"]) - 1,
                    float(row["fix_duration"]),
                )
                assert 0 <= position < len(values)
                if np.isfinite(duration) and duration > 0:
                    values[position] += duration
        observed = {
            (d.participant_id, d.stimulus.stimulus_id): d.values
            for d in self.ds.data(measure="total_time")
        }
        assert set(observed) == set(reference)
        for key, actual in observed.items():
            expected = np.where(reference[key] > 0, reference[key], np.nan)
            np.testing.assert_array_equal(actual, expected, err_msg=str(key))

    def test_no_zero_values(self):
        """Every zero duration reads as missing under the default view.

        A condition flag keeps its zeros: they say the unit was not reached
        in first pass.
        """
        for datum in self.ds.data():
            if datum.measure_name in CONDITIONS:
                continue
            non_nan = datum.values[~np.isnan(datum.values)]
            assert not np.any(non_nan == 0), (
                f"Found zero in {datum.measure_name} for {datum.participant_id}"
            )

    def test_first_stimulus_words(self):
        _assert_first_stimulus_words(self.ds)
