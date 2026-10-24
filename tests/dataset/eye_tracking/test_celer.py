"""Real-data tests for the CELER loader (Berzak et al., 2022).

Slow tests: they load the full IA report.
Pin the paper-reported counts and the structural invariants of the loader.
"""

import os

import numpy as np
import pytest

from replica.dataset import CELER
from replica.dataset.measures import FIRST_PASS_RUN
from tests.dataset._measures import assert_eye_tracking_measures


def _celer_dir(data_root):
    return os.path.join(data_root, "eye-tracking", "original", "celer")


@pytest.mark.needs_data
@pytest.mark.slow
class TestCELERRealData:
    @pytest.fixture(autouse=True)
    def load_dataset(self, data_root):
        path = os.path.join(data_root, "eye-tracking", "original", "celer")
        if not os.path.isdir(path):
            pytest.skip(f"CELER data not found at {path}")
        if not os.path.exists(os.path.join(path, "sent_ia.tsv")):
            pytest.skip(
                f"CELER sent_ia.tsv not found at {path}/sent_ia.tsv. "
                "CELER requires LDC access, see github.com/berzak/celer."
            )
        self.ds = CELER()
        self.ds.load(path)

    def test_participant_count(self):
        """CELER ships data for 365 participants (69 L1 + 296 L2)."""
        assert len(self.ds.participants) == 365

    def test_measures(self):
        """Four word-level reading-time measures plus the ``first_pass`` flag
        read off ``IA_SKIP``."""
        assert_eye_tracking_measures(self.ds)

    def test_default_measure_first_pass_run(self):
        """The first run restricted to first pass is the canonical CELER DV."""
        assert self.ds.default_measure == FIRST_PASS_RUN

    def test_shared_text_metadata_present(self):
        """Each Stimulus carries a ``shared_text`` flag (0 or 1) to split
        shared vs. individual-text trials."""
        seen = set()
        for stim in self.ds.stimuli():
            seen.add(stim.metadata["shared_text"])
            if seen >= {0, 1}:
                break
        assert seen == {0, 1}, (
            f"Expected both shared_text=0 and shared_text=1 stimuli, got {seen}"
        )

    def test_first_run_is_nonneg(self):
        """First-run durations are non-negative (NaN is ok; negative would
        indicate a parse bug)."""
        for d in list(self.ds.data(measure="first_run"))[:200]:
            non_nan = d.values[~np.isnan(d.values)]
            assert (non_nan >= 0).all(), (
                f"negative first_run duration in {d.participant_id} on "
                f"{d.stimulus.stimulus_id}: {non_nan[non_nan < 0]}"
            )

    def test_first_pass_flag_restricts_the_first_run(self):
        """The flag holds 1, 0 or NaN, and ``first_run_in_first_pass`` keeps the
        first run where the flag is 1 and masks it where the flag is 0."""
        # The flag is a condition, so it is read off the full view.
        flags = {
            (d.stimulus.stimulus_id, d.participant_id): d.values
            for d in self.ds.data()
            if d.measure_name == "first_pass"
        }
        first_run = {
            (d.stimulus.stimulus_id, d.participant_id): d.values
            for d in self.ds.data(measure="first_run")
        }
        restricted = self.ds.data(measure="first_run_in_first_pass")
        assert len(restricted) == len(first_run) > 0
        for d in restricted[:200]:
            key = (d.stimulus.stimulus_id, d.participant_id)
            flag, run = flags[key], first_run[key]
            assert set(np.unique(flag[~np.isnan(flag)])) <= {0.0, 1.0}
            np.testing.assert_array_equal(d.values[flag == 1], run[flag == 1])
            assert np.isnan(d.values[flag == 0]).all()


@pytest.mark.needs_data
@pytest.mark.slow
class TestCELERSourceAlignment:
    """Check the entire release against a physical tab-delimited reference."""

    def test_source_words_and_duration_masks_in_both_skip_modes(self, data_root):
        path = _celer_dir(data_root)
        ds = CELER()
        ds.load(path)
        sentences, observations = {}, {}
        with open(os.path.join(path, "sent_ia.tsv"), encoding="utf-8") as source:
            header = source.readline().rstrip("\n").split("\t")
            for line in source:
                fields = line.rstrip("\n").split("\t")
                assert len(fields) == len(header)
                row = dict(zip(header, fields, strict=True))
                sid, index = row["sentenceid"], int(row["IA_ID"])
                entry = sentences.setdefault(sid, {"words": {}, "texts": set()})
                assert entry["words"].get(index, row["IA_LABEL"]) == row["IA_LABEL"]
                entry["words"][index] = row["IA_LABEL"]
                entry["texts"].add(row["sentence"])
                dwell = row["IA_DWELL_TIME"]
                unfixated = dwell not in ("", ".", "NaN") and float(dwell) == 0
                for measure, column in (
                    ("total_time", "IA_DWELL_TIME"),
                    ("first_fixation", "IA_FIRST_FIXATION_DURATION"),
                ):
                    key = (sid, row["DATA_FILE"], measure)
                    observation = observations.setdefault(key, {})
                    assert index not in observation
                    raw = row[column]
                    value = float(raw) if raw not in ("", ".", "NaN") else np.nan
                    observation[index] = 0.0 if np.isnan(value) and unfixated else value
        stimuli = {s.metadata["sentenceid"]: s for s in ds.stimuli()}
        assert set(stimuli) == set(sentences)
        for sid, stimulus in stimuli.items():
            source = sentences[sid]
            words = [source["words"][i] for i in sorted(source["words"])]
            units = stimulus.unit_strings("words")
            # A unit is its interest-area label, widened to the whole
            # whitespace-delimited display token when the label covers only
            # part of it; the units then tile the text.
            assert len(units) == len(words), sid
            for unit, label in zip(units, words, strict=True):
                assert label in unit and not any(c.isspace() for c in unit), sid
            spans = stimulus.layers["words"].spans
            covered = "".join(stimulus.text[sp.start : sp.end] for sp in spans)
            assert covered == "".join(stimulus.text.split()), sid
            assert stimulus.metadata["source_sentence"] in source["texts"], sid
            if stimulus.metadata["text_source"] == "sentence":
                assert stimulus.text in source["texts"], sid
            else:
                # Some sentence fields omit punctuation present in the recorded
                # interest areas. Preserve those labels and retain both sources.
                assert stimulus.metadata["text_source"] == "interest_areas"
                assert stimulus.text == " ".join(units), sid
                assert stimulus.text != stimulus.metadata["source_sentence"], sid
        expected = {}
        for key, values in observations.items():
            indices = sorted(sentences[key[0]]["words"])
            aligned = np.array([values.get(i, np.nan) for i in indices])
            if np.isfinite(aligned).any():
                expected[key] = aligned
        assert expected
        for exclude in (False, True):
            ds.drop_skips = exclude
            loaded = {
                (
                    d.stimulus.metadata["sentenceid"],
                    d.participant_id,
                    d.measure_name,
                ): d.values
                for measure in ("total_time", "first_fixation")
                for d in ds.data(measure)
            }
            assert set(loaded) == set(expected)
            for key, values in loaded.items():
                raw = expected[key]
                wanted = np.where(raw == 0, np.nan, raw) if exclude else raw
                np.testing.assert_array_equal(values, wanted, err_msg=str(key))


@pytest.mark.parametrize("exclude", [False, True])
def test_first_fixation_source_positions_and_missingness(tmp_path, exclude):
    import csv

    # Source rows are shuffled, and IA identifiers are not array positions.
    words = {10: "read", 20: "skip", 30: "unknown", 40: "last"}
    first = {10: "120", 20: ".", 30: ".", 40: "90"}
    dwell = {10: "200", 20: "0", 30: ".", 40: "160"}
    rows = [
        dict(
            sentenceid="s1",
            DATA_FILE="p1",
            IA_ID=i,
            IA_LABEL=words[i],
            sentence="read skip unknown last",
            shared_text=1,
            IA_FIRST_FIXATION_DURATION=first[i],
            IA_DWELL_TIME=dwell[i],
        )
        for i in (40, 20, 10, 30)
    ]
    with (tmp_path / "sent_ia.tsv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    dataset = CELER()
    dataset.load(str(tmp_path))
    dataset.drop_skips = exclude
    [datum] = dataset.data("first_fixation")
    assert datum.stimulus.unit_strings() == list(words.values())
    np.testing.assert_equal(datum.values, [120, np.nan if exclude else 0, np.nan, 90])
    np.testing.assert_equal(
        dataset.exclusion_mask(datum), [False, not exclude, False, True]
    )
