"""Real-data tests for the Brothers & Kuperberg (2021) loader.

Pin the paper-reported counts (216 items x 216 participants → 46,092
critical-region SPR observations).
"""

import os

import numpy as np
import pytest

from replica.dataset import BrothersKuperberg

pytestmark = pytest.mark.needs_data


def test_released_trimming_uses_two_and_a_half_standard_deviations(data_root):
    from pathlib import Path

    import pandas as pd

    path = (
        Path(data_root)
        / "self-paced-reading/original/brothers_kuperberg/SPRT_LogLin_216.csv"
    )
    if not path.exists():
        pytest.skip("Brothers and Kuperberg release unavailable")
    frame = pd.read_csv(path)
    groups = frame.groupby(["SUB", "condition"]).SUM_3RT
    cutoff = groups.transform("mean") + 2.5 * groups.transform("std")
    expected = frame.SUM_3RT.clip(lower=300, upper=cutoff)
    np.testing.assert_allclose(frame.SUM_3RT_trimmed, expected, atol=5e-7, rtol=0)


class TestBrothersKuperbergRealData:
    @pytest.fixture(autouse=True)
    def load_dataset(self, data_root):
        path = os.path.join(
            data_root, "self-paced-reading", "original", "brothers_kuperberg"
        )
        if not os.path.exists(os.path.join(path, "SPRT_LogLin_216.csv")):
            pytest.skip(
                f"brothers_kuperberg data not at {path}. "
                "Run `replica fetch brothers_kuperberg`."
            )
        self.ds = BrothersKuperberg()
        self.ds.load(path)

    def test_stimulus_count(self):
        """216 items x 3 cloze conditions (LC/MC/HC) = 648 distinct
        (item, condition) Stimuli, each carrying the full sentence
        frame from the BK paper's stimuli appendix."""
        assert len(self.ds.stimuli()) == 648

    def test_unique_items(self):
        """216 distinct paper item numbers (corpus name ``LogLin_216``)."""
        items = {s.metadata["item"] for s in self.ds.stimuli()}
        assert len(items) == 216

    def test_participant_count(self):
        """216 participants in the paper's behavioural release."""
        assert len(self.ds.participants) == 216

    def test_total_observations(self):
        """216 items x 216 participants = 46,656 max trials. The published
        data has 46,092, minor attrition expected."""
        n_per_measure = sum(1 for d in self.ds.data() if d.measure_name == "sum_3rt")
        assert n_per_measure == 46092, (
            f"Expected 46092 sum_3rt observations, got {n_per_measure}"
        )

    def test_measures(self):
        """Two DVs: ``sum_3rt`` (raw) + ``sum_3rt_trimmed``."""
        assert sorted(self.ds.measures) == sorted(["sum_3rt", "sum_3rt_trimmed"])

    def test_default_measure(self):
        assert self.ds.default_measure == "sum_3rt"

    def test_three_cloze_conditions(self):
        """Each item is in exactly one of LC / MC / HC. The full set
        must be present in stimulus metadata."""
        conditions = {s.metadata["condition"] for s in self.ds.stimuli()}
        assert conditions == {"LC", "MC", "HC"}

    def test_stimuli_carry_full_sentence(self):
        """Each Stimulus.text is the full sentence the participant read
        self-paced (recovered from the paper's appendix docx). The
        critical word sits at ``metadata['critical_position']`` (1-indexed)
        in the per-word ``words`` layer."""
        for s in self.ds.stimuli():
            n_words = len(s.layers["words"])
            assert n_words > 1, (
                f"{s.stimulus_id} has only {n_words} word(s); BK sentence "
                f"frames are multi-word."
            )
            crit_idx = s.metadata["critical_position"] - 1
            assert 0 <= crit_idx < n_words, (
                f"{s.stimulus_id}: critical_position {crit_idx + 1} out of "
                f"range for {n_words}-word sentence"
            )
            assert (
                s.layers["words"].values[crit_idx].lower()
                == s.metadata["critical_word"].lower()
            ), f"critical_word mismatch in {s.stimulus_id}"

    def test_datum_aligns_with_words_layer(self):
        """Datum's value array is NaN-padded to len(words) with the
        single finite SUM_3RT entry at the critical word's index."""
        for d in list(self.ds.data())[:300]:
            n_words = len(d.stimulus.layers["words"])
            assert len(d.values) == n_words
            finite = np.isfinite(d.values)
            assert finite.sum() == 1, f"expected 1 finite entry, got {finite.sum()}"
            crit_idx = d.stimulus.metadata["critical_position"] - 1
            assert finite[crit_idx]

    def test_metadata_carries_paper_predictors(self):
        """Cloze + trigram predictors are exposed via stimulus.metadata
        so they can be mixed in as PrecomputedFeatures."""
        s = self.ds.stimuli()[0]
        for key in (
            "cloze",
            "log_cloze",
            "trigram",
            "log_trigram",
            "condition",
            "critical_word",
            "critical_position",
        ):
            assert key in s.metadata, f"missing metadata key {key!r}"

    def test_distinct_sentences_per_condition(self):
        """The three conditions per item present different pre-critical
        sentence frames (the whole point of the BK paradigm)."""
        items = {s.metadata["item"] for s in self.ds.stimuli()}
        sampled = list(items)[:20]
        for item in sampled:
            cond_to_text = {
                s.metadata["condition"]: s.text
                for s in self.ds.stimuli()
                if s.metadata["item"] == item
            }
            assert len(set(cond_to_text.values())) == 3, (
                f"item {item}: expected 3 distinct sentences, got "
                f"{set(cond_to_text.values())}"
            )
