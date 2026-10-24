"""Real-data tests for the DeVarda2023 loader.

Pin paper-reported counts and the structural invariants of the aligned
multi-measure release (de Varda et al., 2023, BRM).
"""

import os

import numpy as np
import pytest

from replica.dataset import DeVarda2023
from replica.dataset.measures import FIRST_PASS_RUN

pytestmark = pytest.mark.needs_data


class TestDeVarda2023RealData:
    @pytest.fixture(autouse=True)
    def load_dataset(self, data_root):
        path = os.path.join(data_root, "eye-tracking", "processed", "devarda2023")
        if not os.path.isdir(path):
            pytest.skip(f"DeVarda2023 data not found at {path}")
        self.ds = DeVarda2023()
        try:
            self.ds.load(path)
        except FileNotFoundError as e:
            pytest.skip(f"DeVarda2023 data not loadable: {e}")

    def test_stimulus_count(self):
        """de Varda et al. (2023) align across 205 sentences from Frank et al. (2013, 2015)."""
        assert len(self.ds.stimuli()) == 205

    def test_one_averaged_series_with_item_lists(self):
        assert self.ds.participants == ["averaged"]
        assert {
            v for s in self.ds.stimuli() for v in s.metadata["cloze_lists"].values()
        } == set(range(1, 9))

    def test_eleven_source_means(self):
        expected = {
            "first_fixation_in_first_pass",
            "first_run_in_first_pass",
            "right_bounded",
            "go_past_in_first_pass",
            "spr_rt",
            "elan",
            "lan",
            "n400",
            "epnp",
            "p600",
            "pnp",
        }
        assert set(self.ds.measures) == expected
        assert not self.ds.data("first_run")
        assert not self.ds.data("first_pass")

    def test_default_measure_is_first_pass_run(self):
        """The first run restricted to first pass (the release's
        ``RTfirstpass``) is the most-used DV in this corpus's eye-tracking
        analyses; REPLICA pins it as the default."""
        assert self.ds.default_measure == FIRST_PASS_RUN

    def test_first_stimulus_metadata(self):
        """First stimulus is sent_id=96 ('Arthur placed the bars of chocolate
        on the counter.')."""
        s = self.ds.stimuli()[0]
        assert s.metadata["sent_id"] == 96
        assert s.text.startswith("Arthur placed the bars of chocolate")
        assert s.layers["words"].values[:3] == ["Arthur", "placed", "the"]

    def test_each_measure_has_one_datum_per_sentence(self):
        for measure in self.ds.measures:
            assert len(self.ds.data(measure)) == 205

    def test_data_values_aligned_to_words(self):
        """Each Datum's values array length matches the word count of its
        stimulus, sparseness (NaN at non-test positions) does not change
        the array length."""
        for datum in self.ds.data():
            assert len(datum.values) == len(datum.stimulus.layers["words"]), (
                f"Datum for {datum.participant_id} on {datum.stimulus.stimulus_id}: "
                f"values length {len(datum.values)} != "
                f"word count {len(datum.stimulus.layers['words'])}"
            )

    def test_at_least_one_observation_per_datum(self):
        """The loader filters out datums that are all-NaN; every yielded
        Datum must have at least one observed cell."""
        for datum in self.ds.data():
            assert np.any(~np.isnan(datum.values)), (
                f"Datum for {datum.participant_id} on {datum.stimulus.stimulus_id} "
                f"({datum.measure_name}) is all NaN, should have been filtered out"
            )
