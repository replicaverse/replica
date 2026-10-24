"""Tests for dataset: Dataset hierarchy, covariates, exclusion."""

import numpy as np
import pytest

from replica.core import Datum
from replica.dataset import (
    DatasetInfo,
    EyeTrackingDataset,
    SPRDataset,
)
from replica.dataset.measures import FIRST_PASS_RUN
from tests._stimuli import word_stimulus


class ConcreteSPRDataset(SPRDataset):
    """Concrete SPRDataset for testing (no file I/O)."""

    info = DatasetInfo(name="test_spr")

    def load(self, path=None):
        stim = word_stimulus("The cat sat on mats", "stim_1")
        self._stimuli = [stim]
        self._data = [
            Datum(stim, "spr_rt", np.array([350.0, 250.0, 280.0, 200.0, 310.0]), "p1"),
            Datum(stim, "spr_rt", np.array([400.0, 230.0, 270.0, 210.0, 290.0]), "p2"),
        ]


class ConcreteEyeTrackingDataset(EyeTrackingDataset):
    """Concrete EyeTrackingDataset for testing."""

    info = DatasetInfo(name="test_et")

    def load(self, path=None):
        stim = word_stimulus("The cat sat", "stim_1")
        self._stimuli = [stim]
        self._data = [
            Datum(stim, "first_run", np.array([200.0, 180.0, 250.0]), "p1"),
            Datum(stim, "first_fixation", np.array([150.0, 120.0, 180.0]), "p1"),
        ]


class TestSPRDataset:
    def test_load_and_query(self):
        ds = ConcreteSPRDataset()
        ds.load()

        assert len(ds.stimuli()) == 1
        assert len(ds.data()) == 2
        assert ds.measures == ["spr_rt"]
        assert ds.participants == ["p1", "p2"]
        assert all(d.modality == "behavioral" for d in ds.data())

    def test_filter_by_participant(self):
        ds = ConcreteSPRDataset()
        ds.load()

        data = ds.data(participant="p1")
        assert len(data) == 1
        assert data[0].participant_id == "p1"

    def test_filter_by_measure(self):
        ds = ConcreteSPRDataset()
        ds.load()

        data = ds.data(measure="spr_rt")
        assert len(data) == 2

        data = ds.data(measure="nonexistent")
        assert len(data) == 0

    def test_default_measure(self):
        ds = ConcreteSPRDataset()
        assert ds.default_measure == "spr_rt"


class TestCovariates:
    def test_unit_length(self):
        ds = ConcreteSPRDataset()
        ds.load()
        stim = ds.stimuli()[0]
        covs = ds.covariates(stim)

        # "The"=3, "cat"=3, "sat"=3, "on"=2, "mats"=4
        np.testing.assert_array_equal(covs["unit_length"], [3, 3, 3, 2, 4])

    def test_prev_unit_length(self):
        ds = ConcreteSPRDataset()
        ds.load()
        stim = ds.stimuli()[0]
        covs = ds.covariates(stim)

        assert np.isnan(covs["prev_unit_length"][0])
        np.testing.assert_array_equal(covs["prev_unit_length"][1:], [3, 3, 3, 2])

    def test_unit_position(self):
        ds = ConcreteSPRDataset()
        ds.load()
        stim = ds.stimuli()[0]
        covs = ds.covariates(stim)

        np.testing.assert_array_equal(covs["unit_position"], [0, 1, 2, 3, 4])

    def test_log_frequency_missing(self):
        ds = ConcreteSPRDataset()
        ds.load()
        stim = ds.stimuli()[0]
        covs = ds.covariates(stim)

        assert np.all(np.isnan(covs["log_frequency"]))

    def test_log_frequency_with_table(self):
        ds = ConcreteSPRDataset()
        ds.load()
        ds._word_frequencies = {
            "the": 5.0,
            "cat": 3.0,
            "sat": 2.5,
            "on": 4.0,
            "mats": 1.0,
        }
        stim = ds.stimuli()[0]
        covs = ds.covariates(stim)

        np.testing.assert_array_almost_equal(
            covs["log_frequency"], [5.0, 3.0, 2.5, 4.0, 1.0]
        )


class TestExclusionMask:
    def test_exclude_first_word(self):
        ds = ConcreteSPRDataset()
        ds.load()
        datum = ds.data()[0]
        mask = ds.exclusion_mask(datum)

        assert mask[0] is np.bool_(False)

    def test_exclude_nan(self):
        ds = ConcreteSPRDataset()
        ds.load()
        stim = ds.stimuli()[0]
        values = np.array([300.0, np.nan, 280.0, 200.0, 310.0])
        datum = Datum(stim, "spr_rt", values, "p1")
        mask = ds.exclusion_mask(datum)

        assert mask[1] is np.bool_(False)

    def test_no_rt_window_imposed(self):
        """exclusion_mask is structural only (first-word + NaN). RT-range and
        outlier trimming are researcher choices for the analysis layer, so the
        REPLICA does not drop extreme RTs here (no source-study-agnostic window)."""
        ds = ConcreteSPRDataset()
        ds.load()
        stim = ds.stimuli()[0]
        values = np.array([250.0, 50.0, 4000.0, 200.0, 310.0])
        datum = Datum(stim, "spr_rt", values, "p1")
        mask = ds.exclusion_mask(datum)

        assert mask[0] is np.bool_(False)  # first word excluded
        assert mask[1] is np.bool_(True)  # 50 ms kept (no lower window)
        assert mask[2] is np.bool_(True)  # 4000 ms kept (no upper window)
        assert mask[3] is np.bool_(True)


class TestAdditiveMeasures:
    def test_spr_additive(self):
        ds = ConcreteSPRDataset()
        assert ds.is_additive("spr_rt")

    def test_spr_non_additive(self):
        ds = ConcreteSPRDataset()
        assert not ds.is_additive("first_run")

    def test_et_additive_measures(self):
        ds = ConcreteEyeTrackingDataset()
        assert ds.is_additive("total_time")
        assert ds.is_additive("total_fixation_duration")

    def test_et_non_additive_measures(self):
        ds = ConcreteEyeTrackingDataset()
        assert not ds.is_additive("first_run")
        assert not ds.is_additive("first_fixation")

    def test_additive_measures_frozenset(self):
        assert SPRDataset.additive_measures == frozenset({"spr_rt"})
        assert EyeTrackingDataset.additive_measures == frozenset(
            {"total_time", "total_fixation_duration"}
        )


class TestEyeTrackingDataset:
    def test_additional_covariates(self):
        ds = ConcreteEyeTrackingDataset()
        ds.load()
        stim = ds.stimuli()[0]
        covs = ds.covariates(stim)

        assert "prev_word_fixated" in covs
        assert "saccade_launch_distance" in covs
        # Placeholders are NaN
        assert np.all(np.isnan(covs["prev_word_fixated"]))

    def test_multiple_measures(self):
        ds = ConcreteEyeTrackingDataset()
        ds.load()

        assert "first_run" in ds.measures
        assert "first_fixation" in ds.measures
        assert ds.default_measure == FIRST_PASS_RUN

    def test_first_pass_needs_the_flag(self):
        ds = ConcreteEyeTrackingDataset()
        ds.load()
        with pytest.raises(ValueError, match="first_pass flag"):
            ds.data(measure="first_run_in_first_pass")

    def test_declared_measures_stand_in_only_before_load(self):
        ds = ConcreteEyeTrackingDataset()
        ds._declared_measures = ("first_fixation", "first_run", "total_time")
        assert ds.measures == ["first_fixation", "first_run", "total_time"]
        ds.load()
        assert ds.measures == ["first_fixation", "first_run"]
