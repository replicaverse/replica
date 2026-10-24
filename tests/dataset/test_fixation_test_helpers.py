"""Synthetic checks for the shared fixation-loader assertions."""

import numpy as np
import pytest

from replica.core import AnnotationLayer, Datum, Span, Stimulus
from replica.dataset.contracts import Fixation

from ._fixation_test_base import (
    assert_fixation_round_trip,
    assert_trial_dwell_times,
    group_fixations,
)


class _Dataset:
    def __init__(self, data, fixations):
        self._data = data
        self._fixations = fixations

    def data(self, *, measure):
        return [datum for datum in self._data if datum.measure_name == measure]

    def fixations(self):
        return self._fixations


def _dataset() -> _Dataset:
    stimulus = Stimulus(
        text="a b",
        stimulus_id="s",
        layers={
            "words": AnnotationLayer(
                name="words",
                spans=[Span(0, 1), Span(2, 3)],
                values=["a", "b"],
            )
        },
    )
    fixations = [
        Fixation(
            participant_id="p",
            stimulus_id="s",
            start_time=float(index),
            end_time=float(index + duration),
            duration=duration,
            x=0.0,
            y=0.0,
            unit_index=index,
            char_offset=2 * index,
            fixation_index=index + 1,
            metadata={"trial_index": 1, "dwell_time": duration, "run_id": 1},
        )
        for index, duration in enumerate((2.0, 3.0))
    ]
    data = [
        Datum(
            stimulus=stimulus,
            measure_name=measure,
            values=np.array(values),
            participant_id="p",
        )
        for measure, values in (
            ("total_time", [2.0, 3.0]),
            ("first_run", [2.0, 3.0]),
            ("first_pass", [1.0, 1.0]),
        )
    ]
    return _Dataset(data, fixations)


def test_shared_fixation_assertions_accept_matching_data():
    dataset = _dataset()
    assert len(group_fixations(dataset.fixations())) == 1
    assert_fixation_round_trip(
        dataset,
        source_measure="total_time",
        min_compared=0,
    )
    assert_fixation_round_trip(
        dataset,
        source_measure="first_run",
        min_compared=0,
    )
    assert_fixation_round_trip(
        dataset,
        source_measure="first_pass",
        tolerance=0.5,
        min_compared=0,
    )
    assert_trial_dwell_times(dataset, tolerance=0.0, min_compared=0)


def test_round_trip_helper_reports_reconstruction_mismatches():
    dataset = _dataset()
    dataset._data[0].values[1] = 3.01
    with pytest.raises(AssertionError, match="1/2 cells diverge"):
        assert_fixation_round_trip(
            dataset,
            source_measure="total_time",
            min_compared=0,
        )


@pytest.mark.parametrize("missing_reconstruction", ["nan", "zero"])
def test_missing_reconstruction_cannot_hide_an_observed_fixation(
    missing_reconstruction,
):
    dataset = _dataset()
    dataset._data[0].values[1] = np.nan
    with pytest.raises(AssertionError, match="missingness mismatches"):
        assert_fixation_round_trip(
            dataset,
            source_measure="total_time",
            min_compared=0,
            missing_reconstruction=missing_reconstruction,
        )


def test_observed_reference_requires_an_observed_reconstruction():
    dataset = _dataset()
    dataset._fixations.pop()
    with pytest.raises(AssertionError, match="missingness mismatches"):
        assert_fixation_round_trip(
            dataset,
            source_measure="first_run",
            aggregate_measure="first_fixation",
            min_compared=0,
        )


@pytest.mark.parametrize(
    "measure,policy", [("total_time", "zero"), ("first_run", "nan")]
)
def test_known_skip_reconstruction_follows_its_measure_policy(measure, policy):
    dataset = _dataset()
    dataset._fixations.pop()
    for datum in dataset._data:
        datum.values[1] = np.nan
    assert_fixation_round_trip(
        dataset,
        source_measure=measure,
        aggregate_measure="first_fixation" if policy == "nan" else "total_time",
        missing_reconstruction=policy,
        min_compared=0,
    )


def test_missing_cells_cannot_satisfy_the_minimum_comparison_count():
    dataset = _dataset()
    dataset._fixations.clear()
    dataset._data[0].values[:] = np.nan
    with pytest.raises(AssertionError, match="too few cells compared: 0"):
        assert_fixation_round_trip(dataset, source_measure="total_time", min_compared=0)


def test_measure_assertion_needs_loaded_data():
    from replica.dataset import CELER

    from ._measures import assert_eye_tracking_measures

    with pytest.raises(AssertionError, match="no loaded data"):
        assert_eye_tracking_measures(CELER())
