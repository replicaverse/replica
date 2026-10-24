"""ModelBlocks intervals close windows independently of fixation dwell."""

from dataclasses import replace

import numpy as np
import pytest

from replica.core import AnnotationLayer, Span, Stimulus
from replica.dataset import FixationCalculation
from replica.dataset.contracts import Fixation
from replica.dataset.loading.fixation_utils import reading_measures_from_fixations
from replica.dataset.loading.reaggregation import aggregate_fixations_to_layer


@pytest.fixture
def source():
    stimulus = Stimulus("a b c d", stimulus_id="s")
    stimulus.add_layer(
        AnnotationLayer("words", [Span(i, i + 1) for i in (0, 2, 4, 6)], list("abcd"))
    )
    stimulus.add_layer(
        AnnotationLayer(
            "regions", [Span(0, 3), Span(4, 5), Span(6, 7)], ["a b", "c", "d"]
        )
    )
    intervals = [
        [{"duration": 20, "closes_windows": False}],
        [{"duration": 30, "closes_windows": True}],
        [],
        [{"duration": 10, "closes_windows": False}],
        [],
    ]
    events = [
        Fixation(
            "p",
            "s",
            i * 100,
            (i + 1) * 100,
            100,
            np.nan,
            np.nan,
            u,
            None,
            i + 1,
            metadata={"following_interruptions": gaps},
        )
        for i, (u, gaps) in enumerate(zip([0, 1, 0, 1, 2], intervals, strict=True))
    ]
    return stimulus, events


def test_released_intervals_and_standard_dwell_are_separate(source):
    stimulus, events = source
    calculation = FixationCalculation(
        first_run="modelblocks@1", go_past="modelblocks@1", total_time="modelblocks@1"
    )
    native = aggregate_fixations_to_layer(events, stimulus, calculation=calculation)
    np.testing.assert_equal(native["first_fixation"], [100, 100, 100, np.nan])
    np.testing.assert_equal(native["first_run"], [120, 100, 100, 0])
    np.testing.assert_equal(native["go_past"], [120, 100, 100, 0])
    np.testing.assert_equal(native["total_time"], [220, 240, 100, 0])
    region = aggregate_fixations_to_layer(
        events, stimulus, "regions", calculation=calculation
    )
    np.testing.assert_equal(region["first_fixation"], [100, 100, np.nan])
    np.testing.assert_equal(region["first_run"], [220, 100, 0])
    np.testing.assert_equal(region["go_past"], [220, 100, 0])
    np.testing.assert_equal(region["total_time"], [460, 100, 0])
    standard = aggregate_fixations_to_layer(
        events, stimulus, "regions", calculation=FixationCalculation()
    )
    np.testing.assert_equal(standard["first_run"], [400, 100, 0])
    np.testing.assert_equal(standard["go_past"], [400, 100, 0])
    np.testing.assert_equal(standard["total_time"], [400, 100, 0])
    assert FixationCalculation.from_dict(calculation.to_dict()) == calculation
    assert "modelblocks-release" in calculation.to_dict()["upstream_revisions"]


@pytest.mark.parametrize("measure", ["first_run", "go_past", "total_time"])
def test_selecting_one_algorithm_does_not_replace_other_measures(source, measure):
    stimulus, events = source
    standard = aggregate_fixations_to_layer(
        events, stimulus, calculation=FixationCalculation()
    )
    chosen = aggregate_fixations_to_layer(
        events, stimulus, calculation=FixationCalculation(**{measure: "modelblocks@1"})
    )
    assert not np.array_equal(chosen[measure], standard[measure], equal_nan=True)
    for other in {"first_run", "go_past", "total_time", "first_fixation"} - {measure}:
        np.testing.assert_equal(chosen[other], standard[other])


def test_missing_intervals_and_unresolved_landings_are_errors(source):
    stimulus, events = source
    calculation = FixationCalculation(total_time="modelblocks@1")
    with pytest.raises(ValueError, match="recorded interruption"):
        aggregate_fixations_to_layer(
            [replace(events[0], metadata={})], stimulus, calculation=calculation
        )
    with pytest.raises(ValueError, match="resolved on-word"):
        aggregate_fixations_to_layer(
            [replace(events[0], unit_index=None)], stimulus, calculation=calculation
        )


def test_interval_windows_respect_the_requested_event_order(source):
    _, events = source
    events = [
        replace(event, fixation_index=6 - event.fixation_index) for event in events
    ]
    arrays = reading_measures_from_fixations(
        events,
        4,
        first_run_algorithm="modelblocks@1",
        go_past_algorithm="modelblocks@1",
        order="start_time",
    )
    np.testing.assert_equal(arrays["first_run"], [120, 100, 100, 0])
    np.testing.assert_equal(arrays["go_past"], [120, 100, 100, 0])
