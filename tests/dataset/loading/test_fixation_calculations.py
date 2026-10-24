"""Versioned windows on physical events and mapped units."""

import json
from dataclasses import replace

import numpy as np
import pytest

from replica.core import AnnotationLayer, Datum, Span, Stimulus
from replica.dataset import FixationCalculation, fixation_feature
from replica.dataset.base import EyeTrackingDataset
from replica.dataset.contracts import DatasetInfo, Fixation
from replica.dataset.loading.reaggregation import aggregate_fixations_to_layer
from replica.feature.resolution import resolve_features


def _trial(units, durations):
    text = "a b c d e"
    stimulus = Stimulus(stimulus_id="s", text=text)
    stimulus.add_layer(
        AnnotationLayer(
            "words", [Span(i, i + 1) for i in range(0, 9, 2)], list("abcde")
        )
    )
    stimulus.add_layer(
        AnnotationLayer(
            "phrases",
            [Span(0, 1), Span(2, 5), Span(6, 7), Span(8, 9)],
            ["a", "b c", "d", "e"],
        )
    )
    events = []
    start = 0
    for i, (unit, duration) in enumerate(zip(units, durations, strict=True), 1):
        events.append(
            Fixation(
                "p",
                "s",
                start,
                start + duration,
                duration,
                0,
                0,
                unit,
                None,
                i,
                metadata={"run_id": 99},
            )
        )
        start += duration
    return stimulus, events


@pytest.mark.parametrize(
    "options",
    [
        {"first_run": "popeye_episode"},
        {"first_run": "standard@2"},
        {"go_past": "copco_extractor@2"},
        {"first_run": "popeye_episode@1", "off_text_ends_run": False},
        {"go_past": "copco_extractor@1", "go_past_requires_right_exit": True},
    ],
)
def test_unknown_versions_and_inapplicable_policies_are_rejected(options):
    with pytest.raises(ValueError):
        FixationCalculation(**options)


def test_episode_windows_rebuild_regressions_after_mapping():
    stimulus, events = _trial([3, 2, 1, 2, 4], [100, 70, 50, 90, 80])
    episode = FixationCalculation(first_run="popeye_episode@1")
    standard = FixationCalculation()
    native = aggregate_fixations_to_layer(events, stimulus, calculation=episode)
    np.testing.assert_equal(native["first_run"], [0, 50, 160, 100, 80])
    np.testing.assert_equal(native["first_run_in_first_pass"], [0, 0, 0, 100, 80])
    assert (
        aggregate_fixations_to_layer(events, stimulus, calculation=standard)[
            "first_run"
        ][2]
        == 70
    )
    for calculation in (standard, episode):
        merged = aggregate_fixations_to_layer(
            events, stimulus, "phrases", calculation=calculation
        )
        assert merged["first_fixation"][1] == 70
        assert merged["first_run"][1] == 210
    assert all(e.metadata["run_id"] == 99 for e in events)


def test_episode_windows_handle_off_text_and_trial_boundaries():
    stimulus, events = _trial([2, 1, None, 1, 0, 1], [100, 70, 20, 90, 50, 80])
    calculation = FixationCalculation(first_run="popeye_episode@1")
    arrays = aggregate_fixations_to_layer(events, stimulus, calculation=calculation)
    assert arrays["first_run"][1] == 160
    other_visit = [replace(e, metadata={"trial_index": 2}) for e in events]
    events = [replace(e, metadata={"trial_index": 1}) for e in events]
    repeated = aggregate_fixations_to_layer(
        events + other_visit, stimulus, calculation=calculation
    )
    np.testing.assert_equal(repeated["first_run"], arrays["first_run"])
    np.testing.assert_equal(repeated["total_time"], arrays["total_time"] * 2)


def test_off_text_events_do_not_hide_a_new_regression_episode():
    stimulus, events = _trial([1, 2, None, None, 1], [229, 234, 31, 112, 132])
    arrays = aggregate_fixations_to_layer(
        events, stimulus, calculation=FixationCalculation(first_run="popeye_episode@1")
    )
    assert arrays["first_run"][1] == 229
    assert arrays["total_time"][1] == 361


def test_copco_window_keeps_event_adjacency_and_flooring():
    stimulus, events = _trial([1, 0, 1, 0, 1, 2], [100.9, 50.9, 80.9, 40.9, 60.9, 90.9])
    calculation = FixationCalculation(go_past="copco_extractor@1", floor_durations=True)
    arrays = aggregate_fixations_to_layer(events, stimulus, calculation=calculation)
    np.testing.assert_equal(arrays["go_past"], [50, 230, 90, 0, 0])
    assert arrays["go_past_in_first_pass"][0] == 0
    events[2] = replace(events[2], fixation_index=7)
    # The extractor orders events by their source ordinals (fixation_index).
    changed = aggregate_fixations_to_layer(events, stimulus, calculation=calculation)
    assert changed["go_past"][1] == 150


def test_ambiguous_landings_must_resolve_before_alternative_windows():
    stimulus, events = _trial([None], [100])
    events = [replace(events[0], candidate_unit_indices=(1, 2))]
    calculation = FixationCalculation(first_run="popeye_episode@1")
    with pytest.raises(ValueError, match="resolved target landings"):
        aggregate_fixations_to_layer(events, stimulus, calculation=calculation)
    merged = aggregate_fixations_to_layer(
        events, stimulus, "phrases", calculation=calculation
    )
    assert merged["total_time"][1] == 100
    assert merged["first_fixation"][1] == 100
    assert merged["first_run"][1] == 100


def test_serialized_calculation_roundtrips_and_rejects_changed_pins():
    calculation = FixationCalculation(
        first_run="popeye_episode@1", go_past="copco_extractor@1"
    )
    record = json.loads(json.dumps(calculation.to_dict()))
    assert FixationCalculation.from_dict(record) == calculation
    record["upstream_revisions"]["popEye"] = "another revision"
    with pytest.raises(ValueError, match="inconsistent provenance"):
        FixationCalculation.from_dict(record)
    with pytest.raises(TypeError, match="boolean"):
        FixationCalculation(floor_durations="false")


def test_batch_calculation_validates_a_distinct_native_reference():
    class Events(EyeTrackingDataset):
        info = DatasetInfo(name="events")

        def load(self, path):
            pass

    stimulus, events = _trial([0, 1, 0, 2, 3, 4], [100, 80, 50, 70, 60, 90])
    dataset = Events()
    dataset._stimuli = [stimulus]
    dataset._fixations = events
    reference = Datum(
        stimulus,
        "first_fixation",
        np.array([100, 80, 70, 60, 90.0]),
        "p",
        metadata={"measurement_source": "fixations"},
    )
    dataset.add_datum(reference)
    key = ("s", "p", "words", "go_past")
    calculation = FixationCalculation()
    (values,) = resolve_features(
        dataset,
        [
            fixation_feature(
                dataset,
                "go_past",
                calculation=calculation,
                reference="first_fixation",
            )
        ],
    )
    result = values.blocks[0].source
    assert result.measure_name == "go_past"
    assert result.metadata["native_reference_measure"] == "first_fixation"
    np.testing.assert_equal(values.blocks[0].values, [100, 130, 70, 60, 90])
    np.testing.assert_equal(reference.values, [100, 80, 70, 60, 90])
    wrong = replace(reference, values=reference.values + 1)
    with pytest.raises(ValueError, match="do not reproduce the native measure"):
        dataset._reaggregate_responses(
            [(key, wrong, wrong)], None, {"go_past": calculation}
        )
