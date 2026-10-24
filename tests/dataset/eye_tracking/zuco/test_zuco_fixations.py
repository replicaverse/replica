"""ZuCo physical events must reproduce source assignments without duplicate dwell."""

from pathlib import Path

import numpy as np
import pytest

from replica.core import AnnotationLayer, Span
from replica.dataset import FixationCalculation, describe_dataset
from replica.dataset.eye_tracking.zuco.zuco1 import ZuCo1
from replica.dataset.eye_tracking.zuco.zuco2 import ZuCo2
from replica.feature import resolve


@pytest.fixture(params=[ZuCo1, ZuCo2])
def dataset(request):
    ds = request.param()
    stim = ds._make_stimulus(["ab", "cd", "ef", "gh"], "s")
    ds._append_reading_measures(
        stim,
        "p",
        {
            "first_fixation": np.array([60, 60, 90, 0]),
            "first_run": np.array([60, 130, 90, 0]),
            "go_past": np.array([60, 130, 190, 0]),
            "total_time": np.array([160, 130, 90, 0]),
        },
    )
    ds._store_fixation_trial(
        stim,
        "p",
        {
            "x": [5, 10, 25, 15, 35, 5],
            "y": [5] * 6,
            "duration": [49, 60, 80, 70, 90, 100],
        },
        [[0, 0, 10, 10], [10, 0, 20, 10], [30, 0, 40, 10], [50, 0, 60, 10]],
        [[1, 5], [2, 3], [4], []],
    )
    stim.add_layer(
        AnnotationLayer(
            "regions", [Span(0, 5), Span(6, 8), Span(9, 11)], ["ab cd", "ef", "gh"]
        )
    )
    return ds


def test_physical_events_keep_short_and_off_text_fixations(dataset):
    events = dataset.fixations("p", "s")
    assert len(events) == 6
    assert [e.fixation_index for e in events] == list(range(1, 7))
    assert events[0].duration == 98
    assert events[1].unit_index is None
    assert events[1].candidate_unit_indices == (0, 1)
    assert events[2].unit_index is None and not events[2].candidate_unit_indices
    assert all(e.metadata["synthetic_start_time"] for e in events)
    assert dataset.fixations("absent", "s") == []


def test_missing_raw_events_cannot_fall_back_to_summing_overlapping_totals(dataset):
    dataset._fixation_records.clear()
    with pytest.raises(ValueError, match="not additive"):
        resolve(dataset, dataset.feature("total_time"), layer="regions")


@pytest.mark.parametrize("drop_skips", [False, True])
@pytest.mark.parametrize(
    "measure,expected",
    [
        ("first_fixation", [120, 180, 0]),
        ("first_run", [260, 180, 0]),
        ("go_past", [260, 380, 0]),
        ("total_time", [460, 180, 0]),
    ],
)
def test_native_roundtrip_and_merged_windows(dataset, drop_skips, measure, expected):
    dataset.drop_skips = drop_skips
    datum = dataset.data(measure)[0]
    np.testing.assert_equal(dataset.reaggregate(datum, "words").values, datum.values)
    target = np.array(expected, dtype=float)
    if drop_skips:
        target[target == 0] = np.nan
    np.testing.assert_equal(dataset.reaggregate(datum, "regions").values, target)
    explicit = dataset.reaggregate(datum, "regions", calculation=FixationCalculation())
    np.testing.assert_equal(explicit.values, target)


def test_native_explicit_calculation_keeps_boundary_uncertainty(dataset):
    datum = dataset.data("total_time")[0]
    actual = dataset.reaggregate(datum, "words", calculation=FixationCalculation())
    assert np.isnan(actual.values[:2]).all()
    assert actual.values[2] == 180
    assert actual.metadata["has_native_reference"] is False


def test_incorrect_source_assignments_prevent_reaggregation(dataset):
    record = dataset._fixation_records[("p", "s")]
    record[3][0][0] = 2
    with pytest.raises(ValueError, match="fixPositions"):
        dataset.reaggregate(dataset.data("first_fixation")[0], "regions")


def test_catalog_reports_the_event_stream_as_available(dataset):
    assert dataset.fixations("p", "s")
    key = type(dataset).__name__.lower()
    assert describe_dataset(key).fixations == "available"


@pytest.mark.needs_data
@pytest.mark.slow
@pytest.mark.parametrize(
    "cls,task",
    [(ZuCo1, "NR"), (ZuCo1, "SR"), (ZuCo1, "TSR"), (ZuCo2, "NR"), (ZuCo2, "TSR")],
)
def test_released_word_measures_roundtrip(data_root, cls, task):
    path = Path(data_root) / "eye-tracking/original" / cls.info.name
    if not list(path.glob(f"results*_{task}.mat")):
        pytest.skip("ZuCo release unavailable")
    ds = cls(task=task)
    ds.load(str(path), max_subjects=1, load_eeg=False)
    references = {
        (d.participant_id, d.stimulus.stimulus_id, d.measure_name): d for d in ds._data
    }
    for (participant, sid), events in ds.fixation_trials():
        for measure in ("first_fixation", "first_run", "go_past", "total_time"):
            reference = references.pop((participant, sid, measure))
            for drop_skips in (False, True):
                ds.drop_skips = drop_skips
                datum = ds._skips_as_missing(reference) if drop_skips else reference
                computed = ds.reaggregate(
                    datum, datum.layer, fixations=events, _reference=reference
                )
                np.testing.assert_allclose(
                    computed.values, datum.values, rtol=0, atol=1e-6, equal_nan=True
                )
    assert not references, "Some summary trials have no source fixation record"


@pytest.mark.parametrize("drop_skips", [False, True])
def test_empty_record_establishes_skips_but_absent_events_do_not(dataset, drop_skips):
    datum = dataset.data("total_time")[0]
    with pytest.raises(ValueError, match="absent trial"):
        dataset.reaggregate(datum, "regions", fixations=[])
    key = ("p", "s")
    stimulus, arrays, bounds, positions = dataset._fixation_records[key]
    dataset._store_fixation_trial(
        stimulus, "p", {k: [] for k in arrays}, bounds, [[]] * len(positions)
    )
    for datum in dataset._data:
        datum.values[:] = 0.0
    dataset.drop_skips = drop_skips
    expected = [np.nan if drop_skips else 0.0] * 3
    for name in ("first_fixation", "first_run", "go_past", "total_time"):
        datum = dataset.data(name)[0]
        np.testing.assert_equal(dataset.reaggregate(datum, "regions").values, expected)
        np.testing.assert_equal(
            dataset.reaggregate(
                datum, "regions", calculation=FixationCalculation()
            ).values,
            expected,
        )
    dataset._fixation_records.clear()
    with pytest.raises(NotImplementedError, match="absent"):
        dataset.reaggregate(datum, "regions")
