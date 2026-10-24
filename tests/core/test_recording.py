"""Contracts shared by sampled response modalities."""

from dataclasses import replace

import numpy as np
import pytest

from replica.core import AnnotationLayer, Modality, Recording, Span, Stimulus


def _stimulus() -> Stimulus:
    stimulus = Stimulus("hello", stimulus_id="story")
    stimulus.add_layer(AnnotationLayer("words", [Span(0, 5)], ["hello"]))
    return stimulus


def test_regular_recording_exposes_absolute_times():
    recording = Recording(
        stimulus=_stimulus(),
        values=np.zeros((3, 2)),
        measure_name="voltage",
        modality=Modality.EEG,
        participant_id="p1",
        sample_interval=0.5,
        start_time=1.0,
        dimension_names=("time", "electrode"),
        dimension_labels={"electrode": ("Cz", "Pz")},
    )

    assert recording.modality == "eeg"
    np.testing.assert_array_equal(recording.times, [1.0, 1.5, 2.0])
    explicit = Recording(
        stimulus=recording.stimulus,
        values=recording.values,
        measure_name=recording.measure_name,
        modality=recording.modality,
        sample_interval=recording.sample_interval,
        sample_times=recording.times.astype(np.float32),
    )
    np.testing.assert_array_equal(explicit.times, recording.times)
    assert explicit.sample_times.dtype == np.dtype(np.float32)
    assert explicit.times.dtype == np.dtype(float)
    masked = np.ma.array(np.arange(6).reshape(3, 2), mask=False)
    masked.mask[1, 0] = True
    missing = replace(recording, values=masked)
    np.testing.assert_equal(missing.values, [[0, 1], [np.nan, 3], [4, 5]])
    assert type(missing.values) is np.ndarray
    np.testing.assert_array_equal(masked.data, np.arange(6).reshape(3, 2))
    complex_signal = np.full((3, 2), 1 + 2j, dtype=np.complex64)
    assert replace(recording, values=complex_signal).values is complex_signal


@pytest.mark.parametrize(
    "start, interval, n_samples, dtype",
    [
        (-0.1, 0.001, 101, np.float64),
        (-1.0, 0.1, 101, np.float64),
        (1e16, 2.0, 10, np.float64),
        (0.0, 0.1, 101, np.float32),
        (0.0, 0.002, 10001, np.float32),
    ],
)
def test_regular_clock_roundtrips_through_explicit_timestamps(
    start, interval, n_samples, dtype
):
    recording = Recording(
        stimulus=_stimulus(),
        values=np.zeros(n_samples),
        measure_name="signal",
        modality="eeg",
        start_time=start,
        sample_interval=interval,
    )
    explicit = Recording(
        stimulus=recording.stimulus,
        values=recording.values,
        measure_name=recording.measure_name,
        modality=recording.modality,
        sample_interval=interval,
        sample_times=recording.times.astype(dtype),
    )
    np.testing.assert_array_equal(explicit.times, recording.times.astype(dtype))
    assert explicit.sample_times.dtype == np.dtype(dtype)
    assert explicit.times.dtype == np.dtype(float)
    np.testing.assert_array_equal(replace(explicit).times, explicit.times)


def test_irregular_recording_preserves_sample_times():
    recording = Recording(
        stimulus=_stimulus(),
        values=np.zeros((3, 1)),
        measure_name="signal",
        modality="custom-neural",
        sample_times=np.array([0.0, 0.3, 0.9]),
        dimension_names=("time", "sensor"),
    )

    np.testing.assert_array_equal(recording.times, [0.0, 0.3, 0.9])
    assert recording.sample_interval is None


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"sample_interval": 0.0}, "positive"),
        ({"sample_interval": np.nan}, "finite and positive"),
        ({"sample_interval": np.inf}, "finite and positive"),
        ({"sample_interval": 0.5 + 1j}, "sample_interval.*real"),
        ({"sample_interval": 0.5, "start_time": np.nan}, "start_time.*finite"),
        ({"sample_interval": 0.5, "start_time": np.inf}, "start_time.*finite"),
        ({"sample_interval": 0.5, "start_time": 1j}, "start_time.*real"),
        ({"sample_times": np.array([0.0])}, "length"),
        ({"sample_times": np.array([0, 0.5 + 1j, 1])}, "sample_times.*real"),
        (
            {"sample_times": np.ma.array([0.0, 0.5, 1.0], mask=[False, True, False])},
            "finite",
        ),
        ({"sample_times": np.array([0.0, 0.5, 0.4])}, "increasing"),
        ({"sample_times": [0.0, 1.0, 2.0], "sample_interval": 2.0}, "agree"),
        ({"sample_times": [1e16, 1e16 + 4, 1e16 + 8], "sample_interval": 8.0}, "agree"),
        (
            {
                "sample_times": np.array([0.0, 0.105, 0.2], dtype=np.float32),
                "sample_interval": 0.1,
            },
            "agree",
        ),
        ({"start_time": 1e20, "sample_interval": 0.5}, "increasing"),
        ({"start_time": 1e308, "sample_interval": 1e308}, "finite"),
    ],
)
def test_recording_rejects_invalid_timing(kwargs, message):
    with pytest.raises(ValueError, match=message):
        Recording(
            stimulus=_stimulus(),
            values=np.zeros((3, 1)),
            measure_name="signal",
            modality="eeg",
            dimension_names=("time", "sensor"),
            **kwargs,
        )


def _two_words() -> Stimulus:
    stimulus = Stimulus("hello world", stimulus_id="s")
    stimulus.add_layer(
        AnnotationLayer("words", [Span(0, 5), Span(6, 11)], ["hello", "world"])
    )
    return stimulus


@pytest.mark.parametrize(
    "event_times",
    [[1.25, np.nan], np.ma.array([1.25, 999.0], mask=[False, True])],
    ids=["nan", "masked"],
)
def test_event_times_may_leave_a_unit_untimed(event_times):
    recording = Recording(
        stimulus=_two_words(),
        values=np.zeros((4, 1)),
        measure_name="eeg",
        modality="eeg",
        sample_interval=0.5,
        start_time=1.0,
        event_times=event_times,
    )
    assert recording.event_times.dtype == float
    np.testing.assert_array_equal(recording.event_times, [1.25, np.nan])


def test_a_recording_without_its_own_onsets_leaves_them_unset():
    recording = Recording(
        stimulus=_two_words(),
        values=np.zeros((4, 1)),
        measure_name="eeg",
        modality="eeg",
        sample_interval=0.5,
    )
    assert recording.event_times is None


@pytest.mark.parametrize(
    "event_times, message",
    [
        ([0.1], "one onset per unit"),
        ([0.1, np.inf], "finite or NaN"),
        ([[0.1, 0.2]], "one-dimensional"),
        ([1 + 2j, 3 + 4j], "event_times.*real"),
        (np.array([1 + 2j, 3 + 4j]), "event_times.*real"),
    ],
)
def test_recording_rejects_invalid_event_times(event_times, message):
    with pytest.raises(ValueError, match=message):
        Recording(
            stimulus=_two_words(),
            values=np.zeros((4, 1)),
            measure_name="eeg",
            modality="eeg",
            sample_interval=0.5,
            event_times=event_times,
        )
