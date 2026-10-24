"""Regressor bands, alignment kernels and recording-level folds of the encoding path."""

from dataclasses import replace

import numpy as np
import pytest

from replica.analysis import EncodingConfig, evaluate
from replica.analysis.evaluation.encoding import TemporalKFold, _align, _recording_folds
from replica.analysis.specification.evaluation import EvaluationSpec
from replica.core import AnnotationLayer, Recording
from replica.dataset.base import Dataset
from replica.dataset.contracts import DatasetInfo
from replica.feature import FeatureModelOutput, HiddenState, ModelFeature
from replica.segmentation import WhitespaceSegmenter
from tests._stimuli import word_stimulus

N_WORDS = 60
DT = 0.5
N_SAMPLES = 240
ONSETS = 2.0 * np.arange(N_WORDS) + 0.3
KERNEL = (1.0, 0.5, 0.25)


class _Column:
    """A scalar feature reading one key of the model outputs."""

    def __init__(self, name):
        self.name = name

    def __call__(self, stimulus, outputs, target_layer="words"):
        return np.array([out.values[self.name] for out in outputs])


class _Model:
    supports_output_union = True
    name = "synthetic"
    vocab = None
    metadata = {}

    def __init__(self, x, z):
        self.x, self.z = x, z

    def __call__(self, stimulus, extract=None, target_layer="words"):
        return [
            FeatureModelOutput(
                values={"x": x, "z": z, "hidden_state_0": np.array([x, z])}
            )
            for x, z in zip(self.x, self.z, strict=True)
        ]


def _impulse(values):
    """Impulse series of per-word values on the sample grid."""
    series = np.zeros(N_SAMPLES)
    np.add.at(series, np.round(ONSETS / DT).astype(int), values)
    return series


def _response(x, z, rng):
    """Channels: x through the kernel plus a little z, z alone, noise alone."""

    def convolve(series):
        return sum(
            weight * np.r_[np.zeros(delay), series[: N_SAMPLES - delay]]
            for delay, weight in enumerate(KERNEL)
        )

    noise = rng.normal(scale=0.2, size=(N_SAMPLES, 3))
    return (
        np.column_stack(
            [
                convolve(_impulse(x)) + 0.3 * convolve(_impulse(z)),
                convolve(_impulse(z)),
                np.zeros(N_SAMPLES),
            ]
        )
        + noise
    )


class _Dataset(Dataset):
    """One stimulus, two recordings (runs) of one participant, optionally more."""

    info = DatasetInfo(name="design")

    def __init__(self, extra_participants=(), nan_unit=None):
        super().__init__()
        rng = np.random.default_rng(3)
        self.x = rng.normal(size=N_WORDS)
        self.z = rng.normal(size=N_WORDS)
        if nan_unit is not None:
            self.x[nan_unit] = np.nan
        self.extra = extra_participants
        self.rng = rng

    def load(self, path=""):
        stimulus = word_stimulus(" ".join(f"w{i}" for i in range(N_WORDS)), "story")
        # An alignment span's event time is its midpoint; zero-length spans
        # keep the events at the onsets the response was generated from.
        stimulus.add_layer(
            AnnotationLayer(
                "word_times",
                stimulus.layers["words"].spans,
                [(onset, onset) for onset in ONSETS],
            )
        )
        self._stimuli = [stimulus]
        x = np.nan_to_num(self.x)
        for participant, runs in (("p1", 2), *((p, 1) for p in self.extra)):
            for run in range(runs):
                self.add_recording(
                    Recording(
                        stimulus=stimulus,
                        values=_response(x, self.z, self.rng),
                        measure_name="bold",
                        modality="fmri",
                        participant_id=participant,
                        sample_interval=DT,
                        dimension_names=("time", "channel"),
                        metadata={"run": run},
                    )
                )

    def spec(self, predictors=(), controls=()):
        model = _Model(self.x, self.z)
        return EvaluationSpec(
            predictors=[ModelFeature(model, feature) for feature in predictors],
            controls=[ModelFeature(model, feature) for feature in controls],
            targets=[self.feature("bold", modality="fmri")],
        )


def _config(**overrides):
    settings = {
        "kernel": "impulse",
        "fir_delays": (0, 1, 2),
        "alphas": (1.0,),
        "alpha_chunk_length": 5,
        "alpha_nchunks": 4,
        "n_folds": 3,
        "buffer_trs": 2,
    }
    settings.update(overrides)
    return EncodingConfig(**settings)


def test_controls_form_the_baseline_and_predictors_add_variance():
    dataset = _Dataset()
    dataset.load()
    spec = dataset.spec(predictors=[_Column("x")], controls=[_Column("z")])
    results = evaluate(dataset, spec, _config())
    assert [r.layer for r in results] == [None, None]
    for result in results:
        assert result.metadata["predictors"] == ["x"]
        assert result.metadata["controls"] == ["z"]
        assert [
            (feature["name"], feature["measurement"])
            for feature in result.metadata["feature_measurements"]
        ] == [("z", None), ("x", None)]
        # Channel 0 needs x, channel 1 is z alone, channel 2 is noise.
        delta = result.r2 - result.baseline_r2
        assert delta[0] > 0.5 and result.r2[0] > 0.8
        assert abs(delta[1]) < 0.1 and result.baseline_r2[1] > 0.8
        assert abs(delta[2]) < 0.1
        assert result.metrics["mean_delta_r2"] == pytest.approx(np.mean(delta))
        assert np.isfinite(result.baseline_correlations[:2]).all()


def test_scalar_predictors_without_controls():
    from replica.feature import DerivedFeature

    dataset = _Dataset()
    dataset.load()
    spec = dataset.spec(predictors=[_Column("x")])
    results = evaluate(dataset, spec, _config())
    assert results[0].baseline_r2 is None
    assert results[0].baseline_correlations is None
    assert "mean_delta_r2" not in results[0].metrics
    assert results[0].r2[0] > 0.7

    indicator = DerivedFeature("positive", lambda x: x > 0, spec.predictors[0])
    numeric = DerivedFeature("positive", lambda x: x.astype(float), indicator)
    actual = evaluate(dataset, replace(spec, predictors=[indicator]), _config())
    expected = evaluate(dataset, replace(spec, predictors=[numeric]), _config())
    for boolean, floating in zip(actual, expected, strict=True):
        np.testing.assert_allclose(boolean.correlations, floating.correlations)
        np.testing.assert_allclose(boolean.r2, floating.r2)
        assert boolean.baseline_r2 is None


def test_hidden_states_join_the_scalar_bands():
    dataset = _Dataset()
    dataset.load()
    spec = dataset.spec(
        predictors=[_Column("x"), HiddenState(0)], controls=[_Column("z")]
    )
    (first, _) = evaluate(dataset, spec, _config())
    assert first.layer == 0
    assert first.r2[0] > 0.8


def test_undefined_predictor_values_leave_the_unit_out():
    dataset = _Dataset(nan_unit=5)
    dataset.load()
    results = evaluate(dataset, dataset.spec(predictors=[_Column("x")]), _config())
    assert results[0].metadata["n_units"] == N_WORDS - 1


@pytest.mark.parametrize("missing", [np.nan, np.inf])
def test_nonfinite_responses_require_explicit_preprocessing(missing):
    dataset = _Dataset()
    dataset.load()
    dataset._recordings[0].values[3, 0] = missing
    with pytest.raises(ValueError, match="targets must be finite"):
        evaluate(dataset, dataset.spec(predictors=[_Column("x")]), _config())


def test_a_design_needs_a_predictor():
    dataset = _Dataset()
    dataset.load()
    with pytest.raises(ValueError, match="requires at least one predictor"):
        evaluate(dataset, dataset.spec(), _config())


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"comparison": "individual"}, "only joint"),
        ({"segmentation": WhitespaceSegmenter()}, "does not support segmentation"),
    ],
)
def test_encoding_rejects_unsupported_spec_options_before_extraction(
    changes, message, monkeypatch
):
    dataset = _Dataset()
    dataset.load()
    spec = replace(dataset.spec(predictors=[_Column("x")]), **changes)

    def unexpected_extraction(*args, **kwargs):
        raise AssertionError("Unsupported specification reached feature extraction")

    monkeypatch.setattr(_Model, "__call__", unexpected_extraction)
    with pytest.raises(ValueError, match=message):
        evaluate(dataset, spec, _config())


def test_a_feature_must_return_one_value_per_unit():
    dataset = _Dataset()
    dataset.load()

    class Short(_Column):
        def __call__(self, stimulus, outputs, target_layer="words"):
            return np.zeros(3)

    with pytest.raises(ValueError, match="one first-axis row per observation"):
        evaluate(dataset, dataset.spec(predictors=[Short("x")]), _config())


def test_impulse_kernel_places_units_on_their_nearest_samples():
    recording = Recording(
        stimulus=word_stimulus("a b c d", "s"),
        values=np.zeros((4, 1)),
        measure_name="bold",
        modality="fmri",
        sample_interval=DT,
    )
    values = np.array([[1.0], [2.0], [4.0], [8.0]])
    event_times = np.array([0.0, 0.74, 0.76, 2.4])
    design = _align(values, event_times, recording, _config())
    # 0.74 rounds down to sample 1, 0.76 up to sample 2; 2.4 lies beyond the
    # last sample plus half an interval and is dropped.
    np.testing.assert_array_equal(design[:, 0], [1.0, 2.0, 4.0, 0.0])
    shared = _align(values[:2], np.array([0.5, 0.5]), recording, _config())
    np.testing.assert_array_equal(shared[:, 0], [0.0, 3.0, 0.0, 0.0])


@pytest.mark.parametrize("clock_dtype", [None, np.float64, np.float32])
def test_delays_in_seconds_follow_the_sample_interval(clock_dtype):
    dataset = _Dataset()
    dataset.load()
    interval = 0.1 if clock_dtype is np.float32 else DT
    scale = interval / DT
    if clock_dtype is not None:
        for stimulus in dataset._stimuli:
            stimulus.layers["word_times"].values = [
                (start * scale, end * scale)
                for start, end in stimulus.layers["word_times"].values
            ]
        dataset._recordings = [
            replace(
                recording,
                sample_interval=interval,
                sample_times=(recording.times * scale).astype(clock_dtype),
            )
            for recording in dataset._recordings
        ]
    spec = dataset.spec(predictors=[_Column("x")])
    delays = (0.0, interval, 2 * interval)
    config = _config(fir_delays=delays, delay_unit="seconds")
    results = evaluate(dataset, spec, config)
    assert results[0].metadata["fir_delays"] == list(delays)
    assert results[0].metadata["fir_delays_samples"] == [0, 1, 2]
    assert results[0].r2[0] > 0.8
    irregular = _Dataset()
    irregular.load()
    recording = irregular._recordings[0]
    times = np.arange(N_SAMPLES) * DT
    times[::2] += DT / 4
    irregular._recordings[0] = Recording(
        stimulus=recording.stimulus,
        values=recording.values,
        measure_name="bold",
        modality="fmri",
        participant_id="p1",
        sample_times=times,
        dimension_names=("time", "channel"),
    )
    with pytest.raises(ValueError, match="regular recording"):
        evaluate(irregular, irregular.spec(predictors=[_Column("x")]), config)


def test_recording_folds_deal_round_robin():
    assert _recording_folds(5, 2) == [([1, 3], [0, 2, 4]), ([0, 2, 4], [1, 3])]
    assert _recording_folds(2, 4) == [([1], [0]), ([0], [1])]


def test_recording_cv_holds_out_whole_recordings():
    dataset = _Dataset()
    dataset.load()
    for i, recording in enumerate(dataset._recordings):
        recording.dimension_labels = {
            "channel": ("a", "b", "c"),
            "time": tuple(f"run{i}_{j}" for j in range(recording.n_samples)),
        }
    spec = dataset.spec(predictors=[_Column("x")], controls=[_Column("z")])
    (result,) = evaluate(dataset, spec, _config(cv="recording"))
    assert result.participant_id == "p1"
    assert result.metadata["stimulus_id"] == ["story", "story"]
    assert result.metadata["n_samples"] == 2 * N_SAMPLES
    assert result.metadata["response_dimension_labels"] == {"channel": ("a", "b", "c")}
    assert result.metadata["cv"] == "recording"
    assert result.metadata["recording_folds"] == [
        {"train": [1], "test": [0]},
        {"train": [0], "test": [1]},
    ]
    assert len(result.metadata["selected_alphas"]) == 2
    assert result.r2[0] > 0.8 and result.r2[0] - result.baseline_r2[0] > 0.5


@pytest.mark.parametrize(
    "dimensions, labels",
    [
        (("time", "channel"), {"channel": ("b", "a", "c")}),
        (("time", "voxel"), {"voxel": ("a", "b", "c")}),
    ],
)
def test_recording_cv_rejects_mismatched_response_coordinates(dimensions, labels):
    dataset = _Dataset()
    dataset.load()
    dataset._recordings[0].dimension_labels = {"channel": ("a", "b", "c")}
    dataset._recordings[1] = replace(
        dataset._recordings[1], dimension_names=dimensions, dimension_labels=labels
    )
    with pytest.raises(ValueError, match="align response coordinates"):
        evaluate(
            dataset, dataset.spec(predictors=[_Column("x")]), _config(cv="recording")
        )


def test_recording_cv_needs_two_comparable_recordings():
    single = _Dataset(extra_participants=("p2",))
    single.load()
    spec = single.spec(predictors=[_Column("x")])
    with pytest.raises(ValueError, match="at least two recordings"):
        evaluate(single, spec, _config(cv="recording"))
    dataset = _Dataset()
    dataset.load()
    spec = dataset.spec(predictors=[_Column("x")])
    with pytest.raises(ValueError, match="cv='temporal'"):
        evaluate(dataset, spec, _config(cv="recording"), encoding_fold_callback=print)
    narrow = dataset._recordings[1]
    dataset._recordings[1] = Recording(
        stimulus=narrow.stimulus,
        values=narrow.values[:, :2],
        measure_name="bold",
        modality="fmri",
        participant_id="p1",
        sample_interval=DT,
        dimension_names=("time", "channel"),
    )
    with pytest.raises(ValueError, match="differ in response shape"):
        evaluate(dataset, spec, _config(cv="recording"))


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"kernel": "boxcar"}, "kernel"),
        ({"delay_unit": "ms"}, "delay_unit"),
        ({"cv": "item"}, "cv"),
        ({"normalization": "global"}, "normalization"),
        ({"fir_delays": (0.5,)}, "integers"),
        ({"fir_delays": (-0.5,), "delay_unit": "seconds"}, "numbers"),
    ],
)
def test_config_rejects_unknown_settings(kwargs, message):
    with pytest.raises(ValueError, match=message):
        EncodingConfig(**kwargs)


def test_a_constant_control_changes_nothing():
    """A zero band adds a zero column, so the full model and its chunk
    shuffles match a run without controls to rounding."""
    dataset = _Dataset()
    dataset.load()

    class Zero(_Column):
        def __call__(self, stimulus, outputs, target_layer="words"):
            return np.zeros(len(outputs))

    plain = evaluate(dataset, dataset.spec(predictors=[_Column("x")]), _config())
    padded = evaluate(
        dataset,
        dataset.spec(predictors=[_Column("x")], controls=[Zero("zero")]),
        _config(alphas=(1.0, 3.0, 10.0)),
    )
    plain_alphas = evaluate(
        dataset,
        dataset.spec(predictors=[_Column("x")]),
        _config(alphas=(1.0, 3.0, 10.0)),
    )
    for a, b in zip(plain_alphas, padded, strict=True):
        np.testing.assert_allclose(a.correlations, b.correlations, rtol=1e-12)
        assert a.metadata["selected_alphas"] == b.metadata["selected_alphas"]
    assert plain[0].baseline_r2 is None and padded[0].baseline_r2 is not None
    response = dataset.recordings()[0].values
    observations, predictions = [], []
    for train, test in TemporalKFold(n_folds=3, buffer_trs=2).split(len(response)):
        observations.extend(response[test])
        predictions.extend(np.tile(response[train].mean(axis=0), (len(test), 1)))
    expected = [
        np.corrcoef(np.asarray(observations)[:, i], np.asarray(predictions)[:, i])[0, 1]
        for i in range(response.shape[1])
    ]
    np.testing.assert_allclose(padded[0].baseline_correlations, expected)


def test_recordings_with_their_own_onsets_pool_across_stimuli():
    """ZuCo-style trials: one recording per sentence, onsets on the
    recording, an untimed word, held out sentence by sentence."""
    rng = np.random.default_rng(11)
    dataset = _Dataset()
    dataset._stimuli = []
    dataset._recordings = []
    x_by_stimulus = {}
    for index in range(3):
        n_words = 20 + index
        stimulus = word_stimulus(
            " ".join(f"s{index}w{i}" for i in range(n_words)), f"s{index}"
        )
        onsets = 2.0 * np.arange(n_words) + 0.3
        event_times = onsets.copy()
        event_times[0] = np.nan
        x = rng.normal(size=n_words)
        x_by_stimulus[stimulus.stimulus_id] = x
        impulse = np.zeros(4 * n_words + 8)
        np.add.at(impulse, np.round(onsets[1:] / DT).astype(int), x[1:])
        response = sum(
            weight * np.r_[np.zeros(delay), impulse[: len(impulse) - delay]]
            for delay, weight in enumerate(KERNEL)
        )
        dataset._stimuli.append(stimulus)
        dataset.add_recording(
            Recording(
                stimulus=stimulus,
                values=(response + rng.normal(scale=0.2, size=len(response)))[:, None],
                measure_name="bold",
                modality="fmri",
                participant_id="p1",
                sample_interval=DT,
                event_times=event_times,
                dimension_names=("time", "channel"),
            )
        )

    class Model:
        name = "per-stimulus"
        vocab = None

        def __call__(self, stimulus, extract=None, target_layer="words"):
            return [
                FeatureModelOutput(values={"x": value})
                for value in x_by_stimulus[stimulus.stimulus_id]
            ]

    spec = EvaluationSpec(
        predictors=[ModelFeature(Model(), _Column("x"))],
        targets=[dataset.feature("bold", modality="fmri")],
    )
    (result,) = evaluate(dataset, spec, _config(cv="recording"))
    assert result.metadata["stimulus_id"] == ["s0", "s1", "s2"]
    assert result.metadata["n_units"] == 20 + 21 + 22 - 3
    assert len(result.metadata["selected_alphas"]) == 3
    assert result.r2[0] > 0.8


def test_recording_cv_preserves_unexplained_heldout_mean_shifts():
    dataset = _Dataset()
    dataset.load()
    response = dataset._recordings[1]
    dataset._recordings[1] = replace(response, values=response.values + 100)
    spec = dataset.spec(predictors=[_Column("x")], controls=[_Column("z")])
    (training,) = evaluate(dataset, spec, _config(cv="recording"))
    (within,) = evaluate(
        dataset, spec, _config(cv="recording", normalization="within_split")
    )
    assert training.metadata["normalization"] == "training"
    assert training.metadata["scoring_scale"] == "original_response"
    assert training.r2[0] < 0
    assert within.metadata["scoring_scale"] == "within_split"
    assert within.r2[0] > 0.8
