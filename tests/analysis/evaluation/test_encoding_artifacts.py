"""Numerical replay and isolation of streamed encoding fold artifacts."""

from dataclasses import fields, replace

import numpy as np
import pytest

from replica.analysis import (
    AnalysisConfig,
    EncodingConfig,
    EncodingFoldArtifact,
    evaluate,
)
from replica.analysis.evaluation.encoding import lanczos_interpolate, make_fir_delays
from replica.analysis.specification.evaluation import EvaluationSpec
from replica.core import AnnotationLayer, Recording, Span, Stimulus
from replica.dataset.base import Dataset
from replica.dataset.contracts import DatasetInfo
from replica.feature import FeatureModelOutput, HiddenState, ModelFeature


class _Recordings(Dataset):
    info = DatasetInfo(name="encoding_artifacts")

    def load(self, path=""):
        stimulus = Stimulus(" ".join(["a"] * 36), stimulus_id="story")
        spans = [Span(2 * i, 2 * i + 1) for i in range(36)]
        stimulus.add_layer(AnnotationLayer("words", spans, ["a"] * 36))
        stimulus.add_layer(AnnotationLayer("word_times", spans, np.arange(36) * 1.5))
        self._stimuli = [stimulus]
        rng = np.random.default_rng(47)
        for run in range(2):
            values = rng.normal(size=(36, 2, 2)) + np.arange(36)[:, None, None] / 5
            values[:, 1, 1] = 7
            self.add_recording(
                Recording(
                    stimulus=stimulus,
                    values=values,
                    modality="fmri",
                    measure_name="bold",
                    participant_id="participant",
                    sample_interval=1.5,
                    dimension_names=("time", "region", "channel"),
                    dimension_labels={"region": ("left", "right")},
                    metadata={"source": {"run": run}},
                )
            )


class _Features:
    supports_output_union = True
    name = "deterministic_features"
    vocab = None

    def __init__(self):
        self.values = np.random.default_rng(19).normal(size=(36, 3))
        self.values[:, -1] = 0

    def __call__(self, stimulus, extract=None, target_layer="words"):
        return [
            FeatureModelOutput(
                values={"hidden_state_0": row, "hidden_state_1": row * 2}
            )
            for row in self.values
        ]


@pytest.fixture
def encoding_inputs():
    dataset = _Recordings()
    dataset.load()
    model = _Features()
    features = EvaluationSpec(
        predictors=[ModelFeature(model, HiddenState(layer)) for layer in (0, 1)],
        targets=[dataset.feature("bold", modality="fmri")],
    )
    config = EncodingConfig(
        fir_delays=(0, 2),
        n_folds=3,
        buffer_trs=2,
        alphas=(0.1, 1.0, 10.0),
        alpha_chunk_length=2,
        alpha_nchunks=2,
        random_seed=31,
    )
    return dataset, features, config


def _assert_results_equal(actual, expected):
    assert len(actual) == len(expected)
    for observed, reference in zip(actual, expected, strict=True):
        np.testing.assert_equal(vars(observed), vars(reference))


@pytest.mark.parametrize("normalization", ["training", "within_split", "none"])
def test_fold_artifacts_replay_predictions_and_scores(
    encoding_inputs, tmp_path, normalization
):
    dataset, features, config = encoding_inputs
    config = replace(config, normalization=normalization)
    baseline = evaluate(dataset, features, config, language="en")
    artifacts = []

    def retain(artifact):
        assert isinstance(artifact, EncodingFoldArtifact)
        artifacts.append(artifact)
        arrays = {
            field.name: getattr(artifact, field.name)
            for field in fields(artifact)
            if isinstance(getattr(artifact, field.name), np.ndarray)
        }
        np.savez(tmp_path / f"fold-{len(artifacts)}.npz", **arrays)

    actual = evaluate(
        dataset, features, config, language="en", encoding_fold_callback=retain
    )
    _assert_results_equal(actual, baseline)
    assert len(artifacts) == 12
    recordings = dataset.recordings(measure="bold", modality="fmri")
    for number, artifact in enumerate(artifacts, start=1):
        recording = recordings[artifact.recording_index]
        assert artifact.dataset_name == dataset.info.name
        assert artifact.model_name == features.predictors[0].model.name
        assert artifact.participant_id == recording.participant_id
        assert artifact.stimulus_id == recording.stimulus.stimulus_id
        assert artifact.modality == "fmri"
        assert artifact.response_measure == "bold"
        assert artifact.language == "en"
        assert artifact.config == config
        assert artifact.response_shape == (2, 2)
        assert artifact.response_dimensions == ("time", "region", "channel")
        assert artifact.response_dimension_labels == {"region": ("left", "right")}
        assert artifact.recording_metadata == {
            "source": {"run": artifact.recording_index}
        }
        np.testing.assert_array_equal(
            artifact.test_indices,
            np.arange(artifact.fold_index * 12, (artifact.fold_index + 1) * 12),
        )
        assert np.min(abs(artifact.train_indices[:, None] - artifact.test_indices)) > 2
        np.testing.assert_array_equal(
            artifact.test_times, recording.times[artifact.test_indices]
        )

        feature_values = features.predictors[0].model.values * (artifact.layer + 1)
        design = make_fir_delays(
            lanczos_interpolate(feature_values, recording.times, recording.times),
            config.fir_delays,
        )
        response = recording.values.reshape(36, -1)
        np.testing.assert_array_equal(
            artifact.test_design, design[artifact.test_indices]
        )
        for name, values in (("feature", design), ("response", response)):
            for split in ("train", "test"):
                fitted_split = "train" if normalization == "training" else split
                selected = values[getattr(artifact, f"{fitted_split}_indices")]
                mean = (
                    selected.mean(axis=0)
                    if normalization != "none"
                    else np.zeros(values.shape[1])
                )
                scale = (
                    selected.std(axis=0)
                    if normalization != "none"
                    else np.ones(values.shape[1])
                )
                scale[scale == 0] = 1
                np.testing.assert_array_equal(
                    getattr(artifact, f"{name}_{split}_mean"), mean
                )
                np.testing.assert_array_equal(
                    getattr(artifact, f"{name}_{split}_scale"), scale
                )
        with np.load(tmp_path / f"fold-{number}.npz", allow_pickle=False) as saved:
            replayed = (
                (saved["test_design"] - saved["feature_test_mean"])
                / saved["feature_test_scale"]
            ) @ saved["weights"]
            np.testing.assert_array_equal(replayed, saved["predictions"])
            restored = (
                saved["observations"] * saved["response_test_scale"]
                + saved["response_test_mean"]
            )
            np.testing.assert_allclose(
                restored, response[saved["test_indices"]], atol=1e-15
            )

    for result_index, result in enumerate(actual):
        folds = artifacts[result_index * 3 : (result_index + 1) * 3]
        assert result.metadata["alpha_parameterization"] == "squared"
        assert result.metadata["candidate_alphas"] == list(config.alphas)
        assert result.metadata["selected_alphas"] == [fold.alpha for fold in folds]
        assert all(fold.alpha in config.alphas for fold in folds)
        assert result.metadata["normalization"] == normalization
        assert result.metadata["scoring_scale"] == (
            "within_split" if normalization == "within_split" else "original_response"
        )
        predictions = np.concatenate(
            [
                fold.predictions * fold.response_train_scale + fold.response_train_mean
                if normalization == "training"
                else fold.predictions
                for fold in folds
            ]
        )
        observations = np.concatenate(
            [
                fold.observations * fold.response_test_scale + fold.response_test_mean
                if normalization == "training"
                else fold.observations
                for fold in folds
            ]
        )
        with np.errstate(invalid="ignore", divide="ignore"):
            correlations = [
                np.corrcoef(observations[:, i], predictions[:, i])[0, 1]
                for i in range(4)
            ]
            total = np.sum((observations - observations.mean(axis=0)) ** 2, axis=0)
            expected_r2 = 1 - np.divide(
                np.sum((observations - predictions) ** 2, axis=0),
                total,
                out=np.full_like(total, np.nan),
                where=total > 0,
            )
        np.testing.assert_allclose(correlations, result.correlations, atol=1e-15)
        np.testing.assert_allclose(expected_r2, result.r2, atol=1e-15)
        np.testing.assert_array_equal(
            np.concatenate([fold.test_indices for fold in folds]), np.arange(36)
        )


def test_callback_cannot_mutate_evaluation_state(encoding_inputs):
    dataset, features, config = encoding_inputs
    baseline = evaluate(dataset, features, config)
    source_values = [item.values.copy() for item in dataset.recordings()]

    def overwrite(artifact):
        for field in fields(artifact):
            value = getattr(artifact, field.name)
            if isinstance(value, np.ndarray):
                value[...] = 0
        artifact.recording_metadata["source"]["run"] = "changed"
        artifact.response_dimension_labels.clear()

    actual = evaluate(dataset, features, config, encoding_fold_callback=overwrite)
    _assert_results_equal(actual, baseline)
    for recording, original in zip(dataset.recordings(), source_values, strict=True):
        np.testing.assert_array_equal(recording.values, original)


def test_callback_failure_stops_evaluation(encoding_inputs):
    calls = []

    def fail(artifact):
        calls.append(artifact.fold_index)
        raise OSError("artifact write failed")

    with pytest.raises(OSError, match="artifact write failed"):
        evaluate(*encoding_inputs, encoding_fold_callback=fail)
    assert calls == [0]


def test_scalar_analysis_rejects_encoding_callback(encoding_inputs):
    dataset, features, _config = encoding_inputs
    with pytest.raises(
        ValueError, match="encoding_fold_callback requires EncodingConfig"
    ):
        evaluate(
            dataset, features, AnalysisConfig(), encoding_fold_callback=lambda _: None
        )


def test_alpha_fallback_is_recorded_per_fold(encoding_inputs):
    """A fold too short for the alpha search takes the middle penalty, and the
    result says which folds did."""
    dataset, features, config = encoding_inputs
    for result in evaluate(dataset, features, config):
        assert result.metadata["alpha_selection"] == ["held_out"] * 3
    with pytest.warns(UserWarning, match="alpha_nchunks"):
        results = evaluate(dataset, features, replace(config, alpha_nchunks=50))
    for result in results:
        assert result.metadata["alpha_selection"] == ["fallback"] * 3
        assert result.metadata["selected_alphas"] == [1.0] * 3


def test_resolved_recording_target_values_reach_fits_with_native_clocks(
    encoding_inputs,
):
    from replica.feature import DerivedFeature

    dataset, spec, config = encoding_inputs
    target = DerivedFeature("scaled_bold", lambda values: values * 2, spec.targets[0])
    artifacts = []
    results = evaluate(
        dataset,
        replace(spec, targets=[target]),
        replace(config, normalization="none"),
        encoding_fold_callback=artifacts.append,
    )
    assert all(result.response_measure == "scaled_bold" for result in results)
    recordings = dataset.recordings(measure="bold", modality="fmri")
    for artifact in artifacts:
        native = recordings[artifact.recording_index]
        np.testing.assert_array_equal(
            artifact.test_times, native.times[artifact.test_indices]
        )
        np.testing.assert_array_equal(
            artifact.observations,
            2
            * native.values[artifact.test_indices].reshape(
                len(artifact.test_indices), -1
            ),
        )


@pytest.mark.parametrize(
    "transform",
    [lambda values: values + 1j, lambda values: values.astype(str)],
    ids=["complex", "text"],
)
def test_encoding_rejects_non_real_targets(encoding_inputs, transform):
    from replica.feature import DerivedFeature

    dataset, spec, config = encoding_inputs
    target = DerivedFeature("non_real", transform, spec.targets[0])
    with pytest.raises(ValueError, match="targets require real numeric values"):
        evaluate(dataset, replace(spec, targets=[target]), config)


def test_encoding_rejects_target_lineage_shared_across_temporal_folds(encoding_inputs):
    from replica.contracts.features import block_origins

    dataset, spec, config = encoding_inputs

    class OverlappingTarget:
        name = "overlapping_bold"
        dependencies = (spec.targets[0],)

        def resolve(self, context, inputs):
            values = inputs[0]
            blocks = []
            for block in values.blocks:
                originals = block_origins(values, block)
                shared = frozenset().union(*originals)
                blocks.append(replace(block, origins=tuple(shared for _ in originals)))
            return replace(values, blocks=tuple(blocks))

    with pytest.raises(ValueError, match="both training and test"):
        evaluate(dataset, replace(spec, targets=[OverlappingTarget()]), config)
