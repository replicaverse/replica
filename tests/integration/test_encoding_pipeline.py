"""Integration tests for the fMRI encoding model pipeline.

Uses synthetic data, no real BOLD or HuggingFace models required.
"""

import numpy as np
import pytest

from replica.analysis import EncodingConfig, EncodingResult, evaluate
from replica.analysis.evaluation.encoding import (
    _zscore_columns,
    lanczos_interpolate,
    make_fir_delays,
)
from replica.analysis.specification.evaluation import EvaluationSpec
from replica.core import AnnotationLayer, Recording
from replica.dataset.base import Dataset
from replica.dataset.contracts import DatasetInfo
from replica.feature import FeatureModelOutput, HiddenState, ModelFeature
from tests._stimuli import word_stimulus

# ---------------------------------------------------------------------------
# Fixtures: synthetic dataset + feature model
# ---------------------------------------------------------------------------


def _make_stimulus(n_words=100, duration=150.0):
    """Create a stimulus with words and word_times layers."""
    stim = word_stimulus(" ".join(f"word{i}" for i in range(n_words)), "synthetic")

    # Evenly spaced word times across duration
    dt = duration / n_words
    word_times = [(i * dt, (i + 1) * dt) for i in range(n_words)]
    stim.add_layer(
        AnnotationLayer("word_times", stim.layers["words"].spans, word_times)
    )
    return stim


class SyntheticFMRIDataset(Dataset):
    """Dataset with synthetic native fMRI recordings."""

    info = DatasetInfo(name="synthetic_fmri")

    def __init__(self, n_words=100, n_trs=200, n_voxels=20, n_subjects=2):
        super().__init__()
        self._n_words = n_words
        self._n_trs = n_trs
        self._n_voxels = n_voxels
        self._n_subjects = n_subjects

    def load(self, path: str = "") -> None:
        rng = np.random.default_rng(42)
        stimulus = _make_stimulus(self._n_words, duration=self._n_trs * 1.5)
        self._stimuli = [stimulus]

        for i in range(self._n_subjects):
            bold = rng.standard_normal((self._n_trs, self._n_voxels))
            self.add_recording(
                Recording(
                    stimulus=stimulus,
                    values=bold,
                    measure_name="bold",
                    modality="fmri",
                    participant_id=f"sub-{i:02d}",
                    sample_interval=1.5,
                    dimension_names=("time", "voxel"),
                )
            )


class SyntheticEmbeddingModel:
    """Fake feature model that returns deterministic hidden states."""

    supports_output_union = True
    name = "synthetic_embedding"
    vocab = None

    def __init__(self, embed_dim=10, n_layers=3):
        self._embed_dim = embed_dim
        self._n_layers = n_layers

    def __call__(self, stimulus, extract=None, target_layer="words"):
        rng = np.random.default_rng(123)
        n_words = len(stimulus.layers[target_layer])
        outputs = []
        for _w in range(n_words):
            values = {}
            for layer in range(self._n_layers):
                values[f"hidden_state_{layer}"] = rng.standard_normal(self._embed_dim)
            outputs.append(FeatureModelOutput(values=values))
        return outputs


def _evaluate_sampled(dataset, model, *, layers, fir_delays, **kwargs):
    """Use the only public evaluation path in the integration tests."""
    participants = kwargs.pop("participants", None)
    config = EncodingConfig(
        fir_delays=tuple(fir_delays),
        participants=None if participants is None else tuple(participants),
        **kwargs,
    )
    spec = EvaluationSpec(
        predictors=[ModelFeature(model, HiddenState(layer)) for layer in layers],
        targets=[dataset.feature("bold", modality="fmri")],
    )
    return evaluate(dataset, spec, config)


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestEncodingPipeline:
    def test_basic_pipeline(self):
        """Full pipeline runs and returns results."""
        ds = SyntheticFMRIDataset(n_words=50, n_trs=100, n_voxels=10, n_subjects=1)
        ds.load()

        model = SyntheticEmbeddingModel(embed_dim=5, n_layers=2)

        results = _evaluate_sampled(
            ds,
            model,
            layers=[0, 1],
            fir_delays=range(1, 5),  # fewer delays for speed
            n_folds=2,
            buffer_trs=2,
        )

        # One result per (layer, participant)
        assert len(results) == 2  # 2 layers x 1 participant
        for r in results:
            assert isinstance(r, EncodingResult)
            assert r.dataset_name == "synthetic_fmri"
            assert r.model_name == "synthetic_embedding"
            assert r.correlations.shape == (10,)
            assert "mean_r" in r.metrics
            assert "n_finite_channels" in r.metrics
            assert "n_undefined_channels" in r.metrics
            assert (
                r.metrics["n_finite_channels"] + r.metrics["n_undefined_channels"]
                == r.metrics["n_channels"]
            )
            assert r.metrics["n_channels"] == 10

    def test_multi_subject(self):
        """Pipeline handles multiple subjects."""
        ds = SyntheticFMRIDataset(n_words=50, n_trs=100, n_voxels=5, n_subjects=3)
        ds.load()

        model = SyntheticEmbeddingModel(embed_dim=5, n_layers=1)

        results = _evaluate_sampled(
            ds,
            model,
            layers=[0],
            fir_delays=range(1, 5),
            n_folds=2,
            buffer_trs=2,
        )

        # 1 layer x 3 participants
        assert len(results) == 3
        pids = {r.participant_id for r in results}
        assert pids == {"sub-00", "sub-01", "sub-02"}

    def test_linear_bold_recovers_correlation(self):
        """When BOLD is a linear function of features, positive correlation is recovered."""
        rng = np.random.default_rng(42)

        n_words = 200
        n_trs = 400
        n_voxels = 5
        embed_dim = 8

        stimulus = _make_stimulus(n_words, duration=n_trs * 1.5)

        # Create "features" and linearly generate BOLD from them
        word_features = rng.standard_normal((n_words, embed_dim))

        # Get word midpoints
        word_times_layer = stimulus.layers["word_times"]
        word_midpoints = np.array(
            [(onset + offset) / 2 for _, (onset, offset) in word_times_layer]
        )

        # Interpolate to TR times
        tr_times = np.arange(n_trs) * 1.5
        feats_tr = lanczos_interpolate(word_features, word_midpoints, tr_times)
        feats_delayed = make_fir_delays(feats_tr, range(1, 5))
        feats_z = _zscore_columns(feats_delayed)

        # Generate BOLD as linear function + noise
        true_weights = rng.standard_normal((feats_z.shape[1], n_voxels))
        bold = feats_z @ true_weights + rng.standard_normal((n_trs, n_voxels)) * 0.5

        # Build dataset
        ds = SyntheticFMRIDataset(
            n_words=n_words, n_trs=n_trs, n_voxels=n_voxels, n_subjects=1
        )
        ds.load()
        # Replace BOLD with the test's linear signal
        ds._recordings[0] = Recording(
            stimulus=stimulus,
            values=bold,
            measure_name="bold",
            modality="fmri",
            participant_id="sub-00",
            sample_interval=1.5,
            dimension_names=("time", "voxel"),
        )

        # Feature model returns the same word_features used above
        class LinearFeatureModel:
            name = "linear_test"
            vocab = None

            def __call__(self, stim, extract=None, target_layer="words"):
                outputs = []
                for w in range(len(stim.layers[target_layer])):
                    outputs.append(
                        FeatureModelOutput(values={"hidden_state_0": word_features[w]})
                    )
                return outputs

        results = _evaluate_sampled(
            ds,
            LinearFeatureModel(),
            layers=[0],
            fir_delays=range(1, 5),
            n_folds=4,
            buffer_trs=5,
        )

        assert len(results) == 1
        r = results[0]
        # Should find positive mean correlation for the linear signal
        assert r.metrics["mean_r"] > 0.0
        assert r.metrics["n_positive_channels"] > 0

    def test_participant_filter(self):
        """participants kwarg filters which subjects are evaluated."""
        ds = SyntheticFMRIDataset(n_words=50, n_trs=100, n_voxels=5, n_subjects=3)
        ds.load()

        model = SyntheticEmbeddingModel(embed_dim=5, n_layers=1)

        results = _evaluate_sampled(
            ds,
            model,
            layers=[0],
            fir_delays=range(1, 3),
            n_folds=2,
            buffer_trs=2,
            participants=["sub-01"],
        )

        assert len(results) == 1
        assert results[0].participant_id == "sub-01"


class TestRecordingEventTimes:
    """A recording can provide its own timed units."""

    @staticmethod
    def _dataset(event_times, *, keep_layer=False):
        dataset = SyntheticFMRIDataset(n_words=40, n_voxels=5, n_subjects=1)
        dataset.load()
        stimulus = dataset._stimuli[0]
        if not keep_layer:
            del stimulus.layers["word_times"]
        recording = dataset._recordings[0]
        dataset._recordings[0] = Recording(
            stimulus=stimulus,
            values=recording.values,
            measure_name="bold",
            modality="fmri",
            participant_id=recording.participant_id,
            sample_interval=1.5,
            dimension_names=("time", "voxel"),
            event_times=event_times,
        )
        return dataset

    @staticmethod
    def _correlations(dataset):
        (result,) = _evaluate_sampled(
            dataset, SyntheticEmbeddingModel(), layers=[0], fir_delays=[1, 2]
        )
        return result.correlations

    def test_the_recording_onsets_replace_the_alignment_layer(self):
        with_layer = self._dataset(None, keep_layer=True)
        midpoints = np.array(
            [
                (onset + offset) / 2
                for _, (onset, offset) in with_layer._stimuli[0].layers["word_times"]
            ]
        )
        np.testing.assert_array_equal(
            self._correlations(self._dataset(midpoints)),
            self._correlations(with_layer),
        )

    def test_an_untimed_unit_is_left_out(self):
        times = np.linspace(0.0, 290.0, 40)
        times[-1] = np.nan
        assert self._correlations(self._dataset(times)).shape == (5,)

    def test_a_recording_without_onsets_needs_the_layer(self):
        with pytest.raises(ValueError, match="no temporal alignment layer"):
            self._correlations(self._dataset(None))

    def test_a_recording_with_no_timed_unit_is_rejected(self):
        with pytest.raises(ValueError, match="no timed events"):
            self._correlations(self._dataset(np.full(40, np.nan)))
