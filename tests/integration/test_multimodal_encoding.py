"""MEG and fMRI recordings share the continuous encoder's ``evaluate`` entry point."""

import numpy as np
import pytest

from replica.analysis import EncodingConfig, evaluate
from replica.analysis.specification.evaluation import EvaluationSpec
from replica.core import AnnotationLayer, Modality, Recording, Stimulus
from replica.dataset import Dataset, DatasetInfo
from replica.feature import FeatureModelOutput, HiddenState, ModelFeature
from tests._stimuli import word_stimulus


def _stimulus(stimulus_id: str, n_words: int = 24) -> Stimulus:
    stimulus = word_stimulus(" ".join(f"w{i}" for i in range(n_words)), stimulus_id)
    spans = stimulus.layers["words"].spans
    stimulus.add_layer(
        AnnotationLayer("event_times", spans, list(np.arange(n_words, dtype=float)))
    )
    return stimulus


class MultimodalDataset(Dataset):
    info = DatasetInfo(name="multimodal", modalities=["eeg", "fmri", "meg"])

    def load(self, path: str = "") -> None:
        rng = np.random.default_rng(4)
        for run in range(2):
            stimulus = _stimulus(f"run-{run}")
            self._stimuli.append(stimulus)
            self.add_recording(
                Recording(
                    stimulus=stimulus,
                    values=rng.standard_normal((48, 3, 2)),
                    measure_name="field_strength",
                    modality=Modality.MEG,
                    participant_id="p1",
                    sample_interval=0.5,
                    dimension_names=("time", "sensor", "frequency"),
                    dimension_labels={"frequency": ("low", "high")},
                )
            )
            self.add_recording(
                Recording(
                    stimulus=stimulus,
                    values=rng.standard_normal((48, 4)),
                    measure_name="bold",
                    modality=Modality.FMRI,
                    participant_id="p1",
                    sample_interval=0.5,
                    dimension_names=("time", "vertex"),
                )
            )


class CountingEmbedding:
    name = "counting"
    vocab = None

    def __init__(self):
        self.calls = []

    def __call__(self, stimulus, extract=None, target_layer="words"):
        self.calls.append(stimulus.stimulus_id)
        return [
            FeatureModelOutput(values={"hidden_state_0": np.array([i, i + 1.0])})
            for i in range(len(stimulus.layers[target_layer]))
        ]


def test_multiple_sampled_targets_share_extraction_and_retain_axes():
    dataset = MultimodalDataset()
    dataset.load()
    model = CountingEmbedding()
    results = evaluate(
        dataset,
        EvaluationSpec(
            targets=[
                dataset.feature("field_strength", modality="meg"),
                dataset.feature("bold", modality="fmri"),
            ],
            predictors=[ModelFeature(model, HiddenState(0))],
        ),
        EncodingConfig(
            alignment_layer="event_times",
            fir_delays=(1,),
            n_folds=2,
            buffer_trs=1,
            alphas=(10.0,),
        ),
    )
    assert len(results) == 4
    assert model.calls == ["run-0", "run-1"]
    assert {
        (result.modality, result.response_measure, result.metadata["stimulus_id"])
        for result in results
    } == {
        (modality, quantity, run)
        for modality, quantity in (("meg", "field_strength"), ("fmri", "bold"))
        for run in ("run-0", "run-1")
    }
    for result in results:
        expected = (
            ((3, 2), ("time", "sensor", "frequency"), 6)
            if result.modality == "meg"
            else ((4,), ("time", "vertex"), 4)
        )
        shape, dimensions, channels = expected
        assert result.correlations.shape == (channels,)
        assert result.metadata["response_shape"] == shape
        assert result.metadata["response_dimensions"] == dimensions
        assert result.metrics["n_channels"] == channels


def test_dataset_shared_response_queries():
    dataset = MultimodalDataset()
    dataset.load()

    assert dataset.modalities == ["eeg", "fmri", "meg"]
    assert dataset.measures == ["bold", "field_strength"]
    assert dataset.participants == ["p1"]
    assert len(dataset.responses(modality="meg")) == 2
    assert len(dataset.responses(modality="meg", stimulus_id="run-0")) == 1
    assert dataset.responses(modality="eeg") == []


@pytest.mark.parametrize("quantity, modality", [("bold", "meg"), ("unknown", "fmri")])
def test_unavailable_target_fails_before_model_extraction(quantity, modality):
    dataset = MultimodalDataset()
    dataset.load()
    model = CountingEmbedding()
    spec = EvaluationSpec(
        targets=[dataset.feature(quantity, modality=modality)],
        predictors=[ModelFeature(model, HiddenState(0))],
    )
    with pytest.raises(ValueError, match="unavailable"):
        evaluate(dataset, spec, EncodingConfig())
    assert model.calls == []


@pytest.mark.parametrize("name", [None, ""])
def test_recording_selection_cannot_use_a_wildcard_quantity(name):
    with pytest.raises(ValueError, match="nonempty quantity name"):
        MultimodalDataset().feature(name, modality="meg")
