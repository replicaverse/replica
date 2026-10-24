"""Stored-feature identity, explicit calculations, and acquisition axes."""

import numpy as np
import pytest

from replica.contracts.features import align_values
from replica.core import AnnotationLayer, Datum, Recording, Span, Stimulus
from replica.dataset import Dataset, DatasetInfo
from replica.feature.resolution import resolve_features


class _Dataset(Dataset):
    additive_measures = frozenset({"total_rt"})

    def __init__(self, *, declared=False):
        super().__init__()
        self.info = DatasetInfo(
            name="example",
            language="en",
            material_namespace="materials" if declared else "",
            material_fields=["material"] if declared else [],
        )
        stimulus = Stimulus("a b", stimulus_id="s", metadata={"material": "one"})
        stimulus.add_layer(
            AnnotationLayer("words", [Span(0, 1), Span(2, 3)], ["a", "b"])
        )
        self._stimuli.append(stimulus)

    def load(self, path):
        raise NotImplementedError


def _datum(dataset, name="duration", participant="p", **metadata):
    datum = Datum(
        dataset.stimuli()[0],
        name,
        np.array([10.0, np.nan]),
        participant,
        metadata=metadata,
    )
    dataset.add_datum(datum)
    return datum


def test_selection_preserves_values_and_distinguishes_repeated_presentations():
    dataset = _Dataset()
    first = _datum(dataset, source_trial_index=1)
    _datum(dataset, source_trial_index=2)
    (values,) = resolve_features(dataset, [dataset.feature("duration", name="reading")])
    assert len(values.blocks) == 2
    assert "trial" in values.blocks[0].index.names
    assert values.blocks[0].index.keys != values.blocks[1].index.keys
    assert values.provenance["measure"] == "duration"
    np.testing.assert_equal(values.blocks[0].values, [10.0, np.nan])
    first.values[0] = 99
    assert values.blocks[0].values[0] == 10
    with pytest.raises(ValueError, match="read-only"):
        values.blocks[0].values[0] = 20
    stimulus = dataset.stimuli()[0]
    stimulus.add_layer(AnnotationLayer("sentence", [Span(0, 3)], ["a b"]))
    with pytest.raises(ValueError, match="not additive"):
        resolve_features(dataset, [dataset.feature("duration")], layer="sentence")
    total = _datum(dataset, name="total_rt")
    total.values[:] = [3, 4]
    (sentence,) = resolve_features(
        dataset, [dataset.feature("total_rt")], layer="sentence"
    )
    np.testing.assert_equal(sentence.blocks[0].values, [7])
    assert sentence.blocks[0].layer == "sentence"
    _datum(dataset, source_trial_index=1)
    with pytest.raises(ValueError, match="Duplicate observation"):
        resolve_features(dataset, [dataset.feature("duration")])


def test_cross_dataset_material_identity_does_not_establish_participant_identity():
    selected, source = _Dataset(), _Dataset()
    _datum(source, participant="")
    with pytest.raises(ValueError, match="Cross-dataset"):
        resolve_features(selected, [source.feature("duration")])
    selected, source = _Dataset(declared=True), _Dataset(declared=True)
    _datum(selected)
    _datum(source)
    left, right = resolve_features(
        selected, [selected.feature("duration"), source.feature("duration")]
    )
    assert np.isnan(align_values(right, left)[0].values).all()
    source.stimuli()[0].text = "a c"
    with pytest.raises(ValueError, match="material text"):
        resolve_features(selected, [source.feature("duration")])


def test_recording_clock_and_channel_axes_do_not_supply_amplitude_units():
    dataset = _Dataset()
    recording = Recording(
        dataset.stimuli()[0],
        np.arange(6.0).reshape(3, 2),
        "signal",
        "eeg",
        "p",
        sample_interval=0.01,
        dimension_names=("time", "channel"),
        dimension_labels={"channel": ("Cz", "Pz")},
    )
    dataset.add_recording(recording)
    (values,) = resolve_features(dataset, [dataset.feature("signal", modality="eeg")])
    block = values.blocks[0]
    assert values.measurement is None
    assert block.source is recording
    assert block.value_dims == ("channel",)
    assert block.value_labels == (("Cz", "Pz"),)
    np.testing.assert_equal(block.index.project(["time"]), [[0.0], [0.01], [0.02]])
    _datum(dataset, "signal")
    with pytest.raises(ValueError, match="unit and sampled"):
        resolve_features(dataset, [dataset.feature("signal")])
    (again,) = resolve_features(dataset, [dataset.feature("signal", modality="eeg")])
    np.testing.assert_equal(again.blocks[0].values, recording.values)
