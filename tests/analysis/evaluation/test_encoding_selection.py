"""Participant selection avoids unrelated native responses and model calls."""

from dataclasses import replace

import numpy as np
import pytest

from replica.analysis import EncodingConfig, evaluate
from replica.analysis.evaluation import encoding
from replica.analysis.specification.evaluation import EvaluationSpec
from replica.contracts.features import FeatureValues, block_origins
from replica.core import Recording
from replica.dataset import Dataset, DatasetInfo
from replica.dataset.features import _DatasetFeature
from replica.feature import (
    FeatureModelOutput,
    ModelFeature,
    PrecomputedFeature,
    PrecomputedFeatureModel,
)
from tests._stimuli import word_stimulus


class _Population(Dataset):
    info = DatasetInfo(name="encoding_selection")

    def load(self, path="", *, shared=False):
        self._stimuli = [
            word_stimulus("a b c d e f", stimulus_id=f"s{i}") for i in range(2)
        ]
        recordings = [("p0", 0), ("p1", 1)]
        if shared:
            recordings.append(("p2", 0))
        for participant, stimulus in recordings:
            values = np.column_stack((np.arange(24), np.sin(np.arange(24))))
            self.add_recording(
                Recording(
                    stimulus=self._stimuli[stimulus],
                    values=values,
                    measure_name="bold",
                    modality="fmri",
                    participant_id=participant,
                    sample_interval=1,
                    event_times=np.arange(2, 24, 4),
                    dimension_names=("time", "channel"),
                )
            )


class _Model:
    name = "selection"
    supports_output_union = True
    supports_stimulus_selection = True

    def __init__(self, *, fail_excluded=False):
        self.fail_excluded = fail_excluded
        self.calls = []

    def __call__(self, stimulus, extract=None, target_layer="words"):
        self.calls.append((stimulus.stimulus_id, extract))
        if self.fail_excluded and stimulus.stimulus_id == "s1":
            raise ValueError("No predictor for excluded stimulus")
        offset = 10 if stimulus.stimulus_id == "s1" else 0
        return [
            FeatureModelOutput(values={"x": i + offset, "z": (i + offset) ** 2})
            for i in range(len(stimulus.layers[target_layer]))
        ]


def _config(participants):
    return EncodingConfig(
        participants=participants,
        n_folds=2,
        buffer_trs=0,
        fir_delays=(0,),
        kernel="impulse",
        alphas=(1,),
        alpha_chunk_length=1,
        alpha_nchunks=1,
    )


def _snapshots(monkeypatch):
    copied = []
    original = _DatasetFeature._recording_block

    def recording_block(self, recording, context):
        assert self.dataset is context.dataset
        block = original(self, recording, context)
        assert not np.shares_memory(block.values, recording.values)
        assert not block.values.flags.writeable
        copied.append(recording.participant_id)
        return block

    monkeypatch.setattr(_DatasetFeature, "_recording_block", recording_block)
    return copied


@pytest.mark.parametrize("participants", [("p0",), None])
def test_native_selection_limits_response_copies_and_model_calls(
    monkeypatch, participants
):
    dataset = _Population()
    dataset.load()
    model = _Model(fail_excluded=participants is not None)
    spec = EvaluationSpec(
        targets=[dataset.feature("bold", modality="fmri")],
        predictors=[ModelFeature(model, PrecomputedFeature("x", "x"))],
    )
    copied = _snapshots(monkeypatch)

    results = evaluate(dataset, spec, _config(participants))

    expected = ["p0"] if participants is not None else ["p0", "p1"]
    assert [result.participant_id for result in results] == expected
    assert [stimulus for stimulus, _ in model.calls] == [f"s{p[1]}" for p in expected]
    assert copied == expected
    assert len(dataset.recordings()) == 2


def test_selected_participants_share_one_model_batch_per_material(monkeypatch):
    dataset = _Population()
    dataset.load(shared=True)
    model = _Model(fail_excluded=True)
    spec = EvaluationSpec(
        targets=[dataset.feature("bold", modality="fmri")],
        predictors=[
            ModelFeature(model, PrecomputedFeature("x", "x")),
            ModelFeature(model, PrecomputedFeature("z", "z")),
        ],
    )
    copied = _snapshots(monkeypatch)

    results = evaluate(dataset, spec, _config(("p0", "p2")))

    assert [result.participant_id for result in results] == ["p0", "p2"]
    assert model.calls == [("s0", ["x", "z"])]
    assert copied == ["p0", "p2"]
    np.testing.assert_array_equal(results[0].correlations, results[1].correlations)


@pytest.mark.parametrize("stateful_component", ["model", "extractor"])
def test_undeclared_component_keeps_advancing_rng_outputs_for_later_selection(
    monkeypatch, stateful_component
):
    class AdvancingModel:
        name = "advancing"
        supports_output_union = True

        def __init__(self):
            self.rng = np.random.default_rng(153)
            self.calls = []

        def __call__(self, stimulus, extract=None, target_layer="words"):
            values = self.rng.normal(size=len(stimulus.layers[target_layer]))
            self.calls.append((stimulus.stimulus_id, values))
            return [FeatureModelOutput(values={"x": value}) for value in values]

    class AdvancingExtractor(PrecomputedFeature):
        def __init__(self):
            super().__init__("x", "x")
            self.rng = np.random.default_rng(153)
            self.calls = []

        def __call__(self, stimulus, outputs, target_layer="words"):
            values = self.rng.normal(size=len(stimulus.layers[target_layer]))
            self.calls.append((stimulus.stimulus_id, values))
            return values

    dataset = _Population()
    dataset.load()
    copied = _snapshots(monkeypatch)
    if stateful_component == "model":
        models = [AdvancingModel(), AdvancingModel()]
        extractors = [PrecomputedFeature("x", "x"), PrecomputedFeature("x", "x")]
        components = models
    else:
        outputs = {
            stimulus.stimulus_id: [
                FeatureModelOutput(values={"x": i}) for i in range(6)
            ]
            for stimulus in dataset.stimuli()
        }
        models = [PrecomputedFeatureModel("stored", outputs) for _ in range(2)]
        extractors = [AdvancingExtractor(), AdvancingExtractor()]
        components = extractors
    results = []
    for model, extractor, participants in zip(
        models, extractors, [None, ("p1",)], strict=True
    ):
        copied.clear()
        results.append(
            evaluate(
                dataset,
                EvaluationSpec(
                    targets=[dataset.feature("bold", modality="fmri")],
                    predictors=[ModelFeature(model, extractor)],
                ),
                _config(participants),
            )
        )

    assert copied == ["p1"]
    for component in components:
        assert [stimulus for stimulus, _ in component.calls] == ["s0", "s1"]
    for (_, full), (_, selected) in zip(
        components[0].calls, components[1].calls, strict=True
    ):
        np.testing.assert_array_equal(selected, full)
    assert (
        components[0].rng.bit_generator.state == components[1].rng.bit_generator.state
    )
    full = next(result for result in results[0] if result.participant_id == "p1")
    selected = results[1][0]
    np.testing.assert_array_equal(selected.correlations, full.correlations)
    np.testing.assert_array_equal(selected.r2, full.r2)


def test_precomputed_selection_skips_missing_material_without_opting_in_subclasses():
    class ObservedPrecomputed(PrecomputedFeatureModel):
        def __call__(self, stimulus, extract=None, target_layer="words"):
            calls.append(stimulus.stimulus_id)
            return super().__call__(stimulus, extract, target_layer)

    dataset = _Population()
    dataset.load()
    values = [FeatureModelOutput(values={"x": i}) for i in range(6)]
    calls = []
    models = [
        PrecomputedFeatureModel("stored", {"s1": values}),
        ObservedPrecomputed("stored", {"s0": values, "s1": values}),
    ]
    results = [
        evaluate(
            dataset,
            EvaluationSpec(
                targets=[dataset.feature("bold", modality="fmri")],
                predictors=[ModelFeature(model, PrecomputedFeature("x", "x"))],
            ),
            _config(("p1",)),
        )[0]
        for model in models
    ]
    assert calls == ["s0", "s1"]
    np.testing.assert_array_equal(results[0].correlations, results[1].correlations)


def test_custom_dependency_keeps_materials_outside_selected_recordings(monkeypatch):
    dataset = _Population()
    dataset.load()
    model = _Model()
    leaf = ModelFeature(model, PrecomputedFeature("x", "x"))
    populations = []

    class CenterPopulation:
        name = "centered"
        dependencies = (leaf,)

        def resolve(self, context, inputs):
            values = inputs[0]
            populations.append(
                tuple(block.stimulus.stimulus_id for block in values.blocks)
            )
            mean = np.mean(np.concatenate([block.values for block in values.blocks]))
            origins = frozenset().union(
                *(
                    row
                    for block in values.blocks
                    for row in block_origins(values, block)
                )
            )
            return FeatureValues(
                tuple(
                    replace(
                        block,
                        values=block.values - mean,
                        origins=tuple(origins for _ in block.index.keys),
                    )
                    for block in values.blocks
                ),
                provenance=values.provenance,
            )

    spec = EvaluationSpec(
        targets=[dataset.feature("bold", modality="fmri")],
        predictors=[CenterPopulation()],
    )
    copied = _snapshots(monkeypatch)

    results = evaluate(dataset, spec, _config(("p0",)))

    assert len(results) == 1
    assert populations == [("s0", "s1")]
    assert [stimulus for stimulus, _ in model.calls] == ["s0", "s1"]
    assert copied == ["p0"]


def test_selected_target_does_not_narrow_its_explicit_dependency_binding(monkeypatch):
    dataset = _Population()
    dataset.load()
    target = dataset.feature("bold", modality="fmri")
    populations = []
    identities = {}
    original = encoding._select_recordings

    def select_recordings(feature, values, config):
        identities["target"] = (
            values.provenance["dataset_binding"],
            values.provenance["outcome_identity"],
        )
        return original(feature, values, config)

    monkeypatch.setattr(encoding, "_select_recordings", select_recordings)

    class CopyTarget:
        name = "copy"
        dependencies = (target,)

        def resolve(self, context, inputs):
            populations.append(
                tuple(block.participant_id for block in inputs[0].blocks)
            )
            identities["dependency"] = (
                inputs[0].provenance["dataset_binding"],
                inputs[0].provenance["outcome_identity"],
            )
            return inputs[0]

    copied = _snapshots(monkeypatch)
    with pytest.raises(ValueError, match="cannot depend on target observations"):
        evaluate(
            dataset,
            EvaluationSpec(targets=[target], predictors=[CopyTarget()]),
            _config(("p0",)),
        )
    assert populations == [("p0", "p1")]
    assert copied == ["p0", "p0", "p1"]
    assert identities["target"] == identities["dependency"]
