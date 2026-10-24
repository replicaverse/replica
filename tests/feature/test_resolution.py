"""Source planning, observation alignment and contributor-preserving composition."""

import json
import weakref
from dataclasses import replace

import numpy as np
import pytest

from replica.context import attach_preceding_context
from replica.contracts.features import (
    FeatureBlock,
    FeatureIndex,
    FeatureValues,
    align_values,
    block_origins,
)
from replica.contracts.measurement import FeatureMeasurement
from replica.core import Datum
from replica.dataset import Dataset, DatasetInfo
from replica.feature import (
    ContinuationEntropy,
    ContinuationEntropyLM,
    DerivedFeature,
    Entropy,
    FeatureModelOutput,
    Interaction,
    InUnit,
    Lag,
    MergedFeatureModel,
    ModelFeature,
    PrecedingContextModel,
    PrecomputedFeature,
    PrecomputedFeatureModel,
    SampledEntropy,
    Surprisal,
    UnitLength,
    aggregate,
    named,
    resolve,
    resolve_features,
)

from .._stimuli import word_stimulus


class _Dataset(Dataset):
    info = DatasetInfo(name="feature_contract")

    def __init__(self):
        super().__init__()
        self._stimuli = [word_stimulus("a longer end", stimulus_id="s")]

    def load(self, path):
        raise NotImplementedError


class _Stored(PrecomputedFeatureModel):
    supports_output_union = True

    def __init__(self):
        super().__init__(
            "stored",
            {
                "s": [
                    FeatureModelOutput(
                        values={
                            "log_p_observed": -x * np.log(2),
                            "next_log_probs": np.log([0.25, 0.75]),
                            "custom": x,
                        }
                    )
                    for x in [1.0, 2.0, 3.0]
                ]
            },
        )
        self.calls = []

    def __call__(self, stimulus, extract=None, target_layer="words"):
        self.calls.append(extract)
        return super().__call__(stimulus, extract, target_layer)


def test_bound_sources_share_requests_not_output_names_or_aliases():
    dataset = _Dataset()
    source, other = _Stored(), _Stored()
    other._outputs["s"][0].values["log_p_observed"] = -9 * np.log(2)
    target = ModelFeature(source, Surprisal())
    entropy = ModelFeature(source, Entropy())
    alias = named(target, "target_alias")
    converted = InUnit(target, "nats")
    values = resolve_features(
        dataset,
        [
            target,
            entropy,
            alias,
            converted,
            ModelFeature(other, Surprisal()),
            UnitLength(),
        ],
    )
    assert source.calls == [["log_p_observed", "next_log_probs"]]
    assert other.calls == [["log_p_observed"]]
    np.testing.assert_allclose(values[0].blocks[0].values, [1, 2, 3])
    np.testing.assert_allclose(
        values[1].blocks[0].values, -np.dot([0.25, 0.75], np.log2([0.25, 0.75]))
    )
    np.testing.assert_allclose(
        values[3].blocks[0].values, np.array([1, 2, 3]) * np.log(2)
    )
    assert values[4].blocks[0].values[0] == 9
    assert values[0] is values[2] or values[0].provenance == values[2].provenance
    assert values[0].blocks[0].index is values[5].blocks[0].index
    assert block_origins(values[3], values[3].blocks[0]) == block_origins(
        values[0], values[0].blocks[0]
    )
    a, b = resolve_features(
        dataset,
        [
            ModelFeature(source, PrecomputedFeature("a", "custom")),
            ModelFeature(source, PrecomputedFeature("b", "custom")),
        ],
    )
    assert a.provenance["outcome_identity"] == b.provenance["outcome_identity"]

    for wrapped in (MergedFeatureModel(source), PrecedingContextModel(source)):
        stimulus = dataset.stimuli()[0]
        stimulus.metadata["context_sequence"] = {"group": "story", "position": 1}
        attach_preceding_context([stimulus])
        source._outputs["s#context"] = source._outputs["s"]
        source.calls.clear()
        scored = resolve_features(
            dataset,
            [ModelFeature(wrapped, Surprisal()), ModelFeature(wrapped, Entropy())],
        )
        assert source.calls == [["log_p_observed", "next_log_probs"]]
        for actual, expected in zip(scored, values[:2], strict=True):
            np.testing.assert_array_equal(
                actual.blocks[0].values, expected.blocks[0].values
            )


def test_unverified_sampling_union_and_mutating_definitions_fail_closed():
    dataset = _Dataset()
    source = _Stored()
    source.supports_output_union = False
    with pytest.raises(ValueError, match="supports_output_union"):
        resolve_features(
            dataset,
            [ModelFeature(source, Surprisal()), ModelFeature(source, Entropy())],
        )
    assert not source.calls

    class RequestDependent(PrecomputedFeatureModel):
        calls = 0

        def __call__(self, stimulus, extract=None, target_layer="words"):
            self.calls += 1
            return [
                FeatureModelOutput({"log_p_observed": -float(len(extract or ()))})
                for _ in stimulus.layers[target_layer]
            ]

    unaudited = RequestDependent("custom", {})
    for producer in (
        unaudited,
        MergedFeatureModel(unaudited),
        PrecedingContextModel(unaudited),
    ):
        with pytest.raises(ValueError, match="supports_output_union"):
            resolve_features(
                dataset,
                [
                    ModelFeature(producer, Surprisal()),
                    ModelFeature(producer, Entropy()),
                ],
            )
    assert unaudited.calls == 0

    class UnauditedSampler(ContinuationEntropyLM):
        pass

    sampler = UnauditedSampler.__new__(UnauditedSampler)
    sampler.name = "custom-sampler"
    with pytest.raises(ValueError, match="supports_output_union"):
        resolve_features(
            dataset,
            [
                ModelFeature(sampler, ContinuationEntropy()),
                ModelFeature(sampler, SampledEntropy()),
            ],
        )

    class Mutating(UnitLength):
        def __init__(self):
            self.version = 0

        def configuration(self):
            return {"version": self.version}

        def __call__(self, *args):
            self.version += 1
            return super().__call__(*args)

    with pytest.raises(ValueError, match="configuration changed"):
        resolve(dataset, Mutating())
    cycle = DerivedFeature("cycle", lambda value: value, UnitLength())
    cycle._sources = (cycle,)
    with pytest.raises(ValueError, match="dependency cycle"):
        resolve(dataset, cycle)


def test_continuation_estimate_and_reductions_share_the_same_sample_realization():
    dataset = _Dataset()
    dataset._stimuli.append(word_stimulus("another short text", stimulus_id="t"))

    def sampler(seed):
        # Replace loading and sampling; keep the public driver and output reductions.
        model = ContinuationEntropyLM.__new__(ContinuationEntropyLM)
        model.name = "sample-fixture"
        model._start_prefix_ids, model._context_limit = (0,), 16
        model._seed = seed
        rng = np.random.default_rng(seed)
        calls, draws = [], []

        def prepare(stimulus, layer):
            units = len(stimulus.layers[layer])
            return list(range(units)), units, [(i, i + 1) for i in range(units)], False

        def sample_positions(tokens, positions):
            calls.append(tuple(tokens))
            samples = np.log(rng.uniform(0.1, 0.9, (len(positions), 5)))
            draws.append(samples)
            return {index: samples[index] for index, _ in positions}

        model._prepare_word_sampling = prepare
        model._sample_positions = sample_positions
        return model, calls, draws

    model, calls, draws = sampler(17)
    extractors = [ContinuationEntropy(), SampledEntropy(), SampledEntropy(alpha=2)]
    result = resolve_features(
        dataset, [ModelFeature(model, feature) for feature in extractors]
    )
    assert len(calls) == len(dataset.stimuli())
    for index, samples in enumerate(draws):
        expected_shannon = -samples.mean(axis=1) / np.log(2)
        np.testing.assert_allclose(result[0].blocks[index].values, expected_shannon)
        np.testing.assert_array_equal(
            result[1].blocks[index].values, result[0].blocks[index].values
        )
        np.testing.assert_allclose(
            result[2].blocks[index].values, -np.log2(np.exp(samples).mean(axis=1))
        )

    reversed_model, reversed_calls, _ = sampler(17)
    reversed_result = resolve_features(
        dataset,
        [ModelFeature(reversed_model, feature) for feature in reversed(extractors)],
    )
    assert len(reversed_calls) == len(calls)
    for actual, expected in zip(reversed(reversed_result), result, strict=True):
        for actual_block, expected_block in zip(
            actual.blocks, expected.blocks, strict=True
        ):
            np.testing.assert_array_equal(actual_block.values, expected_block.values)
    alone, alone_calls, _ = sampler(17)
    single = resolve(dataset, ModelFeature(alone, ContinuationEntropy()))
    assert len(alone_calls) == len(calls)
    for actual, expected in zip(single.blocks, result[0].blocks, strict=True):
        np.testing.assert_array_equal(actual.values, expected.values)


def test_shared_model_records_and_source_backed_views_cannot_be_mutated():
    dataset = _Dataset()
    source = _Stored()

    class MutatingExtractor(Surprisal):
        def __call__(self, stimulus, outputs, target_layer="words"):
            outputs[0].values["log_p_observed"] = 0

    with pytest.raises(TypeError):
        resolve(dataset, ModelFeature(source, MutatingExtractor()))
    assert source._outputs["s"][0].values["log_p_observed"] == -np.log(2)
    value = resolve(dataset, UnitLength())
    original = np.array([1.0, 2.0, 3.0])
    view = original.view()
    view.flags.writeable = False
    block = replace(value.blocks[0], values=view)
    original[0] = 99
    assert block.values[0] == 1
    names, keys = list(block.index.names), [list(key) for key in block.index.keys]
    index = FeatureIndex(names, keys)
    labels, dimensions = [["Cz", "Pz"]], ["channel"]
    origins = [set(row) for row in block_origins(value, block)]
    metadata = {"nested": {"labels": ["original"]}}
    vector = FeatureBlock(
        np.ones((3, 2)),
        index,
        stimulus=block.stimulus,
        value_dims=dimensions,
        value_labels=labels,
        origins=origins,
        metadata=metadata,
    )
    supplied = [vector]
    description = FeatureMeasurement("custom", None, {"setting": [1]})
    envelope = FeatureValues(
        supplied, measurement=description, provenance={"nested": metadata}
    )
    names[0], keys[0][0], labels[0][0], dimensions[0] = (
        "changed",
        "changed",
        "changed",
        "changed",
    )
    origins[0].clear()
    metadata["nested"]["labels"][0] = "changed"
    supplied.clear()
    description.parameters["setting"][0] = 99
    assert envelope.measurement.parameters["setting"] == (1,)
    with pytest.raises(TypeError):
        envelope.measurement.parameters["setting"] = ()
    assert envelope.blocks == (vector,)
    assert vector.index == block.index
    assert vector.value_dims == ("channel",)
    assert vector.value_labels == (("Cz", "Pz"),)
    assert vector.origins[0] == block_origins(value, block)[0]
    assert vector.metadata["nested"]["labels"] == ("original",)
    json.dumps(dict(envelope.provenance))
    with pytest.raises(TypeError):
        envelope.provenance["nested"]["nested"]["labels"] = ()

    class Contaminating:
        name = "contaminating"
        dependencies = (UnitLength(),)

        def resolve(self, context, inputs):
            inputs[0].provenance["origin_scopes"]["invented"] = "material"
            return inputs[0]

    with pytest.raises(TypeError):
        resolve(dataset, Contaminating())


def test_observation_composition_reduces_explicitly_and_preserves_lag_origins():
    dataset = _Dataset()
    stimulus = dataset.stimuli()[0]
    for participant, raw in [("p", [10, np.nan, 30]), ("q", [20, 40, 60])]:
        dataset.add_datum(Datum(stimulus, "rt", np.array(raw), participant))
    rt = dataset.feature("rt")
    raw, lagged, interaction, pooled = resolve_features(
        dataset,
        [
            rt,
            Lag(rt),
            Interaction(rt, UnitLength()),
            aggregate(rt, by="unit", statistic="mean", missing="drop"),
        ],
    )
    np.testing.assert_equal(lagged.blocks[0].values, [np.nan, 10, np.nan])
    np.testing.assert_equal(interaction.blocks[0].values, [10, np.nan, 90])
    np.testing.assert_equal(pooled.blocks[0].values, [15, 40, 45])
    assert "participant" not in pooled.blocks[0].index.names
    assert pooled.blocks[0].metadata["contributor_counts"] == (2, 1, 2)
    assert lagged.blocks[0].origins[1] == block_origins(raw, raw.blocks[0])[0]
    contributors = frozenset().union(
        *(block_origins(raw, block)[0] for block in raw.blocks)
    )
    assert contributors < pooled.blocks[0].origins[0]
    assert len(pooled.blocks[0].origins[0] - contributors) == 1

    class Masked:
        name = "masked"
        dependencies = (UnitLength(),)

        def resolve(self, context, inputs):
            value = inputs[0]
            return replace(
                value,
                blocks=(
                    replace(
                        value.blocks[0],
                        eligible=[True, False, True],
                        origins=block_origins(value, value.blocks[0]),
                    ),
                ),
            )

    shifted = resolve(dataset, Lag(Masked())).blocks[0]
    np.testing.assert_equal(shifted.values, [np.nan, 1, 6])
    np.testing.assert_equal(shifted.eligible, [False, True, False])
    with pytest.raises(ValueError, match="aggregate or map"):
        align_values(raw, pooled)


def test_alignment_preserves_categorical_values_and_validates_axis_names():
    dataset = _Dataset()
    base = resolve(dataset, UnitLength())
    block = base.blocks[0]
    order = [2, 0, 1]
    categorical = FeatureValues(
        (
            replace(
                block,
                values=np.array(["c", "a", "b"]),
                index=FeatureIndex(
                    block.index.names, tuple(block.index.keys[i] for i in order)
                ),
                eligible=block.eligible[order],
            ),
        ),
        provenance=base.provenance,
    )
    np.testing.assert_equal(align_values(categorical, base)[0].values, ["a", "b", "c"])
    masked = replace(
        categorical,
        blocks=(
            replace(
                categorical.blocks[0],
                values=np.ma.array(["c", "a", "b"], mask=[False, False, True]),
            ),
        ),
    )
    aligned = align_values(masked, base)[0].values
    assert aligned.dtype == object
    assert aligned[0] == "a" and np.isnan(aligned[1]) and aligned[2] == "c"
    with pytest.raises(ValueError, match="eligibility must be a boolean vector"):
        replace(block, eligible=np.ma.array([True] * 3, mask=[False, True, False]))
    with pytest.raises(ValueError, match="coordinate names"):
        FeatureIndex(("",), ((1,),))
    with pytest.raises(ValueError, match="unique names"):
        FeatureBlock(
            np.ones((3, 2, 2)),
            base.blocks[0].index,
            stimulus=dataset.stimuli()[0],
            value_dims=("channel", "channel"),
        )


def test_masked_stimulus_and_composed_outputs_remain_missing():
    dataset = _Dataset()

    class MaskedLength(UnitLength):
        def __call__(self, stimulus, outputs, target_layer="words"):
            values = super().__call__(stimulus, outputs, target_layer)
            return np.ma.array(values, mask=[False, True, False])

    source = MaskedLength()
    derived = DerivedFeature(
        "masked_length",
        lambda values: np.ma.array(values, mask=[False, True, False]),
        UnitLength(),
    )
    for feature in (source, derived):
        block = resolve(dataset, feature).blocks[0]
        np.testing.assert_equal(block.values, [1.0, np.nan, 3.0])
        assert type(block.values) is np.ndarray


@pytest.mark.parametrize("reverse", [False, True])
def test_vector_composition_rejects_axis_broadcasting_but_accepts_scalars(reverse):
    dataset = _Dataset()

    class Vector:
        dependencies = (UnitLength(),)

        def __init__(self, width):
            self.name = f"vector_{width}"
            self.width = width

        def resolve(self, context, inputs):
            return replace(
                inputs[0],
                blocks=tuple(
                    replace(
                        block,
                        values=np.broadcast_to(
                            np.arange(1, self.width + 1),
                            (len(block.index), self.width),
                        ),
                        value_dims=("channel",),
                        origins=block_origins(inputs[0], block),
                    )
                    for block in inputs[0].blocks
                ),
            )

    wide, narrow = Vector(2), Vector(1)
    operands = (narrow, wide) if reverse else (wide, narrow)
    with pytest.raises(ValueError, match="axis lengths"):
        resolve(dataset, Interaction(*operands))

    operands = (UnitLength(), wide) if reverse else (wide, UnitLength())
    result = resolve(dataset, Interaction(*operands))
    assert result.blocks[0].value_dims == ("channel",)
    np.testing.assert_array_equal(result.blocks[0].values, [[1, 2], [6, 12], [3, 6]])


@pytest.mark.parametrize(
    "source_values",
    [
        np.array([1, 2, 3]),
        np.array(["a", "b", "c"]),
        np.array([1 + 2j, 3 + 4j, 5 + 6j]),
    ],
    ids=["integer", "categorical", "complex"],
)
def test_lag_preserves_value_types_and_missing_boundaries(source_values):
    dataset = _Dataset()

    class Values:
        name = "values"
        dependencies = (UnitLength(),)

        def resolve(self, context, inputs):
            return replace(
                inputs[0],
                blocks=(
                    replace(
                        inputs[0].blocks[0],
                        values=source_values,
                        origins=block_origins(inputs[0], inputs[0].blocks[0]),
                    ),
                ),
            )

    feature = Values()
    source, shifted = resolve_features(dataset, [feature, Lag(feature)])
    block = shifted.blocks[0]
    assert np.isnan(block.values[0])
    np.testing.assert_array_equal(block.values[1:], source_values[:-1])
    np.testing.assert_array_equal(block.eligible, [False, True, True])
    assert block.origins[0] == frozenset()
    assert block.origins[1:] == block_origins(source, source.blocks[0])[:-1]


@pytest.mark.parametrize("conflict", [None, "origin_indexes", "origin_scopes"])
def test_custom_composition_retains_origin_registries_and_aliases(conflict):
    dataset = _Dataset()
    source = UnitLength()

    class Doubled:
        name = "doubled"
        dependencies = (source,)

        def resolve(self, context, inputs):
            value = inputs[0]
            provenance = {}
            if conflict is not None:
                provenance[conflict] = {
                    value.provenance["outcome_identity"]: ("changed",)
                    if conflict == "origin_indexes"
                    else "observations"
                }
            return FeatureValues(
                tuple(
                    replace(
                        block,
                        values=2 * block.values,
                        origins=block_origins(value, block),
                    )
                    for block in value.blocks
                ),
                provenance=provenance,
            )

    doubled = Doubled()
    if conflict is not None:
        with pytest.raises(ValueError, match="Conflicting feature lineage"):
            resolve(dataset, doubled)
        return
    original, result, alias = resolve_features(
        dataset, [source, doubled, named(doubled, "alias")]
    )
    np.testing.assert_array_equal(result.blocks[0].values, [2, 12, 6])
    for field in ("origin_indexes", "origin_scopes"):
        for identity, declaration in original.provenance[field].items():
            assert result.provenance[field][identity] == declaration
    assert alias.blocks[0] is result.blocks[0]
    assert alias.provenance == result.provenance


def test_custom_remapping_cannot_reuse_a_source_observation_identity():
    dataset = _Dataset()
    stimulus = dataset.stimuli()[0]
    for participant, values in (("p", [1, 2, 3]), ("q", [3, 4, 5])):
        dataset.add_datum(Datum(stimulus, "rt", np.array(values), participant))
    source = dataset.feature("rt")
    pooled = aggregate(source, by="unit", statistic="mean", missing="drop")

    class Remapped:
        name = "remapped"
        dependencies = (source, pooled)

        def resolve(self, context, inputs):
            original, reduced = inputs
            return replace(reduced, provenance=original.provenance)

    with pytest.raises(ValueError, match="cannot change its observation coordinates"):
        resolve(dataset, Remapped())
    result = resolve(dataset, pooled)
    np.testing.assert_array_equal(result.blocks[0].values, [2, 3, 4])
    assert "participant" not in result.blocks[0].index.names
    assert any(
        "participant" in coordinates
        for coordinates in result.provenance["origin_indexes"].values()
    )


def test_model_requests_release_raw_arrays_before_the_next_stimulus():
    dataset = _Dataset()
    dataset._stimuli = [word_stimulus("a b", stimulus_id=f"s{i}") for i in range(4)]
    live = []

    class Source:
        name = "streaming"
        supports_output_union = True

        def __init__(self):
            self.calls = 0

        def __call__(self, stimulus, extract=None, target_layer="words"):
            assert all(reference() is None for reference in live)
            self.calls += 1
            return [
                FeatureModelOutput(values={"next_log_probs": np.log([0.25, 0.75])})
                for _ in stimulus.layers[target_layer]
            ]

    class Inspect(Entropy):
        def __call__(self, stimulus, outputs, target_layer="words"):
            live.extend(
                weakref.ref(output.values["next_log_probs"]) for output in outputs
            )
            return super().__call__(stimulus, outputs, target_layer)

    model = Source()
    first, second = resolve_features(
        dataset,
        [
            ModelFeature(model, Inspect()),
            ModelFeature(model, Entropy(), name="entropy_copy"),
        ],
    )
    assert model.calls == 4
    assert all(reference() is None for reference in live)
    for left, right in zip(first.blocks, second.blocks, strict=True):
        np.testing.assert_allclose(
            left.values, -np.dot([0.25, 0.75], np.log2([0.25, 0.75]))
        )
        np.testing.assert_equal(left.values, right.values)

    class RawBypass:
        name = "bypass"
        dependencies = (ModelFeature(model, Entropy()),)

        def resolve(self, context, inputs):
            stimulus = context.stimuli[0]
            context.model_outputs(
                self.dependencies[0].model_requests[0],
                stimulus,
                context.target_layer(stimulus),
            )
            return inputs[0]

    with pytest.raises(ValueError, match="executor source batch"):
        resolve(dataset, RawBypass())
    assert model.calls == 8
    assert all(reference() is None for reference in live)
