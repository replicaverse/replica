"""Call-scoped execution of declared feature graphs and model requests."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np

from replica.contracts.features import (
    UNIT_INDEX_NAMES,
    Feature,
    FeatureBlock,
    FeatureIndex,
    FeatureValues,
)
from replica.segmentation import segment_stimulus

from .core.binding import ModelFeature, _with_lineage


@dataclass(frozen=True)
class _OutputView:
    values: Mapping
    metadata: Mapping


def _freeze(value, memo):
    identity = id(value)
    if identity in memo:
        return memo[identity]
    result: Any
    if isinstance(value, np.ndarray):
        result = value.copy()
        result.flags.writeable = False
    elif isinstance(value, Mapping):
        result = MappingProxyType(
            {key: _freeze(item, memo) for key, item in value.items()}
        )
    elif isinstance(value, (tuple, list)):
        result = tuple(_freeze(item, memo) for item in value)
    else:
        return value
    memo[identity] = result
    return result


def _declared_material(dataset, stimulus):
    declaration = getattr(dataset, "_material_identity", None)
    return declaration(stimulus.metadata) if callable(declaration) else None


class _ResolutionContext:
    def __init__(
        self, dataset, requests, segmentation=None, layer=None, native_layers=()
    ):
        self._bindings = {id(dataset): "source:0"}
        self.dataset = dataset
        self.stimuli = tuple(dataset.stimuli())
        if layer is not None and segmentation is not None:
            raise ValueError("Specify either an existing layer or a segmentation")
        self.segmentation = segmentation
        self.layer = layer
        self._requested_layer = layer
        self._native_layers = frozenset(native_layers)
        self._stimuli = {s.stimulus_id: s for s in self.stimuli}
        if len(self._stimuli) != len(self.stimuli):
            raise ValueError("Material population has duplicate stimulus identities")
        self._layers = {}
        self._requests = requests
        self._outputs = {}
        self._active_batch: tuple[int, str, str] | None = None
        self._indexes = {}

    def source_identity(self, source):
        """Name a source binding within this evaluation, never a persistent cache key."""
        key = id(source)
        if key not in self._bindings:
            self._bindings[key] = f"source:{len(self._bindings)}"
        return self._bindings[key]

    def target_layer(self, stimulus):
        sid = stimulus.stimulus_id
        if sid not in self._layers:
            if self._requested_layer is not None:
                if self._requested_layer not in stimulus.layers:
                    raise ValueError(
                        f"Stimulus lacks requested layer {self._requested_layer!r}"
                    )
                self._layers[sid] = self._requested_layer
            elif self.segmentation is not None:
                self._layers[sid] = segment_stimulus(stimulus, self.segmentation)
            elif self._native_layers:
                if len(self._native_layers) != 1:
                    raise ValueError(
                        "Observations span multiple annotation layers; select an explicit layer or segmentation"
                    )
                native = next(iter(self._native_layers))
                if native not in stimulus.layers:
                    raise ValueError(
                        f"Stimulus lacks native observation layer {native!r}"
                    )
                self._layers[sid] = native
            elif "words" in stimulus.layers:
                self._layers[sid] = "words"
            elif len(stimulus.layers) == 1:
                self._layers[sid] = next(iter(stimulus.layers))
            else:
                raise ValueError(
                    "Select a segmentation for stimuli without an unambiguous native unit layer"
                )
        layer = self._layers[sid]
        if layer not in stimulus.layers and self.segmentation is not None:
            segment_stimulus(stimulus, self.segmentation)
        return layer

    def material_identity(self, stimulus, *, dataset=None):
        source = self.dataset if dataset is None else dataset
        selected = self._stimuli.get(stimulus.stimulus_id)
        if selected is None:
            raise ValueError(
                "Feature source contains a stimulus outside the selected material population"
            )
        if selected.text != stimulus.text:
            raise ValueError(
                "Equal stimulus labels do not establish equal material text"
            )
        source_key = _declared_material(source, stimulus)
        selected_key = _declared_material(self.dataset, selected)
        if source is not self.dataset and (
            source_key is None or source_key != selected_key
        ):
            raise ValueError(
                "Cross-dataset features require a shared declared material identity or an explicit mapping"
            )
        identity = (
            selected_key
            if selected_key is not None
            else f"dataset:{self.source_identity(self.dataset)}"
        )
        return f"{identity}:{hashlib.sha256(stimulus.text.encode()).hexdigest()}"

    def unit_index(self, stimulus, layer, *, dataset=None):
        material = self.material_identity(stimulus, dataset=dataset)
        selected = self._stimuli[stimulus.stimulus_id]
        if layer not in selected.layers and self.segmentation is not None:
            self.target_layer(selected)
        if (
            layer not in selected.layers
            or selected.layers[layer].spans != stimulus.layers[layer].spans
        ):
            raise ValueError(
                "Material annotation spans differ between feature source and selected population"
            )
        cache_key = (material, stimulus.stimulus_id, layer)
        if cache_key not in self._indexes:
            self._indexes[cache_key] = FeatureIndex(
                UNIT_INDEX_NAMES,
                tuple(
                    (material, stimulus.stimulus_id, layer, span.start, span.end, index)
                    for index, span in enumerate(stimulus.layers[layer].spans)
                ),
            )
        return self._indexes[cache_key]

    def model_outputs(self, request, stimulus, layer):
        key = (id(request.model), stimulus.stimulus_id, layer)
        if key != self._active_batch:
            raise ValueError(
                "Raw model outputs are available only inside the executor source batch; use resolved ModelFeature dependencies"
            )
        if id(request.model) not in self._requests:
            raise ValueError("A resolver requested an undeclared model source")
        if key not in self._outputs:
            model, required = self._requests[id(request.model)]
            outputs = model(
                stimulus,
                extract=None if required is None else list(required),
                target_layer=layer,
            )
            if len(outputs) != len(stimulus.layers[layer]):
                raise ValueError(
                    "Model outputs do not match the requested annotation units"
                )
            memo: dict[int, Any] = {}
            self._outputs[key] = tuple(
                _OutputView(
                    _freeze(output.values, memo), _freeze(output.metadata, memo)
                )
                for output in outputs
            )
        return self._outputs[key]


def _plan(features):
    ordered = []
    visiting = set()
    visited = set()
    requests = {}

    def visit(feature):
        key = id(feature)
        if key in visiting:
            raise ValueError("Feature dependency cycle")
        if key in visited:
            return
        if not callable(getattr(feature, "resolve", None)) or not hasattr(
            feature, "dependencies"
        ):
            raise TypeError(
                "Expected a bound Feature; wrap model extractors in ModelFeature(model, extractor)"
            )
        if not isinstance(feature.dependencies, tuple):
            raise TypeError("Feature dependencies must be an ordered tuple")
        visiting.add(key)
        for dependency in feature.dependencies:
            visit(dependency)
        declared_requests = getattr(feature, "model_requests", ())
        if declared_requests and (
            not isinstance(feature, ModelFeature)
            or type(feature).resolve is not ModelFeature.resolve
        ):
            raise TypeError(
                "Bind model requests through ModelFeature(model, extractor), then compose its resolved values"
            )
        for request in declared_requests:
            model_id = id(request.model)
            if model_id not in requests:
                requests[model_id] = (request.model, request.outputs)
            else:
                model, previous = requests[model_id]
                if previous != request.outputs and not getattr(
                    model, "supports_output_union", False
                ):
                    raise ValueError(
                        "Different output requests require a model declaring supports_output_union; shared stochastic outputs must remain unchanged"
                    )
                combined = (
                    None
                    if previous is None or request.outputs is None
                    else tuple(sorted(set(previous) | set(request.outputs)))
                )
                requests[model_id] = (model, combined)
        visiting.remove(key)
        visited.add(key)
        ordered.append(feature)

    for feature in features:
        visit(feature)
    return ordered, requests


def _checked_values(feature, result, dependencies, node, inputs):
    """Validate a node result and retain its position in the declared graph."""
    if not isinstance(result, FeatureValues):
        raise TypeError("Feature.resolve must return FeatureValues")
    if tuple(id(dep) for dep in feature.dependencies) != dependencies:
        raise ValueError("Feature dependencies changed during resolution")
    if (
        inputs
        and not any(result is value for value in inputs)
        and any(block.origins is None for block in result.blocks)
    ):
        raise ValueError(
            "Features with dependencies must return explicit contributor origins "
            "for every block; use block_origins(input_values, input_block) for "
            "pointwise transformations or return an input unchanged"
        )
    return _with_lineage(result, f"feature:{node}", inputs=inputs)


def _extract_model_values(context, ordered, model_materials=None):
    """Extract model-bound leaves one stimulus and source at a time."""
    model_features = [
        feature for feature in ordered if isinstance(feature, ModelFeature)
    ]
    model_blocks: dict[int, list[FeatureBlock]] = {
        id(feature): [] for feature in model_features
    }
    source_features: dict[int, list[ModelFeature]] = {}
    dependencies = {id(dep) for feature in ordered for dep in feature.dependencies}
    for feature in model_features:
        source_features.setdefault(id(feature.model), []).append(feature)
    for stimulus in context.stimuli:
        for source_group in source_features.values():
            active = source_group
            if (
                model_materials is not None
                and stimulus.stimulus_id not in model_materials
                and getattr(source_group[0].model, "supports_stimulus_selection", False)
                and all(
                    getattr(feature.extractor, "supports_stimulus_selection", False)
                    for feature in source_group
                )
            ):
                active = [
                    feature for feature in source_group if id(feature) in dependencies
                ]
            if not active:
                continue
            context._active_batch = (
                id(source_group[0].model),
                stimulus.stimulus_id,
                context.target_layer(stimulus),
            )
            try:
                for feature in active:
                    model_blocks[id(feature)].append(feature._block(context, stimulus))
            finally:
                context._outputs.clear()
                context._active_batch = None
    return {
        id(feature): feature._values(context, tuple(model_blocks[id(feature)]))
        for feature in model_features
    }


def _native_layers(features):
    """Find declared annotation units through a validated dependency graph."""
    layers: set[str] = set()
    pending = list(features)
    seen = set()
    while pending:
        feature = pending.pop()
        if id(feature) in seen:
            continue
        seen.add(id(feature))
        declared = getattr(feature, "native_layers", None)
        if declared is None:
            pending.extend(feature.dependencies)
        else:
            if isinstance(declared, str):
                raise TypeError("native_layers must contain nonempty layer names")
            declared = tuple(declared)
            if any(not isinstance(name, str) or not name for name in declared):
                raise TypeError("native_layers must contain nonempty layer names")
            layers.update(declared)
    return layers


def resolve_features(
    dataset,
    features: Iterable[Feature],
    *,
    segmentation=None,
    layer: str | None = None,
    layer_from: Iterable[Feature] | None = None,
) -> tuple[FeatureValues, ...]:
    """Resolve a graph once with shared, bounded model-output storage.

    Args:
        dataset: Selected material population.
        features: Bound quantities to resolve in result order.
        segmentation: Optional segmentation of selected materials.
        layer: Existing annotation layer; mutually exclusive with segmentation.
        layer_from: Features supplying native annotation units; defaults to features.
    """
    return _resolve_features(
        dataset,
        features,
        segmentation=segmentation,
        layer=layer,
        layer_from=layer_from,
    )


def _resolve_features(
    dataset,
    features,
    *,
    segmentation=None,
    layer=None,
    layer_from=None,
    model_materials=None,
):
    """Execute one graph, optionally narrowing direct model roots to materials.

    Dependencies keep their full declared population. The material restriction
    is used internally after encoding selects native recording targets.
    """
    features = tuple(features)
    ordered, requests = _plan(features)
    anchors = features if layer_from is None else tuple(layer_from)
    nodes = {id(feature) for feature in ordered}
    if any(id(anchor) not in nodes for anchor in anchors):
        raise ValueError("Layer sources must belong to the requested feature graph")
    native = _native_layers(anchors) if layer is None and segmentation is None else ()
    context = _ResolutionContext(dataset, requests, segmentation, layer, native)
    snapshots = {}
    for model_id, (model, _) in requests.items():
        configuration = getattr(model, "configuration", None)
        if callable(configuration):
            snapshots[model_id] = json.dumps(
                configuration(), sort_keys=True, allow_nan=False
            )
    definitions = {}
    for feature in ordered:
        configuration = getattr(feature, "configuration", None)
        definitions[id(feature)] = (
            tuple(id(dep) for dep in feature.dependencies),
            json.dumps(configuration(), sort_keys=True, allow_nan=False)
            if callable(configuration)
            else None,
        )
    resolved: dict[int, FeatureValues] = {}
    # Validate stored inputs and their compositions before expensive inference.
    for node, feature in enumerate(ordered):
        if isinstance(feature, ModelFeature) or any(
            id(dep) not in resolved for dep in feature.dependencies
        ):
            continue
        inputs = tuple(resolved[id(dep)] for dep in feature.dependencies)
        resolved[id(feature)] = _checked_values(
            feature,
            feature.resolve(context, inputs),
            definitions[id(feature)][0],
            node,
            inputs,
        )
    model_values = _extract_model_values(context, ordered, model_materials)
    for node, feature in enumerate(ordered):
        if id(feature) in resolved:
            continue
        dependencies = tuple(feature.dependencies)
        inputs = tuple(resolved[id(value)] for value in dependencies)
        result = (
            model_values[id(feature)]
            if id(feature) in model_values
            else feature.resolve(context, inputs)
        )
        resolved[id(feature)] = _checked_values(
            feature, result, definitions[id(feature)][0], node, inputs
        )
    for feature in ordered:
        dependencies, snapshot = definitions[id(feature)]
        if dependencies != tuple(id(dep) for dep in feature.dependencies):
            raise ValueError("Feature dependencies changed during resolution")
        if snapshot is not None and snapshot != json.dumps(
            feature.configuration(), sort_keys=True, allow_nan=False
        ):
            raise ValueError("Feature configuration changed during resolution")
    for model_id, snapshot in snapshots.items():
        if snapshot != json.dumps(
            requests[model_id][0].configuration(), sort_keys=True, allow_nan=False
        ):
            raise ValueError("Model configuration changed during feature resolution")
    return tuple(resolved[id(feature)] for feature in features)


def resolve(
    dataset, feature: Feature, *, segmentation=None, layer: str | None = None
) -> FeatureValues:
    """Resolve one feature independently of statistical evaluation.

    Args:
        dataset: Selected material population.
        feature: Bound quantity to resolve.
        segmentation: Optional segmentation of selected materials.
        layer: Existing annotation layer; mutually exclusive with segmentation.
    """
    return resolve_features(
        dataset, (feature,), segmentation=segmentation, layer=layer
    )[0]
