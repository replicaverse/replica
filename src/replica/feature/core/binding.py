"""Adapters from per-unit extractors to independently indexed quantities."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from typing import TYPE_CHECKING

from replica.contracts.features import (
    Feature,
    FeatureBlock,
    FeatureContext,
    FeatureValues,
    ModelRequest,
)
from replica.core import Stimulus
from replica.utils.arrays import array_with_missing

if TYPE_CHECKING:
    from .protocols import FeatureExtractor, FeatureModel

from .measurement import describe_feature
from .outputs.requirements import required_outputs


def _description(value) -> dict:
    declared = getattr(value, "configuration", None)
    return {
        "implementation": f"{type(value).__module__}.{type(value).__qualname__}",
        "configuration": declared() if callable(declared) else None,
    }


def _extractor_identity(extractor, context) -> str:
    configuration = getattr(extractor, "configuration", None)
    description = _description(extractor)
    if not callable(configuration):
        description["evaluation_binding"] = context.source_identity(extractor)
    return hashlib.sha256(json.dumps(description, sort_keys=True).encode()).hexdigest()


def resolve_stimulus_feature(
    feature: FeatureExtractor,
    context: FeatureContext,
    inputs: tuple[FeatureValues, ...],
) -> FeatureValues:
    """Resolve a unit calculation over selected materials.

    Args:
        feature: Callable stimulus-derived extractor.
        context: Selected materials and annotation layer.
        inputs: Empty tuple for this source-free calculation.
    """
    if inputs:
        raise ValueError("A stimulus calculation has no feature dependencies")
    blocks = []
    for stimulus in context.stimuli:
        layer = context.target_layer(stimulus)
        values = array_with_missing(feature(stimulus, None, layer))
        if values.ndim != 1:
            raise ValueError("A stimulus calculation must return one scalar per unit")
        blocks.append(
            FeatureBlock(
                values,
                context.unit_index(stimulus, layer),
                stimulus=stimulus,
                layer=layer,
                metadata=stimulus.metadata,
            )
        )
    source = getattr(feature, "_source", None)
    return FeatureValues(
        tuple(blocks),
        measurement=getattr(feature, "measurement", None),
        provenance={
            "identity_scope": "evaluation",
            "input_scope": "material",
            "outcome_identity": f"stimulus:{_extractor_identity(feature, context)}:{context.source_identity(source)}",
            "producer": _description(feature),
            "source": None if source is None else _description(source),
        },
    )


class ModelFeature:
    """Bind a model source to an extractor without assigning a statistical role."""

    dependencies: tuple = ()

    def __init__(
        self,
        model: FeatureModel,
        extractor: FeatureExtractor,
        *,
        name: str | None = None,
    ):
        """Bind the output producer and per-unit calculation.

        Args:
            model: Source of per-unit output records.
            extractor: Calculation consuming those records.
            name: Column alias; defaults to the extractor name.
        """
        if not callable(model) or not callable(extractor):
            raise TypeError("ModelFeature requires a callable model and extractor")
        self.model = model
        self.extractor = extractor
        self.name = extractor.name if name is None else name
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("A feature name must be a nonempty string")

    def configuration(self) -> dict:
        """Snapshot the explicit extraction settings independently of its alias."""
        return {
            "extractor": _description(self.extractor),
            "outputs": required_outputs([self.extractor]),
        }

    @property
    def measurement(self):
        """The extractor's declared quantity and unit, if known."""
        return getattr(self.extractor, "measurement", None)

    @property
    def model_requests(self) -> tuple[ModelRequest, ...]:
        """Source and raw output keys needed by this extractor."""
        keys = required_outputs([self.extractor])
        return (ModelRequest(self.model, None if keys is None else tuple(keys)),)

    def resolve(
        self, context: FeatureContext, inputs: tuple[FeatureValues, ...]
    ) -> FeatureValues:
        """Extract indexed values from the planned source.

        Args:
            context: Selected materials and shared model outputs.
            inputs: Empty tuple for a source-bound leaf.
        """
        if inputs:
            raise ValueError("A model feature has no feature dependencies")
        return self._values(
            context,
            tuple(self._block(context, stimulus) for stimulus in context.stimuli),
        )

    def _block(self, context: FeatureContext, stimulus: Stimulus) -> FeatureBlock:
        layer = context.target_layer(stimulus)
        outputs = context.model_outputs(self.model_requests[0], stimulus, layer)
        values = array_with_missing(self.extractor(stimulus, outputs, layer))
        dims = tuple(getattr(self.extractor, "value_dims", ()))
        if values.ndim > 1 and not dims:
            raise ValueError("A vector extractor must declare its value_dims")
        return FeatureBlock(
            values,
            context.unit_index(stimulus, layer),
            stimulus=stimulus,
            layer=layer,
            value_dims=dims,
            metadata=stimulus.metadata,
        )

    def _values(
        self, context: FeatureContext, blocks: tuple[FeatureBlock, ...]
    ) -> FeatureValues:
        provenance = {
            "identity_scope": "evaluation",
            "input_scope": "material",
            "outcome_identity": f"model:{context.source_identity(self.model)}:{_extractor_identity(self.extractor, context)}",
            "model_name": self.model.name,
            "model_metadata": dict(getattr(self.model, "metadata", {})),
            "source": _description(self.model),
            "extractor": describe_feature(self.extractor),
        }
        encoding_layer = getattr(self.extractor, "encoding_layer", None)
        if encoding_layer is not None:
            provenance["encoding_layer"] = encoding_layer
        return FeatureValues(
            tuple(blocks), measurement=self.measurement, provenance=provenance
        )


class _NamedFeature:
    def __init__(self, feature, name):
        if not isinstance(name, str) or not name:
            raise ValueError("A feature name must be a nonempty string")
        self.name = name
        self.dependencies = (feature,)

    @property
    def measurement(self):
        return getattr(self.dependencies[0], "measurement", None)

    def resolve(self, context, inputs):
        return inputs[0]


def named(feature: Feature, name: str) -> Feature:
    """Give a feature a column alias while preserving its quantity and origins.

    Args:
        feature: Bound quantity to alias.
        name: Nonempty output column name.
    """
    return _NamedFeature(feature, name)


def _with_lineage(
    values: FeatureValues,
    identity: str,
    *,
    inputs: tuple[FeatureValues, ...] = (),
) -> FeatureValues:
    """Complete runtime lineage for a resolved node without inventing source metadata."""
    provenance = dict(values.provenance)
    provenance.setdefault("outcome_identity", identity)
    provenance.setdefault("identity_scope", "evaluation")
    indexes = dict(provenance.get("origin_indexes", {}))
    scopes = dict(provenance.get("origin_scopes", {}))
    for source in inputs:
        for name, registry in (("origin_indexes", indexes), ("origin_scopes", scopes)):
            for origin, declaration in source.provenance.get(name, {}).items():
                if origin in registry and registry[origin] != declaration:
                    raise ValueError(
                        f"Conflicting feature lineage {name} for {origin!r}"
                    )
                registry[origin] = declaration
    identity = provenance["outcome_identity"]
    scopes.setdefault(identity, provenance.get("input_scope", "observations"))
    blocks = []
    for block in values.blocks:
        if identity in indexes and indexes[identity] != block.index.names:
            raise ValueError(
                "A feature outcome_identity cannot change its observation coordinates; "
                "use a new identity for remapped observations"
            )
        indexes[identity] = block.index.names
        if block.origins is not None and not provenance.get("transparent", False):
            origins = tuple(
                origin if (identity, key) in origin else origin | {(identity, key)}
                for origin, key in zip(block.origins, block.index.keys, strict=True)
            )
            blocks.append(
                block
                if all(a is b for a, b in zip(origins, block.origins, strict=True))
                else replace(block, origins=origins)
            )
        else:
            blocks.append(block)
    provenance["origin_indexes"] = indexes
    provenance["origin_scopes"] = scopes
    return replace(values, blocks=tuple(blocks), provenance=provenance)
