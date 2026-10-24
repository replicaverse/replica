"""Indexed quantities shared by data adapters, features and analysis."""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field, replace
from typing import Any, Protocol, runtime_checkable

import numpy as np

from replica.core import Datum, Recording, Stimulus
from replica.utils.arrays import array_with_missing

from .measurement import FeatureMeasurement

ObservationKey = tuple[Hashable, ...]
OutcomeIdentity = tuple[str, ObservationKey]
UNIT_INDEX_NAMES = ("material", "stimulus", "layer", "start", "end", "unit")


class _FrozenMetadata(dict):
    """Read-only metadata mappings that retain ordinary JSON encoding."""

    def _immutable(self, *args, **kwargs):
        raise TypeError("Feature metadata is read-only")

    __setitem__ = __delitem__ = __ior__ = _immutable
    clear = pop = popitem = setdefault = update = _immutable

    def __deepcopy__(self, memo):
        return self

    def __reduce__(self):
        return _freeze_metadata, (dict(self),)


def _freeze_metadata(value):
    if isinstance(value, _FrozenMetadata):
        return value
    if isinstance(value, Mapping):
        return _FrozenMetadata(
            {key: _freeze_metadata(item) for key, item in value.items()}
        )
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_metadata(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze_metadata(item) for item in value)
    if isinstance(value, np.ndarray):
        value = value.copy()
        value.flags.writeable = False
        return value
    return deepcopy(value)


def _tuple(values, label):
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{label} requires a sequence, not a string")
    return tuple(values)


@dataclass(frozen=True)
class FeatureIndex:
    """Named observation coordinates, independent of the measured quantity."""

    names: tuple[str, ...]
    keys: tuple[ObservationKey, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "names", _tuple(self.names, "Coordinate names"))
        object.__setattr__(
            self,
            "keys",
            tuple(
                _tuple(key, "Observation key")
                for key in _tuple(self.keys, "Observation keys")
            ),
        )
        if (
            not self.names
            or any(not isinstance(name, str) or not name for name in self.names)
            or len(set(self.names)) != len(self.names)
        ):
            raise ValueError("Observation coordinate names must be nonempty and unique")
        if any(len(key) != len(self.names) for key in self.keys):
            raise ValueError("Observation keys must match the coordinate names")
        if len(set(self.keys)) != len(self.keys):
            raise ValueError("Duplicate observation keys")

    def project(self, names: Sequence[str]) -> tuple[ObservationKey, ...]:
        """Select coordinates without adding or reducing observations.

        Args:
            names: Coordinate names in the requested order.
        """
        names = tuple(names)
        if names == self.names:
            return self.keys
        positions = tuple(self.names.index(name) for name in names)
        return tuple(tuple(key[pos] for pos in positions) for key in self.keys)

    def __len__(self) -> int:
        return len(self.keys)


@dataclass(frozen=True)
class FeatureBlock:
    """One indexed array with explicit value axes and native record context."""

    values: np.ndarray
    index: FeatureIndex
    stimulus: Stimulus = field(kw_only=True)
    layer: str = field(default="words", kw_only=True)
    participant_id: str | None = field(default=None, kw_only=True)
    metadata: Mapping[str, Any] = field(default_factory=dict, kw_only=True)
    eligible: np.ndarray | None = field(default=None, kw_only=True)
    value_dims: tuple[str, ...] = field(default=(), kw_only=True)
    value_labels: tuple[tuple[Any, ...] | None, ...] = field(default=(), kw_only=True)
    source: Datum | Recording | None = field(default=None, kw_only=True)
    origins: tuple[frozenset[OutcomeIdentity], ...] | None = field(
        default=None, kw_only=True
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "value_dims", _tuple(self.value_dims, "Value dimensions")
        )
        object.__setattr__(
            self,
            "value_labels",
            tuple(
                None if labels is None else _tuple(labels, "Axis labels")
                for labels in _tuple(self.value_labels, "Value labels")
            ),
        )
        if self.origins is not None:
            object.__setattr__(
                self,
                "origins",
                tuple(
                    origins
                    if isinstance(origins, frozenset)
                    else frozenset(_tuple(origins, "Observation origins"))
                    for origins in _tuple(self.origins, "Origins")
                ),
            )
        values = array_with_missing(self.values)
        if values.ndim == 0 or len(values) != len(self.index):
            raise ValueError(
                "Feature values must have one first-axis row per observation"
            )
        if any(
            not isinstance(name, str) or not name for name in self.value_dims
        ) or len(set(self.value_dims)) != len(self.value_dims):
            raise ValueError("Feature value dimensions must be nonempty unique names")
        if len(self.value_dims) != values.ndim - 1:
            raise ValueError(
                "Feature value dimensions must name every non-observation axis"
            )
        if self.value_labels and len(self.value_labels) != len(self.value_dims):
            raise ValueError("Value labels must match value dimensions")
        for axis, labels in enumerate(self.value_labels, start=1):
            if labels is not None and len(labels) != values.shape[axis]:
                raise ValueError(
                    "Value labels must match the corresponding axis length"
                )
        if self.origins is not None and len(self.origins) != len(self.index):
            raise ValueError("Origins must have one entry per observation")
        # A read-only view may still have a writable owner outside this envelope.
        values = values.copy()
        values.flags.writeable = False
        object.__setattr__(self, "values", values)
        eligible = (
            np.ones(len(values), dtype=bool)
            if self.eligible is None
            else array_with_missing(self.eligible)
        )
        if eligible.dtype != np.bool_ or eligible.shape != (len(values),):
            raise ValueError(
                "Feature eligibility must be a boolean vector over observations"
            )
        eligible = eligible.copy()
        eligible.flags.writeable = False
        object.__setattr__(self, "eligible", eligible)
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))


@dataclass(frozen=True)
class FeatureValues:
    """Resolved blocks, their declared meaning and source provenance."""

    blocks: tuple[FeatureBlock, ...]
    measurement: FeatureMeasurement | None = field(default=None, kw_only=True)
    provenance: Mapping[str, Any] = field(default_factory=dict, kw_only=True)

    def __post_init__(self) -> None:
        object.__setattr__(self, "blocks", _tuple(self.blocks, "Feature blocks"))
        if self.measurement is not None:
            object.__setattr__(
                self,
                "measurement",
                replace(
                    self.measurement,
                    parameters=_freeze_metadata(self.measurement.parameters),
                ),
            )
        seen: set[tuple[tuple[str, ...], ObservationKey]] = set()
        for block in self.blocks:
            for key in block.index.keys:
                identity = (block.index.names, key)
                if identity in seen:
                    raise ValueError("Duplicate observation keys across feature blocks")
                seen.add(identity)
        object.__setattr__(self, "provenance", _freeze_metadata(self.provenance))


@dataclass(frozen=True)
class ModelRequest:
    """Raw output requirements from one bound model source."""

    model: Any
    outputs: tuple[str, ...] | None


class FeatureContext(Protocol):
    """Read-only material access and executor-planned source outputs."""

    dataset: object
    stimuli: tuple[Stimulus, ...]
    segmentation: object | None
    layer: str | None

    def source_identity(self, source: object) -> str:
        """Identify one source binding within the current resolution.

        Args:
            source: Bound dataset, model or calculation object.
        """
        ...

    def target_layer(self, stimulus: Stimulus) -> str:
        """Return the selected annotation layer.

        Args:
            stimulus: Material whose annotation units are requested.
        """
        ...

    def material_identity(
        self, stimulus: Stimulus, *, dataset: object | None = None
    ) -> str:
        """Validate material membership and return its scoped identity.

        Args:
            stimulus: Material to match to the selected population.
            dataset: Source dataset; None selects the current population.
        """
        ...

    def unit_index(
        self, stimulus: Stimulus, layer: str, *, dataset: object | None = None
    ) -> FeatureIndex:
        """Return material and span coordinates for the selected units.

        Args:
            stimulus: Material whose units are indexed.
            layer: Annotation layer defining those units.
            dataset: Source dataset; None selects the current population.
        """
        ...

    def model_outputs(
        self, request: ModelRequest, stimulus: Stimulus, layer: str
    ) -> Sequence[Any]:
        """Return read-only outputs within the executor's active model batch.

        Custom resolvers consume resolved ModelFeature dependencies through
        inputs; they cannot request raw outputs outside this batch.

        Args:
            request: Model source and required output keys.
            stimulus: Material to score.
            layer: Annotation units for the outputs.
        """
        ...


@runtime_checkable
class Feature(Protocol):
    """A role-independent quantity with declared feature dependencies.

    Sources may declare ``native_layers``, a collection of annotation-layer names.
    Otherwise unit selection follows their dependencies or an explicit request.

    A resolver with dependencies must supply contributor ``origins`` on every
    output block, or return an input ``FeatureValues`` unchanged. Pointwise
    transformations can retain ``block_origins(input_values, input_block)``;
    changed row mappings must declare their actual contributors. The executor
    retains dependency origin registries, but cannot infer undeclared inputs or
    verify the scientific correctness of an explicit contributor mapping.
    """

    @property
    def name(self) -> str:
        """Column label used in specifications and results."""
        ...

    @property
    def dependencies(self) -> tuple[Feature, ...]:
        """Input quantities in the order passed to resolve()."""
        ...

    def resolve(
        self, context: FeatureContext, inputs: tuple[FeatureValues, ...]
    ) -> FeatureValues:
        """Resolve the quantity from its dependencies.

        Args:
            context: Selected materials and planned source access.
            inputs: Resolved values in dependency order.
        """
        ...


def block_origins(
    values: FeatureValues, block: FeatureBlock
) -> tuple[frozenset[OutcomeIdentity], ...]:
    """Return declared contributors or the source quantity's own observations.

    Args:
        values: Resolved quantity with source provenance.
        block: Observation block belonging to that quantity.
    """
    if block.origins is not None:
        return block.origins
    identity = values.provenance.get("outcome_identity")
    if not isinstance(identity, str) or not identity:
        raise ValueError(
            "Feature provenance requires an outcome_identity for observation lineage"
        )
    return tuple(frozenset(((identity, key),)) for key in block.index.keys)


def align_values(
    values: FeatureValues, anchor: FeatureValues
) -> tuple[FeatureBlock, ...]:
    """Match a quantity to anchor rows, marking absent observations ineligible.

    Source coordinates must be a subset of the anchor coordinates. Extra source
    axes require explicit aggregation or mapping; they are never discarded.

    Args:
        values: Quantity to align, with consistent index and value axes.
        anchor: Existing observations defining output rows and native context.
    """
    if not values.blocks:
        raise ValueError("Cannot align an empty feature")
    names = values.blocks[0].index.names
    shape = values.blocks[0].values.shape[1:]
    dimensions = values.blocks[0].value_dims
    labels = values.blocks[0].value_labels
    lookup = {}
    for block in values.blocks:
        if (
            block.index.names != names
            or block.values.shape[1:] != shape
            or block.value_dims != dimensions
            or block.value_labels != labels
        ):
            raise ValueError(
                "A quantity must have consistent index and value dimensions for alignment"
            )
        assert block.eligible is not None
        for row, (key, row_origins) in enumerate(
            zip(block.index.keys, block_origins(values, block), strict=True)
        ):
            if key in lookup:
                raise ValueError("Ambiguous feature observation lookup")
            lookup[key] = (block.values[row], bool(block.eligible[row]), row_origins)
    result = []
    for block in anchor.blocks:
        if not set(names).issubset(block.index.names):
            raise ValueError(
                "Feature has observation axes absent from the anchor; aggregate or map them explicitly"
            )
        dtype = (
            np.result_type(values.blocks[0].values.dtype, float)
            if values.blocks[0].values.dtype.kind in "biufc"
            else object
        )
        output = np.full((len(block.index), *shape), np.nan, dtype=dtype)
        eligible = np.zeros(len(block.index), dtype=bool)
        origins: list[frozenset[OutcomeIdentity]] = []
        for row, key in enumerate(block.index.project(names)):
            match = lookup.get(key)
            if match is None:
                origins.append(frozenset())
            else:
                output[row], eligible[row], origin = match
                origins.append(origin)
        result.append(
            FeatureBlock(
                output,
                block.index,
                stimulus=block.stimulus,
                layer=block.layer,
                participant_id=block.participant_id,
                metadata=block.metadata,
                eligible=eligible,
                value_dims=dimensions,
                value_labels=labels,
                source=block.source,
                origins=tuple(origins),
            )
        )
    return tuple(result)
