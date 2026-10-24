"""Features backed by stored measurements and explicit fixation calculations."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from replica.contracts.features import (
    FeatureBlock,
    FeatureIndex,
    FeatureMeasurement,
    FeatureValues,
)
from replica.core import resolve_dv
from replica.dataset.contracts import FixationCalculation
from replica.dataset.measures import CANONICAL_MEASURES
from replica.dataset.measures.sources import measure_sources
from replica.segmentation import aggregate_mask, aggregate_values, compute_refinement


@dataclass(frozen=True, eq=False)
class _DatasetFeature:
    dataset: Any = field(repr=False)
    quantity: str
    source: str = "default"
    modality: str | None = None
    name: str = ""
    _recording_participants: tuple[str, ...] | None = field(
        default=None, kw_only=True, repr=False
    )

    def __post_init__(self) -> None:
        if not isinstance(self.quantity, str) or not self.quantity:
            raise ValueError("A dataset feature requires a nonempty quantity name")
        if self.source not in {"default", "released"}:
            raise ValueError("source must be 'default' or 'released'")
        if self.modality is not None and (
            not isinstance(self.modality, str) or not self.modality
        ):
            raise ValueError("modality must be a nonempty string or None")
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name must be a nonempty string")

    @property
    def dependencies(self) -> tuple:
        return ()

    @property
    def measurement(self) -> FeatureMeasurement | None:
        description = CANONICAL_MEASURES.get(resolve_dv(self.quantity)[0])
        if description is None or not description.unit:
            return None
        return FeatureMeasurement(self.quantity, description.unit)

    @property
    def native_layers(self) -> frozenset[str]:
        """Annotation layers of the selected stored observations."""
        datums, _ = self._records()
        return frozenset(datum.layer for datum in datums)

    def _records(self):
        datums = (
            self.dataset.data(measure=self.quantity)
            if self.source == "default"
            else self.dataset.data(measure=self.quantity, source=self.source)
        )
        if self.modality is not None:
            datums = [d for d in datums if d.modality == self.modality]
        recordings = (
            self.dataset.recordings(measure=self.quantity, modality=self.modality)
            if self.source == "default"
            else []
        )
        if self._recording_participants is not None:
            recordings = [
                record
                for record in recordings
                if record.participant_id in self._recording_participants
            ]
        if datums and recordings:
            raise ValueError(
                f"Quantity {self.quantity!r} contains unit and sampled observations; "
                "select an unambiguous modality"
            )
        if not datums and not recordings:
            raise ValueError(
                f"Quantity {self.quantity!r} is unavailable from {self.source!r} "
                f"with modality {self.modality!r}"
            )
        return datums, recordings

    def resolve(self, context, inputs) -> FeatureValues:
        if inputs:
            raise ValueError("Stored features have no dependencies")
        datums, recordings = self._records()
        if recordings:
            blocks = tuple(self._recording_block(r, context) for r in recordings)
            provenance = self._provenance(recordings, blocks, context)
        else:
            prepared = self._prepare(datums, context)
            blocks = tuple(self._datum_block(d, context) for d in prepared)
            provenance = self._provenance(prepared, blocks, context)
        return FeatureValues(
            blocks, measurement=self.measurement, provenance=provenance
        )

    def _prepare(self, datums, context):
        requests = []
        for datum in datums:
            target = (
                datum.layer
                if context.segmentation is None and context.layer is None
                else context.target_layer(datum.stimulus)
            )
            if (
                datum.stimulus.layers[datum.layer].spans
                != datum.stimulus.layers[target].spans
            ):
                key = (
                    datum.stimulus.stimulus_id,
                    datum.participant_id,
                    datum.layer,
                    self.quantity,
                )
                requests.append((key, datum, datum))
        reaggregated = {}
        if requests and hasattr(self.dataset, "_reaggregate_responses"):
            target_layers = {context.target_layer(d.stimulus) for _, d, _ in requests}
            if len(target_layers) != 1:
                raise ValueError("Segmenter produced inconsistent layer names")
            reaggregated = self.dataset._reaggregate_responses(
                requests, target_layers.pop()
            )
        return [
            reaggregated.get(
                (d.stimulus.stimulus_id, d.participant_id, d.layer, self.quantity), d
            )
            for d in datums
        ]

    def _datum_block(self, datum, context):
        target = (
            datum.layer
            if context.segmentation is None and context.layer is None
            else context.target_layer(datum.stimulus)
        )
        values = datum.values
        eligible = (
            self.dataset.exclusion_mask(datum)
            if hasattr(self.dataset, "exclusion_mask")
            else np.ones(len(values), dtype=bool)
        )
        if (
            datum.stimulus.layers[datum.layer].spans
            != datum.stimulus.layers[target].spans
        ):
            if not self.dataset.is_additive(self.quantity):
                raise ValueError(
                    f"Quantity {self.quantity!r} is not additive; raw events are required"
                )
            mapping = compute_refinement(
                datum.stimulus.layers[datum.layer].spans,
                datum.stimulus.layers[target].spans,
            )
            values = aggregate_values(values, mapping)
            eligible = aggregate_mask(eligible, mapping) & ~np.isnan(values)
        index = context.unit_index(datum.stimulus, target, dataset=self.dataset)
        index = self._observation_index(index, datum, context)
        return FeatureBlock(
            values,
            index,
            stimulus=datum.stimulus,
            layer=target,
            participant_id=datum.participant_id or None,
            metadata={**datum.stimulus.metadata, **datum.metadata},
            eligible=eligible,
            source=datum,
            origins=self._origins(index, context, datum.measure_name),
        )

    def _origins(self, index, context, quantity):
        base, condition = resolve_dv(quantity)
        quantities = (base,) if condition is None else (base, condition)
        identities = tuple(
            f"dataset:{context.source_identity(self.dataset)}:measure:{quantity}"
            for quantity in quantities
        )
        return tuple(
            frozenset((identity, key) for identity in identities) for key in index.keys
        )

    def _observation_index(self, index, record, context):
        coordinates: list[tuple[str, Any]] = []
        if record.participant_id:
            coordinates.extend(
                (
                    ("participant_source", context.source_identity(self.dataset)),
                    ("participant", record.participant_id),
                )
            )
        trial = tuple(
            (name, record.metadata[name])
            for name in ("source_trial_index", "trial_id", "run_id", "recording_id")
            if name in record.metadata
        )
        if trial:
            try:
                hash(trial)
            except TypeError as exc:
                raise ValueError("Presentation coordinates must be scalar") from exc
            if not coordinates:
                coordinates.append(
                    ("participant_source", context.source_identity(self.dataset))
                )
            coordinates.append(("trial", trial))
        if not coordinates:
            return index
        names, values = zip(*coordinates, strict=True)
        return FeatureIndex(
            (*index.names, *names), tuple((*key, *values) for key in index.keys)
        )

    def _recording_block(self, recording, context):
        material = context.material_identity(recording.stimulus, dataset=self.dataset)
        index = FeatureIndex(
            ("material", "stimulus", "recording_source", "sample", "time"),
            tuple(
                (
                    material,
                    recording.stimulus.stimulus_id,
                    context.source_identity(recording),
                    i,
                    float(t),
                )
                for i, t in enumerate(recording.times)
            ),
        )
        index = self._observation_index(index, recording, context)
        return FeatureBlock(
            recording.values,
            index,
            stimulus=recording.stimulus,
            layer="time",
            participant_id=recording.participant_id or None,
            metadata=recording.metadata,
            source=recording,
            value_dims=recording.dimension_names[1:],
            value_labels=tuple(
                recording.dimension_labels.get(name)
                for name in recording.dimension_names[1:]
            ),
        )

    def _provenance(self, records, blocks, context):
        return _source_record(self, records, blocks, context)


def _select_recording_features(features, participants):
    """Narrow native target bindings without changing their dependency sources.

    Composed/custom targets retain their declared populations: their operations
    may need contributors outside the eventual participant selection.
    """
    if participants is None or any(
        type(feature) is not _DatasetFeature for feature in features
    ):
        return None
    selected = []
    materials: set[str] = set()
    for feature in features:
        datums, recordings = feature._records()
        if datums:
            return None
        recordings = [r for r in recordings if r.participant_id in participants]
        if not recordings:
            raise ValueError("No recordings for the selected target and participants")
        selected.append(replace(feature, _recording_participants=tuple(participants)))
        materials.update(record.stimulus.stimulus_id for record in recordings)
    return selected, frozenset(materials)


@dataclass(frozen=True, eq=False)
class _FixationFeature(_DatasetFeature):
    calculation: FixationCalculation = field(default_factory=FixationCalculation)
    reference: str | None = None

    @property
    def native_layers(self) -> frozenset[str]:
        """Annotation layers of the selected fixation reference."""
        native = self._native_reference()
        return frozenset(
            d.layer
            for d in native.values()
            if self.modality is None or d.modality == self.modality
        )

    def _native_reference(self):
        if not hasattr(self.dataset, "_reaggregate_responses"):
            raise TypeError(
                "fixation_feature requires a dataset exposing fixation events"
            )
        reference = self.reference or self.quantity
        native = self.dataset._native_responses(reference, source=self.source)
        if not native:
            raise ValueError(
                f"No native reference {reference!r}; select reference= explicitly"
            )
        return native

    def resolve(self, context, inputs) -> FeatureValues:
        if inputs:
            raise ValueError("Fixation features have no feature dependencies")
        native = self._native_reference()
        requests = [
            (
                (d.stimulus.stimulus_id, d.participant_id, d.layer, self.quantity),
                self.dataset._skips_as_missing(d) if self.dataset.drop_skips else d,
                d,
            )
            for d in native.values()
            if self.modality is None or d.modality == self.modality
        ]
        target = None
        if context.segmentation is not None or context.layer is not None:
            layers = {context.target_layer(d.stimulus) for _, d, _ in requests}
            if len(layers) != 1:
                raise ValueError("Segmenter produced inconsistent layer names")
            target = layers.pop()
        results = self.dataset._reaggregate_responses(
            requests,
            target,
            {self.quantity: self.calculation},
        )
        datums = [results[key] for key, _, _ in requests]
        if not datums:
            raise ValueError("No fixation observations match the requested modality")
        blocks = tuple(self._datum_block(d, context) for d in datums)
        return FeatureValues(
            blocks,
            measurement=self.measurement,
            provenance=self._provenance(datums, blocks, context),
        )


def _dataset_feature(
    dataset, quantity: str, *, source="default", modality=None, name=None
):
    """Bind a measurement name or an unambiguous released source field."""
    if not isinstance(quantity, str) or not quantity:
        raise ValueError("A dataset feature requires a nonempty quantity name")
    if source == "released":
        declarations = measure_sources(dataset)
        if quantity not in dataset.measures and quantity not in declarations:
            matches = [
                key
                for key, declaration in declarations.items()
                if quantity in {declaration.source, declaration.released_source}
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"Released field {quantity!r} requires an explicit measurement; "
                    f"matching declarations: {matches}"
                )
            quantity = matches[0]
    return _DatasetFeature(
        dataset, quantity, source, modality, quantity if name is None else name
    )


def fixation_feature(
    dataset,
    quantity: str,
    *,
    calculation: FixationCalculation,
    reference: str | None = None,
    source: str = "default",
    modality: str | None = None,
    name: str | None = None,
):
    """Calculate a fixation quantity after checking a stored native reference.

    Args:
        dataset: Dataset exposing raw fixation events and native measurements.
        quantity: Quantity to calculate.
        calculation: Explicit event assignment and duration policy.
        reference: Native measure to validate; defaults to the requested quantity.
        source: Native reference source, ``default`` or ``released``.
        modality: Acquisition modality to select.
        name: Feature display name; defaults to the quantity name.
    """
    if not isinstance(calculation, FixationCalculation):
        raise TypeError("calculation must be a FixationCalculation")
    return _FixationFeature(
        dataset,
        quantity,
        source,
        modality,
        quantity if name is None else name,
        calculation,
        reference,
    )


def _source_record(feature, records, blocks, context):
    """Record resolved origins and target units alongside each analysis result."""
    dataset = feature.dataset
    declarations = measure_sources(dataset)
    origins = set()
    units = {}
    layers = set()
    for datum, block in zip(records, blocks, strict=True):
        metadata = datum.metadata
        layer = block.layer
        layers.add(layer)
        declared = declarations.get(datum.measure_name) or declarations.get(
            resolve_dv(datum.measure_name)[0]
        )
        origin = {
            k: metadata[k]
            for k in (
                "measurement_source",
                "source_field",
                "source_release",
                "source_algorithm",
                "source_assignment",
                "source_event_filter",
                "synthetic_start_time",
                "calculation",
                "fixation_source_sha256",
                "native_reference_measure",
                "upstream_revisions",
                "source_unit",
                "source_sample_rate_hz",
                "duration_rounding",
                "floor_durations",
                "off_text_ends_run",
                "go_past_includes_off_text",
                "go_past_requires_right_exit",
                "use_source_runs",
            )
            if k in metadata
        }
        site = datum.metadata.get("site", datum.stimulus.metadata.get("site"))
        if site is not None:
            origin["site"] = site
        if declared is not None:
            origin["source_declaration"] = {
                "rule": declared.rule,
                "field": declared.source,
                "released_field": declared.released_source,
                "basis": declared.basis,
            }
        source_metadata = metadata.get("source_metadata")
        if source_metadata is not None:
            origin["native_source"] = {
                k: source_metadata[k]
                for k in (
                    "source_field",
                    "source_release",
                    "source_algorithm",
                    "source_assignment",
                    "source_event_filter",
                    "calculation",
                    "fixation_source_sha256",
                    "upstream_revisions",
                )
                if k in source_metadata
            }
        origins.add(json.dumps(origin, sort_keys=True))
        units[datum.stimulus.stimulus_id] = {
            "text": datum.stimulus.text,
            "spans": [(s.start, s.end) for s in datum.stimulus.layers[layer].spans]
            if layer in datum.stimulus.layers
            else None,
        }
    origin_indexes = {}
    for block in blocks:
        if block.origins is not None:
            for row_origins in block.origins:
                for identity, _ in row_origins:
                    origin_indexes[identity] = block.index.names
    return {
        "origin_indexes": origin_indexes,
        "origin_scopes": {identity: "observations" for identity in origin_indexes},
        "outcome_identity": (
            f"dataset:{context.source_identity(dataset)}:{feature.source}:{feature.quantity}:"
            + (
                hashlib.sha256(
                    json.dumps(feature.calculation.to_dict(), sort_keys=True).encode()
                ).hexdigest()
                if isinstance(feature, _FixationFeature)
                else "stored"
            )
        ),
        "identity_scope": "evaluation",
        "dataset_binding": context.source_identity(dataset),
        "selected_dataset_binding": context.source_identity(context.dataset),
        "input_scope": "observations",
        "measure": feature.quantity,
        "source": feature.source,
        "calculation": feature.calculation.to_dict()
        if isinstance(feature, _FixationFeature)
        else None,
        "dataset": type(dataset).__name__,
        "target_layer": next(iter(layers)) if len(layers) == 1 else sorted(layers),
        "drop_skips": getattr(dataset, "drop_skips", None),
        "exclude_first_unit": getattr(dataset, "exclude_first_unit", None),
        "exclusion_mask": type(dataset).__name__ + ".exclusion_mask"
        if hasattr(dataset, "exclusion_mask")
        else None,
        "origins": [json.loads(origin) for origin in sorted(origins)],
        "target_units_sha256": hashlib.sha256(
            json.dumps(units, sort_keys=True).encode()
        ).hexdigest(),
    }
