"""Keyed composition of resolved quantities."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from replica.contracts.features import (
    FeatureValues,
    OutcomeIdentity,
    align_values,
    block_origins,
)
from replica.utils.arrays import array_with_missing


def _provenance(inputs, operation):
    """Retain source and contributor descriptions for a declared operation."""
    indexes = {}
    scopes = {}
    for value in inputs:
        indexes.update(value.provenance.get("origin_indexes", {}))
        scopes.update(value.provenance.get("origin_scopes", {}))
    return {
        "operation": operation,
        "dependencies": [dict(v.provenance) for v in inputs],
        "origin_indexes": indexes,
        "origin_scopes": scopes,
        "input_scope": "derived",
    }


def _combine(inputs, function, measurement, operation):
    """Apply a pointwise function after unique projection to an existing index."""
    if not inputs or any(not value.blocks for value in inputs):
        raise ValueError("Composition requires nonempty feature inputs")
    anchor = max(inputs, key=lambda value: len(value.blocks[0].index.names))
    aligned = [align_values(value, anchor) for value in inputs]
    result = []
    for blocks in zip(*aligned, strict=True):
        vector = [block for block in blocks if block.value_dims]
        dimensions = vector[0].value_dims if vector else ()
        labels = vector[0].value_labels if vector else ()
        if any(
            block.value_dims != dimensions
            or block.value_labels != labels
            or block.values.shape[1:] != vector[0].values.shape[1:]
            for block in vector
        ):
            raise ValueError(
                "Pointwise composition requires matching named value dimensions and axis lengths"
            )
        arrays = [
            block.values.reshape((len(block.index),) + (1,) * len(dimensions))
            if not block.value_dims and dimensions
            else block.values
            for block in blocks
        ]
        values = array_with_missing(function(*arrays))
        expected = vector[0].values.shape if vector else blocks[0].values.shape
        if values.shape != expected:
            raise ValueError("Pointwise composition changed the declared value shape")
        origins = tuple(
            frozenset().union(*(block.origins[row] for block in blocks))
            for row in range(len(blocks[0].index))
        )
        result.append(
            replace(
                blocks[0],
                values=values,
                eligible=np.logical_and.reduce([block.eligible for block in blocks]),
                value_dims=dimensions,
                value_labels=labels,
                origins=origins,
            )
        )
    return FeatureValues(
        tuple(result),
        measurement=measurement,
        provenance=_provenance(inputs, operation),
    )


def _shifted(value, n, measurement):
    """Shift native unit sequences and their contributor identities together."""
    if not isinstance(n, int) or isinstance(n, bool):
        raise TypeError("Lag positions must be an integer")
    blocks = []
    for block in value.blocks:
        if "unit" not in block.index.names:
            raise ValueError(
                "Positional Lag requires annotation units; use an explicit time mapping for recordings"
            )
        unit_position = block.index.names.index("unit")
        units = [key[unit_position] for key in block.index.keys]
        positions = {unit: row for row, unit in enumerate(units)}
        if units != sorted(units) or len(set(units)) != len(units):
            raise ValueError(
                "Lag requires an ordered native sequence within each observation block"
            )
        dtype = (
            np.result_type(block.values.dtype, float)
            if block.values.dtype.kind in "biufc"
            else object
        )
        values = np.full(block.values.shape, np.nan, dtype=dtype)
        origins: list[frozenset[OutcomeIdentity]] = [frozenset() for _ in units]
        eligible = np.zeros(len(units), dtype=bool)
        original = block_origins(value, block)
        for row, unit in enumerate(units):
            source = unit - n
            if source in positions:
                offset = positions[source]
                values[row] = block.values[offset]
                origins[row] = original[offset]
                eligible[row] = block.eligible[offset]
        blocks.append(
            replace(block, values=values, eligible=eligible, origins=tuple(origins))
        )
    record = _provenance((value,), {"lag": n})
    record.update(
        outcome_identity=value.provenance["outcome_identity"],
        input_scope=value.provenance.get("input_scope", "derived"),
        transparent=True,
    )
    return FeatureValues(tuple(blocks), measurement=measurement, provenance=record)
