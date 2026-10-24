"""Explicit aggregation of observation-indexed quantities."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from replica.contracts.features import (
    UNIT_INDEX_NAMES,
    Feature,
    FeatureBlock,
    FeatureIndex,
    FeatureValues,
    block_origins,
)

from .values import _provenance


class _Aggregate:
    def __init__(self, feature, *, by, statistic, missing, name):
        if statistic not in {"mean", "sum", "median"}:
            raise ValueError("statistic must be 'mean', 'sum', or 'median'")
        if missing not in {"drop", "propagate"}:
            raise ValueError("missing must be 'drop' or 'propagate'")
        self.dependencies = (feature,)
        self.by = (by,) if isinstance(by, str) else tuple(by)
        if not self.by:
            raise ValueError("Aggregation requires explicit grouping coordinates")
        self.statistic = statistic
        self.missing = missing
        self.name = name or f"{feature.name}_{statistic}"

    @property
    def measurement(self):
        return getattr(self.dependencies[0], "measurement", None)

    def resolve(self, context, inputs):
        value = inputs[0]
        if not value.blocks:
            raise ValueError("Cannot aggregate an empty quantity")
        coordinates: list[str] = []
        for name in self.by:
            expanded = (
                UNIT_INDEX_NAMES
                if name == "unit"
                else ("participant_source", "participant")
                if name == "participant"
                else (name,)
            )
            for coordinate in expanded:
                if coordinate not in coordinates:
                    coordinates.append(coordinate)
        names = tuple(coordinates)
        if not set(UNIT_INDEX_NAMES).issubset(names):
            raise ValueError(
                "Initial aggregation requires grouping by canonical unit identity"
            )
        groups = defaultdict(list)
        contexts = {}
        allowed_drop = {"participant", "participant_source", "trial"}
        for block in value.blocks:
            if block.value_dims:
                raise ValueError(
                    "Select scalar values before aggregating observation axes"
                )
            if not set(names).issubset(block.index.names):
                raise ValueError("Aggregation names a coordinate absent from its input")
            if set(block.index.names) - set(names) - allowed_drop:
                raise ValueError(
                    "Aggregation must explicitly retain independent condition axes"
                )
            for row, (key, origins) in enumerate(
                zip(
                    block.index.project(names), block_origins(value, block), strict=True
                )
            ):
                groups[key].append(
                    (block.values[row], bool(block.eligible[row]), origins)
                )
                contexts[key] = block
        outputs = defaultdict(list)
        reductions: dict[str, Callable[[np.ndarray], Any]] = {
            "mean": np.mean,
            "sum": np.sum,
            "median": np.median,
        }
        reduction = reductions[self.statistic]
        for key, observations in groups.items():
            raw = np.array([row[0] for row in observations], dtype=float)
            included = np.array(
                [row[1] for row in observations], dtype=bool
            ) & ~np.isnan(raw)
            valid = included.any() and (self.missing == "drop" or included.all())
            result = float(reduction(raw[included])) if valid else np.nan
            # Retain all contributors, including missing ones that affect selection.
            origins = frozenset().union(*(row[2] for row in observations))
            block = contexts[key]
            group_id = tuple(
                value
                for name, value in zip(names, key, strict=True)
                if name not in {"start", "end", "unit"}
            )
            outputs[group_id].append(
                (key, result, valid, origins, int(included.sum()), len(raw), block)
            )
        blocks = []
        for rows in outputs.values():
            rows.sort(key=lambda row: row[0][names.index("unit")])
            source = rows[0][-1]
            row_coordinates = dict(zip(names, rows[0][0], strict=True))
            blocks.append(
                FeatureBlock(
                    np.array([row[1] for row in rows]),
                    FeatureIndex(names, tuple(row[0] for row in rows)),
                    stimulus=source.stimulus,
                    layer=source.layer,
                    participant_id=row_coordinates.get("participant"),
                    eligible=np.array([row[2] for row in rows], dtype=bool),
                    metadata={
                        **source.stimulus.metadata,
                        "contributor_counts": tuple(row[4] for row in rows),
                        "contributor_totals": tuple(row[5] for row in rows),
                    },
                    origins=tuple(row[3] for row in rows),
                )
            )
        return FeatureValues(
            tuple(blocks),
            measurement=value.measurement,
            provenance=_provenance(
                inputs,
                {"aggregate": self.statistic, "by": names, "missing": self.missing},
            ),
        )


def aggregate(
    feature: Feature,
    *,
    by: str | Sequence[str],
    statistic: str,
    missing: str,
    name: str | None = None,
) -> Feature:
    """Reduce observation groups while retaining contributor identities.

    Args:
        feature: Bound scalar quantity to aggregate.
        by: Retained coordinates, including 'unit' or all of its expanded material
            coordinates. Independent condition axes must also be retained.
        statistic: Reduction: mean, sum or median.
        missing: Drop missing observations or propagate them to the result.
        name: Output column alias.
    """
    return _Aggregate(feature, by=by, statistic=statistic, missing=missing, name=name)
