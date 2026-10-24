"""Validated observations shared by evaluation and study frame adapters."""

from dataclasses import dataclass, field

import numpy as np

from replica.contracts.features import FeatureValues, OutcomeIdentity


@dataclass(frozen=True)
class ObservationTable:
    """Aligned response, feature columns, grouping codes, and source identities."""

    y: np.ndarray
    columns: dict[str, np.ndarray]
    participants: np.ndarray
    items: np.ndarray
    groups: np.ndarray
    sentence_ids: np.ndarray | None
    participant_ids: dict[str, int] = field(default_factory=dict)
    item_ids: dict[str, int] = field(default_factory=dict)
    stimulus_ids: np.ndarray | None = None
    unit_indices: np.ndarray | None = None
    response: dict = field(default_factory=dict)
    factors: dict[str, np.ndarray] = field(default_factory=dict)

    raw_y: np.ndarray | None = None
    row_keys: tuple[tuple, ...] = ()
    target_origins: tuple[frozenset[OutcomeIdentity], ...] = ()
    input_origins: dict[str, tuple[frozenset[OutcomeIdentity], ...]] = field(
        default_factory=dict
    )
    source_values: dict[str, FeatureValues] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if self.raw_y is not None and self.raw_y.shape != self.y.shape:
            raise ValueError("Raw targets must align with prepared targets")
        for identity in (
            self.row_keys,
            self.target_origins,
            *self.input_origins.values(),
        ):
            if identity and len(identity) != len(self.y):
                raise ValueError("Observation provenance must align with targets")
        arrays = [
            self.y,
            *self.columns.values(),
            *self.factors.values(),
            self.participants,
            self.items,
            self.groups,
        ]
        if self.sentence_ids is not None:
            arrays.append(self.sentence_ids)
        if self.stimulus_ids is not None:
            arrays.append(self.stimulus_ids)
        if self.unit_indices is not None:
            arrays.append(self.unit_indices)
        if any(a.shape != (len(self.y),) for a in arrays):
            raise ValueError("Observation columns and grouping codes must align with y")
        if any(np.ma.is_masked(a) for a in arrays):
            raise ValueError("Prepared observations must not contain masked values")
        if not all(np.isfinite(a).all() for a in [self.y, *self.columns.values()]):
            raise ValueError("Prepared response and predictor columns must be finite")
