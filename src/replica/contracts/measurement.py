"""Declared quantities and reporting units."""

from __future__ import annotations

import json
from dataclasses import dataclass, field


@dataclass(frozen=True)
class FeatureMeasurement:
    """Quantity, unit and JSON parameters; not a producer identity."""

    quantity: str | None
    unit: str | None
    parameters: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if any(
            value is not None and (not isinstance(value, str) or not value)
            for value in (self.quantity, self.unit)
        ):
            raise ValueError(
                "A measurement quantity and unit must be nonempty strings or None"
            )
        if not isinstance(self.parameters, dict):
            raise TypeError("FeatureMeasurement parameters must be a dictionary")
        self.to_dict()

    def to_dict(self) -> dict:
        """Return a detached JSON-compatible description."""
        return json.loads(
            json.dumps(
                {
                    "quantity": self.quantity,
                    "unit": self.unit,
                    "parameters": self.parameters,
                },
                allow_nan=False,
            )
        )
