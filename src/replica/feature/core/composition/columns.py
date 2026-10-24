"""Predictors read from model output columns or dataset observations."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from replica.core import Stimulus
from replica.feature.core.measurement import FeatureMeasurement
from replica.feature.core.outputs.units import UnknownValueKey, unit_for
from replica.feature.core.protocols import ModelOutputRecord


class PrecomputedFeature:
    """Feature that reads a named key from FeatureModelOutput.values."""

    @property
    def supports_stimulus_selection(self) -> bool:
        """Reading a column has no state shared between stimulus calls."""
        return type(self) is PrecomputedFeature

    def configuration(self) -> dict:
        """Return computation settings without the column alias."""
        return {"key": self.key}

    def __init__(
        self, name: str, key: str, *, measurement: FeatureMeasurement | None = None
    ):
        """Bind a predictor name to a stored value key.

        Args:
            name: Column name the predictor appears under in the design matrix.
            key: Key read from each ``FeatureModelOutput.values``.
            measurement: Description for a custom column. Canonical value keys
                use their declared storage unit when omitted.
        """
        self.name = name
        self.key = key
        try:
            stored_unit = unit_for(key)
        except UnknownValueKey:
            stored_unit = None
        if stored_unit is not None:
            if measurement is not None and measurement.unit != stored_unit:
                raise ValueError(
                    f"{key!r} is stored in {stored_unit}; use InUnit to convert"
                )
            if measurement is None:
                measurement = FeatureMeasurement(
                    "stored_value", stored_unit, {"key": key}
                )
        self.measurement = measurement

    @property
    def required_outputs(self) -> tuple[str, ...] | list[str] | None:
        """Model value keys consumed by this feature and its dependencies."""
        return (self.key,)

    def __call__(
        self,
        stimulus: Stimulus,
        outputs: Sequence[ModelOutputRecord] | None,
        target_layer: str = "words",
    ) -> np.ndarray:
        """The stored values for the bound key, NaN where a unit lacks it.

        Args:
            stimulus: Text whose units the values belong to.
            outputs: Model records to read the key from; ``None`` raises
                ``ValueError``, since nothing is precomputed without them.
            target_layer: Annotation layer the stored outputs are aligned to.
        """
        if outputs is None:
            raise ValueError(f"PrecomputedFeature '{self.name}' requires outputs")
        return np.array([o.values.get(self.key, np.nan) for o in outputs])
