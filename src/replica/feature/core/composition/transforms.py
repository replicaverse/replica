"""Pointwise and sequential composition of indexed features."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import replace

import numpy as np

from replica.contracts.features import Feature
from replica.feature.core.measurement import (
    FeatureMeasurement,
    _require_features,
    describe_feature,
)

from .values import _combine, _shifted


class Lag:
    """Shift native unit sequences without crossing observation-block boundaries."""

    def __init__(self, feature: Feature, n: int = 1):
        """Select the source and signed positional offset.

        Args:
            feature: Bound quantity to shift.
            n: Positive offsets read preceding units; negative offsets read following units.
        """
        _require_features((feature,))
        if not isinstance(n, int) or isinstance(n, bool):
            raise TypeError("Lag positions must be an integer")
        self._feature = feature
        self._n = n
        self.name = f"lag{n}_{feature.name}"

    @property
    def dependencies(self) -> tuple[Feature, ...]:
        """The quantity whose native sequence is shifted."""
        return (self._feature,)

    @property
    def measurement(self) -> FeatureMeasurement | None:
        """Retain the source quantity/unit and describe the positional offset."""
        source = getattr(self._feature, "measurement", None)
        return (
            None
            if source is None
            else FeatureMeasurement(
                source.quantity,
                source.unit,
                {
                    "operation": "lag",
                    "positions": self._n,
                    "source": describe_feature(self._feature),
                },
            )
        )

    def configuration(self) -> dict:
        """Return the positional transformation settings."""
        return {"positions": self._n}

    def resolve(self, context, inputs):
        """Shift resolved values and contributor identities.

        Args:
            context: Selected material context.
            inputs: The resolved source quantity.
        """
        return _shifted(inputs[0], self._n, self.measurement)


class DerivedFeature:
    """Apply a declared pointwise function to aligned feature quantities."""

    def __init__(
        self,
        name: str,
        fn: Callable,
        *source_features: Feature,
        measurement: FeatureMeasurement | None = None,
    ):
        """Declare an arithmetic operation and its source quantities.

        Args:
            name: Output column alias.
            fn: Pointwise function returning the declared value shape.
            source_features: Bound inputs in argument order.
            measurement: Output semantics, or None when unknown.
        """
        _require_features(source_features)
        if not source_features or not callable(fn):
            raise ValueError("DerivedFeature requires inputs and a callable")
        self.name = name
        self._fn = fn
        self._sources = source_features
        self._measurement = measurement

    @property
    def dependencies(self) -> tuple[Feature, ...]:
        """Pointwise function inputs in argument order."""
        return self._sources

    @property
    def measurement(self) -> FeatureMeasurement | None:
        """Record the declared result semantics and input descriptions."""
        if self._measurement is None:
            return None
        return FeatureMeasurement(
            self._measurement.quantity,
            self._measurement.unit,
            {
                "operation": "caller_supplied",
                "parameters": self._measurement.parameters,
                "sources": [describe_feature(feature) for feature in self._sources],
            },
        )

    def resolve(self, context, inputs):
        """Align inputs and apply the declared function.

        Args:
            context: Selected material context.
            inputs: Resolved inputs in dependency order.
        """
        operation = {
            "callable": f"{type(self._fn).__module__}.{type(self._fn).__qualname__}",
            "persistent_identity": None,
        }
        return _combine(inputs, self._fn, self.measurement, operation)


class Interaction:
    """Multiply aligned features without adding their separate main effects."""

    def __init__(self, a: Feature, b: Feature, name: str | None = None):
        """Declare the two quantities to multiply.

        Args:
            a: Left factor.
            b: Right factor.
            name: Output alias; defaults to the factor names joined by '_x_'.
        """
        _require_features((a, b))
        self._a, self._b = a, b
        self.name = name or f"{a.name}_x_{b.name}"

    @property
    def dependencies(self) -> tuple[Feature, ...]:
        """The two multiplicative factors."""
        return (self._a, self._b)

    @property
    def measurement(self) -> FeatureMeasurement | None:
        """Describe known factor units without inferring dimensional equivalence."""
        left, right = (
            getattr(feature, "measurement", None) for feature in self.dependencies
        )
        if left is None or right is None:
            return None
        unit = (
            None
            if left.unit is None or right.unit is None
            else f"({left.unit})*({right.unit})"
        )
        return FeatureMeasurement(
            "product",
            unit,
            {"factors": [describe_feature(self._a), describe_feature(self._b)]},
        )

    def resolve(self, context, inputs):
        """Multiply values after validated keyed alignment.

        Args:
            context: Selected material context.
            inputs: Resolved left and right factors.
        """
        return _combine(inputs, np.multiply, self.measurement, "product")


_NATS_PER_UNIT = {"nats": 1.0, "bits": math.log(2.0), "hartleys": math.log(10.0)}


class InUnit:
    """Convert a declared logarithmic quantity to bits, nats or hartleys.

    Source values and contributor identities remain unchanged.
    """

    def __init__(self, feature: Feature, unit: str):
        """Select a reporting unit and qualify the feature name with it.

        Args:
            feature: Bound feature with a declared logarithmic unit.
            unit: Reporting unit: ``bits``, ``nats``, or ``hartleys``.
        """
        _require_features([feature])
        measurement = getattr(feature, "measurement", None)
        if not isinstance(measurement, FeatureMeasurement):
            raise ValueError("InUnit requires an explicitly declared measurement")
        if unit not in _NATS_PER_UNIT or measurement.unit not in _NATS_PER_UNIT:
            raise ValueError("InUnit converts only bits, nats, and hartleys")
        self._feature = feature
        self._source_unit = measurement.unit
        self._unit = unit
        self._factor = _NATS_PER_UNIT[measurement.unit] / _NATS_PER_UNIT[unit]
        self.name = f"{feature.name}_{unit}"

    @property
    def dependencies(self) -> tuple:
        """The quantity to convert."""
        return (self._feature,)

    def resolve(self, context, inputs):
        """Convert resolved values while preserving observation contributors.

        Args:
            context: Selected materials and annotation layer.
            inputs: Resolved source values.
        """
        result = _combine(
            inputs,
            lambda values: values * self._factor,
            self.measurement,
            {"unit": self._unit},
        )
        record = dict(result.provenance)
        record.update(
            outcome_identity=inputs[0].provenance["outcome_identity"],
            input_scope=inputs[0].provenance.get("input_scope", "derived"),
            transparent=True,
        )
        return replace(result, provenance=record)

    @property
    def measurement(self) -> FeatureMeasurement:
        """Describe the current source and selected reporting unit."""
        source = getattr(self._feature, "measurement", None)
        if (
            not isinstance(source, FeatureMeasurement)
            or source.unit != self._source_unit
        ):
            raise ValueError("The source measurement unit changed; rebuild InUnit")
        return FeatureMeasurement(
            source.quantity,
            self._unit,
            {
                "conversion": "logarithmic_unit",
                "source": describe_feature(self._feature),
            },
        )
