"""Optional feature descriptions and composition input validation."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

from replica.contracts.measurement import FeatureMeasurement

if TYPE_CHECKING:
    from replica.contracts.features import Feature

    from .protocols import FeatureExtractor


def describe_feature(feature: Feature | FeatureExtractor) -> dict:
    """Describe a feature without guessing undeclared units or configuration.

    Args:
        feature: Extractor with an optional ``measurement`` declaration.
    """
    measurement = getattr(feature, "measurement", None)
    if measurement is not None and not isinstance(measurement, FeatureMeasurement):
        raise TypeError("feature.measurement must be a FeatureMeasurement or None")
    return {
        "name": feature.name,
        "implementation": f"{type(feature).__module__}.{type(feature).__qualname__}",
        "measurement": None if measurement is None else measurement.to_dict(),
    }


def _require_features(features: Iterable[Feature]) -> None:
    """Require complete quantity definitions before constructing a composition."""
    if any(
        not callable(getattr(feature, "resolve", None))
        or not hasattr(feature, "dependencies")
        for feature in features
    ):
        raise TypeError(
            "Composition requires bound Features; use ModelFeature(model, extractor)"
        )
