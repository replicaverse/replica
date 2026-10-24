"""Output requirements shared by features and streaming extraction."""

from collections.abc import Iterable

from replica.feature.core.protocols import FeatureExtractor, OutputRequirements


def required_outputs(features: Iterable[FeatureExtractor]) -> list[str] | None:
    """Union declared requirements, preserving unknown custom-feature behavior.

    Args:
        features: Features whose optional `required_outputs` tuples name the
            model values they read. An absent declaration requests all outputs.
    """
    keys: set[str] = set()
    for feature in features:
        if not isinstance(feature, OutputRequirements):
            return None
        declared = feature.required_outputs
        if declared is None:
            return None
        keys.update(declared)
    return sorted(keys)
