"""Stimulus text helpers shared across feature extractors."""

from replica.core import Stimulus


def _layer_strings(stimulus: Stimulus, target_layer: str) -> list[str]:
    """Extract text strings for units in the given layer."""
    return stimulus.layers[target_layer].extract_strings(stimulus.text)
