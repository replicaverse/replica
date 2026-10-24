"""Caller-defined partitions for pooled unigram sampling."""

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np


def _validate_labels(labels: tuple[str, ...]) -> None:
    if not 1 <= len(labels) <= 256:
        raise ValueError("a partition needs between 1 and 256 strata")
    if any(not isinstance(label, str) or not label for label in labels):
        raise ValueError("stratum labels must be nonempty strings")
    if len(set(labels)) != len(labels):
        raise ValueError("stratum labels must be distinct")


@dataclass(frozen=True, slots=True)
class Stratification:
    """A versioned classifier of decoded prefix tails into ordered labels.

    ``classify`` must be deterministic and return one of ``labels``. Change
    ``name`` when its semantics change: the identity records that version,
    the label order and window size, not the callable's implementation.
    """

    name: str
    labels: tuple[str, ...]
    window_chars: int
    classify: Callable[[str], str] = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "labels", tuple(self.labels))
        _validate_labels(self.labels)
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("stratification name must be a nonempty version")
        if not isinstance(self.window_chars, int) or self.window_chars < 1:
            raise ValueError("window_chars must be a positive integer")
        if not callable(self.classify):
            raise TypeError("classify must be callable")

    @property
    def identity(self) -> str:
        """Digest of the declared classifier version and input layout."""
        payload = (self.name, self.labels, self.window_chars)
        return hashlib.sha256(json.dumps(payload).encode("utf-8")).hexdigest()

    def label(self, prefixes: Sequence[str]) -> np.ndarray:
        """Classify prefix tails, refusing labels outside the partition.

        Args:
            prefixes: Decoded prefixes, truncated here to ``window_chars``.
        """
        lookup = {label: i for i, label in enumerate(self.labels)}
        out = np.empty(len(prefixes), dtype=np.uint8)
        for i, prefix in enumerate(prefixes):
            label = self.classify(prefix[-self.window_chars :])
            if label not in lookup:
                raise ValueError(f"classifier returned unknown stratum {label!r}")
            out[i] = lookup[label]
        return out
