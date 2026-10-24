"""Core data types for the REPLICA.

Implements standoff annotation (Bird & Liberman, 2001): raw text is stored
once, and all annotations (word boundaries, token boundaries, reading times)
are character-offset pointers into that text.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np

from replica.utils.arrays import array_with_missing


class Modality(StrEnum):
    """Stable names for acquisition modalities built into REPLICA.

    Public APIs also accept strings so downstream packages can introduce a
    modality without waiting for a REPLICA release.  The enum provides shared
    spelling for the modalities the core package understands.
    """

    BEHAVIORAL = "behavioral"
    EYE_TRACKING = "eye-tracking"
    EEG = "eeg"
    FMRI = "fmri"
    MEG = "meg"
    ECOG = "ecog"


class Presentation(StrEnum):
    """How a corpus put its stimuli in front of the participant.

    Separate from :class:`Modality`, which says how the response was recorded:
    Narratives is listening recorded with fMRI, ZuCo is natural reading
    recorded with both eye-tracking and EEG.
    """

    NATURAL_READING = "natural reading"
    SELF_PACED_READING = "self-paced reading"
    MAZE = "maze"
    RSVP = "rapid serial visual presentation"
    LISTENING = "listening"


@dataclass(frozen=True)
class Span:
    """Character-level pointer into raw text. Half-open interval [start, end)."""

    start: int
    end: int

    def __post_init__(self):
        if self.start < 0:
            raise ValueError(f"start must be non-negative, got {self.start}")
        if self.end < self.start:
            raise ValueError(
                f"end must be >= start, got start={self.start}, end={self.end}"
            )

    def __len__(self) -> int:
        return self.end - self.start

    @property
    def midpoint(self) -> float:
        """Position halfway between ``start`` and ``end``, in character offsets."""
        return (self.start + self.end) / 2

    def extract(self, text: str) -> str:
        """Extract the substring this span points to.

        Args:
            text: The raw text the span's offsets index into.
        """
        return text[self.start : self.end]

    def contains(self, other: Span) -> bool:
        """True if this span fully contains ``other``.

        Args:
            other: The span to test for containment within this one.
        """
        return self.start <= other.start and other.end <= self.end

    def overlaps(self, other: Span) -> bool:
        """True if the two spans share at least one character position.

        Args:
            other: The span to test for overlap against this one.
        """
        return self.start < other.end and other.start < self.end


@dataclass
class AnnotationLayer:
    """Named layer of (Span, value) pairs over raw text.

    Can hold word boundaries, subword tokens, reading times, or any
    per-region annotation.
    """

    name: str
    spans: list[Span]
    values: list[Any]
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if len(self.spans) != len(self.values):
            raise ValueError(
                f"spans and values must have the same length, "
                f"got {len(self.spans)} spans and {len(self.values)} values"
            )

    def __len__(self) -> int:
        return len(self.spans)

    def __getitem__(self, idx: int) -> tuple[Span, Any]:
        return self.spans[idx], self.values[idx]

    def extract_strings(self, text: str) -> list[str]:
        """Extract the text each span points to.

        Args:
            text: The raw text the layer's spans index into.
        """
        return [span.extract(text) for span in self.spans]


@dataclass
class Stimulus:
    """Text with named annotation layers and metadata.

    Layers refer to the same text through character spans. Metadata holds
    processing annotations and loader-specific fields.
    """

    text: str
    layers: dict[str, AnnotationLayer] = field(default_factory=dict)
    stimulus_id: str = ""
    metadata: dict = field(default_factory=dict)

    def add_layer(self, layer: AnnotationLayer) -> None:
        """Attach an annotation layer.

        Args:
            layer: The layer to store, keyed by its own ``name``.
        """
        self.layers[layer.name] = layer

    def unit_strings(self, layer: str = "words") -> list[str]:
        """Return the strings from the named annotation layer.

        Args:
            layer: Name of the annotation layer to read strings from.
        """
        return self.layers[layer].extract_strings(self.text)


@dataclass
class Datum:
    """One participant's measurement on one stimulus.

    ``values`` is aligned to the annotation layer named by ``layer``,
    one value per unit in that layer.
    Missing observations use ``np.nan``; NumPy masks are converted to NaN.
    """

    stimulus: Stimulus
    measure_name: str
    values: np.ndarray
    participant_id: str = ""
    layer: str = "words"
    metadata: dict = field(default_factory=dict)
    modality: str | Modality = ""

    def __post_init__(self):
        self.values = array_with_missing(self.values)
        self.modality = str(self.modality)
        if self.layer not in self.stimulus.layers:
            raise KeyError(
                f"Stimulus has no layer '{self.layer}' "
                f"(available: {list(self.stimulus.layers.keys())})"
            )
        n_units = len(self.stimulus.layers[self.layer])
        if len(self.values) != n_units:
            raise ValueError(
                f"values length ({len(self.values)}) must match "
                f"number of units in layer '{self.layer}' ({n_units})"
            )


#: Per-unit conditions used to restrict dependent variables.
#: Each flag is a datum named by the key: 1 when true, 0 when false, NaN when
#: unavailable.
CONDITIONS: dict[str, str] = {
    "first_pass": (
        "the unit was fixated before the eye had ever been on a unit to its "
        "right, so a unit first entered by a regression does not qualify"
    ),
}

#: Derived variable names mapped to (measure, condition).
#: Use the measurement where the flag is 1, zero where it is 0, and NaN where
#: it is unavailable. Values stored under the derived name take precedence.
DERIVED_DVS: dict[str, tuple[str, str]] = {
    "first_fixation_in_first_pass": ("first_fixation", "first_pass"),
    "first_run_in_first_pass": ("first_run", "first_pass"),
    "go_past_in_first_pass": ("go_past", "first_pass"),
}


def resolve_dv(name: str) -> tuple[str, str | None]:
    """Return the base measure and optional condition for a dependent variable.

    Args:
        name: A measure name or a key of :data:`DERIVED_DVS`.

    Returns:
        ``(measure, condition)``, the condition ``None`` for a plain measure.
    """
    return DERIVED_DVS.get(name, (name, None))


def _regular_sampling_interval(
    sample_times: np.ndarray, interval: float | None = None
) -> float | None:
    """Return a uniform interval when timestamps agree within rounding error."""
    sample_times = np.asarray(sample_times)
    precision = np.finfo(float).eps
    if sample_times.dtype.kind == "f":
        precision = max(precision, np.finfo(sample_times.dtype).eps)
    times = np.asarray(sample_times, dtype=float)
    if times.ndim != 1 or len(times) < 2:
        return interval
    if interval is None:
        with np.errstate(over="ignore", invalid="ignore"):
            span = times[-1] - times[0]
        interval = float(
            span / (len(times) - 1)
            if np.isfinite(span)
            else times[-1] / (len(times) - 1) - times[0] / (len(times) - 1)
        )
    if not np.isfinite(interval) or interval <= 0:
        return None
    with np.errstate(over="ignore", invalid="ignore"):
        positions = np.arange(len(times))
        expected = times[0] + positions * interval
        overflow = ~np.isfinite(expected)
        if overflow.any():
            expected[overflow] = (
                times[0] / 2 + positions[overflow] * (interval / 2)
            ) * 2
        magnitude = np.maximum(abs(times[0]), np.maximum(np.abs(expected), interval))
        # Roundoff cannot span a neighboring sample even for a large time origin.
        tolerance = np.minimum(4 * precision * magnitude, interval / 4)
        regular = np.all(np.isfinite(expected)) and np.all(
            np.abs(times - expected) <= tolerance
        )
    return interval if regular else None


@dataclass
class Recording:
    """A sampled, potentially multichannel response to one stimulus.

    The first dimension of ``values`` is always time. Remaining dimensions
    are described by ``dimension_names`` and may represent voxels, electrodes,
    sensors, frequency bands, or another modality-specific structure.
    NumPy-masked values and event times become NaN. Timing inputs must be real;
    sample times must be finite.

    A regular recording declares ``sample_interval`` and ``start_time``. An
    explicit clock supplies every timestamp in ``sample_times``. If an interval
    is also supplied, it must agree within the clock's floating-point precision.
    Explicit floating timestamps retain their dtype; :attr:`times` returns
    float64 values. An omitted interval is not inferred.
    Keeping timing on the response object makes continuous EEG, MEG, ECoG, and fMRI
    share one contract without forcing any of them onto word boundaries.

    ``event_times`` holds timing that belongs to the recording rather than
    the stimulus, such as fixation onsets: one onset per unit of the
    stimulus's ``words`` layer, in seconds on the clock of :attr:`times`,
    NaN for a unit the recording never presented.
    """

    stimulus: Stimulus
    values: np.ndarray
    measure_name: str
    modality: str | Modality
    participant_id: str = ""
    sample_interval: float | None = None
    start_time: float = 0.0
    sample_times: np.ndarray | None = None
    event_times: np.ndarray | None = None
    dimension_names: tuple[str, ...] = ()
    dimension_labels: dict[str, tuple[str, ...]] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.values = array_with_missing(self.values)
        self.modality = str(self.modality)

        if self.values.ndim == 0:
            raise ValueError("recording values must have a time dimension")
        if not self.measure_name:
            raise ValueError("recording measure_name must not be empty")
        if not self.modality:
            raise ValueError("recording modality must not be empty")

        self._validate_timing()
        self._validate_events()
        self._validate_dimensions()

    def _validate_events(self) -> None:
        """Validate the recording's own unit onsets, when present."""

        if self.event_times is None:
            return
        self.event_times = array_with_missing(self.event_times)
        if np.iscomplexobj(self.event_times):
            raise ValueError("event_times must be real")
        self.event_times = self.event_times.astype(float, copy=False)
        if self.event_times.ndim != 1:
            raise ValueError("event_times must be one-dimensional")
        if np.isinf(self.event_times).any():
            raise ValueError("event_times must be finite or NaN")
        words = self.stimulus.layers.get("words")
        if words is not None and len(self.event_times) != len(words):
            raise ValueError(
                "event_times must hold one onset per unit of the stimulus's words layer"
            )

    def _validate_timing(self) -> None:
        """Validate regular or irregular sampling metadata."""

        if self.sample_interval is not None and np.iscomplexobj(self.sample_interval):
            raise ValueError("sample_interval must be real")
        if np.iscomplexobj(self.start_time):
            raise ValueError("start_time must be real")
        if self.sample_interval is not None and (
            not np.isfinite(self.sample_interval) or self.sample_interval <= 0
        ):
            raise ValueError("sample_interval must be finite and positive")
        if not np.isfinite(self.start_time):
            raise ValueError("start_time must be finite")
        if self.sample_times is None and self.sample_interval is None:
            raise ValueError("provide sample_interval or sample_times")
        if self.sample_times is not None:
            self.sample_times = array_with_missing(self.sample_times)
            if np.iscomplexobj(self.sample_times):
                raise ValueError("sample_times must be real")
            if self.sample_times.dtype.kind != "f":
                self.sample_times = self.sample_times.astype(float)
            if self.sample_times.ndim != 1:
                raise ValueError("sample_times must be one-dimensional")
            if len(self.sample_times) != self.n_samples:
                raise ValueError(
                    "sample_times length must match the recording time dimension"
                )
        with np.errstate(over="ignore", invalid="ignore"):
            times = self.times
        if not np.isfinite(times).all():
            raise ValueError("recording times must be finite")
        if len(times) > 1 and np.any(times[1:] <= times[:-1]):
            raise ValueError("recording times must be strictly increasing")
        if self.sample_times is not None:
            if (
                self.sample_interval is not None
                and _regular_sampling_interval(self.sample_times, self.sample_interval)
                is None
            ):
                raise ValueError("sample_interval must agree with sample_times")

    def _validate_dimensions(self) -> None:
        """Validate dimension names and optional labels against the array."""

        if not self.dimension_names:
            trailing = tuple(f"dimension_{i}" for i in range(1, self.values.ndim))
            self.dimension_names = ("time", *trailing)
        if len(self.dimension_names) != self.values.ndim:
            raise ValueError(
                "dimension_names must contain one name per values dimension"
            )
        if self.dimension_names[0] != "time":
            raise ValueError("the first dimension name must be 'time'")
        if len(set(self.dimension_names)) != len(self.dimension_names):
            raise ValueError("dimension_names must be unique")

        for dimension, labels in self.dimension_labels.items():
            if dimension not in self.dimension_names:
                raise ValueError(f"labels provided for unknown dimension {dimension!r}")
            axis = self.dimension_names.index(dimension)
            if len(labels) != self.values.shape[axis]:
                raise ValueError(
                    f"{dimension!r} labels length must match dimension size"
                )

    @property
    def n_samples(self) -> int:
        """Number of samples along the time dimension."""
        return self.values.shape[0]

    @property
    def times(self) -> np.ndarray:
        """Absolute time, in seconds, for every sample."""
        if self.sample_times is not None:
            return self.sample_times.astype(float, copy=True)
        interval = self.sample_interval
        if interval is None:  # guarded by _validate_timing
            raise RuntimeError("recording has no sampling information")
        return self.start_time + np.arange(self.n_samples) * interval
