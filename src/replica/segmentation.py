"""Text segmentation for REPLICA.

Segmenters divide raw text into units of analysis. Subclass
:class:`Segmenter` and implement :meth:`segment` to define how text
is split into character-offset spans.

Built-in segmenters cover common cases (whitespace, character-level);
subclass ``Segmenter`` for anything else. A segmenter registered under a name
with :func:`register_segmenter` can also be the unit of a predictor cell:
``replica compute --import <module> --unit <name>`` imports the module that
registers it, attaches its layer before scoring, and names the cell
``_seg-<name>``; the store then checks the cell against that layer.

Example::

    class MySegmenter(Segmenter):
        def segment(self, text: str) -> list[Span]:
            # split on periods
            ...

    spec = EvaluationSpec(targets=[dataset.feature("reading_time")],
                          predictors=[ModelFeature(lm, Surprisal())],
                          segmentation=MySegmenter("units"))
    evaluate(dataset, spec)
"""

from abc import ABC, abstractmethod
from collections.abc import Callable

import numpy as np

from .core import AnnotationLayer, Span, Stimulus
from .utils.arrays import real_array_with_missing

# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class Segmenter(ABC):
    """Base class for text segmenters.

    A segmenter maps raw text to a list of non-overlapping
    :class:`~replica.core.Span` objects defining the units of
    analysis. Subclass and implement :meth:`segment`::

        class HyphenSegmenter(Segmenter):
            def segment(self, text: str) -> list[Span]:
                # split on hyphens, returning non-empty spans
                ...

        seg = HyphenSegmenter("morphemes")
        seg("well-known")  # [Span(0,4), Span(5,10)]
    """

    def __init__(self, name: str):
        """Create a segmenter that labels its output layer *name*.

        Args:
            name: Annotation layer name this segmenter's spans are stored under.
        """
        self.name = name

    @abstractmethod
    def segment(self, text: str) -> list[Span]:
        """Segment *text* into non-overlapping spans.

        Args:
            text: The raw text to split into spans.
        """
        ...

    def __call__(self, text: str) -> list[Span]:
        """Shorthand: ``segmenter(text)`` calls ``segmenter.segment(text)``.

        Args:
            text: The raw text to split into spans.
        """
        return self.segment(text)


# ---------------------------------------------------------------------------
# Named segmentations
# ---------------------------------------------------------------------------

#: Factories of the segmenters a predictor cell may be computed on, by name.
SEGMENTERS: dict[str, Callable[[], Segmenter]] = {}

_NAME_RE = __import__("re").compile(r"^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$")


def register_segmenter(
    name: str,
) -> Callable[[Callable[[], Segmenter]], Callable[[], Segmenter]]:
    """Register a segmenter factory under *name*, the unit name of its cells.

    Args:
        name: Lowercase letters, digits and single hyphens; ``words``,
            ``characters`` and ``tokens`` are the loader's and the model's
            grains and cannot be registered.

    Raises:
        ValueError: On a malformed or reserved name, or a name already taken.
    """
    if not _NAME_RE.match(name) or name in {"words", "characters", "tokens"}:
        raise ValueError(f"segmenter name {name!r} is reserved or malformed")

    def register(factory: Callable[[], Segmenter]) -> Callable[[], Segmenter]:
        if name in SEGMENTERS:
            raise ValueError(f"segmenter {name!r} is already registered")
        SEGMENTERS[name] = factory
        return factory

    return register


def attach_segmentation(name: str, stimuli: list[Stimulus]) -> None:
    """Attach the registered segmentation *name* as a layer of every stimulus.

    Args:
        name: A key of :data:`SEGMENTERS`.
        stimuli: Stimuli to annotate in place; one that already carries the
            layer with the same spans is left untouched.

    Raises:
        KeyError: When no segmenter is registered under *name*.
        ValueError: When a stimulus already carries a different layer of that
            name, since the cell would then not index the segmentation.
    """
    segmenter = SEGMENTERS[name]()
    if segmenter.name != name:
        raise ValueError(f"segmenter {name!r} labels its layer {segmenter.name!r}")
    for stimulus in stimuli:
        if name in stimulus.layers:
            if stimulus.layers[name].spans != segmenter(stimulus.text):
                raise ValueError(
                    f"{stimulus.stimulus_id}: layer {name!r} exists with other spans"
                )
            continue
        segment_stimulus(stimulus, segmenter)


# ---------------------------------------------------------------------------
# Built-in segmenters
# ---------------------------------------------------------------------------


class WhitespaceSegmenter(Segmenter):
    """Segment text on whitespace boundaries.

    Contiguous runs of non-whitespace characters form units.
    Whitespace characters are separators (not part of any unit).
    """

    def __init__(self):
        """Create a segmenter named ``"words"``."""
        super().__init__("words")

    def segment(self, text: str) -> list[Span]:
        """Split *text* into whitespace-delimited spans.

        Args:
            text: The raw text to split.
        """
        spans: list[Span] = []
        i = 0
        n = len(text)
        while i < n:
            while i < n and text[i].isspace():
                i += 1
            if i >= n:
                break
            start = i
            while i < n and not text[i].isspace():
                i += 1
            spans.append(Span(start, i))
        return spans


class CharacterSegmenter(Segmenter):
    """Each character is a unit.

    By default whitespace characters are skipped.  Set
    ``include_whitespace=True`` to include them.
    """

    def __init__(self, *, include_whitespace: bool = False):
        """Create a segmenter named ``"characters"``.

        Args:
            include_whitespace: Whether whitespace characters get their own
                unit instead of being skipped.
        """
        super().__init__("characters")
        self._include_ws = include_whitespace

    def segment(self, text: str) -> list[Span]:
        """Split *text* into one-character spans.

        Args:
            text: The raw text to split.
        """
        if self._include_ws:
            return [Span(i, i + 1) for i in range(len(text))]
        return [Span(i, i + 1) for i in range(len(text)) if not text[i].isspace()]


def from_predicate(
    predicate: Callable[[str, int], bool],
    name: str,
) -> Segmenter:
    """Build a segmenter from a separator predicate.

    Characters where ``predicate(text, i)`` returns True are treated as
    separators and excluded from all spans.  Contiguous runs of
    non-separator characters form units.

    Args:
        predicate: ``(text, position) → bool``.  Returns True for
            separator characters.
        name: Name for the resulting annotation layer.

    Returns:
        A :class:`Segmenter` instance.

    Example::

        word_seg = from_predicate(lambda t, i: t[i].isspace(), "words")
        hyphen_seg = from_predicate(lambda t, i: t[i] == '-', "parts")
    """

    class _PredicateSegmenter(Segmenter):
        def segment(self, text: str) -> list[Span]:
            spans: list[Span] = []
            i = 0
            n = len(text)
            while i < n:
                while i < n and predicate(text, i):
                    i += 1
                if i >= n:
                    break
                start = i
                while i < n and not predicate(text, i):
                    i += 1
                spans.append(Span(start, i))
            return spans

    return _PredicateSegmenter(name)


# ---------------------------------------------------------------------------
# Refinement and aggregation
# ---------------------------------------------------------------------------


def compute_refinement(
    source_spans: list[Span],
    target_spans: list[Span],
) -> list[list[int]]:
    """Compute how source spans map into target spans.

    Checks that *source_spans* **refine** *target_spans*: every source
    span must be fully contained within exactly one target span.

    Args:
        source_spans: Finer-grained spans (e.g. word spans from data).
        target_spans: Coarser-grained spans (e.g. sentence spans from
            a segmenter).

    Returns:
        A list of length ``len(target_spans)``.  Entry *i* is the list
        of source-span indices that fall within target span *i*.

    Raises:
        ValueError: If any source span is not fully contained in exactly
            one target span.
    """
    mapping: list[list[int]] = [[] for _ in target_spans]

    for s_idx, s_span in enumerate(source_spans):
        contained_in: int | None = None
        for t_idx, t_span in enumerate(target_spans):
            if t_span.start <= s_span.start and s_span.end <= t_span.end:
                if contained_in is not None:
                    raise ValueError(
                        f"Source span {s_idx} ({s_span}) is contained in "
                        f"multiple target spans: {contained_in} and {t_idx}. "
                        f"Target spans must be non-overlapping."
                    )
                contained_in = t_idx
        if contained_in is None:
            raise ValueError(
                f"Source span {s_idx} ({s_span.start}:{s_span.end}) is not "
                f"fully contained in any target span.  The data layer's "
                f"granularity is incompatible with the requested "
                f"segmentation."
            )
        mapping[contained_in].append(s_idx)

    return mapping


def aggregate_values(
    values: np.ndarray,
    mapping: list[list[int]],
) -> np.ndarray:
    """Sum source-level values into target-level units.

    For an additive measure (e.g. total fixation duration, SPR reading time)
    the sum within a coarser unit is that unit's value. A NaN source value makes
    its target value NaN.

    Args:
        values: Real per-source-unit values, shape ``(n_source,)``; masks are missing.
        mapping: Output of :func:`compute_refinement`.

    Returns:
        Per-target-unit values, shape ``(n_target,)``.
    """
    values = real_array_with_missing(values)
    result = np.empty(len(mapping))
    for t_idx, s_indices in enumerate(mapping):
        if not s_indices:
            result[t_idx] = np.nan
        else:
            group = values[s_indices]
            if np.any(np.isnan(group)):
                result[t_idx] = np.nan
            else:
                result[t_idx] = np.sum(group)
    return result


def aggregate_mask(
    mask: np.ndarray,
    mapping: list[list[int]],
) -> np.ndarray:
    """Aggregate a boolean exclusion mask from source to target level.

    A target unit is included only if **all** of its constituent source
    units are included (a partial sum is meaningless for additive
    measures).

    Args:
        mask: Per-source-unit inclusion flags, shape ``(n_source,)``.
        mapping: Output of :func:`compute_refinement`.
    """
    result = np.empty(len(mapping), dtype=bool)
    for t_idx, s_indices in enumerate(mapping):
        if not s_indices:
            result[t_idx] = False
        else:
            result[t_idx] = bool(np.all(mask[s_indices]))
    return result


# ---------------------------------------------------------------------------
# Stimulus segmentation
# ---------------------------------------------------------------------------


def _validate_spans(spans: list[Span], text_length: int) -> None:
    """Require ordered, nonempty, disjoint spans within the shared text."""
    previous_end = 0
    for span in spans:
        if span.start < previous_end or span.end > text_length or not len(span):
            raise ValueError(
                "Unit spans must be ordered, nonempty, non-overlapping and within text"
            )
        previous_end = span.end


def segment_stimulus(stimulus: Stimulus, segmenter: Segmenter) -> str:
    """Apply a segmenter to a stimulus, creating a new annotation layer.

    If the stimulus already has a layer with the segmenter's name **and**
    identical spans, the existing layer is reused.  If the layer name
    exists but spans differ, the segmenter's spans are stored under a
    ``_seg_<name>`` layer to avoid overwriting datum-aligned layers.

    Args:
        stimulus: The stimulus to segment.
        segmenter: The segmenter to apply.

    Returns:
        The name of the (possibly new) annotation layer.
    """
    spans = segmenter(stimulus.text)
    _validate_spans(spans, len(stimulus.text))
    layer_name = segmenter.name
    suffix = 1
    while layer_name in stimulus.layers:
        if stimulus.layers[layer_name].spans == spans:
            return layer_name
        layer_name = (
            f"_seg_{segmenter.name}"
            if suffix == 1
            else f"_seg_{segmenter.name}_{suffix}"
        )
        suffix += 1

    strings = [s.extract(stimulus.text) for s in spans]
    stimulus.add_layer(AnnotationLayer(layer_name, spans, strings))
    return layer_name
