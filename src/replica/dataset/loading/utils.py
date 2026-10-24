"""Shared utilities for dataset loaders."""

import contextlib
from collections.abc import (
    Callable,
    Hashable,
    Iterable,
    Mapping,
    MutableSequence,
    Sequence,
)
from typing import Any, TypeVar

import numpy as np

from ...core import Datum, Span, Stimulus

_Key = TypeVar("_Key", bound=Hashable)


def _check_measure_tuple(
    key: Hashable, values: Sequence[float], n_measures: int
) -> None:
    # Eager by design: callers validate every item before the index/skip
    # check, so a malformed tuple fails fast even on rows that would be
    # dropped: a packing-width mismatch is a loader bug.
    if len(values) != n_measures:
        raise ValueError(
            f"measure tuple for {key!r} has length {len(values)}; expected {n_measures}"
        )


def scatter_measure_tuples(
    items: Iterable[tuple[_Key, Sequence[float]]],
    index_of: Mapping[_Key, int],
    n_units: int,
    n_measures: int,
) -> list[np.ndarray]:
    """Scatter per-key measure tuples into ``n_measures`` full-length arrays.

    ``values[k]`` of each ``(key, values)`` record lands at row
    ``index_of[key]`` of array ``k``; keys absent from *index_of* and indices
    outside ``[0, n_units)`` are skipped.

    Args:
        items: ``(key, values)`` records, one tuple of measures per key.
        index_of: Record key to the 0-based unit position it belongs at.
        n_units: Length of each output array.
        n_measures: Number of output arrays, and the required tuple width.

    Returns:
        ``n_measures`` arrays of length ``n_units``, NaN wherever no record
        landed.

    Raises:
        ValueError: When a tuple's width is not *n_measures*.
    """
    arrays = [np.full(n_units, np.nan) for _ in range(n_measures)]
    for key, m_tuple in items:
        _check_measure_tuple(key, m_tuple, n_measures)
        idx = index_of.get(key)
        if idx is not None and 0 <= idx < n_units:
            for k, val in enumerate(m_tuple):
                arrays[k][idx] = val
    return arrays


def scatter_first_occurrence(
    items: Iterable[tuple[_Key, Sequence[float]]],
    index_of: Mapping[_Key, int],
    n_units: int,
    n_measures: int,
) -> list[np.ndarray]:
    """Like :func:`scatter_measure_tuples`, but first occurrence per key wins.

    A repeated key keeps the value of its first appearance in *items*, for
    loaders that record a word's first fixation. Keys absent from *index_of*
    and out-of-range indices are skipped.

    Args:
        items: ``(key, values)`` records in the order occurrence is judged by.
        index_of: Record key to the 0-based unit position it belongs at.
        n_units: Length of each output array.
        n_measures: Number of output arrays, and the required tuple width.

    Returns:
        ``n_measures`` arrays of length ``n_units``, NaN where no record landed.

    Raises:
        ValueError: When a tuple's width is not *n_measures*.
    """
    arrays = [np.full(n_units, np.nan) for _ in range(n_measures)]
    seen: set[Hashable] = set()
    for key, m_tuple in items:
        _check_measure_tuple(key, m_tuple, n_measures)
        idx = index_of.get(key)
        if idx is not None and 0 <= idx < n_units and key not in seen:
            seen.add(key)
            for k, val in enumerate(m_tuple):
                arrays[k][idx] = val
    return arrays


def scatter_values(mapping: Mapping[int, float], n: int) -> np.ndarray:
    """Scatter ``{index: value}`` into a length-``n`` array, NaN where absent.

    Indices outside ``[0, n)`` are skipped.

    Args:
        mapping: 0-based unit position to the value that belongs there.
        n: Length of the output array.
    """
    values = np.full(n, np.nan)
    for idx, val in mapping.items():
        if 0 <= idx < n:
            values[idx] = val
    return values


def append_measure_datums(
    output: MutableSequence[Datum],
    *,
    stimulus: Stimulus,
    participant_id: str,
    measure_names: Sequence[str],
    measure_values: Sequence[np.ndarray],
    metadata: dict | None = None,
) -> None:
    """Append one :class:`Datum` per aligned participant-level measure.

    Args:
        output: List the new datums are appended to.
        stimulus: Stimulus every datum is attached to.
        participant_id: Reader every datum is attributed to.
        measure_names: Measure name per array, in the same order.
        measure_values: Unit-aligned arrays, one per name.
        metadata: Extras copied onto each datum.

    Raises:
        ValueError: When the two sequences differ in length.
    """
    if len(measure_names) != len(measure_values):
        raise ValueError(
            f"got {len(measure_names)} measure names but "
            f"{len(measure_values)} value arrays"
        )
    for measure_name, values in zip(measure_names, measure_values, strict=True):
        output.append(
            Datum(
                stimulus=stimulus,
                measure_name=measure_name,
                values=values,
                participant_id=participant_id,
                metadata=dict(metadata or {}),
            )
        )


def parse_float(s: Any) -> float:
    """Convert a string to float, treating common missing markers as NaN.

    Handles ".", "", "NA", and "NaN" as missing values, conventions used
    across EyeLink exports and various eye-tracking CSV formats.

    Args:
        s: The raw cell; anything unparseable also becomes NaN.
    """
    if s is None or s in (".", "", "NA", "NaN"):
        return np.nan
    try:
        return float(s)
    except (TypeError, ValueError):
        return np.nan


def set_optional_meta(
    meta: dict,
    key: str,
    raw: Any,
    cast: Callable[[Any], Any],
    skip: tuple = (".", "", "NA"),
) -> None:
    """Set ``meta[key] = cast(raw)``, skipping missing markers and bad values.

    ``raw`` values in *skip* are ignored; a ``ValueError`` from *cast*
    leaves *meta* unchanged.

    Args:
        meta: Metadata dict updated in place.
        key: Name the converted value is stored under.
        raw: The cell to convert.
        cast: Converter applied to *raw*.
        skip: Raw values read as missing rather than converted.
    """
    if raw in skip:
        return
    with contextlib.suppress(ValueError):
        meta[key] = cast(raw)


def parse_reading_measure(value: object, *, unfixated: bool = False) -> float:
    """Preserve source duration zeros and distinguish unavailable measurements.

    Zero exclusion belongs to the dataset's ``drop_skips=True`` view,
    allowing the same loaded data to expose either skip policy.

    Args:
        value: The raw cell, in milliseconds where it is a real duration.
        unfixated: Independent source evidence (such as zero total dwell) that
            a missing duration denotes a skipped area. Only then fill it with zero.
    """
    numeric = parse_float(value)
    if np.isnan(numeric) and unfixated:
        return 0.0
    return numeric if np.isfinite(numeric) else np.nan


def first_pass_flag(skipped: object, *, unfixated: bool = False) -> float:
    """The first-pass flag from a release's first-pass skip column.

    Args:
        skipped: Release flag, nonzero when the unit was not fixated before the
            eye had moved past it.
        unfixated: Independent evidence that the unit received no fixation at
            all, so a missing flag still reads as not reached in first pass.

    Returns:
        1.0 when the unit was reached in first pass, 0.0 when it was not, NaN
        when the source flag is unavailable.
    """
    flag = parse_float(skipped)
    if np.isnan(flag):
        return 0.0 if unfixated else np.nan
    return 0.0 if flag else 1.0


def conditional_first_run(
    value: object, *, unfixated: bool = False
) -> tuple[float, float]:
    """First run and first-pass flag from a column reporting the run in first pass only.

    Parse first-pass columns such as ``fdurFP``, ``RTfirstpass``, and ``FPRT``.
    A positive duration establishes the first run and first-pass flag. Zero
    indicates no first-pass reading but does not establish later fixation
    duration. Missing cells remain unavailable without independent skip evidence.

    Args:
        value: The raw cell, in milliseconds.
        unfixated: Independent evidence (such as zero total dwell) that the
            unit received no fixation at all, which makes a zero or missing
            cell an observed first run of zero.

    Returns:
        ``(first_run, first_pass)``: a positive cell is the first run of a unit
        reached in first pass; a zero is a first run of zero on an unfixated
        unit and an unknown first run otherwise, both with the flag at 0; a
        missing cell on a fixated unit leaves both NaN.
    """
    duration = parse_reading_measure(value, unfixated=unfixated)
    if np.isnan(duration):
        return np.nan, np.nan
    if duration > 0:
        return duration, 1.0
    return (0.0 if unfixated else np.nan), 0.0


def build_word_spans(words: list[str], *, separator: str = " ") -> list[Span]:
    """Build character-offset spans for words joined with the given separator.

    Args:
        words: The words in text order.
        separator: Text inserted between words; empty for unspaced scripts.

    Returns:
        List of Span objects, one per word.
    """
    spans: list[Span] = []
    pos = 0
    for w in words:
        spans.append(Span(pos, pos + len(w)))
        pos += len(w) + len(separator)
    return spans
