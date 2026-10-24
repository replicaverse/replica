"""Shared reading-measure primitives for MECO's word-level loaders."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from ...loading.utils import (
    first_pass_flag,
    parse_reading_measure,
    scatter_measure_tuples,
)

# R columns retain the released quantities; standard first-run uses fixations.
MEASURE_COLUMNS: dict[str, str] = {
    "firstfix.dur": "first_fixation",
    "firstrun.dur": "first_run",
    "firstrun.gopast": "go_past_in_first_pass",
    "dur": "total_time",
}

#: First-pass skip flag, inverted into the ``first_pass`` flag.
SKIP_COLUMN = "firstrun.skip"


def clean_word(value: object) -> str:
    """Normalize one MECO AOI label without changing unit boundaries.

    Args:
        value: One ``word`` cell of a MECO frame. Internal whitespace is retained
            inside its measured region. Blank/nonword areas must be handled
            separately with their source measurements before calling this helper.
    """
    word = str(value).strip()
    if not word or word.lower() == "nan":
        raise ValueError("MECO blank area requires explicit nonword handling")
    return " ".join(word.split())


def read_r_dataframe(path: str | Path, key: str = "joint.data") -> Any:
    """Read one named dataframe from an RDA with consistent diagnostics.

    Args:
        path: The ``.rda`` file shipped by a MECO release.
        key: Object name to pull out of the archive. An absent key raises with
            the names the file does carry.

    Returns:
        The named object as a pandas frame; the rest of the archive is dropped.
    """
    import pyreadr

    result = pyreadr.read_r(str(path))
    if key not in result:
        raise ValueError(f"Expected {key!r} in {path}, found: {list(result)}")
    frame = result.pop(key)
    del result
    return frame


def present_measures(frame: Any, *, release_name: str) -> list[tuple[str, str]]:
    """Return supported source/canonical measure pairs, requiring at least one.

    Args:
        frame: Reading-measure frame whose columns are matched against
            ``MEASURE_COLUMNS``.
        release_name: Release label quoted in the error raised when the frame
            carries none of the four supported duration columns.

    Returns:
        ``(source column, canonical measure)`` pairs in ``MEASURE_COLUMNS``
        order, which fixes the column order of the arrays built from them.
    """
    present = [
        (column, measure)
        for column, measure in MEASURE_COLUMNS.items()
        if column in frame.columns
    ]
    if not present:
        raise ValueError(f"{release_name} has no supported reading measures")
    if SKIP_COLUMN in frame.columns:
        present.append((SKIP_COLUMN, "first_pass"))
    return present


def measures_from_row(
    get: Callable[[str], object],
    present: list[tuple[str, str]],
    *,
    unfixated: bool = False,
) -> dict[str, float]:
    """Parse one source row into the canonical measures listed in *present*.

    Args:
        get: Returns the row's raw value for a source column.
        present: Source/canonical pairs from :func:`present_measures`. The
            ``first_pass`` pair inverts the release's skip flag.
        unfixated: The row establishes a skip, so missing durations read as
            zero and a missing flag as not reached in first pass.
    """
    values: dict[str, float] = {}
    for column, measure in present:
        if measure == "first_pass":
            values[measure] = first_pass_flag(get(column), unfixated=unfixated)
        else:
            values[measure] = parse_reading_measure(get(column), unfixated=unfixated)
    return values


def twelve_item_inventory(frame: Any, *, release_name: str) -> list[Any]:
    """Return MECO's numerically sorted twelve-item source inventory.

    Args:
        frame: Reading-measure frame carrying an ``itemid`` column.
        release_name: Release label quoted in the error raised when the frame
            does not hold exactly twelve texts.

    Returns:
        The source ``itemid`` values sorted numerically, so that a site's own
        numbering maps position-wise onto public text IDs 1 to 12.
    """
    item_ids = sorted(frame["itemid"].unique(), key=int)
    if len(item_ids) != 12:
        raise ValueError(
            f"Expected 12 texts in {release_name}, found "
            f"{len(item_ids)} item IDs: {item_ids}"
        )
    return item_ids


def scatter_frame_measures(
    frame: Any,
    present: list[tuple[str, str]],
    word_index: dict[int, int],
    n_words: int,
) -> list[np.ndarray]:
    """Convert one participant's sparse MECO rows to aligned dense arrays.

    Args:
        frame: One participant's rows for one text, one row per source AOI.
        present: Source/canonical measure pairs from :func:`present_measures`,
            whose order fixes the order of the returned arrays.
        word_index: Source ``wordnum`` to its 0-based position in the stimulus.
            A row whose ``wordnum`` is absent is dropped.
        n_words: Length of each output array, the stimulus word count.

    Returns:
        One array per pair in *present*, in milliseconds. Released zeros are
        retained; unavailable or unaligned measurements remain NaN.
    """
    measure_values = {column: frame[column].to_numpy() for column, _ in present}
    word_numbers = frame["wordnum"].to_numpy()
    items = []
    for row_index, word_number in enumerate(word_numbers):
        unfixated = "dur" in measure_values and measure_values["dur"][row_index] == 0.0
        cells = {column: measure_values[column][row_index] for column, _ in present}
        values = measures_from_row(
            cells.__getitem__, present, unfixated=bool(unfixated)
        )
        items.append((int(word_number), tuple(values[m] for _, m in present)))
    return scatter_measure_tuples(items, word_index, n_words, len(present))
