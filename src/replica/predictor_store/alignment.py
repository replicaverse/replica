"""Whether a cell's rows index the units a corpus loader produces.

A cell is aligned with its corpus when every stimulus it covers exists, holds
one row per unit and predictor at indices ``0..n-1``, and names each unit with
the loader's string, compared after NFC normalization. The export, the
readback and the precomputed-feature path all decide alignment here.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral

#: Reason codes, in the order a cell's failures are reported.
REASONS = ("ids_extra", "unit_count", "unit_index_gap", "text_null", "text_mismatch")


@dataclass(frozen=True)
class Misalignment:
    """One reason a cell disagrees with the loader, with the stimuli it affects."""

    reason: str
    stimuli: tuple[str, ...]
    example: str

    def __str__(self) -> str:
        return f"{self.reason} on {len(self.stimuli)} stimuli ({self.example})"


def nfc(text: str) -> str:
    """NFC form of *text*.

    Args:
        text: Any string.
    """
    return unicodedata.normalize("NFC", text)


def same_text(stored: str, loader: str) -> bool:
    """Whether a stored unit string names the loader's unit.

    Args:
        stored: The unit string a cell's row carries.
        loader: The unit string the corpus loader produces.
    """
    return stored == loader or nfc(stored) == nfc(loader)


_NO_TEXT = object()


class RowCheck:
    """Accumulate a cell's rows and judge them against the loader's units.

    Rows arrive one at a time, so a table of millions of rows is checked
    with counters and one bit per unit and predictor.

    Args:
        words: Unit strings of every stimulus the loader produces.
    """

    def __init__(self, words: Mapping[str, Sequence[str]]):
        """Start with no rows seen.

        Args:
            words: Unit strings of every stimulus the loader produces.
        """
        self.words = words
        self.extra: set[str] = set()
        self.null: dict[str, str] = {}
        self.differs: dict[str, str] = {}
        # (stimulus, predictor) -> [rows, lowest index, highest index]
        self.seen: dict[tuple[str, str], list[int]] = {}
        self.positions: dict[tuple[str, str], bytearray] = {}
        self.invalid: dict[str, str] = {}

    def add(self, sid: str, index: int, text: object = _NO_TEXT, key: str = "") -> None:
        """Record one row.

        Args:
            sid: The row's stimulus id.
            index: The row's unit index.
            text: The row's unit string, or ``None`` for a row that should
                name its unit and does not; left out for records that carry
                no unit strings, which skips the text checks.
            key: The row's predictor, so a cell storing several keys counts
                each on its own.
        """
        expected = self.words.get(sid)
        if expected is None:
            self.extra.add(sid)
            return
        if not isinstance(index, Integral) or isinstance(index, bool):
            self.invalid.setdefault(sid, f"{sid}: invalid unit index {index!r}")
            return
        index = int(index)
        seen = self.seen.get((sid, key))
        if seen is None:
            self.seen[sid, key] = [1, index, index]
            self.positions[sid, key] = bytearray((len(expected) + 7) // 8)
        else:
            seen[0] += 1
            seen[1] = min(seen[1], index)
            seen[2] = max(seen[2], index)
        if 0 <= index < len(expected):
            positions = self.positions[sid, key]
            byte, bit = divmod(index, 8)
            mask = 1 << bit
            if positions[byte] & mask:
                self.invalid.setdefault(sid, f"{sid}: duplicate unit index {index}")
            positions[byte] |= mask
        if text is _NO_TEXT or sid in self.differs:
            return
        if not isinstance(text, str):
            self.null.setdefault(sid, f"{sid} unit {index}")
        elif (
            isinstance(text, str)
            and 0 <= index < len(expected)
            and not same_text(text, expected[index])
        ):
            self.differs[sid] = (
                f"{sid} unit {index}: stored {text!r}, loader {expected[index]!r}"
            )

    def problems(self) -> list[Misalignment]:
        """The failed reasons in :data:`REASONS` order, empty when aligned."""
        counts: dict[str, str] = {}
        gaps: dict[str, str] = dict(self.invalid)
        for (sid, _), (rows, low, high) in self.seen.items():
            n = len(self.words[sid])
            if rows != n:
                counts.setdefault(sid, f"{sid}: {rows} units stored, {n} in the loader")
            elif n and (low != 0 or high != n - 1):
                gaps.setdefault(sid, f"{sid}: unit indices {low}..{high} for {n} units")
        # A stimulus off the grid by count or position differs in its strings
        # as a consequence, so the text reasons name only stimuli on the grid.
        off_grid = set(counts) | set(gaps)
        gaps = {sid: why for sid, why in gaps.items() if sid not in counts}
        found = {
            "ids_extra": {sid: f"{sid} is not in the loader" for sid in self.extra},
            "unit_count": counts,
            "unit_index_gap": gaps,
            "text_null": {s: why for s, why in self.null.items() if s not in off_grid},
            "text_mismatch": {
                s: why for s, why in self.differs.items() if s not in off_grid
            },
        }
        return [
            Misalignment(reason, tuple(sorted(failed)), failed[min(failed)])
            for reason in REASONS
            if (failed := found[reason])
        ]


def check_rows(
    words: Mapping[str, Sequence[str]],
    stimulus_ids: Sequence[str],
    unit_indices: Sequence[int],
    texts: Sequence[str | None] | None = None,
    keys: Sequence[str] | None = None,
) -> list[Misalignment]:
    """Compare one cell's rows with the loader's units.

    Args:
        words: Unit strings of every stimulus the loader produces.
        stimulus_ids: Stimulus id of each row.
        unit_indices: Unit index of each row.
        texts: Stored unit string of each row; ``None`` skips the text checks,
            for rows that carry none.
        keys: Predictor of each row; ``None`` treats every row as one key.

    Returns:
        The failed reasons in :data:`REASONS` order, empty when aligned.

    Raises:
        ValueError: Row columns have different lengths.
    """
    for name, values in (("texts", texts), ("keys", keys)):
        if values is not None and len(values) != len(stimulus_ids):
            raise ValueError(f"{name} must have one value per stimulus id")
    check = RowCheck(words)
    for row, (sid, index) in enumerate(zip(stimulus_ids, unit_indices, strict=True)):
        check.add(
            sid,
            index,
            _NO_TEXT if texts is None else texts[row],
            keys[row] if keys is not None else "",
        )
    return check.problems()


def describe(problems: Iterable[Misalignment]) -> str:
    """One line naming every failed reason.

    Args:
        problems: The failures of one cell.
    """
    return "; ".join(str(problem) for problem in problems)
