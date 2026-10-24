"""Check stimulus text, measured units, span alignment, and duplicate IDs.

Checks use the layers referenced by dataset measurements, including character
and multiword units. Reports contain counts and capped examples per issue.
Dataset-specific exceptions are applied by callers; the scan reports all issues.
"""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass

from replica.core import Span
from replica.dataset.base import Dataset

_MAX_EXAMPLES_PER_BUCKET = 5


# Regexes used in the per-unit / per-text checks.
_BAD_CHARS_RE = re.compile(r"[\t\r]|\n{2,}")
# Horizontal Unicode spaces occur inside released multiword regions.
_CORRUPT_WS_RE = re.compile(r"[\t\r\n]| {2,}")


@dataclass(frozen=True)
class StimContentReport:
    """Result of running :func:`stim_content_issues` on one dataset."""

    counts: dict[str, int]
    """Total issue count per bucket name."""

    examples: dict[str, list[str]]
    """Up to ``_MAX_EXAMPLES_PER_BUCKET`` example details per bucket."""

    def issue_buckets(self) -> list[str]:
        """Sorted list of buckets with at least one issue."""
        return sorted(b for b, n in self.counts.items() if n > 0)

    def total_issues(self) -> int:
        """Total issue count across every bucket."""
        return sum(self.counts.values())


def _short(s: str, n: int = 80) -> str:
    """Render *s* for a report line, with whitespace made visible."""
    visible = str(s).replace("\t", "⇥").replace("\n", "⏎").replace("\xa0", "·")
    return visible if len(visible) <= n else visible[:n] + "…"


def _check_units(
    stim_id: str,
    words: list,
    recorder: _Recorder,
) -> None:
    """Per-unit validity: nullish, blank, interior whitespace.

    One example per bucket per stim, but counts reflect every offending unit
    (precedence nullish > blank > interior whitespace, one bucket per unit).
    """
    nullish: list[int] = []
    blank: list[int] = []
    ws_units: list[tuple[int, str]] = []
    for i, w in enumerate(words):
        if w is None or isinstance(w, float):  # NaN slips in as float
            nullish.append(i)
        elif not str(w).strip():
            blank.append(i)
        elif isinstance(w, str) and _CORRUPT_WS_RE.search(w):
            ws_units.append((i, w))

    if nullish:
        i = nullish[0]
        recorder.add(
            "nullish_unit",
            f"{stim_id}: word[{i}]={words[i]!r}",
            count=len(nullish),
        )
    if blank:
        i = blank[0]
        recorder.add(
            "blank_unit",
            f"{stim_id}: word[{i}]={words[i]!r}",
            count=len(blank),
        )
    if ws_units:
        i, w = ws_units[0]
        recorder.add(
            "unit_with_corrupt_whitespace",
            f"{stim_id}: word[{i}]={_short(w)} ({len(ws_units)} total)",
            count=len(ws_units),
        )


def _check_text(stim_id: str, text: str, recorder: _Recorder) -> None:
    """``stimulus.text`` should not carry TAB / CR / double newline."""
    m = _BAD_CHARS_RE.search(text)
    if m is None:
        return
    ctx_start = max(0, m.start() - 20)
    ctx_end = min(len(text), m.end() + 20)
    recorder.add(
        "text_has_tab_or_double_newline",
        f"{stim_id} @char {m.start()}: …{_short(text[ctx_start:ctx_end])}…",
    )


def _uncovered_characters(text: str, spans) -> list[int]:
    """Positions of uncovered characters other than whitespace and punctuation.

    Args:
        text: The stimulus text.
        spans: The layer's spans, or ``None`` when it declares none.
    """
    if not spans:
        return []
    covered = bytearray(len(text))
    for span in spans:
        start = max(0, min(span.start, len(text)))
        end = max(0, min(span.end, len(text)))
        covered[start:end] = b"\x01" * (end - start)
    # Source areas can omit displayed punctuation, such as CELER's outer quotes.
    return [
        i
        for i, c in enumerate(text)
        if not covered[i]
        and not c.isspace()
        and not unicodedata.category(c).startswith("P")
    ]


def _check_span_alignment(
    stim_id: str,
    text: str,
    words: list,
    spans,
    recorder: _Recorder,
) -> None:
    """``words[i] == text[span.start:span.end]`` for all i."""
    if spans is None:
        recorder.add("missing_word_spans", f"{stim_id}: no spans on layer")
        return
    first_mismatch: tuple[int, str, str] | None = None
    n_mismatch = 0
    for i, (sp, w) in enumerate(zip(spans, words, strict=False)):
        try:
            chunk = text[sp.start : sp.end]
        except (AttributeError, IndexError, TypeError):
            chunk = "<index-error>"
        if str(chunk) != str(w):
            n_mismatch += 1
            if first_mismatch is None:
                first_mismatch = (i, chunk, w)
    if n_mismatch:
        i, chunk, w = first_mismatch  # type: ignore[misc]
        recorder.add(
            "span_text_mismatch",
            f"{stim_id}: word[{i}]={_short(w)} but "
            f"text[span]={_short(chunk)} ({n_mismatch} total)",
            count=n_mismatch,
        )


class _Recorder:
    """Track counts and capped examples per issue bucket."""

    def __init__(self, cap: int = _MAX_EXAMPLES_PER_BUCKET) -> None:
        self._cap = cap
        self.counts: dict[str, int] = defaultdict(int)
        self.examples: dict[str, list[str]] = defaultdict(list)

    def add(self, bucket: str, detail: str, count: int = 1) -> None:
        self.counts[bucket] += count
        if len(self.examples[bucket]) < self._cap:
            self.examples[bucket].append(detail)

    def to_report(self) -> StimContentReport:
        return StimContentReport(
            counts=dict(self.counts),
            examples={k: list(v) for k, v in self.examples.items()},
        )


def stim_content_issues(ds: Dataset) -> StimContentReport:
    """Return content issue counts and examples for a loaded dataset.

    Args:
        ds: A loaded :class:`Dataset` (the result of
            ``DatasetRegistration.factory()``).

    Notes:
        The scan is a single pass over ``ds.stimuli()``; cost scales
        linearly with the total measured units. The duplicate-input check
        stores each distinct text and measured-unit layout.
    """
    recorder = _Recorder()
    seen_ids: set[str] = set()
    seen_inputs: dict[tuple[str, tuple[tuple[str, tuple[Span, ...]], ...]], str] = {}
    # Check the annotation layers referenced by the measurements.
    unit_layers = {datum.layer for datum in ds.data()} or {"words"}

    for s in ds.stimuli():
        sid = s.stimulus_id

        if sid in seen_ids:
            recorder.add("duplicate_stimulus_id", f"sid={sid!r} appears twice")
        seen_ids.add(sid)

        present = [name for name in unit_layers if name in s.layers]
        if not present:
            recorder.add(
                "missing_unit_layer",
                f"{sid}: carries none of the measured layers {sorted(unit_layers)}",
            )
            continue

        text = s.text or ""
        _check_text(sid, text, recorder)

        for name in present:
            layer = s.layers[name]
            units = list(layer.values)
            where = f"{sid}[{name}]"
            if not units:
                recorder.add("zero_unit_stim", f"{where}: empty layer")
                continue

            _check_units(where, units, recorder)

            uncovered = _uncovered_characters(text, getattr(layer, "spans", None))
            if uncovered:
                recorder.add(
                    "text_not_covered_by_units",
                    f"{where}: {len(uncovered)} characters in no unit, "
                    f"first at {uncovered[0]} ({text[uncovered[0]]!r})",
                    count=len(uncovered),
                )

            _check_span_alignment(
                where,
                text,
                units,
                getattr(layer, "spans", None),
                recorder,
            )

        # Identical text can have different source measurement regions.
        layout = tuple(
            (name, tuple(s.layers[name].spans or ())) for name in sorted(present)
        )
        input_key = (text, layout)
        prev = seen_inputs.get(input_key)
        if prev is not None and prev != sid:
            recorder.add(
                "duplicate_stim_text",
                f"{sid} text and measured regions identical to {prev!r}",
            )
        else:
            seen_inputs[input_key] = sid

    return recorder.to_report()
