"""Dundee Corpus V.1 (Kennedy et al., 2003), the author-distributed release.

The release ships raw event records: one file per participant and text in
event order (``s<x><nn>ma1p.dat``), plus one word-object file per text
(``items/tx<nn>wrdp.dat``). Its own notes state that no derived measures are
provided because several definitions exist, so the reading measures here are
computed from the event stream.

Access is by permission from the corpus author, so there is no download; point
``$PSYCHO_DATA_DIR/eye-tracking/original/dundee`` at the ``dundee_corpus``
directory of the distribution.

The distinct loader :class:`~replica.dataset.eye_tracking.dundee.shain2024.Dundee` reads
the Shain et al. (2024) preprocessing of the same corpus, which is public and
already carries per-word measures. That preprocessing includes interruption
time in some measures; this loader sums fixation dwell only.
"""

import re
from pathlib import Path

import numpy as np

from replica.dataset.base import EyeTrackingDataset
from replica.dataset.contracts import DatasetInfo, Fixation
from replica.dataset.metadata import io as metadata_io

from ....core import Stimulus
from ...loading.fixation_utils import (
    SyntheticTimer,
    append_fixation,
    first_pass_flags,
    first_run_durations,
    go_past_duration,
    mark_unfixated,
    uncertain_fixation_masks,
)
from ...loading.utils import append_measure_datums, scatter_first_occurrence
from .source import read_display_words

# Event-order (MA1P) fields after the word: the word itself can be absent, so
# rows are split from the right.
_EVENT_FIELDS = 11
_SCREEN, _LINE, _OLEN, _WLEN, _XPOS, _WNUM, _FDUR, _OBLP, _WDLP, _LAUN = range(10)

# The release codes missing values as -99, including an event that fell on no
# word on any line ("off-screen").
_MISSING = -99

# The release was written on DOS: its accented words are CP437, and reading
# them as latin-1 turns them into control characters (`Valéry` -> `Val\x82ry`).
_ENCODING = "cp437"

# The WORD column names the event when it fell on no word. Notes.txt separates
# a blink from a fixation that landed off text; both interrupt a word's reading
# without being a fixation on another word.
_OFF_SCREEN = "*Off-screen"
_BLINK = "*Blink"

_SUBJECT_FILE = re.compile(r"^s([a-z])(\d\d)ma1p\.dat$")

DUNDEE_MEASURES = (
    "first_fixation",
    "first_run",
    "first_pass",
    "go_past",
    "go_past_in_first_pass",
    "total_time",
)

# (kind, unit index, duration, raw fields); kind is fixation, blink or
# off-screen.
_Event = tuple[str, "int | None", float, list[int]]


class DundeeOfficial(EyeTrackingDataset):
    """Dundee Corpus V.1, read from the author-distributed release.

    Measures are computed from the event stream: ``first_fixation`` is the
    first fixation on the word, ``first_run`` its first run whenever that
    occurs, ``first_pass`` whether the eye reached it before going past it,
    ``go_past`` the regression path from first arrival to its right,
    ``go_past_in_first_pass`` the same window only on first-pass entry, and
    ``total_time`` the sum of fixations on that word. Blinks and off-word
    events contribute no word dwell. Off-word fixations end a run; blinks
    preserve its landing sequence. Known skips are raw zeros, excluded by
    default; ``drop_skips=False`` includes them. Unaligned or unavailable
    measurements remain NaN.

    One Stimulus is one of the 20 texts. Fixation ``x``/``y`` are NaN because
    the release reports no screen coordinates; its character position on an
    81-column line is kept in fixation metadata.
    """

    info = DatasetInfo(name="Dundee", language="en")
    _declared_measures = DUNDEE_MEASURES
    additive_measures = frozenset({"total_time"})

    def load(self, path: str, verify: bool = True) -> None:
        """Read the word-object files and every participant's event stream.

        Args:
            path: The distribution's ``dundee_corpus`` directory, holding
                ``english/items/`` and the per-participant ``.dat`` files.
            verify: Check the source files against the checksums the dataset
                YAML declares.
        """
        self._load_metadata(
            path, verify=verify, metadata_path=metadata_io.find_dataset_yaml(path)
        )
        english = Path(path) / "english"
        if not english.is_dir():
            raise FileNotFoundError(
                f"Dundee V.1 release not found at {english}. The corpus is "
                "distributed by its author on request; place the "
                "'dundee_corpus' directory at "
                f"{path} so that {english} holds the .dat files."
            )
        texts = self._read_texts(english / "items")
        self._read_participants(english, texts)

    def _read_texts(self, items: Path) -> dict[int, tuple[Stimulus, int]]:
        """Read original display text and validate its word-object geometry."""
        texts: dict[int, tuple[Stimulus, int]] = {}
        self._source_geometry: dict[int, list[tuple[int, int]]] = {}
        for word_file in sorted(items.glob("tx*wrdp.dat")):
            number = int(word_file.name[2:4])
            original = read_display_words(items, number)
            words = [word.text for word in original]
            self._source_geometry[number] = [
                (word.screen, word.line) for word in original
            ]
            stimulus = self._make_stimulus(
                words,
                f"dundee_text{number:02d}",
                {"text_number": number, "text_source": "Dundee V.1 display files"},
            )
            texts[number] = (stimulus, len(words))
        return texts

    def _read_participants(
        self, english: Path, texts: dict[int, tuple[Stimulus, int]]
    ) -> None:
        self._fixations: list[Fixation] = []
        for event_file in sorted(english.glob("s*ma1p.dat")):
            match = _SUBJECT_FILE.match(event_file.name.lower())
            if match is None:
                continue
            participant = f"s{match.group(1)}"
            number = int(match.group(2))
            entry = texts.get(number)
            if entry is None:
                continue
            stimulus, n_words = entry
            events = self._read_events(
                event_file, n_words, self._source_geometry[number]
            )
            if events:
                self._add_text(stimulus, n_words, participant, events)

    @staticmethod
    def _read_events(
        event_file: Path, n_words: int, geometry: list[tuple[int, int]]
    ) -> list[_Event]:
        """Parse one event file into (kind, unit index, duration, fields).

        NUL-padded rows have no word but can carry elapsed time. Screen/line
        conflicts keep their reported index for auditing affected aggregates;
        the public fixation has no assigned unit.
        """
        events: list[_Event] = []
        for row_number, line in enumerate(
            event_file.read_text(_ENCODING).splitlines(), 1
        ):
            if not line.strip():
                continue
            parts = line.rsplit(None, _EVENT_FIELDS)
            if parts[:2] == ["WORD", "TEXT"]:
                continue
            if len(parts) != _EVENT_FIELDS + 1:
                raise ValueError(
                    f"Malformed Dundee event in {event_file.name}:{row_number}"
                )
            try:
                numbers = [int(value) for value in parts[1:]]
            except ValueError as error:
                raise ValueError(
                    f"Malformed Dundee event in {event_file.name}:{row_number}"
                ) from error
            word = parts[0].strip()
            word_number = numbers[_WNUM]
            if 1 <= word_number <= n_words:
                valid = (numbers[_SCREEN], numbers[_LINE]) == geometry[word_number - 1]
                kind = "fixation" if valid else "unaligned"
                unit: int | None = word_number - 1
            elif word == _BLINK:
                kind, unit = "blink", None
            elif word == _OFF_SCREEN:
                kind, unit = "off-screen", None
            elif word_number == 0 and not word.strip("\x00"):
                if numbers[_FDUR] == 0:
                    continue
                kind, unit = "gap", None
            else:
                raise ValueError(f"Unidentified Dundee event in {event_file.name}")
            if numbers[_FDUR] < 0:
                raise ValueError(f"Negative Dundee duration in {event_file.name}")
            events.append((kind, unit, float(numbers[_FDUR]), numbers))
        return events

    @staticmethod
    def _reading_stream(events: list[_Event]) -> list[tuple[int | None, float]]:
        """Fixations in event order; off-word landings interrupt word runs.

        Notes.txt distinguishes blinks and gaps from fixations. Their elapsed
        time advances the published clock but contributes no fixation dwell.
        """
        return [
            (unit, duration)
            for kind, unit, duration, _ in events
            if kind in {"fixation", "unaligned", "off-screen"}
        ]

    def _add_text(
        self,
        stimulus: Stimulus,
        n_words: int,
        participant: str,
        events: list[_Event],
    ) -> None:
        stream = self._reading_stream(events)
        units = [unit for unit, _ in stream]
        durations = [duration for _, duration in stream]
        fixated = {i: i for i in range(n_words)}

        (first_fixation,) = scatter_first_occurrence(
            (
                (unit, (duration,))
                for kind, unit, duration, _ in events
                if kind in {"fixation", "unaligned"} and unit is not None
            ),
            fixated,
            n_words,
            1,
        )
        total_time = np.full(n_words, np.nan)
        for unit, duration in zip(units, durations, strict=True):
            if unit is None:
                continue
            previous = total_time[unit]
            total_time[unit] = duration + (0.0 if np.isnan(previous) else previous)
        first_run = first_run_durations(
            units, durations, fixated, n_words, off_text_ends_run=True
        )
        first_pass = first_pass_flags(units, n_words)
        go_past = go_past_duration(units, durations, n_words)
        mark_unfixated(
            [first_fixation, first_run, go_past, total_time],
            range(n_words),
            (unit for unit in units if unit is not None),
        )

        conditional_go_past = np.where(first_pass == 1, go_past, 0.0)

        first_event = len(self._fixations)
        self._append_fixations(stimulus, participant, events)
        masks, uncertain_first_pass, uncertain_total = uncertain_fixation_masks(
            [self._fixations[first_event:]], n_words
        )
        masks["go_past_in_first_pass"] = masks["go_past"] | masks["first_pass"]

        unavailable = []
        arrays = dict(
            zip(
                DUNDEE_MEASURES,
                (
                    first_fixation,
                    first_run,
                    first_pass,
                    go_past,
                    conditional_go_past,
                    total_time,
                ),
                strict=True,
            )
        )
        for measure, values in arrays.items():
            for unit in np.flatnonzero(masks[measure] & np.isfinite(values)):
                unavailable.append(
                    {
                        "measure": measure,
                        "unit_index": int(unit),
                        "value": float(values[unit]),
                    }
                )
                values[unit] = np.nan

        append_measure_datums(
            self._data,
            stimulus=stimulus,
            participant_id=participant,
            measure_names=DUNDEE_MEASURES,
            measure_values=(
                first_fixation,
                first_run,
                first_pass,
                go_past,
                conditional_go_past,
                total_time,
            ),
            metadata={
                "unavailable_source_values": unavailable,
                "uncertain_first_pass_units": sorted(uncertain_first_pass),
                "uncertain_total_time_units": sorted(uncertain_total),
            },
        )

    def _append_fixations(
        self,
        stimulus: Stimulus,
        participant: str,
        events: list[_Event],
    ) -> None:
        """Publish fixation events with elapsed blink and gap time in the clock."""
        offsets = [span.start for span in stimulus.layers["words"].spans]
        timer = SyntheticTimer()
        trial = (participant, stimulus.stimulus_id)
        geometry = self._source_geometry[stimulus.metadata["text_number"]]
        index = 0
        for kind, unit, duration, numbers in events:
            start_time, end_time = timer.next(trial, duration)
            if kind in {"blink", "gap"}:
                continue
            candidates = ()
            if kind == "unaligned":
                alternatives = [
                    i
                    for i, position in enumerate(geometry)
                    if position == (numbers[_SCREEN], numbers[_LINE]) and i != unit
                ]
                candidates = (unit, *(alternatives or [None]))
                unit = None
            index += 1
            append_fixation(
                self._fixations,
                participant_id=participant,
                stimulus_id=stimulus.stimulus_id,
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                x=float("nan"),
                y=float("nan"),
                unit_index=unit,
                char_offset=offsets[unit] if unit is not None else None,
                fixation_index=index,
                metadata=self._event_metadata(kind, numbers),
                candidate_unit_indices=candidates,
            )

    @staticmethod
    def _event_metadata(kind: str, numbers: list[int]) -> dict:
        metadata: dict = {"synthetic_start_time": True, "event": kind}
        if kind == "unaligned":
            metadata["source_word_number"] = numbers[_WNUM]
            metadata["alignment_error"] = "source_screen_or_line_mismatch"
        for key, field in (
            ("screen", _SCREEN),
            ("line", _LINE),
            ("xpos_chars", _XPOS),
            ("object_landing_position", _OBLP),
            ("word_landing_position", _WDLP),
            ("saccade_length", _LAUN),
            ("object_length", _OLEN),
            ("word_length", _WLEN),
        ):
            # Off-word events carry -99 here, and stale lengths from the word
            # last fixated; neither describes the event.
            if numbers[field] != _MISSING and kind in {"fixation", "unaligned"}:
                metadata[key] = numbers[field]
        return metadata
