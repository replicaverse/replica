"""CELER: eye-tracking corpus of L1 and L2 English reading (Berzak et al., 2022).

Eye-tracking data for 365 participants (69 L1 English, 296 L2 English) reading
WSJ newswire sentences. Each participant reads 156 sentences: 78 shared
across participants (the "Shared Text" regime) and 78 unique (the "Individual
Text" regime).

Underlying texts come from PTB-WSJ (LDC95T7) and BLLIP (LDC2000T43); the
eye-tracking data ships with the SR DataViewer Interest Area report
(``sent_ia.tsv``). The optional ``sent_fix.tsv`` report supplies physical
fixations for validated reaggregation on custom units.

Reference: Berzak, Y., Nakamura, C., Smith, A., Weng, E., Katz, B.,
    Flynn, S., & Levy, R. (2022). CELER: A 365-Participant Corpus of Eye
    Movements in L1 and L2 English Reading. *Open Mind*.
    https://doi.org/10.1162/opmi_a_00054
Source: https://github.com/berzak/celer (requires LDC access to PTB-WSJ
    and BLLIP before ``python obtain_data.py`` will release the eye-tracking
    archive).
"""

import csv
import sys
from itertools import groupby
from pathlib import Path

import numpy as np

from replica.dataset.base import EyeTrackingDataset
from replica.dataset.contracts import DatasetInfo
from replica.dataset.measures import (
    EYELINK_IA_MEASURES,
    EYELINK_MEASURE_NAMES,
    eyelink_measures,
)
from replica.dataset.metadata import io as metadata_io

from ...core import AnnotationLayer, Datum, Span, Stimulus
from ..loading.fixation_utils import SyntheticTimer, append_fixation
from ..loading.utils import build_word_spans, parse_float, parse_reading_measure

DATA_FILE = "sent_ia.tsv"
FIXATION_FILE = "sent_fix.tsv"
_TRIAL_FIELDS = ("trial", "list", "dataset_version", "EYE_USED")
_SUMMARY_FIELDS = (*_TRIAL_FIELDS, "TRIAL_FIXATION_COUNT", "TRIAL_DWELL_TIME")
_FIXATION_FIELDS = {
    "DATA_FILE",
    "sentenceid",
    *_TRIAL_FIELDS,
    *(
        f"CURRENT_FIX_{name}"
        for name in (
            "INDEX",
            "DURATION",
            "X",
            "Y",
            "INTEREST_AREA_ID",
            "INTEREST_AREA_LABEL",
            "INTEREST_AREA_LEFT",
            "INTEREST_AREA_RIGHT",
            "INTEREST_AREA_RUN_ID",
        )
    ),
}

# Author-documented unread trials, keyed by experimental list and trial number.
# https://github.com/berzak/celer/blob/163e1635cf399ac7367dd845d1bf7f27d092e93b/documentation/known_issues.md
_UNREAD_TRIALS = {
    (37, 45),
    (89, 10),
    (127, 27),
    (169, 1),
    (178, 1),
    (183, 1),
    (277, 40),
    (289, 47),
    (308, 138),
    (317, 106),
    (355, 141),
}


# Missing sentence text can be reconstructed from the ordered interest areas.
_INVALID_SENTENCE = frozenset({"", "nan", "na"})


def _is_valid_sentence(value: str | None) -> bool:
    """True if ``value`` is a usable sentence string (not empty/NaN)."""
    if value is None:
        return False
    return value.strip().lower() not in _INVALID_SENTENCE


def _parse(value: str, *, unfixated: bool = False) -> float:
    """Parse an EyeLink Interest Area duration value.

    Preserve zeros. A missing cell is zero only when the row establishes
    that the word received no fixation; otherwise it stays unavailable.

    Args:
        value: Released duration cell.
        unfixated: The area's total dwell time is zero.
    """
    return parse_reading_measure(value, unfixated=unfixated)


class CELER(EyeTrackingDataset):
    """CELER eye-tracking corpus (Berzak et al., 2022).

    Loader exposes 5 word-level reading-time measures, derived from the
    SR DataViewer Interest Area report:

      * ``first_fixation``, first fixation duration on each word
        (``IA_FIRST_FIXATION_DURATION``).
      * ``first_run``, dwell time of the first run on the word
        (``IA_FIRST_RUN_DWELL_TIME``), whenever that run occurs.
      * ``first_pass``, whether the word was reached in first pass
        (``IA_SKIP`` is 0); ``first_run_in_first_pass`` is gaze duration.
      * ``go_past``, regression-path / go-past duration
        (``IA_REGRESSION_PATH_DURATION``).
      * ``total_time``, total dwell time on the word (``IA_DWELL_TIME``).

    Each unique ``sentenceid`` becomes one Stimulus (the words layer is
    built from the ``IA_LABEL`` column ordered by ``IA_ID``); each
    ``(participant, sentenceid)`` pair contributes one Datum per measure.
    ``participant_id`` is the EDF filename from the source ``DATA_FILE``
    column (e.g. ``"cl2200.edf"``), so it can be joined with
    ``participant_metadata/`` on the same key.

    Default measure is ``first_run_in_first_pass``.

    ``sent_fix.tsv`` is optional for native summaries and required for event
    access. Fixation timestamps are synthetic cumulative dwell because the
    report omits onsets. Source run IDs are retained as provenance; measures
    rebuild runs after mapping events to the requested units.
    """

    info = DatasetInfo(name="celer", language="en")
    _declared_measures = EYELINK_MEASURE_NAMES
    additive_measures = frozenset({"total_time"})

    def fixations(self, participant=None, stimulus_id=None):
        """Read physical fixations for trials represented by interest-area summaries.

        Author-documented unread trials without interest areas are excluded.
        Off-text events within represented trials remain in the stream.

        Args:
            participant: Keep one EDF participant; None keeps all readers.
            stimulus_id: Keep one stimulus; None keeps all sentences.

        Raises:
            NotImplementedError: The separate fixation report is absent.
            ValueError: Events disagree with the summary identities or areas.
        """
        return [
            event
            for _, events in self._fixation_trials(participant, stimulus_id)
            for event in events
        ]

    def fixation_trials(self):
        """Stream complete participant/sentence groups without retaining the corpus."""
        yield from self._fixation_trials()

    def load(self, path: str) -> None:
        """Load CELER from ``path/sent_ia.tsv``.

        Args:
            path: Directory holding ``sent_ia.tsv`` and optionally ``sent_fix.tsv``.
        """
        used_files = [DATA_FILE]
        if (Path(path) / FIXATION_FILE).is_file():
            used_files.append(FIXATION_FILE)
        self._load_metadata(
            path, files=used_files, metadata_path=metadata_io.find_dataset_yaml(path)
        )
        tsv_path = Path(path) / DATA_FILE
        if not tsv_path.exists():
            raise FileNotFoundError(
                f"CELER: {DATA_FILE} not found at {tsv_path}.\n"
                "CELER requires LDC access to PTB-WSJ (LDC95T7) and BLLIP "
                "(LDC2000T43). After completing those, follow:\n"
                "  https://github.com/berzak/celer\n"
                "  → python obtain_data.py     (downloads data_v2.0.zip)\n"
                "  → unzip data_v2.0.zip       (extracts data_v2.0/sent_ia.tsv)\n"
                "Then symlink or copy data_v2.0/sent_ia.tsv into this directory."
            )
        self._load_tsv(tsv_path)
        self._fixation_path = Path(path) / FIXATION_FILE

    def _load_tsv(self, tsv_path: Path) -> None:
        """Stream the IA report and build Stimulus + Datum objects.

        One streaming pass: accumulate per-sentence (words, source row)
        and per-(sentence, participant, measure) value arrays, then a
        second pass over the in-memory dictionaries materialises Stimulus
        and Datum objects (no second disk read).
        """
        # Long sentences exceed the default csv field-size limit; raise it.
        csv.field_size_limit(sys.maxsize)

        # sentenceid → {"text": str, "shared_text": int, "words": {ia_id: ia_label}}
        sentences: dict[str, dict] = {}
        # (sentenceid, participant, canonical_measure) → {ia_id: float}
        per_obs: dict[tuple[str, str, str], dict[int, float]] = {}
        self._source_trials: dict[tuple[str, str], dict] = {}

        with open(tsv_path, encoding="utf-8") as f:
            # DataViewer's quotes are literal stimulus punctuation: CSV quote
            # interpretation swallows later fields and can merge physical rows
            # from different sentences.
            reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
            for row in reader:
                if None in row or any(value is None for value in row.values()):
                    raise ValueError("CELER row does not match the TSV header")
                sid = row["sentenceid"]
                participant = row["DATA_FILE"]
                try:
                    ia_id = int(row["IA_ID"])
                except (KeyError, ValueError):
                    continue
                ia_label = row["IA_LABEL"]
                self._record_source_area((participant, sid), ia_id, row)
                raw_sentence = row.get("sentence", "")

                if sid not in sentences:
                    sentences[sid] = {
                        "text": "",
                        "shared_text": int(row.get("shared_text") or 0),
                        "words": {},
                    }
                sentence_entry = sentences[sid]
                if not _is_valid_sentence(
                    sentence_entry["text"]
                ) and _is_valid_sentence(raw_sentence):
                    sentence_entry["text"] = raw_sentence

                existing = sentence_entry["words"].get(ia_id)
                if existing is not None and existing != ia_label:
                    raise ValueError(f"CELER word label changes at {sid}/{ia_id}")
                sentence_entry["words"][ia_id] = ia_label

                unfixated = _parse(row.get("IA_DWELL_TIME", "")) == 0.0
                values = eyelink_measures(row.get, unfixated=unfixated)
                for canonical, value in zip(EYELINK_MEASURE_NAMES, values, strict=True):
                    per_obs.setdefault((sid, participant, canonical), {})[ia_id] = value

        # Build Stimulus objects in insertion order so results are
        # reproducible across runs.
        self._stimuli = []
        self._data = []

        sid_to_entry: dict[str, tuple[Stimulus, dict[int, int], int]] = {}
        for sid, info in sentences.items():
            if not info["words"]:
                continue
            # Sort IA_IDs into reading order; collapse to a 0-indexed words layer.
            ordered_ids = sorted(info["words"])
            words = [info["words"][i] for i in ordered_ids]
            ia_to_pos = {ia_id: pos for pos, ia_id in enumerate(ordered_ids)}

            source_text = info["text"]
            if not _is_valid_sentence(source_text) and not any(
                w.strip() for w in words
            ):
                continue

            labels = list(words)
            text, spans, words = _build_text_and_spans(source_text, words)

            stimulus = Stimulus(
                text=text,
                stimulus_id=f"celer_{_safe_id(sid)}",
                metadata={
                    "sentenceid": sid,
                    "shared_text": info["shared_text"],
                    "text_source": "sentence"
                    if text == source_text
                    else "interest_areas",
                    "source_sentence": source_text,
                    "interest_area_labels": labels,
                },
            )
            stimulus.add_layer(AnnotationLayer(name="words", spans=spans, values=words))
            self._stimuli.append(stimulus)
            sid_to_entry[sid] = (stimulus, ia_to_pos, len(words))
        self._fixation_stimuli = sid_to_entry

        for (sid, participant, canonical), pos_values in per_obs.items():
            stimulus_entry = sid_to_entry.get(sid)
            if stimulus_entry is None:
                continue
            stimulus, ia_to_pos, n_words = stimulus_entry
            values = np.full(n_words, np.nan)
            for ia_id, v in pos_values.items():
                pos = ia_to_pos.get(ia_id)
                if pos is not None:
                    values[pos] = v
            if not np.any(~np.isnan(values)):
                continue  # The report supplies no values for this measure.
            self._data.append(
                Datum(
                    stimulus=stimulus,
                    measure_name=canonical,
                    values=values,
                    participant_id=participant,
                    metadata={
                        "measurement_source": "released",
                        "source_algorithm": "eyelink_dataviewer",
                        "source_field": next(
                            (
                                k
                                for k, v in EYELINK_IA_MEASURES.items()
                                if v == canonical
                            ),
                            "IA_SKIP",
                        ),
                        "source_trial_index": self._source_trials[participant, sid][
                            "identity"
                        ]["trial"]
                        or None,
                    },
                )
            )

    def _record_source_area(self, key, area, row):
        """Keep independent identities, counts and geometry for event validation."""
        identity = {name: row.get(name, "") for name in _SUMMARY_FIELDS}
        trial = self._source_trials.setdefault(key, {"identity": identity, "areas": {}})
        if trial["identity"] != identity:
            raise ValueError(f"CELER summary presentation identity changes: {key}")
        if area in trial["areas"]:
            raise ValueError(f"Duplicate CELER summary area: {key}, {area}")
        trial["areas"][area] = (
            parse_float(row.get("IA_LEFT", "")),
            parse_float(row.get("IA_RIGHT", "")),
        )

    def _fixation_trials(self, participant=None, stimulus_id=None):
        """Validate report groups against the selected source summaries."""
        path = getattr(self, "_fixation_path", None)
        if path is None or not path.is_file():
            raise NotImplementedError("CELER fixation access requires sent_fix.tsv")
        pending = {
            key
            for key in self._source_trials
            if (participant is None or key[0] == participant)
            and (
                stimulus_id is None
                or self._fixation_stimuli[key[1]][0].stimulus_id == stimulus_id
            )
        }
        seen = set()
        with path.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream, delimiter="\t", quoting=csv.QUOTE_NONE)
            missing = _FIXATION_FIELDS - set(reader.fieldnames or [])
            if missing:
                raise ValueError(
                    f"CELER fixation report is missing columns: {sorted(missing)}"
                )
            for key, rows in groupby(
                reader, lambda r: (r["DATA_FILE"], r["sentenceid"])
            ):
                if participant is not None and key[0] != participant:
                    continue
                if key not in self._source_trials:
                    self._validate_unread_trial(rows)
                    continue
                stimulus = self._fixation_stimuli[key[1]][0]
                if stimulus_id is not None and stimulus.stimulus_id != stimulus_id:
                    continue
                if key in seen:
                    raise ValueError(f"Repeated CELER fixation group: {key}")
                seen.add(key)
                pending.remove(key)
                events = self._read_fixation_trial(key, rows)
                yield (key[0], stimulus.stimulus_id), events
        if pending:
            raise ValueError("CELER summary trials are absent from the fixation report")

    @staticmethod
    def _validate_unread_trial(rows):
        """Allow only documented unread trials with no word-assigned events."""
        for row in rows:
            identity = (int(row["list"]), int(row["trial"]))
            if (
                identity not in _UNREAD_TRIALS
                or row["CURRENT_FIX_INTEREST_AREA_ID"] != "."
            ):
                raise ValueError("CELER fixation trial has no interest-area summary")

    def _read_fixation_trial(self, key, rows):
        """Resolve source areas and require complete event counts and dwell."""
        trial = self._source_trials[key]
        identity = trial["identity"]
        if any(not identity[name] for name in _SUMMARY_FIELDS):
            raise ValueError("CELER event validation requires summary trial metadata")
        stimulus, positions, _ = self._fixation_stimuli[key[1]]
        timer = SyntheticTimer()
        events = []
        previous = 0
        for row in rows:
            if None in row or any(value is None for value in row.values()):
                raise ValueError("CELER fixation row does not match the TSV header")
            if any(row[name] != identity[name] for name in _TRIAL_FIELDS):
                raise ValueError(
                    f"CELER fixation presentation differs from summary: {key}"
                )
            index = int(row["CURRENT_FIX_INDEX"])
            duration = float(row["CURRENT_FIX_DURATION"])
            if index <= previous or not np.isfinite(duration) or duration < 0:
                raise ValueError("CELER fixation indices or durations are invalid")
            previous = index
            unit = self._fixation_unit(row, trial["areas"], positions, stimulus)
            start, end = timer.next(key, duration)
            run = row["CURRENT_FIX_INTEREST_AREA_RUN_ID"]
            append_fixation(
                events,
                participant_id=key[0],
                stimulus_id=stimulus.stimulus_id,
                start_time=start,
                end_time=end,
                duration=duration,
                x=parse_float(row["CURRENT_FIX_X"]),
                y=parse_float(row["CURRENT_FIX_Y"]),
                unit_index=unit,
                char_offset=None,
                fixation_index=index,
                metadata={
                    "trial_index": identity["trial"],
                    "source_run_id": None if run == "." else int(run),
                    "source_area_id": row["CURRENT_FIX_INTEREST_AREA_ID"],
                    "eye_used": identity["EYE_USED"],
                    SyntheticTimer.SENTINEL_KEY: True,
                },
            )
        if len(events) != int(identity["TRIAL_FIXATION_COUNT"]) or not np.isclose(
            sum(event.duration for event in events),
            float(identity["TRIAL_DWELL_TIME"]),
            rtol=0,
            atol=1e-6,
        ):
            raise ValueError(
                f"CELER fixation count or total dwell differs from summary: {key}"
            )
        return events

    @staticmethod
    def _fixation_unit(row, areas, positions, stimulus):
        """Join exact area IDs and labels; nearest areas do not establish landings."""
        area = row["CURRENT_FIX_INTEREST_AREA_ID"]
        if area == ".":
            return None
        area = int(area)
        if area not in areas or area not in positions:
            raise ValueError(f"CELER fixation references an unknown area: {area}")
        unit = positions[area]
        bounds = [
            parse_float(row[f"CURRENT_FIX_INTEREST_AREA_{edge}"])
            for edge in ("LEFT", "RIGHT")
        ]
        label = stimulus.metadata["interest_area_labels"][unit]
        if row["CURRENT_FIX_INTEREST_AREA_LABEL"] != label or not np.allclose(
            bounds, areas[area], rtol=0, atol=1e-6, equal_nan=True
        ):
            raise ValueError(
                f"CELER fixation area label or bounds differ from summary: {area}"
            )
        return unit

    def _reaggregation_provenance(self, datum, layer):
        """Record the standard windows and the absence of observed event onsets."""
        return {
            "source_algorithm": "standard@1",
            "synthetic_start_time": True,
            "source_event_filter": "trials_with_released_interest_areas",
        }


def _build_text_and_spans(
    raw_text: str, words: list[str]
) -> tuple[str, list[Span], list[str]]:
    """Resolve word spans against the raw sentence text.

    Align interest-area labels in order. When the sentence field omits
    punctuation present in the labels, use their joined text and retain the
    original sentence in stimulus metadata. A label that covers only part of
    its whitespace-delimited display token, as the release does for a few
    wholly quoted words whose label drops the quotes, is widened to the whole
    token so the units tile the text.

    Returns:
        The text, one span per unit, and the unit strings after widening.
    """
    if raw_text and ("\t" in raw_text or len(raw_text.split()) > len(words) + 4):
        raw_text = ""
    if raw_text:
        spans: list[Span] = []
        offset = 0
        ok = True
        for w in words:
            pos = raw_text.find(w, offset)
            if pos < 0:
                ok = False
                break
            spans.append(Span(pos, pos + len(w)))
            offset = pos + len(w)
        if ok:
            spans = _widen_to_display_tokens(raw_text, spans)
            return raw_text, spans, [raw_text[sp.start : sp.end] for sp in spans]

    # Retain punctuation that the sentence field omits.
    return " ".join(words), build_word_spans(words), list(words)


def _widen_to_display_tokens(text: str, spans: list[Span]) -> list[Span]:
    """Extend each span over the non-whitespace characters around it.

    A span grows left and right until whitespace, the text edge or the
    neighbouring span, so every character of a whitespace-delimited display
    token belongs to exactly one unit.
    """
    widened: list[Span] = []
    for index, span in enumerate(spans):
        start, end = span.start, span.end
        floor = widened[-1].end if widened else 0
        while start > floor and not text[start - 1].isspace():
            start -= 1
        ceiling = spans[index + 1].start if index + 1 < len(spans) else len(text)
        while end < ceiling and not text[end].isspace():
            end += 1
        widened.append(Span(start, end))
    return widened


def _safe_id(sentenceid: str) -> str:
    """Slug a CELER sentenceid (e.g. ``"en-universal-train.conllu 225"``)
    into a filesystem-safe stimulus_id: spaces/dots/slashes → underscores
    (stimulus_ids key caches and pickle filenames downstream).
    """
    return sentenceid.replace(" ", "_").replace(".", "_").replace("/", "_")
