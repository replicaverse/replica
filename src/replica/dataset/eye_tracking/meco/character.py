"""MECO L1 character-level eye-tracking loader.

Reads the per-fixation, character-aligned export shipped at
``datasets/eye-tracking/processed/meco_char/`` (downloadable via
``replica fetch meco_char``). Each language
has two CSVs (named by the on-disk file code, ``ge`` for German):

    {code}.csv           one row per fixation across all readers / trials
    stimulus_{code}.csv  per-character stimulus table (lang, trial_id,
                         letternum, character)

The loader builds, per trial:

* a ``Stimulus`` whose raw text is the concatenation of all characters in
  trial order, with a ``"characters"`` annotation layer (one Span per char);
* one ``Datum`` per (participant, trial) per measure: ``total_time`` sums
  every fixation on a character, so it is that character's dwell time, while
  ``right_bounded`` sums fixations on the character before the first fixation
  on a later character.

Languages (MECO release codes, not all ISO 639-1): ``de, du, ee, en, fi,
gr, he, it, ko, no, ru, sp, tr``, the 13 wave-1 L1 languages.
"""

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from replica.dataset.base import EyeTrackingDataset
from replica.dataset.contracts import DatasetInfo, Fixation
from replica.dataset.metadata import io as metadata_io

from ....core import AnnotationLayer, Datum, Span, Stimulus
from ...loading.fixation_utils import (
    RunIdInferrer,
    append_fixation,
    right_bounded_durations,
)
from ...loading.utils import parse_float
from .releases import (
    FILE_TO_ISO,
)
from .releases import (
    ISO_TO_FILE as _ISO_TO_FILE,
)
from .releases import (
    LANGUAGES as _SHARED_LANGUAGES,
)

# Public language codes (MECO release; not all ISO 639-1)
LANGUAGES = list(_SHARED_LANGUAGES)


class MECO_char(EyeTrackingDataset):
    """MECO L1 character-level eye-tracking corpus for a single language.

    Each instance handles one language. The default measure is
    ``"total_time"``, the dwell time on a character, summed over every
    fixation that lands on it; ``"right_bounded"`` is also exposed.

    Args:
        language: One of ``MECO_char.LANGUAGES``.

    Example::

        ds = MECO_char(language="en")
        ds.load("datasets/eye-tracking/processed/meco_char")
        results = evaluate(ds, evaluation_spec, AnalysisConfig())
    """

    LANGUAGES = LANGUAGES
    _declared_measures = ("total_time", "right_bounded")

    def __init__(self, language: str = "en"):
        """Bind the instance to one MECO language and its character export.

        Args:
            language: MECO release code in :data:`LANGUAGES`, which selects the
                CSV pair to read and, through ``FILE_TO_ISO``, the ISO code
                reported as the dataset language. An unknown code raises.
        """
        super().__init__()
        if language not in LANGUAGES:
            raise ValueError(
                f"Unknown MECO_char language {language!r}. Valid options: {LANGUAGES}"
            )
        self.language = language
        self.info = DatasetInfo(
            name=f"MECO_char-{language}",
            language=FILE_TO_ISO.get(language, language),
        )
        self.default_measure = "total_time"

    def _source_paths(self, path: str) -> tuple[Path, Path]:
        root = Path(path)
        file_code = _ISO_TO_FILE.get(self.language, self.language)
        stim_csv = root / f"stimulus_{file_code}.csv"
        fix_csv = root / f"{file_code}.csv"
        if not stim_csv.exists():
            raise FileNotFoundError(
                f"Stimulus file not found: {stim_csv}. "
                "Run `replica fetch meco_char` to download."
            )
        if not fix_csv.exists():
            raise FileNotFoundError(
                f"Fixation file not found: {fix_csv}. "
                "Run `replica fetch meco_char` to download."
            )
        return stim_csv, fix_csv

    @staticmethod
    def _detect_start_scale(fix_csv: Path) -> float:
        """Factor converting the ``start`` column to milliseconds.

        English exports ``start`` in seconds (``stop`` empty); the other
        languages export it already in milliseconds (``stop == start + dur``).
        Pick the reading of ``start`` that makes ``start + dur`` match ``stop``;
        default to seconds when no usable ``stop`` is present.
        """
        with open(fix_csv, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                start = parse_float(row.get("start", ""))
                stop = parse_float(row.get("stop", ""))
                dur = parse_float(row.get("dur", ""))
                if not (np.isfinite(start) and np.isfinite(stop) and np.isfinite(dur)):
                    continue
                ms_residual = abs((start + dur) - stop)
                s_residual = abs((start * 1000.0 + dur) - stop)
                return 1000.0 if s_residual < ms_residual else 1.0
        return 1000.0

    def _read_chars_by_trial(self, stim_csv: Path) -> dict[int, list[tuple[int, str]]]:
        chars_by_trial: dict[int, list[tuple[int, str]]] = defaultdict(list)
        with open(stim_csv, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    tid = int(row["trial_id"])
                    ln = int(row["letternum"])
                except (KeyError, ValueError):
                    continue
                chars_by_trial[tid].append((ln, row["character"]))
        return chars_by_trial

    def _build_stimuli_from_chars(
        self, chars_by_trial: dict[int, list[tuple[int, str]]]
    ) -> tuple[dict[int, int], dict[int, dict[int, int]], dict[int, list[int]]]:
        self._stimuli = []
        n_chars_by_trial: dict[int, int] = {}
        letternum_to_index: dict[int, dict[int, int]] = {}
        char_offsets_by_trial: dict[int, list[int]] = {}

        for tid in sorted(chars_by_trial):
            chars = sorted(chars_by_trial[tid], key=lambda x: x[0])
            text = "".join(c for _, c in chars)
            spans, idx_map, offset = [], {}, 0
            for i, (ln, ch) in enumerate(chars):
                start = offset
                end = offset + len(ch)
                spans.append(Span(start, end))
                idx_map[ln] = i
                offset = end
            char_layer = AnnotationLayer(
                name="characters", spans=spans, values=[c for _, c in chars]
            )
            stim = Stimulus(
                text=text,
                stimulus_id=f"meco_char_{self.language}_trial{tid}",
                metadata={"trial_id": tid, "language": self.language},
            )
            stim.add_layer(char_layer)
            self._stimuli.append(stim)
            n_chars_by_trial[tid] = len(chars)
            letternum_to_index[tid] = idx_map
            char_offsets_by_trial[tid] = [span.start for span in spans]

        return n_chars_by_trial, letternum_to_index, char_offsets_by_trial

    def _add_measure_sums(
        self,
        *,
        row: dict[str, str],
        idx: int,
        rid: str,
        tid: int,
        n_chars: int,
        sums: dict[tuple[str, int, str], np.ndarray],
    ) -> None:
        value = parse_float(row["dur"])
        key = (rid, tid, "total_time")
        values = sums.setdefault(key, np.zeros(n_chars))
        values[idx] += value

    def _append_fixation(
        self,
        *,
        row: dict[str, str],
        rid: str,
        tid: int,
        ln: int,
        idx: int,
        run_ids: RunIdInferrer,
        char_offsets_by_trial: dict[int, list[int]],
    ) -> None:
        dur = parse_float(row.get("dur", ""))
        start = parse_float(row.get("start", ""))
        if np.isinf(dur) or np.isinf(start) or dur < 0:
            raise ValueError(
                f"Invalid MECO character fixation timing for reader {rid}, trial {tid}: {row!r}"
            )

        fix_idx_raw = row.get("fixid", "")
        try:
            fix_idx = int(fix_idx_raw)
        except ValueError:
            fix_idx = 0

        run_id = run_ids.next((rid, tid), idx)
        start_ms = start * self._start_scale
        stim_id = f"meco_char_{self.language}_trial{tid}"
        append_fixation(
            self._fixations,
            participant_id=str(rid),
            stimulus_id=stim_id,
            start_time=start_ms,
            end_time=start_ms + dur,
            duration=dur,
            x=parse_float(row.get("x")),
            y=parse_float(row.get("y")),
            unit_index=idx,
            char_offset=char_offsets_by_trial[tid][idx],
            fixation_index=fix_idx,
            unit_layer="characters",
            metadata={
                "trial_id": tid,
                "letternum": ln,
                "run_id": run_id,
                "source_right_bounded_time": parse_float(row.get("right_bounded_time")),
            },
        )

    def _read_fixations_and_sums(
        self,
        *,
        fix_csv: Path,
        n_chars_by_trial: dict[int, int],
        letternum_to_index: dict[int, dict[int, int]],
        char_offsets_by_trial: dict[int, list[int]],
    ) -> dict[tuple[str, int, str], np.ndarray]:
        sums: dict[tuple[str, int, str], np.ndarray] = {}
        self._fixations: list[Fixation] = []
        run_ids = RunIdInferrer()
        self._start_scale = self._detect_start_scale(fix_csv)

        with open(fix_csv, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    tid = int(row["trial_id"])
                    ln = int(row["letternum"])
                except (KeyError, ValueError):
                    continue
                idx_map = letternum_to_index.get(tid)
                if idx_map is None or ln not in idx_map:
                    continue
                idx = idx_map[ln]
                rid = row["reader"]
                self._add_measure_sums(
                    row=row,
                    idx=idx,
                    rid=rid,
                    tid=tid,
                    n_chars=n_chars_by_trial[tid],
                    sums=sums,
                )
                self._append_fixation(
                    row=row,
                    rid=rid,
                    tid=tid,
                    ln=ln,
                    idx=idx,
                    run_ids=run_ids,
                    char_offsets_by_trial=char_offsets_by_trial,
                )
        trials: dict[tuple[str, int], list[Fixation]] = defaultdict(list)
        for fixation in self._fixations:
            trials[(fixation.participant_id, fixation.metadata["trial_id"])].append(
                fixation
            )
        for (rid, tid), events in trials.items():
            if all(np.isfinite(event.start_time) for event in events):
                events = sorted(events, key=lambda event: event.start_time)
            sums[(rid, tid, "right_bounded")] = right_bounded_durations(
                [event.unit_index for event in events],
                [event.duration for event in events],
                n_chars_by_trial[tid],
            )
        return sums

    def _build_data_from_sums(
        self, sums: dict[tuple[str, int, str], np.ndarray]
    ) -> None:
        stim_by_trial = {int(stim.metadata["trial_id"]): stim for stim in self._stimuli}
        self._data = []
        for (rid, tid, measure), arr in sums.items():
            stim = stim_by_trial.get(tid)
            if stim is None:
                continue
            self._data.append(
                Datum(
                    stimulus=stim,
                    measure_name=measure,
                    values=arr,
                    participant_id=str(rid),
                    layer="characters",
                )
            )

    def load(self, path: str) -> None:
        """Load the per-fixation and per-character CSVs for this language.

        Args:
            path: Root directory containing ``{lang}.csv`` and
                ``stimulus_{lang}.csv`` for each supported language.
        """
        stim_csv, fix_csv = self._source_paths(path)
        self._load_metadata(
            path,
            files=[stim_csv.name, fix_csv.name],
            metadata_path=metadata_io.find_dataset_yaml(path),
        )
        # _load_metadata replaces info with yaml contents, where language is
        # the corpus-level "multilingual". Re-pin to the per-instance language.
        self.info.language = FILE_TO_ISO.get(self.language, self.language)
        self.info.name = f"MECO_char-{self.language}"

        n_chars_by_trial, letternum_to_index, char_offsets_by_trial = (
            self._build_stimuli_from_chars(self._read_chars_by_trial(stim_csv))
        )
        sums = self._read_fixations_and_sums(
            fix_csv=fix_csv,
            n_chars_by_trial=n_chars_by_trial,
            letternum_to_index=letternum_to_index,
            char_offsets_by_trial=char_offsets_by_trial,
        )
        self._build_data_from_sums(sums)
