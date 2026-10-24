"""SB-SAT: Stony Brook SAT reading fixation dataset (Ahn et al., 2020).

Eye-tracking data from participants reading SAT practice passages for
comprehension. 1000 Hz EyeLink tracker. Four passages (dickens, flytrap,
genome, northpole) across multiple pages.

Reference: https://doi.org/10.1145/3379156.3391335
Data: https://github.com/ahnchive/SB-SAT (the OSF entry osf.io/jgae7
    ships only raw 1000 Hz samples)
"""

import csv
import os
from itertools import groupby
from typing import Literal

import numpy as np

from replica.dataset.base import EyeTrackingDataset
from replica.dataset.contracts import DatasetInfo, Fixation
from replica.dataset.metadata import io as metadata_io

from ...context import CONTEXT_SEQUENCE
from ...core import Stimulus
from ..loading.fixation_utils import (
    RunIdInferrer,
    SyntheticTimer,
    append_fixation,
    first_pass_flags,
    first_run_durations,
    mark_unfixated,
)
from ..loading.utils import (
    append_measure_datums,
    parse_float,
    scatter_first_occurrence,
    set_optional_meta,
)
from ..loading.word_regions import normalize_c1_controls, preserve_interest_areas

# Navigation interest areas (buttons) present on every page.
_NAVIGATION_IAS = {"previous_PAGE", "next_PAGE", "GO_TO_QUESTION"}

# Verified against the release's stimuli/reading screenshot/reading-*.png.
# Two hyphenated words wrap across lines: both IAs belong to one word.
_LABEL_CORRECTIONS = {
    ("dickens", "3", 45): "Sempere &",
    ("dickens", "5", 112): "Mr.",
    ("dickens", "5", 113): "Dickens",
    ("genome", "2", 70): "species—",
    ("genome", "2", 71): "in",
}
_MERGED_AREAS = {("flytrap", "3", 31): 30, ("genome", "3", 46): 45}
_REPAIR_LABELS = {
    ("dickens", "3", 45): {"Sempere", "Sempere &"},
    ("dickens", "5", 112): {"Mr.Dickens", "Mr."},
    ("dickens", "5", 113): {"Mr.Dickens", "Dickens"},
    ("genome", "2", 70): {"species—in", "species—"},
    ("genome", "2", 71): {"species—in", "in"},
    ("flytrap", "3", 30): {"Burdon-Sanderson’s"},
    ("flytrap", "3", 31): {"Burdon-Sanderson’s"},
    ("genome", "3", 45): {"gee-whiz,"},
    ("genome", "3", 46): {"gee-whiz,"},
}

# Raw fixation tuple layout for SBSAT's large CSV.
_R_IA: Literal[0] = 0
_R_DUR: Literal[1] = 1
_R_DWELL: Literal[2] = 2
_R_X: Literal[3] = 3
_R_Y: Literal[4] = 4
_R_PUPIL: Literal[5] = 5
_R_RUN: Literal[6] = 6
_R_TRIAL: Literal[7] = 7
_R_SOURCE_IA: Literal[8] = 8

_StimKey = tuple[str, str]
_RawFixRow = tuple[int, float, float, float, float, float, str, str, int]
_StimWords = dict[_StimKey, dict[int, str]]
_StimParticipantRows = dict[_StimKey, dict[str, list[_RawFixRow]]]


class SBSAT(EyeTrackingDataset):
    """SB-SAT eye-tracking corpus (Ahn et al., 2020).

    Loads fixation-level data from ``18sat_fixfinal.csv`` and aggregates to
    word-level reading measures:

    - ``first_fixation``: duration of the first fixation landing on a word.
    - ``total_time``: sum of fixation durations on the word across all page
      visits. The release's dwell column covers only one visit.
    - ``first_run``: sum of fixations on the word from first arrival until
      the eye departs to any other word, whenever that arrival occurs.
    - ``first_pass``: whether that arrival fell in first pass; a word first
      entered by a regression is not in first pass, so
      ``first_run_in_first_pass`` treats it as a skip. All are computed from
      the per-fixation stream since SB-SAT's CSV ships no aggregated field.

    Words that were never fixated by a participant are skips, stored as zero.

    Stimuli are identified by (book_name, page). Only rows with
    ``type == "reading"`` are included; navigation interest areas
    (``previous_PAGE``, ``next_PAGE``, ``GO_TO_QUESTION``) are excluded.
    """

    info = DatasetInfo(name="sbsat", language="en")

    def load(self, path: str) -> None:
        """Load SB-SAT fixation data.

        Expected file::

            {path}/18sat_fixfinal.csv

        Args:
            path: Root directory of the downloaded SB-SAT data.
        """

        self._load_metadata(
            path,
            files=["18sat_fixfinal.csv"],
            metadata_path=metadata_io.find_dataset_yaml(
                "eye-tracking/original/sbsat", required=True
            ),
        )

        data_file = os.path.join(path, "18sat_fixfinal.csv")
        if not os.path.exists(data_file):
            raise FileNotFoundError(
                f"SB-SAT fixation file not found at {data_file}\n"
                "Download from https://github.com/ahnchive/SB-SAT "
                "(osf.io/jgae7 ships only raw 1000 Hz samples)"
            )

        self._load_fixation_data(data_file)

    def _load_fixation_data(self, data_file: str) -> None:
        """Parse 18sat_fixfinal.csv and build stimuli + data + raw fixations."""
        stim_words, stim_part_rows = self._read_fixation_rows(data_file)
        stim_keys, stimulus_map = self._build_stimuli(stim_words)
        self._build_data_and_fixations(stim_keys, stimulus_map, stim_part_rows)

    def _read_fixation_rows(
        self, data_file: str
    ) -> tuple[_StimWords, _StimParticipantRows]:
        stim_words: _StimWords = {}
        stim_part_rows: _StimParticipantRows = {}
        with open(data_file, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["type"] != "reading":
                    continue
                on_word = self._is_word_fixation(row)

                ia_id_str = row["CURRENT_FIX_INTEREST_AREA_ID"]
                ia_label = row["CURRENT_FIX_INTEREST_AREA_LABEL"]
                ia_id = int(ia_id_str) if on_word else -1
                book = row["book_name"]
                page = row["page"]
                ia_label = normalize_c1_controls(ia_label)
                source_key = (book, page, ia_id)
                if (
                    source_key in _REPAIR_LABELS
                    and ia_label not in _REPAIR_LABELS[source_key]
                ):
                    raise ValueError(
                        f"SB-SAT repair no longer matches {source_key}: {ia_label!r}"
                    )
                ia_id = _MERGED_AREAS.get(source_key, ia_id)
                ia_label = _LABEL_CORRECTIONS.get(source_key, ia_label)
                participant = row["RECORDING_SESSION_LABEL"]
                stim_key = (book, page)

                word_map = stim_words.setdefault(stim_key, {})
                if on_word and ia_id not in word_map:
                    word_map[ia_id] = ia_label

                part_dict = stim_part_rows.setdefault(stim_key, {})
                rows_list = part_dict.setdefault(participant, [])
                rows_list.append(self._raw_fixation_tuple(row, ia_id))
        return stim_words, stim_part_rows

    @staticmethod
    def _is_word_fixation(row: dict[str, str]) -> bool:
        if row["type"] != "reading":
            return False
        if row["CURRENT_FIX_INTEREST_AREA_ID"] in (".", ""):
            return False
        return row["CURRENT_FIX_INTEREST_AREA_LABEL"] not in _NAVIGATION_IAS

    @staticmethod
    def _raw_fixation_tuple(row: dict[str, str], ia_id: int) -> _RawFixRow:
        return (
            ia_id,
            parse_float(row["CURRENT_FIX_DURATION"]),
            parse_float(row["CURRENT_FIX_INTEREST_AREA_DWELL_TIME"]),
            parse_float(row.get("CURRENT_FIX_X", "")),
            parse_float(row.get("CURRENT_FIX_Y", "")),
            parse_float(row.get("CURRENT_FIX_PUPIL", "")),
            row.get("CURRENT_FIX_INTEREST_AREA_RUN_ID", ""),
            row.get("TRIAL_INDEX", ""),
            int(row["CURRENT_FIX_INTEREST_AREA_ID"])
            if row["CURRENT_FIX_INTEREST_AREA_ID"] not in ("", ".")
            else -1,
        )

    def _build_stimuli(
        self, stim_words: _StimWords
    ) -> tuple[list[_StimKey], dict[_StimKey, Stimulus]]:
        stim_keys = sorted(stim_words.keys(), key=lambda bp: (bp[0], int(bp[1])))
        stimulus_map: dict[_StimKey, Stimulus] = {}
        for stim_key in stim_keys:
            word_map = stim_words[stim_key]
            sorted_pairs = sorted(word_map.items())
            words, ia_ids, ia_to_idx = preserve_interest_areas(sorted_pairs)

            book, page = stim_key
            stimulus = self._make_stimulus(
                words,
                f"sbsat_{book}_{page}",
                {
                    "book_name": book,
                    "page": int(page),
                    CONTEXT_SEQUENCE: {"group": book, "position": int(page)},
                    "ia_ids": ia_ids,
                    "ia_to_idx": ia_to_idx,
                },
            )
            stimulus_map[stim_key] = stimulus
        return stim_keys, stimulus_map

    def _build_data_and_fixations(
        self,
        stim_keys: list[_StimKey],
        stimulus_map: dict[_StimKey, Stimulus],
        stim_part_rows: _StimParticipantRows,
    ) -> None:
        # SBSAT's CSV ships only ``CURRENT_FIX_DURATION`` (no per-fixation
        # start/end timestamps). Trial-relative ``start_time`` is synthesized by
        # accumulating durations within each (participant, trial): the ordering
        # is monotonic but absolute timing is unavailable.
        self._fixations: list[Fixation] = []

        for stim_key in stim_keys:
            stimulus = stimulus_map[stim_key]
            n_words = len(stimulus.layers["words"])
            ia_to_idx = stimulus.metadata["ia_to_idx"]
            char_offsets = [span.start for span in stimulus.layers["words"].spans]

            part_dict = stim_part_rows.get(stim_key, {})
            for participant in sorted(part_dict.keys()):
                self._append_participant_data(
                    stimulus,
                    participant,
                    part_dict[participant],
                    ia_to_idx,
                    n_words,
                    char_offsets,
                )

    def _append_participant_data(
        self,
        stimulus: Stimulus,
        participant: str,
        rows_list: list[_RawFixRow],
        ia_to_idx: dict[int, int],
        n_words: int,
        char_offsets: list[int],
    ) -> None:
        (first_fix,) = scatter_first_occurrence(
            ((r[_R_IA], (r[_R_DUR],)) for r in rows_list),
            ia_to_idx,
            n_words,
            1,
        )
        total_time = np.full(n_words, np.nan)
        for row in rows_list:
            idx = ia_to_idx.get(row[_R_IA])
            if idx is not None and np.isfinite(row[_R_DUR]):
                if np.isnan(total_time[idx]):
                    total_time[idx] = 0.0
                total_time[idx] += row[_R_DUR]
        # Leaving and returning to a page ends the first run even if the
        # last and first fixated word of the two visits happen to coincide.
        first_run = np.full(n_words, np.nan)
        for _, group in groupby(rows_list, key=lambda row: row[_R_TRIAL]):
            trial_rows = list(group)
            trial_run = first_run_durations(
                [r[_R_IA] for r in trial_rows],
                [r[_R_DUR] for r in trial_rows],
                ia_to_idx,
                n_words,
            )
            first_run = np.where(np.isnan(first_run), trial_run, first_run)
        # First pass is judged over the whole story order, visits included.
        first_pass = first_pass_flags(
            [ia_to_idx.get(r[_R_IA]) for r in rows_list], n_words
        )

        self._append_fixations(
            stimulus,
            participant,
            rows_list,
            ia_to_idx,
            char_offsets,
        )
        mark_unfixated(
            [first_fix, total_time, first_run],
            ia_to_idx.values(),
            (ia_to_idx[r[_R_IA]] for r in rows_list if r[_R_IA] in ia_to_idx),
        )
        append_measure_datums(
            self._data,
            stimulus=stimulus,
            participant_id=participant,
            measure_names=("first_fixation", "total_time", "first_run", "first_pass"),
            measure_values=(first_fix, total_time, first_run, first_pass),
        )

    def _append_fixations(
        self,
        stimulus: Stimulus,
        participant: str,
        rows_list: list[_RawFixRow],
        ia_to_idx: dict[int, int],
        char_offsets: list[int],
    ) -> None:
        cur_trial: str | None = None
        fix_within_trial = 0
        timer = SyntheticTimer()
        run_ids = RunIdInferrer()

        for raw in rows_list:
            ia_id = raw[_R_IA]
            fix_dur = raw[_R_DUR]
            dwell = raw[_R_DWELL]
            idx = ia_to_idx.get(ia_id)
            trial_raw = raw[_R_TRIAL]
            if trial_raw != cur_trial:
                cur_trial = trial_raw
                fix_within_trial = 0
            fix_within_trial += 1
            start_time, end_time = timer.next(trial_raw, fix_dur)
            metadata = self._fixation_metadata(raw, dwell, trial_raw)
            run_id = run_ids.next(trial_raw, idx)
            if run_id is not None:
                metadata["run_id"] = run_id

            append_fixation(
                self._fixations,
                participant_id=participant,
                stimulus_id=stimulus.stimulus_id,
                start_time=start_time,
                end_time=end_time,
                duration=fix_dur,
                x=raw[_R_X],
                y=raw[_R_Y],
                unit_index=idx,
                char_offset=char_offsets[idx] if idx is not None else None,
                fixation_index=fix_within_trial,
                metadata=metadata,
            )

    @staticmethod
    def _fixation_metadata(raw: _RawFixRow, dwell: float, trial_raw: str) -> dict:
        meta: dict = {"synthetic_start_time": True, "source_ia_id": raw[_R_SOURCE_IA]}
        set_optional_meta(meta, "trial_index", trial_raw, int)
        set_optional_meta(meta, "source_run_id", raw[_R_RUN], int, skip=(".", "", None))
        pupil = raw[_R_PUPIL]
        if np.isfinite(pupil):
            meta["pupil"] = pupil
        meta["dwell_time"] = dwell
        return meta
