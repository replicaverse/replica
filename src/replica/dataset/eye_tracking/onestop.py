"""OneStop corpus (Berzak et al., 2025).

Eye-tracking data from 360 participants reading English texts under ordinary,
information-seeking, and repeated reading regimes.

Reference: https://doi.org/10.1038/s41597-025-06272-2
Data: https://osf.io/azj2g/
"""

import contextlib
import csv
import os

from replica.dataset.base import EyeTrackingDataset
from replica.dataset.contracts import DatasetInfo, Fixation
from replica.dataset.measures import (
    EYELINK_IA_MEASURES,
    EYELINK_MEASURE_NAMES,
    EYELINK_SKIP_COLUMN,
    eyelink_measures,
)
from replica.dataset.metadata import io as metadata_io

from ...core import Stimulus
from ..loading.fixation_utils import append_fixation
from ..loading.utils import (
    append_measure_datums,
    parse_reading_measure,
    scatter_measure_tuples,
    set_optional_meta,
)

# OneStop's IA report exposes all four canonical EyeLink IA measures.
_NEEDED_COLS: set[str] = {
    "participant_id",
    "TRIAL_INDEX",
    "IA_ID",
    "IA_LABEL",
    "article_id",
    "article_batch",
    "paragraph_id",
    "difficulty_level",
    "question_preview",
    "repeated_reading_trial",
    "practice_trial",
    EYELINK_SKIP_COLUMN,
    *EYELINK_IA_MEASURES,
}

_ParagraphKey = tuple[int, int, str, int]
_StimLookupValue = tuple[Stimulus, dict[int, int], list[int]]


class OneStop(EyeTrackingDataset):
    """OneStop eye-tracking corpus (Berzak et al., 2025).

    Loads the per-word IA report (``ia_Paragraph.csv``), filtered to the
    ordinary-reading regime. Each stimulus corresponds to one paragraph, keyed
    by (article_batch, article_id, difficulty_level, paragraph_id). Words are
    ordered by IA_ID.

    Measures (canonical names):
        - ``first_fixation``: first fixation duration (IA_FIRST_FIXATION_DURATION)
        - ``first_run``: first-run dwell time (IA_FIRST_RUN_DWELL_TIME)
        - ``first_pass``: whether the word was reached in first pass (IA_SKIP is 0)
        - ``total_time``: total dwell time (IA_DWELL_TIME)
        - ``go_past``: go-past / regression-path duration (IA_REGRESSION_PATH_DURATION)

    Example::

        ds = OneStop()
        ds.load("datasets/eye-tracking/original/onestop")
        data = ds.data(measure="first_run_in_first_pass")
    """

    info = DatasetInfo(name="onestop", language="en")
    _declared_measures = EYELINK_MEASURE_NAMES

    def load(self, path: str) -> None:
        """Load OneStop ordinary-reading data.

        Reads ``{path}/ia_Paragraph.csv`` (per word, all three reading regimes)
        and, when present, ``{path}/fixations_Paragraph.csv``. Rows are kept
        where ``question_preview``, ``repeated_reading_trial`` and
        ``practice_trial`` are all ``"False"``.

        Args:
            path: Root directory of the downloaded OneStop data.
        """

        data_file = os.path.join(path, "ia_Paragraph.csv")
        fix_file = os.path.join(path, "fixations_Paragraph.csv")
        used_files = ["ia_Paragraph.csv"]
        if os.path.exists(fix_file):
            used_files.append("fixations_Paragraph.csv")
        self._load_metadata(
            path,
            files=used_files,
            metadata_path=metadata_io.find_dataset_yaml(
                "eye-tracking/original/onestop", required=True
            ),
        )

        if not os.path.exists(data_file):
            raise FileNotFoundError(
                f"OneStop data file not found at {data_file!r}.\n"
                "Run: replica fetch onestop"
            )
        self._load_ia_report(data_file)

        self._fixations: list[Fixation] = []
        if os.path.exists(fix_file):
            self._load_fixations(fix_file)

    def _load_ia_report(self, data_file: str) -> None:
        """Parse EyeLink interest-area report CSV in a single pass.

        Builds stimuli and data simultaneously. For each unique paragraph
        (``article_batch``, ``article_id``, ``difficulty_level``,
        ``paragraph_id``), collects words on first encounter from IA_ID
        ordering, then creates Datum objects for every participant x
        measure combination.
        """

        # Determine column indices from header so unneeded columns can be skipped.
        with open(data_file, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)

        col_idx: dict[str, int] = {}
        for i, col_name in enumerate(header):
            if col_name in _NEEDED_COLS:
                col_idx[col_name] = i

        missing = _NEEDED_COLS - set(col_idx)
        if missing:
            raise ValueError(
                f"Missing required columns in {data_file}: {missing}. "
                f"Available columns (first 20): {header[:20]}"
            )

        # paragraph_key = (article_batch, article_id, difficulty_level, paragraph_id)
        # For stimulus construction: paragraph_key → sorted list of (ia_id, word_str)
        trial_words: dict[_ParagraphKey, dict[str, dict[int, str]]] = {}

        # For data: paragraph_key → participant → ia_id → tuple of measure values.
        # Measures are a fixed-length tuple rather than dict[str, float] to
        # avoid per-row inner-dict overhead at this row count.
        trial_part_data: dict[
            tuple[int, int, str, int],
            dict[str, dict[int, tuple]],
        ] = {}

        # Index lookups
        idx_part = col_idx["participant_id"]
        idx_ia = col_idx["IA_ID"]
        idx_word = col_idx["IA_LABEL"]
        idx_article = col_idx["article_id"]
        idx_batch = col_idx["article_batch"]
        idx_paragraph = col_idx["paragraph_id"]
        idx_difficulty = col_idx["difficulty_level"]
        idx_qpreview = col_idx["question_preview"]
        idx_repeated = col_idx["repeated_reading_trial"]
        idx_practice = col_idx["practice_trial"]
        # The per-row tuple follows ``EYELINK_MEASURE_NAMES`` for array
        # reconstruction.
        _measure_names = EYELINK_MEASURE_NAMES

        with open(data_file, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)  # skip header

            for row in reader:
                # Ordinary-reading filter. The published
                # ``ia_Paragraph_ordinary.csv`` is the subset where both
                # flags are "False".
                if any(
                    row[i] != "False"
                    for i in (idx_qpreview, idx_repeated, idx_practice)
                ):
                    continue

                article_id = int(row[idx_article])
                article_batch = int(row[idx_batch])
                paragraph_id = int(row[idx_paragraph])
                difficulty_level = row[idx_difficulty]
                ia_id = int(row[idx_ia])
                word_str = row[idx_word]
                participant = row[idx_part]

                paragraph_key = (
                    article_batch,
                    article_id,
                    difficulty_level,
                    paragraph_id,
                )

                # Line-wrapped compounds have different IA sequences across
                # readers. Keep each reader's sequence until variant grouping.
                words_dict = trial_words.setdefault(paragraph_key, {}).setdefault(
                    participant, {}
                )
                if ia_id in words_dict and words_dict[ia_id] != word_str:
                    raise ValueError(
                        f"Conflicting IA label for {paragraph_key}, {participant}, {ia_id}"
                    )
                words_dict[ia_id] = word_str

                # Record measures for this participant at this word position
                part_dict = trial_part_data.setdefault(paragraph_key, {})
                ia_dict = part_dict.setdefault(participant, {})
                cells = {
                    column: row[col_idx[column]]
                    for column in (*EYELINK_IA_MEASURES, EYELINK_SKIP_COLUMN)
                }
                ia_dict[ia_id] = eyelink_measures(
                    cells.get,
                    unfixated=parse_reading_measure(cells["IA_DWELL_TIME"]) == 0.0,
                )

        self._build_paragraph_variants(trial_words, trial_part_data, _measure_names)

    def _build_paragraph_variants(self, trial_words, trial_part_data, measure_names):
        """Keep participant measures on the exact IA sequence they saw."""
        for paragraph_key in sorted(trial_words):
            variants = {}
            for participant, words_dict in trial_words[paragraph_key].items():
                sequence = tuple(sorted(words_dict.items()))
                variants.setdefault(sequence, []).append(participant)
            for variant, sequence in enumerate(sorted(variants), start=1):
                self._append_variant(
                    paragraph_key,
                    sequence,
                    variants[sequence],
                    trial_part_data[paragraph_key],
                    measure_names,
                    variant if len(variants) > 1 else None,
                )

    def _append_variant(
        self, paragraph_key, sequence, participants, part_data, measure_names, variant
    ):
        article_batch, article_id, difficulty_level, paragraph_id = paragraph_key
        words = [word for _, word in sequence]
        suffix = f"_v{variant}" if variant is not None else ""
        stimulus = self._make_stimulus(
            words,
            (
                f"onestop_b{article_batch}_a{article_id}"
                f"_{difficulty_level}_p{paragraph_id}{suffix}"
            ),
            {
                "article_batch": article_batch,
                "article_id": article_id,
                "difficulty_level": difficulty_level,
                "paragraph_id": paragraph_id,
                "language": "en",
                "tokenization_variant": variant or 1,
                "participants": sorted(participants),
            },
        )
        ia_to_idx = {ia_id: i for i, (ia_id, _) in enumerate(sequence)}
        stimulus.metadata["ia_to_idx"] = ia_to_idx
        for participant in sorted(participants):
            measure_vals = scatter_measure_tuples(
                part_data[participant].items(),
                ia_to_idx,
                len(words),
                len(measure_names),
            )
            append_measure_datums(
                self._data,
                stimulus=stimulus,
                participant_id=participant,
                measure_names=measure_names,
                measure_values=measure_vals,
            )

    # Fixation-level loading. ``fixations_Paragraph.csv`` shares
    # ``CURRENT_FIX_INTEREST_AREA_INDEX`` numbering with the IA file's
    # ``IA_ID``, so the per-stim ``ia_to_idx`` map resolves directly.
    # Rows are kept only when their paragraph_key matches an ordinary-reading
    # stimulus and the question_preview/repeated_reading_trial flags are False;
    # this drops the non-ordinary-reading regimes.
    #
    # Source columns (subset used):
    #   participant_id, TRIAL_INDEX, article_id   identifying triple
    #   CURRENT_FIX_INDEX                          1-based ordinal in trial
    #   CURRENT_FIX_START / _END / _DURATION       trial-relative timing (ms)
    #   CURRENT_FIX_X / _Y                         pixel coords
    #   CURRENT_FIX_INTEREST_AREA_INDEX            IA the fixation landed in
    #                                              ("." if off-text)
    #   CURRENT_FIX_INTEREST_AREA_RUN_ID           1 = first run on the IA
    #   CURRENT_FIX_INTEREST_AREA_DWELL_TIME       EyeLink's IA dwell within
    #                                              this trial
    _FIX_COLS = (
        "participant_id",
        "TRIAL_INDEX",
        "article_id",
        "article_batch",
        "paragraph_id",
        "difficulty_level",
        "CURRENT_FIX_INDEX",
        "CURRENT_FIX_START",
        "CURRENT_FIX_END",
        "CURRENT_FIX_DURATION",
        "CURRENT_FIX_X",
        "CURRENT_FIX_Y",
        "CURRENT_FIX_INTEREST_AREA_INDEX",
        "CURRENT_FIX_INTEREST_AREA_RUN_ID",
        "CURRENT_FIX_INTEREST_AREA_DWELL_TIME",
        "CURRENT_FIX_PUPIL",
        # Ordinary-reading flags, must match the IA-report filter so the
        # fixation set covers the same trials as the stored measures.
        "question_preview",
        "repeated_reading_trial",
        "practice_trial",
    )

    def _load_fixations(self, fix_file: str) -> None:
        stim_lookup = self._fixation_stimulus_lookup()
        col_idx = self._fixation_column_indices(fix_file)

        with open(fix_file, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                self._append_fixation_row(row, col_idx, stim_lookup)

    def _fixation_stimulus_lookup(
        self,
    ) -> dict[tuple[_ParagraphKey, str], _StimLookupValue]:
        stim_lookup: dict[tuple[_ParagraphKey, str], _StimLookupValue] = {}
        for stim in self._stimuli:
            paragraph_key = (
                stim.metadata["article_batch"],
                stim.metadata["article_id"],
                stim.metadata["difficulty_level"],
                stim.metadata["paragraph_id"],
            )
            ia_to_idx = stim.metadata.get("ia_to_idx", {})
            char_offsets = [s.start for s in stim.layers["words"].spans]
            for participant in stim.metadata["participants"]:
                stim_lookup[(paragraph_key, participant)] = (
                    stim,
                    ia_to_idx,
                    char_offsets,
                )
        return stim_lookup

    def _fixation_column_indices(self, fix_file: str) -> dict[str, int]:
        with open(fix_file, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)

        col_idx: dict[str, int] = {}
        for i, name in enumerate(header):
            if name in self._FIX_COLS:
                col_idx[name] = i
        missing = set(self._FIX_COLS) - set(col_idx)
        if missing:
            raise ValueError(
                f"Missing required columns in {fix_file}: {sorted(missing)}"
            )
        return col_idx

    def _append_fixation_row(
        self,
        row: list[str],
        col_idx: dict[str, int],
        stim_lookup: dict[tuple[_ParagraphKey, str], _StimLookupValue],
    ) -> None:
        # Ordinary-reading filter (identical to ``_load_ia_report``): the
        # paragraph_key alone does not separate an ordinary reading from a
        # repeated reading of the same paragraph, so without this both trials'
        # fixations aggregate onto one paragraph and double the dwell.
        if (
            row[col_idx["question_preview"]] != "False"
            or row[col_idx["repeated_reading_trial"]] != "False"
            or row[col_idx["practice_trial"]] != "False"
        ):
            return
        row_context = self._fixation_row_context(row, col_idx)
        if row_context is None:
            return
        participant, paragraph_key, trial_index, article_id = row_context

        lookup = stim_lookup.get((paragraph_key, participant))
        if lookup is None:
            # Not in the ordinary-reading IA file (information-seeking trials).
            return
        stim, ia_to_idx, char_offsets = lookup

        timing = self._fixation_timing(row, col_idx)
        if timing is None:
            return
        start, end, dur, fix_idx = timing

        x, y = self._fixation_coordinates(row, col_idx)
        unit_index, char_offset = self._fixation_location(
            row, col_idx, ia_to_idx, char_offsets
        )
        metadata = self._fixation_metadata(row, col_idx, trial_index, article_id)
        append_fixation(
            self._fixations,
            participant_id=participant,
            stimulus_id=stim.stimulus_id,
            start_time=start,
            end_time=end,
            duration=dur,
            x=x,
            y=y,
            unit_index=unit_index,
            char_offset=char_offset,
            fixation_index=fix_idx,
            metadata=metadata,
        )

    @staticmethod
    def _fixation_row_context(
        row: list[str], col_idx: dict[str, int]
    ) -> tuple[str, _ParagraphKey, int, int] | None:
        idx_p = col_idx["participant_id"]
        idx_t = col_idx["TRIAL_INDEX"]
        idx_a = col_idx["article_id"]
        idx_batch = col_idx["article_batch"]
        idx_paragraph = col_idx["paragraph_id"]
        idx_difficulty = col_idx["difficulty_level"]
        try:
            article_id = int(row[idx_a])
            article_batch = int(row[idx_batch])
            paragraph_id = int(row[idx_paragraph])
            difficulty_level = row[idx_difficulty]
            trial_index = int(row[idx_t])
        except (ValueError, IndexError):
            return None
        paragraph_key = (
            article_batch,
            article_id,
            difficulty_level,
            paragraph_id,
        )
        return row[idx_p], paragraph_key, trial_index, article_id

    @staticmethod
    def _fixation_timing(
        row: list[str], col_idx: dict[str, int]
    ) -> tuple[float, float, float, int] | None:
        idx_fi = col_idx["CURRENT_FIX_INDEX"]
        idx_st = col_idx["CURRENT_FIX_START"]
        idx_en = col_idx["CURRENT_FIX_END"]
        idx_dur = col_idx["CURRENT_FIX_DURATION"]
        try:
            return (
                float(row[idx_st]),
                float(row[idx_en]),
                float(row[idx_dur]),
                int(row[idx_fi]),
            )
        except (ValueError, IndexError):
            return None

    @staticmethod
    def _fixation_coordinates(
        row: list[str], col_idx: dict[str, int]
    ) -> tuple[float, float]:
        idx_x = col_idx["CURRENT_FIX_X"]
        idx_y = col_idx["CURRENT_FIX_Y"]
        return (
            OneStop._optional_float(row, idx_x),
            OneStop._optional_float(row, idx_y),
        )

    @staticmethod
    def _optional_float(row: list[str], index: int) -> float:
        try:
            return float(row[index])
        except (ValueError, IndexError):
            return float("nan")

    @staticmethod
    def _fixation_location(
        row: list[str],
        col_idx: dict[str, int],
        ia_to_idx: dict[int, int],
        char_offsets: list[int],
    ) -> tuple[int | None, int | None]:
        idx_ia = col_idx["CURRENT_FIX_INTEREST_AREA_INDEX"]
        ia_str = row[idx_ia]
        if ia_str in (".", "", "NA"):
            return None, None
        with contextlib.suppress(ValueError):
            unit_index = ia_to_idx.get(int(ia_str))
            if unit_index is not None and 0 <= unit_index < len(char_offsets):
                return unit_index, char_offsets[unit_index]
        return None, None

    @staticmethod
    def _fixation_metadata(
        row: list[str],
        col_idx: dict[str, int],
        trial_index: int,
        article_id: int,
    ) -> dict:
        idx_run = col_idx["CURRENT_FIX_INTEREST_AREA_RUN_ID"]
        idx_dwell = col_idx["CURRENT_FIX_INTEREST_AREA_DWELL_TIME"]
        idx_pupil = col_idx["CURRENT_FIX_PUPIL"]
        meta: dict = {"trial_index": trial_index, "article_id": article_id}
        set_optional_meta(meta, "run_id", row[idx_run], int)
        set_optional_meta(meta, "dwell_time", row[idx_dwell], float)
        set_optional_meta(meta, "pupil", row[idx_pupil], float)
        return meta
