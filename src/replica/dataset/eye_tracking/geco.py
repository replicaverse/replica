"""GECO: Ghent Eye-tracking Corpus (Cop et al., 2017).

Eye-tracking data from 14 English monolinguals and 19 Dutch-English
bilinguals reading an entire novel (*The Mysterious Affair at Styles*
by Agatha Christie), presented in English and its published Dutch
translation.

Three data files:
- ``MonolingualReadingData.xlsx``: 14 monolingual English readers (English text)
- ``L1ReadingData.xlsx``: 18 released participants from the 19 bilinguals reading Dutch (their L1)
- ``L2ReadingData.xlsx``: 19 bilinguals reading English (their L2)

By default, loads monolingual English data. Use ``group=`` to select.

Reference: https://doi.org/10.3758/s13428-016-0734-0
Data: https://expsy.ugent.be/downloads/geco/
"""

import os

import numpy as np

from replica.dataset.base import EyeTrackingDataset
from replica.dataset.contracts import DatasetInfo
from replica.dataset.metadata import io as metadata_io

from ..loading.utils import (
    append_measure_datums,
    first_pass_flag,
    parse_reading_measure,
    scatter_values,
)
from ..loading.word_regions import preserve_interest_areas

_GROUPS = {
    "monolingual": ("MonolingualReadingData.xlsx", "en"),
    "l1": ("L1ReadingData.xlsx", "nl"),
    "l2": ("L2ReadingData.xlsx", "en"),
}

# Column name → measure name
MEASURE_COLUMNS = {
    "WORD_FIRST_FIXATION_DURATION": "first_fixation",
    "WORD_GAZE_DURATION": "first_run",
    "WORD_GO_PAST_TIME": "go_past",
    "WORD_TOTAL_READING_TIME": "total_time",
}
_SKIP_COL = "WORD_SKIP"
MEASURE_NAMES = (*MEASURE_COLUMNS.values(), "first_pass")

_FIRST_FIX_TIME_COL = "WORD_FIRST_FIXATION_TIME"


def _reconcile_dutch_label(left: str, right: str, source_id: str, *, dutch=True) -> str:
    """Resolve only the release's quote deletion and CP850 decoding conflicts.

    Some L1 exports lose ASCII quotes and decode CP1252 bytes as CP850;
    pp16 retains the Unicode labels. Reconciliation requires an exact match
    after reversing those operations. It never changes an area's measurements.
    """
    if not dutch:
        raise ValueError(
            f"Conflicting GECO labels at {source_id}: {left!r} and {right!r}"
        )
    if source_id == "4-65-102" and {left, right} <= {"klap", "#NAME?"}:
        return "klap"  # pp16's Excel error label; the other readers identify this area.
    for a, b in ((left, right), (right, left)):
        if a.replace('"', "") == b.replace('"', ""):
            return max((a, b), key=lambda value: (len(value), value))
        try:
            decoded = a.encode("cp850").decode("cp1252")
        except UnicodeError:
            continue
        if decoded.replace('"', "") == b.replace('"', ""):
            return max((decoded, b), key=lambda value: (len(value), value))
    raise ValueError(f"Conflicting GECO labels at {source_id}: {left!r} and {right!r}")


def _restore_material_words(words_dict, trial_key, *, monolingual):
    """Restore the two words missing from every monolingual timing row."""
    if not monolingual:
        return
    word = {(3, 50): "No,", (3, 135): "These"}.get(trial_key)
    if word is not None:
        if 1 in words_dict:
            raise ValueError(f"GECO material-only area has a reading row: {trial_key}")
        words_dict[1] = word


def _word_regions(words_dict, trial_key, *, english):
    """Repair release labels while retaining only identifiable measurements.

    The English materials' SENTENCE sheet confirms that 3-132-19/36, labelled
    "Unknown 19/36", are empty areas between sentences.
    Two labels cross a display-line boundary and are repeated on both areas.
    Fixation coordinates place the first area at the preceding line's end
    and the second at the next line's start. The materials' next-word geometry
    leaves room for just the second word there.
    Other multiword labels remain single measured regions.
    """
    regions = dict(words_dict)
    if english:
        if trial_key == (3, 132):
            for pos in (19, 36):
                if regions.get(pos) == f"Unknown {pos}":
                    del regions[pos]
        for key, first, expected in (
            ((2, 57), 43, 'freely..." miserable'),
            ((4, 82), 35, 'again..." that'),
        ):
            label = regions.get(first, "")
            if (
                trial_key == key
                and label in {expected, expected.replace('"', "")}
                and regions.get(first + 1) == label
            ):
                regions[first], regions[first + 1] = label.split()
    return preserve_interest_areas(sorted(regions.items()))


def _first_pass_filter_trial(
    sorted_positions: list[int],
    fix_times: dict[int, float | None],
    pos_to_idx: dict[int, int],
    n_words: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select first-pass word observations and restrict forward jumps.

    Mirrors ``filter_regressions`` in
    ``resource-geco/scripts/process_geco.py`` in modelblocks/modelblocks-release:
    walk fixations in chronological order, keeping only those that
    *advance* a left-to-right ``frontier`` of already-fixated word
    positions. Fixations that arrive at a word already passed (i.e. via
    a regression saccade) are dropped.

    Returns three arrays of length ``n_words`` (or ``len(sorted_positions)``)
    aligned with
    the stimulus's words layer:
      * ``keep_mask`` (bool): True for words that survive the filter.
      * ``wdelta`` (float): saccade length in word positions for kept
        fixations (NaN elsewhere); computed against the previous *kept*
        fixation in chronological order.
      * ``prevwasfix`` (float): ``1.0`` if ``wdelta == 1`` (immediately
        preceding word was the source of the saccade), else ``0.0``;
        NaN at the first kept fixation of the trial.
    """
    n = len(sorted_positions) if n_words is None else n_words
    keep_mask = np.zeros(n, dtype=bool)
    wdelta = np.full(n, np.nan)
    prevwasfix = np.full(n, np.nan)

    # Build chronologically-ordered list of (time, word_pos) for words
    # that actually received a first fixation in this trial.
    fixated = [
        (ft, wp) for wp, ft in fix_times.items() if ft is not None and np.isfinite(ft)
    ]
    fixated.sort(key=lambda x: (x[0], x[1]))

    frontier = -1
    prev_pos: int | None = None
    for _, wp in fixated:
        widx = pos_to_idx.get(wp)
        if wp <= frontier:
            # Regression-arrived fixation, drop.
            continue
        wd = np.nan if prev_pos is None else float(wp - prev_pos)
        frontier = wp
        prev_pos = wp
        # Unmapped source areas still advance the source fixation frontier.
        if widx is None:
            continue
        keep_mask[widx] = True
        if np.isnan(wd):
            wdelta[widx] = np.nan
            prevwasfix[widx] = np.nan
        else:
            wdelta[widx] = wd
            prevwasfix[widx] = 1.0 if wd == 1.0 else 0.0

    # Apply modelblocks' default ``wdelta <= 4`` filter (always-on in
    # ``resource-lmefit/scripts/lmetools.r::cleanupData``). Drops
    # forward jumps of more than four source word positions. The first
    # kept fixation has no predecessor, so its controls stay NaN and it
    # is exempt from this limit.
    nan_wdelta = np.isnan(wdelta)
    over_4 = (~nan_wdelta) & (wdelta > 4)
    if np.any(over_4):
        keep_mask[over_4] = False
        wdelta[over_4] = np.nan
        prevwasfix[over_4] = np.nan
    return keep_mask, wdelta, prevwasfix


class GECO(EyeTrackingDataset):
    """GECO eye-tracking corpus (Cop et al., 2017).

    Loads word-level reading measures from the Ghent Eye-tracking Corpus.

    Args:
        group: Which reader group to load. One of:
            - ``"monolingual"`` (default): 14 English monolinguals reading English
            - ``"l1"``: 18 released participants reading Dutch (their L1)
            - ``"l2"``: 19 bilinguals reading English (their L2)

    Measures:
        - ``first_fixation``: first fixation duration
        - ``first_run``: dwell time of the first run on the word
        - ``first_pass``: whether the word was reached in first pass (WORD_SKIP is 0)
        - ``go_past``: go-past time
        - ``total_time``: total reading time

    Example::

        ds = GECO(group="monolingual")
        ds.load("datasets/eye-tracking/original/geco")
    """

    info = DatasetInfo(name="geco")
    _declared_measures = MEASURE_NAMES
    GROUPS = tuple(_GROUPS)

    def __init__(self, group: str = "monolingual", first_pass_filter: bool = False):
        """Pick the reader group and optional first-pass observation filter.

        Args:
            group: Which reader group of ``GROUPS`` to read, fixing both the
                workbook file and the corpus language.
            first_pass_filter: Order each trial by ``WORD_FIRST_FIXATION_TIME``
                and retain observations only where the first fixation advances
                beyond every word position fixated before it. Also exclude
                forward jumps exceeding four source word positions, and add
                ``wdelta`` (position difference from the previous retained
                first-pass fixation, before applying the jump limit) and
                ``prevwasfix = (wdelta == 1)``. The first retained fixation has
                no predecessor: both controls are NaN and the jump limit does
                not apply. Excluded duration observations become NaN; retained
                duration values and stimulus text are unchanged. Defaults to
                ``False``. This follows the ModelBlocks preprocessing recipe.
        """
        super().__init__()
        if group not in self.GROUPS:
            raise ValueError(
                f"Unknown GECO group {group!r}. Valid: {list(self.GROUPS)}"
            )
        self.group = group
        self.first_pass_filter = bool(first_pass_filter)
        filename, language = _GROUPS[group]
        self._filename = filename
        self.info = DatasetInfo(name=f"geco-{group}", language=language)

    def load(self, path: str) -> None:
        """Load GECO data from XLSX file.

        Args:
            path: Root directory of the downloaded GECO data.
        """

        import openpyxl

        self._load_metadata(
            path,
            files=[self._filename],
            metadata_path=metadata_io.find_dataset_yaml(path),
        )
        self.info.name = f"geco-{self.group}"
        # _load_metadata replaces info with yaml contents. The yaml is
        # corpus-level; re-pin to the per-instance language for the group.
        _, language = _GROUPS[self.group]
        self.info.language = language

        data_file = os.path.join(path, self._filename)
        if not os.path.exists(data_file):
            raise FileNotFoundError(
                f"GECO data file not found: {data_file}\n"
                "Download from https://expsy.ugent.be/downloads/geco/"
            )

        self._load_xlsx(data_file, openpyxl)

    @staticmethod
    def _check_columns(columns: dict, workbook) -> None:
        """Require identity, duration and skip fields before reading observations."""
        required = {
            "PP_NR",
            "PART",
            "TRIAL",
            "WORD_ID_WITHIN_TRIAL",
            "WORD",
            *MEASURE_COLUMNS,
            _SKIP_COL,
        }
        missing = required - set(columns)
        if missing:
            workbook.close()
            raise ValueError(f"GECO: missing columns {sorted(missing)}")

    def _load_xlsx(self, data_file: str, openpyxl) -> None:
        """Parse GECO XLSX and build stimuli + data."""
        wb = openpyxl.load_workbook(data_file, read_only=True)
        ws = wb["DATA"]

        # Build header index from first row
        header = [cell.value for cell in next(ws.iter_rows(max_row=1))]
        col_idx = {name: i for i, name in enumerate(header)}
        self._check_columns(col_idx, wb)

        idx_pp = col_idx["PP_NR"]
        idx_part = col_idx["PART"]
        idx_trial = col_idx["TRIAL"]
        idx_word_pos = col_idx["WORD_ID_WITHIN_TRIAL"]
        idx_word = col_idx["WORD"]
        idx_measures = {col: col_idx[col] for col in MEASURE_COLUMNS if col in col_idx}
        idx_skip = col_idx.get(_SKIP_COL)
        idx_fix_time = (
            col_idx.get(_FIRST_FIX_TIME_COL) if self.first_pass_filter else None
        )
        if self.first_pass_filter and idx_fix_time is None:
            raise ValueError(
                f"GECO first-pass filtering requires {_FIRST_FIX_TIME_COL!r}"
            )

        # trial_key = (part, trial)
        # trial_key → {word_pos: word_str}
        trial_words: dict[tuple[int, int], dict[int, str]] = {}
        # trial_key → participant → {word_pos: {measure: float}}
        trial_part_data: dict[
            tuple[int, int],
            dict[str, dict[int, dict[str, float]]],
        ] = {}
        # Only used when first_pass_filter: trial_key → participant →
        # {word_pos: first_fix_time_ms_or_None}
        trial_part_fix_time: dict[
            tuple[int, int],
            dict[str, dict[int, float | None]],
        ] = {}

        for row in ws.iter_rows(min_row=2, values_only=True):
            participant = str(row[idx_pp])
            part = int(row[idx_part])
            trial = int(row[idx_trial])
            word_pos = int(row[idx_word_pos])
            raw_word = str(row[idx_word]).strip()
            word_str = raw_word
            trial_key = (part, trial)

            # Collect word for this trial position
            words_dict = trial_words.setdefault(trial_key, {})
            if word_pos not in words_dict:
                words_dict[word_pos] = word_str
            elif words_dict[word_pos] != word_str:
                words_dict[word_pos] = _reconcile_dutch_label(
                    words_dict[word_pos],
                    word_str,
                    f"{part}-{trial}-{word_pos}",
                    dutch=self.group == "l1",
                )

            # Collect measures
            measures: dict[str, float] = {}
            for col_name, idx in idx_measures.items():
                measures[MEASURE_COLUMNS[col_name]] = parse_reading_measure(row[idx])
            if measures.get("total_time") == 0.0:
                measures = {
                    name: 0.0 if np.isnan(value) else value
                    for name, value in measures.items()
                }
            if idx_skip is not None:
                measures["first_pass"] = first_pass_flag(
                    row[idx_skip], unfixated=measures.get("total_time") == 0.0
                )

            part_dict = trial_part_data.setdefault(trial_key, {})
            part_dict.setdefault(participant, {})[word_pos] = measures

            if self.first_pass_filter and idx_fix_time is not None:
                fix_time_raw = row[idx_fix_time]
                if fix_time_raw == "." or fix_time_raw is None:
                    fix_time: float | None = None
                else:
                    try:
                        fix_time = float(fix_time_raw)
                    except (TypeError, ValueError):
                        fix_time = None
                ft_dict = trial_part_fix_time.setdefault(trial_key, {})
                ft_dict.setdefault(participant, {})[word_pos] = fix_time

        wb.close()

        # Build Stimulus and Datum objects
        for trial_key in sorted(trial_words):
            part, trial = trial_key
            words_dict = trial_words[trial_key]
            # Two displayed words are absent from every monolingual timing
            # row. The materials' all-areas sheet identifies their areas and
            # the sentence sheet supplies the punctuated spelling. Keep them
            # as context with unavailable responses, including when skips
            # are requested.
            _restore_material_words(
                words_dict, trial_key, monolingual=self.group == "monolingual"
            )
            sorted_positions = sorted(words_dict)
            words, source_positions, pos_to_idx = _word_regions(
                words_dict, trial_key, english=self.group != "l1"
            )
            n_words = len(words)
            stimulus = self._make_stimulus(
                words,
                f"geco_p{part}_t{trial}",
                {
                    "part": part,
                    "trial": trial,
                    # The release's own per-word key, ``WORD_ID`` in the
                    # reading data and ``IA_ID`` in the materials workbooks.
                    # Word positions skip values in some trials, so it cannot
                    # be rebuilt from the word index; carry it to let callers
                    # join a stimulus back to the materials annotation.
                    "word_ids": [f"{part}-{trial}-{pos}" for pos in source_positions],
                },
            )

            part_dict = trial_part_data.get(trial_key, {})
            for participant in sorted(part_dict):
                word_measures = part_dict[participant]

                measure_vals: dict[str, np.ndarray] = {
                    m: scatter_values(
                        {
                            pos_to_idx[word_pos]: measures.get(m, np.nan)
                            for word_pos, measures in word_measures.items()
                            if word_pos in pos_to_idx
                        },
                        n_words,
                    )
                    for m in MEASURE_NAMES
                }

                if self.first_pass_filter:
                    fix_times = trial_part_fix_time.get(trial_key, {}).get(
                        participant, {}
                    )
                    keep_mask, wdelta, prevwasfix = _first_pass_filter_trial(
                        sorted_positions, fix_times, pos_to_idx, n_words
                    )
                    for m_name, values in measure_vals.items():
                        measure_vals[m_name] = np.where(keep_mask, values, np.nan)
                    measure_vals["wdelta"] = wdelta
                    measure_vals["prevwasfix"] = prevwasfix

                append_measure_datums(
                    self._data,
                    stimulus=stimulus,
                    participant_id=participant,
                    measure_names=list(measure_vals),
                    measure_values=list(measure_vals.values()),
                )

    def fixations(
        self,
        participant=None,
        stimulus_id=None,
    ):
        """Reject event-level access: only the interest-area summary is shipped.

        Args:
            participant: Accepted for signature compatibility, never used.
            stimulus_id: Accepted for signature compatibility, never used.

        Raises:
            NotImplementedError: Always; use :meth:`data` for per-word measures.
        """
        raise NotImplementedError(
            "GECO ships only word-level (interest-area summary) reading "
            "measures; a per-fixation report is not available."
        )
