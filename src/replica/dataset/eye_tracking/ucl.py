"""UCL Eye-Tracking Corpus (Frank et al., 2013).

Eye-tracking data: 205-sentence subset, 43 participants.
Measures: first fixation, first run with its first-pass flag, go-past,
right-bounded RT, and total time summed from word-aligned raw fixation
durations when available.

Reference: https://doi.org/10.3758/s13428-012-0313-y
"""

import contextlib
import csv
from pathlib import Path

import numpy as np

from replica.core import AnnotationLayer, Stimulus
from replica.dataset.base import EyeTrackingDataset
from replica.dataset.contracts import DatasetInfo, Fixation
from replica.dataset.eye_tracking.devarda2023.covariates import DeVardaCovariates
from replica.dataset.loading.utils import build_word_spans
from replica.dataset.metadata import io as metadata_io

from ...core import Datum
from ..loading.fixation_utils import (
    RunIdInferrer,
    SyntheticTimer,
    append_fixation,
    first_run_durations,
)
from ..loading.utils import (
    conditional_first_run,
    parse_float,
    parse_reading_measure,
    scatter_values,
)


def _parse_ucl_stimuli(path: Path) -> dict[int, Stimulus]:
    """Parse ``stimuli.txt`` into a mapping from sent_nr -> Stimulus.

    Used by UCL_ET. UCL_SPR builds stimuli from the RT file instead.
    """
    stimuli: dict[int, Stimulus] = {}

    with open(path, encoding="cp1252") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            sent_nr = int(row["sent_nr"])
            sentence = row["sentence"].strip()

            words = sentence.split()
            text = " ".join(words)

            spans = build_word_spans(words)
            word_layer = AnnotationLayer(name="words", spans=spans, values=words)

            stimulus = Stimulus(
                text=text,
                stimulus_id=f"ucl_{sent_nr}",
                metadata={
                    "sent_nr": sent_nr,
                    "question": row.get("question", "-"),
                    "answer": row.get("answer", "-"),
                },
            )
            stimulus.add_layer(word_layer)
            stimuli[sent_nr] = stimulus

    return stimuli


# Map from raw column names to canonical measure names.
MEASURE_COLUMNS = {
    "RTfirstfix": "first_fixation_in_first_pass",
    "RTgopast": "go_past_in_first_pass",
    "RTrightbound": "right_bounded",
}
#: The release's first-pass reading time: the first run of a word reached in
#: first pass, zero otherwise. It yields ``first_run`` and ``first_pass``.
FIRST_PASS_COLUMN = "RTfirstpass"
TABLE_MEASURES = (
    *MEASURE_COLUMNS.values(),
    "first_fixation",
    "first_run",
    "first_pass",
)


class UCL_ET(DeVardaCovariates, EyeTrackingDataset):
    """UCL eye-tracking corpus (Frank et al., 2013).

    205-sentence subset read by 43 participants during eye-tracking. The
    stimulus ids (``ucl_<sent_nr>``) are shared with ``UCL_SPR`` and
    ``Frank2015ERP``, and :meth:`attach_predictability_norms` adds de Varda et al.'s
    (2023) predictability norms as covariates.
    The word table supplies first fixation, first-pass reading time, go-past
    and right-bounded RT, all first-pass conditional, so ``first_pass`` is
    whether the reading time is positive and ``first_run`` is known from the
    table on first-pass words only. When raw fixations are available,
    ``total_time`` is derived; ``first_fixation`` and ``first_run`` are completed
    for words first entered by a regression; the derived first run must reproduce the
    released reading time on every first-pass word.

    Raw fixations are exposed via :meth:`fixations` (from
    ``eyetracking.fix.txt`` when present). Beyond the standard
    ``participant`` / ``stimulus_id`` filters, UCL adds:

      * ``exclude_blink=`` (default ``False``), when ``True``, drop
        fixations whose source ``blink`` column is ``before`` /
        ``after`` / ``both`` (i.e. the eye lid was passing nearby).
        The published per-word measures include these fixations, so the
        default keeps them.
    """

    info = DatasetInfo(name="ucl_et", language="en")
    #: The table measures, plus total time from the fixation file when it is
    #: present.
    _declared_measures = (*TABLE_MEASURES, "total_time", "go_past")

    def load(
        self,
        path: str,
        stimuli_file: str = "stimuli.txt",
        et_file: str = "eyetracking.RT.txt",
        fix_file: str = "eyetracking.fix.txt",
    ) -> None:
        """Load UCL eye-tracking data from disk.

        Args:
            path: Directory containing the data files.
            stimuli_file: Sentence stimuli file (default "stimuli.txt").
            et_file: Per-word RT file (required; default "eyetracking.RT.txt").
            fix_file: Optional per-fixation file (default
                "eyetracking.fix.txt"). If present, populates
                :meth:`fixations`.
        """
        base = Path(path)
        files = {"stimuli.txt": stimuli_file, "eyetracking.RT.txt": et_file}
        if (base / fix_file).exists():
            files["eyetracking.fix.txt"] = fix_file
        self._load_metadata(
            path, files=files, metadata_path=metadata_io.find_dataset_yaml(path)
        )
        self._reset_devarda()

        all_stimuli = _parse_ucl_stimuli(base / stimuli_file)
        self._sent_map = all_stimuli

        self._load_et(base / et_file)

        self._fixations: list[Fixation] = []
        fix_path = base / fix_file
        if fix_path.exists():
            self._load_fixations(fix_path)
            self._load_total_time(fix_path.name)
            self._complete_first_run(fix_path.name)
            self._complete_first_fixation(fix_path.name)
        self._add_arrival_go_past()

    def _complete_first_fixation(self, source_file: str) -> None:
        """Complete first-arrival durations from each trial's ordered events.

        Args:
            source_file: Fixation file used for provenance and validation.
        """
        trials = {
            (datum.participant_id, datum.stimulus.stimulus_id): datum
            for datum in self._data
            if datum.measure_name == "first_fixation"
        }
        first: dict[tuple[str, str], np.ndarray] = {}
        visited: dict[tuple[str, str], set[int]] = {}
        for fixation in self._fixations:
            key = (fixation.participant_id, fixation.stimulus_id)
            if key not in trials:
                continue
            values = first.setdefault(key, np.zeros(len(trials[key].values)))
            seen = visited.setdefault(key, set())
            index = fixation.unit_index
            if index is not None and index not in seen:
                seen.add(index)
                values[index] = fixation.duration
        for key, values in first.items():
            datum = trials[key]
            known = np.isfinite(datum.values)
            if not np.array_equal(values[known], datum.values[known]):
                raise ValueError(f"UCL fixations do not reproduce RTfirstfix for {key}")
            datum.values = values
            datum.metadata.update(
                source_file=source_file,
                aggregation="first word-aligned fixation in source order",
            )

    def _complete_first_run(self, source_file: str) -> None:
        """Sum each word's first run of fixations, whenever the eye arrives.

        Off-text fixations end the current run. Validate the derived duration
        against RTfirstpass on every first-pass word, then fill first-run
        durations for words first reached by regression. Never-fixated words
        receive zero.

        Args:
            source_file: Name of the raw fixation file used for provenance.
        """
        trials = {
            (datum.participant_id, datum.stimulus.stimulus_id): datum
            for datum in self._data
            if datum.measure_name == "first_run"
        }
        streams: dict[tuple[str, str], tuple[list[int | None], list[float]]] = {}
        for fixation in self._fixations:
            key = (fixation.participant_id, fixation.stimulus_id)
            if key in trials:
                units, durations = streams.setdefault(key, ([], []))
                units.append(fixation.unit_index)
                durations.append(fixation.duration)
        for key, (units, durations) in streams.items():
            datum = trials[key]
            n_words = len(datum.values)
            first_run = first_run_durations(
                units,
                durations,
                {index: index for index in range(n_words)},
                n_words,
                off_text_ends_run=True,
            )
            first_run[np.isnan(first_run)] = 0.0
            released = datum.values
            in_first_pass = np.isfinite(released) & (released > 0)
            if not np.array_equal(first_run[in_first_pass], released[in_first_pass]):
                raise ValueError(
                    f"UCL fixations do not reproduce {FIRST_PASS_COLUMN} for {key}"
                )
            datum.values = first_run
            datum.metadata.update(
                source_file=source_file,
                aggregation="first run of word-aligned fixations",
            )

    def _load_total_time(self, source_file: str) -> None:
        """Sum all positive word-aligned fixation durations within each trial.

        Regressions and blink-adjacent fixations contribute to total dwell
        time. Off-text events do not. The synthetic event clock is unused.
        Trials absent from the fixation file receive no derived observation.

        Args:
            source_file: Name of the raw fixation file used for provenance.
        """
        trials = {
            (datum.participant_id, datum.stimulus.stimulus_id): datum
            for datum in self._data
            if datum.measure_name == "first_pass"
        }
        totals: dict[tuple[str, str], np.ndarray] = {}
        for fixation in self._fixations:
            key = (fixation.participant_id, fixation.stimulus_id)
            if key not in trials:
                continue
            values = totals.setdefault(key, np.zeros(len(trials[key].values)))
            if fixation.unit_index is not None:
                if not np.isfinite(fixation.duration) or fixation.duration < 0:
                    values[fixation.unit_index] = np.nan
                else:
                    values[fixation.unit_index] += fixation.duration
        for key, values in totals.items():
            original = trials[key]
            self._data.append(
                Datum(
                    stimulus=original.stimulus,
                    measure_name="total_time",
                    values=values,
                    participant_id=original.participant_id,
                    metadata={
                        **original.metadata,
                        "source_file": source_file,
                        "aggregation": "sum of positive word-aligned fixation durations",
                        "includes_blink_adjacent": True,
                    },
                )
            )

    def _load_et(self, et_path: Path) -> None:
        """Parse eye-tracking data into Datum objects."""
        et_data: dict[tuple[str, int], dict[int, dict[str, float]]] = {}
        sent_nrs_seen: set[int] = set()
        # Comprehension-question outcome per trial ("c", "e", or "-"), kept on
        # the datum metadata for studies that apply the accuracy filter.
        trial_correct: dict[tuple[str, int], str] = {}

        with open(et_path, encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                subj = row["subj_nr"].strip()
                sent_nr = int(row["sent_nr"])
                word_pos = int(row["word_pos"]) - 1

                sent_nrs_seen.add(sent_nr)
                key = (subj, sent_nr)
                if key not in et_data:
                    et_data[key] = {}
                trial_correct[key] = (row.get("correct") or "-").strip()

                measures: dict[str, float] = {}
                for raw_col, canonical in MEASURE_COLUMNS.items():
                    measures[canonical] = parse_reading_measure(row[raw_col])
                measures["first_run"], measures["first_pass"] = conditional_first_run(
                    row[FIRST_PASS_COLUMN]
                )

                conditional = measures["first_fixation_in_first_pass"]
                measures["first_fixation"] = conditional if conditional > 0 else np.nan
                et_data[key][word_pos] = measures

        self._stimuli = []
        for sent_nr in sorted(sent_nrs_seen):
            if sent_nr in self._sent_map:
                self._stimuli.append(self._sent_map[sent_nr])

        self._data = []
        for (subj, sent_nr), positions in et_data.items():
            stimulus = self._sent_map.get(sent_nr)
            if stimulus is None:
                continue

            n_words = len(stimulus.layers["words"])

            for canonical in TABLE_MEASURES:
                values = scatter_values(
                    {
                        word_pos: mdict.get(canonical, np.nan)
                        for word_pos, mdict in positions.items()
                    },
                    n_words,
                )

                self._data.append(
                    Datum(
                        stimulus=stimulus,
                        measure_name=canonical,
                        values=values,
                        participant_id=subj,
                        metadata={
                            "correct": trial_correct.get((subj, sent_nr), "-"),
                            # RTgopast counts all fixations between its endpoints,
                            # including those outside the text (release README).
                            "go_past_includes_off_text": True,
                        },
                    )
                )

    # Fixation-level loading from ``eyetracking.fix.txt``.
    # No per-fixation timestamps in the file; trial-relative ``start_time``
    # is synthesized from cumulative durations within each (subj, sent_nr)
    # group (``metadata['synthetic_start_time'] = True``).
    #
    # Columns:
    #   subj_nr       participant ID
    #   sent_nr       1-based sentence number → maps to one stimulus
    #   gaze_x, gaze_y  pixel coords
    #   fix_duration  ms
    #   letter_pos    character offset within the sentence (1-based)
    #   word_pos      1-based word index within the sentence; "NaN" = off-text
    #   word          the word string at word_pos ("-" for off-text)
    #   blink         "before"/"after"/"both" if bordering a blink; "-" else
    def _load_fixations(self, fix_path: Path) -> None:
        # Per-trial helpers for synthetic timing + run_id inference.
        timer = SyntheticTimer()
        run_ids = RunIdInferrer()
        fix_count: dict[tuple[str, int], int] = {}

        with open(fix_path, encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                try:
                    subj = row["subj_nr"].strip()
                    sent_nr = int(row["sent_nr"])
                    dur = float(row["fix_duration"])
                except (KeyError, ValueError):
                    continue
                stim = self._sent_map.get(sent_nr)
                if stim is None:
                    continue

                # word_pos in source: 1-based int when on a word; the
                # literal string "NaN" when the fixation fell off-text.
                # Don't drop off-text rows, surface them with
                # ``unit_index=None`` to preserve off-text fixation
                # counts (e.g., return-sweep landing studies).
                wp_raw = row.get("word_pos", "").strip()
                n_words = len(stim.layers["words"])
                unit_index: int | None = None
                char_offset: int | None = None
                if wp_raw not in ("", "NaN", "nan", "NA", "."):
                    try:
                        word_pos_raw = int(wp_raw)
                    except ValueError:
                        word_pos_raw = -1
                    if 0 < word_pos_raw <= n_words:
                        unit_index = word_pos_raw - 1
                        char_offset = stim.layers["words"].spans[unit_index].start

                x = parse_float(row.get("gaze_x", ""))
                y = parse_float(row.get("gaze_y", ""))

                # Source codes blink-adjacency as "before"/"after"/"both"
                # (or "-" for no blink). Any non-"-" value means the
                # fixation borders a blink.
                blink_raw = row.get("blink", "-").strip()
                blink = blink_raw in ("before", "after", "both")

                trial_key = (subj, sent_nr)
                start, end = timer.next(trial_key, dur)
                fix_idx = fix_count.get(trial_key, 0) + 1
                fix_count[trial_key] = fix_idx
                # Run-id inference is only meaningful for on-text fixations.
                run_id: int | None = run_ids.next(trial_key, unit_index)

                meta: dict = {
                    "sent_nr": sent_nr,
                    "blink": blink,
                    "synthetic_start_time": True,
                }
                if run_id is not None:
                    meta["run_id"] = run_id
                # letter_pos kept for char-level work.
                lp_raw = row.get("letter_pos", "")
                with contextlib.suppress(ValueError, TypeError):
                    meta["letter_pos"] = int(lp_raw)

                append_fixation(
                    self._fixations,
                    participant_id=subj,
                    stimulus_id=stim.stimulus_id,
                    start_time=start,
                    end_time=end,
                    duration=dur,
                    x=x,
                    y=y,
                    unit_index=unit_index,
                    char_offset=char_offset,
                    fixation_index=fix_idx,
                    metadata=meta,
                )

    def fixations(
        self,
        participant: str | None = None,
        stimulus_id: str | None = None,
        exclude_blink: bool = False,
    ) -> list[Fixation]:
        """Return raw fixation events from ``eyetracking.fix.txt``.

        ``unit_index`` and ``char_offset`` are aligned to the stimulus's
        ``"words"`` layer; an off-text fixation (source ``word_pos = "NaN"``)
        gets ``unit_index = None``. ``metadata["blink"]`` is ``True`` when the
        source ``blink`` column is ``"before"``, ``"after"`` or ``"both"``.

        Args:
            participant: Filter to one participant if set.
            stimulus_id: Filter to one stimulus if set.
            exclude_blink: When ``True``, drop fixations marked as
                blink-adjacent. Default ``False``: the published
                ``eyetracking.RT.txt`` per-word measures include
                blink-adjacent fixations, so the default surfaces the same
                fixation set ``ds.data()`` was computed from.
        """
        result = super().fixations(participant=participant, stimulus_id=stimulus_id)
        if exclude_blink:
            result = [f for f in result if not f.metadata.get("blink", False)]
        return result
