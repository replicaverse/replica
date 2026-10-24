"""Shared task selection, duration conversion and fixation access for ZuCo."""

import logging
import os
import warnings
from collections.abc import Collection
from dataclasses import replace

import numpy as np

from replica.dataset.base import EyeTrackingDataset
from replica.dataset.contracts import Fixation
from replica.dataset.measures import ZUCO_MAT_MEASURES

from ....core import AnnotationLayer, Datum, Stimulus
from ...loading.fixation_utils import SyntheticTimer
from ...loading.utils import append_measure_datums
from .continuous import (
    FragmentMismatch,
    WordFragments,
    sentence_recordings,
    word_fixation_pairs,
)

#: Per-word band-power fields of both releases, keyed by their MATLAB name:
#: the power over the word's total reading time in eight frequency bands.
EEG_BANDS = {
    "TRT_t1": "theta_1",
    "TRT_t2": "theta_2",
    "TRT_a1": "alpha_1",
    "TRT_a2": "alpha_2",
    "TRT_b1": "beta_1",
    "TRT_b2": "beta_2",
    "TRT_g1": "gamma_1",
    "TRT_g2": "gamma_2",
}
#: Electrodes the released band power covers.
N_ELECTRODES = 105

# The author extractor stores EEG event durations in 500-Hz samples.
# firstLevel_NR.m: https://osf.io/gan9v/ (word assignment and feature extraction).
_SAMPLE_RATE_HZ = 500


class ZuCoTaskDataset(EyeTrackingDataset):
    """Eye-tracking release selected by one of its declared reading tasks."""

    TASKS: tuple[str, ...] = ()
    # Inclusive source rectangles can count a physical fixation in two word totals.
    additive_measures = frozenset()

    def __init__(self, task: str = "NR", continuous: bool = False):
        """Pick the reading task whose files are loaded.

        Args:
            task: One of the release's ``TASKS``; ``"NR"`` is normal reading.
            continuous: Also build each sentence's 500 Hz EEG and gaze recordings
                with fixation-locked word onsets; see
                :mod:`replica.dataset.eye_tracking.zuco.continuous`. A reader's
                EEG is about a gigabyte, so select readers with ``participants``.

        Raises:
            ValueError: ``task`` is not one of ``TASKS``.
        """
        super().__init__()
        if task not in self.TASKS:
            raise ValueError(f"task must be one of {self.TASKS}, got {task!r}")
        self.task = task
        self.continuous = continuous
        self._fixation_records: dict[tuple[str, str], tuple] = {}

    def _subject_files(
        self,
        mat_files: list[str],
        participants: Collection[str] | None,
        max_subjects: int | None,
    ) -> list[tuple[str, str]]:
        """Pair each task file with its reader and apply the selection.

        Args:
            mat_files: Sorted ``results<reader>_<task>.mat`` paths.
            participants: Readers to keep; ``None`` keeps all.
            max_subjects: Cap applied after the selection.

        Raises:
            ValueError: A requested reader has no file.
        """
        pairs = [
            (
                os.path.basename(path)
                .replace("results", "")
                .replace(f"_{self.task}.mat", ""),
                path,
            )
            for path in mat_files
        ]
        if participants is not None:
            wanted = set(participants)
            unknown = sorted(wanted - {subject for subject, _ in pairs})
            if unknown:
                raise ValueError(
                    f"ZuCo {self.task}: no task file for participants {unknown}"
                )
            pairs = [pair for pair in pairs if pair[0] in wanted]
        if max_subjects is not None:
            pairs = pairs[:max_subjects]
        return pairs

    def _store_recordings(
        self,
        stimulus: Stimulus,
        participant_id: str,
        eeg: np.ndarray,
        words: list[WordFragments],
        arrays: dict,
        bounds,
        metadata: dict,
    ) -> str | None:
        """Attach a trial's EEG and gaze recordings.

        Returns:
            ``None``, or ``"mismatch"`` when the fragments contradict the trial;
            the debug log names the trial.
        """
        fixations = {
            key: np.asarray(value, dtype=float).ravel() for key, value in arrays.items()
        }
        try:
            recordings = sentence_recordings(
                stimulus,
                participant_id,
                eeg,
                words,
                fixations,
                np.asarray(bounds, dtype=float),
                metadata,
            )
        except FragmentMismatch as exc:
            logging.getLogger("replica").debug(
                "ZuCo %s %s: %s", participant_id, stimulus.stimulus_id, exc
            )
            return "mismatch"
        for recording in recordings:
            self.add_recording(recording)
        return None

    def _add_band_layers(
        self, stimulus: Stimulus, subject_id: str, band_data: dict[str, np.ndarray]
    ) -> None:
        """Attach a reader's per-word band power as layers and mean-power datums.

        Args:
            stimulus: Sentence.
            subject_id: Reader.
            band_data: ``(n_words, n_electrodes)`` power per band, NaN rows for
                unfixated words.
        """
        spans = stimulus.layers["words"].spans
        for band, values in band_data.items():
            stimulus.add_layer(
                AnnotationLayer(f"eeg_{band}_{subject_id}", spans, list(values))
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                mean_power = np.nanmean(values, axis=1)
            self._data.append(
                Datum(
                    stimulus=stimulus,
                    measure_name=f"eeg_{band}",
                    values=mean_power,
                    participant_id=subject_id,
                )
            )

    def _report_skipped(self, participant_id: str, skipped: dict[str, int]) -> None:
        """Log the sentences of a reader that got no recordings, by cause."""
        if any(skipped.values()):
            logging.getLogger("replica").warning(
                "ZuCo %s %s: no recordings for %d sentences without EEG, %d "
                "without fixation fragments and %d whose fragments contradict "
                "the sentence EEG",
                self.task,
                participant_id,
                skipped["eeg"],
                skipped["fragments"],
                skipped["mismatch"],
            )

    def _append_reading_measures(
        self,
        stimulus: Stimulus,
        participant_id: str,
        values: dict[str, np.ndarray],
        *,
        source_sentence_idx: int | None = None,
    ) -> None:
        """Convert released sample counts to milliseconds and retain provenance.

        Args:
            stimulus: Sentence whose word order matches the arrays.
            participant_id: Reader identified by the MATLAB file.
            values: Released duration arrays indexed by canonical measure name.
            source_sentence_idx: Original zero-based row in the participant's
                MATLAB sentence array.
        """
        for source_column, measure in ZUCO_MAT_MEASURES.items():
            source_values = values[measure]
            append_measure_datums(
                self._data,
                stimulus=stimulus,
                participant_id=participant_id,
                measure_names=[measure],
                measure_values=[source_values * (1000 / _SAMPLE_RATE_HZ)],
                metadata={
                    "measurement_source": "released",
                    "source_measure": source_column,
                    "source_algorithm": "zuco_inclusive_word_assignments@1",
                    "source_event_filter": "on_word_and_at_least_100_ms",
                    "source_unit": "samples",
                    "source_sample_rate_hz": _SAMPLE_RATE_HZ,
                    "released_values": source_values.tolist(),
                    **(
                        {"source_sentence_idx": source_sentence_idx}
                        if source_sentence_idx is not None
                        else {}
                    ),
                },
            )

    def _store_fixation_trial(self, stimulus, participant, arrays, bounds, positions):
        """Keep compact source arrays; create physical events only on request."""
        self._fixation_records[(participant, stimulus.stimulus_id)] = (
            stimulus,
            {
                key: np.asarray(value, dtype=float).ravel().copy()
                for key, value in arrays.items()
            },
            np.asarray(bounds, dtype=float).copy(),
            [np.asarray(p, dtype=float).ravel().copy() for p in positions],
        )

    def _trial_fixations(self, key, record):
        """Resolve inclusive rectangles without duplicating physical events."""
        stimulus, arrays, bounds, positions = record
        x, y, durations = (arrays[k] for k in ("x", "y", "duration"))
        n_words = len(stimulus.layers["words"])
        if len(x) != len(y) or len(x) != len(durations):
            raise ValueError("ZuCo fixation coordinates and durations differ in length")
        if bounds.shape != (n_words, 4) or not np.isfinite(bounds).all():
            raise ValueError("ZuCo word rectangles do not match the native units")
        if (bounds[:, :2] > bounds[:, 2:]).any():
            raise ValueError("ZuCo word rectangle has reversed boundaries")
        if not np.isfinite(durations).all() or (durations < 0).any():
            raise ValueError("ZuCo fixation durations must be finite and nonnegative")
        inside, _, assignments = word_fixation_pairs(x, y, durations, bounds)
        if len(positions) != n_words or any(
            not np.array_equal(p, np.flatnonzero(assignments == i) + 1)
            for i, p in enumerate(positions)
        ):
            raise ValueError(
                "ZuCo word fixPositions disagree with source geometry and filtering"
            )
        timer = SyntheticTimer()
        events = []
        for i, duration in enumerate(durations):
            units = tuple(int(u) for u in np.flatnonzero(inside[i]))
            start, end = timer.next(key, float(duration * 2))
            events.append(
                Fixation(
                    participant_id=key[0],
                    stimulus_id=key[1],
                    start_time=start,
                    end_time=end,
                    duration=float(duration * 2),
                    x=float(x[i]),
                    y=float(y[i]),
                    unit_index=units[0] if len(units) == 1 else None,
                    char_offset=None,
                    fixation_index=i + 1,
                    candidate_unit_indices=units if len(units) > 1 else (),
                    metadata={
                        SyntheticTimer.SENTINEL_KEY: True,
                        "source_fixation_index": i + 1,
                        "source_duration_samples": float(duration),
                        "source_word_assignments": units,
                        "source_assignment": "inclusive_word_bounds",
                    },
                )
            )
        return events

    def fixation_trials(self):
        """Yield physical fixation sequences, one participant and sentence at a time.

        Durations are milliseconds. Synthetic timestamps preserve order and dwell
        but omit saccades; the release supplies no event onset timestamps.
        Events below the author's threshold and off-text events remain in this view.

        Raises:
            NotImplementedError: The loaded files have no fixation records.
            ValueError: Source geometry, event arrays or assignment indices disagree.
        """
        if not self._fixation_records:
            raise NotImplementedError(
                "ZuCo fixation records are absent from the loaded files"
            )
        for key, record in self._fixation_records.items():
            yield key, self._trial_fixations(key, record)

    def fixations(self, participant=None, stimulus_id=None):
        """Read physical events, preserving source order and boundary alternatives.

        Args:
            participant: Keep one reader; None keeps all readers.
            stimulus_id: Keep one sentence; None keeps all sentences.

        Raises:
            NotImplementedError: The loaded files have no fixation records.
        """
        if not self._fixation_records:
            raise NotImplementedError(
                "ZuCo fixation records are absent from the loaded files"
            )
        return [
            event
            for key, record in self._fixation_records.items()
            if (participant is None or key[0] == participant)
            and (stimulus_id is None or key[1] == stimulus_id)
            for event in self._trial_fixations(key, record)
        ]

    def _reaggregation_fixations(self, datum, events):
        """Apply the author's 50-sample threshold and on-word selection."""
        return [
            e
            for e in events
            if e.duration >= 100
            and (e.unit_index is not None or e.candidate_unit_indices)
        ]

    def _reaggregation_arrays(self, events, stimulus, layer, **options):
        """Validate native word assignments; use physical events on custom spans."""
        if stimulus.layers[layer].spans == stimulus.layers["words"].spans:
            expanded = []
            for event in events:
                units = event.candidate_unit_indices or (event.unit_index,)
                for unit in units:
                    if unit is None:
                        raise ValueError(
                            "ZuCo native assignment requires an on-word fixation"
                        )
                    expanded.append(
                        replace(
                            event,
                            unit_index=unit,
                            candidate_unit_indices=(),
                            fixation_index=len(expanded) + 1,
                        )
                    )
            events = expanded
        return super()._reaggregation_arrays(events, stimulus, layer, **options)

    def _reaggregation_provenance(self, datum, layer):
        """Distinguish released assignments from physical events on custom units."""
        native = (
            datum.stimulus.layers[layer].spans
            == datum.stimulus.layers[datum.layer].spans
        )
        return {
            "source_algorithm": "zuco_inclusive_word_assignments@1"
            if native
            else "standard@1",
            "source_assignment": "inclusive_word_bounds"
            if native
            else "physical_event_candidates",
            "source_event_filter": "on_word_and_at_least_100_ms",
            "synthetic_start_time": True,
        }

    def _has_empty_fixation_record(self, datum, *, selected):
        """Distinguish an explicit empty MATLAB trial from an absent source."""
        key = (datum.participant_id, datum.stimulus.stimulus_id)
        record = self._fixation_records.get(key)
        if record is None:
            return False
        events = self._trial_fixations(key, record)
        return not (
            self._reaggregation_fixations(datum, events) if selected else events
        )
