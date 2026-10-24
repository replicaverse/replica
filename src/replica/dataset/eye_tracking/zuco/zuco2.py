"""ZuCo 2.0 eye-tracking view (Hollenstein et al., 2020).

Simultaneous EEG and eye-tracking during English sentence reading, in two
tasks (NR, TSR). Per-word measures FFD, GD, GPT and TRT, where GD is
populated for every fixated word and so is first-run dwell; released 500 Hz
sample counts are converted to milliseconds. Per-word band power over 105
electrodes is stored as annotation layers; the band-power view is
``zuco2_eeg``.

The ``results*.mat`` files are MATLAB v7.3, one per subject and task.

Reference: https://aclanthology.org/2020.lrec-1.18
Data: https://osf.io/2urht/
"""

import glob
import os
from collections.abc import Collection

import numpy as np

from replica.dataset.contracts import DatasetInfo
from replica.dataset.measures import ZUCO_MAT_MEASURES
from replica.dataset.metadata import io as metadata_io

from .base import EEG_BANDS, N_ELECTRODES, ZuCoTaskDataset
from .continuous import WordFragments

# MATLAB field → measure name
MEASURE_COLUMNS = dict(ZUCO_MAT_MEASURES)


class ZuCo2(ZuCoTaskDataset):
    """ZuCo 2.0 eye-tracking + EEG dataset (Hollenstein et al., 2020).

    Args:
        task: ``"NR"`` (default) or ``"TSR"``.
        continuous: Also build each sentence's EEG and gaze recordings.

    Usage::

        ds = ZuCo2(task="NR")
        ds.load("datasets/eye-tracking/original/zuco2")
        data = ds.data(measure="first_run")
    """

    info = DatasetInfo(name="zuco2", language="en")
    _declared_measures = (
        *MEASURE_COLUMNS.values(),
        *(f"eeg_{band}" for band in EEG_BANDS.values()),
    )
    default_measure = "first_run"
    TASKS = ("NR", "TSR")

    def load(
        self,
        path: str,
        *,
        max_subjects: int | None = None,
        participants: Collection[str] | None = None,
        load_eeg: bool = True,
    ) -> None:
        """Load ZuCo 2.0 data from the ``results*_{TASK}.mat`` files in ``path``.

        A sentence a subject did not read is a scalar NaN instead of a group and
        is skipped; stimuli come from the first subject with data for the sentence.

        Args:
            path: Directory containing the ``.mat`` files.
            max_subjects: Limit the number of subjects (for testing).
            participants: Subjects to read, named as in the file names (``YAC``);
                ``None`` reads all.
            load_eeg: Also load per-word band power as layers and mean-power
                measures.
        """

        import h5py

        self._declared_measures = (
            *MEASURE_COLUMNS.values(),
            *(f"eeg_{band}" for band in EEG_BANDS.values() if load_eeg),
        )
        pattern = os.path.join(path, f"results*_{self.task}.mat")
        mat_files = sorted(glob.glob(pattern))

        if not mat_files:
            raise FileNotFoundError(
                f"No results*_{self.task}.mat files found in {path}.\n"
                f"Download from https://osf.io/2urht/"
            )

        subjects = self._subject_files(mat_files, participants, max_subjects)
        mat_files = [path for _, path in subjects]

        self._load_metadata(
            path,
            files=[os.path.basename(f) for f in mat_files],
            metadata_path=metadata_io.find_dataset_yaml(path),
        )

        # Preserve the task's sentence inventory and check every reader's text.
        stimuli = self._build_stimuli_from_files(mat_files, h5py)

        # Keep compact fixation arrays alongside the per-word summaries.
        for subject_id, mat_path in subjects:
            with h5py.File(mat_path, "r") as fh:
                self._load_subject(fh, subject_id, stimuli, load_eeg)

    def _read_matlab_string(self, fh, ref) -> str:
        """Dereference an HDF5 object reference to a MATLAB string."""
        data = np.array(fh[ref]).flatten()
        return "".join(chr(int(c)) for c in data)

    @staticmethod
    def _read_array(field):
        """Decode an HDF5 MATLAB array, including its empty-dimension marker."""
        return np.empty(0) if field.attrs.get("MATLAB_empty", 0) else np.asarray(field)

    def _read_scalar(self, fh, ref) -> float | None:
        """Read a duration: empty MATLAB fields are zero; scalar NaN stays missing."""
        field = fh[ref]
        if field.attrs.get("MATLAB_empty", 0):
            return 0.0  # Released empty duration field: no fixation on the word.
        arr = np.array(field)
        if arr.size == 0:
            return 0.0
        if arr.size == 1:
            val = float(arr.flat[0])
            return val if not np.isnan(val) else np.nan
        return None

    @staticmethod
    def _is_valid_sentence(fh, word_ref) -> bool:
        """Check if a sentence entry is a valid Group (not a scalar NaN)."""
        import h5py

        return isinstance(fh[word_ref], h5py.Group)

    def _build_stimuli_from_files(self, mat_files: list[str], h5py_mod) -> list:
        """Build stimuli and verify sentence identity across all selected subjects.

        Return observed stimuli; ``_sent_idx_map`` maps the original sentence
        indices to their positions in that list.
        """
        stimuli: list = []
        for mat_path in mat_files:
            with h5py_mod.File(mat_path, "r") as fh:
                sd = fh["sentenceData"]
                n_sents = len(sd["word"])
                stimuli.extend([None] * max(0, n_sents - len(stimuli)))
                for sent_idx in range(n_sents):
                    word_ref = sd["word"][sent_idx][0]
                    if not self._is_valid_sentence(fh, word_ref):
                        continue
                    word_group = fh[word_ref]
                    words = [
                        self._read_matlab_string(fh, ref[0])
                        for ref in word_group["content"]
                    ]
                    if stimuli[sent_idx] is None:
                        stimuli[sent_idx] = self._make_stimulus(
                            words,
                            f"zuco2_{self.task}_{sent_idx}",
                            {"task": self.task, "sentence_idx": sent_idx},
                        )
                    elif words != stimuli[sent_idx].layers["words"].values:
                        raise ValueError(
                            f"ZuCo2 {os.path.basename(mat_path)}: conflicting text at sentence {sent_idx}"
                        )

        # Remove any sentences that no subject had data for
        valid_indices = [
            i for i, stimulus in enumerate(stimuli) if stimulus is not None
        ]
        stimuli = [stimuli[i] for i in valid_indices]
        # Store mapping from original index → stimulus list position
        self._sent_idx_map = {orig: pos for pos, orig in enumerate(valid_indices)}

        return stimuli

    def _load_subject(
        self,
        fh,
        subject_id: str,
        stimuli: list,
        load_eeg: bool,
    ) -> None:
        """Load one subject's eye-tracking (and optionally EEG) data."""
        sd = fh["sentenceData"]
        n_sents = len(sd["word"])
        skipped = {"eeg": 0, "fragments": 0, "mismatch": 0}

        for sent_idx in range(n_sents):
            # Skip sentences this subject didn't read
            word_ref = sd["word"][sent_idx][0]
            if not self._is_valid_sentence(fh, word_ref):
                continue
            if sent_idx not in self._sent_idx_map:
                raise ValueError(
                    f"ZuCo2 {subject_id}: missing stimulus for sentence {sent_idx}"
                )
            stimulus = stimuli[self._sent_idx_map[sent_idx]]
            word_group = fh[word_ref]
            n_words = len(word_group["content"])

            # Extract eye-tracking measures
            measure_vals: dict[str, np.ndarray] = {
                m: np.full(n_words, np.nan) for m in MEASURE_COLUMNS.values()
            }

            for w_idx in range(n_words):
                for mat_key, measure_name in MEASURE_COLUMNS.items():
                    ref = word_group[mat_key][w_idx][0]
                    val = self._read_scalar(fh, ref)
                    if val is not None:
                        measure_vals[measure_name][w_idx] = val

            self._append_reading_measures(stimulus, subject_id, measure_vals)
            fixation_arrays = bounds = None
            if "allFixations" in sd and "wordbounds" in sd:
                raw = fh[sd["allFixations"][sent_idx, 0]]
                if hasattr(raw, "keys") and all(
                    key in raw for key in ("x", "y", "duration")
                ):
                    positions = []
                    for refs in word_group["fixPositions"]:
                        field = fh[refs[0]]
                        positions.append(self._read_array(field))
                    fixation_arrays = {
                        key: self._read_array(raw[key])
                        for key in ("x", "y", "duration")
                    }
                    bounds = np.asarray(fh[sd["wordbounds"][sent_idx, 0]]).T
                    self._store_fixation_trial(
                        stimulus, subject_id, fixation_arrays, bounds, positions
                    )

            # EEG band power as annotation layers (per-subject)
            if load_eeg:
                self._load_eeg_layers(fh, word_group, stimulus, subject_id, n_words)
            if self.continuous:
                outcome = self._load_sentence_recordings(
                    fh,
                    sd,
                    sent_idx,
                    word_group,
                    stimulus,
                    subject_id,
                    fixation_arrays,
                    bounds,
                )
                if outcome is not None:
                    skipped[outcome] += 1
        self._report_skipped(subject_id, skipped)

    def _load_sentence_recordings(
        self,
        fh,
        sd,
        sent_idx,
        word_group,
        stimulus,
        subject_id,
        fixation_arrays,
        bounds,
    ) -> str | None:
        """Build one sentence's recordings, or name what the release lacks.

        Returns ``"eeg"`` without the sentence EEG, ``"fragments"`` without complete
        fixation fragments and ``"mismatch"`` when they contradict the EEG.
        """
        import h5py

        raw = fh[sd["rawData"][sent_idx, 0]]
        if not isinstance(raw, h5py.Dataset) or raw.ndim != 2 or min(raw.shape) < 2:
            return "eeg"
        if fixation_arrays is None or not {"rawEEG", "rawET"} <= set(word_group):
            return "fragments"
        words = []
        for w_idx in range(len(word_group["content"])):
            positions = self._read_array(fh[word_group["fixPositions"][w_idx][0]])
            positions = tuple(int(p) for p in positions.ravel())
            fragments = []
            for field in ("rawEEG", "rawET"):
                refs = fh[word_group[field][w_idx][0]]
                # An unfixated word stores the empty marker instead of references.
                if refs.dtype != h5py.ref_dtype:
                    fragments.append(())
                else:
                    fragments.append(tuple(np.asarray(fh[ref[0]]) for ref in refs))
            eeg, gaze = fragments
            if (
                len(eeg) != len(positions)
                or len(gaze) != len(positions)
                or any(
                    fragment.ndim != 2 or min(fragment.shape) < 2
                    for fragment in eeg + gaze
                )
            ):
                return "fragments"
            words.append(WordFragments(eeg, gaze, positions))
        return self._store_recordings(
            stimulus,
            subject_id,
            np.asarray(raw),
            words,
            fixation_arrays,
            bounds,
            {"task": self.task, "sentence_idx": sent_idx},
        )

    def _load_eeg_layers(
        self,
        fh,
        word_group,
        stimulus,
        subject_id: str,
        n_words: int,
    ) -> None:
        """Load per-word EEG frequency-band power.

        Creates both:
        - Annotation layers with full electrode data (105-dim per word)
        - Datum objects with mean power across electrodes (scalar per word)
        """
        band_data = {}
        for mat_key, band_name in EEG_BANDS.items():
            if mat_key not in word_group:
                continue
            # (n_words, n_electrodes), NaN for unfixated words
            values = np.full((n_words, N_ELECTRODES), np.nan)
            for w_idx in range(n_words):
                arr = np.array(fh[word_group[mat_key][w_idx][0]])
                if arr.shape == (N_ELECTRODES, 1):
                    values[w_idx, :] = arr[:, 0]
            band_data[band_name] = values
        self._add_band_layers(stimulus, subject_id, band_data)
