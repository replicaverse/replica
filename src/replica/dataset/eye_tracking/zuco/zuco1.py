"""ZuCo 1.0 eye-tracking view (Hollenstein et al., 2018).

Simultaneous EEG and eye-tracking during natural English sentence reading, in
three tasks (NR, SR, TSR). Per-word measures FFD, GD, GPT and TRT, where GD
is populated for every fixated word and so is first-run dwell; released
500 Hz sample counts are converted to milliseconds. The band-power view is
``zuco1_eeg``, the N400/P600 view of Kuribayashi et al. (2025) ``zuco_n400``.

The ``results*.mat`` files are MATLAB v5, read with :func:`scipy.io.loadmat`:
one ``sentenceData`` struct per sentence with its ``content`` and a ``word``
struct array of per-word measures, empty for unfixated words.

Reference: https://doi.org/10.1038/sdata.2018.291
Data: https://osf.io/q3zws/
"""

import glob
import os
from collections.abc import Collection

import numpy as np

from replica.dataset.contracts import DatasetInfo
from replica.dataset.measures import ZUCO_MAT_MEASURES
from replica.dataset.metadata import io as metadata_io

from ....core import Stimulus
from .base import EEG_BANDS, N_ELECTRODES, ZuCoTaskDataset
from .continuous import WordFragments

# MATLAB per-word field -> canonical measure name (source durations in samples).
MEASURE_COLUMNS = dict(ZUCO_MAT_MEASURES)


# Exact NR display text from the authors' task_materials/relations_labels_task2.csv
# (https://osf.io/tu6yx). The MATLAB export lost these four punctuation characters.
_NR_TEXT_REPAIRS = {
    "Hindman died in Baltimore, Maryland and is buried in St. Paul�s Burial Ground there.": "Hindman died in Baltimore, Maryland and is buried in St. Paul’s Burial Ground there.",
    "From 1775 and 1777, Hindman served in Maryland�s revolutionary government, as the state�s Treasurer for the Eastern Shore.": "From 1775 and 1777, Hindman served in Maryland’s revolutionary government, as the state’s Treasurer for the Eastern Shore.",
    "Stella Dorothea Gibbons (5 January 1902�19 December 1989) was an English novelist and poet.": "Stella Dorothea Gibbons (5 January 1902—19 December 1989) was an English novelist and poet.",
}


# Exact TSR display text from relations_labels_task3.csv (https://osf.io/2r95j).
# The task material has a double acute accent (˝) at the replaced character.
_TSR_TEXT_REPAIRS = {
    "He maintained homes in Long Island, New York and Greenwich, Connecticut; the family compound at Kennebunkport, Maine; a 10,000 acre (40 km�) plantation in South Carolina; and an island retreat in Florida.": "He maintained homes in Long Island, New York and Greenwich, Connecticut; the family compound at Kennebunkport, Maine; a 10,000 acre (40 km˝) plantation in South Carolina; and an island retreat in Florida."
}
_TASK_TEXT_REPAIRS = {"NR": _NR_TEXT_REPAIRS, "TSR": _TSR_TEXT_REPAIRS}
_TASK_TEXT_SOURCES = {"NR": "https://osf.io/tu6yx/", "TSR": "https://osf.io/2r95j/"}


def _display_tokens(tokens: list[str], task: str) -> list[str]:
    """Recover only replacement characters established by the task materials."""
    text = " ".join(tokens)
    if "\ufffd" not in text:
        return tokens
    reference = _TASK_TEXT_REPAIRS.get(task, {}).get(text)
    if reference is None:
        raise ValueError("ZuCo text has an unverified replacement character")
    restored = reference.split(" ")
    if len(restored) != len(tokens):
        raise ValueError("ZuCo task text does not preserve the measured word units")
    return restored


def _fragments(value) -> tuple[np.ndarray, ...]:
    """Per-fixation arrays of a squeezed MATLAB v5 word field, time first.

    One fixation is a plain array, several a cell of arrays, none an empty array.
    """
    array = np.asarray(value)
    if array.size == 0:
        return ()
    if array.dtype == object:
        return tuple(np.asarray(item).T for item in array.ravel())
    return (array.T,)


def _band_arrays(words, n_words: int) -> dict[str, np.ndarray]:
    """Per-word band power of a sentence, ``(n_words, 105)`` per released band."""
    band_data = {}
    for mat_key, band in EEG_BANDS.items():
        if not hasattr(words[0], mat_key):
            continue
        values = np.full((n_words, N_ELECTRODES), np.nan)
        for w_idx, word in enumerate(words):
            array = np.asarray(getattr(word, mat_key), dtype=float).ravel()
            if array.shape == (N_ELECTRODES,):
                values[w_idx] = array
        band_data[band] = values
    return band_data


def _scalar(value) -> float:
    """MATLAB word measure -> float; unfixated words store an empty array."""
    arr = np.asarray(value).ravel()
    if arr.size == 0:
        return 0.0
    try:
        return float(arr[0])
    except (TypeError, ValueError):
        return np.nan


def _tsr_zkb_indices(sentences: np.ndarray) -> dict[int, int]:
    """Map ZKB's compacted final TSR block to validated zero-based task positions."""
    # The released file (https://osf.io/rp5vc/) stores task rows 362 through
    # 406 at rows 314 through 358. Empty tail records retain the task text
    # (https://osf.io/2r95j/). Missing rows 314 through 361 have no observations.
    tokens = []
    for sentence in sentences:
        words = np.atleast_1d(getattr(sentence, "word", []))
        tokens.append(
            [str(word.content) for word in words]
            if words.size and hasattr(words[0], "content")
            else []
        )
    if (
        len(sentences) != 407
        or not all(tokens[:359])
        or any(tokens[359:])
        or any(
            " ".join(tokens[index])
            != str(getattr(sentences[index + 48], "content", ""))
            for index in range(314, 359)
        )
    ):
        raise ValueError("ZuCo1 ZKB: unrecognized TSR trial layout")
    return {index: index if index < 314 else index + 48 for index in range(359)}


class ZuCo1(ZuCoTaskDataset):
    """ZuCo 1.0 eye-tracking dataset (Hollenstein et al., 2018).

    Args:
        task: ``"NR"`` (normal reading, default), ``"SR"`` (sentiment reading)
            or ``"TSR"`` (task-specific reading).
        continuous: Also build each sentence's EEG and gaze recordings.

    Usage::

        ds = ZuCo1(task="NR")
        ds.load("datasets/eye-tracking/original/zuco1")
        data = ds.data(measure="first_run")
    """

    info = DatasetInfo(name="zuco1", language="en")
    _declared_measures = (
        *MEASURE_COLUMNS.values(),
        *(f"eeg_{band}" for band in EEG_BANDS.values()),
    )
    default_measure = "first_run"
    TASKS = ("NR", "SR", "TSR")

    def load(
        self,
        path: str,
        *,
        max_subjects: int | None = None,
        participants: Collection[str] | None = None,
        load_eeg: bool = True,
    ) -> None:
        """Load ZuCo 1.0 data from the ``results*_{TASK}.mat`` files in ``path``.

        Args:
            path: Directory containing the ``.mat`` files.
            max_subjects: Limit the number of subjects (for testing).
            participants: Subjects to read, named as in the file names
                (``ZAB``); ``None`` reads all.
            load_eeg: Also load per-word band power as layers and mean-power
                measures.
        """
        self._declared_measures = (
            *MEASURE_COLUMNS.values(),
            *(f"eeg_{band}" for band in EEG_BANDS.values() if load_eeg),
        )
        import scipy.io as sio

        pattern = os.path.join(path, f"results*_{self.task}.mat")
        mat_files = sorted(glob.glob(pattern))
        if not mat_files:
            raise FileNotFoundError(
                f"No results*_{self.task}.mat files found in {path}.\n"
                f"Download the ZuCo 1.0 '{self.task}' Matlab files from "
                f"https://osf.io/q3zws/"
            )
        subjects = self._subject_files(mat_files, participants, max_subjects)
        mat_files = [path for _, path in subjects]

        self._load_metadata(
            path,
            files=[os.path.basename(f) for f in mat_files],
            metadata_path=metadata_io.find_dataset_yaml(path),
        )

        # Share task positions only after validating each reader's word text.
        stimuli: dict[int, Stimulus] = {}
        for subject_id, mat_path in subjects:
            mat = sio.loadmat(mat_path, struct_as_record=False, squeeze_me=True)
            sentence_data = np.atleast_1d(mat["sentenceData"])
            indices = (
                _tsr_zkb_indices(sentence_data)
                if self.task == "TSR" and subject_id == "ZKB"
                else {}
            )

            skipped = {"eeg": 0, "fragments": 0, "mismatch": 0}
            for source_idx, sentence in enumerate(sentence_data):
                words = np.atleast_1d(getattr(sentence, "word", []))
                if words.size == 0 or not hasattr(words[0], "content"):
                    continue  # sentence this subject did not read
                sent_idx = indices.get(source_idx, source_idx)
                source_tokens = [str(getattr(w, "content", "")) for w in words]
                tokens = _display_tokens(source_tokens, self.task)

                stimulus = stimuli.get(sent_idx)
                if stimulus is None:
                    metadata = {"task": self.task, "sentence_idx": sent_idx}
                    if source_tokens != tokens:
                        metadata.update(
                            source_text=" ".join(source_tokens),
                            text_source=_TASK_TEXT_SOURCES[self.task],
                        )
                    stimulus = self._make_stimulus(
                        tokens,
                        f"zuco1_{self.task}_{sent_idx}",
                        metadata,
                    )
                    stimuli[sent_idx] = stimulus
                n_words = len(stimulus.layers["words"].spans)
                if tokens != stimulus.layers["words"].values:
                    raise ValueError(
                        f"ZuCo1 {subject_id}: conflicting text at sentence {sent_idx}"
                    )

                vals = {m: np.full(n_words, np.nan) for m in MEASURE_COLUMNS.values()}
                for w_idx, word in enumerate(words):
                    for mat_key, measure in MEASURE_COLUMNS.items():
                        vals[measure][w_idx] = _scalar(getattr(word, mat_key, np.nan))

                self._append_reading_measures(
                    stimulus, subject_id, vals, source_sentence_idx=source_idx
                )
                raw = getattr(sentence, "allFixations", None)
                if all(hasattr(raw, key) for key in ("x", "y", "duration")):
                    self._store_fixation_trial(
                        stimulus,
                        subject_id,
                        {key: getattr(raw, key) for key in ("x", "y", "duration")},
                        np.atleast_2d(sentence.wordbounds),
                        [getattr(word, "fixPositions", []) for word in words],
                    )
                if load_eeg:
                    self._add_band_layers(
                        stimulus, subject_id, _band_arrays(words, n_words)
                    )
                if self.continuous:
                    outcome = self._load_sentence_recordings(
                        sentence, words, stimulus, subject_id, sent_idx, source_idx
                    )
                    if outcome is not None:
                        skipped[outcome] += 1
            self._report_skipped(subject_id, skipped)

    def _load_sentence_recordings(
        self, sentence, words, stimulus, subject_id, sent_idx, source_idx
    ) -> str | None:
        """Build one sentence's recordings, or name what the release lacks.

        Returns ``"eeg"`` without the sentence EEG, ``"fragments"`` without complete
        fixation fragments and ``"mismatch"`` when they contradict the EEG.
        """
        raw = np.asarray(getattr(sentence, "rawData", ()))
        if raw.ndim != 2 or min(raw.shape) < 2:
            return "eeg"
        fixations = getattr(sentence, "allFixations", None)
        if not all(hasattr(fixations, key) for key in ("x", "y", "duration")):
            return "fragments"
        if not all(hasattr(words[0], field) for field in ("rawEEG", "rawET")):
            return "fragments"
        fragments = []
        for word in words:
            positions = np.asarray(getattr(word, "fixPositions", ()), dtype=float)
            positions = np.atleast_1d(positions)
            positions = tuple(int(p) for p in positions.ravel())
            eeg = _fragments(word.rawEEG)
            gaze = _fragments(word.rawET)
            if (
                len(eeg) != len(positions)
                or len(gaze) != len(positions)
                or any(
                    fragment.ndim != 2 or min(fragment.shape) < 2
                    for fragment in eeg + gaze
                )
            ):
                return "fragments"
            fragments.append(WordFragments(eeg, gaze, positions))
        return self._store_recordings(
            stimulus,
            subject_id,
            raw.T,
            fragments,
            {key: getattr(fixations, key) for key in ("x", "y", "duration")},
            np.atleast_2d(sentence.wordbounds),
            {
                "task": self.task,
                "sentence_idx": sent_idx,
                "source_sentence_idx": source_idx,
            },
        )
