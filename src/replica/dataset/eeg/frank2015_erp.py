"""Frank et al. (2015) ERP amplitudes on the UCL sentences.

24 participants read the 205 sentences of the UCL corpus (Frank et al., 2013)
word by word. ``stimuli_erp.mat`` holds the mean amplitude of six ERP
components per word and participant with the authors' trial exclusions; the
EEGLAB files hold the continuous EEG (250 Hz, 34 channels, band-passed,
mastoid-referenced), read with ``continuous=True``. The release numbers no
sentence, so each is matched by text to its UCL ``sent_nr`` through the UCL
release's ``stimuli.txt``; stimulus ids are shared with ``ucl_et`` and
``ucl_spr``.

Reference: https://doi.org/10.1016/j.bandl.2014.10.006
"""

import logging
from collections.abc import Collection
from pathlib import Path

import numpy as np

from replica.dataset.base import Datum, EEGDataset
from replica.dataset.contracts import DatasetInfo
from replica.dataset.eye_tracking.devarda2023.covariates import DeVardaCovariates
from replica.dataset.metadata import io as metadata_io

from ...core import Modality, Presentation, Recording
from ..eye_tracking.ucl import _parse_ucl_stimuli

#: ERP components in the order of the release's component axis, with the
#: measurement windows of the paper.
COMPONENTS: tuple[str, ...] = ("elan", "lan", "n400", "epnp", "p600", "pnp")
#: Measure carrying each component's pre-onset baseline amplitude.
BASELINES: dict[str, str] = {name: f"{name}_baseline" for name in COMPONENTS}
#: Which released trial flag removes an observation, or none.
EXCLUSIONS: tuple[str, ...] = ("reject", "artefact", "none")
RELEASE_FILE = "stimuli_erp.mat"
README_FILE = "README.txt"
UCL_STIMULI_FILE = "stimuli.txt"
N_PARTICIPANTS = 24
#: A participant's continuous EEG, in EEGLAB format.
EEG_FILE = "EEG{participant:02d}.mat"
EEG_SAMPLE_RATE_HZ = 250
#: Event types above this value mark a sentence's first word; the sentence's
#: release index is the type minus the offset. Smaller types are word positions.
_FIRST_WORD_OFFSET = 50
#: The participant whose right mastoid was not recorded, so the authors
#: re-referenced to the left mastoid alone.
_LEFT_MASTOID_ONLY = "9"


def ucl_sentence_numbers(path: str | Path) -> dict[str, int]:
    """Map each UCL sentence text to its ``sent_nr``.

    Args:
        path: Directory of the UCL release holding ``stimuli.txt``.

    Raises:
        FileNotFoundError: The directory holds no ``stimuli.txt``.
        ValueError: Two UCL sentences share one text, so a text identifies no
            single sentence.
    """
    stimuli_file = Path(path) / UCL_STIMULI_FILE
    if not stimuli_file.is_file():
        raise FileNotFoundError(
            f"{stimuli_file} is missing; the UCL release supplies the sentence "
            "numbers (fetch ucl_et or pass ucl_path)"
        )
    numbers: dict[str, int] = {}
    for sent_nr, stimulus in _parse_ucl_stimuli(stimuli_file).items():
        if stimulus.text in numbers:
            raise ValueError(
                f"UCL sentences {numbers[stimulus.text]} and {sent_nr} share a text"
            )
        numbers[stimulus.text] = sent_nr
    return numbers


class Frank2015ERP(DeVardaCovariates, EEGDataset):
    """Frank et al. (2015) ERP amplitudes, one datum per participant and sentence.

    Measures: the released mean amplitudes ``elan``, ``lan``, ``n400``,
    ``epnp``, ``p600`` and ``pnp`` (0.50 Hz high-pass, modulus transformation)
    and a ``<component>_baseline`` with the mean potential over the 100 ms
    before word onset. Participants are ``"1"`` to ``"24"`` as in the release.

    The release flags trials as ``artefact`` and as ``reject`` (every trial its
    analysis dropped, including each sentence's first and last word);
    ``exclusions`` selects which flag becomes NaN. Unrecorded trials are NaN
    under every policy. Each datum's metadata carries both flags, the sentence's
    session position and the policy.

    Args:
        exclusions: ``"reject"`` (the paper's analysis set), ``"artefact"``
            (the trials de Varda et al. (2023) average over), or ``"none"``.
        continuous: Also read the continuous EEG; see :meth:`load`.
    """

    info = DatasetInfo(
        name="frank2015_erp",
        language="en",
        paradigm="eeg",
        presentation=[str(Presentation.RSVP)],
    )
    default_measure = "n400"
    default_presentation = Presentation.RSVP
    _declared_measures = (*COMPONENTS, *BASELINES.values())

    def __init__(self, exclusions: str = "reject", continuous: bool = False) -> None:
        """Fix which released flag removes an observation.

        Args:
            exclusions: One of :data:`EXCLUSIONS`; anything else raises.
            continuous: Read the continuous EEG as well.
        """
        super().__init__()
        if exclusions not in EXCLUSIONS:
            raise ValueError(
                f"exclusions must be one of {EXCLUSIONS}, got {exclusions!r}"
            )
        self.exclusions = exclusions
        self.continuous = continuous

    def load(
        self,
        path: str,
        release_file: str = RELEASE_FILE,
        ucl_path: str | None = None,
        participants: Collection[str] | None = None,
    ) -> None:
        """Read the release file and build the stimuli and data.

        With ``continuous=True`` the EEG files are read as well: the session's EEG
        from 100 ms before each word to 924 ms after it, NaN elsewhere, with one
        event per word. Each sentence becomes one :class:`~replica.core.Recording`
        over its recorded span, with ``start_time`` and ``event_times`` on the
        session clock, the channel labels, and metadata (session position, release
        index, part of an interrupted recording, reference, unrecorded samples
        inside the span). Word events before a file's first sentence event are
        left out with a warning.

        Args:
            path: Directory holding the release file and its README.
            release_file: File under *path* with the MATLAB arrays.
            ucl_path: Directory of the UCL release whose ``stimuli.txt`` numbers
                the sentences; ``None`` uses the default UCL data location.
            participants: Participants (``"1"`` to ``"24"``) whose EEG is read;
                ``None`` reads all. The amplitudes are loaded for everyone.

        Raises:
            FileNotFoundError: The UCL ``stimuli.txt`` is absent.
            ValueError: A release sentence is not a UCL sentence, the arrays have
                unexpected shapes, a requested participant does not exist, or an
                EEG file's events do not describe its sentences.
        """
        from scipy.io import loadmat

        if ucl_path is None:
            from replica.dataset.locations import UCL_ET_SUBDIR, dataset_path

            ucl_path = str(dataset_path(UCL_ET_SUBDIR))
        numbers = ucl_sentence_numbers(ucl_path)
        files = {RELEASE_FILE: release_file, README_FILE: README_FILE}
        if not (Path(path) / README_FILE).exists():
            del files[README_FILE]
        everyone = [str(number) for number in range(1, N_PARTICIPANTS + 1)]
        selected = everyone if participants is None else [str(p) for p in participants]
        unknown = sorted(set(selected) - set(everyone), key=str)
        if unknown:
            raise ValueError(f"no continuous EEG for participants {unknown}")
        if self.continuous:
            for participant in selected:
                name = EEG_FILE.format(participant=int(participant))
                files[name] = name
        self._load_metadata(
            path, files=files, metadata_path=metadata_io.find_dataset_yaml(path)
        )
        mat = loadmat(Path(path) / release_file)
        cells = mat["sentences"][:, 0]
        if np.asarray(mat["sentence_position"]).shape != (
            N_PARTICIPANTS,
            len(cells),
        ):
            raise ValueError(f"{release_file}: unexpected sentence_position shape")
        self._stimuli = []
        self._data = []
        self._reset_devarda()
        seen: dict[int, int] = {}
        for index, cell in enumerate(cells):
            words = [str(word.item()) for word in cell.reshape(-1)]
            text = " ".join(words)
            sent_nr = numbers.get(text)
            if sent_nr is None:
                raise ValueError(
                    f"sentence {index + 1} of {release_file} is not a UCL "
                    f"sentence: {text!r}"
                )
            if sent_nr in seen:
                raise ValueError(
                    f"sentences {seen[sent_nr]} and {index + 1} of {release_file} "
                    f"are both UCL sentence {sent_nr}"
                )
            seen[sent_nr] = index + 1
            self._build_sentence(mat, index, sent_nr, words)
        if self.continuous:
            self._load_recordings(
                Path(path), selected, np.asarray(mat["sentence_position"])
            )

    def _load_recordings(
        self, path: Path, participants: list[str], positions: np.ndarray
    ) -> None:
        """Read each participant's EEG file into one recording per sentence."""
        from scipy.io import loadmat

        for participant in participants:
            file = path / EEG_FILE.format(participant=int(participant))
            mat = loadmat(file, squeeze_me=True, struct_as_record=False)
            # An interrupted recording continues in EEG_part2 (README).
            parts = sorted(
                (key for key in mat if key == "EEG" or key.startswith("EEG_part")),
                key=lambda key: (key != "EEG", key),
            )
            if not parts:
                raise ValueError(f"{file.name} holds no EEG structure")
            for part, key in enumerate(parts, start=1):
                self._add_part(mat[key], participant, part, positions, file.name)

    def _add_part(
        self, eeg, participant: str, part: int, positions: np.ndarray, name: str
    ) -> None:
        """Cut one EEGLAB structure into the sentences its events mark."""
        if float(eeg.srate) != EEG_SAMPLE_RATE_HZ:
            raise ValueError(
                f"{name}: sampled at {eeg.srate} Hz, not {EEG_SAMPLE_RATE_HZ}"
            )
        data = np.asarray(eeg.data)
        labels = tuple(str(channel.labels) for channel in np.atleast_1d(eeg.chanlocs))
        if data.ndim != 2 or data.shape[0] != len(labels):
            raise ValueError(f"{name}: the data is not one row per labelled channel")
        events = np.atleast_1d(eeg.event)
        types = np.array([int(event.type) for event in events])
        # EEGLAB latencies are one-based sample indices.
        onsets = np.array([int(event.latency) for event in events]) - 1
        if len(onsets) and (
            onsets[0] < 0 or onsets[-1] >= data.shape[1] or (np.diff(onsets) <= 0).any()
        ):
            raise ValueError(f"{name}: event latencies are not increasing samples")
        absent = np.isnan(data).all(axis=0)
        if not np.array_equal(absent, np.isnan(data).any(axis=0)):
            raise ValueError(f"{name}: a sample is missing on some channels only")
        # Recorded blocks are the runs between NaN gaps; a sentence spans one
        # block, or several when a long SOA separates the windows of its words.
        change = np.flatnonzero(np.diff(np.r_[True, absent, True].astype(int)))
        starts, ends = change[0::2], change[1::2]
        blocks = np.searchsorted(starts, onsets, side="right") - 1
        if len(onsets) and ((blocks < 0) | (onsets >= ends[blocks])).any():
            raise ValueError(f"{name}: a word event falls on an unrecorded sample")
        first = np.flatnonzero(types > _FIRST_WORD_OFFSET)
        orphans = len(types) if first.size == 0 else int(first[0])
        if orphans:
            logging.getLogger("replica").warning(
                "%s: %d word events before the first sentence event belong to no "
                "sentence and are left out",
                name,
                orphans,
            )
        if first.size == 0:
            return
        used: set[int] = set()
        presented: set[int] = set()
        for a, b in zip(first, np.r_[first[1:], len(types)], strict=True):
            release_index = int(types[a]) - _FIRST_WORD_OFFSET
            if not 1 <= release_index <= len(self._stimuli):
                raise ValueError(
                    f"{name}: event type {types[a]} names sentence {release_index}, "
                    "which the release does not have"
                )
            if release_index in presented:
                raise ValueError(f"{name}: sentence {release_index} is presented twice")
            presented.add(release_index)
            stimulus = self._stimuli[release_index - 1]
            n_words = len(stimulus.layers["words"])
            if not np.array_equal(types[a + 1 : b], np.arange(2, n_words + 1)):
                raise ValueError(
                    f"{name}: the events of sentence {release_index} are not its "
                    f"word positions 2 to {n_words}"
                )
            span = range(int(blocks[a]), int(blocks[b - 1]) + 1)
            if used.intersection(span):
                raise ValueError(
                    f"{name}: sentence {release_index} shares a recorded block "
                    "with another sentence"
                )
            used.update(span)
            start, end = int(starts[span[0]]), int(ends[span[-1]])
            self.add_recording(
                Recording(
                    stimulus=stimulus,
                    values=np.ascontiguousarray(data[:, start:end].T),
                    measure_name="eeg",
                    modality=Modality.EEG,
                    participant_id=participant,
                    sample_interval=1 / EEG_SAMPLE_RATE_HZ,
                    start_time=start / EEG_SAMPLE_RATE_HZ,
                    event_times=onsets[a:b] / EEG_SAMPLE_RATE_HZ,
                    dimension_names=("time", "channel"),
                    dimension_labels={"channel": labels},
                    metadata={
                        "session_position": int(
                            positions[int(participant) - 1, release_index - 1]
                        ),
                        "release_index": release_index,
                        "part": part,
                        "reference": "left mastoid"
                        if participant == _LEFT_MASTOID_ONLY
                        else "average mastoids",
                        "unrecorded_samples": int(absent[start:end].sum()),
                    },
                )
            )

    def _build_sentence(self, mat: dict, index: int, sent_nr: int, words: list[str]):
        """Register one sentence and the 24 participants' observations on it."""
        n_words = len(words)
        amplitude = np.asarray(mat["ERP"][index, 0], dtype=float)
        baseline = np.asarray(mat["ERPbase"][index, 0], dtype=float)
        artefact = np.asarray(mat["artefact"][index, 0]) == 1
        reject = np.asarray(mat["reject"][index, 0]) == 1
        expected = (n_words, N_PARTICIPANTS, len(COMPONENTS))
        if (
            amplitude.shape != expected
            or baseline.shape != expected
            or artefact.shape != expected[:2]
            or reject.shape != expected[:2]
        ):
            raise ValueError(f"unexpected array shapes for sentence {index + 1}")
        if (artefact & ~reject).any():
            raise ValueError(f"sentence {index + 1}: an artefact trial is not rejected")
        log_frequency = np.asarray(mat["logwordfreq"][index, 0], dtype=float).reshape(
            -1
        )
        if log_frequency.shape != (n_words,):
            raise ValueError(f"unexpected word frequencies for sentence {index + 1}")
        stimulus = self._make_stimulus(
            words,
            f"ucl_{sent_nr}",
            {
                "sent_nr": sent_nr,
                "release_index": index + 1,
                "release_log_frequency": log_frequency.tolist(),
            },
        )
        positions = np.asarray(mat["sentence_position"])
        for participant in range(N_PARTICIPANTS):
            if self.exclusions == "reject":
                excluded = reject[:, participant]
            elif self.exclusions == "artefact":
                excluded = artefact[:, participant]
            else:
                excluded = np.zeros(n_words, dtype=bool)
            for component, name in enumerate(COMPONENTS):
                for measure, source in (
                    (name, amplitude),
                    (BASELINES[name], baseline),
                ):
                    values = source[:, participant, component].copy()
                    values[excluded] = np.nan
                    self._data.append(
                        Datum(
                            stimulus=stimulus,
                            measure_name=measure,
                            values=values,
                            participant_id=str(participant + 1),
                            # Each datum owns its metadata; nothing is shared
                            # between the twelve measures of a trial.
                            metadata={
                                "artefact": artefact[:, participant]
                                .astype(int)
                                .tolist(),
                                "reject": reject[:, participant].astype(int).tolist(),
                                "session_position": int(positions[participant, index]),
                                "exclusions": self.exclusions,
                            },
                        )
                    )

    def covariates(
        self, stimulus, target_layer: str = "words"
    ) -> dict[str, np.ndarray]:
        """Standard covariates, the release's BNC log frequency, and attached norms.

        ``release_log_frequency`` is the log relative frequency of the word in
        the British National Corpus, as the release ships it.

        Args:
            stimulus: Stimulus whose units the covariates are computed over.
            target_layer: Annotation layer naming the units to align to.
        """
        covs = super().covariates(stimulus, target_layer)
        if target_layer == "words" and "release_log_frequency" in stimulus.metadata:
            covs["release_log_frequency"] = np.asarray(
                stimulus.metadata["release_log_frequency"], dtype=float
            )
        return covs
