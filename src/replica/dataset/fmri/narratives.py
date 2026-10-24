"""Load word-aligned BOLD data from the Narratives dataset."""

import csv
import logging
import os
from pathlib import Path

import numpy as np

from replica.dataset.base import Dataset
from replica.dataset.contracts import DatasetInfo
from replica.dataset.metadata import io as metadata_io

from ...core import (
    AnnotationLayer,
    Modality,
    Presentation,
    Recording,
    Stimulus,
    _regular_sampling_interval,
)


class Narratives(Dataset):
    """Surface-projected BOLD data and word alignments for one story.

    BOLD responses are loaded from GIfTI files.

    Args:
        story: Story identifier (default "21styear").
        alignment_word_column: Alignment CSV column used as stimulus text.
            Column 0 contains the surface transcript. Column 1 is an alternate
            aligned token stream that can contain literal ``<unk>`` values.
    """

    info = DatasetInfo(
        name="narratives",
        paradigm="fmri",
        modalities=[str(Modality.FMRI)],
        language="en",
    )
    default_tr = 1.5
    _STORY_TRIMS = {"21styear": (14, 9)}
    default_modality = Modality.FMRI
    # Participants listened to spoken stories; nothing here is read.
    default_presentation = Presentation.LISTENING

    def __init__(
        self,
        story: str = "21styear",
        alignment_word_column: int = 0,
        trim_trs: tuple[int, int] | None = None,
    ):
        """Select the story to load and which alignment column supplies its text.

        Args:
            story: Narratives task label, which names the alignment CSV, the
                report file, and the BOLD run files to match.
            alignment_word_column: 0-based column of the alignment CSV to take
                each word from. Column 0 is the surface transcript; column 1 is
                an alternate token stream that can carry literal ``<unk>``. A
                negative value raises.
            trim_trs: Start and end volumes to remove. The default is (14, 9)
                for 21styear. Other stories require an explicit pair before
                loading BOLD arrays; (0, 0) keeps the complete recording.
        """
        super().__init__()
        if alignment_word_column < 0:
            raise ValueError("alignment_word_column must be non-negative")
        if trim_trs is not None and (
            len(trim_trs) != 2
            or any(type(value) is not int or value < 0 for value in trim_trs)
        ):
            raise ValueError("trim_trs must contain two non-negative integers")
        self.trim_trs = (
            trim_trs if trim_trs is not None else self._STORY_TRIMS.get(story)
        )
        self.story = story
        self.alignment_word_column = alignment_word_column
        self.load_arrays = True
        self._wanted_participants: set[str] | None = None
        self._available_participants: list[str] = []
        self._tr_times: np.ndarray | None = None

    @property
    def participants(self) -> list[str]:
        """Subjects with data for this story, whether or not it was read."""
        return sorted(set(self._available_participants))

    def load(
        self,
        path: str,
        *,
        derivatives_path: str | None = None,
        align_path: str | None = None,
        space: str = "fsaverage6",
        max_subjects: int | None = None,
        participants: tuple[str, ...] | list[str] | None = None,
        load_arrays: bool = True,
    ) -> None:
        """Load BOLD data and word alignment.

        Args:
            path: Narratives data directory, under which the alignment and
                BOLD sources are looked up.
            derivatives_path: Path to AFNI-processed surface derivatives.
                If None, uses ``{path}/afni-nosmooth`` when that directory
                exists, and otherwise BOLD .npy files in a ``{story}/``
                subdirectory of ``path``.
            align_path: Directory containing ``{story}_align.csv`` and
                ``{story}.report``. If None, uses ``path/stimulus_resources``.
            space: Surface space (default "fsaverage6").
            max_subjects: Limit number of subjects loaded (for testing).
                Applies in sorted order, so name ``participants`` instead when
                specific subjects are wanted.
            participants: Read only these subject IDs. Others are still
                counted in :attr:`participants`.
            load_arrays: Read each subject's time course. ``False`` walks the
                cohort and fills :attr:`participants` without reading the
                arrays, for callers that only need to know who is there.

        Raises:
            FileNotFoundError: No BOLD run for the story is found.
        """
        self.load_arrays = load_arrays
        self._wanted_participants = (
            set(participants) if participants is not None else None
        )
        self._available_participants = []
        self._recordings = []
        self._load_metadata(
            path, verify=False, metadata_path=metadata_io.find_dataset_yaml(path)
        )

        # Step 1: Build stimulus from alignment CSV.
        # Resolve in this order:
        #   1. ``align_path/{story}_align.csv`` if align_path was given.
        #   2. ``path/stimulus_resources/{story}_align.csv``.
        #   3. Walk under ``path`` (max depth 4) for ``stimulus_resources/``.
        target = f"{self.story}_align.csv"
        align_file: Path | None = None

        if align_path is not None:
            candidate = Path(align_path) / target
            if candidate.exists():
                align_file = candidate

        if align_file is None:
            candidate = Path(path) / "stimulus_resources" / target
            if candidate.exists():
                align_file = candidate

        if align_file is None:
            root = Path(path)
            for sub in root.rglob("stimulus_resources"):
                # Limit depth to avoid scanning subject BOLD trees.
                if len(sub.relative_to(root).parts) > 4:
                    continue
                candidate = sub / target
                if candidate.exists():
                    align_file = candidate
                    break

        if align_file is None:
            raise FileNotFoundError(
                f"Could not locate {target!r} for story {self.story!r}. "
                f"Looked under {path!r} for a ``stimulus_resources/`` "
                "directory containing the alignment CSV. Pass "
                "``align_path=...`` if it lives elsewhere."
            )

        self._stimuli = []
        stimulus = self._load_alignment(align_file)
        self._tr_times = self._load_report_times(Path(align_file))

        # Step 2: Find and load BOLD data per subject
        default_derivatives = Path(path) / "afni-nosmooth"
        if derivatives_path is None and default_derivatives.is_dir():
            derivatives_path = str(default_derivatives)
        if derivatives_path is not None:
            self._load_bold_gifti(derivatives_path, stimulus, space, max_subjects)
        else:
            self._load_bold_npy(path, stimulus, max_subjects)
        if not self._available_participants:
            source = derivatives_path or os.path.join(path, self.story)
            raise FileNotFoundError(
                f"No BOLD run for story {self.story!r} under {source}. Pass "
                "derivatives_path= to the AFNI surface derivatives, which default "
                f"to {default_derivatives}; see `replica fetch narratives`."
            )

    def _load_alignment(self, align_file: str | Path) -> Stimulus:
        """Parse alignment CSV → Stimulus with words + word_times layers."""
        words = []
        word_times = []

        with open(align_file, newline="") as f:
            for parts in csv.reader(f):
                if len(parts) < 4:
                    continue
                if self.alignment_word_column >= len(parts):
                    raise ValueError(
                        f"alignment row has no column {self.alignment_word_column}: "
                        f"{parts!r}"
                    )
                original_word = parts[self.alignment_word_column]
                onset = float(parts[2])
                offset = float(parts[3])
                words.append(original_word)
                word_times.append((onset, offset))

        stimulus = self._make_stimulus(
            words,
            self.story,
            {
                "story": self.story,
                "alignment_word_column": self.alignment_word_column,
            },
        )
        # Add time layer using the same spans as the words layer
        time_layer = AnnotationLayer(
            "word_times", stimulus.layers["words"].spans, word_times
        )
        stimulus.add_layer(time_layer)

        return stimulus

    def _load_report_times(self, align_file: Path) -> np.ndarray | None:
        """Read acquisition trigger times when the optional report is present.

        Exact triggers are used when supplied; otherwise the loader records
        that it fell back to the nominal TR grid.
        """
        report = align_file.with_name(f"{self.story}.report")
        if not report.exists():
            return None

        trigger_times: list[float] = []
        with open(report) as file:
            for line in file:
                parts = line.split()
                if len(parts) < 2:
                    continue
                if " ".join(parts[1:]) in {"init-trigger", "trigger"}:
                    trigger_times.append(float(parts[0]))

        # Repair a single missing trigger when one interval is unusually long.
        repaired = list(trigger_times)
        intervals = np.diff(trigger_times)
        if len(intervals):
            for index in np.flatnonzero(intervals > intervals.mean() * 1.5):
                repaired.insert(index + 1, trigger_times[index] + self.default_tr)
        return np.asarray(repaired, dtype=float)

    def _timing_metadata(self, n_trs: int) -> tuple[np.ndarray | None, dict]:
        if self._tr_times is not None and len(self._tr_times) == n_trs:
            return self._tr_times, {"timing_source": "acquisition-report"}
        return None, {
            "timing_source": "nominal-tr",
            "timing_report_available": self._tr_times is not None,
        }

    def _trim_and_store_bold(
        self,
        bold: np.ndarray,
        stimulus: Stimulus,
        participant_id: str,
        metadata: dict | None = None,
        sample_times: np.ndarray | None = None,
    ) -> bool:
        """Trim one BOLD run and store it as a native recording."""
        if self.trim_trs is None:
            raise ValueError(
                f"No verified trimming for Narratives story {self.story!r}; set trim_trs"
            )
        trim_start, trim_end = self.trim_trs
        if bold.ndim != 2:
            raise ValueError("BOLD arrays must have dimensions (time, vertex)")
        if trim_start + trim_end >= bold.shape[0]:
            raise ValueError("Narratives trimming would remove the entire BOLD run")

        end = -trim_end if trim_end > 0 else bold.shape[0]
        trimmed = bold[trim_start:end]
        trimmed_times = None
        interval = self.default_tr
        if sample_times is not None:
            sample_times = np.asarray(sample_times)
            if len(sample_times) != bold.shape[0]:
                raise ValueError(
                    "sample_times length must match the untrimmed BOLD array"
                )
            trimmed_times = sample_times[trim_start:end]
            interval = _regular_sampling_interval(trimmed_times)
            if (
                interval is not None
                and _regular_sampling_interval(trimmed_times, self.default_tr)
                is not None
            ):
                interval = self.default_tr

        self.add_recording(
            Recording(
                stimulus=stimulus,
                values=trimmed,
                measure_name="bold",
                modality=Modality.FMRI,
                participant_id=participant_id,
                sample_interval=interval,
                start_time=trim_start * self.default_tr,
                sample_times=trimmed_times,
                dimension_names=("time", "vertex"),
                metadata={
                    "trim_start_samples": trim_start,
                    "trim_end_samples": trim_end,
                    **(metadata or {}),
                },
            )
        )
        return True

    def _load_bold_gifti(
        self,
        derivatives_path: str,
        stimulus: Stimulus,
        space: str,
        max_subjects: int | None,
    ) -> None:
        """Load GIfTI surface BOLD from BIDS derivatives."""
        import nibabel as nib

        # Scan for subject directories that have this story
        bids_root = Path(derivatives_path)
        subject_dirs = sorted(
            d for d in bids_root.iterdir() if d.is_dir() and d.name.startswith("sub-")
        )

        loaded = 0
        for sub_dir in subject_dirs:
            sub_id = sub_dir.name
            func_dir = sub_dir / "func"
            if not func_dir.exists():
                continue

            # Find GIfTI files for this story
            lh_files = sorted(
                func_dir.glob(
                    f"*task-{self.story}*space-{space}_hemi-L_desc-clean.func.gii"
                )
            )
            rh_files = sorted(
                func_dir.glob(
                    f"*task-{self.story}*space-{space}_hemi-R_desc-clean.func.gii"
                )
            )

            if not lh_files or not rh_files:
                if lh_files or rh_files:
                    logging.getLogger("replica").warning(
                        "narratives: missing hemisphere for %s, story %s; skipping run",
                        sub_id,
                        self.story,
                    )
                continue

            # Recorded before the arrays are read: counting the cohort must not
            # require loading the large per-subject time courses.
            self._available_participants.append(sub_id)
            if not self.load_arrays or (
                self._wanted_participants is not None
                and sub_id not in self._wanted_participants
            ):
                continue

            if max_subjects is not None and loaded >= max_subjects:
                continue

            # Load and concatenate hemispheres
            # Use darrays approach (matching reference): each darray is one TR
            try:
                lh_gii = nib.load(str(lh_files[0]))
                rh_gii = nib.load(str(rh_files[0]))
            except Exception as exc:
                # Truncated/partial downloads; skip the run but say so.
                logging.getLogger("replica").warning(
                    "narratives: unreadable GIFTI for %s (%s); skipping run",
                    sub_id,
                    exc,
                )
                continue
            lh_data = np.array(
                [d.data for d in lh_gii.darrays]
            )  # (n_TRs, n_vertices_L)
            rh_data = np.array(
                [d.data for d in rh_gii.darrays]
            )  # (n_TRs, n_vertices_R)

            bold = np.concatenate([lh_data, rh_data], axis=1)
            sample_times, timing = self._timing_metadata(bold.shape[0])
            if self._trim_and_store_bold(
                bold,
                stimulus,
                sub_id,
                {"space": space, **timing},
                sample_times=sample_times,
            ):
                loaded += 1

    def _load_bold_npy(
        self,
        path: str,
        stimulus: Stimulus,
        max_subjects: int | None,
    ) -> None:
        """Load pre-extracted BOLD from .npy files."""
        bold_dir = os.path.join(path, self.story)
        if not os.path.isdir(bold_dir):
            return

        npy_files = sorted(
            f
            for f in os.listdir(bold_dir)
            if f.endswith(".npy") and f.startswith("sub_")
        )

        loaded = 0
        for npy_file in npy_files:
            sub_id = npy_file.replace(".npy", "")
            self._available_participants.append(sub_id)
            if not self.load_arrays or (
                self._wanted_participants is not None
                and sub_id not in self._wanted_participants
            ):
                continue
            if max_subjects is not None and loaded >= max_subjects:
                continue
            bold = np.load(os.path.join(bold_dir, npy_file))
            sample_times, timing = self._timing_metadata(bold.shape[0])
            if self._trim_and_store_bold(
                bold,
                stimulus,
                sub_id,
                timing,
                sample_times=sample_times,
            ):
                loaded += 1
