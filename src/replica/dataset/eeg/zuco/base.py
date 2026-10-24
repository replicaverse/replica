"""Shared EEG band-power view over the ZuCo eye-tracking loaders."""

from collections.abc import Collection

from replica.core import Presentation
from replica.dataset.base import EEGDataset
from replica.dataset.eye_tracking.zuco.base import ZuCoTaskDataset
from replica.dataset.metadata import io as metadata_io


class ZuCoEEGView(EEGDataset):
    """Per-word EEG band power of one ZuCo release.

    Loads through the release's eye-tracking loader and keeps the EEG measures:
    mean power over 105 electrodes in eight bands per word, with electrode-level
    layers ``eeg_<band>_<reader>`` on each stimulus.

    Args:
        task: Reading task, one of the loader's ``TASKS``.
        continuous: Also build each sentence's EEG and gaze recordings; see
            :mod:`replica.dataset.eye_tracking.zuco.continuous`.
    """

    #: Eye-tracking loader of the release.
    loader_cls: type[ZuCoTaskDataset]
    default_measure = "eeg_theta_1"
    default_presentation = Presentation.NATURAL_READING

    def __init__(self, task: str = "NR", continuous: bool = False):
        """Wrap the release's eye-tracking loader for one reading task."""
        super().__init__()
        self.task = task
        self._loader = self.loader_cls(task=task, continuous=continuous)

    def load(
        self,
        path: str,
        *,
        max_subjects: int | None = None,
        participants: Collection[str] | None = None,
    ) -> None:
        """Load the EEG view.

        Args:
            path: Dataset directory; its YAML's ``data_path`` names the .mat
                directory, else the files are expected in *path*.
            max_subjects: Subject cap, for tests.
            participants: Subjects to read, named as in the file names; ``None``
                reads all.
        """
        metadata_path = metadata_io.find_dataset_yaml(path)
        self._load_metadata(path, verify=False, metadata_path=metadata_path)

        # Resolve the directory containing the .mat files
        mat_dir = str(metadata_io.resolve_data_dir(path, metadata_path=metadata_path))

        self._loader.load(
            mat_dir,
            max_subjects=max_subjects,
            participants=participants,
            load_eeg=True,
        )

        # Share stimuli (includes EEG annotation layers) and any recordings
        self._stimuli = self._loader._stimuli
        self._recordings = list(self._loader._recordings)

        # Keep only EEG measures
        self._data = [
            d for d in self._loader._data if d.measure_name.startswith("eeg_")
        ]
