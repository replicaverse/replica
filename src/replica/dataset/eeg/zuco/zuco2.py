"""ZuCo 2.0 EEG frequency-band power dataset.

The EEG view of the ZuCo2 loader: per-word band power (mean across 105
electrodes) with electrode-level annotation layers. The .mat files are those
of the eye-tracking view; the YAML's ``data_path`` points there.

Reference: https://aclanthology.org/2020.lrec-1.18
Data: https://osf.io/2urht/
"""

from replica.dataset.contracts import DatasetInfo
from replica.dataset.eeg.zuco.base import ZuCoEEGView
from replica.dataset.eye_tracking.zuco.zuco2 import ZuCo2


class ZuCo2EEG(ZuCoEEGView):
    """ZuCo 2.0 EEG frequency-band power (Hollenstein et al., 2020).

    Args:
        task: ``"NR"`` (default) or ``"TSR"``.
        continuous: Also build each sentence's EEG and gaze recordings.

    Usage::

        ds = ZuCo2EEG(task="NR")
        ds.load("datasets/eeg/original/zuco2")
        data = ds.data(measure="eeg_theta_1")
    """

    info = DatasetInfo(name="zuco2_eeg", language="en")
    loader_cls = ZuCo2
    TASKS = ZuCo2.TASKS
