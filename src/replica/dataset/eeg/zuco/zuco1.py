"""ZuCo 1.0 EEG frequency-band power dataset.

The EEG view of the ZuCo1 loader: per-word band power (mean across 105
electrodes) with electrode-level annotation layers. The .mat files are those
of the eye-tracking view; the YAML's ``data_path`` points there.

Reference: https://doi.org/10.1038/sdata.2018.291
Data: https://osf.io/q3zws/
"""

from replica.dataset.contracts import DatasetInfo
from replica.dataset.eeg.zuco.base import ZuCoEEGView
from replica.dataset.eye_tracking.zuco.zuco1 import ZuCo1


class ZuCo1EEG(ZuCoEEGView):
    """ZuCo 1.0 EEG frequency-band power (Hollenstein et al., 2018).

    Args:
        task: ``"NR"`` (default), ``"SR"`` or ``"TSR"``.
        continuous: Also build each sentence's EEG and gaze recordings.

    Usage::

        ds = ZuCo1EEG(task="NR")
        ds.load("datasets/eeg/original/zuco1")
        data = ds.data(measure="eeg_theta_1")
    """

    info = DatasetInfo(name="zuco1_eeg", language="en")
    loader_cls = ZuCo1
    TASKS = ZuCo1.TASKS
