"""Check duration units against the released eye-tracker timestamps."""

from pathlib import Path

import numpy as np
import pytest

from replica.dataset.eye_tracking.zuco.zuco1 import MEASURE_COLUMNS, ZuCo1
from replica.dataset.eye_tracking.zuco.zuco2 import ZuCo2

pytestmark = [pytest.mark.needs_data, pytest.mark.slow]


@pytest.mark.parametrize(
    "dataset_class,filename",
    [(ZuCo1, "resultsZAB_NR.mat"), (ZuCo2, "resultsYAC_NR.mat")],
)
def test_durations_use_timestamp_units(data_root, dataset_class, filename):
    root = Path(data_root) / "eye-tracking/original" / dataset_class.info.name
    path = root / filename
    if not path.exists():
        pytest.skip(f"{filename} unavailable")

    if dataset_class is ZuCo1:
        from scipy.io import loadmat

        source = loadmat(path, struct_as_record=False, squeeze_me=True)
        word = source["sentenceData"][0].word[1]
        raw_et = np.asarray(word.rawET)
        if raw_et.dtype == object:
            raw_et = np.asarray(raw_et.ravel()[0])
        timestamps = raw_et[0].copy()
        released = {key: float(getattr(word, key)) for key in MEASURE_COLUMNS}
        del source, word, raw_et
    else:
        import h5py

        with h5py.File(path) as source:
            words = source[source["sentenceData"]["word"][0, 0]]
            raw_et = source[words["rawET"][1, 0]]
            timestamps = np.asarray(source[raw_et[0, 0]])[:, 0]
            released = {
                key: float(np.asarray(source[words[key][1, 0]]).item())
                for key in MEASURE_COLUMNS
            }

    # The extractor takes both endpoints of the EEG-indexed fixation interval.
    assert len(timestamps) - 1 == released["FFD"]
    sample_ms = float(np.median(np.diff(timestamps)))
    assert sample_ms == pytest.approx(2.0, abs=0.001)

    dataset = dataset_class()
    load_options = {"load_eeg": False} if dataset_class is ZuCo2 else {}
    dataset.load(str(root), max_subjects=1, **load_options)
    for source_column, measure in MEASURE_COLUMNS.items():
        datum = dataset.data(measure)[0]
        assert datum.values[1] == pytest.approx(
            released[source_column] * sample_ms, abs=0.01
        )
        assert datum.metadata["source_measure"] == source_column
        assert datum.metadata["source_unit"] == "samples"
        assert datum.metadata["source_sample_rate_hz"] == 500
        assert datum.metadata["released_values"][1] == released[source_column]
