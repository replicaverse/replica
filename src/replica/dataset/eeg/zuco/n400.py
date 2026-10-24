"""ZuCo ERP amplitudes from the Kuribayashi et al. (2025) release."""

from pathlib import Path

import pandas as pd

from replica.core import Presentation
from replica.dataset.base import Datum
from replica.dataset.contracts import DatasetInfo
from replica.dataset.eeg._n400 import N400Dataset


class ZuCoN400(N400Dataset):
    """ZuCo (Hollenstein et al., 2018) N400/P600/ANT amplitudes.

    Full sentences with a subject-averaged amplitude on every word (the release
    has no per-subject EEG for ZuCo), plus the per-component ERP baselines and
    log frequency as covariates.
    """

    info = DatasetInfo(name="zuco_n400", language="en", paradigm="eeg")
    _CORPUS = "ZuCO"
    default_presentation = Presentation.NATURAL_READING
    _MEASURE_COLS = {"n400": "N400", "p600": "P600", "ant": "ANT"}
    # Averaging intervals in milliseconds from the source epoch onset, already applied
    # by the source preprocessing; the loader reads the resulting amplitudes.
    _ERP_WINDOWS_MS = {"n400": (300, 500), "p600": (600, 700), "ant": (200, 400)}
    _COVARIATE_COLS = {
        "n400_baseline": "N400_baseline",
        "p600_baseline": "P600_baseline",
        "ant_baseline": "ANT_baseline",
        "log_frequency": "log_gmean_freq",
    }
    AGGREGATES = ("mean",)

    def __init__(self):
        """Load the subject-averaged grain, the only one ZuCo releases."""
        super().__init__(aggregate="mean")

    def _build(self, ann: pd.DataFrame, ann_path: Path) -> None:
        """One Stimulus per sentence, an amplitude on every word."""
        for (article, sent_id), grp in ann.groupby(["article", "sent_id"]):
            grp = grp.sort_values("tokenN_in_sent")
            words = [str(w) for w in grp["word"]]
            stim = self._make_stimulus(
                words, f"{article}_{sent_id}", {"article": article, "sent_id": sent_id}
            )
            for measure, col in self._MEASURE_COLS.items():
                self._data.append(
                    Datum(
                        stimulus=stim,
                        measure_name=measure,
                        values=grp[col].to_numpy(dtype=float),
                        participant_id="averaged",
                        metadata={
                            "source_column": col,
                            "window_ms": self._ERP_WINDOWS_MS[measure],
                            "channels": {
                                "n400": "central",
                                "p600": "posterior",
                                "ant": "anterior",
                            }[measure],
                        },
                    )
                )
            self._covariates[stim.stimulus_id] = {
                name: grp[col].to_numpy(dtype=float)
                for name, col in self._COVARIATE_COLS.items()
            }
