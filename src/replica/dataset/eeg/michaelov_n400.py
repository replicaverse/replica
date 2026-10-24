"""Critical-word N400 amplitudes from Michaelov et al. (2024)."""

from replica.dataset.contracts import DatasetInfo

from ._n400 import _CriticalWordN400


class MichaelovN400(_CriticalWordN400):
    """Michaelov et al. (2024) N400: critical-word centroparietal cluster-mean N400.

    ``aggregate="electrode"`` keeps each of the nine centroparietal electrodes as
    its own datum so the release's electrode random effect is reproducible.
    """

    info = DatasetInfo(name="michaelov_n400", language="en", paradigm="eeg")
    _CORPUS = "M_N400"
    _MEASURE_COLS = {"n400": "time"}
    _COVARIATE_COLS = {
        "cloze": "Cloze",
        "log_frequency": "ZipfFrequency",
        "orth_neighborhood": "ON",
        "plausibility": "PlausibilityJudgement",
    }
    _TSV_FILE = "michaelov_2024.tsv"
    _TSV_MEASURES = {"n400": "N400"}
    _TSV_COVARIATES = {
        "cloze": "Cloze",
        "log_frequency": "ZipfFrequency",
        "orth_neighborhood": "ON",
        "plausibility": "PlausibilityJudgement",
    }
    _CONDITION_COL = "Condition"
    AGGREGATES = ("subject", "mean", "electrode")
