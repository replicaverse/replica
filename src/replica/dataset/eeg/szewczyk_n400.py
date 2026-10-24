"""Pooled and per-study critical-word N400 amplitudes (Szewczyk & Federmeier, 2022)."""

from replica.dataset.contracts import DatasetInfo

from ._n400 import _CriticalWordN400


class SzewczykN400(_CriticalWordN400):
    """Pooled critical-word N400 across five source studies (Szewczyk & Federmeier, 2022).

    Leave ``study`` None to load all five together (as the release packages them);
    pass one study name (the release ``dataset`` column) for a single source study.
    The five source studies are also registered as individual datasets.
    """

    info = DatasetInfo(name="szewczyk_n400", language="en", paradigm="eeg")
    _CORPUS = "S_N400"
    _MEASURE_COLS = {"n400": "time"}
    _COVARIATE_COLS = {
        "cloze": "cloze_p",
        "n400_baseline": "bline",
        "log_frequency": "logfreq",
        "concreteness": "concr",
        "orth_neighborhood": "old20",
    }
    _TSV_FILE = "szewczyk_2022.tsv"
    _TSV_MEASURES = {"n400": "n400", "n400_baseline": "bline"}
    _TSV_COVARIATES = {
        "cloze": "cloze_p",
        "log_frequency": "logfreq",
        "concreteness": "concr",
        "orth_neighborhood": "old20",
    }
    _CONDITION_COL = None

    def __init__(self, study: str | None = None, aggregate: str = "subject"):
        """Select one source study or the pooled set, and the grain.

        Args:
            study: Name from the release ``dataset`` column, restricting the
                load to that source study. ``None`` pools all five.
            aggregate: ``"subject"`` for one datum per subject per item,
                ``"mean"`` for the item-level subject average. The pooled set
                has no item-level mean, since items recur across studies with
                different subject pools, so that pairing raises ``ValueError``.
        """
        if study is None and aggregate == "mean":
            # Items recur across the five studies with different subject pools, so
            # there is no single subject-average per item to pool. Averaging across
            # studies would yield a value from none of them.
            raise ValueError(
                "the pooled mega-N400 has no single subject-average per item "
                "(items recur across studies); pass a `study` for the item-level "
                "mean, or use aggregate='subject' to pool subjects."
            )
        self.study = study
        super().__init__(aggregate=aggregate)
