"""Brown SPR Corpus (Smith & Levy, 2013).

Self-paced reading times for passages from the Brown corpus of American
English. The original release is private; the only public source is the
Shain et al. (2024) preprocessed release on OSF (project 6wvqe), where the corpus
ships in CDR format (one big ``brown_X.csv`` of predictors per
(subject, word) row plus train/dev/test ``y`` splits).

The loader reads ``brown_X.csv`` directly (it already contains the DV
``fdur`` along with all metadata), builds one Stimulus per ``sentid``
(words in ``sentpos`` order), and one Datum per (subject, sentence) pair.

Reference (corpus): Smith, N. J., & Levy, R. (2013). The effect of word
    predictability on reading time is logarithmic. *Cognition*.
    https://doi.org/10.1016/j.cognition.2013.02.013
Reference (preprocessing): Shain et al. (2024).
    https://osf.io/6wvqe/
"""

from pathlib import Path

import pandas as pd

from replica.dataset.base import SPRDataset, TabularDataset, TabularDatasetConfig
from replica.dataset.contracts import DatasetInfo
from replica.dataset.metadata import io as metadata_io

DATA_FILE = "brown_X.csv"


class BrownSPR(SPRDataset):
    """Brown SPR corpus (Smith & Levy, 2013).

    Loader reads the Shain et al. (2024) preprocessed release of the corpus
    from OSF project 6wvqe. Run
    ``replica fetch brown_spr_shain2024logarithmic``
    to download the four CSVs into
    ``datasets/self-paced-reading/processed/brown_spr_shain2024logarithmic/``.

    Default measure is ``spr_rt`` (per-word self-paced reading time, in ms),
    read from the release's ``fdur`` column.
    Each row of the upstream ``brown_X.csv`` becomes one (participant, word)
    observation; sentences are grouped by ``sentid`` and ordered by
    ``sentpos``.
    """

    info = DatasetInfo(name="brown_spr", language="en")
    _declared_measures = ("spr_rt",)
    additive_measures = frozenset({"spr_rt"})

    def __init__(self) -> None:
        """Select per-word self-paced reading time as the default measure."""
        super().__init__()
        self.default_measure = "spr_rt"

    def load(self, path: str) -> None:
        """Load Brown SPR from ``path/brown_X.csv``, the Shain et al. (2024) format.

        Args:
            path: Directory holding the release's CSVs.
        """
        self._load_metadata(path, metadata_path=metadata_io.find_dataset_yaml(path))
        csv_path = Path(path) / DATA_FILE
        if not csv_path.exists():
            raise FileNotFoundError(
                f"BrownSPR: {DATA_FILE} not found at {csv_path}.\n"
                "The Brown SPR corpus is only publicly distributed as part of "
                "Shain et al. (2024)'s OSF release.\n"
                "Run: replica fetch "
                "brown_spr_shain2024logarithmic"
            )
        self._load_from_csv(csv_path)

    def _load_from_csv(self, csv_path: Path) -> None:
        """Parse the X CSV via the TabularDataset."""

        # ``X.csv`` is space-separated despite its ``.csv`` extension, like
        # the rest of the OSF release.
        df = pd.read_csv(csv_path, sep=r"\s+", engine="python")

        # Reuse the tabular dataset infrastructure: each ``sentid`` is
        # one trial (Stimulus); each (subject, sentid) pair contributes one
        # Datum aligned to the ``sentpos``-ordered word list.
        config = TabularDatasetConfig(
            name="brown_spr",
            language="en",
            text_col="word",
            trial_col="sentid",
            unit_index_col="sentpos",
            measure_col="fdur",
            measure_name="spr_rt",
            participant_col="subject",
        )
        adapter = TabularDataset(df, config)

        # Inherit the constructed stimuli + data into self.
        self._stimuli = adapter._stimuli
        self._data = adapter._data

        # A trial is one sentence, but the corpus spans documents and the
        # release says which: without ``docid`` a context-dependent predictor
        # sees every sentence as passage-initial.
        if "docid" in df.columns:
            doc_of = dict(zip(df["sentid"].tolist(), df["docid"].tolist(), strict=True))
            for stimulus in self._stimuli:
                sentid = stimulus.metadata.get("trial_id")
                if sentid in doc_of:
                    stimulus.metadata["docid"] = doc_of[sentid]
