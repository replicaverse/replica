"""Item-level predictability norms of de Varda et al. (2023) as covariates.

de Varda et al. (2023) collected cloze completions and predictability ratings
for every word after the first of the 205 UCL sentences that Frank et al.
(2013, 2015) recorded eye movements, self-paced reading and ERPs on, and
released them next to participant means of those responses. The norms are the
new content of that release. This module reads them off the aligned table and
attaches them, by exact sentence text, to the stimuli of any loader that
carries those sentences at the participant level.
"""

import csv
from pathlib import Path

import numpy as np

from replica.core import Stimulus
from replica.dataset.base import ReadingTimeDataset

#: Covariate name -> column of the aligned table. ``cloze`` is the smoothed
#: cloze probability and ``cloze_surprisal`` its negative natural log, as the
#: release stores them; the rating columns summarise the predictability
#: ratings; ``cloze_entropy`` and ``cloze_competition`` are the release's own
#: distributional summaries of the cloze responses.
COVARIATE_COLUMNS: dict[str, str] = {
    "cloze": "cloze_p_smoothed",
    "cloze_surprisal": "cloze_s",
    "predictability_rating": "rating_mean",
    "predictability_rating_sd": "rating_sd",
    "cloze_entropy": "entropy",
    "cloze_competition": "competition",
}

#: The official release file, and the local alias accepted for the same table;
#: the official name wins when both are present.
DATA_FILES = ("all_measures.csv", "aligned.csv")

_REQUIRED_COLUMNS = ("sentence", "word", "context_length", *COVARIATE_COLUMNS.values())

#: Cell contents read as a missing norm.
_MISSING = frozenset({"", "NA"})


def find_devarda_table(path: str | Path) -> Path:
    """Locate the aligned table under a directory, or accept the file itself.

    Args:
        path: The de Varda data directory, or the table's own path.

    Raises:
        FileNotFoundError: Neither release file name is present.
    """
    candidate = Path(path)
    if candidate.is_file():
        return candidate
    for name in DATA_FILES:
        if (candidate / name).exists():
            return candidate / name
    raise FileNotFoundError(
        f"none of {DATA_FILES} found in {candidate}. Source (official): "
        "https://github.com/Andrea-de-Varda/prediction-resource (all_measures.csv)."
    )


def has_norm_columns(path: str | Path) -> bool:
    """Whether the table carries the norm columns, as the release does.

    Args:
        path: The de Varda data directory, or the table's own path.
    """
    with open(find_devarda_table(path), newline="") as handle:
        fieldnames = next(csv.reader(handle), [])
    return set(_REQUIRED_COLUMNS) <= set(fieldnames)


def read_devarda_norms(path: str | Path) -> dict[str, dict[int, dict[str, float]]]:
    """Read the norms keyed by sentence text and zero-based word position.

    Each row of the aligned table is one test word. The row's ``word`` must be
    the word at ``context_length`` of its whitespace-split ``sentence``, and no
    position may occur twice for one sentence. An empty cell or ``NA`` is a
    missing norm.

    Args:
        path: The de Varda data directory, or the table's own path.

    Returns:
        ``sentence text -> word position -> covariate name -> value``, with the
        text normalised to single spaces.

    Raises:
        ValueError: The table lacks a required column, or a row's identity is
            inconsistent or repeated.
    """
    table = find_devarda_table(path)
    norms: dict[str, dict[int, dict[str, float]]] = {}
    with open(table, newline="") as handle:
        reader = csv.DictReader(handle)
        missing = set(_REQUIRED_COLUMNS) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{table.name}: missing columns {sorted(missing)}")
        for row in reader:
            text = " ".join(row["sentence"].split())
            words = text.split()
            position = int(row["context_length"])
            if not 0 <= position < len(words) or words[position] != row["word"]:
                raise ValueError(
                    f"{table.name}: word {row['word']!r} is not at position "
                    f"{position} of {text!r}"
                )
            positions = norms.setdefault(text, {})
            if position in positions:
                raise ValueError(
                    f"{table.name}: repeated position {position} of {text!r}"
                )
            positions[position] = {
                name: np.nan if row[column] in _MISSING else float(row[column])
                for name, column in COVARIATE_COLUMNS.items()
            }
    return norms


class DeVardaCovariates(ReadingTimeDataset):
    """Loader base that can carry de Varda et al.'s norms on its stimuli.

    Nothing is attached until :meth:`attach_predictability_norms` runs; the loaders that
    share the UCL sentences (``UCL_ET``, ``UCL_SPR``, ``Frank2015ERP``) leave
    that to the caller, and ``DeVarda2023`` attaches its own table on load.
    """

    def __init__(self) -> None:
        """Start without norms."""
        super().__init__()
        self._reset_devarda()

    def _reset_devarda(self) -> None:
        """Forget attached norms; a loader calls this when it reloads."""
        # stimulus_id -> covariate name -> per-word array; merged in covariates().
        self._devarda: dict[str, dict[str, np.ndarray]] = {}
        self._devarda_attached = False
        #: Sentences of the table that match no stimulus text, after attaching.
        self.devarda_unmatched: list[str] = []

    def attach_predictability_norms(self, path: str | Path | None = None) -> int:
        """Attach de Varda cloze and predictability norms by sentence text.

        Matching is by exact text, so a stimulus whose displayed text differs
        from the UCL sentence (a release typo, say) receives NaN throughout, and
        the unmatched sentences are listed on ``devarda_unmatched``. Positions
        the table does not norm, such as every sentence-initial word, are NaN.
        Attaching again replaces the previous norms.

        Args:
            path: The de Varda data directory or table; ``None`` uses the
                default de Varda data location.

        Returns:
            The number of stimuli that received norms.

        Raises:
            FileNotFoundError: The table is absent from the resolved location.
            ValueError: The table's identities are inconsistent.
        """
        if path is None:
            from replica.dataset.locations import DEVARDA2023_SUBDIR, dataset_path

            path = dataset_path(DEVARDA2023_SUBDIR)
        norms = read_devarda_norms(path)
        matched: set[str] = set()
        self._devarda = {}
        for stimulus in self._stimuli:
            text = " ".join(stimulus.text.split())
            positions = norms.get(text)
            if positions is None:
                continue
            matched.add(text)
            n_units = len(stimulus.layers["words"])
            arrays = {name: np.full(n_units, np.nan) for name in COVARIATE_COLUMNS}
            for position, values in positions.items():
                for name, value in values.items():
                    arrays[name][position] = value
            self._devarda[stimulus.stimulus_id] = arrays
        self._devarda_attached = True
        self.devarda_unmatched = sorted(set(norms) - matched)
        return len(self._devarda)

    def covariates(
        self, stimulus: Stimulus, target_layer: str = "words"
    ) -> dict[str, np.ndarray]:
        """Standard covariates plus the norms, once they are attached.

        Every stimulus then carries the six norm arrays, NaN wherever the
        table norms nothing, so the covariate schema is the same across the
        dataset.

        Args:
            stimulus: Stimulus whose units the covariates are computed over.
            target_layer: Annotation layer naming the units to align to.
        """
        covs = super().covariates(stimulus, target_layer)
        if not self._devarda_attached or target_layer != "words":
            return covs
        attached = self._devarda.get(stimulus.stimulus_id)
        n_units = len(stimulus.layers[target_layer])
        for name in COVARIATE_COLUMNS:
            covs[name] = attached[name].copy() if attached else np.full(n_units, np.nan)
        return covs
