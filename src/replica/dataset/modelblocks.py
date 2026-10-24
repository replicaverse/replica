"""Dataset adaptor for ModelBlocks ``.evmeasures`` reading-time corpora.

The companion to :mod:`replica.feature.modelblocks` (which exposes ModelBlocks
*predictor* columns as Features): this module turns a reading-time *response*
table from a ModelBlocks checkout
(https://github.com/modelblocks/modelblocks-release) into a first-class REPLICA
:class:`~replica.dataset.base.ReadingTimeDataset`.

A ``.evmeasures`` file is a space-delimited per-event table, one row per
``(subject, word)``, produced by the ModelBlocks corpus pipelines
(``process_naturalstories.py``/``process_dundee.py``/``process_provo.py``/...).
Relevant columns:

* ``word``, the token
* ``subject``, participant id
* ``docid`` (or ``discid``), document id; the stimulus grouping
* ``sentid`` / ``sentpos``, sentence id and 1-based within-sentence position;
  together with ``docid`` they order words within a document
* the DV: ``fdur`` (self-paced reading) or ``fdurFP`` / ``fdurGP``
  (eye-tracking first-pass / go-past durations)
* any predictor columns already joined into the table (``wlen``, ``sentpos``,
  ``unigram``, ``totsurp``, ``totsurpS1``, ``prevwasfix``, ``wdelta``, ...)

For an analysis run through ModelBlocks (e.g. the Clark et al. (2025)
reproduction), consuming the ``.evmeasures`` directly gives data, tokenization,
and spillover from the published pipeline. Study-specific censoring must still
be applied by the caller.

``ModelBlocksDataset`` exposes:

* the standard :class:`ReadingTimeDataset` API (``stimuli()``, ``data()``)
  with the DV as a per-word measure;
* :meth:`predictor_model`, a
  :class:`~replica.feature.core.composition.models.PrecomputedFeatureModel` carrying the
  table's stimulus-level predictor columns, one value per word, so the
  ModelBlocks predictors can be used as REPLICA Features without recomputation.
"""

from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from replica.core import Datum, resolve_dv
from replica.dataset.base import ReadingTimeDataset
from replica.dataset.contracts import DatasetInfo
from replica.dataset.loading.utils import conditional_first_run
from replica.dataset.measures import FIRST_PASS_RUN

_MISSING_TOKENS: frozenset[str] = frozenset({"", "NA", "NaN", "nan", "."})
# "null"/"none"/"None" are deliberately not missing markers: they occur as real
# corpus words (GECO "null and void", Dundee/GECO "none") and numeric columns
# coerce them to NaN regardless via pd.to_numeric(errors="coerce").

# Canonical DV label per ModelBlocks DV column, so REPLICA's measure vocabulary
# (``measures.CANONICAL_RT_MEASURES``) stays consistent across loaders. ``fdurFP``
# holds the first run of a word reached in first pass only, so it is stored as
# ``first_run`` with the ``first_pass`` flag and analysed under the
# conditional label.
DV_MEASURE_NAMES: dict[str, str] = {
    "fdur": "spr_rt",
    "fdurFP": FIRST_PASS_RUN,
    "fdurGP": "go_past_in_first_pass",
    "fdurTT": "total_time",
}
# fdurSPsummed is a source-specific eye-tracking run measure. In Provo it
# includes inter-fixation time; retain its source name instead of calling it SPR.

_DOCID_CANDIDATES = ("docid", "discid")


def read_evmeasures(path: str | Path) -> pd.DataFrame:
    """Read a ModelBlocks ``.evmeasures`` table into a DataFrame.

    Space-delimited, single header row. Missing tokens (``NaN``, ``.``, …)
    become ``NaN`` in numeric columns at use time (kept as strings here).

    Args:
        path: The ``.evmeasures`` file to read.

    Raises:
        ValueError: When the file has a header but no rows.
    """
    path = Path(path)
    df = pd.read_csv(path, sep=r"\s+", header=0, dtype=str, na_filter=False)
    if df.empty:
        raise ValueError(f"empty .evmeasures file: {path}")
    return df


def _to_float(series: pd.Series) -> np.ndarray:
    """Coerce a string column to float, mapping ModelBlocks missing tokens to NaN."""
    out = series.where(~series.isin(_MISSING_TOKENS), other=np.nan)
    return pd.to_numeric(out, errors="coerce").to_numpy(dtype=float)


def _scatter_first(n_units: int, pos: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Word-aligned array taking the first non-NaN value at each position.

    Eye-tracking ``.evmeasures`` are per-fixation: a refixated word has several
    rows (in reading order) with the first-pass measure (``fdurFP``/``fdurGP``)
    on the first row and ``NaN`` on the rest. A plain ``arr[pos]=values`` scatter
    is last-wins → it would overwrite the real value with the trailing ``NaN``
    and drop refixated (typically longer) words, biasing the sample. Assigning
    the valid entries in reverse makes the earliest non-NaN row win.
    """
    out = np.full(n_units, np.nan)
    valid = ~np.isnan(values)
    out[pos[valid][::-1]] = values[valid][::-1]
    return out


class ModelBlocksDataset(ReadingTimeDataset):
    """Reading-time dataset built from a ModelBlocks ``.evmeasures`` table.

    One :class:`~replica.core.Stimulus` per ``docid`` (its words ordered
    by ``(sentid, sentpos)``); one :class:`~replica.core.Datum` per
    ``(docid, subject)`` carrying the DV aligned to that document's word
    sequence (``NaN`` where a subject has no row for a word, e.g. unfixated
    words in eye-tracking).

    Each stimulus carries ``metadata['per_word_sentence_ids']`` (globally
    unique integer sentence ids, used by REPLICA's
    ``SubjectSentenceMod10Splitter``) and ``metadata['docid']``.

    Args:
        path: path to the ``.evmeasures`` file (or pass ``dataframe=``).
        dataframe: an already-read table to use in place of *path*; its cells
            are coerced to strings, as :func:`read_evmeasures` leaves them.
        dv_column: ModelBlocks DV column, e.g. ``"fdur"`` (SPR) or
            ``"fdurFP"`` / ``"fdurGP"`` (eye-tracking).
        name: dataset name (defaults to the file stem).
        language: ISO code (default ``"en"``).
        predictor_columns: numeric columns to expose via
            :meth:`predictor_model`; ``None`` (default) auto-selects every
            numeric, stimulus-level column (constant across subjects within a
            ``(docid, sentid, sentpos)`` key).
        datum_columns: per-subject columns (e.g. ``"prevwasfix"``) exposed as
            extra measures, one ``Datum`` per ``(doc, subject)``, for use as
            per-observation controls via ``dataset.feature(...)``.
        subject_col: column holding the participant id.
        word_col: column holding the word form.
        sentid_col: column holding the sentence id.
        sentpos_col: column holding the position within the sentence.
        docid_col: document-id column; ``None`` auto-detects ``docid`` then
            ``discid``, falling back to a single synthetic document.
    """

    _declared_measures: tuple[str, ...] = ()

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        dv_column: str,
        dataframe: pd.DataFrame | None = None,
        name: str | None = None,
        language: str = "en",
        predictor_columns: Sequence[str] | None = None,
        datum_columns: Sequence[str] | None = None,
        subject_col: str = "subject",
        word_col: str = "word",
        sentid_col: str = "sentid",
        sentpos_col: str = "sentpos",
        docid_col: str | None = None,
    ):
        """Read the table and build the stimuli, datums, and predictor arrays.

        The class docstring describes the parameters.
        """
        super().__init__()
        if dataframe is None:
            if path is None:
                raise ValueError("pass either path= or dataframe=")
            df = read_evmeasures(path)
            stem = Path(path).stem
        else:
            df = dataframe.astype(str)
            stem = name or "modelblocks"

        self._dv_column = dv_column
        self.default_measure = DV_MEASURE_NAMES.get(dv_column, dv_column)
        self._dv_measure, self._dv_condition = (
            (self.default_measure, None)
            if dv_column == "fdurGP"
            else resolve_dv(self.default_measure)
        )
        self._declared_measures = (self._dv_measure,) + (
            (self._dv_condition,) if self._dv_condition else ()
        )
        self.additive_measures = (
            frozenset({self._dv_measure})
            if self._dv_measure in ("spr_rt", "total_time")
            else frozenset()
        )

        for col in (dv_column, subject_col, word_col, sentid_col, sentpos_col):
            if col not in df.columns:
                raise KeyError(
                    f"{stem}: required column {col!r} not in .evmeasures "
                    f"(have: {list(df.columns)})"
                )

        if docid_col is None:
            docid_col = next((c for c in _DOCID_CANDIDATES if c in df.columns), None)
        self._docid_col = docid_col

        self.info = DatasetInfo(name=name or stem, language=language)

        self._subject_col = subject_col
        self._word_col = word_col
        self._sentid_col = sentid_col
        self._sentpos_col = sentpos_col

        if predictor_columns is None:
            predictor_columns = self._auto_predictor_columns(df)
        self._predictor_columns = list(predictor_columns)

        # Per-subject (datum-level) columns, e.g. ``prevwasfix``, exposed as
        # extra measures (one Datum per (doc, subject)) for use as
        # per-observation controls via dataset.feature(...).
        self._datum_columns = list(datum_columns or [])
        for col in self._datum_columns:
            if col not in df.columns:
                raise KeyError(f"{stem}: datum column {col!r} not in .evmeasures")

        # stimulus_id -> {column -> per-word float array}; filled by _build.
        self._predictor_arrays: dict[str, dict[str, np.ndarray]] = {}

        self._build(df)

    # ------------------------------------------------------------------ build

    def _auto_predictor_columns(self, df: pd.DataFrame) -> list[str]:
        """Numeric columns that are constant within each (doc,sentid,sentpos)."""
        skip = {
            self._subject_col,
            self._word_col,
            self._dv_column,
        }
        key_cols = [self._sentid_col, self._sentpos_col]
        if self._docid_col:
            key_cols.append(self._docid_col)
        cols: list[str] = []
        for c in df.columns:
            if c in skip or c in key_cols:
                continue
            vals = _to_float(df[c])
            if np.all(np.isnan(vals)):
                continue  # non-numeric / all-missing
            # stimulus-level iff constant within each key
            nun = df.groupby(key_cols, sort=False)[c].nunique(dropna=False)
            if int(nun.max()) <= 1:
                cols.append(c)
        return cols

    def load(self, path: str) -> None:  # pragma: no cover - constructed in __init__
        """Unsupported: the table is read by the constructor instead.

        Args:
            path: Ignored.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError("ModelBlocksDataset is built from its constructor")

    def exclusion_mask(self, datum: Datum) -> np.ndarray:
        """Drop missing DV values; study-specific censoring belongs to the caller.

        Args:
            datum: Observation whose per-word DV values are screened.
        """
        return ~np.isnan(datum.values)

    def _build(self, df: pd.DataFrame) -> None:
        sentid_int = pd.to_numeric(df[self._sentid_col], errors="coerce")
        sentpos_int = pd.to_numeric(df[self._sentpos_col], errors="coerce")
        if sentid_int.isna().any() or sentpos_int.isna().any():
            raise ValueError("non-integer sentid/sentpos in .evmeasures")
        df = df.assign(_sentid=sentid_int.astype(int), _sentpos=sentpos_int.astype(int))
        if self._docid_col is None:
            df = df.assign(_docid="doc0")
            docid_col = "_docid"
        else:
            docid_col = self._docid_col

        df = df.assign(_dv=_to_float(df[self._dv_column]))
        for col in self._datum_columns:
            df[f"_d_{col}"] = _to_float(df[col])

        # Global, deterministic sentence numbering across the whole corpus, the
        # unit the CV partition splits on. Sorted by numeric sentid first so the
        # ids are a dense rank of the raw sentid, matching modelblocks'
        # partition.py cat.codes (docid-string sort would scramble e.g. Dundee's
        # '1','10','2' docids); docid stays in the key for uniqueness.
        ordered = (
            df[[docid_col, "_sentid"]]
            .drop_duplicates()
            .sort_values(["_sentid", docid_col])
        )
        sent_global = {
            (d, s): i
            for i, (d, s) in enumerate(
                zip(ordered[docid_col], ordered["_sentid"], strict=True)
            )
        }

        for docid, doc_df in df.groupby(docid_col, sort=True):
            # Canonical stimulus-level word sequence: unique (sentid, sentpos)
            # in reading order. Predictors are stimulus-level (constant across
            # subjects, enforced by _auto_predictor_columns), so the deduped
            # frame already carries their per-word values in word order.
            unit_df = (
                doc_df.drop_duplicates(subset=["_sentid", "_sentpos"])
                .sort_values(["_sentid", "_sentpos"])
                .reset_index(drop=True)
            )
            words = [str(word) for word in unit_df[self._word_col]]
            if any(not word for word in words):
                raise ValueError(f"Empty stimulus word in ModelBlocks document {docid}")
            n_units = len(words)
            per_word_sentence_ids = [
                sent_global[(docid, int(s))] for s in unit_df["_sentid"]
            ]
            stim = self._make_stimulus(
                words,
                stimulus_id=str(docid),
                metadata={
                    "docid": str(docid),
                    "per_word_sentence_ids": per_word_sentence_ids,
                },
            )
            self._predictor_arrays[str(docid)] = {
                col: _to_float(unit_df[col]) for col in self._predictor_columns
            }

            # Map each event row to its word position via a vectorized merge,
            # then scatter each subject's DV into a word-aligned array.
            unit_pos = unit_df[["_sentid", "_sentpos"]].copy()
            unit_pos["_pos"] = np.arange(n_units)
            placed = doc_df.merge(unit_pos, on=["_sentid", "_sentpos"], how="left")
            for subject, subj_df in placed.groupby(self._subject_col, sort=True):
                pos = subj_df["_pos"].to_numpy()
                dv_values = _scatter_first(n_units, pos, subj_df["_dv"].to_numpy())
                if self._dv_condition:
                    parsed = [conditional_first_run(value) for value in dv_values]
                    dv_values = np.array([run for run, _ in parsed])
                    self._data.append(
                        Datum(
                            stimulus=stim,
                            measure_name=self._dv_condition,
                            values=np.array([flag for _, flag in parsed]),
                            participant_id=str(subject),
                        )
                    )
                self._data.append(
                    Datum(
                        stimulus=stim,
                        measure_name=self._dv_measure,
                        values=dv_values,
                        participant_id=str(subject),
                    )
                )
                for col in self._datum_columns:
                    self._data.append(
                        Datum(
                            stimulus=stim,
                            measure_name=col,
                            values=_scatter_first(
                                n_units, pos, subj_df[f"_d_{col}"].to_numpy()
                            ),
                            participant_id=str(subject),
                        )
                    )

    # -------------------------------------------------------------- predictors

    @property
    def predictor_columns(self) -> list[str]:
        """Table columns :meth:`predictor_model` can expose as per-word values."""
        return list(self._predictor_columns)

    def predictor_model(self, columns: Iterable[str] | None = None):
        """A :class:`PrecomputedFeatureModel` of the table's predictor columns.

        Each requested column becomes a per-word value under its own key, so it
        can be wrapped with
        :class:`~replica.feature.extractors.PrecomputedFeature`.

        Args:
            columns: Subset of :attr:`predictor_columns` to carry; ``None``
                carries all of them.

        Raises:
            KeyError: When a requested column is not a predictor column.
        """
        from replica.feature import FeatureModelOutput, PrecomputedFeatureModel

        cols = list(columns) if columns is not None else list(self._predictor_columns)
        missing = [c for c in cols if c not in self._predictor_columns]
        if missing:
            raise KeyError(
                f"columns {missing!r} not among predictor_columns "
                f"{self._predictor_columns!r}"
            )
        outputs: dict[str, list[FeatureModelOutput]] = {}
        for sid, col_arrays in self._predictor_arrays.items():
            n = len(next(iter(col_arrays.values()))) if col_arrays else 0
            outs = []
            for i in range(n):
                outs.append(
                    FeatureModelOutput(
                        values={c: float(col_arrays[c][i]) for c in cols}
                    )
                )
            outputs[sid] = outs
        return PrecomputedFeatureModel(
            name=f"mb_predictors@{self.info.name}", outputs=outputs
        )


__all__ = ["ModelBlocksDataset", "read_evmeasures", "DV_MEASURE_NAMES"]
