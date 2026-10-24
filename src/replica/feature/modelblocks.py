"""Bridge to the ModelBlocks predictor pipelines and their ``.itemmeasures`` tables.

* :class:`ItemMeasuresTable` + :class:`ModelBlocksFeature`, read a
  pre-computed ``.itemmeasures`` table and expose columns as Features
  (no ModelBlocks install needed).
* :class:`ModelBlocksRunner`, invoke ``make`` in a user-supplied ModelBlocks
  checkout (https://github.com/modelblocks/modelblocks-release) to build
  ``.itemmeasures`` targets on demand.

The ``.itemmeasures`` format: whitespace-separated header + space-separated
rows. A trailing underscore on key columns (``word_``, ``sentid_``, …) marks
``.all-itemmeasures`` outputs vs raw ``.itemmeasures`` (both auto-detected);
multi-document corpora key on ``(docid, sentid, sentpos)``.

This *reads* ModelBlocks predictor tables. For surprisal *computed* natively in
the ModelBlocks convention, see ``feature/surprisal/modelblocks.py``.
"""

from __future__ import annotations

import os
import shlex
import subprocess
from collections.abc import Iterable, Sequence
from itertools import groupby
from pathlib import Path

import numpy as np

from replica.feature.core.annotation._text import _layer_strings
from replica.feature.core.binding import resolve_stimulus_feature
from replica.feature.core.protocols import ModelOutputRecord

from ..core import Stimulus

_MISSING_TOKENS: frozenset[str] = frozenset(
    {"", "NA", "NaN", "nan", "None", "none", "null", "NULL"}
)


# Accept both ``docid`` and ModelBlocks's ``discid`` (discourse id).
_KEY_ALIASES: dict[str, tuple[str, ...]] = {
    "word": ("word", "word_"),
    "sentid": ("sentid", "sentid_"),
    "sentpos": ("sentpos", "sentpos_"),
    "docid": ("docid", "docid_", "discid", "discid_"),
    "subject": ("subject", "subject_"),
}


def _resolve_key(header: list[str], canonical: str) -> str | None:
    for candidate in _KEY_ALIASES[canonical]:
        if candidate in header:
            return candidate
    return None


def _parse_numeric(raw: str) -> float:
    if raw in _MISSING_TOKENS:
        return float("nan")
    return float(raw)


def _require_column_exists(columns: list[str], name: str, source: str) -> None:
    if name not in columns:
        raise KeyError(f"column {name!r} not in {source}; available: {columns!r}")


class ItemMeasuresTable:
    """Parsed wide-format predictor table; construct via :meth:`from_file`.

    Rows are stored long-format (no dedup); :meth:`lookup` returns the first
    row at a key. ``.all-itemmeasures`` files mix stimulus-level columns
    (``wlen_``, ``totsurp_``) with subject-level ones (``fdur_``).

    Attributes:
        path: source file path (for error messages).
        columns: column names in file order.
        rows: list of dicts, one per file row.
        word_col: name of the word column (``"word"`` or ``"word_"``);
            ``None`` if absent.
        sentid_col / sentpos_col: names of the key columns.
        docid_col: name of the ``docid`` column, or ``None``.
        subject_col: name of the ``subject`` column, or ``None``.
    """

    def __init__(
        self,
        *,
        path: Path,
        columns: list[str],
        rows: list[dict[str, str]],
        first_index: dict,
        all_indices: dict,
        word_col: str | None,
        sentid_col: str,
        sentpos_col: str,
        docid_col: str | None,
        subject_col: str | None,
    ):
        """Bind an already-parsed table to its key indices.

        Args:
            path: File the rows were read from, reported in error messages.
            columns: Column names in file order.
            rows: One dict per file row, values kept as raw strings.
            first_index: Row key to the index of its first row.
            all_indices: Row key to every row index under it, which is more
                than one when the file repeats a stimulus per subject.
            word_col: Name of the word column, ``None`` when the file has none.
            sentid_col: Name of the sentence-id column.
            sentpos_col: Name of the within-sentence position column.
            docid_col: Name of the document-id column, ``None`` when the corpus
                is a single document.
            subject_col: Name of the subject column, ``None`` for a
                stimulus-level table.
        """
        self.path = path
        self.columns = columns
        self.rows = rows
        self.word_col = word_col
        self.sentid_col = sentid_col
        self.sentpos_col = sentpos_col
        self.docid_col = docid_col
        self.subject_col = subject_col
        self._first_index = first_index
        self._all_indices = all_indices

    @property
    def has_docid(self) -> bool:
        """True when rows are keyed by document as well as sentence position."""
        return self.docid_col is not None

    @classmethod
    def from_paired_files(
        cls,
        gold_path: str | Path,
        *predictor_paths: str | Path,
    ) -> ItemMeasuresTable:
        """Join key-less predictor files onto a *gold* table by row index.

        For per-predictor files (e.g. ``%.gpt2.itemmeasures`` via
        ``roll_toks.py``), which list columns such as ``word totsurp rolled`` in
        the gold's row order with no keys; a joined ``.all-itemmeasures`` goes
        through :meth:`from_file`.

        Args:
            gold_path: The keyed ``.itemmeasures`` file (``sentid`` /
                ``sentpos``) that supplies the keys and the row order.
            predictor_paths: Key-less files joined onto the gold, each holding
                one row per gold row in the same order; a column the merged
                table already carries is skipped.

        Returns:
            The gold table widened by the new predictor columns, or the gold
            table alone when no predictor paths are given.

        Raises:
            ValueError: On an empty predictor file, a row whose field count
                differs from its header, or a row count differing from the
                gold's.
        """
        gold = cls.from_file(gold_path)
        if not predictor_paths:
            return gold
        all_extra_headers: list[list[str]] = []
        all_extra_rows: list[list[list[str]]] = []
        for p in predictor_paths:
            p = Path(p)
            with p.open() as f:
                lines = f.read().splitlines()
            if not lines:
                raise ValueError(f"empty predictor file: {p}")
            header = lines[0].split()
            rows_data: list[list[str]] = []
            for lineno, line in enumerate(lines[1:], start=2):
                parts = line.split()
                if not parts:
                    continue
                if len(parts) != len(header):
                    raise ValueError(
                        f"{p}:{lineno}: expected {len(header)} columns, "
                        f"got {len(parts)}"
                    )
                rows_data.append(parts)
            if len(rows_data) != len(gold.rows):
                raise ValueError(
                    f"{p} has {len(rows_data)} rows but gold "
                    f"{Path(gold_path).name} has {len(gold.rows)}; "
                    f"row counts must match for a positional join"
                )
            all_extra_headers.append(header)
            all_extra_rows.append(rows_data)

        merged_columns: list[str] = list(gold.columns)
        keep_indices: list[list[int]] = []
        for header in all_extra_headers:
            keep = []
            for col_idx, col in enumerate(header):
                if col in merged_columns:
                    continue
                merged_columns.append(col)
                keep.append(col_idx)
            keep_indices.append(keep)

        new_rows: list[dict[str, str]] = []
        for i, row in enumerate(gold.rows):
            merged_row = dict(row)
            for header, rows_data, keep in zip(
                all_extra_headers, all_extra_rows, keep_indices, strict=False
            ):
                parts = rows_data[i]
                for col_idx in keep:
                    merged_row[header[col_idx]] = parts[col_idx]
            new_rows.append(merged_row)

        return cls(
            path=Path(gold_path),
            columns=merged_columns,
            rows=new_rows,
            first_index=gold._first_index,
            all_indices=gold._all_indices,
            word_col=gold.word_col,
            sentid_col=gold.sentid_col,
            sentpos_col=gold.sentpos_col,
            docid_col=gold.docid_col,
            subject_col=gold.subject_col,
        )

    @classmethod
    def from_file(cls, path: str | Path) -> ItemMeasuresTable:
        """Parse one ``.itemmeasures`` or ``.all-itemmeasures`` file.

        The key columns are auto-detected in the header, with or without their
        trailing underscore. A key may repeat only when the file has a subject
        column, one row per reader.

        Args:
            path: File to read, header line first.

        Raises:
            ValueError: When the file is empty, its header repeats a name or
                lacks ``sentid`` / ``sentpos``, a row's field count differs
                from the header's, a key is non-integer, or a key repeats
                without a subject column.
        """
        path = Path(path)
        with path.open() as f:
            lines = f.read().splitlines()
        if not lines:
            raise ValueError(f"item-measures file is empty: {path}")

        header = lines[0].split()
        if len(set(header)) != len(header):
            dups = sorted({c for c in header if header.count(c) > 1})
            raise ValueError(f"{path}: duplicate column names in header: {dups!r}")

        word_col = _resolve_key(header, "word")
        sentid_col = _resolve_key(header, "sentid")
        sentpos_col = _resolve_key(header, "sentpos")
        docid_col = _resolve_key(header, "docid")
        subject_col = _resolve_key(header, "subject")

        if sentid_col is None or sentpos_col is None:
            raise ValueError(
                f"{path}: must have sentid + sentpos columns; header is {header!r}"
            )

        rows: list[dict[str, str]] = []
        first_index: dict = {}
        all_indices: dict[tuple, list[int]] = {}
        for lineno, line in enumerate(lines[1:], start=2):
            parts = line.split()
            if not parts:
                continue
            if len(parts) != len(header):
                raise ValueError(
                    f"{path}:{lineno}: expected {len(header)} columns, got {len(parts)}"
                )
            row = dict(zip(header, parts, strict=False))
            try:
                sentid_val = int(row[sentid_col])
                sentpos_val = int(row[sentpos_col])
            except ValueError as e:
                raise ValueError(
                    f"{path}:{lineno}: non-integer sentid/sentpos in {row!r}"
                ) from e
            key: tuple
            if docid_col is None:
                key = (sentid_val, sentpos_val)
            else:
                key = (row[docid_col], sentid_val, sentpos_val)

            if key in first_index and subject_col is None:
                existing = rows[first_index[key]]
                raise ValueError(
                    f"{path}:{lineno}: duplicate key {key} but no subject "
                    f"column; rows {existing!r} vs {row!r}"
                )
            row_idx = len(rows)
            rows.append(row)
            all_indices.setdefault(key, []).append(row_idx)
            first_index.setdefault(key, row_idx)

        return cls(
            path=path,
            columns=header,
            rows=rows,
            first_index=first_index,
            all_indices=all_indices,
            word_col=word_col,
            sentid_col=sentid_col,
            sentpos_col=sentpos_col,
            docid_col=docid_col,
            subject_col=subject_col,
        )

    def lookup(
        self,
        sentid: int,
        sentpos: int,
        docid: str | None = None,
    ) -> dict[str, str] | None:
        """Row at the given key, or ``None``; mismatched docid raises.

        Args:
            sentid: Sentence id, as the table's key column spells it.
            sentpos: Position of the word within that sentence.
            docid: Document id, required exactly when the table has a docid
                column and rejected when it has none.

        Returns:
            The first row stored at that key, which for a subject-level table
            is one reader's row.
        """
        if self.docid_col is None:
            if docid is not None:
                raise ValueError(
                    f"{self.path.name}: table has no docid column; don't pass docid="
                )
            key: tuple = (int(sentid), int(sentpos))
        else:
            if docid is None:
                raise ValueError(
                    f"{self.path.name}: table has a {self.docid_col!r} "
                    f"column; pass docid="
                )
            key = (str(docid), int(sentid), int(sentpos))
        idx = self._first_index.get(key)
        return None if idx is None else self.rows[idx]

    def verify_stimulus_level(self, column: str) -> None:
        """Raise if ``column`` varies across subjects for any key.

        Args:
            column: Column asserted to be a property of the stimulus, so every
                subject's row at a key carries the same value.
        """
        _require_column_exists(self.columns, column, self.path.name)
        for key, indices in self._all_indices.items():
            if len(indices) < 2:
                continue
            seen = {self.rows[i][column] for i in indices}
            if len(seen) > 1:
                raise ValueError(
                    f"{self.path.name}: column {column!r} varies across "
                    f"subjects for key {key}: {sorted(seen)!r}; "
                    f"not stimulus-level"
                )

    def word_at(
        self,
        sentid: int,
        sentpos: int,
        docid: str | None = None,
    ) -> str | None:
        """Word form at one key, or ``None`` when the key or column is absent.

        Args:
            sentid: Sentence id, as the table's key column spells it.
            sentpos: Position of the word within that sentence.
            docid: Document id, required exactly when the table has a docid
                column.
        """
        row = self.lookup(sentid, sentpos, docid=docid)
        if row is None or self.word_col is None:
            return None
        return row[self.word_col]

    def column(self, name: str) -> list[str]:
        """Raw string values for one column, in file order.

        Args:
            name: Column to read, spelled as the header spells it.
        """
        _require_column_exists(self.columns, name, self.path.name)
        return [r[name] for r in self.rows]

    def column_array(self, name: str) -> np.ndarray:
        """One column as a 1-D float array (one entry per file row; missing → NaN).

        ``.all-itemmeasures`` files are long-format, so K subjects yield
        K x rows-per-stimulus; use :meth:`lookup` for per-key access.

        Args:
            name: Column to read, whose values have to parse as numbers or as
                one of the recognized missing-value tokens.

        Raises:
            ValueError: On a value that is neither numeric nor a missing-value
                token.
        """
        _require_column_exists(self.columns, name, self.path.name)
        out = np.full(len(self.rows), np.nan)
        for i, r in enumerate(self.rows):
            try:
                out[i] = _parse_numeric(r[name])
            except ValueError as e:
                raise ValueError(
                    f"{self.path.name}: column {name!r} row {i} value "
                    f"{r[name]!r} is not numeric"
                ) from e
        return out


class ModelBlocksFeature:
    """One column of an :class:`ItemMeasuresTable` as a per-word :class:`Feature`.

    Two alignment modes:

    * Single-sentence (default): reads ``sentid`` (and ``docid``) from
      ``stimulus.metadata``; ``sentpos = word_index + sentpos_base``.
    * Multi-sentence (``sentence_layer=`` given): matches each word to the
      sentence whose span contains it, with ``sentpos`` restarting per
      sentence. Required for paragraph-level stimuli (NS, Dundee). The layer
      holds one span per sentence; the word-parallel sentence-id layer that
      :func:`~replica.dataset.catalog.sentences.attach_sentences` builds is
      refused.

    Args:
        table: Parsed predictor table the values are read from.
        column: Column to expose, e.g. ``"dlt"``, ``"totsurp"``.
        name: Name the feature reports; ``None`` uses ``mb_<column>``.
        sentpos_base: Position the first word of a sentence gets, ``1`` in
            ModelBlocks's 1-based indexing.
        strict_word_check: Raise on word mismatch (catches tokenization drift).
        verify_stimulus_level: Assert ``column`` is subject-independent.
        sentence_layer: Annotation layer of sentence spans, each entry a
            mapping with ``"sentid"`` (+ optional ``"docid"``); ``None``
            reads from stimulus metadata.
    """

    dependencies: tuple = ()
    resolve = resolve_stimulus_feature

    def configuration(self) -> dict:
        """Declare the selected table column and lookup policy."""
        return {
            "table_path": str(self._table.path),
            "column": self._column,
            "sentpos_base": self._sentpos_base,
            "sentence_layer": self._sentence_layer,
            "strict_word_check": self._strict_word_check,
        }

    def __init__(
        self,
        table: ItemMeasuresTable,
        column: str,
        *,
        name: str | None = None,
        sentpos_base: int = 1,
        strict_word_check: bool = False,
        verify_stimulus_level: bool = False,
        sentence_layer: str | None = None,
    ):
        """Bind one table column, checking it exists before any stimulus runs."""
        _require_column_exists(table.columns, column, table.path.name)
        if verify_stimulus_level:
            table.verify_stimulus_level(column)
        self._table = table
        self._source = table
        self._column = column
        self._sentpos_base = sentpos_base
        self._strict_word_check = strict_word_check
        self._sentence_layer = sentence_layer
        self.name = name or f"mb_{column}"

    @property
    def required_outputs(self) -> tuple[str, ...]:
        """No model value keys: the column comes from the table."""
        return ()

    def _coerce_int(self, value: object, context: str) -> int:
        try:
            return int(value)  # type: ignore[arg-type]
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"{self.name}: {context} is not int-coercible: {value!r}"
            ) from e

    def _keys_from_metadata(
        self, stimulus: Stimulus, n_words: int
    ) -> list[tuple[int, int, str | None]]:
        meta = stimulus.metadata
        if "sentid" not in meta:
            raise KeyError(
                f"{self.name}: stimulus {stimulus.stimulus_id!r} has no "
                f"'sentid' in metadata"
            )
        sentid = self._coerce_int(
            meta["sentid"],
            f"stimulus {stimulus.stimulus_id!r} metadata 'sentid'",
        )
        docid: str | None = None
        if self._table.has_docid:
            if "docid" not in meta:
                raise KeyError(
                    f"{self.name}: table has docid column but stimulus "
                    f"{stimulus.stimulus_id!r} has no 'docid' in metadata"
                )
            docid = str(meta["docid"])
        return [(sentid, i + self._sentpos_base, docid) for i in range(n_words)]

    def _keys_from_sentence_layer(
        self, stimulus: Stimulus, target_layer: str
    ) -> list[tuple[int, int, str | None]]:
        if self._sentence_layer not in stimulus.layers:
            raise KeyError(
                f"{self.name}: sentence_layer={self._sentence_layer!r} "
                f"is not present on stimulus {stimulus.stimulus_id!r}"
            )
        sent_layer = stimulus.layers[self._sentence_layer]
        word_layer = stimulus.layers[target_layer]
        keys: list[tuple[int, int, str | None]] = []
        last_sent_idx = -1
        pos = self._sentpos_base
        for word_idx, word_span in enumerate(word_layer.spans):
            sent_idx: int | None = None
            for j, sent_span in enumerate(sent_layer.spans):
                if sent_span.contains(word_span):
                    sent_idx = j
                    break
            if sent_idx is None:
                raise KeyError(
                    f"{self.name}: word {word_idx} (span {word_span}) is "
                    f"not contained in any span of layer "
                    f"{self._sentence_layer!r} on stimulus "
                    f"{stimulus.stimulus_id!r}"
                )
            sent_meta = sent_layer.values[sent_idx]
            if not isinstance(sent_meta, dict):
                raise TypeError(
                    f"{self.name}: sentence_layer "
                    f"{self._sentence_layer!r} entry {sent_idx} value "
                    f"is not a mapping: {sent_meta!r}; the layer needs one "
                    "span per sentence valued {'sentid': ..., 'docid': ...} "
                    "(attach_sentences builds a per-word sentence-id layer)"
                )
            if "sentid" not in sent_meta:
                raise KeyError(
                    f"{self.name}: sentence_layer "
                    f"{self._sentence_layer!r} entry {sent_idx} has no "
                    f"'sentid' key: {sent_meta!r}"
                )
            sentid = self._coerce_int(
                sent_meta["sentid"],
                f"sentence_layer entry {sent_idx} 'sentid'",
            )
            docid_meta = sent_meta.get("docid")
            if self._table.has_docid:
                if docid_meta is None:
                    raise KeyError(
                        f"{self.name}: table has docid column but "
                        f"sentence_layer entry {sent_idx} has no 'docid'"
                    )
                docid: str | None = str(docid_meta)
            else:
                docid = None
            if sent_idx != last_sent_idx:
                pos = self._sentpos_base
                last_sent_idx = sent_idx
            else:
                pos += 1
            keys.append((sentid, pos, docid))
        return keys

    def __call__(
        self,
        stimulus: Stimulus,
        outputs: Sequence[ModelOutputRecord] | None,
        target_layer: str = "words",
    ) -> np.ndarray:
        """Per-unit column values for one stimulus, aligned by table key.

        Args:
            stimulus: Stimulus whose units are looked up in the table.
            outputs: Feature-model outputs, unused: the values come from the
                table rather than from a model.
            target_layer: Annotation layer defining the units, one array entry
                per span of that layer.

        Returns:
            One value per unit, ``NaN`` where the table records a missing one.

        Raises:
            KeyError: When a unit has no row in the table.
            ValueError: Under ``strict_word_check``, when a table word differs
                from the stimulus word, or when the value is non-numeric.
        """
        strings = _layer_strings(stimulus, target_layer)
        if not strings:
            return np.array([], dtype=float)
        if self._sentence_layer is not None:
            keys = self._keys_from_sentence_layer(stimulus, target_layer)
        else:
            keys = self._keys_from_metadata(stimulus, len(strings))
        vals = np.full(len(strings), np.nan)
        for i, ((sentid, sentpos, docid), w) in enumerate(
            zip(keys, strings, strict=False)
        ):
            row = self._table.lookup(sentid, sentpos, docid=docid)
            if row is None:
                key_repr = (
                    f"(docid={docid!r}, sentid={sentid}, sentpos={sentpos})"
                    if docid is not None
                    else f"(sentid={sentid}, sentpos={sentpos})"
                )
                raise KeyError(
                    f"{self.name}: no row for {key_repr} in "
                    f"{self._table.path.name}; "
                    f"stimulus {stimulus.stimulus_id!r} word {i}={w!r}"
                )
            if self._strict_word_check and self._table.word_col is not None:
                tab_w = row[self._table.word_col]
                if tab_w != w:
                    raise ValueError(
                        f"{self.name}: word mismatch at sentid={sentid} "
                        f"sentpos={sentpos}: stimulus {w!r} vs table "
                        f"{tab_w!r}. Likely a tokenization mismatch."
                    )
            try:
                vals[i] = _parse_numeric(row[self._column])
            except ValueError as e:
                raise ValueError(
                    f"{self.name}: non-numeric value {row[self._column]!r} "
                    f"at sentid={sentid} sentpos={sentpos}"
                ) from e
        return vals


class ModelBlocksRunner:
    """Drive a ModelBlocks checkout via ``make`` to produce ``.itemmeasures`` files.

    Point this at an existing ModelBlocks installation
    (https://github.com/modelblocks/modelblocks-release) with its
    ``workspace`` bootstrapped. Targets follow ModelBlocks's filename
    grammar (``<corpus>.<spec1>....itemmeasures``).

    Args:
        modelblocks_dir: Checkout root; falls back to ``$MODELBLOCKS_DIR``.
        workspace: Defaults to ``<modelblocks_dir>/workspace``.
        make_executable: String or argv list (e.g. ``"make -j8"``).
        env: Merged on top of :data:`os.environ`.
    """

    def __init__(
        self,
        modelblocks_dir: str | Path | None = None,
        workspace: str | Path | None = None,
        *,
        make_executable: str | Iterable[str] = "make",
        env: dict[str, str] | None = None,
    ):
        """Resolve the checkout and workspace, rejecting one that cannot build.

        Raises:
            ValueError: When no checkout is given and ``$MODELBLOCKS_DIR`` is
                unset, or *make_executable* is empty.
            FileNotFoundError: When the checkout, its ``resource-general``
                subdirectory, or the workspace is missing.
        """
        if modelblocks_dir is None:
            modelblocks_dir = os.environ.get("MODELBLOCKS_DIR")
        if modelblocks_dir is None:
            raise ValueError(
                "modelblocks_dir not given and $MODELBLOCKS_DIR is not set. "
                "Clone https://github.com/modelblocks/modelblocks-release "
                "and pass its path."
            )
        self.modelblocks_dir = Path(modelblocks_dir).resolve()
        self.workspace = (
            Path(workspace).resolve()
            if workspace
            else self.modelblocks_dir / "workspace"
        )
        if isinstance(make_executable, str):
            self.make_argv = shlex.split(make_executable)
        else:
            self.make_argv = list(make_executable)
        if not self.make_argv:
            raise ValueError("make_executable resolved to empty argv")
        self.env = dict(env or {})

        if not self.modelblocks_dir.exists():
            raise FileNotFoundError(
                f"ModelBlocks directory not found: {self.modelblocks_dir}"
            )
        if not (self.modelblocks_dir / "resource-general").is_dir():
            raise FileNotFoundError(
                f"{self.modelblocks_dir} does not look like a ModelBlocks "
                f"checkout (no resource-general/ subdir)."
            )
        if not self.workspace.exists():
            raise FileNotFoundError(
                f"ModelBlocks workspace not found at {self.workspace}. "
                f"Run `make` from {self.modelblocks_dir} to bootstrap it."
            )

    def make(
        self,
        target: str,
        *,
        timeout: float | None = None,
        capture: bool = True,
    ) -> Path:
        """Build a ModelBlocks make-target and return its resolved path.

        Args:
            target: Make target, workspace-relative, in ModelBlocks's filename
                grammar.
            timeout: Seconds to allow the build; ``None`` waits indefinitely.
            capture: Capture the build's output rather than letting it through
                to the terminal, so a failure can quote its stderr.

        Raises:
            RuntimeError: When the build times out or exits non-zero.
            FileNotFoundError: When it succeeds without writing the target.
        """
        target_path = self.workspace / target
        env = {**os.environ, **self.env}
        cmd = [*self.make_argv, target]
        try:
            subprocess.run(
                cmd,
                cwd=self.workspace,
                env=env,
                check=True,
                timeout=timeout,
                capture_output=capture,
                text=True,
            )
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(
                f"ModelBlocks `{' '.join(cmd)}` timed out after {timeout}s. "
                f"Workspace: {self.workspace}."
            ) from e
        except subprocess.CalledProcessError as e:
            stderr = (e.stderr or "").strip()
            raise RuntimeError(
                f"ModelBlocks `{' '.join(cmd)}` failed (exit {e.returncode}). "
                f"Workspace: {self.workspace}. "
                f"Stderr tail:\n{stderr[-2000:]}"
            ) from e
        if not target_path.exists():
            raise FileNotFoundError(
                f"`make {target}` reported success but {target_path} is missing"
            )
        return target_path

    def itemmeasures(
        self,
        target: str,
        *predictor_targets: str,
        **make_kwargs,
    ) -> ItemMeasuresTable:
        """Build an item-measures target and parse it.

        Args:
            target: Make target of the keyed item-measures file.
            predictor_targets: Make targets of key-less predictor files, built
                as well and joined onto *target* by row index.
            make_kwargs: Forwarded to :meth:`make` for every target built, so a
                caller can set ``timeout`` or ``capture``.
        """
        gold_path = self.make(target, **make_kwargs)
        if not predictor_targets:
            return ItemMeasuresTable.from_file(gold_path)
        predictor_paths = [self.make(t, **make_kwargs) for t in predictor_targets]
        return ItemMeasuresTable.from_paired_files(gold_path, *predictor_paths)

    def feature(
        self,
        target: str,
        column: str,
        *,
        predictor_targets: tuple[str, ...] = (),
        name: str | None = None,
        sentpos_base: int = 1,
        strict_word_check: bool = False,
        verify_stimulus_level: bool = False,
        sentence_layer: str | None = None,
        **make_kwargs,
    ) -> ModelBlocksFeature:
        """Build a target and wrap one column as a :class:`Feature`.

        Example::

            runner = ModelBlocksRunner("/path/to/modelblocks-release")
            dlt = runner.feature(
                "genmodel/naturalstories.casp.dlt.itemmeasures",
                column="dlt",
            )

        Args:
            target: Make target of the keyed item-measures file.
            column: Column of that table to expose as the feature's values.
            predictor_targets: Key-less predictor targets joined on by row.
            name: Name the feature reports; ``None`` uses ``mb_<column>``.
            sentpos_base: Position the first word of a sentence gets.
            strict_word_check: Raise when a table word differs from the
                stimulus word.
            verify_stimulus_level: Assert the column is subject-independent.
            sentence_layer: Annotation layer of sentence spans for
                multi-sentence stimuli; ``None`` reads stimulus metadata.
            make_kwargs: Forwarded through :meth:`itemmeasures` to :meth:`make`,
                so a caller can set ``timeout`` or ``capture``.
        """
        table = self.itemmeasures(target, *predictor_targets, **make_kwargs)
        return ModelBlocksFeature(
            table,
            column,
            name=name,
            sentpos_base=sentpos_base,
            strict_word_check=strict_word_check,
            verify_stimulus_level=verify_stimulus_level,
            sentence_layer=sentence_layer,
        )


def write_sentitems(
    stimuli: list[Stimulus],
    path: str | Path,
    *,
    words_layer: str = "words",
    sentences_layer: str | None = None,
) -> Path:
    """Write REPLICA stimuli to a ModelBlocks ``.sentitems`` file.

    Each stimulus becomes one ``!ARTICLE`` block. With ``sentences_layer``,
    each sentence is its own line (parser-recipe layout); otherwise all
    words join on one line (``get_llm_surprisal.py`` layout).

    Args:
        stimuli: Stimuli to emit, one ``!ARTICLE`` block each, in order.
        path: File to write; parent directories must exist.
        words_layer: Layer whose strings become the emitted tokens.
        sentences_layer: Sentence-boundary layer. When present, each sentence
            is its own line. A layer with one span per sentence groups the
            words its spans contain; a layer whose spans equal the word spans
            (the layout :func:`~replica.dataset.catalog.sentences.attach_sentences`
            builds) holds sentence ids, and consecutive words with the same id
            form one sentence.

    Returns:
        The path written, as a :class:`~pathlib.Path`.
    """
    path = Path(path)
    lines: list[str] = []
    for stim in stimuli:
        lines.append("!ARTICLE")
        if sentences_layer is not None and sentences_layer in stim.layers:
            sent_layer = stim.layers[sentences_layer]
            word_layer = stim.layers[words_layer]
            if sent_layer.spans == word_layer.spans:
                pairs = zip(word_layer.spans, sent_layer.values, strict=True)
                for _, sentence in groupby(pairs, key=lambda pair: pair[1]):
                    lines.append(
                        " ".join(span.extract(stim.text) for span, _ in sentence)
                    )
                continue
            for sent_span in sent_layer.spans:
                words = [
                    span.extract(stim.text)
                    for span in word_layer.spans
                    if sent_span.contains(span)
                ]
                if words:
                    lines.append(" ".join(words))
        else:
            lines.append(" ".join(_layer_strings(stim, words_layer)))
    path.write_text("\n".join(lines) + ("\n" if lines else ""))
    return path


__all__ = [
    "ItemMeasuresTable",
    "ModelBlocksFeature",
    "ModelBlocksRunner",
    "write_sentitems",
]
