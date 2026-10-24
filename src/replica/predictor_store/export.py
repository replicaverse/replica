"""Stage a release of the predictor store from compute output directories.

The export reads every completed cell, writes its rows into the staged tree
under the layout of :mod:`~replica.predictor_store.layout`, copies its manifest
and verification record, and writes the index and coverage tables. With the
hub merged in, the rows and index entries of published cells this run does
not replace are carried over, so the staged tree describes the whole store
after the publish.
"""

from __future__ import annotations

import csv
import errno
import json
import logging
import shutil
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from replica.artifacts import manifest_text
from replica.compute.cache import SHARD_RE, canonical_manifest
from replica.compute.units import (
    METHOD_STORAGE,
    TOKEN_UNIT,
)
from replica.feature.core.annotation.characters import CHARACTER_LAYER
from replica.feature.core.outputs.units import unit_for
from replica.utils.naming import canonical_key, canonical_method, canonical_tag

from . import layout
from .alignment import RowCheck, describe
from .cells import QUARANTINED, Cell, Discovery, discover, load_payload, payload_names
from .corpora import Corpora, MisalignedCell, RegistryCorpora, TextIndex, check_payload
from .hub import Hub
from .readme import render_readme

log = logging.getLogger("replica.predictor_store")

_BATCH_ROWS = 500_000

#: Filesystem errors that stop the export: every later cell hits them too.
_FATAL_ERRNOS = frozenset({errno.ENOSPC, errno.EDQUOT, errno.EROFS})


@dataclass
class CellTables:
    """What one cell contributes to the staged tree."""

    scalar_file: str | None
    scalar: pa.Table | None
    arrays: dict[str, pa.Table]
    dropped: dict[str, int]


def _is_scalar_measurement(key: str, value: object) -> bool:
    if isinstance(value, str | bytes):
        return False
    array = np.asarray(value)
    if array.ndim != 0 or array.dtype.kind in "USO":
        return False
    return unit_for(key) not in layout.NON_NUMERIC_UNITS


def _list_array(arrays: list[np.ndarray], depth: int, item: pa.DataType) -> pa.Array:
    """A list array of *depth* levels from equally ranked numpy arrays."""
    dtype = item.to_pandas_dtype()
    if depth == 1:
        lengths = np.fromiter(
            (a.size for a in arrays), dtype=np.int64, count=len(arrays)
        )
        flat = (
            np.concatenate([a.reshape(-1).astype(dtype) for a in arrays])
            if arrays
            else np.zeros(0, dtype=dtype)
        )
        offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int32)
        return pa.ListArray.from_arrays(pa.array(offsets), pa.array(flat, type=item))
    if depth == 2:
        rows = np.fromiter(
            (a.shape[0] for a in arrays), dtype=np.int64, count=len(arrays)
        )
        inner = [row for a in arrays for row in a]
        outer = np.concatenate([[0], np.cumsum(rows)]).astype(np.int32)
        return pa.ListArray.from_arrays(pa.array(outer), _list_array(inner, 1, item))
    raise ValueError(f"arrays of rank {depth} have no store column")


def _scalar_table(cell: Cell, payload: dict, text: TextIndex | None) -> pa.Table | None:
    unit = cell.unit
    tokens = unit == TOKEN_UNIT
    dense_lens = cell.method in layout.LENS_METHODS and not tokens
    columns: dict[str, list] = defaultdict(list)
    for stimulus_id, records in payload.items():
        strings = None
        if not tokens and text is not None:
            strings = text.units(unit, stimulus_id, len(records))
        for index, record in enumerate(records):
            identity = layout.token_identity(record.metadata, index) if tokens else None
            for key, value in record.values.items():
                if not _is_scalar_measurement(key, value):
                    continue
                if dense_lens and layout.is_lens_key(key):
                    continue
                columns["predictor"].append(key)
                columns["stimulus_id"].append(stimulus_id)
                columns["value"].append(float(np.asarray(value)))
                columns["value_unit"].append(unit_for(key))
                if identity is not None:
                    token_index, start, end, token_id, token = identity
                    columns["token_index"].append(token_index)
                    columns["span_start"].append(start)
                    columns["span_end"].append(end)
                    columns["token_id"].append(token_id)
                    columns["token"].append(token)
                else:
                    columns["unit_index"].append(index)
                    columns["text_unit"].append(
                        strings[index] if strings is not None else None
                    )
    if not columns:
        return None
    n = len(columns["value"])
    layout.check_ids(set(columns["stimulus_id"]), cell.path.name)
    columns["dataset"] = [cell.dataset] * n
    columns["tag"] = [cell.tag] * n
    if tokens:
        schema = layout.TOKEN_SCHEMA
    else:
        schema = layout.WORD_SCHEMA
        columns["layer"] = [cell.layer] * n
        columns["storage_class"] = [cell.storage_class] * n
        columns["unit_rule"] = [cell.unit_rule] * n
    return pa.table(
        {
            name: pa.array(columns[name], type=schema.field(name).type)
            for name in schema.names
        },
        schema=schema,
    )


def _unit_columns(
    cell: Cell,
    payload: dict,
    text: TextIndex | None,
    picked: list[tuple[str, int, Any]],
) -> dict[str, list]:
    """Identity columns of array rows: the unit index plus text or token identity."""
    columns: dict[str, list] = defaultdict(list)
    tokens = cell.unit == TOKEN_UNIT
    strings_of: dict[str, list[str] | None] = {}
    for stimulus_id, index, record in picked:
        columns["stimulus_id"].append(stimulus_id)
        columns["unit_index"].append(index)
        if tokens:
            _, start, end, token_id, token = layout.token_identity(
                record.metadata, index
            )
            columns["token_index"].append(index)
            columns["span_start"].append(start)
            columns["span_end"].append(end)
            columns["token_id"].append(token_id)
            columns["token"].append(token)
        else:
            if stimulus_id not in strings_of:
                strings_of[stimulus_id] = (
                    text.units(cell.unit, stimulus_id, len(payload[stimulus_id]))
                    if text is not None
                    else None
                )
            strings = strings_of[stimulus_id]
            columns["text_unit"].append(strings[index] if strings is not None else None)
    return columns


def _family_table(
    cell: Cell,
    family: layout.ArrayFamily,
    payload: dict,
    text: TextIndex | None,
) -> pa.Table | None:
    picked: list[tuple[str, int, Any]] = []
    arrays: list[np.ndarray] = []
    for stimulus_id, records in payload.items():
        for index, record in enumerate(records):
            value = record.values.get(family.key)
            if value is None:
                continue
            picked.append((stimulus_id, index, record))
            arrays.append(np.asarray(value))
    if not arrays:
        return None
    depth = {a.ndim for a in arrays}
    if len(depth) != 1:
        raise ValueError(f"{family.key} mixes ranks {sorted(depth)}")
    rank = depth.pop()
    schema = family.schema(cell.unit, rank)
    columns = _unit_columns(cell, payload, text, picked)
    n = len(arrays)
    columns["dataset"] = [cell.dataset] * n
    columns["tag"] = [cell.tag] * n
    columns["value_unit"] = [unit_for(family.key)] * n
    if family.model_column:
        columns["model"] = [cell.info.model_slot] * n
    data = {
        name: pa.array(columns[name], type=schema.field(name).type)
        for name in schema.names
        if name != family.column
    }
    data[family.column] = _list_array(arrays, rank, family.item)
    return pa.table(data, schema=schema)


def _lens_table(cell: Cell, payload: dict, text: TextIndex | None) -> pa.Table | None:
    picked: list[tuple[str, int, Any]] = []
    vectors: list[np.ndarray] = []
    for stimulus_id, records in payload.items():
        for index, record in enumerate(records):
            keys = layout.lens_layer_keys(list(record.values))
            if not keys:
                continue
            picked.append((stimulus_id, index, record))
            vectors.append(
                np.array([float(record.values[k]) for k in keys], dtype=np.float64)
            )
    if not vectors:
        return None
    schema = layout.lens_schema()
    columns = _unit_columns(cell, payload, text, picked)
    n = len(vectors)
    columns["dataset"] = [cell.dataset] * n
    columns["tag"] = [cell.tag] * n
    columns["value_unit"] = [unit_for(layout.LENS_COLUMN)] * n
    data = {
        name: pa.array(columns[name], type=schema.field(name).type)
        for name in schema.names
        if name != layout.LENS_COLUMN
    }
    data[layout.LENS_COLUMN] = _list_array(vectors, 1, pa.float64())
    return pa.table(data, schema=schema)


def _lookahead_table(
    cell: Cell, payload: dict, text: TextIndex | None
) -> pa.Table | None:
    picked: list[tuple[str, int, Any]] = []
    stacks: dict[str, list[np.ndarray]] = {c: [] for c in layout.LOOKAHEAD_COLUMNS}
    n_samples: list[int] = []
    k_words: list[int] = []
    for stimulus_id, records in payload.items():
        for index, record in enumerate(records):
            values = record.values
            if layout.LOOKAHEAD_INDEX_KEY not in values:
                continue
            picked.append((stimulus_id, index, record))
            arrays = layout.lookahead_arrays(values)
            steps = arrays["log_p_step_nats"]
            n_samples.append(int(steps.shape[0]))
            k_words.append(int(steps.shape[1]))
            for column, array in arrays.items():
                stacks[column].append(array.reshape(-1))
    if not picked:
        return None
    schema = layout.lookahead_schema()
    columns = _unit_columns(cell, payload, text, picked)
    n = len(picked)
    columns["dataset"] = [cell.dataset] * n
    columns["model"] = [cell.info.model_slot] * n
    columns["tag"] = [cell.tag] * n
    columns["n_samples"] = n_samples
    columns["k_words"] = k_words
    columns["value_unit"] = [unit_for(layout.LOOKAHEAD_INDEX_KEY)] * n
    data = {
        name: pa.array(columns[name], type=schema.field(name).type)
        for name in schema.names
        if name not in layout.LOOKAHEAD_COLUMNS
    }
    for column in layout.LOOKAHEAD_COLUMNS:
        data[column] = _list_array(
            stacks[column], 1, schema.field(column).type.value_type
        )
    return pa.table(data, schema=schema)


def _array_keys(payload: dict) -> dict[str, int]:
    """Array-valued keys of a payload and the number of records carrying each."""
    keys: dict[str, int] = defaultdict(int)
    for records in payload.values():
        for record in records:
            for key, value in record.values.items():
                if not isinstance(value, str | bytes) and np.ndim(value) >= 1:
                    keys[key] += 1
    return dict(keys)


def cell_tables(cell: Cell, payload: dict, text: TextIndex | None) -> CellTables:
    """Every table one cell contributes, keyed by its path in the store.

    Args:
        cell: The cell, with its manifest and decoded tag.
        payload: The unpickled ``{stimulus_id: [FeatureModelOutput]}`` mapping.
        text: Unit strings of the corpus, or ``None`` for a token cell.

    Raises:
        ValueError: When a record is malformed, a stimulus id is unusable, or
            the cell holds nothing the store can carry.
    """
    scalar_file = layout.scalar_file(cell.method, cell.unit, cell.dataset, cell.tag)
    scalar = _scalar_table(cell, payload, text) if scalar_file else None
    arrays: dict[str, pa.Table] = {}
    dropped: dict[str, int] = {}
    if cell.method in layout.LENS_METHODS and cell.unit != TOKEN_UNIT:
        table = _lens_table(cell, payload, text)
        if table is not None:
            arrays[f"{layout.LENS_DIR}/{cell.tag}.parquet"] = table
    if cell.method == layout.LOOKAHEAD_METHOD:
        table = _lookahead_table(cell, payload, text)
        if table is not None:
            arrays[f"{layout.LOOKAHEAD_DIR}/{cell.tag}.parquet"] = table
    lookahead_keys = set(layout.LOOKAHEAD_COLUMNS.values())
    for key, count in sorted(_array_keys(payload).items()):
        if cell.method == layout.LOOKAHEAD_METHOD and key in lookahead_keys:
            continue
        family = layout.array_family(cell.method, key)
        if family is None:
            dropped[key] = count
            continue
        table = _family_table(cell, family, payload, text)
        if table is not None:
            arrays[family.file(cell.tag)] = table
    if scalar is None and not arrays:
        raise ValueError("no exportable values")
    return CellTables(scalar_file, scalar, arrays, dropped)


# ---- the index -------------------------------------------------------------


def table_counts(table: pa.Table) -> dict[str, tuple[int, int]]:
    """``{predictor: (n_stimuli, n_rows)}`` of one table.

    An array file has no predictor column; its single entry is keyed by the
    empty string.

    Args:
        table: Rows of one file.
    """
    if "predictor" in table.column_names:
        grouped = table.group_by(["predictor"]).aggregate(
            [("stimulus_id", "count_distinct"), ("stimulus_id", "count")]
        )
        return {
            row["predictor"]: (
                row["stimulus_id_count_distinct"],
                row["stimulus_id_count"],
            )
            for row in grouped.to_pylist()
        }
    return {"": (len(set(table["stimulus_id"].to_pylist())), table.num_rows)}


def _index_row(
    cell: Cell,
    predictor: str,
    file: str,
    n_stimuli: int,
    n_rows: int,
    verification: tuple[str, str] | None,
) -> dict:
    identity = cell.identity
    environment = cell.manifest.get("environment") or {}
    versions = identity.get("versions") or {}
    return {
        "tag": cell.tag,
        "file": file,
        "dataset": cell.dataset,
        "model": cell.info.model,
        "model_kind": cell.info.model_kind,
        "method": cell.method,
        "predictor": predictor,
        "value_unit": unit_for(predictor),
        "n_stimuli": n_stimuli,
        "n_rows": n_rows,
        **cell.index_config(),
        "provenance": "manifest",
        "chunk_size": identity.get("options", {}).get("chunk_size"),
        "device": environment.get("device"),
        "cuda": environment.get("cuda"),
        "torch": versions.get("torch"),
        "transformers": versions.get("transformers"),
        "python": versions.get("python"),
        "replica": versions.get("replica"),
        "code_commit": environment.get("code_commit"),
        "code_dirty": environment.get("code_dirty"),
        "source_sha256": identity.get("source_sha256"),
        "model_revision": (identity.get("model") or {}).get("revision"),
        "computed_at": environment.get("computed_at"),
        "layer": cell.layer,
        "study": cell.study,
        "storage_class": cell.storage_class,
        "unit": cell.unit,
        "unit_rule": cell.unit_rule,
        "context_scope": cell.context_scope,
        "derivation_source": cell.derivation_source,
        "wave": cell.wave,
        "verification": verification[0] if verification else None,
        "verification_file": verification[1] if verification else None,
    }


def _carried_index_row(row: dict, names: Iterable[str]) -> dict:
    """A published index row under the current schema.

    The storage columns come from the method. The wave, verdict and unit rule
    the published row records stay with it: a carried cell was exported by an
    earlier release, and that release's label and verdict are its provenance.
    """
    out = {name: row.get(name) for name in layout.INDEX_SCHEMA.names}
    out["tag"] = canonical_tag(str(row["tag"]))
    if row.get("method") is not None:
        out["method"] = canonical_method(str(row["method"]))
    if row.get("predictor") is not None:
        out["predictor"] = canonical_key(str(row["predictor"]))
    storage = METHOD_STORAGE.get(str(out["method"]))
    dataset = layout.match_dataset(str(out["tag"]), names)
    unit, scope = "words", "stimulus"
    if dataset is not None:
        try:
            info = layout.decode_tag(str(out["tag"]), dataset)
        except ValueError:
            info = None
        if info is not None:
            unit, scope = info.unit, info.context_scope
    # The index names the grain; the tag carries a unit suffix for some
    # methods only, so the tag's unit is the fallback.
    out.update(
        layer=storage.layer if storage else None,
        study=storage.study if storage else None,
        storage_class=storage.storage_class if storage else None,
        unit=row.get("unit") or unit,
        unit_rule=row.get("unit_rule"),
        context_scope=row.get("context_scope") or scope,
        derivation_source=row.get("derivation_source"),
        wave=row.get("wave"),
        verification=row.get("verification"),
        verification_file=row.get("verification_file"),
    )
    return out


# ---- coverage --------------------------------------------------------------


#: Why a cell the export read is not in the staged tree.
_CHANGED_DETAIL = "the pickle changed during the export"
_MISALIGNED_DETAIL = "this run's cell indexes other units than the loader produces"


def _carried_shard_detail(count: int) -> str:
    """Why a shard set's table is in the release although this run staged none.

    Args:
        count: Shard files the output directories hold for that tag.
    """
    return "carried from an earlier release; " + _shard_detail(count)


def _shard_detail(count: int) -> str:
    """Why a tag that exists only as slices is not in the staged tree.

    Args:
        count: Shard files the output directories hold for that tag.
    """
    return (
        f"unmerged shard set of {count} files; replica compute merge-shards checks the "
        "tiling and writes the canonical pickle the export reads"
    )


def _carried_detail(cell: Cell | None) -> str:
    """Why a table the release ships did not come out of this run.

    Args:
        cell: The run's own attempt at the table, when it made one.
    """
    if cell is None:
        return "carried from an earlier release"
    if cell.status == QUARANTINED:
        return "carried from an earlier release; this run quarantined its table: " + (
            cell.failure_detail() or "quarantined"
        )
    return f"carried from an earlier release; {_CHANGED_DETAIL}"


@dataclass(frozen=True)
class PlanRow:
    """One planned compute command and the tables it was to produce."""

    reason_class: str
    method: str
    dataset: str
    model: str
    tags: tuple[str, ...]
    command: str


def read_coverage(paths: Iterable[Path]) -> list[PlanRow]:
    """Read the planned commands of one or more coverage tables.

    A table is tab-separated with the columns class, method, dataset, model,
    expected files (``;``-joined) and command; a header row naming ``class``
    first is skipped.

    Args:
        paths: The tables to read.
    """
    rows: list[PlanRow] = []
    for path in paths:
        with path.open(newline="") as handle:
            for fields in csv.reader(handle, delimiter="\t"):
                if not fields or fields[0] == "class":
                    continue
                if len(fields) != 6:
                    raise ValueError(f"{path}: expected 6 columns, got {len(fields)}")
                cls, method, dataset, model, files, command = fields
                tags = tuple(
                    canonical_tag(
                        SHARD_RE.sub("", f[: -len(".pkl")] if f.endswith(".pkl") else f)
                    )
                    for f in files.split(";")
                    if f
                )
                rows.append(
                    PlanRow(
                        cls, canonical_method(method), dataset, model, tags, command
                    )
                )
    return rows


def _planned_status(
    tag: str,
    found: Discovery,
    stored: set[str],
    command: str,
) -> tuple[str, str | None]:
    """Status and detail of one planned table.

    Args:
        tag: Cell id the plan names.
        found: The cells the output directories hold.
        stored: Tags the staged index holds.
        command: The plan's command, reported when nothing ships the table.
    """
    quarantined = {cell.tag: cell for cell in found.quarantined}
    dropped = {cell.tag: cell for cell in found.dropped}
    misaligned = {cell.tag for cell in found.misaligned}
    if tag in {cell.tag for cell in found.completed}:
        return layout.EXPORTED, None
    if tag in stored:
        if tag in found.shard_sets:
            detail = _carried_shard_detail(found.shard_sets[tag])
        elif tag in misaligned:
            detail = _MISALIGNED_DETAIL
        else:
            detail = _carried_detail(quarantined.get(tag) or dropped.get(tag))
        return layout.CARRIED, detail
    if tag in found.shard_sets:
        return layout.ABSENT, _shard_detail(found.shard_sets[tag])
    if tag in quarantined:
        return layout.QUARANTINED, quarantined[tag].failure_detail() or "quarantined"
    if tag in dropped:
        return layout.ABSENT, _CHANGED_DETAIL
    return layout.ABSENT, _MISALIGNED_DETAIL if tag in misaligned else command


def _tag_row(tag: str, names: Iterable[str], **fields: object) -> dict:
    """A coverage row for a tag no plan names, decoded as far as it can be.

    Args:
        tag: Cell id.
        names: Dataset names a tag may end in.
        **fields: The row's status, reason class and detail.
    """
    dataset = layout.match_dataset(tag, names)
    info = None
    if dataset is not None:
        try:
            info = layout.decode_tag(tag, dataset)
        except ValueError:
            info = None
    return {
        "method": info.method if info else None,
        "model": info.model if info else None,
        "dataset": dataset,
        "unit": info.unit if info else None,
        "tag": tag,
        **fields,
    }


def coverage_rows(
    plans: Sequence[PlanRow],
    found: Discovery,
    stored: set[str] | None = None,
    names: Iterable[str] = (),
    undefined: Callable[[str, str, str, str], str | None] | None = None,
) -> list[dict]:
    """One coverage row per planned table, plus one per produced table no plan names.

    Args:
        plans: The planned commands.
        found: The cells the output directories hold.
        stored: Tags the staged index holds, this run's and the ones carried
            from the store. A planned table this run did not produce but the
            release still ships counts as ``carried``.
        names: Dataset names a tag may end in, to decode a tag no plan names.
        undefined: Why a planned table nothing produced or ships is undefined
            on its corpus, by dataset, method, unit and segmentation; ``None``
            when it is defined. Such a table is ``undefined`` with that reason
            as its detail instead of ``absent`` with the command.
    """
    names = list(names)
    stored = stored or set()
    rows: list[dict] = []
    planned: set[str] = set()
    for plan in plans:
        for tag in plan.tags:
            planned.add(tag)
            unit, segmentation = "words", "whitespace"
            decoded = True
            try:
                info = layout.decode_tag(tag, plan.dataset)
                unit, segmentation = info.unit, info.segmentation
            except ValueError:
                decoded = False
            status, detail = _planned_status(tag, found, stored, plan.command)
            # The command is the detail only when nothing produced or ships the
            # table; a tag the grammar cannot decode names no unit to decide.
            if (
                status == layout.ABSENT
                and detail is plan.command
                and undefined
                and decoded
            ):
                reason = undefined(plan.dataset, plan.method, unit, segmentation)
                if reason is not None:
                    status, detail = layout.UNDEFINED, reason
            rows.append(
                {
                    "method": plan.method,
                    "model": plan.model,
                    "dataset": plan.dataset,
                    "unit": unit,
                    "status": status,
                    "reason_class": plan.reason_class,
                    "detail": detail,
                    "tag": tag,
                }
            )
    for tag, count in sorted(found.shard_sets.items()):
        if tag in planned:
            continue
        carried = tag in stored
        rows.append(
            _tag_row(
                tag,
                names,
                status=layout.CARRIED if carried else layout.ABSENT,
                reason_class=None,
                detail=_carried_shard_detail(count)
                if carried
                else _shard_detail(count),
            )
        )
    produced = (
        (layout.EXPORTED, {cell.tag: cell for cell in found.completed}),
        (layout.QUARANTINED, {cell.tag: cell for cell in found.quarantined}),
        (layout.ABSENT, {cell.tag: cell for cell in found.dropped}),
    )
    for status, cells in produced:
        for tag, cell in sorted(cells.items()):
            if tag in planned:
                continue
            if status == layout.EXPORTED:
                detail = None
            elif tag in stored:
                status, detail = layout.CARRIED, _carried_detail(cell)
            elif status == layout.ABSENT:
                detail = _CHANGED_DETAIL
            else:
                detail = cell.failure_detail()
            rows.append(
                {
                    "method": cell.method,
                    "model": cell.info.model,
                    "dataset": cell.dataset,
                    "unit": cell.unit,
                    "status": status,
                    "reason_class": None,
                    "detail": detail,
                    "tag": tag,
                }
            )
    # A plan may name the same cell twice, once per spelling of an option that
    # does not reach the tag; keep one coverage row per table.
    seen: set[tuple] = set()
    unique = []
    for row in rows:
        key = tuple(sorted(row.items(), key=lambda kv: kv[0]))
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return unique


# ---- the export ------------------------------------------------------------


@dataclass
class ExportReport:
    """What one export wrote, and what it could not."""

    out: str
    cells: int = 0
    by_unit_layer: dict[str, int] = field(default_factory=dict)
    quarantined: int = 0
    coverage_rows: int = 0
    index_rows: int = 0
    carried_index_rows: int = 0
    rows_written: int = 0
    files_written: list[str] = field(default_factory=list)
    files_deleted: list[str] = field(default_factory=list)
    replaced_tags: list[str] = field(default_factory=list)
    exported_tags: list[str] = field(default_factory=list)
    base_revision: str | None = None
    failures: list[str] = field(default_factory=list)
    discovery_problems: list[str] = field(default_factory=list)
    dropped_arrays: dict[str, int] = field(default_factory=dict)
    changed_during_export: list[str] = field(default_factory=list)
    appeared_during_export: list[str] = field(default_factory=list)
    unmerged_shard_sets: dict[str, int] = field(default_factory=dict)
    shard_verification_records: dict[str, int] = field(default_factory=dict)
    carried_count_mismatches: list[str] = field(default_factory=list)
    misaligned_cells: dict[str, str] = field(default_factory=dict)
    misaligned_carried: dict[str, str] = field(default_factory=dict)
    undefined_carried: dict[str, str] = field(default_factory=dict)
    retired_by_request: dict[str, str] = field(default_factory=dict)
    retired_tags: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Whether every discovered cell was exported."""
        return not self.failures and not self.discovery_problems

    def summary(self) -> str:
        """A short account of the export."""
        lines = [
            f"stage: {self.out}",
            f"cells exported: {self.cells} ("
            + ", ".join(f"{k}: {v}" for k, v in sorted(self.by_unit_layer.items()))
            + ")",
            f"quarantined: {self.quarantined}",
            f"coverage rows: {self.coverage_rows}",
            f"index rows: {self.index_rows} ({self.carried_index_rows} carried)",
            f"rows written: {self.rows_written:,}",
            f"files written: {len(self.files_written)}, "
            f"to delete: {len(self.files_deleted)}",
        ]
        if self.unmerged_shard_sets:
            lines.append(
                "unmerged shard sets, not staged (run replica compute merge-shards first): "
                + ", ".join(
                    f"{tag} ({count} shards)"
                    for tag, count in sorted(self.unmerged_shard_sets.items())[:3]
                )
                + (
                    f" and {len(self.unmerged_shard_sets) - 3} more"
                    if len(self.unmerged_shard_sets) > 3
                    else ""
                )
            )
        if self.changed_during_export:
            lines.append(
                f"cells dropped, their pickle changed during the export: "
                f"{len(self.changed_during_export)} "
                f"({', '.join(self.changed_during_export[:3])})"
            )
        if self.appeared_during_export:
            lines.append(
                f"cells that appeared during the export, not staged: "
                f"{len(self.appeared_during_export)} "
                f"({', '.join(self.appeared_during_export[:3])})"
            )
        if self.shard_verification_records:
            lines.append(
                "shard records dropped from the verification file: "
                + ", ".join(
                    f"{wave} ({count})"
                    for wave, count in sorted(self.shard_verification_records.items())
                )
            )
        if self.dropped_arrays:
            lines.append(f"array keys without a folder: {self.dropped_arrays}")
        for mismatch in self.carried_count_mismatches:
            lines.append(f"carried index disagrees with its table: {mismatch}")
        if self.retired_tags:
            lines.append(
                f"published cells retired: {len(self.retired_tags)} "
                f"({', '.join(self.retired_tags[:3])})"
            )
        if self.misaligned_carried and not self.retired_tags:
            lines.append(
                f"published cells whose rows index other units than the loader "
                f"produces (pass --drop-misaligned to retire them): "
                f"{len(self.misaligned_carried)}"
            )
        undefined_kept = [
            t for t in self.undefined_carried if t not in self.retired_tags
        ]
        if undefined_kept:
            lines.append(
                f"published cells carried although the unit contract finds them "
                f"undefined on their corpus: {len(undefined_kept)}"
            )
        for problem in self.discovery_problems:
            lines.append(f"skipped: {problem}")
        for failure in self.failures:
            lines.append(f"failed: {failure}")
        return "\n".join(lines)


class _Writers:
    """Parquet writers opened on demand and closed together."""

    def __init__(self, out: Path):
        self.out = out
        self._open: dict[str, pq.ParquetWriter] = {}
        self.counts: dict[tuple[str, str, str], tuple[int, int]] = {}
        self.rows = 0

    def write(self, file: str, table: pa.Table, tag: str | None) -> None:
        """Append *table* to *file*; with a *tag*, count its rows per predictor."""
        path = self.out / file
        path.parent.mkdir(parents=True, exist_ok=True)
        writer = self._open.get(file)
        if writer is None:
            writer = pq.ParquetWriter(path, table.schema, compression="zstd")
            self._open[file] = writer
        writer.write_table(table)
        self.rows += table.num_rows
        if tag is None:
            return
        for predictor, (n_stimuli, n_rows) in table_counts(table).items():
            key = (tag, file, predictor or layout.index_key_of_file(file) or "")
            earlier = self.counts.get(key, (0, 0))
            self.counts[key] = (earlier[0] + n_stimuli, earlier[1] + n_rows)

    def files(self) -> list[str]:
        """Files opened so far."""
        return sorted(self._open)

    def close(self) -> list[str]:
        """Close every writer and return the files written."""
        for writer in self._open.values():
            writer.close()
        files = sorted(self._open)
        self._open.clear()
        return files


def published_shared_tables(
    hub: Hub,
    file: str,
    dataset: str,
    names: list[str],
    *,
    exclude: set[str] | None = None,
    include: set[str] | None = None,
) -> Iterator[pa.Table]:
    """Published rows of a shared per-corpus table, batch by batch.

    The rows come back under the schema the staged file uses, so a published
    table that lacks a column carries it as a null or, for the storage columns
    of a word table, as the value the row's method fixes.

    Args:
        hub: The published store to read.
        file: Path of the table within the store.
        dataset: Corpus the table belongs to; it names the ``dataset`` column,
            which some published rows spell with their shard slice suffix.
        names: Dataset names a tag may end in, to read each tag's method.
        exclude: Tags to leave out.
        include: Tags to keep, to the exclusion of every other.

    Raises:
        ValueError: When a published stimulus id cannot be joined.
    """
    schema = layout.shared_schema(file)
    storage = "storage_class" in schema.names
    source = pq.ParquetFile(hub.download(file))
    storage_of: dict[str, tuple[int | None, str | None]] = {}
    for batch in source.iter_batches(batch_size=_BATCH_ROWS):
        table = _current_names(pa.Table.from_batches([batch]))
        tags = np.asarray(table["tag"].to_pylist(), dtype=object)
        if include is not None:
            table = table.filter(pa.array(np.isin(tags, sorted(include))))
        elif exclude:
            table = table.filter(pa.array(~np.isin(tags, sorted(exclude))))
        if not table.num_rows:
            continue
        n = table.num_rows
        tags = table["tag"].to_pylist()
        layout.check_ids(set(table["stimulus_id"].to_pylist()), f"{hub.repo}:{file}")
        for tag in set(tags).difference(storage_of):
            method = METHOD_STORAGE.get(_method_of(tag, names))
            storage_of[tag] = (
                (method.layer, method.storage_class) if method else (None, None)
            )
        columns: dict[str, pa.Array] = {}
        for name in schema.names:
            kind = schema.field(name).type
            if name == "dataset":
                columns[name] = pa.array([dataset] * n, type=kind)
            elif name in table.column_names:
                columns[name] = table[name].cast(kind)
            elif storage and name == "layer":
                columns[name] = pa.array([storage_of[t][0] for t in tags], type=kind)
            elif storage and name == "storage_class":
                columns[name] = pa.array([storage_of[t][1] for t in tags], type=kind)
            else:
                columns[name] = pa.nulls(n, type=kind)
        yield pa.table(columns, schema=schema)


def _carry_root_rows(
    hub: Hub,
    file: str,
    dataset: str,
    replaced: set[str],
    writers: _Writers,
    names: list[str],
) -> dict[tuple[str, str], tuple[int, int]]:
    """Write the published rows of *file* whose tag this run does not replace.

    Returns ``{(tag, predictor): (n_stimuli, n_rows)}`` of the rows carried.
    """
    stimuli: dict[tuple[str, str], set[str]] = defaultdict(set)
    rows: dict[tuple[str, str], int] = defaultdict(int)
    for table in published_shared_tables(hub, file, dataset, names, exclude=replaced):
        for tag, pred, sid in zip(
            table["tag"].to_pylist(),
            table["predictor"].to_pylist(),
            table["stimulus_id"].to_pylist(),
            strict=True,
        ):
            stimuli[tag, pred].add(sid)
            rows[tag, pred] += 1
        writers.write(file, table, None)
    return {key: (len(sids), rows[key]) for key, sids in stimuli.items()}


def _current_names(table: pa.Table) -> pa.Table:
    """*table* with its ``tag`` and ``predictor`` columns under current names."""
    for column, canonical in (("tag", canonical_tag), ("predictor", canonical_key)):
        if column not in table.column_names:
            continue
        values = table[column].to_pylist()
        renamed = [None if v is None else canonical(str(v)) for v in values]
        if renamed != values:
            index = table.column_names.index(column)
            table = table.set_column(
                index, column, pa.array(renamed, type=table.schema.field(column).type)
            )
    return table


def _renamed_tags(hub: Hub) -> dict[str, str]:
    """Published tags under a renamed method, old to current."""
    tags = {str(tag) for tag in hub.index()["tag"].to_pylist()}
    return {tag: canonical_tag(tag) for tag in tags if canonical_tag(tag) != tag}


def _method_of(tag: str, names: list[str]) -> str:
    dataset = layout.match_dataset(tag, names)
    if dataset is None:
        return ""
    try:
        return layout.decode_tag(tag, dataset).method
    except ValueError:
        return ""


def read_verifications(
    sources: Iterable[tuple[str, Path]], out: Path, report: ExportReport | None = None
) -> dict[str, tuple[str, str]]:
    """Copy supplied verification records and map each tag to its verdict.

    These labels are not authenticated against the source payload digests.

    A record whose tag carries a shard suffix judges one slice of a corpus. The
    cell those slices merge into carries its own record, so a shard record is
    dropped. Discovery skips shard files for the same reason: a slice compared
    against the whole corpus raises flags that belong to the slice alone. A
    record under an earlier method name is staged under the current one.

    Args:
        sources: ``(wave label, verification.json)`` pairs; each file is a
            list of records carrying ``tag`` and ``verdict``.
        out: The stage root.
        report: Receives the count of dropped shard records per wave.
    """
    verdicts: dict[str, tuple[str, str]] = {}
    for wave, path in sources:
        records = json.loads(path.read_text())
        if not isinstance(records, list):
            raise ValueError(f"{path}: expected a list of per-table records")
        cells = [r for r in records if not SHARD_RE.search(str(r["tag"]))]
        dropped = len(records) - len(cells)
        current = [{**r, "tag": canonical_tag(str(r["tag"]))} for r in cells]
        target = f"{layout.VERIFICATION_DIR}/{wave}.json"
        (out / layout.VERIFICATION_DIR).mkdir(parents=True, exist_ok=True)
        if dropped or current != cells:
            (out / target).write_text(json.dumps(current))
        else:
            shutil.copyfile(path, out / target)
        if dropped:
            if report is not None:
                report.shard_verification_records[wave] = dropped
            log.warning("%s: %d shard records dropped from %s", path, dropped, target)
        for record in current:
            verdicts[str(record["tag"])] = (str(record.get("verdict")), target)
    return verdicts


def _write_side_tables(out: Path, index_rows: list[dict], coverage: list[dict]) -> None:
    pq.write_table(
        pa.Table.from_pylist(
            [
                {name: row.get(name) for name in layout.INDEX_SCHEMA.names}
                for row in index_rows
            ],
            schema=layout.INDEX_SCHEMA,
        ),
        out / layout.INDEX_FILE,
        compression="zstd",
    )
    pq.write_table(
        pa.Table.from_pylist(coverage, schema=layout.COVERAGE_SCHEMA),
        out / layout.COVERAGE_FILE,
        compression="zstd",
    )


def _text_index(
    dataset: str, corpora: Corpora, text_cache: dict[str, TextIndex | Exception]
) -> TextIndex:
    if dataset not in text_cache:
        log.info("loading %s", dataset)
        try:
            text_cache[dataset] = TextIndex(
                corpora.stimuli(dataset), corpora.word_spacing(dataset)
            )
        except Exception as exc:
            # A corpus that fails to load fails the same way for every cell;
            # the failure is kept so it is raised again without a reload.
            text_cache[dataset] = exc
            raise
    cached = text_cache[dataset]
    if isinstance(cached, Exception):
        raise cached
    return cached


def _undefined_reason(
    dataset: str,
    method: str,
    unit: str,
    segmentation: str,
    corpora: Corpora,
    text_cache: dict[str, TextIndex | Exception],
    unit_rule: str | None = None,
) -> str | None:
    """Why the measure of *method* is undefined on *dataset* at *unit*, or ``None``.

    ``None`` also when the corpus cannot be loaded or the method is unknown to
    the compute driver, which the coverage row then reports as absent.
    """
    from replica.compute.units import TOKEN_UNIT, applicability

    try:
        text = _text_index(dataset, corpora, text_cache)
        if unit not in ("words", CHARACTER_LAYER, TOKEN_UNIT):
            text.grid(unit)
        fit = applicability(
            method,
            unit,
            text.stimuli,
            word_spacing=text.word_spacing,
            unit_rule=unit_rule,
            segmentation=segmentation,
        )
    except (OSError, KeyError, ValueError) as exc:
        log.warning("applicability of %s on %s not decided: %s", method, dataset, exc)
        return None
    return None if fit.defined else f"{dataset}: {fit.reason}"


def _carried_undefined_reason(
    dataset: str,
    tag: str,
    method: str | None,
    unit: str,
    unit_rule: str | None,
    corpora: Corpora,
    text_cache: dict[str, TextIndex | Exception],
) -> str | None:
    """Why a published cell has no place under the unit contract, or ``None``.

    A recorded boundary rule other than the method's default is a named rule,
    and the contract judges the cell under it, as the runner would. The
    default rule is recorded for every cell, so it does not show that a
    boundary was chosen, and the contract decides without it. On a corpus without word spacing the word table
    of ``surprisal`` is the ``char_beam`` table, so a published word cell
    under the ``surprisal`` tag has no producer there.
    """
    from replica.compute.units import boundary_rule

    if method is None:
        return None
    try:
        segmentation = layout.decode_tag(tag, dataset).segmentation
    except ValueError:
        segmentation = "whitespace"
    try:
        default_rule = boundary_rule(method, None)
    except (KeyError, ValueError):
        default_rule = None
    named = unit_rule if unit_rule and unit_rule != default_rule else None
    if method == "surprisal" and unit == "words":
        try:
            text = _text_index(dataset, corpora, text_cache)
        except Exception:
            return None
        if not text.word_spacing:
            return (
                f"{dataset}: word surprisal on a corpus without word spacing is "
                "the char_beam table"
            )
        return None
    return _undefined_reason(
        dataset, method, unit, segmentation, corpora, text_cache, unit_rule=named
    )


def _export_cell(
    cell: Cell,
    corpora: Corpora,
    text_cache: dict[str, TextIndex | Exception],
    writers: _Writers,
) -> CellTables:
    text = _text_index(cell.dataset, corpora, text_cache)
    payload = load_payload(cell)
    check_payload(cell, payload, text)
    log.info("exporting %s (%d stimuli)", cell.stem, len(payload))
    tables = cell_tables(cell, payload, text)
    if tables.scalar is not None and tables.scalar_file is not None:
        writers.write(tables.scalar_file, tables.scalar, cell.tag)
    for file, table in tables.arrays.items():
        writers.write(file, table, cell.tag)
    # The manifest is staged with the rows, so a staged cell always has one.
    manifests = writers.out / layout.MANIFESTS_DIR
    manifests.mkdir(parents=True, exist_ok=True)
    (manifests / f"{cell.stem}.json").write_text(manifest_text(cell.manifest))
    return tables


def _rewrite_without(
    out: Path,
    file: str,
    tags: set[str],
    hub: Hub | None,
    names: list[str],
) -> dict[tuple[str, str], tuple[int, int]]:
    """Rewrite one staged file without *tags*, restoring their published rows.

    A per-corpus table holds many cells, so a cell dropped after it was
    written is filtered out of the file rather than taking the file with it.
    When the store has published rows for a dropped cell, they go back in, so
    the release neither ships the half-written cell nor loses the old one.

    Returns:
        ``{(tag, predictor): (n_stimuli, n_rows)}`` of the restored rows.
    """
    path = out / file
    temporary = path.with_name(path.name + ".rewrite")
    source = pq.ParquetFile(path)
    schema = source.schema_arrow
    stimuli: dict[tuple[str, str], set[str]] = defaultdict(set)
    counted: dict[tuple[str, str], int] = defaultdict(int)
    rows = 0
    with pq.ParquetWriter(temporary, schema, compression="zstd") as writer:
        for batch in source.iter_batches(batch_size=_BATCH_ROWS):
            table = pa.Table.from_batches([batch])
            keep = ~np.isin(
                np.asarray(table["tag"].to_pylist(), dtype=object), sorted(tags)
            )
            table = table.filter(pa.array(keep))
            if table.num_rows:
                writer.write_table(table)
                rows += table.num_rows
        if (
            hub is not None
            and layout.is_shared_table(file)
            and file in _published_files(hub)
        ):
            for table in published_shared_tables(
                hub, file, layout.dataset_of_file(file), names, include=tags
            ):
                for tag, predictor, sid in zip(
                    table["tag"].to_pylist(),
                    table["predictor"].to_pylist(),
                    table["stimulus_id"].to_pylist(),
                    strict=True,
                ):
                    stimuli[tag, predictor].add(sid)
                    counted[tag, predictor] += 1
                writer.write_table(table.cast(schema))
                rows += table.num_rows
    if rows:
        temporary.replace(path)
    else:
        temporary.unlink()
        path.unlink()
    return {key: (len(sids), counted[key]) for key, sids in stimuli.items()}


def _drop_changed_cells(
    out: Path,
    dropped: list[Cell],
    writers: _Writers,
    hub: Hub | None,
    names: list[str],
) -> dict[str, dict[tuple[str, str], tuple[int, int]]]:
    """Take the cells whose pickle changed back out of the staged tree.

    Args:
        out: The stage root.
        dropped: Cells to remove, with their rows already written.
        writers: The closed writers, whose counts name each cell's files.
        hub: The published store, to restore the rows of a dropped cell.
        names: Dataset names a tag may end in.

    Returns:
        Per file, the ``{(tag, predictor): (n_stimuli, n_rows)}`` it restored.
    """
    tags = {cell.tag for cell in dropped}
    files = sorted({file for tag, file, _ in writers.counts if tag in tags})
    restored: dict[str, dict[tuple[str, str], tuple[int, int]]] = {}
    for file in files:
        owners = {tag for tag, other, _ in writers.counts if other == file}
        restorable = (
            hub is not None
            and layout.is_shared_table(file)
            and file in _published_files(hub)
        )
        if owners <= tags and not restorable:
            (out / file).unlink(missing_ok=True)
        else:
            counts = _rewrite_without(out, file, tags, hub, names)
            if counts:
                restored[file] = counts
    for key in [key for key in writers.counts if key[0] in tags]:
        del writers.counts[key]
    for cell in dropped:
        (out / layout.MANIFESTS_DIR / f"{cell.stem}.json").unlink(missing_ok=True)
    return restored


def _published_files(hub: Hub) -> set[str]:
    """The hub's file listing plus every file its index names.

    A table the index names is carried even when the listing omits it, and
    a download of it that fails stops the export.
    """
    return hub.files() | set(hub.index()["file"].to_pylist())


def _shared_tables_to_rewrite(
    hub: Hub, replaced: set[str], written: Iterable[str]
) -> set[str]:
    """Shared tables holding fresh rows or published rows of a replaced tag.

    Args:
        hub: The published store.
        replaced: Tags this run stages.
        written: Files this run has written.
    """
    published = _published_files(hub)
    files = {
        file for file in written if layout.is_shared_table(file) and file in published
    }
    index = hub.index()
    renamed = _renamed_tags(hub)
    for file, tag in zip(
        index["file"].to_pylist(), index["tag"].to_pylist(), strict=True
    ):
        if (
            (tag in replaced or tag in renamed)
            and layout.is_shared_table(file)
            and file in published
        ):
            files.add(file)
    return files


def unindexed_shared_tables(hub: Hub, files: Iterable[str]) -> list[str]:
    """Shared tables the store holds without an index row, out of *files*.

    Rewriting one would drop every cell in it, since the index is what says
    which cells the file holds.

    Args:
        hub: The published store.
        files: Shared tables this run is about to write.
    """
    published = hub.files()
    indexed = set(hub.index()["file"].to_pylist())
    return sorted(
        file
        for file in files
        if layout.is_shared_table(file) and file in published and file not in indexed
    )


def read_retirements(path: Path) -> dict[str, str]:
    """The retirement list at *path*: one tag and its reason per line, tab-separated.

    Args:
        path: The list to read; blank lines and lines starting with ``#``
            are skipped.

    Raises:
        ValueError: A line without a tab-separated reason, or a repeated tag.
    """
    retire: dict[str, str] = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        tag, sep, reason = line.partition("\t")
        tag, reason = tag.strip(), reason.strip()
        if not sep or not reason:
            raise ValueError(f"{path}:{number}: expected 'tag<TAB>reason'")
        if tag in retire:
            raise ValueError(f"{path}:{number}: {tag} listed twice")
        retire[tag] = reason
    return retire


def _requested_retirements(
    hub: Hub, retire: Mapping[str, str], staged: set[str], report: ExportReport
) -> set[str]:
    """The carried tags the caller retires, each reason recorded.

    Returns:
        The listed tags the store holds and this run does not stage fresh;
        any other listed tag fails the export.
    """
    index = hub.index()
    published = {
        canonical_tag(str(tag)): layout.dataset_of_file(str(file))
        for tag, file in zip(
            index["tag"].to_pylist(), index["file"].to_pylist(), strict=True
        )
    }
    retired: set[str] = set()
    for tag, reason in sorted(retire.items()):
        tag = canonical_tag(tag)
        if tag in staged:
            report.failures.append(
                f"{tag}: listed for retirement but this run stages it fresh"
            )
        elif tag not in published:
            report.failures.append(f"{tag}: listed for retirement but not published")
        else:
            # The dataset prefix keys the README's per-corpus count, as for
            # the misalignment reasons.
            report.retired_by_request[tag] = f"{published[tag]}: {reason}"
            log.warning("retiring %s: %s", tag, reason)
            retired.add(tag)
    return retired


def _contract_pass(
    dataset: str,
    tags: dict[str, str],
    recorded: dict[str, tuple[str | None, str | None]],
    corpora: Corpora,
    text_cache: dict[str, TextIndex | Exception],
    report: ExportReport,
) -> None:
    """Record the carried cells of *dataset* the unit contract finds undefined.

    The finding is reported and the cell carried unchanged; retirement is
    decided outside the library and passed in as an explicit tag list.
    """
    for tag, unit in sorted(tags.items()):
        method, rule = recorded.get(tag, (None, None))
        reason = _carried_undefined_reason(
            dataset, tag, method, unit, rule, corpora, text_cache
        )
        if reason is not None:
            report.undefined_carried[tag] = reason
            log.warning("carried although undefined: %s: %s", tag, reason)


def _carried_alignment(
    hub: Hub,
    corpora: Corpora,
    text_cache: dict[str, TextIndex | Exception],
    staged: set[str],
    names: list[str],
    report: ExportReport,
    drop: bool,
) -> set[str]:
    """Check every carried word and character cell against its contract and loader.

    A cell whose measure the unit contract finds undefined on its corpus, or
    whose word surprisal the corpus derives from the character table, is
    reported and carried. The rows of each per-corpus table are then read
    once and checked per cell: a cell off the loader's grid fails the export
    or, with *drop*, is retired. A retired tag is returned, and the merge
    removes it as it removes a replaced cell, with no fresh rows in its place.

    Returns:
        The tags to retire.
    """
    # A cell's grain comes from its index row: a root table holds character
    # cells beside word cells, so the file alone does not decide it.
    by_file: dict[str, dict[str, str]] = defaultdict(dict)
    recorded: dict[str, tuple[str | None, str | None]] = {}
    index = hub.index()

    def column(name: str) -> list:
        return (
            index[name].to_pylist()
            if name in index.column_names
            else [None] * index.num_rows
        )

    for file, tag, unit, method, rule in zip(
        index["file"].to_pylist(),
        index["tag"].to_pylist(),
        column("unit"),
        column("method"),
        column("unit_rule"),
        strict=True,
    ):
        tag = canonical_tag(str(tag))
        if tag in staged or not layout.is_shared_table(file):
            continue
        if unit is None:
            unit = layout.decode_tag(tag, layout.dataset_of_file(file)).unit
        if unit == TOKEN_UNIT:
            continue
        by_file[file][tag] = unit
        recorded[tag] = (
            None if method is None else str(method),
            None if rule is None else str(rule),
        )
    retired: set[str] = set()
    for file, tags in sorted(by_file.items()):
        dataset = layout.dataset_of_file(file)
        try:
            text = _text_index(dataset, corpora, text_cache)
        except Exception as exc:  # the other corpora are still checked
            report.failures.append(
                f"{file}: the {dataset} loader failed, so its {len(tags)} carried "
                f"cells could not be checked: {type(exc).__name__}: {exc}"
            )
            continue
        _contract_pass(dataset, tags, recorded, corpora, text_cache, report)
        checks: dict[str, RowCheck] = {}
        for tag, unit in tags.items():
            try:
                checks[tag] = RowCheck(text.grid(unit))
            except ValueError as exc:
                report.failures.append(f"{tag}: {exc}")
        tags = {tag: unit for tag, unit in tags.items() if tag in checks}
        if not tags:
            continue
        for table in published_shared_tables(
            hub, file, dataset, names, include=set(tags)
        ):
            for tag, sid, index, text, key in zip(
                table["tag"].to_pylist(),
                table["stimulus_id"].to_pylist(),
                table["unit_index"].to_pylist(),
                table["text_unit"].to_pylist(),
                table["predictor"].to_pylist(),
                strict=True,
            ):
                checks[tag].add(sid, index, text, key)
        for tag in sorted(tags):
            problems = checks[tag].problems()
            if not problems:
                continue
            detail = f"{dataset}: {describe(problems)}"
            report.misaligned_carried[tag] = detail
            if drop:
                log.warning("retiring %s: %s", tag, detail)
                retired.add(tag)
            else:
                report.failures.append(
                    f"{tag}: published rows index other units than the {dataset} "
                    f"loader produces: {describe(problems)}"
                )
    return retired


def _merge_hub_index(
    hub: Hub,
    fresh_rows: list[dict],
    replaced: set[str],
    written: set[str],
    carried_counts: dict[str, dict[tuple[str, str], tuple[int, int]]],
    names: list[str],
    report: ExportReport,
) -> tuple[list[dict], list[str]]:
    """Fresh index rows plus the published rows this release keeps.

    A published row is kept when its tag is not replaced and its file is
    either untouched or a root table this run rewrote with the row's cell
    carried over. A published file whose tag is replaced and which this run
    did not rewrite is listed for deletion.
    """
    carried: list[dict] = []
    deleted: set[str] = set()
    for row in hub.index().to_pylist():
        file, tag = str(row["file"]), canonical_tag(str(row["tag"]))
        if tag in replaced:
            if file not in written and not layout.is_shared_table(file):
                deleted.add(file)
            continue
        if file in carried_counts:
            counted = carried_counts[file].get(
                (tag, canonical_key(str(row["predictor"])))
            )
            if counted is None:
                report.carried_count_mismatches.append(
                    f"{file}: {tag}/{row['predictor']} indexed but absent from the table"
                )
                continue
            if counted != (row.get("n_stimuli"), row.get("n_rows")):
                report.carried_count_mismatches.append(
                    f"{file}: {tag}/{row['predictor']} indexed {row.get('n_stimuli')}/"
                    f"{row.get('n_rows')}, table holds {counted[0]}/{counted[1]}"
                )
                row = {**row, "n_stimuli": counted[0], "n_rows": counted[1]}
        elif file in written:
            continue
        carried.append(_carried_index_row(row, names))
    report.carried_index_rows = len(carried)
    return carried + fresh_rows, sorted(deleted)


def _restage_renamed_manifests(
    hub: Hub, out: Path, gone: set[str], published: set[str]
) -> set[str]:
    """Stage the manifests of carried cells under a renamed method under the current name.

    Returns:
        The published manifest paths the rename leaves behind.
    """
    left: set[str] = set()
    for old, new in _renamed_tags(hub).items():
        source = f"{layout.MANIFESTS_DIR}/{old}.json"
        if source not in published:
            continue
        left.add(source)
        if new in gone:
            continue
        manifest = canonical_manifest(json.loads(hub.download(source).read_text()))
        target = out / layout.MANIFESTS_DIR / f"{new}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(manifest_text(manifest))
    return left


def _refuse_unindexed_targets(hub: Hub | None, found: Discovery) -> None:
    """Stop before writing a shared table the store holds without an index row.

    Args:
        hub: The published store, or ``None`` when nothing is merged.
        found: The cells this run would stage.

    Raises:
        ValueError: When such a table is among this run's targets.
    """
    if hub is None:
        return
    targets = {
        layout.scalar_file(cell.method, cell.unit, cell.dataset, cell.tag) or ""
        for cell in found.completed
    }
    unindexed = unindexed_shared_tables(hub, targets)
    if unindexed:
        raise ValueError(
            "the store holds these tables without an index row, so this release "
            "cannot say which cells they hold and would drop them: "
            + ", ".join(unindexed)
        )


def _stage_cells(
    found: Discovery,
    corpora: Corpora,
    text_cache: dict[str, TextIndex | Exception],
    writers: _Writers,
    report: ExportReport,
) -> list[Cell]:
    """Write every completed cell, reporting the ones that cannot be read.

    A cell whose records index other units than the loader produces is not
    written; it moves to ``found.misaligned`` and fails the export.

    Args:
        found: The cells the output directories hold.
        corpora: Where the text units come from.
        text_cache: Unit strings per corpus, filled as corpora are loaded.
        writers: The open Parquet writers.
        report: Report to record failures and per-cell notes in.

    Returns:
        The cells whose rows reached the stage.

    Raises:
        OSError: When the filesystem is out of room or over quota, which
            every later cell would hit as well.
    """
    exported: list[Cell] = []
    misaligned: list[Cell] = []
    for cell in found.completed:
        try:
            tables = _export_cell(cell, corpora, text_cache, writers)
        except MisalignedCell as exc:
            report.misaligned_cells[cell.stem] = str(exc)
            report.failures.append(
                f"{cell.stem}: indexes other units than the {cell.dataset} loader "
                f"produces: {exc}"
            )
            log.warning("misaligned %s: %s", cell.stem, exc)
            misaligned.append(cell)
            continue
        except OSError as exc:
            if exc.errno in _FATAL_ERRNOS:
                raise OSError(
                    f"{cell.stem}: {exc.strerror or exc}; the stage cannot be "
                    "written, so the export stops rather than staging part of it"
                ) from exc
            report.failures.append(f"{cell.stem}: {type(exc).__name__}: {exc}")
            log.warning("failed %s: %s", cell.stem, exc)
            continue
        except Exception as exc:  # one unreadable cell must not stop the rest
            report.failures.append(f"{cell.stem}: {type(exc).__name__}: {exc}")
            log.warning("failed %s: %s", cell.stem, exc)
            continue
        exported.append(cell)
        for key, count in tables.dropped.items():
            report.dropped_arrays[key] = report.dropped_arrays.get(key, 0) + count
    if misaligned:
        found.completed = [cell for cell in found.completed if cell not in misaligned]
        found.misaligned = misaligned
    return exported


def _settle_changed_cells(
    out: Path,
    found: Discovery,
    exported: list[Cell],
    writers: _Writers,
    hub: Hub | None,
    names: list[str],
    carried_counts: dict[str, dict[tuple[str, str], tuple[int, int]]],
    report: ExportReport,
) -> list[Cell]:
    """Drop the cells whose pickle moved while the export was reading it.

    The compute driver publishes a cell by renaming a candidate over the
    pickle, so a run still producing cells can replace one under the export.
    Such a cell is taken back out rather than staged part old and part new,
    and the rows the store already had for it go back in.

    Args:
        out: The stage root.
        found: The discovery, whose completed and dropped lists are updated.
        exported: The cells written so far.
        writers: The closed writers.
        hub: The published store, to restore a dropped cell's rows.
        names: Dataset names a tag may end in.
        carried_counts: Per-file carried counts, extended with what is restored.
        report: Report to list the dropped cells in.

    Returns:
        The cells that remain staged.
    """
    changed = [cell for cell in exported if cell.changed()]
    if not changed:
        return exported
    for cell in changed:
        log.warning("%s changed during the export; dropping it", cell.stem)
    restored = _drop_changed_cells(out, changed, writers, hub, names)
    for file, counts in restored.items():
        carried_counts.setdefault(file, {}).update(counts)
    remaining = [cell for cell in exported if cell not in changed]
    found.completed = list(remaining)
    found.dropped = list(changed)
    report.changed_during_export = sorted(cell.stem for cell in changed)
    return remaining


def export(
    out: Path,
    sources: Sequence[tuple[str, Path]],
    *,
    verifications: Sequence[tuple[str, Path]] = (),
    coverage: Sequence[Path] = (),
    hub: Hub | None = None,
    corpora: Corpora | None = None,
    drop_misaligned: bool = False,
    retire: Mapping[str, str] | None = None,
) -> ExportReport:
    """Stage a release of the store under *out*.

    Every fresh cell is checked against its corpus loader, and so is every
    published cell the merge would carry. A fresh cell off the loader's grid
    fails the export and keeps the published rows; a carried one fails the
    export too, unless *drop_misaligned* retires it.

    Args:
        out: Directory to write; created when absent, files inside replaced.
        sources: ``(wave label, output directory)`` pairs holding the pickles.
        verifications: ``(wave label, verification.json)`` pairs of advisory
            labels; these do not replace source-to-stage readback.
        coverage: Coverage tables listing every planned command.
        hub: The published store to merge; ``None`` stages the fresh cells alone.
        corpora: Where the text units come from; the dataset registry by default.
        drop_misaligned: Retire the carried cells whose rows index other units
            than their loader produces: their rows, index entries and
            manifests leave the store.
        retire: Carried cells to retire by tag, each with the reason the
            caller recorded for it; a listed tag the store does not hold, or
            that this run stages fresh, fails the export.

    Returns:
        The report also written to ``export_report.json`` in the stage.
    """
    out.mkdir(parents=True, exist_ok=True)
    (out / layout.READBACK_FILE).unlink(missing_ok=True)
    corpora = corpora or RegistryCorpora()
    names = corpora.names()
    found = discover(list(sources), names)
    report = ExportReport(
        out=str(out),
        discovery_problems=list(found.problems),
        unmerged_shard_sets=dict(found.shard_sets),
    )
    for tag, count in found.shard_sets.items():
        log.warning("%s exists only as %d shard files; not staged", tag, count)
    _refuse_unindexed_targets(hub, found)
    verdicts = read_verifications(verifications, out, report)
    writers = _Writers(out)
    text_cache: dict[str, TextIndex | Exception] = {}
    exported = _stage_cells(found, corpora, text_cache, writers, report)
    staged = {cell.tag for cell in exported}
    retired: set[str] = set()
    carried_counts: dict[str, dict[tuple[str, str], tuple[int, int]]] = {}
    if hub is not None:
        # Requested retirements come first, so a listed cell is neither
        # checked against its loader nor needs --drop-misaligned.
        retired = _requested_retirements(hub, retire or {}, staged, report)
        retired |= _carried_alignment(
            hub, corpora, text_cache, staged | retired, names, report, drop_misaligned
        )
        gone = staged | retired
        for file in sorted(_shared_tables_to_rewrite(hub, gone, writers.files())):
            carried_counts[file] = _carry_root_rows(
                hub, file, layout.dataset_of_file(file), gone, writers, names
            )
    files = writers.close()
    exported = _settle_changed_cells(
        out, found, exported, writers, hub, names, carried_counts, report
    )
    report.appeared_during_export = sorted(
        payload_names(sources)
        - {cell.stem for cell in found.completed}
        - {cell.stem for cell in found.quarantined}
        - {cell.stem for cell in found.dropped}
        - {cell.stem for cell in found.misaligned}
    )
    # Only a staged tag replaces a published cell; a failed or dropped one
    # keeps the rows the store already has.
    replaced = {cell.tag for cell in exported}
    files = [file for file in files if (out / file).exists()]
    # A retired cell is removed from the index and the tables like a replaced
    # one, and its manifest is deleted as well.
    gone = replaced | retired
    index_rows: list[dict] = []
    for (tag, file, predictor), (n_stimuli, n_rows) in sorted(writers.counts.items()):
        cell = next(c for c in exported if c.tag == tag)
        index_rows.append(
            _index_row(cell, predictor, file, n_stimuli, n_rows, verdicts.get(tag))
        )
    if hub is not None:
        index_rows, deleted = _merge_hub_index(
            hub, index_rows, gone, set(files), carried_counts, names, report
        )
        published = hub.files()
        manifests = {f"{layout.MANIFESTS_DIR}/{tag}.json" for tag in retired}
        manifests |= _restage_renamed_manifests(hub, out, gone, published)
        report.files_deleted = sorted(set(deleted) | (manifests & published))
        report.base_revision = hub.revision()
        report.replaced_tags = sorted(
            replaced & {canonical_tag(str(t)) for t in hub.index()["tag"].to_pylist()}
        )
        report.retired_tags = sorted(retired)
    report.index_rows = len(index_rows)
    rows = coverage_rows(
        read_coverage(coverage),
        found,
        {str(row["tag"]) for row in index_rows},
        names,
        undefined=lambda dataset, method, unit, segmentation: _undefined_reason(
            dataset, method, unit, segmentation, corpora, text_cache
        ),
    )
    _write_side_tables(out, index_rows, rows)
    report.coverage_rows = len(rows)
    report.cells = len(exported)
    report.exported_tags = sorted(cell.tag for cell in exported)
    report.quarantined = len(found.quarantined)
    report.rows_written = writers.rows
    for cell in exported:
        key = f"{cell.unit}/layer{cell.layer}"
        report.by_unit_layer[key] = report.by_unit_layer.get(key, 0) + 1
    written = set(files) | {layout.INDEX_FILE, layout.COVERAGE_FILE, layout.README_FILE}
    written |= {f"{layout.MANIFESTS_DIR}/{cell.stem}.json" for cell in exported}
    if hub is not None:
        written |= {
            f"{layout.MANIFESTS_DIR}/{new}.json"
            for new in _renamed_tags(hub).values()
            if (out / layout.MANIFESTS_DIR / f"{new}.json").exists()
        }
    written |= {f"{layout.VERIFICATION_DIR}/{wave}.json" for wave, _ in verifications}
    report.files_written = sorted(written)
    (out / layout.PUBLISH_FILE).write_text(
        json.dumps(
            {
                "repo": hub.repo if hub is not None else None,
                "base_revision": report.base_revision,
                "written": report.files_written,
                "fresh_cells": {cell.tag: cell.digest for cell in exported},
                "deleted": report.files_deleted,
                "replaced_tags": report.replaced_tags,
                "retired_tags": report.retired_tags,
                "failures": report.failures + report.discovery_problems,
            },
            indent=2,
        )
        + "\n"
    )
    (out / layout.REPORT_FILE).write_text(json.dumps(asdict(report), indent=2) + "\n")
    (out / layout.README_FILE).write_text(render_readme(out))
    return report
