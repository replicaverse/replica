"""Read a staged tree back against the pickles it was exported from.

Supported exported values and semantic columns of fresh cells are compared
with their source records and corpus grid. Carried cells without source
pickles are outside this comparison. The index is
then checked against the files it names, and the coverage table against the
cells found. Any difference is a failure.

The comparison runs on Arrow arrays: the pickle's records are laid out as a
table with the store's own columns, both sides are sorted by the same keys,
and the columns are compared in bulk, so a cell of tens of millions of rows
is checked without one Python object per value.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from replica.artifacts import manifest_text
from replica.compute.units import TOKEN_UNIT
from replica.feature.core.outputs.units import unit_for
from replica.utils.naming import canonical_tag

from . import layout
from ._readback import _state, _write_receipt
from .cells import Cell, discover, load_payload
from .corpora import Corpora, RegistryCorpora, TextIndex, check_payload


@dataclass
class VerifyReport:
    """What the readback compared, and where it disagreed."""

    cells: int = 0
    rows_compared: int = 0
    arrays_compared: int = 0
    files_indexed: int = 0
    carried_index_rows: int = 0
    failures: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Whether the stage agrees with the pickles and its own index."""
        return not self.failures

    def summary(self) -> str:
        """A short account of the readback."""
        lines = [
            f"cells compared: {self.cells}",
            f"scalar rows compared: {self.rows_compared:,}",
            f"array rows compared: {self.arrays_compared:,}",
            f"files checked against the index: {self.files_indexed}",
            f"index rows carried from the hub: {self.carried_index_rows}",
        ]
        lines += [f"note: {note}" for note in self.notes]
        lines += [f"FAIL: {failure}" for failure in self.failures]
        lines.append("OK" if self.ok else f"{len(self.failures)} failure(s)")
        return "\n".join(lines)


def _is_measurement(key: str, value: object) -> bool:
    if isinstance(value, str | bytes):
        return False
    array = np.asarray(value)
    return (
        array.ndim == 0
        and array.dtype.kind not in "USO"
        and unit_for(key) not in layout.NON_NUMERIC_UNITS
    )


_TOKEN_IDENTITY = ("span_start", "span_end", "token_id", "token")


def _position_column(cell: Cell) -> str:
    return "token_index" if cell.unit == TOKEN_UNIT else "unit_index"


def _sort_keys(cell: Cell, with_predictor: bool) -> list[tuple[str, str]]:
    keys = [("stimulus_id", "ascending"), (_position_column(cell), "ascending")]
    if with_predictor:
        keys.append(("predictor", "ascending"))
    return keys


def _expected_identity(
    cell: Cell, sid: str, index: int, record, text: TextIndex
) -> dict:
    identity: dict[str, object] = {
        "dataset": cell.dataset,
        "tag": cell.tag,
        "stimulus_id": sid,
    }
    if cell.unit == TOKEN_UNIT:
        names = ("token_index", *_TOKEN_IDENTITY)
        identity.update(
            zip(names, layout.token_identity(record.metadata, index), strict=True)
        )
    else:
        identity.update(unit_index=index, text_unit=text.grid(cell.unit)[sid][index])
    return identity


def _expected_scalars(cell: Cell, payload: dict, text: TextIndex) -> pa.Table:
    """The cell's scalar values laid out as the store's rows, sorted by key."""
    tokens = cell.unit == TOKEN_UNIT
    dense_lens = cell.method in layout.LENS_METHODS and not tokens
    columns: dict[str, list] = defaultdict(list)
    for stimulus_id, records in payload.items():
        for index, record in enumerate(records):
            identity = _expected_identity(cell, stimulus_id, index, record, text)
            for key, value in record.values.items():
                if not _is_measurement(key, value):
                    continue
                if dense_lens and layout.is_lens_key(key):
                    continue
                for name, identity_value in identity.items():
                    columns[name].append(identity_value)
                columns["predictor"].append(key)
                columns["value"].append(float(np.asarray(value)))
                columns["value_unit"].append(unit_for(key))
                if not tokens:
                    columns["layer"].append(cell.layer)
                    columns["storage_class"].append(cell.storage_class)
                    columns["unit_rule"].append(cell.unit_rule)
    schema = layout.TOKEN_SCHEMA if tokens else layout.WORD_SCHEMA
    names = schema.names
    table = pa.table(
        {name: pa.array(columns[name], type=schema.field(name).type) for name in names}
    )
    return table.sort_by(_sort_keys(cell, True))


def _first_difference(left: pa.ChunkedArray, right: pa.ChunkedArray) -> int | None:
    """Index of the first row where the columns differ, ``None`` when equal.

    Floats compare equal when both are NaN; every other type compares by value.
    """
    if pa.types.is_floating(left.type):
        a = left.to_numpy(zero_copy_only=False)
        b = right.to_numpy(zero_copy_only=False)
        valid = np.asarray(left.is_valid()) & np.asarray(right.is_valid())
        same = ((a == b) | (np.isnan(a) & np.isnan(b))) & valid
    else:
        equal = pc.call_function("equal", [left, right])
        same = np.asarray(pc.fill_null(equal, False), dtype=bool)
    same |= np.asarray(left.is_null()) & np.asarray(right.is_null())
    bad = np.flatnonzero(~same)
    return int(bad[0]) if len(bad) else None


def _row(table: pa.Table, index: int) -> dict:
    return table.slice(index, 1).to_pylist()[0]


def _check_scalars(
    cell: Cell, payload: dict, stage: Path, report: VerifyReport, text: TextIndex
) -> set[str]:
    file = layout.scalar_file(cell.method, cell.unit, cell.dataset, cell.tag)
    if file is None:
        return set()
    expected = _expected_scalars(cell, payload, text)
    path = stage / file
    if not path.exists():
        if expected.num_rows:
            report.failures.append(f"{cell.tag}: {file} missing from the stage")
        return set()
    columns = expected.column_names
    staged = pq.read_table(path, columns=columns, filters=[("tag", "==", cell.tag)])
    if staged.num_rows != expected.num_rows:
        report.failures.append(
            f"{cell.tag}: {file} holds {staged.num_rows} rows, pickle has "
            f"{expected.num_rows}"
        )
    keys = [name for name, _ in _sort_keys(cell, True)]
    distinct = staged.group_by(keys).aggregate([]).num_rows
    if distinct != staged.num_rows:
        report.failures.append(
            f"{cell.tag}: {file} holds {staged.num_rows - distinct} duplicate rows"
        )
    staged = staged.sort_by(_sort_keys(cell, True))
    n = min(staged.num_rows, expected.num_rows)
    for name in columns if n else ():
        if staged[name].type != expected[name].type:
            report.failures.append(f"{cell.tag}: {file} {name} type differs")
            continue
        index = _first_difference(staged[name].slice(0, n), expected[name].slice(0, n))
        if index is not None:
            got, want = _row(staged, index), _row(expected, index)
            what = "token identity" if name in _TOKEN_IDENTITY else name
            report.failures.append(
                f"{cell.tag}: {what} of {tuple(want[k] for k in keys)} staged "
                f"{got[name]!r}, pickle {want[name]!r}"
            )
            break
    report.rows_compared += staged.num_rows
    return {file}


def _array_expectations(cell: Cell, payload: dict) -> dict[str, dict[str, dict]]:
    """``{file: {column: {(sid, index): array}}}`` the cell's arrays must appear as."""
    out: dict[str, dict[str, dict]] = defaultdict(lambda: defaultdict(dict))
    dense_lens = cell.method in layout.LENS_METHODS and cell.unit != TOKEN_UNIT
    lookahead = cell.method == layout.LOOKAHEAD_METHOD
    for stimulus_id, records in payload.items():
        for index, record in enumerate(records):
            values = record.values
            key = (stimulus_id, index)
            if dense_lens:
                keys = layout.lens_layer_keys(list(values))
                if keys:
                    file = f"{layout.LENS_DIR}/{cell.tag}.parquet"
                    out[file][layout.LENS_COLUMN][key] = np.array(
                        [float(values[k]) for k in keys]
                    )
            if lookahead and layout.LOOKAHEAD_INDEX_KEY in values:
                file = f"{layout.LOOKAHEAD_DIR}/{cell.tag}.parquet"
                for column, array in layout.lookahead_arrays(values).items():
                    out[file][column][key] = array.reshape(-1)
            for value_key, value in values.items():
                if isinstance(value, str | bytes) or np.ndim(value) < 1:
                    continue
                family = layout.array_family(cell.method, value_key)
                if family is None:
                    continue
                out[family.file(cell.tag)][family.column][key] = np.asarray(value)
    return out


def _flatten(
    column: pa.ChunkedArray,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Flat values, row sizes and child-list lengths at every nesting level."""
    array = column.combine_chunks()
    if array.null_count:
        raise ValueError("contains nulls")
    offsets = np.asarray(array.offsets)
    shapes = [np.diff(offsets)]
    values = array.values
    while pa.types.is_list(values.type) or pa.types.is_large_list(values.type):
        if values.null_count:
            raise ValueError("contains nulls")
        inner_offsets = np.asarray(values.offsets)
        shapes.append(np.diff(inner_offsets))
        offsets = inner_offsets[offsets]
        values = values.values
    if values.null_count:
        raise ValueError("contains nulls")
    return values.to_numpy(zero_copy_only=False), np.diff(offsets), shapes


def _compare_arrays(
    cell: Cell,
    file: str,
    column: str,
    table: pa.Table,
    expected: dict[tuple[str, int], np.ndarray],
    report: VerifyReport,
) -> None:
    keys = list(
        zip(
            table["stimulus_id"].to_pylist(),
            table["unit_index"].to_pylist(),
            strict=True,
        )
    )
    if set(keys) != set(expected) or len(keys) != len(expected):
        report.failures.append(
            f"{cell.tag}: {file} holds {len(keys)} rows, pickle has "
            f"{len(expected)} {column} arrays"
        )
        return
    try:
        flat, sizes, shapes = _flatten(table[column])
    except ValueError as exc:
        report.failures.append(f"{cell.tag}: {file} column {column} {exc}")
        return
    want = [expected[key] for key in keys]
    if any(a.ndim != len(shapes) for a in want):
        report.failures.append(f"{cell.tag}: {file} column {column} ranks differ")
        return
    want_sizes = np.fromiter((a.size for a in want), dtype=np.int64, count=len(want))
    shape_matches = np.array_equal(sizes, want_sizes)
    for axis, lengths in enumerate(shapes):
        expected_lengths = (
            np.concatenate(
                [
                    np.full(np.prod(a.shape[:axis], dtype=np.int64), a.shape[axis])
                    for a in want
                ]
            )
            if want
            else np.zeros(0, dtype=np.int64)
        )
        shape_matches &= np.array_equal(lengths, expected_lengths)
    if not shape_matches:
        report.failures.append(f"{cell.tag}: {file} column {column} shapes differ")
        return
    dtype = flat.dtype
    if dtype.kind in "iu":
        bounds = np.iinfo(dtype)
        if any(
            a.dtype.kind not in "iu"
            or (a.size and (int(a.min()) < bounds.min or int(a.max()) > bounds.max))
            for a in want
        ):
            report.failures.append(
                f"{cell.tag}: {file} column {column} cannot preserve the pickle's integers"
            )
            return
    with np.errstate(over="ignore", invalid="ignore"):
        converted = [a.reshape(-1).astype(dtype) for a in want]
    if dtype.kind == "f" and any(
        a.dtype.kind in "iu"
        and any(
            not np.isfinite(y) or int(x) != int(y)
            for x, y in zip(a.flat, b, strict=True)
        )
        for a, b in zip(want, converted, strict=True)
    ):
        report.failures.append(
            f"{cell.tag}: {file} column {column} cannot preserve the pickle's integers"
        )
        return
    if dtype.kind == "f" and any(
        np.any(np.isfinite(a.reshape(-1)) & ~np.isfinite(b))
        for a, b in zip(want, converted, strict=True)
    ):
        report.failures.append(
            f"{cell.tag}: {file} column {column} overflows finite pickle values"
        )
        return
    want_flat = np.concatenate(converted) if want else np.zeros(0, dtype=dtype)
    if dtype.kind == "f":
        same = (flat == want_flat) | (np.isnan(flat) & np.isnan(want_flat))
    else:
        same = flat == want_flat
    bad = np.flatnonzero(~same)
    if len(bad):
        row = int(np.searchsorted(np.cumsum(sizes), bad[0], side="right"))
        report.failures.append(
            f"{cell.tag}: {file} column {column} differs at {keys[row]}"
        )


def _check_array_metadata(
    cell: Cell,
    payload: dict,
    file: str,
    table: pa.Table,
    report: VerifyReport,
    text: TextIndex,
) -> None:
    key = layout.index_key_of_file(file)
    if key is None:
        raise ValueError(f"no array value key for {file}")
    expected: dict[str, list] = defaultdict(list)
    family = layout.array_family(cell.method, key)
    for sid, index in zip(
        table["stimulus_id"].to_pylist(), table["unit_index"].to_pylist(), strict=True
    ):
        if not isinstance(index, int) or isinstance(index, bool) or index < 0:
            raise ValueError(f"{file}: invalid unit index {index!r}")
        record = payload[sid][index]
        identity = _expected_identity(cell, sid, index, record, text)
        identity.update(unit_index=index, value_unit=unit_for(key))
        if (
            family is not None and family.model_column
        ) or cell.method == layout.LOOKAHEAD_METHOD:
            identity["model"] = cell.info.model_slot
        for name, value in identity.items():
            expected[name].append(value)
    for name, values in expected.items():
        if name not in table.column_names:
            report.failures.append(f"{cell.tag}: {file} lacks {name}")
            continue
        kind = (
            layout.TOKEN_SCHEMA.field(name).type
            if name in layout.TOKEN_SCHEMA.names
            else layout.WORD_SCHEMA.field(name).type
            if name in layout.WORD_SCHEMA.names
            else pa.string()
        )
        want = pa.chunked_array([pa.array(values, type=kind)])
        if table[name].type != kind or _first_difference(table[name], want) is not None:
            report.failures.append(
                f"{cell.tag}: {file} {name} differs from source identity"
            )
    for column in table.schema:
        if pa.types.is_list(column.type) or pa.types.is_large_list(column.type):
            value_key = layout.LOOKAHEAD_COLUMNS.get(column.name, key)
            if (column.metadata or {}).get(b"unit") != unit_for(value_key).encode():
                report.failures.append(
                    f"{cell.tag}: {file} {column.name} unit metadata differs"
                )


def _array_schema(cell: Cell, file: str, columns: dict[str, dict]) -> pa.Schema:
    """The declared storage schema for the source's expected arrays."""
    key = layout.index_key_of_file(file)
    if key is None:
        raise ValueError(f"no array value key for {file}")
    family = layout.array_family(cell.method, key)
    if family is not None:
        ranks = {array.ndim for array in columns[family.column].values()}
        if len(ranks) != 1:
            raise ValueError(f"{family.key} mixes ranks {sorted(ranks)}")
        return family.schema(cell.unit, ranks.pop())
    if file.startswith(f"{layout.LENS_DIR}/"):
        return layout.lens_schema()
    if file.startswith(f"{layout.LOOKAHEAD_DIR}/"):
        return layout.lookahead_schema()
    raise ValueError(f"no array schema for {file}")


def _check_arrays(
    cell: Cell, payload: dict, stage: Path, report: VerifyReport, text: TextIndex
) -> set[str]:
    try:
        expectations = _array_expectations(cell, payload)
    except ValueError as exc:
        report.failures.append(f"{cell.tag}: {exc}")
        return set()
    for file, columns in expectations.items():
        path = stage / file
        if not path.exists():
            report.failures.append(f"{cell.tag}: {file} missing from the stage")
            continue
        table = pq.read_table(path, filters=[("tag", "==", cell.tag)])
        schema = _array_schema(cell, file, columns)
        stored_columns = {
            field.name
            for field in table.schema
            if pa.types.is_list(field.type) or pa.types.is_large_list(field.type)
        }
        if stored_columns != set(columns):
            report.failures.append(
                f"{cell.tag}: {file} array columns differ from source: "
                f"{sorted(stored_columns)} != {sorted(columns)}"
            )
        _check_array_metadata(cell, payload, file, table, report, text)
        if cell.method == layout.LOOKAHEAD_METHOD:
            dimensions = {
                (sid, index): np.asarray(
                    record.values["fwd_word_log_p_step_nats"]
                ).shape
                for sid, records in payload.items()
                for index, record in enumerate(records)
                if layout.LOOKAHEAD_INDEX_KEY in record.values
            }
            for axis, name in enumerate(("n_samples", "k_words")):
                if name not in table.column_names or not pa.types.is_integer(
                    table[name].type
                ):
                    report.failures.append(f"{cell.tag}: {file} lacks integer {name}")
                    continue
                for sid, index, size in zip(
                    table["stimulus_id"].to_pylist(),
                    table["unit_index"].to_pylist(),
                    table[name].to_pylist(),
                    strict=True,
                ):
                    shape = dimensions.get((sid, index))
                    if shape is None or size != shape[axis]:
                        report.failures.append(
                            f"{cell.tag}: {file} {name} differs at {(sid, index)}"
                        )
                        break
        for column, expected in columns.items():
            if column not in table.column_names:
                report.failures.append(f"{cell.tag}: {file} lacks column {column}")
                continue
            if table[column].type != schema.field(column).type:
                report.failures.append(
                    f"{cell.tag}: {file} column {column} type differs from declared "
                    f"storage: {table[column].type} != {schema.field(column).type}"
                )
                continue
            _compare_arrays(cell, file, column, table, expected, report)
            report.arrays_compared += table.num_rows
    return set(expectations)


def _check_manifest(cell: Cell, stage: Path, report: VerifyReport) -> None:
    staged = stage / layout.MANIFESTS_DIR / f"{cell.stem}.json"
    if not staged.exists():
        report.failures.append(f"{cell.tag}: manifest not staged")
    elif staged.read_text() != manifest_text(cell.manifest):
        report.failures.append(f"{cell.tag}: staged manifest differs from the source")


def file_counts(path: Path, file: str) -> dict[tuple[str, str], tuple[int, int]]:
    """``{(tag, predictor): (n_stimuli, n_rows)}`` of one staged file.

    An array file has no predictor column; its rows count under the key the
    index names for its folder.

    Args:
        path: The Parquet file.
        file: Its path relative to the store root.
    """
    names = pq.ParquetFile(path).schema_arrow.names
    keys = ["tag", "predictor"] if "predictor" in names else ["tag"]
    table = pq.read_table(path, columns=[*keys, "stimulus_id"])
    grouped = table.group_by(keys).aggregate(
        [("stimulus_id", "count_distinct"), ("stimulus_id", "count")]
    )
    out: dict[tuple[str, str], tuple[int, int]] = {}
    for row in grouped.to_pylist():
        predictor = row.get("predictor") or layout.index_key_of_file(file) or ""
        out[row["tag"], predictor] = (
            row["stimulus_id_count_distinct"],
            row["stimulus_id_count"],
        )
    return out


def _check_index(
    stage: Path,
    cells: Sequence[Cell],
    compared_files: dict[str, set[str]],
    report: VerifyReport,
) -> None:
    index_path = stage / layout.INDEX_FILE
    if not index_path.exists():
        report.failures.append(f"{layout.INDEX_FILE} missing")
        return
    index = pq.read_table(index_path).to_pylist()
    fresh = {cell.tag: cell for cell in cells}
    by_file: dict[str, dict[tuple[str, str], tuple[int, int]]] = defaultdict(dict)
    for row in index:
        cell = fresh.get(canonical_tag(row["tag"]))
        if cell is not None:
            if row["file"] not in compared_files[cell.tag]:
                report.failures.append(
                    f"index: {cell.tag} {row['file']} is outside its source comparison"
                )
            identity = cell.identity
            expected = {
                "tag": cell.tag,
                "dataset": cell.dataset,
                "method": cell.method,
                "model": cell.info.model,
                "model_kind": cell.info.model_kind,
                "value_unit": unit_for(row["predictor"]),
                "layer": cell.layer,
                "study": cell.study,
                "storage_class": cell.storage_class,
                "unit": cell.unit,
                "unit_rule": cell.unit_rule,
                "context_scope": cell.context_scope,
                "derivation_source": cell.derivation_source,
                "provenance": "manifest",
                "source_sha256": identity.get("source_sha256"),
                "model_revision": (identity.get("model") or {}).get("revision"),
                **cell.index_config(),
            }
            for name, value in expected.items():
                if row.get(name) != value:
                    report.failures.append(
                        f"index: {cell.tag} {name} differs from source identity"
                    )
        key = (str(row["tag"]), str(row["predictor"]))
        if key in by_file[row["file"]]:
            report.failures.append(f"index: duplicate row {row['file']} {key}")
        by_file[row["file"]][key] = (row["n_stimuli"], row["n_rows"])
    staged_files = {
        p.relative_to(stage).as_posix()
        for p in stage.rglob("*.parquet")
        if p.name not in layout.SIDE_FILES
    }
    for file in sorted(staged_files):
        actual = file_counts(stage / file, file)
        indexed = by_file.get(file, {})
        for key in sorted(set(indexed) | set(actual)):
            if indexed.get(key) != actual.get(key):
                report.failures.append(
                    f"index: {file} {key} indexed {indexed.get(key)}, "
                    f"file holds {actual.get(key)}"
                )
        report.files_indexed += 1
    fresh_tags = {cell.tag for cell in cells}
    for row in index:
        if row["file"] in staged_files:
            continue
        if row["tag"] in fresh_tags:
            report.failures.append(
                f"index: {row['file']} is not staged but its cell is a fresh one"
            )
        else:
            report.carried_index_rows += 1


def _check_coverage(
    stage: Path,
    cells: Sequence[Cell],
    quarantined: Sequence[Cell],
    report: VerifyReport,
) -> None:
    path = stage / layout.COVERAGE_FILE
    if not path.exists():
        report.failures.append(f"{layout.COVERAGE_FILE} missing")
        return
    rows = pq.read_table(path).to_pylist()
    status = defaultdict(set)
    for row in rows:
        status[row["tag"]].add(row["status"])
    index_path = stage / layout.INDEX_FILE
    stored: set[str] = set()
    if index_path.exists():
        stored = set(pq.read_table(index_path, columns=["tag"])["tag"].to_pylist())
    for cell in cells:
        if status.get(cell.tag) != {layout.EXPORTED}:
            report.failures.append(
                f"coverage: {cell.tag} exported but marked "
                f"{sorted(status.get(cell.tag, ()))}"
            )
    for cell in quarantined:
        # A table the store already published stays in the release, so a cell
        # this run quarantined is carried.
        expected = layout.CARRIED if cell.tag in stored else layout.QUARANTINED
        if status.get(cell.tag) != {expected}:
            report.failures.append(
                f"coverage: {cell.tag} quarantined but marked "
                f"{sorted(status.get(cell.tag, ()))}, expected {expected}"
            )
    exported = {cell.tag for cell in cells}
    for tag, marks in status.items():
        if layout.EXPORTED in marks and tag not in exported:
            report.failures.append(f"coverage: {tag} marked exported without a cell")


def _check_text_units(stage: Path, report: VerifyReport) -> None:
    """Require a unit string on every word and character row of the stage."""
    for path in sorted(stage.rglob("*.parquet")):
        file = path.relative_to(stage).as_posix()
        if path.name in layout.SIDE_FILES or not layout.is_shared_table(file):
            continue
        if file.startswith(f"{layout.TOKENS_DIR}/"):
            continue
        nulls = 0
        for batch in pq.ParquetFile(path).iter_batches(columns=["text_unit"]):
            nulls += batch.column(0).null_count
        if nulls:
            report.failures.append(f"{file}: {nulls} rows without a unit string")


def verify(
    stage: Path,
    sources: Sequence[tuple[str, Path]],
    *,
    names: Iterable[str] | None = None,
    corpora: Corpora | None = None,
) -> VerifyReport:
    """Compare all fresh cells and record a byte-bound ``readback.json`` receipt.

    Args:
        stage: The staged tree.
        sources: ``(wave label, output directory)`` pairs holding every fresh
            pickle declared in the export plan.
        names: Dataset names a tag may end in; the registry by default.
        corpora: Current corpus loaders; the registry by default.

    Returns:
        The report; ``ok`` is false on any difference. A new verification
        invalidates the previous receipt, including on failure or exception.
        Carried cells are outside the fresh-source numerical comparison.
    """
    receipt = stage / layout.READBACK_FILE
    receipt.unlink(missing_ok=True)
    report = VerifyReport()
    try:
        plan, before = _state(stage)
    except (OSError, ValueError) as exc:
        report.failures.append(f"stage: {exc}")
        return report
    corpora = corpora or RegistryCorpora()
    found = discover(
        list(sources), list(names) if names is not None else corpora.names()
    )
    report.failures += [f"invalid source cell: {problem}" for problem in found.problems]
    actual = {cell.tag: cell.digest for cell in found.completed}
    expected = plan["fresh_cells"]
    if actual != expected:
        changed = sorted(
            tag
            for tag in actual.keys() & expected.keys()
            if actual[tag] != expected[tag]
        )
        report.failures.append(
            "sources do not match the export plan's complete fresh_cells: "
            f"missing={sorted(expected.keys() - actual.keys())}, "
            f"unexpected={sorted(actual.keys() - expected.keys())}, changed={changed}"
        )
        return report
    for failure in plan.get("failures", []):
        report.notes.append(f"export reported: {failure}")
    texts: dict[str, TextIndex] = {}
    compared_files: dict[str, set[str]] = {}
    for cell in found.completed:
        compared_files[cell.tag] = set()
        try:
            if cell.dataset not in texts:
                texts[cell.dataset] = TextIndex(corpora.stimuli(cell.dataset))
            text = texts[cell.dataset]
            payload = load_payload(cell)
            check_payload(cell, payload, text)
            _check_manifest(cell, stage, report)
            compared_files[cell.tag].update(
                _check_scalars(cell, payload, stage, report, text)
            )
            compared_files[cell.tag].update(
                _check_arrays(cell, payload, stage, report, text)
            )
            report.cells += 1
        except (KeyError, ValueError, TypeError, IndexError, OSError) as exc:
            report.failures.append(f"{cell.tag}: {exc}")
    _check_index(stage, found.completed, compared_files, report)
    _check_coverage(stage, found.completed, found.quarantined, report)
    _check_text_units(stage, report)
    required = {file for files in compared_files.values() for file in files}
    required.update(
        f"{layout.MANIFESTS_DIR}/{cell.stem}.json" for cell in found.completed
    )
    if missing := required - set(plan["written"]):
        report.failures.append(f"publish plan omits compared files: {sorted(missing)}")
    try:
        for cell in found.completed:
            if cell.changed():
                report.failures.append(f"{cell.tag}: source changed during readback")
        if _state(stage)[1] != before:
            report.failures.append("stage changed during readback")
    except (OSError, ValueError) as exc:
        report.failures.append(f"readback changed: {exc}")
    if report.ok:
        _write_receipt(stage, before)
    return report
