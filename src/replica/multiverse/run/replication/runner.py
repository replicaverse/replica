"""Execute validated study cells and persist their results without CLI state."""

import csv
import io
import logging
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from replica.core import resolve_dv
from replica.dataset.measures import canonical_measures

from ...results.cell_results import check_append_header
from ...spec import DatasetCell
from .contracts import IncompleteCellError, ReplicationSpec

log = logging.getLogger("replication")


@dataclass(frozen=True)
class CellFailure:
    """One failed computation, identified independently of its log message."""

    study: str
    cell: DatasetCell
    lm: str
    error: str


@dataclass(frozen=True)
class SweepResult:
    """Completed rows, inapplicable cells and failures from a replication sweep."""

    n_rows: int
    n_skipped: int
    failures: tuple[CellFailure, ...]


def cells_for(
    ds: DatasetCell, dvs: tuple[str, ...] | None, measures: Iterable[str]
) -> list[DatasetCell]:
    """Select each requested measure that the corpus actually records.

    Args:
        ds: Corpus cell with its default dependent measure.
        dvs: Canonical measure names to score, or ``None`` for the default.
        measures: Measures the corpus declares, in any spelling.
    """
    if dvs is None:
        return [ds]
    measures = tuple(measures)
    declared = canonical_measures(measures)
    kept = []
    for dv in dvs:
        measure, condition = resolve_dv(dv)
        if dv in declared or (
            measure in declared and (condition is None or condition in measures)
        ):
            kept.append(replace(ds, dv=dv))
    return kept


def _output_paths(
    studies: Mapping[str, ReplicationSpec], output: Path | None, output_dir: Path
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    schemas: dict[Path, tuple[str, ...]] = {}
    for name, spec in studies.items():
        path = (output if output is not None else output_dir / spec.output).resolve()
        if path in schemas and schemas[path] != spec.columns:
            raise ValueError(f"{path}: selected studies have different output schemas")
        check_append_header(path, spec.columns)
        schemas[path] = spec.columns
        paths[name] = path
    return paths


def _selected_cells(
    spec: ReplicationSpec, datasets: set[str] | None, dvs: tuple[str, ...] | None
) -> Iterator[DatasetCell]:
    if dvs:
        from replica.dataset.catalog.description import describe_dataset

    for grid_cell in spec.corpora:
        if datasets is not None and grid_cell.name not in datasets:
            continue
        measures = describe_dataset(grid_cell.name).measures if dvs else ()
        yield from cells_for(grid_cell, dvs, measures)


def _cell_rows(
    spec: ReplicationSpec,
    ds: DatasetCell,
    lm: str,
    tags: set[str] | None,
    *,
    select: tuple[str, ...] | None,
    cdir: Path,
) -> tuple[Sequence[Mapping[str, object]], str | None]:
    """Run one cell, returning its rows and the error of any part that failed."""
    try:
        return spec.run_cell(ds, lm, tags, select=select, cdir=cdir), None
    except IncompleteCellError as exc:
        return exc.rows, f"{type(exc).__name__}: {exc}"


def _cell_text(spec: ReplicationSpec, rows: Sequence[Mapping[str, object]]) -> str:
    spec.validate_rows(rows)
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, delimiter="\t", lineterminator="\n")
    for row in rows:
        writer.writerow([row.get(column) for column in spec.columns])
    return buffer.getvalue()


def run_sweep(
    studies: Mapping[str, ReplicationSpec],
    *,
    cdir: Path,
    tags: set[str] | None,
    lms: tuple[str, ...] | None = None,
    datasets: set[str] | None = None,
    select: tuple[str, ...] | None = None,
    dvs: tuple[str, ...] | None = None,
    output: Path | None = None,
    output_dir: Path = Path("logs/replication"),
) -> SweepResult:
    """Run selected cells, retaining successful results if another cell fails.

    All destination schemas are checked before writing or computing anything.
    A study returns no rows for an inapplicable cell; an exception is recorded as
    a failure, and an :class:`~.contracts.IncompleteCellError` also writes the
    rows it carries. The caller decides whether failures should set an exit
    status.

    Args:
        studies: Validated study adapters, keyed by registered study id.
        cdir: Compute directory passed to each study.
        tags: Available predictor tags, or ``None`` if unknown.
        lms: Model filter, or ``None`` for every model each study declares.
        datasets: Corpus name filter, or ``None`` for all declared corpora.
        select: Study-interpreted selector filter, or ``None`` for all.
        dvs: Canonical dependent measures, or ``None`` for the grid defaults.
        output: Single destination override; selected schemas must agree.
        output_dir: Directory for per-study output basenames.

    Returns:
        Counts of written rows and skipped cells, with every failure identified.
    """
    chosen = {name: spec for name, spec in studies.items() if spec.accepts(select)}
    paths = _output_paths(chosen, output, output_dir)
    n_rows = n_skipped = 0
    failures = []
    for name, spec in chosen.items():
        path = paths[name]
        path.parent.mkdir(parents=True, exist_ok=True)
        is_new = not path.exists() or path.stat().st_size == 0
        with path.open("a", newline="") as handle:
            if is_new:
                handle.write("\t".join(spec.columns) + "\n")
                handle.flush()
            for lm in (model for model in spec.lms if lms is None or model in lms):
                for ds in _selected_cells(spec, datasets, dvs):
                    try:
                        rows, error = _cell_rows(
                            spec, ds, lm, tags, select=select, cdir=cdir
                        )
                        text = _cell_text(spec, rows)
                    except Exception as exc:
                        log.exception(
                            "[cell-fail] %s/%s/%s/%s", name, ds.name, ds.dv, lm
                        )
                        failures.append(
                            CellFailure(name, ds, lm, f"{type(exc).__name__}: {exc}")
                        )
                        continue
                    handle.write(text)
                    handle.flush()
                    n_rows += len(rows)
                    if error is not None:
                        log.error(
                            "[cell-fail] %s/%s/%s/%s: %s",
                            name,
                            ds.name,
                            ds.dv,
                            lm,
                            error,
                        )
                        failures.append(CellFailure(name, ds, lm, error))
                    elif not rows:
                        n_skipped += 1
    return SweepResult(n_rows, n_skipped, tuple(failures))
