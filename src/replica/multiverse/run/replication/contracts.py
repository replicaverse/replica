"""Validated declarations at the boundary between study recipes and the sweep."""

from __future__ import annotations

import inspect
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Protocol, cast

from ...spec import DatasetCell


class CellRunner(Protocol):
    """A study's computation, independent of CLI parsing and result storage."""

    def __call__(
        self,
        ds: DatasetCell,
        lm: str,
        tags: set[str] | None,
        /,
        *,
        select: tuple[str, ...] | None,
        cdir: Path,
    ) -> Sequence[Mapping[str, object]]:
        """Return result rows, or an empty sequence for an inapplicable cell.

        A runner that fails on part of the cell raises
        :class:`IncompleteCellError` carrying the rows that completed.

        Args:
            ds: Corpus and dependent measure to score.
            lm: Language model selected from the study's declared models.
            tags: Available predictor tags, or ``None`` if unknown.
            select: Study-specific selectors, or ``None`` for all.
            cdir: Directory containing computed predictors.
        """
        ...


class IncompleteCellError(RuntimeError):
    """Raised by a cell runner when part of a cell failed to compute."""

    def __init__(self, rows: Sequence[Mapping[str, object]], errors: Sequence[str]):
        """Keep the completed rows with one message per failed part.

        Args:
            rows: Result rows of the parts that completed.
            errors: One message per part that failed.
        """
        self.rows = list(rows)
        self.errors = tuple(errors)
        super().__init__("; ".join(self.errors))


def _names(value: object, field: str) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ValueError(f"{field} must be a sequence of names")
    if any(
        not isinstance(v, str) or not v or any(c in v for c in "\t\r\n") for v in value
    ):
        raise ValueError(
            f"{field} must contain nonempty names without tabs or newlines"
        )
    names = tuple(value)
    if len(names) != len(set(names)):
        raise ValueError(f"{field} contains duplicate names")
    return names


@dataclass(frozen=True)
class ReplicationSpec:
    """A snapshot of a study module's validated sweep interface.

    Formulas and source loading stay inside ``run_cell``. Metadata is copied to
    tuples so later mutation of a module's lists cannot change a running sweep.

    Args:
        output: TSV basename shared by studies with the same result schema.
        columns: Ordered column names; omitted row values become empty fields.
        corpora: Corpus and measure cells the study supports.
        lms: Language models the study supports.
        focal: Selector vocabulary, or ``None`` when the study accepts any.
        run_cell: Study implementation of the cell computation contract.
    """

    output: str
    columns: tuple[str, ...]
    corpora: tuple[DatasetCell, ...]
    lms: tuple[str, ...]
    focal: tuple[str, ...] | None
    run_cell: CellRunner

    @classmethod
    def from_module(
        cls,
        module: ModuleType,
        *,
        default_corpora: Sequence[DatasetCell],
        default_lms: Sequence[str],
    ) -> ReplicationSpec:
        """Adapt the existing module declarations without executing a cell.

        Args:
            module: Module declaring OUTPUT, COLUMNS and run_cell, with optional
                CORPORA, LMS and FOCAL declarations.
            default_corpora: Cells to use when CORPORA is omitted or ``None``.
            default_lms: Models to use when LMS is omitted or ``None``.

        Raises:
            ValueError: Metadata or the runner's call signature is invalid.
        """
        output = getattr(module, "OUTPUT", None)
        if (
            not isinstance(output, str)
            or Path(output).name != output
            or not output.endswith(".tsv")
        ):
            raise ValueError(f"{module.__name__}.OUTPUT must be a TSV basename")
        columns = _names(getattr(module, "COLUMNS", None), "COLUMNS")
        if "dataset" not in columns:
            raise ValueError("COLUMNS must include dataset")
        declared = getattr(module, "CORPORA", None)
        if declared is not None and not isinstance(declared, Sequence):
            raise ValueError("CORPORA must be a sequence of DatasetCell objects")
        corpora = tuple(default_corpora if declared is None else declared)
        if not all(isinstance(cell, DatasetCell) for cell in corpora):
            raise ValueError("CORPORA must contain DatasetCell objects")
        if len(corpora) != len(set(corpora)):
            raise ValueError("CORPORA contains duplicate cells")
        models = getattr(module, "LMS", None)
        lms = _names(default_lms if models is None else models, "LMS")
        focal = getattr(module, "FOCAL", None)
        focal = None if focal is None else _names(focal, "FOCAL")
        runner = getattr(module, "run_cell", None)
        if not callable(runner):
            raise ValueError(f"{module.__name__}.run_cell must be callable")
        try:
            inspect.signature(runner).bind(None, None, None, select=None, cdir=Path())
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{module.__name__}.run_cell has an invalid signature"
            ) from exc
        return cls(output, columns, corpora, lms, focal, cast(CellRunner, runner))

    def accepts(self, select: tuple[str, ...] | None) -> bool:
        """Whether selectors intersect this study's declared vocabulary.

        Args:
            select: Requested selectors; ``None`` selects all.
        """
        return (
            select is None or self.focal is None or bool(set(select) & set(self.focal))
        )

    def validate_rows(self, rows: Sequence[Mapping[str, object]]) -> None:
        """Reject malformed results before any row of the cell is written.

        Args:
            rows: Results from ``run_cell``; optional columns may be omitted.

        Raises:
            ValueError: Rows lack a dataset or contain undeclared column names.
        """
        if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
            raise ValueError("run_cell must return a sequence of result mappings")
        for row in rows:
            if not isinstance(row, Mapping) or not row.get("dataset"):
                raise ValueError("Each result row must identify its dataset")
            extra = row.keys() - set(self.columns)
            if extra:
                raise ValueError(f"Result contains undeclared columns: {sorted(extra)}")
