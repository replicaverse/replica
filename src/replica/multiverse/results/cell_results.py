"""TSV row formatting for grid cells."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Protocol

from .analysis_results import _isfinite


class CellIdentity(Protocol):
    """Coordinates required to serialize a cell's result or failure."""

    paper: str
    position: int
    cell: str
    corpus: str
    dv: str
    predictor_key: str | None


GRID_TSV_COLUMNS = (
    "paper",
    # Index within a recipe walk; -1 for cells addressed by label alone.
    "position",
    # Decision-set label of the cell (e.g. "anchor", "freq+lags").
    "cell",
    "corpus",
    "dv",
    "response",
    "predictor",
    "status",
    "skip_reason",
    "error_type",
    "error_message",
    "n_obs",
    "n_folds",
    "delta_ll",
    "delta_ll_se",
    "delta_r2",
    "delta_r2_se",
    "pvalue",
    "supported",
    "fold_delta_ll",
    "fold_delta_r2",
    # Scale-free effect size. Best-effort: blank when a cell's output lacks
    # the inputs (cohens_f2 needs r2_full, which ΔLL-only and ΔR²-only cells
    # don't carry).
    "cohens_f2",
    # Per-fold test-set sizes (comma-joined, aligned with fold_delta_ll);
    # with fold_delta_ll (per-unit) they reconstruct the per-fold ΔLL totals
    # that paired per-fold significance tests need.
    "fold_n",
)


def default_supported(out: Mapping) -> bool:
    """Fallback support verdict when a runner did not provide one.

    Args:
        out: A cell's output mapping; a positive ``delta_ll`` or ``delta_r2``
            counts as support.
    """
    delta_ll = out.get("delta_ll")
    delta_r2 = out.get("delta_r2")
    return bool((delta_ll or 0) > 0 or (delta_r2 or 0) > 0)


def _none_if_nonfinite(x: float):
    """Map a non-finite float to ``None`` so it serializes as a blank cell."""
    return x if _isfinite(x) else None


def standardized_effects(out: Mapping) -> dict[str, object]:
    """Best-effort scale-free effect size from a cell's output mapping.

    Computes Cohen's-f²-style ΔR²/(1 - R²_full) from ``delta_r2`` and
    ``r2_full``. The value is ``None`` (→ blank cell) when either input is
    missing or non-finite (e.g. cells that report ΔLL but no ``r2_full``).

    Args:
        out: A cell's output mapping, read for ``delta_r2`` and ``r2_full``.

    Returns:
        The ``cohens_f2`` column, ready to merge into a row dict.
    """
    from replica.analysis.inference.effect_size import cohens_f2

    f2 = cohens_f2(out.get("delta_r2"), out.get("r2_full"))
    return {"cohens_f2": _none_if_nonfinite(f2)}


def success_row(
    manifest_row: CellIdentity,
    out: Mapping,
    *,
    paper_tag: str | None = None,
) -> dict[str, object]:
    """Build a grid TSV row from a manifest row and per-cell output.

    Args:
        manifest_row: The planned cell, supplying its identity columns.
        out: What the runner returned for that cell: sizes, effects, per-fold
            series, and optionally its own ``supported`` verdict.
        paper_tag: Written to the ``paper`` column instead of the manifest's
            own, for a run that files cells under a different paper label.
    """
    supported = out.get("supported", default_supported(out))
    return {
        "paper": paper_tag or manifest_row.paper,
        "position": manifest_row.position,
        "cell": manifest_row.cell,
        "corpus": manifest_row.corpus,
        "dv": manifest_row.dv,
        "predictor": manifest_row.predictor_key,
        "response": json.dumps(out.get("response", {}), sort_keys=True),
        "status": "ok",
        "skip_reason": "",
        "error_type": "",
        "error_message": "",
        "n_obs": out.get("n_obs"),
        "n_folds": out.get("n_folds", 10),
        "delta_ll": out.get("delta_ll"),
        "delta_ll_se": out.get("delta_ll_se"),
        "delta_r2": out.get("delta_r2"),
        "delta_r2_se": out.get("delta_r2_se"),
        "pvalue": out.get("pvalue"),
        "supported": "True" if supported else "False",
        "fold_delta_ll": out.get("fold_delta_ll", ""),
        "fold_delta_r2": out.get("fold_delta_r2", ""),
        "fold_n": out.get("fold_n", ""),
        **standardized_effects(out),
    }


def _identity_row(
    manifest_row: CellIdentity,
    status: str,
    paper_tag: str | None,
) -> dict[str, object]:
    """All-blank row carrying only the cell identity, status, and no support."""
    row = {column: "" for column in GRID_TSV_COLUMNS}
    row.update(
        paper=paper_tag or manifest_row.paper,
        position=manifest_row.position,
        cell=manifest_row.cell,
        corpus=manifest_row.corpus,
        dv=manifest_row.dv,
        predictor=manifest_row.predictor_key,
        status=status,
        supported="False",
    )
    return row


def error_row(
    manifest_row: CellIdentity,
    exc: Exception,
    *,
    paper_tag: str | None = None,
) -> dict[str, object]:
    """Build a grid TSV row for a failed cell.

    Args:
        manifest_row: The planned cell, supplying its identity columns.
        exc: The failure, whose class name and text fill ``error_type`` and
            ``error_message``.
        paper_tag: Written to the ``paper`` column instead of the manifest's own.
    """
    row = _identity_row(manifest_row, "error", paper_tag)
    row.update(error_type=type(exc).__name__, error_message=str(exc))
    return row


def skip_row(
    manifest_row: CellIdentity,
    reason: str,
    *,
    paper_tag: str | None = None,
) -> dict[str, object]:
    """Build a grid TSV row for a cell that was not run.

    Args:
        manifest_row: The planned cell, supplying its identity columns.
        reason: Why it was passed over, written to ``skip_reason``.
        paper_tag: Written to the ``paper`` column instead of the manifest's own.
    """
    row = _identity_row(manifest_row, "skip", paper_tag)
    row.update(skip_reason=reason)
    return row


def _tsv_field(value: object) -> str:
    """Stringify a cell, collapsing tab/newline so it stays one TSV column.

    error_message/skip_reason carry raw exception text (multi-line R/pandas
    subprocess errors); an embedded tab or newline would otherwise inject
    columns or physical lines into this plain (unquoted) TSV.
    """
    if value is None:
        return ""
    return str(value).replace("\t", " ").replace("\r", " ").replace("\n", " ")


def format_tsv_row(row: Mapping[str, object]) -> str:
    """Serialize one row dict to a tab-separated line.

    Args:
        row: The row to write; a column it omits comes out blank, and column
            order follows ``GRID_TSV_COLUMNS``.
    """
    return "\t".join(_tsv_field(row.get(column)) for column in GRID_TSV_COLUMNS)


def check_append_header(path: str | Path, columns: Sequence[str]) -> None:
    """Refuse to append rows to a TSV whose header doesn't match *columns*.

    Appending a wider row set onto a file written under an older column
    tuple produces a ragged TSV that breaks every downstream parser.

    Args:
        path: TSV to append to; a missing or empty file passes.
        columns: The schema the caller is about to write.

    Raises:
        ValueError: When the file's header names a different schema.
    """
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return
    with open(p) as f:
        header = f.readline().rstrip("\n")
    expected = "\t".join(columns)
    if header != expected:
        raise ValueError(
            f"{p}: existing header has {len(header.split(chr(9)))} columns, "
            f"current schema has {len(columns)}, appending would produce a "
            "ragged TSV. Move the old file aside or migrate it first."
        )
