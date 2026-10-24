"""Ground registry claim values in stored result artifacts.

An ``artifact`` block on a claim's evidence names the table its transcribed
``values`` came from, pinned by checksum, and optionally how each cell is
re-extracted from it. The chain a verdict rests on is then checkable end to
end: stored table, extracted values, computed comparison, derived label.
A defect annotation (``predictor_correction``/``test_divergence``) may carry
an ``artifacts`` list pinning the corrected-run tables its ``effect``
statement rests on; those are checksum-verified the same way, each with a
``note`` saying what the table carries.

An artifact resolves in one of two ways:

* a repository-relative ``path`` (no ``run`` key): an indexed artifact resolved
  from its pinned results-hub revision or local cache through
  :func:`replica.studies.records.files.locate`;
* a ``run`` plus a run-relative ``path``: a table in the durable results
  store, resolved through the ``REPLICA_RESULTS_STORE`` environment variable
  and verifiable only where the store is mounted.

The extraction kinds cover the registry's cells. ``value`` reads one
number off the single row a ``where`` selector matches, and ``label`` reads
that row's column as text, for a registry cell whose value is a symbol
rather than a number. ``value_below`` is that row's decision: true when its
``column`` is below ``threshold``, the shape of a per-cell significance call;
``value_at_least`` is the complementary decision, true when the column is at
least the threshold, the shape of a per-cell failure to reject.
``signed_below`` is a significance call in a claimed direction: true when the
``column`` is below ``threshold`` and the row's ``sign_column`` has the
``sign`` (``positive`` or ``negative``), so a significant effect the other way
counts as false. ``all_rows_at_least`` is a whole-table decision: true when
every row's ``column`` is at least ``threshold``. ``max_row`` is a ranking
decision: among the rows a ``where`` selector matches, the one with the
largest ``column`` must carry the ``equals`` fields, the shape of a
best-predictor call.
``signed_decision`` retains the direction of a significant effect and labels
nonsignificance ``ns``. ``signed_stars`` applies the same labels using a
minimum published star count.
"""

from __future__ import annotations

import csv
import hashlib
import math
import os
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[4]
STORE_ENV = "REPLICA_RESULTS_STORE"

ARTIFACT_KEYS = {"path", "sha256", "run", "extract", "note"}
EXTRACT_KINDS = {
    "value",
    "label",
    "value_below",
    "value_at_least",
    "signed_below",
    "signed_decision",
    "signed_stars",
    "all_rows_at_least",
    "max_row",
}
ROW_KINDS = {
    "value",
    "label",
    "value_below",
    "value_at_least",
    "signed_below",
    "signed_decision",
    "signed_stars",
}
THRESHOLD_KINDS = {
    "value_below",
    "value_at_least",
    "signed_below",
    "signed_decision",
    "signed_stars",
    "all_rows_at_least",
}


def artifact_path(artifact: dict[str, Any]) -> Path | None:
    """The artifact's file, or ``None`` when its store is not mounted.

    Args:
        artifact: One registry ``artifact`` block. Its ``path`` is repository-
            relative, or run-relative when it also carries a ``run``.
    """
    if "run" in artifact:
        root = os.environ.get(STORE_ENV)
        if not root:
            return None
        return Path(root) / artifact["run"] / artifact["path"]
    path = REPO_ROOT / artifact["path"]
    parts = Path(artifact["path"]).parts
    if path.is_file() or parts[:3] != ("src", "replica", "studies") or len(parts) < 7:
        return path
    if parts[4] != "results":
        return path
    from ..records.files import locate

    try:
        return locate(parts[3], "/".join(parts[5:]))
    except FileNotFoundError:
        return path


def validate_artifact_schema(artifact: dict[str, Any]) -> None:
    """Schema checks that need no file: keys, checksum shape, extract kinds.

    Args:
        artifact: One registry ``artifact`` block.

    Raises:
        ValueError: A key is unknown, ``path``/``sha256`` is missing or the
            wrong shape, or an ``extract`` entry is not a usable spec.
    """
    unknown = set(artifact) - ARTIFACT_KEYS
    if unknown:
        raise ValueError(f"unknown artifact keys {sorted(unknown)}")
    for key in ("path", "sha256"):
        if not isinstance(artifact.get(key), str) or not artifact[key]:
            raise ValueError(f"artifact needs a non-empty {key!r}")
    if len(artifact["sha256"]) != 64:
        raise ValueError("sha256 must be the 64-hex-digit digest")
    for cell, spec in (artifact.get("extract") or {}).items():
        kind = spec.get("kind")
        if kind not in EXTRACT_KINDS:
            raise ValueError(f"{cell}: unknown extract kind {kind!r}")
        if not spec.get("column"):
            raise ValueError(f"{cell}: extract needs a column")
        if kind in ROW_KINDS | {"max_row"} and not spec.get("where"):
            raise ValueError(f"{cell}: a {kind} extract needs a where selector")
        if kind in THRESHOLD_KINDS and "threshold" not in spec:
            raise ValueError(f"{cell}: {kind} needs a threshold")
        if kind == "signed_below" and (
            not spec.get("sign_column")
            or spec.get("sign") not in {"positive", "negative"}
        ):
            raise ValueError(
                f"{cell}: a signed_below extract needs a sign_column and a sign"
            )
        if kind == "max_row" and not spec.get("equals"):
            raise ValueError(f"{cell}: a max_row extract needs an equals mapping")
        if kind in {"signed_decision", "signed_stars"} and not spec.get("sign_column"):
            raise ValueError(f"{cell}: a {kind} extract needs a sign_column")


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def extract_cell(rows: list[dict[str, str]], spec: dict[str, Any]) -> Any:
    """One cell's value off the table, per its extract spec.

    Args:
        rows: The artifact's rows, each column keyed by its header.
        spec: One ``extract`` entry: ``kind``, the ``column`` to read, and a
            ``where`` selector, a ``threshold``, or an ``equals`` mapping as
            the kind requires.

    Returns:
        The selected row's number for a ``value`` extract, its text for
        ``label``, whether that number is below the threshold for
        ``value_below`` (and the row's sign column has the claimed sign for
        ``signed_below``), whether it is at least the threshold for
        ``value_at_least``, or whether every row clears the threshold for
        ``all_rows_at_least``.

    Raises:
        ValueError: A ``where`` selector matches other than exactly one row,
            or no row for ``max_row``.
    """
    column = spec["column"]
    if spec["kind"] == "max_row":
        candidates = [
            row
            for row in rows
            if all(str(row[key]) == str(want) for key, want in spec["where"].items())
        ]
        if not candidates:
            raise ValueError(f"where {spec['where']} matches no rows")
        best = max(candidates, key=lambda row: float(row[column]))
        return all(str(best[key]) == str(want) for key, want in spec["equals"].items())
    if spec["kind"] in ROW_KINDS:
        matches = [
            row
            for row in rows
            if all(str(row[key]) == str(want) for key, want in spec["where"].items())
        ]
        if len(matches) != 1:
            raise ValueError(
                f"where {spec['where']} matches {len(matches)} rows, not 1"
            )
        row = matches[0]
        if spec["kind"] == "label":
            return row[column]
        number = float(row[column])
        if spec["kind"] == "value":
            return number
        if spec["kind"] == "value_at_least":
            return number >= float(spec["threshold"])
        below = number < float(spec["threshold"])
        if spec["kind"] == "value_below":
            return below
        effect = float(row[spec["sign_column"]])
        if spec["kind"] in {"signed_decision", "signed_stars"}:
            return _signed_test_decision(number, effect, spec)
        return below and (effect > 0 if spec["sign"] == "positive" else effect < 0)
    threshold = float(spec["threshold"])
    return all(float(row[column]) >= threshold for row in rows)


def _signed_test_decision(number: float, effect: float, spec: dict) -> str:
    if not math.isfinite(number) or not math.isfinite(effect):
        raise ValueError("A test decision requires finite p and effect values")
    if spec["kind"] == "signed_decision" and not 0 <= number <= 1:
        raise ValueError("A test decision requires a p value between zero and one")
    significant = (
        number >= float(spec["threshold"])
        if spec["kind"] == "signed_stars"
        else number < float(spec["threshold"])
    )
    if not significant:
        return "ns"
    if effect == 0:
        raise ValueError("A significant effect needs a nonzero direction")
    return "+" if effect > 0 else "-"


def _as_bool(value: Any) -> bool | None:
    """The truth value of a boolean, 0, 1, or ``True``/``False`` text; else ``None``."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return {"True": True, "False": False}.get(value)
    if isinstance(value, int | float) and value in (0, 1):
        return bool(value)
    return None


def _decision_rows(path: Path, digest: str) -> list[dict[str, str]]:
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError(f"Decision artifact checksum drift: {path}")
    return _rows(path)


def evidence_decisions(evidence: dict[str, Any]) -> dict[str, Any]:
    """Read declared decisions or extract signed tests from a pinned evidence table.

    Args:
        evidence: Evidence with optional ``decisions`` or ``decision_extract``.
            Extraction defaults override its artifact's per-cell selectors;
            ``cells`` supplies explicit exceptions to those defaults.
    """
    if "decisions" in evidence:
        return evidence["decisions"]
    definition = evidence.get("decision_extract")
    if not definition:
        return evidence.get("values") or {}
    artifact = evidence["artifact"]
    path = artifact_path(artifact)
    if path is None or not path.is_file():
        raise FileNotFoundError(f"Decision artifact unavailable: {artifact['path']}")
    rows = _decision_rows(path, artifact["sha256"])
    defaults = definition.get("default", {})
    overrides = definition.get("cells", {})
    specs = {
        cell: {**spec, **defaults, **overrides.get(cell, {})}
        for cell, spec in artifact["extract"].items()
        if defaults or cell in overrides
    }
    validate_artifact_schema({**artifact, "extract": specs})
    return {
        **(evidence.get("values") or {}),
        **{cell: extract_cell(rows, spec) for cell, spec in specs.items()},
    }


def verify_artifact(
    values: dict[str, Any],
    artifact: dict[str, Any],
    *,
    resolved_path: Path | None = None,
) -> dict[str, str]:
    """Check the checksum and re-extract every spec'd cell.

    A boolean agrees with a boolean, with 0 or 1, or with the text ``True`` or
    ``False``.

    Args:
        values: The claim's transcribed numbers, keyed by cell name.
        artifact: The ``artifact`` block naming the table they came from.
        resolved_path: Explicit local file to verify without resolving or downloading it.

    Returns:
        Per-check outcomes keyed by ``"sha256"`` and cell name, each ``"ok"`` or
        an error message; the caller decides whether to raise. A missing store
        yields a single ``"unresolved"`` entry.
    """
    path = resolved_path if resolved_path is not None else artifact_path(artifact)
    if path is None:
        return {"artifact": "unresolved: results store not mounted"}
    if not path.is_file():
        return {"artifact": f"unresolved: {path} does not exist"}
    outcomes: dict[str, str] = {}
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    outcomes["sha256"] = (
        "ok" if digest == artifact["sha256"] else f"checksum drift: file is {digest}"
    )
    extract = artifact.get("extract") or {}
    if not extract:
        return outcomes
    rows = _rows(path)
    for cell, spec in extract.items():
        if cell not in values:
            outcomes[cell] = "extract names a cell absent from values"
            continue
        try:
            got = extract_cell(rows, spec)
        except (KeyError, ValueError) as error:
            outcomes[cell] = f"extraction failed: {error}"
            continue
        want = values[cell]
        if isinstance(want, bool) or isinstance(got, bool):
            truth = _as_bool(got)
            agree = truth is not None and truth == _as_bool(want)
        elif isinstance(want, str) or isinstance(got, str):
            agree = str(got) == str(want)
        else:
            agree = math.isclose(float(got), float(want), rel_tol=0.0, abs_tol=0.0)
        outcomes[cell] = "ok" if agree else f"transcription drift: table gives {got!r}"
    return outcomes
