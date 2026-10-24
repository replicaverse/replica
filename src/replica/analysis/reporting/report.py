"""Summarize completed analyses without fitting or resolving features again."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import numpy as np

from replica.analysis.contracts.results import AnalysisResult

from ._html import render
from ._observations import observation_inventory, staged_observations


def _portable(value: Any, path: str, omissions: list[str]) -> Any:
    """Convert metadata to finite JSON, identifying omitted values by path."""
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else {"nonfinite": str(value)}
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError(f"Report metadata keys must be strings: {path}")
        return {
            key: _portable(v, f"{path}.{key}", omissions) for key, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_portable(v, f"{path}[{i}]", omissions) for i, v in enumerate(value)]
    omissions.append(path)
    if isinstance(value, np.ndarray):
        return {
            "omitted_array": {"shape": list(value.shape), "dtype": str(value.dtype)}
        }
    return {"unrecorded_type": f"{type(value).__module__}.{type(value).__qualname__}"}


def _undeclared(value: Any, path: str = "configuration") -> list[str]:
    if isinstance(value, dict):
        found = [
            f"{path}.{key}"
            for key, item in value.items()
            if (key.endswith("_options") or key == "options") and item is None
        ]
        return found + [
            field
            for key, item in value.items()
            for field in _undeclared(item, f"{path}.{key}")
        ]
    if isinstance(value, list):
        return [
            field
            for i, item in enumerate(value)
            for field in _undeclared(item, f"{path}[{i}]")
        ]
    return []


def _record(result: AnalysisResult) -> dict[str, Any]:
    if not isinstance(result, AnalysisResult):
        raise TypeError("write_analysis_report accepts scalar AnalysisResult records")
    comparison = result.comparison
    record = {
        "dataset": result.dataset_name,
        "feature_model": result.model_name,
        "target": result.dv_name,
        "comparison_name": result.feature_name,
        "language": result.language,
        "comparison": None
        if comparison is None
        else {"baseline": list(comparison.baseline), "added": list(comparison.added)},
        "metrics": result.metrics,
        "n_observations": result.n_observations,
        "n_scored": result.n_scored,
        "coefficient": result.coefficient,
        "coefficient_mean_within_fit_se": result.coefficient_mean_within_fit_se,
        "coefficients": {
            name: {
                "estimate": value.estimate,
                "mean_within_fit_se": value.mean_within_fit_se,
            }
            for name, value in result.coefficients.items()
        },
        "response": result.response,
        "metadata": result.metadata,
        "folds": None
        if result.fold_results is None
        else [
            {
                "fold_index": fold.fold_index,
                "n_test": fold.n_test,
                "baseline_ll": fold.baseline_ll,
                "full_ll": fold.full_ll,
                "delta_ll": fold.delta_ll,
                "delta_ll_per_observation": fold.delta_ll / fold.n_test
                if fold.n_test
                else None,
                "baseline_r2": fold.baseline_r2,
                "full_r2": fold.full_r2,
                "delta_r2": fold.delta_r2,
                "metadata": fold.metadata,
                "observations": observation_inventory(fold),
            }
            for fold in result.fold_results
        ],
    }
    configuration = result.metadata.get("configuration")
    missing = []
    for name, present in (
        ("configuration", configuration),
        ("comparison", comparison),
        (
            "folds (use AnalysisConfig(per_fold=True) when evaluating)",
            result.fold_results,
        ),
        ("fit diagnostics", result.metadata.get("fit_diagnostics")),
        (
            "row exclusions",
            result.metadata.get(
                "row_exclusions", result.response.get("row_exclusions")
            ),
        ),
        ("feature sources", result.metadata.get("feature_sources")),
        ("target measurement", result.response.get("target_measurement")),
    ):
        if present is None:
            missing.append(name)
    if not result.fold_results or any(
        fold.metadata.get("prediction_diagnostics") is None
        for fold in result.fold_results
    ):
        missing.append(
            "prediction-error summaries for some or all folds "
            "(use AnalysisConfig(per_fold=True, prediction_diagnostics=True))"
        )
    omissions: list[str] = []
    portable = _portable(record, "result", omissions)
    portable["availability"] = {
        "not_recorded": missing,
        "undeclared_options": _undeclared(configuration),
        "omitted_metadata_values": omissions,
    }
    return portable


@contextmanager
def _staged_text(path: Path, text: str):
    temporary = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(text)
        yield temporary
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_analysis_report(
    results: AnalysisResult | Sequence[AnalysisResult],
    path: str | Path,
    *,
    title: str = "Analysis report",
    context: Mapping[str, Any] | None = None,
    observations_path: str | Path | None = None,
) -> Path:
    """Write a self-contained HTML report with downloadable JSON and fold plots.

    Reports existing scalar results without rerunning analysis. Missing metadata
    remains explicit. Per-observation arrays and arrays in metadata are omitted from HTML;
    the report is a summary, not a replay checkpoint. Custom metadata must be
    acyclic. Bounded prediction-error plots require evaluation with
    ``per_fold=True, prediction_diagnostics=True``; fit diagnostics depend on
    what the backend reports. No model checks are inferred from its name.
    No dependencies beyond REPLICA's core installation are required.

    Args:
        results: One scalar result or a sequence. An empty sequence requires
            ``context["failed_analysis"]`` with ``status="failed"``.
        path: HTML destination; its parent directory must exist.
        title: Report heading.
        context: Caller-supplied provenance, such as source and input revisions.
        observations_path: Optional CSV destination for retained scored observations.
            Requires evaluation with ``per_fold=True, retain_predictions=True``.
            Also writes ``<observations_path>.json`` with the same structured
            metadata as the HTML. Every result's folds must retain observations.

    Returns:
        The written path.
    """
    records = [results] if isinstance(results, AnalysisResult) else list(results)
    failure = (context or {}).get("failed_analysis")
    if not records and not (
        isinstance(failure, Mapping) and failure.get("status") == "failed"
    ):
        raise ValueError(
            "An analysis report needs a result or an explicit failed_analysis record"
        )
    if not records and observations_path is not None:
        raise ValueError(
            "A failed analysis without results has no observations to export"
        )
    omissions: list[str] = []
    report = {
        "schema_version": 1,
        "title": title,
        "context": _portable(dict(context or {}), "context", omissions),
        "context_omissions": omissions,
        "results": [_record(result) for result in records],
    }
    destination = Path(path)
    if observations_path is None:
        destination.write_text(render(report), encoding="utf-8")
        return destination
    csv_path = Path(observations_path)
    metadata_path = Path(str(csv_path) + ".json")
    destinations = (destination, csv_path, metadata_path)
    if len({item.resolve() for item in destinations}) != len(destinations):
        raise ValueError(
            "Report, observation CSV and metadata destinations must differ"
        )
    for item in destinations:
        if item.exists() and not item.is_file():
            raise ValueError(f"Report destination is not a file: {item}")
    with staged_observations(records, csv_path) as (csv_temporary, export):
        export["report_links"] = {
            "csv": os.path.relpath(csv_path, destination.parent),
            "metadata": os.path.relpath(metadata_path, destination.parent),
        }
        report["observation_export"] = export
        html = render(report)
        payload = json.dumps(report, allow_nan=False, ensure_ascii=False, indent=2)
        with (
            _staged_text(destination, html) as html_temporary,
            _staged_text(metadata_path, payload + "\n") as metadata_temporary,
        ):
            csv_temporary.replace(csv_path)
            metadata_temporary.replace(metadata_path)
            html_temporary.replace(destination)
    return destination
