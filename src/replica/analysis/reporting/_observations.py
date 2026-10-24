"""Explicit streaming export of retained, aligned scored observations."""

import csv
import hashlib
import json
import math
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import numpy as np

from replica.analysis.contracts.results import AnalysisResult, FoldResult


def _array_description(value: Any) -> dict:
    if isinstance(value, np.ndarray):
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    return {"unrecorded_type": f"{type(value).__module__}.{type(value).__qualname__}"}


def observation_inventory(fold: FoldResult) -> dict | None:
    """Describe retained fields without reading array elements.

    Args:
        fold: Completed fold with optional retained observation arrays.
    """
    observations = getattr(fold, "observations", None)
    keys = fold.observation_keys
    if observations is None:
        return None
    return {
        "n_rows": fold.n_test,
        "fields": {
            name: _array_description(value) for name, value in observations.items()
        },
        "row_ids": None if fold.row_ids is None else _array_description(fold.row_ids),
        "row_keys": None if keys is None else {"count": len(keys)},
        "factors": {
            name: _array_description(value) for name, value in fold.factor_ids.items()
        },
        "delta_loglik": None
        if fold.per_obs_delta_ll is None
        else _array_description(fold.per_obs_delta_ll),
    }


def _identity(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _identity(value.tolist())
    if isinstance(value, np.generic):
        return _identity(value.item())
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else {"nonfinite": str(value)}
    if isinstance(value, (list, tuple)):
        return [_identity(item) for item in value]
    raise TypeError(
        f"Observation identity is not JSON-compatible: {type(value).__name__}"
    )


def _json_cell(value: Any) -> str:
    return json.dumps(
        _identity(value), ensure_ascii=False, allow_nan=False, separators=(",", ":")
    )


def _aligned(value: Any, n_rows: int, name: str, *, numeric: bool) -> None:
    if not isinstance(value, np.ndarray) or value.ndim != 1 or len(value) != n_rows:
        raise ValueError(
            f"Observation export requires an aligned one-dimensional {name} array"
        )
    if numeric and value.dtype.kind not in "biuf":
        raise TypeError(f"Observation export requires real numeric {name} values")


@contextmanager
def staged_observations(
    results: Sequence[AnalysisResult], path: Path
) -> Iterator[tuple[Path, dict]]:
    """Validate and stage CSV rows, leaving destination publication to the caller.

    Args:
        results: Results whose every fold retained observation arrays.
        path: Intended CSV destination, used to locate its temporary file.
    """
    folds = []
    fields: set[str] = set()
    factors: set[str] = set()
    reserved = {
        "result_index",
        "fold_index",
        "row_index",
        "row_id",
        "row_key",
        "delta_loglik",
        "error_baseline",
        "error_full",
    }
    required = {
        "observed",
        "prediction_baseline",
        "prediction_full",
        "in_train",
        "prepared_row_index",
    }
    for result_index, result in enumerate(results):
        if not result.fold_results:
            raise ValueError(f"Result {result_index} has no retained folds to export")
        for fold in result.fold_results:
            observations = getattr(fold, "observations", None)
            if not observations:
                raise ValueError(
                    f"Result {result_index}, fold {fold.fold_index} has no retained observations; "
                    "use AnalysisConfig(per_fold=True, retain_predictions=True) before evaluation"
                )
            if not required.issubset(observations):
                raise ValueError(
                    f"Retained observations lack required fields: {sorted(required - observations.keys())}"
                )
            keys = fold.observation_keys
            if keys is None or len(keys) != fold.n_test:
                raise ValueError(
                    "Observation export requires aligned canonical observation_keys"
                )
            for name, values in observations.items():
                if (
                    not isinstance(name, str)
                    or name in reserved
                    or name.startswith("factor:")
                ):
                    raise ValueError(f"Invalid observation field name: {name!r}")
                _aligned(values, fold.n_test, name, numeric=True)
                fields.add(name)
            for name, values in fold.factor_ids.items():
                if not isinstance(name, str):
                    raise TypeError("Observation factor names must be strings")
                _aligned(values, fold.n_test, name, numeric=False)
                factors.add(name)
            if fold.row_ids is not None and (
                not isinstance(fold.row_ids, np.ndarray)
                or fold.row_ids.ndim not in (1, 2)
                or len(fold.row_ids) != fold.n_test
            ):
                raise ValueError("Observation row_ids must align with scored rows")
            if fold.per_obs_delta_ll is not None:
                _aligned(
                    fold.per_obs_delta_ll, fold.n_test, "delta_loglik", numeric=True
                )
            error_dtypes = tuple(
                np.result_type(
                    observations["observed"].dtype, observations[name].dtype, np.float64
                )
                for name in ("prediction_baseline", "prediction_full")
            )
            folds.append((result_index, fold, observations, keys, error_dtypes))
    field_names = [
        "observed",
        "prediction_baseline",
        "prediction_full",
        "in_train",
        "prepared_row_index",
        *sorted(fields - required),
    ]
    factor_names = sorted(factors)
    columns = [
        "result_index",
        "fold_index",
        "row_index",
        "row_id",
        "row_key",
        *field_names,
        "error_baseline",
        "error_full",
        "delta_loglik",
        *(f"factor:{name}" for name in factor_names),
    ]
    temporary = None
    try:
        with (
            NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="",
                dir=path.parent,
                prefix=f".{path.name}.",
                delete=False,
            ) as stream,
            np.errstate(over="ignore", invalid="ignore"),
        ):
            temporary = Path(stream.name)
            writer = csv.writer(stream)
            writer.writerow(columns)
            for result_index, fold, observations, keys, error_dtypes in folds:
                for index in range(fold.n_test):
                    writer.writerow(
                        [
                            result_index,
                            fold.fold_index,
                            index,
                            ""
                            if fold.row_ids is None
                            else _json_cell(fold.row_ids[index]),
                            _json_cell(keys[index]),
                            *(
                                observations[name][index].item()
                                if name in observations
                                else ""
                                for name in field_names
                            ),
                            np.subtract(
                                observations["observed"][index],
                                observations["prediction_baseline"][index],
                                dtype=error_dtypes[0],
                            ).item(),
                            np.subtract(
                                observations["observed"][index],
                                observations["prediction_full"][index],
                                dtype=error_dtypes[1],
                            ).item(),
                            ""
                            if fold.per_obs_delta_ll is None
                            else fold.per_obs_delta_ll[index].item(),
                            *(
                                _json_cell(fold.factor_ids[name][index])
                                if name in fold.factor_ids
                                else ""
                                for name in factor_names
                            ),
                        ]
                    )
        with temporary.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        yield (
            temporary,
            {
                "format": "csv",
                "path": path.name,
                "metadata_path": path.name + ".json",
                "sha256": digest,
                "n_rows": sum(fold.n_test for _, fold, _, _, _ in folds),
                "columns": columns,
                "row_semantics": "One row per scored observation per result and fold; baseline/full are separate named fields. Repeated folds can repeat observations; rows are not independent replicates.",
                "identity_encoding": "row_key is the exact canonical observation key; row_id contains readable labels. Both and factor:* cells use JSON. Key/label columns are declared in observation_capture metadata. prepared_row_index identifies the prepared row within its recorded cohort. Empty cells mean a field was not retained; JSON null is a retained missing identity.",
                "numeric_encoding": "Full-precision numeric values; nonfinite values use nan, inf, -inf, and booleans use True/False. Empty cells mean the field was not retained for that fold.",
                "error_definition": "error_baseline/full = observed - prediction_baseline/full, on the retained model response scale; derived during export without prediction or fitting.",
                "scales": "See each result's response measurement, configuration and fold metadata. delta_loglik is full minus baseline in nats on the recorded scoring scale, without repeated-CV normalization.",
            },
        )
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
