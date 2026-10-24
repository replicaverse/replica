"""Benchmark runner: evaluate across multiple datasets and feature specs."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

from replica.analysis.contracts.results import AnalysisResult
from replica.analysis.evaluation.encoding import EncodingResult
from replica.analysis.evaluation.evaluate import evaluate
from replica.analysis.specification.evaluation import EvaluationSpec

logger = logging.getLogger("replica")

if TYPE_CHECKING:
    from replica.analysis.specification.config import AnalysisConfig, EncodingConfig
    from replica.dataset import Dataset


def benchmark(
    datasets: Dataset | list[Dataset],
    evaluation_specs: EvaluationSpec
    | Callable[[Dataset], EvaluationSpec]
    | list[EvaluationSpec | Callable[[Dataset], EvaluationSpec]],
    analysis: AnalysisConfig | EncodingConfig | None = None,
    *,
    language_map: dict[str, str] | None = None,
) -> list[AnalysisResult | EncodingResult]:
    """Run evaluate() across multiple datasets and feature specs.

    Computes the Cartesian product of (dataset, evaluation_spec) and calls
    ``evaluate()`` for each pair.

    Args:
        datasets: One or more loaded datasets.
        evaluation_specs: Bound specifications or factories accepting the current
            dataset. A factory explicitly binds dataset features for each corpus;
            bound specifications retain their original source identities.
        analysis: Analysis configuration (shared across all evaluations).
            ``None`` uses defaults.
        language_map: Optional mapping from dataset name to language string.
            Stored on each AnalysisResult.language for cross-language aggregation.

    Returns:
        Flat list of scalar or encoding results across all combinations.
    """
    if not isinstance(datasets, list):
        datasets = [datasets]
    if not isinstance(evaluation_specs, list):
        evaluation_specs = [evaluation_specs]

    n_combos = len(datasets) * len(evaluation_specs)
    logger.info(
        "benchmark | %d dataset(s) x %d spec(s) = %d combination(s)",
        len(datasets),
        len(evaluation_specs),
        n_combos,
    )

    results: list[AnalysisResult | EncodingResult] = []
    combo_idx = 0

    for dataset in datasets:
        lang = language_map.get(dataset.info.name) if language_map else None
        for declaration in evaluation_specs:
            spec = (
                declaration
                if isinstance(declaration, EvaluationSpec)
                else declaration(dataset)
            )
            combo_idx += 1
            logger.info(
                "benchmark | [%d/%d] %s x %s",
                combo_idx,
                n_combos,
                dataset.info.name,
                ", ".join(feature.name for feature in spec.predictors),
            )
            results.extend(
                evaluate(
                    dataset=dataset,
                    spec=spec,
                    analysis=analysis,
                    language=lang,
                )
            )

    logger.info("benchmark | Done, %d total result(s).", len(results))
    return results
