"""Public predictive inference on datasets using explicit model declarations."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

from replica.analysis.contracts.results import ComparisonSpec
from replica.analysis.data.preparation import (
    _all_features,
    _model_name,
    _prepare_target,
    _resolved_inputs,
    _validate_named_analysis,
)
from replica.analysis.specification.config import AnalysisConfig

from .crossed_bootstrap import (
    CrossedBootstrapConfig,
    _new_checkpoint_directory,
    _validate_analysis,
    crossed_bootstrap_test,
)
from .crossed_statistics import METHOD, PredictiveTestResult

if TYPE_CHECKING:
    from replica.analysis.specification.evaluation import EvaluationSpec

    from ...dataset import Dataset


def predictive_test(
    dataset: Dataset,
    spec: EvaluationSpec,
    analysis: AnalysisConfig | None = None,
    *,
    inference: CrossedBootstrapConfig | None = None,
    checkpoint_dir: str | Path | None = "auto",
) -> list[PredictiveTestResult]:
    """Test predictive gain with a full-refit crossed bootstrap by default.

    Requesting inference is explicit: :func:`evaluate` remains an estimate-only
    call. This method refits preprocessing and both models, starting from the
    observations rather than from saved evaluation scores. It supports two
    unseen experimental axes and marginal Gaussian scores. Unsupported designs
    fail before feature extraction; no row-wise or one-axis test is substituted.

    Args:
        dataset: Loaded corpus of individual responses.
        spec: Predictors, baseline controls and response measures. Sequence
            features are prepared on original materials before resampling.
        analysis: Model, response scale and sampling design. None uses the same
            ``AnalysisConfig()`` defaults as ``evaluate``: an ML mixed model,
            raw responses, crossed holdout with 90% training per axis and
            marginal Gaussian scoring.
        inference: Bootstrap budgets and execution options. None selects 500
            draws per series, two cycles per draw and eight original cycles.
        checkpoint_dir: "auto" creates one unique user-cache root, logged before
            fitting and recorded as checkpoint_root in each result. Pass that root
            to resume all comparisons and responses. An explicit path chooses the
            root; numbered subdirectories separate the tests. None disables it.

    Returns:
        One result per comparison and response. Failed cycles yield unavailable
        inference, preserving complete original-data point estimates when possible.
    """
    spec.__post_init__()
    if not spec.predictors:
        raise ValueError("Supply at least one focal predictor")
    analysis = (analysis or AnalysisConfig()).resolve(
        tuple(p.name for p in spec.predictors)
    )
    declaration, _ = _validate_analysis(analysis)
    _validate_named_analysis(analysis, analysis.model, spec)
    settings = inference or CrossedBootstrapConfig()
    root = (
        _new_checkpoint_directory()
        if checkpoint_dir == "auto"
        else (None if checkpoint_dir is None else Path(checkpoint_dir))
    )
    specs = (
        [replace(spec, predictors=[p], comparison="joint") for p in spec.predictors]
        if spec.comparison == "individual"
        else [spec]
    )
    _, resolved = _resolved_inputs(dataset, spec, analysis)
    results = {}
    all_target_names = {target.name for target in spec.targets}
    for dv_index, target in enumerate(spec.targets):
        shared = None
        for comparison_index, contrast in enumerate(specs):
            comparison_spec = replace(contrast, targets=[target])
            comparison = ComparisonSpec(
                tuple(f.name for f in comparison_spec.controls),
                tuple(f.name for f in comparison_spec.predictors),
            )
            names = (*comparison.baseline, *comparison.added)
            selected = {*names, target.name}
            scoped = replace(
                analysis,
                design=declaration.for_model(names, warn=False),
                preprocessing=tuple(
                    step
                    for step in analysis.preprocessing
                    if step.feature.name in selected
                ),
            )
            dv_name = target.name
            if spec.cohort == "shared":
                if shared is None:
                    common = replace(spec, targets=[target], comparison="joint")
                    common_config = replace(
                        analysis,
                        preprocessing=tuple(
                            step
                            for step in analysis.preprocessing
                            if step.feature.name not in all_target_names
                            or step.feature.name == target.name
                        ),
                    )
                    shared = _prepare_target(
                        dataset,
                        common,
                        common_config,
                        target,
                        _all_features(common, common_config),
                        resolved,
                    )
                observations = shared
            else:
                observations = _prepare_target(
                    dataset,
                    comparison_spec,
                    scoped,
                    target,
                    _all_features(comparison_spec, scoped),
                    resolved,
                )
            destination = (
                None
                if root is None
                else root / f"comparison-{comparison_index:03d}-dv-{dv_index:03d}"
            )
            result = crossed_bootstrap_test(
                observations,
                comparison,
                scoped,
                inference=settings,
                checkpoint_dir=destination,
            )
            results[(comparison_index, dv_index)] = replace(
                result,
                metadata={
                    **result.metadata,
                    "dataset_name": dataset.info.name,
                    "model_name": _model_name(
                        resolved[id(feature)]
                        for feature in (
                            *comparison_spec.targets,
                            *_all_features(comparison_spec, scoped),
                        )
                    ),
                    "dv_name": dv_name,
                    "feature_measurements": observations.response[
                        "feature_measurements"
                    ],
                    "checkpoint_root": None if root is None else str(root.resolve()),
                    "configuration": {
                        **scoped.metadata(),
                        "predictive_significance": METHOD,
                    },
                },
            )
    return [results[key] for key in sorted(results)]
