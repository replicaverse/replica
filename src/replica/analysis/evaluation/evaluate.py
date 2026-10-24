"""Run scalar model comparison or sampled-response encoding."""

from __future__ import annotations

import logging
from collections.abc import Callable
from copy import deepcopy
from dataclasses import replace
from typing import (
    TYPE_CHECKING,
    overload,
)

from replica.analysis.contracts.results import AnalysisResult
from replica.analysis.cv.group_holdout import CrossedGroupSplitter
from replica.analysis.cv.scalers import ZScoreScaler
from replica.analysis.data.preparation import (
    _all_features,
    _model_name,
    _prepare_target,
    _resolved_inputs,
    _validate_named_analysis,
)
from replica.analysis.evaluation.encoding import (
    EncodingFoldArtifact,
    EncodingResult,
    _evaluate_sampled_response,
)
from replica.analysis.fitting.diagnostics import execution_environment
from replica.analysis.fitting.regression import (
    _build_design,
    _RegressionSetup,
    _run_custom_analysis_for_dv,
    _run_regression_for_dv,
)
from replica.analysis.specification.config import AnalysisConfig, EncodingConfig

logger = logging.getLogger("replica")


if TYPE_CHECKING:
    from replica.analysis.specification.evaluation import EvaluationSpec
    from replica.dataset import Dataset


@overload
def evaluate(
    dataset: Dataset,
    spec: EvaluationSpec,
    analysis: EncodingConfig,
    *,
    language: str | None = None,
    encoding_fold_callback: Callable[[EncodingFoldArtifact], None] | None = None,
) -> list[EncodingResult]: ...


@overload
def evaluate(
    dataset: Dataset,
    spec: EvaluationSpec,
    analysis: AnalysisConfig | None = None,
    *,
    language: str | None = None,
) -> list[AnalysisResult]: ...


def evaluate(
    dataset: Dataset,
    spec: EvaluationSpec,
    analysis: AnalysisConfig | EncodingConfig | None = None,
    *,
    language: str | None = None,
    encoding_fold_callback: Callable[[EncodingFoldArtifact], None] | None = None,
) -> list[AnalysisResult] | list[EncodingResult]:
    """Evaluate features against a scalar or sampled response.

    Align predictors and responses, apply the declared response policy, and
    compare baseline and full models on held-out observations. Scaling and
    residualization are fitted within each training fold.

    Args:
        dataset: Dataset defining the selected material population.
        spec: Indexed targets, predictors and controls,
            the predictors scored against a baseline built from the controls.
        analysis: An :class:`AnalysisConfig` evaluates scalar, unit-aligned
            dependent variables. An :class:`EncodingConfig` evaluates sampled
            multichannel recordings with ridge encoding, using the predictors,
            controls and selected hidden-state layers from *spec*. None uses ``AnalysisConfig()``:
            an ML mixed model on raw responses, participant intercepts and focal
            slopes with zero correlations, occurrence intercepts, 90% training
            per crossed axis and marginal Gaussian scoring. This call estimates
            predictive gain; ``predictive_test`` runs bootstrap inference with the same defaults.
        language: Stored on each AnalysisResult.language for cross-language
            aggregation (e.g. MECO).
        encoding_fold_callback: Called synchronously after each fitted encoding
            fold with an :class:`EncodingFoldArtifact`. Use it to stream fitted
            weights, held-out data, and preprocessing state to storage. ``None``
            retains only the usual aggregate results. Requires EncodingConfig;
            callback exceptions propagate to the caller.

    Returns:
        Scalar configurations return one :class:`AnalysisResult` per comparison and
        DV. Encoding configurations return one :class:`EncodingResult` per
        layer and recording.

    """
    spec.__post_init__()
    if isinstance(analysis, EncodingConfig):
        return _evaluate_sampled_response(
            dataset,
            spec,
            analysis,
            language=language,
            encoding_fold_callback=encoding_fold_callback,
        )
    if encoding_fold_callback is not None:
        raise ValueError("encoding_fold_callback requires EncodingConfig")
    base = analysis or AnalysisConfig()
    contrasts = (
        [
            replace(spec, predictors=[feature], comparison="joint")
            for feature in spec.predictors
        ]
        if spec.comparison == "individual"
        else [spec]
    )
    configurations = {}
    for target_index, target in enumerate(spec.targets):
        for contrast_index, contrast in enumerate(contrasts):
            current = replace(contrast, targets=[target])
            selected = {
                f.name
                for f in (*current.targets, *current.predictors, *current.controls)
            }
            config = replace(
                base,
                preprocessing=tuple(
                    step for step in base.preprocessing if step.feature.name in selected
                ),
            ).resolve(tuple(feature.name for feature in current.predictors))
            _validate_named_analysis(config, config.model, current)
            configurations[(contrast_index, target_index)] = config
    _, resolved = _resolved_inputs(dataset, spec, base)
    result_batches = {}
    target_names = {feature.name for feature in spec.targets}
    for target_index, target in enumerate(spec.targets):
        shared_table = shared_splits = shared_metadata = shared_splitter = None
        for contrast_index, contrast in enumerate(contrasts):
            current = replace(contrast, targets=[target])
            config = configurations[(contrast_index, target_index)]
            if spec.cohort == "shared":
                if shared_table is None:
                    common = replace(spec, targets=[target], comparison="joint")
                    common_config = replace(
                        base,
                        preprocessing=tuple(
                            step
                            for step in base.preprocessing
                            if step.feature.name not in target_names
                            or step.feature.name == target.name
                        ),
                    ).resolve(tuple(feature.name for feature in spec.predictors))
                    common_features = _all_features(common, common_config)
                    shared_table = _prepare_target(
                        dataset,
                        common,
                        common_config,
                        target,
                        common_features,
                        resolved,
                    )
                table = shared_table
            else:
                table = _prepare_target(
                    dataset,
                    current,
                    config,
                    target,
                    _all_features(current, config),
                    resolved,
                )
            assert config.model is not None
            setup = _RegressionSetup(
                dataset=dataset,
                observations=table,
                all_features=_all_features(current, config),
                features=current.predictors,
                control_features=current.controls,
                feature_model_name=_model_name(
                    resolved[id(feature)]
                    for feature in (*current.targets, *_all_features(current, config))
                ),
                language=language,
                analysis_model=config.model,
                cv_splitter=shared_splitter or config.resolve_cv(),
                scaler=config.scaler or ZScoreScaler(),
                use_training_sigma=config.sigma_source == "training",
                empty_control_baseline=config.empty_control_baseline,
                per_fold=config.per_fold,
                retain_per_obs_delta_ll=config.retain_per_obs_delta_ll,
                prediction_diagnostics=config.prediction_diagnostics,
                retain_predictions=config.retain_predictions,
                dump_merged_path=config.dump_merged_path,
                preprocessing=config.preprocessing,
                declaration=config.design,
                score_scale=config.score_scale,
                target_transform=config.transform_dv,
                default_cv=config.cv is None,
                fixed_splits=shared_splits,
                fixed_split_metadata=shared_metadata,
            )
            if config.analysis_fn is not None:
                target_results = _run_custom_analysis_for_dv(
                    setup, target.name, config.analysis_fn
                )
            else:
                design = _build_design(setup, target.name)
                if design is None:
                    continue
                if spec.cohort == "shared" and shared_splits is None:
                    shared_splits, shared_metadata = (
                        design.splits,
                        design.splitter_metadata,
                    )
                    shared_splitter = setup.cv_splitter
                target_results = _run_regression_for_dv(
                    setup, target.name, design=design
                )
            for result in target_results:
                result.response = {
                    **table.response,
                    "model_name": setup.feature_model_name,
                }
                result.metadata.update(
                    configuration=config.metadata(),
                    feature_measurements=table.response["feature_measurements"],
                    feature_sources=table.response["feature_sources"],
                    cohort_identity=table.response["cohort_identity"],
                    target_identity=table.response["target_identity"],
                    cohort=spec.cohort,
                    score_scale=config.score_scale,
                )
                if config.design is not None:
                    result.metadata.update(
                        design=config.design.metadata(),
                        cv_splitter=type(setup.cv_splitter).__name__,
                        cv_override=config.cv is not None,
                        sigma_source=config.sigma_source,
                        scoring=f"{config.design.scoring}_gaussian",
                        cv_seed=getattr(setup.cv_splitter, "seed", None),
                        cv_cycles=1
                        if isinstance(setup.cv_splitter, CrossedGroupSplitter)
                        else None,
                    )
                    if isinstance(setup.cv_splitter, CrossedGroupSplitter):
                        result.metadata["crossed_cv"] = setup.cv_splitter.summary()
            result_batches[(contrast_index, target_index)] = target_results
    results = [
        result for key in sorted(result_batches) for result in result_batches[key]
    ]
    environment = execution_environment()
    for result in results:
        result.metadata["execution_environment"] = deepcopy(environment)
    return results
