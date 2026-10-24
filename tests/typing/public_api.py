"""Checked examples of the public API for third-party extensions."""

from collections.abc import Sequence
from typing import assert_type

import numpy as np

from replica import (
    AnalysisConfig,
    AnalysisResult,
    CrossedBootstrapConfig,
    Dataset,
    EncodingConfig,
    EncodingResult,
    EvaluationSpec,
    FeatureMeasurement,
    FeatureModelOutput,
    InUnit,
    ModelFeature,
    ObservationTable,
    PredictiveTestResult,
    Stimulus,
    Surprisal,
    evaluate,
    predictive_test,
    prepare_observations,
)
from replica.feature.core.protocols import FeatureExtractor, ModelOutputRecord


class CustomModel:
    name = "custom"
    vocab = None

    def __call__(
        self,
        stimulus: Stimulus,
        extract: list[str] | None = None,
        target_layer: str = "words",
    ) -> list[FeatureModelOutput]:
        return [
            FeatureModelOutput(values={"log_p_observed": -1.0})
            for _ in stimulus.layers[target_layer].spans
        ]


class CustomExtractor:
    name = "custom_score"

    def __call__(
        self,
        stimulus: Stimulus,
        outputs: Sequence[ModelOutputRecord] | None,
        target_layer: str = "words",
    ) -> np.ndarray:
        if outputs is None:
            raise ValueError("custom_score requires outputs")
        return np.array([output.values["log_p_observed"] for output in outputs])


class CustomAnalysis:
    name = "custom"

    def fit_predict(
        self, y_train: np.ndarray, X_train: np.ndarray, X_test: np.ndarray
    ) -> np.ndarray:
        return np.full(len(X_test), y_train.mean())


class CustomLikelihood:
    name = "custom_likelihood"
    owns_loglik = True

    def fit_score(
        self,
        y_train: np.ndarray,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        *,
        is_full: bool,
    ) -> tuple[float, np.ndarray]:
        prediction = np.full(len(X_test), y_train.mean())
        return float(-np.abs(y_test - prediction).sum()), prediction


def public_contract(dataset: Dataset, encoding: EncodingConfig) -> None:
    record: ModelOutputRecord = FeatureModelOutput(values={"log_p_observed": -1.0})
    extractor: FeatureExtractor = CustomExtractor()
    assert_type(record, ModelOutputRecord)
    ModelFeature(CustomModel(), extractor)
    converted = InUnit(ModelFeature(CustomModel(), Surprisal()), "nats")
    assert_type(converted.measurement, FeatureMeasurement)
    spec = EvaluationSpec(targets=[dataset.feature("response")], predictors=[converted])
    config = AnalysisConfig(design=None, transform_dv=np.log2, model=CustomAnalysis())
    assert_type(evaluate(dataset, spec, config), list[AnalysisResult])
    likelihood = AnalysisConfig(design=None, model=CustomLikelihood())
    assert_type(evaluate(dataset, spec, likelihood), list[AnalysisResult])
    assert_type(evaluate(dataset, spec, encoding), list[EncodingResult])
    assert_type(prepare_observations(dataset, spec, config), ObservationTable)


def predictive_contract(
    dataset: Dataset, spec: EvaluationSpec, config: AnalysisConfig
) -> None:
    inference = CrossedBootstrapConfig()
    assert_type(inference.fit_budget(100), int)
    assert_type(
        predictive_test(dataset, spec, config, inference=inference),
        list[PredictiveTestResult],
    )
