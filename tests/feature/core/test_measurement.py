"""Reporting units, explicit feature descriptions, and composition boundaries."""

import json

import numpy as np
import pytest

from replica.feature import (
    DerivedFeature,
    Entropy,
    FeatureMeasurement,
    FeatureModelOutput,
    InformationValue,
    Interaction,
    InUnit,
    Lag,
    LogFrequency,
    ModelFeature,
    NextWordInformationValue,
    PrecomputedFeature,
    PrecomputedFeatureModel,
    RenyiEntropy,
    SampledEntropy,
    SimilarityAdjustedSurprisal,
    Surprisal,
    UnigramSurprisal,
    UnitLength,
    UserTable,
    describe_feature,
    resolve,
)
from tests._stimuli import word_stimulus


class _Material:
    def __init__(self, stimulus):
        self._stimulus = stimulus

    def stimuli(self):
        return [self._stimulus]


def _bound(extractor, stimulus=None, outputs=()):
    stimulus = stimulus or word_stimulus("word")
    return ModelFeature(
        PrecomputedFeatureModel("stored", {stimulus.stimulus_id: list(outputs)}),
        extractor,
    )


def _values(feature, stimulus):
    return resolve(_Material(stimulus), feature).blocks[0].values


@pytest.mark.parametrize(
    "feature, expected_nats",
    [
        (Surprisal(), -np.log(0.25)),
        (Entropy(), -0.25 * np.log(0.25) - 0.75 * np.log(0.75)),
        (RenyiEntropy(0.5), 2 * np.log(np.sqrt(0.25) + np.sqrt(0.75))),
        (SimilarityAdjustedSurprisal(), -np.log(0.6)),
    ],
)
def test_nonuniform_measurements_convert_without_changing_the_raw_outputs(
    feature, expected_nats
):
    # The zero-mass outcome exercises 0 log 0; it is not a missing observation.
    log_probs = np.array([np.log(0.25), np.log(0.75), -np.inf])
    outputs = [
        FeatureModelOutput(
            values={
                "log_p_observed": np.log(0.25),
                "next_log_probs": log_probs.copy(),
                "similarity_adjusted_surprisal": -np.log(0.6),
            }
        )
    ]
    stimulus = word_stimulus("word")
    converted = InUnit(_bound(feature, stimulus, outputs), "nats")
    np.testing.assert_allclose(_values(converted, stimulus), [expected_nats])
    np.testing.assert_allclose(
        _values(InUnit(converted, "bits"), stimulus), [expected_nats / np.log(2)]
    )
    assert converted.dependencies[0].model_requests[0].outputs == tuple(
        sorted(feature.required_outputs)
    )
    np.testing.assert_array_equal(outputs[0].values["next_log_probs"], log_probs)
    assert outputs[0].values["log_p_observed"] == np.log(0.25)


def test_retained_samples_and_stored_estimates_share_the_reporting_boundary():
    samples = np.log([0.25, 0.75, 0.75, 0.75])
    expected = -samples.mean()
    outputs = [
        FeatureModelOutput(
            values={
                "log_p_samples_nats": samples.copy(),
                "mc_word_entropy": expected / np.log(2),
            }
        )
    ]
    stimulus = word_stimulus("word")
    for feature in [SampledEntropy(), PrecomputedFeature("stored", "mc_word_entropy")]:
        np.testing.assert_allclose(
            _values(InUnit(_bound(feature, stimulus, outputs), "nats"), stimulus),
            [expected],
        )
    np.testing.assert_array_equal(outputs[0].values["log_p_samples_nats"], samples)


def test_frequency_conversion_keeps_sign_and_correct_pmi_arithmetic():
    source = UserTable({"word": 0.25})
    stimulus = word_stimulus("word")
    outputs = [FeatureModelOutput(values={"log_p_observed": np.log(0.5)})]
    np.testing.assert_allclose(
        _values(InUnit(LogFrequency(source), "nats"), stimulus), [np.log(0.25)]
    )
    pmi = DerivedFeature(
        "pmi",
        lambda unigram, surprisal: unigram - surprisal,
        UnigramSurprisal(source),
        _bound(Surprisal(), stimulus, outputs),
        measurement=FeatureMeasurement("pointwise_mutual_information", "bits"),
    )
    np.testing.assert_allclose(
        _values(InUnit(pmi, "nats"), stimulus), [np.log(0.5 / 0.25)]
    )
    description = describe_feature(pmi)["measurement"]
    assert description["parameters"]["operation"] == "caller_supplied"
    assert description["parameters"]["sources"][0]["measurement"]["unit"] == "bits"
    assert (
        describe_feature(DerivedFeature("opaque", lambda x: x, _bound(Surprisal())))[
            "measurement"
        ]
        is None
    )


def test_conversion_preserves_missing_values_and_composition_descriptions():
    stimulus = word_stimulus("one two three")
    outputs = [
        FeatureModelOutput(values={"surprisal": x}) for x in [np.nan, np.inf, 2.0]
    ]
    feature = InUnit(_bound(Surprisal(), stimulus, outputs), "nats")
    np.testing.assert_equal(_values(feature, stimulus), [np.nan, np.inf, 2 * np.log(2)])
    lagged = Lag(feature)
    np.testing.assert_equal(_values(lagged, stimulus), [np.nan, np.nan, np.inf])
    assert lagged.measurement.unit == "nats"
    assert lagged.measurement.parameters["source"]["name"] == "surprisal_nats"
    product = Interaction(feature, UnitLength())
    assert product.measurement.unit == "(nats)*(characters)"
    with pytest.raises(ValueError, match="converts only"):
        InUnit(product, "bits")
    record = json.loads(json.dumps(describe_feature(product), allow_nan=False))
    assert record["measurement"]["quantity"] == "product"


def test_custom_features_must_declare_units_and_keep_unknown_output_requirements():
    class Custom:
        name = "custom"

        def __call__(self, stimulus, outputs, target_layer="words"):
            return np.array([1.0])

    feature = Custom()
    assert describe_feature(feature)["measurement"] is None
    with pytest.raises(ValueError, match="explicitly declared"):
        InUnit(_bound(feature), "nats")
    feature.measurement = FeatureMeasurement("surprisal", "bits")
    converted = InUnit(_bound(feature, outputs=[FeatureModelOutput()]), "nats")
    assert converted.dependencies[0].model_requests[0].outputs is None
    np.testing.assert_equal(_values(converted, word_stimulus("word")), [np.log(2)])
    assert PrecomputedFeature("custom", "arbitrary_column").measurement is None
    with pytest.raises(ValueError, match="converts only"):
        InUnit(_bound(feature), "seconds")


def test_measurement_descriptions_are_detached_and_reject_unrecordable_parameters():
    feature = InUnit(_bound(RenyiEntropy(0.5)), "nats")
    record = describe_feature(feature)
    source = record["measurement"]["parameters"]["source"]
    assert source["measurement"]["parameters"]["order"] == 0.5
    source["measurement"]["parameters"]["order"] = 2.0
    assert (
        describe_feature(feature)["measurement"]["parameters"]["source"]["measurement"][
            "parameters"
        ]["order"]
        == 0.5
    )
    with pytest.raises(TypeError):
        FeatureMeasurement("custom", "bits", {"function": lambda x: x})
    with pytest.raises(ValueError):
        FeatureMeasurement("custom", "bits", {"order": np.nan})


def test_conversion_tracks_parameters_and_refuses_relabeling_storage_units():
    entropy = RenyiEntropy(0.5)
    converted = InUnit(_bound(entropy), "nats")
    entropy.alpha = 2.0
    record = describe_feature(converted)
    assert (
        record["measurement"]["parameters"]["source"]["measurement"]["parameters"][
            "order"
        ]
        == 2.0
    )
    with pytest.raises(ValueError, match="use InUnit"):
        PrecomputedFeature(
            "surprisal",
            "surprisal",
            measurement=FeatureMeasurement("surprisal", "nats"),
        )
    source = PrecomputedFeature(
        "custom", "custom", measurement=FeatureMeasurement("entropy", "bits")
    )
    converted = InUnit(
        _bound(source, outputs=[FeatureModelOutput(values={"custom": 1.0})]), "nats"
    )
    source.measurement = FeatureMeasurement("entropy", "hartleys")
    with pytest.raises(ValueError, match="unit changed"):
        _values(converted, word_stimulus("word"))


@pytest.mark.parametrize(
    "feature, attribute, replacement",
    [
        (InformationValue(), "distance", "length"),
        (SimilarityAdjustedSurprisal(), "similarity", "pos"),
        (NextWordInformationValue(), "similarity", "orthographic"),
    ],
)
def test_stored_feature_descriptions_follow_the_selected_output_key(
    feature, attribute, replacement
):
    description = describe_feature(feature)
    (key,) = feature.required_outputs
    outputs = [FeatureModelOutput(values={key: 0.2})]
    setattr(feature, attribute, replacement)
    np.testing.assert_array_equal(feature(word_stimulus("word"), outputs), [0.2])
    assert describe_feature(feature) == description
