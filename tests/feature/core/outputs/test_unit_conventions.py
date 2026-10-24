"""Default reporting units and missing-output contracts.

Raw log probabilities use nats. The entropy and surprisal extractors tested
here default to bits; similarity-adjusted surprisal uses nats. Explicit unit
conversion and nonuniform numerical references live in test_measurement.py.
"""

import numpy as np
import pytest

from replica.core import Stimulus
from replica.feature import (
    ContinuationEntropy,
    ContinuationEntropyBuggy,
    Entropy,
    ExpectedProbability,
    FeatureModelOutput,
    FeatureNotAvailable,
    MCWordEntropy,
    MergedFeatureModel,
    PrecomputedFeatureModel,
    Rank,
    RenyiEntropy,
    Surprisal,
    SurprisalReduction,
    TokenRenyiEntropySum,
)
from replica.feature.entropy import ReproductionOnlyWarning
from tests._stimuli import word_stimulus


def _stim(n_units: int) -> Stimulus:
    return word_stimulus(" ".join(f"w{i}" for i in range(n_units)))


def _uniform_outputs(n_units: int, vocab_n: int) -> list[FeatureModelOutput]:
    """Outputs with a uniform next_log_probs over vocab_n (each log p = -ln N
    nats) and log_p_observed = -ln 2 nats at t>0 (NaN at t=0)."""
    nlp = np.full(vocab_n, -np.log(vocab_n))  # uniform distribution, in nats
    return [
        FeatureModelOutput(
            values={
                "log_p_observed": np.nan if t == 0 else -np.log(2.0),
                "next_log_probs": nlp,
            }
        )
        for t in range(n_units)
    ]


# --- base-pin: entropy of a uniform distribution is log2(N) bits ---


def test_entropy_of_uniform_is_log2_bits():
    feature = Entropy()
    assert feature.name == "entropy"
    vals = feature(_stim(3), _uniform_outputs(3, 8))
    assert np.allclose(vals, 3.0, atol=1e-9), vals


@pytest.mark.parametrize("alpha", [0.0, 1.0, 2.0])
def test_renyi_of_uniform_is_log2_bits(alpha):
    # Cover support-count, Shannon-limit, and general-order branches.
    vals = RenyiEntropy(alpha=alpha)(_stim(3), _uniform_outputs(3, 8))
    assert np.allclose(vals, 3.0, atol=1e-9), (alpha, vals)


def test_expected_probability_of_uniform_is_one_over_n():
    vals = ExpectedProbability()(_stim(3), _uniform_outputs(3, 8))
    assert np.allclose(vals[1:], 1 / 8, atol=1e-12), vals


# --- base-pin: surprisal converts nats log_p_observed to bits ---


@pytest.mark.parametrize(
    "feature, name, expected",
    [
        (Surprisal(), "surprisal", 1.0),
        (SurprisalReduction(), "surprisal_reduction", 2.0),
    ],
)
def test_log_probability_features_report_bits_and_preserve_missing_values(
    feature, name, expected
):
    assert feature.name == name
    vals = feature(_stim(3), _uniform_outputs(3, 8))
    assert vals.shape == (3,)
    assert np.isnan(vals[0])
    np.testing.assert_allclose(vals[1:], expected, atol=1e-12)


# --- contract: every distribution feature raises without next_log_probs ---


@pytest.mark.parametrize(
    "feat",
    [
        Entropy(),
        RenyiEntropy(alpha=0.5),
        ExpectedProbability(),
        Rank(),
        SurprisalReduction(),
    ],
    ids=["entropy", "renyi", "expected_prob", "rank", "surprisal_reduction"],
)
def test_distribution_feature_raises_without_next_log_probs(feat):
    outs = [
        FeatureModelOutput(
            values={
                "log_p_observed": np.nan if t == 0 else -1.0,
                "next_log_probs": None,
            }
        )
        for t in range(3)
    ]
    with pytest.raises(FeatureNotAvailable):
        feat(_stim(3), outs)


# --- contract: surprisal works without a distribution (vocab=None case) ---


def test_surprisal_works_without_next_log_probs():
    outs = [
        FeatureModelOutput(
            values={"log_p_observed": np.nan if t == 0 else -np.log(2.0)}
        )
        for t in range(3)
    ]
    vals = Surprisal()(_stim(3), outs)
    assert np.isnan(vals[0])
    assert np.allclose(vals[1:], 1.0, atol=1e-12), vals


# --- contract: a stored surprisal in bits is read as is ---


def test_surprisal_reads_the_stored_bits_column():
    outs = [
        FeatureModelOutput(values={"surprisal": 5.0}),
        FeatureModelOutput(values={"log_p_observed": -np.log(2.0), "surprisal": None}),
        FeatureModelOutput(values={"log_p_observed": -np.log(2.0)}),
    ]
    vals = Surprisal()(_stim(3), outs)
    # The bits column is read where it holds a value; the nats column is
    # converted where it is absent or empty.
    assert np.allclose(vals, [5.0, 1.0, 1.0], atol=1e-12), vals


def _table(name: str, values: dict) -> PrecomputedFeatureModel:
    return PrecomputedFeatureModel(
        name, {"t": [FeatureModelOutput(values=dict(values)) for _ in range(3)]}
    )


def test_merged_live_log_probability_replaces_the_table_surprisal():
    # A derived word table carries both forms (1 bit); the live model 3 bits.
    table = _table("table", {"log_p_observed": -np.log(2.0), "surprisal": 1.0})
    live = _table("live", {"log_p_observed": -3 * np.log(2.0)})
    merged = MergedFeatureModel(table, live)(_stim(3))
    assert np.allclose(Surprisal()(_stim(3), merged), 3.0, atol=1e-12)
    # A later member that stores both forms keeps its own surprisal.
    merged = MergedFeatureModel(live, table)(_stim(3))
    assert np.allclose(Surprisal()(_stim(3), merged), 1.0, atol=1e-12)


# Missing output keys raise.


def test_surprisal_raises_without_any_log_probability():
    with pytest.raises(FeatureNotAvailable, match="log_p_observed"):
        Surprisal()(_stim(3), [FeatureModelOutput() for _ in range(3)])


def _stored_estimate(feature_class):
    if feature_class is ContinuationEntropyBuggy:
        with pytest.warns(ReproductionOnlyWarning):
            return feature_class(allow_reproduction=True)
    return feature_class()


@pytest.mark.parametrize(
    "feature_class", [ContinuationEntropy, MCWordEntropy, ContinuationEntropyBuggy]
)
def test_stored_estimates_raise_when_no_output_carries_them(feature_class):
    feat = _stored_estimate(feature_class)
    outs = [FeatureModelOutput(values={"log_p_observed": -1.0}) for _ in range(3)]
    with pytest.raises(FeatureNotAvailable, match=feat.name):
        feat(_stim(3), outs)


def test_stored_estimates_keep_stored_nan_values():
    outs = [
        FeatureModelOutput(values={"mc_word_entropy": np.nan}),
        FeatureModelOutput(values={"mc_word_entropy": 2.0}),
        FeatureModelOutput(values={}),
    ]
    vals = MCWordEntropy()(_stim(3), outs)
    assert np.isnan(vals[0]) and vals[1] == 2.0 and np.isnan(vals[2])


def test_summed_renyi_hint_names_both_constructor_flags():
    with pytest.warns(ReproductionOnlyWarning):
        feat = TokenRenyiEntropySum(allow_reproduction=True)
    outs = [FeatureModelOutput(values={}, metadata={"model_name": "lm"})]
    with pytest.raises(FeatureNotAvailable, match="token_renyi_alphas=.*sum=True"):
        feat(_stim(1), outs)
