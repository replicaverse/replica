"""Numerical pins for the MC Rényi aggregation estimator.

Pure-math tests (no model load): for a perfectly-representative sample the
importance-sampling estimator must equal the closed-form Rényi entropy. They
sit apart from ``test_entropy_predictors.py``, which is ``needs_model``, so
they run in the fast tier.
"""

# Mixing log bases in the aggregation scales p^(alpha-1) to
# p^((alpha-1)/ln 2); these tests pin the aggregation against the analytical
# Rényi entropy of a hand-computed distribution.

import numpy as np
import pytest

from replica.feature import (
    FeatureModelOutput,
    FeatureNotAvailable,
    PrecomputedFeatureModel,
    SampledEntropy,
)
from replica.feature.entropy.estimators import (
    _renyi_entropy_bits_from_log_p_nats,
)
from tests._stimuli import word_stimulus


class TestRenyiAggregationNumerical:
    """For a perfectly-representative sample of a known distribution, the
    importance-sampling estimator reduces to the exact analytical Rényi
    entropy. Any deviation, such as mixed log bases, shows up as a
    numerical mismatch."""

    @pytest.fixture
    def p(self):
        # Three-outcome distribution with a non-trivial mass ratio so
        # Σ p^alpha differs visibly across alpha.
        return np.array([0.5, 0.25, 0.25])

    @pytest.fixture
    def log_p_nats(self, p):
        # "Perfect" N=4 sample: outcome i appears N · p_i times. Then
        # the MC estimator Σ p(s^(i))^(alpha-1) / N exactly equals Σ p^alpha.
        return np.array(
            [
                np.log(0.5),
                np.log(0.5),  # 2 of the p=0.5 outcome
                np.log(0.25),  # 1 of the first p=0.25 outcome
                np.log(0.25),  # 1 of the second p=0.25 outcome
            ]
        )

    @pytest.mark.parametrize("alpha", [0.0, 1.0, 2.0, 0.5, 3.0])
    def test_matches_closed_form(self, p, log_p_nats, alpha):
        """For a perfectly-representative sample the estimator equals the
        exact Rényi entropy H_alpha = log_2(Σ p^alpha)/(1-alpha) (Shannon at alpha=1), e.g.
        collision -log_2 Σp² at alpha=2, 2·log_2 Σ√p at alpha=0.5. Mixed log
        bases change the estimate at every alpha other than 1."""
        if alpha == 1.0:
            h_true = float(-np.sum(p * np.log2(p)))
        else:
            h_true = float(np.log2(np.sum(p**alpha)) / (1.0 - alpha))
        h_mc = _renyi_entropy_bits_from_log_p_nats(log_p_nats, alpha=alpha)
        np.testing.assert_allclose(h_mc, h_true, atol=1e-12)
        assert _renyi_entropy_bits_from_log_p_nats(np.array([0.0]), alpha) == 0.0
        assert np.isnan(_renyi_entropy_bits_from_log_p_nats(np.empty(0), alpha))

        stimulus = word_stimulus("a b")
        original_samples = log_p_nats.copy()
        outputs = [
            FeatureModelOutput(values={"log_p_samples_nats": np.empty(0)}),
            FeatureModelOutput(
                values={"log_p_samples_nats": log_p_nats, "mc_word_entropy": 1.5},
                metadata={"alpha": 1.0, "model_name": "retained-samples"},
            ),
        ]
        model = PrecomputedFeatureModel(
            "retained-samples", {stimulus.stimulus_id: outputs}
        )
        feature = SampledEntropy(alpha)
        estimates = feature(
            stimulus, model(stimulus, extract=list(feature.required_outputs))
        )
        assert np.isnan(estimates[0])
        np.testing.assert_allclose(estimates[1], h_true, atol=1e-12)
        np.testing.assert_array_equal(
            outputs[1].values["log_p_samples_nats"], original_samples
        )
        assert outputs[1].values["mc_word_entropy"] == 1.5
        assert outputs[1].metadata == {"alpha": 1.0, "model_name": "retained-samples"}

    def test_alpha_1_matches_renyi_limit(self, p, log_p_nats):
        """The alpha=1 fast-path must equal the alpha→1 limit of the alpha≠1
        branch. Use alpha=1+ε and alpha=1-ε and check both converge to the
        alpha=1 fast-path output."""
        h_one = _renyi_entropy_bits_from_log_p_nats(log_p_nats, alpha=1.0)
        h_lo = _renyi_entropy_bits_from_log_p_nats(log_p_nats, alpha=1.0 - 1e-6)
        h_hi = _renyi_entropy_bits_from_log_p_nats(log_p_nats, alpha=1.0 + 1e-6)
        # Loose tolerance: numerical derivative is ill-conditioned at alpha=1.
        np.testing.assert_allclose(h_lo, h_one, atol=1e-4)
        np.testing.assert_allclose(h_hi, h_one, atol=1e-4)


@pytest.mark.parametrize("alpha", [-1, np.nan, np.inf, -np.inf])
def test_sampled_entropy_rejects_invalid_order(alpha):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        SampledEntropy(alpha)
    with pytest.raises(ValueError, match="finite and nonnegative"):
        _renyi_entropy_bits_from_log_p_nats(np.array([-1.0]), alpha)


@pytest.mark.parametrize(
    "samples",
    [
        0.0,
        [[-1.0]],
        [np.nan],
        [-np.inf],
        [np.inf],
        [0.1],
        [-1.0, 0.1],
        [np.nextafter(0.0, 1.0)],
        [-1 + 2j],
        ["-1"],
        pytest.param(
            np.array([-np.finfo(np.longdouble).max]),
            marks=pytest.mark.skipif(
                np.finfo(np.longdouble).max <= np.finfo(np.float64).max,
                reason="longdouble has no wider range than float64",
            ),
            id="float64-overflow",
        ),
    ],
)
@pytest.mark.parametrize("alpha", [0.5, 1.0])
def test_sampled_entropy_rejects_invalid_samples(samples, alpha):
    with pytest.raises(ValueError, match="finite nonpositive"):
        _renyi_entropy_bits_from_log_p_nats(samples, alpha)
    outputs = [FeatureModelOutput(values={"log_p_samples_nats": samples})]
    with pytest.raises(ValueError, match=r"finite nonpositive.*unit 0"):
        SampledEntropy(alpha)(word_stimulus("a"), outputs)


def test_sampled_entropy_missing_samples_are_not_undefined_positions():
    stimulus = word_stimulus("a")
    with pytest.raises(ValueError, match="requires model outputs"):
        SampledEntropy()(stimulus, None)
    with pytest.raises(FeatureNotAvailable, match="log_p_samples_nats"):
        SampledEntropy()(stimulus, [FeatureModelOutput()])


@pytest.mark.parametrize(
    "values,metadata",
    [
        ({"continuation_entropy_buggy": 1.0}, {}),
        ({}, {"model_name": "cont_entropy_buggy_gpt2_N64_L10"}),
    ],
)
def test_sampled_entropy_refuses_known_incorrect_sampling(values, metadata):
    stimulus = word_stimulus("a")
    output = FeatureModelOutput(
        values={"log_p_samples_nats": [-1.0], **values}, metadata=metadata
    )
    model = PrecomputedFeatureModel(
        "retained-samples", {stimulus.stimulus_id: [output]}
    )
    feature = SampledEntropy()
    with pytest.raises(ValueError, match="sampling and log probabilities do not match"):
        feature(stimulus, model(stimulus, extract=list(feature.required_outputs)))
