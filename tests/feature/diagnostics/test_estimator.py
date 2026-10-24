"""Estimator-quality diagnostics: synthetic sanity + the three-way story.

The headline test reproduces, through ``diagnose_estimator``, the distinct
statistical character of three Monte-Carlo estimators that look identical at a
single N: the Shannon continuation-entropy MC mean (unbiased), the Renyi MC
estimator (log of a sample mean: biased at finite N but consistent, Jensen),
and the top-k truncated sampler (inconsistent: converges to a cross-entropy).
"""

import numpy as np
import pytest

from replica.feature.diagnostics import MarkovReferenceLM
from replica.feature.diagnostics.references._closed_forms import (
    continuation_entropy,
    continuation_renyi,
    topk_renorm,
)
from replica.feature.diagnostics.sampling.estimator import diagnose_estimator


class TestSynthetic:
    THETA = 3.0

    def _grid(self):
        return (16, 64, 256, 1024)

    @pytest.mark.statistical
    def test_unbiased_consistent(self):
        def est(n, seed):
            return self.THETA + np.random.default_rng(seed).normal(0, 2 / np.sqrt(n))

        rep = diagnose_estimator(est, self.THETA, n_grid=self._grid(), n_reps=400)
        assert rep.classification == "no_bias_detected", rep.to_text()
        assert -1.3 <= rep.variance_slope <= -0.7

    @pytest.mark.statistical
    def test_biased_but_consistent(self):
        # bias = -3/n, vanishes as O(1/N)
        def est(n, seed):
            return (
                self.THETA
                - 3.0 / n
                + np.random.default_rng(seed).normal(0, 1 / np.sqrt(n))
            )

        rep = diagnose_estimator(est, self.THETA, n_grid=self._grid(), n_reps=600)
        assert rep.classification == "decreasing_bias", rep.to_text()
        assert rep.bias_slope < -0.6  # roughly 1/N

    @pytest.mark.statistical
    def test_inconsistent_persistent_bias(self):
        def est(n, seed):
            return (
                self.THETA + 0.5 + np.random.default_rng(seed).normal(0, 1 / np.sqrt(n))
            )

        rep = diagnose_estimator(est, self.THETA, n_grid=self._grid(), n_reps=400)
        assert rep.classification == "persistent_bias", rep.to_text()

    @pytest.mark.statistical
    def test_ci_coverage_is_measured_and_near_nominal(self):
        def est_with_se(n, seed):
            rng = np.random.default_rng(seed)
            se = 1.5 / np.sqrt(n)
            return self.THETA + rng.normal(0, se), se

        rep = diagnose_estimator(
            est_with_se, self.THETA, n_grid=self._grid(), n_reps=800
        )
        for L in rep.levels:
            assert 0.90 <= L.coverage <= 0.99, (L.n, L.coverage)

    def test_n_grid_must_increase(self):
        with pytest.raises(ValueError, match="increasing"):
            diagnose_estimator(lambda n, s: 1.0, 1.0, n_grid=(64, 16))

    @pytest.mark.parametrize(
        "observed_bias,limiting_bias,label",
        [(1.0, 0.0, "persistent_bias"), (0.0, 1.0, "no_bias_detected")],
    )
    def test_classification_describes_only_the_tested_budgets(
        self, observed_bias, limiting_bias, label
    ):
        def estimate(n, seed):
            bias = observed_bias if n <= 64 else limiting_bias
            return bias + (-1) ** seed * 0.01 / np.sqrt(n)

        report = diagnose_estimator(estimate, 0.0, n_grid=(16, 32, 64), n_reps=20)
        assert report.classification == label
        assert abs(estimate(1_000_000, 0) - limiting_bias) < 0.0001

    @pytest.mark.parametrize(
        "bias,label", [(0.0, "no_bias_detected"), (1e-14, "persistent_bias")]
    )
    def test_constant_bias_is_not_reported_as_decreasing(self, bias, label):
        report = diagnose_estimator(
            lambda n, seed: bias, 0.0, n_grid=(16, 32, 64), n_reps=20
        )
        assert report.classification == label


# ---------------------------------------------------------------------------
# The real story: three MC estimators of a continuation quantity on the toy
# model, where theta is known in closed form.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def chain():
    ref = MarkovReferenceLM()
    return ref, ref.transition, ref.vocab.index("ga")  # a peaked context


def _sample_logp(T, state, n, rng, sampler):
    """n joint log2-probs of length-2 continuations sampled from `sampler`
    (rows), each scored under the full chain T. sampler == T is the correct
    estimator; a truncated sampler is the bug."""
    V = T.shape[0]
    cdf = np.cumsum(sampler, axis=1)
    w = (rng.random(n)[:, None] > np.cumsum(sampler[state])).sum(1).clip(0, V - 1)
    logp = np.log2(T[state, w])
    nxt = (rng.random(n)[:, None] > cdf[w]).sum(1).clip(0, V - 1)
    logp = logp + np.log2(T[w, nxt])
    return logp


@pytest.mark.statistical
class TestThreeWayStory:
    L = 2
    ALPHA = 0.5
    K = 4
    GRID = (16, 64, 256, 1024, 4096)

    def test_shannon_mc_is_unbiased_consistent(self, chain):
        ref, T, s = chain
        theta = continuation_entropy(T, T[s][None], self.L)[0]

        def est(n, seed):
            lp = _sample_logp(T, s, n, np.random.default_rng(seed), T)
            return float(-lp.mean()), float(lp.std(ddof=1) / np.sqrt(n))

        rep = diagnose_estimator(est, theta, n_grid=self.GRID, n_reps=500)
        assert rep.classification == "no_bias_detected", rep.to_text()
        assert -1.3 <= rep.variance_slope <= -0.7
        # CLT: the sampling distribution is approximately normal at large N
        assert rep.normality_p > 0.01 or rep.largest.n >= 4096

    def test_renyi_mc_is_biased_but_consistent(self, chain):
        ref, T, s = chain
        theta = continuation_renyi(T, T[s][None], self.ALPHA, self.L)[0]

        def est(n, seed):
            lp = _sample_logp(T, s, n, np.random.default_rng(seed), T)
            return float(
                (1 / (1 - self.ALPHA)) * np.log2(np.mean(2 ** ((self.ALPHA - 1) * lp)))
            )

        # Renyi has high variance at small N; enough reps to detect its bias.
        rep = diagnose_estimator(est, theta, n_grid=self.GRID, n_reps=2000)
        assert rep.classification == "decreasing_bias", rep.to_text()
        assert rep.levels[0].bias < 0  # downward Jensen bias
        assert rep.bias_slope < -0.5  # roughly 1/N (vs -0.5 noise floor)

    def test_truncated_sampler_is_inconsistent(self, chain):
        ref, T, s = chain
        theta = continuation_entropy(T, T[s][None], self.L)[0]  # the true entropy
        Q = topk_renorm(T, self.K)

        def est(n, seed):
            lp = _sample_logp(T, s, n, np.random.default_rng(seed), Q)
            return float(-lp.mean())

        rep = diagnose_estimator(est, theta, n_grid=self.GRID, n_reps=500)
        assert rep.classification == "persistent_bias", rep.to_text()
        # bias barely moves across two decades of N
        assert abs(rep.levels[-1].bias - rep.levels[0].bias) < 0.05
