"""MC convergence diagnostics: variance scaling and bias detection."""

import numpy as np
import pytest

from replica.feature.diagnostics import (
    MarkovReferenceLM,
    default_specs,
    mc_convergence,
    simulate_mc_entropy,
)

EXACT = np.linspace(1.0, 5.0, 8)


def _unbiased(n, seed):
    rng = np.random.default_rng(seed)
    return EXACT + rng.normal(0.0, 2.0 / np.sqrt(n), size=EXACT.shape)


class TestSynthetic:
    @pytest.mark.statistical
    def test_unbiased_mc_passes_both_checks(self):
        rep = mc_convergence(_unbiased, EXACT, n_grid=(64, 256, 1024, 4096), n_reps=16)
        assert rep.variance_ok
        assert not rep.biased

    @pytest.mark.statistical
    def test_offset_estimator_is_flagged_biased(self):
        rep = mc_convergence(
            lambda n, seed: _unbiased(n, seed) + 0.5,
            EXACT,
            n_grid=(64, 256, 1024),
            n_reps=8,
        )
        assert rep.biased

    @pytest.mark.statistical
    def test_non_scaling_variance_is_flagged(self):
        # Variance independent of N, the signature of correlated samples
        # (e.g. a buffer reshape that reuses draws across "samples").
        def est(n, seed):
            rng = np.random.default_rng(seed)
            return EXACT + rng.normal(0.0, 0.1, size=EXACT.shape)

        rep = mc_convergence(est, EXACT, n_grid=(64, 256, 1024), n_reps=16)
        assert not rep.variance_ok

    def test_n_grid_must_strictly_increase(self):
        with pytest.raises(ValueError, match="increasing"):
            mc_convergence(_unbiased, EXACT, n_grid=(256, 64))
        with pytest.raises(ValueError, match="increasing"):
            mc_convergence(_unbiased, EXACT, n_grid=(64, 64))

    def test_nan_estimate_raises_instead_of_passing(self):
        # A NaN must raise before it reaches the t-stat: a NaN t-stat would
        # silently pass the bias gate.
        def est(n, seed):
            vals = _unbiased(n, seed) + 10.0
            vals[0] = np.nan
            return vals

        with pytest.raises(ValueError, match="non-finite"):
            mc_convergence(est, EXACT, n_grid=(64, 256), n_reps=8)

    @pytest.mark.statistical
    def test_nan_in_exact_is_masked_out(self):
        exact = EXACT.copy()
        exact[0] = np.nan

        def est(n, seed):
            vals = _unbiased(n, seed)
            vals[0] = np.nan  # estimator may be undefined there too
            return vals

        rep = mc_convergence(est, exact, n_grid=(64, 256, 1024), n_reps=8)
        assert not rep.biased

    def test_deterministic_estimator_is_reported_not_crashed(self):
        rep = mc_convergence(
            lambda n, seed: EXACT.copy(), EXACT, n_grid=(64, 256), n_reps=8
        )
        assert rep.deterministic
        assert not rep.variance_ok
        assert not rep.biased
        assert "deterministic" in rep.to_text()

    def test_deterministic_offset_is_still_flagged_biased(self):
        # Zero across-seed spread with nonzero mean error is a definite
        # bias, even though its t-stat is undefined.
        rep = mc_convergence(
            lambda n, seed: EXACT + 0.5, EXACT, n_grid=(64, 256), n_reps=8
        )
        assert rep.biased

    def test_too_few_unit_reps_raises(self):
        # One unit x 8 reps is below the 48 unit-reps the slope and bias
        # checks need.
        with pytest.raises(ValueError, match="unit-reps"):
            mc_convergence(
                lambda n, seed: np.array([1.0]),
                np.array([1.0]),
                n_grid=(64, 256),
                n_reps=8,
            )


@pytest.mark.statistical
class TestOnReferenceChain:
    """The numpy MC simulator against the closed-form continuation specs."""

    @pytest.fixture(scope="class")
    @classmethod
    def setup(cls):
        ref = MarkovReferenceLM()
        stim = ref.probe_stimulus()
        exact = {s.name: s.compute(ref, stim) for s in default_specs()}
        return ref, stim, exact

    def test_correct_sampler_is_unbiased(self, setup):
        ref, stim, exact = setup
        rep = mc_convergence(
            lambda n, seed: simulate_mc_entropy(
                ref, stim, L=2, n_samples=n, rng=np.random.default_rng(seed)
            ),
            exact["continuation_entropy_L2"],
            n_grid=(64, 256),
            n_reps=8,
        )
        assert not rep.biased
        assert rep.variance_ok

    def test_topk_sampler_is_biased_against_true_entropy(self, setup):
        ref, stim, exact = setup
        rep = mc_convergence(
            lambda n, seed: simulate_mc_entropy(
                ref,
                stim,
                L=2,
                n_samples=n,
                rng=np.random.default_rng(seed),
                sample_top_k=4,
            ),
            exact["continuation_entropy_L2"],
            n_grid=(64, 256),
            n_reps=8,
        )
        assert rep.biased
