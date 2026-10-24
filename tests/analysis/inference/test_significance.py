"""Tests for replica.analysis.inference.significance."""

import itertools
import math
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest

from replica.analysis.inference.significance import (
    EXACT_SIGN_FLIP_MAX_N,
    SupportResult,
    bh_correct,
    concat_fold_per_obs_delta_ll,
    is_supported_at_alpha,
    paired_permutation_pvalue,
    per_datapoint_permutation_pvalue,
    pvalue_to_stars,
    signflip_normal_pvalue,
    support_under_regime,
)

# ---------------------------------------------------------------------------
# paired_permutation_pvalue
# ---------------------------------------------------------------------------


class TestPairedPermutation:
    @pytest.mark.parametrize("alternative", ["two-sided", "greater", "less"])
    def test_exact_matches_integer_enumeration_across_units(self, alternative):
        for values in ([1, 1, 1], [1, 2, 4, -1]):
            observed = sum(values)
            null = [
                sum(sign * value for sign, value in zip(signs, values, strict=True))
                for signs in itertools.product((-1, 1), repeat=len(values))
            ]
            if alternative == "two-sided":
                expected = sum(abs(value) >= abs(observed) for value in null)
            elif alternative == "greater":
                expected = sum(value >= observed for value in null)
            else:
                expected = sum(value <= observed for value in null)
            expected /= len(null)
            for scale in (1e-308, 1e-200, 1e-20, 1.0, 1e200, 1e308 / max(values)):
                with np.errstate(over="raise", invalid="raise"):
                    actual = paired_permutation_pvalue(
                        np.asarray(values) * scale, alternative=alternative
                    )
                assert actual == expected

    @pytest.mark.parametrize("alternative", ["two-sided", "greater", "less"])
    @pytest.mark.parametrize(
        "values",
        [
            [1.0, *([1e-13] * 9)],
            [1.0, -1.0, 1e-300, -2e-300],
            [np.finfo(float).max, -np.finfo(float).max, np.nextafter(0.0, 1.0)],
            [0.1, 0.2, -0.3, 0.0],
        ],
    )
    def test_exact_matches_rational_oracle_with_unequal_magnitudes(
        self, values, alternative
    ):
        rational = [Fraction(float(value)) for value in values]
        observed = sum(rational)
        null = [
            sum(sign * value for sign, value in zip(signs, rational, strict=True))
            for signs in itertools.product((-1, 1), repeat=len(values))
        ]
        if alternative == "two-sided":
            expected = sum(abs(value) >= abs(observed) for value in null)
        elif alternative == "greater":
            expected = sum(value >= observed for value in null)
        else:
            expected = sum(value <= observed for value in null)
        assert paired_permutation_pvalue(values, alternative=alternative) == (
            expected / len(null)
        )

    def test_all_positive_two_sided_max_significance(self):
        # All folds positive, most extreme observation possible under
        # exact enumeration. Only one of 2^n patterns (all +1) yields
        # a mean as large as the observed, so p = 2/2^n (both extremes).
        x = [0.1] * 5  # 32 sign patterns; only ±1 all-same are as extreme
        p = paired_permutation_pvalue(x, alternative="two-sided")
        # ±1 all-same → 2 patterns out of 32 → 0.0625
        assert p == pytest.approx(2 / 32)

    def test_one_sided_halves_two_sided_for_uniform_sign(self):
        # When all per-fold values share the sign as the alternative,
        # the one-sided p should be half the two-sided p.
        x = [0.1, 0.3, 0.2, 0.15, 0.05]
        p2 = paired_permutation_pvalue(x, alternative="two-sided")
        p1 = paired_permutation_pvalue(x, alternative="greater")
        # 2 patterns ≥ observed under 2-sided (mean = 0.16, only +1+1+1+1+1
        # and -1-1-1-1-1 hit |stat| >= 0.16); under 1-sided only +1s.
        assert p1 == pytest.approx(p2 / 2)

    def test_real_valued_extremes_are_not_lost_to_roundoff(self):
        # These ten positive fold values exercise the case where ``x.mean()``
        # and the permutation dot product differ by a few ulps.
        x = [
            0.017026862119400087,
            0.026842071912215982,
            0.020980953044044577,
            0.03502336542534671,
            0.026614523762685215,
            0.041662095595055675,
            0.008039534453889526,
            0.032359763242776236,
            0.02081256589360478,
            0.03263440731664503,
        ]
        assert paired_permutation_pvalue(x, alternative="two-sided") == pytest.approx(
            2 / 1024
        )

    def test_zero_observed_two_sided_is_one(self):
        # Observed mean exactly zero → trivially the most-extreme stat
        # is zero, so every pattern's |stat| >= |obs|. p = 1.
        x = [1.0, -1.0, 0.5, -0.5]
        p = paired_permutation_pvalue(x, alternative="two-sided")
        assert p == 1.0

    def test_less_alternative_negates_greater(self):
        x = [0.1, 0.2, -0.05, 0.15, 0.3]
        pg = paired_permutation_pvalue(x, alternative="greater")
        pl = paired_permutation_pvalue(x, alternative="less")
        # Greater + less should sum to ≥ 1 (overlap at obs).
        assert pg + pl >= 1.0 - 1e-9

    def test_nan_entries_ignored(self):
        x_with_nan = [0.1, float("nan"), 0.2, float("nan"), 0.15]
        x_clean = [0.1, 0.2, 0.15]
        p_nan = paired_permutation_pvalue(x_with_nan)
        p_clean = paired_permutation_pvalue(x_clean)
        assert p_nan == p_clean

    def test_fewer_than_two_returns_nan(self):
        assert math.isnan(paired_permutation_pvalue([]))
        assert math.isnan(paired_permutation_pvalue([1.0]))

    def test_monte_carlo_converges_to_exact(self):
        x = np.array([0.1, 0.05, 0.2, -0.05, 0.15, 0.08, 0.12, 0.04, 0.18, 0.06])
        p_exact = paired_permutation_pvalue(x, alternative="two-sided", method="exact")
        p_mc = paired_permutation_pvalue(
            x,
            alternative="two-sided",
            method="monte-carlo",
            n_resamples=100_000,
            rng=np.random.default_rng(0),
        )
        assert abs(p_exact - p_mc) < 0.005

    def test_invalid_alternative_raises(self):
        with pytest.raises(ValueError, match="unknown alternative"):
            paired_permutation_pvalue([0.1, 0.2], alternative="not-a-thing")  # type: ignore

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="unknown method"):
            paired_permutation_pvalue([0.1, 0.2], method="cheat")  # type: ignore

    def test_exact_enumeration_is_refused_past_its_bound(self):
        """Exact enumeration visits 2**n sign patterns one at a time; past the
        bound only the Monte Carlo branch finishes."""
        x = np.linspace(0.1, 1.0, EXACT_SIGN_FLIP_MAX_N + 1)
        with pytest.raises(ValueError, match="monte-carlo"):
            paired_permutation_pvalue(x)
        # A zero effect needs no enumeration, whatever its length.
        assert paired_permutation_pvalue(np.zeros_like(x)) == 1.0
        p = paired_permutation_pvalue(
            x, method="monte-carlo", n_resamples=1000, rng=np.random.default_rng(0)
        )
        assert p == pytest.approx(1 / 1001)


@pytest.mark.parametrize("paired", [False, True])
def test_monte_carlo_rejects_invalid_budget_before_data_shortcuts(paired):
    for values in ([], [0.0, 0.0], [1.0, -0.5]):
        for invalid in (0, -1, True, 1.5):
            kwargs = {"method": "monte-carlo"} if paired else {}
            function = (
                paired_permutation_pvalue
                if paired
                else per_datapoint_permutation_pvalue
            )
            with pytest.raises(
                ValueError, match="n_resamples must be a positive integer"
            ):
                function(values, n_resamples=invalid, **kwargs)
            if not paired:
                with pytest.raises(
                    ValueError, match="max_elems must be a positive integer"
                ):
                    function(values, max_elems=invalid)
    if paired:
        assert paired_permutation_pvalue([1.0, 1.0], n_resamples=0) == 0.5


# ---------------------------------------------------------------------------
# per_datapoint_permutation_pvalue  (REPLICA-native test)
# ---------------------------------------------------------------------------


class TestPerDatapointPermutation:
    def test_matches_exact_on_small_n(self):
        # On a small vector the chunked MC must converge to the exact
        # per-fold helper (same underlying sign-flip null).
        rng = np.random.default_rng(0)
        x = rng.normal(0.0, 1.0, size=12)
        p_exact = paired_permutation_pvalue(x.tolist(), method="exact")
        p_mc = per_datapoint_permutation_pvalue(
            x, rng=np.random.default_rng(1), n_resamples=200_000
        )
        assert abs(p_exact - p_mc) < 0.005

    def test_clear_positive_shift_significant(self):
        rng = np.random.default_rng(2)
        x = rng.normal(0.3, 1.0, size=4000)
        p = per_datapoint_permutation_pvalue(
            x, rng=np.random.default_rng(3), n_resamples=20_000
        )
        assert p < 0.001

    def test_null_not_significant(self):
        rng = np.random.default_rng(4)
        x = rng.normal(0.0, 1.0, size=4000)
        p = per_datapoint_permutation_pvalue(
            x, rng=np.random.default_rng(5), n_resamples=20_000
        )
        assert p > 0.05

    def test_saturating_effect_never_zero(self):
        """A permutation p-value is never exactly 0: the observed pattern
        is itself a valid permutation. A huge effect saturates at the
        add-one floor 1/(n_resamples+1)."""
        x = np.full(2000, 5.0)  # mean far beyond any sign-flip can reach
        n_res = 10_000
        p = per_datapoint_permutation_pvalue(
            x, rng=np.random.default_rng(0), n_resamples=n_res
        )
        assert p == pytest.approx(1.0 / (n_res + 1))
        assert p > 0.0

    def test_one_sided_halves_two_sided_for_uniform_sign(self):
        rng = np.random.default_rng(6)
        x = np.abs(rng.normal(0.0, 1.0, size=2000)) + 0.05  # all positive
        p2 = per_datapoint_permutation_pvalue(
            x, alternative="two-sided", rng=np.random.default_rng(7)
        )
        p1 = per_datapoint_permutation_pvalue(
            x, alternative="greater", rng=np.random.default_rng(7)
        )
        # both saturate at 0 here; assert greater is never larger
        assert p1 <= p2 + 1e-12

    def test_chunking_matches_unchunked(self):
        # A tiny max_elems forces many chunks; the p-value must agree
        # (within MC noise) with the single-chunk computation.
        rng = np.random.default_rng(8)
        n_points = 2_000
        n_resamples = 2_000
        x = rng.normal(0.04, 1.0, size=n_points)
        p_one = per_datapoint_permutation_pvalue(
            x,
            rng=np.random.default_rng(9),
            n_resamples=n_resamples,
            # Exactly one batch, about 32 MiB as float64, which fits the CI
            # runners' memory.
            max_elems=n_points * n_resamples,
        )
        p_chunked = per_datapoint_permutation_pvalue(
            x,
            rng=np.random.default_rng(9),
            n_resamples=n_resamples,
            max_elems=2 * n_points,
        )
        assert abs(p_one - p_chunked) < 0.01

    def test_nan_entries_dropped(self):
        x = [0.2, float("nan"), 0.3, float("inf"), 0.1]
        p1 = per_datapoint_permutation_pvalue(
            x, rng=np.random.default_rng(0), n_resamples=5000
        )
        p2 = per_datapoint_permutation_pvalue(
            [0.2, 0.3, 0.1], rng=np.random.default_rng(0), n_resamples=5000
        )
        assert p1 == p2

    def test_fewer_than_two_returns_nan(self):
        assert math.isnan(per_datapoint_permutation_pvalue([]))
        assert math.isnan(per_datapoint_permutation_pvalue([0.5]))

    def test_zero_observed_two_sided_is_one(self):
        assert per_datapoint_permutation_pvalue([1.0, -1.0, 0.5, -0.5]) == 1.0

    def test_invalid_alternative_raises(self):
        with pytest.raises(ValueError, match="unknown alternative"):
            per_datapoint_permutation_pvalue([0.1, 0.2], alternative="nope")  # type: ignore


@pytest.mark.parametrize("kind", ["paired", "per-observation"])
@pytest.mark.parametrize("alternative", ["two-sided", "greater", "less"])
def test_monte_carlo_extreme_scales_preserve_the_seeded_sign_test(kind, alternative):
    values = np.array([1, 2, 4, -1])
    rng = np.random.default_rng(5)
    if kind == "paired":
        signs = rng.choice((-1, 1), size=(257, len(values)))
        function, options = paired_permutation_pvalue, {"method": "monte-carlo"}
        ordinary_numerators = {"two-sided": 89, "greater": 43, "less": 216}
    else:
        signs = 2 * rng.integers(0, 2, size=(257, len(values)), dtype=np.int8) - 1
        function, options = per_datapoint_permutation_pvalue, {}
        ordinary_numerators = {"two-sided": 99, "greater": 49, "less": 210}
    null = signs @ values
    if alternative == "two-sided":
        hits = np.count_nonzero(np.abs(null) >= abs(values.sum()))
    elif alternative == "greater":
        hits = np.count_nonzero(null >= values.sum())
    else:
        hits = np.count_nonzero(null <= values.sum())
    for scale in (1.0, np.nextafter(0.0, 1.0), 1e308 / 4):
        with np.errstate(over="raise", invalid="raise"):
            actual = function(
                values * scale,
                alternative=alternative,
                n_resamples=257,
                rng=np.random.default_rng(5),
                **options,
            )
        assert actual == (hits + 1) / 258
    ordinary = function(
        [0.11, -0.03, 0.07, 0.09, -0.08],
        alternative=alternative,
        n_resamples=257,
        rng=np.random.default_rng(5),
        **options,
    )
    assert ordinary == ordinary_numerators[alternative] / 258


# ---------------------------------------------------------------------------
# signflip_normal_pvalue
# ---------------------------------------------------------------------------


class TestSignflipNormal:
    @pytest.mark.parametrize("alternative", ["two-sided", "greater", "less"])
    def test_normal_score_is_independent_of_units(self, alternative):
        z = math.sqrt(3)
        expected = (
            math.erfc(z / math.sqrt(2))
            if alternative == "two-sided"
            else 0.5 * math.erfc((z if alternative == "greater" else -z) / math.sqrt(2))
        )
        for scale in (1e-308, 1e-200, 1e-20, 1.0, 1e200, 1e308):
            with np.errstate(over="raise", invalid="raise"):
                actual = signflip_normal_pvalue(np.full(3, scale), alternative)
            assert actual == pytest.approx(expected)

    def test_agrees_with_mc_on_large_n(self):
        # Same null/statistic as the MC per-datapoint test → same verdict on
        # a large vector (the regime where the pooled grid test uses it).
        rng = np.random.default_rng(0)
        x = rng.normal(0.03, 1.0, size=5_000)
        p_mc = per_datapoint_permutation_pvalue(
            x,
            alternative="greater",
            rng=np.random.default_rng(1),
            n_resamples=5_000,
        )
        p_norm = signflip_normal_pvalue(x, alternative="greater")
        assert (p_mc < 0.05) == (p_norm < 0.05)
        assert abs(p_mc - p_norm) < 0.02

    def test_clear_positive_shift_significant(self):
        rng = np.random.default_rng(2)
        x = rng.normal(0.3, 1.0, size=4000)
        assert signflip_normal_pvalue(x, alternative="greater") < 1e-6

    def test_null_not_significant(self):
        rng = np.random.default_rng(3)
        x = rng.normal(0.0, 1.0, size=4000)
        assert signflip_normal_pvalue(x, alternative="two-sided") > 0.05

    def test_z_formula_matches_hand_computation(self):
        # p_two = erfc(|Σx| / sqrt(2·Σx²))
        x = np.array([0.4, -0.1, 0.2, 0.3, -0.05])
        z = x.sum() / math.sqrt(float(np.dot(x, x)))
        expected = math.erfc(abs(z) / math.sqrt(2.0))
        assert signflip_normal_pvalue(x, "two-sided") == pytest.approx(expected)

    def test_one_sided_is_half_two_sided_when_positive(self):
        rng = np.random.default_rng(4)
        x = rng.normal(0.1, 1.0, size=5000)  # positive mean
        p2 = signflip_normal_pvalue(x, "two-sided")
        pg = signflip_normal_pvalue(x, "greater")
        assert pg == pytest.approx(p2 / 2.0, rel=1e-9)

    def test_nan_entries_dropped(self):
        a = signflip_normal_pvalue([0.2, float("nan"), 0.3, float("inf"), 0.1])
        b = signflip_normal_pvalue([0.2, 0.3, 0.1])
        assert a == pytest.approx(b)

    def test_fewer_than_two_returns_nan(self):
        assert math.isnan(signflip_normal_pvalue([]))
        assert math.isnan(signflip_normal_pvalue([0.5]))

    def test_all_zero_returns_nan(self):
        assert math.isnan(signflip_normal_pvalue([0.0, 0.0, 0.0]))

    def test_invalid_alternative_raises(self):
        with pytest.raises(ValueError, match="unknown alternative"):
            signflip_normal_pvalue([0.1, 0.2], alternative="nope")  # type: ignore


# ---------------------------------------------------------------------------
# concat_fold_per_obs_delta_ll
# ---------------------------------------------------------------------------


class TestConcatFoldPerObsDeltaLL:
    def test_concatenates_in_order(self):
        folds = [
            SimpleNamespace(per_obs_delta_ll=np.array([1.0, 2.0])),
            SimpleNamespace(per_obs_delta_ll=np.array([3.0])),
        ]
        np.testing.assert_array_equal(
            concat_fold_per_obs_delta_ll(folds), [1.0, 2.0, 3.0]
        )

    def test_skips_none_folds(self):
        folds = [
            SimpleNamespace(per_obs_delta_ll=np.array([1.0])),
            SimpleNamespace(per_obs_delta_ll=None),
            SimpleNamespace(per_obs_delta_ll=np.array([2.0])),
        ]
        np.testing.assert_array_equal(concat_fold_per_obs_delta_ll(folds), [1.0, 2.0])

    def test_empty_inputs_return_empty(self):
        assert concat_fold_per_obs_delta_ll([]).size == 0
        assert concat_fold_per_obs_delta_ll(None).size == 0
        assert (
            concat_fold_per_obs_delta_ll([SimpleNamespace(per_obs_delta_ll=None)]).size
            == 0
        )


# ---------------------------------------------------------------------------
# bh_correct
# ---------------------------------------------------------------------------


class TestBHCorrect:
    def test_empty_input(self):
        c, r = bh_correct([])
        assert c.size == 0 and r.size == 0

    def test_single_pvalue_unchanged(self):
        c, r = bh_correct([0.04], alpha=0.05)
        assert c[0] == pytest.approx(0.04)
        assert r[0]

    def test_uniform_pvalues_step_up(self):
        # 5 equal p=0.01 → BH-adjusted = 0.01 * 5/rank, monotone-capped
        # to the same min: q = (0.05, 0.025, 0.0167, 0.0125, 0.01) →
        # after monotone smoothing all become 0.01.
        c, r = bh_correct([0.01] * 5, alpha=0.05)
        assert np.allclose(c, 0.01)
        assert r.all()

    def test_canonical_bh_example(self):
        # Benjamini-Hochberg (1995), p=[0.01..0.05, 0.5], m=6, alpha=0.05.
        # Thresholds alpha·i/m = [0.0083, 0.0167, 0.025, 0.0333, 0.0417, 0.05];
        # every p_(i) exceeds its threshold → no rejections. Corrected
        # p_i = min over the tail of p_i·m/rank → 0.06 for the first five.
        c, r = bh_correct([0.01, 0.02, 0.03, 0.04, 0.05, 0.5], alpha=0.05)
        assert c[0] == pytest.approx(0.06)
        assert not r.any()

    def test_nan_passes_through(self):
        c, r = bh_correct([0.01, float("nan"), 0.02], alpha=0.05)
        assert math.isnan(c[1])
        assert not r[1]
        # The 2 finite values still get BH-corrected as a batch of m=2.
        # corrected: [0.02, _, 0.02] → both reject at alpha=0.05.
        assert c[0] == pytest.approx(0.02)
        assert c[2] == pytest.approx(0.02)

    def test_monotone_correction(self):
        # Adjusted p-values must be monotone non-decreasing in the
        # sorted-input order.
        rng = np.random.default_rng(0)
        raw = sorted(rng.uniform(0, 1, size=20))
        c, _ = bh_correct(raw, alpha=0.05)
        assert all(c[i] <= c[i + 1] + 1e-12 for i in range(len(c) - 1))


# ---------------------------------------------------------------------------
# SupportResult + support_under_regime
# ---------------------------------------------------------------------------


class TestSupport:
    def test_supportresult_sign_check(self):
        r = SupportResult(effect=+0.05, p_value=0.01, claim_direction=+1, alpha=0.05)
        assert r.is_supported
        # wrong-sign cell at same p → not supported
        r2 = SupportResult(effect=-0.05, p_value=0.01, claim_direction=+1, alpha=0.05)
        assert not r2.is_supported
        # right sign, p above threshold → not supported
        r3 = SupportResult(effect=+0.05, p_value=0.06, claim_direction=+1, alpha=0.05)
        assert not r3.is_supported

    def test_supportresult_zero_direction_skips_sign(self):
        r_pos = SupportResult(effect=+0.05, p_value=0.01, claim_direction=0)
        r_neg = SupportResult(effect=-0.05, p_value=0.01, claim_direction=0)
        assert r_pos.is_supported and r_neg.is_supported

    def test_supportresult_nan_unsupported(self):
        r = SupportResult(effect=float("nan"), p_value=0.01, claim_direction=+1)
        assert not r.is_supported

    def test_regime_two_sided_uncorrected(self):
        results = support_under_regime(
            effects=[+0.1, +0.05, -0.02],
            raw_pvalues=[0.01, 0.06, 0.03],
            claim_directions=+1,
            sided="two-sided",
            correction="none",
        )
        # cell 0: positive sign + p<0.05 → supported
        assert results[0].is_supported
        # cell 1: positive sign + p=0.06 → not supported
        assert not results[1].is_supported
        # cell 2: negative sign → not supported regardless of p
        assert not results[2].is_supported

    def test_regime_one_sided_halves_p_when_sign_matches(self):
        # raw two-sided p = 0.06 → one-sided p = 0.03 if sign matches
        results = support_under_regime(
            effects=[+0.1],
            raw_pvalues=[0.06],
            claim_directions=+1,
            sided="greater",
            correction="none",
        )
        assert results[0].p_value == pytest.approx(0.03)
        assert results[0].is_supported

    def test_regime_one_sided_penalises_wrong_sign(self):
        # Effect has wrong sign → one-sided "greater" p ≈ 1 - raw/2
        results = support_under_regime(
            effects=[-0.1],
            raw_pvalues=[0.04],
            claim_directions=+1,
            sided="greater",
            correction="none",
        )
        assert results[0].p_value == pytest.approx(0.98)
        assert not results[0].is_supported

    @pytest.mark.parametrize(
        "sided, effect, expected",
        [
            ("greater", +0.5, 0.0005),
            ("greater", -0.5, 0.9995),
            ("less", -0.5, 0.0005),
            ("less", +0.5, 0.9995),
        ],
    )
    def test_regime_without_a_claimed_sign_tests_its_own_tail(
        self, sided, effect, expected
    ):
        """With no claimed sign, a one-sided regime is oriented by its own
        direction, so an effect on the other side keeps a large p-value."""
        result = support_under_regime(
            [effect], [0.001], claim_directions=0, sided=sided
        )[0]
        assert result.p_value == pytest.approx(expected)
        assert result.is_supported is (expected < 0.05)

    def test_regime_bh_correction_applied(self):
        # 5 raw p's = 0.01 → BH-adjusted = 0.01 (monotone min). All supported.
        results = support_under_regime(
            effects=[+0.1] * 5,
            raw_pvalues=[0.01] * 5,
            claim_directions=+1,
            sided="two-sided",
            correction="bh",
        )
        assert all(r.is_supported for r in results)
        for r in results:
            assert r.p_value == pytest.approx(0.01)
            assert r.is_corrected
            assert r.regime_label == "two-sided BH"

    def test_regime_label_combinations(self):
        labels = {
            (s, c) for s in ("two-sided", "greater", "less") for c in ("none", "bh")
        }
        for sided, corr in labels:
            r = support_under_regime(
                [0.1],
                [0.01],
                +1,
                sided=sided,
                correction=corr,
            )[0]
            expected = sided + (" BH" if corr == "bh" else "")
            assert r.regime_label == expected


# ---------------------------------------------------------------------------
# pvalue_to_stars
# ---------------------------------------------------------------------------


class TestPvalueToStars:
    def test_thresholds(self):
        assert pvalue_to_stars(0.0005) == "***"
        assert pvalue_to_stars(0.005) == "**"
        assert pvalue_to_stars(0.03) == "*"
        assert pvalue_to_stars(0.2) == ""

    def test_boundaries_are_strict(self):
        # Exactly at a threshold falls to the weaker bucket (< is strict).
        assert pvalue_to_stars(0.001) == "**"
        assert pvalue_to_stars(0.01) == "*"
        assert pvalue_to_stars(0.05) == ""

    def test_custom_ns_label(self):
        assert pvalue_to_stars(0.2, ns="n.s.") == "n.s."
        assert pvalue_to_stars(0.0005, ns="n.s.") == "***"

    def test_nonfinite_is_empty(self):
        assert pvalue_to_stars(float("nan")) == ""
        assert pvalue_to_stars(float("inf"), ns="n.s.") == ""


# ---------------------------------------------------------------------------
# is_supported_at_alpha
# ---------------------------------------------------------------------------


class TestIsSupportedAtAlpha:
    @pytest.mark.parametrize(
        "effect, p, alpha, expected",
        [
            (0.05, 0.01, 0.05, True),  # positive and significant
            (-0.05, 0.01, 0.05, False),  # wrong sign
            (0.05, 0.06, 0.05, False),  # not significant
            (0.05, 0.05, 0.05, False),  # alpha is strict (p == alpha)
            (0.05, 0.0499, 0.05, True),  # just under alpha
            (0.0, 0.001, 0.05, False),  # zero effect
            (0.05, 0.005, 0.01, True),  # custom alpha passes
            (0.05, 0.02, 0.01, False),  # custom alpha fails
            (float("nan"), 0.01, 0.05, False),  # non-finite effect
            (0.05, float("nan"), 0.05, False),  # non-finite p
        ],
        ids=[
            "pos_sig",
            "wrong_sign",
            "not_sig",
            "alpha_strict_eq",
            "alpha_strict_under",
            "zero_effect",
            "custom_alpha_pass",
            "custom_alpha_fail",
            "nan_effect",
            "nan_p",
        ],
    )
    def test_decision(self, effect, p, alpha, expected):
        assert bool(is_supported_at_alpha(effect, p, alpha=alpha)) is expected

    def test_agrees_with_supportresult(self):
        for eff, p in [(0.05, 0.01), (-0.05, 0.01), (0.05, 0.2), (0.0, 0.001)]:
            sr = SupportResult(effect=eff, p_value=p, claim_direction=+1, alpha=0.05)
            assert is_supported_at_alpha(eff, p) == sr.is_supported


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
