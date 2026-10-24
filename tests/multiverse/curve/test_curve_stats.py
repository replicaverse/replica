"""Tests for replica.multiverse.curve.stats."""

import math
from fractions import Fraction
from itertools import product

import numpy as np
import pandas as pd
import pytest

from replica.multiverse.curve.stats import (
    family_wise_signflip,
    joint_signflip_inference,
    spec_curve_descriptives,
    swap_attribution,
)

# ---------------------------------------------------------------------------
# spec_curve_descriptives
# ---------------------------------------------------------------------------


class TestSpecCurveDescriptives:
    def test_median_mean_and_range(self):
        s = spec_curve_descriptives([0.1, 0.2, 0.3, 0.4])
        assert s.n_specs == 4
        assert s.median_effect == pytest.approx(0.25)
        assert s.mean_effect == pytest.approx(0.25)
        assert s.effect_min == pytest.approx(0.1)
        assert s.effect_max == pytest.approx(0.4)

    def test_frac_in_direction_positive_default(self):
        # 3 of 4 positive → predicted direction +1 satisfied by 3/4.
        s = spec_curve_descriptives([0.1, 0.2, -0.3, 0.4])
        assert s.frac_in_direction == pytest.approx(0.75)

    def test_frac_in_direction_negative_claim(self):
        s = spec_curve_descriptives([-0.1, -0.2, 0.3, -0.4], directions=-1)
        assert s.frac_in_direction == pytest.approx(0.75)

    def test_frac_significant_requires_direction_and_alpha(self):
        # Effects all positive; only two have p < .05.
        s = spec_curve_descriptives(
            [0.1, 0.2, 0.3, -0.4],
            pvalues=[0.01, 0.20, 0.04, 0.001],
            alpha=0.05,
        )
        # Significant-in-direction: (0.1,p=.01) and (0.3,p=.04) → 2/4.
        # The negative effect with p=.001 is significant but wrong direction.
        assert s.frac_significant == pytest.approx(0.5)

    def test_no_pvalues_gives_nan_frac_significant(self):
        s = spec_curve_descriptives([0.1, 0.2])
        assert math.isnan(s.frac_significant)

    def test_drops_nonfinite_effects(self):
        s = spec_curve_descriptives([0.1, float("nan"), 0.3])
        assert s.n_specs == 2
        assert s.median_effect == pytest.approx(0.2)

    def test_empty_all_nan(self):
        s = spec_curve_descriptives([])
        assert s.n_specs == 0
        assert math.isnan(s.median_effect)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            spec_curve_descriptives([0.1, 0.2], pvalues=[0.01])


# ---------------------------------------------------------------------------
# joint_signflip_inference
# ---------------------------------------------------------------------------


class TestJointSignflipInference:
    def test_strong_positive_effect_is_significant(self):
        # Five cells each a constant positive ΔLL: per-cell mean ≈ 0.1, far
        # outside the sign-flip null (centred at 0). Median across cells is
        # extreme → tiny p.
        cells = [np.full(50, 0.1) for _ in range(5)]
        res = joint_signflip_inference(
            cells, statistic="median", n_resamples=2000, rng=np.random.default_rng(0)
        )
        assert res.n_specs == 5
        assert res.observed == pytest.approx(0.1)
        assert res.p_value < 0.01

    def test_symmetric_null_is_not_significant(self):
        # Exactly symmetric cells (mean 0) → observed median 0; the null
        # reaches |0| on every draw.
        base = np.array([1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 1.5, -1.5])
        cells = [base.copy() for _ in range(4)]
        res = joint_signflip_inference(
            cells, statistic="median", n_resamples=2000, rng=np.random.default_rng(1)
        )
        assert res.observed == pytest.approx(0.0)
        assert res.p_value > 0.05

    def test_share_positive_statistic(self):
        # All 8 cells positive → observed share 1.0. Under the null each cell's
        # mean is positive with prob ~0.5, so P(all 8) ≈ 0.5**8 ≈ 0.004.
        cells = [np.full(40, 0.2) for _ in range(8)]
        res = joint_signflip_inference(
            cells,
            statistic="share_positive",
            alternative="greater",
            n_resamples=4000,
            rng=np.random.default_rng(2),
        )
        assert res.observed == pytest.approx(1.0)
        assert res.p_value < 0.02

    def test_negative_effect_two_sided_detects_greater_does_not(self):
        cells = [np.full(50, -0.1) for _ in range(5)]
        two = joint_signflip_inference(
            cells,
            statistic="median",
            alternative="two-sided",
            n_resamples=2000,
            rng=np.random.default_rng(3),
        )
        greater = joint_signflip_inference(
            cells,
            statistic="median",
            alternative="greater",
            n_resamples=2000,
            rng=np.random.default_rng(3),
        )
        assert two.p_value < 0.01
        assert greater.p_value > 0.5

    def test_deterministic_with_seed(self):
        cells = [np.array([0.1, -0.05, 0.2, 0.0, 0.15]) for _ in range(3)]
        a = joint_signflip_inference(
            cells, n_resamples=1000, rng=np.random.default_rng(9)
        )
        b = joint_signflip_inference(
            cells, n_resamples=1000, rng=np.random.default_rng(9)
        )
        assert a.observed == pytest.approx(b.observed)
        assert a.p_value == pytest.approx(b.p_value)

    def test_short_cells_dropped(self):
        cells = [np.full(30, 0.1), np.array([0.5])]  # second has < 2 entries
        res = joint_signflip_inference(
            cells, n_resamples=500, rng=np.random.default_rng(0)
        )
        assert res.n_specs == 1

    def test_empty_input_nan(self):
        res = joint_signflip_inference([], n_resamples=100)
        assert res.n_specs == 0
        assert math.isnan(res.observed)
        assert math.isnan(res.p_value)

    def test_unknown_statistic_raises(self):
        with pytest.raises(ValueError):
            joint_signflip_inference(
                [np.full(10, 0.1)], statistic="bogus", n_resamples=10
            )

    def test_independent_mode_recorded(self):
        res = joint_signflip_inference(
            [np.full(20, 0.1)], n_resamples=200, rng=np.random.default_rng(0)
        )
        assert res.mode == "independent"

    def test_share_positive_two_sided_rejected(self):
        # share_positive's sign-flip null is centred near 0.5, so a |stat|
        # >= |obs| two-sided tail (centred at 0) is malformed, must raise.
        with pytest.raises(ValueError, match="share_positive"):
            joint_signflip_inference(
                [np.full(20, 0.1)],
                statistic="share_positive",
                alternative="two-sided",
                n_resamples=10,
            )
        with pytest.raises(ValueError, match="share_positive"):
            joint_signflip_inference(
                [np.full(20, 0.1)],
                statistic="share_positive",
                alternative="two-sided",
                shared_observations=True,
                n_resamples=10,
            )


class TestSynchronizedJointSignflip:
    def test_shared_positive_cells_significant(self):
        cells = [np.full(50, 0.1) for _ in range(4)]
        res = joint_signflip_inference(
            cells,
            shared_observations=True,
            n_resamples=2000,
            rng=np.random.default_rng(0),
        )
        assert res.mode == "synchronized"
        assert res.observed == pytest.approx(0.1)
        assert res.p_value < 0.01

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            joint_signflip_inference(
                [np.full(30, 0.1), np.full(20, 0.1)],
                shared_observations=True,
                n_resamples=100,
            )

    def test_independent_is_anticonservative_vs_synchronized(self):
        # Five identical cells (perfectly correlated). The independent null
        # treats them as five independent draws and tightens the median's
        # null, giving a smaller, anti-conservative p-value. The synchronized
        # null preserves the correlation and gives a larger, valid p-value, so
        # independent p < synchronized p here.
        cell = np.array([1.0] * 35 + [-1.0] * 25)  # mean = 10/60 ≈ 0.167
        cells = [cell.copy() for _ in range(5)]
        indep = joint_signflip_inference(
            cells, statistic="median", n_resamples=4000, rng=np.random.default_rng(1)
        )
        sync = joint_signflip_inference(
            cells,
            statistic="median",
            shared_observations=True,
            n_resamples=4000,
            rng=np.random.default_rng(1),
        )
        assert indep.p_value < sync.p_value

    def test_synchronized_one_cell_matches_observed(self):
        cells = [np.full(40, 0.2)]
        res = joint_signflip_inference(
            cells,
            shared_observations=True,
            n_resamples=1000,
            rng=np.random.default_rng(0),
        )
        assert res.n_specs == 1
        assert res.observed == pytest.approx(0.2)

    def test_synchronized_drops_rows_nonfinite_in_any_cell(self):
        # Row 0 is NaN in the second cell → dropped from both to keep alignment.
        a = np.array([5.0, 0.1, 0.1, 0.1])
        b = np.array([float("nan"), 0.1, 0.1, 0.1])
        res = joint_signflip_inference(
            [a, b],
            shared_observations=True,
            n_resamples=500,
            rng=np.random.default_rng(0),
        )
        # Both cells now mean over rows 1-3 = 0.1 (the 5.0 outlier row dropped).
        assert res.observed == pytest.approx(0.1)

    def test_synchronized_empty_nan(self):
        res = joint_signflip_inference([], shared_observations=True, n_resamples=100)
        assert res.n_specs == 0
        assert math.isnan(res.observed)


# ---------------------------------------------------------------------------
# family_wise_signflip (maxT + share-significant)
# ---------------------------------------------------------------------------


def _enumerated_family_wise(cells, signs, alternative, alpha):
    """Rank all sign patterns with exact rational, scale-free statistics."""
    from scipy.stats import norm

    def statistics(values):
        values = [Fraction(float(value)) for value in values]
        n = len(values)
        total = sum(values)
        squares = sum(value * value for value in values)
        # Signed squared sum / sum of squares is monotone in mean / SE.
        rank = total * abs(total) / squares if squares else Fraction(0)
        variance = (n * squares - total * total) / (n * n * (n - 1))
        z = (
            float(total / n) / math.sqrt(float(variance))
            if variance
            else math.copysign(math.inf, total)
            if total
            else 0.0
        )
        return rank, z

    def direction(value):
        return (
            abs(value)
            if alternative == "two-sided"
            else (value if alternative == "greater" else -value)
        )

    observed = [statistics(cell) for cell in cells]
    null = [[statistics(cell * sign) for cell in cells] for sign in signs]
    maximum = [max(direction(rank) for rank, _ in draw) for draw in null]
    adjusted = [
        (1 + sum(value >= direction(rank) for value in maximum)) / (len(signs) + 1)
        for rank, _ in observed
    ]
    critical = norm.ppf(1 - alpha / (2 if alternative == "two-sided" else 1))
    share = sum(direction(z) > critical for _, z in observed) / len(cells)
    share_hits = sum(
        sum(direction(z) > critical for _, z in draw) / len(cells) >= share
        for draw in null
    )
    return (
        [z for _, z in observed],
        adjusted,
        share,
        (share_hits + 1) / (len(signs) + 1),
    )


class TestFamilyWiseSignflip:
    @pytest.mark.parametrize("alternative", ["greater", "less", "two-sided"])
    @pytest.mark.parametrize(
        "cells",
        [
            [[0.1] * 6],
            [[-0.1] * 6],
            [[0.0] * 6],
            [[1.0, -1.0, 1.0, -1.0]],
            [np.random.default_rng(1).normal(size=8)],
            [[3, -1, 2, -4], [0.3] * 4, [0] * 4, [-2, 4, -1, 3]],
            [[1, 1 + delta, 1 - delta, 1] for delta in (1e-3, 1e-6, 1e-9)],
            [[1, -(1 + 1e-9), 1 - 1e-9, -1]],
        ],
        ids=[
            "positive",
            "negative",
            "zero",
            "signed",
            "roundoff-tie",
            "multiple",
            "near-constant",
            "near-constant-null",
        ],
    )
    def test_exact_enumeration_oracle(self, monkeypatch, cells, alternative):
        from replica.multiverse.curve import stats

        cells = np.asarray(cells, dtype=float)
        signs = np.asarray(list(product((-1.0, 1.0), repeat=cells.shape[1])))
        monkeypatch.setattr(
            stats, "sign_chunks", lambda *args: iter(np.array_split(signs, 3))
        )
        z, adjusted, share, share_p = _enumerated_family_wise(
            cells, signs, alternative, alpha=0.05
        )
        for groups in (None, np.arange(cells.shape[1])):
            actual = family_wise_signflip(
                cells, alternative=alternative, n_resamples=len(signs), groups=groups
            )
            assert actual.cell_z == pytest.approx(z)
            assert actual.adj_pvalues == pytest.approx(adjusted)
            assert actual.share_significant == share
            assert actual.share_significant_p == share_p

    @pytest.mark.parametrize("scale", [1e-200, 1e200])
    def test_standardization_is_independent_of_units(self, scale):
        cells = np.array([[3, -1, 2, 4], [-2, 4, 1, 3]], dtype=float)
        expected = family_wise_signflip(cells, n_resamples=100)
        actual = family_wise_signflip(cells * scale, n_resamples=100)
        assert actual.cell_z == pytest.approx(expected.cell_z)
        assert actual.adj_pvalues == expected.adj_pvalues
        assert actual.share_significant_p == expected.share_significant_p

    def test_strong_effect_all_specs_fwer_significant(self):
        # Four row-aligned cells with an unambiguous positive effect (per-cell
        # z ≈ 7–8, comfortably past the maxT threshold even after correcting
        # for four competing nulls): every spec should survive the family-wise
        # control and the share-significant joint test should be tiny. The
        # effect is deliberately large so sampling noise can't drop a cell
        # below threshold (a borderline z≈3 effect does so ~1 seed in 4).
        rng = np.random.default_rng(0)
        cells = [rng.normal(0.12, 1.0, 4000) for _ in range(4)]
        r = family_wise_signflip(cells, n_resamples=2000, rng=np.random.default_rng(1))
        assert r.n_specs == 4
        assert r.n_sig_fwer == 4
        assert r.min_adj_p < 0.01
        assert r.share_significant == pytest.approx(1.0)
        assert r.share_significant_p < 0.01

    def test_pure_noise_not_family_wise_significant(self):
        # A representative null draw of six true-zero-effect cells: no spec
        # should clear the family-wise control. (maxT controls FWER at alpha,
        # so by design ~5% of null draws still yield one rejection; this seed
        # gives a typical non-tail draw.)
        rng = np.random.default_rng(0)
        cells = [rng.normal(0.0, 1.0, 3000) for _ in range(6)]
        r = family_wise_signflip(cells, n_resamples=2000, rng=np.random.default_rng(3))
        assert r.n_sig_fwer == 0
        assert r.min_adj_p > 0.05

    def test_maxt_isolates_the_one_real_spec(self):
        # One strong cell hidden among five noise cells: the maxT control must
        # flag exactly the strong one (its adjusted p stays small even after
        # correcting for the five nulls competing for the maximum).
        rng = np.random.default_rng(4)
        cells = [rng.normal(0.09, 1.0, 3000)] + [
            rng.normal(0.0, 1.0, 3000) for _ in range(5)
        ]
        r = family_wise_signflip(cells, n_resamples=3000, rng=np.random.default_rng(5))
        assert r.n_sig_fwer == 1
        assert r.adj_pvalues[0] < 0.05
        assert min(r.adj_pvalues[1:]) > 0.05

    def test_fwer_min_p_is_minimum_adjusted_p(self):
        rng = np.random.default_rng(6)
        cells = [rng.normal(0.04, 1.0, 2000) for _ in range(3)]
        r = family_wise_signflip(cells, n_resamples=1500, rng=np.random.default_rng(7))
        assert r.min_adj_p == pytest.approx(min(r.adj_pvalues))

    def test_wrong_direction_not_significant_under_greater(self):
        # A real negative effect must not count as family-wise significant for
        # the one-sided "greater" claim.
        rng = np.random.default_rng(8)
        cells = [rng.normal(-0.06, 1.0, 3000) for _ in range(4)]
        r = family_wise_signflip(
            cells, alternative="greater", n_resamples=2000, rng=np.random.default_rng(9)
        )
        assert r.n_sig_fwer == 0
        two = family_wise_signflip(
            cells,
            alternative="two-sided",
            n_resamples=2000,
            rng=np.random.default_rng(9),
        )
        assert two.n_sig_fwer == 4  # two-sided detects the negative effect

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            family_wise_signflip([np.zeros(10), np.zeros(11)], n_resamples=100)

    def test_unknown_alternative_raises(self):
        with pytest.raises(ValueError):
            family_wise_signflip([np.zeros(10)], alternative="sideways")

    @pytest.mark.parametrize(
        "option,value",
        [
            ("n_resamples", -2),
            ("n_resamples", -1),
            ("n_resamples", 0),
            ("max_elems", 0),
            ("alpha", 0),
            ("alpha", 1),
            ("alpha", np.nan),
            ("alpha", np.inf),
        ],
    )
    @pytest.mark.parametrize(
        "cells,groups",
        [
            ([[1.0, 2.0]], None),
            ([], None),
            ([[1.0]], None),
            ([[1.0, 2.0]], ["a", "b"]),
            ([[]], []),
        ],
    )
    def test_invalid_options_rejected_before_data_shortcuts(
        self, option, value, cells, groups
    ):
        with pytest.raises(ValueError, match=option):
            family_wise_signflip(cells, groups=groups, **{option: value})

    def test_empty_is_nan(self):
        r = family_wise_signflip([], n_resamples=100)
        assert r.n_specs == 0
        assert math.isnan(r.min_adj_p)

    def test_deterministic_with_seed(self):
        rng = np.random.default_rng(10)
        cells = [rng.normal(0.03, 1.0, 1500) for _ in range(3)]
        a = family_wise_signflip(cells, n_resamples=1200, rng=np.random.default_rng(11))
        b = family_wise_signflip(cells, n_resamples=1200, rng=np.random.default_rng(11))
        assert a.adj_pvalues == b.adj_pvalues
        assert a.min_adj_p == pytest.approx(b.min_adj_p)
        assert a.share_significant_p == pytest.approx(b.share_significant_p)


# ---------------------------------------------------------------------------
# swap_attribution
# ---------------------------------------------------------------------------


def _grid_2x2(effect_fn, reps=4):
    rows = []
    for _ in range(reps):
        for a in ("paper", "replica"):
            for b in ("paper", "replica"):
                rows.append({"dv_swap": a, "freq_swap": b, "effect": effect_fn(a, b)})
    return pd.DataFrame(rows)


class TestSwapAttribution:
    def test_single_driver_gets_all_variance(self):
        # Effect depends only on dv_swap → it explains everything, no residual.
        df = _grid_2x2(lambda a, b: 0.0 if a == "paper" else 1.0)
        attr = swap_attribution(df, "effect", ["dv_swap", "freq_swap"])
        assert attr.eta_squared["dv_swap"] == pytest.approx(1.0)
        assert attr.eta_squared["freq_swap"] == pytest.approx(0.0, abs=1e-9)
        assert attr.residual_eta_squared == pytest.approx(0.0, abs=1e-9)
        assert attr.r_squared == pytest.approx(1.0)
        assert attr.n_obs == 16

    def test_dominant_driver_with_noise(self):
        rng = np.random.default_rng(0)
        df = _grid_2x2(
            lambda a, b: (
                (1.0 if a == "replica" else 0.0) + 0.05 * rng.standard_normal()
            ),
            reps=20,
        )
        attr = swap_attribution(df, "effect", ["dv_swap", "freq_swap"])
        assert attr.eta_squared["dv_swap"] > attr.eta_squared["freq_swap"]
        assert attr.eta_squared["dv_swap"] > 0.8

    def test_inactive_dimension_is_zero_not_error(self):
        df = pd.DataFrame(
            {
                "dv_swap": ["paper", "replica", "paper", "replica"],
                "freq_swap": ["paper", "paper", "paper", "paper"],  # constant
                "effect": [0.0, 1.0, 0.0, 1.0],
            }
        )
        attr = swap_attribution(df, "effect", ["dv_swap", "freq_swap"])
        assert attr.eta_squared["freq_swap"] == 0.0
        assert math.isnan(attr.pvalues["freq_swap"])
        assert attr.eta_squared["dv_swap"] == pytest.approx(1.0)

    def test_nonfinite_effects_dropped(self):
        df = pd.DataFrame(
            {
                "dv_swap": ["paper", "replica", "paper", "replica"],
                "freq_swap": ["paper", "replica", "replica", "paper"],
                "effect": [0.0, 1.0, float("nan"), 1.0],
            }
        )
        attr = swap_attribution(df, "effect", ["dv_swap", "freq_swap"])
        assert attr.n_obs == 3

    def test_constant_effect_attributes_nothing(self):
        df = _grid_2x2(lambda a, b: 0.5)
        attr = swap_attribution(df, "effect", ["dv_swap", "freq_swap"])
        assert attr.eta_squared["dv_swap"] == pytest.approx(0.0)
        assert attr.residual_eta_squared == pytest.approx(1.0)

    def test_column_order_preserved(self):
        df = _grid_2x2(lambda a, b: 0.0 if a == "paper" else 1.0)
        attr = swap_attribution(df, "effect", ["freq_swap", "dv_swap"])
        assert list(attr.eta_squared.keys()) == ["freq_swap", "dv_swap"]

    def test_missing_swap_value_does_not_corrupt_comparison(self):
        # A NaN swap value must not silently drop a row from only some fits;
        # it becomes its own "nan" level and the η²/residual stay in [0, 1].
        df = pd.DataFrame(
            {
                "dv_swap": ["paper", "replica", "replica", float("nan")],
                "freq_swap": ["paper", "replica", "paper", "replica"],
                "effect": [0.0, 1.0, 0.9, 0.2],
            }
        )
        attr = swap_attribution(df, "effect", ["dv_swap", "freq_swap"])
        assert attr.n_obs == 4
        assert 0.0 <= attr.residual_eta_squared <= 1.0
        assert 0.0 <= attr.r_squared <= 1.0

    def test_collinear_swaps_r_squared_still_correct(self):
        # dv_swap and freq_swap perfectly aligned (a 1-D cumulative path):
        # each spec is (paper,paper) or (REPLICA,REPLICA). The model explains the
        # effect (r² ≈ 1) even though the two collinear dims cannot be
        # uniquely credited (their semipartial η² are ~0).
        rows = [
            {"dv_swap": s, "freq_swap": s, "effect": 0.0 if s == "paper" else 1.0}
            for s in ("paper", "replica")
            for _ in range(8)
        ]
        attr = swap_attribution(pd.DataFrame(rows), "effect", ["dv_swap", "freq_swap"])
        assert attr.r_squared == pytest.approx(1.0)
        assert attr.eta_squared["dv_swap"] == pytest.approx(0.0, abs=1e-9)
        assert attr.eta_squared["freq_swap"] == pytest.approx(0.0, abs=1e-9)


class TestChunkedResampling:
    """Small max_elems forces multi-chunk accumulation; the estimate must
    agree with the single-chunk run (both are MC estimates of one p)."""

    def _vectors(self):
        rng = np.random.default_rng(3)
        return [rng.normal(0.05, 1.0, size=40) for _ in range(5)]

    def test_joint_signflip_chunked_matches_single_chunk(self):
        vectors = self._vectors()
        one = joint_signflip_inference(
            vectors, n_resamples=4000, rng=np.random.default_rng(0)
        )
        chunked = joint_signflip_inference(
            vectors, n_resamples=4000, rng=np.random.default_rng(0), max_elems=97
        )
        assert chunked.p_value == pytest.approx(one.p_value, abs=0.05)
        assert chunked.observed == one.observed

    def test_shared_observation_chunking_matches_single_chunk(self):
        vectors = [v.copy() for v in self._vectors()]
        one = joint_signflip_inference(
            vectors,
            shared_observations=True,
            n_resamples=4000,
            rng=np.random.default_rng(0),
        )
        chunked = joint_signflip_inference(
            vectors,
            shared_observations=True,
            n_resamples=4000,
            rng=np.random.default_rng(0),
            max_elems=97,
        )
        assert chunked.p_value == pytest.approx(one.p_value, abs=0.05)


def test_mean_statistic_aggregates_the_cell_means():
    # Cells at +1 and -0.5: mean statistic = 0.25, and clearly positive cells
    # dominate the sign-flip null.
    vectors = [np.full(30, 1.0), np.full(30, -0.5)]

    result = joint_signflip_inference(
        vectors, statistic="mean", n_resamples=500, rng=np.random.default_rng(0)
    )

    assert result.observed == pytest.approx(0.25)
    assert result.statistic == "mean"


def test_share_positive_with_shared_observations_gives_no_free_evidence():
    # Shared sign draws keep the p-value near 0.5 for duplicated observations.
    rng = np.random.default_rng(1)
    base = rng.normal(0.5, 0.2, size=50)
    vectors = [base, base + 0.01, base - 0.01]

    result = joint_signflip_inference(
        vectors,
        statistic="share_positive",
        shared_observations=True,
        alternative="greater",
        n_resamples=500,
        rng=np.random.default_rng(0),
    )

    assert result.observed == 1.0
    assert result.mode == "synchronized"
    assert 0.3 < result.p_value < 0.7
