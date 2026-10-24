"""The permutation procedures the CDR-family papers use to decide claims.

Each era of the reference tests a different statistic, and the differences
decide claims: a per-observation paired swap for the early papers and a
permutation of ensemble member columns whose statistic is a median for the
later ones. These tests pin the properties that distinguish them.
"""

import numpy as np

from replica.analysis.inference.significance import (
    ensemble_column_permutation,
    paired_swap_permutation,
)


class TestPairedSwapPermutation:
    def test_identical_models_are_not_distinguished(self):
        """Without the reference's explicit guard, a zero difference would
        report the smallest attainable p-value, since no strict hit can occur."""
        rng = np.random.default_rng(0)
        values = rng.normal(size=400)
        result = paired_swap_permutation(
            values, values.copy(), n_iter=500, rng=np.random.default_rng(1)
        )
        assert result.difference == 0.0
        assert result.p_value == 1.0

    def test_a_clear_advantage_is_detected(self):
        rng = np.random.default_rng(0)
        a = rng.normal(size=400)
        b = a - 0.5
        result = paired_swap_permutation(
            a, b, mode="loglik", n_iter=1000, rng=np.random.default_rng(1)
        )
        assert result.difference > 0
        assert result.p_value < 0.01

    def test_pvalue_can_never_be_zero(self):
        a = np.arange(200.0)
        b = a - 100.0
        result = paired_swap_permutation(a, b, n_iter=100, rng=np.random.default_rng(0))
        assert result.p_value == 1 / 101

    def test_nonfinite_rows_drop_from_both_models(self):
        a = np.array([1.0, np.nan, 3.0, 4.0])
        b = np.array([0.0, 1.0, np.inf, 2.0])
        result = paired_swap_permutation(a, b, n_iter=50, rng=np.random.default_rng(0))
        assert result.n == 2
        assert result.n_dropped == 2

    def test_directional_gate_short_circuits_the_wrong_sign(self):
        """The first model is the ablated one; testing it is pointless when
        it is already the better of the two."""
        ablated = np.array([1.0, 1.0, 1.0])
        fuller = np.array([2.0, 2.0, 2.0])
        # As log-likelihoods the ablated model is worse, which is the
        # direction worth testing.
        tested = paired_swap_permutation(
            ablated,
            fuller,
            mode="loglik",
            directional_gate=True,
            n_iter=100,
            rng=np.random.default_rng(0),
        )
        assert tested.p_value < 1.0
        # As losses the same numbers make the ablated model better, so the
        # comparison stops without resampling.
        gated = paired_swap_permutation(
            ablated,
            fuller,
            mode="loss",
            directional_gate=True,
            n_iter=100,
            rng=np.random.default_rng(0),
        )
        assert gated.p_value == 1.0

    def test_ties_are_not_counted_as_hits(self):
        """The reference counts strictly greater, so an exact tie is not extreme."""
        a = np.array([2.0, -1.0])
        b = np.zeros(2)
        result = paired_swap_permutation(
            a,
            b,
            mode="loglik",
            n_iter=400,
            rng=np.random.default_rng(0),
            keep_null=True,
        )
        assert result.difference == 1.0
        assert result.null is not None
        # Sign flips of (2, -1) give totals of +-1 and +-3. Only the +-3 draws
        # exceed the observed 1; the +-1 draws tie it and must not count.
        assert set(np.unique(np.abs(result.null))) == {1.0, 3.0}
        ties = int(np.count_nonzero(np.abs(result.null) == 1.0))
        assert ties > 0
        expected_hits = int(np.count_nonzero(np.abs(result.null) > 1.0))
        assert result.p_value == (expected_hits + 1) / (result.n_iter + 1)

    def test_one_tailed_test_follows_a_positive_effect(self):
        a = np.array([2.0, -1.0])
        b = np.zeros(2)
        result = paired_swap_permutation(
            a,
            b,
            mode="loglik",
            n_tails=1,
            n_iter=400,
            rng=np.random.default_rng(0),
            keep_null=True,
        )
        assert result.null is not None
        expected_hits = int(np.count_nonzero(result.null >= result.difference))
        assert result.p_value == (expected_hits + 1) / (result.n_iter + 1)

    def test_one_tailed_test_follows_a_negative_effect(self):
        a = np.array([-2.0, 1.0])
        b = np.zeros(2)
        result = paired_swap_permutation(
            a,
            b,
            mode="loglik",
            n_tails=1,
            n_iter=400,
            rng=np.random.default_rng(0),
            keep_null=True,
        )
        assert result.null is not None
        expected_hits = int(np.count_nonzero(result.null <= result.difference))
        assert result.p_value == (expected_hits + 1) / (result.n_iter + 1)

    def test_tail_count_must_be_one_or_two(self):
        with np.testing.assert_raises_regex(ValueError, "n_tails"):
            paired_swap_permutation(
                np.array([1.0]),
                np.array([0.0]),
                n_tails=3,
                n_iter=1,
            )


class TestEnsembleColumnPermutation:
    def test_single_member_matches_the_paired_swap(self):
        rng = np.random.default_rng(0)
        a = rng.normal(size=300)
        b = a - 0.2
        one = ensemble_column_permutation(
            a, b, n_iter=400, rng=np.random.default_rng(7)
        )
        two = paired_swap_permutation(
            a, b, mode="loglik", n_iter=400, rng=np.random.default_rng(7)
        )
        assert one.p_value == two.p_value
        assert one.difference == two.difference

    def test_statistic_is_the_median_over_members(self):
        a = np.tile(np.array([[1.0, 2.0, 30.0]]), (10, 1))
        b = np.zeros((10, 3))
        result = ensemble_column_permutation(
            a, b, n_iter=10, rng=np.random.default_rng(0)
        )
        # Column totals are 10, 20, 300; the statistic is their median, 20.
        assert result.a_stat == 20.0
        assert result.b_stat == 0.0

    def test_members_are_exchanged_within_an_observation(self):
        rng = np.random.default_rng(0)
        a = rng.normal(size=(200, 4)) + 1.0
        b = rng.normal(size=(200, 4))
        result = ensemble_column_permutation(
            a, b, n_iter=500, rng=np.random.default_rng(1)
        )
        assert result.difference > 0
        assert result.p_value < 0.05

    def test_nonfinite_handling_is_selectable(self):
        a = np.array([[1.0, 2.0], [np.nan, 1.0], [2.0, 2.0]])
        b = np.zeros((3, 2))
        dropped = ensemble_column_permutation(
            a, b, n_iter=10, rng=np.random.default_rng(0)
        )
        assert dropped.n == 2 and dropped.n_dropped == 1
        kept = ensemble_column_permutation(
            a, b, n_iter=10, drop_nonfinite=False, rng=np.random.default_rng(0)
        )
        assert kept.n == 3 and kept.n_dropped == 0

    def test_one_tailed_test_follows_the_observed_direction(self):
        """An ablated model with a clearly lower likelihood has a negative
        difference; the one-tailed count must look at that tail, inclusively,
        as the single-member test and the reference do."""
        rng = np.random.default_rng(0)
        fuller = rng.normal(size=(400, 2))
        ablated = fuller - 0.3 + 0.1 * rng.normal(size=(400, 2))
        result = ensemble_column_permutation(
            ablated,
            fuller,
            mode="loglik",
            n_tails=1,
            n_iter=500,
            rng=np.random.default_rng(1),
            keep_null=True,
        )
        assert result.difference < 0
        assert result.null is not None
        expected_hits = int(np.count_nonzero(result.null <= result.difference))
        assert result.p_value == (expected_hits + 1) / (result.n_iter + 1)
        assert result.p_value < 0.01
        one_member = ensemble_column_permutation(
            ablated[:, :1],
            fuller[:, :1],
            mode="loglik",
            n_tails=1,
            n_iter=500,
            rng=np.random.default_rng(1),
        )
        assert one_member.p_value < 0.01

    def test_squared_error_is_gated_and_averaged_as_a_loss(self):
        """``mse`` is the reference's name for a loss: a higher value is worse,
        and rows are averaged."""
        rng = np.random.default_rng(0)
        fuller = rng.random((300, 2))
        ablated = fuller + 0.5
        for mode in ("loss", "mse"):
            result = ensemble_column_permutation(
                ablated,
                fuller,
                mode=mode,
                directional_gate=True,
                n_iter=200,
                rng=np.random.default_rng(1),
            )
            assert result.a_stat == float(np.median(ablated.mean(axis=0)))
            assert result.p_value < 0.05, mode
        swapped = paired_swap_permutation(
            ablated[:, 0], fuller[:, 0], mode="mse", n_iter=10
        )
        assert swapped.a_stat == float(ablated[:, 0].mean())

    def test_unknown_modes_are_rejected(self):
        a, b = np.ones(5), np.zeros(5)
        with np.testing.assert_raises_regex(ValueError, "mode"):
            ensemble_column_permutation(np.ones((5, 2)), np.zeros((5, 2)), mode="corr")
        with np.testing.assert_raises_regex(ValueError, "mode"):
            paired_swap_permutation(a, b, mode="likelihood")
