"""Tests for replica.analysis.inference.effect_size."""

import math

import numpy as np
import pytest
from scipy import stats

from replica.analysis.inference.effect_size import cohens_f2, nadeau_bengio_ci

# ---------------------------------------------------------------------------
# cohens_f2
# ---------------------------------------------------------------------------


class TestCohensF2:
    def test_basic_ratio(self):
        # ΔR²=0.02 on a full model explaining 0.20 → 0.02/0.80 = 0.025.
        assert cohens_f2(0.02, 0.20) == pytest.approx(0.025)

    def test_full_r2_at_one_is_nan(self):
        assert math.isnan(cohens_f2(0.01, 1.0))

    def test_full_r2_above_one_is_nan(self):
        assert math.isnan(cohens_f2(0.01, 1.2))

    def test_negative_delta_r2_gives_negative_f2(self):
        assert cohens_f2(-0.02, 0.20) == pytest.approx(-0.025)

    def test_nonfinite_inputs_nan(self):
        assert math.isnan(cohens_f2(None, 0.2))
        assert math.isnan(cohens_f2(0.02, None))


# ---------------------------------------------------------------------------
# nadeau_bengio_ci
# ---------------------------------------------------------------------------


class TestNadeauBengioCI:
    def test_wider_than_naive_t_ci(self):
        # The correction inflates variance, so the interval must be wider
        # than the naive Student-t interval on the same fold values.
        vals = [0.10, 0.12, 0.08, 0.11, 0.09, 0.13, 0.07, 0.10, 0.12, 0.09]
        _, nlo, nhi = nadeau_bengio_ci(vals)
        x = np.asarray(vals)
        se = x.std(ddof=1) / math.sqrt(x.size)
        naive_half = stats.t.ppf(0.975, x.size - 1) * se
        assert (nhi - nlo) > 2 * naive_half

    def test_default_test_ratio_is_one_over_k_minus_one(self):
        # For 10 folds, test_ratio defaults to 1/9; supplying it explicitly
        # must reproduce the same interval.
        vals = [0.10, 0.12, 0.08, 0.11, 0.09, 0.13, 0.07, 0.10, 0.12, 0.09]
        auto = nadeau_bengio_ci(vals)
        explicit = nadeau_bengio_ci(vals, test_ratio=1.0 / 9.0)
        assert auto == pytest.approx(explicit)

    def test_mean_matches(self):
        vals = [0.1, 0.2, 0.3]
        mean, _, _ = nadeau_bengio_ci(vals)
        assert mean == pytest.approx(float(np.mean(vals)))

    def test_single_fold_nan_bounds(self):
        mean, lo, hi = nadeau_bengio_ci([0.2])
        assert mean == pytest.approx(0.2)
        assert math.isnan(lo) and math.isnan(hi)
