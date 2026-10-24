"""OutlierFilter: absolute floor + per-participant MAD/SD trims.

Pins two properties: (1) the trim is per-participant, and (2) both MAD
and the conventional SD method are selectable. The
headline test shows a global trim does the wrong thing across readers of
different speed.
"""

import numpy as np
import pytest

from replica.analysis.data.filters import OutlierFilter


def test_absolute_floor_drops_below_threshold():
    f = OutlierFilter(floor=80.0)
    y = np.array([50.0, 79.0, 80.0, 200.0])
    assert f.keep_mask(y).tolist() == [False, False, True, True]


def test_default_k_is_2p5():
    assert OutlierFilter(method="mad").k == 2.5


def test_method_validation():
    import pytest

    with pytest.raises(ValueError, match="method must be"):
        OutlierFilter(method="zscore")


def test_inactive_keeps_everything():
    f = OutlierFilter()
    assert not f.is_active
    y = np.array([1.0, 1e9, np.nan])
    assert f.keep_mask(y).all()


def test_mad_per_participant_vs_global():
    """A fast reader (A) and a slow reader (B). A's 420 ms is an outlier *for A*
    but ordinary on the pooled scale; A's 100-130 ms are normal *for A* but
    extreme vs the pooled median. Per-participant MAD drops only A's 420;
    a global MAD does the opposite (drops A's normal RTs, keeps the outlier)."""
    y = np.array([100.0, 110, 120, 130, 420, 400, 410, 420, 430, 440])
    pid = np.array(["A"] * 5 + ["B"] * 5)

    per_p = OutlierFilter(method="mad", k=2.5).keep_mask(y, participants=pid)
    assert per_p.tolist() == [
        True,
        True,
        True,
        True,
        False,
        True,
        True,
        True,
        True,
        True,
    ]

    glob = OutlierFilter(method="mad", k=2.5).keep_mask(y)  # no participants
    assert glob.tolist() == [
        False,
        False,
        False,
        False,
        True,
        True,
        True,
        True,
        True,
        True,
    ]
    # The two grains genuinely disagree (this is the whole point).
    assert per_p.tolist() != glob.tolist()


def test_sd_per_participant_trims_within_reader():
    # Many tight inliers + one gross outlier, so the (non-robust) SD still
    # catches it; B has zero within-reader variance and is kept entirely.
    y = np.array([300.0] * 9 + [2000.0] + [500.0] * 5)
    pid = np.array(["A"] * 10 + ["B"] * 5)
    keep = OutlierFilter(method="sd", k=2.5).keep_mask(y, participants=pid)
    assert not keep[9]  # A's 2000 ms outlier dropped
    assert keep.sum() == 14  # everything else kept


def test_sd_is_less_robust_than_mad():
    """Leys et al. (2013): with few points one outlier inflates the SD enough to
    mask itself, while the robust MAD still flags it. Justifies the MAD default."""
    y = np.array([300.0, 305, 310, 315, 900])
    pid = np.array(["A"] * 5)
    sd_keep = OutlierFilter(method="sd", k=2.5).keep_mask(y, participants=pid)
    mad_keep = OutlierFilter(method="mad", k=2.5).keep_mask(y, participants=pid)
    assert sd_keep.all()  # SD fails to flag the 900 ms point
    assert not mad_keep[4]  # MAD flags it


def test_canonical_options_support_global_mad():
    f = OutlierFilter(floor=50.0, ceiling=3000.0, method="mad", k=3.0)
    assert f.floor == 50.0 and f.ceiling == 3000.0
    assert f.method == "mad" and f.k == 3.0
    y = np.array([10.0, 100, 110, 120, 130, 5000])
    keep = f.keep_mask(y)
    assert not keep[0] and not keep[-1]


def test_custom_quantile_rule_is_grouped_and_combined_with_bounds():
    seen = []

    def middle(values):
        seen.append(values.copy())
        low, high = np.quantile(values, [0.25, 0.75])
        return (values >= low) & (values <= high)

    values = np.array([0.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 13.0, np.nan])
    groups = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1])
    keep = OutlierFilter(floor=2, ceiling=11, method=middle).keep_mask(values, groups)
    assert keep[:8].tolist() == [False, False, True, False, False, True, False, False]
    assert [v.tolist() for v in seen] == [
        [0.0, 1.0, 2.0, 3.0],
        [10.0, 11.0, 12.0, 13.0],
    ]


@pytest.mark.parametrize("invalid", [True, [True], [1, 0], [[True, False]]])
def test_custom_rule_rejects_non_boolean_or_misaligned_masks(invalid):
    with pytest.raises(ValueError, match="boolean mask.*same shape"):
        OutlierFilter(method=lambda values: invalid).keep_mask(np.array([1.0, 2.0]))


def test_custom_rule_prepares_rows_before_transform_and_records_identity():
    import json
    from dataclasses import replace

    from replica.analysis import OLS, AnalysisConfig, prepare_observations
    from replica.analysis.specification.evaluation import EvaluationSpec
    from replica.feature import UnitLength
    from tests.analysis._fixtures import make_toy_dataset

    def lower_half(values):
        return values <= np.median(values)

    dataset = make_toy_dataset(n_stimuli=2, n_words=4, n_participants=2)
    features = EvaluationSpec(
        targets=[dataset.feature("spr_rt")], predictors=[UnitLength()]
    )
    config = AnalysisConfig(
        model=OLS(), design=None, outlier_method=lower_half, transform_dv=np.log
    )
    rows = prepare_observations(dataset, features, config)
    original = prepare_observations(
        dataset, features, replace(config, outlier_method=None, transform_dv=None)
    )
    expected = OutlierFilter(method=lower_half).keep_mask(
        original.y, original.participants
    )
    np.testing.assert_array_equal(rows.y, np.log(original.y[expected]))
    np.testing.assert_array_equal(rows.unit_indices, original.unit_indices[expected])
    metadata = config.metadata()
    assert metadata["outlier_filter"]["method"].endswith(".lower_half")
    json.dumps(metadata)
