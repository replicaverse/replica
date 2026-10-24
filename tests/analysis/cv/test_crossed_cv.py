"""Joint holdout excludes each identity, including partially held-out rows."""

import numpy as np
import pytest

from replica.analysis import AnalysisConfig, CrossedGroupSplitter, Design


def test_crossed_rectangles_score_every_observation_once_without_identity_leakage():
    readers, sources, words = np.indices((6, 5, 3)).reshape(3, -1)
    # Unequal and incomplete reader/source blocks must retain their identities.
    keep = ~((readers == 0) & (sources == 2)) & ~((readers == 3) & (words == 0))
    labels = np.column_stack((readers[keep], sources[keep]))
    splitter = CrossedGroupSplitter((2, 3))
    tested = np.zeros(len(labels), dtype=int)
    for train, test in splitter.split(np.zeros(len(labels)), labels, None, None):
        tested += test
        assert not (train & test).any()
        for axis in range(2):
            assert not set(labels[train, axis]) & set(labels[test, axis])
        assert (~(train | test)).any()
    assert np.all(tested == 1)
    assert len(splitter.fold_metadata()) == 6


def test_default_design_audits_both_axes_and_single_axis_remains_explicit():
    design = Design(comparison="predictor")
    assert design.cv_units == ("participant", "stimulus")
    assert design.permutation_unit is None
    labels = {"participant": np.array([0, 0, 1, 1]), "stimulus": np.array([0, 1, 0, 1])}
    train = np.array([True, False, True, False])
    with pytest.warns(UserWarning, match="participant"):
        messages = design.check_splits([(train, ~train)], labels)
    assert len(messages) == 1
    assert Design(cv_unit="stimulus").check_splits([(train, ~train)], labels) == []


def test_crossed_cv_rejects_unidentifiable_axes_and_empty_training_rectangles():
    with pytest.raises(ValueError, match="at least two groups"):
        list(CrossedGroupSplitter().split(np.zeros(3), np.arange(3), np.zeros(3), None))
    labels = np.array([[0, 0], [0, 1], [1, 0]])
    with pytest.raises(ValueError, match="no training rows"):
        list(CrossedGroupSplitter(2).split(np.zeros(3), labels, None, None))


def _crossed_groups(readers=20, sources=20):
    """Provide whole reader/source identities on a complete response grid."""
    r, s = np.indices((readers, sources)).reshape(2, -1)
    return {"participant": r, "source": s}


def test_default_target_and_explicit_folds_have_identical_partitions():
    groups = _crossed_groups()
    y = np.zeros(400)
    default = CrossedGroupSplitter()
    expected = CrossedGroupSplitter(10)
    configured = AnalysisConfig().resolve_cv()
    assert default.seed == configured.seed == 42
    assert default.preview(groups) == configured.preview(groups)
    for actual, reference in zip(
        default.split(y, groups, None, None),
        expected.split(y, groups, None, None),
        strict=True,
    ):
        for a, b in zip(actual, reference, strict=True):
            np.testing.assert_array_equal(a, b)
    summary = default.summary()
    assert summary["requested_train_fraction"] == {"participant": 0.9, "source": 0.9}
    assert summary["resolved_folds"] == {"participant": 10, "source": 10}
    assert summary["fits_per_model_per_cycle"] == 100
    assert summary["train_fraction_rows_range"] == [0.81, 0.81]


@pytest.mark.parametrize("fraction,folds", [(0.5, 2), (0.8, 5), (0.81, 6), (0.9, 10)])
def test_fraction_rounding_targets_nominal_training_fraction(fraction, folds):
    summary = CrossedGroupSplitter(train_fraction=fraction).preview(_crossed_groups())
    assert summary["resolved_folds"] == {"participant": folds, "source": folds}
    assert summary["fits_per_model_per_cycle"] == folds**2


def test_named_targets_follow_factor_names_instead_of_mapping_order():
    splitter = CrossedGroupSplitter(train_fraction={"source": 0.9, "participant": 0.8})
    summary = splitter.preview(_crossed_groups())
    assert summary["resolved_folds"] == {"participant": 5, "source": 10}
    assert summary["train_fraction_rows_range"] == [0.72, 0.72]
    assert summary["fits_per_model_per_cycle"] == 50
    with pytest.raises(ValueError, match="factor names"):
        splitter.preview({"participant": np.arange(20), "wrong_source": np.arange(20)})


def test_small_group_warning_and_uneven_actual_fractions_are_reported():
    with pytest.warns(UserWarning, match="source.*4 groups"):
        summary = CrossedGroupSplitter().preview(_crossed_groups(sources=4))
    assert summary["resolved_folds"] == {"participant": 10, "source": 4}
    assert summary["train_fraction_groups_range"]["source"] == [0.75, 0.75]
    assert summary["train_fraction_rows_range"] == [0.675, 0.675]
    summary = CrossedGroupSplitter().preview(_crossed_groups(readers=45, sources=12))
    assert summary["nominal_train_fraction"]["source"] == 0.9
    assert summary["train_fraction_groups_range"]["source"] == pytest.approx(
        [10 / 12, 11 / 12]
    )
    assert summary["train_fraction_groups_range"]["participant"] == pytest.approx(
        [40 / 45, 41 / 45]
    )


def test_seeded_partitions_reproduce_preview_and_ignore_row_order_and_response_values():
    groups = _crossed_groups()
    splitter = CrossedGroupSplitter(seed=17)
    preview = splitter.preview(groups)
    original = list(splitter.split(np.full(400, np.nan), groups, None, None))
    assert splitter.summary() == preview
    permutation = np.random.default_rng(2).permutation(400)
    reordered = list(
        splitter.split(
            np.zeros(400), {k: v[permutation] for k, v in groups.items()}, None, None
        )
    )
    for (train, test), (new_train, new_test) in zip(original, reordered, strict=True):
        np.testing.assert_array_equal(train[permutation], new_train)
        np.testing.assert_array_equal(test[permutation], new_test)
        for labels in groups.values():
            assert not set(labels[train]) & set(labels[test])
    assert np.all(sum(test for _, test in original) == 1)
    other = list(CrossedGroupSplitter(seed=18).split(np.zeros(400), groups, None, None))
    assert any(
        not np.array_equal(a[1], b[1]) for a, b in zip(original, other, strict=True)
    )


def test_preview_counts_only_nonempty_intersections_and_observed_training_groups():
    groups = _crossed_groups(readers=4, sources=4)
    keep = (groups["participant"] % 2) == (groups["source"] % 2)
    groups = {k: v[keep] for k, v in groups.items()}
    summary = CrossedGroupSplitter(2, seed=None).preview(groups)
    assert summary["possible_test_intersections"] == 4
    assert summary["fits_per_model_per_cycle"] == 2
    assert summary["train_fraction_rows_range"] == [0.5, 0.5]
    assert summary["train_fraction_groups_range"] == {
        "participant": [0.5, 0.5],
        "source": [0.5, 0.5],
    }


@pytest.mark.parametrize(
    "fraction", [0.49, 1, float("nan"), float("inf"), True, (), {}]
)
def test_invalid_training_targets_fail_before_splitting(fraction):
    with pytest.raises(ValueError):
        CrossedGroupSplitter(train_fraction=fraction)


def test_conflicting_configuration_and_incomplete_summaries_are_rejected():
    with pytest.raises(ValueError, match="either"):
        CrossedGroupSplitter(5, train_fraction=0.9)
    with pytest.raises(ValueError, match="integer"):
        CrossedGroupSplitter(2.5)
    with pytest.raises(ValueError, match="seed"):
        CrossedGroupSplitter(seed=-1)
    with pytest.raises(ValueError, match="one fold count"):
        CrossedGroupSplitter((2, 3, 4)).preview(_crossed_groups())
    with pytest.raises(ValueError, match="aligned"):
        CrossedGroupSplitter(2).preview({"a": np.arange(4), "b": np.arange(3)})
    splitter = CrossedGroupSplitter(2)
    with pytest.raises(RuntimeError, match="Complete"):
        splitter.summary()
    iterator = splitter.split(np.zeros(400), _crossed_groups(), None, None)
    next(iterator)
    with pytest.raises(RuntimeError, match="Complete"):
        splitter.summary()
