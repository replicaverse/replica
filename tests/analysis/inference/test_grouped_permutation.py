"""Complete group sums and observation identities govern grouped inference."""

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from replica.analysis.inference.grouped_permutation import grouped_permutation_test
from replica.analysis.inference.significance import (
    paired_permutation_pvalue,
    per_datapoint_permutation_pvalue,
    sum_by_group,
)
from replica.multiverse.curve.stats import (
    family_wise_signflip,
    grouped_family_wise_signflip,
)


def result():
    values = np.array([3.0, -1.0, -4.0, 1.0, 2.0])
    groups = np.array(["a", "a", "b", "b", "c"])
    rows = np.array([[str(i), g, "0"] for i, g in enumerate(groups)])
    fold = SimpleNamespace(
        per_obs_delta_ll=values,
        factor_ids={"stimulus": groups, "participant": rows[:, 0]},
        row_ids=rows,
    )
    return SimpleNamespace(
        fold_results=[fold], metadata={"design": {"permutation_unit": "stimulus"}}
    )


def test_grouped_test_matches_explicit_sum():
    analysis = result()
    inference = grouped_permutation_test(
        analysis, method="exact", alternative="greater"
    )
    assert inference.n_groups == 3
    assert inference.delta_ll == 1
    assert inference.pvalue == paired_permutation_pvalue(
        [2, -3, 2], alternative="greater"
    )
    fold = analysis.fold_results[0]
    assert per_datapoint_permutation_pvalue(
        fold.per_obs_delta_ll, groups=fold.factor_ids["stimulus"]
    ) == per_datapoint_permutation_pvalue([2, -3, 2])


def test_unit_override_is_recorded():
    with pytest.warns(UserWarning, match="overrides declared"):
        test = grouped_permutation_test(result(), unit="participant")
    assert test.unit == "participant"
    assert test.override_warning


def test_incomplete_or_repeated_observations_rejected():
    with pytest.raises(ValueError, match="finite losses"):
        sum_by_group([1, np.nan], ["a", "a"])
    with pytest.raises(ValueError, match="align"):
        sum_by_group([1, 2], ["a"])
    analysis = result()
    analysis.fold_results *= 2
    with pytest.raises(ValueError, match="Repeated scored"):
        grouped_permutation_test(analysis)


def test_family_wise_aligns_row_ids_and_shares_group_signs():
    first = result()
    second = deepcopy(first)
    fold = second.fold_results[0]
    order = np.array([4, 2, 0, 3, 1])
    fold.per_obs_delta_ll = (fold.per_obs_delta_ll * 1.5)[order]
    fold.row_ids = fold.row_ids[order]
    fold.factor_ids = {n: v[order] for n, v in fold.factor_ids.items()}
    actual = grouped_family_wise_signflip([first, second], n_resamples=200)
    expected = family_wise_signflip([[2, -3, 2], [3, -4.5, 3]], n_resamples=200)
    assert actual.cell_z == pytest.approx(expected.cell_z)
    assert actual == replace(
        expected,
        cell_z=actual.cell_z,
        permutation_unit="stimulus",
        inference_scopes=actual.inference_scopes,
    )
    assert all(not s["cv_scope_recorded"] for s in actual.inference_scopes)
    fold.row_ids[0, 0] = "bad"
    with pytest.raises(ValueError, match="identical row identities"):
        grouped_family_wise_signflip([first, second])


@pytest.mark.parametrize("labels", [[1, "1"], [1, 1.0], [True, 1]])
def test_mixed_label_types_do_not_merge(labels):
    with pytest.raises(ValueError, match="mix scalar types"):
        sum_by_group([1.0, 2.0], labels)


def test_numpy_float_missing_group_rejected():
    with pytest.raises(ValueError, match="non-finite"):
        sum_by_group([1.0, 2.0], np.array([np.nan, 2], dtype=np.float32))


@pytest.mark.parametrize("field", ["labels", "losses"])
def test_masked_group_inputs_do_not_expose_hidden_values(field):
    values = np.array([1.0, 999.0, 2.0])
    groups = np.array(["a", "hidden", "b"])
    if field == "labels":
        groups = np.ma.array(groups, mask=[False, True, False])
    else:
        values = np.ma.array(values, mask=[False, True, False])
    for operation in (sum_by_group, per_datapoint_permutation_pvalue):
        with pytest.raises(ValueError, match="Missing labels|finite losses"):
            operation(values, groups=groups)


@pytest.mark.parametrize("field", ["factor", "row", "loss"])
def test_grouped_inference_rejects_masks_before_concatenating_folds(field):
    analysis = result()
    fold = analysis.fold_results[0]
    if field == "factor":
        fold.factor_ids["stimulus"] = np.ma.array(
            fold.factor_ids["stimulus"], mask=[True, False, False, False, False]
        )
    elif field == "row":
        fold.row_ids = np.ma.array(fold.row_ids, mask=False)
        fold.row_ids.mask[0, 1] = True
    else:
        fold.per_obs_delta_ll = np.ma.array(
            fold.per_obs_delta_ll, mask=[True, False, False, False, False]
        )
    with pytest.raises(ValueError, match="complete losses"):
        grouped_permutation_test(analysis)


def test_unmasked_group_arrays_retain_their_values():
    expected = sum_by_group([1.0, 2.0, 3.0], ["a", "b", "a"])
    actual = sum_by_group(
        np.ma.array([1.0, 2.0, 3.0], mask=False),
        np.ma.array(["a", "b", "a"], mask=False),
    )
    for left, right in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(left, right)


def test_grouped_monte_carlo_counts_ties_like_exact_enumeration(monkeypatch):
    import itertools

    from replica.analysis.inference import significance

    signs = np.array(list(itertools.product((-1.0, 1.0), repeat=8)))
    monkeypatch.setattr(significance, "sign_chunks", lambda *args: iter([signs]))
    values = np.random.default_rng(1).normal(size=8)
    exact = paired_permutation_pvalue(values, method="exact")
    actual = per_datapoint_permutation_pvalue(
        values, groups=list("abcdefgh"), n_resamples=len(signs)
    )
    assert actual == pytest.approx((exact * len(signs) + 1) / (len(signs) + 1))


@pytest.mark.parametrize("scale", [1e-200, 1e200])
@pytest.mark.parametrize("alternative", ["two-sided", "greater", "less"])
def test_grouped_inference_is_invariant_to_loss_units(scale, alternative):
    first = result()
    first.fold_results[0].per_obs_delta_ll = np.array([3.0, 1.0, 4.0, 1.0, 2.0])
    if alternative == "less":
        first.fold_results[0].per_obs_delta_ll *= -1
    second = deepcopy(first)
    second.fold_results[0].per_obs_delta_ll *= scale
    for method in ("exact", "monte-carlo"):
        options = dict(method=method, alternative=alternative, n_resamples=1000)
        expected_p = grouped_permutation_test(first, **options).pvalue
        assert expected_p < 0.5
        assert expected_p == (grouped_permutation_test(second, **options).pvalue)
    options = dict(alternative=alternative, n_resamples=1000)
    expected = grouped_family_wise_signflip([first, first], **options)
    actual = grouped_family_wise_signflip([first, second], **options)
    assert actual.cell_z == pytest.approx(expected.cell_z)
    assert actual.adj_pvalues == expected.adj_pvalues
    assert actual.share_significant_p == expected.share_significant_p


def test_group_sum_preserves_cancellation_independently_of_row_order():
    for values in ([1e16, 1, -1e16], [1e16, -1e16, 1]):
        _, sums = sum_by_group(values, ["a"] * 3)
        np.testing.assert_array_equal(sums, [1])


def test_nonrepresentable_group_sum_rejected():
    with pytest.raises(ValueError, match="finite group sums"):
        family_wise_signflip([[1e308, 1e308, 1]], groups=["a", "a", "b"])


@pytest.mark.parametrize("missing", [None, np.nan, ""])
def test_missing_stimulus_row_identity_rejected(missing):
    analysis = result()
    fold = analysis.fold_results[0]
    fold.row_ids = fold.row_ids.astype(object)
    fold.row_ids[0, 1] = missing
    with pytest.raises(ValueError, match="row identit"):
        grouped_permutation_test(analysis)


def test_aggregate_rows_allow_empty_participant_identity():
    analysis = result()
    fold = analysis.fold_results[0]
    fold.row_ids[:, 0] = ""
    fold.row_ids[:, 2] = np.arange(len(fold.row_ids)).astype(str)
    del fold.factor_ids["participant"]
    assert np.isfinite(grouped_permutation_test(analysis, method="exact").pvalue)


@pytest.mark.parametrize("budget", [0, -1, 1.5, True])
def test_grouped_resample_budget_must_be_positive_integer(budget):
    with pytest.raises(ValueError, match="positive integer"):
        grouped_permutation_test(result(), n_resamples=budget)
    with pytest.raises(ValueError, match="positive integer"):
        family_wise_signflip([[0, 0]], groups=["a", "b"], n_resamples=budget)


def test_numeric_row_ids_cannot_alias_text_ids_across_specs():
    first, second = result(), result()
    second.fold_results[0].row_ids = second.fold_results[0].row_ids.astype(object)
    second.fold_results[0].row_ids[0, 0] = 0
    with pytest.raises(ValueError, match="row identities"):
        grouped_family_wise_signflip([first, second])


def test_mixed_factor_types_across_folds_are_not_coerced_to_text():
    analysis = result()
    fold = deepcopy(analysis.fold_results[0])
    fold.row_ids[:, 2] = "1"
    fold.factor_ids["stimulus"] = np.arange(len(fold.row_ids))
    analysis.fold_results.append(fold)
    with pytest.raises(ValueError, match="mix scalar types"):
        grouped_permutation_test(analysis)


@pytest.mark.parametrize("alpha", [np.nan, np.inf, 0, 1, -0.1])
def test_grouped_family_wise_rejects_invalid_alpha_before_empty_shortcut(alpha):
    with pytest.raises(ValueError, match="alpha"):
        family_wise_signflip([[]], groups=[], alpha=alpha)


@pytest.mark.parametrize("max_elems", [0, -1, 1.5, True])
def test_grouped_inference_rejects_invalid_memory_budget(max_elems):
    with pytest.raises(ValueError, match="max_elems"):
        per_datapoint_permutation_pvalue([], groups=[], max_elems=max_elems)
    with pytest.raises(ValueError, match="max_elems"):
        family_wise_signflip([[]], groups=[], max_elems=max_elems)


@pytest.mark.parametrize("familywise", [False, True])
@pytest.mark.parametrize("explicit_unit", [False, True])
def test_crossed_cv_requires_acknowledgement_even_for_declared_unit(
    familywise, explicit_unit
):
    analysis = result()
    analysis.metadata["design"].update(
        cv_unit=("participant", "stimulus"), condition_on=()
    )
    original = deepcopy(analysis.metadata)
    test = grouped_family_wise_signflip if familywise else grouped_permutation_test
    argument = [analysis] if familywise else analysis
    options = {"unit": "stimulus"} if explicit_unit else {}
    with pytest.raises(ValueError, match="allow_scope_mismatch=True"):
        test(argument, n_resamples=20, **options)
    with pytest.warns(
        UserWarning, match="do not resample declared CV factors"
    ) as caught:
        inference = test(argument, allow_scope_mismatch=True, n_resamples=20, **options)
    assert len(caught) == 1
    scope = inference.inference_scopes[0] if familywise else inference.inference_scope
    assert scope["cv_units"] == ("participant", "stimulus")
    assert scope["permutation_unit"] == "stimulus"
    assert scope["unresampled_cv_units"] == ("participant",)
    assert scope["scope_mismatch_acknowledged"]
    assert scope["warning"] == str(caught[0].message)
    assert analysis.metadata == original  # Never relabel the original CV claim.


def test_matched_scope_records_conditioning_without_certifying_validity():
    analysis = result()
    analysis.metadata["design"].update(
        cv_unit="stimulus", condition_on=("participant",)
    )
    inference = grouped_permutation_test(analysis, method="exact")
    scope = inference.inference_scope
    assert scope["cv_units"] == ("stimulus",)
    assert scope["condition_on"] == ("participant",)
    assert scope["unresampled_cv_units"] == ()
    assert not scope["scope_mismatch_acknowledged"]
    assert scope["warning"] is None
    assert "does not establish" in scope["interpretation"]


def test_family_scope_check_covers_every_analysis_and_records_each_scope():
    first, second = result(), result()
    first.metadata["design"].update(cv_unit="stimulus", condition_on=("participant",))
    second.metadata["design"].update(cv_unit=["participant", "stimulus"])
    with pytest.raises(ValueError, match="participant"):
        grouped_family_wise_signflip([first, second], n_resamples=20)
    with pytest.warns(UserWarning, match="do not resample"):
        inference = grouped_family_wise_signflip(
            [first, second], allow_scope_mismatch=True, n_resamples=20
        )
    assert [s["unresampled_cv_units"] for s in inference.inference_scopes] == [
        (),
        ("participant",),
    ]
    assert [s["condition_on"] for s in inference.inference_scopes] == [
        ("participant",),
        (),
    ]


def test_inference_override_cannot_silently_change_single_axis_scope():
    analysis = result()
    analysis.metadata["design"]["cv_unit"] = "participant"
    with pytest.raises(ValueError, match="do not resample"):
        grouped_permutation_test(analysis, unit="stimulus", method="exact")


@pytest.mark.parametrize("acknowledgement", ["false", "true", 1, None])
def test_scope_acknowledgement_requires_a_boolean(acknowledgement):
    with pytest.raises(ValueError, match="must be a boolean"):
        grouped_permutation_test(result(), allow_scope_mismatch=acknowledgement)
