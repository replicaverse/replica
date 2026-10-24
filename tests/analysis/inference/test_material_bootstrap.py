"""Material isolation and independent checks of the full-refit calculation."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from replica.analysis.cv.materials import material_folds, material_id
from replica.analysis.inference.material_bootstrap import Contrast, GroupedOLS


def test_declared_materials_and_missing_identifiers():
    pages = [
        SimpleNamespace(stimulus_id=f"sbsat_book_{p}", metadata={"cv_group": "book"})
        for p in (1, 2)
    ]
    assert material_id(pages[0]) == material_id(pages[1])
    own = [SimpleNamespace(stimulus_id=f"text_{v}", metadata={}) for v in (1, 2)]
    assert material_id(own[0]) != material_id(own[1])
    for key in (" ", float("nan"), np.float32("nan"), pd.NA, ["a"], {"a": 1}):
        with pytest.raises(ValueError, match="cv_group"):
            material_id(SimpleNamespace(stimulus_id="x", metadata={"cv_group": key}))
    for sid in ("", None, float("nan"), pd.NA):
        with pytest.raises(ValueError, match="identit"):
            material_id(SimpleNamespace(stimulus_id=sid, metadata={}))
    assert material_id(SimpleNamespace(stimulus_id=7, metadata={"cv_group": 3})) == "3"
    with pytest.raises(ValueError, match="identit"):
        material_folds(["text", None])
    with pytest.raises(ValueError, match="two distinct"):
        material_folds(["text", "text"])


@pytest.mark.parametrize("sigma", ["mle", "df"])
@pytest.mark.parametrize(
    "case",
    [
        "ordinary",
        "rescaled",
        "extreme_units",
        "dependent",
        "small_groups",
        "cancellation",
        "noisy_cancellation",
    ],
)
def test_weighted_refits_equal_explicit_row_copies(sigma, case):
    rng = np.random.default_rng(123)
    sizes = [1, 2, 3, 4] if case == "small_groups" else [13, 17, 21, 25]
    groups = np.repeat(["a", "b", "c", "d"], sizes)
    x = rng.normal(size=len(groups))
    y = 4 + 2 * x + rng.normal(size=len(groups))
    reference_design = np.column_stack(
        [np.ones(len(x)), x, x if case == "dependent" else x**2]
    )
    if case in ("cancellation", "noisy_cancellation"):
        reference_design[:, 1:] = 1e6 + rng.normal(size=(len(x), 2))
        y = reference_design[:, 1] - reference_design[:, 2]
        if case == "noisy_cancellation":
            y += rng.normal(scale=0.1, size=len(x))
    units = {"rescaled": [1e-12, 1e12], "extreme_units": [1e-200, 1e200]}.get(
        case, [1, 1]
    )
    predictors = dict(zip(["x", "z"], (reference_design[:, 1:] * units).T, strict=True))
    engine = GroupedOLS(y, predictors, groups, n_folds=3, sigma_correction=sigma)
    multiplicities = np.array([[1, 1, 1, 1], [2, 0, 1, 1], [0, 2, 2, 0]])
    scores, _ = engine.scores(("x", "z"), multiplicities)
    for b, w in enumerate(multiplicities):
        rows = np.repeat(np.arange(len(y)), w[engine.group_codes])
        total_ll = 0
        for fold in range(engine.n_folds):
            test = rows[engine.row_folds[rows] == fold]
            train = rows[engine.row_folds[rows] != fold]
            if not len(test):
                continue
            assert not set(groups[train]) & set(groups[test])
            fit = sm.OLS(y[train], reference_design[train]).fit()
            augmented_rank = np.linalg.matrix_rank(
                np.column_stack([reference_design[train], y[train]])
            )
            if fit.df_resid <= 0 or augmented_rank == fit.model.rank:
                total_ll = np.nan
                break
            denom = len(train) if sigma == "mle" else fit.df_resid
            variance = np.sum(fit.resid**2) / denom
            total_ll += np.sum(
                -0.5
                * (
                    np.log(2 * np.pi * variance)
                    + (y[test] - fit.predict(reference_design[test])) ** 2 / variance
                )
            )
        np.testing.assert_allclose(scores[b], total_ll / len(rows), atol=1e-10)


def test_residual_scores_survive_large_response_offset():
    rng = np.random.default_rng(7)
    groups = np.repeat(np.arange(100).astype(str), 40)
    x = rng.integers(-10, 10, size=len(groups))
    y = 2 * x + rng.choice([-0.125, 0.125], size=len(groups))
    weights = np.ones((1, 100))
    centered = GroupedOLS(y, {"x": x}, groups).scores(("x",), weights)
    shifted = GroupedOLS(y + 2**40, {"x": x}, groups).scores(("x",), weights)
    np.testing.assert_array_equal(shifted[0], centered[0])
    np.testing.assert_array_equal(shifted[1], centered[1])


def test_empty_training_resamples_are_invalid_and_empty_test_folds_contribute_zero():
    x = np.arange(12.0)
    engine = GroupedOLS(np.sin(x), {"x": x}, np.repeat(["a", "b", "c"], 4))
    scores, folds = engine.scores(("x",), np.array([[0, 0, 0], [2, 0, 0], [1, 1, 0]]))
    assert np.isnan(scores[:2]).all()
    assert np.isnan(folds[1]).all()
    assert np.isfinite(scores[2])
    assert folds[2, engine.group_folds[2]] == 0


@pytest.mark.parametrize("argument", ["response", "predictor", "weights"])
@pytest.mark.parametrize("kind", ["masked", "complex"])
def test_material_inputs_reject_lossy_array_conversion(argument, kind):
    values = {
        "response": np.arange(8.0),
        "predictor": np.arange(8.0),
        "weights": np.ones((1, 2)),
    }
    value = values[argument]
    values[argument] = np.ma.array(value, mask=True) if kind == "masked" else value + 1j
    with pytest.raises(ValueError, match="unmasked real"):
        engine = GroupedOLS(
            values["response"], {"x": values["predictor"]}, ["a"] * 4 + ["b"] * 4
        )
        engine.scores(("x",), values["weights"])


def test_fold_compression_preserves_residuals_across_multiple_blocks():
    rng = np.random.default_rng(12)
    groups = np.repeat(np.arange(800).astype(str), 5)
    x = rng.normal(size=len(groups))
    y = 2 + x + rng.normal(size=len(groups))
    engine = GroupedOLS(y, {"x": x}, groups, n_folds=2)
    weights = rng.integers(0, 4, size=(1, engine.n_groups))
    actual, _ = engine.scores(("x",), weights)
    row_weights = weights[0, engine.group_codes]
    expected = 0.0
    for fold in range(2):
        test = engine.row_folds == fold
        design = sm.add_constant(x)
        fit = sm.WLS(y[~test], design[~test], weights=row_weights[~test]).fit()
        variance = np.average(fit.resid**2, weights=row_weights[~test])
        residuals = y[test] - fit.predict(design[test])
        expected += np.sum(
            row_weights[test]
            * -0.5
            * (np.log(2 * np.pi * variance) + residuals**2 / variance)
        )
    assert actual[0] == pytest.approx(expected / row_weights.sum(), abs=1e-12)


def test_correction_retains_finite_fold_resolution_and_resamples_training():
    rng = np.random.default_rng(12)
    groups = np.repeat(np.arange(10).astype(str), 15)
    x = rng.normal(size=len(groups))
    y = 20 * x + rng.normal(size=len(groups))
    engine = GroupedOLS(y, {"x": x}, groups)
    contrasts = [Contrast("addition", (), ("x",)), Contrast("identity", ("x",), ("x",))]
    rows, draws = engine.compare(contrasts, n_resamples=99, seed=3)
    assert rows[0]["p_fold_exact"] == 2 / 1024
    assert rows[0]["pvalue"] >= rows[0]["p_fold_exact"]
    assert rows[0]["pvalue"] >= rows[0]["p_bootstrap"]
    assert rows[1]["pvalue"] == 1
    assert np.all(draws[:, 1] == 0)
    assert np.std(draws[:, 0]) > 0
    repeated, again = engine.compare(contrasts, n_resamples=99, seed=3, batch_size=1)
    np.testing.assert_allclose(draws, again, rtol=1e-9, atol=1e-10)
    assert rows[0]["pvalue"] == repeated[0]["pvalue"]


def test_too_few_materials_do_not_manufacture_ten_folds():
    groups = np.repeat(["a", "b", "c", "d"], 20)
    rng = np.random.default_rng(0)
    x = rng.normal(size=len(groups))
    y = x + rng.normal(size=len(groups))
    engine = GroupedOLS(y, {"x": x}, groups)
    rows, _ = engine.compare([Contrast("x", (), ("x",))], n_resamples=49)
    assert rows[0]["n_folds"] == 4
    assert rows[0]["fold_p_floor"] == 0.125
    assert rows[0]["pvalue"] >= 0.125
