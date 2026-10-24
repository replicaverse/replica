"""Held-out group predictions against independent mgcv coefficient sums."""

from importlib.resources import files

import numpy as np
import pytest

from replica.analysis.models import RGamModel
from replica.analysis.models.support.r import _r_context
from replica.utils.r import RScriptError

pytestmark = pytest.mark.needs_r


@pytest.fixture(params=[False, True], ids=["bam", "discrete_bam"])
def backend(request):
    """Both supported bam fitting modes use the same prediction contract."""
    return {"use_bam": True, "discrete": request.param, "method": "fREML"}


@pytest.fixture
def curve_data():
    """Crossed curves with a nonzero offset and all four test group patterns."""
    rng = np.random.default_rng(426)
    subject, item, _ = np.indices((6, 8, 6)).reshape(3, -1)
    x, z, offset = rng.uniform(-2, 2, (3, len(subject)))
    y = (
        4
        + np.sin(x)
        - z
        + offset
        + rng.normal(0, 3, 6)[subject]
        + rng.normal(0, 2, 6)[subject] * x**2
        + rng.normal(0, 2, 8)[item]
        + rng.normal(0, 1, (6, 8))[subject, item]
        + rng.normal(0, 0.2, len(subject))
    )
    train = np.column_stack([x, z, offset, subject, item])
    test = np.tile(train[[0, 50, 150, 280]], (4, 1))
    test[4:8, -2] = 99
    test[8:12, -1] = 99
    test[12:, -2:] = 99
    return y, train, test


def _reference(y, train, test, formula, backend, smooth_groups, *, family="gaussian()"):
    """Zero specified coefficient blocks in an independent, training-only fit.

    Each smooth's dependencies are declared by the test: 0 for population,
    1 for subject, 2 for item, and 3 for both. No production classifier or
    prediction helper participates in this reference.
    """
    unseen = (~np.isin(test[:, -2], train[:, -2])).astype(int)
    unseen += 2 * ~np.isin(test[:, -1], train[:, -1])
    with _r_context() as ro:
        columns = ["x0", "x1", "x2", "participant_id", "item_id"]
        for name, X in (("train", train), ("test", test)):
            data = {col: ro.FloatVector(X[:, i]) for i, col in enumerate(columns)}
            if name == "train":
                data["y"] = ro.FloatVector(y)
            ro.globalenv[f".shrink_{name}"] = ro.r["data.frame"](**data)
        ro.globalenv[".shrink_formula"] = formula
        ro.globalenv[".shrink_groups"] = ro.IntVector(smooth_groups)
        ro.globalenv[".shrink_unseen"] = ro.IntVector(unseen)
        ro.r(
            "for (col in c('participant_id', 'item_id')) {"
            ".shrink_train[[col]] <- factor(.shrink_train[[col]])}; "
            ".shrink_safe <- .shrink_test; "
            "for (col in c('participant_id', 'item_id')) {"
            ".shrink_safe[[col]] <- factor(.shrink_safe[[col]], "
            "levels=levels(.shrink_train[[col]])); "
            ".shrink_safe[[col]][is.na(.shrink_safe[[col]])] <- "
            "levels(.shrink_train[[col]])[1]}"
        )
        ro.r(
            "suppressPackageStartupMessages(library(mgcv)); "
            ".shrink_model <- bam(as.formula(.shrink_formula), "
            f"data=.shrink_train, method='fREML', family={family}, "
            f"discrete={'TRUE' if backend['discrete'] else 'FALSE'})"
        )
        expected = np.asarray(
            ro.r(
                "local({"
                "m <- .shrink_model; "
                "stopifnot(length(m$smooth) == length(.shrink_groups)); "
                "X <- predict(m, .shrink_safe, type='lpmatrix', discrete=FALSE); "
                "for (i in seq_along(m$smooth)) {"
                "s <- m$smooth[[i]]; "
                "rows <- bitwAnd(.shrink_unseen, .shrink_groups[i]) > 0; "
                "X[rows, s$first.para:s$last.para] <- 0}; "
                "eta <- drop(X %*% coef(m)) + .shrink_test$x2; "
                "m$family$linkinv(eta)})"
            ),
            dtype=float,
        )
        sigma = float(ro.r("sqrt(.shrink_model$sig2)")[0])
    return expected, sigma


@pytest.mark.parametrize(
    "terms, groups",
    [
        pytest.param(
            "s(participant_id, bs='re') + s(x0, participant_id, bs='re')",
            [1, 1],
            id="random_intercept_and_slope",
        ),
        pytest.param("s(x0, participant_id, bs='fs', m=1, k=4)", [1], id="fs"),
        pytest.param("s(x0, by=participant_id, k=4)", [1] * 6, id="factor_by"),
        pytest.param(
            "t2(x1, participant_id, bs=c('tp', 're'), k=c(4, 6))",
            [1],
            id="tensor",
        ),
        pytest.param(
            "s(x0, participant_id, bs='fs', m=1, k=4) + "
            "s(participant_id, item_id, bs='re')",
            [1, 3],
            id="crossed_interaction",
        ),
        pytest.param(
            "s(participant_id, by=item_id, bs='re')", [3] * 8, id="crossed_by"
        ),
        pytest.param(
            "s(x0, participant_id, bs='fs', m=1, k=4) + "
            "s(x1, item_id, bs='fs', m=1, k=4)",
            [1, 2],
            id="both_group_curves",
        ),
        pytest.param(
            "s(x0, participant_id, bs='fs', m=1, k=4) + s(x1, bs='re')",
            [1, 0],
            id="retain_ungrouped_random_effect",
        ),
    ],
)
def test_shrink_unseen_matches_coefficient_reference(
    backend, curve_data, terms, groups
):
    """Drop only coefficients depending on an unseen group in each test row."""
    y, train, test = curve_data
    formula = f"y ~ s(x0, k=4) + x1 + offset(x2) + {terms} + s(item_id, bs='re')"
    expected, expected_sigma = _reference(
        y, train, test, formula, backend, [0, *groups, 2]
    )
    model = RGamModel(formula=formula, re_predict="shrink_unseen", **backend)
    actual, sigma = model.fit_predict_with_sigma(y, train, test)
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-7)
    assert sigma == pytest.approx(expected_sigma, rel=1e-7)

    # Ordinary conditional prediction is authoritative when every ID is known.
    conditional = RGamModel(formula=formula, re_predict="conditional", **backend)
    known = conditional.fit_predict(y, train, test[:4])
    np.testing.assert_allclose(actual[:4], known, rtol=1e-7, atol=1e-7)


@pytest.mark.parametrize("group", ["participant_id", "item_id"])
def test_shrink_unseen_preserves_and_validates_fixed_groups(backend, curve_data, group):
    """Known fixed group effects survive; unseen fixed levels have no estimate."""
    y, train, test = curve_data
    if group == "item_id":
        train, test = train[:, [0, 1, 2, 4, 3]], test[:, [0, 1, 2, 4, 3]]
    formula = (
        f"y ~ s(x0, k=4) + x1 + offset(x2) + {group} + "
        f"s(x0, {group}, bs='fs', m=1, k=4)"
    )
    groups = [0, 1 if group == "participant_id" else 2]
    known = test[np.isin(test[:, -2 if group == "participant_id" else -1], range(6))]
    expected, _ = _reference(y, train, known, formula, backend, groups)
    model = RGamModel(formula=formula, re_predict="shrink_unseen", **backend)
    np.testing.assert_allclose(model.fit_predict(y, train, known), expected, atol=1e-7)
    with pytest.raises(RScriptError, match=f"shrink_unseen.*unseen {group} levels"):
        model.fit_predict(y, train, test)


def test_shrink_unseen_uses_selected_baseline_formula(backend, curve_data):
    """Grouping IDs used only in the full formula do not constrain a baseline."""
    y, train, test = curve_data
    reduced_train, reduced_test = train[:, [0, 3, 4]], test[:, [0, 3, 4]].copy()
    reduced_test[:, -2:] = 99
    model = RGamModel(
        formula="y ~ s(x0, k=4) + x1 + participant_id",
        baseline_formula="y ~ s(x0, k=4)",
        col_names={"x0": "x0", "x1": "x1"},
        re_predict="shrink_unseen",
        **backend,
    )
    actual = model.fit_predict(y, reduced_train, reduced_test)
    expected = RGamModel(formula="y ~ s(x0, k=4)", **backend).fit_predict(
        y, reduced_train[:, :1], reduced_test[:, :1]
    )
    np.testing.assert_allclose(actual, expected, atol=1e-7)


def test_shrink_unseen_without_grouping_columns(backend, curve_data):
    """A population-only model keeps every term, including numeric bs='re'."""
    y, train, test = curve_data
    formula = "y ~ s(x0, k=4) + s(x1, bs='re') + offset(x2)"
    model = RGamModel(formula=formula, re_predict="shrink_unseen", **backend)
    expected, _ = _reference(y, train, test, formula, backend, [0, 0])
    actual = model.fit_predict(y, train[:, :3], test[:, :3])
    np.testing.assert_allclose(actual, expected, atol=1e-7)


@pytest.mark.parametrize("missing", ["response", "predictor"])
def test_shrink_unseen_uses_groups_in_retained_training_rows(
    backend, curve_data, missing
):
    """A group whose observations mgcv omits has no fitted group effect."""
    y, train, test = curve_data
    omitted = (train[:, -2] == 0) | (train[:, -1] == 0)
    formula = (
        "y ~ s(x0, k=4) + x1 + offset(x2) + "
        "s(x0, participant_id, bs='fs', m=1, k=4) + "
        "s(item_id, bs='re') + s(participant_id, item_id, bs='re')"
    )
    expected, expected_sigma = _reference(
        y[~omitted], train[~omitted], test, formula, backend, [0, 1, 2, 3]
    )
    if missing == "response":
        y[omitted] = np.nan
    else:
        train[omitted, 1] = np.nan
    model = RGamModel(formula=formula, re_predict="shrink_unseen", **backend)
    actual, sigma = model.fit_predict_with_sigma(y, train, test)
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-7)
    assert sigma == pytest.approx(expected_sigma, rel=1e-7)


@pytest.mark.parametrize("group", ["participant_id", "item_id"])
def test_shrink_unseen_rejects_fixed_groups_omitted_from_fit(
    backend, curve_data, group
):
    """Raw training IDs cannot justify an unestimated fixed group coefficient."""
    y, train, test = curve_data
    y[train[:, -2 if group == "participant_id" else -1] == 0] = np.nan
    model = RGamModel(
        formula=f"y ~ s(x0, k=4) + {group}", re_predict="shrink_unseen", **backend
    )
    # These IDs all occur in the raw training frame; some have no usable rows.
    with pytest.raises(RScriptError, match=f"shrink_unseen.*unseen {group} levels"):
        model.fit_predict(y, train, test[:4])


@pytest.mark.parametrize(
    "terms, groups",
    [
        ("s(x0, factor(participant_id), bs='fs', m=1, k=4)", [0, 1]),
        ("s(x0, by=factor(participant_id), k=4)", [0, *([1] * 6)]),
    ],
    ids=["fs", "factor_by"],
)
def test_shrink_unseen_recognizes_group_expressions(curve_data, terms, groups):
    """Transformed groups retain the dependency on their original ID column."""
    y, train, test = curve_data
    backend = {"use_bam": True, "discrete": False, "method": "fREML"}
    formula = f"y ~ s(x0, k=4) + x1 + offset(x2) + {terms}"
    expected, _ = _reference(y, train, test, formula, backend, groups)
    model = RGamModel(formula=formula, re_predict="shrink_unseen", **backend)
    np.testing.assert_allclose(model.fit_predict(y, train, test), expected, atol=1e-7)


def test_shrink_unseen_is_invariant_to_batching_and_labels(backend, curve_data):
    """Row order, singleton batches and arbitrary group labels have no effect."""
    y, train, test = curve_data
    model = RGamModel(
        formula="y ~ s(x0, k=4) + x1 + offset(x2) + "
        "s(x0, participant_id, bs='fs', m=1, k=4) + s(item_id, bs='re')",
        re_predict="shrink_unseen",
        **backend,
    )
    expected = model.fit_predict(y, train, test)
    order = np.random.default_rng(0).permutation(len(test))
    np.testing.assert_allclose(
        model.fit_predict(y, train, test[order]), expected[order], atol=1e-7
    )
    for i in (0, 4, 8, 12):
        singleton = model.fit_predict(y, train, test[i : i + 1])
        assert singleton.shape == (1,)
        np.testing.assert_allclose(singleton, expected[i : i + 1], atol=1e-7)
    relabeled_train, relabeled_test = train.copy(), test.copy()
    relabeled_train[:, -2:] = 50 - train[:, -2:]
    relabeled_test[:, -2:] = 50 - test[:, -2:]
    np.testing.assert_allclose(
        model.fit_predict(y, relabeled_train, relabeled_test), expected, atol=1e-6
    )


def test_shrink_unseen_response_link_and_placeholder_invariance(backend, curve_data):
    """The helper preserves offsets and applies the response link after summing."""
    y, train, test = curve_data
    y = np.exp(y / 8)
    formula = (
        "y ~ s(x0, k=4) + x1 + offset(x2) + "
        "s(x0, participant_id, bs='fs', m=1, k=4) + s(item_id, bs='re')"
    )
    expected, _ = _reference(
        y, train, test, formula, backend, [0, 1, 2], family="Gamma(link='log')"
    )
    with _r_context() as ro:
        ro.r(
            files("replica.analysis.models.additive")
            .joinpath("gam_prediction.R")
            .read_text()
        )
        for reverse_rows in (False, True):
            if reverse_rows:
                ro.r(".shrink_train <- .shrink_train[nrow(.shrink_train):1, ]")
            actual = np.asarray(
                ro.r(
                    "replica_gam_shrink_unseen(.shrink_model, "
                    ".shrink_train, .shrink_test)"
                ),
                dtype=float,
            )
            np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-7)
