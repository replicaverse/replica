"""Refitting, resampling, restart integrity and default predictive inference."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import norm, t

from replica import CrossedBootstrapConfig, PredictiveTestResult, predictive_test
from replica.analysis import (
    AnalysisConfig,
    CrossedGroupSplitter,
    Design,
    Factor,
    RandomEffect,
    Residualize,
)
from replica.analysis.contracts.grouped import GroupedFit
from replica.analysis.data.observations import ObservationTable
from replica.analysis.inference.crossed_bootstrap import (
    _context,
    _cycle,
    _partitions,
    _sample,
    crossed_bootstrap_test,
)
from replica.analysis.inference.crossed_statistics import SERIES, _summarize
from replica.analysis.specification.evaluation import ComparisonSpec, EvaluationSpec
from replica.feature import ModelFeature, Surprisal, UnitLength, named
from tests.analysis._fixtures import ConstantLM, make_toy_dataset


class GaussianBackend:
    """Fit actual OLS predictions while auditing the named-frame contract."""

    calls = []
    fail = False

    def validate_design(self, design):
        pass

    def fit_grouped(self, y, train, test):
        if self.fail:
            raise ValueError("deliberate fit failure")
        for name in ("reader", "source", "item"):
            if name in train.factors:
                assert not set(train.factors[name]) & set(test.factors[name])
        self.calls.append((y.copy(), train, test))
        x = np.column_stack([np.ones(len(y)), *(train.columns[n] for n in train.fixed)])
        xt = np.column_stack(
            [
                np.ones(len(next(iter(test.factors.values())))),
                *(test.columns[n] for n in train.fixed),
            ]
        )
        beta = np.linalg.lstsq(x, y, rcond=None)[0]
        sigma = max(float(np.sqrt(np.mean((y - x @ beta) ** 2))), 1e-6)
        return GroupedFit(
            xt @ beta,
            sigma,
            beta,
            np.zeros_like(beta),
            predictive_variance=np.full(len(xt), sigma**2 + 0.2),
        )


def fixture():
    rng = np.random.default_rng(19)
    r, s, w = np.indices((8, 7, 4)).reshape(3, -1)
    x = rng.normal(size=28)[s * 4 + w]
    c = rng.normal(size=len(x))
    y = 0.4 * x + 0.3 * c + rng.normal(size=len(x))
    # Unequal intersections exercise observation weighting and missingness.
    keep = ~((r == 0) & (s < 2)) & ~((w == 3) & (s % 2 == 0))
    arrays = [v[keep] for v in (r, s, w, x, c, y)]
    r, s, w, x, c, y = arrays
    factors = {
        "reader": r.astype(str),
        "source": s.astype(str),
        "item": (s * 4 + w).astype(str),
    }
    data = ObservationTable(y, {"x": x, "c": c}, r, s * 4 + w, s, None, factors=factors)
    design = Design(
        factors=(
            Factor("reader", "participant"),
            Factor("source", "stimulus"),
            Factor("item", "unit:item", role="model"),
        ),
        random_effects=(
            RandomEffect("reader", ("x",), correlated=False),
            RandomEffect("item"),
        ),
        cv_unit=("reader", "source"),
        comparison="predictor",
    )
    config = AnalysisConfig(
        model=GaussianBackend(), design=design, cv=CrossedGroupSplitter((2, 3))
    )
    return data, ComparisonSpec(("c",), ("x",)), config


SMALL = CrossedBootstrapConfig(n_resamples=3, cycles_per_draw=2, original_cycles=3)


def test_target_slot_preserves_arbitrary_input_names():
    data, comparison, config = fixture()
    key = ("identity", 0, 0)
    expected = _cycle(_context(data, comparison, config, SMALL, None), key)
    renamed = replace(
        data,
        columns={"x": data.columns["x"], "__target__": data.columns["c"]},
    )
    actual = _cycle(
        _context(renamed, ComparisonSpec(("__target__",), ("x",)), config, SMALL, None),
        key,
    )
    assert expected["error"] is None and actual["error"] is None
    assert actual["gain"] == pytest.approx(expected["gain"], rel=0, abs=1e-12)

    before = len(GaussianBackend.calls)
    with pytest.raises(ValueError, match="Target name conflicts"):
        _context(
            replace(data, response={"target": "c"}), comparison, config, SMALL, None
        )
    assert len(GaussianBackend.calls) == before


@pytest.mark.parametrize("field", ["y", "column", "groups", "factor"])
def test_prepared_observations_reject_active_masks(field):
    data, _, _ = fixture()
    values = {
        "y": data.y,
        "column": data.columns["x"],
        "groups": data.groups,
        "factor": data.factors["reader"],
    }[field]
    masked = np.ma.array(values, mask=False)
    masked.mask[0] = True
    if field == "column":
        change = {"columns": {**data.columns, "x": masked}}
    elif field == "factor":
        change = {"factors": {**data.factors, "reader": masked}}
    else:
        change = {field: masked}
    with pytest.raises(ValueError, match="masked"):
        replace(data, **change)
    masked.mask[:] = False
    replace(data, **change)


def test_defaults_and_explicit_budget_validation():
    default = CrossedBootstrapConfig()
    assert (default.n_resamples, default.cycles_per_draw, default.original_cycles) == (
        500,
        2,
        8,
    )
    assert default.fit_budget(100) == 601600
    for kw in (
        {"n_resamples": True},
        {"cycles_per_draw": 1},
        {"original_cycles": 1},
        {"n_jobs": 0},
        {"seed": -1},
        {"confidence_level": 1},
    ):
        with pytest.raises(ValueError):
            CrossedBootstrapConfig(**kw)


def test_bootstrap_multiplicity_and_random_group_copies():
    data, comparison, config = fixture()
    ctx = _context(data, comparison, config, SMALL, None)
    rows, factors, copies = _sample(ctx, "joint", 0)
    rng = np.random.default_rng(np.random.SeedSequence([SMALL.seed, 0, 0]))
    readers = rng.integers(8, size=8)
    sources = rng.integers(7, size=7)
    rcount = np.bincount(readers, minlength=8)
    scount = np.bincount(sources, minlength=7)
    expected = rcount[data.participants] * scount[data.groups]
    np.testing.assert_array_equal(np.bincount(rows, minlength=len(data.y)), expected)
    for name, copy_axis in [("reader", 0), ("item", 1)]:
        original = data.factors[name][rows]
        for level in np.unique(original):
            chosen = original == level
            assert len(np.unique(factors[name][chosen])) == len(
                np.unique(copies[copy_axis][chosen])
            )
    # Missing intersections remain missing; all numeric predictors are copied unchanged.
    assert len(rows) == expected.sum()
    assignments = _partitions(ctx, "joint", 0, 0)
    for code in np.unique(rows):
        assert all(len(set(a[rows][rows == code])) == 1 for a in assignments)
    assert np.unique(assignments[0]).size == 2
    assert np.unique(assignments[1]).size == 3


def test_fixed_original_pool_keeps_absent_slots_and_fresh_cycles():
    data, comparison, config = fixture()
    ctx = _context(
        data, comparison, replace(config, cv=CrossedGroupSplitter((8, 7))), SMALL, None
    )
    rows, _, _ = _sample(ctx, "joint", 0)
    assert np.unique(data.participants[rows]).size < 8
    assert ctx.folds == (8, 7)
    first = _partitions(ctx, "joint", 0, 0)
    second = _partitions(ctx, "joint", 0, 1)
    assert any(not np.array_equal(a, b) for a, b in zip(first, second, strict=True))
    assert len(np.unique(first[0])) == 8


@pytest.mark.parametrize("residualize", [False, True])
def test_full_refitting_training_scaling_and_weighted_aggregation(residualize):
    data, comparison, config = fixture()
    if residualize:
        config = replace(
            config,
            preprocessing=(
                Residualize(
                    named(UnitLength(), "x"),
                    against=[named(UnitLength(), "c")],
                    policy="centered_slope",
                ),
            ),
        )
    GaussianBackend.calls.clear()
    ctx = _context(data, comparison, config, SMALL, None)
    result = _cycle(ctx, ("identity", 0, 0))
    assert result["error"] is None
    assert result["n_scored"] == len(data.y)
    assert len(GaussianBackend.calls) == 12
    assignments = _partitions(ctx, "identity", 0, 0)
    weighted = []
    unweighted = []
    call = 0
    for i in range(2):
        for j in range(3):
            train = (assignments[0] != i) & (assignments[1] != j)
            test = (assignments[0] == i) & (assignments[1] == j)
            columns = dict(data.columns)
            if residualize:
                reference = data.columns["c"]
                # Independent OLS oracle: use its slope, retaining the target mean.
                slope = np.linalg.lstsq(
                    np.column_stack([np.ones(train.sum()), reference[train]]),
                    data.columns["x"][train],
                    rcond=None,
                )[0][1]
                columns["x"] = data.columns["x"] - slope * reference
            scores = []
            for fixed in [("c",), ("c", "x")]:
                y, frame, held = GaussianBackend.calls[call]
                call += 1
                assert frame.fixed == fixed
                if fixed == ("c",):
                    assert frame.design.random_effects[0].slopes == ()
                for name in fixed:
                    np.testing.assert_allclose(
                        frame.columns[name].mean(), 0, atol=1e-12
                    )
                    expected = (
                        columns[name][test] - columns[name][train].mean()
                    ) / columns[name][train].std()
                    np.testing.assert_allclose(held.columns[name], expected)
                x = np.column_stack(
                    [np.ones(len(y)), *(frame.columns[n] for n in fixed)]
                )
                xt = np.column_stack(
                    [np.ones(test.sum()), *(held.columns[n] for n in fixed)]
                )
                beta = np.linalg.lstsq(x, y, rcond=None)[0]
                sigma2 = np.mean((y - x @ beta) ** 2)
                scores.append(
                    norm.logpdf(data.y[test], xt @ beta, np.sqrt(sigma2 + 0.2))
                )
            gains = scores[1] - scores[0]
            weighted.extend(gains)
            unweighted.append(gains.mean())
    assert result["gain"] == pytest.approx(np.mean(weighted))
    assert not np.isclose(result["gain"], np.mean(unweighted))


def test_all_cycles_refit_and_checkpoint_resume(tmp_path):
    data, comparison, config = fixture()
    GaussianBackend.calls.clear()
    result = crossed_bootstrap_test(
        data, comparison, config, inference=SMALL, checkpoint_dir=tmp_path
    )
    expected_cycles = 3 + 3 * 3 * 2
    assert len(result.metadata["cycles"]) == expected_cycles
    assert result.metadata["fitted_models"] == len(GaussianBackend.calls)
    count = len(GaussianBackend.calls)
    again = crossed_bootstrap_test(
        data, comparison, config, inference=SMALL, checkpoint_dir=tmp_path
    )
    assert again == result and len(GaussianBackend.calls) == count
    with pytest.raises(ValueError, match="different inputs"):
        crossed_bootstrap_test(
            replace(data, y=data.y + 1),
            comparison,
            config,
            inference=SMALL,
            checkpoint_dir=tmp_path,
        )
    file = tmp_path / "joint-000000-000.json"
    saved = json.loads(file.read_text())
    saved["record"]["gain"] = 100
    file.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="Corrupt checkpoint"):
        crossed_bootstrap_test(
            data, comparison, config, inference=SMALL, checkpoint_dir=tmp_path
        )


def test_failures_are_unavailable_and_not_redrawn(tmp_path):
    data, comparison, config = fixture()
    model = GaussianBackend()
    model.fail = True
    result = crossed_bootstrap_test(
        data,
        comparison,
        replace(config, model=model),
        inference=SMALL,
        checkpoint_dir=tmp_path,
    )
    assert not result.available and result.pvalue is None and result.gain is None
    assert len(result.metadata["failures"]) == 21
    assert (
        len({(r["series"], r["draw"], r["cycle"]) for r in result.metadata["cycles"]})
        == 21
    )


def test_summary_matches_frozen_assessment_fixtures():
    path = Path(__file__).parent / "fixtures" / "crossed_bootstrap_arithmetic.json"
    fixture = json.loads(path.read_text())
    for case in fixture["cases"]:
        result = _summarize(case["original"], case["samples"], case["groups"], 0.95)
        expected = case["expected"]
        assert result.available == expected["available"]
        assert result.gain == pytest.approx(expected["gain"])
        if result.available:
            assert result.pvalue == pytest.approx(expected["p"], rel=1e-12)
            assert result.standard_error == pytest.approx(expected["se"], rel=1e-12)
            assert result.confidence_interval == pytest.approx(
                (expected["low"], expected["high"])
            )
            assert result.degrees_of_freedom == expected["df"]


def test_summary_partition_correction_axis_symmetry_and_scale():
    rng = np.random.default_rng(971)
    original = rng.normal(0.03, 0.003, 8)
    samples = {
        s: rng.normal(0.03, 0.06, (40, 1)) + rng.normal(0, 0.01, (40, 3))
        for s in SERIES
    }
    result = _summarize(original, samples, (60, 12), 0.95)
    assert result.available
    for name, values in samples.items():
        corrected = (
            3 * values.mean(axis=1).var(ddof=1) - values.var(axis=0, ddof=1).mean()
        ) / 2
        # The covariance implementation is an unbiased finite-sample correction.
        assert result.variance_components[name] == pytest.approx(corrected, abs=1e-14)
    swapped = _summarize(
        original,
        {
            "axis_0": samples["axis_1"],
            "axis_1": samples["axis_0"],
            "joint": samples["joint"],
        },
        (12, 60),
        0.95,
    )
    scaled = _summarize(
        -10 * original, {k: -10 * v for k, v in samples.items()}, (60, 12), 0.95
    )
    assert result.pvalue == pytest.approx(swapped.pvalue)
    assert result.pvalue == pytest.approx(scaled.pvalue)
    assert result.standard_error * 10 == pytest.approx(scaled.standard_error)
    assert result.pvalue == pytest.approx(
        2 * t.sf(abs(result.gain) / result.standard_error, 7)
    )
    assert result.degrees_of_freedom == 7
    with pytest.raises(ValueError, match="Failed or missing"):
        _summarize([1, np.nan], samples, (60, 12), 0.95)


def test_public_entry_rejects_wrong_target_before_extraction():
    dataset = make_toy_dataset()
    lm = ConstantLM()
    features = EvaluationSpec(
        targets=[dataset.feature(dataset.default_measure)],
        predictors=[ModelFeature(lm, Surprisal())],
    )
    config = AnalysisConfig(
        model=GaussianBackend(),
        design=Design(comparison="predictor", cv_unit="stimulus"),
    )
    with pytest.raises(ValueError, match="two unseen"):
        predictive_test(dataset, features, config, inference=SMALL)
    assert lm.call_count == 0


def test_public_entry_and_recreated_graph_resume(tmp_path):
    lm = ConstantLM()
    data = make_toy_dataset(n_stimuli=6, n_participants=6, measures=["spr_rt"])
    config = AnalysisConfig(
        model=GaussianBackend(), design=Design(comparison="predictor"), n_folds=(2, 3)
    )
    results = predictive_test(
        data,
        EvaluationSpec(
            targets=[data.feature(data.default_measure)],
            predictors=[ModelFeature(lm, Surprisal())],
        ),
        config,
        inference=SMALL,
        checkpoint_dir=tmp_path,
    )
    assert len(results) == 1 and isinstance(results[0], PredictiveTestResult)
    assert results[0].metadata["dv_name"] == "spr_rt"
    assert lm.call_count == 6
    assert results[0].metadata["inference"]["n_resamples"] == 3

    fitted = len(GaussianBackend.calls)
    rebuilt = make_toy_dataset(n_stimuli=6, n_participants=6, measures=["spr_rt"])
    other_model = ConstantLM()
    resumed = predictive_test(
        rebuilt,
        EvaluationSpec(
            targets=[rebuilt.feature("spr_rt")],
            predictors=[ModelFeature(other_model, Surprisal())],
        ),
        replace(config, model=GaussianBackend()),
        inference=SMALL,
        checkpoint_dir=tmp_path,
    )
    assert resumed == results
    assert len(GaussianBackend.calls) == fitted
    assert other_model.call_count == 6


def test_parallel_execution_is_numerically_identical():
    data, comparison, config = fixture()
    serial = crossed_bootstrap_test(data, comparison, config, inference=SMALL)
    parallel = crossed_bootstrap_test(
        data, comparison, config, inference=replace(SMALL, n_jobs=2)
    )
    assert serial.gain == parallel.gain
    assert serial.pvalue == parallel.pvalue
    assert serial.metadata["cycles"] == parallel.metadata["cycles"]


def test_unsupported_random_group_requires_a_copy_policy():
    data, comparison, config = fixture()
    # A repeated word type can cross both axes, unlike a source word occurrence.
    factors = {**data.factors, "item": (data.items % 4).astype(str)}
    with pytest.raises(ValueError, match="not nested"):
        crossed_bootstrap_test(
            replace(data, factors=factors), comparison, config, inference=SMALL
        )


def test_unresolved_convergence_and_rank_loss_make_inference_unavailable():
    from replica.analysis.contracts.grouped import validate_grouped_fit

    fit = GroupedFit(np.zeros(2), 1, np.ones(2), np.ones(2))
    with pytest.raises(ValueError, match="coefficients must follow"):
        validate_grouped_fit(fit, 2, n_fixed=2)
    for bad in (
        replace(fit, beta=np.array([1, np.nan])),
        replace(fit, metadata={"optimizer_convergence": [1]}),
        replace(fit, metadata={"convergence_messages": ["failed to converge"]}),
    ):
        with pytest.raises(ValueError):
            validate_grouped_fit(bad, 2, 1)
    validate_grouped_fit(
        replace(
            fit,
            metadata={
                "singular": True,
                "convergence_messages": ["boundary (singular) fit"],
            },
        ),
        2,
        1,
    )


@pytest.mark.needs_r
def test_lmer_reporting_cache_does_not_change_checkpoint_identity(tmp_path):
    from replica.analysis import RLmerModel
    from replica.analysis.inference.crossed_bootstrap import _checkpoint

    data, comparison, config = fixture()
    backend = RLmerModel(reml=False)
    config = replace(config, model=backend)
    context = _context(data, comparison, config, SMALL, None)
    _, manifest = _checkpoint(context, tmp_path)
    backend._last_fixed = (np.array([1.0, 2.0]), np.array([0.1, 0.2]))
    restarted = _context(data, comparison, config, SMALL, None)
    assert restarted.model._last_fixed is None
    assert backend._last_fixed is not None
    assert _checkpoint(restarted, tmp_path)[1] == manifest
    changed = replace(config, model=RLmerModel(optimizer="nlminb"))
    with pytest.raises(ValueError, match="different inputs"):
        _checkpoint(_context(data, comparison, changed, SMALL, None), tmp_path)


@pytest.mark.needs_r
def test_refitted_resample_scores_match_independent_r_formulas():
    from replica.analysis import RLmerModel
    from replica.analysis.models.support.r import _r_context

    _, comparison, config = fixture()
    rng = np.random.default_rng(5138)
    reader, source, word = np.indices((24, 12, 8)).reshape(3, -1)
    item = source * 8 + word
    x = rng.normal(size=96)[item]
    c = rng.normal(size=96)[item]
    y = (
        0.7 * x
        + 0.3 * c
        + rng.normal(0, 0.8, 24)[reader]
        + rng.normal(0, 0.4, 24)[reader] * x
        + rng.normal(0, 0.6, 96)[item]
        + rng.normal(0, 0.8, len(x))
    )
    data = ObservationTable(
        y,
        {"x": x, "c": c},
        reader,
        item,
        source,
        None,
        factors={
            "reader": reader.astype(str),
            "source": source.astype(str),
            "item": item.astype(str),
        },
    )
    config = replace(
        config, model=RLmerModel(nlminb_fallback=False), cv=CrossedGroupSplitter((2, 2))
    )
    ctx = _context(data, comparison, config, SMALL, None)
    key = ("joint", 1, 0)
    actual = _cycle(ctx, key)
    assert actual["error"] is None
    rows, factors, _ = _sample(ctx, *key[:2])
    assignments = [a[rows] for a in _partitions(ctx, *key)]
    totals = []
    with _r_context() as ro:
        independent = ro.r("""function(y, c, x, reader, item, training, testing) {
          d <- data.frame(y=y,c=c,x=x,reader=factor(reader),item=factor(item))
          tr <- d[training,]; te <- d[testing,]
          for (v in c('c','x')) {
            center <- mean(tr[[v]])
            scale <- sqrt(mean((tr[[v]]-center)^2)); if (scale==0) scale <- 1
            tr[[v]] <- (tr[[v]]-center)/scale; te[[v]] <- (te[[v]]-center)/scale
          }
          control <- lme4::lmerControl(optimizer='bobyqa', optCtrl=list(maxfun=50000))
          baseline <- lme4::lmer(y~c+(1|reader)+(1|item),data=tr,REML=FALSE,control=control)
          full <- lme4::lmer(y~c+x+(1|reader)+(0+x|reader)+(1|item),data=tr,REML=FALSE,control=control)
          score <- function(fit) {
            mu <- predict(fit,newdata=te,re.form=NA,allow.new.levels=TRUE)
            v <- rep(sigma(fit)^2,nrow(te))
            for (G in lme4::VarCorr(fit)) {
              z <- if(colnames(G)[1]=='x') te$x else rep(1,nrow(te))
              v <- v + as.numeric(G[1,1])*z^2
            }
            dnorm(te$y,mu,sqrt(v),log=TRUE)
          }
          score(full)-score(baseline)
        }""")
        for i in range(2):
            for j in range(2):
                train = (assignments[0] != i) & (assignments[1] != j)
                test = (assignments[0] == i) & (assignments[1] == j)
                if not test.any():
                    continue
                gains = independent(
                    ro.FloatVector(data.y[rows]),
                    ro.FloatVector(data.columns["c"][rows]),
                    ro.FloatVector(data.columns["x"][rows]),
                    ro.StrVector(factors["reader"]),
                    ro.StrVector(factors["item"]),
                    ro.BoolVector(train),
                    ro.BoolVector(test),
                )
                totals.extend(np.asarray(gains, float))
    assert len(totals) == actual["n_scored"]
    # Allow optimizer-level numerical variation between independent fits.
    assert actual["gain"] == pytest.approx(np.mean(totals), rel=0, abs=1e-7)

    # Spawned workers initialize independent R sessions and reproduce the same fit.
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context

    from replica.analysis.inference.crossed_bootstrap import _initialize_worker, _worker

    with ProcessPoolExecutor(
        max_workers=2,
        mp_context=get_context("spawn"),
        initializer=_initialize_worker,
        initargs=(ctx,),
    ) as executor:
        independent_workers = list(executor.map(_worker, [key, key]))
    assert all(r == actual for r in independent_workers)


def test_public_default_selects_bootstrap_with_500_draws(monkeypatch):
    dataset = make_toy_dataset(n_participants=3)
    import importlib

    module = importlib.import_module("replica.analysis.inference.predictive")
    selected = []

    def capture(observations, comparison, analysis, **kwargs):
        selected.append(kwargs["inference"])
        return PredictiveTestResult(0.01, False, "dispatch check")

    monkeypatch.setattr(module, "crossed_bootstrap_test", capture)
    lm = ConstantLM()
    config = AnalysisConfig(
        model=GaussianBackend(), design=Design(comparison="predictor")
    )
    result = predictive_test(
        dataset,
        EvaluationSpec(
            targets=[dataset.feature(dataset.default_measure)],
            predictors=[ModelFeature(lm, Surprisal())],
        ),
        config,
    )
    assert selected == [CrossedBootstrapConfig()]
    assert selected[0].n_resamples == 500
    assert result[0].gain == 0.01


def test_bootstrap_failure_preserves_complete_original_estimate(monkeypatch):
    import importlib

    module = importlib.import_module("replica.analysis.inference.crossed_bootstrap")
    original_cycle = module._cycle

    def fail_one(ctx, key):
        record = original_cycle(ctx, key)
        if key == ("joint", 0, 0):
            record.update(gain=None, error="deliberate failure")
        return record

    monkeypatch.setattr(module, "_cycle", fail_one)
    data, comparison, config = fixture()
    result = crossed_bootstrap_test(data, comparison, config, inference=SMALL)
    assert result.gain is not None and not result.available
    assert result.pvalue is None and result.confidence_interval is None
    assert len(result.metadata["failures"]) == 1
