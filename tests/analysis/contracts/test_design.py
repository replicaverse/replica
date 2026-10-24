"""Named factors remain aligned through preparation, fitting and inference."""

import warnings
from dataclasses import replace

import numpy as np
import pytest

from replica.analysis import (
    OLS,
    AnalysisConfig,
    CrossedGroupSplitter,
    Design,
    Factor,
    RandomEffect,
    RLmerModel,
    evaluate,
    grouped_permutation_test,
    prepare_observations,
)
from replica.analysis.contracts.grouped import GroupedFrame
from replica.analysis.specification.evaluation import EvaluationSpec
from replica.feature import ModelFeature, Surprisal, UnitPosition
from tests.analysis._fixtures import ConstantLM, VaryingLM, make_toy_dataset


def declaration(*, comparison="fixed_effect", **kwargs):
    kwargs.setdefault("cv_unit", "stimulus")
    kwargs.setdefault("permutation_unit", "stimulus")
    kwargs.setdefault("scoring", "residual")
    return Design(
        factors=(
            Factor("participant", "participant"),
            Factor("stimulus", "stimulus"),
            Factor("word_type", "word", role="model"),
        ),
        random_effects=(
            RandomEffect("participant", ("surprisal",)),
            RandomEffect("stimulus", ("surprisal",)),
            RandomEffect("word_type"),
        ),
        condition_on=("participant",),
        comparison=comparison,
        **kwargs,
    )


def test_sampling_roles():
    with pytest.raises(ValueError, match="Model-only"):
        declaration(cv_unit="word_type")
    with pytest.raises(ValueError, match="Undeclared"):
        Design(condition_on=("language",))


def test_factor_alignment_after_masks():
    ds = make_toy_dataset(n_participants=3)
    spec = EvaluationSpec(
        targets=[ds.feature(ds.default_measure)],
        predictors=[ModelFeature(VaryingLM(), Surprisal())],
    )
    config = AnalysisConfig(
        design=declaration(), outlier_floor=300, outlier_ceiling=700
    )
    obs = prepare_observations(ds, spec, config)
    assert set(obs.factors) == {"participant", "stimulus", "word_type"}
    stimuli = {s.stimulus_id: s for s in ds.stimuli()}
    expected = [
        stimuli[s].unit_strings()[i]
        for s, i in zip(obs.stimulus_ids, obs.unit_indices, strict=True)
    ]
    np.testing.assert_array_equal(obs.factors["word_type"], expected)
    np.testing.assert_array_equal(obs.factors["stimulus"], obs.stimulus_ids)
    assert len(obs.y) < 8 * 5 * 3


def synthetic_frames(design=None):
    rng = np.random.default_rng(14)
    p, s, w = np.indices((8, 12, 6)).reshape(3, -1)
    x = rng.normal(size=len(p))
    y = (
        3
        + 0.7 * x
        + rng.normal(0, 0.8, 8)[p]
        + rng.normal(0, 0.4, 12)[s]
        + rng.normal(0, 0.3, 6)[w]
        + rng.normal(0, 0.4, len(p))
    )
    keep = s < 10
    design = design or declaration()
    factors = {
        "participant": p.astype(str),
        "stimulus": s.astype(str),
        "word_type": w.astype(str),
    }
    frames = [
        GroupedFrame(
            {"surprisal": x[m]},
            {n: v[m] for n, v in factors.items()},
            ("surprisal",),
            design,
        )
        for m in (keep, ~keep)
    ]
    return y[keep], frames


@pytest.mark.needs_r
def test_named_lmer_matches_direct_r():
    from replica.analysis.models.support.r import _r_context

    y, (train, test) = synthetic_frames()
    model = RLmerModel(nlminb_fallback=False)
    fit = model.fit_grouped(y, train, test)
    assert fit.metadata["scoring"] == "residual_gaussian"
    with _r_context() as ro:
        # Build independent reference frames from the source arrays.
        for key, frame in ((".ref_train", train), (".ref_test", test)):
            columns = {"x0": ro.FloatVector(frame.columns["surprisal"])}
            columns.update(
                {
                    f"g{i}": ro.StrVector(frame.factors[name])
                    for i, name in enumerate(("participant", "stimulus", "word_type"))
                }
            )
            if key == ".ref_train":
                columns["y"] = ro.FloatVector(y)
            ro.globalenv[key] = ro.DataFrame(columns)
        # Independent formula specifies the issue's three-factor structure.
        ro.r(
            'ref <- lme4::lmer(y ~ x0 + (1+x0|g0) + (1+x0|g1) + (1|g2), data=.ref_train, REML=FALSE, control=lme4::lmerControl(optimizer="bobyqa",optCtrl=list(maxfun=50000)))'
        )
        expected = np.asarray(
            ro.r(
                "predict(ref,newdata=.ref_test,re.form=~(1+x0|g0),allow.new.levels=TRUE)"
            )
        )
        coefficients = np.asarray(ro.r("lme4::fixef(ref)"))
        sigma = float(ro.r("sigma(ref)")[0])
    np.testing.assert_allclose(fit.predictions, expected, atol=1e-9)
    np.testing.assert_allclose(fit.beta, coefficients, atol=1e-9)
    assert fit.sigma == pytest.approx(sigma)
    assert fit.beta[1] == pytest.approx(0.7, abs=0.1)


@pytest.mark.needs_r
def test_conditioning_and_unseen_levels():
    y, (train, test) = synthetic_frames()
    model = RLmerModel(nlminb_fallback=False)
    conditional = model.fit_grouped(y, train, test)
    zero = replace(train.design, condition_on=())
    fixed = model.fit_grouped(
        y, replace(train, design=zero), replace(test, design=zero)
    )
    assert np.std(conditional.predictions - fixed.predictions) > 0.2
    strict = replace(train.design, condition_on=("stimulus",), unseen="error")
    with pytest.raises(ValueError, match="Unseen conditioned"):
        model.fit_grouped(
            y, replace(train, design=strict), replace(test, design=strict)
        )
    unseen = replace(strict, unseen="zero")
    new = model.fit_grouped(
        y, replace(train, design=unseen), replace(test, design=unseen)
    )
    np.testing.assert_allclose(new.predictions, fixed.predictions)


@pytest.mark.needs_r
def test_evaluate_three_factors():
    ds = make_toy_dataset(n_participants=5, n_words=8)
    spec = EvaluationSpec(
        targets=[ds.feature(ds.default_measure)],
        predictors=[ModelFeature(VaryingLM(), Surprisal())],
        controls=[UnitPosition()],
    )
    config = AnalysisConfig(
        model=RLmerModel(nlminb_fallback=False),
        design=declaration(),
        n_folds=2,
        per_fold=True,
        retain_per_obs_delta_ll=True,
    )
    result = evaluate(ds, spec, analysis=config)[0]
    assert np.isfinite(result.delta_ll)
    inference = grouped_permutation_test(result, method="exact")
    assert inference.n_groups == 8
    assert inference.delta_ll == pytest.approx(result.delta_ll)
    assert 0 <= inference.pvalue <= 1
    assert result.metadata["design"]["condition_on"] == ("participant",)
    for fold in result.fold_results:
        assert fold.row_ids.shape == (fold.n_test, 3)
        assert len(fold.factor_ids["word_type"]) == len(fold.per_obs_delta_ll)
        assert "(1 + x1 | g0)" in fold.metadata["baseline"]["formula"]
        assert not fold.metadata["split_warnings"]


@pytest.mark.needs_r
def test_categorical_encoding_and_unseen_rejection():
    y, (train, test) = synthetic_frames()
    design = replace(train.design, categorical=("word_type",))
    model = RLmerModel(nlminb_fallback=False)
    fit = model.fit_grouped(
        y, replace(train, design=design), replace(test, design=design)
    )
    assert fit.beta.shape == (2,)
    assert fit.metadata["categorical_levels"]["word_type"] == list("012345")
    assert "g22" in fit.metadata["fixed_coefficients"]
    unknown = dict(
        test.factors, word_type=np.full(len(test.factors["word_type"]), "unknown")
    )
    with pytest.raises(ValueError, match="Unseen fixed categorical"):
        model.fit_grouped(
            y,
            replace(train, design=design),
            replace(test, design=design, factors=unknown),
        )


def test_backend_rejection_precedes_feature_work():
    from replica.analysis import BrmsModel, RGamModel, RustLmerModel, StatsmodelsLME
    from replica.analysis.models import MarginalLMEModel, RobustCrossedLmer

    ds = make_toy_dataset()
    lm = ConstantLM()
    spec = EvaluationSpec(
        targets=[ds.feature(ds.default_measure)],
        predictors=[ModelFeature(lm, Surprisal())],
    )
    for model in (
        OLS(),
        RGamModel(),
        RustLmerModel(),
        StatsmodelsLME(),
        MarginalLMEModel(),
        RobustCrossedLmer(n_controls=0),
    ):
        with pytest.raises(ValueError, match="does not support named Design"):
            evaluate(
                ds, spec, analysis=AnalysisConfig(model=model, design=declaration())
            )
    # Brms needs explicit priors to construct: method absence is its capability declaration.
    assert not hasattr(BrmsModel, "fit_grouped")
    assert lm.call_count == 0


def test_splitter_override_warns():
    design = declaration()
    labels = {"stimulus": np.array(["a", "a", "b", "b"])}
    train = np.array([True, False, True, False])
    with pytest.warns(UserWarning, match="shares declared CV unit"):
        messages = design.check_splits([(train, ~train)], labels)
    assert len(messages) == 1


def test_metadata_factor_sources():
    ds = make_toy_dataset()
    datum = ds.data()[0]
    datum.metadata["site"] = "lab"
    datum.stimulus.metadata["language"] = "en"
    datum.stimulus.layers["words"].metadata["lemma"] = list("abcde")
    positions = np.array([1, 3])
    for source, expected in [
        ("datum:site", ["lab"] * 2),
        ("stimulus:language", ["en"] * 2),
        ("unit:lemma", ["b", "d"]),
    ]:
        np.testing.assert_array_equal(
            Factor("f", source).values(datum, "words", positions), expected
        )


def test_masked_factor_metadata_stays_missing_until_row_selection():
    from replica.analysis.contracts.labels import categorical_labels

    datum = make_toy_dataset().data()[0]
    datum.stimulus.layers["words"].metadata["group"] = np.ma.array(
        [1, 999, 2, 3, 4], mask=[False, True, False, False, False]
    )
    factor = Factor("group", "unit:group")
    labels = factor.values(datum, "words", np.array([0, 1, 2]))
    with pytest.raises(ValueError, match="Missing"):
        categorical_labels(labels)
    np.testing.assert_array_equal(
        categorical_labels(factor.values(datum, "words", np.array([0, 2]))),
        ["1", "2"],
    )


@pytest.mark.parametrize("values", [[1, 999, 2], ["a", "hidden", "b"]])
def test_masked_factor_labels_are_rejected(values):
    from replica.analysis.contracts.labels import categorical_labels

    with pytest.raises(ValueError, match="Missing labels"):
        categorical_labels(np.ma.array(values, mask=[False, True, False]))


class RecordingModel(RLmerModel):
    def __init__(self):
        super().__init__()
        self.frames = []

    def fit_grouped(self, y_train, train, test):
        from replica.analysis.contracts.grouped import GroupedFit

        self.frames.append((train, test))
        size = len(next(iter(test.factors.values())))
        return GroupedFit(
            np.full(size, y_train.mean()),
            float(y_train.std()),
            np.zeros(len(train.fixed) + 1),
            np.ones(len(train.fixed) + 1),
            predictive_variance=np.full(size, y_train.var()),
        )


def test_default_named_cv_holds_out_readers_and_stimuli():
    dataset = make_toy_dataset(n_participants=4)
    model = RecordingModel()
    design = Design(
        factors=(Factor("participant", "participant"), Factor("stimulus", "stimulus")),
        comparison="predictor",
        random_effects=(RandomEffect("participant", ("surprisal",)),),
    )
    result = evaluate(
        dataset,
        EvaluationSpec(
            targets=[dataset.feature(dataset.default_measure)],
            predictors=[ModelFeature(VaryingLM(), Surprisal())],
        ),
        AnalysisConfig(model=model, design=design, n_folds=2, per_fold=True),
    )[0]
    assert len(result.fold_results) == 4
    assert result.metadata["split_warnings"] == []
    for train, test in model.frames:
        for factor in ("participant", "stimulus"):
            assert not set(train.factors[factor]) & set(test.factors[factor])
        assert train.design.condition_on == ()


@pytest.mark.parametrize("explicit", [False, True])
def test_named_fraction_configuration_and_direct_splitter_agree(explicit):
    dataset = make_toy_dataset(n_participants=10, n_stimuli=10)
    design = Design(
        factors=(Factor("participant", "participant"), Factor("source", "stimulus")),
        cv_unit=("participant", "source"),
        comparison="predictor",
        random_effects=(RandomEffect("participant", ("surprisal",)),),
    )
    fractions = {"source": 0.9, "participant": 0.8}
    options = (
        {"cv": CrossedGroupSplitter(train_fraction=fractions)}
        if explicit
        else {"train_fraction": fractions}
    )
    model = RecordingModel()
    result = evaluate(
        dataset,
        EvaluationSpec(
            targets=[dataset.feature(dataset.default_measure)],
            predictors=[ModelFeature(VaryingLM(), Surprisal())],
        ),
        AnalysisConfig(model=model, design=design, per_fold=True, **options),
    )[0]
    assert len(result.fold_results) == 50
    assert len(model.frames) == 100
    assert result.metadata["crossed_cv"]["resolved_folds"] == {
        "participant": 5,
        "source": 10,
    }
    assert result.metadata["crossed_cv"]["train_fraction_rows_range"] == [0.72, 0.72]
    for train, test in model.frames:
        for factor in design.cv_units:
            assert not set(train.factors[factor]) & set(test.factors[factor])


def test_config_resolver_uses_crossed_defaults_and_explicit_overrides():
    assert AnalysisConfig().resolve_cv().train_fraction == (0.9,)
    assert AnalysisConfig(n_folds=5).resolve_cv().n_folds == (5,)
    assert AnalysisConfig(design=None).resolve_cv().n_folds == 10
    config = AnalysisConfig(design=Design(comparison="predictor"))
    assert config.resolve_cv().train_fraction == (0.9,)
    assert AnalysisConfig(
        design=config.design, n_folds=(2, 10)
    ).resolve_cv().n_folds == (2, 10)
    with pytest.raises(ValueError, match="either"):
        AnalysisConfig(design=config.design, n_folds=2, train_fraction=0.9)
    with pytest.raises(ValueError, match="crossed Design"):
        AnalysisConfig(design=None, train_fraction=0.9)
    with pytest.raises(ValueError, match="explicit cv"):
        AnalysisConfig(cv=CrossedGroupSplitter(), train_fraction=0.9)


def test_default_cv_uses_declared_unit_and_preserves_random_only_slopes():
    ds = make_toy_dataset(n_participants=4)
    model = RecordingModel()
    spec = EvaluationSpec(
        targets=[ds.feature(ds.default_measure)],
        predictors=[ModelFeature(VaryingLM(), Surprisal())],
    )
    config = AnalysisConfig(
        model=model,
        design=declaration(cv_unit="participant"),
        n_folds=2,
        per_fold=True,
        retain_per_obs_delta_ll=True,
    )
    result = evaluate(ds, spec, analysis=config)[0]
    for train, test in model.frames:
        assert not set(train.factors["participant"]) & set(test.factors["participant"])
        assert "surprisal" in train.columns
    assert model.frames[0][0].fixed == ()
    assert model.frames[1][0].fixed == ("surprisal",)
    assert result.metadata["split_warnings"] == []


def test_strict_comparison_warning_requires_choice_before_feature_extraction():
    dataset = make_toy_dataset()
    lm = ConstantLM()
    model = RecordingModel()
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="Design.comparison was not set")
        with pytest.raises(UserWarning, match="defaulting to 'predictor'"):
            evaluate(
                dataset,
                EvaluationSpec(
                    targets=[dataset.feature(dataset.default_measure)],
                    predictors=[ModelFeature(lm, Surprisal())],
                ),
                AnalysisConfig(model=model, design=declaration(comparison=None)),
            )
    assert lm.call_count == 0
    assert model.frames == []


@pytest.mark.parametrize("comparison", [None, "predictor", "fixed_effect"])
def test_comparison_resolves_columns_slopes_and_provenance(comparison):
    dataset = make_toy_dataset(n_participants=4)
    model = RecordingModel()
    control = UnitPosition()
    design = replace(
        declaration(comparison=comparison),
        random_effects=(
            RandomEffect("participant", (control.name, "surprisal")),
            RandomEffect("stimulus"),
        ),
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = evaluate(
            dataset,
            EvaluationSpec(
                targets=[dataset.feature(dataset.default_measure)],
                predictors=[ModelFeature(VaryingLM(), Surprisal())],
                controls=[ModelFeature(VaryingLM(), control)],
            ),
            AnalysisConfig(model=model, design=design, n_folds=2, per_fold=True),
        )[0]
    assert len(caught) == (1 if comparison is None else 0)
    if caught:
        assert "Design.comparison was not set" in str(caught[0].message)
    for i, (train, test) in enumerate(model.frames):
        included = i % 2 == 1 or comparison == "fixed_effect"
        expected = (control.name, "surprisal") if included else (control.name,)
        for frame in (train, test):
            assert tuple(frame.columns) == expected
            assert frame.design.random_effects[0].slopes == expected
            assert frame.design.condition_on == ("participant",)
            assert frame.design.cv_unit == design.cv_unit
        assert not set(train.factors["stimulus"]) & set(test.factors["stimulus"])
        side = "full" if i % 2 else "baseline"
        assert result.fold_results[i // 2].metadata[side]["design"] == (
            train.design.metadata()
        )
    assert result.metadata["design"]["comparison"] == comparison
    assert result.metadata["design"]["effective_comparison"] == (
        comparison or "predictor"
    )
    # Both sides use the same rows and the same training-derived control scaling.
    for baseline, full in zip(model.frames[::2], model.frames[1::2], strict=True):
        for base_frame, full_frame in zip(baseline, full, strict=True):
            np.testing.assert_array_equal(
                base_frame.columns[control.name], full_frame.columns[control.name]
            )
            for name in base_frame.factors:
                np.testing.assert_array_equal(
                    base_frame.factors[name], full_frame.factors[name]
                )


def test_predictor_comparison_four_models_keep_only_their_own_slopes():
    surprisal = ("surprisal", "surprisal_lag1", "surprisal_lag2")
    entropy = ("entropy", "entropy_lag1", "entropy_lag2")
    design = Design(
        factors=(Factor("participant", "participant"), Factor("stimulus", "stimulus")),
        random_effects=(
            RandomEffect("participant", surprisal + entropy, correlated=False),
            RandomEffect("stimulus"),
        ),
        condition_on=("participant",),
        comparison="predictor",
    )
    columns = {n: np.arange(4.0) for n in ("length", *surprisal, *entropy)}
    factors = {
        "participant": np.array(["a", "a", "b", "b"]),
        "stimulus": np.array(["x", "y", "x", "y"]),
    }
    for predictors in ((), surprisal, entropy, surprisal + entropy):
        frame = GroupedFrame(columns, factors, ("length", *predictors), design)
        assert tuple(frame.columns) == ("length", *predictors)
        assert frame.design.random_effects[0] == RandomEffect(
            "participant", predictors, correlated=False
        )
        assert frame.design.random_effects[1] == RandomEffect("stimulus")
    assert design.random_effects[0].slopes == surprisal + entropy
    assert len(columns) == 7


def test_slope_only_factor_is_removed_from_conditioning_not_sampling():
    design = Design(
        factors=(Factor("participant", "participant"), Factor("stimulus", "stimulus")),
        random_effects=(
            RandomEffect("participant", ("surprisal",), intercept=False),
            RandomEffect("stimulus"),
        ),
        condition_on=("participant", "stimulus"),
        cv_unit="participant",
        permutation_unit="stimulus",
        comparison="predictor",
    )
    baseline = design.for_model(())
    assert baseline.random_effects == (RandomEffect("stimulus"),)
    assert baseline.condition_on == ("stimulus",)
    assert baseline.factors == design.factors
    assert baseline.cv_unit == "participant"
    assert baseline.permutation_unit == "stimulus"


@pytest.mark.needs_r
def test_removed_slope_only_factor_does_not_reject_unseen_levels():
    design = Design(
        factors=(Factor("participant", "participant"), Factor("stimulus", "stimulus")),
        random_effects=(
            RandomEffect("participant", ("surprisal",), intercept=False),
            RandomEffect("stimulus"),
        ),
        condition_on=("participant",),
        unseen="error",
        comparison="predictor",
    )
    y, (full_train, full_test) = synthetic_frames(design)
    train = GroupedFrame({}, full_train.factors, (), design)
    test = GroupedFrame(
        {},
        {
            **full_test.factors,
            "participant": np.full(len(full_test.factors["participant"]), "new"),
        },
        (),
        design,
    )
    fit = RLmerModel(nlminb_fallback=False).fit_grouped(y, train, test)
    assert fit.metadata["formula"] == "y ~ 1 + (1 | g1)"
    assert fit.metadata["conditioning_formula"] is None
    assert fit.metadata["numeric_columns"] == {}


@pytest.mark.needs_r
@pytest.mark.parametrize("reml", [False, True])
def test_removing_last_random_term_preserves_residual_scale_convention(reml):
    design = Design(
        factors=(Factor("participant", "participant"), Factor("stimulus", "stimulus")),
        random_effects=(RandomEffect("participant", ("surprisal",), intercept=False),),
        condition_on=("participant",),
        comparison="predictor",
    )
    y = np.array([1.0, 2.0, 4.0, 7.0, 8.0, 11.0])
    x = np.arange(len(y), dtype=float)
    factors = {
        "participant": np.array(["a", "b"] * 3),
        "stimulus": np.array(["s"] * 6),
    }
    frame = GroupedFrame({"length": x}, factors, ("length",), design)
    fit = RLmerModel(reml=reml, nlminb_fallback=False).fit_grouped(y, frame, frame)
    matrix = np.column_stack([np.ones(len(y)), x])
    beta = np.linalg.lstsq(matrix, y, rcond=None)[0]
    residuals = y - beta[0] - beta[1] * x
    expected_sigma = np.sqrt(np.sum(residuals**2) / (len(y) - (2 if reml else 0)))
    np.testing.assert_allclose(fit.beta, beta)
    np.testing.assert_allclose(fit.predictions, y - residuals)
    assert fit.sigma == pytest.approx(expected_sigma)
    expected_se = expected_sigma * np.sqrt(np.diag(np.linalg.inv(matrix.T @ matrix)))
    np.testing.assert_allclose(fit.se, expected_se)
    assert fit.metadata["formula"] == "y ~ 1 + x0"
    assert fit.metadata["conditioning_formula"] is None


def test_direct_grouped_frame_defaults_to_removal_with_warning():
    _, (train, _) = synthetic_frames()
    with pytest.warns(UserWarning, match="Design.comparison was not set"):
        baseline = replace(train, fixed=(), design=declaration(comparison=None))
    assert baseline.columns == {}
    assert all(not effect.slopes for effect in baseline.design.random_effects)
    assert baseline.design.comparison == "predictor"


@pytest.mark.parametrize("slopes", [(), ("unit_position",)])
def test_unspecified_comparison_warns_even_when_no_slopes_are_removed(slopes):
    dataset = make_toy_dataset(n_participants=3)
    design = Design(
        factors=(Factor("participant", "participant"), Factor("stimulus", "stimulus")),
        random_effects=(RandomEffect("participant", slopes),),
    )
    model = RecordingModel()
    with pytest.warns(UserWarning, match="Design.comparison was not set") as caught:
        evaluate(
            dataset,
            EvaluationSpec(
                targets=[dataset.feature(dataset.default_measure)],
                predictors=[ModelFeature(VaryingLM(), Surprisal())],
                controls=[UnitPosition()],
            ),
            AnalysisConfig(model=model, design=design, n_folds=2),
        )
    assert len(caught) == 1
    for train, test in model.frames:
        assert (
            train.design.random_effects
            == test.design.random_effects
            == design.random_effects
        )
        assert train.design.comparison == test.design.comparison == "predictor"


def test_invalid_comparison_is_rejected():
    with pytest.raises(ValueError, match="comparison must be"):
        Design(comparison="unknown")


@pytest.mark.needs_r
@pytest.mark.parametrize("comparison", ["predictor", "fixed_effect"])
def test_baseline_comparison_matches_independent_r(comparison):
    from replica.analysis.models.support.r import _r_context

    design = declaration(comparison=comparison)
    design = replace(
        design,
        random_effects=tuple(
            replace(r, correlated=False) for r in design.random_effects
        ),
    )
    y, (train, test) = synthetic_frames(design)
    fit = RLmerModel(nlminb_fallback=False).fit_grouped(
        y, replace(train, fixed=()), replace(test, fixed=())
    )
    assert fit.metadata["design"] == design.for_model(()).metadata()
    assert ("surprisal" in fit.metadata["numeric_columns"]) == (
        comparison == "fixed_effect"
    )
    with _r_context() as ro:
        for key, frame in ((".ref_train", train), (".ref_test", test)):
            columns = {"x": ro.FloatVector(frame.columns["surprisal"])}
            columns.update(
                {
                    f"g{i}": ro.StrVector(frame.factors[n])
                    for i, n in enumerate(("participant", "stimulus", "word_type"))
                }
            )
            if key == ".ref_train":
                columns["y"] = ro.FloatVector(y)
            ro.globalenv[key] = ro.DataFrame(columns)
        random = "(1|g0) + (1|g1) + (1|g2)"
        conditioning = "~(1|g0)"
        if comparison == "fixed_effect":
            random = "(1|g0) + (0+x|g0) + (1|g1) + (0+x|g1) + (1|g2)"
            conditioning += " + (0+x|g0)"
        ro.r(
            f"ref <- lme4::lmer(y ~ 1 + {random}, data=.ref_train, REML=FALSE, "
            'control=lme4::lmerControl(optimizer="bobyqa",optCtrl=list(maxfun=50000)))'
        )
        predictions = np.asarray(
            ro.r(
                f"predict(ref,newdata=.ref_test,re.form={conditioning},allow.new.levels=TRUE)"
            )
        )
        coefficients = np.asarray(ro.r("lme4::fixef(ref)"))
        sigma = float(ro.r("sigma(ref)")[0])
    np.testing.assert_allclose(fit.predictions, predictions, atol=1e-8)
    np.testing.assert_allclose(fit.beta, coefficients, atol=1e-8)
    assert fit.sigma == pytest.approx(sigma)


def test_explicit_cv_warning_persists_without_fold_results():
    dataset = make_toy_dataset(n_participants=2)
    from replica.analysis import FullDataSplitter

    spec = EvaluationSpec(
        targets=[dataset.feature(dataset.default_measure)],
        predictors=[ModelFeature(VaryingLM(), Surprisal())],
    )
    config = AnalysisConfig(
        model=RecordingModel(), design=declaration(), cv=FullDataSplitter()
    )
    with pytest.warns(UserWarning, match="shares declared CV unit"):
        result = evaluate(dataset, spec, analysis=config)[0]
    assert result.metadata["cv_override"]
    assert result.metadata["split_warnings"]


@pytest.mark.parametrize("per_fold", [False, True])
def test_omitted_known_reader_effects_warn_before_fitting_and_are_recorded(per_fold):
    ds = make_toy_dataset(n_participants=3)
    spec = EvaluationSpec(
        targets=[ds.feature(ds.default_measure)],
        predictors=[ModelFeature(VaryingLM(), Surprisal())],
    )
    design = replace(declaration(), condition_on=())
    model = RecordingModel()
    config = AnalysisConfig(model=model, design=design, n_folds=2, per_fold=per_fold)
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="Test readers in factor")
        with pytest.raises(UserWarning, match="not in condition_on"):
            evaluate(ds, spec, analysis=config)
    assert not model.frames
    with pytest.warns(UserWarning, match="not in condition_on") as caught:
        result = evaluate(ds, spec, analysis=config)[0]
    assert len(caught) == 1
    assert result.metadata["conditioning_warnings"] == [str(caught[0].message)]
    assert result.metadata["design"]["condition_on"] == ()
    for fold in result.fold_results or []:
        assert (
            fold.metadata["conditioning_warnings"]
            == result.metadata["conditioning_warnings"]
        )


def test_conditioning_check_uses_reader_source_and_actual_overlap():
    design = Design(
        factors=(
            Factor("reader", "participant"),
            Factor("text", "stimulus"),
            Factor("item", "unit:item_id", role="model"),
        ),
        random_effects=(RandomEffect("reader"), RandomEffect("item")),
        cv_unit="text",
        comparison="predictor",
    )
    factors = {
        "reader": np.array(["a", "b", "a", "b"]),
        "text": np.array(["x", "x", "y", "y"]),
        "item": np.array(["x1", "x1", "y1", "y1"]),
    }
    train = np.array([True, True, False, False])
    splits = [(train, ~train)]
    with pytest.warns(UserWarning, match="factor 'reader'"):
        assert design.check_conditioning(splits, factors)
    assert design.condition_on == ()  # No automatic conditioning on unseen items.
    conditional = replace(design, condition_on=("reader",), unseen="error")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert conditional.check_conditioning(splits, factors) == []
        unseen_readers = {**factors, "reader": np.array(["a", "b", "c", "d"])}
        assert design.check_conditioning(splits, unseen_readers) == []
        no_reader_effect = replace(design, random_effects=(RandomEffect("item"),))
        assert no_reader_effect.check_conditioning(splits, factors) == []
    assert not caught


def test_rust_known_stall_is_rejected_before_fit(monkeypatch):
    from types import SimpleNamespace

    from replica.analysis.models.mixed_effects import lmer_rs

    monkeypatch.setattr(lmer_rs, "version", lambda _: "0.2.5")
    model = lmer_rs.RustLmerModel()
    model._random_terms = [
        "(1 | participant_id)",
        "(0 + x0 | participant_id)",
        "(1 | item_id)",
    ]
    with pytest.raises(ValueError, match="stalls"):
        model._control(SimpleNamespace())


def test_rust_controls_pass_through_and_validate(monkeypatch):
    from types import SimpleNamespace

    from replica.analysis.models.mixed_effects import lmer_rs

    monkeypatch.setattr(lmer_rs, "version", lambda _: "0.2.6")
    model = lmer_rs.RustLmerModel(
        max_iterations=20, tolerance=1e-9, max_inner_iterations=5
    )
    model._random_terms = ["(1 | participant_id)"]
    control = model._control(SimpleNamespace(FitControl=lambda **kw: kw))
    assert control["max_iterations"] == 20
    assert control["max_inner_iterations"] == 5
    assert control["tolerance"] == 1e-9
    with pytest.raises(ValueError, match="positive integer"):
        lmer_rs.RustLmerModel(max_iterations=0)


def test_mixed_metadata_labels_rejected_across_datums():
    ds = make_toy_dataset(n_participants=2)
    for i, datum in enumerate(ds.data()):
        datum.metadata["group"] = 1 if i % 2 else "1"
    design = replace(
        declaration(),
        factors=(*declaration().factors, Factor("group", "datum:group", role="model")),
    )
    with pytest.raises(ValueError, match="mix scalar types"):
        prepare_observations(
            ds,
            EvaluationSpec(
                targets=[ds.feature(ds.default_measure)],
                predictors=[ModelFeature(VaryingLM(), Surprisal())],
            ),
            AnalysisConfig(design=design),
        )


def test_grouped_model_contract_requires_validation():
    dataset = make_toy_dataset()

    class IncompleteBackend:
        def fit_grouped(self, y_train, train, test):
            raise AssertionError("must reject before fitting")

    lm = ConstantLM()
    with pytest.raises(ValueError, match="does not support named Design"):
        evaluate(
            dataset,
            EvaluationSpec(
                targets=[dataset.feature(dataset.default_measure)],
                predictors=[ModelFeature(lm, Surprisal())],
            ),
            AnalysisConfig(model=IncompleteBackend(), design=declaration()),
        )
    assert lm.call_count == 0


@pytest.mark.needs_r
def test_uncorrelated_conditional_predictions_match_all_effects():
    from replica.analysis.models.support.r import _r_context

    y, (train, test) = synthetic_frames()
    design = replace(
        train.design,
        random_effects=tuple(
            replace(r, correlated=False) for r in train.design.random_effects
        ),
        condition_on=("participant", "stimulus", "word_type"),
        unseen="zero",
    )
    fit = RLmerModel(nlminb_fallback=False).fit_grouped(
        y, replace(train, design=design), replace(test, design=design)
    )
    with _r_context() as ro:
        expected = np.asarray(
            ro.r(
                "predict(.lmer_model,newdata=.test_data,re.form=NULL,allow.new.levels=TRUE)"
            )
        )
    np.testing.assert_allclose(fit.predictions, expected, atol=1e-9)


@pytest.mark.needs_r
def test_metadata_records_actual_optimizer():
    class AlternateOptimizer(RLmerModel):
        def _fit_lmer(self, ro, formula_str):
            requested = self.optimizer
            try:
                self.optimizer = "Nelder_Mead"
                super()._fit_lmer(ro, formula_str)
            finally:
                self.optimizer = requested

    y, (train, test) = synthetic_frames()
    fit = AlternateOptimizer(nlminb_fallback=False).fit_grouped(y, train, test)
    assert fit.metadata["optimizer_requested"] == "bobyqa"
    assert fit.metadata["optimizer_used"] == "Nelder_Mead"


@pytest.mark.parametrize("sigma", [np.nan, np.inf, 0, -1])
def test_grouped_fit_rejects_invalid_gaussian_scale(sigma):
    from replica.analysis.contracts.grouped import GroupedFit

    with pytest.raises(ValueError, match="sigma must be finite and positive"):
        GroupedFit(np.ones(2), sigma, np.zeros(1), np.ones(1))


@pytest.mark.parametrize("missing", [np.nan, np.inf, "masked"])
def test_grouped_training_cannot_silently_omit_responses(missing):
    from replica.analysis.models.mixed_effects.grouped import fit_lmer_grouped

    y, (train, test) = synthetic_frames()
    if isinstance(missing, str):
        y = np.ma.array(y, mask=False)
        y.mask[0] = True
    else:
        y[0] = missing
    with pytest.raises(ValueError, match="Training responses must be finite"):
        fit_lmer_grouped(RLmerModel(), None, y, train, test)


def test_grouped_frame_rejects_masked_numeric_predictors():
    _, (train, _) = synthetic_frames()
    values = np.ma.array(train.columns["surprisal"], mask=False)
    values.mask[0] = True
    with pytest.raises(ValueError, match="must be finite"):
        replace(train, columns={"surprisal": values})
    values.mask[:] = False
    actual = replace(train, columns={"surprisal": values})
    np.testing.assert_array_equal(
        actual.columns["surprisal"], train.columns["surprisal"]
    )


@pytest.mark.parametrize("field", ["predictions", "beta", "se", "predictive_variance"])
def test_grouped_fit_rejects_masks_before_converting_backend_arrays(field):
    from replica.analysis.contracts.grouped import GroupedFit

    valid = GroupedFit(
        np.ones(2), 1, np.zeros(1), np.ones(1), predictive_variance=np.ones(2)
    )
    values = np.ma.array(getattr(valid, field), mask=False)
    values.mask[0] = True
    with pytest.raises(ValueError, match="masked"):
        replace(valid, **{field: values})
    values.mask[:] = False
    actual = replace(valid, **{field: values})
    np.testing.assert_array_equal(getattr(actual, field), getattr(valid, field))


def test_grouped_fit_rejects_wrong_coefficient_order_contract():
    dataset = make_toy_dataset()

    class TruncatedCoefficients(RecordingModel):
        def fit_grouped(self, y_train, train, test):
            fitted = super().fit_grouped(y_train, train, test)
            return replace(fitted, beta=np.empty(0), se=np.empty(0))

    with pytest.raises(ValueError, match="coefficients must follow"):
        evaluate(
            dataset,
            EvaluationSpec(
                targets=[dataset.feature(dataset.default_measure)],
                predictors=[ModelFeature(VaryingLM(), Surprisal())],
            ),
            AnalysisConfig(
                model=TruncatedCoefficients(), design=declaration(), n_folds=2
            ),
        )


@pytest.mark.parametrize(
    "change",
    [
        {"predictions": [np.nan]},
        {"predictions": [[1]]},
        {"beta": [np.inf]},
        {"se": [-1]},
        {"se": [1, 2]},
    ],
)
def test_grouped_fit_rejects_invalid_backend_arrays(change):
    from replica.analysis.contracts.grouped import GroupedFit

    valid = GroupedFit(np.ones(2), 1, np.zeros(1), np.ones(1))
    with pytest.raises(ValueError, match="Grouped"):
        replace(valid, **change)


def test_numeric_and_categorical_names_cannot_collide_before_extraction():
    dataset = make_toy_dataset()
    lm = ConstantLM()
    design = replace(
        declaration(),
        factors=(*declaration().factors, Factor("surprisal", "word", role="model")),
        categorical=("surprisal",),
    )
    with pytest.raises(ValueError, match="names must be distinct"):
        evaluate(
            dataset,
            EvaluationSpec(
                targets=[dataset.feature(dataset.default_measure)],
                predictors=[ModelFeature(lm, Surprisal())],
            ),
            AnalysisConfig(model=RecordingModel(), design=design),
        )
    assert lm.call_count == 0
