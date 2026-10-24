"""Fit diagnostics preserve call identity, scored responses and numerical behavior."""

import warnings
from copy import deepcopy
from dataclasses import asdict

import numpy as np
import pytest

from replica import (
    AnalysisConfig,
    Design,
    EvaluationSpec,
    Factor,
    FullDataSplitter,
    ItemSplitter,
    MultiSeedSplitter,
    Residualize,
    evaluate,
)
from replica.analysis.contracts.grouped import GroupedFit
from replica.analysis.contracts.models import FoldScore
from replica.feature import ModelFeature, Surprisal, UnitPosition
from tests.analysis._fixtures import VaryingLM, make_toy_dataset


class _FitSpy:
    name = "diagnostic_spy"

    def __init__(self):
        self.calls = []
        self.predictions = []
        self.returned_predictions = []
        self.snapshot_calls = []
        self.rng = np.random.default_rng(927)
        self.current = {"selected_parameters": {"call": 0}, "details": {"markers": []}}

    def _predict(self, y_train, x_test, n_columns):
        self.calls.append((len(y_train), n_columns, len(x_test)))
        number = len(self.calls)
        # Reuse nested mutable objects deliberately: shallow snapshots are wrong.
        self.current["selected_parameters"]["call"] = number
        self.current["details"]["markers"][:] = [number]
        prediction = (
            y_train.mean()
            + 0.05 * x_test.sum(axis=1)
            + self.rng.normal(0, 0.1, len(x_test))
        )
        self.predictions.append(prediction.copy())
        self.returned_predictions.append(prediction)
        return prediction

    def last_fit_diagnostics(self):
        self.snapshot_calls.append(len(self.calls))
        return self.current


class _MatrixSpy(_FitSpy):
    def fit_predict(self, y_train, x_train, x_test):
        return self._predict(y_train, x_test, x_train.shape[1])

    def fit_predict_with_sigma(self, y_train, x_train, x_test):
        return self.fit_predict(y_train, x_train, x_test), 1.0


class _LikelihoodSpy(_FitSpy):
    owns_loglik = True

    def fit_score(self, y_train, x_train, x_test, y_test, *, is_full):
        prediction = self._predict(y_train, x_test, x_train.shape[1])
        pointwise = -0.5 * (y_test - prediction) ** 2 - 0.123
        return FoldScore(float(pointwise.sum()), prediction, pointwise)


class _GroupedSpy(_FitSpy):
    def validate_design(self, design):
        pass

    def last_fit_diagnostics(self):
        pytest.fail("GroupedFit.metadata is authoritative, not mutable backend state")

    def fit_grouped(self, y_train, train, test):
        x_test = np.column_stack([test.columns[name] for name in train.fixed])
        prediction = self._predict(y_train, x_test, len(train.fixed))
        return GroupedFit(
            prediction,
            1.0,
            np.zeros(len(train.fixed) + 1),
            np.ones(len(train.fixed) + 1),
            metadata={
                "optimizer_convergence": [0],
                "convergence_messages": [],
                "fit_diagnostics": self.current,
            },
        )


_BACKENDS = [_MatrixSpy, _LikelihoodSpy, _GroupedSpy]


def _evaluate(
    model,
    *,
    controls=True,
    per_fold=False,
    diagnostics=False,
    retain_predictions=False,
    preprocess=False,
    cv=None,
    n_participants=1,
    empty_control_baseline="mean",
    sigma_source="training",
):
    dataset = make_toy_dataset(n_stimuli=6, n_words=5, n_participants=n_participants)
    target = dataset.feature("spr_rt")
    position = UnitPosition()
    design = (
        Design(
            factors=(Factor("stimulus", "stimulus"),),
            random_effects=(),
            cv_unit="stimulus",
            comparison="predictor",
            scoring="residual",
        )
        if isinstance(model, _GroupedSpy)
        else None
    )
    result = evaluate(
        dataset,
        EvaluationSpec(
            targets=[target],
            predictors=[ModelFeature(VaryingLM(), Surprisal())],
            controls=[position] if controls else [],
        ),
        AnalysisConfig(
            model=model,
            design=design,
            cv=cv,
            n_folds=3,
            transform_dv=np.log2,
            per_fold=per_fold,
            retain_per_obs_delta_ll=per_fold,
            prediction_diagnostics=diagnostics,
            retain_predictions=retain_predictions,
            empty_control_baseline=empty_control_baseline,
            sigma_source=sigma_source,
            preprocessing=(
                (Residualize(target, (position,), policy="centered_slope"),)
                if preprocess
                else ()
            ),
        ),
    )[0]
    return dataset, result


@pytest.mark.parametrize("backend", _BACKENDS)
def test_fit_snapshots_keep_each_side_and_fold_without_retaining_folds(backend):
    model = backend()
    _, result = _evaluate(model)
    assert result.fold_results is None
    assert model.calls == [(16, 1, 8), (16, 2, 8)] * 3
    assert model.snapshot_calls == ([] if backend is _GroupedSpy else list(range(1, 7)))
    model.current["selected_parameters"]["call"] = -1
    model.current["details"]["markers"].append(-1)
    snapshots = result.metadata["fit_diagnostics"]
    assert [record["fold_index"] for record in snapshots] == [0, 1, 2]
    for fold, record in enumerate(snapshots):
        for side, number in (("baseline", 2 * fold + 1), ("full", 2 * fold + 2)):
            assert record[side]["selected_parameters"] == {"call": number}
            assert record[side]["details"] == {"markers": [number]}


@pytest.mark.parametrize("needs_grouping", [False, True])
def test_analytic_baseline_never_reads_previous_backend_fit(needs_grouping):
    model = _MatrixSpy()
    model.needs_grouping = needs_grouping
    dataset, result = _evaluate(model, controls=False)
    assert model.calls == [(16, 1 + 2 * needs_grouping, 8)] * 3
    assert model.snapshot_calls == [1, 2, 3]
    y = np.log2(np.concatenate([datum.values[1:] for datum in dataset.data()]))
    stimuli = np.repeat(np.arange(6), 4)
    for number, record in enumerate(result.metadata["fit_diagnostics"], 1):
        training = y[stimuli % 3 != number - 1]
        sigma = np.std(training, ddof=1)
        assert record["baseline"] == {
            "method": "analytic_mean",
            "convergence_applicable": False,
            "rank": 1,
            "n_columns": 1,
            "coefficients": {
                "names": ["(Intercept)"],
                "estimates": [training.mean()],
                "naming": "design_columns",
            },
            "residual_scale": {
                "sigma": sigma,
                "variance": sigma**2,
                "convention": "sample_training_sd",
            },
        }
        assert record["full"]["selected_parameters"] == {"call": number}


@pytest.mark.parametrize("needs_grouping", [False, True])
@pytest.mark.parametrize("sigma_source", ["training", "test"])
def test_empty_control_model_baseline_fits_and_scores_the_backend(
    needs_grouping, sigma_source
):
    model = _MatrixSpy()
    model.needs_grouping = needs_grouping
    dataset, result = _evaluate(
        model,
        controls=False,
        per_fold=True,
        empty_control_baseline="model",
        sigma_source=sigma_source,
    )
    n_groups = 2 * needs_grouping
    assert model.calls == [(16, n_groups, 8), (16, n_groups + 1, 8)] * 3
    assert result.metadata["configuration"]["empty_control_baseline"] == "model"
    y = np.log2(np.concatenate([datum.values[1:] for datum in dataset.data()]))
    stimuli = np.repeat(np.arange(6), 4)
    for index, fold in enumerate(result.fold_results):
        held = stimuli % 3 == index
        predicted = model.predictions[2 * index]
        assert not np.allclose(predicted, y[~held].mean())
        residual = y[held] - predicted
        sigma = 1.0 if sigma_source == "training" else np.sqrt(np.mean(residual**2))
        expected = (
            -0.5 * np.log(2 * np.pi) - np.log(sigma) - 0.5 * (residual / sigma) ** 2
        )
        assert fold.baseline_ll == pytest.approx(expected.sum())
        baseline = result.metadata["fit_diagnostics"][index]["baseline"]
        assert baseline["selected_parameters"] == {"call": 2 * index + 1}


@pytest.mark.parametrize(
    "kwargs",
    [
        {"empty_control_baseline": "invalid", "design": None},
        {"empty_control_baseline": "model"},
        {"empty_control_baseline": "model", "design": None, "analysis_fn": lambda: []},
        {"empty_control_baseline": "model", "design": None, "model": _LikelihoodSpy()},
    ],
)
def test_empty_control_baseline_rejects_inapplicable_configuration(kwargs):
    with pytest.raises(ValueError, match="empty_control_baseline"):
        AnalysisConfig(**kwargs)


@pytest.mark.parametrize("provider", ["missing", "none"])
def test_unavailable_backend_diagnostics_stay_none(monkeypatch, provider):
    model = _MatrixSpy()
    if provider == "missing":
        monkeypatch.delattr(_FitSpy, "last_fit_diagnostics")
    else:
        monkeypatch.setattr(model, "last_fit_diagnostics", lambda: None)
    _, result = _evaluate(model)
    assert result.metadata["fit_diagnostics"] == [
        {"fold_index": fold, "baseline": None, "full": None} for fold in range(3)
    ]
    assert len(model.calls) == 6


@pytest.mark.parametrize("backend", _BACKENDS)
def test_prediction_capture_preserves_fits_rng_scores_and_transformed_errors(
    backend, monkeypatch
):
    global_rng = deepcopy(np.random.get_state())
    disabled, enabled = backend(), backend()
    with monkeypatch.context() as guard:
        guard.setattr(
            "replica.analysis.fitting.regression.prediction_diagnostics",
            lambda *_args, **_kwargs: pytest.fail("Disabled diagnostics were computed"),
        )
        guard.setattr(
            "replica.analysis.fitting.regression.retained_predictions",
            lambda *_args: pytest.fail("Disabled observation arrays were copied"),
        )
        _, original = _evaluate(
            disabled,
            per_fold=True,
            preprocess=True,
            n_participants=2,
            cv=MultiSeedSplitter([10, 20], lambda _: ItemSplitter(3)),
        )
    dataset, captured = _evaluate(
        enabled,
        per_fold=True,
        diagnostics=True,
        retain_predictions=True,
        preprocess=True,
        n_participants=2,
        cv=MultiSeedSplitter([10, 20], lambda _: ItemSplitter(3)),
    )
    assert disabled.calls == enabled.calls == [(32, 1, 16), (32, 2, 16)] * 6
    assert disabled.rng.bit_generator.state == enabled.rng.bit_generator.state
    np.testing.assert_equal(np.random.get_state(), global_rng)
    np.testing.assert_array_equal(disabled.predictions, enabled.predictions)

    # Independent oracle: fixture responses, explicit stimulus membership and
    # zero-indexed unit positions; no preparation or diagnostic helper is reused.
    y = np.log2(np.concatenate([datum.values[1:] for datum in dataset.data()]))
    positions = np.tile(np.arange(1, 5), 12)
    stimuli = np.repeat(np.arange(6), 8)
    source_ids = [
        {
            "participant": datum.participant_id,
            "stimulus": datum.stimulus.stimulus_id,
            "unit": unit,
            "layer": "words",
            "start": 5 * unit,
            "end": 5 * unit + 4,
        }
        for datum in dataset.data()
        for unit in range(1, 5)
    ]
    for index, fold in enumerate(captured.fold_results):
        held = stimuli % 3 == index % 3
        slope = np.polyfit(positions[~held], y[~held], 1)[0]
        target = y[held] - slope * positions[held]
        summary = fold.metadata["prediction_diagnostics"]
        assert summary["population"] == "scored_rows"
        assert (
            summary["response_scale"] == "model_after_transform_and_fold_preprocessing"
        )
        assert summary["n_rows"] == 16 and summary["n_train_overlap"] == 0
        assert fold.metadata["seed"] == [10, 20][index // 3]
        assert fold.metadata["inner_fold_index"] == index % 3
        scaling = fold.metadata["scaling"]
        assert (
            scaling["baseline"]["fitted"]["n_columns"],
            scaling["full"]["fitted"]["n_columns"],
        ) == (1, 2)
        np.testing.assert_array_equal(
            scaling["baseline"]["fitted"]["center"], [positions[~held].mean()]
        )
        np.testing.assert_array_equal(
            scaling["baseline"]["fitted"]["scale"], [positions[~held].std()]
        )
        np.testing.assert_array_equal(
            scaling["full"]["fitted"]["center"][:1],
            scaling["baseline"]["fitted"]["center"],
        )
        groups = fold.metadata["groups"]["stimulus"]
        assert (
            groups["n_train_groups"],
            groups["n_scored_groups"],
            groups["n_unseen_scored_groups"],
            groups["n_rows_in_unseen_groups"],
        ) == (4, 2, 2, 16)
        assert (
            groups["train_group_size"]
            == groups["scored_group_size"]
            == {"min": 8, "max": 8}
        )
        observed = fold.observations
        np.testing.assert_allclose(observed["observed"], target, atol=1e-13, rtol=0)
        np.testing.assert_array_equal(
            observed["prepared_row_index"], np.flatnonzero(held)
        )
        assert not observed["in_train"].any()
        names = fold.metadata["observation_capture"]["row_key_names"]
        expected_ids = [source_ids[i] for i in np.flatnonzero(held)]
        actual_keys = [
            dict(zip(names, key, strict=True)) for key in fold.observation_keys
        ]
        for actual, expected in zip(actual_keys, expected_ids, strict=True):
            assert {name: actual[name] for name in expected} == expected
            assert actual["material"] and actual["participant_source"]
        assert fold.row_ids.tolist() == [
            [row["participant"], row["stimulus"], str(row["unit"])]
            for row in expected_ids
        ]
        for offset, side in enumerate(("baseline", "full")):
            errors = target - enabled.predictions[2 * index + offset]
            np.testing.assert_array_equal(
                observed[f"prediction_{side}"], enabled.predictions[2 * index + offset]
            )
            assert summary[side]["n_finite_errors"] == 16
            assert summary[side]["error_mean"] == pytest.approx(
                errors.mean(), abs=1e-13
            )
            assert summary[side]["error_rms"] == pytest.approx(
                np.sqrt(np.mean(errors**2)), abs=1e-13
            )
            assert summary[side]["normal_qq"]["available"] is (
                backend is not _LikelihoodSpy
            )

    assert [fold.observation_keys for fold in captured.fold_results[:3]] == [
        fold.observation_keys for fold in captured.fold_results[3:]
    ]

    before, after = asdict(original), asdict(captured)
    for record in (before, after):
        for owner in (record["metadata"], record["response"]):
            owner.get("configuration", {}).pop("prediction_diagnostics", None)
            owner.get("configuration", {}).pop("retain_predictions", None)
        for attempt in record["metadata"]["fold_attempts"]:
            attempt.pop("elapsed_seconds", None)
    for fold in before["fold_results"]:
        assert "prediction_diagnostics" not in fold["metadata"]
    for source, fold in zip(before["fold_results"], after["fold_results"], strict=True):
        del fold["metadata"]["prediction_diagnostics"]
        del fold["metadata"]["observation_capture"]
        assert source["observations"] is source["observation_keys"] is None
        for name in ("observations", "observation_keys", "row_ids", "factor_ids"):
            fold[name] = source[name]
    np.testing.assert_equal(before, after)

    for prediction in enabled.returned_predictions:
        prediction[:] = np.nan
    for index, fold in enumerate(captured.fold_results):
        for offset, side in enumerate(("baseline", "full")):
            np.testing.assert_array_equal(
                fold.observations[f"prediction_{side}"],
                enabled.predictions[2 * index + offset],
            )

    if backend is _GroupedSpy:
        enabled.current["selected_parameters"]["call"] = -1
        enabled.current["details"]["markers"].append(-1)
        for index, fold in enumerate(captured.fold_results):
            for offset, side in enumerate(("baseline", "full"), 1):
                retained = fold.metadata[side]
                snapshot = captured.metadata["fit_diagnostics"][index][side]
                assert "fit_diagnostics" not in retained
                assert retained["diagnostics"] == snapshot
                assert snapshot["selected_parameters"] == {"call": 2 * index + offset}
                assert snapshot["details"] == {"markers": [2 * index + offset]}


def test_in_sample_diagnostics_record_scored_overlap():
    _, result = _evaluate(
        _MatrixSpy(),
        per_fold=True,
        diagnostics=True,
        retain_predictions=True,
        cv=FullDataSplitter(),
    )
    fold = result.fold_results[0]
    assert result.n_observations == 24
    assert fold.metadata["n_train"] == fold.n_test == 24
    assert fold.metadata["prediction_diagnostics"]["population"] == "scored_rows"
    assert fold.metadata["prediction_diagnostics"]["n_train_overlap"] == 24
    assert fold.observations["in_train"].all()


@pytest.mark.parametrize(
    "options", [{"per_fold": False}, {"per_fold": True, "analysis_fn": lambda: []}]
)
def test_prediction_capture_requires_retained_builtin_folds(options):
    with pytest.raises(ValueError, match="prediction_diagnostics"):
        AnalysisConfig(design=None, prediction_diagnostics=True, **options)


class _AttemptSplitter:
    def split(self, y, groups, participants, items):
        first, last = groups < 2, groups >= 4
        yield ~first, first
        yield np.zeros(len(y), dtype=bool), np.ones(len(y), dtype=bool)
        yield ~last, last


@pytest.mark.parametrize("fails", [False, True])
def test_attempt_records_keep_skips_warnings_and_original_failure(fails):
    problem = RuntimeError("deliberate full-fit failure")

    class WarnFit(_MatrixSpy):
        attempted = 0

        def fit_predict(self, y_train, x_train, x_test):
            self.attempted += 1
            warnings.warn(
                f"backend attempt {self.attempted}", UserWarning, stacklevel=1
            )
            if fails and self.attempted == 4:
                raise problem
            return super().fit_predict(y_train, x_train, x_test)

    model = WarnFit()
    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter("always", UserWarning)
        if fails:
            with pytest.raises(RuntimeError) as caught:
                _evaluate(model, per_fold=True, cv=_AttemptSplitter())
            assert caught.value is problem
            diagnostic = caught.value.analysis_diagnostics
            assert diagnostic["status"] == "failed"
            assert diagnostic["dataset"] == "toy"
            assert diagnostic["target"] == "spr_rt"
            assert diagnostic["environment"]["python"]
        else:
            _, result = _evaluate(model, per_fold=True, cv=_AttemptSplitter())
            diagnostic = result.metadata
            assert [fold.fold_index for fold in result.fold_results] == [0, 2]
            assert diagnostic["execution_environment"]["python"]
    attempts = diagnostic["fold_attempts"]
    assert model.attempted == 4 and len(model.calls) == (3 if fails else 4)
    assert [str(item.message) for item in emitted] == [
        f"backend attempt {i}" for i in range(1, 5)
    ]
    assert [item["status"] for item in attempts] == [
        "completed",
        "skipped",
        "failed" if fails else "completed",
    ]
    assert [
        (item["fold_index"], item["n_train"], item["n_test"]) for item in attempts
    ] == [(0, 16, 8), (1, 0, 24), (2, 16, 8)]
    assert attempts[1]["reason"] == "empty_training" and not attempts[1]["warnings"]
    for index, start in ((0, 1), (2, 3)):
        assert attempts[index]["warnings"] == [
            {"category": "UserWarning", "message": f"backend attempt {i}"}
            for i in (start, start + 1)
        ]
    if fails:
        assert attempts[-1]["stage"] == "full_fit_and_predict"
        assert attempts[-1]["error"] == {
            "type": "builtins.RuntimeError",
            "message": str(problem),
        }


@pytest.mark.parametrize("policy", ["default", "error"])
def test_warning_capture_preserves_filter_policy(policy):
    class WarnFit(_MatrixSpy):
        def fit_predict(self, y_train, x_train, x_test):
            warnings.warn("same backend location", UserWarning, stacklevel=1)
            return super().fit_predict(y_train, x_train, x_test)

    model = WarnFit()
    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter(policy, UserWarning)
        hook = warnings.showwarning
        filters = list(warnings.filters)
        if policy == "error":
            with pytest.raises(UserWarning, match="same backend location") as caught:
                _evaluate(model)
            assert not model.calls
            assert (
                caught.value.analysis_diagnostics["fold_attempts"][0]["status"]
                == "failed"
            )
        else:
            _, result = _evaluate(model)
            assert len(model.calls) == 6
            assert len(emitted) == 1
            assert [
                len(item["warnings"]) for item in result.metadata["fold_attempts"]
            ] == [1, 0, 0]
        assert warnings.showwarning is hook and warnings.filters == filters


@pytest.mark.parametrize(
    "options", [{"per_fold": False}, {"per_fold": True, "analysis_fn": lambda: []}]
)
def test_observation_capture_requires_retained_builtin_folds(options):
    with pytest.raises(ValueError, match="retain_predictions"):
        AnalysisConfig(design=None, retain_predictions=True, **options)


@pytest.mark.parametrize("metadata_error", [OSError, ValueError])
def test_optional_environment_metadata_cannot_break_fit_or_mask_failure(
    monkeypatch, metadata_error
):
    from replica.analysis.fitting import diagnostics

    original = diagnostics.importlib.metadata.version

    def version(distribution):
        if distribution == "numpy":
            raise metadata_error("unreadable optional version metadata")
        return original(distribution)

    monkeypatch.setattr(diagnostics.importlib.metadata, "version", version)
    _, result = _evaluate(_MatrixSpy())
    environment = result.metadata["execution_environment"]
    assert environment["packages"]["numpy"] is None
    assert "numpy" in str(environment["metadata_errors"])
    problem = RuntimeError("original fit failure")

    class FailedFit(_MatrixSpy):
        def fit_predict(self, *_):
            raise problem

    with pytest.raises(RuntimeError) as caught:
        _evaluate(FailedFit())
    assert caught.value is problem
    failure_environment = caught.value.analysis_diagnostics["environment"]
    assert failure_environment["packages"]["numpy"] is None
    assert "numpy" in str(failure_environment["metadata_errors"])
