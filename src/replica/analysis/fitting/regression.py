"""Fit and score scalar comparisons using resolved analysis settings."""

from __future__ import annotations

import logging
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, field, replace
from typing import (
    TYPE_CHECKING,
    cast,
)

import numpy as np

from replica.analysis.contracts.design import Design
from replica.analysis.contracts.events import ResponseEvents
from replica.analysis.contracts.grouped import (
    GroupedFrame,
    GroupedModel,
    validate_grouped_fit,
)
from replica.analysis.contracts.models import AnalysisModel, FoldScore, LikelihoodModel
from replica.analysis.contracts.results import (
    AnalysisResult,
    Coefficient,
    ComparisonSpec,
    FoldResult,
)
from replica.analysis.cv.group_holdout import CrossedGroupSplitter
from replica.analysis.cv.scalers import Scaler
from replica.analysis.cv.splitters import CVSplitter
from replica.analysis.data.events import build_response_events
from replica.analysis.data.observations import ObservationTable
from replica.analysis.data.preparation import validate_fold_dependencies
from replica.analysis.fitting.diagnostics import (
    emitted_warnings,
    execution_environment,
    group_summary,
    name_fit_columns,
    prediction_diagnostics,
    retained_predictions,
    snapshot_fit_diagnostics,
)
from replica.analysis.fitting.folds import (
    _append_grouping_columns,
    _average_feature_stat,
    _build_design_matrices,
    _cv_splits,
    _dump_design_matrix,
    _fit_baseline_predictions,
    _fit_full_predictions,
)
from replica.analysis.fitting.scoring import (
    _gaussian_log_likelihood,
    _gaussian_log_likelihood_pointwise,
    _r_squared,
)
from replica.analysis.fitting.transforms import (
    FoldContext,
    apply_fold_transforms,
)
from replica.analysis.specification.preprocessing import Residualize
from replica.contracts.features import Feature

logger = logging.getLogger("replica")


if TYPE_CHECKING:
    from replica.dataset import Dataset


@dataclass(frozen=True)
class _RegressionSetup:
    """Prepared observations and resolved choices for one target contrast."""

    dataset: Dataset
    observations: ObservationTable
    all_features: list[Feature]
    features: list[Feature]
    control_features: list[Feature]
    feature_model_name: str
    language: str | None
    analysis_model: AnalysisModel | LikelihoodModel | GroupedModel
    cv_splitter: CVSplitter
    scaler: Scaler
    use_training_sigma: bool
    per_fold: bool
    retain_per_obs_delta_ll: bool
    dump_merged_path: str | None
    empty_control_baseline: str = "mean"
    preprocessing: tuple[Residualize, ...] = ()
    score_scale: str = "transformed"
    target_transform: Callable | None = None
    declaration: Design | None = None
    default_cv: bool = True
    fixed_splits: list[tuple[np.ndarray, np.ndarray]] | None = None
    fixed_split_metadata: list[dict] | None = None
    prediction_diagnostics: bool = False
    retain_predictions: bool = False

    @property
    def target_layer(self):
        layers = {
            block.layer
            for values in self.observations.source_values.values()
            for block in values.blocks
        }
        if len(layers) != 1:
            raise ValueError("Continuous-time analysis requires one annotation layer")
        return next(iter(layers))

    @property
    def feature_arrays(self):
        arrays: dict[str, dict[str, np.ndarray]] = {}
        for feature in (*self.features, *self.control_features):
            arrays[feature.name] = {}
            for block in self.observations.source_values[feature.name].blocks:
                if block.participant_id is not None or block.values.ndim != 1:
                    raise ValueError(
                        "Continuous-time impulses require scalar stimulus features"
                    )
                sid = block.stimulus.stimulus_id
                if sid in arrays[feature.name]:
                    raise ValueError("Ambiguous stimulus impulse values")
                arrays[feature.name][sid] = block.values
        return arrays

    def flat_arrays(self, dv_name, *, maps_out, events_out):
        table = self.observations
        maps_out.update(
            participant_ids_map=table.participant_ids,
            item_ids_map=table.item_ids,
            row_exclusions=table.response["row_exclusions"],
        )
        if events_out is not None:
            inverse = {v: k for k, v in table.participant_ids.items()}
            events_out.update(
                unit_index=table.unit_indices,
                stimulus_id=table.stimulus_ids,
                participant_id=np.asarray([inverse[i] for i in table.participants]),
                factors=table.factors,
            )
        return (
            table.y,
            table.columns,
            table.participants,
            table.items,
            table.groups,
            table.sentence_ids,
        )


def _run_custom_analysis_for_dv(
    setup: _RegressionSetup, dv_name: str, analysis_fn: Callable
) -> list[AnalysisResult]:
    """Run a custom analysis function for one DV."""
    # The same DV outlier filter as the regression path, so row counts match
    # across both paths.
    maps: dict = {}
    y, feat_flat, participants, items, _groups, _sentence_ids = setup.flat_arrays(
        dv_name, maps_out=maps, events_out=None
    )

    X_features = {f.name: feat_flat[f.name] for f in setup.features}
    X_controls = {f.name: feat_flat[f.name] for f in setup.control_features}

    # Raw (un-re-coded) participant / item identifiers, so a custom analysis can
    # reproduce a paper's fold scheme keyed on the original ids (``items`` are
    # re-coded to a 0-based index and lose the original values).
    participant_ids = _raw_ids_from_map(maps, participants, "participant_ids_map")
    item_ids = _raw_ids_from_map(maps, items, "item_ids_map")

    return analysis_fn(
        y=y,
        X_features=X_features,
        X_controls=X_controls,
        participants=participants,
        items=items,
        participant_ids=participant_ids,
        item_ids=item_ids,
        dataset_name=setup.dataset.info.name,
        model_name=setup.feature_model_name,
        dv_name=dv_name,
    )


@dataclass
class _RegressionTotals:
    baseline_ll: float = 0.0
    full_ll: float = 0.0
    baseline_r2_weighted: float = 0.0
    full_r2_weighted: float = 0.0
    n_test: int = 0

    def add(
        self,
        *,
        baseline_ll: float,
        full_ll: float,
        baseline_r2: float,
        full_r2: float,
        n_test: int,
    ) -> None:
        self.baseline_ll += baseline_ll
        self.full_ll += full_ll
        self.baseline_r2_weighted += baseline_r2 * n_test
        self.full_r2_weighted += full_r2 * n_test
        self.n_test += n_test

    def normalized(self, n_obs: int) -> tuple[float, float, float, float, int]:
        """Normalize repeated-CV totals to one effective pass through the data."""
        coverage = self.n_test / n_obs if n_obs > 0 else 1.0
        if coverage <= 1.0 + 1e-6:
            return (
                self.baseline_ll,
                self.full_ll,
                self.baseline_r2_weighted,
                self.full_r2_weighted,
                self.n_test,
            )
        return (
            self.baseline_ll / coverage,
            self.full_ll / coverage,
            self.baseline_r2_weighted / coverage,
            self.full_r2_weighted / coverage,
            n_obs,
        )


def _validate_predictions(y: np.ndarray, *predictions: np.ndarray) -> None:
    for prediction in predictions:
        if not isinstance(prediction, np.ndarray) or prediction.shape != y.shape:
            raise ValueError(f"Model predictions must have shape {y.shape}")
        if not np.isfinite(prediction).all():
            raise ValueError("Model predictions must be finite on every scored row")


def _make_fold_result(
    *,
    fold_idx: int,
    n_test: int,
    y_test: np.ndarray,
    y_pred_base: np.ndarray,
    y_pred_full: np.ndarray,
    sigma_base: float | np.ndarray | None,
    sigma_full: float | np.ndarray | None,
    metadata: dict,
    retain_per_obs_delta_ll: bool,
    original_scale: tuple | None = None,
    retain_prediction_diagnostics: bool = False,
    retain_predictions: bool = False,
    n_train_overlap: int = 0,
) -> tuple[FoldResult, float, float, float, float]:
    """Build one FoldResult and return its aggregate metrics."""
    _validate_predictions(y_test, y_pred_base, y_pred_full)
    fold_base_ll = _gaussian_log_likelihood(y_test, y_pred_base, sigma=sigma_base)
    fold_full_ll = _gaussian_log_likelihood(y_test, y_pred_full, sigma=sigma_full)
    r2_y, r2_base, r2_full = y_test, y_pred_base, y_pred_full
    if original_scale is not None:
        raw, offset, inverse, jacobian = original_scale
        r2_y = raw
        r2_base, r2_full = (
            np.asarray(inverse(y_pred_base + offset)),
            np.asarray(inverse(y_pred_full + offset)),
        )
        _validate_predictions(raw, r2_base, r2_full)
        correction = np.asarray(jacobian(raw), dtype=float)
        if correction.shape != raw.shape or not np.isfinite(correction).all():
            raise ValueError(
                "Target log_abs_det_jacobian must return one finite value per original observation"
            )
        fold_base_ll += float(correction.sum())
        fold_full_ll += float(correction.sum())
    fold_base_r2 = _r_squared(r2_y, r2_base)
    fold_full_r2 = _r_squared(r2_y, r2_full)

    per_obs_delta_ll = None
    if retain_per_obs_delta_ll:
        base_pt = _gaussian_log_likelihood_pointwise(
            y_test, y_pred_base, sigma=sigma_base
        )
        full_pt = _gaussian_log_likelihood_pointwise(
            y_test, y_pred_full, sigma=sigma_full
        )
        per_obs_delta_ll = np.asarray(full_pt - base_pt, dtype=float)

    if retain_prediction_diagnostics:
        metadata = {
            **metadata,
            "prediction_diagnostics": prediction_diagnostics(
                y_test,
                y_pred_base,
                y_pred_full,
                sigma_base=sigma_base,
                sigma_full=sigma_full,
                normal_reference=True,
                n_train_overlap=n_train_overlap,
            ),
        }

    return (
        FoldResult(
            fold_index=fold_idx,
            n_test=n_test,
            baseline_ll=fold_base_ll,
            full_ll=fold_full_ll,
            delta_ll=fold_full_ll - fold_base_ll,
            baseline_r2=fold_base_r2,
            full_r2=fold_full_r2,
            delta_r2=fold_full_r2 - fold_base_r2,
            metadata=dict(metadata),
            per_obs_delta_ll=per_obs_delta_ll,
            observations=retained_predictions(y_test, y_pred_base, y_pred_full)
            if retain_predictions
            else None,
        ),
        fold_base_ll,
        fold_full_ll,
        fold_base_r2,
        fold_full_r2,
    )


def _raw_ids_from_map(maps: dict, encoded: np.ndarray, key: str) -> np.ndarray:
    """Recover raw ids from integer-encoded ones via a ``maps_out`` map."""
    inv = {c: raw for raw, c in maps.get(key, {}).items()}
    return np.array([inv.get(int(c)) for c in encoded], dtype=object)


def _unpack_fold_score(score):
    """Accept either ``(loglik, y_pred)`` or a :class:`FoldScore`."""
    if isinstance(score, FoldScore):
        return score.loglik, score.y_pred, score.per_obs_loglik
    loglik, y_pred = score
    return loglik, y_pred, None


def _score_owns_loglik_fold(
    model: LikelihoodModel,
    y_train,
    y_test,
    ctrl_designs,
    full_designs,
    *,
    fold_idx,
    n_test,
    per_fold,
    metadata,
    retain_per_obs_delta_ll=False,
    extra_kwargs=None,
    retain_prediction_diagnostics=False,
    retain_predictions=False,
    attempt=None,
    n_train_overlap=0,
) -> _FoldOutcome:
    """Score one fold with a model that owns its held-out likelihood.

    Retains each backend snapshot before the next fit replaces its state.
    """
    extra = dict(extra_kwargs or {})
    if attempt is not None:
        attempt["stage"] = "baseline_fit_and_score"
    base_ll, y_pred_base, base_per_obs = _unpack_fold_score(
        model.fit_score(
            y_train, ctrl_designs[0], ctrl_designs[1], y_test, is_full=False, **extra
        )
    )
    baseline_diagnostics = snapshot_fit_diagnostics(model)
    if attempt is not None:
        attempt["stage"] = "full_fit_and_score"
    full_ll, y_pred_full, full_per_obs = _unpack_fold_score(
        model.fit_score(
            y_train, full_designs[0], full_designs[1], y_test, is_full=True, **extra
        )
    )
    full_diagnostics = snapshot_fit_diagnostics(model)
    _validate_predictions(y_test, y_pred_base, y_pred_full)
    base_r2 = _r_squared(y_test, y_pred_base)
    full_r2 = _r_squared(y_test, y_pred_full)
    # The full design was fit last, so a reporting model reports that fit.
    reporter = getattr(model, "last_fixed_effects", None)
    beta, se = reporter() or (None, None) if reporter is not None else (None, None)
    fold_result = None
    if per_fold:
        if retain_prediction_diagnostics:
            metadata = {
                **metadata,
                "prediction_diagnostics": prediction_diagnostics(
                    y_test,
                    y_pred_base,
                    y_pred_full,
                    normal_reference=False,
                    n_train_overlap=n_train_overlap,
                ),
            }
        # A per-observation delta is available only when the model decomposes
        # its likelihood per observation; a per-group block likelihood cannot.
        per_obs_delta = None
        if (
            retain_per_obs_delta_ll
            and base_per_obs is not None
            and full_per_obs is not None
        ):
            per_obs_delta = np.asarray(full_per_obs) - np.asarray(base_per_obs)
        fold_result = FoldResult(
            fold_index=fold_idx,
            n_test=n_test,
            baseline_ll=base_ll,
            full_ll=full_ll,
            delta_ll=full_ll - base_ll,
            baseline_r2=base_r2,
            full_r2=full_r2,
            delta_r2=full_r2 - base_r2,
            metadata=metadata,
            per_obs_delta_ll=per_obs_delta,
            observations=retained_predictions(y_test, y_pred_base, y_pred_full)
            if retain_predictions
            else None,
        )
    return _FoldOutcome(
        fold_result,
        base_ll,
        full_ll,
        base_r2,
        full_r2,
        n_test,
        beta,
        se,
        {
            "fold_index": fold_idx,
            "baseline": baseline_diagnostics,
            "full": full_diagnostics,
        },
    )


@dataclass
class _Design:
    """One DV's regression inputs: rows, design matrices, folds, transforms."""

    y: np.ndarray
    feat_flat: dict[str, np.ndarray]
    participants: np.ndarray
    items: np.ndarray
    n_ctrl: int
    ctrl_names: list[str]
    X_ctrl_raw: np.ndarray
    X_full_raw: np.ndarray
    splits: list[tuple[np.ndarray, np.ndarray]]
    splitter_metadata: list[dict] | None
    ctrl_transforms: list
    full_transforms: list
    use_grouping: bool
    response_events: ResponseEvents | None
    factors: dict[str, np.ndarray]
    row_ids: np.ndarray | None
    split_warnings: list[str]
    conditioning_warnings: list[str]
    row_exclusions: dict
    fold_attempts: list[dict] = field(default_factory=list)


@dataclass(frozen=True)
class _FoldOutcome:
    """What one fitted fold contributes to the totals."""

    fold_result: FoldResult | None
    baseline_ll: float
    full_ll: float
    baseline_r2: float
    full_r2: float
    n_test: int
    beta: np.ndarray | None = None
    se: np.ndarray | None = None
    diagnostics: dict = field(default_factory=dict)


def _event_measure(setup, target_name):
    """Resolve a native response timeline without rebinding foreign observations."""
    from replica.core import Datum

    values = setup.observations.source_values[target_name]
    pending = [values.provenance]
    bindings = []
    while pending:
        source = pending.pop()
        if "dataset_binding" in source:
            bindings.append(
                source["dataset_binding"] == source.get("selected_dataset_binding")
            )
        pending.extend(source.get("dependencies", ()))
    if not bindings or not all(bindings):
        raise ValueError("Target uses a different dataset event timeline")
    if any(not isinstance(block.source, Datum) for block in values.blocks):
        raise ValueError(
            "Continuous-time targets require a native datum event timeline"
        )
    measures = {cast(Datum, block.source).measure_name for block in values.blocks}
    if len(measures) != 1:
        raise ValueError("Continuous-time targets require one native response timeline")
    return next(iter(measures))


def _build_design(setup: _RegressionSetup, dv_name: str) -> _Design | None:
    """Flat arrays, design matrices, folds and per-fold transforms for one DV.

    ``None`` when there is no predictor to evaluate: design matrices cannot
    stack an empty feature set.
    """
    maps: dict = {}
    # Continuous-time models are scored against an impulse stream that does not
    # share the flat rows' row space, so they ask for the event channel and the
    # builder additionally reports each row's stimulus and unit.
    model = setup.analysis_model
    wants_events = getattr(model, "needs_events", False)
    rows: dict = {}
    y, feat_flat, participants, items, groups, sentence_ids = setup.flat_arrays(
        dv_name,
        maps_out=maps,
        events_out=rows
        if wants_events or setup.declaration is not None or setup.retain_predictions
        else None,
    )
    observations = setup.observations
    y, feat_flat = observations.y, observations.columns
    if not setup.features:
        return None

    ctrl_names, X_ctrl_raw, feat_cols, X_full_raw, n_ctrl = _build_design_matrices(
        feat_flat, setup.features, setup.control_features, len(y)
    )

    # Raw subject ids let sentence-based splitters reproduce a study's own
    # np.unique-coded (subject, sentence) fold partition.
    cv_groups = groups
    if setup.declaration is not None and (
        setup.default_cv or getattr(setup.cv_splitter, "uses_declared_factors", False)
    ):
        columns = [
            np.unique(observations.factors[unit], return_inverse=True)[1]
            for unit in setup.declaration.cv_units
        ]
        if getattr(setup.cv_splitter, "uses_named_factors", False):
            cv_groups = dict(zip(setup.declaration.cv_units, columns, strict=True))
        else:
            cv_groups = columns[0] if len(columns) == 1 else np.column_stack(columns)
    splits = (
        setup.fixed_splits
        if setup.fixed_splits is not None
        else _cv_splits(
            setup.cv_splitter,
            y,
            cv_groups,
            participants,
            items,
            sentence_ids,
            participant_ids=_raw_ids_from_map(
                maps, participants, "participant_ids_map"
            ),
        )
    )
    validate_fold_dependencies(
        observations,
        splits,
        in_sample=getattr(setup.cv_splitter, "in_sample", False),
        declaration=setup.declaration,
    )
    split_warnings = (
        []
        if setup.declaration is None
        else setup.declaration.check_splits(splits, observations.factors)
    )
    conditioning_warnings = (
        []
        if setup.declaration is None
        else setup.declaration.for_model(
            tuple([*ctrl_names, *(f.name for f in setup.features)]), warn=False
        ).check_conditioning(splits, observations.factors)
    )
    splitter_metadata: list[dict] | None = setup.fixed_split_metadata
    if setup.fixed_splits is None and hasattr(setup.cv_splitter, "fold_metadata"):
        splitter_metadata = setup.cv_splitter.fold_metadata()
    if isinstance(setup.cv_splitter, CrossedGroupSplitter):
        summary = setup.cv_splitter.summary()
        logger.info(
            "Crossed CV: %s folds; %d fits/model/cycle; training-row fraction %s",
            summary["resolved_folds"],
            summary["fits_per_model_per_cycle"],
            summary["train_fraction_rows_range"],
        )

    _dump_design_matrix(
        setup.dump_merged_path,
        ctrl_names=ctrl_names,
        X_ctrl_raw=X_ctrl_raw,
        features=setup.features,
        feat_cols=feat_cols,
        participants=participants,
        items=items,
        y=y,
    )

    response_events = (
        build_response_events(
            dataset=setup.dataset,
            dv_name=_event_measure(setup, dv_name),
            rows=rows,
            feature_arrays=setup.feature_arrays,
            control_names=tuple(f.name for f in setup.control_features),
            focal_names=tuple(f.name for f in setup.features),
            y=y,
            group=participants,
            layer=setup.target_layer,
        )
        if wants_events
        else None
    )
    return _Design(
        y=y,
        feat_flat=feat_flat,
        participants=participants,
        items=items,
        n_ctrl=n_ctrl,
        ctrl_names=ctrl_names,
        X_ctrl_raw=X_ctrl_raw,
        X_full_raw=X_full_raw,
        splits=splits,
        splitter_metadata=splitter_metadata,
        ctrl_transforms=[setup.scaler],
        full_transforms=[setup.scaler],
        use_grouping=setup.declaration is None
        and getattr(model, "needs_grouping", False),
        factors=observations.factors,
        row_ids=np.column_stack(
            [
                rows["participant_id"],
                rows["stimulus_id"],
                rows["unit_index"].astype(str),
            ]
        )
        if setup.declaration is not None or setup.retain_predictions
        else None,
        split_warnings=split_warnings,
        conditioning_warnings=conditioning_warnings,
        row_exclusions=maps["row_exclusions"],
        response_events=response_events,
    )


def _fold_matrices(setup, design, train_mask, test_mask):
    """Fit declared projections once, then scale each statistical design."""
    columns = dict(design.feat_flat)
    target_name = setup.observations.response["target"]
    columns[target_name] = design.y
    fitted = []
    for step in setup.preprocessing:
        if step.feature.name not in columns:
            continue
        before = columns[step.feature.name]
        train, test, state = step.apply(
            before,
            [columns[f.name] for f in step.against],
            train_mask,
            test_mask,
        )
        if not np.isfinite(train).all() or not np.isfinite(test).all():
            raise ValueError(
                "Preprocessing produced nonfinite values; rows cannot be dropped after splitting"
            )
        transformed = before.copy()
        transformed[train_mask], transformed[test_mask] = train, test
        columns[step.feature.name] = transformed
        fitted.append({**step.configuration(), "fitted": state})
    ctx = FoldContext(train_mask, test_mask, columns)
    controls = [columns[name] for name in design.ctrl_names]
    x_ctrl = np.column_stack(controls) if controls else np.empty((len(design.y), 0))
    x_full = np.column_stack([*controls, *(columns[f.name] for f in setup.features)])
    if controls:
        ctrl_train, ctrl_test = apply_fold_transforms(
            x_ctrl, design.ctrl_transforms, ctx
        )
        control_scaling = snapshot_fit_diagnostics(setup.scaler)
    else:
        control_scaling = None
        ctrl_train = np.empty((int(train_mask.sum()), 0))
        ctrl_test = np.empty((int(test_mask.sum()), 0))
    full_train, full_test = apply_fold_transforms(x_full, design.full_transforms, ctx)
    scaling = {
        "baseline": {"columns": list(design.ctrl_names), "fitted": control_scaling},
        "full": {
            "columns": [*design.ctrl_names, *(f.name for f in setup.features)],
            "fitted": snapshot_fit_diagnostics(setup.scaler),
        },
    }
    if design.use_grouping:
        full_train, full_test = _append_grouping_columns(
            full_train,
            full_test,
            train_mask,
            test_mask,
            design.participants,
            design.items,
        )
        if design.n_ctrl or setup.empty_control_baseline == "model":
            ctrl_train, ctrl_test = _append_grouping_columns(
                ctrl_train,
                ctrl_test,
                train_mask,
                test_mask,
                design.participants,
                design.items,
            )
    return (
        ctrl_train,
        ctrl_test,
        full_train,
        full_test,
        columns[target_name],
        fitted,
        scaling,
    )


def _original_scale(setup, fold_y, test):
    if setup.score_scale != "original":
        return None
    from replica.analysis.specification.config import _identity_response

    transform = setup.target_transform
    inverse = (
        _identity_response if transform is _identity_response else transform.inverse
    )
    jacobian = (
        np.zeros_like
        if transform is _identity_response
        else transform.log_abs_det_jacobian
    )
    raw = setup.observations.raw_y
    if raw is None:
        raise ValueError("Original-scale scoring requires retained raw target values")
    return raw[test], setup.observations.y[test] - fold_y[test], inverse, jacobian


def _fit_fold(
    setup: _RegressionSetup,
    design: _Design,
    fold_idx: int,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    attempt: dict | None = None,
) -> _FoldOutcome:
    """Fit the baseline and full models on one fold and score the held-out rows."""
    n_test = int(np.sum(test_mask))
    if attempt is not None:
        attempt["stage"] = "preprocessing"
    X_ctrl_train, X_ctrl_test, X_full_train, X_full_test, y, transforms, scaling = (
        _fold_matrices(setup, design, train_mask, test_mask)
    )
    y_train, y_test = y[train_mask], y[test_mask]
    design = replace(design, y=y)
    meta = (
        design.splitter_metadata[fold_idx]
        if design.splitter_metadata is not None
        else {}
    )
    meta = {
        **meta,
        "preprocessing": transforms,
        "scaling": scaling,
        "n_train": int(train_mask.sum()),
        "n_test": n_test,
    }
    model = setup.analysis_model
    if setup.per_fold:
        meta["groups"] = group_summary(
            design.factors
            or {"participant": design.participants, "stimulus": design.items},
            train_mask,
            test_mask,
        )
    if setup.declaration is not None:
        return _fit_named_fold(
            setup,
            design,
            fold_idx,
            train_mask,
            test_mask,
            (X_ctrl_train, X_ctrl_test, X_full_train, X_full_test),
            dict(meta),
            attempt=attempt,
        )

    if getattr(model, "owns_loglik", False):
        # Models supply either pointwise predictive scores or group-block
        # likelihoods, bypassing the default plug-in Gaussian scorer.
        if design.use_grouping and design.n_ctrl == 0:
            # An intercept-only baseline still needs its random effects.
            X_ctrl_train, X_ctrl_test = _append_grouping_columns(
                X_ctrl_train,
                X_ctrl_test,
                train_mask,
                test_mask,
                design.participants,
                design.items,
            )
        extra = (
            {}
            if design.response_events is None
            else {
                "events": design.response_events,
                "train_mask": train_mask,
                "test_mask": test_mask,
            }
        )
        return _score_owns_loglik_fold(
            cast(LikelihoodModel, model),
            y_train,
            y_test,
            (X_ctrl_train, X_ctrl_test),
            (X_full_train, X_full_test),
            fold_idx=fold_idx,
            n_test=n_test,
            per_fold=setup.per_fold,
            metadata=meta,
            retain_per_obs_delta_ll=setup.retain_per_obs_delta_ll,
            extra_kwargs=extra,
            retain_prediction_diagnostics=setup.prediction_diagnostics,
            retain_predictions=setup.retain_predictions,
            attempt=attempt,
            n_train_overlap=int(np.sum(train_mask & test_mask))
            if setup.prediction_diagnostics
            else 0,
        )

    prediction_model = cast(AnalysisModel, model)
    if attempt is not None:
        attempt["stage"] = "baseline_fit_and_predict"
    y_pred_base, sigma_base = _fit_baseline_predictions(
        prediction_model,
        y_train,
        n_test,
        X_ctrl_train,
        X_ctrl_test,
        use_training_sigma=setup.use_training_sigma,
        force_model=setup.empty_control_baseline == "model",
    )
    baseline_diagnostics = (
        {
            "method": "analytic_mean",
            "convergence_applicable": False,
            "rank": 1,
            "n_columns": 1,
            "coefficients": {
                "names": ["(Intercept)"],
                "estimates": [float(y_train.mean())],
                "naming": "design_columns",
            },
            "residual_scale": {
                "sigma": sigma_base,
                "variance": None if sigma_base is None else sigma_base**2,
                "convention": "sample_training_sd"
                if sigma_base is not None
                else "not_retained",
            },
        }
        if X_ctrl_train.shape[1] == 0 and setup.empty_control_baseline == "mean"
        else snapshot_fit_diagnostics(model)
    )
    if attempt is not None:
        attempt["stage"] = "full_fit_and_predict"
    y_pred_full, beta, se, sigma_full = _fit_full_predictions(
        prediction_model,
        y_train,
        X_full_train,
        X_full_test,
        use_training_sigma=setup.use_training_sigma,
    )
    full_diagnostics = name_fit_columns(
        snapshot_fit_diagnostics(model),
        [*design.ctrl_names, *(f.name for f in setup.features)],
    )
    baseline_diagnostics = name_fit_columns(
        baseline_diagnostics, list(design.ctrl_names)
    )
    fold_result, fb, ff, br2, fr2 = _make_fold_result(
        fold_idx=fold_idx,
        n_test=n_test,
        y_test=y_test,
        y_pred_base=y_pred_base,
        y_pred_full=y_pred_full,
        sigma_base=sigma_base,
        sigma_full=sigma_full,
        metadata=meta,
        retain_per_obs_delta_ll=setup.per_fold and setup.retain_per_obs_delta_ll,
        original_scale=_original_scale(setup, design.y, test_mask),
        retain_prediction_diagnostics=setup.prediction_diagnostics,
        retain_predictions=setup.retain_predictions,
        n_train_overlap=int(np.sum(train_mask & test_mask))
        if setup.prediction_diagnostics
        else 0,
    )
    return _FoldOutcome(
        fold_result if setup.per_fold else None,
        fb,
        ff,
        br2,
        fr2,
        n_test,
        beta,
        se,
        {
            "fold_index": fold_idx,
            "baseline": baseline_diagnostics,
            "full": full_diagnostics,
        },
    )


def _fit_named_fold(
    setup: _RegressionSetup,
    design: _Design,
    fold_idx: int,
    train: np.ndarray,
    test: np.ndarray,
    matrices: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    metadata: dict,
    attempt: dict | None = None,
) -> _FoldOutcome:
    """Score a named model comparison while retaining row and factor alignment."""
    ctrl_train, ctrl_test, full_train, full_test = matrices
    names = [*design.ctrl_names, *(f.name for f in setup.features)]
    declaration = setup.declaration
    model = setup.analysis_model
    assert declaration is not None and isinstance(model, GroupedModel)
    assert design.row_ids is not None

    def fit(is_full: bool):
        if attempt is not None:
            attempt["stage"] = (
                "full_fit_and_predict" if is_full else "baseline_fit_and_predict"
            )
        fixed = tuple(names if is_full else design.ctrl_names)
        # An unspecified comparison has already warned before feature extraction.
        resolved = declaration.for_model(fixed, warn=False)
        frames = []
        for mask, full, ctrl in (
            (train, full_train, ctrl_train),
            (test, full_test, ctrl_test),
        ):
            columns = dict(zip(names, full.T, strict=True))
            columns.update(zip(design.ctrl_names, ctrl.T, strict=True))
            frames.append(
                GroupedFrame(
                    columns,
                    {n: v[mask] for n, v in design.factors.items()},
                    fixed,
                    resolved,
                )
            )
        fitted = model.fit_grouped(design.y[train], frames[0], frames[1])
        diagnostics = validate_grouped_fit(fitted, int(test.sum()), len(fixed))
        return replace(
            fitted,
            metadata={
                **{
                    key: value
                    for key, value in fitted.metadata.items()
                    if key != "fit_diagnostics"
                },
                "diagnostics": diagnostics,
                "design": frames[0].design.metadata(),
            },
        )

    baseline, full = fit(False), fit(True)
    metadata.update(
        design=declaration.metadata(),
        split_warnings=design.split_warnings,
        conditioning_warnings=design.conditioning_warnings,
        baseline=baseline.metadata,
        full=full.metadata,
        sigma_source="training" if setup.use_training_sigma else "test",
    )
    result, fb, ff, br2, fr2 = _make_fold_result(
        fold_idx=fold_idx,
        n_test=int(test.sum()),
        y_test=design.y[test],
        y_pred_base=baseline.predictions,
        y_pred_full=full.predictions,
        sigma_base=baseline.scoring_sigma(declaration.scoring)
        if setup.use_training_sigma
        else None,
        sigma_full=full.scoring_sigma(declaration.scoring)
        if setup.use_training_sigma
        else None,
        metadata=metadata,
        retain_per_obs_delta_ll=setup.per_fold and setup.retain_per_obs_delta_ll,
        original_scale=_original_scale(setup, design.y, test),
        retain_prediction_diagnostics=setup.prediction_diagnostics,
        retain_predictions=setup.retain_predictions,
        n_train_overlap=int(np.sum(train & test))
        if setup.prediction_diagnostics
        else 0,
    )
    if result.per_obs_delta_ll is not None:
        result.factor_ids = {n: v[test].copy() for n, v in design.factors.items()}
        result.row_ids = design.row_ids[test].copy()
    return _FoldOutcome(
        result if setup.per_fold else None,
        fb,
        ff,
        br2,
        fr2,
        int(test.sum()),
        full.beta,
        full.se,
        {
            "fold_index": fold_idx,
            "baseline": baseline.metadata["diagnostics"],
            "full": full.metadata["diagnostics"],
        },
    )


def _fit_folds(setup: _RegressionSetup, design: _Design) -> list[_FoldOutcome]:
    """Fit every fold with rows on both sides, in the splitter's order."""
    outcomes: list[_FoldOutcome] = []
    n_folds = len(design.splits)
    for fold_idx, (train_mask, test_mask) in enumerate(design.splits):
        attempt = {
            "fold_index": fold_idx,
            "status": "started",
            "n_train": int(train_mask.sum()),
            "n_test": int(test_mask.sum()),
        }
        design.fold_attempts.append(attempt)
        logger.info(
            "    fold %d/%d  (train=%d, test=%d)",
            fold_idx + 1,
            n_folds,
            int(np.sum(train_mask)),
            int(np.sum(test_mask)),
        )
        if np.sum(train_mask) == 0 or np.sum(test_mask) == 0:
            attempt.update(
                status="skipped",
                reason="empty_training" if not train_mask.any() else "empty_scoring",
                warnings=[],
            )
            logger.warning("Ignoring empty train/test partition in fold %d", fold_idx)
            continue
        with emitted_warnings() as captured:
            try:
                outcome = _fit_fold(
                    setup, design, fold_idx, train_mask, test_mask, attempt
                )
                fold = outcome.fold_result
                if fold is not None and setup.retain_predictions:
                    assert fold.observations is not None and design.row_ids is not None
                    selected = np.flatnonzero(test_mask)
                    fold.observations.update(
                        in_train=train_mask[test_mask].copy(),
                        prepared_row_index=selected,
                    )
                    fold.row_ids = design.row_ids[test_mask].copy()
                    fold.factor_ids = {
                        name: values[test_mask].copy()
                        for name, values in design.factors.items()
                    }
                    fold.observation_keys = tuple(
                        setup.observations.row_keys[i] for i in selected
                    )
                    fold.metadata["observation_capture"] = {
                        "response_scale": "model_after_transform_and_fold_preprocessing",
                        "population": "scored_rows",
                        "row_id_columns": ["participant", "stimulus", "unit"],
                        "row_key_names": setup.observations.response.get(
                            "observation_index_names"
                        ),
                    }
                attempt.update(
                    status="completed", stage="completed", warnings=list(captured)
                )
                outcomes.append(outcome)
            except Exception as error:
                attempt.update(
                    status="failed",
                    warnings=list(captured),
                    error={
                        "type": f"{type(error).__module__}.{type(error).__qualname__}",
                        "message": str(error),
                    },
                )
                record = {
                    "status": "failed",
                    "dataset": setup.dataset.info.name,
                    "target": setup.observations.response.get("target"),
                    "comparison": {
                        "baseline": list(design.ctrl_names),
                        "added": [f.name for f in setup.features],
                    },
                    "fold_attempts": deepcopy(design.fold_attempts),
                    "environment": execution_environment(),
                }
                BaseException.__setattr__(error, "analysis_diagnostics", record)
                raise
    return outcomes


def _assemble_analyses(
    setup: _RegressionSetup,
    dv_name: str,
    design: _Design,
    outcomes: list[_FoldOutcome],
) -> list[AnalysisResult]:
    """Total the folds into one :class:`AnalysisResult` per joint comparison."""
    totals = _RegressionTotals()
    for o in outcomes:
        totals.add(
            baseline_ll=o.baseline_ll,
            full_ll=o.full_ll,
            baseline_r2=o.baseline_r2,
            full_r2=o.full_r2,
            n_test=o.n_test,
        )
    all_betas = [o.beta for o in outcomes if o.beta is not None]
    all_ses = [o.se for o in outcomes if o.se is not None]
    fold_results_list = [o.fold_result for o in outcomes if o.fold_result is not None]

    return [
        _make_regression_analysis(
            totals=totals,
            n_obs=len(design.y),
            ctrl_names=design.ctrl_names
            + (
                [] if setup.declaration is None else list(setup.declaration.categorical)
            ),
            features=setup.features,
            all_betas=all_betas,
            all_ses=all_ses,
            n_ctrl=design.n_ctrl,
            dataset_name=setup.dataset.info.name,
            feature_model_name=setup.feature_model_name,
            dv_name=dv_name,
            language=setup.language,
            fold_results_list=fold_results_list,
            per_fold=setup.per_fold,
        )
    ]


def _run_regression_for_dv(
    setup: _RegressionSetup, dv_name: str, *, design: _Design | None = None
) -> list[AnalysisResult]:
    """Cross-validated regression for one target: prepare, fit folds and total."""
    if design is None:
        design = _build_design(setup, dv_name)
    if design is None:
        return []
    outcomes = _fit_folds(setup, design)
    results = _assemble_analyses(setup, dv_name, design, outcomes)
    for result in results:
        result.metadata["row_exclusions"] = design.row_exclusions
        result.metadata["coefficient_uncertainty"] = "mean_within_fit_standard_error"
        result.metadata["fit_diagnostics"] = [o.diagnostics for o in outcomes]
        result.metadata["fold_attempts"] = deepcopy(design.fold_attempts)
    if setup.declaration is not None:
        for result in results:
            result.metadata["split_warnings"] = design.split_warnings
            result.metadata["conditioning_warnings"] = design.conditioning_warnings
    return results


def _make_regression_analysis(
    *,
    totals: _RegressionTotals,
    n_obs: int,
    ctrl_names: list[str],
    features: list[Feature],
    all_betas: list[np.ndarray],
    all_ses: list[np.ndarray],
    n_ctrl: int,
    dataset_name: str,
    feature_model_name: str,
    dv_name: str,
    language: str | None,
    fold_results_list: list[FoldResult],
    per_fold: bool,
) -> AnalysisResult:
    if totals.n_test == 0:
        raise ValueError(
            "No observations were scored; check the CV splits and number of stimulus groups"
        )
    (
        baseline_ll_total,
        full_ll_total,
        baseline_r2_total,
        full_r2_total,
        n_test_total,
    ) = totals.normalized(n_obs)

    delta_ll = full_ll_total - baseline_ll_total

    baseline_r2 = baseline_r2_total / n_test_total if n_test_total > 0 else 0.0
    full_r2 = full_r2_total / n_test_total if n_test_total > 0 else 0.0
    delta_r2 = full_r2 - baseline_r2

    # A likelihood belongs to the whole added set, while slopes name terms.
    comparison = ComparisonSpec(tuple(ctrl_names), tuple(f.name for f in features))
    coefficients = {
        feat.name: Coefficient(
            _average_feature_stat(all_betas, 1 + n_ctrl + i),
            _average_feature_stat(all_ses, 1 + n_ctrl + i),
        )
        for i, feat in enumerate(features)
    }
    first = coefficients[features[0].name]

    return AnalysisResult(
        dataset_name=dataset_name,
        model_name=feature_model_name,
        feature_name=comparison.name,
        dv_name=dv_name,
        metrics={
            "baseline_ll": baseline_ll_total,
            "full_ll": full_ll_total,
            "delta_ll": delta_ll,
            # Per held-out observation: divide by the rows that
            # contributed to delta_ll (== n_obs for full-coverage CV,
            # smaller for a partial-coverage FixedSplitter fold).
            "delta_ll_per_unit": (delta_ll / n_test_total if n_test_total > 0 else 0.0),
            "baseline_r2": baseline_r2,
            "full_r2": full_r2,
            "delta_r2": delta_r2,
        },
        n_observations=n_obs,
        n_scored=n_test_total,
        comparison=comparison,
        coefficients=coefficients,
        coefficient=first.estimate,
        coefficient_se=first.standard_error,
        language=language,
        fold_results=fold_results_list if per_fold else None,
    )
