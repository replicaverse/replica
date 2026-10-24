"""Full-refit crossed bootstrap for predictive comparisons on new groups."""

from __future__ import annotations

import hashlib
import inspect
import json
import logging
import math
import multiprocessing
import pickle
import platform
import tempfile
from collections.abc import Callable, Iterator, Mapping
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from importlib.metadata import version
from numbers import Integral
from pathlib import Path

import numpy as np

from replica.analysis.contracts.design import Design
from replica.analysis.contracts.grouped import (
    GroupedFrame,
    GroupedModel,
    validate_grouped_fit,
)
from replica.analysis.contracts.labels import categorical_labels
from replica.analysis.contracts.results import ComparisonSpec
from replica.analysis.cv.group_holdout import CrossedGroupSplitter
from replica.analysis.data.observations import ObservationTable
from replica.analysis.data.preparation import (
    _input_holdout_origins,
    validate_fold_origins,
)
from replica.analysis.fitting.scoring import _gaussian_log_likelihood_pointwise
from replica.analysis.fitting.transforms import (
    FoldContext,
    _ResidualizeColumns,
    apply_fold_transforms,
)
from replica.analysis.specification.config import AnalysisConfig
from replica.analysis.specification.preprocessing import _apply_projection

from ...paths import user_cache_root
from ..cv.scalers import ZScoreScaler
from .crossed_statistics import METHOD, SERIES, PredictiveTestResult, _summarize

logger = logging.getLogger("replica")
PROTOCOL = "crossed-cv-full-refit-v1"


@dataclass(frozen=True)
class CrossedBootstrapConfig:
    """Budgets for explicitly requested crossed predictive inference.

    Args:
        n_resamples: Draws in each of three series. The default 500 rounds the
            499-draw assessment budget; it is not a universal precision guarantee.
        cycles_per_draw: Independent complete CV partitions of each resample.
        original_cycles: Independent complete partitions of the original data.
        seed: Reproducible streams keyed by draw and cycle, independent of workers.
        n_jobs: Independent worker processes. One avoids implicit parallel jobs.
        confidence_level: Pointwise interval coverage requested from the t-reference.
    """

    n_resamples: int = 500
    cycles_per_draw: int = 2
    original_cycles: int = 8
    seed: int = 42
    n_jobs: int = 1
    confidence_level: float = 0.95

    def __post_init__(self):
        for name, minimum in (
            ("n_resamples", 2),
            ("cycles_per_draw", 2),
            ("original_cycles", 2),
            ("seed", 0),
            ("n_jobs", 1),
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, Integral)
                or value < minimum
            ):
                raise ValueError(f"{name} must be an integer >= {minimum}")
            object.__setattr__(self, name, int(value))
        if isinstance(self.confidence_level, bool) or not 0 < self.confidence_level < 1:
            raise ValueError("confidence_level must lie strictly between zero and one")

    def fit_budget(self, folds_per_cycle: int) -> int:
        """Return the two-model fit bound before retries or empty test folds.

        Args:
            folds_per_cycle: Product of the resolved counts on the two axes.
        """
        if (
            isinstance(folds_per_cycle, bool)
            or not isinstance(folds_per_cycle, Integral)
            or folds_per_cycle < 1
        ):
            raise ValueError("folds_per_cycle must be a positive integer")
        return int(
            2
            * folds_per_cycle
            * (self.original_cycles + 3 * self.n_resamples * self.cycles_per_draw)
        )


@dataclass
class _Context:
    y: np.ndarray
    columns: dict
    factors: dict
    codes: dict
    axes: tuple[str, ...]
    group_counts: tuple[int, ...]
    folds: tuple[int, ...]
    dependencies: dict
    comparison: ComparisonSpec
    design: Design
    model: GroupedModel
    scaler: object
    residualization: dict
    settings: CrossedBootstrapConfig
    cv_seed: int
    preview: dict
    runtime: dict
    preprocessing: tuple[dict, ...]
    target_name: str
    target_origins: tuple
    input_origins: dict
    holdout_origins: dict


def _runtime(model):
    """Record the numerical libraries used by fitting and inference."""
    from replica.analysis.models.mixed_effects.lmer import RLmerModel
    from replica.analysis.models.support.r import _r_context

    runtime = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": version("scipy"),
        "backend": f"{type(model).__module__}.{type(model).__qualname__}",
    }
    if isinstance(model, RLmerModel):
        with _r_context() as ro:
            runtime["R"] = str(ro.r("R.version.string")[0])
            runtime["lme4"] = str(ro.r('as.character(packageVersion("lme4"))')[0])
    return runtime


def _validate_analysis(analysis):
    """Reject unsupported targets before feature extraction or fitting."""
    if not isinstance(analysis, AnalysisConfig) or analysis.design is None:
        raise ValueError(
            "Predictive inference requires AnalysisConfig with a named Design"
        )
    design = analysis.design
    if len(design.cv_units) != 2 or design.condition_on:
        raise ValueError(
            "Default predictive inference requires two unseen CV axes and no conditioning"
        )
    if design.scoring != "marginal" or analysis.sigma_source != "training":
        raise ValueError(
            "The assessed predictive test requires marginal scoring from training fits"
        )
    if design.comparison not in (None, "predictor"):
        raise ValueError(
            "Predictive inference requires comparison='predictor' (complete removal)"
        )
    if design.permutation_unit is not None:
        raise ValueError(
            "A permutation_unit conflicts with the crossed bootstrap; set it to None"
        )
    if analysis.analysis_fn is not None:
        raise ValueError("Custom analysis_fn is not supported by full-refit inference")
    if not isinstance(analysis.model, GroupedModel):
        raise ValueError(
            "Select a named Gaussian model, such as RLmerModel(reml=False)"
        )
    if getattr(analysis.model, "reml", False):
        raise ValueError("The assessed predictive test uses ML, not REML")
    analysis.model.validate_design(design)
    splitter = analysis.resolve_cv()
    if type(splitter) is not CrossedGroupSplitter:
        raise ValueError("Full-refit inference requires CrossedGroupSplitter")
    return design, splitter


def _preprocessing_plan(
    analysis, observations, comparison, residualization, target_name
):
    """Validate detached preprocessing against the comparison and context columns."""
    preprocessing = tuple(step.configuration() for step in analysis.preprocessing)
    fixed = {*comparison.baseline, *comparison.added, target_name}
    available = {*observations.columns, target_name}
    for step in preprocessing:
        if step["feature"] not in fixed or not set(step["against"]) <= available:
            raise ValueError(
                "Preprocessing names must belong to this comparison and its context"
            )
        if step["feature"] == target_name and set(step["against"]) & set(
            comparison.added
        ):
            raise ValueError("Target preprocessing cannot depend on focal predictors")
    if set(residualization) & {step["feature"] for step in preprocessing}:
        raise ValueError(
            "Residualization cannot be declared through both APIs for one column"
        )
    return preprocessing


def _target_name(observations):
    """Choose an unambiguous target column for fold preprocessing."""
    if "target" in observations.response:
        target_name = observations.response["target"]
        if not isinstance(target_name, str) or not target_name:
            raise ValueError("Target metadata must supply a nonempty name")
        if target_name in observations.columns:
            raise ValueError("Target name conflicts with an input column")
    else:
        target_name = "__target__"
        while target_name in observations.columns:
            target_name += "_"
    return target_name


def _context(observations, comparison, analysis, settings, residualization):
    """Freeze aligned inputs and validate random-factor nesting."""
    if isinstance(analysis, AnalysisConfig):
        analysis = analysis.resolve(comparison.added)
    design, splitter = _validate_analysis(analysis)
    fixed = (*comparison.baseline, *comparison.added)
    if not comparison.added or len(set(fixed)) != len(fixed):
        raise ValueError(
            "Supply distinct baseline terms and a nonempty added predictor family"
        )
    missing = set(fixed) - observations.columns.keys()
    if missing:
        raise ValueError(f"Missing comparison columns: {sorted(missing)}")
    if not len(observations.y):
        raise ValueError("No eligible observations")
    full_design = design.for_model(fixed)
    design = replace(design, comparison=full_design.comparison)
    if set(full_design.random_effects) != set(design.random_effects):
        raise ValueError("All declared random slopes must belong to this comparison")
    factors = {
        f.name: categorical_labels(observations.factors[f.name], name=f.name)
        for f in design.factors
    }
    codes = {k: np.unique(v, return_inverse=True)[1] for k, v in factors.items()}
    axes = design.cv_units
    counts = tuple(len(np.unique(codes[n])) for n in axes)
    if min(counts) < 2:
        raise ValueError("Each held-out axis needs at least two original groups")
    dependencies: dict[str, tuple[int, ...]] = {}
    for effect in design.random_effects:
        name = effect.factor
        if name in axes:
            dependencies[name] = (axes.index(name),)
            continue
        group = codes[name]
        dependencies[name] = tuple(
            i
            for i, axis in enumerate(axes)
            if len(np.unique(np.column_stack([group, codes[axis]]), axis=0))
            == len(np.unique(group))
        )
        if not dependencies[name]:
            raise ValueError(
                f"Random factor {name!r} is not nested within either held-out axis; "
                "its bootstrap copy policy needs an explicit extension"
            )
    if set(design.categorical) & set(dependencies):
        raise ValueError("A copied random factor cannot also supply fixed categories")
    groups = {n: codes[n] for n in axes}
    preview = deepcopy(splitter).preview(groups)
    plan = dict(residualization or {})
    if set(plan) - set(fixed) or set(plan.values()) - observations.columns.keys():
        raise ValueError(
            "Residualization names must refer to supplied comparison/context columns"
        )
    target_name = _target_name(observations)
    preprocessing = _preprocessing_plan(
        analysis, observations, comparison, plan, target_name
    )
    # Only equality of original contributors is needed in workers/checkpoints.
    # Deterministic labels retain that relation without storing source objects.
    identities: dict[object, int] = {}

    def canonical(rows):
        return tuple(
            frozenset(
                identities.setdefault(origin, len(identities))
                for origin in sorted(row, key=str)
            )
            for row in rows
        )

    target_origins = canonical(observations.target_origins)
    input_origins = {
        name: canonical(rows) for name, rows in observations.input_origins.items()
    }
    holdout_origins = {
        name: canonical(rows)
        for name, rows in _input_holdout_origins(observations, design).items()
    }
    return _Context(
        observations.y.copy(),
        {k: v.copy() for k, v in observations.columns.items()},
        factors,
        codes,
        axes,
        counts,
        tuple(preview["resolved_folds"][n] for n in axes),
        dependencies,
        comparison,
        design,
        _refitting_model(analysis.model),
        deepcopy(analysis.scaler) if analysis.scaler is not None else ZScoreScaler(),
        plan,
        settings,
        splitter.seed or 0,
        preview,
        _runtime(analysis.model),
        preprocessing,
        target_name,
        target_origins,
        input_origins,
        holdout_origins,
    )


def _refitting_model(model):
    """Copy the backend without lmer's unused last-coefficient reporting cache."""
    from replica.analysis.models.mixed_effects.lmer import RLmerModel

    model = deepcopy(model)
    if isinstance(model, RLmerModel):
        model._last_fixed = None
    return model


def _sample(ctx, series, draw):
    """Expand observed intersections and assign distinct random-effect copies."""
    nr, nt = ctx.group_counts
    r, t = (ctx.codes[n] for n in ctx.axes)
    if series == "identity":
        indices = np.arange(len(ctx.y))
        copies = (r, t)
    else:
        rng = np.random.default_rng(
            np.random.SeedSequence([ctx.settings.seed, 0, draw])
        )
        readers, sources = rng.integers(nr, size=nr), rng.integers(nt, size=nt)
        if series == "axis_1":
            readers = np.arange(nr)
        if series == "axis_0":
            sources = np.arange(nt)
        rows = [np.flatnonzero(r == reader) for reader in readers]
        stage = np.concatenate(rows)
        reader_copies = np.repeat(np.arange(nr), [len(v) for v in rows])
        source_positions = [np.flatnonzero(t[stage] == source) for source in sources]
        selected = np.concatenate(source_positions)
        indices = stage[selected]
        copies = (
            reader_copies[selected],
            np.repeat(np.arange(nt), [len(v) for v in source_positions]),
        )
    factors = {n: v[indices] for n, v in ctx.factors.items()}
    for name, dependencies in ctx.dependencies.items():
        key = np.column_stack(
            [ctx.codes[name][indices], *(copies[i] for i in dependencies)]
        )
        factors[name] = np.unique(key, axis=0, return_inverse=True)[1].astype(str)
    return indices, factors, copies


def _partitions(ctx, series, draw, cycle):
    """Partition every original identity, including zero-multiplicity slots."""
    sid = ("identity", *SERIES).index(series)
    rng = np.random.default_rng(
        np.random.SeedSequence(
            [
                ctx.settings.seed,
                1,
                ctx.cv_seed,
                sid,
                draw,
                cycle,
            ]
        )
    )
    assignments = []
    for name, n, k in zip(ctx.axes, ctx.group_counts, ctx.folds, strict=True):
        folds = np.arange(n) % k
        rng.shuffle(folds)
        assignments.append(folds[ctx.codes[name]])
    return assignments


def _preprocess_columns(preprocessing, columns, train, test):
    """Apply each training-fitted projection to one bootstrap fold."""
    result = dict(columns)
    for step in preprocessing:
        before = result[step["feature"]]
        left, right, _ = _apply_projection(
            step, before, [result[name] for name in step["against"]], train, test
        )
        if not np.isfinite(left).all() or not np.isfinite(right).all():
            raise ValueError(
                "Preprocessing produced nonfinite values in a bootstrap fold"
            )
        transformed = before.copy()
        transformed[train], transformed[test] = left, right
        result[step["feature"]] = transformed
    return result


def _cycle(ctx, key):
    """Fit both models on every occupied test rectangle of one complete cycle."""
    series, draw, cycle = key
    record = {"series": series, "draw": draw, "cycle": cycle, "error": None}
    fits = singular = n_test_total = 0
    gain_sum = 0.0
    rows, factors, copies = _sample(ctx, series, draw)
    record["n_observations"] = len(rows)
    assignments = [a[rows] for a in _partitions(ctx, series, draw, cycle)]
    digest = hashlib.sha256()
    for v in [rows, *copies, *assignments]:
        digest.update(np.asarray(v, dtype="<i8").tobytes())
    record["assignment_sha256"] = digest.hexdigest()
    columns = {n: v[rows] for n, v in ctx.columns.items()}
    names = (*ctx.comparison.baseline, *ctx.comparison.added)
    columns[ctx.target_name] = ctx.y[rows]
    coverage = np.zeros(len(rows), dtype=int)
    model = deepcopy(ctx.model)
    versions = set()
    try:
        if not len(rows):
            raise ValueError("Empty resampled observation population")
        for i in range(ctx.folds[0]):
            for j in range(ctx.folds[1]):
                test = (assignments[0] == i) & (assignments[1] == j)
                if not test.any():
                    continue
                train = (assignments[0] != i) & (assignments[1] != j)
                if not train.any():
                    raise ValueError("No training rectangle in this resample")
                coverage[test] += 1
                validate_fold_origins(
                    tuple(ctx.target_origins[row] for row in rows)
                    if ctx.target_origins
                    else (),
                    {
                        name: tuple(origins[row] for row in rows)
                        for name, origins in ctx.input_origins.items()
                    },
                    [(train, test)],
                    holdout_origins={
                        name: tuple(origins[row] for row in rows)
                        for name, origins in ctx.holdout_origins.items()
                    },
                )
                fold_columns = _preprocess_columns(
                    ctx.preprocessing, columns, train, test
                )
                matrix = np.column_stack([fold_columns[n] for n in names])
                fold_y = fold_columns[ctx.target_name]
                fold_ctx = FoldContext(train, test, fold_columns)
                transforms = [
                    _ResidualizeColumns(
                        [
                            (idx, ctx.residualization[n])
                            for idx, n in enumerate(names)
                            if n in ctx.residualization
                        ]
                    ),
                    deepcopy(ctx.scaler),
                ]
                full_train, full_test = apply_fold_transforms(
                    matrix, transforms, fold_ctx
                )
                n_control = len(ctx.comparison.baseline)
                if n_control:
                    control_transforms = [
                        _ResidualizeColumns(
                            [
                                (idx, ctx.residualization[n])
                                for idx, n in enumerate(ctx.comparison.baseline)
                                if n in ctx.residualization
                            ]
                        ),
                        deepcopy(ctx.scaler),
                    ]
                    control_train, control_test = apply_fold_transforms(
                        matrix[:, :n_control],
                        control_transforms,
                        fold_ctx,
                    )
                else:
                    control_train, control_test = full_train[:, :0], full_test[:, :0]
                scores = []
                for fixed in (ctx.comparison.baseline, names):
                    frames = []
                    for mask, full, controls in (
                        (train, full_train, control_train),
                        (test, full_test, control_test),
                    ):
                        values = dict(zip(names, full.T, strict=True))
                        values.update(
                            zip(ctx.comparison.baseline, controls.T, strict=True)
                        )
                        frames.append(
                            GroupedFrame(
                                values,
                                {n: v[mask] for n, v in factors.items()},
                                fixed,
                                ctx.design,
                            )
                        )
                    fitted = model.fit_grouped(fold_y[train], *frames)
                    validate_grouped_fit(fitted, int(test.sum()), len(fixed))
                    score = _gaussian_log_likelihood_pointwise(
                        fold_y[test],
                        fitted.predictions,
                        fitted.scoring_sigma("marginal"),
                    )
                    if not np.isfinite(score).all():
                        raise ValueError("Nonfinite held-out score")
                    scores.append(score)
                    fits += 1
                    singular += bool(fitted.metadata.get("singular", False))
                    if fitted.metadata.get("lme4_version"):
                        versions.add(fitted.metadata["lme4_version"])
                gain_sum += float((scores[1] - scores[0]).sum())
                n_test_total += int(test.sum())
        if not np.all(coverage == 1):
            raise ValueError(
                "Every resampled response must be scored exactly once per cycle"
            )
        record["gain"] = gain_sum / n_test_total
    except Exception as error:
        r_failure = type(error).__name__ == "RRuntimeError" and type(
            error
        ).__module__.startswith("rpy2.")
        if not r_failure and not isinstance(
            error, (ValueError, RuntimeError, np.linalg.LinAlgError)
        ):
            raise
        record.update(gain=None, error=f"{type(error).__name__}: {error}")
    record.update(
        n_scored=n_test_total,
        fitted_models=fits,
        singular_fits=singular,
        lme4_versions=sorted(versions),
    )
    return record


_WORKER_STATE: dict[str, _Context] = {}


def _initialize_worker(ctx):
    """Give each spawned process its own backend and prepared observations."""
    _WORKER_STATE["context"] = ctx


def _worker(key):
    """Execute a cycle without sharing an embedded R session across workers."""
    return _cycle(_WORKER_STATE["context"], key)


def _keys(settings):
    """Enumerate all required cycles without data-dependent stopping."""
    return [("identity", 0, c) for c in range(settings.original_cycles)] + [
        (s, b, c)
        for s in SERIES
        for b in range(settings.n_resamples)
        for c in range(settings.cycles_per_draw)
    ]


def _new_checkpoint_directory():
    """Create and announce a unique restart root under the writable user cache."""
    root = user_cache_root() / "inference"
    root.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="crossed-", dir=root))
    logger.info("Crossed bootstrap checkpoints: %s", directory)
    return directory


def _checkpoint(ctx, directory):
    """Bind restart files to data, method, budgets, transformations and models."""
    if directory is None:
        return None, None
    if directory == "auto":
        directory = _new_checkpoint_directory()
    directory = Path(directory)
    frozen = replace(ctx, settings=replace(ctx.settings, n_jobs=1))
    try:
        payload = pickle.dumps(frozen, protocol=5)
    except (TypeError, pickle.PicklingError, AttributeError) as error:
        raise ValueError(
            "Checkpointing requires serializable model and scaler options"
        ) from error
    code_hash = hashlib.sha256()
    paths = set(Path(__file__).parents[1].rglob("*.py"))
    paths.add(Path(inspect.getfile(type(ctx.model))))
    paths.add(Path(inspect.getfile(type(ctx.scaler))))
    for path in sorted(paths):
        code_hash.update(path.name.encode())
        code_hash.update(path.read_bytes())
    manifest = {
        "protocol": PROTOCOL,
        "method": METHOD,
        "input_sha256": hashlib.sha256(payload).hexdigest(),
        "implementation_sha256": code_hash.hexdigest(),
        "numpy_version": np.__version__,
    }
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "manifest.json"
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise ValueError(
                "Checkpoint belongs to different inputs, options, or implementation"
            )
    else:
        with path.open("x") as stream:
            json.dump(manifest, stream, indent=2)
    return directory, manifest


def _record_path(directory, key):
    """Name a checkpoint by scientific task identity, never by worker number."""
    return directory / f"{key[0]}-{key[1]:06d}-{key[2]:03d}.json"


def _run(ctx, checkpoint_dir, progress):
    """Reuse authenticated completed cycles and run remaining independent tasks."""
    directory, manifest = _checkpoint(ctx, checkpoint_dir)
    binding = (
        None
        if manifest is None
        else hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    )
    records, pending = {}, []
    for key in _keys(ctx.settings):
        path = _record_path(directory, key) if directory is not None else None
        if path is not None and path.exists():
            saved = json.loads(path.read_text())
            if saved.get("manifest_sha256") != binding:
                raise ValueError(f"Checkpoint input binding mismatch: {path}")
            body = json.dumps(saved["record"], sort_keys=True, allow_nan=False)
            if saved["sha256"] != hashlib.sha256(body.encode()).hexdigest():
                raise ValueError(f"Corrupt checkpoint: {path}")
            record = saved["record"]
            if tuple(record[n] for n in ("series", "draw", "cycle")) != key:
                raise ValueError(f"Checkpoint task identity mismatch: {path}")
            records[key] = record
        else:
            pending.append(key)
    executor = None
    output: Iterator[dict]
    if ctx.settings.n_jobs == 1:
        output = (_cycle(ctx, key) for key in pending)
    else:
        executor = ProcessPoolExecutor(
            max_workers=ctx.settings.n_jobs,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_initialize_worker,
            initargs=(ctx,),
        )
        output = executor.map(_worker, pending, chunksize=1)
    try:
        for key, record in zip(pending, output, strict=True):
            if directory is not None:
                body = json.dumps(record, sort_keys=True, allow_nan=False)
                saved = {
                    "manifest_sha256": binding,
                    "record": record,
                    "sha256": hashlib.sha256(body.encode()).hexdigest(),
                }
                path = _record_path(directory, key)
                temporary = path.with_suffix(".tmp")
                temporary.write_text(json.dumps(saved, allow_nan=False) + "\n")
                temporary.replace(path)
            records[key] = record
            if progress is not None:
                progress(deepcopy(record))
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
    return [records[k] for k in _keys(ctx.settings)], manifest, directory


def crossed_bootstrap_test(
    observations: ObservationTable,
    comparison: ComparisonSpec,
    analysis: AnalysisConfig,
    *,
    inference: CrossedBootstrapConfig | None = None,
    checkpoint_dir: str | Path | None = "auto",
    residualization: Mapping[str, str] | None = None,
    progress: Callable[[dict], None] | None = None,
) -> PredictiveTestResult:
    """Refit a paired crossed bootstrap on already prepared individual responses.

    Use :func:`predictive_test` to prepare a Dataset and EvaluationSpec directly.
    This explicit call defaults to the assessed crossed-bootstrap construction.
    It never resamples saved predictions or silently falls back to another test.

    Args:
        observations: Eligible unaveraged rows with original sequence predictors
            and named factor labels. The response transform and fixed eligibility
            are already applied; training-dependent scaling is repeated per fit.
        comparison: Baseline numeric columns and the focal family added jointly.
        analysis: Named two-axis Design, Gaussian backend fitted by ML, training
            scaler and CrossedGroupSplitter choices. Marginal scoring is required.
        inference: Draw/cycle budgets, seed and worker count; None uses defaults.
        checkpoint_dir: "auto" creates a unique directory under the REPLICA
            user cache and logs its path before fitting. Pass that path to resume,
            another path for durable storage, or None to disable checkpoints.
            Reuse verifies inputs, settings and code and preserves failed cycles.
        residualization: Optional column-to-reference mapping for train-only
            residualization, applied before scaling in every fold.
        progress: Optional callback receiving each newly completed cycle record.

    Returns:
        An unadjusted two-sided test and pointwise interval, or an explicit
        unavailable result if a required cycle fails or variance is unsupported.
    """
    settings = inference or CrossedBootstrapConfig()
    ctx = _context(observations, comparison, analysis, settings, residualization)
    budget = settings.fit_budget(math.prod(ctx.folds))
    logger.info(
        "Crossed bootstrap: %s folds, up to %d fits before retries", ctx.folds, budget
    )
    records, manifest, directory = _run(ctx, checkpoint_dir, progress)
    failures = [r for r in records if r["error"]]
    original = [r["gain"] for r in records if r["series"] == "identity"]
    metadata = {
        "protocol": PROTOCOL,
        "method": METHOD,
        "inference": asdict(settings),
        "design": ctx.design.metadata(),
        "comparison": asdict(comparison),
        "axis_names": list(ctx.axes),
        "group_counts": list(ctx.group_counts),
        "n_observations": len(ctx.y),
        "crossed_cv": ctx.preview,
        "fit_budget": budget,
        "fitted_models": sum(r["fitted_models"] for r in records),
        "singular_fits": sum(r["singular_fits"] for r in records),
        "cycles": records,
        "failures": failures,
        "checkpoint_manifest": manifest,
        "checkpoint_dir": None if directory is None else str(directory.resolve()),
        "response": observations.response,
        "runtime": ctx.runtime,
        "scoring": "marginal_gaussian",
        "partition_policy": "fresh balanced partitions of all original identities, including absent slots",
        "copy_policy": "distinct random-effect copies within inferred and verified nesting",
        "rng": "numpy PCG64 / SeedSequence by purpose, series, draw and cycle",
        "multiplicity_adjusted": False,
        "calibration_scope": "Assessment used 499 draws, C=2, M=8 and 2x10 folds; 40/41 checks passed, overall criterion unmet. Transfer is not automatic.",
    }
    if failures:
        point = (
            float(np.mean(original)) if all(v is not None for v in original) else None
        )
        return PredictiveTestResult(
            point, False, "recorded fitting or resample failure", metadata=metadata
        )
    samples = {
        s: np.array([r["gain"] for r in records if r["series"] == s]).reshape(
            settings.n_resamples,
            settings.cycles_per_draw,
        )
        for s in SERIES
    }
    result = _summarize(original, samples, ctx.group_counts, settings.confidence_level)
    return replace(result, metadata={**result.metadata, **metadata})
