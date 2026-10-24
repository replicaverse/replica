"""Analysis configuration for evaluate().

Bundles the regression model, cross-validation strategy, scaling, DV
transform, and sigma estimation into a single reusable object.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from numbers import Real
from typing import TYPE_CHECKING, Literal

import numpy as np

from replica.analysis.contracts.configuration import component_options
from replica.analysis.contracts.design import Design
from replica.analysis.cv.group_holdout import CrossedGroupSplitter
from replica.analysis.cv.scalers import ZScoreScaler
from replica.analysis.cv.splitters import ItemSplitter
from replica.analysis.specification.preprocessing import Residualize

if TYPE_CHECKING:
    from replica.analysis.contracts.grouped import GroupedModel
    from replica.analysis.contracts.models import AnalysisModel, LikelihoodModel
    from replica.analysis.cv.scalers import Scaler
    from replica.analysis.cv.splitters import CVSplitter


@dataclass(frozen=True)
class AnalysisConfig:
    """Specification of how to run the regression analysis.

    Bundles the statistical methodology parameters that vary across papers
    but are fixed within a single paper's analysis. Immutable so it can
    be safely shared across ``evaluate()`` calls.

    Use ``dataclasses.replace()`` to create variants::

        from dataclasses import replace
        base = AnalysisConfig()
        variant = replace(base, train_fraction=0.8)

    Defaults fit an ML mixed model to raw responses, with crossed holdout of
    participants and stimuli and pointwise marginal Gaussian scoring. The default
    Design adds participant intercepts and focal-predictor slopes with zero
    correlations, plus occurrence intercepts. Reproductions declare their recipe
    explicitly, including ``design=None`` for models using the scalar interface.

    Args:
        preprocessing: Ordered training-fold transformations, before scaling.
            Backends with opaque internal tuning cannot refit these on inner folds
            and reject them.
        missing: Drop incomplete observations or reject them before fitting.
        score_scale: Gaussian scoring on the transformed or original target
            scale. Original-scale scoring requires transform_dv.inverse(values)
            and transform_dv.log_abs_det_jacobian(raw_values), except for identity.
            Backend-owned likelihoods require transformed-scale scoring.
        model: Regression backend. None selects ``RLmerModel(reml=False)``.
            Paper reproductions should declare their estimator explicitly.
        design: Named factors, model terms, prediction and sampling policy.
            Defaults to :class:`Design`. An explicit None selects the scalar
            model and grouping interface for alternative analysis recipes.
        cv: None selects the declared named Design holdout: crossed CV for
            multiple factors, or ItemSplitter for one factor. Without a named
            Design, ItemSplitter uses the explicitly selected scalar interface.
        n_folds: Explicit maximum folds per held-out factor when ``cv`` is None.
            A tuple selects separate crossed counts. None uses the crossed
            splitter's 90% training target, or ten folds for legacy/single-axis
            analyses. Mutually exclusive with ``train_fraction``.
        train_fraction: Nominal training fraction for default crossed CV,
            shared across axes, supplied per axis, or keyed by declared factor
            name. Configure an explicit ``cv`` through its own constructor.
        scaler: ``None`` → ``ZScoreScaler()``.
        transform_dv: Applied to the DV before regression. None keeps the raw
            response scale; a callable such as ``np.log`` selects a transform.
            Outputs must be real and preserve shape. Masked or nonfinite results
            follow the ``missing`` policy.
        sigma_source: ``"training"`` computes sigma from training-set
            residuals; ``"test"`` uses the test-set residual estimate and
            requires ``design=None``.
        empty_control_baseline: With no controls, ``"mean"`` predicts the
            training mean; ``"model"`` fits the configured backend, including
            its grouping effects. Applies only to scalar prediction backends
            with ``design=None`` and the built-in analysis.
        per_fold: Store a FoldResult per CV fold on each AnalysisResult.
        retain_per_obs_delta_ll: When True (and ``per_fold`` is True),
            each ``FoldResult`` stores the per-observation held-out ΔLL
            vector ``per_obs_delta_ll`` (sums to the fold's ``delta_ll``).
            Useful for diagnostics and recipe replay. These dependent
            scores are not an independent sample for a row-wise test; request
            ``predictive_test`` for full-refit crossed inference.
        prediction_diagnostics: Retain bounded per-fold summaries of scored
            prediction errors for reports. Requires ``per_fold=True`` and the
            built-in analysis; does not retain response or prediction vectors.
        retain_predictions: Retain aligned scored responses, predictions and
            observation identities for optional export. Requires ``per_fold=True``
            and the built-in analysis. Memory grows with scored rows across folds.
        analysis_fn: Custom analysis run once per DV in place of the built-in
            regression loop; requires ``design=None``. It receives keyword arguments ``y``,
            ``X_features`` and ``X_controls`` (feature name to column),
            ``participants`` and ``items`` (integer codes), ``participant_ids``
            and ``item_ids`` (raw IDs), ``dataset_name``, ``model_name`` and
            ``dv_name``, and returns a list of :class:`AnalysisResult`. The outlier
            filter and ``transform_dv`` still apply to ``y``; the model, CV,
            scaling, sigma and per-fold settings are ignored.
    """

    model: AnalysisModel | LikelihoodModel | GroupedModel | None = None
    preprocessing: tuple[Residualize, ...] = ()
    missing: Literal["drop", "error"] = "drop"
    score_scale: Literal["transformed", "original"] = "transformed"
    cv: CVSplitter | None = None
    n_folds: int | tuple[int, ...] | None = None
    scaler: Scaler | None = None  # None → ZScoreScaler()
    transform_dv: Callable | None = None
    sigma_source: Literal["training", "test"] = "training"
    per_fold: bool = False
    retain_per_obs_delta_ll: bool = False
    analysis_fn: Callable | None = None
    dump_merged_path: str | None = None
    """If set, write the post-merge long-form design matrix (one row per
    observation, columns = all controls + all features + participant_id +
    item_id + transformed y) to this path as a Parquet file before
    fitting. For offline inspection of what the regression sees."""
    # ------------------------------------------------------------------
    # Outlier-filter knobs (applied to the raw DV, before ``transform_dv``)
    #
    # All default to ``None``; drivers that need DV cleaning set them on
    # their AnalysisConfig.
    # ------------------------------------------------------------------
    outlier_floor: float | None = None
    """Inclusive lower bound on raw responses, combined with the trim mask."""
    outlier_ceiling: float | None = None
    """Inclusive upper bound on raw responses, combined with the trim mask."""
    outlier_method: str | Callable[[np.ndarray], np.ndarray] | None = None
    """Per-participant trim: ``"mad"`` (median +/- k*1.4826*MAD, Leys et al.
    (2013)), ``"sd"`` (mean +/- k*SD), or a callable returning a boolean keep
    mask for each finite raw-response group. ``None`` skips statistical trimming."""
    outlier_k: float = 2.5
    """Cutoff multiplier for built-in MAD/SD methods (2.5 by default)."""

    design: Design | None = field(default_factory=Design)
    train_fraction: float | tuple[float, ...] | Mapping[str, float] | None = None
    prediction_diagnostics: bool = False
    retain_predictions: bool = False
    empty_control_baseline: Literal["mean", "model"] = "mean"

    def __post_init__(self):
        if self.prediction_diagnostics or self.retain_predictions:
            option = (
                "retain_predictions"
                if self.retain_predictions
                else "prediction_diagnostics"
            )
            if not self.per_fold:
                raise ValueError(f"{option} requires per_fold=True")
            if self.analysis_fn is not None:
                raise ValueError(f"{option} requires the built-in analysis")
        object.__setattr__(self, "preprocessing", tuple(self.preprocessing))
        if any(not isinstance(step, Residualize) for step in self.preprocessing):
            raise TypeError("preprocessing entries must be Residualize declarations")
        if self.missing not in ("drop", "error"):
            raise ValueError("missing must be drop or error")
        if self.score_scale not in ("transformed", "original"):
            raise ValueError("score_scale must be transformed or original")
        if self.model is None:
            from replica.analysis.models.mixed_effects.lmer import RLmerModel

            object.__setattr__(self, "model", RLmerModel(reml=False))
        if self.empty_control_baseline not in ("mean", "model"):
            raise ValueError("empty_control_baseline must be mean or model")
        if self.empty_control_baseline == "model" and (
            self.design is not None
            or self.analysis_fn is not None
            or getattr(self.model, "owns_loglik", False)
        ):
            raise ValueError(
                "empty_control_baseline='model' requires design=None, "
                "the built-in analysis and a prediction backend"
            )
        if self.transform_dv is None:
            object.__setattr__(self, "transform_dv", _identity_response)
        elif not callable(self.transform_dv):
            raise TypeError("transform_dv must be a callable or None for raw responses")
        self.resolve_cv()
        if self.sigma_source not in ("training", "test"):
            raise ValueError(
                f"sigma_source must be 'training' or 'test', got {self.sigma_source!r}"
            )

    def resolve_cv(self) -> CVSplitter:
        """Resolve the CV policy once, using the same defaults as direct splitters."""
        if self.cv is not None:
            if self.train_fraction is not None:
                raise ValueError("Set train_fraction on the explicit cv splitter")
            return self.cv
        if self.design is not None and len(self.design.cv_units) > 1:
            return CrossedGroupSplitter(
                self.n_folds, train_fraction=self.train_fraction
            )
        if self.train_fraction is not None:
            raise ValueError("train_fraction requires a crossed Design or explicit cv")
        if self.n_folds is not None and not isinstance(self.n_folds, int):
            raise ValueError("A single held-out axis requires one integer n_folds")
        return ItemSplitter(10 if self.n_folds is None else self.n_folds)

    def resolve(self, predictors: tuple[str, ...]) -> AnalysisConfig:
        """Resolve the same focal random structure for scoring and inference.

        Args:
            predictors: Names of focal columns, including their declared lags.
        """
        if self.design is None:
            return self
        design = self.design.resolve(predictors)
        return self if design is self.design else replace(self, design=design)

    def metadata(self) -> dict:
        """Describe resolved conventions; missing component options are None.

        Custom components and callable transforms can declare JSON-compatible
        options through ``configuration()`` on their concrete class.
        """
        model = self.model
        model_options = component_options(model)
        splitter = self.resolve_cv()
        scaler = ZScoreScaler() if self.scaler is None else self.scaler
        transform = self.transform_dv
        outliers = {
            key: getattr(self, "outlier_" + key)
            for key in ("floor", "ceiling", "method", "k")
        }
        if callable(self.outlier_method):
            method = self.outlier_method
            outliers["method"] = (
                f"{getattr(method, '__module__', type(method).__module__)}."
                f"{getattr(method, '__qualname__', type(method).__qualname__)}"
            )
            outliers["method_options"] = component_options(method)
        return {
            "preprocessing": [step.configuration() for step in self.preprocessing],
            "missing": self.missing,
            "score_scale": self.score_scale,
            "backend": f"{type(model).__module__}.{type(model).__qualname__}",
            "backend_options": model_options,
            "fit_criterion": ("REML" if model_options["reml"] else "ML")
            if model_options is not None and "reml" in model_options
            else None,
            "residual_variance_convention": getattr(model, "_sigma_correction", None),
            "response_transform": "identity"
            if transform is _identity_response
            else getattr(
                transform,
                "__qualname__",
                getattr(transform, "__name__", type(transform).__qualname__),
            ),
            "response_transform_options": {}
            if transform is _identity_response
            else component_options(transform),
            "scaler": "ZScoreScaler"
            if self.scaler is None
            else type(self.scaler).__name__,
            "scaler_options": component_options(scaler),
            "sigma_source": self.sigma_source,
            "empty_control_baseline": self.empty_control_baseline
            if self.design is None
            and self.analysis_fn is None
            and not getattr(model, "owns_loglik", False)
            else None,
            "prediction_diagnostics": self.prediction_diagnostics,
            "retain_predictions": self.retain_predictions,
            "cv_splitter": type(splitter).__name__,
            "cv_options": component_options(splitter),
            "outlier_filter": outliers,
            "predictive_significance": "not_assessed"
            if self.analysis_fn is None
            else "custom_analysis",
        }


def _identity_response(values):
    """Keep responses on their original scale."""
    return values


@dataclass(frozen=True)
class EncodingConfig:
    """Configuration for sampled-response encoding through ``evaluate()``.

    Fits the recordings selected by ``EvaluationSpec.targets``. It builds
    a regressor design on each recording's clock and scores a cross-validated
    ridge encoder per channel. The design is the controls,
    the scalar predictors and one explicit vector feature per fit, aligned with
    ``kernel`` and expanded by ``fir_delays``; controls alone form the
    baseline whose R² the full model is compared with. Non-time response
    dimensions are flattened into channels and kept in the result metadata.
    Targets retain their declared recording clocks and channel dimensions.
    Responses must be finite; missing values require explicit preprocessing.

    Args:
        fir_delays: Non-negative delays of the FIR design, in ``delay_unit``.
        n_folds: Folds per recording (``cv="temporal"``) or per participant
            (``cv="recording"``).
        buffer_trs: Samples excluded around each held-out block.
        lanczos_window: Number of lobes in the interpolation kernel, at least 1.
        alphas: Ridge penalty scales considered within each training fold;
            the effective L2 penalty is ``alpha ** 2``.
        kernel: How unit features reach the response clock: ``"lanczos"``
            interpolates them to the sample grid, ``"impulse"`` places each
            unit on its nearest sample and zero elsewhere.
        delay_unit: ``"samples"`` takes ``fir_delays`` as integer samples;
            ``"seconds"`` converts them per recording from its sample interval.
        cv: ``"temporal"`` holds out contiguous blocks inside each recording;
            ``"recording"`` holds out whole recordings of a participant, whose
            response dimensions and channel labels must match.
        normalization: ``"training"`` fits feature and response means/scales on
            training rows and applies them to held-out rows. ``"within_split"``
            standardizes each temporal split, or each recording in recording CV,
            separately. ``"none"`` leaves the design and responses unchanged.
            Training/none results are scored in original response coordinates;
            within-split results are scored in separately standardized coordinates.
        participants: Optional participant subset, a sequence of IDs matched
            exactly. ``None`` selects all. When every target directly selects
            stored recordings, only selected responses are copied. Unrelated
            stimuli are skipped only for model sources and extractors declaring
            ``supports_stimulus_selection=True``. Explicit feature
            dependencies and composed targets retain their declared populations.
        alignment_layer: Stimulus layer containing event timestamps.
        feature_layer: Stimulus layer passed to the feature model.
        alpha_chunk_length: Held-out chunk length for ridge-alpha selection,
            at least 1.
        alpha_nchunks: Number of alpha-selection chunks, at least 1. A
            training fold with too few complete chunks or no rows left after
            holdout takes the middle alpha and records ``"fallback"`` in the
            result metadata's ``alpha_selection`` list.
        random_seed: Chunk-shuffle seed. ``None`` is non-deterministic.
    """

    fir_delays: tuple[float, ...] = tuple(range(1, 9))
    n_folds: int = 4
    buffer_trs: int = 5
    lanczos_window: int = 3
    alphas: tuple[float, ...] = field(
        default_factory=lambda: tuple(float(value) for value in np.logspace(1, 3, 10))
    )
    participants: tuple[str, ...] | None = None
    alignment_layer: str = "word_times"
    feature_layer: str = "words"
    alpha_chunk_length: int = 40
    alpha_nchunks: int = 20
    random_seed: int | None = 0
    kernel: str = "lanczos"
    delay_unit: str = "samples"
    cv: str = "temporal"
    normalization: str = "training"

    def __post_init__(self) -> None:
        if not self.fir_delays:
            raise ValueError("fir_delays must contain at least one delay")
        if not self.alphas:
            raise ValueError("alphas must contain at least one ridge penalty")
        if self.n_folds < 2:
            raise ValueError("n_folds must be at least 2")
        if self.buffer_trs < 0:
            raise ValueError("buffer_trs must be non-negative")
        if self.kernel not in ("lanczos", "impulse"):
            raise ValueError(
                f"kernel must be 'lanczos' or 'impulse', got {self.kernel!r}"
            )
        if self.delay_unit not in ("samples", "seconds"):
            raise ValueError(
                f"delay_unit must be 'samples' or 'seconds', got {self.delay_unit!r}"
            )
        if self.cv not in ("temporal", "recording"):
            raise ValueError(f"cv must be 'temporal' or 'recording', got {self.cv!r}")
        if self.normalization not in ("training", "within_split", "none"):
            raise ValueError(
                "normalization must be 'training', 'within_split', or 'none'"
            )
        integral = self.delay_unit == "samples"
        if any(
            isinstance(delay, bool)
            or not isinstance(delay, (int, np.integer) if integral else Real)
            or delay < 0
            for delay in self.fir_delays
        ):
            raise ValueError(
                "fir_delays must be non-negative "
                f"{'integers' if integral else 'numbers'}, got {self.fir_delays!r}"
            )
        if self.lanczos_window < 1:
            raise ValueError("lanczos_window must be at least 1")
        if self.alpha_chunk_length < 1:
            raise ValueError("alpha_chunk_length must be at least 1")
        if self.alpha_nchunks < 1:
            raise ValueError("alpha_nchunks must be at least 1")
        if isinstance(self.participants, str):
            raise ValueError(
                "participants must be a sequence of participant IDs, not one "
                f"string; use ({self.participants!r},)"
            )
