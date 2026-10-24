"""Encoding-model evaluation for continuous-response modalities.

The counterpart of ``evaluate()`` for sampled responses: per-event hidden
states are interpolated to the response sample times, FIR-delayed,
ridge-regressed against the response channels and scored by channel-wise
Pearson correlation under temporal k-fold CV, for any :class:`Recording`
(fMRI, EEG, MEG, ECoG).

"The reference" in comments is the encoding code of the Oota et al. (2023)
release (``brain_predictions/`` with ``npp`` and ``ridge_utils``), which
follows the Huth lab's ridge code.
"""

from __future__ import annotations

import random
import warnings
from collections.abc import Callable, Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import numpy as np

from replica.analysis.data.preparation import (
    _model_name,
    target_outcome_origins,
    validate_fold_origins,
)
from replica.analysis.specification.config import EncodingConfig
from replica.contracts.features import (
    UNIT_INDEX_NAMES,
    FeatureBlock,
    FeatureIndex,
    FeatureValues,
    align_values,
    block_origins,
)
from replica.core import Recording
from replica.dataset.features import _select_recording_features
from replica.feature.core.measurement import describe_feature
from replica.feature.resolution import _resolve_features

if TYPE_CHECKING:
    from replica.analysis.specification.evaluation import EvaluationSpec
    from replica.core import Stimulus
    from replica.dataset import Dataset


def _artifact_metadata(value):
    """Detach nested metadata into containers owned by the callback."""
    if isinstance(value, Mapping):
        return {key: _artifact_metadata(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_artifact_metadata(item) for item in value)
    return deepcopy(value)


@dataclass
class EncodingResult:
    """Result of one fit: a layer (or none) on one recording or participant.

    ``correlations`` and ``r2`` score the full model per channel; the
    baseline fields hold the controls-only model and are ``None`` without
    controls.
    """

    dataset_name: str
    model_name: str
    layer: int | None
    participant_id: str
    correlations: np.ndarray  # flattened non-time response dimensions
    modality: str
    response_measure: str
    language: str | None = None
    metrics: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    r2: np.ndarray | None = None
    baseline_correlations: np.ndarray | None = None
    baseline_r2: np.ndarray | None = None

    @property
    def mean_correlation(self) -> float:
        """Held-out Pearson r averaged over the channels where it is defined."""
        return self.metrics.get("mean_r", 0.0)


@dataclass(frozen=True)
class EncodingFoldArtifact:
    """Fitted state and held-out data from one temporal encoding fold.

    ``recording_index`` is the position after participant filtering and
    ``fold_index`` follows the temporal splitter; row indices refer to the
    original recording. Response channels flatten ``response_shape`` in C order;
    design columns follow ``config.fir_delays``, then hidden-state coordinates.

    ``test_design`` holds the unscaled interpolated FIR features, so predictions
    replay as ``((test_design - feature_test_mean) / feature_test_scale) @
    weights``. ``config.normalization`` identifies the transformation: training
    statistics apply to both splits by default; ``within_split`` fits each split
    separately, and ``none`` records zero means and unit scales. Predictions use
    training-response coordinates and observations use the recorded test-response
    coordinates. Scales are population standard deviations with zero replaced by
    one. Shapes: ``test_design`` (test rows, FIR features),
    ``weights`` (FIR features, channels), ``predictions`` and ``observations``
    (test rows, channels); means and scales hold one value per feature or
    channel. ``response_dimensions`` includes time, ``response_shape`` does not.

    Arrays and metadata are detached from the evaluation, so a callback may keep
    them. Refitting still needs the source features and responses.
    """

    dataset_name: str
    model_name: str
    layer: int | None
    participant_id: str
    stimulus_id: str
    recording_index: int
    fold_index: int
    modality: str
    response_measure: str
    language: str | None
    config: EncodingConfig
    response_shape: tuple[int, ...]
    response_dimensions: tuple[str, ...]
    response_dimension_labels: dict
    recording_metadata: dict
    train_indices: np.ndarray
    test_indices: np.ndarray
    test_times: np.ndarray
    alpha: float
    test_design: np.ndarray
    weights: np.ndarray
    predictions: np.ndarray
    observations: np.ndarray
    feature_train_mean: np.ndarray
    feature_train_scale: np.ndarray
    feature_test_mean: np.ndarray
    feature_test_scale: np.ndarray
    response_train_mean: np.ndarray
    response_train_scale: np.ndarray
    response_test_mean: np.ndarray
    response_test_scale: np.ndarray


def _lanczosfun(cutoff: float, t: np.ndarray, window: int = 3) -> np.ndarray:
    """Lanczos kernel: windowed sinc function.

    Args:
        cutoff: Cutoff frequency (cycles per unit time).
        t: Time offsets (will be scaled by cutoff).
        window: Number of lobes.

    Returns:
        Kernel values, same shape as t.
    """
    t = np.asarray(t, dtype=float) * cutoff
    pi = np.pi

    # Compute sinc * windowed sinc
    with np.errstate(divide="ignore", invalid="ignore"):
        val = window * np.sin(pi * t) * np.sin(pi * t / window) / (pi**2 * t**2)

    # Handle t == 0
    val[t == 0] = 1.0
    # Zero outside window
    val[np.abs(t) > window] = 0.0

    return val


def lanczos_interpolate(
    data: np.ndarray,
    old_times: np.ndarray,
    new_times: np.ndarray,
    window: int = 3,
) -> np.ndarray:
    """Lanczos interpolation from irregular to regular time grid.

    Resamples rows of ``data`` from ``old_times`` to ``new_times`` using
    a Lanczos (windowed sinc) kernel.

    Args:
        data: Feature matrix, shape (n_old, n_features). If 1-D, treated
            as (n_old, 1) and result is squeezed back.
        old_times: Time of each row in data, shape (n_old,).
        new_times: Target time grid, shape (n_new,). Assumed evenly spaced.
        window: Number of sinc lobes (default 3, matching reference).

    Returns:
        Interpolated matrix, shape (n_new, n_features).
    """
    data = np.asarray(data, dtype=float)
    squeeze = False
    if data.ndim == 1:
        data = data[:, np.newaxis]
        squeeze = True

    old_times = np.asarray(old_times, dtype=float)
    new_times = np.asarray(new_times, dtype=float)

    # Cutoff frequency from target grid spacing
    cutoff = float(1.0 / np.mean(np.diff(new_times)))

    # Build sinc interpolation matrix
    sincmat = np.zeros((len(new_times), len(old_times)))
    for i in range(len(new_times)):
        sincmat[i, :] = _lanczosfun(cutoff, new_times[i] - old_times, window)

    result = sincmat @ data
    if squeeze:
        result = result.squeeze(axis=1)
    return result


def make_fir_delays(data: np.ndarray, delays: Iterable[int]) -> np.ndarray:
    """Create FIR delay design matrix by shifting data forward.

    For each delay d, shifts data forward by d rows and zero-fills the
    gap at the start. The result is the horizontal concatenation of all
    delayed copies.

    Args:
        data: Shape (n_trs, n_features).
        delays: Delay values in TRs (e.g. range(1, 9) for 1-8).

    Returns:
        Shape (n_trs, n_features * len(delays)).
    """
    data = np.asarray(data, dtype=float)
    if data.ndim == 1:
        data = data[:, np.newaxis]

    n_trs, n_feat = data.shape
    delays = list(delays)
    result = np.zeros((n_trs, n_feat * len(delays)))

    for i, d in enumerate(delays):
        col_start = i * n_feat
        col_end = (i + 1) * n_feat
        if d >= n_trs:
            # All zeros
            continue
        result[d:, col_start:col_end] = data[: n_trs - d]

    return result


def _zscore_columns(data: np.ndarray) -> np.ndarray:
    """Per-column z-scoring. Columns with zero std are left as zero."""
    return _standardize_columns(data)[0]


def _standardize_columns(
    data: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return z-scored columns and the mean and effective scale used."""
    data = np.asarray(data, dtype=float)
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    std[std == 0] = 1.0  # avoid division by zero
    return (data - mean) / std, mean, std


def _normalize_splits(train: np.ndarray, test: np.ndarray, policy: str):
    """Transform a partition and retain the statistics used for each side."""
    if policy == "none":
        identity = (np.zeros(train.shape[1]), np.ones(train.shape[1]))
        return train, test, identity, identity
    normalized_train, mean, scale = _standardize_columns(train)
    if policy == "training":
        return normalized_train, (test - mean) / scale, (mean, scale), (mean, scale)
    normalized_test, test_mean, test_scale = _standardize_columns(test)
    return normalized_train, normalized_test, (mean, scale), (test_mean, test_scale)


def _columnwise_pearson(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """Pearson correlation for matching columns, preserving undefined cells.

    A correlation is undefined when either column is constant or contains a
    non-finite value.  Returning ``NaN`` for those columns matches
    ``numpy.corrcoef`` and, for cortical data, prevents medial-wall
    vertices with zero BOLD variance from being silently counted as zero
    correlations in whole-brain averages.
    """
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if actual.ndim != 2 or predicted.ndim != 2:
        raise ValueError("actual and predicted must both be two-dimensional")
    if actual.shape != predicted.shape:
        raise ValueError(
            f"actual and predicted shapes differ: {actual.shape} != {predicted.shape}"
        )

    actual_centered = actual - actual.mean(axis=0)
    predicted_centered = predicted - predicted.mean(axis=0)
    actual_std = actual_centered.std(axis=0)
    predicted_std = predicted_centered.std(axis=0)
    valid = (
        np.isfinite(actual).all(axis=0)
        & np.isfinite(predicted).all(axis=0)
        & (actual_std > 0)
        & (predicted_std > 0)
    )

    correlations = np.full(actual.shape[1], np.nan, dtype=float)
    correlations[valid] = (
        actual_centered[:, valid] * predicted_centered[:, valid]
    ).mean(axis=0) / (actual_std[valid] * predicted_std[valid])
    return correlations


class TemporalKFold:
    """K-fold temporal CV with buffer around test fold edges.

    Removes ``buffer_trs`` training samples on both sides of each test
    fold to prevent temporal autocorrelation leakage. Uses contiguous
    folds in time order.
    """

    def __init__(self, n_folds: int = 4, buffer_trs: int = 5):
        """Set the fold count and the width of the discarded edge.

        Args:
            n_folds: Number of contiguous test blocks the timeline is cut into.
            buffer_trs: Samples dropped from training on each side of the test
                block, in acquisition samples.
        """
        self.n_folds = n_folds
        self.buffer_trs = buffer_trs

    def split(self, n_trs: int) -> Iterable[tuple[np.ndarray, np.ndarray]]:
        """Yield (train_indices, test_indices) for each fold.

        Args:
            n_trs: Length of the recording in acquisition samples, indexed
                from zero.
        """
        from sklearn.model_selection import KFold

        all_indices = np.arange(n_trs)
        kf = KFold(n_splits=self.n_folds, shuffle=False)

        for train_idx, test_idx in kf.split(all_indices):
            # Remove buffer TRs from training set around test boundaries
            test_start = test_idx[0]
            test_end = test_idx[-1]

            # Exclude buffer_trs before test fold
            exclude = set()
            for r in range(test_start - self.buffer_trs, test_start):
                if r >= 0:
                    exclude.add(r)

            # Exclude buffer_trs after test fold
            for r in range(test_end + 1, test_end + 1 + self.buffer_trs):
                if r < n_trs:
                    exclude.add(r)

            train_idx_clean = np.array([i for i in train_idx if i not in exclude])

            yield train_idx_clean, test_idx


def _ridge_svd(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    alpha: float,
    singcutoff: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray]:
    """SVD-based ridge regression.

    Args:
        X_train: (n_train, n_features)
        y_train: (n_train, n_voxels)
        X_test: (n_test, n_features)
        alpha: Ridge penalty, entering squared: each singular direction is
            scaled by ``S / (S**2 + alpha**2)``.
        singcutoff: Singular values below this are dropped.

    Returns:
        (y_pred, weights) where y_pred is (n_test, n_voxels) and
        weights is (n_features, n_voxels).
    """
    U, S, Vh = np.linalg.svd(X_train, full_matrices=False)

    # Truncate tiny singular values
    good = singcutoff < S
    U = U[:, good]
    S = S[good]
    Vh = Vh[good]

    # Precompute U^T @ y_train
    UR = U.T @ y_train

    # Ridge weights: Vh^T @ diag(S/(S^2 + alpha^2)) @ U^T @ y
    D = S / (S**2 + alpha**2)
    wt = Vh.T @ (D[:, np.newaxis] * UR)

    y_pred = X_test @ wt
    return y_pred, wt


def _select_alpha_bootstrap(
    X_train: np.ndarray,
    y_train: np.ndarray,
    alphas: np.ndarray,
    chunklen: int = 40,
    nchunks: int = 20,
    singcutoff: float = 1e-10,
    rng: random.Random | None = None,
    *,
    normalize_training: bool = False,
) -> tuple[float, str]:
    """Select the ridge alpha on one held-out split of the training data.

    Holds out shuffled contiguous chunks of the supplied training data and picks
    the alpha with the highest mean correlation over channels. If the requested
    holdout leaves no training rows or needs more complete chunks than available,
    the middle alpha is used, with a warning.

    Args:
        X_train: Features, (n_train, n_features).
        y_train: Responses, (n_train, n_channels).
        alphas: Ridge penalty scales to score; the L2 penalty is ``alpha ** 2``.
        chunklen: Chunk length for held-out validation (default 40).
        nchunks: Number of chunks to hold out (default 20).
        singcutoff: SVD truncation threshold.
        rng: Shuffle state; the default is a local seed of zero.
        normalize_training: Fit normalization on inner-training rows only;
            otherwise use the supplied arrays without rescaling.

    Returns:
        The alpha, and ``"held_out"`` when it was selected on the held-out
        chunks or ``"fallback"`` when it is the middle alpha.
    """
    n_trs = X_train.shape[0]
    all_inds = list(range(n_trs))

    # Discard an incomplete final chunk, shuffle complete chunks, and hold out
    # the first ``nchunks``.
    chunks = list(zip(*[iter(range(n_trs))] * chunklen, strict=False))

    if len(chunks) < nchunks or n_trs == nchunks * chunklen:
        middle = float(alphas[len(alphas) // 2])
        warnings.warn(
            f"ridge alpha selection needs alpha_nchunks={nchunks} complete "
            f"chunks of {chunklen} samples and remaining training rows, but the "
            f"fold has {n_trs} rows ({len(chunks)} complete chunks); "
            f"using the middle alpha {middle}",
            stacklevel=3,
        )
        return middle, "fallback"

    if rng is None:
        rng = random.Random(0)
    rng.shuffle(chunks)
    held_chunks = chunks[:nchunks]
    held_inds: list[int] = []
    for chunk in held_chunks:
        held_inds.extend(chunk)

    held_inds_set = set(held_inds)
    train_inds = [i for i in all_inds if i not in held_inds_set]

    RR_X = X_train[train_inds]
    PR_X = X_train[held_inds]
    RR_y = y_train[train_inds]
    PR_y = y_train[held_inds]
    if normalize_training:
        RR_X, PR_X, _, _ = _normalize_splits(RR_X, PR_X, "training")
        RR_y, PR_y, _, _ = _normalize_splits(RR_y, PR_y, "training")

    # SVD once for all alphas
    U, S, Vh = np.linalg.svd(RR_X, full_matrices=False)
    good = singcutoff < S
    U = U[:, good]
    S = S[good]
    Vh = Vh[good]

    UR = U.T @ RR_y
    PVh = PR_X @ Vh.T

    # Reference: zs(Presp) and zs(pred) for correlation
    zPR_y = _zscore_columns(PR_y)

    best_mean_corr = -np.inf
    best_alpha = alphas[0]

    for alpha in alphas:
        D = S / (S**2 + alpha**2)
        pred = PVh @ (D[:, np.newaxis] * UR)
        zpred = _zscore_columns(pred)
        corr = np.mean(zPR_y * zpred, axis=0)
        corr = np.nan_to_num(corr)
        mean_corr = float(np.mean(corr))

        if mean_corr > best_mean_corr:
            best_mean_corr = mean_corr
            best_alpha = alpha

    return float(best_alpha), "held_out"


def _alignment_times(stimulus: Stimulus, layer_name: str) -> np.ndarray:
    """Return one event time per annotation from points or onset/offset pairs."""
    if layer_name not in stimulus.layers:
        raise ValueError(
            f"Stimulus {stimulus.stimulus_id!r} has no temporal alignment layer "
            f"{layer_name!r}"
        )

    times: list[float] = []
    layer = stimulus.layers[layer_name]
    for index in range(len(layer)):
        _span, value = layer[index]
        if np.isscalar(value):
            times.append(float(np.asarray(value).item()))
            continue
        try:
            onset, offset = value
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{layer_name!r} values must be timestamps or onset/offset pairs"
            ) from exc
        times.append((float(onset) + float(offset)) / 2)

    result = np.asarray(times, dtype=float)
    if not np.isfinite(result).all():
        raise ValueError(f"{layer_name!r} contains non-finite event times")
    return result


@dataclass(frozen=True)
class _Bands:
    """Per-unit regressors of one stimulus: controls, predictors, hidden layers."""

    controls: np.ndarray
    predictors: np.ndarray
    layers: dict[int, np.ndarray]


@dataclass(frozen=True)
class _Fit:
    """Held-out predictions and standardization of one fold; weights are not kept."""

    predictions: np.ndarray
    observations: np.ndarray
    alpha: float
    selection: str
    feature_train: tuple[np.ndarray, np.ndarray]
    feature_test: tuple[np.ndarray, np.ndarray]
    response_train: tuple[np.ndarray, np.ndarray]
    response_test: tuple[np.ndarray, np.ndarray]


def _select_recordings(target, values, config):
    """Retain native recording clocks and the resolved target's values and axes."""
    selected = []
    for block in values.blocks:
        source = block.source
        if not isinstance(source, Recording):
            raise ValueError("Encoding targets require native Recording context")
        if (
            config.participants is not None
            and source.participant_id not in config.participants
        ):
            continue
        if not {"sample", "time"}.issubset(block.index.names):
            raise ValueError("Encoding targets must retain their native sample index")
        coordinates = block.index.project(("sample", "time"))
        expected = tuple(zip(range(source.n_samples), source.times, strict=True))
        if coordinates != expected:
            raise ValueError(
                "Encoding targets must preserve the native recording clock"
            )
        if block.values.ndim < 2:
            raise ValueError(
                "Encoding responses need time plus at least one response dimension"
            )
        if block.values.dtype.kind not in "biuf":
            raise ValueError("Encoding targets require real numeric values")
        if not block.eligible.all() or not np.isfinite(block.values).all():
            raise ValueError(
                "Encoding targets must be finite and eligible; handle missing values explicitly before evaluation"
            )
        recording = replace(
            source,
            values=block.values,
            measure_name=target.name,
            dimension_names=("time", *block.value_dims),
            dimension_labels={
                name: labels
                for name, labels in zip(
                    block.value_dims, block.value_labels, strict=True
                )
                if labels is not None
            }
            if block.value_labels
            else {},
            metadata=dict(block.metadata),
        )
        selected.append((recording, block))
    if not selected:
        raise ValueError("No recordings for the selected target and participants")
    return selected


def _unit_anchor(response, block, config):
    """Use the recording's material and presentation coordinates on its unit grid."""
    layer = config.feature_layer
    if layer not in response.stimulus.layers:
        raise ValueError(f"Missing feature layer {layer!r}")
    coordinate_names = tuple(
        name
        for name in block.index.names
        if name not in {"material", "stimulus", "recording_source", "sample", "time"}
    )
    keys = block.index.project(("material", *coordinate_names))
    if not keys or any(key != keys[0] for key in keys):
        raise ValueError("A recording must have one material and presentation identity")
    material, *presentation = keys[0]
    index = FeatureIndex(
        (*UNIT_INDEX_NAMES, *coordinate_names),
        tuple(
            (
                material,
                response.stimulus.stimulus_id,
                layer,
                span.start,
                span.end,
                i,
                *presentation,
            )
            for i, span in enumerate(response.stimulus.layers[layer].spans)
        ),
    )
    return FeatureValues(
        (
            FeatureBlock(
                np.empty(len(index)),
                index,
                stimulus=response.stimulus,
                layer=layer,
                participant_id=response.participant_id,
                metadata=block.metadata,
            ),
        )
    )


def _unit_bands(spec, config, response, block, resolved) -> _Bands:
    """Align resolved scalar or vector quantities to a recording's event units."""
    anchor = _unit_anchor(response, block, config)
    n_units = len(anchor.blocks[0].index)
    controls: list[np.ndarray] = []
    predictors: list[np.ndarray] = []
    layers = {}
    for role, features in ((controls, spec.controls), (predictors, spec.predictors)):
        for feature in features:
            quantity = resolved[id(feature)]
            aligned = align_values(quantity, anchor)[0]
            if aligned.values.dtype.kind not in "biuf":
                raise ValueError(f"Encoding requires numeric feature {feature.name!r}")
            array = np.asarray(aligned.values, dtype=float).reshape(n_units, -1).copy()
            assert aligned.eligible is not None
            array[~aligned.eligible] = np.nan
            layer = quantity.provenance.get("encoding_layer")
            if role is predictors and layer is not None:
                if layer in layers:
                    raise ValueError(
                        "Each encoding layer must identify one predictor; combine vector features explicitly"
                    )
                layers[layer] = array
            else:
                role.append(array)

    def stack(arrays):
        return np.hstack(arrays) if arrays else np.empty((n_units, 0))

    return _Bands(stack(controls), stack(predictors), layers)


def _timed_units(
    response: Recording, bands: _Bands, config: EncodingConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Unit onsets on the response clock, and the mask of units the design uses.

    The recording's own ``event_times`` win over the stimulus alignment layer.
    A unit without an onset or with a non-finite control or predictor value
    is left out.
    """
    if response.event_times is not None:
        event_times = np.asarray(response.event_times, dtype=float)
    else:
        event_times = _alignment_times(response.stimulus, config.alignment_layer)
    n_units = len(event_times)
    for matrix in (bands.controls, bands.predictors, *bands.layers.values()):
        if len(matrix) != n_units:
            raise ValueError(
                f"Feature model returned {len(matrix)} rows for {n_units} aligned "
                f"events in stimulus {response.stimulus.stimulus_id!r}"
            )
    timed = (
        np.isfinite(event_times)
        & np.isfinite(bands.controls).all(axis=1)
        & np.isfinite(bands.predictors).all(axis=1)
    )
    if not timed.any():
        raise ValueError(
            f"Recording of {response.stimulus.stimulus_id!r} for participant "
            f"{response.participant_id!r} has no timed events"
        )
    return event_times, timed


def _delays_in_samples(config: EncodingConfig, response: Recording) -> tuple[int, ...]:
    """The FIR delays of one recording, in samples."""
    if config.delay_unit == "samples":
        return tuple(int(delay) for delay in config.fir_delays)
    if response.sample_interval is None:
        raise ValueError("delay_unit='seconds' needs a regular recording")
    return tuple(
        int(round(delay / response.sample_interval)) for delay in config.fir_delays
    )


def _align(
    values: np.ndarray, event_times: np.ndarray, response: Recording, config
) -> np.ndarray:
    """Put per-unit values on the response clock with the configured kernel."""
    times = response.times
    if config.kernel == "lanczos":
        return lanczos_interpolate(
            values, event_times, times, window=config.lanczos_window
        )
    # Impulse: each unit lands on its nearest sample; units beyond half a
    # sample outside the recording are dropped.
    design = np.zeros((len(times), values.shape[1]))
    spacing = (
        response.sample_interval
        if response.sample_interval is not None
        else float(np.median(np.diff(times)))
        if len(times) > 1
        else 1.0
    )
    inside = (event_times >= times[0] - spacing / 2) & (
        event_times <= times[-1] + spacing / 2
    )
    onsets = event_times[inside]
    right = np.clip(np.searchsorted(times, onsets), 1, len(times) - 1)
    left = right - 1
    nearest = np.where(onsets - times[left] <= times[right] - onsets, left, right)
    np.add.at(design, nearest, values[inside])
    return design


def _design(
    bands: _Bands,
    timed: np.ndarray,
    event_times: np.ndarray,
    response: Recording,
    config: EncodingConfig,
    layer: int | None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """FIR designs of one recording: the full model of a layer, and the baseline."""
    delays = _delays_in_samples(config, response)
    parts = [bands.controls[timed], bands.predictors[timed]]
    if layer is not None:
        parts.append(bands.layers[layer][timed])
    full = make_fir_delays(
        _align(np.hstack(parts), event_times[timed], response, config), delays
    )
    if not bands.controls.shape[1]:
        return full, None
    baseline = make_fir_delays(
        _align(bands.controls[timed], event_times[timed], response, config), delays
    )
    return full, baseline


def _fit_fold(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    config: EncodingConfig,
    rng: random.Random,
) -> tuple[_Fit, np.ndarray]:
    """Select the ridge alpha on the training split, fit and predict.

    Returns the fold record and the ridge weights, which the caller drops
    once any artifact has been emitted: they are large.
    """
    selection_X, selection_y = X_train, y_train
    X_train, X_test, feature_train, feature_test = _normalize_splits(
        X_train, X_test, config.normalization
    )
    y_train, y_test, response_train, response_test = _normalize_splits(
        y_train, y_test, config.normalization
    )
    training_normalization = config.normalization == "training"
    alpha, selection = _select_alpha_bootstrap(
        selection_X if training_normalization else X_train,
        selection_y if training_normalization else y_train,
        np.asarray(config.alphas, dtype=float),
        chunklen=config.alpha_chunk_length,
        nchunks=config.alpha_nchunks,
        rng=rng,
        normalize_training=training_normalization,
    )
    y_pred, weights = _ridge_svd(X_train, y_train, X_test, alpha)
    fit = _Fit(
        y_pred,
        y_test,
        alpha,
        selection,
        tuple(feature_train),
        tuple(feature_test),
        tuple(response_train),
        tuple(response_test),
    )
    return fit, weights


def _scores(fits: list[_Fit], config: EncodingConfig) -> tuple[np.ndarray, np.ndarray]:
    """Score pooled predictions in original or explicitly within-split coordinates."""
    observations = np.vstack([fit.observations for fit in fits])
    predictions = np.vstack([fit.predictions for fit in fits])
    if config.normalization == "training":
        start = 0
        for fit in fits:
            stop = start + len(fit.observations)
            observations[start:stop] *= fit.response_test[1]
            observations[start:stop] += fit.response_test[0]
            predictions[start:stop] *= fit.response_train[1]
            predictions[start:stop] += fit.response_train[0]
            start = stop
    correlations = _columnwise_pearson(observations, predictions)
    # One (samples, channels) temporary at a time: the arrays are large.
    work = observations - predictions
    np.square(work, out=work)
    residual = work.sum(axis=0)
    work = observations - observations.mean(axis=0)
    np.square(work, out=work)
    total = work.sum(axis=0)
    del work
    with np.errstate(divide="ignore", invalid="ignore"):
        r2 = np.where(total > 0, 1 - residual / total, np.nan)
    return correlations, r2


def _metrics(
    correlations: np.ndarray, r2: np.ndarray, baseline_r2: np.ndarray | None
) -> dict:
    """Channel-averaged scores; means run over the channels where defined."""

    def mean(values: np.ndarray) -> float:
        finite = np.isfinite(values)
        return float(np.mean(values[finite])) if finite.any() else float("nan")

    finite = np.isfinite(correlations)
    positive = finite & (correlations > 0)
    metrics = {
        "mean_r": mean(correlations),
        "mean_positive_r": mean(correlations[positive])
        if positive.any()
        else float("nan"),
        "n_channels": len(correlations),
        "n_positive_channels": int(positive.sum()),
        "n_finite_channels": int(finite.sum()),
        "n_undefined_channels": int((~finite).sum()),
        "mean_r2": mean(r2),
    }
    if baseline_r2 is not None:
        metrics["mean_baseline_r2"] = mean(baseline_r2)
        metrics["mean_delta_r2"] = mean(r2 - baseline_r2)
    return metrics


def _recording_folds(
    n_recordings: int, n_folds: int
) -> list[tuple[list[int], list[int]]]:
    """Deal recordings round-robin to folds; each fold holds its own out."""
    n = min(n_folds, n_recordings)
    return [
        (
            [i for i in range(n_recordings) if i % n != fold],
            [i for i in range(n_recordings) if i % n == fold],
        )
        for fold in range(n)
    ]


def _validate_sample_folds(origins, folds):
    """Reject original target contributors shared by fitted and held-out samples."""
    masks = []
    for train, test in folds:
        train_mask = np.zeros(len(origins), dtype=bool)
        test_mask = np.zeros(len(origins), dtype=bool)
        train_mask[train], test_mask[test] = True, True
        masks.append((train_mask, test_mask))
    validate_fold_origins(origins, {}, masks)


class _Evaluation:
    """One ``evaluate()`` call over sampled responses."""

    def __init__(
        self,
        dataset: Dataset,
        spec: EvaluationSpec,
        config: EncodingConfig,
        language: str | None,
        callback: Callable[[EncodingFoldArtifact], None] | None,
        target,
        resolved,
    ) -> None:
        self.dataset = dataset
        self.spec = spec
        self.config = config
        self.language = language
        self.callback = callback
        self.target = target
        self.resolved = resolved
        self.target_values = resolved[id(target)]
        selected = _select_recordings(target, self.target_values, config)
        self.recordings = [recording for recording, _ in selected]
        self.blocks = {id(recording): block for recording, block in selected}
        layers = [
            resolved[id(feature)].provenance.get("encoding_layer")
            for feature in spec.predictors
        ]
        self.fits = [layer for layer in layers if layer is not None] or [None]
        self.model_name = _model_name(
            [resolved[id(feature)] for feature in (*spec.predictors, *spec.controls)]
        )
        self.bands: dict[int, _Bands] = {}
        target_origins = set().union(
            *(
                set().union(*target_outcome_origins(self.target_values, block))
                for _, block in selected
            )
        )
        for feature in (*spec.predictors, *spec.controls):
            values = resolved[id(feature)]
            if any(
                origin & target_origins
                for block in values.blocks
                for origin in block_origins(values, block)
            ):
                raise ValueError("Encoding inputs cannot depend on target observations")
        # The baseline draws its own chunk shuffles, so the full model's
        # match a run without controls.
        self.alpha_rng = random.Random(config.random_seed)
        self.baseline_rng = random.Random(config.random_seed)

    def bands_for(self, response) -> _Bands:
        """Align feature values once for each recording's presentation."""
        key = id(response)
        if key not in self.bands:
            self.bands[key] = _unit_bands(
                self.spec, self.config, response, self.blocks[key], self.resolved
            )
        return self.bands[key]

    def run(self) -> list[EncodingResult]:
        """Score every fit the config asks for."""
        if self.config.cv == "temporal":
            return [
                analysis
                for index, response in enumerate(self.recordings)
                for analysis in self._temporal(index, response)
            ]
        groups: dict[str, list[Recording]] = {}
        for response in self.recordings:
            groups.setdefault(response.participant_id, []).append(response)
        return [
            analysis
            for participant, responses in groups.items()
            for analysis in self._pooled(participant, responses)
        ]

    def _temporal(self, index: int, response: Recording) -> list[EncodingResult]:
        """Buffered temporal folds inside one recording, one fit per layer."""
        config = self.config
        bands = self.bands_for(response)
        event_times, timed = _timed_units(response, bands, config)
        Y = np.asarray(response.values, dtype=float).reshape(response.n_samples, -1)
        folds = list(
            TemporalKFold(n_folds=config.n_folds, buffer_trs=config.buffer_trs).split(
                response.n_samples
            )
        )
        _validate_sample_folds(
            target_outcome_origins(self.target_values, self.blocks[id(response)]), folds
        )
        _, X_base = _design(bands, timed, event_times, response, config, None)
        baseline = None
        if X_base is not None:
            baseline = _scores(
                [
                    _fit_fold(
                        X_base[train],
                        X_base[test],
                        Y[train],
                        Y[test],
                        config,
                        self.baseline_rng,
                    )[0]
                    for train, test in folds
                ],
                config,
            )
        analyses = []
        for layer in self.fits:
            X, _ = _design(bands, timed, event_times, response, config, layer)
            fits = []
            for fold_index, (train, test) in enumerate(folds):
                fit, weights = _fit_fold(
                    X[train],
                    X[test],
                    Y[train],
                    Y[test],
                    config,
                    self.alpha_rng,
                )
                fits.append(fit)
                if self.callback is not None:
                    self.callback(
                        self._artifact(
                            response,
                            index,
                            layer,
                            fold_index,
                            train,
                            test,
                            X,
                            fit,
                            weights,
                        )
                    )
                del weights
            analyses.append(
                self._analysis(
                    response.participant_id,
                    layer,
                    fits,
                    baseline,
                    response,
                    {
                        "n_trs": response.n_samples,
                        "n_samples": response.n_samples,
                        "n_units": int(timed.sum()),
                        "fir_delays_samples": list(
                            _delays_in_samples(config, response)
                        ),
                        "stimulus_id": response.stimulus.stimulus_id,
                        "recording_metadata": dict(response.metadata),
                        "stimulus_metadata": dict(response.stimulus.metadata),
                    },
                )
            )
        return analyses

    def _pooled(
        self, participant: str, responses: list[Recording]
    ) -> list[EncodingResult]:
        """Hold out whole recordings and normalize under the configured policy."""
        config = self.config
        if len(responses) < 2:
            raise ValueError(
                f"cv='recording' needs at least two recordings per participant; "
                f"{participant!r} has {len(responses)}"
            )
        if len({r.values.shape[1:] for r in responses}) != 1:
            raise ValueError(
                f"the recordings of {participant!r} differ in response shape"
            )
        first = responses[0]
        for response in responses[1:]:
            if tuple(response.dimension_names[1:]) != tuple(
                first.dimension_names[1:]
            ) or any(
                tuple(response.dimension_labels.get(dim, ()))
                != tuple(first.dimension_labels.get(dim, ()))
                for dim in first.dimension_names[1:]
            ):
                raise ValueError(
                    f"the recordings of {participant!r} differ in response "
                    "dimensions or channel labels; align response coordinates "
                    "before pooling"
                )
        prepared = []
        for response in responses:
            bands = self.bands_for(response)
            event_times, timed = _timed_units(response, bands, config)
            Y = np.asarray(response.values, dtype=float).reshape(response.n_samples, -1)
            if config.normalization == "within_split":
                Y = _standardize_columns(Y)[0]
            prepared.append((response, bands, event_times, timed, Y))
        folds = _recording_folds(len(responses), config.n_folds)
        _validate_sample_folds(
            tuple(
                frozenset().union(
                    *target_outcome_origins(
                        self.target_values, self.blocks[id(response)]
                    )
                )
                for response in responses
            ),
            folds,
        )

        def designs(layer: int | None, which: int) -> list[np.ndarray | None]:
            return [
                None
                if (
                    matrix := _design(bands, timed, times, response, config, layer)[
                        which
                    ]
                )
                is None
                else (
                    _standardize_columns(matrix)[0]
                    if config.normalization == "within_split"
                    else matrix
                )
                for response, bands, times, timed, _ in prepared
            ]

        fold_config = (
            replace(config, normalization="none")
            if config.normalization == "within_split"
            else config
        )

        def fit_all(X: list[np.ndarray | None], rng: random.Random) -> list[_Fit]:
            matrices = []
            for matrix in X:
                if matrix is None:
                    raise ValueError("Cannot fit encoding without a design matrix")
                matrices.append(matrix)
            return [
                _fit_fold(
                    np.vstack([matrices[i] for i in train]),
                    np.vstack([matrices[i] for i in test]),
                    np.vstack([prepared[i][4] for i in train]),
                    np.vstack([prepared[i][4] for i in test]),
                    fold_config,
                    rng,
                )[0]
                for train, test in folds
            ]

        baseline = None
        X_base = designs(None, 1)
        if X_base[0] is not None:
            baseline = _scores(fit_all(X_base, self.baseline_rng), config)
        return [
            self._analysis(
                participant,
                layer,
                fit_all(designs(layer, 0), self.alpha_rng),
                baseline,
                first,
                {
                    "n_samples": int(sum(r.n_samples for r in responses)),
                    "response_dimension_labels": {
                        dim: labels
                        for dim, labels in first.dimension_labels.items()
                        if dim != "time"
                    },
                    "recording_folds": [
                        {"train": train, "test": test} for train, test in folds
                    ],
                    "n_units": int(sum(int(timed.sum()) for *_, timed, _ in prepared)),
                    "fir_delays_samples": [
                        list(_delays_in_samples(config, r)) for r in responses
                    ],
                    "stimulus_id": [r.stimulus.stimulus_id for r in responses],
                    "recording_metadata": [dict(r.metadata) for r in responses],
                    "stimulus_metadata": [dict(r.stimulus.metadata) for r in responses],
                },
            )
            for layer in self.fits
        ]

    def _artifact(
        self, response, index, layer, fold_index, train, test, X, fit, weights
    ):
        config = self.config
        return EncodingFoldArtifact(
            dataset_name=self.dataset.info.name,
            model_name=self.model_name,
            layer=layer,
            participant_id=response.participant_id,
            stimulus_id=response.stimulus.stimulus_id,
            recording_index=index,
            fold_index=fold_index,
            modality=response.modality,
            response_measure=response.measure_name,
            language=self.language,
            config=config,
            response_shape=response.values.shape[1:],
            response_dimensions=response.dimension_names,
            response_dimension_labels=_artifact_metadata(response.dimension_labels),
            recording_metadata=_artifact_metadata(response.metadata),
            train_indices=train.copy(),
            test_indices=test.copy(),
            test_times=response.times[test],
            alpha=fit.alpha,
            test_design=X[test],
            weights=weights,
            predictions=fit.predictions.copy(),
            observations=fit.observations.copy(),
            feature_train_mean=fit.feature_train[0].copy(),
            feature_train_scale=fit.feature_train[1].copy(),
            feature_test_mean=fit.feature_test[0].copy(),
            feature_test_scale=fit.feature_test[1].copy(),
            response_train_mean=fit.response_train[0].copy(),
            response_train_scale=fit.response_train[1].copy(),
            response_test_mean=fit.response_test[0].copy(),
            response_test_scale=fit.response_test[1].copy(),
        )

    def _analysis(
        self, participant, layer, fits, baseline, response, extra
    ) -> EncodingResult:
        config = self.config
        correlations, r2 = _scores(fits, config)
        baseline_correlations, baseline_r2 = baseline if baseline else (None, None)
        return EncodingResult(
            dataset_name=self.dataset.info.name,
            model_name=self.model_name,
            layer=layer,
            participant_id=participant,
            correlations=correlations,
            modality=response.modality,
            response_measure=response.measure_name,
            language=self.language,
            metrics=_metrics(correlations, r2, baseline_r2),
            metadata={
                "n_folds": config.n_folds,
                "cv": config.cv,
                "normalization": config.normalization,
                "scoring_scale": (
                    "within_split"
                    if config.normalization == "within_split"
                    else "original_response"
                ),
                "kernel": config.kernel,
                "fir_delays": list(config.fir_delays),
                "delay_unit": config.delay_unit,
                "alignment_layer": config.alignment_layer,
                "feature_layer": config.feature_layer,
                "controls": [feature.name for feature in self.spec.controls],
                "predictors": [feature.name for feature in self.spec.predictors],
                "feature_measurements": [
                    describe_feature(feature)
                    for feature in [
                        *self.spec.controls,
                        *self.spec.predictors,
                    ]
                ],
                "response_shape": response.values.shape[1:],
                "response_dimensions": response.dimension_names,
                "response_dimension_labels": dict(response.dimension_labels),
                "alpha_parameterization": "squared",
                "candidate_alphas": list(config.alphas),
                "selected_alphas": [fit.alpha for fit in fits],
                "alpha_selection": [fit.selection for fit in fits],
                "alpha_chunk_length": config.alpha_chunk_length,
                "alpha_nchunks": config.alpha_nchunks,
                "random_seed": config.random_seed,
                "feature_sources": {
                    feature.name: dict(self.resolved[id(feature)].provenance)
                    for feature in (*self.spec.controls, *self.spec.predictors)
                },
                "target_source": dict(self.target_values.provenance),
                **extra,
            },
            r2=r2,
            baseline_correlations=baseline_correlations,
            baseline_r2=baseline_r2,
        )


def _evaluate_sampled_response(
    dataset: Dataset,
    spec: EvaluationSpec,
    config: EncodingConfig,
    *,
    language: str | None = None,
    encoding_fold_callback: Callable[[EncodingFoldArtifact], None] | None = None,
) -> list[EncodingResult]:
    """Sampled-response implementation behind ``evaluate()``.

    Per recording (``cv="temporal"``) or participant (``cv="recording"``) and
    per layer: build the FIR design from the controls, predictors and the
    layer's hidden states on the response clock, fit a ridge per fold with
    the alpha selected on the training split, and score the held-out
    predictions per channel. The controls alone are fitted once more as the
    baseline.

    Returns:
        One EncodingResult per layer and recording, or per layer and
        participant with ``cv="recording"``.
    """
    spec.__post_init__()
    if spec.comparison != "joint":
        raise ValueError("Encoding supports only joint predictor comparisons")
    if spec.segmentation is not None:
        raise ValueError(
            "Encoding uses event alignment and does not support segmentation"
        )
    if spec.observations is not None:
        raise ValueError(
            "Encoding uses each target's native recording index; observations is not supported"
        )
    if not spec.predictors:
        raise ValueError("Encoding requires at least one predictor")
    if encoding_fold_callback is not None and config.cv != "temporal":
        raise ValueError("encoding_fold_callback needs cv='temporal'")
    selection = _select_recording_features(spec.targets, config.participants)
    targets, materials = (spec.targets, None) if selection is None else selection
    requested = [*targets, *spec.predictors, *spec.controls]
    values = _resolve_features(
        dataset, requested, layer=config.feature_layer, model_materials=materials
    )
    resolved = {
        id(feature): value
        for feature, value in zip(
            [*spec.targets, *spec.predictors, *spec.controls], values, strict=True
        )
    }
    return [
        result
        for target in spec.targets
        for result in _Evaluation(
            dataset, spec, config, language, encoding_fold_callback, target, resolved
        ).run()
    ]
