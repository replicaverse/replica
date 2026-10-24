"""Per-fold regression mechanics for :func:`evaluate`.

The low-level fold helpers: design-matrix assembly, CV splitting, per-fold
orthogonalization planning, optional design-matrix dump, baseline/full
prediction fitting, and feature-stat averaging. The fold orchestration
(AnalysisResult / FoldResult assembly) lives in ``fitting/regression.py``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from replica.analysis.contracts.models import AnalysisModel, StatsPredictor
from replica.analysis.cv.splitters import CVSplitter, SentenceAwareSplitter
from replica.analysis.fitting.scoring import _fit_predict_sigma
from replica.contracts.features import Feature

logger = logging.getLogger("replica")


def _build_design_matrices(
    feat_flat: dict[str, np.ndarray],
    features: list[Feature],
    control_features: list[Feature],
    n_rows: int,
) -> tuple[list[str], np.ndarray, list[np.ndarray], np.ndarray, int]:
    """Build unstandardized baseline and full design matrices."""
    ctrl_names = [f.name for f in control_features]
    X_ctrl_raw = (
        np.column_stack([feat_flat[n] for n in ctrl_names])
        if ctrl_names
        else np.empty((n_rows, 0))
    )

    n_ctrl = X_ctrl_raw.shape[1]
    feat_cols = [feat_flat[f.name] for f in features]
    if n_ctrl > 0:
        X_full_raw = np.column_stack([X_ctrl_raw, *feat_cols])
    elif len(feat_cols) == 1:
        X_full_raw = feat_cols[0].reshape(-1, 1)
    else:
        X_full_raw = np.column_stack(feat_cols)
    return ctrl_names, X_ctrl_raw, feat_cols, X_full_raw, n_ctrl


def _cv_splits(
    cv_splitter: CVSplitter,
    y: np.ndarray,
    groups: np.ndarray,
    participants: np.ndarray,
    items: np.ndarray,
    sentence_ids: np.ndarray | None,
    participant_ids: np.ndarray | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Resolve CV splits, passing sentence/participant ids only to splitters
    that need them (sentence-based partitions).
    """
    if (
        isinstance(cv_splitter, SentenceAwareSplitter)
        and cv_splitter.needs_sentence_ids
        and sentence_ids is not None
    ):
        splits = list(
            cv_splitter.split(
                y,
                groups,
                participants,
                items,
                sentence_ids=sentence_ids,
                participant_ids=participant_ids,
            )
        )
    else:
        splits = list(cv_splitter.split(y, groups, participants, items))
    for i, (train, test) in enumerate(splits):
        if any(
            not isinstance(mask, np.ndarray)
            or mask.dtype != np.bool_
            or mask.shape != y.shape
            for mask in (train, test)
        ):
            raise ValueError(
                f"CV fold {i} must provide boolean masks of shape {y.shape}"
            )
    return splits


def _dump_design_matrix(
    dump_path: str | None,
    *,
    ctrl_names: list[str],
    X_ctrl_raw: np.ndarray,
    features: list[Feature],
    feat_cols: list[np.ndarray],
    participants: np.ndarray,
    items: np.ndarray,
    y: np.ndarray,
) -> None:
    """Optionally write the merged design matrix for inspection."""
    if not dump_path:
        return
    cols: dict[str, np.ndarray] = {}
    for i, name in enumerate(ctrl_names):
        cols[name] = X_ctrl_raw[:, i]
    for j, feat in enumerate(features):
        cols[feat.name] = feat_cols[j]
    cols["participant_id"] = participants
    cols["item_id"] = items
    cols["y"] = y
    df = pd.DataFrame(cols)
    path = Path(dump_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path, index=False)
    except Exception as exc:
        # E.g. a column dtype parquet cannot serialize; TSV is the
        # fallback. Name the cause so it isn't a silent format swap.
        path = path.with_suffix(".tsv")
        logger.warning(
            "    [dump] parquet unavailable (%s); wrote TSV -> %s", exc, path
        )
        df.to_csv(path, sep="\t", index=False)
    logger.info(
        "    [dump] merged design matrix -> %s  (%d rows x %d cols)",
        path,
        len(df),
        df.shape[1],
    )


def _append_grouping_columns(
    X_train: np.ndarray,
    X_test: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    participants: np.ndarray,
    items: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Append participant/item grouping columns for mixed-effect backends."""
    grp_train = np.column_stack([participants[train_mask], items[train_mask]])
    grp_test = np.column_stack([participants[test_mask], items[test_mask]])
    return np.column_stack([X_train, grp_train]), np.column_stack([X_test, grp_test])


def _fit_baseline_predictions(
    analysis_model: AnalysisModel,
    y_train: np.ndarray,
    y_test_size: int,
    X_ctrl_train: np.ndarray,
    X_ctrl_test: np.ndarray,
    *,
    use_training_sigma: bool,
    force_model: bool = False,
) -> tuple[np.ndarray, float | None]:
    """Fit/predict the baseline model for one fold."""
    if X_ctrl_train.shape[1] == 0 and not force_model:
        pred = np.full(y_test_size, np.mean(y_train))
        sigma = float(np.std(y_train, ddof=1)) if use_training_sigma else None
        return pred, sigma
    if use_training_sigma:
        return _fit_predict_sigma(analysis_model, y_train, X_ctrl_train, X_ctrl_test)
    return analysis_model.fit_predict(y_train, X_ctrl_train, X_ctrl_test), None


def _fit_full_predictions(
    analysis_model: AnalysisModel,
    y_train: np.ndarray,
    X_full_train: np.ndarray,
    X_full_test: np.ndarray,
    *,
    use_training_sigma: bool,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, float | None]:
    """Fit/predict the full model for one fold, including optional stats."""
    if isinstance(analysis_model, StatsPredictor) and getattr(
        analysis_model, "provides_stats", True
    ):
        y_pred, beta, se, sigma_from_fit = analysis_model.fit_predict_with_stats(
            y_train, X_full_train, X_full_test
        )
        sigma = sigma_from_fit if use_training_sigma else None
        return y_pred, beta, se, sigma
    if use_training_sigma:
        y_pred, sigma = _fit_predict_sigma(
            analysis_model, y_train, X_full_train, X_full_test
        )
        return y_pred, None, None, sigma
    return (
        analysis_model.fit_predict(y_train, X_full_train, X_full_test),
        None,
        None,
        None,
    )


def _average_feature_stat(
    stats: list[np.ndarray],
    coeff_idx: int,
) -> float:
    """Average one coefficient/statistic across folds when available."""
    usable = [s for s in stats if coeff_idx < len(s)]
    return float(np.mean([s[coeff_idx] for s in usable])) if usable else float("nan")
