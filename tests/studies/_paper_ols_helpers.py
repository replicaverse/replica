import numpy as np
import pandas as pd

from replica.analysis.cv import SequentialBlockSplitter
from replica.analysis.evaluation.evaluate import evaluate
from replica.analysis.models import OLS


def _gaussian_ll_sum(y: np.ndarray, mu: np.ndarray, sigma: float) -> float:
    """Sum of Gaussian log-likelihood, evaluated with the exact reduction."""
    n = len(y)
    sigma2 = sigma * sigma
    if sigma2 == 0:
        return 0.0
    residuals = y - mu
    return float(
        -0.5 * n * np.log(2 * np.pi * sigma2) - 0.5 * np.sum(residuals**2) / sigma2
    )


def direct_fold_delta_lls(
    y: np.ndarray,
    X_full: np.ndarray,
    X_base: np.ndarray,
    n_folds: int = 10,
) -> list[float]:
    """Per-fold mean(LL_full) - mean(LL_base) on identical CV splits.

    The shared Pimentel et al. (2023) / Clark et al. (2025) OLS recipe, through
    :meth:`OLS.fit_predict_with_stats` so y_pred and sigma use the same numpy
    reductions as ``evaluate()``.

    Returns:
        Per sequential fold, the mean test-observation Gaussian log-likelihood
        of the full fit minus that of the base fit; each OLS fit estimates an
        MLE sigma from its train residuals.
    """
    ols = OLS(sigma_correction="mle")
    splitter = SequentialBlockSplitter(n_folds=n_folds)
    dummy = np.zeros(len(y), dtype=int)
    splits = list(splitter.split(y, dummy, dummy, dummy))

    fold_lls: list[float] = []
    for train_mask, test_mask in splits:
        y_train, y_test = y[train_mask], y[test_mask]
        n_test = int(test_mask.sum())

        pred_f, _, _, sigma_f = ols.fit_predict_with_stats(
            y_train, X_full[train_mask], X_full[test_mask]
        )
        pred_b, _, _, sigma_b = ols.fit_predict_with_stats(
            y_train, X_base[train_mask], X_base[test_mask]
        )

        full_ll = _gaussian_ll_sum(y_test, pred_f, sigma_f)
        base_ll = _gaussian_ll_sum(y_test, pred_b, sigma_b)
        fold_lls.append((full_ll - base_ll) / n_test)
    return fold_lls


def direct_from_dataframe(
    df: pd.DataFrame,
    target_cols: list[str],
    baseline_cols: list[str],
    dv_col: str = "time",
    n_folds: int = 10,
) -> dict:
    """Convenience wrapper around ``direct_fold_delta_lls`` for tabular data."""
    y = df[dv_col].values.astype(float)
    X_full = np.column_stack(
        [df[c].values.astype(float) for c in baseline_cols + target_cols]
    )
    X_base = np.column_stack([df[c].values.astype(float) for c in baseline_cols])
    fold_lls = direct_fold_delta_lls(y, X_full, X_base, n_folds=n_folds)
    return {
        "n_obs": len(df),
        "n_folds": len(fold_lls),
        "fold_delta_lls": fold_lls,
        "mean_delta_ll": float(np.mean(fold_lls)),
    }


def fold_delta_lls_from_replica(
    dataset,
    evaluation_spec,
    analysis_config,
) -> tuple[list[float], list[int]]:
    """Run the evaluate and return (per_fold_mean_delta_ll, n_test_per_fold).

    Requires ``analysis_config.per_fold=True``. REPLICA stores per-fold
    LL sums; converted to per-fold means by dividing by ``n_test``,
    which equals ``mean(LL_full) - mean(LL_base)`` exactly because
    both submodels are evaluated on the same test rows.
    """
    results = evaluate(dataset, evaluation_spec, analysis_config)
    if not results:
        raise RuntimeError("evaluate() returned no AnalysisResult")
    a = results[0]
    if not a.fold_results:
        raise RuntimeError("evaluate() did not populate fold_results")
    fold_means = [fr.delta_ll / fr.n_test for fr in a.fold_results]
    fold_ns = [fr.n_test for fr in a.fold_results]
    return fold_means, fold_ns
