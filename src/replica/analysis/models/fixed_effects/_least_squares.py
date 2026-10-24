"""Column-normalized least squares and rank-based residual variance."""

import numpy as np


def column_norms(design: np.ndarray) -> np.ndarray:
    """Euclidean column norms without squaring unscaled entries."""
    largest = np.max(np.abs(design), axis=-2)
    scaled = np.divide(
        design,
        largest[..., None, :],
        out=np.zeros_like(design),
        where=largest[..., None, :] > 0,
    )
    with np.errstate(over="ignore"):
        norms = largest * np.linalg.norm(scaled, axis=-2)
    if not np.isfinite(norms).all():
        raise ValueError("Design column norms exceed floating-point range")
    return norms


def normalized_design(design: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return unit-norm columns and their scales; zero columns stay zero."""
    scale = column_norms(design)
    scale = np.where(scale > 0, scale, 1.0)
    return design / scale[..., None, :], scale


def least_squares(
    design: np.ndarray,
    response: np.ndarray,
    *,
    n_observations: np.ndarray | int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return coefficients, rank and a unit-variance covariance factor.

    Columns have unit Euclidean norm before SVD. Dependent columns use the
    minimum-norm solution in those coordinates. The cutoff is machine epsilon
    times the larger of the observation and coefficient counts.

    Args:
        design: Design matrices, with observations on the penultimate axis.
        response: Corresponding response vectors.
        n_observations: Original row counts when fitting QR-compressed data.
    """
    normalized, scale = normalized_design(design)
    u, singular, vh = np.linalg.svd(normalized, full_matrices=False)
    count = design.shape[-2] if n_observations is None else n_observations
    cutoff = np.finfo(float).eps * np.maximum(count, design.shape[-1])
    retained = singular > (cutoff * singular[..., 0])[..., None]
    inverse = np.divide(1.0, singular, out=np.zeros_like(singular), where=retained)
    basis = np.swapaxes(vh, -1, -2) * inverse[..., None, :]
    basis /= scale[..., :, None]
    projected = np.einsum("...ij,...i->...j", u, response)
    beta = np.einsum("...ij,...j->...i", basis, projected)
    return beta, retained.sum(axis=-1), basis


def residual_variance(rss, n_observations, rank, correction: str):
    """Return residual variance, or NaN when its denominator is nonpositive."""
    denominator = n_observations - rank if correction == "df" else n_observations
    return np.divide(
        rss,
        denominator,
        out=np.full(np.shape(rss), np.nan),
        where=denominator > 0,
    )
