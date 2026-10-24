"""The assessed variance correction for fully refitted two-factor CV gains."""

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import t

METHOD = "finite-crossed-lower-priority-partition-df-v1"
SERIES = ("axis_0", "axis_1", "joint")


@dataclass(frozen=True)
class PredictiveTestResult:
    """Mean paired gain and approximate inference, with explicit availability.

    Gain is in nats per response. Intervals and p-values are unadjusted for
    multiple comparisons. An unavailable result never substitutes another test.
    """

    gain: float | None
    available: bool
    reason: str | None = None
    standard_error: float | None = None
    pvalue: float | None = None
    confidence_interval: tuple[float, float] | None = None
    degrees_of_freedom: int | None = None
    variance_components: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)


def _summarize(original, samples, groups, confidence_level):
    """Combine complete cycle gains using the frozen finite-sample rule."""
    original = np.asarray(original, dtype=float)
    if original.ndim != 1 or len(original) < 2:
        raise ValueError("Supply at least two original-data cycle gains")
    if set(samples) != set(SERIES):
        raise ValueError("Supply both one-axis bootstrap series and the joint series")
    samples = {k: np.asarray(v, dtype=float) for k, v in samples.items()}
    shapes = {v.shape for v in samples.values()}
    if len(shapes) != 1:
        raise ValueError("Bootstrap series must share draw and cycle counts")
    shape = next(iter(shapes))
    if len(shape) != 2 or min(shape) < 2:
        raise ValueError("Each series needs at least two draws and two cycles per draw")
    if not all(np.isfinite(a).all() for a in [original, *samples.values()]):
        raise ValueError("Failed or missing cycles cannot be omitted from inference")
    if len(groups) != 2 or any(
        isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 2
        for n in groups
    ):
        raise ValueError("Two original group counts of at least two are required")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must lie strictly between zero and one")
    gain = float(original.mean())
    partition = float(original.var(ddof=1) / len(original))
    components = {}
    for name, values in samples.items():
        # Off-diagonal covariance removes independently redrawn partition noise.
        covariance = np.cov(values, rowvar=False, ddof=1)
        mask = ~np.eye(shape[1], dtype=bool)
        components[name] = float(covariance[mask].mean())
    vr, vt, vrt = (components[s] for s in SERIES)
    ar, at = (n / (n - 1) for n in groups)
    raw = (ar + ar * at) * vr + (at + ar * at) * vt - ar * at * vrt
    lower, upper = max(0.0, ar * vr, at * vt), max(ar, at) * vrt
    bounded = max(lower, min(upper, raw))
    variance = bounded + partition
    # The assessed candidate preserved the prior rule's unavailable outcomes.
    prior_variance = min(vrt, max(ar * vr, at * vt, raw)) + partition
    details = {
        **components,
        "original_partition_variance": partition,
        "raw": raw,
        "lower": lower,
        "upper": upper,
        "bounds_conflict": lower > upper,
        "bounded": bounded,
        "variance": variance,
        "prior_variance": prior_variance,
    }
    metadata = {
        "method": METHOD,
        "alternative": "two-sided",
        "confidence_level": confidence_level,
        "draws_per_series": shape[0],
        "cycles_per_draw": shape[1],
        "original_cycles": len(original),
        "group_counts": list(groups),
        "multiplicity_adjusted": False,
    }
    if not np.isfinite(prior_variance) or prior_variance <= 0:
        return PredictiveTestResult(
            gain,
            False,
            "nonpositive or nonfinite variance under the assessed availability rule",
            variance_components=details,
            metadata=metadata,
        )
    if not np.isfinite(variance) or variance <= 0:
        return PredictiveTestResult(
            gain,
            False,
            "nonpositive or nonfinite corrected variance",
            variance_components=details,
            metadata=metadata,
        )
    se = float(np.sqrt(variance))
    df = min(*groups, len(original)) - 1
    pvalue = float(2 * t.sf(abs(gain) / se, df))
    width = float(t.ppf((1 + confidence_level) / 2, df) * se)
    return PredictiveTestResult(
        gain,
        True,
        standard_error=se,
        pvalue=pvalue,
        confidence_interval=(gain - width, gain + width),
        degrees_of_freedom=df,
        variance_components=details,
        metadata=metadata,
    )
