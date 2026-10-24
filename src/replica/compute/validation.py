"""Check predictor structure and domains, with advisory distribution diagnostics.

Malformed values block cache reuse and publication. Magnitude, missingness,
and mean-outlier findings describe the data without deciding its scientific
eligibility.
"""

import math
import pickle
from dataclasses import dataclass, field
from pathlib import Path

# Advisory defaults use the stored value's units; callers may override them.
_EXTREME_VALUE_THRESHOLD = 100.0
_OUTLIER_SIGMA = 5.0
_NAN_BURST_FRACTION = 0.5

_SOFT_ISSUE_BUCKETS: frozenset = frozenset(
    {"mean_outlier", "single_unit_nan", "all_nan_stim", "nan_burst", "extreme_per_word"}
)


@dataclass
class PredictorPickleReport:
    """Result of running :func:`predictor_pickle_issues` on one pickle."""

    path: Path
    n_stims: int = 0
    issues: dict[str, list[str]] = field(default_factory=dict)
    primary_keys: set[str] = field(default_factory=set)

    def add(self, bucket: str, detail: str, cap: int = 5) -> None:
        """Append one example detail to *bucket*, capped at *cap* entries.

        Args:
            bucket: Issue category to file this detail under, e.g.
                ``"nan_burst"`` or ``"extreme_per_word"``.
            detail: Human-readable description of the offending stim or word.
            cap: Maximum number of example details kept for this bucket;
                later occurrences in an already-full bucket are dropped.
        """
        bucket_list = self.issues.setdefault(bucket, [])
        if len(bucket_list) < cap:
            bucket_list.append(detail)

    @property
    def is_clean(self) -> bool:
        """True iff *no* issues at all (including soft ones).

        Useful for audit-listing output. Quarantine decisions should
        use :attr:`has_hard_issues` instead so soft issues like
        ``mean_outlier`` warn without blocking the pickle.
        """
        return not self.issues

    @property
    def has_hard_issues(self) -> bool:
        """True iff any *hard* issue is present.

        Hard issues block the pickle from being used (rename to
        ``.suspect.pkl``); soft issues (see :data:`_SOFT_ISSUE_BUCKETS`)
        are advisory only.
        """
        return any(b not in _SOFT_ISSUE_BUCKETS for b in self.issues)

    @property
    def soft_only(self) -> bool:
        """True iff issues exist but all are soft (advisory only)."""
        return bool(self.issues) and not self.has_hard_issues

    def record(self, *, failed: int, missing: int) -> dict:
        """Manifest block of this audit.

        Args:
            failed: Stimuli that raised.
            missing: Requested stimuli absent from the payload.
        """
        return {
            "primary_keys": sorted(self.primary_keys),
            "hard_issues": sorted(
                bucket for bucket in self.issues if bucket not in _SOFT_ISSUE_BUCKETS
            ),
            "soft_issues": sorted(
                bucket for bucket in self.issues if bucket in _SOFT_ISSUE_BUCKETS
            ),
            "examples": {
                bucket: list(items) for bucket, items in sorted(self.issues.items())
            },
            "failed_stimuli": int(failed),
            "missing_stimuli": int(missing),
        }

    def summary_line(self) -> str:
        """One-line rendering of this report for a human-readable audit listing.

        Reads ``"<name>: clean (<n> stims)"`` when there are no issues, else
        ``"<name>: [<tag>] <bucket>=<count>, ..."`` where ``tag`` marks
        whether every issue is soft (see :attr:`soft_only`) or at least one
        is hard.
        """
        if self.is_clean:
            return f"  {self.path.name}: clean ({self.n_stims} stims)"
        bullets = ", ".join(f"{k}={len(v)}" for k, v in self.issues.items())
        tag = "soft-only" if self.soft_only else "HARD"
        return f"  {self.path.name}: [{tag}] {bullets}"


def _primary_value_key(values: dict) -> str | None:
    """Pick the 'main' numeric key from a FeatureModelOutput.values dict."""
    preferred = (
        "log_p_observed",
        "mean_token_log_p",
        "token_entropy",
        "token_entropy_sum",
        "token_renyi_entropy_sum",
        "continuation_entropy",
        "continuation_entropy_buggy",
        "mc_word_entropy",
        "unigram_surprisal",
        "surprisal",
        "forward_marginal_entropy",
        "joint_word_entropy",
        "decontextualised_distance",
        "similarity_adjusted_surprisal",
        "similarity_adjusted_surprisal_orth",
        "similarity_adjusted_surprisal_pos",
        # Information value writes one key per (aggregation, distance); the
        # semantic mean is the one the paper leads with.
        "information_value_mean_cosine",
    )
    for k in preferred:
        if k in values:
            return k
    for k in values:
        if k.startswith("token_renyi_entropy_alpha"):
            return k
    # Word-level information value writes one key per layer; any layer serves.
    for k in values:
        if k.startswith("infovalue_layer"):
            return k
    for k in values:
        if k.startswith(
            (
                "restricted_first_token_entropy",
                "restricted_first_token_renyi_entropy_alpha",
                "next_word_entropy",
                "next_word_renyi_entropy_alpha",
            )
        ):
            return k
    # Lens cells write one key per layer; the final layer is the only one
    # whose magnitudes behave like ordinary surprisal, so it carries the
    # per-word checks (early layers are legitimately extreme).
    lens_layers = [
        (int(k.rsplit("_L", 1)[1]), k)
        for k in values
        if k.startswith(("logit_lens_log_p_L", "tuned_lens_log_p_L"))
    ]
    if lens_layers:
        return max(lens_layers)[1]
    return None


def _values_dict(output) -> dict:
    """The per-word values dict of one output.

    Canonical outputs are :class:`FeatureModelOutput` (``.values`` is a dict);
    plain-dict outputs are also handled (there ``.values`` is the dict's bound
    method). An unrecognized shape yields an empty dict rather than raising.
    """
    v = getattr(output, "values", None)
    if isinstance(v, dict):
        return v
    return output if isinstance(output, dict) else {}


def _scalarize(v):
    """Reduce a value to a representative scalar for the per-word checks. Array-valued profiles (e.g. ``forward_marginal_entropy``, a
    per-depth vector) collapse to their first depth = the exact next-word
    entropy; scalars pass through. Returns ``None`` if not numeric.
    """
    if v is None:
        return None
    if hasattr(v, "__len__") and not isinstance(v, (str, bytes)):
        try:
            return float(v[0]) if len(v) else None
        except (TypeError, ValueError, IndexError, KeyError):
            return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _collect_stim_values(
    stim_outputs, primary_key: str | None = None
) -> tuple[list[float], str | None]:
    """Return (per-word values, primary key) for one stim's outputs."""
    if not stim_outputs:
        return ([], None)
    primary = primary_key or _primary_value_key(_values_dict(stim_outputs[0]))
    if primary is None:
        return ([], None)
    vals: list[float] = []
    for u in stim_outputs:
        v = _scalarize(_values_dict(u).get(primary))
        vals.append(math.nan if v is None or math.isnan(v) else v)
    return (vals, primary)


def _check_primary_values(
    report: PredictorPickleReport, sid: str, outputs, primary_key: str | None
) -> None:
    """Reject missing or nonnumeric values for the selected key."""
    if primary_key is None:
        return
    for index, output in enumerate(outputs or ()):
        values = _values_dict(output)
        if primary_key not in values:
            report.add(
                "missing_primary_value", f"{sid}: word[{index}] lacks {primary_key!r}"
            )
        elif _scalarize(values[primary_key]) is None:
            report.add(
                "invalid_primary_value",
                f"{sid}: word[{index}] has nonnumeric {primary_key!r}",
            )


#: Fewest values on which a stimulus's NaN share is judged. One value cannot
#: separate a missing value the estimator defines (a first unit scored without
#: a beginning-of-sequence token, a documented per-word failure) from
#: corruption.
_MIN_JUDGED = 2


def _judged_values(vals: list[float], primary: str) -> list[float]:
    """Values entering the NaN accounting.

    Word information value has no left context at the first unit, so a NaN
    there is excluded before counting.
    """
    if primary.startswith("infovalue_layer") and math.isnan(vals[0]):
        return vals[1:]
    return vals


def _nan_issue(
    sid: str, vals: list[float], primary: str, nan_burst_fraction: float
) -> tuple[str, str] | None:
    """The NaN finding of one stimulus.

    ``all_nan_stim`` or ``nan_burst`` when at least :data:`_MIN_JUDGED` values
    are judged; the advisory ``single_unit_nan`` when the only judged value is
    NaN; otherwise none.
    """
    counted = _judged_values(vals, primary)
    n = len(counted)
    n_nan = sum(1 for v in counted if math.isnan(v))
    if n < _MIN_JUDGED:
        if n == 1 and n_nan == 1:
            return "single_unit_nan", f"{sid}: its only unit is NaN"
        return None
    if n_nan == n:
        return "all_nan_stim", f"{sid}: {n}/{n} NaN"
    if n_nan / n > nan_burst_fraction:
        return "nan_burst", f"{sid}: {n_nan}/{n} NaN ({100 * n_nan / n:.0f}%)"
    return None


def _value_issue(sid: str, vals: list[float], primary: str) -> tuple[str, str] | None:
    """Check numeric domains independently of magnitude or the filename."""
    log_probability = primary in {
        "log_p_observed",
        "mean_token_log_p",
    } or primary.startswith(("logit_lens_log_p_L", "tuned_lens_log_p_L"))
    surprisal = primary in {"unigram_surprisal", "surprisal"} or primary.startswith(
        "similarity_adjusted_surprisal"
    )
    nonnegative = (
        surprisal
        or primary
        in {
            "token_entropy",
            "token_entropy_sum",
            "token_renyi_entropy_sum",
            "continuation_entropy",
            "continuation_entropy_buggy",
            "mc_word_entropy",
            "forward_marginal_entropy",
            "joint_word_entropy",
        }
        or primary.startswith(
            (
                "token_renyi_entropy_alpha",
                "restricted_first_token_entropy",
                "restricted_first_token_renyi_entropy_alpha",
                "next_word_entropy",
                "next_word_renyi_entropy_alpha",
            )
        )
    )
    for i, value in enumerate(vals):
        if math.isnan(value):
            continue
        detail = f"{sid}: word[{i}]={value!r} for {primary}"
        if (log_probability and value > 0) or (nonnegative and value < 0):
            return "invalid_value_domain", detail
        # A zero-probability event has -inf log probability or +inf surprisal.
        if math.isinf(value) and not (log_probability or surprisal):
            return "nonfinite_value", detail
    return None


def _extreme_issue(sid: str, vals: list[float], threshold: float | None) -> str | None:
    """First finite magnitude above the caller's advisory threshold."""
    if threshold is not None:
        for i, value in enumerate(vals):
            if math.isfinite(value) and abs(value) > threshold:
                return f"{sid}: word[{i}]={value:.1f} (|v|>{threshold})"
    return None


def _per_stim_means(stim_vals: dict[str, list[float]]) -> dict[str, float]:
    """Mean of finite values per stimulus for outlier diagnostics."""
    out: dict[str, float] = {}
    for sid, vals in stim_vals.items():
        finite = [v for v in vals if math.isfinite(v)]
        out[sid] = sum(finite) / len(finite) if finite else math.nan
    return out


def _validate_options(
    primary_key, extreme_value_threshold, outlier_sigma, nan_burst_fraction
):
    """Validate caller-supplied diagnostic thresholds and the selected key."""
    if primary_key is not None and (
        not isinstance(primary_key, str) or not primary_key
    ):
        raise ValueError("primary_key must be a nonempty string or None")
    if extreme_value_threshold is not None and (
        not math.isfinite(extreme_value_threshold) or extreme_value_threshold <= 0
    ):
        raise ValueError("extreme_value_threshold must be positive and finite or None")
    if not math.isfinite(outlier_sigma) or outlier_sigma <= 0:
        raise ValueError("outlier_sigma must be positive and finite")
    if not 0 <= nan_burst_fraction <= 1:
        raise ValueError("nan_burst_fraction must lie in [0, 1]")


def predictor_pickle_issues(
    path: str | Path,
    *,
    primary_key: str | None = None,
    extreme_value_threshold: float | None = _EXTREME_VALUE_THRESHOLD,
    outlier_sigma: float = _OUTLIER_SIGMA,
    nan_burst_fraction: float = _NAN_BURST_FRACTION,
) -> PredictorPickleReport:
    """Scan a predictor pickle and return a per-bucket issue report.

    Args:
        path: Path to the pickle (``compute_dir()/*.pkl``).
        primary_key: Numeric value to audit in every output record. Required
            for custom value names; ``None`` selects a built-in primary key.
            A missing or nonnumeric declared value is a hard issue. Numeric
            arrays use their first element, as for built-in depth profiles.
        extreme_value_threshold: Advisory magnitude threshold in stored units;
            ``None`` disables it. The default is 100, not a mathematical bound.
        outlier_sigma: Number of sigma above the corpus per-stim mean to
            flag a ``mean_outlier``.
        nan_burst_fraction: Advisory missingness fraction, between zero and one.
            An entirely missing table remains unusable and is a hard issue.

    Returns:
        :class:`PredictorPickleReport`.
    """
    _validate_options(
        primary_key, extreme_value_threshold, outlier_sigma, nan_burst_fraction
    )
    path = Path(path)
    report = PredictorPickleReport(path=path)
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
    except Exception as e:
        # Corrupt bytes can surface as almost any exception from the unpickler.
        report.add("unreadable", f"{type(e).__name__}: {e}")
        return report

    if not isinstance(data, dict) or not data:
        report.add("empty_pickle", f"data type={type(data).__name__}, n_keys=0")
        return report

    report.n_stims = len(data)

    per_stim_values: dict[str, list[float]] = {}
    for sid, outs in data.items():
        vals, primary = _collect_stim_values(outs, primary_key)
        if primary is not None:
            report.primary_keys.add(primary)
        _check_primary_values(report, sid, outs, primary)
        if not vals or primary is None:
            report.add("zero_value_stim", f"{sid}: no usable values")
            continue
        per_stim_values[sid] = vals

        issue = _nan_issue(sid, vals, primary, nan_burst_fraction)
        if issue is not None:
            report.add(*issue)
            if issue[0] == "all_nan_stim":
                continue

        invalid = _value_issue(sid, vals, primary)
        if invalid is not None:
            report.add(*invalid)
        extreme = _extreme_issue(sid, vals, extreme_value_threshold)
        if extreme is not None:
            report.add("extreme_per_word", extreme)

    # Individual missing observations are valid; a table with no values is unusable.
    means = _per_stim_means(per_stim_values)
    finite_means = [m for m in means.values() if not math.isnan(m)]
    if per_stim_values and not any(
        not math.isnan(value) for vals in per_stim_values.values() for value in vals
    ):
        report.add(
            "all_nan_pickle", f"all values are missing in {len(per_stim_values)} stims"
        )
    if len(finite_means) >= 4:
        mu = sum(finite_means) / len(finite_means)
        var = sum((m - mu) ** 2 for m in finite_means) / max(1, len(finite_means) - 1)
        sigma = math.sqrt(var)
        if sigma > 0:
            cutoff = outlier_sigma * sigma
            for sid, m in means.items():
                if math.isnan(m):
                    continue
                if abs(m - mu) > cutoff:
                    report.add(
                        "mean_outlier",
                        f"{sid}: mean={m:.2f} corpus_mean={mu:.2f} "
                        f"|Δ|={abs(m - mu):.2f} (>{outlier_sigma}sigma={cutoff:.2f})",
                    )

    return report
