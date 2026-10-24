"""Distribution fingerprinting: which target quantity does a predictor compute?

``fingerprint()`` runs a predictor over an exactly-solvable reference model
and ranks every candidate :class:`~replica.feature.diagnostics.specs.TargetSpec`
in four tiers:

- exact, numerically equal to the spec.
- affine, equal up to ``a·x + b``; the slope is matched against known
  constants (``ln 2`` → nats, negative → sign flip, ...).
- approx, slope ≈ 1, high R², RMS deviation from the identity line
  ≤ 0.25·std: agreement within the noise tolerances. A nearby target's
  values can fit affinely while failing this identity check.
- unmatched, correlation evidence only.

Degenerate (constant / all-NaN) predictors are flagged before matching;
low coverage and unit-drift hazards are surfaced in the verdict.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy import stats as _scipy_stats

from replica.feature.diagnostics._statistics import affine_fit
from replica.feature.diagnostics.references.markov import MarkovReferenceLM

from ...core import Stimulus
from ..extractors import Feature
from .specs import TargetSpec, default_specs

__all__ = ["FingerprintReport", "SpecMatch", "fingerprint"]

_EXACT_RTOL = 1e-6
_EXACT_ATOL = 1e-9
_AFFINE_MIN_R2 = 1.0 - 1e-9
_APPROX_MIN_R2 = 0.95
_APPROX_SLOPE = (0.9, 1.1)
_APPROX_IDENTITY_RMS = 0.25  # rms(y - x) as a fraction of std(x)
_MIN_COMPARED = 8  # no tier below this many jointly-finite positions
_LOW_COVERAGE = 0.8  # flag verdicts built on < this fraction of positions

# Affine slopes with a known interpretation (matched at 1% tolerance).
_KNOWN_SLOPES = [
    (
        1.0,
        "slope ≈ 1: affine-but-not-exact, numerically degraded copy "
        "(float32?) or a small additive bias",
    ),
    (np.log(2.0), "slope ≈ ln 2: predictor appears to be in nats, spec is bits"),
    (1.0 / np.log(2.0), "slope ≈ 1/ln 2: inverse base conversion (double log2?)"),
    (np.log10(2.0), "slope ≈ log10 2: predictor appears to be in hartleys/dits"),
    (-1.0, "slope ≈ -1: sign flip (log-probability instead of surprisal?)"),
    (-np.log(2.0), "slope ≈ -ln 2: sign flip and nats"),
    (
        -np.log10(2.0),
        "slope ≈ -log10 2: predictor appears to be a log10 probability "
        "(sign-flipped hartleys/dits, e.g. a raw log-frequency)",
    ),
]


@dataclass
class SpecMatch:
    """Comparison of the predictor against one TargetSpec."""

    spec: TargetSpec
    n_compared: int
    n_total: int
    max_abs_err: float = np.nan
    max_rel_err: float = np.nan
    slope: float = np.nan
    intercept: float = np.nan
    r2: float = np.nan
    pearson_r: float = np.nan
    spearman_rho: float = np.nan
    tier: str | None = None  # "exact" | "affine" | "approx" | None

    @property
    def name(self) -> str:
        """Name of the spec this match was scored against."""
        return self.spec.name


@dataclass
class FingerprintReport:
    """Ranked comparison of one predictor against the whole spec battery.

    ``matches`` is sorted best first, by tier and then by R². ``degenerate``
    short-circuits the ranking, so it comes back with ``matches`` empty.
    """

    predictor_name: str
    n_positions: int
    matches: list[SpecMatch] = field(default_factory=list)
    verdict: str = ""
    degenerate: bool = False

    @property
    def best(self) -> SpecMatch | None:
        """Top-ranked match, or ``None`` when nothing was scored."""
        return self.matches[0] if self.matches else None

    def to_text(self) -> str:
        """Render the verdict and one table row per candidate spec."""
        lines = [
            f"Fingerprint of {self.predictor_name!r} ({self.n_positions} positions)",
            f"Verdict: {self.verdict}",
        ]
        if self.matches:
            header = (
                f"{'spec':<38} {'tier':<7} {'n':>7} {'R²':>10} {'slope':>9} "
                f"{'icept':>8} {'max_rel_err':>12} {'rho':>7}"
            )
            lines += ["", header, "-" * len(header)]
            for m in self.matches:
                n = f"{m.n_compared}/{m.n_total}"
                lines.append(
                    f"{m.name:<38} {m.tier or '-':<7} {n:>7} {m.r2:>10.6f} "
                    f"{m.slope:>9.4f} {m.intercept:>8.3f} "
                    f"{m.max_rel_err:>12.3g} {m.spearman_rho:>7.3f}"
                )
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.to_text()


def _predictor_values(
    predictor,
    reference: MarkovReferenceLM,
    stimuli: list[Stimulus],
    target_layer: str,
) -> tuple[np.ndarray, str]:
    """Resolve the predictor to one concatenated per-unit array."""
    n_units = [len(s.layers[target_layer]) for s in stimuli]
    if isinstance(predictor, np.ndarray):
        values, name = np.asarray(predictor, dtype=np.float64), "array"
        if values.ndim != 1:
            raise ValueError(
                f"predictor array must be 1-D (one value per unit), "
                f"got shape {values.shape}"
            )
    elif isinstance(predictor, (list, tuple)) and all(
        isinstance(a, np.ndarray) for a in predictor
    ):
        if len(predictor) != len(stimuli):
            raise ValueError(f"got {len(predictor)} arrays for {len(stimuli)} stimuli")
        for i, (a, n) in enumerate(zip(predictor, n_units, strict=True)):
            if a.ndim != 1 or len(a) != n:
                raise ValueError(
                    f"array {i} has shape {a.shape} but stimulus "
                    f"{stimuli[i].stimulus_id!r} has {n} units"
                )
        values, name = np.concatenate([np.asarray(a) for a in predictor]), "array"
    elif callable(predictor):
        # Feature protocol: run the reference model, then extract.
        arrays = [
            np.asarray(
                predictor(s, reference(s, target_layer=target_layer), target_layer)
            )
            for s in stimuli
        ]
        values = np.concatenate(arrays)
        name = getattr(predictor, "name", type(predictor).__name__)
    else:
        raise TypeError(
            "predictor must be a Feature, an ndarray, or a list of ndarrays; "
            f"got {type(predictor).__name__}"
        )
    if len(values) != sum(n_units):
        raise ValueError(
            f"predictor produced {len(values)} values but the probe stimuli "
            f"have {sum(n_units)} units"
        )
    return values, name


def _interpret_slope(slope: float) -> str | None:
    for ref, note in _KNOWN_SLOPES:
        if abs(slope - ref) <= 0.01 * abs(ref):
            return note
    return None


def _score_one(spec: TargetSpec, pred: np.ndarray, exact: np.ndarray) -> SpecMatch:
    mask = np.isfinite(pred) & np.isfinite(exact)
    m = SpecMatch(spec=spec, n_compared=int(mask.sum()), n_total=len(pred))
    if m.n_compared < 3:
        return m
    x, y = exact[mask], pred[mask]
    m.max_abs_err = float(np.max(np.abs(y - x)))
    m.max_rel_err = float(np.max(np.abs(y - x) / np.maximum(np.abs(x), 1e-12)))

    if np.std(x) > 0:
        m.slope, m.intercept, m.r2 = affine_fit(x, y)
        if np.std(y) > 0:
            m.pearson_r = float(np.corrcoef(x, y)[0, 1])
            m.spearman_rho = float(_scipy_stats.spearmanr(x, y).statistic)

    if m.n_compared < _MIN_COMPARED:
        return m  # stats reported, but too few positions for any tier
    if np.allclose(y, x, rtol=_EXACT_RTOL, atol=_EXACT_ATOL):
        m.tier = "exact"
    elif np.isfinite(m.r2) and m.r2 >= _AFFINE_MIN_R2:
        m.tier = "affine"
    elif (
        np.isfinite(m.r2)
        and m.r2 >= _APPROX_MIN_R2
        and _APPROX_SLOPE[0] <= m.slope <= _APPROX_SLOPE[1]
        and np.std(x) > 0
        and float(np.sqrt(np.mean((y - x) ** 2))) <= _APPROX_IDENTITY_RMS * np.std(x)
    ):
        m.tier = "approx"
    return m


_TIER_ORDER = {"exact": 0, "affine": 1, "approx": 2, None: 3}


def _scale_caution(report: FingerprintReport) -> str:
    """Tier-first ranking can crown a confusable spec while a unit-drifted
    predictor of another spec fits far better affinely (e.g. entropy in
    nats lands on entropy_top4's scale). Surface that instead of letting
    the tier win silently.
    """
    best = report.best
    finite = [m for m in report.matches if np.isfinite(m.r2)]
    if not finite:
        return ""
    top = max(finite, key=lambda m: m.r2)
    # A NaN best.r2 (constant spec array on the probe) has to suppress the
    # caution; the affirmative >= is False for NaN, which does that.
    if top.name == best.name or not (top.r2 >= max(best.r2 + 0.01, 0.99)):
        return ""
    note = _interpret_slope(top.slope)
    hint = f"; {note}" if note else ""
    return (
        f" CAUTION: {top.name!r} fits better affinely "
        f"(R²={top.r2:.4f}, slope={top.slope:.4f}{hint}), "
        "verify units/scale before trusting the attribution."
    )


def _verdict(report: FingerprintReport) -> str:
    best = report.best
    if best is None or best.n_compared < _MIN_COMPARED:
        return (
            f"insufficient finite positions to fingerprint "
            f"(need ≥ {_MIN_COMPARED} jointly finite with a spec)"
        )
    prefix = ""
    if best.n_compared < _LOW_COVERAGE * best.n_total:
        prefix = (
            f"LOW COVERAGE, predictor is non-finite at "
            f"{best.n_total - best.n_compared} of {best.n_total} positions; "
            "the verdict only describes the rest: "
        )
    ties = [
        m.name
        for m in report.matches[1:]
        if best.tier is not None and m.tier == best.tier
    ]
    tie_note = (
        f"; also {best.tier}-matches {', '.join(repr(n) for n in ties)}"
        ", use mc_convergence() against each to discriminate"
        if ties
        else ""
    )
    if best.tier == "exact":
        return prefix + (
            f"matches {best.name!r} exactly "
            f"(max rel err {best.max_rel_err:.2g}): {best.spec.description}" + tie_note
        )
    if best.tier == "affine":
        note = _interpret_slope(best.slope)
        msg = (
            f"matches {best.name!r} up to an affine transform "
            f"(slope={best.slope:.4f}, intercept={best.intercept:.4f})"
        )
        return prefix + (f"{msg}; {note}" if note else msg) + tie_note
    if best.tier == "approx":
        return prefix + (
            f"approximates {best.name!r} (slope≈1, R²={best.r2:.4f}); "
            "consistent with an unbiased noisy estimator of "
            f"{best.spec.description}" + tie_note
        )
    msg = (
        f"no exact, affine, or unbiased-estimator match; closest is "
        f"{best.name!r} (R²={best.r2:.3f}, rho={best.spearman_rho:.3f})."
    )
    if np.isfinite(best.r2) and best.r2 >= _APPROX_MIN_R2:
        note = _interpret_slope(best.slope)
        hint = f" ({note})" if note else ""
        msg += (
            f" The fit is strong but off the identity line "
            f"(slope={best.slope:.4f}, intercept={best.intercept:.4f}){hint}, "
            "a noisy estimator in different units, or a biased one."
        )
    else:
        msg += " The predictor may target a quantity outside the battery, or be buggy."
    return prefix + msg


def fingerprint(
    predictor: Feature | np.ndarray | list[np.ndarray],
    reference: MarkovReferenceLM | None = None,
    stimuli: list[Stimulus] | None = None,
    specs: list[TargetSpec] | None = None,
    target_layer: str = "words",
) -> FingerprintReport:
    """Identify which target quantity ``predictor`` computes.

    Args:
        predictor: a REPLICA ``Feature`` (run against the reference model's
            outputs), or per-unit values as one concatenated ndarray /
            a list of per-stimulus ndarrays aligned to ``stimuli``.
        reference: exactly-solvable reference model
            (default: ``MarkovReferenceLM()``).
        stimuli: probe stimuli (default: the reference's de Bruijn probe,
            which covers every (context, next-word) pair).
        specs: candidate quantities (default: :func:`default_specs`).
        target_layer: annotation layer defining the unit granularity.
    """
    reference = reference if reference is not None else MarkovReferenceLM()
    stimuli = stimuli if stimuli is not None else [reference.probe_stimulus()]
    specs = specs if specs is not None else default_specs()

    values, name = _predictor_values(predictor, reference, stimuli, target_layer)
    report = FingerprintReport(predictor_name=name, n_positions=len(values))

    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        report.degenerate = True
        report.verdict = "degenerate: predictor produced no finite values"
        return report
    # Spread relative to magnitude: a tiny-but-informative predictor is
    # not constant; a constant-plus-float-noise one is.
    if np.ptp(finite) <= 1e-12 * float(np.abs(finite).max()):
        report.degenerate = True
        report.verdict = (
            f"degenerate: predictor is constant ({finite[0]:.6g} at every "
            "position), it carries no information about any distribution"
        )
        return report

    exact_arrays = {
        spec.name: np.concatenate(
            [spec.compute(reference, s, target_layer) for s in stimuli]
        )
        for spec in specs
    }
    report.matches = sorted(
        (_score_one(spec, values, exact_arrays[spec.name]) for spec in specs),
        key=lambda m: (_TIER_ORDER[m.tier], -(m.r2 if np.isfinite(m.r2) else -np.inf)),
    )
    report.verdict = _verdict(report)
    if report.best is not None and report.best.tier is not None:
        report.verdict += _scale_caution(report)
    return report
