"""Assemble the multiverse report from grid cell TSVs.

Reads cell TSVs, derives swap columns from spec fields, and outputs two tables:
- Claims: one row per (paper, corpus, dv, predictor), with spec-curve
  descriptives (Cohen's f², ΔLL/N), and joint sign-flip test.
- Attribution: one row per paper, showing variance decomposition across swap
  dimensions.

ΔLL/N is only comparable within DV family. The joint test uses only
cells selected by the caller's comparability rules with verified cell identities
and identical ordered observation IDs; family-wise (maxT) statistics use that
same aligned set. Joint inference requires a caller-supplied delta loader and an
explicit declaration that individual observations are independent.
"""

import json
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from .stats import (
    JointInferenceResult,
    family_wise_signflip,
    joint_signflip_inference,
    spec_curve_descriptives,
    swap_attribution,
)


class CellLabels(Protocol):
    """Labels for positions in a caller-defined sequence of analysis decisions."""

    def label(self, position: int, mode: str = "cumulative") -> str:
        """Return a cell label.

        Args:
            position: Position in the sequence.
            mode: Caller-defined composition mode.
        """
        ...


class ReportRegistry(Protocol):
    """Resolve labels and attribute-bearing specs for report rows."""

    def recipe(self, name: str) -> CellLabels:
        """Return the labels for a named recipe.

        Args:
            name: Caller-defined recipe name.
        """
        ...

    def recipe_spec(
        self, recipe: str, paper: str, position: int, *, mode: str
    ) -> object:
        """Resolve the attributes of one recipe cell.

        Args:
            recipe: Recipe name.
            paper: Configuration key from the row's paper column.
            position: Cell position in that recipe.
            mode: Caller-defined composition mode.
        """
        ...

    def cells_for_mode(
        self, mode: str
    ) -> Callable[[str], Iterable[tuple[str, object]]]:
        """Return a resolver from configuration key to named atomic cells.

        Args:
            mode: Caller-defined atomic composition mode.
        """
        ...


# One claim groups fitted specifications; a cell also has a position and label.
CLAIM_KEY = ("paper", "corpus", "dv", "predictor")
CELL_KEY = (*CLAIM_KEY, "position", "cell")


@dataclass(frozen=True)
class CellDeltas:
    """A cell's held-out likelihood differences and their source identities.

    Args:
        cell_key: Stored identity in ``CELL_KEY`` order; include the cell label
            because atomic cells can share position -1.
        observation_ids: Unique identifiers in the exact order of ``values``.
            Use complete observation keys, such as (participant, item, token).
        values: One-dimensional held-out likelihood differences.
    """

    cell_key: tuple[Hashable, ...]
    observation_ids: Sequence[Hashable]
    values: ArrayLike


@dataclass(frozen=True)
class ReportSpec:
    """The dimensions and comparability rules of one multiverse report.

    Args:
        swap_fields: Cell-spec attributes to expose as report columns.
        attribution: Output dimension -> source columns combined into one
            categorical factor. Empty uses each swap field separately.
        joint_criteria: Required column values for joint likelihood inference.
            Callers must select cells on the same response scale.
        load_deltas: Optional loader of a row's CellDeltas; returns None when
            unavailable. Without a loader, the report is descriptive only.
            Identities must come from the source metadata, not be inferred from
            array length or copied from the requested row. The report verifies
            the cell key and exact observation order before joint inference.
        likelihood_scale_fields: Columns that must match the claim's anchor
            before a relative likelihood change is meaningful.
        exclude_atomic_cell: Optional explicit filter for duplicate or unwanted
            atomic cell labels; true excludes the label.
        assume_independent_observations: Explicitly enable observation-wise sign
            flips for independent observations. Matching IDs do not establish
            independence. For grouped observations, use
            :func:`replica.multiverse.curve.stats.grouped_family_wise_signflip`.
    """

    swap_fields: tuple[str, ...]
    joint_criteria: Mapping[str, object]
    load_deltas: Callable[[Mapping[str, object]], CellDeltas | None] | None = None
    attribution: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    likelihood_scale_fields: tuple[str, ...] = ()
    exclude_atomic_cell: Callable[[str], bool] | None = None
    assume_independent_observations: bool = False

    def __post_init__(self) -> None:
        if self.load_deltas is not None and not callable(self.load_deltas):
            raise TypeError(
                "load_deltas must be None or a callable returning CellDeltas or None"
            )
        if not isinstance(self.assume_independent_observations, bool):
            raise TypeError("assume_independent_observations must be boolean")
        if len(set(self.swap_fields)) != len(self.swap_fields):
            raise ValueError("swap_fields must be unique")
        if any(not columns for columns in self.attribution.values()):
            raise ValueError(
                "each attribution dimension needs at least one source column"
            )


def _with_metrics(g: pd.DataFrame) -> pd.DataFrame:
    """Numeric effect columns incl. derived ΔLL/N."""
    g = g.copy()
    g["dll_per_obs"] = pd.to_numeric(g["delta_ll"], errors="coerce") / (
        pd.to_numeric(g["n_obs"], errors="coerce")
    )
    for col in ("delta_r2", "cohens_f2", "pvalue"):
        if col in g.columns:
            g[col] = pd.to_numeric(g[col], errors="coerce")
    return g


def load_grid_cells(cells_dir: Path | str) -> pd.DataFrame:
    """Concatenate every per-task cell TSV under *cells_dir*.

    Duplicate rows for one cell identity resolve to the last one.

    Args:
        cells_dir: Directory holding one ``.tsv`` file per grid task, each
            carrying the ``position``, ``cell``, and claim-key columns.
    """
    files = sorted(Path(cells_dir).glob("*.tsv"))
    if not files:
        raise FileNotFoundError(f"no cell TSVs under {cells_dir}")
    df = pd.concat([pd.read_csv(f, sep="\t") for f in files], ignore_index=True)
    required = {"position", "cell", *CLAIM_KEY}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            f"cell TSVs under {cells_dir} are missing required grid columns: {missing}"
        )
    # Cell identity is the decision-set label: atomic cells all carry
    # position=-1, so keying on the numeric position alone would collapse the
    # whole grid to one decision cell per (paper, corpus, dv, predictor).
    key = ["paper", "position", "cell", *CLAIM_KEY[1:]]
    return df.drop_duplicates(subset=key, keep="last")


def add_swap_columns(
    df: pd.DataFrame,
    mode: str,
    *,
    registry: ReportRegistry,
    report: ReportSpec,
    recipe: str | None = None,
    config_tag: Callable[[str], str] = str,
) -> pd.DataFrame:
    """Annotate each row with its spec's swap-dimension values.

    Recipe modes derive the values per (paper, position) from
    ``registry.recipe_spec`` over the named *recipe*; a stored ``cell`` label
    that differs from the recipe's label marks cells from another grid
    revision.

    Args:
        df: Cell table with ``paper``, ``position``, and ``cell`` columns.
        mode: Grid run mode; a value starting with ``"atomic-"`` routes to
            the atomic-cell variant, any other value requires *recipe*.
        registry: Registry that resolves a (paper, position) or (paper, cell)
            pair to its configured field values.
        report: Swap fields and explicit atomic-cell exclusions.
        recipe: Recipe name to resolve positions through; required unless
            *mode* is an atomic mode.
        config_tag: Maps a row's ``paper`` value to the registry key it is
            looked up under; identity by default, and a grid whose contrast
            rows carry suffixed tags passes a mapping.

    Returns:
        *df* with one added column per configured swap field.

    Raises:
        ValueError: When a row's stored ``cell`` label disagrees with the
            recipe's label for its position.
    """
    if mode.startswith("atomic-"):
        return _add_swap_columns_atomic(df, mode, registry, config_tag, report)
    if recipe is None:
        raise ValueError(f"mode {mode!r} needs a recipe name")
    rec = registry.recipe(recipe)
    stored = df["cell"].dropna()
    expected = stored.index.map(lambda i: rec.label(int(df.loc[i, "position"]), mode))
    bad = stored[stored.to_numpy() != expected.to_numpy()]
    if len(bad):
        raise ValueError(
            "cell label disagrees with the current recipe's labels "
            f"(different grid revision?): e.g. position "
            f"{df.loc[bad.index[0], 'position']} stored "
            f"{bad.iloc[0]!r}"
        )
    cache: dict[tuple[str, int], dict[str, str]] = {}

    def vals(tag: str, position: int) -> dict[str, str]:
        key = (config_tag(tag), int(position))
        if key not in cache:
            spec = registry.recipe_spec(recipe, key[0], key[1], mode=mode)
            cache[key] = {f: str(getattr(spec, f)) for f in report.swap_fields}
        return cache[key]

    out = df.copy()
    for f in report.swap_fields:
        out[f] = [
            vals(t, s)[f] for t, s in zip(out["paper"], out["position"], strict=True)
        ]
    return out


def _add_swap_columns_atomic(
    df: pd.DataFrame,
    mode: str,
    registry: ReportRegistry,
    config_tag: Callable[[str], str],
    report: ReportSpec,
) -> pd.DataFrame:
    """Atomic-grid variant: swap values come from the (paper, cell) spec in
    ``registry.cells_for_mode(mode)`` (rows carry position=-1 and the
    decision-cell label in ``cell``).
    """
    fn = registry.cells_for_mode(mode)
    if "cell" not in df.columns:
        raise ValueError("atomic mode needs a 'cell' column in the cell TSVs")
    if report.exclude_atomic_cell is not None:
        excluded = df["cell"].astype(str).map(report.exclude_atomic_cell)
        df = df[~excluded]
    specs: dict[tuple[str, str], dict[str, str]] = {}

    def vals(tag: str, cell: str) -> dict[str, str]:
        key = (config_tag(tag), str(cell))
        if key not in specs:
            spec = dict(fn(key[0])).get(key[1])
            if spec is None:
                raise ValueError(f"unknown atomic cell {key[1]!r} for {key[0]}")
            specs[key] = {f: str(getattr(spec, f)) for f in report.swap_fields}
        return specs[key]

    out = df.copy()
    for f in report.swap_fields:
        out[f] = [vals(t, c)[f] for t, c in zip(out["paper"], out["cell"], strict=True)]
    return out


def _valid_identity(value: object) -> bool:
    """Reject missing or non-reflexive IDs, including compound-key members."""
    if value is None or (isinstance(value, str) and not value):
        return False
    if isinstance(value, (tuple, frozenset)):
        return bool(value) and all(_valid_identity(member) for member in value)
    try:
        hash(value)
        # Missing sentinels such as NaN do not compare equal to themselves.
        return bool(value == value)  # noqa: PLR0124
    except (TypeError, ValueError):
        return False


def _claim_aligned_vectors(
    g: pd.DataFrame,
    report: ReportSpec,
) -> tuple[list[np.ndarray], list[int], list[str]]:
    """Load eligible cells and verify their identities and observation order."""
    if report.load_deltas is None:
        return [], [], []
    vectors: list[np.ndarray] = []
    positions: list[int] = []
    labels: list[str] = []
    expected_ids: tuple[Hashable, ...] | None = None
    seen: set[tuple[Hashable, ...]] = set()
    for _, row in g.iterrows():
        if any(
            str(row[column]) != str(value)
            for column, value in report.joint_criteria.items()
        ):
            continue
        key = tuple(row[column] for column in CELL_KEY)
        if not _valid_identity(key):
            raise ValueError("cell identity fields must be non-missing and hashable")
        if key in seen:
            raise ValueError(f"duplicate cell identity in joint inference: {key!r}")
        seen.add(key)
        deltas = report.load_deltas(row.to_dict())
        if deltas is None:
            continue
        loaded_key = tuple(deltas.cell_key)
        if not _valid_identity(loaded_key):
            raise ValueError("loaded cell identity must be non-missing and hashable")
        if loaded_key != key:
            raise ValueError(f"loaded delta cell identity does not match {key!r}")
        raw = np.asarray(deltas.values)
        if np.iscomplexobj(raw):
            raise ValueError(f"cell {key!r}: delta values must be real")
        values = np.asarray(raw, dtype=float)
        ids = tuple(deltas.observation_ids)
        if values.ndim != 1 or len(ids) != values.size:
            raise ValueError(
                f"cell {key!r}: observation IDs must match a 1D delta array"
            )
        if not all(_valid_identity(identity) for identity in ids):
            raise ValueError(
                f"cell {key!r}: observation IDs must be non-missing and hashable"
            )
        unique_ids = set(ids)
        if len(unique_ids) != len(ids):
            raise ValueError(f"cell {key!r}: observation IDs must be unique")
        if not ids:
            continue
        if expected_ids is not None and ids != expected_ids:
            raise ValueError(
                f"cell {key!r}: observation IDs or order differ across cells"
            )
        expected_ids = ids
        vectors.append(values)
        positions.append(int(row["position"]))
        labels.append(str(row["cell"]))
    return vectors, positions, labels


def _claim_joint(
    vectors: list[np.ndarray],
    *,
    n_resamples: int,
    rng: np.random.Generator | None,
) -> JointInferenceResult | None:
    """Synchronized joint sign-flip over a claim's row-aligned cells.

    Statistic: ``median`` of per-cell mean ΔLL. A magnitude statistic is
    required: ``share_positive`` keeps only each flipped mean's sign, which is
    symmetric about 0, so its null share is bimodal and the test has no power.
    Pair with :func:`family_wise_signflip` for the minP / share-significant
    statistics on the same aligned cells.
    """
    if not vectors:
        return None
    return joint_signflip_inference(
        vectors,
        statistic="median",
        alternative="greater",
        shared_observations=True,
        n_resamples=n_resamples,
        rng=rng,
    )


def summarize_claims(
    df: pd.DataFrame,
    *,
    report: ReportSpec,
    n_resamples: int = 10_000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """One multiverse-summary row per claim over a swap-annotated cell table.

    *df* must carry the configured swap columns (see
    :func:`add_swap_columns`).

    Args:
        df: Swap-annotated cell table; rows outside ``status == "ok"`` are
            dropped before grouping.
        report: Eligibility rules, optional identity-bearing delta loader, and
            explicit independence declaration for observation-wise inference.
        n_resamples: Sign-flip resamples for the per-claim joint and
            family-wise tests.
        alpha: Significance threshold for ``frac_significant`` and the
            family-wise ``n_specs_fwer_sig``/``joint_share_significant``.
        rng: Random generator for the sign-flip tests; a fixed seed when
            omitted, for reproducible p-values.

    Returns:
        One row per (paper, corpus, dv, predictor) claim, with spec-curve
        descriptives and joint sign-flip statistics. ``joint_cells`` is a JSON
        list of cell labels in the same order as ``joint_positions``. Without a
        loader, joint statistics are unavailable (NaN and zero cell counts).

    Raises:
        ValueError: A loaded cell has the wrong identity, duplicate observations,
            invalid dimensions, or observations that differ in identity or order
            from another eligible cell. Incomparable cells must be excluded
            explicitly through ``joint_criteria``. Also raised when joint data
            is supplied without declaring independent observations.
    """
    ok = _with_metrics(df[df["status"] == "ok"])
    rows = []
    for key, g in ok.groupby(list(CLAIM_KEY), dropna=False):
        g = g.sort_values("position")
        pv = g["pvalue"]
        d_dll = spec_curve_descriptives(g["dll_per_obs"], pv, alpha=alpha)
        d_f2 = spec_curve_descriptives(g["cohens_f2"], pv, alpha=alpha)
        # ΔR² curve: the native effect of R²-based runners, which report
        # no ΔLL; unitless, so comparable across the DV swap.
        d_dr2 = spec_curve_descriptives(g["delta_r2"], pv, alpha=alpha)
        vectors, joint_positions, joint_cells = _claim_aligned_vectors(g, report)
        if vectors and not report.assume_independent_observations:
            raise ValueError(
                "joint sign flips require assume_independent_observations=True; "
                "aligned IDs alone do not establish independence. For grouped "
                "data, use grouped_family_wise_signflip with retained analyses."
            )
        joint = _claim_joint(vectors, n_resamples=n_resamples, rng=rng)
        fwe = (
            family_wise_signflip(
                vectors,
                alternative="greater",
                n_resamples=n_resamples,
                alpha=alpha,
                rng=rng,
            )
            if vectors
            else None
        )
        rows.append(
            {
                **dict(zip(CLAIM_KEY, key, strict=True)),
                "n_specs": int(len(g)),
                "positions": ",".join(str(s) for s in g["position"]),
                # ΔLL/N curve (sign fields robust; magnitude only within DV family)
                "dll_per_obs_median": d_dll.median_effect,
                "dll_per_obs_min": d_dll.effect_min,
                "dll_per_obs_max": d_dll.effect_max,
                "frac_in_direction": d_dll.frac_in_direction,
                "frac_significant": d_dll.frac_significant,
                # Cohen's f² curve (scale-free; n may be < n_specs)
                "n_specs_f2": d_f2.n_specs,
                "cohens_f2_median": d_f2.median_effect,
                "cohens_f2_min": d_f2.effect_min,
                "cohens_f2_max": d_f2.effect_max,
                # ΔR² curve (unitless; fills R²-native runners' rows)
                "n_specs_dr2": d_dr2.n_specs,
                "delta_r2_median": d_dr2.median_effect,
                "delta_r2_min": d_dr2.effect_min,
                "delta_r2_max": d_dr2.effect_max,
                "frac_in_direction_dr2": d_dr2.frac_in_direction,
                # Synchronized joint test over the row-aligned specs
                # (median of per-cell mean ΔLL, one-sided greater)
                "joint_median_dll": joint.observed if joint else float("nan"),
                "joint_p": joint.p_value if joint else float("nan"),
                "n_specs_joint": joint.n_specs if joint else 0,
                "joint_positions": ",".join(str(s) for s in joint_positions),
                "joint_cells": json.dumps(joint_cells),
                # Family-wise (maxT) control + share-significant joint test over
                # the same aligned specs.
                "fwer_min_p": fwe.min_adj_p if fwe else float("nan"),
                "n_specs_fwer_sig": fwe.n_sig_fwer if fwe else 0,
                "joint_share_significant": (
                    fwe.share_significant if fwe else float("nan")
                ),
                "joint_share_significant_p": (
                    fwe.share_significant_p if fwe else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows)


def paired_swaps(
    ok: pd.DataFrame, *, report: ReportSpec
) -> tuple[pd.DataFrame, dict[str, str]]:
    """One row per (paper, swap, claim), paired against the anchor cell.

    Recipe-mode grids only, since the anchor is the row with ``position == 0``.
    The effect metric is the held-out ΔR² of the focal predictor, or ΔLL/N for
    a paper whose runner reports no ΔR², where ``rel`` is NaN for the log2-DV
    swap because its likelihoods are on another DV scale.

    Args:
        ok: Swap-annotated cell table restricted to rows with
            ``status == "ok"``; must carry ``position``, ``cell``,
            ``supported``, and the configured likelihood-scale columns.
        report: Columns whose anchor values define a comparable likelihood scale.

    Returns:
        The long table (paper, cell, rel, flip), with NaN ``rel`` and ``flip``
        for a claim that has no anchor row, and the effect metric used per
        paper.
    """
    key = ["corpus", "dv", "predictor"]
    frames = []
    metrics: dict[str, str] = {}
    for paper, g in ok.groupby("paper"):
        g = _with_metrics(g)
        metric = "delta_r2" if g["delta_r2"].notna().all() else "dll_per_obs"
        metrics[str(paper)] = metric
        a = g[g["position"] == 0].set_index(key)
        anchor_eff = a[metric]
        anchor_sup = (a["supported"].astype(str) == "True").astype(float)
        f = g[g["position"] > 0]
        idx = pd.MultiIndex.from_frame(f[key])
        ae = anchor_eff.reindex(idx).to_numpy()
        fe = f[metric].to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.where(np.abs(ae) > 0, (fe - ae) / np.abs(ae), np.nan)
        if metric == "dll_per_obs":
            comparable = np.ones(len(f), dtype=bool)
            for column in report.likelihood_scale_fields:
                comparable &= f[column].to_numpy() == a[column].reindex(idx).to_numpy()
            rel = np.where(comparable, rel, np.nan)
        asup = anchor_sup.reindex(idx).to_numpy()
        fsup = (f["supported"].astype(str) == "True").to_numpy().astype(float)
        flip = np.where(np.isnan(asup), np.nan, (fsup != asup).astype(float))
        frames.append(
            pd.DataFrame(
                {
                    "paper": str(paper),
                    "cell": f["cell"].to_numpy(),
                    "rel": rel,
                    "flip": flip,
                }
            )
        )
    return pd.concat(frames, ignore_index=True), metrics


def swap_summary_tables(
    ok: pd.DataFrame,
    *,
    report: ReportSpec,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, str]]:
    """(swap x paper) tables of median % effect change and conclusion-flip share.

    ``n`` counts the paired claims behind each flip share; rows whose anchor
    is missing don't enter the denominator.

    Args:
        ok: Swap-annotated cell table restricted to rows with
            ``status == "ok"``, as consumed by :func:`paired_swaps`.
        report: Comparability rules forwarded to :func:`paired_swaps`.

    Returns:
        ``(pct, flips, n, metrics)``: the median relative effect change and
        flip-share tables indexed by swap cell x paper, the paired-claim
        count behind each flip share, and the effect metric
        (from :func:`paired_swaps`) used per paper.
    """
    long, metrics = paired_swaps(ok, report=report)
    grp = long.groupby(["cell", "paper"])
    pct = grp["rel"].median().unstack() * 100
    flips = grp["flip"].mean().unstack()
    n = grp["flip"].count().unstack()
    return pct, flips, n, metrics


def attribution_by_paper(
    df: pd.DataFrame,
    effect_col: str = "cohens_f2",
    extra_dims: tuple[str, ...] = ("corpus", "dv", "predictor"),
    *,
    report: ReportSpec,
) -> pd.DataFrame:
    """Per-paper swap attribution of *effect_col* across all ok cells.

    *extra_dims* enter the ANOVA alongside the swap dimensions so corpus/
    predictor variance is attributed to them instead of inflating the
    residual.

    Args:
        df: Cell table with a ``status`` column and the configured dimensions.
        report: Attribution dimensions and their source columns.
        effect_col: Numeric column to attribute variance in, for example
            Cohen's f² or ΔLL/N; rows where it is not finite are dropped.
        extra_dims: Non-swap columns entered into the same ANOVA, so their
            variance is not folded into the residual.

    Returns:
        One row per paper, with the fields of
        :class:`~replica.multiverse.curve.stats.SwapAttribution` flattened
        into ``eta2_<dim>`` / ``p_<dim>`` columns.
    """
    ok = df[df["status"] == "ok"].copy()
    if effect_col == "dll_per_obs" and effect_col not in ok.columns:
        ok = _with_metrics(ok)
    groups = report.attribution or {name: (name,) for name in report.swap_fields}
    derived = {}
    for name, columns in groups.items():
        if len(columns) == 1:
            derived[name] = ok[columns[0]]
        else:
            # Tuples distinguish categorical combinations even when labels contain '|'.
            derived[name] = list(
                zip(*(ok[column].astype(str) for column in columns), strict=True)
            )
    ok = ok.assign(**derived)
    dims = [*groups, *extra_dims]
    rows = []
    for tag, g in ok.groupby("paper"):
        att = swap_attribution(g, effect_col, dims)
        row: dict = {
            "paper": tag,
            "effect": effect_col,
            "n_cells": att.n_obs,
            "r_squared": att.r_squared,
            "residual_eta_squared": att.residual_eta_squared,
        }
        row.update({f"eta2_{d}": att.eta_squared[d] for d in dims})
        row.update({f"p_{d}": att.pvalues[d] for d in dims})
        rows.append(row)
    return pd.DataFrame(rows)
