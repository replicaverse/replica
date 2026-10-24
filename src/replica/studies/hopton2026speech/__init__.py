"""Hopton et al. (2026a): Estimating Surprisal from the Speech Signal."""

from __future__ import annotations

import functools
from pathlib import Path

import numpy as np
import pandas as pd

from replica.analysis.contracts.results import AnalysisResult
from replica.analysis.fitting.scoring import _gaussian_log_likelihood
from replica.analysis.specification.config import AnalysisConfig
from replica.analysis.specification.evaluation import EvaluationSpec
from replica.feature import ModelFeature

from ...analysis.models import RGamModel
from ...dataset.base import TabularDataset, TabularDatasetConfig
from ...feature.extractors import PrecomputedFeature
from ..base import Study, build_grouped_model
from ..records.resources import resource_file

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------

#: The four speech LMs, named as in the feather columns.
SPEECH_LMS = ("gslm", "spirit", "llamamimi", "llamamimi8b")

#: Word-level also carries a GPT-2 text baseline (appendix); it has no
#: marginalized variant and exists only at the word granularity.
WORD_MODELS = ("gpt2",) + SPEECH_LMS

GRANULARITIES = ("word", "phone", "frame")

#: Pupillary lag in ms. 1000 ms is the main text (Winn 2016); 900/1300 are the
#: physiologically plausible bracket from van Rij et al. (2019).
OFFSETS = (900, 1000, 1300)
MAIN_OFFSET = 1000

#: ``surp`` (current unit) and ``surp_prev`` (K=1 spillover) are the two
#: predictor columns every model_data exposes, in surprisal (positive) units.
SURP = "surp"
PREV_SURP = "prev_surp"

#: The paper's GAMM. Both surprisal smooths enter the full model together; the
#: baseline is the two random-effect smooths alone. ``s(x, k=4)`` with no ``bs``
#: is mgcv's default thin-plate spline, matching ``analysis_projection_offsets.R``.
#: Backend columns ``participant_id`` and ``item_id`` map to Listener and item.
_FULL_FORMULA = (
    "y ~ s(surp, k=4) + s(prev_surp, k=4) "
    "+ s(participant_id, bs='re') + s(item_id, bs='re')"
)
_RE_BASELINE = "y ~ s(participant_id, bs='re') + s(item_id, bs='re')"

#: Appendix time-smooth variant: a normalized trial-time nuisance smooth
#: ``s(norm_time, k=4)`` is added to both the full model and the baseline,
#: symmetrically, matching ``analysis_projection_offsets.R`` under
#: ``INCLUDE_TIME_SMOOTH`` (``norm_time = pupil_time_ms / duration_ms``, raw).
NORM_TIME = "norm_time"
_FULL_FORMULA_TS = (
    "y ~ s(surp, k=4) + s(prev_surp, k=4) + s(norm_time, k=4) "
    "+ s(participant_id, bs='re') + s(item_id, bs='re')"
)
_RE_BASELINE_TS = (
    "y ~ s(norm_time, k=4) + s(participant_id, bs='re') + s(item_id, bs='re')"
)


def _identity(y: np.ndarray) -> np.ndarray:
    """Keep pupil dilation on its native scale for the published recipe."""
    return y


@functools.lru_cache(maxsize=8)
def _input_table(name: str) -> pd.DataFrame:
    """Read a checksum-verified study input, cached across calls."""
    return pd.read_parquet(resource_file("hopton2026speech", f"data/{name}.parquet"))


# ----------------------------------------------------------------------------
# Feather selection per (model, granularity)
# ----------------------------------------------------------------------------


def _validate_condition(model: str, granularity: str, variant: str) -> None:
    """Reject conditions absent from the study's experimental design."""
    if model not in WORD_MODELS:
        raise ValueError(f"Unknown speech-study model: {model!r}")
    if granularity not in GRANULARITIES:
        raise ValueError(f"Unknown speech-study granularity: {granularity!r}")
    if variant not in {"marg", "unmarg"}:
        raise ValueError(f"Unknown speech-study variant: {variant!r}")
    if model == "gpt2" and (granularity != "word" or variant != "unmarg"):
        raise ValueError("GPT-2 supports word-level unmarginalized surprisal only")
    if variant == "marg" and granularity != "word":
        raise ValueError("Marginalized surprisal is available at word level only")


def _feather_spec(model: str, granularity: str) -> tuple[str, str, list[str]]:
    """The input table, its unit-index column, and the value columns.

    Value columns are speech-LM log-probs (negated to surprisal downstream).
    Word carries both the unmarginalized and the importance-sampled (marg.)
    columns for a speech LM, or the single GPT-2 column; phone and frame carry
    only the unmarginalized column. Frame reads that model's own feather, whose
    per-item ``max(end_time)`` also sets its audio-duration offset.
    """
    if granularity == "word":
        if model == "gpt2":
            return "combined_word", "word_idx", ["gpt2_log_prob"]
        return (
            "combined_word",
            "word_idx",
            [
                f"{model}_unmarg_log_prob",
                f"{model}_is_log_prob",
            ],
        )
    if granularity == "phone":
        return "combined_phone", "phone_idx", [f"{model}_unmarg_log_prob"]
    return f"{model}_frame", "frame_idx", [f"{model}_unmarg_log_prob"]


_TOKEN_COL = {"word": "word", "phone": "phone", "frame": "frame_idx"}


def _paper_delta_ll(
    *,
    y,
    X_features,
    X_controls,
    participants,
    items,
    participant_ids,  # noqa: ARG001 - part of the analysis_fn kwargs contract, unused here
    item_ids,
    dataset_name,
    model_name,
    dv_name,
    item_fold,
    feature_name,
    time_smooth=False,
):
    """Score item-fold CV with the paper's pooled held-out residual scale.

    Seen listeners retain their fitted random effects; unseen test items use
    zero random effects. Residual standard deviations use ``ddof=1``.
    """
    y = np.asarray(y, dtype=float)
    surp = np.asarray(X_controls[SURP], dtype=float)
    prev = np.asarray(X_features[PREV_SURP], dtype=float)

    # The R z-scores surp / prev once over the whole model_data.
    def _z(v: np.ndarray) -> np.ndarray:
        sd = v.std(ddof=1)
        return (v - v.mean()) / sd if sd > 0 else np.zeros_like(v)

    surp_z, prev_z = _z(surp), _z(prev)
    part = np.asarray(participants, dtype=float)
    item = np.asarray(items, dtype=float)
    # Appendix time-smooth: norm_time enters raw (R does not z-score it) as an
    # extra s(norm_time, k=4) in both the full model and the baseline, placed at
    # x0 so one col_names maps across the full and baseline fits alike.
    norm = np.asarray(X_controls[NORM_TIME], dtype=float) if time_smooth else None

    def _fold_of(raw: object) -> int:
        key = str(raw)
        if key not in item_fold and key.startswith("trial_"):
            key = key[len("trial_") :]
        return item_fold[key]

    fold = np.array([_fold_of(s) for s in item_ids], dtype=int)

    model = RGamModel(
        formula=_FULL_FORMULA_TS if time_smooth else _FULL_FORMULA,
        baseline_formula=_RE_BASELINE_TS if time_smooth else _RE_BASELINE,
        col_names=(
            {"x0": NORM_TIME, "x1": SURP, "x2": PREV_SURP}
            if time_smooth
            else {"x0": SURP, "x1": PREV_SURP}
        ),
        re_predict="shrink_unseen",
        sigma_mode="sig2",
        method="fREML",
        use_bam=True,
        discrete=True,
        n_threads=4,
    )

    n = len(y)
    yhat_full = np.full(n, np.nan)
    yhat_base = np.full(n, np.nan)
    for k in np.unique(fold):
        tr = fold != k
        te = fold == k
        if time_smooth:
            x_full_tr = np.column_stack(
                [norm[tr], surp_z[tr], prev_z[tr], part[tr], item[tr]]
            )
            x_full_te = np.column_stack(
                [norm[te], surp_z[te], prev_z[te], part[te], item[te]]
            )
            x_base_tr = np.column_stack([norm[tr], part[tr], item[tr]])
            x_base_te = np.column_stack([norm[te], part[te], item[te]])
        else:
            x_full_tr = np.column_stack([surp_z[tr], prev_z[tr], part[tr], item[tr]])
            x_full_te = np.column_stack([surp_z[te], prev_z[te], part[te], item[te]])
            x_base_tr = np.column_stack([part[tr], item[tr]])
            x_base_te = np.column_stack([part[te], item[te]])
        yhat_full[te] = model.fit_predict(y[tr], x_full_tr, x_full_te)
        yhat_base[te] = model.fit_predict(y[tr], x_base_tr, x_base_te)

    # One pooled held-out-residual sigma per model, R's sd() (n-1).
    sigma_full = float(np.std(y - yhat_full, ddof=1))
    sigma_base = float(np.std(y - yhat_base, ddof=1))
    full_ll = _gaussian_log_likelihood(y, yhat_full, sigma=sigma_full)
    base_ll = _gaussian_log_likelihood(y, yhat_base, sigma=sigma_base)
    delta_ll = full_ll - base_ll
    return [
        AnalysisResult(
            dataset_name=dataset_name,
            model_name=model_name,
            feature_name=feature_name,
            dv_name=dv_name,
            metrics={
                "delta_ll": delta_ll,
                "delta_ll_per_unit": delta_ll / n,
                "baseline_ll": base_ll,
                "full_ll": full_ll,
            },
            n_observations=n,
        )
    ]


class Hopton2026Speech(Study):
    """Hopton et al. (2026a), speech-LM surprisal and pupil dilation."""

    name = "hopton2026speech"
    bibtex = r"""
    @misc{hopton2026speech,
      title = {Estimating Surprisal from the Speech Signal},
      author = {Hopton, Zachary and Kiegeland, Samuel and Re, Francesco Ignazio and Haller, Patrick and Winn, Matthew B. and Cotterell, Ryan and Chodroff, Eleanor},
      year = {2026},
      archivePrefix = {arXiv},
      eprint = {2610.18439},
      url = {https://arxiv.org/abs/2610.18439}
    }
    """
    dependencies: list = []

    # ------------------------------------------------------------------
    # Data: the pupil <-> surprisal rolling join (validated to Table 6 N-obs)
    # ------------------------------------------------------------------

    def _joined(self, model: str, granularity: str, offset: int) -> pd.DataFrame:
        """Join every pupil sample to the unit active at ``t - offset``.

        Ports ``build_{word,phone,frame}_model_data``: a rolling
        (last-observation-carried-backward) join of the pupil samples onto unit
        onsets, kept only where the sample falls inside the unit window, with a
        K=1 spillover lag and the log-probs negated to surprisal. The
        pupil clock is ``Time2StimOffset + duration_ms(item) - offset``, where
        ``duration_ms`` is ``1000 * max(end_time)`` over this granularity's own
        feather.
        """
        feat, idx, vals = _feather_spec(model, granularity)
        d = _input_table(feat).copy()
        d["item"] = d["item"].astype(str)

        # Per-item audio duration from this feather's latest unit offset.
        dur = (d.groupby("item")["end_time"].max() * 1000.0).rename("duration_ms")

        # Unit table: ms boundaries, drop untimed units, K=1 lag by unit index.
        d["start_ms"] = d["start_time"] * 1000.0
        d["end_ms"] = d["end_time"] * 1000.0
        d = d[d["start_ms"].notna()].sort_values(["item", idx])
        for v in vals:
            d[f"{v}_prev"] = d.groupby("item")[v].shift(1)
        d["onset_ms"] = d["start_ms"]
        token = _TOKEN_COL[granularity]
        keep = ["item", idx, token, "start_ms", "end_ms", "onset_ms"]
        keep += vals + [f"{v}_prev" for v in vals]
        # Frame's token column is the unit index (frame_idx); dedupe so the
        # selection is not a duplicate-named (2-D) frame.
        keep = list(dict.fromkeys(keep))
        units = d[keep].sort_values("onset_ms")

        # Pupil clock, then the rolling backward join per item.
        pup = _input_table("pupil").rename(columns={"stimulus": "item"}).copy()
        pup["item"] = pup["item"].astype(str)
        pup = pup.merge(dur, left_on="item", right_index=True, how="left")
        pup["pupil_time_ms"] = pup["Time2StimOffset"] + pup["duration_ms"] - offset
        pup = pup[pup["pupil_time_ms"].notna()].sort_values("pupil_time_ms")

        j = pd.merge_asof(
            pup,
            units,
            by="item",
            left_on="pupil_time_ms",
            right_on="onset_ms",
            direction="backward",
        )
        # Containment: the sample must fall inside the active unit's window.
        inside = (j["pupil_time_ms"] >= j["start_ms"]) & (
            j["pupil_time_ms"] <= j["end_ms"]
        )
        j = j[inside & j["pupil"].notna()].copy()

        # Sign flip: surprisal = -log_prob (for the value and its lag).
        for v in vals:
            j[v] = -j[v]
            j[f"{v}_prev"] = -j[f"{v}_prev"]
        return j

    def _model_data(
        self,
        *,
        model: str,
        granularity: str,
        variant: str = "unmarg",
        offset: int = MAIN_OFFSET,
        data_root: Path | str | None = None,
    ) -> pd.DataFrame:
        """One model's analysis frame: rows are (item, Listener, pupil sample).

        Applies the exact ``run_cv_bootstrap`` filter for the cell, then exposes
        generic ``surp`` / ``prev_surp`` columns (surprisal units) and a
        per-item ``unit`` index (dense rank of the 40 ms grid, listener-invariant
        so the same grid point is one unit across listeners). ``variant`` selects
        marginalized (``"marg"``) vs realization-specific (``"unmarg"``)
        surprisal at the word level; phone/frame are unmarginalized only.
        """
        _validate_condition(model, granularity, variant)
        j = self._joined(model, granularity, offset)

        if model == "gpt2":
            col = "gpt2_log_prob"
            keep = j[col].notna() & j[f"{col}_prev"].notna()
            surp_col = col
        elif granularity == "word":
            um = f"{model}_unmarg_log_prob"
            im = f"{model}_is_log_prob"
            # Shared design: both estimates present for current and prev, the
            # unmarginalized surprisal is nonzero, and the single largest
            # marginalized surprisal is dropped as an outlier.
            keep = (
                j[um].notna()
                & j[f"{um}_prev"].notna()
                & j[im].notna()
                & j[f"{im}_prev"].notna()
                & (j[um] != 0)
                & (j[f"{um}_prev"] != 0)
            )
            sub = j[keep]
            keep = keep & (j[im] != sub[im].max())
            surp_col = um if variant == "unmarg" else im
        else:  # phone / frame, unmarginalized only
            um = f"{model}_unmarg_log_prob"
            keep = (
                j[um].notna()
                & j[f"{um}_prev"].notna()
                & (j[um] != 0)
                & (j[f"{um}_prev"] != 0)
            )
            surp_col = um

        m = j[keep].copy()
        out = m[["item", "Listener", "Time2StimOffset", "pupil"]].copy()
        out[SURP] = m[surp_col].to_numpy()
        out[PREV_SURP] = m[f"{surp_col}_prev"].to_numpy()
        out["token"] = m[_TOKEN_COL[granularity]].astype(str).to_numpy()
        # Listener-invariant unit id: dense rank of the grid time within an item.
        out["unit"] = (
            out.groupby("item")["Time2StimOffset"].rank(method="dense").astype(int)
        )
        # Normalized trial time (pupil_time_ms / duration_ms), raw, for the
        # appendix time-smooth baseline. Constant within (item, unit).
        out[NORM_TIME] = (m["pupil_time_ms"] / m["duration_ms"]).to_numpy()
        return out.reset_index(drop=True)

    # ------------------------------------------------------------------
    # Study interface
    # ------------------------------------------------------------------

    def dataset(
        self,
        *,
        model: str = "gslm",
        granularity: str = "word",
        variant: str = "unmarg",
        offset: int = MAIN_OFFSET,
        data_root: Path | str | None = None,
    ) -> TabularDataset:
        """Per-sample pupil frame for one (model, granularity, variant, offset).

        Units are 40 ms grid points; the participant axis is the listener; the
        measure is pupil dilation.

        Args:
            model: Language-model identifier selecting the surprisal columns.
            granularity: Surprisal unit: word, phone, or acoustic frame.
            variant: Marginalized or realization-specific surprisal condition.
            offset: Delay in milliseconds between the linguistic unit and pupil response.
            data_root: Optional root for external study dependencies.
        """
        df = self._model_data(
            model=model,
            granularity=granularity,
            variant=variant,
            offset=offset,
            data_root=data_root,
        )
        return TabularDataset(
            df,
            TabularDatasetConfig(
                name=self.name,
                language="en",
                text_col="token",
                trial_col="item",
                unit_index_col="unit",
                measure_col="pupil",
                measure_name="pupil",
                participant_col="Listener",
            ),
        )

    def evaluation_spec(
        self,
        *,
        model: str = "gslm",
        granularity: str = "word",
        variant: str = "unmarg",
        offset: int = MAIN_OFFSET,
        time_smooth: bool = False,
        data_root: Path | str | None = None,
        dataset,
    ) -> EvaluationSpec:
        """The paper's joint predictors: current surprisal plus its K=1 spillover.

        Both smooths enter one model, scored jointly against the RE-only baseline
        by :func:`_paper_delta_ll`. That function reads ``surp`` from the controls
        and ``prev_surp`` from the predictors and sets the formulas and their
        ``col_names`` itself, so the split between the two only routes columns.

        Args:
            dataset: Loaded dataset supplying the target observations.
            model: Language-model identifier selecting the surprisal columns.
            granularity: Surprisal unit: word, phone, or acoustic frame.
            variant: Marginalized or realization-specific surprisal condition.
            offset: Delay in milliseconds between the linguistic unit and pupil response.
            time_smooth: Include the appendix smooth of normalized within-item time.
            data_root: Optional root for external study dependencies.
        """
        df = self._model_data(
            model=model,
            granularity=granularity,
            variant=variant,
            offset=offset,
            data_root=data_root,
        )
        unit_df = df.drop_duplicates(subset=["item", "unit"]).sort_values(
            ["item", "unit"]
        )
        tag = f"{model}_{granularity}_{variant}"
        # The appendix time-smooth adds norm_time (constant per item/unit) as an
        # extra control column, so it reaches the analysis_fn aligned to y.
        columns = [SURP, PREV_SURP] + ([NORM_TIME] if time_smooth else [])
        fmodel = build_grouped_model(
            unit_df,
            groupby_col="item",
            sort_col="unit",
            columns=columns,
            model_name=tag,
        )
        controls = [ModelFeature(fmodel, PrecomputedFeature(SURP, SURP))]
        if time_smooth:
            controls.append(
                ModelFeature(fmodel, PrecomputedFeature(NORM_TIME, NORM_TIME))
            )
        return EvaluationSpec(
            controls=controls,
            predictors=[ModelFeature(fmodel, PrecomputedFeature(PREV_SURP, PREV_SURP))],
            targets=[dataset.feature("pupil")],
        )

    def analysis_config(
        self,
        *,
        model: str = "gslm",
        granularity: str = "word",
        variant: str = "unmarg",
        offset: int = MAIN_OFFSET,
        time_smooth: bool = False,
        data_root: Path | str | None = None,
    ) -> AnalysisConfig:
        """The paper's GAMM, scored by :func:`_paper_delta_ll` with item-fold CV.

        ``full``: ``pupil ~ s(surp,k=4) + s(prev_surp,k=4) + s(Listener,re)
        + s(item,re)``; ``baseline``: ``pupil ~ s(Listener,re) + s(item,re)``.
        Folds use the archived R map for this cell (``<granularity>_<model>``)
        and offset; Marg. and Unmarg. of one LM share a cell.

        Args:
            model: Language-model identifier selecting the surprisal columns.
            granularity: Surprisal unit: word, phone, or acoustic frame.
            variant: Marginalized or realization-specific surprisal condition.
            offset: Delay in milliseconds between the linguistic unit and pupil response.
            time_smooth: Include the appendix smooth of normalized within-item time.
            data_root: Optional root for external study dependencies.
        """
        _validate_condition(model, granularity, variant)
        cell = f"{granularity}_{model}"
        folds = _input_table("folds")
        fmap = folds[(folds["cell"] == cell) & (folds["offset_ms"] == offset)]
        if fmap.empty:
            raise ValueError(
                f"no archived fold map for cell {cell!r} at offset {offset}"
            )
        item_fold = dict(
            zip(fmap["item"].astype(str), fmap["fold"].astype(int), strict=True)
        )
        feature_name = f"{model}_{granularity}_{variant}"

        def _fn(**kwargs) -> list:
            return _paper_delta_ll(
                item_fold=item_fold,
                feature_name=feature_name,
                time_smooth=time_smooth,
                **kwargs,
            )

        return AnalysisConfig(design=None, transform_dv=_identity, analysis_fn=_fn)


# ----------------------------------------------------------------------------
# Coverage (Table 6): offset-independent, over lists 3-6, from the feather alone
# ----------------------------------------------------------------------------


def word_coverage() -> pd.DataFrame:
    """Reproduce the Table 6 word coverage: n estimated / n total (lists 3-6).

    Total is word tokens with valid timing over lists 3-6; estimated is those
    the IS estimator scored.
    """
    w = _input_table("combined_word")
    ok = w["word_idx"].notna() & w["start_time"].notna()
    total = int(ok.sum())
    rows = []
    for m in SPEECH_LMS:
        est = int((ok & w[f"{m}_is_log_prob"].notna()).sum())
        rows.append(
            {
                "model": m,
                "n_tokens_est": est,
                "n_tokens_total": total,
                "coverage": est / total,
            }
        )
    return pd.DataFrame(rows)
