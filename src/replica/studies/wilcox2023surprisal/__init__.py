"""Wilcox et al. (2023): Testing the Predictions of Surprisal Theory in 11 Languages.

Reproduction scope, anchors and deviations: ``README.md``.
"""

import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from replica.analysis.specification.config import AnalysisConfig
from replica.analysis.specification.evaluation import EvaluationSpec
from replica.feature import ModelFeature

from ...analysis.cv import NoScaler, SequentialBlockSplitter
from ...analysis.models import OLS
from ...dataset.base import TabularDataset, TabularDatasetConfig
from ...external import repo_dependencies, repo_path
from ...feature import (
    Feature,
    FeatureModel,
    PrecomputedFeatureModel,
)
from ...feature.extractors import (
    Entropy,
    Lag,
    PrecomputedFeature,
    Surprisal,
    TokenEntropy,
)
from ..base import Study, build_grouped_model

LANGUAGES = ["du", "en", "fi", "ge", "gr", "he", "it", "ko", "ru", "sp", "tr"]
LMS = ["mgpt_sc", "mgpt_lc", "monot_30m", "monot_all"]
DVS = ["gaze_rt", "total_rt", "firstfix_rt"]

CORRECTED_ENTROPY_MODEL = "ai-forever/mGPT"
DEFAULT_LM = "mgpt_lc"
DEFAULT_DV = "gaze_rt"

SURP_COLS = ["surp", "prev_surp", "prev2_surp"]
ENT_COLS = ["ent", "prev_ent", "prev2_ent"]
CTRL_COLS = ["freq", "len", "prev_freq", "prev_len", "prev2_freq", "prev2_len"]

# The linear-versus-nonlinear link comparison (analysis_link.Rmd): gaze
# duration only, two spillover-free focal terms, tensor-smooth controls.
LINK_PRED_COLS = ["surp", "prev_surp"]
LINK_CTRL_COLS = ["freq", "len", "prev_freq", "prev_len"]
LINK_DV = "gaze_rt"


def _r_parse_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Convert string columns to float with R's own parser via rpy2.

    The released analysis consumed ``read.csv``'s values, and R's parser
    lands one ulp off the correctly rounded double on some released
    strings; mgcv's GCV search amplifies one ulp into fold-level
    differences, so the reproduction fits the values R's parser produces.
    """
    from replica.analysis.models.support.r import _r_context

    df = df.copy()
    with _r_context() as ro:
        for col in columns:
            ro.globalenv[".vals"] = ro.StrVector(
                [v if isinstance(v, str) and v != "" else "NA" for v in df[col]]
            )
            df[col] = np.asarray(
                ro.r("suppressWarnings(as.numeric(.vals))"), dtype=float
            )
    return df


LINK_BASELINE_FORMULA = (
    "y ~ te(freq, len, bs = 'cr') + te(prev_freq, prev_len, bs = 'cr')"
)
LINK_FORMULAS = {
    "linear": "y ~ surp + prev_surp + " + LINK_BASELINE_FORMULA[4:],
    "nonlinear": (
        "y ~ s(surp, bs = 'cr', k = 6) + s(prev_surp, bs = 'cr', k = 6) + "
        + LINK_BASELINE_FORMULA[4:]
    ),
}


def _features(
    cols: list[str],
    model: FeatureModel | None,
    table: FeatureModel,
    token_entropy: bool = False,
) -> list[Feature]:
    """The named released columns, as features.

    With a live ``model``, the surprisal and entropy triples become live
    extractors on it instead: the paper's ``prev_``/``prev2_`` columns are
    one- and two-word spillover, so ``Lag`` reproduces them from the live
    values. Frequency and length always stay the authors' released columns.
    When the entropy contrast runs live, its surprisal baseline is the live
    model's own, so the comparison stays within one model. ``token_entropy``
    reads the entropy off the model's first-subword token distribution
    (:class:`TokenEntropy`), the quantity the paper defines and the released
    column sums over subwords, instead of the word-marginalised
    :class:`Entropy`.
    """
    if model is None:
        return [ModelFeature(table, PrecomputedFeature(c, c)) for c in cols]
    live: list[Feature] = []
    rest = list(cols)
    entropy = TokenEntropy if token_entropy else Entropy
    for group, extractor in ((SURP_COLS, Surprisal), (ENT_COLS, entropy)):
        if set(group) <= set(rest):
            base = ModelFeature(model, extractor())
            live += [base, Lag(base, 1), Lag(base, 2)]
            rest = [c for c in rest if c not in group]
    return live + [ModelFeature(table, PrecomputedFeature(c, c)) for c in rest]


class Wilcox2023Surprisal(Study):
    """Wilcox et al. (2023), surprisal & entropy across 11 MECO languages."""

    name = "wilcox2023surprisal"
    bibtex = r"""
    @article{wilcox-etal-2023-testing,
      title = {Testing the Predictions of Surprisal Theory in 11 Languages},
      author = {Wilcox, Ethan G. and
          Pimentel, Tiago and
          Meister, Clara and
          Cotterell, Ryan and
          Levy, Roger P.},
      journal = {Transactions of the Association for Computational Linguistics},
      volume = {11},
      year = {2023},
      url = {https://aclanthology.org/2023.tacl-1.82/},
      doi = {10.1162/tacl_a_00612},
      pages = {1451--1470}
    }
    """
    dependencies = repo_dependencies("xlang-processing")

    def _data_dir(self, data_root: Path | str | None = None) -> Path:
        root = self._resolve_root(data_root)
        return (
            root
            / "datasets"
            / "eye-tracking"
            / "processed"
            / "meco_wilcox2023surprisal"
        )

    def _ensure_symlinks(self, data_root: Path | str | None = None) -> None:
        src = (
            repo_path("xlang-processing", data_root, fetch=False)
            / "data"
            / "merged_data"
        )
        dst = self._data_dir(data_root)
        dst.mkdir(parents=True, exist_ok=True)
        # Relative so the processed tree stays movable with its data root.
        rel = Path(os.path.relpath(src, dst))
        for csv in src.glob("*.csv"):
            link = dst / csv.name
            if link.is_symlink() or link.exists():
                continue
            link.symlink_to(rel / csv.name)

    def _load_language(
        self,
        lang: str,
        data_root: Path | str | None = None,
        lm: str = DEFAULT_LM,
    ) -> pd.DataFrame:
        """Filters mirror ``analysis_surp.Rmd``: freq>0, finite, ``model==lm``,
        ``drop_na()``.
        """
        self.ensure_data(data_root)
        self._ensure_symlinks(data_root)
        df = pd.read_csv(self._data_dir(data_root) / f"{lang}.csv")
        df = df[(df["freq"] > 0) & (df["prev_freq"] > 0) & (df["prev2_freq"] > 0)]
        df = df[
            np.isfinite(df["freq"])
            & np.isfinite(df["prev_freq"])
            & np.isfinite(df["prev2_freq"])
        ]
        df = df[df["model"] == lm]
        # Wide dropna mirrors the R recipe and is pinned by the paper anchor.
        return df.dropna().reset_index(drop=True)

    def _full_frame(
        self,
        lang: str,
        data_root: Path | str | None,
        lm: str,
    ) -> pd.DataFrame:
        """Every interest area of every trial, dense in ``ianum``.

        A live model conditions on the stimulus text, and ``_load_language``
        drops the first two interest areas of each trial, so building stimuli
        from it would hand the model a text missing its opening words. Here
        the excluded rows stay as units and instead carry NaN in every
        analysis column, which keeps them out of the regression while the
        text stays whole. Density in ``ianum`` is what makes a positional
        :class:`Lag` the authors' ``lag()``, which is by interest-area index.
        """
        self.ensure_data(data_root)
        self._ensure_symlinks(data_root)
        df = pd.read_csv(self._data_dir(data_root) / f"{lang}.csv")
        df = df[df["model"] == lm]

        kept = self._load_language(lang, data_root, lm=lm)
        excluded = ~df.set_index(["trialid", "ianum"]).index.isin(
            set(zip(kept["trialid"], kept["ianum"], strict=True))
        )
        df = df.copy()
        df.loc[excluded, SURP_COLS + ENT_COLS + CTRL_COLS + DVS] = np.nan

        # The release itself omits a handful of interest areas; they become
        # "?" units so the index stays an offset.
        filled = []
        for trial_id, trial in df.groupby("trialid"):
            dense = (
                trial.set_index("ianum")
                .reindex(range(1, int(trial["ianum"].max()) + 1))
                .reset_index()
            )
            dense["trialid"] = trial_id
            filled.append(dense)
        return pd.concat(filled, ignore_index=True)

    def dataset(
        self,
        language: str = "en",
        data_root: Path | str | None = None,
        lm: str = DEFAULT_LM,
        dv: str = DEFAULT_DV,
        full_text: bool = False,
    ) -> TabularDataset:
        """Load one MECO language's released interest-area table.

        ``full_text=True`` keeps the interest areas the analysis excludes as
        units, so the stimulus carries the whole trial. Pair it with
        ``evaluation_spec(model=...)``; the two frames have different unit
        counts, so a mismatched pairing raises rather than misaligning.

        Args:
            language: MECO language code, one of :data:`LANGUAGES`.
            data_root: Directory the study's dependency paths resolve
                against.
            lm: Released language-model tag whose rows are kept (the
                ``model`` column filter).
            dv: Reading-time measure loaded as the dataset's DV column.
            full_text: Keep every interest area, including the ones the
                default recipe excludes, so the stimulus text stays whole.

        Returns:
            A dataset over the interest areas :meth:`evaluation_spec` expects,
            dense in interest-area index when *full_text* is true.
        """
        df = (
            self._full_frame(language, data_root, lm)
            if full_text
            else self._load_language(language, data_root, lm=lm)
        )
        return TabularDataset(
            df,
            TabularDatasetConfig(
                name=f"MECO-{language}", language=language, measure_col=dv
            ),
        )

    def _spec(
        self,
        language: str,
        controls: list[str],
        predictors: list[str],
        dv: str,
        lm: str,
        data_root: Path | str | None,
        model: FeatureModel | None = None,
        token_entropy: bool = False,
        *,
        dataset,
    ) -> EvaluationSpec:
        df = (
            self._load_language(language, data_root, lm=lm)
            if model is None
            else self._full_frame(language, data_root, lm)
        )
        table = build_grouped_model(
            df,
            groupby_col="trialid",
            sort_col="ianum",
            columns=predictors + controls,
            model_name=lm,
        )
        return EvaluationSpec(
            predictors=_features(predictors, model, table, token_entropy),
            controls=_features(controls, model, table, token_entropy),
            targets=[dataset.feature(dv)],
        )

    def evaluation_spec(
        self,
        language: str = "en",
        data_root: Path | str | None = None,
        lm: str = DEFAULT_LM,
        dv: str = DEFAULT_DV,
        contrast: str = "surp",
        model: FeatureModel | None = None,
        token_entropy: bool = False,
        *,
        dataset,
    ) -> EvaluationSpec:
        """Configure the surprisal or entropy-add contrast for one language.

        The "replace" panel of ``analysis_ent.Rmd`` has a non-nested baseline, so
        it cannot be a single ``evaluate()`` call; use
        :meth:`evaluate_entropy_replace_dll`.

        Args:
            dataset: Loaded dataset supplying the target observations.
            language: MECO language code, one of :data:`LANGUAGES`.
            data_root: Directory the study's dependency paths resolve
                against.
            lm: Released language-model tag the controls (and, with no
                *model*, the focal predictors) are read from.
            dv: Reading-time measure the spec is scored against.
            contrast: ``"surp"`` for the surprisal contrast of
                ``analysis_surp.Rmd``, ``"ent_add"`` for the
                entropy-plus-surprisal-baseline contrast, the "add" panel of
                ``analysis_ent.Rmd``.
            model: Comprehension model that supplies the focal surprisal or
                entropy and its two spillover lags in place of the released LM
                column, while the authors' controls and DV stay as released;
                ``None`` uses the released *lm* table throughout.
            token_entropy: With a live *model*, take its first-subword token
                entropy as the entropy predictor, the paper's definition,
                rather than the word-marginalised entropy.

        Raises:
            ValueError: When *contrast* is not ``"surp"`` or ``"ent_add"``.
        """
        if contrast == "surp":
            return self._spec(
                language,
                CTRL_COLS,
                SURP_COLS,
                dv,
                lm,
                data_root,
                model,
                token_entropy,
                dataset=dataset,
            )
        if contrast == "ent_add":
            return self._spec(
                language,
                CTRL_COLS + SURP_COLS,
                ENT_COLS,
                dv,
                lm,
                data_root,
                model,
                token_entropy,
                dataset=dataset,
            )
        raise ValueError(f"unknown contrast {contrast!r}; expected 'surp' or 'ent_add'")

    def analysis_config(self) -> AnalysisConfig:
        """Configure the shared surprisal/entropy regression recipe.

        Mirrors the R ``lm()`` recipe: plain OLS, no scaling, raw DV, and the
        released analyses' sequential ten-fold split.
        """
        return AnalysisConfig(
            design=None,
            model=OLS(),
            cv=SequentialBlockSplitter(10),
            scaler=NoScaler(),
            transform_dv=lambda x: x,
        )

    def evaluate_entropy_replace_dll(
        self,
        language: str = "en",
        data_root: Path | str | None = None,
        lm: str = DEFAULT_LM,
        dv: str = DEFAULT_DV,
        model: FeatureModel | None = None,
        token_entropy: bool = False,
    ) -> dict:
        """Paper's "replace" contrast as ``DLL(ent+ctrl) - DLL(surp+ctrl)``.

        Composed from two nested :func:`evaluate` calls; Gaussian LLs are
        additive so this equals fitting the non-nested pair directly.

        Args:
            language: MECO language code, one of :data:`LANGUAGES`.
            data_root: Directory the study's dependency paths resolve
                against.
            lm: Released language-model tag the controls and focal columns
                are read from.
            dv: Reading-time measure the contrast is scored against.
            model: Live model supplying surprisal and entropy in place of the
                released columns, as in :meth:`evaluation_spec`.
            token_entropy: With a live *model*, use its first-subword token
                entropy, as in :meth:`evaluation_spec`.

        Returns:
            ``delta_ll`` (the replace contrast), plus the underlying
            ``surp_only``/``ent_only`` :func:`evaluate` results and
            ``n_observations``.
        """
        from replica.analysis.evaluation.evaluate import evaluate

        cfg = self.analysis_config()
        ds = self.dataset(
            language=language,
            data_root=data_root,
            lm=lm,
            dv=dv,
            full_text=model is not None,
        )
        surp = evaluate(
            ds,
            self._spec(
                language,
                CTRL_COLS,
                SURP_COLS,
                dv,
                lm,
                data_root,
                model,
                token_entropy,
                dataset=ds,
            ),
            cfg,
        )
        ent = evaluate(
            ds,
            self._spec(
                language,
                CTRL_COLS,
                ENT_COLS,
                dv,
                lm,
                data_root,
                model,
                token_entropy,
                dataset=ds,
            ),
            cfg,
        )
        return {
            "delta_ll": ent[0].delta_ll - surp[0].delta_ll,
            "surp_only": surp,
            "ent_only": ent,
            "n_observations": surp[0].n_observations,
        }

    def per_obs_delta_ll_all(
        self,
        language: str = "en",
        data_root: Path | str | None = None,
        lm: str = DEFAULT_LM,
        dv: str = DEFAULT_DV,
        model: FeatureModel | None = None,
        token_entropy: bool = False,
    ) -> dict[str, np.ndarray]:
        """All three contrasts' per-datapoint held-out ΔLL, in fold order.

        The ``ent_replace`` pair is non-nested, so its vector is the difference
        of two nested vectors, whose shared controls-only baseline cancels row by
        row; its surprisal-only fit is the ``surp`` contrast's, so the three
        contrasts take three fits.

        Args:
            language: MECO language code, one of :data:`LANGUAGES`.
            data_root: Directory the study's dependency paths resolve
                against.
            lm: Released language-model tag the controls and focal columns
                are read from.
            dv: Reading-time measure the contrasts are scored against.
            model: Live model supplying surprisal and entropy in place of the
                released columns, as in :meth:`evaluation_spec`.
            token_entropy: With a live *model*, use its first-subword token
                entropy, as in :meth:`evaluation_spec`.

        Returns:
            ``{"surp": ..., "ent_add": ..., "ent_replace": ...}``, one
            per-observation ΔLL vector per contrast: the vectors the paper's
            significance test consumes, since ``model_cross_val`` returns one
            held-out log density per test row and the test runs on
            target-minus-baseline.
        """
        from replica.analysis.evaluation.evaluate import evaluate

        from ...analysis.inference.significance import concat_fold_per_obs_delta_ll

        cfg = replace(
            self.analysis_config(), per_fold=True, retain_per_obs_delta_ll=True
        )
        ds = self.dataset(
            language=language,
            data_root=data_root,
            lm=lm,
            dv=dv,
            full_text=model is not None,
        )

        def vector(predictors: list[str], controls: list[str]) -> np.ndarray:
            spec = self._spec(
                language, controls, predictors, dv, lm, data_root, model, token_entropy
            )
            return concat_fold_per_obs_delta_ll(evaluate(ds, spec, cfg)[0].fold_results)

        surp = vector(SURP_COLS, CTRL_COLS)
        return {
            "surp": surp,
            "ent_add": vector(ENT_COLS, CTRL_COLS + SURP_COLS),
            "ent_replace": vector(ENT_COLS, CTRL_COLS) - surp,
        }

    def per_obs_delta_ll(
        self,
        language: str = "en",
        data_root: Path | str | None = None,
        lm: str = DEFAULT_LM,
        dv: str = DEFAULT_DV,
        contrast: str = "surp",
    ) -> np.ndarray:
        """One contrast's per-datapoint held-out ΔLL.

        See :meth:`per_obs_delta_ll_all` for how each contrast is computed.

        Args:
            language: MECO language code, one of :data:`LANGUAGES`.
            data_root: Directory the study's dependency paths resolve
                against.
            lm: Released language-model tag the controls and focal columns
                are read from.
            dv: Reading-time measure the contrast is scored against.
            contrast: One of ``"surp"``, ``"ent_add"``, ``"ent_replace"``.

        Raises:
            ValueError: When *contrast* is not a known contrast name.
        """
        vectors = self.per_obs_delta_ll_all(
            language=language, data_root=data_root, lm=lm, dv=dv
        )
        if contrast not in vectors:
            raise ValueError(f"unknown contrast {contrast!r}")
        return vectors[contrast]

    def _load_language_link(
        self,
        lang: str,
        data_root: Path | str | None = None,
        lm: str = DEFAULT_LM,
    ) -> pd.DataFrame:
        """Filters mirror ``analysis_link.Rmd``: the three frequency filters
        and ``model==lm`` only; unlike the other contrasts there is no
        ``drop_na()``, so a row NA-free in the link columns stays even when
        an entropy column is missing. The analysis columns go through
        :func:`_r_parse_columns` so the fitted values are R's parse.
        """
        self.ensure_data(data_root)
        self._ensure_symlinks(data_root)
        link_cols = [LINK_DV, *LINK_CTRL_COLS, *LINK_PRED_COLS]
        df = pd.read_csv(
            self._data_dir(data_root) / f"{lang}.csv",
            dtype={c: str for c in link_cols},
        )
        df = _r_parse_columns(df, link_cols)
        df = df[(df["freq"] > 0) & (df["prev_freq"] > 0) & (df["prev2_freq"] > 0)]
        df = df[
            np.isfinite(df["freq"])
            & np.isfinite(df["prev_freq"])
            & np.isfinite(df["prev2_freq"])
        ]
        return df[df["model"] == lm].reset_index(drop=True)

    def link_dataset(
        self,
        language: str = "en",
        data_root: Path | str | None = None,
        lm: str = DEFAULT_LM,
    ) -> TabularDataset:
        """Load the link-comparison table for one MECO language.

        Args:
            language: MECO language code, one of :data:`LANGUAGES`.
            data_root: Directory the study's dependency paths resolve
                against.
            lm: Released language-model tag whose rows are kept.

        Returns:
            A dataset over the link comparison's interest areas, with its DV
            column set to :data:`LINK_DV`.
        """
        df = self._load_language_link(language, data_root, lm)
        return TabularDataset(
            df,
            TabularDatasetConfig(
                name=f"MECO-{language}", language=language, measure_col=LINK_DV
            ),
        )

    def link_evaluation_spec(
        self,
        language: str = "en",
        data_root: Path | str | None = None,
        lm: str = DEFAULT_LM,
        *,
        dataset,
    ) -> EvaluationSpec:
        """Configure the released surprisal predictors for the link comparison.

        Args:
            dataset: Loaded dataset supplying the target observations.
            language: MECO language code, one of :data:`LANGUAGES`.
            data_root: Directory the study's dependency paths resolve
                against.
            lm: Released language-model tag the predictors and controls are
                read from.

        Returns:
            A spec pairing :data:`LINK_PRED_COLS` against
            :data:`LINK_CTRL_COLS`, scored on :data:`LINK_DV`.
        """
        df = self._load_language_link(language, data_root, lm)
        table = build_grouped_model(
            df,
            groupby_col="trialid",
            sort_col="ianum",
            columns=LINK_PRED_COLS + LINK_CTRL_COLS,
            model_name=lm,
        )
        return EvaluationSpec(
            predictors=[
                ModelFeature(table, PrecomputedFeature(c, c)) for c in LINK_PRED_COLS
            ],
            controls=[
                ModelFeature(table, PrecomputedFeature(c, c)) for c in LINK_CTRL_COLS
            ],
            targets=[dataset.feature(LINK_DV)],
        )

    def link_analysis_config(self, arm: str) -> AnalysisConfig:
        """Configure one arm of the linear-vs-nonlinear link comparison.

        ``analysis_link.Rmd``'s engine for one arm: mgcv ``gam()`` at its
        default GCV.Cp method, both arms over the shared tensor-smooth
        baseline, raw DV, the Rmd's sequential ten folds.

        Args:
            arm: ``"linear"`` or ``"nonlinear"``, selecting which of
                :data:`LINK_FORMULAS` fits the focal surprisal terms.

        Raises:
            ValueError: When *arm* is neither ``"linear"`` nor ``"nonlinear"``.
        """
        from ...analysis.models import RGamModel

        if arm not in LINK_FORMULAS:
            raise ValueError(f"unknown arm {arm!r}; expected 'linear' or 'nonlinear'")
        col_names = {f"x{i}": c for i, c in enumerate(LINK_CTRL_COLS + LINK_PRED_COLS)}
        return AnalysisConfig(
            design=None,
            model=RGamModel(
                formula=LINK_FORMULAS[arm],
                baseline_formula=LINK_BASELINE_FORMULA,
                col_names=col_names,
                method="GCV.Cp",
            ),
            cv=SequentialBlockSplitter(10),
            scaler=NoScaler(),
            transform_dv=lambda x: x,
            per_fold=True,
            retain_per_obs_delta_ll=True,
        )

    def link_per_obs_delta_ll(
        self,
        language: str = "en",
        data_root: Path | str | None = None,
        lm: str = DEFAULT_LM,
    ) -> dict[str, np.ndarray]:
        """Each link arm's per-datapoint held-out ΔLL over the shared baseline.

        Two ``evaluate()`` calls, one per arm; the baseline fit is identical
        in both, so ``linear - nonlinear`` cancels it row by row, which is
        the vector the Rmd's ``perm.test`` consumes.

        Args:
            language: MECO language code, one of :data:`LANGUAGES`.
            data_root: Directory the study's dependency paths resolve
                against.
            lm: Released language-model tag the predictors and controls are
                read from.

        Returns:
            ``{"linear": ..., "nonlinear": ...}``, one per-observation ΔLL
            vector per arm.
        """
        from replica.analysis.evaluation.evaluate import evaluate

        from ...analysis.inference.significance import concat_fold_per_obs_delta_ll

        ds = self.link_dataset(language, data_root, lm)
        spec = self.link_evaluation_spec(language, data_root, lm, dataset=ds)
        return {
            arm: concat_fold_per_obs_delta_ll(
                evaluate(ds, spec, self.link_analysis_config(arm))[0].fold_results
            )
            for arm in ("linear", "nonlinear")
        }

    def link_cell(
        self,
        language: str = "en",
        data_root: Path | str | None = None,
        lm: str = DEFAULT_LM,
        num_sim: int = 1000,
        seed: int = 444,
    ) -> dict:
        """One (language, lm) cell of the link comparison, release-faithful.

        Applies the release's own outlier rule, including its asymmetry, and
        both ``perm.test`` calls through the Rmd's ``jmuOutlier``:
        ``p_unpaired`` is what the release's ``paried = T`` call computes,
        ``p_paired`` the paired test that call names.

        Args:
            language: MECO language code, one of :data:`LANGUAGES`.
            data_root: Directory the study's dependency paths resolve
                against.
            lm: Released language-model tag the predictors and controls are
                read from.
            num_sim: Monte Carlo draws for each ``perm.test`` call.
            seed: Seed applied before each ``perm.test`` call, so both
                p-values are individually reproducible.

        Returns:
            The cell's observation counts, filter threshold, per-arm mean
            ΔLL, and both permutation p-values.
        """
        vectors = self.link_per_obs_delta_ll(language, data_root, lm)
        lin, non = vectors["linear"], vectors["nonlinear"]
        keep = np.isfinite(lin) & np.isfinite(non)
        n_obs = int(lin.size)
        lin, non = lin[keep], non[keep]
        comp = lin - non
        thresh = float(comp.mean() + 3 * comp.std(ddof=1))
        outlier = np.abs(comp) > thresh
        lin_f, non_f = lin[~outlier], non[~outlier]
        p_unpaired, p_paired = jmu_perm_test_pair(
            lin_f, non_f, num_sim=num_sim, seed=seed
        )
        comp_f = lin_f - non_f
        return {
            "lang": language,
            "lm": lm,
            "dv": LINK_DV,
            "n_obs": n_obs,
            "n_tested": int(lin_f.size),
            "n_dropped_by_filter": int(outlier.sum()),
            "filter_threshold": thresh,
            "mean_dll_linear": float(lin_f.mean()),
            "mean_dll_nonlinear": float(non_f.mean()),
            "mean_dll_comp": float(comp_f.mean()),
            "se_dll_comp": float(comp_f.std(ddof=1) / np.sqrt(comp_f.size)),
            "p_unpaired": p_unpaired,
            "p_paired": p_paired,
            "num_sim": num_sim,
        }

    def significance(
        self,
        language: str = "en",
        data_root: Path | str | None = None,
        lm: str = DEFAULT_LM,
        dv: str = DEFAULT_DV,
        contrast: str = "surp",
        n_resamples: int = 100_000,
        seed: int = 0,
    ) -> dict:
        """The paper's significance test for one cell.

        Runs :func:`per_datapoint_permutation_pvalue` on
        :meth:`per_obs_delta_ll`, the two-sided sign-flip test of
        ``analysis_surp.Rmd``/``analysis_ent.Rmd``. ``p_normal`` is the same
        test's RNG-free normal limit; see :func:`signflip_moments`.

        Args:
            language: MECO language code, one of :data:`LANGUAGES`.
            data_root: Directory the study's dependency paths resolve
                against.
            lm: Released language-model tag the controls and focal columns
                are read from.
            dv: Reading-time measure the contrast is scored against.
            contrast: One of ``"surp"``, ``"ent_add"``, ``"ent_replace"``.
            n_resamples: Sign-flip resamples drawn for ``p_perm``.
            seed: Seed for the sign-flip resamples.

        Returns:
            :func:`signflip_summary`'s moments and both p-values for this
            cell's per-observation ΔLL vector.
        """
        return signflip_summary(
            self.per_obs_delta_ll(
                language=language,
                data_root=data_root,
                lm=lm,
                dv=dv,
                contrast=contrast,
            ),
            n_resamples=n_resamples,
            seed=seed,
        )


def jmu_perm_test_pair(
    x: np.ndarray,
    y: np.ndarray,
    num_sim: int = 1000,
    seed: int = 444,
) -> tuple[float, float]:
    """The Rmd's two ``jmuOutlier::perm.test`` p-values, unpaired then paired.

    Runs the release's own test engine through rpy2, seeded per call so both
    p-values are reproducible; the release seeds once globally, so its exact
    draws are not recoverable and the comparand is the decision.

    Args:
        x: First sample's values.
        y: Second sample's values.
        num_sim: Monte Carlo draws for each ``perm.test`` call.
        seed: Seed applied before each ``perm.test`` call.

    Returns:
        ``(p_unpaired, p_paired)``.
    """
    from replica.analysis.models.support.r import _r_context

    with _r_context() as ro:
        ro.r("suppressPackageStartupMessages(library(jmuOutlier))")
        ro.globalenv[".x"] = ro.FloatVector(np.asarray(x, dtype=float))
        ro.globalenv[".y"] = ro.FloatVector(np.asarray(y, dtype=float))
        ro.globalenv[".num_sim"] = float(num_sim)
        ro.r(f"set.seed({int(seed)})")
        p_unpaired = float(
            np.array(ro.r("perm.test(.x, .y, num.sim=.num_sim)$p.value"))[0]
        )
        ro.r(f"set.seed({int(seed)})")
        p_paired = float(
            np.array(ro.r("perm.test(.x, .y, num.sim=.num_sim, paired=TRUE)$p.value"))[
                0
            ]
        )
    return p_unpaired, p_paired


def signflip_moments(deltas: np.ndarray) -> dict:
    """The tested vector's moments and the RNG-free sign-flip statistic.

    Every entry is a deterministic function of the vector, so these are the
    quantities two engines can be held to value-for-value; ``p_normal`` is
    the normal limit of the same permutation test.

    Args:
        deltas: Per-observation held-out ΔLL values; non-finite entries are
            dropped before the moments are computed.

    Returns:
        Sample size, mean/sd/se/min/max, the sign-flip z-statistic, and its
        normal-approximation p-value.
    """
    from ...analysis.inference.significance import signflip_normal_pvalue

    x = np.asarray(deltas, dtype=float)
    x = x[np.isfinite(x)]
    return {
        "n": int(x.size),
        "mean_dll": float(x.mean()),
        "sd_dll": float(x.std(ddof=1)),
        "se_dll": float(x.std(ddof=1) / np.sqrt(x.size)),
        "min_dll": float(x.min()),
        "max_dll": float(x.max()),
        "z_signflip": float(x.sum() / np.sqrt(np.dot(x, x))),
        "p_normal": signflip_normal_pvalue(x),
    }


def signflip_summary(
    deltas: np.ndarray,
    n_resamples: int = 100_000,
    seed: int = 0,
) -> dict:
    """:func:`signflip_moments` plus the simulated sign-flip p-value.

    Args:
        deltas: Per-observation held-out ΔLL values; non-finite entries are
            dropped before the test runs.
        n_resamples: Sign-flip resamples drawn for the simulated p-value.
        seed: Seed for the sign-flip resamples.

    Returns:
        :func:`signflip_moments`'s fields, plus ``p_perm`` and
        ``n_resamples``.
    """
    from ...analysis.inference.significance import per_datapoint_permutation_pvalue

    x = np.asarray(deltas, dtype=float)
    x = x[np.isfinite(x)]
    return {
        **signflip_moments(x),
        "p_perm": per_datapoint_permutation_pvalue(
            x, n_resamples=n_resamples, rng=np.random.default_rng(seed)
        ),
        "n_resamples": n_resamples,
    }


def cached_outputs(
    study: Wilcox2023Surprisal,
    model: FeatureModel,
    language: str,
    data_root: Path | None = None,
) -> PrecomputedFeatureModel:
    """Score every trial of one language once and serve the outputs from memory.

    The nine fits behind a language's three measures and three contrasts read
    the same per-word values, so the model runs once per language.

    Args:
        study: The study whose dense ``full_text`` frame defines the trials.
        model: The live model to score them with.
        language: MECO language code.
        data_root: Directory the study's dependency paths resolve against.
    """
    dataset = study.dataset(
        language=language, data_root=data_root, lm=DEFAULT_LM, full_text=True
    )
    outputs = {s.stimulus_id: model(s) for s in dataset.stimuli()}
    return PrecomputedFeatureModel(getattr(model, "name", "live"), outputs)


def corrected_entropy_model() -> FeatureModel:
    """mGPT as the paper scores it, with first-subword token entropy on."""
    from replica.feature.backends.huggingface import HuggingFaceLM

    return HuggingFaceLM(
        CORRECTED_ENTROPY_MODEL,
        add_bos=False,
        provide_word_dist=False,
        token_level_entropy=True,
    )
