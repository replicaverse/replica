"""Hopton et al. (2026b): Revisiting the Estimation of Unigram Surprisal in Computational Psycholinguistics.

Extends Opedal et al. (2024) Experiment 2 by replacing their ``wordfreq``
unigram surprisal with a marginalized Monte Carlo (MMC) estimate sampled from
mGPT, and refits the GAMMs of reading time. The fixture TSVs are outputs of the
independent R port.

Reproduction scope, recipe, anchors and deviations: README.md.

Bibtex:

    @misc{hopton2026unigram,
      title = {Revisiting the Estimation of Unigram Surprisal in Computational Psycholinguistics},
      author = {Hopton, Zachary William and Re, Francesco Ignazio and Kiegeland, Samuel and Opedal, Andreas and Chodroff, Eleanor and Cotterell, Ryan},
      year = {2026},
      archivePrefix = {arXiv},
      eprint = {2610.18414},
      url = {https://arxiv.org/abs/2610.18414}
    }
"""

from __future__ import annotations

import functools
import hashlib
from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from replica.analysis.specification.config import AnalysisConfig
from replica.analysis.specification.evaluation import EvaluationSpec

from ...analysis.cv import ZScoreScaler
from ...analysis.models import RGamModel
from ...dataset.base import TabularDataset, TabularDatasetConfig
from ...external import repo_dependencies
from ...external.adapters.context_reading import data_dir
from ..base import Study
from ..records.resources import resource_file

LANGUAGES = ["du", "en", "fi", "ge", "gr", "he", "it", "ko", "ru", "sp", "tr"]

MEASURES = ("total_rt", "gaze_rt", "firstfix_rt")

#: ``Standard`` reads the released wordfreq columns, ``RB_Estimate`` the archived
#: MMC estimates. The names match the ``condition`` column of the fixtures.
CONDITIONS = ("Standard", "RB_Estimate")

ESTIMATORS = (
    "reference",
    "nmc",
    "mmc_pathwise",
    "mmc_expectation_equivalent",
    "priority_mmc_pathwise",
    "priority_mmc_expectation_equivalent",
)

COMPS = ("surp", "freq", "pmi", "surp+freq", "pmi+freq")

MODEL_NAME = "mgpt_lc"

#: Leading-whitespace variant of the MMC estimate. The main text uses the
#: space column; the appendix ablation uses the other.
_ESTIMATE_COLUMNS = {
    True: "rb_surp_N500_realigned_space",
    False: "rb_surp_N500_realigned_no_space",
}

#: Per-condition predictor columns, mirroring ``condition_specs`` in the R port.
_CONDITION_COLUMNS = {
    "Standard": {
        "freq": "freq",
        "prev_freq": "prev_freq",
        "pmi": "pmi",
        "prev_pmi": "prev_pmi",
    },
    "RB_Estimate": {
        "freq": "freq_BW",
        "prev_freq": "prev_freq_BW",
        "pmi": "pmi_BW",
        "prev_pmi": "prev_pmi_BW",
    },
}


def _identity(y: np.ndarray) -> np.ndarray:
    """DV transform: the paper fits raw reading time."""
    return y


@functools.lru_cache(maxsize=8)
def _input_table(name: str) -> pd.DataFrame:
    """Read a checksum-verified study table, cached across calls."""
    return pd.read_parquet(resource_file("hopton2026unigram", f"data/{name}.parquet"))


class PrecomputedFoldSplitter:
    """Replays a fold assignment computed elsewhere, aligned to the flat rows.

    ``evaluate()`` flattens observations as ``(stimulus, participant, unit)``
    and hands splitters only ``(y, groups, participants, items)``, with no unit
    index, so a fold map keyed by ``(participant, trial, unit)`` cannot be
    resolved inside :meth:`split`. The study therefore resolves it up front and
    passes the per-row fold ids here.

    ``expected_participants`` / ``expected_items`` are checked against what
    ``evaluate()`` actually passes. A misalignment would otherwise be silent:
    folds would shuffle and every delta log-likelihood would move without any
    error, so the mismatch is raised instead.
    """

    def __init__(
        self,
        fold_ids: np.ndarray,
        *,
        expected_participants: np.ndarray,
        expected_items: np.ndarray,
    ):
        self.fold_ids = np.asarray(fold_ids)
        self._expected_participants = np.asarray(expected_participants)
        self._expected_items = np.asarray(expected_items)

    @property
    def n_folds(self) -> int:
        return int(len(np.unique(self.fold_ids)))

    def configuration(self) -> dict:
        """Identify the ordered fold assignment and grouping codes."""
        rows = np.column_stack(
            [self.fold_ids, self._expected_participants, self._expected_items]
        ).astype("<i8")
        return {
            "n_folds": self.n_folds,
            "assignment_sha256": hashlib.sha256(rows.tobytes()).hexdigest(),
        }

    def split(
        self,
        y: np.ndarray,
        groups: np.ndarray,
        participants: np.ndarray,
        items: np.ndarray,
    ) -> Iterable[tuple[np.ndarray, np.ndarray]]:
        if len(y) != len(self.fold_ids):
            raise ValueError(
                f"PrecomputedFoldSplitter: {len(self.fold_ids)} fold ids for "
                f"{len(y)} observations. The study resolved folds against a "
                f"different row set than evaluate() built."
            )
        for name, got, want in (
            ("participants", participants, self._expected_participants),
            ("items", items, self._expected_items),
        ):
            if not np.array_equal(np.asarray(got), want):
                n_diff = int(np.sum(np.asarray(got) != want))
                raise ValueError(
                    f"PrecomputedFoldSplitter: {name} sequence differs from the "
                    f"order the folds were resolved against ({n_diff} rows). "
                    f"The flat-array order changed; re-resolve the fold map."
                )
        for fold in np.unique(self.fold_ids):
            test_mask = self.fold_ids == fold
            yield ~test_mask, test_mask


class Hopton2026Unigram(Study):
    """Hopton et al. (2026b), unigram-surprisal estimation and reading time."""

    name = "hopton2026unigram"
    bibtex = r"""
    @misc{hopton2026unigram,
      title = {Revisiting the Estimation of Unigram Surprisal in Computational Psycholinguistics},
      author = {Hopton, Zachary William and Re, Francesco Ignazio and Kiegeland, Samuel and Opedal, Andreas and Chodroff, Eleanor and Cotterell, Ryan},
      year = {2026},
      archivePrefix = {arXiv},
      eprint = {2610.18414},
      url = {https://arxiv.org/abs/2610.18414}
    }
    """
    dependencies = repo_dependencies("context-reading-time")

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def _load_language(
        self,
        language: str = "en",
        *,
        measure: str = "gaze_rt",
        whitespace: bool = True,
        source: str = "committed",
        data_root: Path | str | None = None,
        estimator: str = "reference",
    ) -> pd.DataFrame:
        """One language's per-participant frame, prepared as the R port does.

        ``source="committed"`` joins the pinned per-unit CSV to the archived
        per-participant reading times and MMC estimates. ``source="meco"`` is
        reserved for deriving the reading times from REPLICA's MECO loader; it
        needs a reconciliation between MECO v2.1 (what the loader ships) and
        v1.2 plus the authors' trial-identity fixes (what the pinned CSVs were
        built from), so it raises ``NotImplementedError``.
        """
        if source == "meco":
            raise NotImplementedError(
                "source='meco' needs a MECO v2.1 / v1.2 reconciliation; "
                "the indexed study inputs are the reference path."
            )
        if source != "committed":
            raise ValueError(f"source must be 'committed' or 'meco', got {source!r}")
        if language not in LANGUAGES:
            raise ValueError(f"language must be one of {LANGUAGES}, got {language!r}")
        if measure not in MEASURES:
            raise ValueError(f"measure must be one of {MEASURES}, got {measure!r}")

        if estimator not in ESTIMATORS:
            raise ValueError(
                f"estimator must be one of {ESTIMATORS}, got {estimator!r}"
            )
        if estimator != "reference" and (measure != "gaze_rt" or not whitespace):
            raise ValueError(
                "Paper estimator inputs support gaze_rt with whitespace=True"
            )

        self.ensure_data(data_root)
        estimate_col = (
            "nmc_surp" if estimator == "nmc" else _ESTIMATE_COLUMNS[bool(whitespace)]
        )

        per_unit = pd.read_csv(data_dir(data_root) / f"{language}.csv")
        rt = _input_table("rt_by_participant")
        est = _input_table(
            "mmc_estimates" if estimator == "reference" else f"{estimator}_estimates"
        )
        folds = _input_table(
            "folds" if estimator == "reference" else f"folds_{estimator}"
        )

        rt = rt[rt["lang"] == language].drop(columns=["lang"])
        est = est[est["lang"] == language].drop(columns=["lang"])
        folds = folds[
            (folds["lang"] == language) & (folds["psychometric"] == measure)
        ].drop(columns=["lang", "psychometric"])

        df = (
            rt.merge(
                per_unit.drop(
                    columns=[c for c in ("lang", *MEASURES) if c in per_unit.columns]
                ),
                on=["trialid", "ianum"],
                how="inner",
            )
            .merge(
                est,
                on=["trialid", "ianum", "model_ia"],
                how="left",
                validate="many_to_one",
            )
            .rename(columns={estimate_col: "_mmc"})
            # Both whitespace variants ship in one table; drop the one this
            # call did not select so the frame carries a single estimate.
            .drop(
                columns=[c for c in _ESTIMATE_COLUMNS.values() if c != estimate_col],
                errors="ignore",
            )
        )
        df = self._prepare(df)
        df = df[df["model"] == MODEL_NAME]

        # Restrict to the rows the R port fit: the intersection of the two
        # conditions' complete cases, materialized as the fold map's obs_ids.
        # Pre-filtering here means evaluate()'s own NaN masking is a no-op, so
        # the flat rows and the archived folds describe the same row set.
        required = [
            measure,
            "surp",
            "prev_surp",
            "freq",
            "prev_freq",
            "pmi",
            "prev_pmi",
            "freq_BW",
            "prev_freq_BW",
            "pmi_BW",
            "prev_pmi_BW",
        ]
        complete = df.loc[np.isfinite(df[required].to_numpy(dtype=float)).all(axis=1)]
        if set(complete.obs_id) != set(folds.obs_id):
            raise ValueError(
                "The estimator's complete cases do not match its archived fold map"
            )
        df = complete.merge(folds, on="obs_id", how="inner", validate="one_to_one")
        return df.reset_index(drop=True)

    @staticmethod
    def _prepare(df: pd.DataFrame) -> pd.DataFrame:
        """Port of ``prepare_dataset`` in the R reference.

        ``obs_id`` retains the author's row identities, since the R original
        assigns it by row number over the raw per-participant CSV, before the
        ``prev_freq > 0`` filter. Everything else follows the R order: the
        zero-frequency floor is taken over the raw positive frequencies, and
        the spillover MMC column is a within-trial lag rather than a stored
        column, which is why the two conditions derive spillover differently.
        """
        lowest_freq = df.loc[df["freq"] > 0, "freq"].min()
        df = df[df["prev_freq"] > 0].copy()
        df["freq"] = -np.log2(df["freq"].where(df["freq"] != 0, lowest_freq))
        df["prev_freq"] = -np.log2(df["prev_freq"])

        df = df.sort_values(
            ["subid", "model", "trialid", "ianum", "obs_id"], kind="mergesort"
        )
        grouped = df.groupby(["subid", "model", "trialid"], sort=True, dropna=False)
        df["freq_BW"] = df["_mmc"]
        df["prev_freq_BW"] = grouped["_mmc"].shift(1)

        df["pmi"] = df["freq"] - df["surp"]
        df["prev_pmi"] = df["prev_freq"] - df["prev_surp"]
        df["pmi_BW"] = df["freq_BW"] - df["surp"]
        df["prev_pmi_BW"] = df["prev_freq_BW"] - df["prev_surp"]
        return df.drop(columns=[c for c in df.columns if c.startswith("prev2")])

    # ------------------------------------------------------------------
    # Study interface
    # ------------------------------------------------------------------

    def dataset(
        self,
        language: str = "en",
        measure: str = "gaze_rt",
        **kwargs,
    ) -> TabularDataset:
        df = self._load_language(language, measure=measure, **kwargs)
        dataset = TabularDataset(
            df,
            TabularDatasetConfig(
                name=f"MECO-{language}"
                + (
                    f"/{kwargs['estimator']}"
                    if kwargs.get("estimator", "reference") != "reference"
                    else ""
                ),
                language=language,
                measure_col=measure,
                participant_col="subid",
            ),
        )
        columns = [
            "surp",
            "prev_surp",
            *(
                column
                for mapping in _CONDITION_COLUMNS.values()
                for column in mapping.values()
            ),
        ]
        observations = df.assign(subid=df.subid.astype(str)).set_index(
            ["trialid", "subid", "ianum"]
        )
        if not observations.index.is_unique:
            raise ValueError("Expected one observation per participant, trial and word")
        unit_orders = {
            trial: np.sort(rows.ianum.unique()) for trial, rows in df.groupby("trialid")
        }
        for datum in dataset.data(measure=measure):
            trial = datum.stimulus.metadata["trial_id"]
            rows = observations.xs(
                (trial, datum.participant_id), level=("trialid", "subid")
            ).reindex(unit_orders[trial])
            # Spillover values can differ between participants at the same word.
            for column in columns:
                dataset.add_datum(
                    replace(
                        datum,
                        measure_name=column,
                        values=rows[column].to_numpy(dtype=float),
                    )
                )
        return dataset

    def evaluation_spec(
        self,
        language: str = "en",
        condition: str = "RB_Estimate",
        comp: str = "surp+freq",
        measure: str = "gaze_rt",
        *,
        dataset,
        **kwargs,
    ) -> EvaluationSpec:
        """Predictors for one comparison against the fitted random-effects baseline."""
        if condition not in CONDITIONS:
            raise ValueError(
                f"condition must be one of {CONDITIONS}, got {condition!r}"
            )
        if comp not in COMPS:
            raise ValueError(f"comp must be one of {COMPS}, got {comp!r}")

        cols = _CONDITION_COLUMNS[condition]
        surp = dataset.feature("surp")
        prev_surp = dataset.feature("prev_surp")
        freq = dataset.feature(cols["freq"], name="freq")
        prev_freq = dataset.feature(cols["prev_freq"], name="prev_freq")
        pmi = dataset.feature(cols["pmi"], name="pmi")
        prev_pmi = dataset.feature(cols["prev_pmi"], name="prev_pmi")

        predictors = {
            "surp": [surp, prev_surp],
            "freq": [freq, prev_freq],
            "pmi": [pmi, prev_pmi],
            "surp+freq": [surp, prev_surp, freq, prev_freq],
            "pmi+freq": [pmi, prev_pmi, freq, prev_freq],
        }[comp]

        return EvaluationSpec(
            predictors=predictors,
            controls=[],
            targets=[dataset.feature(measure)],
        )

    def analysis_config(
        self,
        language: str = "en",
        measure: str = "gaze_rt",
        **kwargs,
    ) -> AnalysisConfig:
        """GAMM config replaying R's folds.

        The splitter needs the flat-row order, so this rebuilds the dataset to
        resolve it. Both calls hit the cached parquet reads.
        """
        df = self._load_language(language, measure=measure, **kwargs)
        ds = self.dataset(language, measure=measure, **kwargs)
        fold_ids, participants, items = _resolve_flat_folds(ds, df, measure)

        return AnalysisConfig(
            design=None,
            model=RGamModel(
                smooth_all=True,
                smooth_k=6,
                use_bam=True,
                discrete=True,
                prediction_discrete=True,
                method="fREML",
                n_threads=4,
                re_subject_intercept=True,
                re_item_intercept=True,
                re_predict="conditional",
            ),
            cv=PrecomputedFoldSplitter(
                fold_ids,
                expected_participants=participants,
                expected_items=items,
            ),
            scaler=ZScoreScaler(ddof=1),
            empty_control_baseline="model",
            transform_dv=_identity,
            sigma_source="training",
            per_fold=True,
            retain_per_obs_delta_ll=True,
        )

    def word_scores(
        self,
        result,
        language: str = "en",
        measure: str = "gaze_rt",
        **kwargs,
    ) -> pd.DataFrame:
        """Average retained participant ΔLL within each trial/word, as in the R port."""
        df = self._load_language(language, measure=measure, **kwargs)
        ds = self.dataset(language, measure=measure, **kwargs)
        if result.dataset_name != ds.info.name or result.dv_name != measure:
            raise ValueError("The result belongs to a different dataset or response")
        frame = _flat_frame(ds, df, measure)
        fold_ids, participants, items = _resolve_flat_folds(ds, df, measure)
        expected_cv = PrecomputedFoldSplitter(
            fold_ids, expected_participants=participants, expected_items=items
        ).configuration()
        if result.metadata.get("configuration", {}).get("cv_options") != expected_cv:
            raise ValueError(
                "The result did not use this study's archived fold assignment"
            )
        folds = sorted(frame.fold.unique())
        if result.fold_results is None or len(result.fold_results) != len(folds):
            raise ValueError("Word scores require all archived folds with retained ΔLL")
        scores = np.full(len(frame), np.nan)
        for index, (fold_id, fold) in enumerate(
            zip(folds, result.fold_results, strict=True)
        ):
            mask = frame.fold.to_numpy() == fold_id
            values = fold.per_obs_delta_ll
            if (
                fold.fold_index != index
                or values is None
                or len(values) != mask.sum()
                or fold.n_test != mask.sum()
            ):
                raise ValueError("Retained scores do not match the archived fold order")
            scores[mask] = values
        if not np.isfinite(scores).all():
            raise ValueError("Word scores require finite ΔLL for every observation")
        return (
            frame.assign(dll=scores)
            .groupby(["trialid", "ianum"], as_index=False)
            .dll.mean()
        )


def _flat_frame(ds: TabularDataset, df: pd.DataFrame, measure: str) -> pd.DataFrame:
    """Align archived observation identities and folds to dataset traversal."""
    data_by_sid: dict[str, list] = {}
    for datum in ds.data(measure=measure):
        data_by_sid.setdefault(datum.stimulus.stimulus_id, []).append(datum)
    ianum_by_trial = {
        int(trial): np.sort(group.ianum.unique())
        for trial, group in df.groupby("trialid")
    }
    keys = []
    for stimulus in ds.stimuli():
        trial = int(stimulus.metadata["trial_id"])
        for datum in data_by_sid.get(stimulus.stimulus_id, []):
            for index in np.flatnonzero(np.isfinite(datum.values)):
                keys.append(
                    (
                        datum.participant_id or "",
                        trial,
                        int(ianum_by_trial[trial][index]),
                    )
                )
    columns = ["subid", "trialid", "ianum"]
    expected = df[columns + ["fold"]].copy()
    expected["subid"] = expected.subid.astype(str)
    frame = pd.DataFrame(keys, columns=columns).merge(
        expected, on=columns, how="left", sort=False, validate="one_to_one"
    )
    if frame.fold.isna().any() or len(frame) != len(expected):
        raise ValueError("Dataset observations do not match the archived fold map")
    return frame


def _resolve_flat_folds(
    ds: TabularDataset, df: pd.DataFrame, measure: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fold, participant and trial codes in evaluation order."""
    frame = _flat_frame(ds, df, measure)
    return (
        frame.fold.to_numpy(dtype=int),
        pd.factorize(frame.subid, sort=False)[0],
        pd.factorize(frame.trialid, sort=False)[0],
    )
