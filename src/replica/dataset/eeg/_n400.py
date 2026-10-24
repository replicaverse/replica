"""Shared loading and covariates for the released N400 amplitude corpora."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from replica.dataset.metadata import io as metadata_io

from ...external import repo_path
from ..base import Datum, EEGDataset


class N400Dataset(EEGDataset):
    """Shared base for the three N400 corpora in the release.

    Holds release-path resolution and the classical N400 predictors as
    covariates. A subclass fixes the corpus and its
    columns and implements :meth:`_build`; instantiate a subclass
    (:class:`ZuCoN400`, :class:`SzewczykN400`, :class:`MichaelovN400`).
    """

    _ANNOTATION = "all.txt.averaged_rt.annotation"
    default_measure = "n400"

    # Subclasses set: corpus directory, and canonical name -> annotation column
    # for the measures this corpus stores and for its item-level covariates.
    _CORPUS: str
    _MEASURE_COLS: dict[str, str]
    _COVARIATE_COLS: dict[str, str]
    AGGREGATES: tuple[str, ...] = ("mean",)

    # One of S_N400's source studies, or None for ZuCo/M_N400 and the pooled S_N400.
    study: str | None = None

    def __init__(self, aggregate: str = "mean"):
        """Fix the grain the corpus is loaded at.

        Args:
            aggregate: Grain to expose, one of the subclass's ``AGGREGATES``.
                ``"mean"`` is the subject-averaged item-level view,
                ``"subject"`` one datum per subject per item, ``"electrode"``
                one datum per electrode. Anything else raises ``ValueError``.
        """
        super().__init__()
        if aggregate not in self.AGGREGATES:
            raise ValueError(
                f"aggregate must be one of {self.AGGREGATES} for "
                f"{type(self).__name__}, got {aggregate!r}"
            )
        self.aggregate = aggregate
        # stimulus_id -> {covariate name: per-word array}; merged in covariates().
        self._covariates: dict[str, dict[str, np.ndarray]] = {}
        self._declared_measures = self._measures()

    def _measures(self) -> tuple[str, ...]:
        """Measures this corpus declares at the current grain."""
        return tuple(self._MEASURE_COLS)

    def _active_covariate_cols(self) -> dict[str, str]:
        """Covariate name -> source column for the current grain."""
        return self._COVARIATE_COLS

    def covariates(
        self, stimulus, target_layer: str = "words"
    ) -> dict[str, np.ndarray]:
        """Standard covariates plus the annotation's item-level N400 predictors.

        The added names are the corpus's cloze, frequency, concreteness,
        neighborhood and plausibility columns, whichever of them it stores.

        Args:
            stimulus: Stimulus whose units the covariates are computed over.
            target_layer: Annotation layer naming the units to align to.
        """
        covs = super().covariates(stimulus, target_layer)
        covs.update(self._covariates.get(stimulus.stimulus_id, {}))
        return covs

    def default_controls(self) -> list[str]:
        """N400 controls, including the participant baseline when released."""
        controls = [
            c
            for c in self._active_covariate_cols()
            if not c.startswith(("p600", "ant"))
        ]
        if "n400_baseline" in self._measures() and "n400_baseline" not in controls:
            controls.append("n400_baseline")
        return controls

    def _annotation_path(self, path: str) -> Path:
        """Locate the release annotation for this corpus.

        Prefers *path* (the ``datasets/eeg/processed`` symlink set up alongside
        the other study-sourced corpora); falls back to the study-dependency
        checkout so the dataset also loads straight from a fetched study
        dependency, before that symlink exists.
        """
        local = Path(path) / self._ANNOTATION
        if local.exists():
            return local
        return (
            repo_path("surprisal_internal_layers", fetch=False)
            / "data"
            / self._CORPUS
            / self._ANNOTATION
        )

    def _input_files(self) -> list[str]:
        """Files read by the selected observation grain."""
        return [self._ANNOTATION]

    def load(self, path: str) -> None:
        """Read the release annotation and build the stimuli and data.

        Rows are restricted to ``self.study`` when the subclass sets one, then
        ordered by article, sentence, and token so every corpus builds in
        reading order.

        Args:
            path: Dataset directory holding the release annotation; the
                study-dependency checkout is the fallback when it is absent.
        """
        ann_path = self._annotation_path(path)
        self._load_metadata(
            path,
            files=self._input_files(),
            data_path=str(ann_path.resolve().parent),
            metadata_path=metadata_io.find_dataset_yaml(path),
        )
        ann = pd.read_csv(ann_path)
        if self.study is not None:
            ann = ann[ann["dataset"] == self.study]
            if ann.empty:
                raise ValueError(f"no rows for study {self.study!r} in {self._CORPUS}")
        ann = ann.sort_values(["article", "sent_id", "tokenN_in_sent"]).reset_index(
            drop=True
        )
        self._build(ann, ann_path)

    def _build(self, ann: pd.DataFrame, ann_path: Path) -> None:
        raise NotImplementedError

    def _item_metadata(self, article, sent_id, grp, condition_col: str | None = None):
        meta = {"article": article, "sent_id": sent_id}
        if self.study is not None:
            meta["study"] = self.study
        col = condition_col or ("Condition" if "Condition" in grp.columns else None)
        if col and col in grp.columns:
            meta["condition"] = str(grp[col].iloc[0])
        return meta

    @staticmethod
    def _on_token(n: int, token: int, value: float) -> np.ndarray:
        """Length-``n`` array with ``value`` on ``token`` and NaN elsewhere."""
        arr = np.full(n, np.nan)
        arr[token] = value
        return arr

    @staticmethod
    def _text_to_position(ann: pd.DataFrame, tokens: dict) -> dict[str, tuple]:
        """Full-sentence text -> (article, sent_id, critical token), so per-subject
        TSV rows join to the surprisal-aligned ``tokens.json`` sentence by text.
        """
        out: dict[str, tuple] = {}
        for (article, sent_id), grp in ann.groupby(["article", "sent_id"]):
            token = int(grp["tokenN_in_sent"].iloc[0])
            text = " ".join(str(w) for w in tokens[str(article)][int(sent_id)])
            out[text] = (str(article), int(sent_id), token)
        return out


class _CriticalWordN400(N400Dataset):
    """Shared build for the critical-word corpora (S_N400, M_N400).

    The N400 sits on one token of the full sentence rebuilt from ``tokens.json``.
    Loads per-subject trials from the release TSV by default, or the
    subject-averaged item-level view with ``aggregate="mean"``.
    """

    # Subclasses set the release TSV and its columns.
    _TSV_FILE: str
    _TSV_SUBJECT: str = "Subject"
    _TSV_MEASURES: dict[str, str]
    _TSV_COVARIATES: dict[str, str]
    _CONDITION_COL: str | None = None
    AGGREGATES: tuple[str, ...] = ("subject", "mean")

    def __init__(self, aggregate: str = "subject"):
        super().__init__(aggregate=aggregate)

    def exclusion_mask(self, datum):
        # The amplitude sits on exactly one token (NaN elsewhere); keep it even
        # when it is sentence-initial. The base EEG mask also drops position 0
        # (no-spillover-baseline), which would delete the whole critical-word datum.
        return ~np.isnan(datum.values)

    def _measures(self) -> tuple[str, ...]:
        # Per subject the TSV carries the ERP measures; the averaged/electrode
        # views expose only the N400.
        return tuple(self._TSV_MEASURES) if self.aggregate == "subject" else ("n400",)

    def _active_covariate_cols(self) -> dict[str, str]:
        return (
            self._COVARIATE_COLS if self.aggregate == "mean" else self._TSV_COVARIATES
        )

    def _input_files(self) -> list[str]:
        """Annotation and text, plus participant data outside the mean view."""
        files = [self._ANNOTATION, "tokens.json"]
        if self.aggregate != "mean":
            files.append(self._TSV_FILE)
        return files

    def _build(self, ann: pd.DataFrame, ann_path: Path) -> None:
        with open(ann_path.resolve().parent / "tokens.json") as f:
            tokens = json.load(f)
        if self.aggregate == "mean":
            self._build_critical_word_mean(ann, tokens)
        else:
            self._build_per_subject(ann_path, ann, tokens)

    def _build_critical_word_mean(self, ann: pd.DataFrame, tokens: dict) -> None:
        """Subject-averaged view: one item-level N400 per stimulus (the M_N400
        electrode cluster mean), with the item-level predictors as covariates.
        """
        col = self._MEASURE_COLS["n400"]
        for (article, sent_id), grp in ann.groupby(["article", "sent_id"]):
            words = [str(w) for w in tokens[str(article)][int(sent_id)]]
            n = len(words)
            token = int(grp["tokenN_in_sent"].iloc[0])
            meta = self._item_metadata(article, sent_id, grp)
            stim = self._make_stimulus(words, f"{article}_{sent_id}", meta)
            amplitude = np.full(n, np.nan)
            amplitude[token] = float(grp[col].mean())  # cluster mean for M_N400
            self._data.append(
                Datum(
                    stimulus=stim,
                    measure_name="n400",
                    values=amplitude,
                    participant_id="averaged",
                ),
            )
            self._covariates[stim.stimulus_id] = {
                name: self._on_token(n, token, float(grp[ccol].mean()))
                for name, ccol in self._COVARIATE_COLS.items()
            }

    def _build_per_subject(
        self, ann_path: Path, ann: pd.DataFrame, tokens: dict
    ) -> None:
        """Per-subject view, one Datum per subject per item with the N400 on the
        critical word. For M_N400, ``aggregate="subject"`` averages the nine
        electrodes into the centroparietal cluster; ``aggregate="electrode"``
        keeps each electrode as its own datum (electrode on ``Datum.metadata``).
        """
        txt2pos = self._text_to_position(ann, tokens)
        tsv = pd.read_csv(ann_path.resolve().parent / self._TSV_FILE, sep="\t")
        covariate_columns = list(self._TSV_COVARIATES.values())
        variation = tsv.groupby("FullSentence")[covariate_columns].nunique(dropna=False)
        conflicting = set(variation.index[variation.max(axis=1) > 1])
        if self.study is not None:
            tsv = tsv[tsv["dataset"] == self.study]
        for key, sgrp in tsv.groupby(
            ["FullSentence", *covariate_columns], dropna=False
        ):
            sentence = key[0]
            pos = txt2pos.get(str(sentence).strip())
            if pos is None:
                continue
            article, sent_id, token = pos
            words = [str(w) for w in tokens[str(article)][int(sent_id)]]
            n = len(words)
            meta = self._item_metadata(article, sent_id, sgrp, self._CONDITION_COL)
            stimulus_id = f"{article}_{sent_id}"
            if sentence in conflicting:
                # Identical text can carry different norms in separate source studies.
                profile = json.dumps([float(value) for value in key[1:]])
                stimulus_id += "_" + hashlib.sha256(profile.encode()).hexdigest()[:12]
            stim = self._make_stimulus(words, stimulus_id, meta)
            self._covariates[stim.stimulus_id] = {
                name: self._on_token(n, token, float(sgrp[col].iloc[0]))
                for name, col in self._TSV_COVARIATES.items()
            }
            for subject, subj_grp in sgrp.groupby(self._TSV_SUBJECT):
                # Study membership per observation, so the pooled mega-N400 can fit
                # a study random effect: an item recurs across studies, so its study
                # is stored on the Datum.
                dmeta = (
                    {"study": str(subj_grp["dataset"].iloc[0])}
                    if "dataset" in subj_grp.columns
                    else {}
                )
                if self.aggregate == "electrode":
                    # keep each electrode as its own datum (the release's M_N400
                    # regression has an electrode random effect); electrode on Datum.metadata
                    # so subject stays the participant grouping.
                    col = self._TSV_MEASURES["n400"]
                    for electrode, e_grp in subj_grp.groupby("Electrode"):
                        self._data.append(
                            Datum(
                                stimulus=stim,
                                measure_name="n400",
                                values=self._on_token(
                                    n, token, float(e_grp[col].mean())
                                ),
                                participant_id=str(subject),
                                metadata={"electrode": str(electrode), **dmeta},
                            )
                        )
                    continue
                for measure, col in self._TSV_MEASURES.items():
                    self._data.append(
                        Datum(
                            stimulus=stim,
                            measure_name=measure,
                            values=self._on_token(
                                n, token, float(subj_grp[col].mean())
                            ),
                            participant_id=str(subject),
                            metadata=dmeta,
                        )
                    )
