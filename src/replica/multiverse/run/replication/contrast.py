"""Scoring one focal contrast per corpus into the shared master schema.

A study declares a :class:`ContrastRecipe` in its own ``replication`` module:
the focal predictors, the baseline design, the analysis config, and whether the
fit needs long-form rows. A study whose claim is not a nested contrast
implements ``run_cell`` directly.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path

from replica.analysis.evaluation.evaluate import evaluate
from replica.analysis.specification.config import AnalysisConfig
from replica.dataset.catalog.registry import REGISTRY

from ...results.analysis_results import row_from_analysis
from ...spec import DatasetCell
from ..aggregation import aggregate_over_participants
from ..features import Predictor
from .contracts import IncompleteCellError
from .sources import resolve_predictor_source

log = logging.getLogger("replication")


@dataclass(frozen=True)
class Focal:
    """The contrast under test: its predictor key, definition, and lag."""

    key: str
    predictor: Predictor
    lag: int = 0

    @property
    def is_surprisal(self) -> bool:
        """True when the focal predictor is surprisal itself."""
        return self.key == "surprisal"


@dataclass(frozen=True)
class ContrastRecipe:
    """One study's nested-contrast design.

    ``stage`` keys the row series in every saved master table and in the claim
    registry. ``predictors`` maps the study's own keys to predictor definitions.
    ``build_model`` assembles the caller-selected predictor sources. Optional
    ``model_aliases`` maps application model keys to canonical identifiers.
    ``summarize`` optionally replaces descriptive row formatting with the caller's
    inference and output policy, using the ``row_from_analysis`` call signature.
    ``focal`` selects those keys, optionally suffixed
    ``@lag<k>`` for the same focal entered k words back.
    """

    stage: str
    focal: tuple[str, ...]
    build_spec: Callable[..., object]
    config: Callable[[], AnalysisConfig]
    predictors: Mapping[str, Predictor]
    build_model: Callable[..., object]
    aggregate: bool = True
    model_aliases: Mapping[str, str] | None = None
    summarize: Callable[..., dict] | None = None


def paper_faithful(base: AnalysisConfig) -> AnalysisConfig:
    """A study's own config with no outlier filter, keeping per-observation delta-LL.

    Both sign-flip regimes, per fold and per datapoint, need those values on
    every row.

    Args:
        base: The study's own analysis config, left unchanged; a modified copy
            is returned.
    """
    return replace(
        base,
        per_fold=True,
        retain_per_obs_delta_ll=True,
        outlier_floor=None,
        outlier_ceiling=None,
        outlier_method=None,
        dump_merged_path=None,
    )


def _prepared_dataset(ds: DatasetCell, *, aggregate: bool):
    """The corpus, shaped the way the recipe's regression needs it.

    Aggregating means over fixators per (stimulus, word), matching the reader-
    averaged inputs an OLS paper regresses on. A per-subject random effect needs
    the long form instead, plus the per-word sentence ids its splitter reads.
    """
    dataset = REGISTRY[ds.name].factory()
    if aggregate:
        aggregate_over_participants(dataset, measure=ds.dv, strategy="excl")
    else:
        from replica.analysis.cv import populate_sentence_ids

        populate_sentence_ids(dataset)
    return dataset


def run_contrast_cells(
    recipe: ContrastRecipe,
    ds: DatasetCell,
    lm: str,
    *,
    cdir: Path,
    select: tuple[str, ...] | None = None,
) -> list[dict]:
    """Master-schema rows for one (corpus, LM), one per focal predictor.

    A cell whose sources are absent is skipped with a logged reason, so one
    missing pickle cannot end a sweep. A focal that fails to fit or score is
    logged and the remaining focals still run.

    Args:
        recipe: The study's contrast design: focal keys, spec builder, config.
        ds: Corpus cell naming the registry dataset and its dependent variable.
        lm: Language-model tag whose predictor sources are used.
        cdir: Compute directory the predictor sources resolve against.
        select: Focal keys to keep, for sharding; ``None`` runs every focal the
            recipe declares.

    Raises:
        IncompleteCellError: At least one focal failed; it carries the rows of
            the focals that completed.
    """
    rows: list[dict] = []
    errors: list[str] = []
    keys = [k for k in recipe.focal if select is None or k in select]
    for key in keys:
        base_key, _, lag_text = key.partition("@lag")
        focal_lag = int(lag_text) if lag_text else 0
        if (
            resolve_predictor_source(
                cdir,
                lm,
                recipe.predictors[base_key].pickle_suffix,
                ds.name,
                aliases=recipe.model_aliases,
            )
            is None
        ):
            log.warning(
                "  [skip] %s/%s/%s/%s: pickle missing", recipe.stage, ds.name, key, lm
            )
            continue
        try:
            model = recipe.build_model(
                recipe.stage,
                lm,
                ds.name,
                cdir=cdir,
                focal=recipe.predictors[base_key],
            )
            if model is None:
                log.warning(
                    "  [skip] %s/%s/%s/%s: a required source is missing",
                    recipe.stage,
                    ds.name,
                    key,
                    lm,
                )
                continue
            dataset = _prepared_dataset(ds, aggregate=recipe.aggregate)
            focal = Focal(base_key, recipe.predictors[base_key], focal_lag)
            spec = recipe.build_spec(model, focal, ds, dataset=dataset)
            analyses = evaluate(dataset, spec, recipe.config())
            if not analyses:
                raise RuntimeError("evaluate produced 0 analyses")
            summarize = (
                recipe.summarize if recipe.summarize is not None else row_from_analysis
            )
            row = summarize(ds, key, recipe.stage, analyses[0], dataset, lm=lm)
        except Exception as exc:
            log.exception("  [cell-fail] %s/%s/%s/%s", recipe.stage, ds.name, key, lm)
            errors.append(f"{key}: {type(exc).__name__}: {exc}")
            continue
        rows.append(row)
        log.info(
            "  %s/%s/%s/%s  dLL=%.4f  p=%.4g  p_obs=%.4g  n=%s",
            recipe.stage,
            ds.name,
            key,
            lm,
            row["delta_ll"] if row["delta_ll"] is not None else float("nan"),
            row["pvalue"] if row["pvalue"] is not None else float("nan"),
            row["pvalue_per_datapoint"]
            if row["pvalue_per_datapoint"] is not None
            else float("nan"),
            row["n_obs"],
        )
    if errors:
        raise IncompleteCellError(rows, errors)
    return rows
