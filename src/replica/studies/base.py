"""Base classes for per-paper study recipes.

Each :class:`Study` bundles data dependencies, dataset loading,
feature extraction, and analysis config. The three components are
independently usable::

    ds = rep.dataset(...)
    spec = rep.evaluation_spec(dataset=ds, ...)
    config = rep.analysis_config()
    results = evaluate(ds, spec, config)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from ..external import (
    DataDependency,
    DependencyMismatchError,
    fetch_dependency,
    resolve_data_root,
    verify_dependency,
)
from ..utils.bibtex import bibtex_field as _bibtex_field
from ..utils.bibtex import clean_tex as _clean_tex

if TYPE_CHECKING:
    import pandas as pd

    from replica.analysis.specification.config import AnalysisConfig, EncodingConfig
    from replica.analysis.specification.evaluation import EvaluationSpec

    from ..dataset import Dataset
    from ..feature import PrecomputedFeatureModel

logger = logging.getLogger("replica")

#: Dependencies verified in this process, keyed by resolved path and pin.
_VERIFIED_DEPENDENCIES: set[tuple[str, str, str | None, str | None]] = set()


@dataclass(frozen=True)
class TabularRecipe:
    """Declarative spec for a single-``TabularDataset`` study.

    A :class:`TabularStudy` builds ``dataset()`` and ``evaluation_spec()``
    from this; the paper supplies only ``load_df()`` for its source-specific
    dataframe munging. The generated ``evaluation_spec()`` still accepts per-call
    ``predictors``/``controls`` overrides (defaulting to the recipe's tuples).

    Args:
        dataset_name: ``TabularDatasetConfig.name``.
        measure_col: DV column (also ``TabularDatasetConfig.measure_col``).
        predictors: Default predictor column names.
        controls: Default control column names.
        language: Dataset language tag.
        model_name: ``PrecomputedFeatureModel`` name (defaults to ``dataset_name``).
    """

    dataset_name: str
    measure_col: str
    predictors: tuple[str, ...]
    controls: tuple[str, ...] = ()
    language: str = "en"
    model_name: str | None = None


bibtex_field = _bibtex_field
clean_tex = _clean_tex


class Study(ABC):
    """Base class for per-paper study modules.

    A study module packages one paper's recipe (data dependencies, dataset,
    features, analysis config). It anchors that paper's reproduction tests
    (re-obtaining the paper's numbers from its own data and recipe) and is
    reused by the grid for replication (additional datasets and metrics) and
    robustness checks (one analysis decision varied at a time).

    Subclasses define ``name``, ``bibtex``, ``dependencies``, and the three
    abstract methods :meth:`dataset`, :meth:`evaluation_spec`, :meth:`analysis_config`.
    """

    name: str
    bibtex: str  # The source of truth for all citation info
    dependencies: list[DataDependency]

    @property
    def paper(self) -> str:
        """Human-readable citation derived from bibtex."""
        authors = bibtex_field(self.bibtex, "author") or "Unknown"
        title = bibtex_field(self.bibtex, "title") or "Unknown"
        year = bibtex_field(self.bibtex, "year") or "????"

        authors = clean_tex(authors)
        title = clean_tex(title)
        year = year.strip().strip(",").strip('"').strip("'")

        return f"{authors} ({year}). {title}"

    @abstractmethod
    def dataset(self, **kwargs) -> Dataset:
        """Load the paper's dataset.

        Args:
            kwargs: Cell selectors the concrete override defines, typically a
                corpus or language key and a ``data_root``; this signature only
                reserves the slot.
        """
        ...

    @abstractmethod
    def evaluation_spec(self, *, dataset: Dataset, **kwargs) -> EvaluationSpec:
        """Bind the paper's features and targets to loaded observations.

        Args:
            dataset: Dataset supplying the target observations.
            kwargs: Predictor and control selectors the concrete override
                defines, typically the language model, the reading measure and
                a ``data_root``; this signature only reserves the slot.
        """
        ...

    @abstractmethod
    def analysis_config(self) -> AnalysisConfig | EncodingConfig:
        """Return the paper's analysis configuration."""
        ...

    def _resolve_root(self, data_root: Path | str | None = None) -> Path:
        """Resolves ``dep.path`` against: arg → ``$PSYCHO_REPL_DATA`` → repo root."""
        return resolve_data_root(data_root)

    def ensure_data(self, data_root: Path | str | None = None) -> None:
        """Fetch missing dependencies and verify their declared identities.

        Args:
            data_root: Where the dependencies live; ``None`` resolves through
                :func:`resolve_data_root`.

        Raises:
            ValueError: A present file's digest differs from its pinned one.
        """
        root = self._resolve_root(data_root)
        for dep in self.dependencies:
            full_path = root / dep.path
            if not full_path.exists():
                logger.info(
                    "Study %s: fetching %s from %s",
                    self.name,
                    dep.path,
                    dep.source,
                )
                fetch_dependency(dep, root)
            # A pinned Git tree is verified by reading it into a scratch index,
            # seconds per call on a network file system; a study loads its data
            # once per cell, so each dependency is verified once per process.
            key = (str(full_path.resolve()), dep.source, dep.commit, dep.checksum)
            if key in _VERIFIED_DEPENDENCIES:
                continue
            verify_dependency(dep, root)
            _VERIFIED_DEPENDENCIES.add(key)

    def verify(self, data_root: Path | str | None = None) -> bool:
        """Check file checksums and pinned Git trees, including declared patches.

        Fetches nothing, so it is safe to call where the data may be absent.

        Args:
            data_root: Where the dependencies live; ``None`` resolves through
                :func:`resolve_data_root`.
        """
        root = self._resolve_root(data_root)
        try:
            for dep in self.dependencies:
                key = (
                    str((root / dep.path).resolve()),
                    dep.source,
                    dep.commit,
                    dep.checksum,
                )
                if key in _VERIFIED_DEPENDENCIES:
                    continue
                verify_dependency(dep, root)
                _VERIFIED_DEPENDENCIES.add(key)
        except (FileNotFoundError, DependencyMismatchError):
            return False
        return True


def build_tabular_model(
    dataset: Dataset,
    df: pd.DataFrame,
    columns: list[str],
    model_name: str,
) -> PrecomputedFeatureModel:
    """PrecomputedFeatureModel for a TabularDataset.

    One stimulus per DataFrame row; stimulus ids are ``"trial_{row_index}"``.

    Args:
        dataset: A loaded TabularDataset.
        df: The DataFrame used to build the dataset.
        columns: Column names to include as feature values.
        model_name: Name for the PrecomputedFeatureModel.
    """
    from ..feature import (
        FeatureModelOutput,
        PrecomputedFeatureModel,
    )

    outputs: dict[str, list[FeatureModelOutput]] = {}
    for stim in dataset.stimuli():
        sid = stim.stimulus_id
        row_idx = int(sid.replace("trial_", ""))
        vals = {c: float(df.iloc[row_idx][c]) for c in columns}
        outputs[sid] = [FeatureModelOutput(values=vals)]
    return PrecomputedFeatureModel(model_name, outputs)


def build_grouped_model(
    df: pd.DataFrame,
    groupby_col: str,
    sort_col: str,
    columns: list[str],
    model_name: str,
    *,
    sort: bool = True,
) -> PrecomputedFeatureModel:
    """PrecomputedFeatureModel where each group becomes one multi-output stimulus.

    Stimulus IDs are ``"trial_{group_key}"``.

    Args:
        df: Long-form table, one row per unit.
        groupby_col: Column whose value identifies a trial, e.g. ``"trialid"``.
        sort_col: Column giving each unit's position in its trial, e.g.
            ``"ianum"``.
        columns: Column names to carry through as feature values.
        model_name: Name the model reports in analysis results.
        sort: If True, iterate groups in sorted-key order and sort rows
            within each group. If False, preserve input row order, useful
            when honouring an externally imposed sequence (e.g. a row order
            shuffled by an R port).
    """
    from ..feature import (
        FeatureModelOutput,
        PrecomputedFeatureModel,
    )

    outputs: dict[str, list[FeatureModelOutput]] = {}
    for group_key, group_df in df.groupby(groupby_col, sort=sort):
        if sort:
            group_df = group_df.sort_values(sort_col)
        sid = f"trial_{group_key}"
        matrix = group_df.loc[:, columns].to_numpy(dtype=float)
        trial_outputs = [
            FeatureModelOutput(values=dict(zip(columns, row_values, strict=True)))
            for row_values in matrix
        ]
        outputs[sid] = trial_outputs
    return PrecomputedFeatureModel(model_name, outputs)


class TabularStudy(Study):
    """A :class:`Study` whose data is one DataFrame-backed TabularDataset.

    Subclasses set the :class:`TabularRecipe` ``recipe`` and implement
    :meth:`load_df` (the paper's source-specific munging); the base generates
    :meth:`dataset` and :meth:`evaluation_spec` from the recipe. ``analysis_config``
    stays abstract, since CV/model choices vary per paper.
    """

    recipe: TabularRecipe

    @abstractmethod
    def load_df(self, data_root: Path | str | None = None) -> pd.DataFrame:
        """Return the analysis-ready DataFrame (one row per observation).

        Args:
            data_root: Where the study's dependencies live; ``None`` falls
                back to ``$PSYCHO_REPL_DATA``, then the repository root.
        """
        ...

    def _tabular_config(self):
        from ..dataset.base import TabularDatasetConfig

        return TabularDatasetConfig(
            name=self.recipe.dataset_name,
            language=self.recipe.language,
            measure_col=self.recipe.measure_col,
        )

    def dataset(self, data_root: Path | str | None = None, **kwargs) -> Dataset:
        """Wrap :meth:`load_df` in the ``TabularDataset`` the recipe describes.

        Accepts no option beyond ``data_root``; any other keyword argument
        raises, so a misspelled option fails rather than being silently
        ignored.

        Args:
            data_root: Where the study's dependencies live; ``None`` falls
                back to ``$PSYCHO_REPL_DATA``, then the repository root.
            kwargs: Collects any other keyword so it can be rejected; nothing
                here is forwarded to the ``TabularDataset``.

        Raises:
            TypeError: A keyword argument other than ``data_root`` was passed.
        """
        from ..dataset.base import TabularDataset

        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"unsupported dataset() option(s): {unexpected}")
        return TabularDataset(self.load_df(data_root), self._tabular_config())

    def evaluation_spec(
        self,
        predictors: list[str] | None = None,
        controls: list[str] | None = None,
        data_root: Path | str | None = None,
        *,
        dataset,
        **kwargs,
    ) -> EvaluationSpec:
        """Build the recipe's spec, with the DV read straight off the table.

        Accepts no option beyond the three below; any other keyword argument
        raises, so a misspelled option fails rather than being silently
        ignored.

        Args:
            dataset: Loaded dataset supplying the target observations.
            predictors: Column names to score; ``None`` takes the recipe's own.
            controls: Column names for the baseline; ``None`` takes the
                recipe's own.
            data_root: Where the study's dependencies live; ``None`` falls
                back to ``$PSYCHO_REPL_DATA``, then the repository root.
            kwargs: Collects any other keyword so it can be rejected; nothing
                here reaches the recipe or the returned ``EvaluationSpec``.

        Raises:
            TypeError: A keyword argument other than the three above was passed.
        """
        from replica.analysis.specification.evaluation import EvaluationSpec
        from replica.feature import ModelFeature

        from ..feature.extractors import PrecomputedFeature

        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"unsupported evaluation_spec() option(s): {unexpected}")

        r = self.recipe
        preds = list(r.predictors if predictors is None else predictors)
        ctrls = list(r.controls if controls is None else controls)
        config = self._tabular_config()
        fm = build_grouped_model(
            self.load_df(data_root),
            config.trial_col,
            config.unit_index_col,
            [*ctrls, *preds],
            r.model_name or r.dataset_name,
        )
        return EvaluationSpec(
            predictors=[ModelFeature(fm, PrecomputedFeature(p, p)) for p in preds],
            controls=[ModelFeature(fm, PrecomputedFeature(c, c)) for c in ctrls],
            targets=[dataset.feature(r.measure_col)],
        )
