"""User-facing discovery and loading helpers for integrated datasets.

The registry remains the source of truth for concrete benchmark keys.  This
module adds a readable description layer over it, including constructor/load
parameters and the conditional variants of dataset families such as MECO.
"""

import inspect
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Literal, cast

from replica.core import resolve_dv
from replica.dataset.base import Dataset, EyeTrackingDataset
from replica.dataset.catalog.registry import GROUPS, REGISTRY, DatasetRegistration
from replica.dataset.eeg.frank2015_erp import EXCLUSIONS as _ERP_EXCLUSIONS
from replica.dataset.eye_tracking.meco import MECO, MECO_RELEASES
from replica.dataset.eye_tracking.meco.releases import L1W2_SITE_LANGUAGES
from replica.dataset.eye_tracking.meco.releases import LANGUAGES as _MECO_LANGUAGES
from replica.dataset.maze.natural_stories_maze import _POST_ERROR_POLICIES
from replica.dataset.measures import with_derived

ParameterStage = Literal["init", "load"]


@dataclass(frozen=True)
class DatasetParameter:
    """One accepted dataset construction or loading parameter."""

    name: str
    stage: ParameterStage
    description: str
    default: Any = None
    required: bool = False
    choices: tuple[Any, ...] = ()
    choice_descriptions: tuple[tuple[Any, str], ...] = ()


@dataclass(frozen=True)
class DatasetVariant:
    """One conditional configuration within a dataset family."""

    name: str
    description: str
    default_subdir: str
    parameters: tuple[DatasetParameter, ...] = ()
    download_keys: tuple[str, ...] = ()
    measures: tuple[str, ...] = ()
    default_measure: str = ""
    presentation: tuple[str, ...] = ()
    fixations: str = ""
    measures_by_representation: tuple[tuple[str, tuple[str, ...]], ...] = ()


@dataclass(frozen=True)
class DatasetDescription:
    """Structured, printable description of a dataset key or group."""

    name: str
    kind: Literal["dataset", "family", "group"]
    loader: str = ""
    paradigm: str = ""
    default_subdir: str = ""
    parameters: tuple[DatasetParameter, ...] = ()
    variants: tuple[DatasetVariant, ...] = ()
    members: tuple[str, ...] = ()
    selected_options: tuple[tuple[str, Any], ...] = ()
    download_keys: tuple[str, ...] = ()
    measures: tuple[str, ...] = ()
    default_measure: str = ""
    presentation: tuple[str, ...] = ()
    fixations: str = ""

    def __str__(self) -> str:
        return format_dataset_description(self)


_SKIP_PARAMETER = DatasetParameter(
    name="drop_skips",
    stage="load",
    description=(
        "Exclude skipped duration observations (default); False includes them "
        "as 0 ms. Set through load_dataset or on the loaded dataset."
    ),
    default=True,
    choices=(True, False),
)


_DESCRIPTIONS: dict[tuple[str, str], str] = {
    ("Dundee", "word_path"): (
        "Original Dundee english/items directory with display and word-object "
        "files; its parent must contain the original MA1 event files."
    ),
    ("GECO", "group"): "Reader group and stimulus language.",
    ("GECO", "first_pass_filter"): (
        "Keep first-pass observations with forward jumps of at most four source "
        "word positions; add wdelta and prevwasfix controls."
    ),
    ("PROVO", "first_pass_filter"): (
        "Keep first-pass observations with forward jumps of at most four aligned "
        "word positions; add wdelta and prevwasfix controls."
    ),
    (
        "RSC",
        "release",
    ): "OSF release: the published 96-participant table or the extended one.",
    ("ZuCo1", "task"): (
        "Reading task: normal reading, sentiment reading or task-specific reading."
    ),
    (
        "ZuCo1",
        "max_subjects",
    ): "Optional subject limit, primarily useful for smoke tests.",
    (
        "ZuCo1",
        "participants",
    ): "Optional subject ids to read, as named in the task files; the rest are left out.",
    (
        "ZuCo1",
        "continuous",
    ): "Also build each sentence's 500 Hz EEG and gaze recordings with fixation-locked word onsets.",
    (
        "ZuCo1",
        "load_eeg",
    ): "Also load EEG frequency-band measures and annotation layers.",
    ("ZuCo1EEG", "task"): (
        "Reading task: normal reading, sentiment reading or task-specific reading."
    ),
    (
        "ZuCo1EEG",
        "max_subjects",
    ): "Optional subject limit, primarily useful for smoke tests.",
    (
        "ZuCo1EEG",
        "participants",
    ): "Optional subject ids to read, as named in the task files; the rest are left out.",
    (
        "ZuCo1EEG",
        "continuous",
    ): "Also build each sentence's 500 Hz EEG and gaze recordings with fixation-locked word onsets.",
    ("ZuCo2", "task"): "Reading task: normal reading or task-specific reading.",
    (
        "ZuCo2",
        "max_subjects",
    ): "Optional subject limit, primarily useful for smoke tests.",
    (
        "ZuCo2",
        "load_eeg",
    ): "Also load EEG frequency-band measures and annotation layers.",
    (
        "ZuCo2",
        "participants",
    ): "Optional subject ids to read, as named in the task files; the rest are left out.",
    (
        "ZuCo2",
        "continuous",
    ): "Also build each sentence's 500 Hz EEG and gaze recordings with fixation-locked word onsets.",
    ("ZuCo2EEG", "task"): "Reading task: normal reading or task-specific reading.",
    (
        "ZuCo2EEG",
        "max_subjects",
    ): "Optional subject limit, primarily useful for smoke tests.",
    (
        "ZuCo2EEG",
        "participants",
    ): "Optional subject ids to read, as named in the task files; the rest are left out.",
    (
        "ZuCo2EEG",
        "continuous",
    ): "Also build each sentence's 500 Hz EEG and gaze recordings with fixation-locked word onsets.",
    (
        "Narratives",
        "story",
    ): "Narratives story identifier; availability depends on local alignment files.",
    (
        "Narratives",
        "derivatives_path",
    ): "Optional path to surface-projected BOLD derivatives.",
    ("Narratives", "align_path"): "Optional path containing story alignment files.",
    ("Narratives", "space"): "Surface-space identifier used by the derivatives.",
    (
        "Narratives",
        "max_subjects",
    ): "Optional subject limit, primarily useful for smoke tests.",
    ("Narratives", "alignment_word_column"): (
        "Alignment CSV column that supplies each word: 0 is the surface "
        "transcript, 1 an aligned token stream that can contain <unk>."
    ),
    ("Narratives", "trim_trs"): (
        "Volumes removed from the start and end of each run; required for "
        "stories other than 21styear."
    ),
    ("Narratives", "participants"): (
        "Subject IDs whose BOLD arrays are read; all subjects stay listed."
    ),
    ("Narratives", "load_arrays"): (
        "Read the BOLD arrays; False lists the cohort without reading them."
    ),
    ("NaturalStoriesMaze", "analysed_sample"): (
        "Keep only the native-English readers above 80% accuracy that the "
        "study analyses; False keeps every recruited reader."
    ),
    ("NaturalStoriesMaze", "post_error"): (
        "Words to discard after a wrong choice; the wrong choice itself is "
        "always discarded."
    ),
    ("EMTeC", "measurement_source"): (
        "Measures computed from the fixation events, or the released summaries."
    ),
    ("PoTeC", "measurement_source"): (
        "Measures computed from the corrected fixation events, or the released "
        "summaries."
    ),
    ("RSC", "exclude_inconsistent_go_past"): (
        "Exclude regression paths shorter than the same observation's first run "
        "from analysis; the released values stay in data()."
    ),
    ("MECOL1W2", "site"): (
        "Upstream collection-site/cohort code; it also selects the language."
    ),
    ("MECO_L1", "language"): "MECO release language code.",
    ("MECO_char", "language"): "MECO release language code.",
    ("DundeeOfficial", "verify"): (
        "Check the source files against the checksums the dataset YAML declares."
    ),
    ("EMTeC", "data_file"): "Reading-measures CSV under the data directory.",
    (
        "NaturalStories",
        "stories_file",
    ): "Tokenized stories file under the data directory.",
    (
        "NaturalStories",
        "rt_file",
    ): "Processed reading-time file under the data directory.",
    ("NaturalStoriesMaze", "rt_file"): (
        "Serialized trial table of the release, under the data directory."
    ),
    ("UCL_ET", "stimuli_file"): "Sentence stimuli file under the data directory.",
    ("UCL_ET", "et_file"): "Per-word reading-time file under the data directory.",
    ("UCL_ET", "fix_file"): (
        "Optional per-fixation file under the data directory; when present it "
        "supplies fixations()."
    ),
    ("UCL_SPR", "stimuli_file"): (
        "File with each sentence's comprehension question and answer, under the "
        "data directory."
    ),
    ("UCL_SPR", "rt_file"): (
        "Per-word reading-time file under the data directory, one row per "
        "subject, sentence and word position."
    ),
    (
        "SzewczykN400",
        "study",
    ): "Source study within the mega-N400 release; required for aggregate='mean'.",
    (
        "SzewczykN400",
        "aggregate",
    ): "Observation grain; the pooled study=None view supports only 'subject'.",
    ("MichaelovN400", "aggregate"): "Observation grain for N400 amplitudes.",
    ("Frank2015ERP", "exclusions"): (
        "Released trial flag that becomes NaN: 'reject' (the paper's analysis "
        "set), 'artefact' (the trials de Varda et al. average over) or 'none'."
    ),
    ("Frank2015ERP", "release_file"): (
        "MATLAB release file under the data directory with the ERP arrays."
    ),
    ("Frank2015ERP", "continuous"): (
        "Also read the continuous EEG, one recording per participant and sentence."
    ),
    ("Frank2015ERP", "participants"): (
        "Optional participants ('1' to '24') whose continuous EEG is read."
    ),
    ("Frank2015ERP", "ucl_path"): (
        "Directory of the UCL release whose stimuli.txt numbers the sentences; "
        "the registered ucl_et location by default."
    ),
}

_ZUCO_BANDS = (
    "eeg_theta_1",
    "eeg_theta_2",
    "eeg_alpha_1",
    "eeg_alpha_2",
    "eeg_beta_1",
    "eeg_beta_2",
    "eeg_gamma_1",
    "eeg_gamma_2",
)

_MEASURE_OVERRIDES: dict[str, tuple[str, ...]] = {
    "SBSAT": ("first_fixation", "first_run", "first_pass", "total_time"),
    "ZuCo1EEG": _ZUCO_BANDS,
    "ZuCo2EEG": _ZUCO_BANDS,
    "Narratives": ("bold",),
}

_FIXATION_STATUS: dict[str, str] = {
    "BSC": "available",
    "CELER": "available when sent_fix.tsv is present",
    "CopCo": "available",
    "DeVarda2023": "unavailable: upstream cells are aggregated",
    "Dundee": "available",
    "DundeeOfficial": "available",
    "EMTeC": "available",
    "GECO": "unavailable: no per-fixation release",
    "MECOL1W2": "lazy/filterable",
    "ZuCo1": "available",
    "MECOL2W1": "lazy/filterable",
    "MECOL2W2": "lazy/filterable",
    "MECO_L1": "lazy/filterable",
    "MECO_char": "available",
    "OneStop": "available",
    "PoTeC": "available",
    "PROVO": "available when the optional fixation file is present",
    "RastrOS": "unavailable: the release contains word-level summaries only",
    "RSC": "unavailable: the fixation reports cover a different recording set",
    "SBSAT": "available",
    "UCL_ET": "available",
    "ZuCo2": "available",
}


#: Fixed choices of loader options, keyed by (loader class, parameter).
_CHOICES: dict[tuple[str, str], tuple[Any, ...]] = {
    ("EMTeC", "measurement_source"): ("fixations", "released"),
    ("PoTeC", "measurement_source"): ("fixations", "released"),
    ("NaturalStoriesMaze", "post_error"): _POST_ERROR_POLICIES,
    ("MECOL1W2", "site"): tuple(L1W2_SITE_LANGUAGES),
    ("MECO_L1", "language"): tuple(_MECO_LANGUAGES),
    ("MECO_char", "language"): tuple(_MECO_LANGUAGES),
    ("Frank2015ERP", "exclusions"): _ERP_EXCLUSIONS,
}


def _choices(loader: type[Dataset], parameter: inspect.Parameter) -> tuple[Any, ...]:
    loader_options: Any = loader
    if (loader.__name__, parameter.name) in _CHOICES:
        return _CHOICES[loader.__name__, parameter.name]
    if parameter.name == "group" and loader.__name__ == "GECO":
        return tuple(loader_options.GROUPS)
    if parameter.name == "task" and loader.__name__ in {
        "ZuCo1",
        "ZuCo1EEG",
        "ZuCo2",
        "ZuCo2EEG",
    }:
        return tuple(loader_options.TASKS)
    if parameter.name == "release" and loader.__name__ == "RSC":
        return tuple(loader_options.RELEASES)
    if parameter.name == "aggregate" and hasattr(loader, "AGGREGATES"):
        return tuple(loader.AGGREGATES)
    if parameter.name == "study" and loader.__name__ == "SzewczykN400":
        return tuple(
            spec.init_kwargs["study"]
            for spec in REGISTRY.values()
            if spec.loader_class == "SzewczykN400" and "study" in spec.init_kwargs
        )
    if isinstance(parameter.default, bool):
        return (False, True)
    return ()


def _parameters(loader: type[Dataset]) -> tuple[DatasetParameter, ...]:
    found: list[DatasetParameter] = []
    seen: set[str] = set()
    for stage, callable_ in (("init", loader), ("load", loader.load)):
        for parameter in inspect.signature(callable_).parameters.values():
            if parameter.name in {"self", "path"} or parameter.name in seen:
                continue
            if parameter.kind in {
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            }:
                continue
            seen.add(parameter.name)
            required = parameter.default is inspect.Parameter.empty
            found.append(
                DatasetParameter(
                    name=parameter.name,
                    stage=cast(ParameterStage, stage),
                    description=_DESCRIPTIONS.get(
                        (loader.__name__, parameter.name),
                        f"Optional {stage} parameter accepted by {loader.__name__}.",
                    ),
                    default=None if required else parameter.default,
                    required=required,
                    choices=_choices(loader, parameter),
                )
            )
    if issubclass(loader, EyeTrackingDataset) and "drop_skips" not in seen:
        found.append(_SKIP_PARAMETER)
    return tuple(found)


def _declared_measures(loader: type[Dataset], dataset: Dataset) -> tuple[str, ...]:
    measures = set(dataset.measures)
    module = import_module(loader.__module__)
    measure_columns = getattr(module, "MEASURE_COLUMNS", None)
    if isinstance(measure_columns, dict):
        measures.update(measure_columns.values())
    measures.update(_MEASURE_OVERRIDES.get(loader.__name__, ()))
    default_measure = getattr(dataset, "default_measure", None)
    if default_measure:
        measure, condition = resolve_dv(default_measure)
        if dataset.default_measure not in measures:
            measures.update([measure] + ([condition] if condition else []))
    return tuple(sorted(with_derived(measures)))


#: Corpora whose loader also reads a second release: Shain's Dundee release
#: aligns against the author-distributed Dundee V.1 items and MA1 event files,
#: and the Frank et al. (2015) ERP release numbers no sentence, so its loader
#: reads the UCL stimuli file to share the UCL sentence ids.
_COMPANION_KEYS: dict[str, tuple[str, ...]] = {
    "dundee": ("dundee", "dundee_official"),
    "frank2015_erp": ("frank2015_erp", "ucl_et"),
}


def _download_keys(name: str) -> tuple[str, ...]:
    from replica.dataset.catalog.sources import DATASETS, resolve_dataset_name

    if name.startswith("meco_char_"):
        return ("meco_char",)
    if name.startswith("meco_l1_w2_"):
        return ("meco_l1_w2", "meco_l1_w2_fixations")
    if name in {"meco_l2_w1", "meco_l2_w2"}:
        return (name, f"{name}_fixations")
    if name.startswith("meco_"):
        return ("meco", "meco_l1_w1_fixations")
    if name in {"zuco1_eeg", "zuco2_eeg"} and name[:-4] in DATASETS:
        return (name[:-4],)
    if name in _COMPANION_KEYS:
        return _COMPANION_KEYS[name]
    if name in DATASETS:
        return (name,)
    # Entries served by a shared download reach it through a fetch alias.
    aliased = resolve_dataset_name(name)
    return (aliased,) if aliased in DATASETS else ()


def _meco_description() -> DatasetDescription:
    languages = tuple(_MECO_LANGUAGES)
    representations = ("word", "character")
    site_labels = tuple(L1W2_SITE_LANGUAGES.items())
    word_measures = tuple(MECO().measures)
    return DatasetDescription(
        name="meco",
        kind="family",
        loader="MECO",
        paradigm="eye-tracking",
        variants=(
            DatasetVariant(
                "l1_w1",
                "Native-language readers, Wave 1.",
                "eye-tracking/original/meco (word) or eye-tracking/processed/meco_char (character)",
                (
                    DatasetParameter(
                        "language",
                        "init",
                        "Native-language code.",
                        default="en",
                        choices=languages,
                    ),
                    DatasetParameter(
                        "representation",
                        "init",
                        "Source word areas or the character/fixation-derived view.",
                        default="word",
                        choices=representations,
                    ),
                ),
                download_keys=("meco", "meco_char", "meco_l1_w1_fixations"),
                measures=(),
                measures_by_representation=(
                    (
                        "word",
                        word_measures,
                    ),
                    ("character", ("right_bounded", "total_time")),
                ),
                default_measure="first_run_in_first_pass (word) / total_time (character)",
                fixations="lazy/filterable for word; available for character",
            ),
            DatasetVariant(
                "l1_w2",
                "Native-language readers, Wave 2, selected by collection site.",
                "eye-tracking/original/meco_l1_w2",
                (
                    DatasetParameter(
                        "site",
                        "init",
                        "Upstream collection-site/cohort code (labels show language).",
                        required=True,
                        choices=tuple(L1W2_SITE_LANGUAGES),
                        choice_descriptions=site_labels,
                    ),
                ),
                download_keys=("meco_l1_w2", "meco_l1_w2_fixations"),
                measures=word_measures,
                default_measure="first_run_in_first_pass",
                fixations="lazy/filterable",
            ),
            DatasetVariant(
                "l2_w1",
                "Second-language readers of fixed English stimuli, Wave 1.",
                "eye-tracking/original/meco_l2_w1",
                download_keys=("meco_l2_w1", "meco_l2_w1_fixations"),
                measures=word_measures,
                default_measure="first_run_in_first_pass",
                fixations="lazy/filterable",
            ),
            DatasetVariant(
                "l2_w2",
                "Second-language readers of fixed English stimuli, Wave 2.",
                "eye-tracking/original/meco_l2_w2",
                download_keys=("meco_l2_w2", "meco_l2_w2_fixations"),
                measures=word_measures,
                default_measure="first_run_in_first_pass",
                fixations="lazy/filterable",
            ),
        ),
        parameters=(
            DatasetParameter(
                "release",
                "init",
                "Reader group and collection wave.",
                default="l1_w1",
                choices=tuple(MECO_RELEASES),
            ),
            _SKIP_PARAMETER,
        ),
        members=tuple(name for name in REGISTRY if name.startswith("meco_")),
    )


def available_datasets(*, include_groups: bool = False) -> list[str]:
    """Return sorted user-facing dataset keys.

    The ``"meco"`` family entry is included alongside concrete registry keys.
    Pass ``include_groups=True`` to append registry group aliases as well.

    Args:
        include_groups: Also list the group aliases that stand for several keys.
    """
    keys = set(REGISTRY)
    keys.add("meco")
    if include_groups:
        keys.update(GROUPS)
    return sorted(keys)


def describe_dataset(name: str) -> DatasetDescription:
    """Describe how to construct and load one dataset, family, or group.

    Args:
        name: A registry key, a group alias, or ``"meco"`` for the MECO family.

    Raises:
        KeyError: When *name* is none of the three.
    """
    if name == "meco":
        return _meco_description()
    if name in GROUPS:
        return DatasetDescription(
            name=name,
            kind="group",
            members=tuple(GROUPS[name]),
        )
    if name not in REGISTRY:
        raise KeyError(
            f"Unknown dataset {name!r}; available datasets: {available_datasets()}. "
            f"Groups: {sorted(GROUPS)}"
        )
    spec = REGISTRY[name]
    loader = spec.loader_cls()
    dataset = loader(**spec.init_kwargs)
    return DatasetDescription(
        name=name,
        kind="dataset",
        loader=loader.__name__,
        paradigm=spec.paradigm,
        default_subdir=spec.default_subdir,
        parameters=_parameters(loader),
        selected_options=tuple(spec.init_kwargs.items()),
        download_keys=_download_keys(name),
        measures=_declared_measures(loader, dataset),
        default_measure=getattr(dataset, "default_measure", "") or "",
        presentation=tuple(dataset.presentation),
        fixations=_FIXATION_STATUS.get(loader.__name__, "not applicable"),
    )


def _meco_spec(options: dict[str, Any]) -> tuple[Dataset, DatasetRegistration]:
    factory_names = set(inspect.signature(MECO).parameters)
    unknown = sorted(set(options) - factory_names)
    if unknown:
        raise TypeError(
            f"Unknown MECO option(s): {unknown}; supported: {sorted(factory_names)}"
        )
    dataset = MECO(**options)
    release = options.get("release", "l1_w1")
    if release == "l1_w1":
        language = options.get("language") or "en"
        prefix = (
            "meco_char_" if options.get("representation") == "character" else "meco_"
        )
        registry_name = f"{prefix}{language}"
    elif release == "l1_w2":
        registry_name = f"meco_l1_w2_{options.get('site')}"
    else:
        registry_name = f"meco_{release}"
    return dataset, REGISTRY[registry_name]


def load_dataset(
    name: str,
    *,
    path: str | Path | None = None,
    repo_root: str | Path | None = None,
    **options: Any,
) -> Dataset:
    """Construct and load a dataset by catalog key.

    Use either ``path`` or ``repo_root``; without either, the registry path
    under ``PSYCHO_DATA_DIR`` is used.

    Args:
        name: A registry key or ``"meco"``; group aliases are rejected.
        path: Directory to read the corpus from, overriding the registry's
            default location.
        repo_root: Checkout to resolve the registry's default subdirectory
            against, in place of ``PSYCHO_DATA_DIR``.
        options: Loader settings, each routed to the constructor or to
            ``load()`` by whichever public signature declares it. A name in
            neither, or one the registry key already pins to another value, is
            rejected. ``drop_skips`` is the exception: it sets
            :attr:`EyeTrackingDataset.drop_skips` on the loaded corpus and is
            refused for a corpus that is not eye-tracking.
    """
    if path is not None and repo_root is not None:
        raise ValueError("Pass either path= or repo_root=, not both")
    drop_skips = options.pop("drop_skips", None)
    if name in GROUPS:
        raise ValueError(
            f"{name!r} is a dataset group, not one dataset; members: {GROUPS[name]}"
        )

    if name == "meco":
        dataset, spec = _meco_spec(options)
        load_options: dict[str, Any] = {}
    else:
        if name not in REGISTRY:
            raise KeyError(
                f"Unknown dataset {name!r}; available: {available_datasets()}"
            )
        spec = REGISTRY[name]
        loader = spec.loader_cls()
        init_signature = inspect.signature(loader)
        load_signature = inspect.signature(loader.load)
        init_names = set(init_signature.parameters)
        load_names = set(load_signature.parameters) - {"self", "path"}
        supported = init_names | load_names
        unknown = sorted(set(options) - supported)
        if unknown:
            raise TypeError(
                f"Unknown option(s) for {name!r}: {unknown}; "
                f"supported: {sorted(supported)}"
            )
        for key, pinned_value in spec.init_kwargs.items():
            if key in options and options[key] != pinned_value:
                raise ValueError(
                    f"Dataset key {name!r} fixes {key}={pinned_value!r}; "
                    "choose the corresponding registry key or the 'meco' family"
                )
        init_options = dict(spec.init_kwargs)
        init_options.update(
            {key: value for key, value in options.items() if key in init_names}
        )
        load_options = {
            key: value for key, value in options.items() if key in load_names
        }
        dataset = loader(**init_options)

    if path is None:
        root = Path(repo_root) if repo_root is not None else None
        resolved_path = spec.default_path(root)
    else:
        resolved_path = Path(path)
    dataset.load(str(resolved_path), **load_options)
    if drop_skips is not None:
        if not isinstance(dataset, EyeTrackingDataset):
            raise TypeError(
                f"{name!r} is not an eye-tracking corpus; drop_skips applies to "
                "eye-tracking durations only"
            )
        dataset.drop_skips = bool(drop_skips)
    return dataset


def _format_parameter(parameter: DatasetParameter, indent: str = "  ") -> list[str]:
    if parameter.required:
        default = "required"
    else:
        default = f"default {parameter.default!r}"
    line = f"{indent}{parameter.name} [{parameter.stage}; {default}]"
    if parameter.choices:
        labels = dict(parameter.choice_descriptions)
        rendered = [
            f"{choice!r} ({labels[choice]})" if choice in labels else repr(choice)
            for choice in parameter.choices
        ]
        line += f": {', '.join(rendered)}"
    return [line, f"{indent}  {parameter.description}"]


def _format_variant(variant: DatasetVariant) -> list[str]:
    """Describe one release and its representation-specific measurements."""
    lines = [f"  {variant.name}: {variant.description}"]
    lines.append(f"    data: {variant.default_subdir}")
    lines.append(f"    download: {', '.join(variant.download_keys)}")
    if variant.measures:
        lines.append(f"    measures: {', '.join(variant.measures)}")
    for representation, measures in variant.measures_by_representation:
        lines.append(f"    {representation} measures: {', '.join(measures)}")
    lines.append(f"    default measure: {variant.default_measure}")
    lines.append(f"    fixations: {variant.fixations}")
    for parameter in variant.parameters:
        lines.extend(_format_parameter(parameter, indent="    "))
    return lines


def format_dataset_description(description: DatasetDescription) -> str:
    """Render a :class:`DatasetDescription` for terminals and notebooks.

    Args:
        description: The description to lay out as plain text.

    Returns:
        A newline-joined block; empty fields are omitted.
    """
    lines = [f"{description.kind.title()}: {description.name}"]
    if description.loader:
        lines.append(f"Loader: {description.loader}")
    if description.paradigm:
        lines.append(f"Paradigm: {description.paradigm}")
    if description.presentation:
        lines.append(f"Presentation: {', '.join(description.presentation)}")
    if description.default_subdir:
        lines.append(f"Default data path: {description.default_subdir}")
    if description.kind == "dataset":
        download = ", ".join(description.download_keys) or "not managed by the fetcher"
        lines.append(f"Download key(s): {download}")
    if description.measures:
        lines.append(f"Measures: {', '.join(description.measures)}")
    if description.default_measure:
        lines.append(f"Default measure: {description.default_measure}")
    if description.fixations:
        lines.append(f"Fixations: {description.fixations}")
    if description.selected_options:
        selected = ", ".join(
            f"{key}={value!r}" for key, value in description.selected_options
        )
        lines.append(f"Selected options: {selected}")
    if description.parameters:
        lines.append("Options:")
        for parameter in description.parameters:
            lines.extend(_format_parameter(parameter))
    if description.variants:
        lines.append("Variants:")
        for variant in description.variants:
            lines.extend(_format_variant(variant))
    if description.members:
        lines.append(
            f"Members ({len(description.members)}): {', '.join(description.members)}"
        )
    return "\n".join(lines)


def print_dataset_description(name: str) -> None:
    """Print the catalog description for ``name``.

    Args:
        name: A registry key, a group alias, or ``"meco"``.
    """
    print(describe_dataset(name))
