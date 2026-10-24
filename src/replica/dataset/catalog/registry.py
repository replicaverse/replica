"""Registry of the integrated datasets.

Each :class:`DatasetRegistration` describes a ``(loader_module, loader_class,
default_subdir, init_kwargs)`` entry; multi-language corpora (MECO L1,
MECO_char) are expanded into one entry per language. ``factory()``
returns a Dataset already loaded against its canonical on-disk path;
consumers that don't want to load can use :meth:`DatasetRegistration.loader_cls`
directly.

``replica.compute`` consumes this registry.
``replica.dataset.catalog.sources`` owns download-source metadata.
"""

import importlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from replica.dataset.eye_tracking.meco.releases import (
    L1W2_SITE_LANGUAGES as _MECO_L1W2_SITES,
)
from replica.dataset.eye_tracking.meco.releases import LANGUAGES as _MECO_LANGS
from replica.dataset.eye_tracking.meco.releases import (
    WORD_RELEASES as _MECO_WORD_RELEASES,
)
from replica.dataset.locations import (
    DATA_DIR_ENV,
    DEVARDA2023_SUBDIR,
    REPO_ROOT,
    UCL_ET_SUBDIR,
    data_dir,
    dataset_path,
)
from replica.paths import user_cache_root

__all__ = [
    "COMPUTE_DIR_ENV",
    "DATA_DIR_ENV",
    "GROUPS",
    "REGISTRY",
    "REPO_ROOT",
    "DatasetRegistration",
    "compute_dir",
    "data_dir",
    "expand",
    "get",
    "names",
]

COMPUTE_DIR_ENV = "PSYCHO_COMPUTE_DIR"


def compute_dir() -> Path:
    """Root for pre-computed predictor caches (surprisal/entropy pickles).

    ``$PSYCHO_COMPUTE_DIR`` if set (``~`` expanded; not created here), else
    ``<repo>/compute`` in a checkout, else ``$XDG_CACHE_HOME/replica/compute``
    (the user cache default when ``XDG_CACHE_HOME`` is unset).
    """
    env = os.environ.get(COMPUTE_DIR_ENV)
    if env:
        return Path(env).expanduser().resolve()
    return REPO_ROOT / "compute" if REPO_ROOT else user_cache_root() / "compute"


@dataclass(frozen=True)
class DatasetRegistration:
    """A single registered dataset."""

    name: str
    """Stable string id used by every consumer (replica.compute, tests)."""

    loader_module: str
    """Importable module path of the loader class, e.g.\
    ``replica.dataset.eye_tracking.provo``."""

    loader_class: str
    """Name of the loader class within that module, e.g.\\ ``"PROVO"``."""

    default_subdir: str
    """Path relative to :func:`data_dir` where data is expected. Used by
    ``factory()`` and by tests when ``--data-root`` is not given."""

    init_kwargs: dict = field(default_factory=dict)
    """Constructor kwargs (e.g.\\ ``{"language": "en"}`` for MECO)."""

    paradigm: str = "eye-tracking"
    """For grouping in the public API (eye-tracking, self-paced-reading, maze,
    eeg, fmri)."""

    default_context_scope: Literal["stimulus", "sentence", "story"] = "stimulus"
    """Default predictor context for methods that support this scope.

    Callers can override it; context-independent methods use stimulus scope.
    """

    def loader_cls(self):
        """Resolve the loader class by importing it lazily."""
        mod = importlib.import_module(self.loader_module)
        return getattr(mod, self.loader_class)

    def default_path(self, repo_root: Path | None = None) -> Path:
        """Resolve the on-disk path for this dataset.

        ``<repo_root>/datasets/<default_subdir>`` when *repo_root* is given (a
        test fixture path), else ``<data_dir()>/<default_subdir>``.

        Args:
            repo_root: Checkout holding a ``datasets`` directory to resolve
                against; ``None`` uses :func:`data_dir`.
        """
        return dataset_path(self.default_subdir, repo_root)

    def factory(self, repo_root: Path | None = None, *, data_root: Path | None = None):
        """Construct + load the dataset against its default path.

        Args:
            repo_root: Checkout holding a ``datasets`` directory to load from;
                ``None`` uses :func:`data_dir`.
            data_root: Explicit raw-data root, mutually exclusive with repo_root.
        """
        cls = self.loader_cls()
        ds = cls(**self.init_kwargs)
        if data_root is not None and repo_root is not None:
            raise ValueError("Specify data_root or repo_root, not both")
        path = (
            Path(data_root) / self.default_subdir
            if data_root is not None
            else self.default_path(repo_root)
        )
        ds.load(str(path))
        return ds


# ============================================================================
# Per-dataset entries
# ============================================================================
#
# Corpora integrated end-to-end (both load and evaluate). Each name is the
# stable id used by tools downstream.

_ET_PREFIX = "replica.dataset.eye_tracking"
_SPR_PREFIX = "replica.dataset.spr"
_EEG_PREFIX = "replica.dataset.eeg"
_FMRI_PREFIX = "replica.dataset.fmri"
_MAZE_PREFIX = "replica.dataset.maze"

_BASE_ENTRIES: list[DatasetRegistration] = [
    # Eye-tracking, English
    DatasetRegistration(
        name="provo",
        loader_module=f"{_ET_PREFIX}.provo",
        loader_class="PROVO",
        default_subdir="eye-tracking/original/provo",
    ),
    DatasetRegistration(
        # Shain's public CSV supplies measures; original Dundee display,
        # word-object and event files are also required for verified alignment.
        name="dundee",
        loader_module=f"{_ET_PREFIX}.dundee.shain2024",
        loader_class="Dundee",
        default_subdir="eye-tracking/processed/dundee_shain2024logarithmic",
    ),
    DatasetRegistration(
        name="dundee_official",
        loader_module=f"{_ET_PREFIX}.dundee.official",
        loader_class="DundeeOfficial",
        default_subdir="eye-tracking/original/dundee",
    ),
    DatasetRegistration(
        name="geco",
        loader_module=f"{_ET_PREFIX}.geco",
        loader_class="GECO",
        default_subdir="eye-tracking/original/geco",
    ),
    DatasetRegistration(
        name="ucl_et",
        loader_module=f"{_ET_PREFIX}.ucl",
        loader_class="UCL_ET",
        default_subdir=UCL_ET_SUBDIR,
    ),
    DatasetRegistration(
        name="emtec",
        loader_module=f"{_ET_PREFIX}.emtec",
        loader_class="EMTeC",
        default_subdir="eye-tracking/original/emtec",
    ),
    DatasetRegistration(
        name="onestop",
        loader_module=f"{_ET_PREFIX}.onestop",
        loader_class="OneStop",
        default_subdir="eye-tracking/original/onestop",
    ),
    DatasetRegistration(
        name="copco",
        loader_module=f"{_ET_PREFIX}.copco",
        loader_class="CopCo",
        default_subdir="eye-tracking/original/copco",
    ),
    DatasetRegistration(
        name="potec",
        loader_module=f"{_ET_PREFIX}.potec",
        loader_class="PoTeC",
        default_subdir="eye-tracking/original/potec",
    ),
    DatasetRegistration(
        name="sbsat",
        loader_module=f"{_ET_PREFIX}.sbsat",
        loader_class="SBSAT",
        default_subdir="eye-tracking/original/sbsat",
        default_context_scope="story",
    ),
    DatasetRegistration(
        name="bsc",
        loader_module=f"{_ET_PREFIX}.bsc",
        loader_class="BSC",
        default_subdir="eye-tracking/original/bsc",
    ),
    DatasetRegistration(
        name="rastros",
        loader_module=f"{_ET_PREFIX}.rastros",
        loader_class="RastrOS",
        default_subdir="eye-tracking/original/rastros",
    ),
    DatasetRegistration(
        # Russian Sentence Corpus. Both OSF releases live in one directory
        # and are selected by the loader's ``release`` argument, so the two
        # entries share a default_subdir (and one fetch key).
        name="rsc",
        loader_module=f"{_ET_PREFIX}.rsc",
        loader_class="RSC",
        default_subdir="eye-tracking/original/rsc",
    ),
    DatasetRegistration(
        name="rsc_extended",
        loader_module=f"{_ET_PREFIX}.rsc",
        loader_class="RSC",
        default_subdir="eye-tracking/original/rsc",
        init_kwargs={"release": "extended"},
    ),
    DatasetRegistration(
        name="meco_l2_w1",
        loader_module=f"{_ET_PREFIX}.meco.waves",
        loader_class="MECOL2W1",
        default_subdir=_MECO_WORD_RELEASES["meco_l2_w1"].data_subdir,
    ),
    DatasetRegistration(
        name="meco_l2_w2",
        loader_module=f"{_ET_PREFIX}.meco.waves",
        loader_class="MECOL2W2",
        default_subdir=_MECO_WORD_RELEASES["meco_l2_w2"].data_subdir,
    ),
    DatasetRegistration(
        # CELER (Berzak et al., 2022). Author-permission only: requires LDC
        # PTB + BLLIP access, so ``replica fetch celer``
        # prints manual download instructions instead of auto-downloading.
        # The loader reads ``sent_ia.tsv`` (the per-IA report) from this path.
        name="celer",
        loader_module=f"{_ET_PREFIX}.celer",
        loader_class="CELER",
        default_subdir="eye-tracking/original/celer",
    ),
    DatasetRegistration(
        # The de Varda et al. (2023) "aligned" corpus is itself a paper-specific aggregation
        # of Frank et al. (2013, 2015) data, so it lives under processed/.
        name="devarda2023",
        loader_module=f"{_ET_PREFIX}.devarda2023",
        loader_class="DeVarda2023",
        default_subdir=DEVARDA2023_SUBDIR,
    ),
    DatasetRegistration(
        # ZuCo 1.0: simultaneous eye-tracking + EEG during natural reading; the
        # eye-tracking view (MATLAB v5 .mat). The EEG band-power view is
        # registered as zuco1_eeg and the N400/P600 ERP view as zuco_n400.
        name="zuco1",
        loader_module=f"{_ET_PREFIX}.zuco.zuco1",
        loader_class="ZuCo1",
        default_subdir="eye-tracking/original/zuco1",
    ),
    DatasetRegistration(
        # ZuCo 2.0: simultaneous eye-tracking + EEG during natural reading; the
        # eye-tracking view. The EEG band-power view is registered as zuco2_eeg.
        name="zuco2",
        loader_module=f"{_ET_PREFIX}.zuco.zuco2",
        loader_class="ZuCo2",
        default_subdir="eye-tracking/original/zuco2",
    ),
    # Maze
    DatasetRegistration(
        name="natural_stories_maze",
        loader_module=f"{_MAZE_PREFIX}.natural_stories_maze",
        loader_class="NaturalStoriesMaze",
        default_subdir="maze/original/natural_stories_maze",
        paradigm="maze",
    ),
    # Self-paced reading
    DatasetRegistration(
        name="natural_stories",
        loader_module=f"{_SPR_PREFIX}.natural_stories",
        loader_class="NaturalStories",
        default_subdir="self-paced-reading/original/natural_stories",
        paradigm="self-paced-reading",
    ),
    DatasetRegistration(
        name="ucl_spr",
        loader_module=f"{_SPR_PREFIX}.ucl",
        loader_class="UCL_SPR",
        default_subdir="self-paced-reading/original/ucl",
        paradigm="self-paced-reading",
    ),
    DatasetRegistration(
        name="brothers_kuperberg",
        loader_module=f"{_SPR_PREFIX}.brothers_kuperberg",
        loader_class="BrothersKuperberg",
        default_subdir="self-paced-reading/original/brothers_kuperberg",
        paradigm="self-paced-reading",
    ),
    DatasetRegistration(
        # Brown SPR has no public archive of the raw Smith & Levy (2013) data;
        # default_subdir points at the Shain et al. (2024) preprocessed release
        # on OSF 6wvqe (under processed/), the only fetchable source.
        name="brown_spr",
        loader_module=f"{_SPR_PREFIX}.brown_spr",
        loader_class="BrownSPR",
        default_subdir="self-paced-reading/processed/brown_spr_shain2024logarithmic",
        paradigm="self-paced-reading",
    ),
    # EEG (N400 / ERP amplitudes). Three corpora repackaged with the Kuribayashi
    # et al. (2025) release; default_subdir points at a symlink into that release.
    DatasetRegistration(
        name="zuco_n400",
        loader_module=f"{_EEG_PREFIX}.zuco.n400",
        loader_class="ZuCoN400",
        default_subdir="eeg/processed/zuco_n400",
        paradigm="eeg",
    ),
    # The mega-N400 pools five distinct published source studies (a
    # ``dataset`` column splits them): registered whole (``szewczyk_n400``, all
    # five together as the release packages them) and one entry per source study.
    DatasetRegistration(
        name="szewczyk_n400",
        loader_module=f"{_EEG_PREFIX}.szewczyk_n400",
        loader_class="SzewczykN400",
        default_subdir="eeg/processed/szewczyk_n400",
        paradigm="eeg",
    ),
    *(
        DatasetRegistration(
            name=_name,
            loader_module=f"{_EEG_PREFIX}.szewczyk_n400",
            loader_class="SzewczykN400",
            default_subdir=f"eeg/processed/{_name}",
            init_kwargs={"study": _study},
            paradigm="eeg",
        )
        for _name, _study in (
            ("federmeier2007_n400", "Federmeier et al. (2007)"),
            ("hubbard2019_n400", "Hubbard et al. (2019)"),
            ("wlotko_federmeier2012_n400", "Wlotko & Federmeier (2012)"),
            ("szewczyk_federmeier2022_n400", "Szewczyk & Federmeier (2022)"),
            ("szewczyk2022_n400", "Szewczyk et al. (2022)"),
        )
    ),
    DatasetRegistration(
        name="michaelov_n400",
        loader_module=f"{_EEG_PREFIX}.michaelov_n400",
        loader_class="MichaelovN400",
        default_subdir="eeg/processed/michaelov_n400",
        paradigm="eeg",
    ),
    DatasetRegistration(
        # Frank et al. (2015): per-participant ERP amplitudes on the UCL
        # eye-tracking sentences, from the authors' own release; shares its
        # stimulus ids with ucl_et and ucl_spr.
        name="frank2015_erp",
        loader_module=f"{_EEG_PREFIX}.frank2015_erp",
        loader_class="Frank2015ERP",
        default_subdir="eeg/original/frank2015_erp",
        paradigm="eeg",
    ),
    DatasetRegistration(
        # ZuCo 1.0 EEG band-power view; shares the .mat files with the zuco1
        # eye-tracking loader (resolved via the YAML data_path field).
        name="zuco1_eeg",
        loader_module=f"{_EEG_PREFIX}.zuco.zuco1",
        loader_class="ZuCo1EEG",
        default_subdir="eeg/original/zuco1",
        paradigm="eeg",
    ),
    DatasetRegistration(
        # ZuCo 2.0 EEG band-power view; shares the .mat files with the zuco2
        # eye-tracking loader (resolved via the YAML data_path field).
        name="zuco2_eeg",
        loader_module=f"{_EEG_PREFIX}.zuco.zuco2",
        loader_class="ZuCo2EEG",
        default_subdir="eeg/original/zuco2",
        paradigm="eeg",
    ),
    # fMRI
    DatasetRegistration(
        # Narratives (Nastase et al., 2021, OpenNeuro ds002345); one story per
        # entry (default 21styear). BOLD prediction via evaluate_encoding.
        name="narratives",
        loader_module=f"{_FMRI_PREFIX}.narratives",
        loader_class="Narratives",
        default_subdir="fmri/original/narratives",
        paradigm="fmri",
    ),
]


def _expand_meco() -> list[DatasetRegistration]:
    """Per-language MECO_L1 and MECO_char entries and per-site MECOL1W2 entries."""
    out: list[DatasetRegistration] = []
    for lang in _MECO_LANGS:
        out.append(
            DatasetRegistration(
                name=f"meco_{lang}",
                loader_module=f"{_ET_PREFIX}.meco.l1",
                loader_class="MECO_L1",
                default_subdir="eye-tracking/original/meco",
                init_kwargs={"language": lang},
            )
        )
        out.append(
            DatasetRegistration(
                name=f"meco_char_{lang}",
                loader_module=f"{_ET_PREFIX}.meco.character",
                loader_class="MECO_char",
                default_subdir="eye-tracking/processed/meco_char",
                init_kwargs={"language": lang},
            )
        )
    for site in _MECO_L1W2_SITES:
        out.append(
            DatasetRegistration(
                name=f"meco_l1_w2_{site}",
                loader_module=f"{_ET_PREFIX}.meco.waves",
                loader_class="MECOL1W2",
                default_subdir=_MECO_WORD_RELEASES["meco_l1_w2"].data_subdir,
                init_kwargs={"site": site},
            )
        )
    return out


REGISTRY: dict[str, DatasetRegistration] = {
    spec.name: spec for spec in (_BASE_ENTRIES + _expand_meco())
}

# Convenience group aliases consumed by the feature-computation CLI.
# Reading-time paradigms (a reading-time DV like first_run / RT applies);
# EEG (signed amplitude) and fMRI (BOLD) are excluded from the reading groups.
# The maze paradigm is not a member either, so natural_stories_maze is named
# on its own.
_READING_PARADIGMS = ("eye-tracking", "self-paced-reading")
_MECO_L1_W1_NAMES = [f"meco_{lang}" for lang in _MECO_LANGS]
_MECO_CHAR_NAMES = [f"meco_char_{lang}" for lang in _MECO_LANGS]
_MECO_L1_W2_NAMES = [f"meco_l1_w2_{site}" for site in _MECO_L1W2_SITES]
_MECO_L2_NAMES = ["meco_l2_w1", "meco_l2_w2"]

GROUPS: dict[str, list[str]] = {
    "meco_l1_w1_all": list(_MECO_L1_W1_NAMES),
    "meco_char_all": list(_MECO_CHAR_NAMES),
    "meco_l1_w2_all": list(_MECO_L1_W2_NAMES),
    "meco_l2_all": list(_MECO_L2_NAMES),
    "meco_cohorts_all": (_MECO_L1_W1_NAMES + _MECO_L1_W2_NAMES + _MECO_L2_NAMES),
    # Paradigm-scoped groups, so a modality can be targeted without pulling in
    # datasets whose dependent variable differs.
    "reading": [n for n, s in REGISTRY.items() if s.paradigm in _READING_PARADIGMS],
    "eeg": [n for n, s in REGISTRY.items() if s.paradigm == "eeg"],
    "fmri": [n for n, s in REGISTRY.items() if s.paradigm == "fmri"],
    # Word-granularity reading corpora (excludes MECO_char, maze, EEG, and fMRI).
    "all_word_level": [
        s.name for s in _BASE_ENTRIES if s.paradigm in _READING_PARADIGMS
    ]
    + _MECO_L1_W1_NAMES
    + _MECO_L1_W2_NAMES,
    "all": list(REGISTRY),
}


def names() -> list[str]:
    """All registered dataset names, sorted."""
    return sorted(REGISTRY)


def expand(name_or_group: str) -> list[str]:
    """Resolve a name or group-alias into a flat list of registry names.

    Raises ``KeyError`` if the input is neither.

    Args:
        name_or_group: A key of :data:`REGISTRY` or of :data:`GROUPS`.
    """
    if name_or_group in GROUPS:
        return list(GROUPS[name_or_group])
    if name_or_group in REGISTRY:
        return [name_or_group]
    raise KeyError(
        f"Unknown dataset {name_or_group!r}. "
        f"Known datasets: {names()}. Groups: {sorted(GROUPS)}."
    )


def get(name: str) -> DatasetRegistration:
    """Return the spec for one dataset name. Raises ``KeyError`` if absent.

    Args:
        name: A key of :data:`REGISTRY`; group aliases are not accepted.
    """
    if name not in REGISTRY:
        raise KeyError(f"Unknown dataset {name!r}; known: {names()}")
    return REGISTRY[name]
