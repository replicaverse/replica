"""Tests for the shared dataset registry.

The registry is a single source of truth for "which datasets does the
REPLICA claim to support" so feature computation, dataset fetching, and the
test suite cannot disagree.

These tests pin the contract:
  * Names are unique, filesystem-friendly, and rooted under datasets/.
  * Every loader_module / loader_class imports and constructs.
  * Group aliases resolve to known names; MECO groups cover 13 langs.
  * compute_lm_features.py imports the shared registry.

Each whole-registry property gets one test that reports all violations
in a single assertion, per-entry parametrization (which would be the
canonical pytest style) is overkill for properties that hold by
construction across all entries.
"""

import re

import pytest

from replica.dataset.catalog.registry import (
    _BASE_ENTRIES,
    GROUPS,
    REGISTRY,
    DatasetRegistration,
    _expand_meco,
    data_dir,
    expand,
    get,
)
from replica.dataset.eye_tracking.meco.releases import (
    L1W2_SITE_LANGUAGES,
)
from replica.dataset.eye_tracking.meco.releases import (
    LANGUAGES as _MECO_LANGS,
)


def test_no_duplicate_names():
    # REGISTRY is a dict, so it silently dedups any collision at construction.
    # Scan the source spec lists instead to catch a duplicate name regression.
    names = [s.name for s in _BASE_ENTRIES + _expand_meco()]
    dupes = sorted({n for n in names if names.count(n) > 1})
    assert not dupes, f"duplicate names in registry spec lists: {dupes}"


def test_every_loader_class_resolves_and_constructs():
    """Every registered loader_module/loader_class must (a) import,
    (b) be callable, (c) accept its declared init_kwargs at construct
    time. We collect all failures into one assertion so a fresh CI
    failure pinpoints every broken entry at once."""
    failures: list[str] = []
    for name, spec in REGISTRY.items():
        try:
            cls = spec.loader_cls()
        except Exception as e:
            failures.append(
                f"{name}: import {spec.loader_module}.{spec.loader_class} → {type(e).__name__}: {e}"
            )
            continue
        if not callable(cls):
            failures.append(f"{name}: {cls!r} is not callable")
            continue
        try:
            cls(**spec.init_kwargs)
        except Exception as e:
            failures.append(
                f"{name}: {spec.loader_class}(**{spec.init_kwargs}) → {type(e).__name__}: {e}"
            )
    assert not failures, "registry entries failed to resolve:\n  " + "\n  ".join(
        failures
    )


def test_groups_resolve_to_known_names():
    failures: list[str] = []
    for group, members in GROUPS.items():
        if not members:
            failures.append(f"group {group!r} is empty")
            continue
        if len(members) != len(set(members)):
            failures.append(f"group {group!r} contains duplicate members")
        for m in members:
            if m not in REGISTRY:
                failures.append(f"group {group!r} → unknown name {m!r}")
    assert not failures, "group aliases broken:\n  " + "\n  ".join(failures)


def test_meco_groups_cover_all_13_languages():
    expected_word = {f"meco_{lang}" for lang in _MECO_LANGS}
    expected_char = {f"meco_char_{lang}" for lang in _MECO_LANGS}
    assert set(GROUPS["meco_l1_w1_all"]) == expected_word
    assert set(GROUPS["meco_l1_w1_all"]) == expected_word
    assert set(GROUPS["meco_char_all"]) == expected_char
    assert set(GROUPS["meco_l1_w2_all"]) == {
        f"meco_l1_w2_{site}" for site in L1W2_SITE_LANGUAGES
    }
    assert set(GROUPS["meco_l2_all"]) == {"meco_l2_w1", "meco_l2_w2"}


def test_meco_registry_loaders_live_in_one_package():
    modules = {
        spec.loader_module
        for name, spec in REGISTRY.items()
        if name.startswith("meco_")
    }
    assert modules == {
        "replica.dataset.eye_tracking.meco.character",
        "replica.dataset.eye_tracking.meco.l1",
        "replica.dataset.eye_tracking.meco.waves",
    }


def test_expand_handles_single_names_and_groups():
    assert expand("provo") == ["provo"]
    assert set(expand("meco_l1_w1_all")) == {f"meco_{lang}" for lang in _MECO_LANGS}
    with pytest.raises(KeyError):
        expand("not_a_real_name")


def test_get_returns_specs_and_raises_on_unknown():
    spec = get("provo")
    assert isinstance(spec, DatasetRegistration)
    assert spec.name == "provo"
    with pytest.raises(KeyError):
        get("not_a_real_name")


def test_all_names_match_format_and_root_under_datasets():
    """compute_lm_features.py interpolates registry names into pickle
    filenames, so non-printable or whitespace-bearing names would
    break. Every default_subdir must resolve under the data_dir() tree."""
    bad_name = [n for n in REGISTRY if not re.match(r"^[a-z0-9_]+$", n)]
    root = data_dir().resolve()
    bad_subdir = []
    for spec in REGISTRY.values():
        try:
            resolved = spec.default_path().resolve()
            resolved.relative_to(root)
        except ValueError:
            bad_subdir.append(f"{spec.name} → {spec.default_subdir!r}")
    assert not bad_name, f"non-fs-friendly registry names: {bad_name}"
    assert not bad_subdir, "default_subdir not under data_dir():\n  " + "\n  ".join(
        bad_subdir
    )


def test_fetch_configs_align_with_registry():
    """The fetch config and the registry must not drift: every downloadable key
    is a registry name (the single source of truth for its on-disk path) or one
    of a small allowlist of shared-download bundles / non-dataset resources.
    Registry-name configs must not carry their own ``dir`` (``dataset_directory``
    reads it
    from ``default_subdir``); a stray ``dir`` would be silently ignored and could
    drift from the registry.
    """
    from replica.dataset.catalog import sources as catalog

    # Downloads not tied 1:1 to a single registry entry (keep their own dir).
    BUNDLES = {
        "meco",
        "meco_l1_w2",
        "meco_char",
        "subtlex_en",
        "meco_l1_w1_fixations",
        "meco_l1_w2_fixations",
        "meco_l2_w1_fixations",
        "meco_l2_w2_fixations",
    }

    bad_key, redundant_dir = [], []
    for name, cfg in catalog.DATASETS.items():
        if name not in REGISTRY and name not in BUNDLES:
            bad_key.append(name)
        elif name in REGISTRY and "dir" in cfg:
            redundant_dir.append(name)
    bad_alias = [
        alias
        for alias, target in catalog.ALIASES.items()
        if target not in catalog.DATASETS
    ]

    assert not bad_key, f"fetch keys not in registry or bundle allowlist: {bad_key}"
    assert not redundant_dir, (
        "registry-name fetch configs carry a redundant 'dir' that "
        "dataset_directory ignores "
        f"(path comes from the registry, so this can silently drift): {redundant_dir}"
    )
    assert not bad_alias, f"fetch aliases pointing nowhere: {bad_alias}"


def test_compute_lm_features_uses_registry():
    """The feature producer uses the shared dataset registry."""
    from replica.compute import runner

    assert runner.REGISTRY is REGISTRY, (
        "replica.compute.runner.REGISTRY drifted from the shared "
        "replica.dataset.catalog.registry.REGISTRY."
    )
