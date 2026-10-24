"""A corpus read from a preprocessed release must say so in its metadata.

``DatasetInfo.original`` defaults to True, so an unset field is
indistinguishable from a claim to be the original release. The data path
carries the answer already: ``yaml_gen`` documents the
``{paradigm}/{original|processed}/{name}`` layout.
"""

from pathlib import Path

import pytest

from replica.dataset.catalog import registry
from replica.dataset.metadata.io import find_dataset_yaml, load_yaml


def _tier(subdir: str) -> str:
    parts = subdir.split("/")
    return parts[1] if len(parts) > 1 else ""


def _releases() -> dict[tuple[str, Path], list[str]]:
    """Registry keys by (data path, YAML).

    Keys that share both check the same file on the same tier. The tier comes
    from the data path, so keys that share a YAML but load from different tiers
    stay separate.
    """
    releases: dict[tuple[str, Path], list[str]] = {}
    for name in registry.names():
        spec = registry.get(name)
        yaml_path = find_dataset_yaml(spec.default_path())
        if yaml_path is not None:
            releases.setdefault((spec.default_subdir, yaml_path), []).append(name)
    return releases


RELEASES = _releases()


def _cases(tier: str | None = None) -> list:
    """One case per (data path, YAML) pair on ``tier``, or on every tier."""
    return [
        pytest.param(", ".join(names), subdir, yaml_path, id=names[0])
        for (subdir, yaml_path), names in RELEASES.items()
        if tier is None or _tier(subdir) == tier
    ]


@pytest.mark.parametrize(("name", "subdir", "yaml_path"), _cases("processed"))
def test_processed_release_declares_its_derivation(name, subdir, yaml_path):
    info = load_yaml(yaml_path)
    assert info.original is False, (
        f"{name} loads from {subdir} but declares original=True, "
        "so its metadata presents a preprocessed release as the original one"
    )
    assert info.derived_from, f"{name} is preprocessed but names no source"
    assert info.processing, f"{name} is preprocessed but describes no processing"


@pytest.mark.parametrize(("name", "subdir", "yaml_path"), _cases("original"))
def test_original_release_claims_no_derivation(name, subdir, yaml_path):
    info = load_yaml(yaml_path)
    assert info.original is True, (
        f"{name} loads from {subdir} but declares original=False"
    )


def test_every_registered_corpus_is_on_a_known_tier():
    unknown = {
        name: registry.get(name).default_subdir
        for name in registry.names()
        if _tier(registry.get(name).default_subdir) not in {"original", "processed"}
    }
    assert not unknown, f"data paths outside the documented layout: {unknown}"


DISTRIBUTIONS = {"authors", "third-party", "replica"}
AVAILABILITY = {"free", "on request", "licence required"}


@pytest.mark.parametrize(("name", "subdir", "yaml_path"), _cases())
def test_every_corpus_says_who_released_its_files(name, subdir, yaml_path):
    """Who distributed the data is a separate question from how it was shaped.

    A third party's redistribution can be unprocessed, and an authors' release
    can be heavily processed, so this cannot be read off the data path.
    """
    info = load_yaml(yaml_path)
    assert info.distribution in DISTRIBUTIONS, (
        f"{name} declares distribution {info.distribution!r}, "
        f"not one of {sorted(DISTRIBUTIONS)}"
    )
    if info.distribution == "authors":
        assert not info.distributed_by, (
            f"{name} is an authors' release but names a distributor"
        )
    else:
        assert info.distributed_by, (
            f"{name} is a {info.distribution} release but does not say whose"
        )


@pytest.mark.parametrize(("name", "subdir", "yaml_path"), _cases())
def test_every_corpus_says_how_anyone_obtains_its_files(name, subdir, yaml_path):
    """Availability is stated per release, apart from any licence it grants.

    A release can be a free download and state no licence, so the two fields
    answer different questions.
    """
    info = load_yaml(yaml_path)
    assert info.availability in AVAILABILITY, (
        f"{name} declares availability {info.availability!r}, "
        f"not one of {sorted(AVAILABILITY)}"
    )
