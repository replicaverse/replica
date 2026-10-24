"""Pull-and-pin policy guards for external checkouts.

Tracked code may only reference ``external_repos/`` paths that are
materialized by the registry in ``replica.external``: an
official source, a pinned commit for every git checkout, and any local
compatibility fixes shipped as in-repo patch files.
"""

import re

from replica.external import EXTERNAL_REPOS, patch_path
from replica.external.dependency import _ARCHIVE_SUFFIXES
from replica.studies import ALL_STUDIES

_SHA_RE = re.compile(r"[0-9a-f]{40}")


def _registry_deps():
    for name, deps in EXTERNAL_REPOS.items():
        for dep in deps:
            yield name, dep


def test_registry_git_sources_pin_a_commit():
    for name, dep in _registry_deps():
        if dep.source.startswith("git:"):
            assert dep.commit and _SHA_RE.fullmatch(dep.commit), (
                f"registry entry {name!r} ({dep.source}) must pin a full commit hash"
            )


def test_registry_file_downloads_pin_a_checksum():
    """A fetch skips a file that exists, so only its checksum can reject a
    partial or changed download; archives unpack to directories instead."""
    for name, dep in _registry_deps():
        downloaded = dep.source.startswith(("url:", "osf:", "release:"))
        if downloaded and not dep.source.lower().endswith(_ARCHIVE_SUFFIXES):
            assert dep.checksum, (
                f"registry entry {name!r} downloads {dep.path} without a checksum"
            )


def test_registry_paths_live_under_their_name():
    for name, dep in _registry_deps():
        assert dep.path == f"external_repos/{name}" or dep.path.startswith(
            f"external_repos/{name}/"
        ), f"registry entry {name!r} declares a path outside its checkout: {dep.path}"


def test_registry_patch_files_exist():
    for name, dep in _registry_deps():
        if dep.patch:
            assert patch_path(dep).is_file(), (
                f"registry entry {name!r} names missing patch {dep.patch}"
            )


def test_study_external_repo_deps_come_from_registry():
    """Studies must declare external_repos checkouts via the registry, so
    every checkout stays pinned and fetchable from one place."""
    registered = {(dep.path, dep.source) for _, dep in _registry_deps()}
    for study_name, study in ALL_STUDIES.items():
        for dep in study.dependencies:
            if not dep.path.startswith("external_repos/"):
                continue
            assert (dep.path, dep.source) in registered, (
                f"study {study_name!r} declares {dep.path} outside replica.external"
            )
            if dep.source.startswith("git:"):
                assert dep.commit, (
                    f"study {study_name!r} has unpinned git dep {dep.source}"
                )
