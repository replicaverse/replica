"""Pinned input verification uses temporary Git indexes and preserves user work."""

import subprocess
from dataclasses import replace

import pytest

from replica.external import (
    DataDependency,
    DependencyMismatchError,
    verify_dependency,
)


def _git(path, *args):
    return subprocess.run(
        ["git", "-C", str(path), *args], check=True, text=True, capture_output=True
    ).stdout.strip()


@pytest.fixture
def checkout(tmp_path):
    repo = tmp_path / "source"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "config", "user.email", "test@example.invalid")
    (repo / "source.txt").write_text("original\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "initial")
    pin = _git(repo, "rev-parse", "HEAD")
    return (
        tmp_path,
        repo,
        DataDependency(
            path="source", source="git:https://example.invalid/source.git", commit=pin
        ),
    )


def test_empty_directory_is_not_a_verified_checkout(checkout):
    root, repo, dep = checkout
    empty = root / "empty"
    empty.mkdir()
    with pytest.raises(DependencyMismatchError):
        verify_dependency(replace(dep, path="empty"), root)
    verify_dependency(dep, root)
    with pytest.raises(DependencyMismatchError):
        verify_dependency(replace(dep, commit="0" * 40), root)


def test_tracked_and_staged_changes_fail_without_modifying_the_checkout(checkout):
    root, repo, dep = checkout
    source = repo / "source.txt"
    source.write_text("user work\n")
    _git(repo, "add", "source.txt")
    index_before = (repo / ".git/index").read_bytes()
    with pytest.raises(DependencyMismatchError, match="tracked files"):
        verify_dependency(dep, root)
    assert source.read_text() == "user work\n"
    assert (repo / ".git/index").read_bytes() == index_before


def test_declared_patch_is_verified_and_untracked_results_are_allowed(
    checkout, monkeypatch
):
    root, repo, dep = checkout
    source = repo / "source.txt"
    source.write_text("compatible\n")
    patches = root / "patches"
    patches.mkdir()
    (patches / "compat.patch").write_text(_git(repo, "diff") + "\n")
    (repo / "generated.txt").write_text("results\n")
    verify_dependency(replace(dep, patch=str(patches / "compat.patch")), root)
    with pytest.raises(DependencyMismatchError):
        verify_dependency(dep, root)
