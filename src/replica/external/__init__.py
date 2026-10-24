"""Pinned external dependencies, checkout resolution and input adapters.

:mod:`.dependency` declares and fetches one file or directory
(:class:`DataDependency`); :mod:`.repositories` is the registry of pinned upstream
repositories that studies, loaders and predictor sources resolve through
:func:`repo_path`. :mod:`.adapters` contains upstream-specific helpers and patches.
"""

from .dependency import (
    DataDependency,
    fetch_dependency,
    file_sha256,
    patch_path,
    resolve_data_root,
    sha256_checksum,
)
from .repositories import EXTERNAL_REPOS, repo_dependencies, repo_path
from .verification import DependencyMismatchError, verify_dependency

__all__ = [
    "EXTERNAL_REPOS",
    "DataDependency",
    "DependencyMismatchError",
    "verify_dependency",
    "fetch_dependency",
    "file_sha256",
    "patch_path",
    "repo_dependencies",
    "repo_path",
    "resolve_data_root",
    "sha256_checksum",
]
