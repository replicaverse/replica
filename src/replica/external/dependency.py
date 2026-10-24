"""External dependency declarations, acquisition and file verification.

A :class:`DataDependency` names one file or directory relative to a data
root and where it comes from; :func:`fetch_dependency` materializes it and
:func:`resolve_data_root` fixes the root every study and loader shares.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from ..paths import source_checkout, user_data_root
from ..utils import digest
from ..utils.transfer import download_and_extract, download_url

# Suffixes that trigger download-then-extract (single top-level dir flattened).
_ARCHIVE_SUFFIXES = (".zip", ".tar.xz", ".tar.gz", ".tgz", ".tar.bz2", ".tar")

#: Root the ``patch`` of a dependency resolves against: the ``replica`` package.
PACKAGE_DIR = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class DataDependency:
    """One file or directory needed by a study.

    Args:
        path: Path relative to ``data_root`` where the data should live.
        source: How to fetch it. Supported prefixes:
            - ``"git:user/repo"``, shallow git clone
            - ``"osf:file_id"``, download from OSF
            - ``"url:https://..."``, direct HTTP download
            - ``"release:user/repo:tag:asset"``, GitHub release asset
            - ``"manual"``, must be obtained manually
        checksum: Expected SHA-256 checksum (``"sha256:hex"``). For files only.
        commit: Git commit hash to pin to (for git sources).
        patch: Compatibility diff applied after checkout. Relative paths resolve
            inside the ``replica`` package.
    """

    path: str
    source: str
    checksum: str | None = None
    commit: str | None = None
    patch: str | None = None


def patch_path(dep: DataDependency) -> Path:
    """The patch file of a dependency; an absolute ``patch`` is used as given.

    Args:
        dep: A dependency whose ``patch`` is set.
    """
    path = Path(dep.patch or "")
    return path if path.is_absolute() else PACKAGE_DIR / path


def file_sha256(file_path: Path) -> str:
    """Streamed SHA-256 hex digest of *file_path*.

    Args:
        file_path: File to hash.
    """
    return digest.file_sha256(file_path)


def sha256_checksum(file_path: Path) -> str:
    """Checksum of *file_path* in the ``sha256:hex`` form the pins use.

    Args:
        file_path: File to hash.
    """
    return digest.prefixed_sha256(file_path)


def _default_data_root() -> Path:
    """The source checkout, else the user data root (``$XDG_DATA_HOME/replica``)."""
    return source_checkout() or user_data_root()


def resolve_data_root(data_root: Path | str | None = None) -> Path:
    """The root that study dependency paths resolve against.

    Args:
        data_root: Directory the studies' ``dep.path`` entries resolve against.
            ``None`` falls through to ``$PSYCHO_REPL_DATA``, then the source
            checkout, then ``$XDG_DATA_HOME/replica``.
    """
    if data_root is not None:
        return Path(data_root)
    env = os.environ.get("PSYCHO_REPL_DATA")
    if env:
        return Path(env).expanduser()
    return _default_data_root()


def fetch_dependency(dep: DataDependency, root: Path) -> None:
    """Fetch a single dependency into ``root / dep.path``.

    A git clone lands through a sibling temporary directory and an atomic
    rename, so an interrupted fetch leaves no half-populated target.

    Args:
        dep: What to fetch, and from which of the supported source prefixes.
        root: Data root its ``path`` is resolved against.

    Raises:
        FileNotFoundError: The source is ``manual``, so nothing can fetch it.
        ValueError: The source prefix is not one of the supported ones.
    """
    source = dep.source
    target = root / dep.path

    if source.startswith("git:"):
        repo = source[4:]
        url = f"https://github.com/{repo}.git"
        # Clone to sibling tempdir, then atomic rename, partial state never survives.
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=str(target.parent)) as tmp:
            clone_dir = Path(tmp) / "clone"
            cmd = ["git", "clone"] if dep.commit else ["git", "clone", "--depth", "1"]
            cmd.extend([url, str(clone_dir)])
            subprocess.run(cmd, check=True, timeout=600)
            if dep.commit:
                subprocess.run(
                    ["git", "-C", str(clone_dir), "checkout", dep.commit],
                    check=True,
                    timeout=60,
                )
            if dep.patch:
                subprocess.run(
                    [
                        "git",
                        "-C",
                        str(clone_dir),
                        "apply",
                        str(patch_path(dep)),
                    ],
                    check=True,
                    timeout=60,
                )
            os.replace(clone_dir, target)

    elif source.startswith("osf:"):
        file_id = source[4:]
        url = f"https://osf.io/download/{file_id}/"
        target.parent.mkdir(parents=True, exist_ok=True)
        download_url(url, target)

    elif source.startswith("url:"):
        url = source[4:]
        _download_asset(url, url, target)

    elif source.startswith("release:"):
        parts = source[8:].split(":")  # user/repo:tag:asset_name
        if len(parts) != 3:
            raise ValueError(f"Invalid release source: {source}")
        repo, tag, asset = parts
        url = f"https://github.com/{repo}/releases/download/{tag}/{asset}"
        _download_asset(url, asset, target)

    elif source == "manual":
        raise FileNotFoundError(
            f"Data at {dep.path} must be obtained manually. "
            f"See the study documentation."
        )

    else:
        raise ValueError(f"Unknown source type: {source}")


def _download_asset(url: str, asset_name: str, target: Path) -> None:
    """Download ``url``; if archive, extract and flatten a single top-level dir."""
    if asset_name.lower().endswith(_ARCHIVE_SUFFIXES):
        download_and_extract(url, target)
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        download_url(url, target)
