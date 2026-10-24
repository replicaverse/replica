"""Verify materialized reproduction inputs without modifying their checkouts."""

import os
import subprocess
import tempfile
from pathlib import Path

from .dependency import DataDependency, patch_path, sha256_checksum


class DependencyMismatchError(ValueError):
    """An available reproduction input differs from its declared identity."""


def verify_dependency(dep: DataDependency, root: Path) -> None:
    """Verify a file checksum or pinned Git tree and its compatibility patch.

    Generated untracked files are allowed. Tracked files must match the pin plus
    the declared patch, including when the user's real index has staged changes.
    Verification uses a temporary index and never resets the checkout.

    Args:
        dep: Dependency whose declared identity is authoritative.
        root: Root against which its relative path is resolved.
    """
    path = root / dep.path
    if not path.exists():
        raise FileNotFoundError(path)
    if dep.checksum and path.is_file() and sha256_checksum(path) != dep.checksum:
        raise DependencyMismatchError(f"Checksum mismatch for {path}")
    if not dep.source.startswith("git:"):
        return
    if dep.commit is None:
        raise DependencyMismatchError(f"Git dependency {path} has no pinned commit")
    head = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if head.returncode or head.stdout.strip() != dep.commit:
        raise DependencyMismatchError(
            f"{path}: expected Git HEAD {dep.commit}; checkout is missing or differs"
        )
    # A parent repository must not make a plain subdirectory look like a clone.
    top = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=True,
    )
    if Path(top.stdout.strip()).resolve() != path.resolve():
        raise DependencyMismatchError(f"{path} is not the root of the pinned checkout")
    with tempfile.TemporaryDirectory(prefix="replica-verify-") as tmp:
        env = dict(os.environ, GIT_INDEX_FILE=str(Path(tmp) / "index"))
        command = ["git", "-C", str(path)]
        subprocess.run(
            [*command, "read-tree", dep.commit],
            env=env,
            check=True,
            capture_output=True,
        )
        if dep.patch:
            subprocess.run(
                [*command, "apply", "--cached", str(patch_path(dep))],
                env=env,
                check=True,
                capture_output=True,
            )
        diff = subprocess.run(
            [*command, "diff", "--quiet", "--"],
            env=env,
            check=False,
            capture_output=True,
        )
        if diff.returncode:
            raise DependencyMismatchError(
                f"{path}: tracked files differ from the pin and declared patch"
            )
