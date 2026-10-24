"""Resolve checksum-bound study inputs from packages or archived bundles."""

import errno
import re
import tempfile
from pathlib import Path

from ...paths import user_cache_root
from . import files, store


def resource_file(study: str, name: str) -> Path:
    """Return an indexed input or figure, verifying its SHA-256 on each access.

    Bundled resources are fetched at their immutable revision and extracted
    atomically into the user cache. A corrupt local or cached copy raises
    ``ValueError``; it is never silently replaced.

    Args:
        study: Study package identifier.
        name: Resource path relative to the study package.
    """
    if not re.fullmatch(r"[a-z][a-z0-9_]*", study) or not files._resource_path(name):
        raise ValueError("Invalid study resource path")
    document = files.read_index(study)
    try:
        resource = document["resources"][name]
    except KeyError:
        raise FileNotFoundError(f"{study}: no indexed resource {name!r}") from None
    archives = document.get("archives", {})
    package = files.STUDIES_DIR / study
    problems = files._resource_problems(
        study, package, {name: resource}, [name], archives
    )
    if problems:
        raise ValueError("; ".join(problems))
    digest = resource.get("archive")
    if digest is not None:
        problems = files._archive_problems(study, {digest: archives[digest]})
        if problems:
            raise ValueError("; ".join(problems))
    path = package / name
    if path.is_file():
        return path
    binding = archives[digest]
    destination = user_cache_root() / "study-bundles" / study / digest
    if not destination.exists():
        reference = dict(binding["reference"], study=study)
        archive = store.fetch_bundle(reference)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
            extracted = store.extract_archive(
                archive, Path(temporary) / "payload", sha256=digest
            )
            try:
                extracted.rename(destination)
            except OSError as exc:
                if exc.errno not in {errno.EEXIST, errno.ENOTEMPTY}:
                    raise
    path = destination / name
    if (
        destination.is_symlink()
        or any((destination / parent).is_symlink() for parent in Path(name).parents)
        or path.is_symlink()
        or not path.is_file()
        or files._sha256(path) != resource["sha256"]
    ):
        raise ValueError(f"{study}/{name}: cached resource differs from its digest")
    return path
