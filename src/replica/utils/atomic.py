"""Atomic file writes: a reader finds the previous content or the complete new one."""

import os
import pickle
from pathlib import Path


def temporary_path(path: Path) -> Path:
    """The file :func:`atomic_pickle` writes before renaming it to *path*.

    Args:
        path: File the write targets.
    """
    return path.with_name(path.name + ".tmp")


def atomic_pickle(obj: object, path: Path, *, fsync: bool = False) -> None:
    """Pickle *obj* to :func:`temporary_path` and rename it over *path*.

    Args:
        obj: Object to pickle at the default protocol.
        path: File to write.
        fsync: Flush temporary file contents to disk before the rename.
            The parent directory entry is not synchronized.
    """
    tmp = temporary_path(path)
    with open(tmp, "wb") as stream:
        pickle.dump(obj, stream)
        if fsync:
            stream.flush()
            os.fsync(stream.fileno())
    os.replace(tmp, path)
