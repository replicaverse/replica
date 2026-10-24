"""SHA-256 digests of files and text, in the two spellings the package records."""

import hashlib
from pathlib import Path


def file_sha256(path: str | Path) -> str:
    """Hex SHA-256 of a file, streamed in 1 MiB blocks.

    Args:
        path: File to hash.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def prefixed_sha256(path: str | Path) -> str:
    """The ``sha256:<hex>`` form dataset metadata and dependency pins record.

    Args:
        path: File to hash.
    """
    return f"sha256:{file_sha256(path)}"


def text_sha256(text: str) -> str:
    """Hex SHA-256 of a string's UTF-8 bytes.

    Args:
        text: Text to hash.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
