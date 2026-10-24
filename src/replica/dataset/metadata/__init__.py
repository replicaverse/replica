"""Read, write and verify dataset metadata."""

from .io import (
    ChecksumMismatchError,
    compute_aggregate_sha256,
    compute_checksums,
    compute_sha256,
    find_dataset_yaml,
    from_dict,
    load_yaml,
    resolve_data_dir,
    save_yaml,
    to_dict,
    verify_checksums,
)

__all__ = [
    "compute_sha256",
    "compute_aggregate_sha256",
    "compute_checksums",
    "ChecksumMismatchError",
    "verify_checksums",
    "to_dict",
    "from_dict",
    "resolve_data_dir",
    "save_yaml",
    "load_yaml",
    "find_dataset_yaml",
]
