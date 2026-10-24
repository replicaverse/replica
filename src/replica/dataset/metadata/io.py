"""Dataset metadata: YAML serialization, checksum computation and verification."""

import dataclasses
import fnmatch
import hashlib
from pathlib import Path
from typing import Any

import yaml

from replica.dataset.contracts import DatasetInfo, FileEntry, IndexMapping
from replica.utils.digest import prefixed_sha256

# ---------------------------------------------------------------------------
# Checksums
# ---------------------------------------------------------------------------


def compute_sha256(file_path: str | Path) -> str:
    """Return ``"sha256:<hex_digest>"`` for *file_path*.

    Args:
        file_path: File to hash.
    """
    return prefixed_sha256(file_path)


def compute_aggregate_sha256(directory: str | Path, rel_paths: list[str]) -> str:
    r"""Aggregate SHA-256 over multiple files (deterministic, order-free).

    Hashes the sorted ``"<relpath>\\t<sha256>\\n"`` lines so that adding,
    removing or modifying any file changes the result.

    Args:
        directory: Root the relative paths are resolved against.
        rel_paths: Paths of the files to cover, relative to *directory*.
    """
    base = Path(directory)
    lines = sorted(f"{rel}\t{compute_sha256(base / rel)}\n" for rel in rel_paths)
    h = hashlib.sha256()
    for line in lines:
        h.update(line.encode("utf-8"))
    return f"sha256:{h.hexdigest()}"


def compute_checksums(directory: str | Path, files: list[FileEntry]) -> list[FileEntry]:
    """Return a new list of :class:`FileEntry` with checksums filled in.

    Mirrors :func:`verify_checksums`: glob entries (``n_files > 0``) get the
    aggregate checksum over their matched files; single-file entries get the
    plain SHA-256. All other fields are preserved so the result round-trips.

    Args:
        directory: Root the entries' paths are resolved against.
        files: Entries to checksum, in the order the result keeps.
    """
    base = Path(directory)
    result: list[FileEntry] = []
    for entry in files:
        if entry.n_files > 0:
            matched = _expand_glob(base, entry.path)
            checksum = compute_aggregate_sha256(base, matched)
        else:
            checksum = compute_sha256(base / entry.path)
        result.append(dataclasses.replace(entry, checksum=checksum))
    return result


class ChecksumMismatchError(ValueError):
    """Raised when a file's checksum does not match the expected value."""

    def __init__(self, path: str, expected: str, actual: str) -> None:
        """Record which file mismatched and both digests.

        Args:
            path: Entry path that failed, relative to the dataset directory.
            expected: Digest the metadata declares.
            actual: Digest recomputed from the files on disk.
        """
        self.path = path
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"Checksum mismatch for {path}: expected {expected}, got {actual}"
        )


def _expand_glob(directory: Path, pattern: str) -> list[str]:
    """Return sorted relative paths under *directory* matching *pattern*.

    Pattern is shell-style (``fnmatch``) over relative paths, so a leading
    subdirectory like ``"FixationReports/FIX_report_P*.txt"`` matches files
    inside that subdirectory only.
    """
    candidates: list[str] = []
    for p in directory.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(directory).as_posix()
        if fnmatch.fnmatchcase(rel, pattern):
            candidates.append(rel)
    return sorted(candidates)


def verify_checksums(directory: str | Path, files: list[FileEntry]) -> None:
    """Verify checksums for all entries that have a non-empty checksum.

    For glob entries (``n_files > 0``), the matching files are enumerated, the
    aggregate checksum is recomputed, and the file count is verified.

    Args:
        directory: Root the entries' paths are resolved against.
        files: Entries to check; an entry with an empty checksum is skipped.

    Raises:
        ChecksumMismatchError: If any checksum (single or aggregate) does not match.
        FileNotFoundError: If a referenced file is missing or the glob expands
            to a different number of files than declared.
    """
    base = Path(directory)
    for entry in files:
        if not entry.checksum:
            continue
        if entry.n_files > 0:
            matched = _expand_glob(base, entry.path)
            if len(matched) != entry.n_files:
                raise FileNotFoundError(
                    f"Glob {entry.path!r} matched {len(matched)} files, "
                    f"expected {entry.n_files}"
                )
            actual = compute_aggregate_sha256(base, matched)
        else:
            full_path = base / entry.path
            if not full_path.exists():
                raise FileNotFoundError(
                    f"{entry.path!r} is declared in the dataset metadata but "
                    f"missing from {base}. Re-run the dataset fetch, or pass "
                    f"the files this loader reads to _load_metadata."
                )
            actual = compute_sha256(full_path)
        if actual != entry.checksum:
            raise ChecksumMismatchError(entry.path, entry.checksum, actual)


# ---------------------------------------------------------------------------
# YAML serialization
# ---------------------------------------------------------------------------

# Defaults used to decide which fields to omit from YAML output
_DEFAULT_INFO = DatasetInfo(name="")


_DEFAULT_FILE_ENTRY = FileEntry(path="")


def _file_entry_to_dict(fe: FileEntry) -> dict:
    """Serialize a FileEntry, omitting default-valued fields except ``path``."""
    result: dict = {"path": fe.path}
    for f in dataclasses.fields(FileEntry):
        if f.name == "path":
            continue
        val = getattr(fe, f.name)
        if val == getattr(_DEFAULT_FILE_ENTRY, f.name):
            continue
        result[f.name] = val
    return result


def to_dict(info: DatasetInfo) -> dict:
    """Convert a :class:`DatasetInfo` to a dict, omitting default-valued fields.

    ``name`` is always included.

    Args:
        info: The metadata to serialize; nested file entries and the index
            mapping are converted along with it.
    """
    result: dict = {"name": info.name}
    for f in dataclasses.fields(DatasetInfo):
        if f.name == "name":
            continue
        val = getattr(info, f.name)
        default_val = getattr(_DEFAULT_INFO, f.name)
        if val == default_val:
            continue
        # Convert nested dataclasses
        if f.name == "files" and val:
            result["files"] = [_file_entry_to_dict(fe) for fe in val]
        elif f.name == "index_mapping" and val is not None:
            result["index_mapping"] = dataclasses.asdict(val)
        else:
            result[f.name] = val
    return result


def from_dict(data: dict) -> DatasetInfo:
    """Build a :class:`DatasetInfo` from a dict (e.g. parsed YAML).

    Unknown keys are silently ignored (forward compatibility).

    Args:
        data: Mapping of :class:`DatasetInfo` field names to values; ``files``
            and ``index_mapping`` may be nested dicts.

    Raises:
        ValueError: If ``name`` is missing.
    """
    if "name" not in data:
        raise ValueError("DatasetInfo requires a 'name' field")

    known_fields = {f.name for f in dataclasses.fields(DatasetInfo)}
    kwargs: dict = {}
    for key, val in data.items():
        if key not in known_fields:
            continue
        if key == "files" and isinstance(val, list):
            fe_fields = {f.name for f in dataclasses.fields(FileEntry)}
            kwargs["files"] = [
                FileEntry(**{k: v for k, v in fe.items() if k in fe_fields})
                for fe in val
            ]
        elif key == "index_mapping" and isinstance(val, dict):
            im_fields = {f.name for f in dataclasses.fields(IndexMapping)}
            kwargs["index_mapping"] = IndexMapping(
                **{k: v for k, v in val.items() if k in im_fields}
            )
        else:
            kwargs[key] = val
    return DatasetInfo(**kwargs)


def resolve_data_dir(
    path: str | Path, *, metadata_path: str | Path | None = None
) -> Path:
    """Resolve ``data_path`` from explicitly selected catalog-layout metadata.

    A view that shares another paradigm's release has no directory of its own:
    ZuCo 2.0's EEG view reads the eye-tracking ``.mat`` files, and says so with
    ``data_path`` in its YAML. Relative declarations use the third parent of
    *path*: ``<root>`` in ``<root>/<paradigm>/<stage>/<name>``. Without metadata,
    a declaration, or an existing target directory, return *path*.

    Args:
        path: The view's own data directory in the catalog layout.
        metadata_path: YAML file declaring the shared data location, if any.
    """
    directory = Path(path)
    if metadata_path is None:
        return directory
    raw = yaml.safe_load(Path(metadata_path).read_text()) or {}
    declared = raw.get("data_path")
    if not declared:
        return directory
    # Declared relative to the datasets root: <root>/<paradigm>/<stage>/<name>.
    resolved = directory.parent.parent.parent / declared
    return resolved if resolved.is_dir() else directory


def save_yaml(info: DatasetInfo, path: str | Path) -> None:
    """Write *info* to a YAML file at *path*.

    Args:
        info: The metadata to write; default-valued fields are left out.
        path: Destination file, overwritten if it exists.
    """
    d = to_dict(info)
    with open(path, "w") as f:
        yaml.dump(d, f, default_flow_style=False, sort_keys=False)


def load_yaml(path: str | Path) -> DatasetInfo:
    """Load a :class:`DatasetInfo` from a YAML file.

    Args:
        path: YAML file to read.

    Raises:
        ValueError: When the file parses to nothing.
    """
    with open(path) as f:
        data = yaml.safe_load(f)
    if data is None:
        raise ValueError(f"Empty YAML file: {path}")
    return from_dict(data)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def find_dataset_yaml(directory: str | Path, *, required: bool = False) -> Path | None:
    """Return the YAML metadata file describing the dataset in *directory*.

    The file is ``src/replica/dataset/metadata/yaml/{paradigm}/{name}.yaml``, where
    ``paradigm`` is the grandparent directory name and ``name`` is the basename
    of *directory*.

    Args:
        directory: The dataset's data directory, supplying both lookup keys.
        required: Raise if the packaged metadata is absent.

    Returns:
        The YAML path, or ``None`` if no YAML is found.

    Raises:
        FileNotFoundError: Required packaged metadata is absent.
    """
    directory = Path(directory)
    name = directory.name

    pkg_yaml_root = Path(__file__).parent / "yaml"

    # Paradigm = grandparent dir of the data directory.
    # E.g. datasets/eye-tracking/original/provo  →  parents[1] = "eye-tracking".
    parents = directory.parents
    paradigm = parents[1].name if len(parents) >= 2 else ""

    candidate = pkg_yaml_root / paradigm / f"{name}.yaml"
    if candidate.is_file():
        return candidate
    if required:
        raise FileNotFoundError(f"Required dataset metadata is missing: {candidate}")
    return None


def file_presence(path: Path, files: list[dict[str, Any]]) -> dict[str, int]:
    """Count how many of the YAML's declared data files are on disk.

    A corpus loads from a partial download, so its counts can describe the
    download rather than the release. A glob entry counts once, any match.

    Args:
        path: Data directory the declared paths are relative to.
        files: The YAML's file entries, each with a ``path`` that is a relative
            file name or a glob.

    Returns:
        ``declared`` entries carrying a path, and how many of them are
        ``present``.
    """
    declared = present = 0
    for entry in files:
        pattern = str(entry.get("path") or "")
        if not pattern:
            continue
        declared += 1
        if any(char in pattern for char in "*?["):
            present += 1 if any(path.glob(pattern)) else 0
        else:
            present += 1 if (path / pattern).is_file() else 0
    return {"declared": declared, "present": present}
