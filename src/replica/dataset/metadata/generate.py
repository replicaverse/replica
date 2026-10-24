"""(Re)generate the in-package YAML metadata for a dataset directory.

Walks the directory tree, computes SHA-256 checksums for all data files
(csv, tsv, rda, tok, txt, xlsx, docx, mat, dat, itm), and writes/updates
``replica/dataset/metadata/yaml/{paradigm}/{name}.yaml`` next to this module. Only the
``files`` section is rewritten; every other key of an existing YAML is kept.

A glob the YAML already declares keeps its pattern and covers every file it
matches. Other files are grouped by collapsed pattern (runs of digits replaced
with ``*``, under the same parent directory). Groups of size >=
``GROUP_THRESHOLD`` are emitted as a single glob FileEntry with an aggregate
checksum, so e.g. PoTeC's per-reader fixation files become a handful of
entries instead of expanding the yaml to thousands of lines.

This is a maintainer operation because it rewrites the shipped checksums. Run
it with ``python -m replica.dataset.metadata.generate``.
"""

import argparse
import dataclasses
import re
import sys
from collections import defaultdict
from pathlib import Path

import yaml

from replica.dataset.contracts import DatasetInfo, FileEntry
from replica.dataset.metadata.io import (
    _expand_glob,
    _file_entry_to_dict,
    compute_aggregate_sha256,
    compute_sha256,
    find_dataset_yaml,
    from_dict,
)

DATA_EXTENSIONS = {
    ".csv",
    ".tsv",
    ".rda",
    ".tok",
    ".txt",
    ".xlsx",
    ".docx",
    ".mat",
    ".dat",
    ".itm",
}

# The in-package YAML tree, beside this module; matches the layout read by
# :func:`replica.dataset.metadata.io.find_dataset_yaml`.
_PKG_YAML_ROOT = Path(__file__).resolve().parent / "yaml"


def _yaml_destination(directory: Path) -> Path:
    """Return ``replica/dataset/metadata/yaml/{paradigm}/{name}.yaml``.

    ``paradigm`` is the grandparent dir of *directory* (i.e. the
    ``datasets/{paradigm}/{original|processed}/{name}/`` layout); ``name``
    is the basename. The destination directory is created if missing.
    """
    name = directory.name
    # Compute paradigm exactly as metadata.find_dataset_yaml reads it (from the
    # unresolved path), so the write target and the loader's lookup agree.
    parents = directory.parents
    paradigm = parents[1].name if len(parents) >= 2 else ""
    out_dir = _PKG_YAML_ROOT / paradigm
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{name}.yaml"


#: Files matching the same digit-collapsed pattern are folded into one glob
#: entry once the group size reaches this threshold.
GROUP_THRESHOLD = 5

_DIGIT_RUN = re.compile(r"\d+")


def _collapse_pattern(rel_path: str) -> str:
    """Replace digit runs in the basename with ``*`` (parent stays literal)."""
    p = Path(rel_path)
    return (p.parent / _DIGIT_RUN.sub("*", p.name)).as_posix()


def _collect_data_files(directory: Path) -> list[str]:
    """Return sorted relative paths of data files under *directory*."""
    rels: list[str] = []
    for p in directory.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in DATA_EXTENSIONS:
            continue
        rels.append(p.relative_to(directory).as_posix())
    return sorted(rels)


def _file_entries(
    directory: Path, rels: list[str], previous: list[FileEntry]
) -> list[FileEntry]:
    """Checksummed entries for *rels*, carrying over what *previous* declared.

    A declared glob keeps its pattern; other entries keep their role and
    provenance when the regenerated path equals the declared one.
    """
    entries: list[FileEntry] = []
    claimed: set[str] = set()
    for entry in previous:
        if entry.n_files <= 0:
            continue
        members = _expand_glob(directory, entry.path)
        if members:
            claimed.update(members)
            entries.append(
                dataclasses.replace(
                    entry,
                    checksum=compute_aggregate_sha256(directory, members),
                    n_files=len(members),
                )
            )

    # Group the other files by collapsed pattern. Singleton groups stay
    # individual.
    groups: dict[str, list[str]] = defaultdict(list)
    for rel in rels:
        if rel not in claimed:
            groups[_collapse_pattern(rel)].append(rel)

    by_path = {e.path: e for e in previous}
    for pattern, members in sorted(groups.items()):
        if len(members) >= GROUP_THRESHOLD:
            checksum = compute_aggregate_sha256(directory, members)
            existing = by_path.get(pattern)
            entries.append(
                FileEntry(
                    path=pattern,
                    role=existing.role if existing else "measurements",
                    checksum=checksum,
                    original_path=existing.original_path if existing else "",
                    provenance_unavailable=(
                        existing.provenance_unavailable if existing else ""
                    ),
                    n_files=len(members),
                )
            )
        else:
            for rel in members:
                checksum = compute_sha256(directory / rel)
                existing = by_path.get(rel)
                entries.append(
                    FileEntry(
                        path=rel,
                        role=existing.role if existing else "measurements",
                        checksum=checksum,
                        original_path=existing.original_path if existing else "",
                        provenance_unavailable=(
                            existing.provenance_unavailable if existing else ""
                        ),
                    )
                )
    return entries


def generate_yaml(directory: Path, name: str | None = None) -> DatasetInfo:
    """Generate or update dataset.yaml for *directory*.

    Args:
        directory: Dataset directory to walk; also fixes where the YAML is
            written and which existing YAML is read back.
        name: Dataset name to record; ``None`` keeps the existing one, or the
            directory's basename when there is no YAML yet.

    Returns:
        The written metadata, with ``files`` rebuilt from what is on disk and
        each entry's role and provenance carried over where it already had one.
    """
    existing_path = find_dataset_yaml(directory)
    raw: dict = {}
    if existing_path is not None:
        raw = yaml.safe_load(existing_path.read_text()) or {}
    raw["name"] = name if name is not None else raw.get("name", directory.name)
    previous = from_dict({"name": raw["name"], "files": raw.get("files") or []})

    rels = _collect_data_files(directory)
    entries = _file_entries(directory, rels, previous.files)
    raw["files"] = [_file_entry_to_dict(entry) for entry in entries]
    out_path = _yaml_destination(directory)
    with open(out_path, "w") as f:
        yaml.dump(raw, f, default_flow_style=False, sort_keys=False)
    if not entries:
        print(f"  No data files found in {directory}")
    n_total = sum(e.n_files or 1 for e in entries)
    n_globs = sum(1 for e in entries if e.n_files)
    print(
        f"  Wrote {out_path} "
        f"({len(entries)} entries, {n_total} files, {n_globs} glob entries)"
    )
    return from_dict(raw)


def main() -> None:
    """Command line entry point: rewrite one dataset directory's YAML."""
    parser = argparse.ArgumentParser(
        description="Generate or update dataset.yaml with checksums"
    )
    parser.add_argument(
        "directory",
        type=Path,
        help="Dataset directory to process",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Override the dataset name",
    )
    args = parser.parse_args()

    if not args.directory.is_dir():
        print(f"Error: {args.directory} is not a directory", file=sys.stderr)
        sys.exit(1)

    generate_yaml(args.directory, args.name)


if __name__ == "__main__":
    main()
