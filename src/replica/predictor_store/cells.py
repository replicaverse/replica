"""Discovery of the cells an output directory holds.

A completed cell is a ``<tag>.pkl`` whose ``<tag>.pkl.manifest.json`` reports
``completed`` and matches the payload digest. A quarantined cell is a
``<tag>.suspect.pkl`` with its own manifest and, where the run left one, a
``<tag>.failed.log`` naming the stimuli that failed. Every other file in the
directory (locks, partial checkpoints, run records, subdirectories) is left
alone.
"""

from __future__ import annotations

import json
import pickle
from collections.abc import Iterable
from dataclasses import dataclass, field
from inspect import signature
from pathlib import Path

from replica.artifacts import SCHEMA_VERSION
from replica.compute.cache import (
    BOS_SENSITIVE,
    SHARD_RE,
    canonical_manifest,
    method_cache_tag,
    parse_cache_tag,
)
from replica.compute.units import METHOD_STORAGE
from replica.feature.core.outputs.units import unit_for
from replica.utils.digest import file_sha256
from replica.utils.naming import canonical_keys, canonical_tag, safe_lm

from .layout import TagInfo, decode_tag, match_dataset

COMPLETED, QUARANTINED = "completed", "quarantined"
_SUSPECT = ".suspect.pkl"
_MANIFEST = ".manifest.json"
_FAILED = ".failed.log"
# Directory names a run nests its outputs in; the wave label is the first
# ancestor that is none of these.
_NESTING = frozenset({"out", "out_mila", "verify", "config", "logs"})


@dataclass(frozen=True)
class Cell:
    """One pickle and what its manifest and tag say about it."""

    stem: str
    tag: str
    path: Path
    manifest: dict
    status: str
    wave: str
    info: TagInfo
    fail_log: Path | None = None
    size: int = 0
    digest: str = ""

    def changed(self) -> bool:
        """Whether the pickle differs from the one this cell was read from.

        A running compute driver can rename a new candidate over the pickle
        during an export. A different size settles it; otherwise the payload is
        hashed again, since a rewrite can keep both the size and a coarse
        modification time.
        """
        try:
            stat = self.path.stat()
        except OSError:
            return True
        if stat.st_size != self.size:
            return True
        return file_sha256(self.path) != self.digest

    @property
    def identity(self) -> dict:
        """The manifest's identity block."""
        return dict(self.manifest["identity"])

    def index_config(self) -> dict:
        """Encoded settings at their authenticated source precision."""
        options = self.identity.get("options", {})
        return {
            key: options[key] if key in self.info.config else None
            for key in self.info.index_config()
        }

    @property
    def dataset(self) -> str:
        """Corpus the cell was computed on."""
        return self.info.dataset

    @property
    def method(self) -> str:
        """Compute method, as the manifest records it."""
        return str(self.identity["method"])

    @property
    def unit(self) -> str:
        """Grain of the records: ``words``, ``tokens`` or ``characters``."""
        return str(self.identity.get("options", {}).get("unit", "words"))

    @property
    def layer(self) -> int | None:
        """Store layer: 1 for library conventions, 2 for study-associated methods.

        ``None`` for a method the compute driver does not register, such as
        the pooled marginal Monte Carlo sources.
        """
        recorded = self.identity.get("layer")
        if recorded is not None:
            return int(recorded)
        storage = METHOD_STORAGE.get(self.method)
        return storage.layer if storage else None

    @property
    def storage_class(self) -> str | None:
        """``positional`` or ``unit_bound``; ``None`` for an unregistered method."""
        recorded = self.identity.get("storage_class")
        if recorded is not None:
            return str(recorded)
        storage = METHOD_STORAGE.get(self.method)
        return storage.storage_class if storage else None

    @property
    def study(self) -> str | None:
        """Study a layer-2 cell reconstructs, ``None`` on layer 1."""
        recorded = self.identity.get("study")
        if recorded is not None:
            return str(recorded)
        storage = METHOD_STORAGE.get(self.method)
        return storage.study if storage else None

    @property
    def unit_rule(self) -> str | None:
        """Boundary rule a unit-bound estimator used, ``None`` when positional."""
        rule = self.identity.get("unit_rule")
        return None if rule is None else str(rule)

    @property
    def context_scope(self) -> str:
        """Context the model conditioned on: ``stimulus``, ``sentence`` or ``story``."""
        return str(self.identity.get("options", {}).get("context_scope", "stimulus"))

    @property
    def derivation_source(self) -> str | None:
        """Tag of the finer table a derived word table was aggregated from."""
        derivation = self.manifest.get("derivation")
        if not derivation:
            return None
        source = str(derivation["source"])
        return SHARD_RE.sub(
            "", source[: -len(".pkl")] if source.endswith(".pkl") else source
        )

    def failure_detail(self) -> str | None:
        """First lines of the failure log, or the manifest status."""
        if self.fail_log is not None and self.fail_log.exists():
            lines = self.fail_log.read_text().splitlines()
            head = "; ".join(lines[:3])
            if len(lines) > 3:
                head += f"; and {len(lines) - 3} more"
            return head
        return None


@dataclass
class Discovery:
    """Cells found across the output directories, and files that were not cells."""

    completed: list[Cell] = field(default_factory=list)
    quarantined: list[Cell] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    dropped: list[Cell] = field(default_factory=list)
    misaligned: list[Cell] = field(default_factory=list)
    shard_sets: dict[str, int] = field(default_factory=dict)

    @property
    def tags(self) -> set[str]:
        """Tags of the completed cells."""
        return {cell.tag for cell in self.completed}


def wave_label(path: Path) -> str:
    """Name of the run a path belongs to: its first ancestor outside the nesting.

    Args:
        path: An output directory or a verification file.
    """
    current = path.resolve()
    if current.is_file():
        current = current.parent
    while current.name in _NESTING and current.parent != current:
        current = current.parent
    return current.name


def parse_source(argument: str) -> tuple[str, Path]:
    """Split a ``LABEL=PATH`` command-line argument, or label a bare path.

    Args:
        argument: The argument as typed.
    """
    if "=" in argument and not Path(argument).exists():
        label, _, raw = argument.partition("=")
        return label, Path(raw)
    path = Path(argument)
    return wave_label(path), path


def _read_manifest(path: Path, expected_status: str) -> tuple[dict | None, str | None]:
    sidecar = path.with_name(path.name + _MANIFEST)
    if not sidecar.exists():
        return None, f"{path.name}: no manifest"
    try:
        manifest = json.loads(sidecar.read_text())
        identity = manifest["identity"]
        status = manifest["status"]
        digest = manifest["payload_sha256"]
        if not isinstance(identity, dict):
            raise ValueError("identity must be an object")
        declared_units = identity.get("units")
        if declared_units is not None:
            if not isinstance(declared_units, dict):
                raise ValueError("identity.units must be an object")
            for key, unit in canonical_keys(declared_units).items():
                if unit != unit_for(key):
                    raise ValueError(
                        f"manifest unit for {key!r} conflicts with the unit registry"
                    )
        if not isinstance(identity.get("options", {}), dict):
            raise ValueError("identity.options must be an object")
        model = identity.get("model")
        if model is not None and not isinstance(model, dict):
            raise ValueError("identity.model must be an object or null")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return None, f"{path.name}: unreadable manifest ({exc})"
    if identity.get("schema_version") != SCHEMA_VERSION:
        return None, f"{path.name}: manifest schema {identity.get('schema_version')!r}"
    if status != expected_status:
        return None, f"{path.name}: manifest status {status!r}"
    if digest != file_sha256(path):
        return None, f"{path.name}: payload digest differs from the manifest"
    return manifest, None


def _check_tag_options(info: TagInfo, options: dict) -> None:
    missing = sorted(set(info.config) - options.keys())
    if missing:
        raise ValueError(f"manifest options missing encoded settings: {missing}")
    arguments = {
        key: value
        for key, value in options.items()
        if key in signature(method_cache_tag).parameters and key != "method"
    }
    # An unseeded filename does not claim a particular source seed.
    if "seed" not in info.config:
        arguments.pop("seed", None)
    try:
        _, encoded = parse_cache_tag(method_cache_tag(info.method, **arguments))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"manifest options cannot encode the filename: {exc}") from exc
    if encoded != info.config:
        raise ValueError(
            f"manifest options conflict with filename settings: "
            f"{encoded!r} != {info.config!r}"
        )
    for key, expected in (("unit", info.unit), ("context_scope", info.context_scope)):
        if key in options and options[key] != expected:
            raise ValueError(f"manifest option {key} conflicts with the filename")


def _check_start_options(identity: dict) -> None:
    """Reject a recorded prefix when a BOS-sensitive method disables it."""
    if (
        identity.get("method") not in BOS_SENSITIVE
        or identity.get("options", {}).get("add_bos") is not False
    ):
        return
    start = (identity.get("model") or {}).get("start")
    if start is None:
        return
    if not isinstance(start, dict):
        raise ValueError("identity.model.start must be an object or null")
    ids = start.get("ids")
    if ids is not None and not isinstance(ids, list):
        raise ValueError("identity.model.start.ids must be an array or null")
    if ids:
        raise ValueError("manifest model.start.ids contradicts add_bos=False")


def _cell(
    path: Path, stem: str, status: str, wave: str, names: Iterable[str]
) -> tuple[Cell | None, str | None]:
    manifest, problem = _read_manifest(path, status)
    if manifest is None:
        return None, problem
    # Files keep the name the run gave them; the cell carries the current one.
    fail_log = path.with_name(stem + _FAILED)
    stem = canonical_tag(stem)
    tag = SHARD_RE.sub("", stem)
    dataset = match_dataset(tag, names)
    if dataset is None:
        return None, f"{path.name}: no registry dataset in the tag"
    try:
        manifest = canonical_manifest(manifest)
        identity = manifest["identity"]
        _check_start_options(identity)
        requested = identity.get("options", {}).get("model_id")
        model_id = requested
        if model_id is None:
            model_id = (identity.get("model") or {}).get("source")
        info = decode_tag(tag, dataset, model_id=model_id)
        _check_tag_options(info, identity.get("options", {}))
        # A resolved source can have a canonical Hub name different from the
        # requested ID that formed the filename.
        if requested is not None and safe_lm(requested) != info.model_slot:
            raise ValueError("manifest model_id conflicts with the filename model slot")
    except ValueError as exc:
        return None, f"{path.name}: {exc}"
    method = manifest["identity"].get("method")
    if method != info.method:
        return None, f"{path.name}: tag names {info.method}, manifest {method}"
    stat = path.stat()
    return (
        Cell(
            stem=stem,
            tag=tag,
            path=path,
            manifest=manifest,
            status=status,
            wave=wave,
            info=info,
            fail_log=fail_log if fail_log.exists() else None,
            size=stat.st_size,
            digest=manifest["payload_sha256"],
        ),
        None,
    )


def payload_names(sources: Iterable[tuple[str, Path]]) -> set[str]:
    """Stems of the completed payloads the given directories hold when called.

    Reads the directory listing alone, so it is cheap enough to run again at
    the end of a long export and see which cells appeared meanwhile.

    Args:
        sources: ``(wave label, directory)`` pairs.
    """
    found: set[str] = set()
    for _wave, directory in sources:
        if not directory.is_dir():
            continue
        for path in directory.iterdir():
            if (
                path.is_file()
                and path.suffix == ".pkl"
                and not path.name.endswith((_SUSPECT, ".candidate.pkl"))
                and not SHARD_RE.search(path.stem)
            ):
                found.add(canonical_tag(path.stem))
    return found


def load_payload(cell: Cell) -> dict:
    """The cell's records by stimulus id, their value keys under current names.

    Args:
        cell: A completed cell.

    Raises:
        ValueError: When the payload is not a mapping or value keys have colliding aliases.
    """
    with cell.path.open("rb") as stream:
        payload = pickle.load(stream)
    if not isinstance(payload, dict):
        raise ValueError("payload is not a mapping")
    for records in payload.values():
        for record in records:
            record.values = canonical_keys(record.values)
    return payload


def discover(sources: list[tuple[str, Path]], names: Iterable[str]) -> Discovery:
    """Find the completed and quarantined cells of the given output directories.

    Args:
        sources: ``(wave label, directory)`` pairs; only the directory's own
            files are read, never its subdirectories.
        names: Dataset names a tag may end in.

    Raises:
        ValueError: When two directories hold a completed cell of the same tag
            with different payloads or computational identities.
    """
    names = list(names)
    found = Discovery()
    completed: dict[str, Cell] = {}
    quarantined: dict[str, Cell] = {}
    shards: dict[str, int] = {}
    for wave, directory in sources:
        if not directory.is_dir():
            raise FileNotFoundError(f"{directory} is not a directory")
        for path in sorted(directory.iterdir()):
            if not path.is_file():
                continue
            if path.name.endswith(_SUSPECT):
                stem, status = path.name[: -len(_SUSPECT)], QUARANTINED
            elif path.suffix == ".pkl" and not path.name.endswith(".candidate.pkl"):
                stem, status = path.stem, COMPLETED
            else:
                continue
            # A shard file holds one slice of a corpus. replica.compute.shards checks
            # that a set tiles the corpus from stimulus zero and writes the
            # canonical pickle; until it has run, the slices are not a cell.
            if SHARD_RE.search(stem):
                tag = canonical_tag(SHARD_RE.sub("", stem))
                shards[tag] = shards.get(tag, 0) + 1
                continue
            cell, problem = _cell(path, stem, status, wave, names)
            if cell is None:
                found.problems.append(str(problem))
                continue
            bucket = completed if status == COMPLETED else quarantined
            earlier = bucket.get(cell.stem)
            if earlier is not None and status == COMPLETED:
                if (
                    earlier.manifest["payload_sha256"]
                    != cell.manifest["payload_sha256"]
                ):
                    raise ValueError(
                        f"{cell.stem} is completed in both {earlier.path.parent} "
                        f"and {directory} with different payloads"
                    )
                if earlier.identity != cell.identity:
                    raise ValueError(
                        f"{cell.stem} has conflicting computational identities in "
                        f"{earlier.path.parent} and {directory}"
                    )
                continue
            if earlier is None:
                bucket[cell.stem] = cell
    found.completed = sorted(completed.values(), key=lambda c: c.stem)
    done = {cell.tag for cell in found.completed}
    # A canonical pickle beside leftover slices is the cell; the slices are not.
    found.shard_sets = {
        tag: count for tag, count in sorted(shards.items()) if tag not in done
    }
    found.quarantined = sorted(
        (cell for cell in quarantined.values() if cell.tag not in done),
        key=lambda c: c.stem,
    )
    return found
