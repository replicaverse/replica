"""Machine-readable facts about every integrated dataset.

The data-free tier (registry, dataset YAML, download catalog) runs anywhere.
The measured tier loads each corpus to count it, so it runs only where the data
is, and persists to ``dataset_counts.json`` for consumers that cannot load it.
Comparing that snapshot against the registry catches a corpus added without it.
"""

import argparse
import dataclasses
import hashlib
import inspect
import json
import os
import subprocess
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from replica.dataset.catalog import registry
from replica.dataset.catalog.description import describe_dataset
from replica.dataset.metadata.io import (
    file_presence,
    find_dataset_yaml,
    load_yaml,
    resolve_data_dir,
)

SNAPSHOT_PATH: Path = Path(__file__).with_name("dataset_counts.json")

# Prefix marking the child process's payload, so a loader's own stdout chatter
# cannot be parsed as the result.
_MARKER = "__DATASET_COUNTS__"

DEFAULT_TIMEOUT = 900

# Substrings identifying a load failure caused by missing data rather than by
# broken code. Anything else is reported as an error and left visible.
_DATA_ABSENT = ("FileNotFoundError", "No such file", "Errno 2")

# Narratives reads BOLD from the release's AFNI derivatives, and counting its
# cohort must not pull ~0.7 GB per subject.
_LOAD_KWARGS: dict[str, dict[str, Any]] = {
    "narratives": {
        "derivatives_path": "fmri/original/narratives/afni-nosmooth",
        "load_arrays": False,
    },
}


#: Participant ids a loader uses when the release ships averaged series.
_AVERAGED_IDS = frozenset({"averaged", "mean", "average"})


def counts_for(dataset: Any) -> dict[str, Any]:
    """Count participants, distinct stimuli, annotation units and characters.

    Stimuli are de-duplicated by text, giving corpus sizes rather than trial
    counts. ``units`` names the word layer where a corpus has one.

    Args:
        dataset: An already-loaded dataset, or anything exposing ``stimuli()``
            and ``participants``.

    Returns:
        Participant, stimulus, unit, and character totals, plus per-layer unit
        counts and a flag for releases that ship one averaged series per item.
    """
    unique = {s.text: s for s in dataset.stimuli()}
    by_layer: dict[str, int] = {}
    for stimulus in unique.values():
        for name, layer in stimulus.layers.items():
            by_layer[name] = by_layer.get(name, 0) + len(layer)
    if "words" in by_layer:
        units = by_layer["words"]
    else:
        units = next(iter(by_layer.values())) if len(by_layer) == 1 else 0

    ids = list(getattr(dataset, "participants", ()) or ())
    return {
        "participants": len(ids),
        # A release that ships one averaged series per item counts as one
        # participant; say so rather than let the 1 read as a cohort size.
        "participants_averaged": bool(ids) and set(ids) <= _AVERAGED_IDS,
        "stimuli": len(unique),
        "units": units,
        "units_by_layer": by_layer,
        "characters": sum(len(text) for text in unique),
    }


def load_for_counting(key: str) -> Any:
    """Construct and load one registered dataset against its default path.

    Args:
        key: Registry name of the corpus to load.
    """
    spec = registry.get(key)
    dataset = spec.loader_cls()(**spec.init_kwargs)
    kwargs = dict(_LOAD_KWARGS.get(key, {}))
    if "derivatives_path" in kwargs:
        kwargs["derivatives_path"] = str(
            registry.data_dir() / kwargs["derivatives_path"]
        )
    dataset.load(str(spec.default_path()), **kwargs)
    return dataset


def _classify(message: str) -> str:
    lowered = message.lower()
    if any(marker.lower() in lowered for marker in _DATA_ABSENT):
        return "data-absent"
    return "error"


def _probe_counts(key: str, timeout: int) -> tuple[dict[str, Any] | None, str, str]:
    """Count one corpus in a subprocess; return (counts, status, detail).

    Isolation keeps a hanging loader from taking the sweep with it. Checksums
    are skipped: a full re-hash changes no count.
    """
    env = {**os.environ, "PSYCHO_SKIP_CHECKSUMS": "1"}
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "replica.dataset.catalog.facts", "--count-one", key],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return None, "error", f"timed out after {timeout}s"

    for line in proc.stdout.splitlines():
        if line.startswith(_MARKER):
            payload = json.loads(line[len(_MARKER) :])
            if "counts" in payload:
                return payload["counts"], "ok", ""
            detail = payload.get("error", "unknown failure")
            return None, _classify(detail), detail

    detail = (proc.stderr.strip().splitlines() or ["no output"])[-1]
    return None, _classify(detail), detail


def _access(download_keys: tuple[str, ...]) -> tuple[bool, str]:
    """Return whether access is gated, and the catalog's access instructions."""
    from replica.dataset.catalog.sources import DATASETS

    for key in download_keys:
        entry = DATASETS.get(key)
        if entry and entry.get("source") == "manual":
            return True, str(entry.get("instructions", "")).strip()
    return False, ""


def _yaml_facts(path: Path) -> tuple[str, dict[str, Any] | None]:
    """Resolve the paradigm-scoped YAML for a data directory, without the data.

    Fields are read as declared: ``metadata.to_dict`` drops default-valued
    fields and would hide ``language: en``.
    """
    yaml_path = find_dataset_yaml(path)
    if yaml_path is None:
        return "", None
    relative = yaml_path.relative_to(Path(__file__).parent.parent / "metadata" / "yaml")
    return str(relative), dataclasses.asdict(load_yaml(yaml_path))


def _facts_one(
    key: str,
    *,
    measure: bool,
    timeout: int,
    snapshot: dict[str, Any] | None,
) -> dict[str, Any]:
    spec = registry.get(key)
    path = spec.default_path()
    record: dict[str, Any] = {
        "key": key,
        "paradigm": spec.paradigm,
        "loader": spec.loader_class,
        "loader_module": spec.loader_module,
        "default_subdir": spec.default_subdir,
        "init_kwargs": dict(spec.init_kwargs),
        "path": str(path),
        "path_exists": path.exists(),
    }

    try:
        described = describe_dataset(key)
    except Exception as exc:  # report the failure and keep the other facts
        record["describe_error"] = f"{type(exc).__name__}: {exc}"
        described = None
    if described is not None:
        record.update(
            {
                "measures": list(described.measures),
                "default_measure": described.default_measure,
                "presentation": list(described.presentation),
                "fixations": described.fixations,
                "download_keys": list(described.download_keys),
                "parameters": [dataclasses.asdict(p) for p in described.parameters],
                "variants": [dataclasses.asdict(v) for v in described.variants],
            }
        )
        gated, instructions = _access(described.download_keys)
        record["gated"] = gated
        record["access_instructions"] = instructions

    record["yaml"], record["metadata"] = _yaml_facts(path)
    # Census the files only while measuring, so the file report and the counts
    # always come from the same machine; otherwise a builder's partial copy
    # would annotate sizes measured elsewhere.
    declared_files = (record["metadata"] or {}).get("files") or []
    data_dir = resolve_data_dir(path, metadata_path=find_dataset_yaml(path))
    if not declared_files and data_dir != path:
        # A view that shares another release's files inherits its census, so a
        # partial download is annotated on both views.
        _, shared = _yaml_facts(data_dir)
        declared_files = (shared or {}).get("files") or []
    if measure and declared_files and data_dir.exists():
        record["files_on_disk"] = file_presence(data_dir, declared_files)

    counts: dict[str, Any] | None = None
    source = ""
    if measure:
        if not data_dir.exists():
            record["probe_status"] = "data-absent"
            record["probe_detail"] = f"{spec.default_subdir} is not present"
        else:
            counts, status, detail = _probe_counts(key, timeout)
            record["probe_status"] = status
            record["probe_detail"] = detail
            source = "measured" if counts else ""
    if counts is None and snapshot:
        stored = snapshot.get("counts", {}).get(key)
        if stored:
            counts, source = stored, "snapshot"
    if snapshot and "files_on_disk" not in record:
        # A docs build sees no data, so file presence comes from the snapshot
        # written where the counts were computed.
        stored_files = snapshot.get("files", {}).get(key)
        if stored_files:
            record["files_on_disk"] = stored_files
    record["counts"] = counts
    record["counts_source"] = source
    return record


def load_snapshot(path: Path | None = None) -> dict[str, Any]:
    """Read the committed counts snapshot, or an empty one if absent.

    Per-count configuration and manifest paths are relative to the ``replica``
    package; input files and directory listings are relative to the data root.
    Declared checksums come from the configuration and any input manifests.
    Observed-file hashes may omit native-library reads. Source fingerprints
    identify the named functions and loader modules, not transitive dependencies.

    Args:
        path: Snapshot file to read; ``None`` reads the one shipped beside this
            module.
    """
    path = path or SNAPSHOT_PATH
    if not path.is_file():
        return {"provenance": {}, "counts": {}, "unmeasured": {}, "files": {}}
    return json.loads(path.read_text())


def snapshot_drift(
    snapshot: dict[str, Any] | None = None,
    *,
    measured_counts: dict[str, dict[str, Any]] | None = None,
) -> dict[str, list[str]]:
    """Registered corpora the snapshot misses, and entries the registry lacks.

    Either one fails a docs build rather than rendering a wrong size.

    Args:
        snapshot: An already-read snapshot; ``None`` reads the committed one.
        measured_counts: Fresh counts to compare with stored values, when available.

    Returns:
        ``missing`` for registered keys the snapshot covers neither as counted
        nor as unmeasured, ``stale`` for snapshot keys the registry dropped.
        When fresh counts are supplied, ``changed`` lists value or schema drift.
    """
    snap = snapshot if snapshot is not None else load_snapshot()
    covered = set(snap.get("counts", {})) | set(snap.get("unmeasured", {}))
    live = set(registry.names())
    drift = {
        "missing": sorted(live - covered),
        "stale": sorted(covered - live),
    }
    if measured_counts is not None:
        drift["changed"] = sorted(
            key
            for key, counts in measured_counts.items()
            if snap.get("counts", {}).get(key) != counts
        )
    return drift


def dataset_facts(
    keys: list[str] | None = None,
    *,
    measure: bool = False,
    timeout: int = DEFAULT_TIMEOUT,
    use_snapshot: bool = True,
    on_record: Callable[[str, dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Collect facts for *keys* (default: every registered corpus).

    ``measure=False`` fills counts from the snapshot; ``measure=True`` loads
    each corpus, calling ``on_record`` per corpus so a sweep reports progress.

    Args:
        keys: Registry names to cover; ``None`` covers every registered corpus.
        measure: Load each corpus in a subprocess and count it, rather than
            reading its counts off the snapshot.
        timeout: Seconds one corpus's counting subprocess may run before it is
            recorded as an error.
        use_snapshot: Fall back to the committed snapshot wherever a corpus was
            not counted in this call.
        on_record: Called with each key and its finished record as the sweep
            goes, for progress output and incremental persistence.

    Returns:
        ``{"datasets": {key: record}}``, one record per requested corpus.
    """
    snapshot = load_snapshot() if use_snapshot else None
    names = list(keys) if keys else registry.names()
    collected = {}
    for key in names:
        record = _facts_one(key, measure=measure, timeout=timeout, snapshot=snapshot)
        collected[key] = record
        if on_record is not None:
            on_record(key, record)
    return {"datasets": collected}


def _provenance() -> dict[str, str]:
    """Identify the count function independently of checkout history."""
    return {
        "generated": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "producer": f"{counts_for.__module__}.{counts_for.__qualname__}",
        "producer_sha256": hashlib.sha256(
            inspect.getsource(counts_for).encode("utf-8")
        ).hexdigest(),
        "input_checksums": "not-verified",
        "data_root": "PSYCHO_DATA_DIR"
        if os.environ.get(registry.DATA_DIR_ENV)
        else "default data directory",
    }


def _load_attempted(record: dict[str, Any]) -> bool:
    """Whether a facts record comes from a pass that tried to load its corpus."""
    return "probe_status" in record or record.get("counts_source") == "measured"


def write_snapshot(
    facts: dict[str, Any], path: Path | None = None, *, merge: bool = False
) -> Path:
    """Persist the measured tier of *facts* plus provenance.

    ``merge`` keeps entries *facts* does not cover, so a resumed sweep adds.

    Args:
        facts: A ``dataset_facts`` result whose records carry counts, probe
            status, file presence and optional ``count_provenance`` from the
            measurement.
        path: Destination file; ``None`` writes the one shipped beside this
            module.
        merge: Start from the file already there instead of rebuilding it.

    Returns:
        The path written.

    Raises:
        ValueError: A rebuild (``merge=False``) got a record that attempted no
            load, whose counts it would drop.
    """
    if not merge:
        unattempted = sorted(
            key
            for key, record in facts["datasets"].items()
            if not _load_attempted(record)
        )
        if unattempted:
            raise ValueError(
                f"rebuilding the snapshot needs measured records; {unattempted} "
                "attempted no load (measure them, or pass merge=True)"
            )
    path = path or SNAPSHOT_PATH
    existing = load_snapshot(path) if merge else {}
    counts: dict[str, Any] = dict(existing.get("counts", {}))
    unmeasured: dict[str, str] = dict(existing.get("unmeasured", {}))
    files: dict[str, Any] = dict(existing.get("files", {}))
    provenance = _provenance()
    count_provenance = {
        key: existing.get("count_provenance", {}).get(
            key, existing.get("provenance", {})
        )
        for key in counts
    }
    for key, record in facts["datasets"].items():
        # Only a record from a load attempt rewrites its counts, and a rebuild
        # refuses any other record, so a file-presence refresh cannot discard
        # measured sizes.
        if _load_attempted(record):
            if record.get("counts_source") == "measured":
                counts[key] = record["counts"]
                count_provenance[key] = record.get("count_provenance", provenance)
                unmeasured.pop(key, None)
            else:
                status = record["probe_status"]
                detail = record.get("probe_detail", "")
                unmeasured[key] = f"{status}: {detail}" if detail else status
                counts.pop(key, None)
                count_provenance.pop(key, None)
        if record.get("files_on_disk"):
            files[key] = record["files_on_disk"]
    payload = {
        "provenance": provenance,
        "count_provenance": count_provenance,
        "counts": counts,
        "unmeasured": unmeasured,
        "files": files,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def _record_line(key: str, record: dict[str, Any]) -> str:
    counts = record.get("counts")
    if counts:
        return (
            f"  {key:28s} part={counts['participants']:6d} "
            f"stim={counts['stimuli']:6d} units={counts['units']:8d} "
            f"chars={counts['characters']:9d} [{record['counts_source']}]"
        )
    status = record.get("probe_status", record.get("counts_source") or "-")
    detail = record.get("probe_detail") or "no counts (pass --measure)"
    return f"  {key:28s} {status}: {detail}"


def _summarize(facts: dict[str, Any]) -> str:
    lines = []
    by_status: dict[str, int] = {}
    for key, record in sorted(facts["datasets"].items()):
        status = record.get("probe_status", record.get("counts_source") or "-")
        by_status[status] = by_status.get(status, 0) + 1
        lines.append(_record_line(key, record))
    lines.append("")
    lines.append("  " + ", ".join(f"{k}={v}" for k, v in sorted(by_status.items())))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Command line entry point, also used for the per-corpus child process.

    Args:
        argv: Command-line arguments; defaults to ``sys.argv[1:]``.

    Returns:
        0; a corpus that fails to load is reported rather than raised.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count-one", metavar="KEY", help=argparse.SUPPRESS)
    parser.add_argument("datasets", nargs="*", help="Corpora (default: all)")
    parser.add_argument(
        "--measure", action="store_true", help="Load each corpus and count it"
    )
    parser.add_argument(
        "--write-snapshot",
        action="store_true",
        help=f"Write measured counts to {SNAPSHOT_PATH.name} (requires --measure)",
    )
    parser.add_argument("--json", action="store_true", help="Emit the full facts")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument(
        "--only-missing",
        action="store_true",
        help="Measure only corpora the snapshot has no counts for, and merge",
    )
    args = parser.parse_args(argv)
    if args.write_snapshot and not args.measure:
        parser.error("--write-snapshot requires --measure")

    if args.count_one:
        try:
            payload: dict[str, Any] = {
                "counts": counts_for(load_for_counting(args.count_one))
            }
        except Exception as exc:  # reported to the parent process
            head = (str(exc).splitlines() or [""])[0]
            payload = {"error": f"{type(exc).__name__}: {head}"}
        print(_MARKER + json.dumps(payload))
        return 0

    keys = args.datasets or registry.names()
    if args.only_missing:
        measured = set(load_snapshot().get("counts", {}))
        keys = [key for key in keys if key not in measured]
        print(f"  measuring {len(keys)} corpora without counts", flush=True)

    def progress(key: str, record: dict[str, Any]) -> None:
        print(_record_line(key, record), flush=True)
        # Persist as we go: a sweep over every corpus takes hours, and an
        # interrupted run should leave its finished counts behind.
        if args.write_snapshot:
            write_snapshot({"datasets": {key: record}}, merge=True)

    facts = dataset_facts(
        keys,
        measure=args.measure,
        timeout=args.timeout,
        on_record=progress if args.measure and not args.json else None,
    )
    if args.json:
        print(json.dumps(facts, indent=2, sort_keys=True))
    else:
        print(_summarize(facts))
    if args.write_snapshot:
        # Measuring a subset adds to the snapshot; only a full sweep may
        # rebuild it, so naming one corpus cannot discard the rest.
        merge = bool(args.only_missing or args.datasets)
        path = write_snapshot(facts, merge=merge)
        print(f"\nwrote {path} ({'merged' if merge else 'rebuilt'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
