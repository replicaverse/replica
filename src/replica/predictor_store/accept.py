"""Publish a quarantined cell after its investigation: ``replica predictors accept``.

The payload is published as ``<tag>.pkl`` unchanged; the manifest becomes
``completed`` with the audit and the reason, so the export stages the cell and
a later run of the same identity reuses it.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

from replica.artifacts import SCHEMA_VERSION, manifest_path, write_manifest
from replica.compute.validation import predictor_pickle_issues
from replica.utils.digest import file_sha256

SUSPECT = ".suspect.pkl"

#: Audit buckets no acceptance covers: the payload holds nothing to publish.
UNPUBLISHABLE = frozenset({"unreadable", "empty_pickle"})


def accept(path: Path, reason: str) -> Path:
    """Publish the quarantined cell at *path* with *reason* recorded.

    Args:
        path: The ``<tag>.suspect.pkl`` of the cell.
        reason: What the investigation found; recorded under ``audit.accepted``.

    Returns:
        The published ``<tag>.pkl``.

    Raises:
        ValueError: Not a quarantined cell of the current schema, a changed or
            unreadable payload, an empty reason, or a run short of its stimuli.
        FileExistsError: A completed ``<tag>.pkl`` exists.
    """
    path = Path(path)
    if not path.name.endswith(SUSPECT):
        raise ValueError(f"{path.name} is not a quarantined cell ({SUSPECT})")
    if not reason.strip():
        raise ValueError("an acceptance needs a reason")
    sidecar = manifest_path(path)
    manifest = _quarantined_manifest(path, sidecar)
    stem = path.name[: -len(SUSPECT)]
    target = path.with_name(stem + ".pkl")
    if target.exists():
        raise FileExistsError(
            f"{target.name} exists; the quarantined cell is superseded"
        )
    audit = manifest.get("audit")
    if audit is None:
        audit = _audit_from_files(path, stem, manifest)
    unpublishable = UNPUBLISHABLE.intersection(audit["hard_issues"])
    if unpublishable:
        raise ValueError(
            f"{path.name}: the payload is {', '.join(sorted(unpublishable))}"
        )
    if audit["failed_stimuli"] or audit["missing_stimuli"]:
        raise ValueError(
            f"{path.name}: {audit['failed_stimuli']} stimuli failed; the payload "
            f"holds {audit['missing_stimuli']} fewer than requested; recompute "
            "the cell instead"
        )
    audit = {
        **audit,
        "accepted": {
            "reason": reason.strip(),
            "at": datetime.now(UTC).isoformat(timespec="seconds"),
        },
    }
    # The payload is linked, then its manifest written, then the quarantined
    # files removed, so an interruption leaves a complete cell at every step.
    os.link(path, target)
    write_manifest(
        target,
        manifest["identity"],
        status="completed",
        environment=manifest.get("environment"),
        window=manifest.get("window"),
        sampling=manifest.get("sampling"),
        derivation=manifest.get("derivation"),
        multiword=manifest.get("multiword"),
        audit=audit,
        external=manifest.get("external"),
    )
    path.unlink()
    sidecar.unlink()
    # The checkpoint of the run duplicates rows of the published cell.
    partial = target.with_name(target.name + ".partial")
    partial.unlink(missing_ok=True)
    manifest_path(partial).unlink(missing_ok=True)
    return target


def _quarantined_manifest(path: Path, sidecar: Path) -> dict:
    """The manifest of a quarantined cell whose payload matches it."""
    try:
        manifest = json.loads(sidecar.read_text())
        identity = manifest["identity"]
        schema = identity["schema_version"]
        n_stimuli = identity["n_stimuli"]
        status = manifest["status"]
        digest = manifest["payload_sha256"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ValueError(
            f"{path.name}: unreadable or incomplete manifest ({exc})"
        ) from exc
    if schema != SCHEMA_VERSION:
        raise ValueError(
            f"{path.name}: manifest schema {schema!r} is not {SCHEMA_VERSION}"
        )
    if not isinstance(n_stimuli, int):
        raise ValueError(f"{path.name}: manifest records no stimulus count")
    if status != "quarantined":
        raise ValueError(f"{path.name}: manifest status is {status!r}")
    if digest != file_sha256(path):
        raise ValueError(f"{path.name}: payload differs from its manifest's digest")
    return manifest


def _audit_from_files(path: Path, stem: str, manifest: dict) -> dict:
    """Audit block of a cell whose manifest has none, from the payload.

    A failure log beside the cell names stimuli the run did not score, so the
    cell is refused before the payload is read.
    """
    fail_log = path.with_name(stem + ".failed.log")
    if fail_log.exists():
        raise ValueError(
            f"{path.name}: the run left {fail_log.name}; recompute the cell instead"
        )
    report = predictor_pickle_issues(path)
    missing = max(0, manifest["identity"]["n_stimuli"] - report.n_stims)
    return report.record(failed=0, missing=missing)
