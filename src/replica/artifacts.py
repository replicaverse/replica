"""Shared manifest schema and integrity checks for computed payloads.

Readers and producers use this contract without importing model construction.
"""

import json
from pathlib import Path

from .utils.digest import file_sha256

SCHEMA_VERSION = 1


def manifest_text(manifest: dict) -> str:
    """The bytes a manifest is written and staged as.

    Args:
        manifest: The manifest to serialise.
    """
    return json.dumps(manifest, indent=2) + "\n"


def manifest_path(path: Path) -> Path:
    """Sidecar location for a completed artifact or checkpoint.

    Args:
        path: Payload file.
    """
    return path.with_name(path.name + ".manifest.json")


def write_manifest(
    path: Path,
    identity: dict,
    *,
    status: str,
    environment: dict | None = None,
    window: dict | None = None,
    sampling: dict | None = None,
    derivation: dict | None = None,
    multiword: dict | None = None,
    audit: dict | None = None,
    external: dict | None = None,
) -> None:
    """Atomically bind an existing payload to its identity and checksum.

    Args:
        path: Fully written payload file.
        identity: Resolved run identity.
        status: completed, partial or quarantined.
        environment: The producing process's ``environment_record``.
        window: The context window rule a token table's forward pass applied.
        sampling: For a sampled positional token table: the positions
            estimated, the seed and how the draws are seeded.
        derivation: For a table aggregated from a finer one: the source
            table's name and digest, the aggregation rules and the commit.
        multiword: The rule units of several whitespace words were valued
            under, and their count.
        audit: The pickle audit's buckets, examples and failed and missing
            counts, plus the acceptance that published a quarantined cell.
        external: For a payload produced outside the compute driver: the
            reason, the time of the stamp and the inputs with their digests.
    """
    manifest = {
        "identity": identity,
        "status": status,
        "payload_sha256": file_sha256(path),
    }
    if environment is not None:
        manifest["environment"] = environment
    if window is not None:
        manifest["window"] = window
    if sampling is not None:
        manifest["sampling"] = sampling
    if derivation is not None:
        manifest["derivation"] = derivation
    if multiword is not None:
        manifest["multiword"] = multiword
    if audit is not None:
        manifest["audit"] = audit
    if external is not None:
        manifest["external"] = external
    target = manifest_path(path)
    temp = target.with_suffix(".tmp")
    temp.write_text(manifest_text(manifest))
    temp.replace(target)


def validate_artifact(path: Path) -> dict | None:
    """Verify a payload's provenance sidecar and return the identity it records.

    Args:
        path: Payload to read. A missing sidecar returns None; an invalid or
            incomplete sidecar raises ValueError instead of claiming provenance.
    """
    sidecar = manifest_path(path)
    if not sidecar.exists():
        return None
    try:
        manifest = json.loads(sidecar.read_text())
        identity = manifest["identity"]
        if (
            identity["schema_version"] != SCHEMA_VERSION
            or manifest["status"] != "completed"
            or manifest["payload_sha256"] != file_sha256(path)
        ):
            raise ValueError("unsupported, incomplete or mismatched manifest")
        return identity
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ValueError(f"{path}: invalid artifact manifest ({exc})") from exc


def matching_manifest(path: Path, identity: dict, *, status: str) -> bool:
    """True when the sidecar matches *identity*, *status* and the payload's checksum.

    Args:
        path: Payload to validate.
        identity: Identity the current run requires.
        status: Required lifecycle status.
    """
    try:
        manifest = json.loads(manifest_path(path).read_text())
        return (
            manifest["identity"] == identity
            and manifest["status"] == status
            and manifest["payload_sha256"] == file_sha256(path)
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False
