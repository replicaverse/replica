"""Byte receipts and upload snapshots for verified predictor stages."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path, PurePosixPath
from tempfile import NamedTemporaryFile
from typing import Any

from replica.utils.digest import file_sha256
from replica.utils.naming import canonical_tag

from . import layout

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_EVIDENCE = {layout.PUBLISH_FILE, layout.READBACK_FILE, layout.HISTORY_FILE}


def _relative_file(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or PurePosixPath(value).is_absolute()
        or PurePosixPath(value).as_posix() != value
        or any(part in {".", ".."} for part in value.split("/"))
        or any(char in value for char in "\\*?[]\x00\n\r")
    ):
        raise ValueError(f"invalid stage-relative file: {value!r}")
    return value


def _digest_map(value: object) -> dict[str, str]:
    if not isinstance(value, dict) or any(
        not isinstance(key, str)
        or not key
        or not isinstance(digest, str)
        or _SHA256.fullmatch(digest) is None
        for key, digest in value.items()
    ):
        raise ValueError("readback requires a mapping of names to SHA-256 digests")
    return value


def _read_plan(stage: Path) -> tuple[dict[str, Any], bytes]:
    raw = (stage / layout.PUBLISH_FILE).read_bytes()
    plan = json.loads(raw)
    if not isinstance(plan, dict) or "fresh_cells" not in plan:
        raise ValueError("stage lacks fresh_cells; re-export and verify it")
    fresh = _digest_map(plan["fresh_cells"])
    if any(canonical_tag(tag) != tag for tag in fresh):
        raise ValueError("fresh_cells must use canonical tags")
    for key in ("written", "deleted"):
        files = plan.get(key)
        if not isinstance(files, list):
            raise ValueError(f"publish plan lacks a {key} file list")
        checked = [_relative_file(file) for file in files]
        if len(set(checked)) != len(checked) or set(checked) & _EVIDENCE:
            raise ValueError(f"invalid or repeated {key} files in publish plan")
    required = {layout.INDEX_FILE, layout.COVERAGE_FILE, layout.README_FILE}
    if not required <= set(plan["written"]):
        raise ValueError("publish plan omits required stage files")
    if set(plan["written"]) & set(plan["deleted"]):
        raise ValueError("publish plan both writes and deletes a file")
    return plan, raw


def _stage_file(stage: Path, file: str) -> Path:
    path = stage / file
    if not path.resolve().is_relative_to(stage.resolve()):
        raise ValueError(f"stage file escapes its directory: {file}")
    return path


def _state(stage: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    plan, raw = _read_plan(stage)
    return plan, {
        "schema_version": 1,
        "status": "passed",
        "plan_sha256": hashlib.sha256(raw).hexdigest(),
        "fresh_cells": plan["fresh_cells"],
        "files": {
            file: file_sha256(_stage_file(stage, file)) for file in plan["written"]
        },
    }


def _copy_verified(stage: Path, destination: Path) -> dict[str, Any]:
    """Copy only receipt-bound bytes, rejecting stale or incomplete evidence."""
    try:
        raw_receipt = (stage / layout.READBACK_FILE).read_bytes()
    except FileNotFoundError as exc:
        raise ValueError(
            "stage lacks readback.json; run verify before publishing"
        ) from exc
    receipt = json.loads(raw_receipt)
    plan, raw_plan = _read_plan(stage)
    if (
        not isinstance(receipt, dict)
        or type(receipt.get("schema_version")) is not int
        or receipt["schema_version"] != 1
        or receipt.get("status") != "passed"
        or receipt.get("plan_sha256") != hashlib.sha256(raw_plan).hexdigest()
        or receipt.get("fresh_cells") != plan["fresh_cells"]
    ):
        raise ValueError(
            "readback receipt does not match the publish plan; rerun verify"
        )
    files = _digest_map(receipt.get("files"))
    if set(files) != set(plan["written"]):
        raise ValueError("readback receipt does not cover all written files")
    for file, expected in files.items():
        target = destination / file
        target.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        with _stage_file(stage, file).open("rb") as source, target.open("wb") as output:
            for block in iter(lambda: source.read(1 << 20), b""):
                digest.update(block)
                output.write(block)
        if digest.hexdigest() != expected:
            raise ValueError(f"{file}: bytes changed since readback; rerun verify")
    (destination / layout.PUBLISH_FILE).write_bytes(raw_plan)
    (destination / layout.READBACK_FILE).write_bytes(raw_receipt)
    return plan


def _write_receipt(stage: Path, state: dict[str, Any]) -> None:
    with NamedTemporaryFile(
        mode="w", dir=stage, prefix=".readback-", delete=False
    ) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(json.dumps(state, indent=2) + "\n")
            handle.close()
            temporary.replace(stage / layout.READBACK_FILE)
        finally:
            temporary.unlink(missing_ok=True)
