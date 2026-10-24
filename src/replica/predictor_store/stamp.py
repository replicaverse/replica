"""Bind a pickle produced outside the compute driver to a manifest.

``stamp`` reads the tag for the dataset, method and configuration and takes
the model identity from the caller. It obtains the corpus digest and stimulus
count from the loader, audits the payload as the driver does, and writes the
manifest the export requires:
``completed`` beside the pickle, or ``quarantined`` beside the pickle renamed
to ``<tag>.suspect.pkl`` when the audit finds hard issues or the payload is
short of the corpus.
"""

from __future__ import annotations

import os
import pickle
from datetime import UTC, datetime
from pathlib import Path

from replica.artifacts import SCHEMA_VERSION, manifest_path, write_manifest
from replica.compute.artifacts import (
    environment_record,
    package_versions,
    source_digest,
)
from replica.compute.units import METHOD_STORAGE
from replica.compute.validation import predictor_pickle_issues
from replica.dataset.catalog.provenance import corpus_fingerprint
from replica.utils.digest import file_sha256
from replica.utils.naming import safe_lm

from .corpora import Corpora, RegistryCorpora
from .layout import decode_tag, match_dataset

SUSPECT = ".suspect.pkl"


def stamp(
    path: Path,
    *,
    model_id: str,
    inputs: list[Path],
    reason: str,
    corpora: Corpora | None = None,
) -> Path:
    """Write the manifest of the externally produced cell at *path*.

    Args:
        path: The cell's ``<tag>.pkl``, without a manifest.
        model_id: Exact model or frequency-source identity supplied by the
            producer; its filename encoding must match the tag's model slot.
        inputs: Files the payload was derived from, recorded with their digests.
        reason: What produced the payload; recorded under ``external``.
        corpora: Where the loader's stimuli come from; the dataset registry
            when ``None``.

    Returns:
        The payload the manifest binds: *path*, or its ``.suspect.pkl`` when
        the cell is quarantined.

    Raises:
        ValueError: A manifest exists, the tag names no registry dataset, the
            reason is empty, an input is missing, the payload is unreadable or
            holds stimuli the loader does not produce, or the supplied model
            identity is empty or disagrees with the filename.
        FileExistsError: The quarantine payload or its manifest already exists.
    """
    path = Path(path)
    if not path.name.endswith(".pkl") or path.name.endswith(SUSPECT):
        raise ValueError(f"{path.name} is not a cell pickle")
    if not reason.strip():
        raise ValueError("a stamp needs a reason")
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("model_id must be a nonempty string")
    if manifest_path(path).exists():
        raise ValueError(
            f"{path.name} carries a manifest; stamp binds an unstamped payload"
        )
    missing_inputs = [str(p) for p in inputs if not Path(p).is_file()]
    if missing_inputs:
        raise ValueError(f"input not found: {', '.join(missing_inputs)}")
    corpora = corpora or RegistryCorpora()
    tag = path.name[: -len(".pkl")]
    dataset = match_dataset(tag, corpora.names())
    if dataset is None:
        raise ValueError(f"{path.name}: no registry dataset in the tag")
    info = decode_tag(tag, dataset, model_id=model_id)
    if safe_lm(model_id) != info.model_slot:
        raise ValueError("model_id conflicts with the filename model slot")
    payload = _payload(path)
    stimuli = corpora.stimuli(dataset)
    ids = [s.stimulus_id for s in stimuli]
    extra = sorted(set(payload) - set(ids))
    if extra:
        raise ValueError(
            f"{path.name}: {len(extra)} stimuli the loader does not produce ({extra[0]})"
        )
    storage = METHOD_STORAGE.get(info.method)
    identity = {
        "schema_version": SCHEMA_VERSION,
        "method": info.method,
        "storage_class": storage.storage_class if storage else None,
        "layer": storage.layer if storage else None,
        "options": {
            **dict(info.config),
            "model_id": model_id,
            "unit": info.unit,
            "context_scope": info.context_scope,
        },
        "model": {"source": model_id},
        "corpus_sha256": corpus_fingerprint(
            dataset, stimuli, info.unit, info.context_scope
        ),
        "n_stimuli": len(stimuli),
        "source_sha256": source_digest(),
        "versions": package_versions(),
    }
    if storage is not None and storage.study is not None:
        identity["study"] = storage.study
    external = {
        "reason": reason.strip(),
        "at": datetime.now(UTC).isoformat(timespec="seconds"),
        "inputs": [
            {"path": Path(p).name, "sha256": file_sha256(Path(p))} for p in inputs
        ],
    }
    report = predictor_pickle_issues(path)
    missing = len(ids) - len(payload)
    incomplete = missing > 0 or not payload
    audit = (
        report.record(failed=0, missing=missing)
        if report.issues or incomplete
        else None
    )
    environment = environment_record()
    if report.has_hard_issues or incomplete:
        target = path.with_name(tag + SUSPECT)
        if any(
            candidate.exists() or candidate.is_symlink()
            for candidate in (target, manifest_path(target))
        ):
            raise FileExistsError(f"{target.name}: quarantine destination is occupied")
        os.link(path, target)
        write_manifest(
            target,
            identity,
            status="quarantined",
            environment=environment,
            external=external,
            audit=audit,
        )
        path.unlink()
        return target
    write_manifest(
        path,
        identity,
        status="completed",
        environment=environment,
        external=external,
        audit=audit,
    )
    return path


def _payload(path: Path) -> dict:
    try:
        with path.open("rb") as stream:
            payload = pickle.load(stream)
    except (OSError, pickle.UnpicklingError, EOFError, AttributeError) as exc:
        raise ValueError(f"{path.name}: unreadable payload ({exc})") from exc
    if not isinstance(payload, dict):
        raise ValueError(
            f"{path.name}: payload is {type(payload).__name__}, not a stimulus mapping"
        )
    return payload
