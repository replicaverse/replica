"""Validate explicit website release receipts.

A release receipt approves a frozen registry entry and its source references.
Missing receipts keep computed results off the site. Artifact storage, methodological
eligibility and agreement with a paper do not authorize website publication.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

from .registry import CORRECTION_KEYS
from .values import validate_artifact_schema, verify_artifact

STAGES = frozenset(
    {"repetition", "reproduction", "replication_repetition", "replication"}
)
REPO_ROOT = Path(__file__).resolve().parents[4]


def evidence_digest(claim: dict, sources: dict, results: dict, stage: str) -> str:
    """Bind a stage approval to the claim, references and reviewed transfer data.

    The complete claim is bound because corrections, scope and comparison rules
    can affect more than one stage. Release declarations are excluded so adding
    another approval does not invalidate the first one.

    Args:
        claim: Registered claim, including its evidence and comparison rules.
        sources: The study’s pinned paper and implementation references.
        results: Transfer results after protocol review.
        stage: Registry stage being approved.
    """
    if stage not in STAGES:
        raise ValueError(f"Unknown website stage: {stage}")
    payload = {
        "stage": stage,
        "claim": {k: v for k, v in claim.items() if k != "website_release"},
        "sources": sources,
        "transfer": results.get("claims", {}).get(claim["id"]),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def released_stages(
    registry: dict, results: dict, *, root: Path | None = None
) -> dict[str, dict[str, dict]]:
    """Read checked, repository-local receipts; reject stale or malformed approvals.

    Receipts are small committed review records, so rendering requires neither
    result downloads nor statistical execution. Their verification text
    identifies the independent checks; their digest pins the exact public values,
    artifact references, methodology and paper version that were reviewed.

    Args:
        registry: Claims and pinned study sources.
        results: Transfer results after protocol review.
        root: Checkout root for receipts and local artifacts.
    """
    root = REPO_ROOT if root is None else root
    released: dict[str, dict[str, dict]] = {}
    for claim in registry["claims"]:
        declarations = claim.get("website_release", {})
        if not isinstance(declarations, dict) or set(declarations) - STAGES:
            raise ValueError(f"Invalid website release stages: {claim['id']}")
        for stage, reference in declarations.items():
            if not isinstance(reference, dict) or set(reference) != {"path", "sha256"}:
                raise ValueError("Website release needs a receipt path and SHA-256")
            path = (root / reference["path"]).resolve()
            expected_root = (root / "src/replica/studies" / claim["study"]).resolve()
            if not path.is_relative_to(expected_root) or not path.is_file():
                raise ValueError(
                    "Website receipt must exist within its study directory"
                )
            digest = reference["sha256"]
            if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ValueError("Invalid website receipt SHA-256")
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != digest:
                raise ValueError(
                    f"Website receipt checksum drift: {claim['id']} / {stage}"
                )
            receipt = json.loads(raw)
            # A study can keep all reviewed stages in one checksummed record.
            # Each selected receipt still passes the same independent checks.
            if (
                isinstance(receipt, dict)
                and receipt.get("schema") == "replica.website-release.v1"
            ):
                if (
                    set(receipt) != {"schema", "study", "claims"}
                    or receipt["study"] != claim["study"]
                    or not isinstance(receipt["claims"], dict)
                    or not isinstance(receipt["claims"].get(claim["id"]), dict)
                ):
                    raise ValueError("Invalid study website release record")
                receipt = receipt["claims"][claim["id"]].get(stage)
            required = {
                "status",
                "evidence_sha256",
                "scope",
                "verification",
                "verified_artifacts",
            }
            if not isinstance(receipt, dict) or set(receipt) != required:
                raise ValueError("Invalid website receipt fields")
            if receipt["status"] != "finalized" or any(
                not isinstance(receipt[key], str) or not receipt[key].strip()
                for key in ("scope", "verification")
            ):
                raise ValueError(
                    "Website release requires finalized independent verification"
                )
            sources = registry.get("study_sources", {}).get(claim["study"])
            if not sources or not sources.get("paper", {}).get("audit_document"):
                raise ValueError("Website release requires the pinned paper reference")
            response_groups(claim, require_complete=True)
            verify_release_artifacts(claim, sources, results, stage, receipt, root)
            if receipt["evidence_sha256"] != evidence_digest(
                claim, sources, results, stage
            ):
                raise ValueError(
                    f"Website evidence changed since review: {claim['id']} / {stage}"
                )
            released.setdefault(claim["id"], {})[stage] = receipt
    return released


def stage_artifacts(claim: dict, stage: str, results: dict) -> list[tuple[dict, dict]]:
    """Find the artifacts that ground the released stage and its paper values.

    Args:
        claim: Claim whose stage is being released.
        stage: Registry stage to inspect.
        results: Reviewed transfer results.
    """
    nodes = [claim["evidence"]["paper"]]
    if stage == "repetition":
        nodes.append(claim["evidence"].get("repetition", claim["evidence"]["replica"]))
    elif stage == "reproduction":
        corrections = [
            claim[key]["corrected"]
            for key in CORRECTION_KEYS
            if claim.get(key, {}).get("corrected")
        ]
        nodes.extend(corrections or [claim["evidence"]["replica"]])
    else:
        entry = results.get("claims", {}).get(claim["id"], {})
        nodes.append(
            entry.get(stage, {}) if stage == "replication_repetition" else entry
        )
    out = []
    for node in nodes:
        artifacts = (
            [node["artifact"]] if node.get("artifact") else node.get("artifacts", [])
        )
        if not isinstance(artifacts, list):
            raise ValueError("Stage artifacts must be a list")
        out.extend((node.get("values") or {}, artifact) for artifact in artifacts)
    if not out or not (nodes[-1].get("artifact") or nodes[-1].get("artifacts")):
        raise ValueError("Finalized results require a pinned stage artifact")
    return out


def artifact_identity(artifact: dict) -> str:
    """Identify a registry artifact without resolving the results store.

    Args:
        artifact: Artifact with a path and optional durable-store run.
    """
    return (
        f"{artifact['run']}/{artifact['path']}"
        if artifact.get("run")
        else artifact["path"]
    )


def verify_release_artifacts(
    claim: dict, sources: dict, results: dict, stage: str, receipt: dict, root: Path
) -> None:
    """Recheck readable artifacts and retain explicit attestations for the rest.

    A missing repository-local artifact is only accepted when the receipt pins
    its verified bytes. No hub lookup is attempted. Local files and mounted
    store files must still match, including transcribed values with extraction
    rules. The receipt also attests the audited source document's exact bytes.

    Args:
        claim: Claim named by the receipt.
        sources: Pinned study references.
        results: Reviewed transfer results.
        stage: Approved registry stage.
        receipt: Final review and verified artifact digests.
        root: Checkout root for local artifact verification.
    """
    artifacts = stage_artifacts(claim, stage, results)
    document = sources["paper"]["audit_document"]
    references = [(artifact_identity(a), a.get("sha256")) for _, a in artifacts]
    references.append((document["url"], document.get("sha256")))
    expected = {}
    for identity, digest in references:
        if identity in expected and expected[identity] != digest:
            raise ValueError(f"Conflicting released artifact pins: {identity}")
        expected[identity] = digest
    if any(
        not isinstance(v, str) or not re.fullmatch(r"[0-9a-f]{64}", v)
        for v in expected.values()
    ):
        raise ValueError("Released artifacts and paper reference require SHA-256 pins")
    if receipt["verified_artifacts"] != expected:
        raise ValueError(
            "Website receipt does not attest the exact stage artifacts and paper reference"
        )
    for values, artifact in artifacts:
        validate_artifact_schema(artifact)
        if artifact.get("run"):
            mounted = os.environ.get("REPLICA_RESULTS_STORE")
            path = (
                Path(mounted) / artifact["run"] / artifact["path"] if mounted else None
            )
        else:
            path = root / artifact["path"]
        if path is not None and path.is_file():
            outcomes = verify_artifact(values, artifact, resolved_path=path)
            if any(value != "ok" for value in outcomes.values()):
                raise ValueError(
                    f"Released artifact verification failed: {artifact_identity(artifact)}: {outcomes}"
                )
    if document.get("path"):
        path = root / document["path"]
        if (
            path.is_file()
            and hashlib.sha256(path.read_bytes()).hexdigest() != document["sha256"]
        ):
            raise ValueError("Released paper source artifact checksum drift")


def response_groups(
    claim: dict, *, require_complete: bool = False
) -> dict[str, list[str]]:
    """Partition comparison cells by declared response or explicit comparison unit.

    ``comparand.comparison_groups`` labels non-response or joint comparisons. Missing tags
    are kept individually for paper-only display and cannot be released as a
    computed response summary. No response is inferred from a cell identifier.

    Args:
        claim: Claim with declared cells and their tags.
        require_complete: Reject unlabelled cells before publishing computed outcomes.
    """
    comparand = claim["comparand"]
    tags = comparand.get("cell_tags", {})
    explicit = comparand.get("comparison_groups", {})
    if set(explicit) - set(comparand["cells"]):
        raise ValueError("Comparison groups name undeclared cells")
    responses = {tag["dv"] for tag in tags.values() if tag.get("dv")}
    if {r.replace("_", " ").casefold() for r in responses}.intersection(
        str(label).replace("_", " ").casefold() for label in explicit.values()
    ):
        raise ValueError("Comparison labels cannot collide with response measures")
    groups: dict[str, list[str]] = {}
    for cell in comparand["cells"]:
        response = tags.get(cell, {}).get("dv")
        if response and cell in explicit:
            raise ValueError(
                "A cell cannot have both a response and a comparison group"
            )
        label = response or explicit.get(cell)
        if not label:
            if require_complete:
                raise ValueError(
                    f"Missing response or comparison group: {claim['id']} / {cell}"
                )
            label = f"Paper comparison {comparand['cells'].index(cell) + 1}"
        if not isinstance(label, str) or not label.strip():
            raise ValueError("Comparison group labels must be nonempty strings")
        groups.setdefault(label, []).append(cell)
    return groups


def subset_claim(claim: dict[str, Any], cells: list[str]) -> dict[str, Any]:
    """Select a response's cells while preserving existing counting rules.

    Args:
        claim: Claim with stage evidence and correction annotations.
        cells: Cells assigned to one response or comparison group.
    """
    selected = deepcopy(claim)
    selected["comparand"]["cells"] = cells
    selected["comparand"]["cell_tags"] = {
        key: value
        for key, value in selected["comparand"].get("cell_tags", {}).items()
        if key in cells
    }
    significance = selected["comparand"].get("significance")
    if significance and "cells" in significance:
        significance["cells"] = [key for key in significance["cells"] if key in cells]

    def restrict(node: Any) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "values" and isinstance(value, dict):
                    node[key] = {k: v for k, v in value.items() if k in cells}
                else:
                    restrict(value)
        elif isinstance(node, list):
            for value in node:
                restrict(value)

    restrict(selected["evidence"])
    for key in CORRECTION_KEYS:
        if key in selected:
            restrict(selected[key])
    return selected
