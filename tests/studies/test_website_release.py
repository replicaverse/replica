"""Release receipts: what approves a stage for publication and what invalidates the approval."""

import hashlib
import json

import pytest

from replica.studies.claims.website_release import (
    artifact_identity,
    evidence_digest,
    released_stages,
    response_groups,
    stage_artifacts,
)


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    import replica.studies.claims.website_release as release

    monkeypatch.setattr(release, "REPO_ROOT", tmp_path)
    claim = {
        "id": "example_claim",
        "study": "example",
        "claim": "Paper claim.",
        "anchor": "Table 1",
        "coverage": "full",
        "comparand": {
            "cells": ["a", "b", "c", "d", "e"],
            "unit": "decision",
            "statistic": "test decisions",
            "cell_tags": {
                key: {"dv": "first_fixation" if key in "ab" else "total_time"}
                for key in "abcde"
            },
            "significance": {"positive_values": [True]},
            "match": {"kind": "exact"},
        },
        "evidence": {
            "paper": {
                "values": dict(
                    zip("abcde", [True, False, True, True, True], strict=True)
                )
            },
            "replica": {
                "status": "available",
                "values": dict(
                    zip("abcde", [True, False, False, False, False], strict=True)
                ),
                "provenance": "from_spec_compute",
                "consumes_official_results": False,
            },
        },
    }
    path = tmp_path / "stage.tsv"
    path.write_text("value\n1\n")
    artifact = {
        "path": "stage.tsv",
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    claim["evidence"]["replica"]["artifact"] = artifact
    sources = {
        "paper": {
            "abstract": "Published abstract.",
            "audit_document": {
                "url": "https://example.org/paper",
                "version": "v1",
                "sha256": "a" * 64,
            },
        }
    }
    registry = {"claims": [claim], "study_sources": {"example": sources}}
    results = {
        "claims": {
            claim["id"]: {
                "artifact": artifact,
                "total": None,
                "replication_repetition": {
                    "artifact": artifact,
                    "total": None,
                    "stage": "replication_repetition",
                    "analysis_method": "released",
                    "provenance": "author_code_compute",
                    "review": {"status": "verified"},
                    "source_run": "fixture",
                    "scope": "fixture",
                },
            }
        }
    }
    return registry, results, tmp_path


def approve(evidence, stage):
    registry, results, root = evidence
    claim = registry["claims"][0]
    receipt = {
        "status": "finalized",
        "scope": "The declared five test decisions.",
        "verification": "Independent replay checked the pinned artifacts and documented disagreements.",
        "evidence_sha256": evidence_digest(
            claim, registry["study_sources"]["example"], results, stage
        ),
    }
    receipt["verified_artifacts"] = {
        artifact_identity(a): a["sha256"]
        for _, a in stage_artifacts(claim, stage, results)
    }
    document = registry["study_sources"]["example"]["paper"]["audit_document"]
    receipt["verified_artifacts"][document["url"]] = document["sha256"]
    relative = f"src/replica/studies/example/{stage}-review.json"
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt))
    claim.setdefault("website_release", {})[stage] = {
        "path": relative,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    return path


def test_study_record_preserves_individual_release_checks(evidence):
    registry, results, root = evidence
    claim = registry["claims"][0]
    stages = ("repetition", "reproduction")
    receipts = {
        stage: json.loads(approve(evidence, stage).read_text()) for stage in stages
    }
    expected = released_stages(registry, results, root=root)
    relative = "src/replica/studies/example/website_release.json"
    path = root / relative
    record = {
        "schema": "replica.website-release.v1",
        "study": "example",
        "claims": {claim["id"]: receipts},
    }
    path.write_text(json.dumps(record))
    for stage in stages:
        claim["website_release"][stage] = {
            "path": relative,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    assert released_stages(registry, results, root=root) == expected

    # Updating the file hash cannot make an altered stage approval valid.
    record["claims"][claim["id"]]["reproduction"]["evidence_sha256"] = "0" * 64
    path.write_text(json.dumps(record))
    for reference in claim["website_release"].values():
        reference["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="evidence changed since review"):
        released_stages(registry, results, root=root)


@pytest.mark.parametrize("defect", ["study", "claim", "stage", "schema"])
def test_study_record_cannot_substitute_another_review(evidence, defect):
    registry, results, root = evidence
    claim = registry["claims"][0]
    path = approve(evidence, "repetition")
    receipt = json.loads(path.read_text())
    record = {
        "schema": "replica.website-release.v1",
        "study": "example",
        "claims": {claim["id"]: {"repetition": receipt}},
    }
    if defect == "study":
        record["study"] = "another_study"
    elif defect == "claim":
        record["claims"] = {"another_claim": {"repetition": receipt}}
    elif defect == "stage":
        record["claims"][claim["id"]] = {"reproduction": receipt}
    else:
        record["schema"] = "unknown"
    path.write_text(json.dumps(record))
    claim["website_release"]["repetition"]["sha256"] = hashlib.sha256(
        path.read_bytes()
    ).hexdigest()
    with pytest.raises(
        ValueError, match="Invalid .*receipt fields|Invalid study website"
    ):
        released_stages(registry, results, root=root)


def test_response_partition_is_complete_and_explicit(evidence):
    registry, _, _ = evidence
    claim = registry["claims"][0]
    assert response_groups(claim) == {
        "first_fixation": ["a", "b"],
        "total_time": ["c", "d", "e"],
    }
    del claim["comparand"]["cell_tags"]["e"]["dv"]
    with pytest.raises(ValueError, match="Missing response"):
        response_groups(claim, require_complete=True)
    claim["comparand"]["comparison_groups"] = {"e": "Joint response test"}
    assert response_groups(claim, require_complete=True)["Joint response test"] == ["e"]


def test_readable_artifact_drift_fails_even_when_registry_is_unchanged(evidence):
    registry, results, root = evidence
    approve(evidence, "reproduction")
    (root / "stage.tsv").write_text("value\n2\n")
    with pytest.raises(ValueError, match="artifact verification failed"):
        released_stages(registry, results)


def test_attested_unavailable_artifact_requires_no_download(evidence, monkeypatch):
    registry, results, root = evidence
    approve(evidence, "reproduction")
    (root / "stage.tsv").unlink()
    import replica.studies.claims.values as artifacts

    monkeypatch.setattr(
        artifacts,
        "artifact_path",
        lambda a: pytest.fail("Public builds must not resolve result downloads"),
    )
    assert set(released_stages(registry, results)["example_claim"]) == {"reproduction"}


def test_joint_group_cannot_hide_response_or_omit_finalized_cells(evidence):
    registry, results, _ = evidence
    claim = registry["claims"][0]
    del claim["comparand"]["cell_tags"]["e"]["dv"]
    approve(evidence, "replication")
    with pytest.raises(ValueError, match="Missing response"):
        released_stages(registry, results)
    claim["comparand"]["comparison_groups"] = {"e": "first_fixation"}
    with pytest.raises(ValueError, match="collide"):
        response_groups(claim, require_complete=True)


def test_receipt_cannot_override_artifact_value_extraction(evidence):
    registry, results, root = evidence
    claim = registry["claims"][0]
    artifact = claim["evidence"]["replica"]["artifact"]
    path = root / "stage.tsv"
    path.write_text("word\tp\na\t0.01\n")
    artifact["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    artifact["extract"] = {
        "a": {
            "kind": "value_below",
            "where": {"word": "a"},
            "column": "p",
            "threshold": 0.05,
        }
    }
    claim["evidence"]["replica"]["values"]["a"] = False
    approve(evidence, "reproduction")
    with pytest.raises(ValueError, match="transcription drift"):
        released_stages(registry, results)


@pytest.mark.parametrize("reference", ["paper_artifact", "paper_document"])
def test_offline_receipt_rejects_conflicting_pins_for_one_identity(evidence, reference):
    registry, results, root = evidence
    claim = registry["claims"][0]
    artifact = claim["evidence"]["replica"]["artifact"]
    (root / artifact["path"]).unlink()
    if reference == "paper_artifact":
        claim["evidence"]["paper"]["artifact"] = {**artifact, "sha256": "a" * 64}
    else:
        registry["study_sources"]["example"]["paper"]["audit_document"]["url"] = (
            artifact["path"]
        )
    approve(evidence, "reproduction")
    with pytest.raises(ValueError, match="Conflicting released artifact pins"):
        released_stages(registry, results)


def test_offline_receipt_accepts_repeated_identical_artifact_pin(evidence):
    registry, results, root = evidence
    claim = registry["claims"][0]
    artifact = claim["evidence"]["replica"]["artifact"]
    (root / artifact["path"]).unlink()
    claim["evidence"]["paper"]["artifact"] = artifact.copy()
    approve(evidence, "reproduction")
    assert set(released_stages(registry, results)[claim["id"]]) == {"reproduction"}


def test_release_authenticates_every_table_in_a_transfer_family(evidence):
    registry, results, root = evidence
    entry = results["claims"]["example_claim"]["replication_repetition"]
    first = entry.pop("artifact")
    second_path = root / "second.tsv"
    second_path.write_text("value\n0\n")
    second = {
        "path": "second.tsv",
        "sha256": hashlib.sha256(second_path.read_bytes()).hexdigest(),
    }
    entry["artifacts"] = [first, second]
    approve(evidence, "replication_repetition")
    assert (
        "replication_repetition" in released_stages(registry, results)["example_claim"]
    )
    second_path.write_text("value\n1\n")
    with pytest.raises(ValueError, match="verification failed"):
        released_stages(registry, results)
