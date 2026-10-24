"""Artifact grounding of registry claim values (``claim_values.py``).

Artifact schemas are checked without downloads. Full checksum and extraction
checks use the prepared HF cache; missing required indexed artifacts fail.
Explicit external-store records are selected with the data-dependent tier.
"""

import hashlib

import pytest

from replica.studies.claims.registry import load_registry
from replica.studies.claims.values import (
    REPO_ROOT,
    validate_artifact_schema,
    verify_artifact,
)

TABLE = "lang\tlm\tp\teffect\nen\tmgpt\t0.99\t0.2\nko\tmgpt\t0.41\t-0.3\n"


@pytest.fixture()
def tracked_artifact(tmp_path, monkeypatch):
    """A tiny repo-relative table plus its pinned checksum."""
    path = tmp_path / "cells.tsv"
    path.write_text(TABLE)
    monkeypatch.setattr(
        "replica.studies.claims.values.REPO_ROOT", tmp_path, raising=True
    )
    return {
        "path": "cells.tsv",
        "sha256": hashlib.sha256(TABLE.encode()).hexdigest(),
    }


def test_value_and_decision_extraction_agree_with_the_table(tracked_artifact):
    tracked_artifact["extract"] = {
        "en_p": {"kind": "value", "column": "p", "where": {"lang": "en"}},
        "ko_significant": {
            "kind": "value_below",
            "column": "p",
            "threshold": 0.5,
            "where": {"lang": "ko"},
        },
        "none_significant": {
            "kind": "all_rows_at_least",
            "column": "p",
            "threshold": 0.05,
        },
    }
    validate_artifact_schema(tracked_artifact)
    outcomes = verify_artifact(
        {"en_p": 0.99, "ko_significant": True, "none_significant": True},
        tracked_artifact,
    )
    assert outcomes == {
        "sha256": "ok",
        "en_p": "ok",
        "ko_significant": "ok",
        "none_significant": "ok",
    }
    assert verify_artifact({"ko_significant": False}, tracked_artifact)[
        "ko_significant"
    ].startswith("transcription drift")


def test_booleans_agree_only_with_zero_one_or_exact_text(tmp_path):
    path = tmp_path / "flags.tsv"
    path.write_text(
        "cell\tflag\tword\n"
        "one\t1\tTrue\n"
        "zero\t0\tFalse\n"
        "missing\tnan\t1\n"
        "negative\t-1\ttrue\n"
    )
    extract = {
        f"{row}_{kind}": {"kind": kind, "column": column, "where": {"cell": row}}
        for row in ("one", "zero", "missing", "negative")
        for kind, column in (("value", "flag"), ("label", "word"))
    }
    extract["zero_below"] = {
        "kind": "value_below",
        "column": "flag",
        "threshold": 0.5,
        "where": {"cell": "zero"},
    }
    artifact = {
        "path": "flags.tsv",
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "extract": extract,
    }

    def agreeing(values):
        outcomes = verify_artifact(values, artifact, resolved_path=path)
        return {cell for cell, outcome in outcomes.items() if outcome == "ok"}

    assert agreeing(dict.fromkeys(extract, True)) == {
        "sha256",
        "one_value",
        "one_label",
        "zero_below",
    }
    assert agreeing(dict.fromkeys(extract, False)) == {
        "sha256",
        "zero_value",
        "zero_label",
    }
    for transcribed, agrees in [
        (1, True),
        (0.5, False),
        ("True", True),
        ("False", False),
    ]:
        assert ("zero_below" in agreeing({"zero_below": transcribed})) is agrees


def test_signed_and_at_least_extracts_read_the_claimed_direction(tracked_artifact):
    """A significant effect the other way is not a hit; a null cell is a failure to reject."""
    tracked_artifact["extract"] = {
        "ko_positive": {
            "kind": "signed_below",
            "column": "p",
            "threshold": 0.5,
            "sign_column": "effect",
            "sign": "positive",
            "where": {"lang": "ko"},
        },
        "ko_negative": {
            "kind": "signed_below",
            "column": "p",
            "threshold": 0.5,
            "sign_column": "effect",
            "sign": "negative",
            "where": {"lang": "ko"},
        },
        "en_null": {
            "kind": "value_at_least",
            "column": "p",
            "threshold": 0.05,
            "where": {"lang": "en"},
        },
    }
    validate_artifact_schema(tracked_artifact)
    outcomes = verify_artifact(
        {"ko_positive": False, "ko_negative": True, "en_null": True}, tracked_artifact
    )
    assert outcomes == {
        "sha256": "ok",
        "ko_positive": "ok",
        "ko_negative": "ok",
        "en_null": "ok",
    }
    with pytest.raises(ValueError, match="sign_column"):
        validate_artifact_schema(
            {
                **tracked_artifact,
                "extract": {
                    "x": {
                        "kind": "signed_below",
                        "column": "p",
                        "threshold": 0.05,
                        "where": {"lang": "ko"},
                    }
                },
            }
        )


def test_label_extract_compares_symbols_as_text(tracked_artifact):
    spec = {"kind": "label", "column": "lang", "where": {"lm": "mgpt", "p": "0.41"}}
    artifact = {**tracked_artifact, "extract": {"which": spec}}
    assert verify_artifact({"which": "ko"}, artifact)["which"] == "ok"
    assert verify_artifact({"which": "en"}, artifact)["which"].startswith(
        "transcription drift"
    )


def test_max_row_extract_names_the_largest_matching_row(tracked_artifact):
    spec = {
        "kind": "max_row",
        "column": "p",
        "where": {"lm": "mgpt"},
        "equals": {"lang": "en"},
    }
    artifact = {**tracked_artifact, "extract": {"en_largest": spec}}
    assert verify_artifact({"en_largest": True}, artifact)["en_largest"] == "ok"
    artifact["extract"]["en_largest"]["equals"] = {"lang": "ko"}
    assert verify_artifact({"en_largest": True}, artifact)["en_largest"].startswith(
        "transcription drift"
    )
    with pytest.raises(ValueError, match="equals"):
        validate_artifact_schema(
            {
                **tracked_artifact,
                "extract": {
                    "c": {"kind": "max_row", "column": "p", "where": {"lm": "mgpt"}}
                },
            }
        )


def test_transcription_and_checksum_drift_are_reported(tracked_artifact):
    tracked_artifact["extract"] = {
        "en_p": {"kind": "value", "column": "p", "where": {"lang": "en"}}
    }
    outcomes = verify_artifact({"en_p": 0.5}, tracked_artifact)
    assert outcomes["sha256"] == "ok"
    assert outcomes["en_p"].startswith("transcription drift")
    tracked_artifact["sha256"] = "0" * 64
    outcomes = verify_artifact({"en_p": 0.99}, tracked_artifact)
    assert outcomes["sha256"].startswith("checksum drift")


def test_ambiguous_selectors_and_bad_schemas_are_rejected(tracked_artifact):
    tracked_artifact["extract"] = {
        "p": {"kind": "value", "column": "p", "where": {"lm": "mgpt"}}
    }
    outcomes = verify_artifact({"p": 0.99}, tracked_artifact)
    assert outcomes["p"].startswith("extraction failed")
    with pytest.raises(ValueError, match="unknown extract kind"):
        validate_artifact_schema(
            {"path": "x", "sha256": "0" * 64, "extract": {"c": {"kind": "sum"}}}
        )
    with pytest.raises(ValueError, match="64-hex"):
        validate_artifact_schema({"path": "x", "sha256": "abc"})


def test_missing_store_reads_as_unresolved_not_failure(monkeypatch):
    monkeypatch.delenv("REPLICA_RESULTS_STORE", raising=False)
    outcomes = verify_artifact(
        {"c": 1.0}, {"run": "some_run", "path": "results/x.tsv", "sha256": "0" * 64}
    )
    assert outcomes == {"artifact": "unresolved: results store not mounted"}


def test_repo_root_points_at_the_checkout():
    assert (REPO_ROOT / "pyproject.toml").is_file()


def _artifacts():
    for claim in load_registry()["claims"]:
        for source, evidence in claim["evidence"].items():
            artifact = evidence.get("artifact")
            if artifact is not None:
                yield claim, source, evidence, artifact
        for reference in claim.get("reference_evidence", []):
            yield claim, reference["id"], reference, reference["artifact"]
        # A correction's corrected decisions are evidence of the same shape.
        for key in ("predictor_correction", "test_divergence", "method_defect"):
            corrected = (claim.get(key) or {}).get("corrected")
            if corrected is not None:
                yield claim, f"{key}.corrected", corrected, corrected["artifact"]


def _annotation_artifacts():
    """Corrected-run tables a defect annotation's effect statement rests on."""
    for claim in load_registry()["claims"]:
        for key in ("predictor_correction", "test_divergence", "method_defect"):
            annotation = claim.get(key) or {}
            for index, artifact in enumerate(annotation.get("artifacts") or []):
                yield claim, f"{key}[{index}]", artifact


ARTIFACTS = list(_artifacts())
IDS = [f"{claim['id']}:{source}" for claim, source, _, _ in ARTIFACTS]
ANNOTATION_ARTIFACTS = list(_annotation_artifacts())
ANNOTATION_IDS = [f"{claim['id']}:{where}" for claim, where, _ in ANNOTATION_ARTIFACTS]


def _artifact_checks(rows):
    # A run-relative artifact explicitly requires the external results store.
    return [
        pytest.param(*row, marks=pytest.mark.needs_data if "run" in row[-1] else ())
        for row in rows
    ]


@pytest.mark.parametrize("claim,source,evidence,artifact", ARTIFACTS, ids=IDS)
@pytest.mark.artifact
def test_artifact_blocks_are_well_formed(claim, source, evidence, artifact):
    validate_artifact_schema(artifact)
    assert evidence.get("values"), (
        f"{claim['id']} [{source}]: an artifact grounds transcribed values, "
        "so the values block must exist"
    )


@pytest.mark.parametrize(
    "claim,source,evidence,artifact", _artifact_checks(ARTIFACTS), ids=IDS
)
@pytest.mark.artifact
def test_artifact_checksums_and_extractions_hold(claim, source, evidence, artifact):
    outcomes = verify_artifact(evidence["values"], artifact)
    unresolved = outcomes.get("artifact", "")
    if unresolved.startswith("unresolved: results store not mounted"):
        pytest.skip(unresolved)
    bad = {key: value for key, value in outcomes.items() if value != "ok"}
    assert not bad, (claim["id"], source, bad)


@pytest.mark.parametrize(
    "claim,where,artifact", _artifact_checks(ANNOTATION_ARTIFACTS), ids=ANNOTATION_IDS
)
@pytest.mark.artifact
def test_annotation_artifacts_are_pinned_and_hold(claim, where, artifact):
    """A defect's corrected-run tables: schema-valid, noted, checksum-true."""
    validate_artifact_schema(artifact)
    assert artifact.get("note", "").strip(), (
        f"{claim['id']} [{where}]: an effect artifact needs a note saying "
        "what the table carries"
    )
    outcomes = verify_artifact({}, artifact)
    unresolved = outcomes.get("artifact", "")
    if unresolved.startswith("unresolved: results store not mounted"):
        pytest.skip(unresolved)
    bad = {key: value for key, value in outcomes.items() if value != "ok"}
    assert not bad, (claim["id"], where, bad)


def _replica_claims_with_values():
    for claim in load_registry()["claims"]:
        evidence = claim["evidence"]["replica"]
        if evidence.get("values"):
            yield claim, evidence


REPLICA_VALUED = list(_replica_claims_with_values())


@pytest.mark.parametrize(
    "claim,evidence", REPLICA_VALUED, ids=[c["id"] for c, _ in REPLICA_VALUED]
)
@pytest.mark.artifact
def test_every_replica_value_rests_on_a_pinned_indexed_extract(claim, evidence):
    """Every claimed value has a checksummed artifact and a complete extraction rule."""
    artifact = evidence.get("artifact")
    assert artifact is not None, f"{claim['id']}: REPLICA values carry no artifact"
    assert "run" not in artifact, (
        f"{claim['id']}: the REPLICA artifact must resolve through the result index"
    )
    from pathlib import Path

    from replica.studies.records.files import read_index

    parts = Path(artifact["path"]).parts
    assert parts[:3] == ("src", "replica", "studies") and parts[4] == "results"
    relative = "/".join(parts[5:])
    owners = [
        entry["hub"][relative[len(run) + 1 :]]
        for run, entry in read_index(parts[3])["runs"].items()
        if relative.startswith(run + "/")
        and relative[len(run) + 1 :] in entry.get("hub", {})
    ]
    assert owners == [artifact["sha256"]]
    extract = artifact.get("extract") or {}
    missing = set(evidence["values"]) - set(extract)
    assert not missing, f"{claim['id']}: cells without an extract: {sorted(missing)}"


@pytest.mark.parametrize(
    "effect,pvalue,want",
    [(1, 0.01, "+"), (-1, 0.01, "-"), (-1, 0.05, "ns"), (1, 0.9, "ns")],
)
def test_signed_decision_keeps_opposite_significant_effects(effect, pvalue, want):
    from replica.studies.claims.values import extract_cell

    rows = [{"id": "a", "effect": str(effect), "p": str(pvalue)}]
    spec = {
        "kind": "signed_decision",
        "column": "p",
        "sign_column": "effect",
        "threshold": 0.05,
        "where": {"id": "a"},
    }
    assert extract_cell(rows, spec) == want


@pytest.mark.parametrize(
    "pvalue,effect",
    [("nan", "1"), ("0.01", "nan"), ("-0.1", "1"), ("1.1", "1"), ("0.01", "0")],
)
def test_invalid_signed_test_never_becomes_a_null(pvalue, effect):
    from replica.studies.claims.values import extract_cell

    with pytest.raises(ValueError):
        extract_cell(
            [{"id": "a", "p": pvalue, "effect": effect}],
            {
                "kind": "signed_decision",
                "column": "p",
                "sign_column": "effect",
                "threshold": 0.05,
                "where": {"id": "a"},
            },
        )


def test_evidence_decisions_verify_table_before_reading(tracked_artifact, tmp_path):
    from replica.studies.claims.values import evidence_decisions

    tracked_artifact["extract"] = {
        "a": {
            "kind": "value_below",
            "column": "p",
            "threshold": 0.5,
            "where": {"lang": "ko"},
        }
    }
    evidence = {
        "artifact": tracked_artifact,
        "values": {"a": True},
        "decision_extract": {
            "default": {"kind": "signed_decision", "sign_column": "effect"}
        },
    }
    assert evidence_decisions(evidence) == {"a": "-"}
    path = tmp_path / "cells.tsv"
    path.write_text(path.read_text().replace("-0.3", "0.31"))
    with pytest.raises(ValueError, match="checksum drift"):
        evidence_decisions(evidence)


@pytest.mark.parametrize(
    "claim,source,evidence,artifact",
    [row for row in ARTIFACTS if row[2].get("decision_extract")],
    ids=[
        f"{c['id']}:{source}"
        for c, source, evidence, _ in ARTIFACTS
        if evidence.get("decision_extract")
    ],
)
@pytest.mark.artifact
def test_declared_decision_extractions_are_complete_and_signed(
    claim, source, evidence, artifact
):
    from replica.studies.claims.values import evidence_decisions

    definition = evidence["decision_extract"]
    assert set(definition) <= {"default", "cells"}
    assert set(definition.get("cells", {})) <= set(evidence["values"])
    decisions = evidence_decisions(evidence)
    assert set(decisions) == set(evidence["values"])
    assert set(decisions.values()) <= {"+", "-", "ns"}, (claim["id"], source)
