"""Publishing a quarantined cell after its investigation."""

import json

import pytest

from replica.artifacts import manifest_path, validate_artifact
from replica.predictor_store import cli, layout
from replica.predictor_store.accept import accept
from replica.predictor_store.export import export
from tests.predictor_store._cells import DATASET, word_payload, write_cell

TAG = f"gpt2_restricted_first_token_entropy_{DATASET}"


def _quarantined(out, **kwargs):
    return write_cell(
        out,
        TAG,
        word_payload("restricted_first_token_entropy", 2.0),
        method="restricted_first_token_entropy",
        status="quarantined",
        **kwargs,
    )


def _with_audit(path, **audit):
    sidecar = manifest_path(path)
    manifest = json.loads(sidecar.read_text())
    manifest["audit"] = {
        "hard_issues": ["nan_burst"],
        "soft_issues": [],
        "examples": {"nan_burst": ["toy_1: 2/2 NaN"]},
        "failed_stimuli": 0,
        "missing_stimuli": 0,
        **audit,
    }
    sidecar.write_text(json.dumps(manifest))
    return manifest


def test_accept_publishes_the_payload_unchanged_with_the_reason(tmp_path):
    out = tmp_path / "wave" / "out"
    suspect = _quarantined(out)
    before = _with_audit(suspect)
    payload = suspect.read_bytes()

    published = accept(suspect, "  WT103 lacks the fragments; the NaN is definitional ")

    assert published == out / f"{TAG}.pkl" and published.read_bytes() == payload
    assert not suspect.exists() and not manifest_path(suspect).exists()
    manifest = json.loads(manifest_path(published).read_text())
    assert manifest["status"] == "completed"
    assert manifest["identity"] == before["identity"]
    assert manifest["environment"] == before["environment"]
    assert manifest["payload_sha256"] == before["payload_sha256"]
    audit = manifest["audit"]
    assert audit["hard_issues"] == ["nan_burst"]
    assert audit["accepted"]["reason"] == (
        "WT103 lacks the fragments; the NaN is definitional"
    )
    assert audit["accepted"]["at"].endswith("+00:00")
    assert validate_artifact(published) == before["identity"]


def test_accept_audits_a_cell_whose_manifest_has_no_block(tmp_path):
    out = tmp_path / "wave" / "out"
    suspect = _quarantined(out)
    assert "audit" not in json.loads(manifest_path(suspect).read_text())

    published = accept(suspect, "clean on re-audit")

    audit = json.loads(manifest_path(published).read_text())["audit"]
    assert audit["hard_issues"] == [] and audit["soft_issues"] == []
    assert audit["failed_stimuli"] == 0 and audit["missing_stimuli"] == 0
    assert audit["accepted"]["reason"] == "clean on re-audit"


@pytest.mark.parametrize(
    "audit,detail",
    [
        ({"failed_stimuli": 1}, "1 stimuli failed; the payload holds 0 fewer"),
        ({"missing_stimuli": 2}, "0 stimuli failed; the payload holds 2 fewer"),
    ],
)
def test_an_incomplete_cell_is_refused(tmp_path, audit, detail):
    out = tmp_path / "wave" / "out"
    suspect = _quarantined(out)
    _with_audit(suspect, **audit)
    with pytest.raises(ValueError, match=detail):
        accept(suspect, "looks fine")
    assert suspect.exists() and manifest_path(suspect).exists()


def test_an_incomplete_cell_without_the_block_is_refused_from_its_files(tmp_path):
    out = tmp_path / "wave" / "out"
    suspect = _quarantined(out, failed=[("toy_3", "RuntimeError: out of\nmemory")])
    with pytest.raises(ValueError, match="the run left .*failed.log"):
        accept(suspect, "looks fine")
    short = _quarantined(tmp_path / "short")
    sidecar = manifest_path(short)
    manifest = json.loads(sidecar.read_text())
    manifest["identity"]["n_stimuli"] += 1
    sidecar.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="holds 1 fewer than requested"):
        accept(short, "looks fine")


def test_an_unreadable_or_empty_payload_is_refused(tmp_path):
    out = tmp_path / "wave" / "out"
    suspect = _quarantined(out)
    _with_audit(suspect, hard_issues=["empty_pickle"])
    with pytest.raises(ValueError, match="payload is empty_pickle"):
        accept(suspect, "a reason")


def test_a_manifest_without_identity_or_of_another_schema_is_refused(tmp_path):
    out = tmp_path / "wave" / "out"
    suspect = _quarantined(out)
    sidecar = manifest_path(suspect)
    manifest = json.loads(sidecar.read_text())
    manifest["identity"]["schema_version"] = 0
    sidecar.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="schema 0 is not 1"):
        accept(suspect, "a reason")
    del manifest["identity"]
    sidecar.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="unreadable or incomplete manifest"):
        accept(suspect, "a reason")


def test_an_accepted_cell_is_reused_by_a_run_of_the_same_identity(tmp_path):
    from replica.compute import runner

    out = tmp_path / "wave" / "out"
    suspect = _quarantined(out)
    # The toy payload is clean, so the recorded hard issue is a superset of
    # what the audit finds now; an acceptance covers it.
    _with_audit(suspect, hard_issues=["nan_burst"])
    partial = out / f"{TAG}.pkl.partial"
    partial.write_bytes(b"x")
    manifest_path(partial).write_text("{}")
    published = accept(suspect, "definitional NaN")
    identity = json.loads(manifest_path(published).read_text())["identity"]
    assert not partial.exists() and not manifest_path(partial).exists()
    assert runner._should_skip_existing(published, force=False, identity=identity)
    sidecar = manifest_path(published)
    manifest = json.loads(sidecar.read_text())
    del manifest["audit"]["accepted"]
    sidecar.write_text(json.dumps(manifest))
    assert runner._should_skip_existing(published, force=False, identity=identity)
    assert not runner._should_skip_existing(published, force=True, identity=identity)


def test_the_command_line_reports_a_refusal(tmp_path, capsys):
    out = tmp_path / "wave" / "out"
    suspect = _quarantined(out)
    _with_audit(suspect, failed_stimuli=2)
    assert cli.main(["accept", str(suspect), "--reason", "a reason"]) == 1
    assert capsys.readouterr().err.startswith("refused: ")
    assert suspect.exists()


def test_accept_refuses_other_files_an_empty_reason_and_a_superseded_cell(tmp_path):
    out = tmp_path / "wave" / "out"
    suspect = _quarantined(out)
    _with_audit(suspect)
    with pytest.raises(ValueError, match="needs a reason"):
        accept(suspect, "   ")
    completed = write_cell(
        out,
        TAG,
        word_payload("restricted_first_token_entropy", 2.0),
        method="restricted_first_token_entropy",
    )
    with pytest.raises(ValueError, match="not a quarantined cell"):
        accept(completed, "a reason")
    with pytest.raises(FileExistsError, match="superseded"):
        accept(suspect, "a reason")
    assert suspect.exists()


def test_a_changed_payload_or_status_is_refused(tmp_path):
    out = tmp_path / "wave" / "out"
    suspect = _quarantined(out)
    _with_audit(suspect)
    suspect.write_bytes(suspect.read_bytes() + b"\0")
    with pytest.raises(ValueError, match="differs from its manifest"):
        accept(suspect, "a reason")
    other = _quarantined(tmp_path / "other")
    sidecar = manifest_path(other)
    manifest = json.loads(sidecar.read_text())
    manifest["status"] = "partial"
    sidecar.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="status is 'partial'"):
        accept(other, "a reason")


def test_an_accepted_cell_is_exported_with_its_acceptance(tmp_path, corpora, capsys):
    out = tmp_path / "wave" / "out"
    suspect = _quarantined(out)
    _with_audit(suspect)
    assert cli.main(["accept", str(suspect), "--reason", "definitional NaN"]) == 0
    assert capsys.readouterr().out.startswith("published ")

    stage = tmp_path / "stage"
    report = export(stage, [("wave", out)], corpora=corpora)

    assert report.ok and report.cells == 1 and report.quarantined == 0
    staged = json.loads((stage / layout.MANIFESTS_DIR / f"{TAG}.json").read_text())
    assert staged["status"] == "completed"
    assert staged["audit"]["accepted"]["reason"] == "definitional NaN"
