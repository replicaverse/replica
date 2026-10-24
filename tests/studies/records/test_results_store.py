"""Saving a run into the store and publishing it, on a synthetic tree.

The store is a temporary directory named by ``REPLICA_RESULTS_STORE``; the
upload hook records what would go to the hub and the download hook reads it
back from the store, so no network is involved.
"""

import hashlib
import json

import pytest

from replica.studies.records import cli as C
from replica.studies.records import files as R
from replica.studies.records import store as S

STUDY = "fakestudy"
RUN = "replication/reanalysis/reading_time/r0123456789ab"


@pytest.fixture
def tree(tmp_path, monkeypatch):
    src = tmp_path / "src"
    package = src / "replica" / "studies" / STUDY
    (src / "replica" / "analysis").mkdir(parents=True)
    (src / "replica" / "studies").mkdir()
    (src / "replica" / "__init__.py").write_text("")
    (src / "replica" / "analysis" / "__init__.py").write_text("")
    (src / "replica" / "analysis" / "evaluate.py").write_text("X = 1\n")
    (src / "replica" / "studies" / "__init__.py").write_text("")
    package.mkdir()
    (package / "__init__.py").write_text("from ...analysis.evaluate import X\n")
    (package / "results").mkdir()
    (package / "results" / "index.json").write_text(
        json.dumps({"schema": R.SCHEMA, "runs": {}}, indent=2)
    )
    out = tmp_path / "job"
    (out / "nested").mkdir(parents=True)
    (out / "cells.tsv").write_text("a\tb\n1\t2\n3\t4\n")
    (out / "summary.json").write_text("{}\n")
    (out / "nested" / "folds.tsv").write_text("f\n1\n")
    (out / "driver.log").write_text("done\n")
    (out / "frame.parquet").write_bytes(b"pq")
    (out / ".partial").write_text("x")
    uploads = []
    monkeypatch.setattr(R, "STUDIES_DIR", src / "replica" / "studies")
    monkeypatch.setattr(R, "SRC_DIR", src)
    monkeypatch.setenv(R.STORE_ENV, str(tmp_path / "store"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))

    def upload(folder, path_in_repo, message):
        uploads.append(
            (
                {
                    p.relative_to(folder).as_posix(): p.read_bytes()
                    for p in folder.rglob("*")
                    if p.is_file()
                },
                path_in_repo,
            )
        )
        return "f" * 40

    monkeypatch.setattr(S, "_upload", upload)
    monkeypatch.setattr(
        R,
        "_hub_download",
        lambda remote, **kwargs: str(
            tmp_path
            / "store"
            / "run_1"
            / "results"
            / remote.removeprefix(f"studies/{STUDY}/results/{RUN}/")
        ),
    )
    R._verified.clear()
    return tmp_path, out, uploads


def test_save_sorts_files_by_kind_and_writes_a_path_free_manifest(tree):
    root, out, _ = tree
    run_dir = S.save(
        "run_1", [out], study=STUDY, command="python -m x", inputs={"corpus": "0" * 64}
    )
    assert sorted(
        p.relative_to(run_dir).as_posix() for p in run_dir.rglob("*") if p.is_file()
    ) == [
        "MANIFEST.json",
        "frames/frame.parquet",
        "logs/driver.log",
        "results/cells.tsv",
        "results/nested/folds.tsv",
        "results/summary.json",
    ]
    manifest = S.read_manifest("run_1")
    assert manifest["study"] == STUDY
    assert manifest["produced_by"]["inputs"] == {"corpus": "0" * 64}
    assert manifest["produced_by"]["footprint"] == R.code_footprint(STUDY)
    cells = next(e for e in manifest["files"] if e["path"] == "results/cells.tsv")
    assert cells["rows"] == 2
    assert cells["sha256"] == hashlib.sha256(b"a\tb\n1\t2\n3\t4\n").hexdigest()
    assert str(root) not in json.dumps(manifest)
    assert set(manifest) == {
        "run_id",
        "study",
        "saved_utc",
        "command",
        "produced_by",
        "files",
    }


def test_save_again_keeps_earlier_files_and_verify_sees_tampering(tree):
    _, out, _ = tree
    S.save("run_1", [out], study=STUDY)
    (out / "cells.tsv").write_text("a\tb\n9\t9\n")
    S.save("run_1", [out / "cells.tsv"], study=STUDY)
    manifest = S.read_manifest("run_1")
    assert {e["path"] for e in manifest["files"]} == {
        "frames/frame.parquet",
        "logs/driver.log",
        "results/cells.tsv",
        "results/nested/folds.tsv",
        "results/summary.json",
    }
    assert S.verify("run_1") == []
    (S.store_root() / "run_1" / "results" / "summary.json").write_text("{tampered}")
    (S.store_root() / "run_1" / "logs" / "driver.log").unlink()
    assert S.verify("run_1") == [
        "logs/driver.log: missing",
        "results/summary.json: digest differs",
    ]
    assert S.list_runs() == [
        {
            "run_id": "run_1",
            "study": STUDY,
            "saved_utc": manifest["saved_utc"],
            "files": 5,
            "stamped": True,
        }
    ]


def test_save_again_refuses_changed_or_missing_earlier_files(tree):
    _, out, _ = tree
    S.save("run_1", [out], study=STUDY)
    manifest = S.read_manifest("run_1")
    stored = S.store_root() / "run_1"
    cells = (stored / "results" / "cells.tsv").read_bytes()
    (out / "cells.tsv").write_text("a\tb\n9\t9\n")
    (stored / "results" / "summary.json").write_text("{tampered}")
    with pytest.raises(ValueError, match="results/summary.json"):
        S.save("run_1", [out / "cells.tsv"], study=STUDY)
    assert S.read_manifest("run_1") == manifest
    assert (stored / "results" / "cells.tsv").read_bytes() == cells
    assert S.verify("run_1") == ["results/summary.json: digest differs"]
    (stored / "results" / "summary.json").write_text("{}\n")
    (stored / "logs" / "driver.log").unlink()
    with pytest.raises(ValueError, match="logs/driver.log"):
        S.save("run_1", [out / "cells.tsv"], study=STUDY)
    assert S.read_manifest("run_1") == manifest
    assert S.verify("run_1") == ["logs/driver.log: missing"]


def test_publish_uploads_payloads_and_keeps_only_metadata_in_checkout(
    tree, monkeypatch
):
    root, out, uploads = tree
    document = R.read_index(STUDY)
    document["hub_repo"] = "example/first-store"
    R.write_index(STUDY, document)
    monkeypatch.setenv("REPLICA_RESULTS_HF_REPO", "example/study-bundles")
    S.save("run_1", [out], study=STUDY)
    entry = S.publish(
        "run_1",
        study=STUDY,
        run=RUN,
        role="evidence",
        description="Reading-time fit",
        note="fresh",
    )
    assert len(uploads) == 2
    monkeypatch.delenv("REPLICA_RESULTS_HF_REPO")
    assert R.hub_repo(R.read_index(STUDY)) == "example/study-bundles"
    assert all(
        json.loads(payload["index.json"])["hub_repo"] == "example/study-bundles"
        for payload, _ in uploads
    )
    payload, destination = uploads[0]
    assert set(uploads[1][0]) == {"index.json", "README.md"}
    assert json.loads(uploads[1][0]["index.json"])["hub_revision"] == "f" * 40
    assert destination == f"studies/{STUDY}/results"
    assert set(payload) == {
        "index.json",
        "README.md",
        *(f"{RUN}/{n}" for n in entry["files"]),
    }
    assert json.loads(payload["index.json"])["runs"][RUN] == entry
    assert set(entry["hub"]) == {"cells.tsv", "nested/folds.tsv", "summary.json"}
    assert (
        entry["store"] == "run_1"
        and entry["role"] == "evidence"
        and entry["note"] == "fresh"
    )
    assert entry["produced_by"] == S.read_manifest("run_1")["produced_by"]
    checkout = R.STUDIES_DIR / STUDY / "results" / RUN
    assert not checkout.exists()
    assert {p.name for p in (R.STUDIES_DIR / STUDY / "results").iterdir()} == {
        "index.json",
        "README.md",
    }
    assert R.validate_index(STUDY) == []
    assert R.stale_runs(STUDY) == []
    assert R.results_file(STUDY, RUN, "summary.json") == (
        S.store_root() / "run_1" / "results" / "summary.json"
    )
    assert R.results_file(STUDY, RUN, "nested/folds.tsv") == (
        S.store_root() / "run_1" / "results" / "nested" / "folds.tsv"
    )
    before = R.read_index(STUDY)
    monkeypatch.setenv("REPLICA_RESULTS_HF_REPO", "example/another-store")
    with pytest.raises(ValueError, match="Cannot redirect"):
        S.publish(
            "run_1",
            study=STUDY,
            run=RUN,
            role="evidence",
            description="Reading-time fit",
        )
    assert len(uploads) == 2 and R.read_index(STUDY) == before


def test_publish_has_no_checkout_payload_option(tree):
    _, out, uploads = tree
    S.save("run_1", [out], study=STUDY)
    with pytest.raises(TypeError, match="keep"):
        S.publish(
            "run_1",
            study=STUDY,
            run=RUN,
            role="check",
            description="Reading-time fit",
            keep=("cells.tsv",),
        )
    assert uploads == []


def test_publish_refuses_a_foreign_or_unstamped_run(tree):
    _, out, _ = tree
    S.save("run_1", [out], study=STUDY)
    with pytest.raises(ValueError, match="was saved for"):
        S.publish(
            "run_1",
            study="other",
            run=RUN,
            role="check",
            description="Reading-time fit",
        )
    manifest_path = S.store_root() / "run_1" / S.MANIFEST
    manifest = json.loads(manifest_path.read_text())
    manifest["produced_by"] = None
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="no produced_by"):
        S.publish(
            "run_1", study=STUDY, run=RUN, role="check", description="Reading-time fit"
        )


def test_hub_only_run_is_valid_and_fetchable(tree, monkeypatch):
    _, out, _ = tree
    S.save("run_1", [out], study=STUDY)
    S.publish(
        "run_1", study=STUDY, run=RUN, role="check", description="Reading-time fit"
    )
    assert not (R.STUDIES_DIR / STUDY / "results" / RUN).exists()
    assert R.validate_index(STUDY) == []
    monkeypatch.delenv(R.STORE_ENV)
    fetched = S.fetch(STUDY, RUN)
    assert sorted(p.name for p in fetched) == ["cells.tsv", "folds.tsv", "summary.json"]
    assert (
        R.hub_url(STUDY, f"{RUN}/cells.tsv")
        == f"{R.HUB_URL}/blob/{'f' * 40}/studies/{STUDY}/results/{RUN}/cells.tsv"
    )
    with pytest.raises(FileNotFoundError, match="no run"):
        S.fetch(STUDY, "robustness/none")


def test_verification_fetch_selects_required_archives_and_propagates_errors(
    tree, monkeypatch, capsys
):
    root, out, _ = tree
    S.save("run_1", [out], study=STUDY)
    S.publish(
        "run_1", study=STUDY, run=RUN, role="check", description="Reading-time fit"
    )
    document = R.read_index(STUDY)
    document["archives"] = {
        "a" * 64: {"verification_required": True, "reference": {"sha256": "a" * 64}},
        "b" * 64: {"verification_required": False, "reference": {"sha256": "b" * 64}},
        "c" * 64: {"reference": {"sha256": "c" * 64}},
    }
    R.write_index(STUDY, document)
    fetched = []

    def fetch_bundle(reference):
        fetched.append(reference)
        return root / "verified-source.tar.gz"

    monkeypatch.setattr(S, "fetch_bundle", fetch_bundle)
    assert len(S.fetch(STUDY, RUN)) == 3
    assert fetched == []
    assert C.main(["fetch", STUDY, RUN, "--verification"]) == 0
    assert fetched == [{"sha256": "a" * 64}]
    assert "verified-source.tar.gz" in capsys.readouterr().out

    def corrupt_bundle(reference):
        raise ValueError("bundle checksum mismatch")

    monkeypatch.setattr(S, "fetch_bundle", corrupt_bundle)
    with pytest.raises(ValueError, match="bundle checksum mismatch"):
        S.fetch(STUDY, verification=True)


def test_cli_round_trip(tree, capsys):
    _, out, _ = tree
    assert (
        C.main(
            [
                "save",
                "run_1",
                "--study",
                STUDY,
                "--from",
                str(out),
                "--input",
                "corpus=" + "1" * 64,
            ]
        )
        == 0
    )
    assert C.main(["verify", "run_1"]) == 0
    assert C.main(["list"]) == 0
    assert "run_1  fakestudy" in capsys.readouterr().out
    assert (
        C.main(
            [
                "publish",
                "run_1",
                "--study",
                STUDY,
                "--run",
                RUN,
                "--description",
                "Reading-time fit",
                "--role",
                "scoped",
            ]
        )
        == 0
    )
    assert "3 hub files" in capsys.readouterr().out
    assert C.main(["fetch", STUDY]) == 0
    assert capsys.readouterr().out.count("\n") == 3
    (S.store_root() / "run_1" / "results" / "cells.tsv").write_text("changed\n")
    assert C.main(["verify", "run_1"]) == 1


def test_cli_reports_a_missing_store_without_a_traceback(monkeypatch):
    monkeypatch.delenv(R.STORE_ENV, raising=False)
    with pytest.raises(SystemExit, match=R.STORE_ENV):
        C.main(["list"])


@pytest.mark.parametrize(
    "run,role,description",
    [
        ("replication/reanalysis", "evidence", "Reading-time fit"),
        ("replication/reanalysis/native/r0123456789ab", "evidence", "Reading-time fit"),
        (RUN, "published", "Reading-time fit"),
        (RUN, "evidence", None),
    ],
)
def test_publish_rejects_invalid_records_before_upload(tree, run, role, description):
    _, out, uploads = tree
    S.save("run_1", [out], study=STUDY)
    with pytest.raises(ValueError):
        S.publish("run_1", study=STUDY, run=run, role=role, description=description)
    assert uploads == []
    assert R.read_index(STUDY)["runs"] == {}


def test_publish_rejects_changed_payload_before_upload(tree):
    _, out, uploads = tree
    S.save("run_1", [out], study=STUDY)
    (S.store_root() / "run_1" / "results" / "cells.tsv").write_text("changed\n")
    with pytest.raises(ValueError, match="Store bytes differ"):
        S.publish(
            "run_1",
            study=STUDY,
            run=RUN,
            role="evidence",
            description="Reading-time fit",
        )
    assert uploads == []


def test_failed_upload_leaves_checkout_and_index_unchanged(tree, monkeypatch):
    _, out, _ = tree
    S.save("run_1", [out], study=STUDY)
    previous = R.index_path(STUDY).read_bytes()

    def fail(*args):
        raise RuntimeError("Concurrent publication")

    monkeypatch.setattr(S, "_upload", fail)
    with pytest.raises(RuntimeError, match="Concurrent publication"):
        S.publish(
            "run_1",
            study=STUDY,
            run=RUN,
            role="evidence",
            description="Reading-time fit",
        )
    assert R.index_path(STUDY).read_bytes() == previous
    assert not (R.STUDIES_DIR / STUDY / "results" / RUN).exists()


@pytest.mark.parametrize("field", ["files", "producer"])
def test_publish_cannot_replace_an_existing_result_identity(tree, field):
    _, out, uploads = tree
    S.save("run_1", [out], study=STUDY)
    S.publish(
        "run_1", study=STUDY, run=RUN, role="scoped", description="Reading-time fit"
    )
    previous = R.index_path(STUDY).read_bytes()
    count = len(uploads)
    manifest_path = S.store_root() / "run_1" / S.MANIFEST
    manifest = json.loads(manifest_path.read_text())
    if field == "producer":
        manifest["produced_by"]["commit"] = "1" * 40
    else:
        path = S.store_root() / "run_1" / "results/cells.tsv"
        path.write_text("a\tb\n9\t9\n")
        item = next(e for e in manifest["files"] if e["path"] == "results/cells.tsv")
        item["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        item["bytes"] = path.stat().st_size
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="new result record identifier"):
        S.publish(
            "run_1", study=STUDY, run=RUN, role="scoped", description="Reading-time fit"
        )
    assert len(uploads) == count
    assert R.index_path(STUDY).read_bytes() == previous


def test_publish_rejects_loose_package_outputs_before_upload(tree):
    _, out, uploads = tree
    S.save("run_1", [out], study=STUDY)
    stray = R.STUDIES_DIR / STUDY / "summary.tsv"
    stray.write_text("value\n1\n")
    with pytest.raises(ValueError, match="artifact outside indexed results"):
        S.publish(
            "run_1", study=STUDY, run=RUN, role="scoped", description="Reading-time fit"
        )
    assert not uploads


def test_upload_preserves_published_archive_bindings(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import huggingface_hub

    digest = "a" * 64
    reference = {
        "repo_id": "example/results",
        "revision": "b" * 40,
        "path": f"bundles/{STUDY}/{digest}.tar.gz",
        "sha256": digest,
        "bytes": 100,
    }
    previous = tmp_path / "previous.json"
    previous.write_text(
        json.dumps(
            {
                "runs": {},
                "archives": {
                    digest: {
                        "description": "Reading-time execution",
                        "reference": reference,
                    }
                },
            }
        )
    )
    folder = tmp_path / "upload"
    folder.mkdir()
    (folder / "index.json").write_text(json.dumps({"runs": {}, "archives": {}}))
    head = SimpleNamespace(
        sha="c" * 40,
        siblings=[SimpleNamespace(rfilename=f"studies/{STUDY}/results/index.json")],
    )
    api = SimpleNamespace(repo_info=lambda *args, **kwargs: head)
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: api)
    monkeypatch.setattr(
        huggingface_hub, "hf_hub_download", lambda *args, **kwargs: str(previous)
    )
    with pytest.raises(ValueError, match="replace an archive binding"):
        S._upload(folder, f"studies/{STUDY}/results", "Publish results")


def test_save_rejects_missing_dependency_before_writing_any_run(tree):
    root, out, _ = tree
    with pytest.raises(ValueError, match="must exist relative to the replica package"):
        S.save("bad_dependency", [out], study=STUDY, depends_on=["evaluate.py"])
    assert not (root / "store/bad_dependency").exists()
    saved = S.save(
        "good_dependency", [out], study=STUDY, depends_on=["analysis/evaluate.py"]
    )
    manifest = json.loads((saved / S.MANIFEST).read_text())
    assert manifest["produced_by"]["depends_on"] == ["analysis/evaluate.py"]
