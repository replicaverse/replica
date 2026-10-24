"""Publishing a staged tree through a stand-in hub client."""

import importlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from replica.predictor_store import layout
from replica.predictor_store.export import export
from replica.predictor_store.publish import file_history, publish
from replica.predictor_store.verify import verify
from replica.utils.digest import file_sha256
from tests.predictor_store._cells import (
    DATASET,
    FakeHub,
    ToyCorpora,
    token_payload,
    word_payload,
    write_cell,
)

REPO = "example/predictors"


def _entry(path, blob, sha=None):
    lfs = SimpleNamespace(sha256=sha) if sha else None
    return SimpleNamespace(path=path, blob_id=blob, lfs=lfs)


class FakeApi:
    """Records the upload and answers the history walk from a fixed tree."""

    def __init__(self):
        self.calls = []
        self.uploaded = {}
        first = datetime(2026, 9, 1, tzinfo=UTC)
        second = datetime(2026, 9, 21, tzinfo=UTC)
        self.commits = {
            None: [SimpleNamespace(commit_id="1" * 40, created_at=first)],
            "refs/pr/3": [
                SimpleNamespace(commit_id="2" * 40, created_at=second),
                SimpleNamespace(commit_id="1" * 40, created_at=first),
            ],
        }
        self.trees = {
            "1" * 40: [
                _entry("tags.parquet", "b1", "s1"),
                _entry("provo.parquet", "b2", "s2"),
                _entry("mcword_samples/old.parquet", "b3", "s3"),
                SimpleNamespace(path="mcword_samples", tree_id="t1"),
            ],
            "2" * 40: [
                _entry("tags.parquet", "b4", "s4"),
                _entry("provo.parquet", "b2", "s2"),
                _entry("tokens/provo.parquet", "b5", "s5"),
            ],
        }

    def upload_folder(self, **kwargs):
        self.calls.append(("upload_folder", kwargs))
        root = Path(kwargs["folder_path"])
        self.uploaded = {
            file: (root / file).read_bytes() for file in kwargs["allow_patterns"]
        }
        return SimpleNamespace(
            oid="2" * 40,
            pr_url=f"https://huggingface.co/datasets/{REPO}/discussions/3",
            pr_revision="refs/pr/3",
            pr_num=3,
        )

    def upload_file(self, **kwargs):
        self.calls.append(("upload_file", kwargs))
        return SimpleNamespace(oid="3" * 40)

    def list_repo_commits(self, repo, *, repo_type, revision=None):
        assert repo == REPO and repo_type == "dataset"
        return list(self.commits[revision])

    def list_repo_tree(self, repo, *, repo_type, recursive, revision):
        assert repo == REPO and repo_type == "dataset" and recursive
        return list(self.trees[revision])


def _verify_stage(stage, sources=None):
    return verify(
        stage,
        [("w", stage.parent / "out")] if sources is None else sources,
        corpora=ToyCorpora(),
    )


def _edit_plan(stage, **updates):
    path = stage / layout.PUBLISH_FILE
    plan = json.loads(path.read_text())
    plan.update(updates)
    path.write_text(json.dumps(plan))


@pytest.fixture
def stage(tmp_path):
    out, root = tmp_path / "out", tmp_path / "stage"
    write_cell(out, f"gpt2_surprisal_{DATASET}", word_payload(), method="surprisal")
    write_cell(
        out,
        f"gpt2_surprisal_tokens_{DATASET}",
        token_payload(),
        method="surprisal",
        unit="tokens",
    )
    hub = FakeHub(tmp_path / "hub", [], {})
    assert export(root, [("w", out)], corpora=ToyCorpora(), hub=hub).ok
    _edit_plan(root, deleted=["mcword_samples/old.parquet"])
    assert _verify_stage(root).ok
    return root


def test_publish_uploads_the_written_files_and_deletes_the_replaced_ones(stage):
    api = FakeApi()
    result = publish(stage, REPO, create_pr=True, message="Release", api=api)
    (name, upload), (name_2, history) = api.calls
    assert name == "upload_folder" and name_2 == "upload_file"
    assert upload["repo_id"] == REPO and upload["repo_type"] == "dataset"
    assert upload["folder_path"] != str(stage)
    assert not Path(upload["folder_path"]).exists()
    assert upload["create_pr"] is True
    assert upload["commit_message"] == "Release"
    assert upload["parent_commit"] == "a" * 40
    plan = json.loads((stage / layout.PUBLISH_FILE).read_text())
    assert set(upload["allow_patterns"]) == set(plan["written"]) | {
        layout.PUBLISH_FILE,
        layout.READBACK_FILE,
    }
    receipt = json.loads(api.uploaded[layout.READBACK_FILE])
    assert receipt["plan_sha256"] == file_sha256(stage / layout.PUBLISH_FILE)
    assert receipt["files"] == {
        file: file_sha256(stage / file) for file in plan["written"]
    }
    assert upload["delete_patterns"] == ["mcword_samples/old.parquet"]
    assert history["path_in_repo"] == layout.HISTORY_FILE
    assert history["revision"] == "refs/pr/3"
    assert result["pr_revision"] == "refs/pr/3" and result["commit"] == "2" * 40
    rows = {
        r["file"]: r for r in pq.read_table(stage / layout.HISTORY_FILE).to_pylist()
    }
    assert result["history_files"] == len(rows) == 4
    assert rows["provo.parquet"]["published_in"] == "1" * 40
    assert rows["provo.parquet"]["last_modified_in"] == "1" * 40
    assert rows["tags.parquet"]["last_modified_in"] == "2" * 40
    assert rows["tags.parquet"]["published_in"] == "1" * 40
    assert rows["tokens/provo.parquet"]["published_at"].startswith("2026-09-21")
    assert rows["mcword_samples/old.parquet"]["present"] is False
    assert rows["tokens/provo.parquet"]["present"] is True


def test_publish_refuses_a_stage_with_failures_or_another_repo(stage):
    plan = json.loads((stage / layout.PUBLISH_FILE).read_text())
    plan["failures"] = ["gpt2_surprisal_provo.pkl: ValueError: no exportable values"]
    (stage / layout.PUBLISH_FILE).write_text(json.dumps(plan))
    assert _verify_stage(stage).ok
    api = FakeApi()
    with pytest.raises(ValueError, match="1 failure"):
        publish(stage, REPO, api=api)
    assert api.calls == []
    with pytest.raises(ValueError, match="merged against"):
        publish(stage, "someone/else", api=api, allow_failures=True)
    result = publish(stage, REPO, api=api, allow_failures=True)
    assert result["commit"] == "2" * 40


def test_publish_refuses_an_unmerged_stage_unless_it_is_the_initial_release(stage):
    """A stage exported without the hub holds only its own run's cells."""
    plan = json.loads((stage / layout.PUBLISH_FILE).read_text())
    plan.update(repo=None, base_revision=None)
    (stage / layout.PUBLISH_FILE).write_text(json.dumps(plan))
    assert _verify_stage(stage).ok
    api = FakeApi()
    with pytest.raises(ValueError, match="without --merge-hub"):
        publish(stage, REPO, create_pr=False, api=api)
    assert api.calls == []
    publish(stage, REPO, create_pr=False, api=api, initial_release=True)
    (name, upload), _ = api.calls
    assert name == "upload_folder" and upload["parent_commit"] is None


def test_file_history_walks_the_default_branch_without_a_revision():
    api = FakeApi()
    rows = file_history(api, REPO)
    assert [r["file"] for r in rows] == [
        "mcword_samples/old.parquet",
        "provo.parquet",
        "tags.parquet",
    ]
    assert all(r["present"] for r in rows)


def _change_value(stage):
    path = stage / f"{DATASET}.parquet"
    table = pq.read_table(path)
    values = table["value"].to_pylist()
    values[0] = -999.0
    table = table.set_column(
        table.schema.get_field_index("value"), "value", pa.array(values)
    )
    pq.write_table(table, path)


@pytest.mark.parametrize("change", ["value", "file", "plan", "receipt", "missing"])
def test_publish_refuses_unverified_bytes_even_with_allow_failures(stage, change):
    if change == "value":
        _change_value(stage)
    elif change == "file":
        (stage / layout.README_FILE).write_text("changed\n")
    elif change == "plan":
        _edit_plan(stage, deleted=[])
    elif change == "receipt":
        path = stage / layout.READBACK_FILE
        receipt = json.loads(path.read_text())
        receipt["files"].pop(layout.README_FILE)
        path.write_text(json.dumps(receipt))
    else:
        (stage / layout.READBACK_FILE).unlink()
    api = FakeApi()
    with pytest.raises(ValueError, match="readback"):
        publish(stage, REPO, api=api, allow_failures=True)
    assert api.calls == []


def test_failed_repeat_verify_invalidates_a_previous_success(stage):
    _change_value(stage)
    report = _verify_stage(stage)
    assert not report.ok and any("staged -999.0" in f for f in report.failures)
    assert not (stage / layout.READBACK_FILE).exists()
    api = FakeApi()
    with pytest.raises(ValueError, match="readback"):
        publish(stage, REPO, api=api, allow_failures=True)
    assert api.calls == []


@pytest.mark.parametrize("operation", [export, verify])
def test_exception_in_export_or_verify_invalidates_the_receipt(stage, operation):
    class BrokenCorpora(ToyCorpora):
        def names(self):
            raise RuntimeError("loader failed")

    with pytest.raises(RuntimeError, match="loader failed"):
        operation(stage, [("w", stage.parent / "out")], corpora=BrokenCorpora())
    assert not (stage / layout.READBACK_FILE).exists()


@pytest.mark.parametrize("omit", ["one", "all"])
def test_missing_fresh_sources_cannot_be_relabelled_as_carried(stage, tmp_path, omit):
    coverage = pq.read_table(stage / layout.COVERAGE_FILE)
    rows = coverage.to_pylist()
    for row in rows:
        row["status"] = layout.CARRIED
    pq.write_table(
        pa.Table.from_pylist(rows, schema=coverage.schema), stage / layout.COVERAGE_FILE
    )
    subset = tmp_path / "subset"
    subset.mkdir()
    if omit == "one":
        for path in (stage.parent / "out").glob(f"gpt2_surprisal_{DATASET}.pkl*"):
            shutil.copyfile(path, subset / path.name)
    report = _verify_stage(stage, [("w", subset)])
    assert not report.ok
    assert any("complete fresh_cells" in f for f in report.failures)
    assert not (stage / layout.READBACK_FILE).exists()


@pytest.mark.parametrize("change", ["stage", "source", "plan"])
def test_changes_during_readback_cannot_mint_a_receipt(stage, monkeypatch, change):
    module = importlib.import_module("replica.predictor_store.verify")
    original = module._check_text_units

    def change_after_comparison(root, report):
        original(root, report)
        if change == "stage":
            _change_value(root)
        elif change == "source":
            path = stage.parent / "out" / f"gpt2_surprisal_{DATASET}.pkl"
            path.write_bytes(path.read_bytes() + b"changed")
        else:
            _edit_plan(root, deleted=[])

    monkeypatch.setattr(module, "_check_text_units", change_after_comparison)
    report = _verify_stage(stage)
    assert not report.ok and any(
        "changed during readback" in f for f in report.failures
    )
    assert not (stage / layout.READBACK_FILE).exists()


def test_publication_is_portable_and_uploads_a_separate_verified_snapshot(
    stage, tmp_path
):
    moved = tmp_path / "other-host" / "stage"
    shutil.copytree(stage, moved)
    shutil.rmtree(stage.parent / "out")
    expected = (moved / f"{DATASET}.parquet").read_bytes()

    class MutatingApi(FakeApi):
        def upload_folder(self, **kwargs):
            _change_value(moved)
            return super().upload_folder(**kwargs)

    api = MutatingApi()
    publish(moved, REPO, api=api)
    assert api.uploaded[f"{DATASET}.parquet"] == expected
    assert (moved / f"{DATASET}.parquet").read_bytes() != expected
    assert not Path(api.calls[0][1]["folder_path"]).exists()


def test_temporary_snapshot_is_removed_on_upload_failure(stage):
    class FailingApi(FakeApi):
        def upload_folder(self, **kwargs):
            super().upload_folder(**kwargs)
            raise RuntimeError("upload failed")

    api = FailingApi()
    with pytest.raises(RuntimeError, match="upload failed"):
        publish(stage, REPO, api=api)
    assert not Path(api.calls[0][1]["folder_path"]).exists()


def test_mutation_during_snapshot_copy_is_rejected_before_api_construction(
    stage, monkeypatch
):
    module = importlib.import_module("replica.predictor_store._readback")
    original = module._stage_file

    def change_before_copy(root, file):
        if file == f"{DATASET}.parquet":
            _change_value(root)
        return original(root, file)

    monkeypatch.setattr(module, "_stage_file", change_before_copy)
    monkeypatch.setattr("huggingface_hub.HfApi", lambda: pytest.fail("API constructed"))
    with pytest.raises(ValueError, match="bytes changed"):
        publish(stage, REPO)


@pytest.mark.parametrize(
    "path", ["../outside", "/absolute", "tokens/../outside", "*.parquet"]
)
def test_plan_paths_are_literal_and_relative(stage, path):
    _edit_plan(stage, deleted=[path])
    report = _verify_stage(stage)
    assert not report.ok and any("stage-relative" in f for f in report.failures)


def test_old_plan_requires_reexport_and_compared_files_must_be_written(stage):
    path = stage / layout.PUBLISH_FILE
    original = json.loads(path.read_text())
    legacy = {key: value for key, value in original.items() if key != "fresh_cells"}
    path.write_text(json.dumps(legacy))
    assert "re-export" in _verify_stage(stage).failures[0]
    original["written"].remove(f"{DATASET}.parquet")
    path.write_text(json.dumps(original))
    assert any("omits compared files" in f for f in _verify_stage(stage).failures)
