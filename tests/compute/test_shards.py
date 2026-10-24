"""Shard-pickle merging and provenance under ``replica.compute``."""

import hashlib
import json
import pickle

import pytest

from replica.compute.shards import merge_shards


def _write(path, payload):
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    _manifest(path, hashlib.sha256(path.name.encode()).hexdigest(), len(payload))


@pytest.mark.parametrize("cli", [False, True])
def test_merges_disjoint_shards(tmp_path, cli):
    _write(tmp_path / "tag.shard00000_00002.pkl", {"s0": [1], "s1": [2]})
    _write(tmp_path / "tag.shard00002_00004.pkl", {"s2": [3], "s3": [4]})
    out = tmp_path / "tag.pkl"

    pattern = str(tmp_path / "tag.shard*.pkl")
    if cli:
        from replica.cli import main

        main(
            [
                "compute",
                "merge-shards",
                "--pattern",
                pattern,
                "--out",
                str(out),
                "--expect-stimuli",
                "4",
            ]
        )
    else:
        assert merge_shards(pattern, out, expect_stimuli=4) == 4
    with open(out, "rb") as f:
        assert pickle.load(f) == {"s0": [1], "s1": [2], "s2": [3], "s3": [4]}


def test_overlapping_shards_abort(tmp_path):
    _write(tmp_path / "tag.shard00000_00002.pkl", {"s0": [1]})
    _write(tmp_path / "tag.shard00001_00003.pkl", {"s0": [9]})  # duplicate id

    with pytest.raises(SystemExit, match="overlapping"):
        merge_shards(str(tmp_path / "tag.shard*.pkl"), tmp_path / "tag.pkl")


def test_no_matching_shards_abort(tmp_path):
    with pytest.raises(SystemExit, match="no shards"):
        merge_shards(str(tmp_path / "nope.shard*.pkl"), tmp_path / "out.pkl")


def test_delete_shards_after_merge(tmp_path):
    p1 = tmp_path / "tag.shard00000_00001.pkl"
    _write(p1, {"s0": [1]})

    merge_shards(
        str(tmp_path / "tag.shard*.pkl"),
        tmp_path / "tag.pkl",
        delete_shards=True,
        expect_stimuli=1,
    )

    assert not p1.exists()


def _manifest(path, corpus, n, *, source="s" * 64, device="gpu-a", status="completed"):
    from replica.artifacts import write_manifest

    identity = {
        "schema_version": 1,
        "method": "surprisal",
        "options": {"model_id": "gpt2"},
        "model": {"revision": "r"},
        "corpus_sha256": corpus,
        "n_stimuli": n,
        "source_sha256": source,
        "versions": {"torch": "2.13"},
    }
    write_manifest(
        path,
        identity,
        status=status,
        environment={"code_commit": "c" * 40, "device": device},
    )


def test_merged_pickle_inherits_the_provenance_its_shards_share(tmp_path):
    import json

    from replica.artifacts import manifest_path, validate_artifact

    a, b = tmp_path / "tag.shard00000_00002.pkl", tmp_path / "tag.shard00002_00004.pkl"
    _write(a, {"s0": [1], "s1": [2]})
    _write(b, {"s2": [3], "s3": [4]})
    _manifest(a, "a" * 64, 2, device="gpu-a")
    _manifest(b, "b" * 64, 2, device="gpu-b")
    out = tmp_path / "tag.pkl"

    merge_shards(str(tmp_path / "tag.shard*.pkl"), out, expect_stimuli=4)

    identity = validate_artifact(out)
    assert identity["n_stimuli"] == 4 and identity["shards"] == 2
    assert identity["source_sha256"] == "s" * 64
    assert identity["corpus_sha256"] not in ("a" * 64, "b" * 64)
    merged = json.loads(manifest_path(out).read_text())
    environment = merged["environment"]
    assert "device" not in environment
    assert environment["code_commit"] == "c" * 40
    assert environment["shards"] == [
        {"shard": a.name, "environment": {"code_commit": "c" * 40, "device": "gpu-a"}},
        {"shard": b.name, "environment": {"code_commit": "c" * 40, "device": "gpu-b"}},
    ]
    assert "multiword" not in merged


def test_merged_pickle_preserves_scientific_metadata(tmp_path):
    import json

    from replica.artifacts import manifest_path

    block = {"over": "whitespace_words", "rule": "sum", "multiword_units": 3}
    a, b = tmp_path / "tag.shard00000_00002.pkl", tmp_path / "tag.shard00002_00004.pkl"
    _write(a, {"s0": [1], "s1": [2]})
    _write(b, {"s2": [3], "s3": [4]})
    _manifest(a, "a" * 64, 2, device="gpu-a")
    _manifest(b, "b" * 64, 2, device="gpu-b")
    for shard in (a, b):
        sidecar = manifest_path(shard)
        manifest = json.loads(sidecar.read_text())
        manifest["multiword"] = dict(block)
        manifest["window"] = {"max_positions": 512, "rule": "clark"}
        manifest["sampling"] = {"seed": 7, "scheme": "model-bundle-v1"}
        sidecar.write_text(json.dumps(manifest))
    out = tmp_path / "tag.pkl"
    merge_shards(str(tmp_path / "tag.shard*.pkl"), out, expect_stimuli=4)
    merged = json.loads(manifest_path(out).read_text())
    for key in ("multiword", "window", "sampling"):
        assert merged[key] == manifest[key]


def test_merged_pickle_unites_the_audits_of_accepted_shards(tmp_path):
    import json

    from replica.artifacts import manifest_path

    a, b = tmp_path / "tag.shard00000_00002.pkl", tmp_path / "tag.shard00002_00004.pkl"
    _write(a, {"s0": [1], "s1": [2]})
    _write(b, {"s2": [3], "s3": [4]})
    _manifest(a, "a" * 64, 2)
    _manifest(b, "b" * 64, 2)
    sidecar = manifest_path(a)
    manifest = json.loads(sidecar.read_text())
    manifest["audit"] = {
        "hard_issues": ["nan_burst"],
        "soft_issues": [],
        "examples": {"nan_burst": ["s0: 1/1 NaN"]},
        "failed_stimuli": 0,
        "missing_stimuli": 0,
        "accepted": {"reason": "definitional", "at": "2026-10-06T20:00:00+00:00"},
    }
    sidecar.write_text(json.dumps(manifest))
    out = tmp_path / "tag.pkl"
    merge_shards(str(tmp_path / "tag.shard*.pkl"), out, expect_stimuli=4)
    audit = json.loads(manifest_path(out).read_text())["audit"]
    assert audit["hard_issues"] == ["nan_burst"] and audit["soft_issues"] == []
    assert audit["examples"] == {"nan_burst": ["s0: 1/1 NaN"]}
    assert audit["failed_stimuli"] == 0 and audit["missing_stimuli"] == 0
    assert audit["accepted"] == [
        {
            "shard": a.name,
            "reason": "definitional",
            "at": "2026-10-06T20:00:00+00:00",
        }
    ]


@pytest.mark.parametrize(
    "fault",
    [
        "checksum",
        "missing",
        "identity",
        "window",
        "sampling",
        "multiword",
        "derivation",
        "external",
        "count",
        "count_type",
    ],
)
def test_invalid_shards_preserve_existing_output_and_inputs(tmp_path, fault):
    from replica.artifacts import manifest_path

    a, b = tmp_path / "tag.shard00000_00001.pkl", tmp_path / "tag.shard00001_00002.pkl"
    _write(a, {"s0": [1]})
    _write(b, {"s1": [2]})
    out = tmp_path / "tag.pkl"
    _write(out, {"previous": [7]})
    if fault == "checksum":
        b.write_bytes(pickle.dumps({"s1": [999]}))
    elif fault == "missing":
        manifest_path(b).unlink()
    else:
        sidecar = manifest_path(b)
        manifest = json.loads(sidecar.read_text())
        if fault == "identity":
            manifest["identity"]["source_sha256"] = "different"
        elif fault in {"count", "count_type"}:
            manifest["identity"]["n_stimuli"] = 3 if fault == "count" else "1"
        else:
            manifest[fault] = {"different": True}
        sidecar.write_text(json.dumps(manifest))
    preserved = [a, b, out, manifest_path(a), manifest_path(out)]
    if manifest_path(b).exists():
        preserved.append(manifest_path(b))
    before = {path: path.read_bytes() for path in preserved}
    message = {
        "checksum": "mismatched manifest",
        "missing": "missing artifact manifest",
        "identity": "identities disagree",
        "count": "payload count",
        "count_type": "n_stimuli must be a positive integer",
    }.get(fault, fault)
    with pytest.raises(ValueError, match=message):
        merge_shards(
            str(tmp_path / "tag.shard*.pkl"), out, delete_shards=True, expect_stimuli=2
        )
    assert {path: path.read_bytes() for path in preserved} == before


def test_merge_respects_locks_aliases_and_restores_failed_publication(
    tmp_path, monkeypatch
):
    import fcntl
    import os
    import sys
    from pathlib import Path

    from replica.artifacts import manifest_path

    shard = tmp_path / "tag.shard00000_00001.pkl"
    out = tmp_path / "tag.pkl"
    _write(shard, {"s0": [1]})
    _write(out, {"previous": [7]})
    preserved = [shard, manifest_path(shard), out, manifest_path(out)]
    before = {path: path.read_bytes() for path in preserved}
    pattern = str(tmp_path / "tag.shard*.pkl")
    with monkeypatch.context() as unsupported:
        unsupported.setitem(sys.modules, "fcntl", None)
        with pytest.raises(RuntimeError, match="POSIX file locking"):
            merge_shards(pattern, out, delete_shards=True, expect_stimuli=1)
    assert {path: path.read_bytes() for path in preserved} == before
    for owner in (shard, out):
        with owner.with_name(owner.name + ".lock").open("a") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            with pytest.raises(RuntimeError, match="locked"):
                merge_shards(pattern, out, delete_shards=True, expect_stimuli=1)
    with pytest.raises(ValueError, match="alias"):
        merge_shards(pattern, shard, delete_shards=True, expect_stimuli=1)
    alias = tmp_path / "alias.pkl"
    for link in (lambda: alias.symlink_to(shard), lambda: os.link(shard, alias)):
        link()
        with pytest.raises(ValueError, match="alias|symbolic"):
            merge_shards(pattern, alias, delete_shards=True, expect_stimuli=1)
        alias.unlink()
    replace = Path.replace

    def fail_sidecar(path, target):
        if path.name == "payload.pkl.manifest.json":
            raise OSError("injected publication failure")
        return replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_sidecar)
    with pytest.raises(OSError, match="injected"):
        merge_shards(pattern, out, delete_shards=True, expect_stimuli=1)
    assert {path: path.read_bytes() for path in preserved} == before

    def fail_restore(path, target):
        if path.name in {"payload.pkl.manifest.json", "before-0"}:
            raise OSError("injected persistent failure")
        return replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_restore)
    with pytest.raises(OSError, match="recovery files retained"):
        merge_shards(pattern, out, delete_shards=True, expect_stimuli=1)
    (recovery,) = tmp_path.glob(".tag.pkl.merge-*")
    assert (recovery / "before-0").read_bytes() == before[out]
    assert manifest_path(out).read_bytes() == before[manifest_path(out)]
    assert shard.read_bytes() == before[shard]
    assert manifest_path(shard).read_bytes() == before[manifest_path(shard)]


_TABLE = "gpt2_surprisal_tokens_provo"


def _compute_shards(monkeypatch, tmp_path):
    """Two one-stimulus shards of a token table, computed by the runner."""
    from replica.compute import runner
    from tests.feature.core.annotation.test_token_tables import _install

    common = _install(monkeypatch, tmp_path)
    common["model_id"] = "gpt2"
    for stim_slice in ((0, 1), (1, 2)):
        outcome = runner.compute_for(
            "provo", method="surprisal", unit="tokens", stim_slice=stim_slice, **common
        )
        assert outcome.status == "completed", outcome.error
    return runner, common


def test_an_unsliced_run_reuses_the_merged_cell(tmp_path, monkeypatch):
    """The merged sidecar records the identity an unsliced run requires."""
    from types import SimpleNamespace

    import pyarrow.parquet as pq

    from replica.artifacts import manifest_path
    from replica.predictor_store import layout
    from replica.predictor_store.cells import discover
    from replica.predictor_store.export import export

    runner, common = _compute_shards(monkeypatch, tmp_path)
    out = tmp_path / f"{_TABLE}.pkl"
    producers = []
    for index, shard in enumerate(sorted(tmp_path.glob("*.shard*.pkl"))):
        sidecar = manifest_path(shard)
        manifest = json.loads(sidecar.read_text())
        manifest["environment"] = {
            "device": f"gpu-{index}",
            "code_commit": str(index) * 40,
        }
        sidecar.write_text(json.dumps(manifest))
        producers.append({"shard": shard.name, "environment": manifest["environment"]})

    merge_shards(str(tmp_path / f"{_TABLE}.shard*.pkl"), out, delete_shards=True)

    assert not list(tmp_path.glob("*.shard*.pkl"))
    assert not list(tmp_path.glob("*.shard*.manifest.json"))
    whole = runner.compute_for("provo", method="surprisal", unit="tokens", **common)
    assert whole.status == "cached" and whole.path == out
    found = discover([("merged", tmp_path)], ["provo"])
    assert len(found.completed) == 1 and not found.problems
    stage = tmp_path / "stage"
    corpora = SimpleNamespace(
        names=lambda: ["provo"],
        stimuli=lambda name: runner._load_dataset(name).stimuli,
        word_spacing=lambda name: True,
    )
    report = export(stage, [("merged", tmp_path)], corpora=corpora)
    assert report.ok and report.cells == 1, report.summary()
    saved = json.loads((stage / layout.MANIFESTS_DIR / f"{_TABLE}.json").read_text())
    assert saved["environment"]["shards"] == producers
    index_rows = pq.read_table(stage / layout.INDEX_FILE).to_pylist()
    assert index_rows and all(
        row["device"] is None and row["code_commit"] is None for row in index_rows
    )


def test_shards_of_other_stimuli_fail_before_publication(tmp_path, monkeypatch):
    """A changed corpus cannot reuse shards from the earlier input."""
    from copy import deepcopy

    from replica.artifacts import manifest_path
    from tests._stimuli import word_stimulus
    from tests.feature.core.annotation.test_token_tables import STIMULI

    runner, _ = _compute_shards(monkeypatch, tmp_path)
    changed = [STIMULI[0], word_stimulus("A second, longer one.", "s1")]
    monkeypatch.setattr(
        runner,
        "_load_dataset",
        lambda name: runner._LoadedDataset(deepcopy(changed), 2, "en", True),
    )
    out = tmp_path / f"{_TABLE}.pkl"

    with pytest.raises(ValueError, match="loaded corpus"):
        merge_shards(str(tmp_path / f"{_TABLE}.shard*.pkl"), out)
    assert not out.exists() and not manifest_path(out).exists()
