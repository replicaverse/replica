"""Complete archive integrity and immutable publication."""

import hashlib
import io
import json
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from replica.studies.records import store


def _bundle(tmp_path, *, extras=None, changed=False, study="example2026"):
    payload = b"cell\tvalue\nfirst\t0.25\n"
    manifest = {
        "format": store.BUNDLE_FORMAT,
        "study": study,
        "producer": {"commit": "historical-producer", "environment": "frozen"},
        "files": {
            "outputs/folds.tsv": {
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        },
    }
    path = tmp_path / "run.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        rows = [
            (store.MANIFEST, json.dumps(manifest).encode()),
            ("outputs/folds.tsv", payload + b"changed" if changed else payload),
            *(extras or []),
        ]
        for name, data in rows:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return path, manifest


def test_bundle_checks_every_member_without_restamping(tmp_path):
    archive, manifest = _bundle(tmp_path)
    before = archive.read_bytes()
    assert store.verify_bundle(archive) == manifest
    assert archive.read_bytes() == before


@pytest.mark.parametrize(
    "extras,changed",
    [
        ([("REPLAY.md", b"unlisted")], False),
        ([("../outside", b"unsafe")], False),
        ([("outputs/folds.tsv", b"duplicate")], False),
        (None, True),
    ],
)
def test_invalid_bundle_is_rejected(tmp_path, extras, changed):
    archive, _ = _bundle(tmp_path, extras=extras, changed=changed)
    with pytest.raises(ValueError):
        store.verify_bundle(archive)


def _hub(
    monkeypatch,
    archive,
    *,
    exists=False,
    corrupt=False,
    head="a" * 40,
    upload_error=None,
):
    import huggingface_hub

    calls = []
    uploaded = "b" * 40
    remote = archive.parent / "remote.tar.gz"

    class API:
        def repo_info(self, *args, **kwargs):
            return SimpleNamespace(sha=head)

        def file_exists(self, *args, **kwargs):
            calls.append(("exists", kwargs))
            return exists

        def upload_file(self, **kwargs):
            calls.append(("upload", kwargs))
            if upload_error:
                raise upload_error
            remote.write_bytes(Path(kwargs["path_or_fileobj"]).read_bytes())
            return SimpleNamespace(oid=uploaded)

    def download(**kwargs):
        calls.append(("download", kwargs))
        if corrupt:
            path = archive.parent / "bad.tar.gz"
            path.write_bytes(b"bad download")
            return str(path)
        return str(remote if remote.exists() else archive)

    monkeypatch.setattr(huggingface_hub, "HfApi", API)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    return calls, head, uploaded


def test_publication_records_uploaded_commit_and_verifies_download(
    tmp_path, monkeypatch
):
    archive, manifest = _bundle(tmp_path)
    calls, head, uploaded = _hub(monkeypatch, archive)
    monkeypatch.setattr(
        store.files,
        "produced_by_now",
        lambda *a, **k: pytest.fail("Publication must not invent producer provenance"),
    )
    reference = store.publish_bundle(
        archive, study="example2026", repo_id="org/results"
    )
    assert reference["revision"] == uploaded
    assert reference["path"] == (
        f"bundles/example2026/{hashlib.sha256(archive.read_bytes()).hexdigest()}.tar.gz"
    )
    upload = next(kwargs for kind, kwargs in calls if kind == "upload")
    assert upload["parent_commit"] == head
    download = next(kwargs for kind, kwargs in calls if kind == "download")
    assert download["revision"] == uploaded
    assert download["force_download"] is True
    assert store.verify_bundle(archive) == manifest


def test_identical_bundle_reuses_verified_remote_revision(tmp_path, monkeypatch):
    archive, _ = _bundle(tmp_path)
    calls, head, _ = _hub(monkeypatch, archive, exists=True)
    reference = store.publish_bundle(archive, study="example2026")
    assert reference["revision"] == head
    assert not any(kind == "upload" for kind, _ in calls)
    assert any(kind == "download" for kind, _ in calls)


def test_bad_remote_copy_cannot_return_a_publication_reference(tmp_path, monkeypatch):
    archive, _ = _bundle(tmp_path)
    _hub(monkeypatch, archive, corrupt=True)
    with pytest.raises(ValueError, match="Downloaded bundle"):
        store.publish_bundle(archive, study="example2026")


def test_fetch_rejects_mutable_revision_before_network(tmp_path, monkeypatch):
    archive, _ = _bundle(tmp_path)
    calls, _, _ = _hub(monkeypatch, archive)
    with pytest.raises(ValueError, match="immutable revision"):
        store.fetch_bundle({"revision": "main", "sha256": "c" * 64, "bytes": 1})
    assert not calls


def test_changed_head_rejects_publication_without_download(tmp_path, monkeypatch):
    archive, _ = _bundle(tmp_path)
    calls, _, _ = _hub(monkeypatch, archive, upload_error=RuntimeError("head changed"))
    with pytest.raises(RuntimeError, match="head changed"):
        store.publish_bundle(archive, study="example2026")
    assert not any(kind == "download" for kind, _ in calls)


def test_unresolved_head_cannot_disable_publication_guard(tmp_path, monkeypatch):
    archive, _ = _bundle(tmp_path)
    calls, _, _ = _hub(monkeypatch, archive, head=None)
    with pytest.raises(ValueError, match="immutable commit"):
        store.publish_bundle(archive, study="example2026")
    assert not calls


def test_replacing_original_path_cannot_change_verified_upload(tmp_path, monkeypatch):
    archive, _ = _bundle(tmp_path)
    expected = hashlib.sha256(archive.read_bytes()).hexdigest()
    other = tmp_path / "other"
    other.mkdir()
    replacement, _ = _bundle(other, study="different2026")
    verify = store.verify_bundle

    def replace_after_check(path):
        result = verify(path)
        archive.write_bytes(replacement.read_bytes())
        return result

    _hub(monkeypatch, archive)
    monkeypatch.setattr(store, "verify_bundle", replace_after_check)
    reference = store.publish_bundle(archive, study="example2026")
    assert reference["sha256"] == expected
    assert reference["study"] == "example2026"
    assert verify(tmp_path / "remote.tar.gz")["study"] == "example2026"


@pytest.mark.parametrize(
    "manifest", [[], None, {"files": []}, {"files": {"a": "digest"}}]
)
def test_unversioned_or_nonobject_manifests_are_rejected(tmp_path, manifest):
    path = tmp_path / "invalid.tar.gz"
    data = json.dumps(manifest).encode()
    with tarfile.open(path, "w:gz") as archive:
        member = tarfile.TarInfo(store.MANIFEST)
        member.size = len(data)
        archive.addfile(member, io.BytesIO(data))
    with pytest.raises(ValueError, match="manifest"):
        store.verify_bundle(path)


@pytest.mark.parametrize("kind", ["missing", "symlink", "hardlink"])
def test_missing_or_linked_payload_is_rejected(tmp_path, kind):
    _, manifest = _bundle(tmp_path)
    path = tmp_path / "invalid.tar.gz"
    data = json.dumps(manifest).encode()
    with tarfile.open(path, "w:gz") as archive:
        member = tarfile.TarInfo(store.MANIFEST)
        member.size = len(data)
        archive.addfile(member, io.BytesIO(data))
        if kind != "missing":
            member = tarfile.TarInfo("outputs/folds.tsv")
            member.type = tarfile.SYMTYPE if kind == "symlink" else tarfile.LNKTYPE
            member.linkname = "outside"
            archive.addfile(member)
    with pytest.raises(ValueError):
        store.verify_bundle(path)


@pytest.mark.parametrize(
    "key", ["study", "repo_id", "path", "bytes", "sha256", "revision"]
)
def test_incomplete_reference_is_rejected_before_network(tmp_path, monkeypatch, key):
    archive, _ = _bundle(tmp_path)
    calls, _, _ = _hub(monkeypatch, archive)
    reference = {
        "study": "example2026",
        "repo_id": "org/results",
        "path": f"bundles/example2026/{'a' * 64}.tar.gz",
        "sha256": "a" * 64,
        "bytes": 1,
        "revision": "b" * 40,
    }
    del reference[key]
    with pytest.raises(ValueError):
        store.fetch_bundle(reference)
    assert not calls


@pytest.fixture
def bundled_resource(tmp_path, monkeypatch):
    from replica.studies.records import files

    study = "example2026"
    archive, manifest = _bundle(tmp_path, study=study)
    digest = files._sha256(archive)
    name, entry = next(iter(manifest["files"].items()))
    document = {
        "schema": files.SCHEMA,
        "runs": {},
        "resources": {
            name: dict(
                role="input",
                description="Fold scores",
                sha256=entry["sha256"],
                archive=digest,
            )
        },
        "archives": {
            digest: {
                "description": "Fold-score inputs",
                "records": {name: entry["sha256"]},
                "reference": {
                    "study": study,
                    "repo_id": "example/study-bundles",
                    "revision": "a" * 40,
                    "path": f"bundles/{study}/{digest}.tar.gz",
                    "sha256": digest,
                    "bytes": archive.stat().st_size,
                },
            }
        },
    }
    package = tmp_path / "studies" / study
    (package / "results").mkdir(parents=True)
    monkeypatch.setattr(files, "STUDIES_DIR", package.parent)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    files.write_index(study, document)
    calls = []

    def download(remote, **kwargs):
        calls.append((remote, kwargs))
        return str(archive)

    monkeypatch.setattr(files, "_hub_download", download)
    return study, name, document, calls


def test_resource_bundle_verifies_and_reuses_cached_members(
    bundled_resource, monkeypatch
):
    from replica.studies.records import files
    from replica.studies.records.resources import resource_file

    study, name, document, calls = bundled_resource
    assert files.validate_package(study, paths=["results/index.json"]) == []
    path = resource_file(study, name)
    assert path.read_bytes() == b"cell\tvalue\nfirst\t0.25\n"
    assert calls[0][1]["revision"] == "a" * 40
    assert calls[0][1]["repo_id"] == "example/study-bundles"
    monkeypatch.setattr(
        files,
        "_hub_download",
        lambda *a, **kw: pytest.fail("cached input must work offline"),
    )
    assert resource_file(study, name) == path
    path.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="cached resource differs"):
        resource_file(study, name)


@pytest.mark.parametrize("fault", ["member", "revision", "local", "provenance"])
def test_resource_rejects_invalid_binding_or_local_input(bundled_resource, fault):
    from replica.studies.records import files
    from replica.studies.records.resources import resource_file

    study, name, document, calls = bundled_resource
    binding = next(iter(document["archives"].values()))
    if fault == "member":
        binding["records"][name] = "b" * 64
    elif fault == "revision":
        binding["reference"]["revision"] = "main"
    elif fault == "provenance":
        binding["reference"]["path"] = (
            f"provenance/{binding['reference']['sha256']}.tar.gz"
        )
    else:
        path = files.STUDIES_DIR / study / name
        path.parent.mkdir(parents=True)
        path.write_bytes(b"corrupt")
    files.write_index(study, document)
    assert files.validate_package(study, paths=["results/index.json"])
    with pytest.raises(ValueError):
        resource_file(study, name)
    assert calls == []


def test_resource_accepts_concurrent_complete_cache_install(
    bundled_resource, monkeypatch
):
    import shutil

    from replica.paths import user_cache_root
    from replica.studies.records import files
    from replica.studies.records.resources import resource_file

    study, name, document, calls = bundled_resource
    digest = document["resources"][name]["archive"]
    destination = user_cache_root() / "study-bundles" / study / digest
    extract = store.extract_archive

    def install_winner(*args, **kwargs):
        payload = extract(*args, **kwargs)
        shutil.copytree(payload, destination)
        return payload

    monkeypatch.setattr(store, "extract_archive", install_winner)
    assert (
        files._sha256(resource_file(study, name))
        == document["resources"][name]["sha256"]
    )
    assert len(calls) == 1
