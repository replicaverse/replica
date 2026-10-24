"""Catalog coverage, immutable identities and result ownership."""

import hashlib
import io
import json
import subprocess
import tarfile
from copy import deepcopy

import pytest

from replica.studies.records import catalog

REVISION = "a" * 40
REPO = "example/results"
PREFIX = "src/replica/studies/example2026/results/"
RUN = "reproduction/repetition/reading_time/r0123456789ab"
CHECK = "reproduction/repetition/reading_time/rabcdef012345"


def _git(root, *args):
    return subprocess.check_output(["git", *args], cwd=root, stderr=subprocess.PIPE)


def _commit(root):
    _git(root, "add", ".")
    _git(
        root,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.org",
        "commit",
        "-qm",
        "Add artifacts",
    )


def _identity(name, body):
    return {
        "path": name,
        "bytes": len(body),
        "sha256": hashlib.sha256(body).hexdigest(),
        "git_blob": hashlib.sha1(f"blob {len(body)}\0".encode() + body).hexdigest(),
    }


@pytest.fixture
def source(tmp_path):
    _git(tmp_path, "init", "-q")
    _git(
        tmp_path,
        "remote",
        "add",
        "origin",
        "https://github.com/replicaverse/replica-dev.git",
    )
    root = tmp_path / PREFIX
    (root / RUN).mkdir(parents=True)
    (root / CHECK).mkdir(parents=True)
    body = b"value\n1\n"
    (root / RUN / "cells.tsv").write_bytes(body)
    (root / CHECK / "detail.tsv").write_bytes(body)
    remote = _identity(f"studies/example2026/results/{RUN}/cells.tsv", body)
    (root / "index.json").write_text(
        json.dumps(
            {
                "runs": {
                    RUN: {
                        "role": "evidence",
                        "description": "Reading-time fit",
                        "files": {"cells.tsv": remote["sha256"]},
                        "produced_by": None,
                        "hub": {"cells.tsv": remote["sha256"]},
                    },
                    CHECK: {
                        "role": "check",
                        "produced_by": None,
                        "description": "Independent fit",
                        "files": {"detail.tsv": remote["sha256"]},
                    },
                }
            }
        )
    )
    (root / "README.md").write_text("# Results\n")
    _commit(tmp_path)
    snapshot = {"repo_id": REPO, "revision": REVISION, "objects": [remote]}
    receipts = [{**remote, "repo_id": REPO, "revision": REVISION, "verified": True}]
    return tmp_path, snapshot, receipts


def test_run_ownership_and_missing_producer_are_preserved(source):
    result = catalog.build_catalog(*source)
    assert [r["run"] for r in result["runs"]] == [RUN, CHECK]
    assert [len(r["files"]) for r in result["runs"]] == [1, 1]
    assert all(r["produced_by"] is None for r in result["runs"])
    assert len(result["source_metadata"]) == 2
    assert result["files"][0]["references"][0]["run"] == RUN
    assert result["files"][0]["classification"] == "indexed"
    assert str(source[0]) not in json.dumps(result)
    assert catalog.build_catalog(*source) == result


def test_catalog_rejects_outputs_outside_result_records(source):
    root, snapshot, receipts = source
    path = root / "src/replica/studies/example2026/figures/summary.csv"
    path.parent.mkdir()
    path.write_text("value\n1\n")
    _commit(root)
    with pytest.raises(ValueError, match="artifact outside indexed results"):
        catalog.build_catalog(root, snapshot, receipts)


@pytest.mark.parametrize(
    "field,value",
    [
        ("revision", "b" * 40),
        ("repo_id", "different/results"),
        ("sha256", "0" * 64),
        ("git_blob", "0" * 40),
        ("bytes", 999),
        ("verified", False),
    ],
)
def test_changed_verification_identity_is_rejected(source, field, value):
    root, snapshot, receipts = source
    receipts[0][field] = value
    with pytest.raises(ValueError, match="Verification identity"):
        catalog.build_catalog(root, snapshot, receipts)


def test_missing_and_duplicate_receipts_are_rejected(source):
    root, snapshot, receipts = source
    for changed in ([], receipts * 2):
        with pytest.raises(ValueError, match="Verification must cover|Duplicate"):
            catalog.build_catalog(root, snapshot, changed)


def test_unindexed_and_modified_local_results_are_rejected(source):
    root, snapshot, receipts = source
    path = root / PREFIX / "unlisted.tsv"
    path.write_text("value\n2\n")
    _commit(root)
    with pytest.raises(ValueError, match="no indexed run"):
        catalog.build_catalog(root, snapshot, receipts)
    path.write_text("value\n3\n")
    with pytest.raises(ValueError, match="no tracked changes"):
        catalog.build_catalog(root, snapshot, receipts)


def test_retained_archive_origins_required_and_rendered_in_path_order(
    provenance_archive,
):
    root, snapshot, receipts, _ = provenance_archive
    extra = next(r for r in receipts if r["path"].startswith("provenance/"))
    with pytest.raises(ValueError, match="origin record"):
        catalog.build_catalog(root, snapshot, receipts)
    retained = {
        "retained_records": {
            extra["path"]: {
                "description": "Fitted result with source metadata.",
                "metadata_paths": [extra["path"]],
            }
        }
    }
    result = catalog.build_catalog(root, snapshot, receipts, retained)
    assert [r["path"] for r in result["files"]] == sorted(
        r["path"] for r in result["files"]
    )
    archived = next(r for r in result["files"] if r["path"] == extra["path"])
    assert archived["classification"] == "retained"
    readme = catalog.render_readme(result)
    assert (
        f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{extra['path']}"
        in readme
    )
    bad = deepcopy(retained)
    bad["retained_records"][extra["path"]]["superseded_by"] = extra["path"]
    with pytest.raises(ValueError, match="self-referencing"):
        catalog.build_catalog(root, snapshot, receipts, bad)


def test_loose_hub_files_are_rejected_even_with_origin_records(source):
    root, snapshot, receipts = source
    extra = _identity("example2026/earlier/provenance.json", b"{}")
    snapshot["objects"].append(extra)
    receipts.append({**extra, "repo_id": REPO, "revision": REVISION, "verified": True})
    retained = {
        "retained_records": {
            extra["path"]: {
                "description": "Source metadata",
                "metadata_paths": [extra["path"]],
            }
        }
    }
    with pytest.raises(ValueError, match="Noncanonical Hub object path"):
        catalog.build_catalog(root, snapshot, receipts, retained)


def test_download_checks_regular_git_objects(monkeypatch, tmp_path):
    import huggingface_hub

    path = tmp_path / "README.md"
    path.write_bytes(b"original")
    item = _identity("README.md", path.read_bytes())
    item["sha256"] = None
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *a, **kw: str(path))
    receipt = catalog.verify_object(REPO, REVISION, item)
    assert receipt["sha256"] == hashlib.sha256(b"original").hexdigest()
    path.write_bytes(b"modified")
    with pytest.raises(ValueError, match="differs from Hub identity"):
        catalog.verify_object(REPO, REVISION, item)
    with pytest.raises(ValueError, match="immutable"):
        catalog.verify_object(REPO, "main", item)


def test_bundle_members_are_verified_and_required(monkeypatch, source):
    import huggingface_hub

    root, snapshot, receipts = source
    path = root / "bundle.tar.gz"
    body = b"result"
    manifest = {
        "format": "replica.run-bundle.v1",
        "study": "example2026",
        "files": {
            "run/results.tsv": {
                "bytes": len(body),
                "sha256": hashlib.sha256(body).hexdigest(),
            }
        },
    }
    with tarfile.open(path, "w:gz") as archive:
        for name, data in [
            ("MANIFEST.json", json.dumps(manifest).encode()),
            ("run/results.tsv", body),
        ]:
            member = tarfile.TarInfo(name)
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
    item = _identity("placeholder", path.read_bytes())
    item["path"] = f"bundles/example2026/{item['sha256']}.tar.gz"
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *a, **kw: str(path))
    receipt = catalog.verify_object(REPO, REVISION, item)
    assert receipt["manifest"] == manifest
    snapshot["objects"].append(item)
    receipts.append(receipt)
    reference = root / PREFIX / "index.json"
    document = json.loads(reference.read_text())
    document["archives"] = {
        item["sha256"]: {
            "description": "Reading-time execution outputs",
            "reference": {**item, "repo_id": REPO, "revision": REVISION},
            "records": {
                "run/results.tsv": manifest["files"]["run/results.tsv"]["sha256"]
            },
        }
    }
    reference.write_text(json.dumps(document))
    path.unlink()
    _commit(root)
    result = catalog.build_catalog(root, snapshot, receipts)
    assert result["files"][0]["members"] == manifest["files"]
    document["archives"][item["sha256"]]["records"]["run/results.tsv"] = "0" * 64
    reference.write_text(json.dumps(document))
    _commit(root)
    with pytest.raises(ValueError, match="Archive member record differs"):
        catalog.build_catalog(root, snapshot, receipts)
    del receipt["manifest"]
    with pytest.raises(ValueError, match="member verification is absent"):
        catalog.build_catalog(root, snapshot, receipts)


def test_local_provenance_retains_identity_roles_without_restamping(source):
    root, snapshot, receipts = source
    path = root / PREFIX / RUN / "source.json"
    path.write_text(
        json.dumps({"producer_commit": "b" * 40, "packaging_commit": "c" * 40})
    )
    index = root / PREFIX / "index.json"
    document = json.loads(index.read_text())
    document["runs"][RUN]["files"]["source.json"] = hashlib.sha256(
        path.read_bytes()
    ).hexdigest()
    index.write_text(json.dumps(document))
    _commit(root)
    result = catalog.build_catalog(root, snapshot, receipts)
    run = result["runs"][0]
    assert run["produced_by"] is None
    assert run["provenance"]["stamp_status"] == "absent"
    record = run["provenance"]["local_records"][0]
    assert record["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert record["recorded_fields"] == [
        {"pointer": "/packaging_commit", "value": "c" * 40},
        {"pointer": "/producer_commit", "value": "b" * 40},
    ]


@pytest.fixture
def provenance_archive(source, monkeypatch):
    import huggingface_hub

    root, snapshot, receipts = source
    file = snapshot["objects"][0]
    name = "records/source.json"
    body = json.dumps(
        {
            "public_commit": "b" * 40,
            "files": [
                {
                    "path": "results.tsv",
                    "bytes": file["bytes"],
                    "sha256": file["sha256"],
                }
            ],
        }
    ).encode()
    member = {"bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()}
    manifest = {
        "format": "replica.run-bundle.v1",
        "scope": "run_provenance",
        "studies": ["example2026"],
        "files": {name: member},
    }
    path = root / "provenance.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        for key, data in [
            ("MANIFEST.json", json.dumps(manifest).encode()),
            (name, body),
        ]:
            entry = tarfile.TarInfo(key)
            entry.size = len(data)
            archive.addfile(entry, io.BytesIO(data))
    identity = _identity("placeholder", path.read_bytes())
    identity["path"] = f"provenance/{identity['sha256']}.tar.gz"
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *a, **kw: str(path))
    receipt = catalog.verify_object(REPO, REVISION, identity)
    snapshot["objects"].append(identity)
    receipts.append(receipt)
    retained = {
        "retained_records": {
            identity["path"]: {
                "description": "Original manifests for matched results.",
                "metadata_paths": [identity["path"]],
            }
        },
        "run_records": {
            f"example2026/{RUN}": {
                "records": [
                    {
                        "archive": identity["path"],
                        "path": name,
                        **member,
                        "recorded_fields": receipt["provenance_records"][name][
                            "recorded_fields"
                        ],
                        "relation": "file_identity",
                        "matches": [
                            {
                                "file": PREFIX + RUN + "/cells.tsv",
                                "recorded_path": "results.tsv",
                            }
                        ],
                    }
                ]
            }
        },
    }
    return root, snapshot, receipts, retained


def test_manifest_identity_links_remain_separate_from_producer_stamps(
    provenance_archive,
):
    result = catalog.build_catalog(*provenance_archive)
    run = result["runs"][0]
    assert run["produced_by"] is None
    assert run["provenance"]["archive_records"][0]["recorded_fields"] == [
        {"pointer": "/public_commit", "value": "b" * 40}
    ]
    archived = next(f for f in result["files"] if f["path"].startswith("provenance/"))
    assert archived["study"] is None
    assert "Run provenance" in catalog.render_readme(result)


def test_archive_provenance_survives_removal_of_checkout_payload(provenance_archive):
    root, snapshot, receipts, retained = provenance_archive
    before = catalog.build_catalog(root, snapshot, receipts, retained)["runs"][0]
    (root / PREFIX / RUN / "cells.tsv").unlink()
    _commit(root)
    after = catalog.build_catalog(root, snapshot, receipts, retained)["runs"][0]
    assert after["files"] == []
    assert after["hub_files"] == before["hub_files"]
    assert after["provenance"] == before["provenance"]
    assert after["produced_by"] == before["produced_by"]
    record = retained["run_records"][f"example2026/{RUN}"]["records"][0]
    record["matches"][0]["recorded_path"] = "unrelated.tsv"
    with pytest.raises(ValueError, match="indexed result bytes"):
        catalog.build_catalog(root, snapshot, receipts, retained)


@pytest.mark.parametrize(
    "change", ["identity", "matched_file", "source_file", "member_hash", "study"]
)
def test_unbound_provenance_is_rejected(provenance_archive, change):
    root, snapshot, receipts, retained = provenance_archive
    receipts, retained = deepcopy(receipts), deepcopy(retained)
    record = retained["run_records"][f"example2026/{RUN}"]["records"][0]
    if change == "identity":
        record["recorded_fields"][0]["value"] = "c" * 40
    elif change == "matched_file":
        record["matches"][0]["file"] = "unrelated.tsv"
    elif change == "source_file":
        record["matches"][0]["recorded_path"] = "unrelated.tsv"
    elif change == "member_hash":
        record["sha256"] = "0" * 64
    else:
        receipts[-1]["manifest"]["studies"] = ["another2026"]
    with pytest.raises(
        ValueError,
        match="identities differ|indexed result bytes|archived member|different-study",
    ):
        catalog.build_catalog(root, snapshot, receipts, retained)


def test_provenance_published_only_on_hub_is_linked(source, monkeypatch):
    import huggingface_hub

    root, snapshot, receipts = source
    body = json.dumps({"producer_commit": "c" * 40}).encode()
    remote = _identity(f"studies/example2026/results/{RUN}/provenance.json", body)
    downloaded = root / "download.json"
    downloaded.write_bytes(body)
    monkeypatch.setattr(
        huggingface_hub, "hf_hub_download", lambda *a, **kw: str(downloaded)
    )
    receipts.append(catalog.verify_object(REPO, REVISION, remote))
    snapshot["objects"].append(remote)
    index = root / PREFIX / "index.json"
    data = json.loads(index.read_text())
    data["runs"][RUN]["hub"]["provenance.json"] = remote["sha256"]
    data["runs"][RUN]["files"]["provenance.json"] = remote["sha256"]
    index.write_text(json.dumps(data))
    downloaded.unlink()
    _commit(root)
    result = catalog.build_catalog(root, snapshot, receipts)
    run = result["runs"][0]
    assert run["produced_by"] is None
    assert run["provenance"]["local_records"] == []
    assert run["provenance"]["hub_records"] == [
        {
            "path": remote["path"],
            "bytes": remote["bytes"],
            "sha256": remote["sha256"],
            "recorded_fields": [{"pointer": "/producer_commit", "value": "c" * 40}],
        }
    ]


@pytest.mark.parametrize(
    "origin, expected",
    [
        ("https://github.com/replicaverse/replica.git", "replicaverse/replica"),
        ("git@github.com:replicaverse/replica-dev.git", "replicaverse/replica-dev"),
        ("ssh://git@github.com/replicaverse/replica", "replicaverse/replica"),
    ],
)
def test_source_repository_comes_from_the_origin_or_the_caller(
    source, origin, expected
):
    root, snapshot, receipts = source
    _git(root, "remote", "set-url", "origin", origin)
    result = catalog.build_catalog(root, snapshot, receipts)
    assert result["source_repository"] == expected
    readme = catalog.render_readme(result)
    assert f"https://github.com/{expected}/blob/{result['source_commit']}" in readme
    explicit = catalog.build_catalog(root, snapshot, receipts, repository="other/name")
    assert explicit["source_repository"] == "other/name"
    assert "https://github.com/other/name/blob/" in catalog.render_readme(explicit)


def test_source_repository_is_never_guessed(source):
    root, snapshot, receipts = source
    _git(root, "remote", "set-url", "origin", "https://gitlab.example.org/x/y.git")
    with pytest.raises(ValueError, match="not a GitHub repository URL"):
        catalog.build_catalog(root, snapshot, receipts)
    _git(root, "remote", "remove", "origin")
    with pytest.raises(ValueError, match="no origin remote"):
        catalog.build_catalog(root, snapshot, receipts)
    for bad in ("just-a-name", "", "../repo", "owner/..", "./.", "owner/.", "o wner/x"):
        with pytest.raises(ValueError, match="owner/name"):
            catalog.build_catalog(root, snapshot, receipts, repository=bad)
    profile = catalog.build_catalog(
        root, snapshot, receipts, repository="owner/.github"
    )
    assert profile["source_repository"] == "owner/.github"
    _git(
        root,
        "remote",
        "add",
        "origin",
        "https://x-access-token:secret@gitlab.example.org/x/y.git",
    )
    with pytest.raises(ValueError) as failure:
        catalog.build_catalog(root, snapshot, receipts)
    assert "secret" not in str(failure.value) and "gitlab" not in str(failure.value)


def test_retained_sources_keep_their_own_repository(provenance_archive):
    root, snapshot, receipts, _ = provenance_archive
    extra = next(r for r in receipts if r["path"].startswith("provenance/"))
    retained = {
        "retained_records": {
            extra["path"]: {
                "description": "Fitted result with source metadata.",
                "sources": [
                    {
                        "commit": "a" * 40,
                        "path": "studies/example2026/freeze.py",
                        "repository": "other/archive",
                    }
                ],
            }
        }
    }
    result = catalog.build_catalog(root, snapshot, receipts, retained)
    bundle = {
        "path": "bundles/example.tar.gz",
        "study": "example2026",
        "sha256": "f" * 64,
        "bytes": 1,
        "references": [],
        "record": {
            "sources": [
                {"commit": "a" * 40, "path": "x.py", "repository": "other/archive"},
                {"commit": "b" * 40, "path": "y.py"},
            ]
        },
    }
    readme = catalog.render_readme({**result, "files": [*result["files"], bundle]})
    assert "https://github.com/other/archive/blob/" + "a" * 40 + "/x.py" in readme
    assert (
        f"https://github.com/{result['source_repository']}/blob/" + "b" * 40 + "/y.py"
        in readme
    )
    bad = deepcopy(retained)
    bad["retained_records"][extra["path"]]["sources"][0]["repository"] = "nope"
    with pytest.raises(ValueError, match="owner/name"):
        catalog.build_catalog(root, snapshot, receipts, bad)


def test_retained_sources_inherit_the_catalog_they_came_from(provenance_archive):
    root, snapshot, receipts, _ = provenance_archive
    extra = next(r for r in receipts if r["path"].startswith("provenance/"))
    retained = {
        "retained_records": {
            extra["path"]: {
                "description": "Fitted result with source metadata.",
                "sources": [
                    {"commit": "a" * 40, "path": "studies/example2026/freeze.py"}
                ],
            }
        }
    }
    first = catalog.build_catalog(
        root, snapshot, receipts, retained, repository="old/repository"
    )
    assert (
        first["retained_records"][extra["path"]]["sources"][0].get("repository") is None
    )
    before = deepcopy(first)
    second = catalog.build_catalog(
        root, snapshot, receipts, first, repository="new/repository"
    )
    assert first == before
    kept = second["retained_records"][extra["path"]]["sources"][0]
    assert kept["repository"] == "old/repository"
    bundle = {
        "path": "bundles/example.tar.gz",
        "study": "example2026",
        "sha256": "f" * 64,
        "bytes": 1,
        "references": [],
        "record": second["retained_records"][extra["path"]],
    }
    readme = catalog.render_readme({**second, "files": [*second["files"], bundle]})
    assert "https://github.com/old/repository/blob/" + "a" * 40 in readme
    assert "https://github.com/new/repository/blob/" + "a" * 40 not in readme
    for inherited in ("", "../x"):
        with pytest.raises(ValueError, match="owner/name"):
            catalog.build_catalog(
                root, snapshot, receipts, {**first, "source_repository": inherited}
            )
