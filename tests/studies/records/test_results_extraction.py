"""Archive restoration rejects altered bytes, unsafe members and partial inventories."""

import hashlib
import io
import json
import tarfile

import pytest

from replica.studies.records import store


def _archive(path, members):
    with tarfile.open(path, "w") as archive:
        for name, kind in members:
            member = tarfile.TarInfo(name)
            member.type = kind
            if kind == tarfile.REGTYPE:
                member.size = 4
                archive.addfile(member, io.BytesIO(b"data"))
            else:
                member.linkname = "elsewhere"
                archive.addfile(member)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_extract_archive_snapshot(tmp_path, monkeypatch):
    """Replacing the caller's archive after snapshotting cannot replace loaded bytes."""
    archive = tmp_path / "source.tar"
    digest = _archive(archive, [("nested/data", tarfile.REGTYPE)])
    copy = store.shutil.copyfile

    def replace_source(source, target):
        copy(source, target)
        source.write_bytes(b"replaced after copy")

    monkeypatch.setattr(store.shutil, "copyfile", replace_source)
    destination = tmp_path / "restored"
    assert store.extract_archive(archive, destination, sha256=digest) == destination
    assert (destination / "nested/data").read_bytes() == b"data"


@pytest.mark.parametrize(
    "members",
    [
        [("../outside", tarfile.REGTYPE)],
        [("/outside", tarfile.REGTYPE)],
        [("a//b", tarfile.REGTYPE)],
        [("a\\b", tarfile.REGTYPE)],
        [("same", tarfile.REGTYPE), ("same", tarfile.REGTYPE)],
        [("same", tarfile.DIRTYPE), ("same/", tarfile.DIRTYPE)],
        [("link", tarfile.SYMTYPE)],
        [("link", tarfile.LNKTYPE)],
        [("pipe", tarfile.FIFOTYPE)],
        [("parent", tarfile.REGTYPE), ("parent/child", tarfile.REGTYPE)],
    ],
)
def test_extract_rejects_unsafe_inventory(tmp_path, members):
    """Validate the whole inventory before creating the output directory."""
    archive = tmp_path / "source.tar"
    digest = _archive(archive, members)
    destination = tmp_path / "restored"
    with pytest.raises(ValueError):
        store.extract_archive(archive, destination, sha256=digest)
    assert not destination.exists()


def test_extract_identity_and_destination(tmp_path):
    """Changed bytes and an existing extraction destination fail closed."""
    archive = tmp_path / "source.tar"
    digest = _archive(archive, [("data", tarfile.REGTYPE)])
    for bad in ("unpinned", "b" * 64):
        with pytest.raises(ValueError):
            store.extract_archive(archive, tmp_path / "new", sha256=bad)
    destination = tmp_path / "existing"
    destination.mkdir()
    with pytest.raises(FileExistsError):
        store.extract_archive(archive, destination, sha256=digest)


@pytest.mark.parametrize("change", [None, "extra", "missing", "bytes", "link", "size"])
def test_directory_complete_inventory(tmp_path, change):
    """The common manifest schema authenticates exactly the extracted file set."""
    data = tmp_path / "data"
    data.write_bytes(b"data")
    manifest = {
        "files": {"data": {"bytes": 4, "sha256": hashlib.sha256(b"data").hexdigest()}}
    }
    (tmp_path / "MANIFEST.json").write_text(json.dumps(manifest))
    if change == "extra":
        (tmp_path / "extra").write_bytes(b"unexpected")
    elif change == "missing":
        data.unlink()
    elif change == "bytes":
        data.write_bytes(b"else")
    elif change == "link":
        (tmp_path / "link").symlink_to(data)
    elif change == "size":
        manifest["files"]["data"]["bytes"] = True
    if change is None:
        assert store.verify_directory(tmp_path, manifest=manifest) == manifest
    else:
        with pytest.raises(ValueError):
            store.verify_directory(tmp_path, manifest=manifest)
