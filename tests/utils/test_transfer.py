"""Downloads and archive extraction write only complete, contained files.

A download replaces its target once the transfer completes; extraction rejects
members that escape the destination.
"""

import io
import tarfile
import zipfile
from http.client import HTTPResponse

import pytest

from replica.utils import transfer
from replica.utils.transfer import archive_output_path, extract_archive


class _Response:
    """A streamed HTTP body that fails after its first chunk when *broken*."""

    headers: dict[str, str] = {}

    def __init__(self, broken: bool):
        self._chunks = [b"first", b"second"]
        self._broken = broken

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self, _size: int = -1) -> bytes:
        if not self._chunks:
            return b""
        if self._broken and len(self._chunks) == 1:
            raise ConnectionResetError("connection reset mid-transfer")
        return self._chunks.pop(0)


@pytest.mark.parametrize("previous", [None, b"existing file"])
def test_an_interrupted_download_leaves_no_partial_file(
    monkeypatch, tmp_path, previous
):
    """A fetch that fails mid-stream leaves the target as it was."""
    target = tmp_path / "data.jsonl"
    if previous is not None:
        target.write_bytes(previous)
    monkeypatch.setattr(transfer, "urlopen", lambda *a, **k: _Response(broken=True))

    with pytest.raises(ConnectionResetError):
        transfer.download_url("https://example.test/data.jsonl", target)

    assert (target.read_bytes() if target.exists() else None) == previous
    assert sorted(path.name for path in tmp_path.iterdir()) == (
        [] if previous is None else ["data.jsonl"]
    )


@pytest.mark.parametrize("headers", [{}, {"Content-Length": "11"}])
def test_a_completed_download_replaces_the_target(monkeypatch, tmp_path, headers):
    target = tmp_path / "data.jsonl"
    target.write_bytes(b"existing file")
    response = _Response(broken=False)
    response.headers = headers
    monkeypatch.setattr(transfer, "urlopen", lambda *a, **k: response)

    transfer.download_url("https://example.test/data.jsonl", target)

    assert target.read_bytes() == b"firstsecond"
    assert [path.name for path in tmp_path.iterdir()] == ["data.jsonl"]


@pytest.mark.parametrize("previous", [None, b"existing file"])
def test_truncated_http_body_preserves_target(monkeypatch, tmp_path, previous):
    class Socket:
        def makefile(self, _mode):
            return io.BytesIO(b"HTTP/1.1 200 OK\r\nContent-Length: 20\r\n\r\nshort")

    response = HTTPResponse(Socket())
    response.begin()
    target = tmp_path / "data.jsonl"
    if previous is not None:
        target.write_bytes(previous)
    monkeypatch.setattr(transfer, "urlopen", lambda *a, **k: response)

    with pytest.raises(OSError, match="received 5 bytes.*expected 20"):
        transfer.download_url("https://example.test/data.jsonl", target)

    assert (target.read_bytes() if target.exists() else None) == previous
    assert list(tmp_path.iterdir()) == ([] if previous is None else [target])


def test_overlapping_downloads_use_independent_temporary_files(monkeypatch, tmp_path):
    target = tmp_path / "data.jsonl"

    class OverlappingResponse(_Response):
        def read(self, _size=-1):
            if len(self._chunks) == 1:
                transfer.download_url("https://example.test/second", target)
                assert target.read_bytes() == b"other payload"
            return super().read(_size)

    first = OverlappingResponse(broken=False)
    second = _Response(broken=False)
    second._chunks = [b"other payload"]
    responses = iter([first, second])
    monkeypatch.setattr(transfer, "urlopen", lambda *a, **k: next(responses))

    transfer.download_url("https://example.test/first", target)

    assert target.read_bytes() == b"firstsecond"
    assert list(tmp_path.iterdir()) == [target]


def test_extract_archive_accepts_regular_zip_member(tmp_path):
    archive = tmp_path / "safe.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("nested/data.txt", "safe")

    destination = tmp_path / "destination"
    extract_archive(archive, destination)

    assert (destination / "nested/data.txt").read_text() == "safe"


def test_extract_archive_rejects_zip_traversal(tmp_path):
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("../outside.txt", "unsafe")

    with pytest.raises(ValueError, match="escapes destination"):
        extract_archive(archive, tmp_path / "destination")

    assert not (tmp_path / "outside.txt").exists()


def test_extract_archive_rejects_tar_traversal(tmp_path):
    archive = tmp_path / "unsafe.tar"
    with tarfile.open(archive, "w") as bundle:
        payload = b"unsafe"
        member = tarfile.TarInfo("../outside.txt")
        member.size = len(payload)
        bundle.addfile(member, io.BytesIO(payload))

    with pytest.raises(ValueError, match="escapes destination"):
        extract_archive(archive, tmp_path / "destination")

    assert not (tmp_path / "outside.txt").exists()


def test_extract_archive_rejects_tar_symlink(tmp_path):
    archive = tmp_path / "link.tar"
    with tarfile.open(archive, "w") as bundle:
        member = tarfile.TarInfo("link")
        member.type = tarfile.SYMTYPE
        member.linkname = "../outside.txt"
        bundle.addfile(member)

    with pytest.raises(ValueError, match="unsafe non-file"):
        extract_archive(archive, tmp_path / "destination")


def test_archive_output_path_accepts_nested_member(tmp_path):
    output = archive_output_path(tmp_path, "nested/data.csv")

    assert output == tmp_path / "nested/data.csv"


@pytest.mark.parametrize("member", ["../outside.csv", "nested/../../outside.csv"])
def test_archive_output_path_rejects_parent_traversal(tmp_path, member):
    with pytest.raises(ValueError, match="escapes destination"):
        archive_output_path(tmp_path, member)
