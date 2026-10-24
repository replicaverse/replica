"""File digests preserve published SHA-256 values and the manifest spelling."""

import hashlib

import pytest

from replica.utils.digest import file_sha256, prefixed_sha256


@pytest.mark.parametrize(
    "content,expected",
    [
        (b"", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"),
        (b"abc", "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"),
    ],
)
def test_known_file_digests_and_prefix(tmp_path, content, expected):
    path = tmp_path / "payload"
    path.write_bytes(content)
    assert file_sha256(path) == expected
    assert file_sha256(str(path)) == expected
    assert prefixed_sha256(path) == f"sha256:{expected}"


@pytest.mark.parametrize(
    "size", [(1 << 20) - 1, 1 << 20, (1 << 20) + 1, (2 << 20) + 73]
)
def test_binary_digest_across_streaming_boundaries(tmp_path, size):
    content = (bytes(range(256)) * (size // 256 + 1))[:size]
    path = tmp_path / "payload"
    path.write_bytes(content)
    assert file_sha256(path) == hashlib.sha256(content).hexdigest()
