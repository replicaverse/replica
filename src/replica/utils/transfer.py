"""Shared HTTP transfer and safe archive-extraction primitives."""

import os
import shutil
import ssl
import stat
import tarfile
import tempfile
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Protocol, cast
from urllib.request import Request, urlopen

import certifi

CHUNK_SIZE = 64 * 1024


class _DownloadResponse(Protocol):
    headers: Mapping[str, str]

    def read(self, size: int = -1) -> bytes: ...


def ensure_ssl_cert_chain() -> None:
    """Use certifi's CA bundle when no explicit TLS certificate file is set."""
    if os.environ.get("SSL_CERT_FILE"):
        return
    os.environ["SSL_CERT_FILE"] = certifi.where()


def archive_output_path(directory: Path, relative_name: str) -> Path:
    """Resolve an archive member below ``directory`` and reject traversal.

    Args:
        directory: Extraction root; the resolved member has to stay inside it.
        relative_name: Member name as the archive records it, which may carry
            ``..`` segments or an absolute prefix.

    Returns:
        The resolved destination path for that member.

    Raises:
        ValueError: When the member resolves outside ``directory``.
    """
    root = directory.resolve()
    output = (root / relative_name).resolve()
    if output != root and not output.is_relative_to(root):
        raise ValueError(f"archive member escapes destination: {relative_name!r}")
    return output


def _stream_response(
    response: _DownloadResponse,
    output_path: Path,
    *,
    show_progress: bool,
) -> None:
    total = response.headers.get("Content-Length")
    expected = int(total) if total is not None else None
    total_mb = expected / (1024 * 1024) if expected is not None else None
    downloaded = 0
    last_reported = 0.0

    with output_path.open("wb") as output:
        while chunk := response.read(CHUNK_SIZE):
            output.write(chunk)
            downloaded += len(chunk)
            downloaded_mb = downloaded / (1024 * 1024)
            if show_progress and downloaded_mb - last_reported >= 10:
                suffix = f" / {total_mb:.1f} MB" if total_mb else " MB"
                print(f"    {downloaded_mb:.1f}{suffix}")
                last_reported = downloaded_mb

    if expected is not None and downloaded != expected:
        raise OSError(
            f"Incomplete download: received {downloaded} bytes, expected {expected}"
        )
    if show_progress:
        print(f"    Done: {downloaded / (1024 * 1024):.1f} MB")


def download_url(
    url: str,
    output_path: Path,
    *,
    user_agent: str = "replica/1.0",
    ssl_context: ssl.SSLContext | None = None,
    timeout: float = 300,
    show_progress: bool = False,
) -> None:
    """Download ``url`` to ``output_path`` using a bounded streaming request.

    The bytes stream into a temporary directory beside ``output_path``.
    It replaces the target after the transfer completes and its byte count
    matches ``Content-Length``, when supplied.

    Args:
        url: Resource to fetch over HTTP or HTTPS.
        output_path: File to write; its parent directory must exist and any
            existing file is replaced when the download completes.
        user_agent: Value of the ``User-Agent`` header, which some hosts
            require before they serve the file.
        ssl_context: TLS context for the request; ``None`` uses the default
            one, which :func:`ensure_ssl_cert_chain` can point at certifi.
        timeout: Seconds to wait for the connection and for each read.
        show_progress: Print the megabytes written every 10 MB and on
            completion.

    Raises:
        OSError: The response body does not match its declared byte count.
    """
    request = Request(url, headers={"User-Agent": user_agent})
    with tempfile.TemporaryDirectory(
        dir=output_path.parent, prefix=f".{output_path.name}."
    ) as temporary:
        partial = Path(temporary) / "download.part"
        with urlopen(request, context=ssl_context, timeout=timeout) as response:
            _stream_response(
                cast(_DownloadResponse, response),
                partial,
                show_progress=show_progress,
            )
        os.replace(partial, output_path)


def extract_zip_member(
    archive: zipfile.ZipFile,
    member: str | zipfile.ZipInfo,
    output_path: Path,
) -> None:
    """Stream one regular ZIP member to an explicit output path.

    Args:
        archive: Open ZIP file the member is read from.
        member: Member name, or the ``ZipInfo`` already read from the archive.
        output_path: File to write; parent directories are created.

    Raises:
        ValueError: When the member's mode bits mark it a symlink.
    """
    info = archive.getinfo(member) if isinstance(member, str) else member
    mode = info.external_attr >> 16
    if stat.S_ISLNK(mode):
        raise ValueError(f"unsafe symlink in zip archive: {info.filename!r}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with archive.open(info) as source, output_path.open("wb") as output:
        shutil.copyfileobj(source, output, length=CHUNK_SIZE)


def extract_archive(archive: Path, destination: Path) -> None:
    """Extract ZIP or TAR data while rejecting links and escaping paths.

    Args:
        archive: Archive file; a ``.zip`` suffix selects the ZIP reader, every
            other name is opened as a tarball with compression auto-detected.
        destination: Directory to extract into, created when absent.

    Raises:
        ValueError: When a member escapes ``destination``, or when a tar member
            is neither a directory nor a readable regular file.
    """
    destination.mkdir(parents=True, exist_ok=True)
    if archive.name.lower().endswith(".zip"):
        with zipfile.ZipFile(archive) as bundle:
            for info in bundle.infolist():
                output = archive_output_path(destination, info.filename)
                if info.is_dir():
                    output.mkdir(parents=True, exist_ok=True)
                else:
                    extract_zip_member(bundle, info, output)
        return

    with tarfile.open(archive) as bundle:
        for member in bundle.getmembers():
            output = archive_output_path(destination, member.name)
            if member.isdir():
                output.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                raise ValueError(f"unsafe non-file in tar archive: {member.name!r}")
            source = bundle.extractfile(member)
            if source is None:
                raise ValueError(f"could not read tar member: {member.name!r}")
            output.parent.mkdir(parents=True, exist_ok=True)
            with source, output.open("wb") as target:
                shutil.copyfileobj(source, target, length=CHUNK_SIZE)


def download_and_extract(url: str, target: Path) -> None:
    """Download an archive and atomically install its extracted contents.

    Extraction happens in a temporary directory beside ``target`` and the
    result is moved into place in one step, so a failed transfer leaves no
    half-populated directory behind. A single top-level directory in the
    archive is unwrapped, and macOS metadata entries are discarded.

    Args:
        url: Archive to fetch; its filename decides which extractor runs.
        target: Directory path to install the contents at; it must not exist
            yet, since the final rename cannot replace a non-empty directory.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=str(target.parent)) as temporary:
        temporary_path = Path(temporary)
        archive = temporary_path / Path(url).name
        download_url(url, archive)
        extracted = temporary_path / "extracted"
        extract_archive(archive, extracted)

        for junk in (extracted / ".DS_Store", extracted / "__MACOSX"):
            if junk.is_dir():
                shutil.rmtree(junk)
            elif junk.exists():
                junk.unlink()

        entries = list(extracted.iterdir())
        source = entries[0] if len(entries) == 1 and entries[0].is_dir() else extracted
        os.replace(source, target)
