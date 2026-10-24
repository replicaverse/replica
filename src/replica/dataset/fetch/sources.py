"""Network transfer and archive extraction primitives for dataset fetching."""

import shutil
import ssl
import tempfile
import zipfile
from pathlib import Path

import certifi

from ...external import repo_path
from ...utils.transfer import (
    archive_output_path,
    download_url,
    extract_zip_member,
)

USER_AGENT = "replica dataset fetcher"


def _download(
    url: str,
    output_path: Path,
    *,
    ssl_context: ssl.SSLContext | None = None,
) -> None:
    download_url(
        url,
        output_path,
        user_agent=USER_AGENT,
        ssl_context=ssl_context,
        show_progress=True,
    )


def download_osf(file_id: str, output_path: Path) -> None:
    """Download a file from OSF by file ID.

    Args:
        file_id: OSF identifier of the file, the last segment of its download URL.
        output_path: File to write the bytes to.
    """
    url = f"https://osf.io/download/{file_id}/"
    print(f"  Downloading {output_path.name} from OSF ({file_id}) ...")
    _download(url, output_path)


def download_github_raw(repo: str, branch: str, path: str, output_path: Path) -> None:
    """Download a raw file from GitHub.

    Args:
        repo: Repository slug, ``owner/name``.
        branch: Git ref to read at, a branch name or a commit sha.
        path: Location of the file within the repository.
        output_path: File to write the bytes to.
    """
    url = f"https://raw.githubusercontent.com/{repo}/{branch}/{path}"
    print(f"  Downloading {output_path.name} from GitHub ({repo}) ...")
    _download(url, output_path)


def download_osf_zip(file_id: str, filename: str, directory: Path) -> None:
    """Download a ZIP from OSF and extract all contents to *directory*.

    Members are flattened: subdirectory prefixes are dropped, and directory
    entries and ``__MACOSX`` metadata are skipped. When the download turns out
    to be a plain file rather than an archive, it is saved under *filename*
    without its ``.zip`` suffix.

    Args:
        file_id: OSF identifier of the archive, the last segment of its
            download URL.
        filename: Name the archive carries on OSF, used for progress output and
            as the fallback name for a non-archive download.
        directory: Directory the extracted members are written into.
    """
    url = f"https://osf.io/download/{file_id}/"
    print(f"  Downloading {filename} from OSF ({file_id}) ...")
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        _download(url, tmp_path)
        # Check if it's actually a ZIP
        if zipfile.is_zipfile(tmp_path):
            with zipfile.ZipFile(tmp_path) as zf:
                for member in zf.namelist():
                    # Skip directories and hidden files
                    if member.endswith("/") or member.startswith("__MACOSX"):
                        continue
                    # Extract to flat directory (strip any subdirectory prefix)
                    out_name = Path(member).name
                    out = directory / out_name
                    extract_zip_member(zf, member, out)
                    print(f"  Extracted {out_name}")
        else:
            out = directory / filename.removesuffix(".zip")
            shutil.move(str(tmp_path), str(out))
            print(f"  Saved {out.name}")
            return  # skip unlink: shutil.move leaves no temp file behind
    finally:
        tmp_path.unlink(missing_ok=True)


def download_zip(url: str, directory: Path, file_mapping: dict[str, str]) -> None:
    """Download a ZIP from *url* and extract files per *file_mapping*.

    A member is matched by its full name or by its trailing path segment, so a
    mapping key need not spell out the archive's internal directories. A key
    that matches nothing is reported and skipped.

    Args:
        url: Location of the archive to download.
        directory: Directory the selected members are written into.
        file_mapping: ZIP member name -> local filename under *directory*.
    """
    print(f"  Downloading ZIP from {url[:80]}...")
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        _download(url, tmp_path)
        with zipfile.ZipFile(tmp_path) as zf:
            members = zf.namelist()
            for zip_name, local_name in file_mapping.items():
                # Find the member (may be nested in a subdirectory)
                match = None
                for m in members:
                    if m == zip_name or m.endswith("/" + zip_name):
                        match = m
                        break
                if match is None:
                    print(f"  WARNING: {zip_name} not found in ZIP")
                    continue
                out = archive_output_path(directory, local_name)
                extract_zip_member(zf, match, out)
                print(f"  Extracted {local_name}")
    finally:
        tmp_path.unlink(missing_ok=True)


def download_direct_url(url: str, output_path: Path) -> None:
    """Download a file from a direct URL.

    Verifies TLS against the certifi CA bundle (also wired globally by
    ``ensure_ssl_cert_chain``); this trusts the direct-URL hosts' chain without
    disabling verification.

    Args:
        url: Location of the file to download.
        output_path: File to write the bytes to.
    """
    print(f"  Downloading {output_path.name} ...")
    ctx = ssl.create_default_context(cafile=certifi.where())
    _download(url, output_path, ssl_context=ctx)


def link_study_release(repo: str, member: str, output_path: Path) -> None:
    """Point *output_path* at one file inside a study's released data.

    Several corpora are small extracts from one large release that a study
    already declares as a data dependency, so the release is fetched once and
    shared by those corpora.

    Args:
        repo: Key in :data:`replica.external.EXTERNAL_REPOS` naming the
            release to take the file from; a missing checkout is fetched.
        member: Location of the file inside that release.
        output_path: Symlink to create, replacing anything already there. On a
            filesystem without symlinks the file is copied instead.
    """
    source = repo_path(repo) / member
    if not source.exists():
        raise FileNotFoundError(
            f"{member} is not in the {repo} release at {source.parent}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.is_symlink() or output_path.exists():
        output_path.unlink()
    try:
        output_path.symlink_to(source)
    except OSError:  # a filesystem without symlinks
        shutil.copy2(source, output_path)
