"""Save a finished run into the results store and publish it to the hub.

The store is the directory ``REPLICA_RESULTS_STORE`` names. Each run is
``<store>/<run_id>/`` holding ``MANIFEST.json``, ``results/`` (tables),
``frames/`` (intermediates) and ``logs/``. The manifest lists every file with
its digest and row count and carries the ``produced_by`` stamp taken when the
run was saved, in the tree that produced it. It records no host, working
directory or source path, so it can be published as it is.

``publish`` uploads tables and their index under ``studies/<study>/results/``
and registers the run in the study's ``index.json`` with its role, its store id, the digest of every
hub file and the stamp. ``fetch`` brings a run's hub files into the local
cache through :func:`replica.studies.records.files.results_file`.

``publish_bundle`` archives a complete sealed run and preserves its producer
metadata. It returns a verified immutable reference for ``fetch_bundle``;
study evidence indexes are unchanged.
"""

from __future__ import annotations

import json
import os
import shutil
import tarfile
import tempfile
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

from . import files
from .layout import parse_run, render_readme

MANIFEST = "MANIFEST.json"
BUNDLE_FORMAT = "replica.run-bundle.v1"
#: Name prefixes of working copies, which ``save`` skips.
_SKIP_PREFIXES = (".",)


def store_root() -> Path:
    """The store directory, from ``REPLICA_RESULTS_STORE``.

    Raises:
        RuntimeError: The variable is unset.
    """
    root = os.environ.get(files.STORE_ENV)
    if not root:
        raise RuntimeError(f"set {files.STORE_ENV} to the results store directory")
    return Path(root)


def _classify(path: Path) -> str:
    if path.suffix == ".log":
        return "logs"
    if path.suffix in {".tsv", ".csv", ".json", ".md", ".txt"}:
        return "results"
    return "frames"


def _row_count(path: Path) -> int | None:
    if path.suffix not in {".tsv", ".csv"}:
        return None
    with path.open(encoding="utf-8", errors="replace") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def _destination(run_dir: Path, source_root: Path | None, path: Path) -> Path:
    """Where a file lands: its class directory, keeping any subdirectory below the source."""
    inside = path.relative_to(source_root) if source_root else Path(path.name)
    return run_dir / _classify(path) / inside


def read_manifest(run_id: str) -> dict[str, Any]:
    """The manifest of one store run.

    Args:
        run_id: Store run identifier.
    """
    return json.loads((store_root() / run_id / MANIFEST).read_text())


def save(
    run_id: str,
    sources: list[Path],
    *,
    study: str,
    command: str | None = None,
    inputs: dict[str, str] | None = None,
    driver_commit: str | None = None,
    depends_on: list[str] | None = None,
) -> Path:
    """Copy a run's outputs into the store and write its manifest.

    A file already in the store with the same digest is left alone, so a save
    can run against a live output directory and again at completion. Files an
    earlier save recorded stay listed with their recorded digests when a later
    save names fewer sources, and the save stops if one is missing or changed.

    Args:
        run_id: Store run identifier.
        sources: Output directories or single files.
        study: Study package whose code produced the run.
        command: What produced the run, as free text.
        inputs: Label -> sha256 of each input the run read.
        driver_commit: Commit of a driver repository outside this package.
        depends_on: Package-relative files the run's content depends on;
            ``None`` covers the study's footprint files.

    Returns:
        The run's directory in the store.
    """
    missing = [
        name
        for name in (depends_on or [])
        if not (files.SRC_DIR / "replica" / name).is_file()
    ]
    if missing:
        raise ValueError(
            "Explicit producer dependencies must exist relative to the replica package: "
            + ", ".join(missing)
        )
    run_dir = store_root() / run_id
    planned: list[tuple[Path, Path]] = []
    for source in sources:
        if not source.exists():
            raise FileNotFoundError(source)
        candidates = (
            [source]
            if source.is_file()
            else sorted(p for p in source.rglob("*") if p.is_file())
        )
        planned += [
            (path, _destination(run_dir, None if source.is_file() else source, path))
            for path in candidates
            if not path.name.startswith(_SKIP_PREFIXES)
        ]
    named = {target.relative_to(run_dir).as_posix() for _, target in planned}
    entries: list[dict[str, Any]] = []
    manifest_path = run_dir / MANIFEST
    if manifest_path.is_file():
        for entry in json.loads(manifest_path.read_text())["files"]:
            if entry["path"] in named:
                continue
            kept = run_dir / entry["path"]
            if not kept.is_file() or files._sha256(kept) != entry["sha256"]:
                raise ValueError(
                    f"{run_id}/{entry['path']} is missing or differs from the "
                    f"digest in {MANIFEST}; run `replica results verify {run_id}`"
                )
            entries.append(entry)
    written: dict[str, str] = {}
    for path, target in planned:
        digest = files._sha256(path)
        relative = target.relative_to(run_dir).as_posix()
        if relative in written and written[relative] != digest:
            raise ValueError(f"two different files map to {relative}")
        written[relative] = digest
        target.parent.mkdir(parents=True, exist_ok=True)
        if not (target.is_file() and files._sha256(target) == digest):
            shutil.copy2(path, target)
        entries.append(
            {
                "path": relative,
                "bytes": target.stat().st_size,
                "rows": _row_count(target),
                "sha256": digest,
            }
        )
    manifest = {
        "run_id": run_id,
        "study": study,
        "saved_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "command": command,
        "produced_by": files.produced_by_now(
            study, inputs, driver_commit, depends_on=depends_on
        ),
        "files": sorted(entries, key=lambda e: e["path"]),
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return run_dir


def verify(run_id: str) -> list[str]:
    """Every file of a run that is missing or differs from its manifest.

    Args:
        run_id: Store run identifier.
    """
    run_dir = store_root() / run_id
    problems = []
    for entry in read_manifest(run_id)["files"]:
        path = run_dir / entry["path"]
        if not path.is_file():
            problems.append(f"{entry['path']}: missing")
        elif files._sha256(path) != entry["sha256"]:
            problems.append(f"{entry['path']}: digest differs")
    return problems


def list_runs() -> list[dict[str, Any]]:
    """One summary per store run: id, study, save time, file count, stamp."""
    summaries = []
    for manifest_path in sorted(store_root().glob(f"*/{MANIFEST}")):
        manifest = json.loads(manifest_path.read_text())
        summaries.append(
            {
                "run_id": manifest_path.parent.name,
                "study": manifest.get("study"),
                "saved_utc": manifest.get("saved_utc"),
                "files": len(manifest.get("files", [])),
                "stamped": bool(manifest.get("produced_by")),
            }
        )
    return summaries


def _upload(folder: Path, path_in_repo: str, message: str) -> str:
    from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

    api = HfApi()
    document = json.loads((folder / "index.json").read_text())
    repo_id = files.hub_repo(document)
    head = api.repo_info(repo_id, repo_type="dataset", revision=files.HF_REVISION)
    names = {
        p.rfilename
        for p in api.repo_info(repo_id, repo_type="dataset", revision=head.sha).siblings
    }
    remote_index = f"{path_in_repo}/index.json"
    if remote_index in names:
        previous = json.loads(
            Path(
                hf_hub_download(
                    repo_id, remote_index, repo_type="dataset", revision=head.sha
                )
            ).read_text()
        )
        for run, entry in previous["runs"].items():
            current = document["runs"].get(run)
            if (
                current is None
                or current["files"] != entry["files"]
                or current.get("produced_by") != entry.get("produced_by")
            ):
                raise ValueError(
                    f"Publication would replace an immutable record: {run}"
                )
        for digest, entry in previous.get("archives", {}).items():
            current = document.get("archives", {}).get(digest, {})
            if current.get("reference") != entry["reference"] or any(
                current.get("records", {}).get(name) != value
                for name, value in entry.get("records", {}).items()
            ):
                raise ValueError(
                    f"Publication would replace an archive binding: {digest}"
                )
    commit = api.create_commit(
        repo_id=repo_id,
        repo_type="dataset",
        revision=files.HF_REVISION,
        parent_commit=head.sha,
        operations=[
            CommitOperationAdd(
                path_in_repo=f"{path_in_repo}/{path.relative_to(folder).as_posix()}",
                path_or_fileobj=path,
            )
            for path in sorted(folder.rglob("*"))
            if path.is_file()
        ],
        commit_message=message,
    )
    return commit.oid


def _publish_tables(
    study: str, run: str, document: dict[str, Any], tables: Path
) -> str:
    document["hub_repo"] = files.hub_repo(document)
    with tempfile.TemporaryDirectory() as temporary:
        staged = Path(temporary)
        shutil.copytree(tables, staged / run)
        (staged / "index.json").write_text(json.dumps(document, indent=2) + "\n")
        readme = files.STUDIES_DIR / study / "results" / "README.md"
        if readme.is_file():
            shutil.copy2(readme, staged / "README.md")
        text = render_readme(
            study, staged, f"https://huggingface.co/datasets/{files.hub_repo(document)}"
        )
        (staged / "README.md").write_text(text)
        # The per-study index and payload are committed together on the Hub.
        revision = _upload(staged, f"studies/{study}/results", f"Publish {study}/{run}")
        if revision:
            document["hub_revision"] = revision
            with tempfile.TemporaryDirectory() as metadata_directory:
                metadata = Path(metadata_directory)
                (metadata / "index.json").write_text(
                    json.dumps(document, indent=2) + "\n"
                )
                (metadata / "README.md").write_text(text)
                text = render_readme(
                    study,
                    metadata,
                    f"https://huggingface.co/datasets/{files.hub_repo(document)}",
                )
                (metadata / "README.md").write_text(text)
                _upload(
                    metadata, f"studies/{study}/results", f"Pin {study} result payloads"
                )
    return text


def publish(
    run_id: str,
    *,
    study: str,
    run: str,
    role: str,
    note: str | None = None,
    description: str | None = None,
) -> dict[str, Any]:
    """Upload a run's tables to the hub and register the run in the study index.

    Args:
        run_id: Store run identifier.
        study: Study package name.
        run: Run path relative to the study's ``results/``.
        role: The run's role, one of :data:`replica.studies.records.layout.ROLES`.
        note: Text for the index entry.
        description: Calculation and input configuration represented by this record.

    Returns:
        The index entry written.
    """
    parse_run(run)
    if not description or not description.strip():
        raise ValueError("A result record requires a description")
    package_problems = files.validate_package(study)
    if package_problems:
        raise ValueError("Invalid study package: " + "; ".join(package_problems))
    if role not in files.ROLES or run.startswith("references/") != (
        role == "published"
    ):
        raise ValueError("Role and result placement disagree")
    manifest = read_manifest(run_id)
    if manifest.get("study") != study:
        raise ValueError(
            f"{run_id} was saved for {manifest.get('study')!r}, not {study!r}"
        )
    if not manifest.get("produced_by"):
        raise ValueError(f"{run_id} carries no produced_by stamp")
    stamp_problems = files._stamp_problems(manifest["produced_by"], recording=True)
    if stamp_problems:
        raise ValueError("Invalid producer stamp: " + "; ".join(stamp_problems))
    tables = store_root() / run_id / "results"
    hub = {
        entry["path"].removeprefix("results/"): entry["sha256"]
        for entry in manifest["files"]
        if entry["path"].startswith("results/")
    }
    if not hub:
        raise ValueError(f"{run_id} holds no tables under results/")
    for name in hub:
        _archive_name(name)
    actual = {
        p.relative_to(tables).as_posix() for p in tables.rglob("*") if p.is_file()
    }
    if actual != set(hub):
        raise ValueError("Store tables must match the complete file manifest")
    problems = verify(run_id)
    if problems:
        raise ValueError("Store bytes differ: " + "; ".join(problems))
    document = files.read_index(study)
    if (
        document.get("hub_repo")
        and document["hub_repo"] != files.hub_repo(document)
        and any(entry.get("hub") for entry in document["runs"].values())
    ):
        raise ValueError(
            "Cannot redirect an index with published payloads to another repository"
        )
    previous = document["runs"].get(run)
    if previous is not None and (
        previous["files"] != hub
        or previous.get("produced_by") != manifest["produced_by"]
    ):
        raise ValueError(
            "Changed outputs or producer require a new result record identifier"
        )
    entry = {
        "role": role,
        "description": description,
        "produced_by": manifest["produced_by"],
        "files": hub,
        "hub": hub,
        "store": run_id,
    }
    if note is not None:
        entry["note"] = note
    document["runs"][run] = entry
    document["schema"] = files.SCHEMA
    text = _publish_tables(study, run, document, tables)
    files.write_index(study, document)
    (files.STUDIES_DIR / study / "results" / "README.md").write_text(text)
    return entry


def fetch(
    study: str, run: str | None = None, *, verification: bool = False
) -> list[Path]:
    """Bring every hub file of one run, or of all runs, into the local cache.

    Args:
        study: Study package name.
        run: Run path relative to the study's ``results/``; all runs when ``None``.
        verification: Also fetch archives declared necessary for result verification.
    """
    document = files.read_index(study)
    entries = document["runs"]
    if run is not None and run not in entries:
        raise FileNotFoundError(f"{study}: no run {run!r} in {files.INDEX}")
    paths = []
    for name, entry in entries.items():
        if run is not None and name != run:
            continue
        for file_name in entry.get("hub") or {}:
            paths.append(files.results_file(study, name, file_name))
    if verification:
        for archive in document.get("archives", {}).values():
            if archive.get("verification_required") is True:
                paths.append(fetch_bundle(archive["reference"]))
    return paths


def verify_bundle(archive: Path) -> dict[str, Any]:
    """Verify every file in a sealed archive without extracting or restamping it.

    Args:
        archive: Tar archive with a root MANIFEST.json declaring BUNDLE_FORMAT.
            Its files mapping records the SHA-256 and byte length of every
            other member. Existing run manifests belong inside the payload.

    Returns:
        The unchanged manifest, after all member identities pass.

    Raises:
        ValueError: A member is unsafe, duplicated, unlisted or inconsistent
            with the manifest. Links and special files are not accepted.
    """
    import hashlib

    with tarfile.open(archive, mode="r:*") as bundle:
        members = bundle.getmembers()
        names = set()
        for member in members:
            path = PurePosixPath(member.name)
            if (
                path.is_absolute()
                or ".." in path.parts
                or path.as_posix() != member.name
                or "\\" in member.name
                or not member.isfile()
                or member.name in names
            ):
                raise ValueError(f"Invalid or duplicate bundle member: {member.name}")
            names.add(member.name)
        if MANIFEST not in names:
            raise ValueError(f"Bundle lacks {MANIFEST}")
        stream = bundle.extractfile(MANIFEST)
        assert stream is not None
        with stream:
            manifest = json.load(stream)
        if not isinstance(manifest, dict):
            raise ValueError("Bundle manifest must be a JSON object")
        if manifest.get("format") != BUNDLE_FORMAT:
            raise ValueError(f"Bundle manifest must declare {BUNDLE_FORMAT}")
        entries = _manifest_entries(manifest, names - {MANIFEST})
        for name, entry in entries.items():
            member = bundle.getmember(name)
            stream = bundle.extractfile(member)
            assert stream is not None
            checksum = hashlib.sha256()
            with stream:
                while chunk := stream.read(1024 * 1024):
                    checksum.update(chunk)
            if member.size != entry["bytes"] or checksum.hexdigest() != entry["sha256"]:
                raise ValueError(f"Bundle member differs from its manifest: {name}")
    return manifest


def fetch_bundle(reference: dict[str, Any], *, force_download: bool = False) -> Path:
    """Download and verify a full run archive at an immutable Hub revision.

    Args:
        reference: Repository, revision, path, sha256 and bytes returned by
            publish_bundle. Branch names and mutable revisions are rejected.
        force_download: Download a new copy even when the cache has the file.

    Returns:
        Cached archive whose outer digest and complete member manifest pass.

    Raises:
        ValueError: The reference is incomplete or the archive does not match.
    """
    import re

    revision = reference.get("revision", "")
    digest = reference.get("sha256", "")
    study = reference.get("study", "")
    repo_id = reference.get("repo_id")
    if (
        not isinstance(revision, str)
        or not re.fullmatch(r"[0-9a-f]{40}", revision)
        or not isinstance(digest, str)
        or not files._DIGEST.fullmatch(digest)
        or not isinstance(reference.get("bytes"), int)
        or isinstance(reference.get("bytes"), bool)
        or reference["bytes"] < 0
        or not isinstance(study, str)
        or not re.fullmatch(r"[a-z][a-z0-9_]*", study)
        or not isinstance(repo_id, str)
        or len(repo_id.split("/")) != 2
        or not all(repo_id.split("/"))
        or reference.get("path") != f"bundles/{study}/{digest}.tar.gz"
    ):
        raise ValueError("Bundle reference needs an immutable revision and digest")
    path = Path(
        files._hub_download(
            reference["path"],
            repo_id=reference["repo_id"],
            revision=revision,
            force_download=force_download,
        )
    )
    if path.stat().st_size != reference["bytes"] or files._sha256(path) != digest:
        raise ValueError("Downloaded bundle differs from its immutable reference")
    if verify_bundle(path).get("study") != study:
        raise ValueError("Downloaded bundle names a different study")
    return path


def publish_bundle(
    archive: Path, *, study: str, repo_id: str | None = None
) -> dict[str, Any]:
    """Publish a complete run as a verified archive.

    The archive digest determines its Hub path. Existing identical content is
    reused; a changed repository head rejects the upload for a safe retry.
    Included producer records are kept as recorded. The checks cover integrity
    only; scientific validity and redistribution rights are outside their scope.

    Args:
        archive: Completed tar archive accepted by verify_bundle.
        study: Study identifier matching the archive manifest.
        repo_id: Existing results repository; defaults to HF_REPO.

    Returns:
        Immutable archive reference, issued only after a fresh download verifies.
        It does not register website evidence.

    Raises:
        ValueError: Study identity or archive integrity fails.
    """
    import re

    if not re.fullmatch(r"[a-z][a-z0-9_]*", study):
        raise ValueError("Invalid study identifier")
    with tempfile.TemporaryDirectory(prefix="replica-run-bundle-") as directory:
        snapshot = Path(directory) / "bundle.tar.gz"
        shutil.copyfile(archive, snapshot)
        manifest = verify_bundle(snapshot)
        if manifest.get("study") != study:
            raise ValueError("Archive manifest names a different study")
        return _publish_bundle_snapshot(snapshot, study, repo_id or files.HF_REPO)


def _publish_bundle_snapshot(archive: Path, study: str, repo_id: str) -> dict[str, Any]:
    """Publish an isolated archive and verify the committed remote copy.

    Args:
        archive: Verified archive in a temporary directory.
        study: Validated study identifier.
        repo_id: Existing dataset repository.

    Returns:
        Reference pinned to the commit containing the verified archive.

    Raises:
        ValueError: The repository did not resolve to an immutable revision.
    """
    import re

    from huggingface_hub import HfApi

    digest = files._sha256(archive)
    size = archive.stat().st_size
    api = HfApi()
    info = api.repo_info(repo_id, repo_type="dataset", revision="main")
    remote = f"bundles/{study}/{digest}.tar.gz"
    revision = info.sha
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Results repository did not resolve to an immutable commit")
    if not api.file_exists(repo_id, remote, repo_type="dataset", revision=revision):
        commit = api.upload_file(
            repo_id=repo_id,
            repo_type="dataset",
            path_or_fileobj=archive,
            path_in_repo=remote,
            revision="main",
            parent_commit=revision,
            commit_message=f"Archive {study} run bundle {digest[:12]}",
        )
        revision = commit.oid
    reference = {
        "study": study,
        "repo_id": repo_id,
        "revision": revision,
        "path": remote,
        "sha256": digest,
        "bytes": size,
    }
    fetch_bundle(reference, force_download=True)
    return reference


def _manifest_entries(manifest: dict[str, Any], names: set[str]) -> dict[str, Any]:
    """Validate the shared complete-file inventory and byte identities.

    Args:
        manifest: Authenticated manifest with a files mapping.
        names: Actual regular-file paths, excluding the root manifest.

    Returns:
        Validated manifest entries.
    """
    entries = manifest.get("files")
    if not isinstance(entries, dict) or not entries:
        raise ValueError("Manifest needs a nonempty files mapping")
    if set(entries) != names:
        raise ValueError("Files differ from the manifest inventory")
    for name, entry in entries.items():
        _archive_name(name)
        if (
            not isinstance(entry, dict)
            or not isinstance(entry.get("bytes"), int)
            or isinstance(entry["bytes"], bool)
            or entry["bytes"] < 0
            or not isinstance(entry.get("sha256"), str)
            or not files._DIGEST.fullmatch(entry["sha256"])
        ):
            raise ValueError(f"Invalid manifest entry: {name}")
    return entries


def _archive_name(name: str) -> PurePosixPath:
    """Require one normalized relative archive path.

    Args:
        name: POSIX member or manifest path.

    Returns:
        Validated relative path.
    """
    path = PurePosixPath(name)
    if (
        not path.parts
        or path.is_absolute()
        or ".." in path.parts
        or path.as_posix() != name
        or "\\" in name
    ):
        raise ValueError(f"Invalid archive path: {name}")
    return path


def verify_directory(directory: Path, *, manifest: dict[str, Any]) -> dict[str, Any]:
    """Verify an extracted directory against an authenticated file inventory.

    This accepts the same files mapping as verify_bundle, including complete
    inventories without a transport format field. The caller must
    authenticate the manifest itself through a pinned parent archive or digest.
    The root MANIFEST.json is excluded; every other file must be listed.

    Args:
        directory: Extracted payload to check; links and special files fail.
        manifest: Authenticated manifest containing bytes and SHA-256 per file.

    Returns:
        The supplied manifest after all inventory and content checks pass.

    Raises:
        ValueError: Paths, inventory or file contents differ.
    """
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("Expected an ordinary directory")
    paths = {}
    for path in directory.rglob("*"):
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise ValueError(f"Invalid extracted path: {path}")
        name = path.relative_to(directory).as_posix()
        _archive_name(name)
        if path.is_file() and name != MANIFEST:
            paths[name] = path
    entries = _manifest_entries(manifest, set(paths))
    for name, path in paths.items():
        entry = entries[name]
        if (
            path.stat().st_size != entry["bytes"]
            or files._sha256(path) != entry["sha256"]
        ):
            raise ValueError(f"Extracted file differs from its manifest: {name}")
    return manifest


def extract_archive(archive: Path, destination: Path, *, sha256: str) -> Path:
    """Extract a digest-bound archive snapshot into a new directory.

    Works for verified run bundles and their nested source/data archives. It
    validates the entire member inventory before extraction, allowing only
    normalized directories and regular files. It does not infer scientific
    acceptance or trust a digest read from an unauthenticated source.

    Args:
        archive: Tar archive whose identity is supplied by a trusted receipt.
        destination: New extraction directory; existing paths are rejected.
        sha256: Required lowercase SHA-256 of the complete archive.

    Returns:
        Destination containing the verified snapshot's members.

    Raises:
        ValueError: The digest, member paths or member types are invalid.
        FileExistsError: The destination already exists.
    """
    if not isinstance(sha256, str) or not files._DIGEST.fullmatch(sha256):
        raise ValueError("An exact archive SHA-256 is required")
    with tempfile.TemporaryDirectory(prefix="replica-extract-") as directory:
        snapshot = Path(directory) / "archive.tar"
        shutil.copyfile(archive, snapshot)
        if files._sha256(snapshot) != sha256:
            raise ValueError("Archive differs from the requested SHA-256")
        with tarfile.open(snapshot, "r:*") as source:
            members = source.getmembers()
            names: dict[str, bool] = {}
            for member in members:
                name = member.name.removesuffix("/") if member.isdir() else member.name
                _archive_name(name)
                if not (member.isfile() or member.isdir()) or name in names:
                    raise ValueError(f"Invalid or duplicate archive member: {name}")
                names[name] = member.isdir()
            for name in names:
                for parent in PurePosixPath(name).parents:
                    if parent.as_posix() in names and not names[parent.as_posix()]:
                        raise ValueError("Archive file is also a parent directory")
            destination.mkdir(parents=True, exist_ok=False)
            source.extractall(destination, filter="data")
    return destination
