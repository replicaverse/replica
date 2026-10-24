"""Index verified results-store objects and committed study artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import tarfile
from copy import deepcopy
from pathlib import Path
from typing import Any

from .layout import hub_path, parse_run
from .store import _archive_name, _manifest_entries, verify_bundle

SCHEMA = "replica.results-catalog.v4"
#: GitHub repository names: owner/name, neither segment the path "." or "..".
_REPOSITORY = re.compile(r"(?!\.\.?/)[A-Za-z0-9_.-]+/(?!\.\.?$)[A-Za-z0-9_.-]+")
_ORIGIN = re.compile(
    r"^(?:https://github\.com/|git@github\.com:|ssh://git@github\.com/)"
    r"(?P<repository>[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+?)(?:\.git)?/?$"
)
OUTPUTS = ("README.md", "catalog.json")


def _read(path: Path) -> Any:
    return json.loads(path.read_text())


def _revision(value: str) -> None:
    if not re.fullmatch(r"[0-9a-f]{40}", value):
        raise ValueError("An immutable full commit identifier is required")


def _references(value: Any) -> list[dict[str, Any]]:
    found = []
    if isinstance(value, dict):
        if {"repo_id", "revision", "path", "sha256", "bytes"} <= value.keys():
            found.append(value)
        for child in value.values():
            found.extend(_references(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(_references(child))
    return found


def snapshot_store(repo_id: str, *, revision: str = "main") -> dict[str, Any]:
    """List all objects at an immutable results repository revision.

    Args:
        repo_id: Hugging Face dataset repository identifier.
        revision: Branch or commit to resolve to an immutable revision.
    """
    from huggingface_hub import HfApi

    api = HfApi()
    head = api.repo_info(repo_id, repo_type="dataset", revision=revision)
    info = api.repo_info(
        repo_id, repo_type="dataset", revision=head.sha, files_metadata=True
    )
    return {
        "repo_id": repo_id,
        "revision": info.sha,
        "objects": sorted(
            [
                {
                    "path": r.rfilename,
                    "bytes": r.size,
                    "git_blob": r.blob_id,
                    "sha256": r.lfs.sha256 if r.lfs else None,
                }
                for r in info.siblings
            ],
            key=lambda r: r["path"],
        ),
    }


def verify_object(repo_id: str, revision: str, item: dict[str, Any]) -> dict[str, Any]:
    """Download one pinned object and verify its bytes and bundle members.

    Args:
        repo_id: Results dataset repository.
        revision: Full Hub commit identifier.
        item: Path, byte length, Git blob identifier and optional LFS SHA-256.
    """
    from huggingface_hub import hf_hub_download

    _revision(revision)
    name = str(_archive_name(item["path"]))
    path = Path(hf_hub_download(repo_id, name, repo_type="dataset", revision=revision))
    size = path.stat().st_size
    sha = hashlib.sha256()
    blob = hashlib.sha1(f"blob {size}\0".encode())
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            sha.update(block)
            blob.update(block)
    expected = item.get("sha256") or item["git_blob"]
    actual = sha.hexdigest() if item.get("sha256") else blob.hexdigest()
    if size != item["bytes"] or actual != expected:
        raise ValueError(f"Object differs from Hub identity: {name}")
    result: dict[str, Any] = {
        "path": name,
        "repo_id": repo_id,
        "revision": revision,
        "bytes": size,
        "sha256": sha.hexdigest(),
        "git_blob": item["git_blob"],
        "verified": True,
    }
    if name.startswith(("bundles/", "provenance/")):
        manifest = verify_bundle(path)
        if not _bundle_path_matches(name, sha.hexdigest(), manifest):
            raise ValueError(f"Bundle identity differs from its path: {name}")
        result["manifest"] = manifest
        result["provenance_records"] = _bundle_records(path, manifest)
    elif "/" in name and name.endswith(".json") and size <= 8 * 1024 * 1024:
        result["provenance_record"] = {"recorded_fields": _record_fields(_read(path))}
    return result


def _bundle_path_matches(name: str, digest: str, manifest: dict) -> bool:
    if name.startswith("bundles/"):
        return name == f"bundles/{manifest.get('study')}/{digest}.tar.gz"
    studies = manifest.get("studies")
    return (
        name == f"provenance/{digest}.tar.gz"
        and manifest.get("scope") == "run_provenance"
        and isinstance(studies, list)
        and bool(studies)
        and all(
            isinstance(study, str) and re.fullmatch(r"[a-z][a-z0-9_]*", study)
            for study in studies
        )
    )


def _verified_objects(snapshot: dict, receipts: list[dict]) -> dict[str, dict]:
    objects = {str(_archive_name(r["path"])): r for r in snapshot["objects"]}
    verified = {str(_archive_name(r["path"])): r for r in receipts}
    if len(objects) != len(snapshot["objects"]) or len(verified) != len(receipts):
        raise ValueError("Duplicate object or verification path")
    if objects.keys() != verified.keys():
        raise ValueError("Verification must cover every store object exactly")
    for name, item in objects.items():
        check = verified[name]
        if (
            check.get("verified") is not True
            or check["revision"] != snapshot["revision"]
            or check["repo_id"] != snapshot["repo_id"]
            or check["bytes"] != item["bytes"]
            or check["git_blob"] != item["git_blob"]
            or not re.fullmatch(r"[0-9a-f]{64}", check["sha256"])
            or (item.get("sha256") and check["sha256"] != item["sha256"])
        ):
            raise ValueError(f"Verification identity differs: {name}")
        if name.startswith(("bundles/", "provenance/")):
            manifest = check.get("manifest", {})
            if not _bundle_path_matches(name, check["sha256"], manifest):
                raise ValueError(f"Bundle member verification is absent: {name}")
            _manifest_entries(manifest, set(manifest.get("files", {})))
    return verified


def _local_file(root: Path, name: str) -> dict[str, Any]:
    path = root / name
    if path.is_symlink():
        raise ValueError(f"Tracked artifact must be a regular file: {name}")
    body = path.read_bytes()
    return {
        "path": name,
        "bytes": len(body),
        "sha256": hashlib.sha256(body).hexdigest(),
    }


def _indexed_runs(
    root: Path,
    name: str,
    document: dict,
    tracked: list[str],
    verified: dict,
    links: dict,
    commit: str,
) -> list[dict]:
    study = Path(name).parents[1].name
    prefix = f"src/replica/studies/{study}/results/"
    owners = {}
    for local in tracked:
        if not local.startswith(prefix) or local in (
            prefix + "index.json",
            prefix + "README.md",
        ):
            continue
        matches = [r for r in document["runs"] if local.startswith(f"{prefix}{r}/")]
        if not matches:
            raise ValueError(f"Result artifact has no indexed run: {local}")
        owners[local] = max(matches, key=len)
    runs = []
    for run, entry in sorted(document["runs"].items()):
        parse_run(run)
        if not entry.get("description") or not entry.get("files"):
            raise ValueError(f"Result record lacks its manifest: {study}/{run}")
        if run.startswith("references/") != (entry["role"] == "published"):
            raise ValueError(f"Reference role and placement disagree: {study}/{run}")
        runs.append(
            {
                "study": study,
                "run": run,
                "role": entry["role"],
                "description": entry["description"],
                "analysis": parse_run(run)[0],
                "record": parse_run(run)[1],
                "produced_by": entry.get("produced_by"),
                "validated_by": entry.get("validated_by"),
                "files": [
                    _local_file(root, p) for p in sorted(owners) if owners[p] == run
                ],
                "hub_files": [],
            }
        )
        for file, digest in (entry.get("hub") or {}).items():
            remote = f"{hub_path(study, run)}/{file}"
            if remote not in verified or verified[remote]["sha256"] != digest:
                raise ValueError(f"Indexed object is absent or differs: {remote}")
            links[remote].append(
                {"commit": commit, "path": name, "run": run, "role": entry["role"]}
            )
            runs[-1]["hub_files"].append(
                {"path": remote, "bytes": verified[remote]["bytes"], "sha256": digest}
            )
        actual = {
            p["path"].removeprefix(prefix + run + "/"): p["sha256"]
            for p in runs[-1]["files"]
        }
        actual.update(entry.get("hub") or {})
        if actual != entry["files"]:
            raise ValueError(f"File manifest differs from indexed files: {study}/{run}")
    return runs


def _validate_packages(root: Path, tracked: list[str]) -> None:
    """Require ownership for artifacts throughout each study package."""
    from .files import validate_package

    packages = {name.split("/")[3] for name in tracked if len(name.split("/")) > 4}
    for study in sorted(packages):
        prefix = f"src/replica/studies/{study}/"
        paths = [
            name.removeprefix(prefix) for name in tracked if name.startswith(prefix)
        ]
        problems = validate_package(study, paths, package=root / prefix)
        if problems:
            raise ValueError(
                f"Study package artifacts are invalid: {study}: {problems}"
            )


def _source_records(
    root: Path, verified: dict, repo_id: str, commit: str
) -> tuple[list, dict, list]:
    tracked = (
        subprocess.check_output(
            ["git", "ls-files", "-z", "src/replica/studies"], cwd=root
        )
        .decode()
        .split("\0")
    )
    links: dict[str, list[dict]] = {name: [] for name in verified}
    runs = []
    metadata = []
    _validate_packages(root, tracked)
    for name in sorted(filter(None, tracked)):
        if re.fullmatch(
            r"src/replica/studies/[^/]+/results/(README.md|index.json)", name
        ):
            metadata.append(_local_file(root, name))
            remote = name.removeprefix("src/replica/")
            if remote in verified:
                if verified[remote]["sha256"] != metadata[-1]["sha256"]:
                    raise ValueError(f"Mirrored study index differs: {remote}")
                links[remote].append({"commit": commit, "path": name})
        if not name.endswith(".json"):
            continue
        document = _read(root / name)
        if re.fullmatch(r"src/replica/studies/[^/]+/results/index.json", name):
            runs.extend(
                _indexed_runs(root, name, document, tracked, verified, links, commit)
            )
        for ref in _references(document):
            if ref["repo_id"] != repo_id:
                continue
            _revision(ref["revision"])
            remote = str(_archive_name(ref["path"]))
            if remote not in verified:
                raise ValueError(f"Referenced archive is absent: {remote}")
            receipt = verified[remote]
            if (ref["sha256"], ref["bytes"]) != (receipt["sha256"], receipt["bytes"]):
                raise ValueError(f"Archive reference differs: {remote}")
            links[remote].append(
                {"commit": commit, "path": name, "revision": ref["revision"]}
            )
        if re.fullmatch(r"src/replica/studies/[^/]+/results/index.json", name):
            for entry in document.get("archives", {}).values():
                receipt = verified[entry["reference"]["path"]]
                members = receipt.get("manifest", {}).get("files", {})
                for member, digest in entry.get("records", {}).items():
                    if members.get(member, {}).get("sha256") != digest:
                        raise ValueError(f"Archive member record differs: {member}")
    covered = {r["path"] for run in runs for r in run["files"]} | {
        r["path"] for r in metadata
    }
    results = {n for n in tracked if re.match(r"src/replica/studies/[^/]+/results/", n)}
    if covered != results:
        raise ValueError(f"Results lack index coverage: {sorted(results - covered)}")
    return runs, links, metadata


def _repository(name: str) -> str:
    """Validate a GitHub repository name of the form owner/name."""
    if not _REPOSITORY.fullmatch(name):
        raise ValueError(f"Repository must be owner/name, got {name!r}")
    return name


def _origin_repository(root: Path) -> str:
    """The GitHub repository the checkout's origin points at, as owner/name.

    Args:
        root: Git checkout whose ``origin`` names the repository.

    Raises:
        ValueError: The checkout has no origin or its URL is not a GitHub
            repository; pass the repository explicitly instead.
    """
    result = subprocess.run(
        ["git", "remote", "get-url", "origin"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise ValueError(f"{root} has no origin remote; pass --repository owner/name")
    match = _ORIGIN.match(result.stdout.strip())
    if match is None:
        # The URL is left out of the message: it may carry a token.
        raise ValueError(
            "origin is not a GitHub repository URL; pass --repository owner/name"
        )
    return _repository(match.group("repository"))


def _retained_with_repository(retained: dict) -> dict:
    """A copy of the retained records whose sources name their repository.

    A source keeps an explicit repository; one without takes the repository of
    the catalog it was retained from; a record from a catalog that recorded
    none stays without, and renders against the current catalog.
    """
    records = deepcopy(retained.get("retained_records", {}))
    inherited = retained.get("source_repository")
    if inherited is not None:
        _repository(inherited)
        for record in records.values():
            for source in record.get("sources", []):
                source.setdefault("repository", inherited)
    return records


def _retained_records(records: dict, verified: dict) -> None:
    if records.keys() - verified.keys():
        raise ValueError("Origin records name absent objects")
    for name, record in records.items():
        if not record.get("description") or not (
            record.get("sources") or record.get("metadata_paths")
        ):
            raise ValueError(f"Object needs documented origin records: {name}")
        for source in record.get("sources", []):
            _revision(source["commit"])
            _archive_name(source["path"])
            if "repository" in source:
                _repository(source["repository"])
        for path in record.get("metadata_paths", []):
            if path not in verified:
                raise ValueError(f"Origin metadata is absent: {path}")
        replacement = record.get("superseded_by")
        if replacement and (replacement not in verified or replacement == name):
            raise ValueError(
                f"Replacement is absent or self-referencing: {replacement}"
            )
        seen = {name}
        while replacement:
            if replacement in seen:
                raise ValueError(f"Replacement cycle: {name}")
            seen.add(replacement)
            replacement = records.get(replacement, {}).get("superseded_by")


def _catalog_files(verified: dict, links: dict, records: dict) -> list[dict]:
    members: dict[tuple, list[dict]] = {}
    for archive, check in sorted(verified.items()):
        for member, identity in sorted(
            check.get("manifest", {}).get("files", {}).items()
        ):
            members.setdefault((identity["sha256"], identity["bytes"]), []).append(
                {"archive": archive, "member": member}
            )
    result = []
    for name, check in sorted(verified.items()):
        record = records.get(name)
        metadata = name in (*OUTPUTS, ".gitattributes")
        if not links[name] and not record and not metadata:
            raise ValueError(f"Object needs an origin record: {name}")
        parts = name.split("/")
        if not metadata and parts[0] not in {"studies", "bundles", "provenance"}:
            raise ValueError(f"Noncanonical Hub object path: {name}")
        if parts[0] == "studies":
            if len(parts) < 4 or parts[2] != "results" or not links[name]:
                raise ValueError(f"Hub study file has no indexed owner: {name}")
        result.append(
            {
                "path": name,
                "bytes": check["bytes"],
                "sha256": check["sha256"],
                "git_blob": check["git_blob"],
                "identical_members": members.get((check["sha256"], check["bytes"]), []),
                "study": parts[1]
                if parts[0] in {"bundles", "studies"}
                else parts[0]
                if len(parts) > 1 and parts[0] != "provenance"
                else None,
                "classification": "metadata"
                if metadata
                else "indexed"
                if links[name]
                else "retained",
                "references": sorted(
                    links[name], key=lambda v: json.dumps(v, sort_keys=True)
                ),
                "record": record,
                "verification": "bundle_members"
                if "manifest" in check
                else "file_bytes",
                **(
                    {"members": check["manifest"]["files"]}
                    if "manifest" in check
                    else {}
                ),
            }
        )
    return result


def _record_fields(value: Any, pointer: str = "") -> list[dict[str, str]]:
    names = {
        "commit",
        "base_commit",
        "public_commit",
        "producer_commit",
        "scientific_producer_commit",
        "scientific_producer",
        "producer",
        "author_commit",
        "author_revision",
        "official_commit",
        "source_revision",
        "packaging_commit",
        "verifier_commit",
        "validator_commit",
        "driver_commit",
        "replica_commit",
        "code_commit",
        "environment_sha256",
        "environment_digest",
    }
    fields = []
    if isinstance(value, dict):
        for key, child in sorted(value.items()):
            location = pointer + "/" + key.replace("~", "~0").replace("/", "~1")
            if (
                key in names
                and isinstance(child, str)
                and re.fullmatch(r"[0-9a-f]{7,64}", child)
            ):
                fields.append({"pointer": location, "value": child})
            fields.extend(_record_fields(child, location))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            fields.extend(_record_fields(child, f"{pointer}/{index}"))
    return fields


def _file_identities(document: Any) -> dict[str, dict]:
    if not isinstance(document, dict):
        return {}
    entries = document.get("files", document.get("artifacts", []))
    if isinstance(entries, dict):
        entries = [
            {"path": name, **value}
            for name, value in entries.items()
            if isinstance(value, dict)
        ]
    if not isinstance(entries, list):
        return {}
    return {
        entry["path"]: {"bytes": entry["bytes"], "sha256": entry["sha256"]}
        for entry in entries
        if isinstance(entry, dict)
        and isinstance(entry.get("path"), str)
        and isinstance(entry.get("bytes"), int)
        and entry["bytes"] >= 0
        and isinstance(entry.get("sha256"), str)
        and re.fullmatch(r"[0-9a-f]{64}", entry["sha256"])
    }


def _bundle_records(path: Path, manifest: dict) -> dict[str, dict]:
    records = {}
    with tarfile.open(path, "r|gz") as archive:
        for member in archive:
            if not member.name.endswith(".json") or member.size > 8 * 1024 * 1024:
                continue
            expected = manifest["files"].get(member.name)
            if expected is None:
                continue
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError(f"Archive record is not a file: {member.name}")
            body = stream.read()
            if (
                len(body) != expected["bytes"]
                or hashlib.sha256(body).hexdigest() != expected["sha256"]
            ):
                raise ValueError(f"Archive record bytes differ: {member.name}")
            document = json.loads(body)
            fields = _record_fields(document)
            records[member.name] = {
                **expected,
                "recorded_fields": fields,
                "file_identities": _file_identities(document),
            }
    return records


def _indexed_identities(run: dict) -> dict[str, dict]:
    """Result identities keyed by their logical source-package paths."""
    files = {
        f["path"]: {"bytes": f["bytes"], "sha256": f["sha256"]} for f in run["files"]
    }
    for file in run["hub_files"]:
        files["src/replica/" + file["path"]] = {
            "bytes": file["bytes"],
            "sha256": file["sha256"],
        }
    return files


def _run_provenance(
    root: Path, runs: list[dict], verified: dict, retained: dict
) -> None:
    annotations = retained.get("run_records", {})
    known = {f"{run['study']}/{run['run']}" for run in runs}
    if annotations.keys() - known:
        raise ValueError("Provenance records name absent runs")
    copies: dict[tuple, list[dict]] = {}
    for archive, receipt in sorted(verified.items()):
        manifest = receipt.get("manifest", {})
        for path, identity in sorted(manifest.get("files", {}).items()):
            key = (manifest.get("study"), identity["bytes"], identity["sha256"])
            copies.setdefault(key, []).append({"archive": archive, "path": path})
    for run in runs:
        documents = []
        archived = []
        for file in run["files"]:
            path = file["path"]
            if path.endswith(".json"):
                fields = _record_fields(_read(root / path))
                if fields or re.search(
                    r"(?:source|provenance|inputs|generation|environment|manifest)",
                    Path(path).name,
                    re.I,
                ):
                    documents.append({**file, "recorded_fields": fields})
        for path, identity in _indexed_identities(run).items():
            matches = copies.get(
                (run["study"], identity["bytes"], identity["sha256"]), []
            )
            if matches:
                archived.append({"file": path, "copies": matches})
        annotation = annotations.get(f"{run['study']}/{run['run']}", {})
        records = annotation.get("records", [])
        _check_run_records(records, verified, run)
        hub_records = [
            {**file, **verified[file["path"]]["provenance_record"]}
            for file in run["hub_files"]
            if "provenance_record" in verified[file["path"]]
            and (
                verified[file["path"]]["provenance_record"]["recorded_fields"]
                or re.search(
                    r"(?:source|provenance|inputs|generation|environment|manifest)",
                    Path(file["path"]).name,
                    re.I,
                )
            )
        ]
        run["provenance"] = {
            "stamp_status": "recorded" if run["produced_by"] else "absent",
            "local_records": documents,
            "hub_records": hub_records,
            "archive_records": records,
            "identical_archive_members": archived,
            "limitations": annotation.get("limitations", []),
        }


def _check_run_records(records: list[dict], verified: dict, run: dict) -> None:
    study = run["study"]
    files = _indexed_identities(run)
    for record in records:
        archive = record["archive"]
        manifest = verified.get(archive, {}).get("manifest", {})
        if manifest.get("study") != study and study not in manifest.get("studies", []):
            raise ValueError(
                f"Run provenance names an absent or different-study archive: {archive}"
            )
        member = manifest["files"].get(str(_archive_name(record["path"])))
        if member != {"bytes": record["bytes"], "sha256": record["sha256"]}:
            raise ValueError(
                f"Run provenance differs from its archived member: {record['path']}"
            )
        authenticated = (
            verified[archive].get("provenance_records", {}).get(record["path"])
        )
        if (
            authenticated is None
            or record.get("recorded_fields", []) != authenticated["recorded_fields"]
        ):
            raise ValueError(
                "Recorded identities differ from the authenticated archive record"
            )
        if record.get("relation") not in {"file_identity", "archive_context"}:
            raise ValueError(
                "Run provenance needs an explicit relationship to the artifact"
            )
        matches = record.get("matches", [])
        if record["relation"] == "file_identity" and not matches:
            raise ValueError("File identity provenance requires matched result files")
        for match in matches:
            recorded = authenticated["file_identities"].get(match["recorded_path"])
            if match["file"] not in files or files[match["file"]] != recorded:
                raise ValueError(
                    "Provenance manifest does not identify the indexed result bytes"
                )
        # Values remain attributed statements from the original record.
        # Their field names distinguish source, producer and packaging identities.
        for field in record.get("recorded_fields", []):
            if not isinstance(field.get("pointer"), str) or not re.fullmatch(
                r"[0-9a-f]{7,64}", field.get("value", "")
            ):
                raise ValueError("Invalid recorded identity field")


def build_catalog(
    root: Path,
    snapshot: dict[str, Any],
    receipts: list[dict[str, Any]],
    retained: dict[str, Any] | None = None,
    repository: str | None = None,
) -> dict[str, Any]:
    """Join verified store objects to committed run and archive records.

    Args:
        root: Clean Git checkout containing the study packages.
        snapshot: Results repository, immutable revision and complete objects list.
        receipts: Byte and member verification receipts from verify_object.
        retained: Earlier catalog or supplemental records preserving artifact origins.
        repository: GitHub repository (owner/name) that holds the source commit;
            defaults to the checkout's origin.
    """
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    repository = (
        _origin_repository(root) if repository is None else _repository(repository)
    )
    if subprocess.check_output(["git", "diff", "HEAD", "--name-only"], cwd=root):
        raise ValueError("Catalog source must have no tracked changes")
    _revision(snapshot["revision"])
    verified = _verified_objects(snapshot, receipts)
    runs, links, metadata = _source_records(root, verified, snapshot["repo_id"], commit)
    retained = retained or {}
    _run_provenance(root, runs, verified, retained)
    records = _retained_with_repository(retained)
    _retained_records(records, verified)
    result = {
        "format": SCHEMA,
        "source_commit": commit,
        "source_repository": repository,
        "repo_id": snapshot["repo_id"],
        "revision": snapshot["revision"],
        "generated_files": list(OUTPUTS),
        "retained_records": dict(sorted(records.items())),
        "run_records": dict(sorted(retained.get("run_records", {}).items())),
        "runs": runs,
        "source_metadata": metadata,
        "files": _catalog_files(verified, links, records),
    }
    if subprocess.check_output(["git", "diff", commit, "--name-only"], cwd=root):
        raise ValueError("Catalog source changed during indexing")
    return result


def render_readme(catalog: dict[str, Any]) -> str:
    """Render the study index and immutable artifact links.

    Args:
        catalog: Verified catalog produced by build_catalog.
    """
    repository = catalog["source_repository"]
    code = f"https://github.com/{repository}/blob/{catalog['source_commit']}"
    hub = f"https://huggingface.co/datasets/{catalog['repo_id']}/resolve/{catalog['revision']}"
    lines = [
        "---",
        "pretty_name: REPLICA results",
        "---",
        "",
        "# REPLICA results",
        "",
        "Study results, fitted-state archives, source identities and replay evidence.",
        "[catalog.json](catalog.json) indexes every stored object and each committed result run.",
        "",
        "Reproduction uses original data; replication uses new data. Repetition retains",
        "the released procedure; reanalysis applies corrections. Each run records its",
        "evidence role separately. Byte verification establishes artifact integrity.",
        "Scientific scope and agreement are stated in the study's procedure documentation.",
        "A null producer field means the run index supplies no producer stamp; consult",
        "the linked archive and source records for additional provenance.",
        "Each run lists local provenance documents, attributed identity fields and",
        "byte-identical archived copies. Archive-context records can describe several",
        "computations; their identities are never substituted for a run's producer stamp.",
        "",
        f"Code: `{catalog['source_commit']}`. Payload revision: `{catalog['revision']}`.",
        "Catalog outputs are listed separately from payload identities. An earlier catalog",
        "in the payload inventory describes the revision recorded inside that file.",
        "",
        "| Study | Indexed runs | Archives | Other stored files | Bytes |",
        "|---|---:|---:|---:|---:|",
    ]
    studies = {r["study"] for r in catalog["runs"]} | {
        r["study"] for r in catalog["files"] if r["study"]
    }
    for study in sorted(studies):
        objects = [r for r in catalog["files"] if r["study"] == study]
        archives = sum(r["path"].startswith("bundles/") for r in objects)
        count = sum(r["study"] == study for r in catalog["runs"])
        lines.append(
            f"| [{study}]({code}/src/replica/studies/{study}/README.md) | {count} | {archives} | {len(objects) - archives} | {sum(r['bytes'] for r in objects)} |"
        )
    lines += [
        "",
        "## Run provenance",
        "",
        "| Study | Run | Producer stamp | Local records | Hub records | Archive records | Files with archived copies |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for run in catalog["runs"]:
        provenance = run["provenance"]
        path = f"src/replica/studies/{run['study']}/results/index.json"
        lines.append(
            f"| {run['study']} | [{run['analysis']}: {run['description']}]({code}/{path}) | {provenance['stamp_status']} | "
            f"{len(provenance['local_records'])} | {len(provenance['hub_records'])} | {len(provenance['archive_records'])} | "
            f"{len(provenance['identical_archive_members'])} |"
        )
    lines += [
        "",
        "## Archives",
        "",
        "| Study | Archive | Source records |",
        "|---|---|---|",
    ]
    for item in catalog["files"]:
        if not item["path"].startswith("bundles/"):
            continue
        origins = item["references"] or (item.get("record") or {}).get("sources", [])
        sources = ", ".join(
            f"[{r['path']}](https://github.com/{r.get('repository', repository)}"
            f"/blob/{r['commit']}/{r['path']})"
            for r in origins
        )
        lines.append(
            f"| {item['study']} | [{item['sha256'][:12]}]({hub}/{item['path']}) | {sources} |"
        )
    lines += [
        "",
        "## Retained artifacts",
        "",
        "These objects have origin records outside the current run indexes.",
        "Replacement links describe a changed source reference; they do not assert identical contents.",
        "",
        "| Object | Scope |",
        "|---|---|",
    ]
    for name, record in catalog["retained_records"].items():
        lines.append(f"| [{name}]({hub}/{name}) | {record['description']} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    """Build catalog files from a frozen store snapshot and verification receipts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--receipts", required=True, type=Path)
    parser.add_argument("--retained", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--repository",
        metavar="OWNER/NAME",
        help="GitHub repository holding the source commit (default: the origin)",
    )
    args = parser.parse_args()
    catalog = build_catalog(
        args.source,
        _read(args.snapshot),
        _read(args.receipts),
        _read(args.retained) if args.retained else None,
        repository=args.repository,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "catalog.json").write_text(
        json.dumps(catalog, indent=2, sort_keys=True) + "\n"
    )
    (args.output / "README.md").write_text(render_readme(catalog))


if __name__ == "__main__":
    main()
