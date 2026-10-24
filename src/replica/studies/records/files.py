"""Where a study's result files are and which code produced them.

A run's files may sit in three places: the checkout, beside the study's code;
the durable results store a cluster mounts (``REPLICA_RESULTS_STORE``); and
the results hub, the Hugging Face dataset repository ``HF_REPO``, for files
``index.json`` lists with their checksums under ``hub``. ``results_file``
looks in that order and checks a hub download against the recorded digest.
``run_dir`` gives directory readers one tree holding the checkout files and
the hub files side by side. A run whose files are all on the hub has an
index entry and no checkout directory.

``index.json`` also records what produced each run under ``produced_by``: the
package commit, whether the tree was dirty, when, and a footprint, one digest
over the files the run depends on: the paths listed under ``depends_on``,
relative to the ``replica`` package (an empty list for a run that no package
file produced), or, when the stamp has no ``depends_on``, the study package and
every ``replica`` module it imports, including packaged dataset YAML when metadata
I/O is imported and registered patches when the external registry is imported.
``stale_runs`` compares that
footprint with the current tree. ``null`` marks a run whose
producer is not recorded; it cannot be checked and is reported as unstamped.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from ...paths import user_cache_root
from ...utils.digest import file_sha256
from ...utils.git import git_output
from ...utils.imports import import_names
from .layout import INDEX, ROLES, description_conflicts, hub_path, parse_run, runs

STUDIES_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = STUDIES_DIR.parents[1]
SCHEMA = 3
STORE_ENV = "REPLICA_RESULTS_STORE"
HF_REPO = os.environ.get(
    "REPLICA_RESULTS_HF_REPO", "replicaverse/replica-study-bundles"
)
HF_REVISION = os.environ.get("REPLICA_RESULTS_HF_REVISION", "main")
HUB_URL = f"https://huggingface.co/datasets/{HF_REPO}"
#: Parts of a study package that do not shape its numbers.
_OUTSIDE_FOOTPRINT = {
    "results",
    "figures",
    "__pycache__",
    "README.md",
    "EVIDENCE.md",
    "claims.yaml",
    "replication_results.yaml",
    "architecture.toml",
}
#: Study registries, policy and dependency declarations, not result artifacts.
_METADATA_FILES = {
    "claims.yaml",
    "replication_results.yaml",
    "architecture.toml",
    "requirements.txt",
}
_STAMP_KEYS = {"commit", "dirty", "produced_at", "footprint", "inputs"}
_OPTIONAL_STAMP_KEYS = {"driver_commit", "restamped", "depends_on"}
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_REPO_ID = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")
_ENTRY_KEYS = {
    "role",
    "note",
    "description",
    "files",
    "produced_by",
    "validated_by",
    "hub",
    "store",
}
_verified: set[tuple[str, str]] = set()


def _sha256(path: Path, digests: dict[Path, str] | None = None) -> str:
    """The file's digest, reused from ``digests`` when one call hashes many studies."""
    if digests is not None and path in digests:
        return digests[path]
    value = file_sha256(path)
    if digests is not None:
        digests[path] = value
    return value


def studies_with_results() -> list[str]:
    """Every study package that records runs in an ``index.json``."""
    return sorted(p.parents[1].name for p in STUDIES_DIR.glob(f"*/results/{INDEX}"))


def index_path(study: str) -> Path:
    """The ``index.json`` of one study.

    Args:
        study: Study package name.
    """
    return STUDIES_DIR / study / "results" / INDEX


def read_index(study: str) -> dict[str, Any]:
    """The whole index document of one study.

    Args:
        study: Study package name.
    """
    return json.loads(index_path(study).read_text())


def write_index(study: str, document: dict[str, Any]) -> None:
    """Write an index document with its runs in sorted order.

    Args:
        study: Study package name.
        document: The index document, ``{"schema": ..., "runs": {...}}``.
    """
    document["runs"] = dict(sorted(document["runs"].items()))
    index_path(study).write_text(json.dumps(document, indent=2) + "\n")


def _entry(study: str, run: str) -> dict[str, Any]:
    try:
        return read_index(study)["runs"][run]
    except (FileNotFoundError, KeyError):
        raise FileNotFoundError(f"{study}: no run {run!r} in {INDEX}") from None


def _download_error(exc: Exception) -> tuple[bool, str | None]:
    """Classify Hub failures without exposing request URLs or credentials."""
    import ssl
    from importlib import import_module

    from huggingface_hub.utils import (
        FileMetadataError,
        HfHubHTTPError,
        LocalEntryNotFoundError,
        OfflineModeIsEnabled,
    )

    transient: tuple[type[BaseException], ...] = (TimeoutError, ConnectionError)
    permanent: tuple[type[BaseException], ...] = (ssl.SSLError, OfflineModeIsEnabled)
    http_errors: tuple[type[BaseException], ...] = (HfHubHTTPError,)
    try:
        from requests import exceptions as requests_errors
    except ImportError:
        pass
    else:
        transient += (requests_errors.Timeout, requests_errors.ConnectionError)
        permanent += (requests_errors.SSLError, requests_errors.ProxyError)
        http_errors += (requests_errors.HTTPError,)
    for module in ("httpx", "httpx2"):
        try:
            client = import_module(module)
        except ImportError:
            continue
        transient += (
            client.TimeoutException,
            client.NetworkError,
            client.RemoteProtocolError,
        )
        permanent += (client.ProxyError, client.UnsupportedProtocol)
        http_errors += (client.HTTPStatusError,)

    chain: list[BaseException] = []
    error: BaseException | None = exc
    while error is not None and all(error is not previous for previous in chain):
        chain.append(error)
        error = error.__cause__
    known = (
        transient
        + permanent
        + http_errors
        + (
            LocalEntryNotFoundError,
            FileMetadataError,
        )
    )
    if not any(isinstance(error, known) for error in chain):
        return False, None
    details = []
    statuses = []
    for error in chain:
        label = type(error).__name__
        response = getattr(error, "response", None)
        if isinstance(error, http_errors) and response is not None:
            statuses.append(response.status_code)
            label += f" (HTTP {response.status_code})"
        details.append(label)
    retryable = not any(isinstance(error, permanent) for error in chain) and (
        all(status in {408, 429, 500, 502, 503, 504} for status in statuses)
        if statuses
        else any(isinstance(error, transient) for error in chain)
    )
    return retryable, " caused by ".join(details)


def _hub_download(
    remote: str,
    *,
    revision: str = "main",
    repo_id: str | None = None,
    force_download: bool = False,
) -> str:
    """Retry transient transfers of the same artifact, preserving its revision."""
    import logging
    import time

    from huggingface_hub import hf_hub_download

    repo_id = repo_id or HF_REPO
    for attempt in range(1, 4):
        try:
            return hf_hub_download(
                repo_id=repo_id,
                filename=remote,
                repo_type="dataset",
                revision=revision,
                force_download=force_download,
            )
        except Exception as exc:
            retryable, detail = _download_error(exc)
            if detail is None:
                raise
            identity = f"{repo_id}/{remote} at revision {revision}"
            if not retryable or attempt == 3:
                error_type = (
                    FileNotFoundError if isinstance(exc, FileNotFoundError) else OSError
                )
                raise error_type(
                    f"Cannot download {identity}: {detail} (attempt {attempt}/3)"
                ) from exc
            delay = 5 * attempt
            logging.getLogger(__name__).warning(
                "Download failed for %s: %s; retrying in %ss (attempt %s/3)",
                identity,
                detail,
                delay,
                attempt,
            )
            time.sleep(delay)
    raise AssertionError("unreachable")


def hub_repo(document: dict[str, Any]) -> str:
    """Resolve a study's explicit repository or the configured default.

    Args:
        document: Study result index.
    """
    value = os.environ.get("REPLICA_RESULTS_HF_REPO") or document.get(
        "hub_repo", HF_REPO
    )
    if not isinstance(value, str) or not _REPO_ID.fullmatch(value):
        raise ValueError("hub_repo must be an owner/repository identifier")
    return value


def _fetch(study: str, run: str, name: str, digest: str) -> Path:
    document = read_index(study)
    revision = os.environ.get("REPLICA_RESULTS_HF_REVISION") or document.get(
        "hub_revision", HF_REVISION
    )
    path = Path(
        _hub_download(
            f"{hub_path(study, run)}/{name}",
            revision=revision,
            repo_id=hub_repo(document),
        )
    )
    key = (str(path), digest)
    if key not in _verified:
        actual = _sha256(path)
        if actual != digest:
            raise ValueError(
                f"{study}/results/{run}/{name}: the hub copy has digest {actual}, "
                f"{INDEX} records {digest}"
            )
        _verified.add(key)
    return path


def results_file(study: str, run: str, name: str) -> Path:
    """One result file, from the checkout, the mounted store, or the hub.

    Args:
        study: Study package name.
        run: Run path relative to the study's ``results/``.
        name: File path relative to the run.

    Raises:
        FileNotFoundError: The file is in none of the three places.
        OSError: The Hub transfer failed.
        ValueError: An available copy does not match the recorded digest.
    """
    entry = _entry(study, run)
    parse_run(run)
    if (
        not name
        or Path(name).is_absolute()
        or any(p in {"", ".", ".."} for p in name.split("/"))
    ):
        raise ValueError(f"Invalid result file path: {name!r}")
    expected = entry.get("files", {}).get(name)
    local = STUDIES_DIR / study / "results" / run / name
    store = os.environ.get(STORE_ENV)
    candidates = [local]
    if store and entry.get("store"):
        candidates.append(Path(store) / entry["store"] / "results" / name)
    for candidate in candidates:
        if candidate.is_file():
            if expected is None or _sha256(candidate) != expected:
                raise ValueError(
                    f"{study}/{run}/{name}: local or stored copy differs from its manifest digest"
                )
            return candidate
    digest = (entry.get("hub") or {}).get(name)
    if digest is None:
        raise FileNotFoundError(
            f"{study}/results/{run}/{name}: absent from the checkout and not "
            f"listed under hub in {INDEX}"
        )
    return _fetch(study, run, name, digest)


def _link(target: Path, source: Path) -> None:
    if target.is_symlink() and target.resolve() == source.resolve():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid4().hex}")
    try:
        temporary.symlink_to(source)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def run_dir(study: str, run: str) -> Path:
    """A directory holding every file of a run.

    A run with no hub files is its checkout directory. Otherwise a view under
    the user cache links the checkout files and the fetched hub files together.

    Args:
        study: Study package name.
        run: Run path relative to the study's ``results/``.
    """
    local = STUDIES_DIR / study / "results" / run
    entry = _entry(study, run)
    hub = entry.get("hub") or {}
    if entry.get("files") and all((local / name).is_file() for name in entry["files"]):
        for name in entry["files"]:
            results_file(study, run, name)
        return local
    if not hub:
        if local.is_dir():
            return local
        raise FileNotFoundError(f"{study}/results/{run}: not in the checkout")
    view = user_cache_root() / "results" / study / run
    for name in hub:
        _link(view / name, results_file(study, run, name))
    if local.is_dir():
        for path in local.rglob("*"):
            if path.is_file():
                _link(view / path.relative_to(local), path)
    return view


def locate(study: str, relative: str) -> Path:
    """Resolve ``<run>/<name>`` when the run boundary is not known.

    Args:
        study: Study package name.
        relative: File path relative to the study's ``results/``.
    """
    for run in sorted(read_index(study)["runs"], key=len, reverse=True):
        if relative.startswith(run + "/"):
            return results_file(study, run, relative[len(run) + 1 :])
    raise FileNotFoundError(f"{study}/results/{relative}: no run in {INDEX} holds it")


def hub_url(study: str, relative: str) -> str | None:
    """The hub page of an indexed file or directory, or ``None`` when unlisted.

    Args:
        study: Study package name.
        relative: File path relative to the study's ``results/``.
    """
    document = read_index(study)
    entries = document["runs"]
    base = f"https://huggingface.co/datasets/{hub_repo(document)}"
    revision = os.environ.get("REPLICA_RESULTS_HF_REVISION") or document.get(
        "hub_revision", HF_REVISION
    )
    relative = relative.rstrip("/")
    for run in sorted(entries, key=len, reverse=True):
        if relative == run and entries[run].get("hub"):
            return f"{base}/tree/{revision}/{hub_path(study, run)}"
        if relative.startswith(run + "/"):
            name = relative[len(run) + 1 :]
            if name in (entries[run].get("hub") or {}):
                return f"{base}/blob/{revision}/{hub_path(study, run)}/{name}"
            if any(file.startswith(name + "/") for file in entries[run].get("hub", {})):
                return f"{base}/tree/{revision}/{hub_path(study, run)}/{name}"
    return None


def _imports(path: Path) -> set[str]:
    # Relative imports resolve against the package: the directory of a module,
    # the package itself for its ``__init__``; both drop the last part.
    package = ".".join(path.relative_to(SRC_DIR).with_suffix("").parts[:-1])
    return {
        name
        for name in import_names(path.read_text(), package)
        if name == "replica" or name.startswith("replica.")
    }


def _module_files(name: str) -> list[Path]:
    """The files Python runs to import ``name``: package inits, then the module."""
    files: list[Path] = []
    parts = name.split(".")
    for depth in range(1, len(parts) + 1):
        base = SRC_DIR.joinpath(*parts[:depth])
        if base.is_dir():
            init = base / "__init__.py"
            if init.is_file():
                files.append(init)
        elif base.with_suffix(".py").is_file():
            files.append(base.with_suffix(".py"))
            break
        else:
            break
    return files


def _external_patch_files() -> list[Path]:
    """Resolve declared checkout patches within this source tree."""
    from ...external.repositories import EXTERNAL_REPOS

    package = SRC_DIR / "replica"
    paths = []
    for name, dependencies in EXTERNAL_REPOS.items():
        for dependency in dependencies:
            if not dependency.patch:
                continue
            relative = Path(dependency.patch)
            path = package / relative
            if relative.is_absolute() or not path.resolve().is_relative_to(
                package.resolve()
            ):
                raise ValueError(
                    f"{name}: patch must stay inside the replica package: {relative}"
                )
            if not path.is_file():
                raise FileNotFoundError(f"{name}: missing declared patch {path}")
            paths.append(path)
    return paths


def footprint_files(study: str) -> dict[str, Path]:
    """Every file whose content the footprint covers, keyed by source path.

    The study package minus its results, figures and README, plus the closure
    of ``replica`` modules its Python files import, package inits included.
    Metadata I/O imports include all packaged dataset YAML. External checkout
    registry imports include their declared patches. Explicit ``depends_on``
    lists bypass this conservative resource discovery.

    Args:
        study: Study package name.
    """
    package = STUDIES_DIR / study
    files: dict[str, Path] = {}

    def add(path: Path) -> bool:
        key = path.relative_to(SRC_DIR).as_posix()
        if key in files:
            return False
        files[key] = path
        return True

    pending: list[Path] = []
    for path in sorted(package.rglob("*")):
        if not path.is_file() or path.suffix in {".pyc", ".pyi"}:
            continue
        if _OUTSIDE_FOOTPRINT & set(path.relative_to(package).parts):
            continue
        add(path)
        if path.suffix == ".py":
            pending.append(path)
    pending += _module_files(".".join(package.relative_to(SRC_DIR).parts))
    while pending:
        path = pending.pop()
        add(path)
        for name in _imports(path):
            for file in _module_files(name):
                if add(file) and file.suffix == ".py":
                    pending.append(file)
    if "replica/external/repositories.py" in files:
        for path in _external_patch_files():
            add(path)
    if "replica/dataset/metadata/io.py" in files:
        for path in (SRC_DIR / "replica/dataset/metadata/yaml").rglob("*.yaml"):
            if path.is_file():
                add(path)
    return dict(sorted(files.items()))


def code_footprint(
    study: str,
    digests: dict[Path, str] | None = None,
    depends_on: list[str] | None = None,
) -> str:
    """One digest over the paths and contents of the files a run depends on.

    Args:
        study: Study package name.
        digests: File digests shared across studies checked in one call, so
            the library layers are hashed once.
        depends_on: Paths relative to the ``replica`` package that the run's
            content depends on; ``None`` means the study's footprint files and
            an empty list a run no package file produced. A listed file that
            does not exist hashes as absent.
    """
    if depends_on is None:
        files = footprint_files(study)
    else:
        files = {name: SRC_DIR / "replica" / name for name in sorted(depends_on)}
    digest = hashlib.sha256()
    for key, path in files.items():
        value = _sha256(path, digests) if path.is_file() else "absent"
        digest.update(f"{key}\n{value}\n".encode())
    return digest.hexdigest()


def produced_by_now(
    study: str,
    inputs: dict[str, str] | None = None,
    driver_commit: str | None = None,
    depends_on: list[str] | None = None,
) -> dict[str, Any]:
    """The stamp for a run produced by the current tree.

    Args:
        study: Study package name.
        inputs: Label -> sha256 of each input the run read, when known.
        driver_commit: Commit of a driver repository outside this package.
        depends_on: Paths relative to the ``replica`` package whose content
            the run depends on; ``None`` covers the study's footprint files and
            an empty list marks a run no package file produced.
    """
    root = SRC_DIR.parent
    status = git_output(root, "status", "--porcelain", "--", "src/replica")
    stamp: dict[str, Any] = {
        "commit": git_output(root, "rev-parse", "HEAD"),
        "dirty": None if status is None else bool(status),
        "produced_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "footprint": code_footprint(study, depends_on=depends_on),
        "inputs": dict(inputs or {}),
    }
    if driver_commit:
        stamp["driver_commit"] = driver_commit
    if depends_on is not None:
        stamp["depends_on"] = sorted(depends_on)
    return stamp


def stamp(
    study: str,
    run: str,
    produced_by: dict[str, Any],
    *,
    validate: bool = False,
    replace: bool = False,
) -> None:
    """Record what produced a run in the study's index.

    ``validate=True`` records a source assessment separately from the producer.
    It does not record a scientific rerun or establish numerical agreement.

    Args:
        study: Study package name.
        run: Run path relative to the study's ``results/``.
        produced_by: The stamp, as :func:`produced_by_now` builds it.
        validate: Record assessed source and evidence, keeping the producer.
        replace: Replace a recorded producer; otherwise only a run whose
            producer is ``null`` takes one.
    """
    if validate and replace:
        raise ValueError("validate and replace are mutually exclusive")
    document = read_index(study)
    if run not in document["runs"]:
        raise FileNotFoundError(f"{study}: no run {run!r} in {INDEX}")
    entry = document["runs"][run]
    problems = _stamp_problems(produced_by, recording=True)
    if problems:
        raise ValueError(f"{study}/{run}: {'; '.join(problems)}")
    if validate:
        if not entry.get("produced_by"):
            raise ValueError(
                f"{study}/{run}: validation requires an existing producer record"
            )
        entry["validated_by"] = produced_by
    else:
        if entry.get("produced_by") and not replace:
            raise ValueError(
                f"{study}/{run} already records its producer; validate it with "
                "--validate, or replace the record with --replace-producer"
            )
        entry["produced_by"] = produced_by
        entry.pop("validated_by", None)
    write_index(study, document)


def _stamp_problems(value: Any, *, recording: bool = False) -> list[str]:
    # A stamp being recorded must list depends_on paths relative to the
    # replica package; records already in an index keep the paths they hold.
    if not isinstance(value, dict):
        return ["produced_by must be an object or null"]
    problems = [f"produced_by lacks {key}" for key in sorted(_STAMP_KEYS - set(value))]
    problems += [
        f"produced_by has unknown key {key}"
        for key in sorted(set(value) - _STAMP_KEYS - _OPTIONAL_STAMP_KEYS)
    ]
    if not _DIGEST.match(str(value.get("footprint", ""))):
        problems.append("footprint is not a sha256 digest")
    if not isinstance(value.get("inputs", {}), dict) or any(
        not _DIGEST.match(str(v)) for v in value.get("inputs", {}).values()
    ):
        problems.append("inputs must map labels to sha256 digests")
    depends_on = value.get("depends_on")
    if depends_on is not None and (
        not isinstance(depends_on, list)
        or any(not isinstance(v, str) or not v for v in depends_on)
    ):
        problems.append("depends_on must be a list of package paths")
    elif depends_on is not None and recording:
        problems += [
            f"depends_on path {v!r} is not relative to the replica package"
            for v in depends_on
            if not _resource_path(v) or v.split("/")[0] in {"src", "replica"}
        ]
    return problems


def _manifest_problems(results: Path, run: str, entry: dict[str, Any]) -> list[str]:
    problems = []
    identities = entry.get("files")
    if not isinstance(identities, dict) or not identities:
        problems.append(f"{run}: missing file manifest")
        identities = {}
    for name, digest in identities.items():
        if (
            not isinstance(name, str)
            or Path(name).is_absolute()
            or any(p in {".", "..", ""} for p in name.split("/"))
        ):
            problems.append(f"{run}: invalid file path {name!r}")
            continue
        if not _DIGEST.fullmatch(str(digest)):
            problems.append(f"{run}: invalid file digest for {name}")
        local = results / run / name
        if local.is_file():
            if local.is_symlink() or _sha256(local) != digest:
                problems.append(f"{run}: {name} differs from the file manifest")
        elif (entry.get("hub") or {}).get(name) != digest:
            problems.append(f"{run}: {name} has no matching checkout or Hub file")
    for name, digest in (entry.get("hub") or {}).items():
        if identities.get(name) != digest:
            problems.append(f"{run}: Hub file {name} differs from the file manifest")
    return problems


def _entry_problems(results: Path, run: str, entry: dict[str, Any]) -> list[str]:
    problems = []
    try:
        parse_run(run)
    except ValueError as exc:
        problems.append(str(exc))
    if run.startswith("references/") != (entry.get("role") == "published"):
        problems.append(f"{run}: reference role and placement disagree")
    if (
        not isinstance(entry.get("description"), str)
        or not entry["description"].strip()
    ):
        problems.append(f"{run}: missing description")
    problems.extend(_manifest_problems(results, run, entry))
    for key in sorted(set(entry) - _ENTRY_KEYS):
        problems.append(f"{run}: unknown key {key}")
    if entry.get("role") not in ROLES:
        problems.append(f"{run}: role {entry.get('role')!r}")
    if "produced_by" not in entry:
        problems.append(f"{run}: no produced_by (null marks an unstamped run)")
    elif entry["produced_by"] is not None:
        problems += [f"{run}: {p}" for p in _stamp_problems(entry["produced_by"])]
    validation = entry.get("validated_by")
    if validation is not None:
        if not entry.get("produced_by"):
            problems.append(f"{run}: validation requires an existing producer record")
        problems += [f"{run}: {p}" for p in _stamp_problems(validation)]
    for name, digest in (entry.get("hub") or {}).items():
        if not _DIGEST.match(str(digest)):
            problems.append(f"{run}: hub digest of {name} is not sha256")
        local = results / run / name
        if local.is_file() and _sha256(local) != digest:
            problems.append(f"{run}: {name} differs between checkout and hub")
    return problems


def validate_index(study: str) -> list[str]:
    """Everything wrong with a study's index, as messages; empty when valid.

    Args:
        study: Study package name.
    """
    results = STUDIES_DIR / study / "results"
    document = read_index(study)
    problems = validate_package(study)
    if document.get("schema") != SCHEMA:
        problems.append(f"schema is {document.get('schema')!r}, expected {SCHEMA}")
    if "hub_repo" in document and (
        not isinstance(document["hub_repo"], str)
        or not _REPO_ID.fullmatch(document["hub_repo"])
    ):
        problems.append("hub_repo must be an owner/repository identifier")
    revision = document.get("hub_revision")
    if revision is not None and not re.fullmatch(r"[0-9a-f]{40}", str(revision)):
        problems.append("hub_revision must be an immutable commit")
    entries = document.get("runs", {})
    tree = set(runs(results))
    for run in sorted(tree - set(entries)):
        problems.append(f"{run}: only in the tree")
    for run in sorted(set(entries) - tree):
        if not entries[run].get("hub"):
            problems.append(f"{run}: only in {INDEX} and not on the hub")
    problems.extend(description_conflicts(entries))
    for run, entry in entries.items():
        problems.extend(_entry_problems(results, run, entry))
    for path in results.rglob("*"):
        if not path.is_file() or path in {results / INDEX, results / "README.md"}:
            continue
        relative = path.relative_to(results).as_posix()
        owners = [
            run
            for run, entry in entries.items()
            if relative.startswith(run + "/")
            and relative[len(run) + 1 :] in (entry.get("files") or {})
        ]
        if len(owners) != 1:
            problems.append(
                f"{relative}: file must belong to exactly one result record"
            )
    return problems


def validate_package(
    study: str, paths: list[str] | None = None, *, package: Path | None = None
) -> list[str]:
    """Check package resources and archive bindings outside result records.

    Args:
        study: Study package name.
        paths: Package-relative tracked paths; defaults to Git's file listing.
        package: Study directory in an alternate checkout.
    """
    package = package or STUDIES_DIR / study
    index = package / "results" / INDEX
    document = json.loads(index.read_text()) if index.is_file() else {}
    if paths is None:
        tracked = git_output(package, "ls-files", "-z", ".")
        paths = (
            tracked.split("\0")
            if tracked is not None
            else [
                p.relative_to(package).as_posix()
                for p in package.rglob("*")
                if p.is_file()
            ]
        )
    problems = []
    resources = document.get("resources", {})
    if not isinstance(resources, dict):
        return ["resources must be an object"]
    for name in filter(None, paths):
        path = Path(name)
        if "__pycache__" in path.parts:
            continue
        if path.parts[0] == "results":
            if not document:
                problems.append("results directory requires an index")
            continue
        if path.suffix in {".py", ".R", ".r", ".sh", ".patch"}:
            continue
        if name in _METADATA_FILES:
            continue
        if path.suffix == ".md":
            if re.search(
                r"(?:^|[_-])(?:WIP|STATUS|HANDOFF|PLAN)(?:[_-]|$)|\d{8}",
                path.stem,
                re.I,
            ):
                problems.append(
                    f"{name}: operational document outside source documentation"
                )
            continue
        if name not in resources:
            problems.append(
                f"{name}: artifact outside indexed results or declared resources"
            )
    problems.extend(
        _resource_problems(
            study, package, resources, paths, document.get("archives", {})
        )
    )
    problems.extend(_archive_problems(study, document.get("archives", {})))
    return problems


def _resource_problems(
    study: str, package: Path, resources: dict, paths: list[str], archives: Any
) -> list[str]:
    """Check declared input and figure identities against the tracked files."""
    problems = []
    for name, resource in resources.items():
        if not _resource_path(name) or name.startswith("results/"):
            problems.append(f"{name}: invalid resource path")
            continue
        path = package / name
        if not isinstance(resource, dict):
            problems.append(f"{name}: resource must be an object")
            continue
        if resource.get("role") not in {"input", "figure"} or not resource.get(
            "description"
        ):
            problems.append(
                f"{name}: resource requires an input/figure role and description"
            )
        if resource.get("role") == "figure" and path.suffix not in {
            ".png",
            ".jpg",
            ".jpeg",
            ".svg",
            ".pdf",
        }:
            problems.append(f"{name}: figure resource must be an image")
        digest = resource.get("sha256")
        if not isinstance(digest, str) or not _DIGEST.fullmatch(digest):
            problems.append(f"{name}: resource requires a SHA-256 digest")
        archive = resource.get("archive")
        if archive is not None:
            binding = (
                archives.get(archive)
                if isinstance(archives, dict) and isinstance(archive, str)
                else None
            )
            records = binding.get("records", {}) if isinstance(binding, dict) else {}
            if not isinstance(records, dict) or records.get(name) != digest:
                problems.append(
                    f"{name}: resource lacks an exact archive member binding"
                )
            reference = (
                binding.get("reference", {}) if isinstance(binding, dict) else {}
            )
            if (
                not isinstance(reference, dict)
                or reference.get("path") != f"bundles/{study}/{archive}.tar.gz"
            ):
                problems.append(f"{name}: resource requires a study-bundle archive")
        if (archive is None or path.exists() or path.is_symlink()) and (
            not path.is_file() or path.is_symlink() or _sha256(path) != digest
        ):
            problems.append(f"{name}: resource differs from its declared digest")
        if archive is None and name not in paths:
            problems.append(f"{name}: declared resource is not tracked")
    return problems


def _resource_path(name: str) -> bool:
    return (
        isinstance(name, str)
        and bool(name)
        and "\\" not in name
        and not Path(name).is_absolute()
        and all(p not in {"", ".", ".."} for p in name.split("/"))
    )


def _archive_problems(study: str, archives: Any) -> list[str]:
    if not isinstance(archives, dict):
        return ["archives must be an object"]
    problems = []
    for digest, entry in archives.items():
        if not isinstance(entry, dict) or not entry.get("description"):
            problems.append(f"{digest}: archive requires a description")
            continue
        if not isinstance(entry.get("verification_required", False), bool):
            problems.append(f"{digest}: verification_required must be a boolean")
        ref = entry.get("reference", {})
        if (
            not isinstance(ref, dict)
            or not _DIGEST.fullmatch(str(digest))
            or ref.get("sha256") != digest
            or not re.fullmatch(r"[0-9a-f]{40}", str(ref.get("revision", "")))
            or not isinstance(ref.get("repo_id"), str)
            or not _REPO_ID.fullmatch(ref["repo_id"])
            or ref.get("study", study) != study
            or isinstance(ref.get("bytes"), bool)
            or not isinstance(ref.get("bytes"), int)
            or ref["bytes"] <= 0
            or ref.get("path")
            not in {f"bundles/{study}/{digest}.tar.gz", f"provenance/{digest}.tar.gz"}
        ):
            problems.append(f"{digest}: invalid immutable archive reference")
        records = entry.get("records", {})
        if not isinstance(records, dict) or any(
            not _resource_path(name) or not _DIGEST.fullmatch(str(value))
            for name, value in records.items()
        ):
            problems.append(f"{digest}: invalid archive member records")
    return problems


def unstamped_runs(study: str) -> list[str]:
    """Runs whose ``produced_by`` is ``null``.

    Args:
        study: Study package name.
    """
    entries = read_index(study)["runs"]
    return [run for run, entry in entries.items() if entry.get("produced_by") is None]


def stale_runs(
    study: str, digests: dict[Path, str] | None = None
) -> list[tuple[str, str]]:
    """Stamped runs whose footprint differs from the current tree.

    Args:
        study: Study package name.
        digests: File digests shared across studies checked in one call.

    Returns:
        ``(run, message)`` pairs naming the commit of the record compared:
        the run's validation when it has one, else its producer.
    """
    stamped = {
        run: ("validated", entry["validated_by"])
        if entry.get("validated_by")
        else ("produced", entry["produced_by"])
        for run, entry in read_index(study)["runs"].items()
        if entry.get("produced_by")
    }
    if not stamped:
        return []
    current: dict[tuple[str, ...] | None, str] = {}
    stale = []
    for run, (record, recorded) in stamped.items():
        depends_on = recorded.get("depends_on")
        key = tuple(depends_on) if depends_on is not None else None
        if key not in current:
            current[key] = code_footprint(study, digests, depends_on)
        if recorded["footprint"] != current[key]:
            commit = (recorded.get("commit") or "unknown commit")[:12]
            stale.append((run, f"{record} at {commit} on {recorded['produced_at']}"))
    return stale
