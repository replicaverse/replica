"""Upload a staged tree to the hub and refresh the file history.

The upload is one commit built by ``upload_folder``: it adds the files the
export wrote and deletes the published files whose cell the export replaced
without rewriting them. ``history.parquet`` is rebuilt afterwards, since the
publishing commit exists only then, and committed on the same branch.
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from . import layout
from ._readback import _copy_verified


def file_history(api: Any, repo: str, revision: str | None = None) -> list[dict]:
    """First-publish and last-change commit of every path a repository ever held.

    Walks the commits oldest first and lists the tree at each one.

    Args:
        api: A ``huggingface_hub.HfApi`` or an object with the same
            ``list_repo_commits`` and ``list_repo_tree`` methods.
        repo: Dataset repository id.
        revision: Branch or pull-request ref whose history to walk; the
            default branch when ``None``.
    """
    commits = sorted(
        api.list_repo_commits(repo, repo_type="dataset", revision=revision),
        key=lambda c: c.created_at,
    )
    first: dict[str, tuple[str, str]] = {}
    last: dict[str, tuple[str, str]] = {}
    digest_of: dict[str, str] = {}
    present: set[str] = set()
    for commit in commits:
        present = set()
        entries = api.list_repo_tree(
            repo, repo_type="dataset", recursive=True, revision=commit.commit_id
        )
        when = commit.created_at.isoformat(timespec="seconds")
        for entry in entries:
            if getattr(entry, "blob_id", None) is None:
                continue
            lfs = getattr(entry, "lfs", None)
            digest = getattr(lfs, "sha256", None) or entry.blob_id
            first.setdefault(entry.path, (commit.commit_id, when))
            if digest_of.get(entry.path) != digest:
                last[entry.path] = (commit.commit_id, when)
                digest_of[entry.path] = digest
            present.add(entry.path)
    return [
        {
            "file": path,
            "published_in": first[path][0],
            "published_at": first[path][1],
            "last_modified_in": last[path][0],
            "last_modified_at": last[path][1],
            "present": path in present,
        }
        for path in sorted(first)
    ]


def publish(
    stage: Path,
    repo: str,
    *,
    create_pr: bool = True,
    message: str | None = None,
    allow_failures: bool = False,
    initial_release: bool = False,
    api: Any = None,
) -> dict:
    """Upload a verified snapshot and refresh ``history.parquet``.

    A matching successful readback receipt is required before any API call.
    The upload copies the receipt-bound files into temporary storage, requiring
    space for one additional stage; original source pickles are not needed.

    Args:
        stage: Tree written by the export, holding ``publish.json``.
        repo: Dataset repository id to upload to.
        create_pr: Open a pull request instead of committing to ``main``.
        message: Commit message; a summary of the release by default.
        allow_failures: Upload although the export reported cells it could
            not stage. Never bypasses readback verification.
        initial_release: Upload a stage exported without ``--merge-hub``,
            whose index and shared tables replace the repository's.
        api: A ``huggingface_hub.HfApi`` or a stand-in with the same
            ``upload_folder``, ``upload_file``, ``list_repo_commits`` and
            ``list_repo_tree`` methods.

    Returns:
        The commit id, the pull-request url and ref when one was opened, and
        the number of files in the refreshed history.

    Raises:
        ValueError: When the export left failures and *allow_failures* is
            false, when the stage names another repository, or when the stage
            has no base revision and *initial_release* is false, or when its
            readback receipt is missing, invalid or stale.
    """
    with TemporaryDirectory(prefix="replica-publish-") as directory:
        snapshot = Path(directory)
        plan = _copy_verified(stage, snapshot)
        if plan.get("failures") and not allow_failures:
            raise ValueError(
                f"the export left {len(plan['failures'])} failure(s); fix them or pass "
                "allow_failures"
            )
        if plan.get("repo") not in (None, repo):
            raise ValueError(f"the stage was merged against {plan['repo']}, not {repo}")
        if not plan.get("base_revision") and not initial_release:
            raise ValueError(
                "the stage was exported without --merge-hub, so its tags.parquet and "
                "shared tables hold only this run's cells and would replace the "
                "store's; export it with --merge-hub, or pass initial_release to "
                "publish it as the first release of a repository"
            )
        if api is None:
            from huggingface_hub import HfApi

            api = HfApi()
        if message is None:
            message = (
                f"Publish {len(plan['written'])} files, replacing "
                f"{len(plan.get('replaced_tags', []))} cells"
            )
        info = api.upload_folder(
            repo_id=repo,
            folder_path=str(snapshot),
            repo_type="dataset",
            commit_message=message,
            create_pr=create_pr,
            parent_commit=plan.get("base_revision"),
            allow_patterns=[
                *plan["written"],
                layout.PUBLISH_FILE,
                layout.READBACK_FILE,
            ],
            delete_patterns=list(plan["deleted"]) or None,
        )
    revision = getattr(info, "pr_revision", None) if create_pr else None
    rows = file_history(api, repo, revision)
    history = stage / layout.HISTORY_FILE
    pq.write_table(
        pa.Table.from_pylist(rows, schema=layout.HISTORY_SCHEMA),
        history,
        compression="zstd",
    )
    api.upload_file(
        path_or_fileobj=str(history),
        path_in_repo=layout.HISTORY_FILE,
        repo_id=repo,
        repo_type="dataset",
        revision=revision,
        commit_message="Refresh the publish history",
    )
    return {
        "commit": getattr(info, "oid", None),
        "pr_url": getattr(info, "pr_url", None),
        "pr_revision": revision,
        "history_files": len(rows),
    }
