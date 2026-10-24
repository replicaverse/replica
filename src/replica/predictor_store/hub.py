"""Read access to the published store: its revision, index and tables."""

from __future__ import annotations

import re
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from replica.external.hub import HF_REPO as DEFAULT_REPO
from replica.external.hub import resolve_store_revision

from .layout import INDEX_FILE

_COMMIT_RE = re.compile(r"[0-9a-f]{40}")


class Hub:
    """The published store at one commit.

    Every download goes through the same resolved commit, so a carried row
    and the index it is checked against come from one revision.
    """

    def __init__(self, repo: str = DEFAULT_REPO, revision: str | None = None):
        """Bind to a repository, resolving ``main`` to a commit on first use.

        Args:
            repo: Dataset repository id.
            revision: Commit to read; ``None`` resolves the head of ``main``,
                or the cached snapshot when the hub is offline.
        """
        self.repo = repo
        self._revision = revision
        self._index: pa.Table | None = None
        self._files: set[str] | None = None

    def revision(self) -> str:
        """The commit every read of this object comes from."""
        if self._revision is None:
            self._revision = resolve_store_revision(self.repo)
        if not _COMMIT_RE.fullmatch(self._revision):
            raise ValueError(
                f"{self.repo} did not resolve to a commit: {self._revision!r}"
            )
        return self._revision

    def download(self, filename: str) -> Path:
        """Local path of one published file at the bound revision.

        Args:
            filename: Path within the repository.
        """
        from huggingface_hub import hf_hub_download

        return Path(
            hf_hub_download(
                self.repo, filename, repo_type="dataset", revision=self.revision()
            )
        )

    def index(self) -> pa.Table:
        """The published ``tags.parquet``."""
        if self._index is None:
            self._index = pq.read_table(self.download(INDEX_FILE))
        return self._index

    def files(self) -> set[str]:
        """Every file the repository holds at the bound revision.

        Only the hub's listing says which files exist: the index omits a table
        published without an index row, and the local cache omits every file
        not downloaded before, so a failed or offline listing raises OSError.
        """
        if self._files is None:
            from huggingface_hub import HfApi

            revision = self.revision()
            try:
                entries = HfApi().list_repo_tree(
                    self.repo, repo_type="dataset", recursive=True, revision=revision
                )
                self._files = {
                    entry.path
                    for entry in entries
                    if getattr(entry, "blob_id", None) is not None
                }
            except OSError as exc:
                raise OSError(
                    f"cannot list the files of {self.repo} at {revision}: {exc}. "
                    "A merge needs the hub's complete listing; retry online."
                ) from exc
        return self._files
