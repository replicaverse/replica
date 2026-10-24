"""Pinned predictor repository reads and auxiliary reproduction inputs."""

import functools
import logging
import os
import re
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download, try_to_load_from_cache

log = logging.getLogger("replica.predictors")
HF_REPO = os.environ.get("PSYCHO_PREDICTORS_HF_REPO", "replicaverse/replica-predictors")
HF_REVISION = os.environ.get("PSYCHO_PREDICTORS_HF_REVISION", "main")
_COMMIT_RE = re.compile(r"[0-9a-f]{40}")


def resolve_store_revision(repo: str, ref: str = "main") -> str:
    """Resolve a ref of a predictor store repository to a commit id.

    A commit id resolves to itself. In offline mode the snapshot holding the
    cached index names the commit, so an uncached index raises OSError.

    Args:
        repo: Dataset repository id.
        ref: Branch, tag or commit id.
    """
    if _COMMIT_RE.fullmatch(ref):
        revision = ref
    elif os.environ.get("HF_HUB_OFFLINE", "").upper() in {"1", "ON", "YES", "TRUE"}:
        cached = try_to_load_from_cache(
            repo, "tags.parquet", revision=ref, repo_type="dataset"
        )
        if not isinstance(cached, str):
            raise OSError(
                f"offline and the index of {repo} at {ref} is not cached; pass a "
                "commit revision or go online"
            )
        revision = Path(cached).parent.name
    else:
        revision = HfApi().repo_info(repo, repo_type="dataset", revision=ref).sha
    if not revision or not _COMMIT_RE.fullmatch(revision):
        raise ValueError(f"{repo} did not resolve to a commit: {revision!r}")
    return revision


@functools.lru_cache(maxsize=1)
def _store_revision() -> str:
    """Resolve the configured store to one commit for all reads in this process.

    Set PSYCHO_PREDICTORS_HF_REVISION to a commit for reproducible runs.
    """
    revision = resolve_store_revision(HF_REPO, HF_REVISION)
    log.info("Predictor store %s at %s", HF_REPO, revision)
    return revision


def predictor_store_identity() -> dict[str, str]:
    """Return a fresh report of the immutable store selected for this process."""
    return {"repo_id": HF_REPO, "revision": _store_revision()}


def _hub_file(filename: str) -> str:
    identity = predictor_store_identity()
    return hf_hub_download(
        identity["repo_id"],
        filename,
        repo_type="dataset",
        revision=identity["revision"],
    )


def fetch_repro_asset(hf_filename: str, *, env_var: str | None = None) -> str | None:
    """Resolve a large reproduction input to a local file path.

    Returns ``$env_var`` if that variable is set, otherwise downloads
    ``hf_filename`` from the predictor HF repo (:data:`HF_REPO`), cached by
    ``huggingface_hub``. Returns ``None`` when neither is available, letting
    callers fall back (to wordfreq, or to skipping an ARPA overlay).

    Args:
        hf_filename: Path of the asset within the predictor HF dataset repo.
        env_var: Environment variable that, when set, is returned directly
            instead of downloading the asset from the Hub.
    """
    if env_var and os.environ.get(env_var):
        return os.environ[env_var]
    try:
        return _hub_file(hf_filename)
    except OSError as e:
        log.warning(
            "reproduction asset %s unavailable from %s (%s); set %s to a local path",
            hf_filename,
            HF_REPO,
            e,
            env_var or "an override env var",
        )
        return None
