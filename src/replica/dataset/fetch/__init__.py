"""Orchestrate dataset downloads and verify acquired files.

The ``replica fetch`` / ``replica datasets`` console commands
(:mod:`replica.cli`) drive this module.
"""

import sys
from collections.abc import Callable
from functools import partial
from pathlib import Path

from replica.dataset.catalog.registry import data_dir as _data_dir
from replica.dataset.metadata.generate import generate_yaml

from ...utils.transfer import ensure_ssl_cert_chain
from ..catalog.sources import DATASETS, dataset_directory, resolve_dataset_name
from .sources import (
    download_direct_url,
    download_github_raw,
    download_osf,
    download_osf_zip,
    download_zip,
    link_study_release,
)


def print_manual_instructions(config: dict) -> None:
    """Print the access steps for a dataset that cannot be downloaded.

    Args:
        config: Catalog entry with ``source`` ``"manual"``; its
            ``instructions`` text tells the user how to obtain the files.
    """
    print(f"\n{config['instructions']}\n")


# ---------------------------------------------------------------------------
# Fetch logic
# ---------------------------------------------------------------------------


def fetch_dataset(
    name: str, *, force: bool = False, regenerate_yaml: bool = False
) -> None:
    """Fetch a single dataset by name.

    By default the download is verified against the shipped checksums.
    ``regenerate_yaml=True``, the maintainer path, rewrites that baseline from
    the download instead; it is off by default because a bad download would
    otherwise be baselined as correct.

    Args:
        name: Catalog key or fetch alias of the dataset.
        force: Re-download files that are already on disk instead of keeping them.
        regenerate_yaml: Rewrite the in-package dataset YAML from the downloaded
            files, replacing the shipped checksums, instead of verifying against
            them.
    """
    name = resolve_dataset_name(name)
    config = DATASETS[name]
    directory = _data_dir() / dataset_directory(name)

    print(f"\n=== {name} ===")

    if config["source"] == "manual":
        print_manual_instructions(config)
        return

    ensure_ssl_cert_chain()
    directory.mkdir(parents=True, exist_ok=True)
    _fetch_dataset_source(config, directory, force=force)
    if regenerate_yaml:
        _regenerate_yaml(directory, config)
    elif config.get("verify_yaml", True):
        _verify_download(directory)
    _verify_config_checksums(directory, config)


def _verify_config_checksums(directory: Path, config: dict) -> None:
    """Verify optional download-specific SHA-256 pins outside dataset YAML."""
    from replica.dataset.metadata.io import ChecksumMismatchError, compute_sha256

    checksums = config.get("sha256", {})
    for filename, expected_hex in checksums.items():
        actual = compute_sha256(directory / filename)
        expected = f"sha256:{expected_hex}"
        if actual != expected:
            raise ChecksumMismatchError(filename, expected, actual)
    if checksums:
        print(f"  verified {len(checksums)} source file(s) against SHA-256 pins")


def _regenerate_yaml(directory: Path, config: dict) -> None:
    """Maintainer step: recompute the shipped YAML checksum baseline from the
    files in *directory* and write it into the in-package YAML tree. Errors
    propagate (a maintainer wants to know the refresh failed).
    """
    if not config.get("regenerate_yaml", True):
        print("  (dataset.yaml shipped with archive; not regenerating)")
        return
    print("  Updating dataset.yaml ...")
    generate_yaml(directory)


def _verify_download(directory: Path) -> None:
    """Verify the downloaded files against the shipped checksums.

    Raises on a mismatch, a corrupt or truncated download.
    """
    from replica.dataset.metadata.io import (
        find_dataset_yaml,
        load_yaml,
        verify_checksums,
    )

    yaml_path = find_dataset_yaml(directory)
    if yaml_path is None:
        print("  (no shipped checksums to verify against)")
        return
    info = load_yaml(yaml_path)
    pinned = [entry for entry in info.files if entry.checksum]
    if not pinned:
        print("  (no shipped checksums to verify against)")
        return
    verify_checksums(directory, pinned)
    print(f"  verified {len(pinned)} manifest entry/entries against shipped checksums")
    unpinned = len(info.files) - len(pinned)
    if unpinned:
        print(
            f"  {unpinned} manifest entry/entries have no checksum; content not verified"
        )


def _fetch_dataset_source(config: dict, directory: Path, *, force: bool) -> None:
    if config["source"] == "osf":
        _fetch_osf_files(config, directory, force=force)

    elif config["source"] == "github_raw":
        _fetch_github_raw_files(config, directory, force=force)

    elif config["source"] == "osf_zip":
        _fetch_osf_zip_files(config, directory, force=force)

    elif config["source"] == "zip":
        _fetch_zip_files(config, directory, force=force)

    elif config["source"] == "direct_url":
        _fetch_direct_url_files(config, directory, force=force)

    elif config["source"] == "study_release":
        _fetch_study_release_files(config, directory, force=force)


def _download_if_needed(
    label: str, out: Path, force: bool, downloader: Callable[[], None]
) -> None:
    if out.exists() and not force:
        print(f"  {label} already exists (use --force to re-download)")
        return
    downloader()


def _fetch_osf_files(config: dict, directory: Path, *, force: bool) -> None:
    for file_id, filename in config["files"]:
        out = directory / filename
        out.parent.mkdir(parents=True, exist_ok=True)
        _download_if_needed(
            filename,
            out,
            force,
            partial(download_osf, file_id, out),
        )


def _fetch_github_raw_files(config: dict, directory: Path, *, force: bool) -> None:
    repo = config["repo"]
    branch = config.get("revision", config["branch"])
    for remote_path, filename in config["files"]:
        out = directory / filename
        _download_if_needed(
            filename,
            out,
            force,
            partial(download_github_raw, repo, branch, remote_path, out),
        )


def _fetch_osf_zip_files(config: dict, directory: Path, *, force: bool) -> None:
    for file_id, filename in config["files"]:
        # Extracted member names are unknown a priori; a per-ZIP marker keeps
        # re-runs idempotent (skip unless force, like the other sources).
        marker = directory / f".{filename}.extracted"
        if marker.exists() and not force:
            print(f"  {filename} already extracted (use --force to re-download)")
            continue
        download_osf_zip(file_id, filename, directory)
        marker.touch()
    _fetch_extra_osf_files(config, directory, force=force)
    for url, filename in config.get("extra_urls", []):
        out = directory / filename
        out.parent.mkdir(parents=True, exist_ok=True)
        _download_if_needed(
            filename, out, force, partial(download_direct_url, url, out)
        )


def _fetch_extra_osf_files(config: dict, directory: Path, *, force: bool) -> None:
    for file_id, rel_path in config.get("extra_osf_files", []):
        out = directory / rel_path
        out.parent.mkdir(parents=True, exist_ok=True)
        _download_if_needed(
            rel_path,
            out,
            force,
            partial(download_osf, file_id, out),
        )


def _fetch_zip_files(config: dict, directory: Path, *, force: bool) -> None:
    needed = {}
    for zip_name, local_name in config["files"].items():
        out = directory / local_name
        if out.exists() and not force:
            print(f"  {local_name} already exists (use --force to re-download)")
        else:
            needed[zip_name] = local_name
    if needed:
        download_zip(config["url"], directory, needed)


def _fetch_study_release_files(config: dict, directory: Path, *, force: bool) -> None:
    """Take the corpus out of a study release, fetching that release once."""
    repo = config["repo"]
    for member, filename in config["files"]:
        out = directory / filename
        _download_if_needed(
            filename,
            out,
            force,
            partial(link_study_release, repo, member, out),
        )


def _fetch_direct_url_files(config: dict, directory: Path, *, force: bool) -> None:
    for url, filename in config["files"]:
        out = directory / filename
        _download_if_needed(
            filename,
            out,
            force,
            partial(download_direct_url, url, out),
        )


def _file_census(config: dict, directory: Path) -> dict[str, int]:
    """Count the files a fetch of this entry verifies, and those on disk.

    These are the dataset YAML's file entries, or the entry's own SHA-256 pins
    when it verifies outside the YAML.
    """
    from replica.dataset.metadata.io import file_presence, find_dataset_yaml, load_yaml

    entries = [{"path": name} for name in config.get("sha256", {})]
    yaml_path = find_dataset_yaml(directory)
    if config.get("verify_yaml", True) and yaml_path is not None:
        entries += [{"path": entry.path} for entry in load_yaml(yaml_path).files]
    return file_presence(directory, entries)


def list_datasets() -> None:
    """Print status of all datasets, with next-step hints."""
    print("\nDatasets (status, next step):\n")
    for name, config in DATASETS.items():
        directory = _data_dir() / dataset_directory(name)

        if config["source"] == "manual":
            status = "[manual]"
            hint = f"run `replica fetch {name}` for access steps"
        elif not directory.exists():
            status = "[not downloaded]"
            hint = f"run `replica fetch {name}`"
        else:
            census = _file_census(config, directory)
            if census["declared"] and census["present"] == census["declared"]:
                status = "[downloaded]"
                hint = f"path: {dataset_directory(name)}"
            elif census["present"]:
                status = "[partial]"
                hint = f"run `replica fetch {name}`"
            else:
                status = "[not downloaded]"
                hint = f"run `replica fetch {name}`"

        print(f"  {name:20s} {status:18s} {hint}")
    print()


# ---------------------------------------------------------------------------
# Direct module entry point. The installed command delegates to the same
# argument surface in :mod:`replica.cli`.
# ---------------------------------------------------------------------------


def run_fetch(
    *,
    names: list[str],
    fetch_all: bool = False,
    list_only: bool = False,
    force: bool = False,
    regenerate_yaml: bool = False,
    on_empty: Callable[[], None] | None = None,
) -> None:
    """Shared fetch driver for the CLI.

    Lists if *list_only*; else fetches *names* (or every dataset if
    *fetch_all*). *regenerate_yaml* is the maintainer baseline-refresh path.
    Calls *on_empty* and exits non-zero when neither names nor *fetch_all*
    is given.

    Args:
        names: Catalog keys or fetch aliases to fetch, in order.
        fetch_all: Fetch every catalog entry that opts into bulk fetching,
            ignoring *names*.
        list_only: Print the status of every dataset and return without fetching.
        force: Re-download files that are already on disk.
        regenerate_yaml: Rewrite the shipped checksum baseline from the download
            rather than verifying against it.
        on_empty: Called when there is nothing to fetch, to show usage before
            the non-zero exit; ``None`` exits silently.
    """
    if list_only:
        list_datasets()
        return

    if fetch_all:
        names = [
            name
            for name, config in DATASETS.items()
            if config.get("include_in_all", True)
        ]
    elif not names:
        if on_empty is not None:
            on_empty()
        sys.exit(1)

    unknown = [n for n in names if resolve_dataset_name(n) not in DATASETS]
    if unknown:
        sys.exit(
            f"unknown dataset(s): {', '.join(unknown)}\n"
            f"available: {', '.join(DATASETS.keys())}"
        )

    for name in names:
        fetch_dataset(name, force=force, regenerate_yaml=regenerate_yaml)

    print("\nDone.")
