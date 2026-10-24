#!/usr/bin/env python3
"""Validate dataset metadata, provenance fields, and available file checksums.

The optional --check-urls flag also checks upstream URL availability.
"""

import argparse
import re
import ssl
import sys
from pathlib import Path
from urllib.request import Request, urlopen

from replica.dataset.catalog.registry import data_dir
from replica.dataset.metadata.io import (
    _expand_glob,
    compute_aggregate_sha256,
    compute_sha256,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_YAML_DIR = REPO_ROOT / "src" / "replica" / "dataset" / "metadata" / "yaml"

# ---------------------------------------------------------------------------
# YAML loading (avoid dependency on DatasetInfo round-trip)
# ---------------------------------------------------------------------------


def load_raw_yaml(path: Path) -> dict:
    """Load a dataset.yaml as a raw dict.

    Args:
        path: The metadata YAML to read, one dataset per file.

    Raises:
        ValueError: The file parses to nothing.
    """
    import yaml

    with open(path) as f:
        data = yaml.safe_load(f)
    if data is None:
        raise ValueError(f"Empty YAML: {path}")
    return data


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

REQUIRED_FIELDS = ["name", "citation", "bibtex", "paradigm", "description"]

#: The stimulus-type vocabulary documented on ``DatasetInfo.stimulus_type``.
VALID_STIMULUS_TYPES = {"single-unit", "phrase", "sentence", "paragraph", "passage"}
#: The availability vocabulary documented on ``DatasetInfo.availability``.
VALID_AVAILABILITY = {"free", "on request", "licence required"}
REQUIRED_ORIGINAL_FIELDS = ["download_url"]  # only for original datasets

VALID_PARADIGMS = {
    "eye-tracking",
    "self-paced reading",
    "self-paced-reading",
    "maze",
    "eeg",
    "EEG",
    "fmri",
    # Language-model training corpora: provenance-only entries with no
    # registry loader, consumed by studies that train their own models.
    "lm",
}


def extract_bibtex_key(bibtex: str) -> str | None:
    """Extract the citation key from a bibtex entry.

    Args:
        bibtex: A bibtex entry, of which only the ``@type{key,`` header is read.

    Returns:
        The key, or ``None`` when no entry header is present.
    """
    m = re.search(r"@\w+\{([^,]+),", bibtex)
    return m.group(1).strip() if m else None


def check_bibtex_fields(bibtex: str) -> list[str]:
    """Check that bibtex has author, title, year.

    Args:
        bibtex: A bibtex entry, searched case-insensitively for each field name.

    Returns:
        One message per absent field; empty when all three are present.
    """
    errors = []
    for field in ("author", "title", "year"):
        if not re.search(rf"{field}\s*=", bibtex, re.IGNORECASE):
            errors.append(f"bibtex missing '{field}' field")
    return errors


def check_url_reachable(url: str, timeout: int = 10) -> str | None:
    """Return error string if URL is not reachable, None if ok.

    Tries HEAD first and falls back to GET, since some hosts reject HEAD. A
    value that does not start with ``http`` is treated as prose and skipped.

    Args:
        url: The address to request.
        timeout: Seconds to wait on each of the two attempts.
    """
    if not url.startswith("http"):
        return None  # skip non-URLs (e.g. "Contact authors ...")
    try:
        req = Request(
            url, method="HEAD", headers={"User-Agent": "dataset-yaml-verifier"}
        )
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        with urlopen(req, timeout=timeout, context=ctx) as resp:
            if resp.status >= 400:
                return f"HTTP {resp.status}"
    except Exception:
        # Try GET as fallback (some servers reject HEAD)
        try:
            req = Request(url, headers={"User-Agent": "dataset-yaml-verifier"})
            with urlopen(req, timeout=timeout, context=ctx) as resp:
                if resp.status >= 400:
                    return f"HTTP {resp.status}"
        except Exception as e2:
            return str(e2)[:100]
    return None


def verify_one(
    yaml_path: Path, *, check_urls: bool = False, data_root: Path | None = None
) -> tuple[list[str], list[str]]:
    """Verify a single dataset.yaml. Returns (errors, warnings).

    Args:
        yaml_path: Packaged metadata or a dataset-local YAML.
        data_root: Dataset root for packaged metadata; defaults to the loader
            registry's configured data directory. Local YAMLs use sibling files.
        check_urls: Also request ``url`` and ``download_url`` over the network.
    """
    errors: list[str] = []
    warnings: list[str] = []
    data = load_raw_yaml(yaml_path)
    dataset_dir = _dataset_directory(yaml_path, data, data_root)
    is_original = data.get("original", True)

    errors.extend(_required_field_errors(data, is_original=is_original))
    errors.extend(_citation_errors(data))
    errors.extend(_url_metadata_errors(data))
    if check_urls:
        errors.extend(_url_reachability_errors(data))
    errors.extend(_bibtex_quality_errors(data))
    errors.extend(_paradigm_errors(data))
    errors.extend(_stimulus_type_errors(data))
    errors.extend(_availability_errors(data))
    warnings.extend(_language_warnings(data))
    file_errors, file_warnings = _file_entry_findings(
        data.get("files", []), dataset_dir, is_original=is_original
    )
    errors.extend(file_errors)
    warnings.extend(file_warnings)

    return errors, warnings


def _dataset_directory(yaml_path: Path, data: dict, root: Path | None) -> Path:
    """Resolve packaged metadata to the same release layout as the loaders."""
    try:
        relative = yaml_path.resolve().relative_to(PACKAGE_YAML_DIR.resolve())
    except ValueError:
        return yaml_path.parent
    root = data_dir() if root is None else root.expanduser()
    if data.get("data_path"):
        return root / data["data_path"]
    stage = "original" if data.get("original", True) else "processed"
    return root / relative.parent / stage / relative.stem


def _required_field_errors(data: dict, *, is_original: bool) -> list[str]:
    required = REQUIRED_FIELDS[:]
    if is_original:
        required += REQUIRED_ORIGINAL_FIELDS
    errors = []
    for field in required:
        val = data.get(field, "")
        if not val or (isinstance(val, str) and not val.strip()):
            errors.append(f"missing or empty required field: {field}")
    return errors


def _citation_errors(data: dict) -> list[str]:
    citation = data.get("citation", "")
    bibtex = data.get("bibtex", "")
    if not citation or not bibtex:
        return []
    bib_key = extract_bibtex_key(bibtex)
    if bib_key and bib_key != citation:
        return [f"citation '{citation}' does not match bibtex key '{bib_key}'"]
    return []


def _url_metadata_errors(data: dict) -> list[str]:
    errors = []
    url = data.get("url", "")
    paper_url_patterns = ("doi.org", "arxiv.org", "aclanthology.org")
    if url and url.startswith("http") and not any(p in url for p in paper_url_patterns):
        errors.append(f"url should be a DOI/paper link, got: {url}")
    download_url = data.get("download_url", "")
    if download_url and "doi.org" in download_url:
        errors.append(f"download_url should not be a DOI, got: {download_url}")
    return errors


def _url_reachability_errors(data: dict) -> list[str]:
    errors = []
    for field in ("url", "download_url"):
        url = data.get(field, "")
        if not (url and url.startswith("http")):
            continue
        err = check_url_reachable(url)
        if err:
            errors.append(f"{field} unreachable: {err}")
    return errors


def _bibtex_quality_errors(data: dict) -> list[str]:
    bibtex = data.get("bibtex", "")
    return check_bibtex_fields(bibtex) if bibtex else []


def _paradigm_errors(data: dict) -> list[str]:
    paradigm = data.get("paradigm", "")
    if paradigm and paradigm not in VALID_PARADIGMS:
        return [f"unknown paradigm: '{paradigm}'"]
    return []


def _stimulus_type_errors(data: dict) -> list[str]:
    kind = data.get("stimulus_type", "")
    if kind and kind not in VALID_STIMULUS_TYPES:
        return [f"unknown stimulus_type: '{kind}'"]
    return []


def _availability_errors(data: dict) -> list[str]:
    availability = data.get("availability", "")
    if availability and availability not in VALID_AVAILABILITY:
        return [f"unknown availability: '{availability}'"]
    return []


def _language_warnings(data: dict) -> list[str]:
    """Warn when 'language' is undeclared: DatasetInfo.language defaults silently to 'en'."""
    lang = data.get("language", "")
    if not lang or (isinstance(lang, str) and not lang.strip()):
        return ["missing 'language' field (DatasetInfo defaults silently to 'en')"]
    return []


def _file_entry_findings(
    files: list[dict], dataset_dir: Path, *, is_original: bool
) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    if not files:
        warnings.append("no files declared; no checksums can be verified")
    for file_entry in files:
        entry_errors, entry_warnings = _file_entry_findings_one(
            file_entry, dataset_dir, is_original=is_original
        )
        errors.extend(entry_errors)
        warnings.extend(entry_warnings)
    return errors, warnings


def _file_entry_findings_one(
    file_entry: dict, dataset_dir: Path, *, is_original: bool
) -> tuple[list[str], list[str]]:
    path = file_entry.get("path", "")
    if not path:
        return ["file entry missing 'path'"], []

    errors: list[str] = []
    warnings: list[str] = []
    n_files = file_entry.get("n_files", 0)
    checksum = file_entry.get("checksum", "")
    original_path = file_entry.get("original_path", "")
    unavailable = file_entry.get("provenance_unavailable", "")
    has_unavailable_reason = isinstance(unavailable, str) and bool(unavailable.strip())
    if original_path and unavailable:
        errors.append(
            f"file '{path}' sets both original_path and provenance_unavailable"
        )
    if "provenance_unavailable" in file_entry and not has_unavailable_reason:
        errors.append(
            f"file '{path}' provenance_unavailable must be a non-empty reason"
        )
    if (
        is_original
        and n_files == 0
        and not original_path
        and "provenance_unavailable" not in file_entry
    ):
        errors.append(
            f"file '{path}' missing original_path or provenance_unavailable reason"
        )
    if not checksum:
        warnings.append(f"file '{path}' has no checksum; content is not verified")
    elif not (
        _expand_glob(dataset_dir, path)
        if n_files > 0
        else (dataset_dir / path).is_file()
    ):
        warnings.append(
            f"file '{path}' unavailable in {dataset_dir}; checksum not verified"
        )
    errors.extend(_checksum_errors(dataset_dir, path, n_files, checksum))
    return errors, warnings


def _checksum_errors(
    dataset_dir: Path, path: str, n_files: int, checksum: str
) -> list[str]:
    if not checksum:
        return []
    if n_files > 0:
        return _glob_checksum_errors(dataset_dir, path, n_files, checksum)
    actual_file = dataset_dir / path
    if not actual_file.exists():
        return []
    actual = compute_sha256(actual_file)
    if actual == checksum:
        return []
    return [
        f"checksum mismatch for {path}: "
        f"expected {checksum[:20]}..., got {actual[:20]}..."
    ]


def _glob_checksum_errors(
    dataset_dir: Path, path: str, n_files: int, checksum: str
) -> list[str]:
    matched = _expand_glob(dataset_dir, path)
    if not matched:
        return []
    if len(matched) != n_files:
        return [f"glob '{path}': matched {len(matched)} files, expected {n_files}"]
    actual = compute_aggregate_sha256(dataset_dir, matched)
    if actual == checksum:
        return []
    return [f"aggregate checksum mismatch for '{path}' ({n_files} files)"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def find_all_dataset_yamls() -> list[Path]:
    """Find every dataset YAML the repository tracks.

    Scans the in-package metadata tree at
    ``src/replica/dataset/metadata/yaml/{paradigm}/{name}.yaml``.
    """
    return sorted(PACKAGE_YAML_DIR.rglob("*.yaml"))


def main() -> None:
    """Verify the selected YAMLs, print per-file findings, exit 1 on any error.

    A missing path or an empty selection is an error, so a passing check has
    read at least one file.
    """
    parser = argparse.ArgumentParser(
        description="Verify dataset.yaml files for consistency"
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Specific dataset.yaml files to check (default: all)",
    )
    parser.add_argument(
        "--check-urls",
        action="store_true",
        help="Also verify that URLs are reachable (slow)",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        help="Dataset root (default: the loader registry data directory)",
    )
    args = parser.parse_args()

    yamls = args.paths if args.paths else find_all_dataset_yamls()
    missing = [str(path) for path in yamls if not path.is_file()]
    if missing:
        print(f"no such file: {', '.join(missing)}", file=sys.stderr)
        sys.exit(1)
    if not yamls:
        print(f"no dataset YAML found under {PACKAGE_YAML_DIR}", file=sys.stderr)
        sys.exit(1)

    total_errors = 0
    total_warnings = 0
    for yaml_path in yamls:
        try:
            rel = yaml_path.relative_to(REPO_ROOT)
        except ValueError:
            rel = yaml_path
        errors, warnings = verify_one(
            yaml_path, check_urls=args.check_urls, data_root=args.data_root
        )
        if errors:
            print(f"\nFAIL  {rel}")
            for e in errors:
                print(f"  - {e}")
            for w in warnings:
                print(f"  ⚠ {w}")
            total_errors += len(errors)
            total_warnings += len(warnings)
        elif warnings:
            print(f"  OK  {rel}  ({len(warnings)} warning(s))")
            for w in warnings:
                print(f"  ⚠ {w}")
            total_warnings += len(warnings)
        else:
            print(f"  OK  {rel}")

    print(f"\n{'=' * 60}")
    print(
        f"Checked {len(yamls)} files, "
        f"{total_errors} error(s), {total_warnings} warning(s)"
    )

    if total_errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
