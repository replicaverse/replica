"""Canonical paths for study references, analyses and result records."""

import json
import re
from pathlib import Path
from typing import Any

AXES = ("reproduction", "replication")
STAGES = ("repetition", "reanalysis")
BUCKETS = (*AXES, "robustness", "references")
ROLES = ("evidence", "scoped", "check", "published")
INDEX = "index.json"
_NAME = re.compile(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)*\Z")
_AMBIGUOUS = {"original", "associations", "headline", "native", "author", "authors"}


def parse_run(run: str) -> tuple[str, str]:
    """Return the analysis and record identifiers of a canonical result path.

    Args:
        run: Path relative to a study's results directory.

    Raises:
        ValueError: The path lacks an axis, analysis or record identifier.
    """
    parts = run.split("/")
    if parts[0] in AXES:
        valid = len(parts) == 4 and parts[1] in STAGES
    elif parts[0] == "references":
        valid = len(parts) == 4 and parts[1] in ("paper", "released")
    else:
        valid = parts[0] == "robustness" and len(parts) == 3
    if not valid or any(not _NAME.fullmatch(part) for part in parts):
        raise ValueError(f"Noncanonical result path: {run!r}")
    analysis, record = parts[-2:]
    if _AMBIGUOUS.intersection(analysis.split("_")):
        raise ValueError(f"Analysis must name a calculation: {analysis!r}")
    if not re.fullmatch(r"r[0-9a-f]{12,64}", record):
        raise ValueError(f"Result record must have an immutable identifier: {record!r}")
    return analysis, record


def hub_path(study: str, run: str) -> str:
    """Return the shared Hub and checkout suffix for one result record.

    Args:
        study: Canonical study package identifier.
        run: Canonical path within its results directory.
    """
    if not _NAME.fullmatch(study):
        raise ValueError(f"Invalid study identifier: {study!r}")
    parse_run(run)
    return f"studies/{study}/results/{run}"


def stage_label(run: str) -> str:
    """Describe the data and analysis represented by a result path.

    Args:
        run: Canonical path within a study's results directory.
    """
    parse_run(run)
    head = run.split("/")
    if head[0] == "references":
        return "paper reference" if head[1] == "paper" else "released output reference"
    if head[0] == "robustness":
        return "changed analysis choice"
    data = "original data" if head[0] == "reproduction" else "new data"
    analysis = "released analysis" if head[1] == "repetition" else "corrected analysis"
    return f"{data}, {analysis}"


def runs(results: Path) -> list[str]:
    """List result records at the required depth in each canonical bucket.

    Args:
        results: Study results directory.
    """
    found = []
    for bucket in BUCKETS:
        pattern = "*/*" if bucket == "robustness" else "*/*/*"
        found.extend(
            p.relative_to(results).as_posix()
            for p in (results / bucket).glob(pattern)
            if p.is_dir()
        )
    return sorted(found)


def load_index(results: Path) -> dict[str, dict[str, Any]]:
    """Read the study's result manifests, keyed by canonical result path.

    Args:
        results: Study results directory.
    """
    path = results / INDEX
    return json.loads(path.read_text())["runs"] if path.is_file() else {}


def description_conflicts(index: dict[str, dict[str, Any]]) -> list[str]:
    """Report duplicate result descriptions within each data and analysis stage.

    Args:
        index: Result manifests keyed by canonical result path.
    """
    descriptions: dict[tuple[str, str], str] = {}
    problems = []
    for run, entry in index.items():
        description = entry.get("description")
        if isinstance(description, str) and description.strip():
            key = (run.rsplit("/", 2)[0], " ".join(description.casefold().split()))
            if key in descriptions:
                problems.append(f"{run}: description also labels {descriptions[key]}")
            descriptions[key] = run
    return problems


_LAYOUT = (
    "[index.json](index.json) records each result's files, SHA-256 digests, "
    "producer and immutable HF revision. Result payloads are downloaded into "
    "the cache with `replica results fetch {study}`.",
    "Reproduction uses original data; replication uses new data. Repetition "
    "retains the released analysis; reanalysis corrects it. The claim registries "
    "select evidence for each measure.",
)
_SECTIONS = (
    ("reproduction/repetition", "Reproduction / Repetition"),
    ("reproduction/reanalysis", "Reproduction / Reanalysis"),
    ("replication/repetition", "Replication / Repetition"),
    ("replication/reanalysis", "Replication / Reanalysis"),
    ("references/paper", "Paper references"),
    ("references/released", "Released output references"),
    ("robustness", "Robustness"),
)
#: Openings of paragraphs the generator owns, so an older wording is replaced.
_OWNED_OPENINGS = (
    "[index.json]",
    "Reproduction uses original data",
    "No indexed results.",
    *("## " + label for _, label in _SECTIONS),
    "## Archives",
    "## Package resources",
    "The claim registry",
    "Each index entry",
    "Scientific coverage",
    "The current website scope",
    "The two axes are",
    "The directories follow the website's two axes",
    "Compact tables and their provenance",
    "Each run holds compact tables",
    "A reference or validation artifact",
    "Preserve explicit partial coverage",
)
ROLE_LABEL = {
    "evidence": "stage evidence",
    "scoped": "scoped comparison",
    "check": "supporting check",
    "published": "published values",
}


def _paragraphs(text: str) -> list[str]:
    return [p.strip() for p in text.strip().split("\n\n") if p.strip()]


def _kept_prose(existing: str) -> list[str]:
    """Paragraphs a study wrote itself: neither owned prose, title nor table."""
    kept = []
    for paragraph in _paragraphs(existing):
        if paragraph.startswith("# ") or paragraph.startswith("|"):
            continue
        if paragraph.startswith(_OWNED_OPENINGS):
            continue
        kept.append(paragraph)
    return kept


def render_readme(study: str, results: Path, hub_url: str) -> str:
    """Render indexed results by data and analysis axis, with pinned HF links.

    Args:
        study: Study package name.
        results: Directory containing the result index and README.
        hub_url: Base URL of the results dataset repository.
    """
    document = json.loads((results / INDEX).read_text())
    index = document["runs"]
    problems = description_conflicts(index)
    if problems:
        raise ValueError("; ".join(problems))
    revision = document.get("hub_revision", "main")
    existing = (
        (results / "README.md").read_text() if (results / "README.md").exists() else ""
    )
    parts = [
        f"# {study} results",
        *[line.format(study=study) for line in _LAYOUT],
        *_kept_prose(existing),
    ]
    for prefix, label in _SECTIONS:
        selected = sorted(
            (run for run in index if run.startswith(prefix + "/")),
            key=lambda run: (parse_run(run)[0], index[run]["description"].casefold()),
        )
        if not selected and prefix.split("/")[0] not in AXES:
            continue
        parts.append("## " + label)
        if not selected:
            parts.append("No indexed results.")
            continue
        has_notes = any(index[run].get("note") for run in selected)
        rows = [
            "| Analysis | Result | Role | Files |" + (" Note |" if has_notes else ""),
            "|---|---|---|---|" + ("---|" if has_notes else ""),
        ]
        for run in selected:
            entry = index[run]
            analysis, _ = parse_run(run)
            target = f"{hub_url}/tree/{revision}/{hub_path(study, run)}"
            row = f"| {analysis} | [{entry['description']}]({target}) | {ROLE_LABEL[entry['role']]} | {len(entry['files'])} |"
            if has_notes:
                row += f" {entry.get('note', '')} |"
            rows.append(row)
        parts.append("\n".join(rows))
    parts.extend(_metadata_tables(study, document))
    return "\n\n".join(parts) + "\n"


def _metadata_tables(study: str, document: dict[str, Any]) -> list[str]:
    sections = []
    archives = document.get("archives", {})
    if archives:
        rows = ["| Archive |", "|---|"]
        for entry in sorted(
            archives.values(), key=lambda entry: entry["description"].casefold()
        ):
            ref = entry["reference"]
            url = f"https://huggingface.co/datasets/{ref['repo_id']}/resolve/{ref['revision']}/{ref['path']}"
            rows.append(f"| [{entry['description']}]({url}) |")
        sections.extend(["## Archives", "\n".join(rows)])
    resources = document.get("resources", {})
    if resources:
        rows = ["| Resource | Role | Description |", "|---|---|---|"]
        for name, entry in sorted(resources.items()):
            if entry.get("archive"):
                ref = archives[entry["archive"]]["reference"]
                url = f"https://huggingface.co/datasets/{ref['repo_id']}/blob/{ref['revision']}/{ref['path']}"
            else:
                url = f"https://github.com/replicaverse/replica/blob/main/src/replica/studies/{study}/{name}"
            rows.append(
                f"| [{name}]({url}) | {entry['role']} | {entry['description']} |"
            )
        sections.extend(["## Package resources", "\n".join(rows)])
    return sections
