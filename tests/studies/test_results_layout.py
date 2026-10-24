"""Every study's tracked results sit in a stage bucket and carry a role.

The website reads a run's stage from its placement, so a run outside the
buckets, or one whose role is unrecorded, would be evidence the site cannot
place. The index and the tree must agree in both directions.
"""

import json
import re
from pathlib import Path

import pytest

from replica import studies as S
from replica.studies.records.files import hub_url, validate_package
from replica.studies.records.layout import (
    AXES,
    BUCKETS,
    INDEX,
    ROLES,
    STAGES,
    load_index,
    runs,
)

_STUDIES = Path(S.__file__).resolve().parent
_REPO = _STUDIES.parents[2]
_WITH_RESULTS = sorted(
    p.parent.name for p in _STUDIES.glob("*/results") if load_index(p)
)


@pytest.mark.parametrize(
    "study", sorted(p.name for p in _STUDIES.iterdir() if (p / "__init__.py").is_file())
)
def test_package_artifacts_are_indexed_or_declared_resources(study: str):
    assert validate_package(study) == []


@pytest.mark.parametrize("study", _WITH_RESULTS)
def test_only_buckets_sit_under_results(study: str):
    results = _STUDIES / study / "results"
    stray = sorted(
        p.name for p in results.iterdir() if p.is_dir() and p.name not in BUCKETS
    )
    assert not stray, f"{study}: directories outside the buckets: {stray}"
    for axis in AXES:
        if (results / axis).is_dir():
            inner = sorted(
                p.name for p in (results / axis).iterdir() if p.name not in STAGES
            )
            assert not inner, f"{study}/{axis}: not a stage: {inner}"


@pytest.mark.parametrize("study", _WITH_RESULTS)
def test_index_and_tree_agree(study: str):
    results = _STUDIES / study / "results"
    index = json.loads((results / INDEX).read_text())["runs"]
    on_disk = set(runs(results))
    assert on_disk <= set(index), (
        f"{study}: runs the index lacks: {on_disk - set(index)}"
    )
    unbacked = [run for run in set(index) - on_disk if not index[run].get("hub")]
    assert not unbacked, (
        f"{study}: index runs neither on disk nor on the hub: {unbacked}"
    )
    for run, entry in index.items():
        assert entry["files"] == entry["hub"], f"{study}/{run}: incomplete HF manifest"
        assert entry.get("role") in ROLES, f"{study}/{run}: role {entry.get('role')!r}"


@pytest.mark.parametrize("study", _WITH_RESULTS)
def test_every_indexed_run_is_reachable_from_the_results_readme(study: str):
    """Every record remains discoverable when its payload is outside the checkout."""
    results = _STUDIES / study / "results"
    text = (results / "README.md").read_text()
    for run, entry in load_index(results).items():
        assert f"[{entry['description']}]({hub_url(study, run)})" in text, (study, run)
    labels = re.findall(r"\[([^\]]+)\]\(", text)
    assert not any(re.fullmatch(r"r?[0-9a-f]{12,64}", label) for label in labels)


@pytest.mark.parametrize("study", _WITH_RESULTS)
def test_every_results_link_resolves(study: str):
    """A README link into results/ names a file in the checkout or on the hub."""
    study_dir = _STUDIES / study

    def resolves(relative: str) -> bool:
        return (study_dir / "results" / relative).is_file()

    bad = []
    documentation = "".join(
        (study_dir / name).read_text()
        for name in ("README.md", "EVIDENCE.md")
        if (study_dir / name).is_file()
    )
    for match in re.finditer(r"\]\(results/([^)#]+)\)", documentation):
        if not resolves(match.group(1)):
            bad.append(f"results/{match.group(1)}")
    results_readme = study_dir / "results" / "README.md"
    for match in re.finditer(r"\]\(([^)#h][^)#]*)\)", results_readme.read_text()):
        if not resolves(match.group(1)):
            bad.append(f"results/{match.group(1)}")
    assert not bad, f"{study}: links into results/ that do not resolve: {bad}"


@pytest.mark.parametrize("study", _WITH_RESULTS)
def test_archive_links_use_the_recorded_repository_and_revision(study: str):
    results = _STUDIES / study / "results"
    document = json.loads((results / INDEX).read_text())
    text = (results / "README.md").read_text()
    for entry in document.get("archives", {}).values():
        ref = entry["reference"]
        url = f"https://huggingface.co/datasets/{ref['repo_id']}/resolve/{ref['revision']}/{ref['path']}"
        assert f"[{entry['description']}]({url})" in text


def test_results_readmes_are_generated():
    from scripts.readmes.regen_study_readmes import main

    assert main(["--results", "--check"]) == 0
