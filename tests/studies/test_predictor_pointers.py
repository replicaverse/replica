"""A study README's ``## Predictors`` section must point at things that exist.

The section records, per study, the authors' convention, the REPLICA code that
reproduces it, and the published cell that is faithful to it. This guard walks
every relative link in the section and requires its target on disk, so a stale
pointer fails.

Cell tags are checked against the published store only when it is reachable, so
the guard stays useful offline.
"""

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.needs_data

STUDIES = Path(__file__).resolve().parents[2] / "src" / "replica" / "studies"
SECTION = re.compile(r"^## Predictors$(.*?)^## ", re.MULTILINE | re.DOTALL)
LINK = re.compile(r"\]\(([^)]+)\)")
BACKTICKED = re.compile(r"`([^`]+)`")
#: Column of the section's table holding the published cell, 0-based over the
#: cells of a ``|``-delimited row.
CELL_COLUMN = 3
#: A cell entry that names no single published tag: an absence, or a family
#: standing in for many tags.
NOT_A_TAG = ("none", "n/a", "not verified", "*")


def readmes_with_section() -> list[Path]:
    out = []
    for readme in sorted(STUDIES.glob("*/README.md")):
        if SECTION.search(readme.read_text(encoding="utf-8")):
            out.append(readme)
    return out


def section_of(readme: Path) -> str:
    match = SECTION.search(readme.read_text(encoding="utf-8"))
    assert match, f"{readme} has no '## Predictors' section"
    return match.group(1)


def named_cells(section: str) -> set[str]:
    """Concrete published tags from the table's faithful-cell column.

    Wildcards and absences are skipped: they name a family or a gap.
    """
    tags: set[str] = set()
    for line in section.splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) <= CELL_COLUMN:
            continue
        entry = cells[CELL_COLUMN]
        if any(marker in entry.lower() for marker in NOT_A_TAG):
            continue
        tags.update(BACKTICKED.findall(entry))
    return tags


def test_some_study_documents_its_predictors():
    """The section is the convention record; losing all of them is a regression."""
    assert readmes_with_section(), "no study README has a '## Predictors' section"


@pytest.mark.parametrize("readme", readmes_with_section(), ids=lambda p: p.parent.name)
def test_predictor_links_resolve(readme: Path):
    broken = []
    for target in LINK.findall(section_of(readme)):
        if target.startswith(("http://", "https://", "#")):
            continue
        path = (readme.parent / target).resolve()
        if not path.exists():
            broken.append(target)
    assert not broken, f"{readme.parent.name}: unresolved pointers {broken}"


@pytest.mark.parametrize("readme", readmes_with_section(), ids=lambda p: p.parent.name)
def test_named_cells_are_published(readme: Path):
    """Any cell named as faithful must exist in the store's own index."""
    tags = sorted(named_cells(section_of(readme)))
    if not tags:
        pytest.skip("no published cell named")
    from replica.predictors import published_tags

    published = published_tags()
    if published is None:
        pytest.skip("predictor store unreachable")
    missing = [t for t in tags if t not in published]
    assert not missing, f"{readme.parent.name}: cells not in the store {missing}"
