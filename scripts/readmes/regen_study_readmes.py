#!/usr/bin/env python3
"""Rewrite the generated parts of every study's documentation.

A study with claims documents itself in two files: ``README.md`` holds the
recipe sections (Abstract, Predictors, BibTeX) and ``EVIDENCE.md`` holds the
Headline claims, the Scope of reproduction, the stage sections and the Notes.
The "Headline claims" section is rendered from the registry, and
``tests/studies/test_study_readmes.py`` requires it to equal
``render_readme_claims(study)`` byte for byte. A study can select concise
claim tables with ``sources.readme_claims: compact`` in its claims file. The
table of contents between the title and the first section of each file lists
that file's second-level headings, and the same test requires it to equal
:func:`readme_toc`. Optional study-owned ``readme_evidence.py`` renderers
supply delimited numerical evidence blocks. A study-owned ``readme.py`` with
``update(text)`` renders the complete document, which :func:`split_document`
divides into the two files.

Usage::

    python scripts/readmes/regen_study_readmes.py          # rewrite in place
    python scripts/readmes/regen_study_readmes.py --check  # exit 1 if any differs
"""

from __future__ import annotations

import argparse
import importlib
import json
import re
import runpy
import sys
from pathlib import Path

from replica.studies.claims.registry import load_registry, render_readme_claims

STUDIES_DIR = Path(__file__).resolve().parents[2] / "src" / "replica" / "studies"
EVIDENCE = "EVIDENCE.md"
#: The sections that stay in README.md; every other section is evidence.
README_SECTIONS = ("Abstract", "Predictors", "BibTeX")
_SECTION = re.compile(r"^## Headline claims\n(.*?)(?=^## )", re.MULTILINE | re.DOTALL)
_TITLE = re.compile(r"^# [^\n]+\n")
_HEADING = re.compile(r"^## ([^\n]+)$", re.MULTILINE)


def heading_slug(heading: str) -> str:
    """The anchor GitHub gives a heading: lower case, punctuation dropped, spaces as hyphens.

    Args:
        heading: The heading text without its ``#`` marks.
    """
    kept = re.sub(r"[^\w\- ]", "", heading.lower())
    return kept.replace(" ", "-")


def readme_toc(text: str) -> str:
    """The table of contents a README carries: one link per second-level heading.

    Args:
        text: The whole README.

    Returns:
        The list, one bullet per heading in document order, without a trailing
        newline.
    """
    return "\n".join(
        f"- [{heading}](#{heading_slug(heading)})" for heading in _HEADING.findall(text)
    )


def with_toc(text: str) -> str:
    """The README with its table of contents current.

    Whatever sits between the title and the first section is replaced, so a
    stale list or a missing one comes out the same.

    Args:
        text: The whole README, opening with its one first-level title.
    """
    title = _TITLE.match(text)
    first = _HEADING.search(text)
    if title is None or first is None:
        raise ValueError("README needs a title and at least one section")
    return (
        text[: title.end()] + "\n" + readme_toc(text) + "\n\n" + text[first.start() :]
    )


def with_evidence(text: str, study_dir: Path) -> str:
    """Replace the evidence block supplied by a study's standalone renderer.

    The optional ``readme_evidence.py`` exports ``BEGIN``, ``END`` and
    ``render()``. Loading its file directly keeps study analysis backends out of
    the documentation process. Renderers use absolute imports and read indexed
    evidence without changing it.

    Args:
        text: README whose surrounding prose must be preserved.
        study_dir: Study package containing the README and optional renderer.
    """
    renderer = study_dir / "readme_evidence.py"
    if not renderer.is_file():
        return text
    module = runpy.run_path(str(renderer))
    begin, end = module["BEGIN"], module["END"]
    if text.count(begin) != 1 or text.count(end) != 1:
        raise ValueError(f"{study_dir}: expected one evidence block")
    start, stop = text.index(begin), text.index(end) + len(end)
    if start >= text.index(end):
        raise ValueError(f"{study_dir}: evidence markers are out of order")
    rendered = module["render"]()
    if (
        not rendered.startswith(begin)
        or not rendered.endswith(end)
        or rendered.count(begin) != 1
        or rendered.count(end) != 1
    ):
        raise ValueError(f"{renderer}: renderer must return one delimited block")
    return text[:start] + rendered + text[stop:]


def split_document(text: str) -> tuple[str, str]:
    """Divide one complete study document into its README and EVIDENCE texts.

    Both files start with the document's title; each receives the sections
    that belong to it in the document's order, without a table of contents.

    Args:
        text: A document with one H1 title and H2 sections.
    """
    title = re.match(r"^# [^\n]+\n", text)
    if title is None:
        raise ValueError("document must start with an H1 title")
    starts = [m.start() for m in _HEADING.finditer(text)] + [len(text)]
    sections = [text[a:b] for a, b in zip(starts, starts[1:], strict=False)]
    readme = [
        sec for sec in sections if _HEADING.match(sec).group(1) in README_SECTIONS
    ]
    evidence = [
        sec for sec in sections if _HEADING.match(sec).group(1) not in README_SECTIONS
    ]

    def join(parts: list[str]) -> str:
        return title.group(0) + "\n" + "".join(parts)

    return join(readme), join(evidence)


def regenerate(study: str, check: bool = False) -> bool:
    """Bring a study's generated documentation current; return whether it differs.

    Args:
        study: Package name under ``replica.studies``.
        check: Report whether the files differ without writing either of them.
    """
    study_dir = STUDIES_DIR / study
    readme_path, evidence_path = study_dir / "README.md", study_dir / EVIDENCE
    readme = readme_path.read_text()
    evidence = evidence_path.read_text() if evidence_path.is_file() else None
    new_readme, new_evidence = readme, evidence
    if (study_dir / "readme.py").is_file():
        renderer = importlib.import_module(f"replica.studies.{study}.readme")
        new_readme, new_evidence = split_document(renderer.update(readme))
    if study in {c["study"] for c in load_registry()["claims"]}:
        if new_evidence is None:
            raise ValueError(
                f"{evidence_path}: a study with claims needs an evidence file"
            )
        match = _SECTION.search(new_evidence)
        if match is None:
            raise ValueError(f"{evidence_path}: no '## Headline claims' section")
        rendered = "\n" + render_readme_claims(study).rstrip("\n") + "\n\n"
        new_evidence = (
            new_evidence[: match.start(1)] + rendered + new_evidence[match.end(1) :]
        )
    if new_evidence is not None:
        new_evidence = with_toc(with_evidence(new_evidence, study_dir))
    else:
        new_readme = with_evidence(new_readme, study_dir)
    new_readme = with_toc(new_readme)
    changed = new_readme != readme or (
        new_evidence is not None and new_evidence != evidence
    )
    if changed and not check:
        if new_readme != readme:
            readme_path.write_text(new_readme)
        if new_evidence is not None and new_evidence != evidence:
            evidence_path.write_text(new_evidence)
    return changed


def regenerate_results(check: bool) -> list[str]:
    """Rewrite the generated parts of every study's ``results/README.md``.

    Result records and their pinned links are grouped by data and analysis
    axis; notes outside the generated sections are kept.

    Args:
        check: Report the READMEs that differ instead of writing them.

    Returns:
        The studies whose results README differed.
    """
    from replica.studies.records.files import hub_repo
    from replica.studies.records.layout import render_readme

    stale = []
    for results in sorted(STUDIES_DIR.glob("*/results")):
        if not (results / "index.json").is_file():
            continue
        target = results / "README.md"
        document = json.loads((results / "index.json").read_text())
        text = render_readme(
            results.parent.name,
            results,
            f"https://huggingface.co/datasets/{hub_repo(document)}",
        )
        if target.exists() and target.read_text() == text:
            continue
        stale.append(results.parent.name)
        if not check:
            target.write_text(text)
    return stale


def main(argv: list[str] | None = None) -> int:
    """Regenerate every study README, or with ``--results`` every results README.

    Args:
        argv: Command-line arguments; defaults to ``sys.argv[1:]``.

    Returns:
        0 when the READMEs are current after the run, 1 under ``--check`` if
        any README needed regeneration.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="report stale READMEs without writing"
    )
    parser.add_argument("--study", help="regenerate only this study")
    parser.add_argument(
        "--results",
        action="store_true",
        help="regenerate the results READMEs instead of the study READMEs",
    )
    args = parser.parse_args(argv)
    if args.results:
        if args.study:
            parser.error("--results regenerates every results README")
        stale = regenerate_results(args.check)
        for study in stale:
            print(("stale: " if args.check else "rewrote: ") + study)
        return 1 if (args.check and stale) else 0
    studies = sorted(
        p.parent.name
        for p in STUDIES_DIR.glob("*/README.md")
        if (p.parent / "__init__.py").exists()
    )
    if args.study:
        if args.study not in studies:
            parser.error(f"Unknown study: {args.study}")
        studies = [args.study]
    stale = [study for study in studies if regenerate(study, check=args.check)]
    for study in stale:
        print(("stale: " if args.check else "rewrote: ") + study)
    return 1 if (args.check and stale) else 0


if __name__ == "__main__":
    sys.exit(main())
