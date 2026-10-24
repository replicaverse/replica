"""Check study documentation against citations, source records and tracked files."""

import ast
import hashlib
import json
import os
import re
import subprocess
from functools import cache
from pathlib import Path
from urllib.parse import unquote, urlsplit

import pytest

from replica import studies as S
from replica.external import EXTERNAL_REPOS
from replica.studies import STUDY_IDS
from replica.studies.base import bibtex_field, clean_tex
from replica.studies.claims.registry import render_readme_claims

_STUDIES_DIR = Path(S.__file__).resolve().parent
_REPO = _STUDIES_DIR.parents[2]
_README_SECTIONS = ("Abstract", "Predictors", "BibTeX")
_EVIDENCE_SECTIONS = ("Headline claims", "Scope of reproduction", "Notes")
#: The study packages, as the lazy loader discovers them.
_NAMES = list(STUDY_IDS)

# SHA-256 of the abstract text served for the URL in the canonical paper/
# BibTeX entry. DOI URLs use CSL-JSON content negotiation. Abstract sections are
# deliberately one unwrapped paragraph: any character-level drift changes the
# digest. The oota2023linguistic URL is a PDF, so its digest uses the reflowed Abstract
# paragraph from that PDF. The Information Value hash uses its official ACL
# Anthology record.
_ABSTRACT_SHA256_BY_URL = {
    "https://arxiv.org/abs/2610.18439": "88c0f2bcea1261e071925c4777951bea5d2d85a9d9f10cb88b627ac11c415e77",
    "https://arxiv.org/abs/2610.18414": "cd0ed8dcc9b501cb885c2c8cc9d9c2fe0f30bf8083ffc3a4f0006a2d523e66cd",
    "https://aclanthology.org/2025.ijcnlp-short.4/": "1a25e8678a722ddeff7b3bccff183c89cba821571a45e0e676b8de46ccfb2d3d",
    "https://aclanthology.org/2023.emnlp-main.343/": "d245a252452eb70be81f4358dad08a1cffe2cf5fa35a851dd945a64234e0968b",
    "https://aclanthology.org/2024.findings-emnlp.682/": "667a856bd99c90c89f19c7501cf3ebce719f5cb25c0b1fe95a36d0759ed5eacc",
    "https://aclanthology.org/2024.emnlp-main.1032/": "5a67163b99837e3c49b74b51d0d68161af96c818c2b81f432b33500faf789020",
    "https://aclanthology.org/2021.acl-long.405/": "269377d4a3adf51eea1bcab684ac472e82ca7a54d32508e80437c40ccc7b05e3",
    "https://aclanthology.org/2025.tacl-1.78/": "14beebb04ac35adcb95549cb87c5f63f56057d9786e27d36d1f997379492f97d",
    "https://aclanthology.org/2024.emnlp-main.921/": "c175211530316a98ccb54f665a57a8370a2e6a29bfa2f7e4fb79f10d69a1417a",
    "https://aclanthology.org/2026.acl-long.1534/": "363dbcbdbbcddede0005d267e7d7da5b663f912d306adc4e8ba1ee8eaed97462",
    "https://proceedings.neurips.cc/paper_files/paper/2023/hash/3a0e2de215bd17c39ad08ba1d16c1b12-Abstract-Conference.html": "392ab37691b6593371e26298340a71f9aae0072c49297c5b5df8c6c12ac49d5d",
    "https://aclanthology.org/2024.emnlp-main.179/": "c7b414c66fa210c96ec94027a36018f2bb1a80e36347f315d682b7435db822b5",
    "https://doi.org/10.1162/tacl_a_00603": "9b343ff397ec74870a410d09cb303a5938bcda2a60f506757567e5ab6ce208d7",
    "https://aclanthology.org/2025.acl-long.1474/": "d10ad822c2dac44975dda0d5b66737721f92bd09ed5fbc7aede4ed991dc27768",
    "https://aclanthology.org/D18-1288": "18b78e9e40baaa7de762cd67ee17c37dc63328d23f9c28c9e3bb9e032d321620",
    "https://doi.org/10.1016/j.cognition.2021.104735": "c08b6832ce9e41a3095d2811f611c706aba3c3df644b221f06178168617246ed",
    "https://doi.org/10.1162/opmi_a_00119": "511d8391b9289dc135f0c88afad3ca5e6b8b3181922790b93d309c1eef92949d",
    "https://www.pnas.org/doi/abs/10.1073/pnas.2307876121": "89f1e68a306779833c4a2961c28ec00f406a5fb5cae140e25cdefcd4b7e4b7fe",
    "https://aclanthology.org/2023.tacl-1.82/": "3001f638f6903ba5e9cd04dc6e4a582fe94452e8f7094996b08f0b1d60d94aa2",
    "https://lanni-ni.github.io/hsp2025.pdf": "fb83e9979654b03964907a3db92cf2c891a5cba0aeefbf28da6b6feaa1f47116",
    "https://arxiv.org/abs/2608.23818": "27d86e4b80e19944c39e7ca551ec17a8eace689a93ccc1e14d4e9d5337025013",
    "https://aclanthology.org/2023.findings-emnlp.1052/": "5068287ebbbfa30ce276ed3ff698fa4f42c175f0755ac9a4335a098efa9e1ce6",
}
# README paths must exist in git or be materialized by the external registry.
_PATH_RE = re.compile(r"(?:local/|scripts/|docs/|re_runs/|external_repos/)[\w./_-]+")
_GITHUB_BLOB_RE = re.compile(
    r"https://github\.com/[^/]+/[^/]+/blob/(?P<ref>[^/]+)/[^\s)]+"
)
_GITHUB_TREE_RE = re.compile(r"https://github\.com/[^/]+/[^/]+/tree/(?P<ref>[^/\s)]+)")
_CODE_FILE_RE = re.compile(
    r"`(?P<file>[^`]+\.(?:py|R|Rmd|ipynb|yml|yaml|ini|csv|txt|jsonl))`"
)
_DECLARED_MISSING_AUTHOR_FILES = {
    # The scorer internals live in a submodule whose repository is deleted,
    # so there is nothing to link to.
    "meister2024similarity": {"measures_simplified.py"},
    # The gam_data frames are generated on demand by build_gam_data.
    "shain2024logarithmic": {
        "test_surp_lin.py",
        "test_surp_lin_curr.py",
        "gam_data/{corpus}_X_{train-dev,test}.csv",
    },
    # The preprocessed Dundee file the authors' preproc_EN reads is not in
    # their release.
    "xu2023linearity": {"big_dundee_et.csv"},
}


def _tracked() -> set[str]:
    out = subprocess.run(
        ["git", "-C", str(_REPO), "ls-files"],
        capture_output=True,
        text=True,
        check=False,
    )
    return set(out.stdout.split())


def _normalise(bibtex: str) -> str:
    """Compare bibtex ignoring indentation and blank lines."""
    return "\n".join(
        line.strip() for line in bibtex.strip().splitlines() if line.strip()
    )


@cache
def _study_bibtex(name: str) -> str:
    """Read the study's citation without importing its data or model backends."""
    citations = []
    for path in (_STUDIES_DIR / name).glob("*.py"):
        for node in ast.parse(path.read_text()).body:
            if not isinstance(node, ast.ClassDef) or node.name.lower() != name:
                continue
            for statement in node.body:
                if isinstance(statement, ast.Assign) and any(
                    isinstance(target, ast.Name) and target.id == "bibtex"
                    for target in statement.targets
                ):
                    citations.append(ast.literal_eval(statement.value))
    assert len(citations) == 1 and isinstance(citations[0], str), name
    return citations[0]


def _doc_files(name: str) -> list[Path]:
    """The README and, for a study with claims, the evidence file."""
    files = [_STUDIES_DIR / name / "README.md"]
    evidence = _STUDIES_DIR / name / "EVIDENCE.md"
    if evidence.is_file():
        files.append(evidence)
    return files


def _doc_text(name: str) -> str:
    """Both documentation files of a study, concatenated."""
    return "\n".join(path.read_text() for path in _doc_files(name))


def _file_parts(path: Path, required: tuple[str, ...]) -> tuple[str, dict[str, str]]:
    """Check one documentation file's schema and return its title and sections."""
    name = path.parent.name
    text = path.read_text()
    first = re.match(r"^# ([^\n]+)\n", text)
    assert first, f"{name}/{path.name}: must start with one H1 study title"
    assert len(re.findall(r"^# [^\n]+$", text, re.M)) == 1, (
        f"{name}/{path.name}: must contain exactly one H1"
    )
    headings = re.findall(r"^## ([^\n]+)$", text, re.M)
    assert set(required) <= set(headings), (
        f"{name}/{path.name}: missing required sections {set(required) - set(headings)}"
    )
    assert headings[0] == required[0], (
        f"{name}/{path.name}: must begin with {required[0]}"
    )
    assert len(headings) == len(set(headings)), (
        f"{name}/{path.name}: duplicate sections"
    )
    parts = {}
    for index, heading in enumerate(headings):
        start = re.search(rf"^## {re.escape(heading)}\n", text, re.M).end()
        if index + 1 < len(headings):
            stop = (
                start
                + re.search(
                    rf"^## {re.escape(headings[index + 1])}\n", text[start:], re.M
                ).start()
            )
        else:
            stop = len(text)
        parts[heading] = text[start:stop].strip("\n")
    return first.group(1), parts


def _readme_parts(name: str) -> tuple[str, dict[str, str]]:
    """The study's title and every section of its documentation, schema checked."""
    files = _doc_files(name)
    if len(files) == 1:
        return _file_parts(files[0], _README_SECTIONS)
    title, parts = _file_parts(files[0], _README_SECTIONS)
    evidence_title, evidence = _file_parts(files[1], _EVIDENCE_SECTIONS)
    assert evidence_title == title, f"{name}: EVIDENCE.md title differs from README.md"
    assert not (parts.keys() & evidence.keys()), (
        f"{name}: sections duplicated across README.md and EVIDENCE.md"
    )
    return title, {**parts, **evidence}


@pytest.mark.parametrize("name", _NAMES)
def test_readme_title_and_required_sections(name: str):
    title, parts = _readme_parts(name)
    expected_title = clean_tex(bibtex_field(_study_bibtex(name), "title") or "")
    assert title == expected_title
    assert all(parts[heading] for heading in _README_SECTIONS)
    if len(_doc_files(name)) == 2:
        assert set(_README_SECTIONS) <= set(parts) and set(_EVIDENCE_SECTIONS) <= set(
            parts
        )


@pytest.mark.parametrize(
    "name",
    [name for name in _NAMES if (_STUDIES_DIR / name / "claims.yaml").is_file()],
)
def test_readme_headline_claims_match_registry(name: str):
    """Checked-in public claims are generated from the audited registry."""
    _, parts = _readme_parts(name)
    assert parts["Headline claims"] == render_readme_claims(name)


@pytest.mark.parametrize("name", _NAMES)
def test_readme_citation_matches_bibtex(name: str):
    """The rendered citation matches the study's BibTeX."""
    text = _doc_text(name)
    blocks = re.findall(r"```bibtex\n(.*?)```", text, re.S)
    assert len(blocks) == 1, f"{name}: expected exactly one bibtex block"
    assert _normalise(blocks[0]) == _normalise(_study_bibtex(name)), (
        f"{name}: README citation has drifted from the bibtex attribute"
    )


@pytest.mark.parametrize("name", _NAMES)
def test_readme_abstract_matches_declared_source(name: str):
    """Abstracts link to their source or match an audited quotation."""
    _, parts = _readme_parts(name)
    abstract = parts["Abstract"]
    url = bibtex_field(_study_bibtex(name), "url")
    link = re.fullmatch(r"\[[^\]\n]+\]\((https?://[^\s)]+)\)", abstract)
    if link is not None:
        assert link.group(1) == url, f"{name}: abstract link differs from citation URL"
        return
    record_path = _STUDIES_DIR / name / "abstract_source.json"
    if record_path.exists():
        record = json.loads(record_path.read_text())
        assert record["kind"] == "author_supplied"
        assert record["status"] == "unpublished"
        source = record["source"]
        assert source["repository"] == "https://github.com/replicaverse/replica"
        assert re.fullmatch(r"[0-9a-f]{40}", source["commit"])
        assert source["path"] == str(
            (_STUDIES_DIR / name / "README.md").relative_to(_REPO)
        )
        assert source["section"] == "Abstract"
        assert hashlib.sha256(abstract.encode()).hexdigest() == record["sha256"]
        return
    if url is None:
        assert abstract.casefold().startswith("unavailable:"), (
            f"{name}: abstract needs a source record or an unavailable status"
        )
        return
    assert url in _ABSTRACT_SHA256_BY_URL, f"{name}: unaudited abstract URL {url}"
    assert "\n" not in abstract, f"{name}: exact abstract must be one paragraph"
    digest = hashlib.sha256(abstract.encode()).hexdigest()
    assert digest == _ABSTRACT_SHA256_BY_URL[url], (
        f"{name}: abstract does not exactly match the audited text at {url}"
    )


@pytest.mark.parametrize("name", _NAMES)
def test_readme_references_are_resolvable(name: str):
    """No pointers to gitignored or machine-local paths."""
    text = _doc_text(name)
    assert "/cluster/" not in text, f"{name}: README hard-codes a cluster path"
    tracked = _tracked()
    fetchable = {f"external_repos/{n}" for n in EXTERNAL_REPOS}
    bad = []
    for line_no, line in enumerate(text.splitlines(), 1):
        url_spans = [match.span() for match in re.finditer(r"https?://[^\s)]+", line)]
        url_spans.extend(match.span() for match in _RELATIVE_TARGET_RE.finditer(line))
        for match in _PATH_RE.finditer(line):
            if any(start <= match.start() < stop for start, stop in url_spans):
                continue
            ref = match.group(0).rstrip(".,);:`'\"")
            if ref in tracked or any(
                t.startswith(ref.rstrip("/") + "/") for t in tracked
            ):
                continue
            if any(ref == f or ref.startswith(f + "/") for f in fetchable):
                continue
            bad.append(f"line {line_no}: {ref}")
    for match in _RELATIVE_TARGET_RE.finditer(text):
        target = match.group(1)
        parsed = urlsplit(target)
        if parsed.scheme or parsed.netloc or not parsed.path:
            continue
        resolved = os.path.normpath(_STUDIES_DIR / name / unquote(parsed.path))
        ref = os.path.relpath(resolved, _REPO)
        if ref not in tracked and not any(
            t.startswith(ref.rstrip("/") + "/") for t in tracked
        ):
            if not any(ref == f or ref.startswith(f + "/") for f in fetchable):
                bad.append(target)
    assert not bad, f"{name}: README points at unavailable paths: {bad}"


@pytest.mark.parametrize("name", _NAMES)
def test_github_code_links_are_commit_pinned_and_line_anchored(name: str):
    """Author-code citations remain immutable and land on a code snippet."""
    text = _doc_text(name)
    bad = []
    for match in _GITHUB_BLOB_RE.finditer(text):
        url = match.group(0)
        if not re.fullmatch(r"[0-9a-f]{40}", match.group("ref")):
            bad.append(f"not pinned to a full commit: {url}")
        if not re.search(r"#L\d+(?:-L\d+)?$", url):
            bad.append(f"not line-anchored: {url}")
    for match in _GITHUB_TREE_RE.finditer(text):
        if not re.fullmatch(r"[0-9a-f]{40}", match.group("ref")):
            bad.append(f"tree not pinned to a full commit: {match.group(0)}")
    assert not bad, f"{name}: invalid GitHub code links: {bad}"


@pytest.mark.parametrize("name", _NAMES)
def test_code_file_references_are_linked_or_explicitly_missing(name: str):
    """A named source file is clickable unless the release omits that file."""
    text = _doc_text(name)
    bare = set()
    for match in _CODE_FILE_RE.finditer(text):
        prefix = text[: match.start()]
        if prefix.rfind("[") <= prefix.rfind("]"):
            bare.add(match.group("file"))
    allowed = _DECLARED_MISSING_AUTHOR_FILES.get(name, set())
    assert bare <= allowed, f"{name}: unlinked code-file references: {bare - allowed}"


_RELATIVE_TARGET_RE = re.compile(r"\]\((?!https?://|#)([^)#]+)(?:#[^)]*)?\)")
