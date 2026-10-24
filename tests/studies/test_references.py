"""Bibliography metadata and references used by study and feature tests."""

import re
from pathlib import Path

import pytest
import yaml

from replica.studies import ALL_STUDIES
from replica.studies.base import bibtex_field
from tests.feature.diagnostics.support.catalog import _REFS

pytestmark = pytest.mark.artifact

_REFERENCES = yaml.safe_load(
    Path(__file__).with_name("fixtures").joinpath("references.yaml").read_text()
)


def test_bibliography_matches_studies():
    assert set(ALL_STUDIES) <= _REFERENCES.keys()
    for name, study in ALL_STUDIES.items():
        assert (
            re.sub(r"\s+", " ", _REFERENCES[name]).strip()
            == re.sub(r"\s+", " ", study.bibtex).strip()
        ), name


def test_bibliography_entries_have_author_title_and_year():
    for name, bibtex in _REFERENCES.items():
        for field in ("author", "title", "year"):
            assert bibtex_field(bibtex, field), (name, field)


def test_diagnostic_references_resolve():
    import replica.dataset.metadata

    root = Path(replica.dataset.metadata.__file__).parent / "yaml"
    keys = set(_REFERENCES)
    for path in root.rglob("*.yaml"):
        metadata = yaml.safe_load(path.read_text()) or {}
        if metadata.get("bibtex"):
            keys.add(f"dataset:{metadata.get('citation') or path.stem}")
    assert {key for key, _ in _REFS.values()} <= keys
