"""Optional dependency groups remain valid, reachable and acyclic."""

import tomllib
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

_PROJECT = tomllib.loads(
    (Path(__file__).resolve().parents[2] / "pyproject.toml").read_text()
)["project"]
_EXTRAS = _PROJECT["optional-dependencies"]


def _references():
    project = canonicalize_name(_PROJECT["name"])
    return {
        name: {
            extra
            for text in requirements
            if canonicalize_name((requirement := Requirement(text)).name) == project
            for extra in requirement.extras
        }
        for name, requirements in _EXTRAS.items()
    }


def test_requirements_are_valid_and_extra_names_are_canonical():
    for text in _PROJECT["dependencies"]:
        assert Requirement(text).url is None, f"PyPI rejects direct URLs: {text}"
    for name, requirements in _EXTRAS.items():
        assert name == canonicalize_name(name)
        assert requirements, f"Empty optional dependency group: {name}"
        for text in requirements:
            assert Requirement(text).url is None, f"PyPI rejects direct URLs: {text}"


def test_recursive_extra_references_exist_and_have_no_cycles():
    references = _references()

    def visit(name, path):
        assert name in references, f"Unknown extra: {' -> '.join((*path, name))}"
        assert name not in path, f"Extra cycle: {' -> '.join((*path, name))}"
        for child in references[name]:
            visit(child, (*path, name))

    for name in references:
        visit(name, ())


def test_all_extra_reaches_every_optional_group():
    references = _references()
    seen = set()
    pending = ["all"]
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        pending.extend(references[name])
    assert seen == set(_EXTRAS), f"Extras missing from all: {set(_EXTRAS) - seen}"
