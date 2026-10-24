"""``Study.ensure_data`` verifies each dependency once per process."""

from pathlib import Path

import pytest

from replica.external.dependency import DataDependency, sha256_checksum
from replica.studies import base
from replica.studies.base import Study


class _OneFile(Study):
    name = "one_file"
    bibtex = "@misc{one_file, title={One file}}"

    def __init__(self, dep: DataDependency):
        self.dependencies = [dep]

    def dataset(self, *args, **kwargs):
        raise NotImplementedError

    def evaluation_spec(self, *args, **kwargs):
        raise NotImplementedError

    def analysis_config(self, *args, **kwargs):
        raise NotImplementedError


@pytest.mark.resource_free("Dependency verification uses a stub and a temporary file.")
def test_ensure_data_verifies_a_dependency_once_per_process(tmp_path, monkeypatch):
    target = tmp_path / "table.tsv"
    target.write_text("a\tb\n1\t2\n")
    dep = DataDependency(
        path=Path("table.tsv"),
        source="file:table.tsv",
        checksum=sha256_checksum(target),
    )
    calls: list[Path] = []
    monkeypatch.setattr(
        base, "verify_dependency", lambda dep, root: calls.append(root / dep.path)
    )
    monkeypatch.setattr(base, "_VERIFIED_DEPENDENCIES", set())
    study = _OneFile(dep)
    study.ensure_data(tmp_path)
    study.ensure_data(tmp_path)
    assert calls == [target]
