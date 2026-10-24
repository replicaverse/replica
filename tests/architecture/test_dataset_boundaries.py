"""Shared data contracts and text segmentation remain independent of loaders."""

import tomllib
from pathlib import Path

import pytest

from scripts.checks.source.import_boundaries import _check

_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("indirect", [False, True], ids=["direct", "via-helper"])
@pytest.mark.parametrize(
    "source,rule",
    [
        *[
            (f"dataset.{name}", "dataset-contracts-have-no-loader-dependencies")
            for name in ("contracts", "measures", "metadata", "fixation_utils")
        ],
        ("segmentation", "text-segmentation-is-independent-of-datasets"),
    ],
)
def test_shared_contracts_reject_loader_imports(tmp_path, source, rule, indirect):
    rules = tomllib.loads(
        (_ROOT / "scripts/checks/source/architecture.toml").read_text()
    )["rules"]
    policy = {"rules": [r for r in rules if r["name"] == rule]}
    package = tmp_path / "src/replica"
    if source.startswith("dataset."):
        package /= "dataset"
    package.mkdir(parents=True)
    for name in ("contracts", "measures", "metadata", "fixation_utils", "base"):
        (package / f"{name}.py").write_text("")
    import_loader = "from replica.dataset.base import Dataset\n"
    (package / f"{source.split('.')[-1]}.py").write_text(
        "from . import helper\n" if indirect else import_loader
    )
    if indirect:
        (package / "helper.py").write_text(import_loader)
    assert any(rule in error for error in _check(tmp_path, policy)["errors"])
