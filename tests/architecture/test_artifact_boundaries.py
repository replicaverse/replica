"""Readers can validate manifests without loading producer dependencies."""

import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from scripts.checks.source.import_boundaries import _check

_ROOT = Path(__file__).resolve().parents[2]


def test_manifest_round_trip_without_site_packages(tmp_path):
    probe = """
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from replica.artifacts import (
    SCHEMA_VERSION, matching_manifest, validate_artifact, write_manifest,
)
path = Path(sys.argv[2]) / 'payload'
path.write_bytes(b'computed values')
identity = {'schema_version': SCHEMA_VERSION, 'method': 'surprisal'}
write_manifest(path, identity, status='completed')
assert validate_artifact(path) == identity
assert matching_manifest(path, identity, status='completed')
path.write_bytes(b'corruption')
assert not matching_manifest(path, identity, status='completed')
try:
    validate_artifact(path)
except ValueError:
    pass
else:
    raise AssertionError('reader accepted corruption')
assert not any(name.startswith(('replica.compute', 'replica.feature'))
               for name in sys.modules)
"""
    subprocess.run(
        [sys.executable, "-S", "-c", probe, str(_ROOT / "src"), str(tmp_path)],
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("indirect", [False, True], ids=["direct", "via-helper"])
@pytest.mark.parametrize(
    "source,target,rule",
    [
        ("artifacts", "replica.compute.methods", "artifact-contract-is-independent"),
        ("artifacts", "numpy", "artifact-contract-is-independent"),
        (
            "dataset/loading/helper",
            "replica.dataset.measures.sources",
            "dataset-loading-does-not-import-measure-catalog",
        ),
        (
            "dataset/loading/helper",
            "replica.dataset.catalog.registry",
            "dataset-loading-helpers-have-no-orchestration-dependencies",
        ),
        (
            "dataset/catalog/registry",
            "replica.dataset.fetch",
            "dataset-catalog-does-not-execute-downloads",
        ),
        (
            "analysis/contracts/grouped",
            "replica.analysis.specification.config",
            "analysis-contracts-have-no-orchestration-dependencies",
        ),
        (
            "analysis/models/backend",
            "replica.analysis.data.events",
            "analysis-models-have-no-preparation-dependencies",
        ),
        (
            "feature/model",
            "replica.predictors",
            "feature-models-do-not-import-predictor-readers",
        ),
        (
            "predictors",
            "replica.compute.artifacts",
            "predictor-readers-have-no-producer-dependencies",
        ),
        (
            "predictors",
            "replica.feature",
            "predictor-readers-have-no-producer-dependencies",
        ),
    ],
)
def test_artifact_boundaries_reject_producer_dependencies(
    tmp_path, source, target, rule, indirect
):
    policy = tomllib.loads(
        (_ROOT / "scripts/checks/source/architecture.toml").read_text()
    )
    policy = {"rules": [r for r in policy["rules"] if r["name"] == rule]}
    assert policy["rules"], rule
    package = tmp_path / "src/replica"
    (package / "feature").mkdir(parents=True)
    (package / "feature/__init__.py").write_text(
        "from .frequency.registry import SOURCES\n"
    )
    (package / source).parent.mkdir(parents=True, exist_ok=True)
    (package / f"{source}.py").write_text(
        "from . import helper\n" if indirect else f"import {target}\n"
    )
    if indirect:
        (package / source).parent.joinpath("helper.py").write_text(f"import {target}\n")
    assert any(rule in error for error in _check(tmp_path, policy)["errors"])
