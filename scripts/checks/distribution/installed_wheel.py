"""Exercise an installed wheel from outside its checkout and test environment."""

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from importlib.metadata import version
from importlib.resources import files
from importlib.resources.abc import Traversable
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from replica import (
    OLS,
    AnalysisConfig,
    EvaluationSpec,
    FeatureModelOutput,
    ModelFeature,
    PrecomputedFeature,
    PrecomputedFeatureModel,
    RowSplitter,
    TabularDataset,
    TabularDatasetConfig,
    __version__,
    evaluate,
    prepare_observations,
)
from replica.dataset.catalog.registry import compute_dir, data_dir
from replica.external import EXTERNAL_REPOS, patch_path, resolve_data_root
from replica.paths import source_checkout


def _check_r_resources(directory: Traversable) -> int:
    count = 0
    for resource in directory.iterdir():
        if resource.is_dir():
            count += _check_r_resources(resource)
        elif resource.name.lower().endswith(".r"):
            resource.read_bytes()
            count += 1
    return count


def _check_installed_inputs(studies: Traversable) -> int:
    """Count the installed study inputs, each checked against its index digest."""
    count = 0
    for study in studies.iterdir():
        index = study.joinpath("results", "index.json")
        if not index.is_file():
            continue
        resources = json.loads(index.read_text(encoding="utf-8")).get("resources", {})
        for name, resource in resources.items():
            blob = study.joinpath(*name.split("/"))
            if resource.get("role") != "input" or not blob.is_file():
                continue
            digest = hashlib.sha256(blob.read_bytes()).hexdigest()
            assert digest == resource["sha256"], (
                f"{study.name}: {name} differs from its index"
            )
            count += 1
    return count


def main() -> None:
    """Check base imports, evaluation, resources, paths and the installed CLI."""
    assert source_checkout() is None, "smoke test must run outside a source checkout"
    assert importlib.util.find_spec("torch") is None, (
        "wheel smoke must use base dependencies only"
    )
    assert data_dir() == Path(os.environ["XDG_DATA_HOME"]) / "replica/datasets"
    assert compute_dir() == Path(os.environ["XDG_CACHE_HOME"]) / "replica/compute"
    assert resolve_data_root() == Path(os.environ["XDG_DATA_HOME"]) / "replica"
    root = files("replica")
    # The separate source-to-wheel check verifies completeness. Here, exercise
    # resource access through the installed package without a second inventory.
    assert _check_r_resources(root) > 0, "No installed R resources found"
    assert root.joinpath("__init__.pyi").is_file()
    assert _check_installed_inputs(root.joinpath("studies")) > 0, (
        "No study input installed"
    )
    for name, dependencies in EXTERNAL_REPOS.items():
        for dependency in dependencies:
            if dependency.patch:
                patch = patch_path(dependency)
                assert patch.is_file(), f"{name}: missing installed patch {patch}"
                assert patch.read_bytes(), f"{name}: empty installed patch {patch}"
    x = np.arange(30, dtype=float)
    frame = pd.DataFrame(
        dict(trialid=np.arange(30), ianum=0, ia="word", rt=x * 3 + np.sin(x))
    )
    dataset = TabularDataset(
        frame, TabularDatasetConfig(name="smoke", measure_col="rt")
    )
    model = PrecomputedFeatureModel(
        "smoke",
        {
            f"trial_{i}": [FeatureModelOutput(values={"x": value})]
            for i, value in enumerate(x)
        },
    )
    spec = EvaluationSpec(
        targets=[dataset.feature("rt")],
        predictors=[ModelFeature(model, PrecomputedFeature("x", "x"))],
    )
    config = AnalysisConfig(
        model=OLS(), design=None, cv=RowSplitter(3), transform_dv=lambda values: values
    )
    (result,) = evaluate(dataset, spec, config)
    assert result.n_scored == 30 and result.delta_ll > 0
    assert prepare_observations(dataset, spec, config).y.shape == (30,)
    cli = Path(sys.executable).with_name(
        "replica.exe" if os.name == "nt" else "replica"
    )
    installed_version = subprocess.run(
        [str(cli), "version"], check=True, capture_output=True, text=True
    )
    assert installed_version.stdout.strip() == version("replicaverse")
    assert __version__ == version("replicaverse"), (
        "package and metadata versions differ"
    )
    subprocess.run(
        [str(cli), "datasets", "--loadable"], check=True, capture_output=True
    )
    check = subprocess.run(
        [str(cli), "check"], capture_output=True, text=True, check=False
    )
    assert check.returncode != 0 and "source checkout" in check.stderr
    with TemporaryDirectory() as store:
        environment = {**os.environ, "REPLICA_RESULTS_STORE": store}
        listing = subprocess.run(
            [str(cli), "results", "list"],
            env=environment,
            capture_output=True,
            text=True,
            check=True,
        )
        assert listing.stdout == ""
        missing = subprocess.run(
            [str(cli), "results", "verify", "missing-run"],
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        assert missing.returncode != 0
        assert "replica results verify:" in missing.stderr
        assert "Traceback" not in missing.stderr
    print("Installed wheel: base evaluation, resources, paths and CLI passed")


if __name__ == "__main__":
    main()
