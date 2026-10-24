"""Execute the getting-started feature inspection and default analysis."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

import replica
import replica.feature
from replica.analysis import AnalysisConfig, evaluate
from tests.analysis._fixtures import VaryingLM, make_toy_dataset
from tests.analysis.inference.test_crossed_bootstrap import GaussianBackend


def test_getting_started_configuration_runs_with_ordinary_defaults(monkeypatch):
    path = Path(__file__).resolve().parents[2] / "docs/tutorials/getting_started.ipynb"
    notebook = json.loads(path.read_text())
    cells = [
        "".join(c["source"])
        for c in notebook["cells"]
        if c["cell_type"] == "code"
        and any(
            operation in "".join(c["source"])
            for operation in ("responses = resolve", "results = evaluate")
        )
    ]
    monkeypatch.setattr(replica.feature, "HuggingFaceLM", lambda _: VaryingLM())
    observed = []

    def run(dataset, features, config=None):
        config = config or AnalysisConfig()
        observed.append(config)
        return evaluate(dataset, features, replace(config, model=GaussianBackend()))

    monkeypatch.setattr(replica, "evaluate", run)
    namespace = {
        "ds": make_toy_dataset(
            n_participants=10,
            n_stimuli=10,
            measures=["first_fixation", "first_run_in_first_pass"],
        )
    }
    for cell in cells:
        exec(compile(cell, str(path), "exec"), namespace)
    records = namespace["ds"].data("first_run_in_first_pass")
    for block, record in zip(namespace["responses"].blocks, records, strict=True):
        np.testing.assert_array_equal(block.values, record.values)
    assert len(observed) == 1
    config = observed[0]
    assert config.model.reml is False
    assert config.resolve_cv().seed == 42
    assert config.design.comparison is None
    assert config.design.scoring == "marginal"
    assert len(namespace["results"]) == 2
    assert all(np.isfinite(result.delta_ll_per_unit) for result in namespace["results"])
