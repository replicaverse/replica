"""Every successful fast CI job includes the R-backed tests."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def test_r_tier_is_required_in_every_fast_job():
    workflow = yaml.safe_load((ROOT / ".github/workflows/quality.yml").read_text())
    job = workflow["jobs"]["tests"]
    assert {"tier": "fast"} in job["strategy"]["matrix"]["include"]
    assert "if" not in job
    assert not job.get("continue-on-error", False)

    steps = job["steps"]
    install = next(step for step in steps if "r-base-core" in step.get("run", ""))
    tests = next(step for step in steps if "needs_r" in step.get("run", ""))
    assert steps.index(install) < steps.index(tests)
    for step in (install, tests):
        assert step["if"] == "matrix.tier == 'fast'"
        assert not step.get("continue-on-error", False)
    assert "python -m pytest tests/" in tests["run"]
    assert "--cov-append" in tests["run"]
