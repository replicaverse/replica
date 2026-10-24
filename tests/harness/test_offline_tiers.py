"""Local and CI selections share the same offline exclusions."""

import pytest

pytest_plugins = ["pytester"]


@pytest.mark.parametrize(
    ("arguments", "passed", "deselected"),
    [
        (("--offline-tier", "fast"), 1, 9),
        (("--offline-tier", "regression"), 3, 7),
        (("--offline-tier", "all"), 4, 6),
        (("--offline-tier", "all", "-m", "dependency"), 1, 9),
        ((), 10, 0),
    ],
)
def test_tier_selection_and_extra_marker_filter(
    pytester, arguments, passed, deselected
):
    pytester.makeconftest(
        "from tests.conftest import pytest_addoption, pytest_configure\n"
    )
    pytester.makeini("""
[pytest]
markers =
    artifact: retained artifacts
    statistical: statistical checks
    dependency: optional dependencies
    slow: long running checks
    needs_model: pretrained model
    needs_r: R backend
    needs_data: corpus data
    needs_network: network service
    needs_lmer_rs: Rust backend
""")
    pytester.makepyfile("""
import pytest
@pytest.mark.parametrize("case", [
    "fast",
    pytest.param("artifact", marks=pytest.mark.artifact),
    pytest.param("statistical", marks=pytest.mark.statistical),
    pytest.param("dependency", marks=pytest.mark.dependency),
    pytest.param("slow", marks=pytest.mark.slow),
    pytest.param("model", marks=pytest.mark.needs_model),
    pytest.param("r", marks=pytest.mark.needs_r),
    pytest.param("data", marks=pytest.mark.needs_data),
    pytest.param("network", marks=pytest.mark.needs_network),
    pytest.param("rust", marks=pytest.mark.needs_lmer_rs),
])
def test_selected(case): pass
""")
    result = pytester.runpytest(*arguments, "--strict-markers", "-q")
    result.assert_outcomes(passed=passed, deselected=deselected)
