"""``--require-external`` turns every skip of a selected test into a failure."""

import pytest

pytest_plugins = ["pytester"]

_CONFTEST = """
from tests.conftest import (
    pytest_addoption,
    pytest_make_collect_report,
    pytest_runtest_makereport,
)
"""


def _run(pytester, *args):
    pytester.makeconftest(_CONFTEST)
    pytester.makepyfile(
        test_module_skip="""
import pytest
pytest.importorskip("replica_absent_module")
def test_unreachable(): pass
""",
        test_function_skip="""
import pytest
def test_function():
    pytest.importorskip("replica_absent_module")
""",
        test_present="def test_present(): pass\n",
    )
    return pytester.runpytest(*args)


def test_a_module_skipped_at_import_fails_the_run(pytester):
    result = _run(
        pytester, "--require-external", "test_module_skip.py", "test_present.py"
    )
    assert result.ret == pytest.ExitCode.INTERRUPTED
    result.stdout.fnmatch_lines(["*Required module was skipped: test_module_skip.py*"])


def test_a_test_skipped_in_its_body_fails(pytester):
    result = _run(pytester, "--require-external", "test_function_skip.py")
    assert result.ret == pytest.ExitCode.TESTS_FAILED
    result.stdout.fnmatch_lines(
        ["*Required test was skipped: test_function_skip.py::test_function*"]
    )


def test_skips_stay_skips_without_the_flag(pytester):
    result = _run(pytester)
    assert result.ret == pytest.ExitCode.OK
    result.assert_outcomes(passed=1, skipped=2)
