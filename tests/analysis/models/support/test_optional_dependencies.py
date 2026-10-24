"""Missing R dependencies report the supported extra and retain the cause."""

import builtins

import pytest

from replica.analysis.models.support.r import _r_context


@pytest.mark.parametrize(
    "failure",
    [
        ModuleNotFoundError("No module named 'rpy2'"),
        ImportError("R shared library failed"),
    ],
)
def test_r_installation_failure_preserves_diagnostics(monkeypatch, failure):
    original = builtins.__import__

    def unavailable(name, *args, **kwargs):
        if name == "rpy2.robjects":
            raise failure
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", unavailable)
    with pytest.raises(ImportError, match=r"pip install -e '\.\[r\]'") as error:
        with _r_context():
            pytest.fail("The missing backend must not run")
    assert error.value.__cause__ is failure
