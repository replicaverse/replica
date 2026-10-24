"""Data-availability helpers skip on an absent path and let loader errors fail."""

import pytest

from tests._require import load_or_skip


def _raised(call):
    """Run *call* and return what it raised, a skip included."""
    try:
        call()
    except BaseException as exc:  # a skip is the outcome under test
        return exc
    return None


class _Loader:
    def __init__(self, error=None):
        self.error = error
        self.loaded_from = None

    def load(self, path):
        if self.error is not None:
            raise self.error
        self.loaded_from = path


def test_load_or_skip_skips_an_absent_path(tmp_path):
    with pytest.raises(pytest.skip.Exception, match="data not present"):
        load_or_skip(_Loader(), tmp_path / "absent")


@pytest.mark.parametrize(
    "error", [FileNotFoundError("member missing"), ImportError("no reader")]
)
def test_load_or_skip_propagates_a_loader_error(tmp_path, error):
    assert _raised(lambda: load_or_skip(_Loader(error), tmp_path)) is error


def test_load_or_skip_loads_a_present_path(tmp_path):
    assert load_or_skip(_Loader(), tmp_path).loaded_from == str(tmp_path)


def test_registered_smoke_load_skips_only_an_absent_directory(tmp_path):
    from tests.integration.test_all_datasets import _load_registered

    with pytest.raises(pytest.skip.Exception, match="provo data not found"):
        _load_registered("provo", tmp_path)
    (tmp_path / "eye-tracking" / "original" / "provo").mkdir(parents=True)
    assert isinstance(_raised(lambda: _load_registered("provo", tmp_path)), OSError)
