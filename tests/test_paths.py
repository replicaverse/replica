"""Checkout resolution and the writable user roots that follow the XDG rules."""

from pathlib import Path

import pytest

from replica.paths import checkout_root, user_cache_root, user_data_root

_ROOTS = [
    (user_data_root, "XDG_DATA_HOME", ".local/share"),
    (user_cache_root, "XDG_CACHE_HOME", ".cache"),
]


@pytest.mark.parametrize(("root", "variable", "default"), _ROOTS)
@pytest.mark.parametrize("value", ["", "relative/dir"])
def test_an_empty_or_relative_value_counts_as_unset(
    monkeypatch, root, variable, default, value
):
    monkeypatch.setenv(variable, value)
    assert root() == Path.home() / default / "replica"


@pytest.mark.parametrize(("root", "variable", "default"), _ROOTS)
def test_an_absolute_value_is_the_base(monkeypatch, tmp_path, root, variable, default):
    monkeypatch.setenv(variable, str(tmp_path))
    assert root() == tmp_path / "replica"
    monkeypatch.delenv(variable)
    assert root() == Path.home() / default / "replica"


@pytest.mark.parametrize(("root", "variable"), [pair[:2] for pair in _ROOTS])
def test_a_home_relative_value_expands(monkeypatch, root, variable):
    monkeypatch.setenv(variable, "~/base")
    assert root() == Path.home() / "base" / "replica"


def test_resolves_to_source_checkout(monkeypatch):
    monkeypatch.delenv("REPLICA_CHECKOUT", raising=False)
    root = checkout_root()
    assert (root / "pyproject.toml").is_file()
    assert (root / "scripts").is_dir()


def test_env_override_wins(monkeypatch, tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\n")
    (tmp_path / "src/replica").mkdir(parents=True)
    monkeypatch.setenv("REPLICA_CHECKOUT", str(tmp_path))
    assert checkout_root() == tmp_path.resolve()


def test_env_override_must_be_a_checkout(monkeypatch, tmp_path):
    monkeypatch.setenv("REPLICA_CHECKOUT", str(tmp_path))
    with pytest.raises(RuntimeError, match="checkout"):
        checkout_root()


def test_walk_resolves_to_this_checkout_root(monkeypatch):
    monkeypatch.delenv("REPLICA_CHECKOUT", raising=False)
    assert checkout_root() == Path(__file__).resolve().parents[1]
