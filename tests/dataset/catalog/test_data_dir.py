"""Pin the ``PSYCHO_DATA_DIR`` environment-variable contract.

REPLICA resolves on-disk dataset locations through
``replica.dataset.locations.data_dir()`` (also exported by the registry):

1. ``$PSYCHO_DATA_DIR`` if set → that path directly (expands ``~``).
2. Otherwise ``<repo>/datasets`` in a source checkout.
3. Otherwise ``$XDG_DATA_HOME/replica/datasets`` (or the user data default).

Loader paths come from ``DatasetRegistration.default_path`` which composes
``data_dir()`` with the spec's relative ``default_subdir``. This file
pins the resolution so future changes can't silently break the
checkout-vs-external-drive separation.
"""

from pathlib import Path

import pytest

from replica.dataset import locations
from replica.dataset.catalog.registry import REGISTRY, REPO_ROOT, compute_dir, data_dir


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("PSYCHO_DATA_DIR", raising=False)
    monkeypatch.delenv("PSYCHO_COMPUTE_DIR", raising=False)


def test_data_dir_defaults_to_repo_datasets():
    assert data_dir() == REPO_ROOT / "datasets"


@pytest.mark.parametrize("relative", [False, True])
def test_data_dir_env_var_override(monkeypatch, tmp_path, relative):
    monkeypatch.chdir(tmp_path)
    target = Path("external_data_root") if relative else tmp_path / "external_data_root"
    monkeypatch.setenv("PSYCHO_DATA_DIR", str(target))
    # Path doesn't have to exist; the resolver doesn't create it (that's
    # the fetcher's job).
    assert data_dir() == target.resolve()


@pytest.mark.parametrize("xdg", [None, "relative", "absolute"])
def test_installed_data_dir_uses_user_data(monkeypatch, tmp_path, xdg):
    monkeypatch.setattr(locations, "REPO_ROOT", None)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    if xdg is not None:
        monkeypatch.setenv(
            "XDG_DATA_HOME", str(tmp_path / "xdg") if xdg == "absolute" else xdg
        )
    expected = tmp_path / ("xdg" if xdg == "absolute" else ".local/share")
    assert data_dir() == expected / "replica/datasets"


def test_data_dir_env_var_expands_tilde(monkeypatch):
    monkeypatch.setenv("PSYCHO_DATA_DIR", "~/foo_dataset_root")
    assert str(data_dir()).startswith(str(Path.home()))


def test_default_subdir_strips_leading_datasets_prefix():
    """Every registered spec's default_subdir must not start with
    'datasets/'; that prefix is added by ``data_dir()``. If a spec kept
    it, a ``$PSYCHO_DATA_DIR`` setting would produce
    ``<repo>/datasets/datasets/...`` double-prefixing."""
    for name, spec in REGISTRY.items():
        assert not spec.default_subdir.startswith("datasets/"), (
            f"{name}: default_subdir {spec.default_subdir!r} still has "
            "the 'datasets/' prefix, it must be relative to data_dir()."
        )


def test_default_path_uses_env_var(monkeypatch, tmp_path):
    target = tmp_path / "external"
    monkeypatch.setenv("PSYCHO_DATA_DIR", str(target))
    spec = REGISTRY["provo"]
    assert spec.default_path() == target.resolve() / spec.default_subdir


def test_default_path_repo_root_kwarg(monkeypatch, tmp_path):
    """``repo_root=tmp_path`` roots the dataset path at an explicit checkout;
    fixture loading depends on this semantic.
    """
    monkeypatch.setenv("PSYCHO_DATA_DIR", str(tmp_path / "ignored"))
    spec = REGISTRY["provo"]
    p = spec.default_path(repo_root=tmp_path)
    assert p == tmp_path / "datasets" / spec.default_subdir


def test_compute_dir_defaults_to_repo_compute():
    assert compute_dir() == REPO_ROOT / "compute"


def test_compute_dir_env_var_override(monkeypatch, tmp_path):
    target = tmp_path / "external_compute_root"
    monkeypatch.setenv("PSYCHO_COMPUTE_DIR", str(target))
    assert compute_dir() == target.resolve()


def test_data_and_compute_dirs_are_independent(monkeypatch, tmp_path):
    """Setting one env var must not affect the other, raw data often
    lives on slow disk and compute caches on fast SSD."""
    data_target = tmp_path / "external_data"
    monkeypatch.setenv("PSYCHO_DATA_DIR", str(data_target))
    # PSYCHO_COMPUTE_DIR not set → falls back to <repo>/compute
    assert data_dir() == data_target.resolve()
    assert compute_dir() == REPO_ROOT / "compute"


def test_compute_lm_features_uses_compute_dir_default(monkeypatch, tmp_path):
    """The feature producer resolves its default through ``compute_dir()``."""
    target = tmp_path / "external_compute"
    monkeypatch.setenv("PSYCHO_COMPUTE_DIR", str(target))

    from replica.compute import cli as compute_cli

    assert compute_cli.compute_dir is compute_dir
    assert compute_cli.compute_dir() == target.resolve()


def test_fetch_dataset_imports_data_dir():
    """The fetch module must resolve its destination via the same
    ``data_dir`` the registry uses, so a single env var controls both the
    downloader's destination and the loader's read path. If they ever
    diverge, fetched files land somewhere loaders don't look.
    """
    from replica.dataset import fetch as fetch_mod

    # fetch imports it under the local name ``_data_dir``.
    assert fetch_mod._data_dir is data_dir
    assert data_dir is locations.data_dir
