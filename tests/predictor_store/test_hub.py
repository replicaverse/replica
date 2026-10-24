"""The published-store reader, against stand-in hub clients."""

from types import SimpleNamespace

import pytest
from huggingface_hub.errors import OfflineModeIsEnabled

from replica.external import hub as assets
from replica.predictor_store.hub import Hub


class _OfflineApi:
    """A client whose repository listing fails as it does offline."""

    def list_repo_tree(self, *args, **kwargs):
        raise OfflineModeIsEnabled("offline")


def test_a_failed_listing_raises_rather_than_reading_the_cache(tmp_path, monkeypatch):
    """The local cache holds only the files downloaded before."""
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "tags.parquet").write_bytes(b"index")
    monkeypatch.setattr("huggingface_hub.HfApi", _OfflineApi)
    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download",
        lambda repo, filename, **kwargs: str(snapshot / filename),
    )
    hub = Hub("example/predictors", "a" * 40)
    with pytest.raises(OSError, match="cannot list the files of example/predictors"):
        hub.files()


def test_the_head_of_main_resolves_as_the_reader_resolves_it(monkeypatch):
    """The export and the predictor reader share one revision resolver."""
    calls = []

    class Api:
        def repo_info(self, repo, **kwargs):
            calls.append((repo, kwargs))
            return SimpleNamespace(sha="b" * 40)

    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.setattr(assets, "HfApi", Api)
    hub = Hub("example/predictors")
    assert hub.revision() == hub.revision() == "b" * 40
    assert calls == [
        ("example/predictors", {"repo_type": "dataset", "revision": "main"})
    ]


def test_offline_the_cached_index_names_the_commit(tmp_path, monkeypatch):
    snapshot = tmp_path / "snapshots" / ("c" * 40)
    snapshot.mkdir(parents=True)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(
        assets,
        "try_to_load_from_cache",
        lambda repo, filename, **kwargs: str(snapshot / filename),
    )
    assert Hub("example/predictors").revision() == "c" * 40
    monkeypatch.setattr(assets, "try_to_load_from_cache", lambda *a, **k: None)
    with pytest.raises(OSError, match="is not cached"):
        Hub("example/predictors").revision()


def test_a_revision_passed_to_the_hub_must_be_a_commit():
    with pytest.raises(ValueError, match="did not resolve to a commit"):
        Hub("example/predictors", "main").revision()
