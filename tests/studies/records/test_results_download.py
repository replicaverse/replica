"""Pinned Hub transfers recover transient failures without masking corrupt results."""

from importlib import import_module
from types import SimpleNamespace
from unittest.mock import Mock

import huggingface_hub
import pytest
from huggingface_hub.utils import (
    HfHubHTTPError,
    LocalEntryNotFoundError,
    OfflineModeIsEnabled,
    get_session,
)

from replica.studies.records import cli as results_cli
from replica.studies.records import files
from replica.studies.records import store as results_store

if type(get_session()).__module__.startswith("requests"):
    from requests.exceptions import (
        ConnectionError as NetworkError,
    )
    from requests.exceptions import (
        HTTPError as HTTPStatusError,
    )
    from requests.exceptions import (
        ProxyError,
        ReadTimeout,
        SSLError,
    )
else:
    from ssl import SSLError

    client = import_module(type(get_session()).__module__.split(".")[0])
    NetworkError = client.ConnectError
    HTTPStatusError = client.HTTPStatusError
    ProxyError = client.ProxyError
    ReadTimeout = client.ReadTimeout


REMOTE = "studies/example/results/reproduction/repetition/reading_time/run/cells.tsv"
REVISION = "a" * 40


def _http_failure(status, backend="hub"):
    response = SimpleNamespace(status_code=status, headers={}, request=None)
    if backend == "native":
        return HTTPStatusError(
            "sensitive request details", request=None, response=response
        )
    return HfHubHTTPError("sensitive request details", response=response)


def _wrapped(cause):
    error = LocalEntryNotFoundError("not cached")
    error.__cause__ = cause
    return error


@pytest.fixture
def transfer(monkeypatch):
    download = Mock()
    sleep = Mock()
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    monkeypatch.setattr("time.sleep", sleep)
    return download, sleep


@pytest.mark.parametrize(
    "error",
    [
        _wrapped(_http_failure(503)),
        _wrapped(_http_failure(429)),
        _http_failure(502),
        _wrapped(_http_failure(503, "native")),
        _wrapped(_http_failure(504, "native")),
        _wrapped(ReadTimeout("sensitive request details")),
        _wrapped(NetworkError("sensitive request details")),
        NetworkError("sensitive request details"),
        ReadTimeout("sensitive request details"),
        TimeoutError("sensitive request details"),
    ],
)
def test_transient_download_retries_same_artifact(transfer, caplog, error):
    download, sleep = transfer
    download.side_effect = [error, "/cache/cells.tsv"]
    assert files._hub_download(REMOTE, revision=REVISION) == "/cache/cells.tsv"
    assert download.call_count == 2
    first, second = download.call_args_list
    assert first == second
    assert first.kwargs == {
        "repo_id": files.HF_REPO,
        "filename": REMOTE,
        "repo_type": "dataset",
        "revision": REVISION,
        "force_download": False,
    }
    sleep.assert_called_once_with(5)
    assert REMOTE in caplog.text and REVISION in caplog.text
    assert "sensitive request details" not in caplog.text


def test_exhausted_download_identifies_file_revision_and_cause(transfer, caplog):
    download, sleep = transfer
    error = _wrapped(_http_failure(503))
    download.side_effect = error
    with pytest.raises(FileNotFoundError) as caught:
        files._hub_download(REMOTE, revision=REVISION)
    assert download.call_count == 3
    assert [call.args[0] for call in sleep.call_args_list] == [5, 10]
    message = str(caught.value)
    assert REMOTE in message and REVISION in message
    assert "LocalEntryNotFoundError caused by HfHubHTTPError (HTTP 503)" in message
    assert "attempt 3/3" in message
    assert caught.value.__cause__ is error
    assert "sensitive request details" not in message + caplog.text


@pytest.mark.parametrize(
    "error",
    [
        _http_failure(401),
        _http_failure(403),
        _wrapped(_http_failure(404)),
        _http_failure(400),
        _wrapped(_http_failure(401, "native")),
        _wrapped(_http_failure(404, "native")),
        _wrapped(OfflineModeIsEnabled("offline")),
        LocalEntryNotFoundError("not cached"),
        SSLError("invalid certificate"),
        ProxyError("invalid proxy"),
    ],
)
def test_permanent_download_failure_is_not_retried(transfer, error):
    download, sleep = transfer
    download.side_effect = error
    with pytest.raises(OSError, match=REMOTE) as caught:
        files._hub_download(REMOTE, revision=REVISION)
    if isinstance(error, FileNotFoundError):
        assert isinstance(caught.value, FileNotFoundError)
    download.assert_called_once()
    sleep.assert_not_called()


def test_unrelated_errors_are_not_reclassified(transfer):
    download, sleep = transfer
    error = TypeError("invalid argument")
    download.side_effect = error
    with pytest.raises(TypeError) as caught:
        files._hub_download(REMOTE, revision=REVISION)
    assert caught.value is error
    download.assert_called_once()
    sleep.assert_not_called()


@pytest.mark.parametrize(
    "error", [_http_failure(401), _wrapped(OfflineModeIsEnabled("offline"))]
)
def test_cli_reports_download_context_for_io_failures(transfer, monkeypatch, error):
    download, sleep = transfer
    download.side_effect = error

    def fetch(*args, **kwargs):
        files._hub_download(REMOTE, revision=REVISION)

    monkeypatch.setattr(results_store, "fetch", fetch)
    with pytest.raises(SystemExit) as caught:
        results_cli.main(["fetch", "example2026"])
    message = str(caught.value)
    assert message.startswith("replica results fetch: Cannot download")
    assert REMOTE in message and REVISION in message
    assert "sensitive request details" not in message
    download.assert_called_once()
    sleep.assert_not_called()


def test_recovered_transfer_still_checks_recorded_digest(
    transfer, tmp_path, monkeypatch
):
    download, sleep = transfer
    downloaded = tmp_path / "cells.tsv"
    downloaded.write_text("altered contents\n")
    download.side_effect = [_wrapped(_http_failure(503)), str(downloaded)]
    monkeypatch.setattr(files, "read_index", lambda study: {"hub_revision": REVISION})
    with pytest.raises(ValueError, match="the hub copy has digest"):
        files._fetch(
            "example2026",
            "reproduction/repetition/reading_time/r0123456789ab",
            "cells.tsv",
            "0" * 64,
        )
    assert download.call_count == 2
    sleep.assert_called_once_with(5)
