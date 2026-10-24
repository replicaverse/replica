"""The precision flags reach the runner and the pickle manifest's model identity."""

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from replica.compute import artifacts, cli, methods
from replica.compute.methods import ComputeContext


def _context(**kwargs):
    return replace(
        ComputeContext("gpt2", 5, 64, 10, 1.0, 0, True, "en", None), **kwargs
    )


def test_precision_options_are_passed_only_when_set():
    assert methods.precision(_context()) == {}
    assert methods.precision(_context(dtype="bfloat16")) == {"dtype": "bfloat16"}
    assert methods.precision(_context(device_map="auto")) == {"device_map": "auto"}


def test_the_surprisal_backend_receives_the_precision(monkeypatch):
    received = {}
    monkeypatch.setattr(
        methods, "surprisal_model", lambda model_id, **kwargs: received.update(kwargs)
    )
    methods.METHOD_SPECS["surprisal"].build_model(
        _context(dtype="float16", device_map="auto")
    )
    assert received["dtype"] == "float16" and received["device_map"] == "auto"
    received.clear()
    methods.METHOD_SPECS["surprisal"].build_model(_context())
    assert "dtype" not in received and "device_map" not in received


def test_identity_records_the_precision_only_when_set(monkeypatch):
    from transformers import AutoConfig

    monkeypatch.setattr(artifacts, "resolve_model_bundle", lambda *a: ("snapshot", {}))
    monkeypatch.setattr(artifacts, "_start_record", lambda *a: {"ids": [5]})
    monkeypatch.setattr(
        AutoConfig, "from_pretrained", lambda *a: SimpleNamespace(n_positions=2048)
    )
    _, plain = artifacts.resolve_model_snapshot("surprisal", _context())
    _, half = artifacts.resolve_model_snapshot(
        "surprisal", _context(dtype="bfloat16", device_map="auto")
    )
    assert "precision" not in plain
    assert half["precision"] == {"dtype": "bfloat16", "device_map": "auto"}
    assert plain != half


def test_the_command_line_forwards_the_flags(tmp_path, monkeypatch):
    received = {}

    def fake_compute_for(*args, **kwargs):
        received.update(kwargs)
        raise RuntimeError("stop after recording")

    monkeypatch.setattr(cli, "expand_datasets", lambda name: ["fake"])
    monkeypatch.setattr(cli, "compute_for", fake_compute_for)
    manifest = tmp_path / "run.json"
    args = [
        "--dataset",
        "fake",
        "--lm",
        "gpt2",
        "--out-dir",
        str(tmp_path),
        "--run-manifest",
        str(manifest),
        "--dtype",
        "bfloat16",
        "--device-map",
        "auto",
    ]
    with pytest.raises(SystemExit):
        cli.main(args)
    assert received["dtype"] == "bfloat16" and received["device_map"] == "auto"
    assert json.loads(manifest.read_text())["outcomes"][0]["status"] == "failed"
