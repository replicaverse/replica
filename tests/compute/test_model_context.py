"""Context overrides reach scoring and cannot reuse another window's identity."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from replica.compute import artifacts, cli, methods
from replica.compute.methods import ComputeContext


def _context(**kwargs):
    return replace(
        ComputeContext("ai-forever/mGPT", 5, 64, 10, 1.0, 0, True, "en", None),
        **kwargs,
    )


@pytest.mark.parametrize("method", ["surprisal", "char_beam"])
def test_snapshot_records_the_effective_window(monkeypatch, method):
    from transformers import AutoConfig

    monkeypatch.setattr(artifacts, "resolve_model_bundle", lambda *a: ("snapshot", {}))
    monkeypatch.setattr(artifacts, "_start_record", lambda *a: {"ids": [5]})
    monkeypatch.setattr(
        AutoConfig, "from_pretrained", lambda *a: SimpleNamespace(n_positions=2048)
    )
    _, default = artifacts.resolve_model_snapshot(method, _context())
    assert default["context_window"] == 512
    with pytest.warns(UserWarning, match="1,?024-token context"):
        _, override = artifacts.resolve_model_snapshot(
            method, _context(context_window=1024)
        )
    assert override["context_window"] == 1024
    assert default != override


def test_snapshot_preserves_previously_recorded_settings(monkeypatch):
    from transformers import AutoConfig, AutoTokenizer

    from tests.feature.backends.test_start_state import _mgpt

    monkeypatch.setattr(
        artifacts, "resolve_model_bundle", lambda *a: ("ai-forever/mGPT", {})
    )
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *a: _mgpt())
    monkeypatch.setattr(
        AutoConfig, "from_pretrained", lambda *a: SimpleNamespace(n_positions=2048)
    )
    context = _context(model_identity={"start": {"ids": [0]}, "context_window": 1024})
    with pytest.warns(UserWarning):
        _, identity = artifacts.resolve_model_snapshot("surprisal", context)
    assert identity["start"]["ids"] == [0]
    assert identity["context_window"] == 1024


def test_snapshot_requires_and_records_a_window_for_an_unknown_model(monkeypatch):
    from transformers import AutoConfig

    monkeypatch.setattr(artifacts, "resolve_model_bundle", lambda *a: ("snapshot", {}))
    monkeypatch.setattr(artifacts, "_start_record", lambda *a: {"ids": [0]})
    monkeypatch.setattr(AutoConfig, "from_pretrained", lambda *a: SimpleNamespace())
    context = _context(model_id="new/model")
    with pytest.raises(ValueError, match="no usable context limit.*context_window"):
        artifacts.resolve_model_snapshot("surprisal", context)
    _, identity = artifacts.resolve_model_snapshot(
        "surprisal", replace(context, context_window=64)
    )
    assert identity["context_window"] == 64


def test_empty_prefix_is_validated_against_the_resolved_revision(monkeypatch):
    from transformers import AutoConfig, AutoTokenizer

    loads = []

    def tokenizer(path):
        loads.append(path)
        return SimpleNamespace(name_or_path=path, bos_token_id=None)

    monkeypatch.setattr(
        artifacts, "resolve_model_bundle", lambda *a: ("pinned-snapshot", {})
    )
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", tokenizer)
    monkeypatch.setattr(
        AutoConfig, "from_pretrained", lambda *a: SimpleNamespace(n_positions=512)
    )
    _, identity = artifacts.resolve_model_snapshot(
        "surprisal",
        _context(
            model_id="model-without-bos",
            model_revision="pinned-revision",
            model_identity={"start": {"ids": []}},
        ),
    )
    assert loads and set(loads) == {"pinned-snapshot"}
    assert identity["start"]["ids"] == []


@pytest.mark.parametrize("method", ["modelblocks_surprisal", "word_information_value"])
def test_unsupported_window_is_rejected_before_loading(method):
    with pytest.raises(ValueError, match="does not support context_window"):
        methods.build_model(method, _context(context_window=1024))
    with pytest.raises(ValueError, match="does not support context_window"):
        artifacts.resolve_model_snapshot(method, _context(context_window=1024))


def test_recorded_window_reaches_the_builder_and_refuses_conflicts(monkeypatch):
    context = _context(model_identity={"context_window": 512})
    model = methods.build_model("surprisal", context)
    assert model.context_window == 512
    with pytest.raises(ValueError, match="disagrees with the model identity"):
        methods.build_model("surprisal", replace(context, context_window=1024))
    captured = {}
    monkeypatch.setattr(
        methods, "surprisal_model", lambda *a, **kw: captured.update(kw)
    )
    methods.build_model("char_beam", context)
    assert captured["context_window"] == 512


def test_unresolved_identity_window_is_refused():
    with pytest.raises(ValueError, match="requires a resolved context_window"):
        methods.build_model(
            "surprisal", _context(model_identity={"context_window": None})
        )


@pytest.mark.parametrize("window", [3, 4])
def test_scoring_uses_requested_window_and_records_it(window):
    import torch

    from tests._stimuli import word_stimulus
    from tests.feature.core.annotation.test_token_tables import _lm

    lm = _lm(limit=8)
    lm.context_window = window
    calls = []
    original = lm._model.__class__.__call__

    class RecordingModel(lm._model.__class__):
        def __call__(self, input_ids):
            calls.append(input_ids.shape[-1])
            return original(self, input_ids)

    lm._model = RecordingModel(8)
    stimulus = word_stimulus("ab cd ef gh ij kl mn op")
    outputs = lm.score_tokens(stimulus)
    assert max(calls) <= window and len(calls) > 1
    assert all(row.metadata["context_window"] == window for row in outputs)
    assert all(
        torch.isfinite(torch.tensor(row.values["log_p_observed"])) for row in outputs
    )
    assert lm.window()["max_positions"] == window


def test_cli_records_the_window_used_for_scoring(monkeypatch, tmp_path):
    from tests.feature.core.annotation.test_token_tables import (
        _install,
        _lm,
        _load,
        _manifest,
    )

    def builder(context):
        model = _lm()
        model.context_window = context.context_window
        return model

    _install(monkeypatch, tmp_path, builder=builder)
    monkeypatch.setattr(cli, "expand_datasets", lambda name: [name])
    cli.main(
        [
            "--dataset",
            "corpus",
            "--lm",
            "test-tokens",
            "--method",
            "surprisal",
            "--unit",
            "tokens",
            "--context-window",
            "4",
            "--out-dir",
            str(tmp_path),
        ]
    )
    path = tmp_path / "test-tokens_surprisal_tokens_corpus.pkl"
    assert _manifest(path)["identity"]["options"]["context_window"] == 4
    assert {
        row.metadata["context_window"] for rows in _load(path).values() for row in rows
    } == {4}
