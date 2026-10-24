"""Requests the command line refuses, and the stimulus slice it clamps."""

import json
from dataclasses import replace

import pytest

from replica.compute import cli, runner
from replica.compute.methods import ComputeContext

from .test_compute_lm_features_checkpoint import (
    _install_fake_dataset,
    _install_fake_extract_keys,
    _install_fake_lm,
    _install_fake_tag,
)


def _outcomes(tmp_path, *extra, code=1):
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
        *extra,
    ]
    if code:
        with pytest.raises(SystemExit) as error:
            cli.main(args)
        assert error.value.code == code
    else:
        cli.main(args)
    return json.loads(manifest.read_text())["outcomes"]


@pytest.fixture
def fake_registry(monkeypatch):
    monkeypatch.setattr(cli, "expand_datasets", lambda name: ["fake"])
    monkeypatch.setattr(
        runner,
        "resolve_model_snapshot",
        lambda method, context: (context.model_id, {"revision": "test-commit"}),
    )
    with _install_fake_dataset(runner, ["s0", "s1"]):
        yield


def test_a_unit_bound_method_refuses_a_token_table(tmp_path, fake_registry):
    [outcome] = _outcomes(tmp_path, "--method", "mc_word_entropy", "--unit", "tokens")
    assert outcome["status"] == "failed"
    assert "token-unit methods" in outcome["error"]


def test_a_positional_method_refuses_a_unit_rule(tmp_path, fake_registry):
    [outcome] = _outcomes(
        tmp_path, "--method", "surprisal", "--unit-rule", "whitespace_continuation"
    )
    assert outcome["status"] == "failed"
    assert "positional and takes no unit rule" in outcome["error"]


def test_a_slice_past_the_corpus_end_is_clamped(tmp_path, fake_registry):
    with (
        _install_fake_lm(runner),
        _install_fake_extract_keys(runner),
        _install_fake_tag(runner),
    ):
        [outcome] = _outcomes(
            tmp_path,
            "--method",
            "modelblocks_surprisal",
            "--stim-slice",
            "1:999",
            code=0,
        )
    assert outcome["status"] == "completed"
    assert outcome["path"].endswith("fake_cell.shard00001_00002.pkl")


def test_a_slice_beyond_the_corpus_selects_nothing(tmp_path, fake_registry):
    [outcome] = _outcomes(
        tmp_path, "--method", "modelblocks_surprisal", "--stim-slice", "5:9"
    )
    assert outcome["status"] == "failed"
    assert "No stimuli selected" in outcome["error"]


def test_slice_bounds_are_clamped_to_the_corpus():
    assert runner._apply_stimulus_slice(["a", "b"], 2, (-3, 999)) == (
        ["a", "b"],
        ".shard00000_00002",
    )
    assert runner._apply_stimulus_slice(["a", "b"], 2, (1, 999)) == (
        ["b"],
        ".shard00001_00002",
    )
    assert runner._apply_stimulus_slice(["a", "b"], 2, (5, 9)) == (
        [],
        ".shard00002_00002",
    )


def test_layer_segmentation_names_only_the_cell_it_reaches():
    context = ComputeContext("gpt2", 5, 64, 10, 1.0, None, True, "en", None)
    layered = replace(context, segmentation="layer")
    assert runner.cache_stem("word_information_value", layered, "provo").endswith(
        "_layer_provo"
    )
    assert runner.cache_stem("surprisal", layered, "provo") == runner.cache_stem(
        "surprisal", context, "provo"
    )


@pytest.mark.parametrize(
    "method",
    ["first_token_entropy_buggy", "renyi_entropy_buggy", "continuation_entropy_buggy"],
)
def test_reproduction_methods_require_cli_and_python_opt_in(
    tmp_path, monkeypatch, method
):
    from replica.compute import methods

    def unexpected(*args, **kwargs):
        pytest.fail("A refused reproduction must not load data or models")

    monkeypatch.setattr(cli, "expand_datasets", unexpected)
    monkeypatch.setattr(runner, "_load_dataset", unexpected)
    monkeypatch.setattr(methods, "HuggingFaceLM", unexpected)
    monkeypatch.setattr(methods, "ContinuationEntropyLMBuggy", unexpected)
    with pytest.raises(SystemExit) as error:
        cli.main(["--dataset", "provo", "--lm", "gpt2", "--method", method])
    assert error.value.code == 2
    with pytest.raises(ValueError, match="allow_reproduction=True"):
        runner.compute_for(
            "provo",
            method=method,
            model_id="gpt2",
            K=5,
            n_samples=4,
            max_tokens=2,
            alpha=0.5,
            seed=0,
            out_dir=tmp_path,
            add_bos=True,
            force=False,
        )
    context = ComputeContext("gpt2", 5, 4, 2, 0.5, 0, True, "en", None)
    with pytest.raises(ValueError, match="allow_reproduction=True"):
        methods.build_model(method, context)


def test_cli_passes_reproduction_opt_in_to_the_computation(tmp_path, monkeypatch):
    received = []
    monkeypatch.setattr(cli, "expand_datasets", lambda name: [name])

    def compute(dataset, **kwargs):
        received.append(kwargs["allow_reproduction"])
        return runner.ComputeOutcome(
            dataset, kwargs["method"], "cached", tmp_path / "cell.pkl"
        )

    monkeypatch.setattr(cli, "compute_for", compute)
    _outcomes(
        tmp_path,
        "--method",
        "first_token_entropy_buggy",
        "--allow-reproduction",
        code=0,
    )
    assert received == [True]


def test_reproduction_opt_in_preserves_numerical_identity():
    from replica.compute import artifacts
    from tests._stimuli import word_stimulus

    context = ComputeContext("gpt2", 5, 4, 2, 0.5, 0, True, "en", None)
    opted_in = replace(context, allow_reproduction=True)
    stimulus = word_stimulus("a word", "test")
    for method in ("surprisal", "first_token_entropy_buggy"):
        assert artifacts.artifact_identity(
            method, context, [stimulus], {}
        ) == artifacts.artifact_identity(method, opted_in, [stimulus], {})
        assert runner.cache_stem(method, context, "provo") == runner.cache_stem(
            method, opted_in, "provo"
        )
