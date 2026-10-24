"""Regression checks for cache provenance, publication and batch failures."""

import fcntl
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from replica import artifacts as manifests
from replica.compute import artifacts, cli, runner
from replica.compute.methods import ComputeContext
from replica.core import AnnotationLayer, Span, Stimulus
from replica.feature import FeatureModelOutput
from replica.utils.atomic import atomic_pickle


def _context():
    return ComputeContext("gpt2", 5, 64, 10, 1.0, 0, True, "en", None)


def _stim(text="ab cd", spans=None):
    return Stimulus(
        text,
        {
            "words": AnnotationLayer(
                "words", spans or [Span(0, 2), Span(3, 5)], ["ab", "cd"]
            )
        },
        "s",
    )


def test_cache_names_separate_window_and_segmentation():
    base = _context()
    names = {
        runner.cache_stem(
            "word_information_value",
            replace(base, window_policy=w, segmentation=s),
            "provo",
        )
        for w in ("reference", "cap")
        for s in ("whitespace", "layer")
    }
    assert len(names) == 4
    assert names == {
        "gpt2_word_infoval_N64_M10_s0_provo",
        "gpt2_word_infoval_N64_M10_s0_cap_provo",
        "gpt2_word_infoval_N64_M10_s0_layer_provo",
        "gpt2_word_infoval_N64_M10_s0_cap_layer_provo",
    }


def test_manifest_invalidates_changed_text_spans_options_revision_and_payload(tmp_path):
    context = _context()
    model = {"revision": "a" * 40}
    identity = artifacts.artifact_identity("surprisal", context, [_stim()], model)
    path = tmp_path / "cell.pkl"
    atomic_pickle({"s": [FeatureModelOutput(values={"log_p_observed": -1})]}, path)
    manifests.write_manifest(path, identity, status="completed")
    manifest = json.loads(manifests.manifest_path(path).read_text())
    assert manifest["payload_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert manifests.matching_manifest(path, identity, status="completed")
    assert manifests.validate_artifact(path) == identity
    variants = [
        artifacts.artifact_identity("surprisal", context, [_stim("ef gh")], model),
        artifacts.artifact_identity(
            "surprisal", context, [_stim(spans=[Span(0, 1), Span(2, 5)])], model
        ),
        artifacts.artifact_identity(
            "surprisal", replace(context, segmentation="layer"), [_stim()], model
        ),
        artifacts.artifact_identity(
            "surprisal", context, [_stim()], {"revision": "b" * 40}
        ),
    ]
    for variant in variants:
        assert not manifests.matching_manifest(path, variant, status="completed")
    path.write_bytes(b"corrupt")
    assert not manifests.matching_manifest(path, identity, status="completed")
    with pytest.raises(ValueError, match="invalid artifact manifest"):
        manifests.validate_artifact(path)


def test_legacy_and_mismatched_checkpoints_are_not_resumed(tmp_path):
    path = tmp_path / "cell.partial"
    atomic_pickle({"s": [FeatureModelOutput(values={"log_p_observed": -1})]}, path)
    assert runner._load_partial_outputs(path, force=False, identity={}) == {}
    manifests.write_manifest(path, {"old": True}, status="partial")
    assert runner._load_partial_outputs(path, force=False, identity={}) == {}


def test_three_openers_share_the_same_lock_after_release(tmp_path):
    path = tmp_path / "cell.lock"
    first = runner._acquire_lock(path)
    with path.open("a") as delayed:
        inode = path.stat().st_ino
        runner._release_lock(first, path)
        fcntl.flock(delayed, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            assert path.stat().st_ino == inode
            assert runner._acquire_lock(path) is None
        finally:
            fcntl.flock(delayed, fcntl.LOCK_UN)


def test_quarantined_candidate_is_never_published_canonically(tmp_path, monkeypatch):
    from replica.compute import validation

    path = tmp_path / "cell.pkl"
    original = validation.predictor_pickle_issues

    def inspect_candidate(candidate):
        assert candidate != path
        assert not path.exists()
        return original(candidate)

    monkeypatch.setattr(validation, "predictor_pickle_issues", inspect_candidate)
    published = runner._write_and_audit_outputs(
        path, tmp_path / "partial", {}, identity={}, expected=1
    )
    assert published.name == "cell.suspect.pkl"
    assert published.exists()
    assert not path.exists()


@pytest.mark.parametrize("boundary", ["payload", "manifest"])
def test_failed_publication_retains_checkpoint_and_can_be_retried(
    tmp_path, monkeypatch, boundary
):
    path, partial = tmp_path / "cell.pkl", tmp_path / "cell.pkl.partial"
    old = {"schema_version": 1, "generation": "old"}
    new = {"schema_version": 1, "generation": "new"}
    original = {"s": [FeatureModelOutput(values={"log_p_observed": -1.0})]}
    replacement = {"s": [FeatureModelOutput(values={"log_p_observed": -2.0})]}
    atomic_pickle(original, path)
    manifests.write_manifest(path, old, status="completed")
    atomic_pickle(replacement, partial)
    manifests.write_manifest(partial, new, status="partial")
    checkpoint = partial.read_bytes()
    checkpoint_manifest = manifests.manifest_path(partial).read_bytes()
    target = path if boundary == "payload" else manifests.manifest_path(path)
    replace = Path.replace

    def interrupted(source, destination):
        if destination == target:
            raise OSError("injected publication failure")
        return replace(source, destination)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "replace", interrupted)
        with pytest.raises(OSError, match="injected publication"):
            runner._write_and_audit_outputs(
                path, partial, replacement, identity=new, expected=1
            )
    assert not manifests.matching_manifest(path, new, status="completed")
    assert partial.read_bytes() == checkpoint
    assert manifests.manifest_path(partial).read_bytes() == checkpoint_manifest
    if boundary == "payload":
        assert manifests.validate_artifact(path) == old
    else:
        with pytest.raises(ValueError, match="invalid artifact manifest"):
            manifests.validate_artifact(path)
    assert (
        runner._write_and_audit_outputs(
            path, partial, replacement, identity=new, expected=1
        )
        == path
    )
    assert manifests.validate_artifact(path) == new
    assert not partial.exists()
    assert not manifests.manifest_path(partial).exists()


def test_compute_cli_records_every_request_and_exits_nonzero(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "expand_datasets", lambda name: ["broken", "good"])

    def compute(dataset, **kwargs):
        if dataset == "broken":
            raise ValueError("bad input")
        return runner.ComputeOutcome(dataset, "surprisal", "completed", tmp_path)

    monkeypatch.setattr(cli, "compute_for", compute)
    manifest = tmp_path / "run.json"
    args = [
        "--dataset",
        "any",
        "--lm",
        "gpt2",
        "--out-dir",
        str(tmp_path),
        "--run-manifest",
        str(manifest),
    ]
    with pytest.raises(SystemExit) as error:
        cli.main(args)
    assert error.value.code == 1
    report = json.loads(manifest.read_text())
    assert [o["status"] for o in report["outcomes"]] == ["failed", "completed"]
    assert report["requested"] == 2
    cli.main([*args, "--allow-incomplete"])
    assert json.loads(manifest.read_text())["allow_incomplete"] is True


def test_fingerprint_ignores_generated_layers_but_tracks_sentence_boundaries():
    import numpy as np

    context = replace(_context(), context_scope="sentence")
    stimulus = _stim()
    stimulus.add_layer(
        AnnotationLayer(
            "sentences",
            list(stimulus.layers["words"].spans),
            [np.int64(1), np.int64(1)],
        )
    )
    original = artifacts.artifact_identity("surprisal", context, [stimulus], {})
    stimulus.add_layer(
        AnnotationLayer("generated_tokenizer_layer", [Span(0, 5)], [object()])
    )
    assert artifacts.artifact_identity("surprisal", context, [stimulus], {}) == original
    stimulus.layers["sentences"].values[1] = np.int64(2)
    assert artifacts.artifact_identity("surprisal", context, [stimulus], {}) != original


def test_environment_record_names_the_code_state_and_device():
    record = artifacts.environment_record()
    assert set(record) == {
        "code_commit",
        "code_dirty",
        "device",
        "model_device",
        "cuda",
        "computed_at",
    }
    import torch

    if torch.cuda.is_available():
        assert record["device"] == torch.cuda.get_device_name(
            torch.cuda.current_device()
        )
        assert record["cuda"] == torch.version.cuda
    else:
        assert record["device"] == "cpu" and record["cuda"] is None
    assert record["model_device"] is None
    assert artifacts.environment_record("cuda:0")["model_device"] == "cuda:0"
    commit = record["code_commit"]
    assert commit is None or (len(commit) == 40 and int(commit, 16) >= 0)
    assert (record["code_dirty"] is None) == (commit is None)
    assert isinstance(record["device"], str) and record["device"]
    from datetime import datetime

    assert datetime.fromisoformat(record["computed_at"]).tzinfo is not None


def test_environment_rides_beside_the_identity_without_touching_it(tmp_path):
    context = _context()
    identity = artifacts.artifact_identity("surprisal", context, [_stim()], {})
    path = tmp_path / "cell.pkl"
    atomic_pickle({"s": [FeatureModelOutput(values={"x": 1})]}, path)
    environment = {"code_commit": "f" * 40, "device": "cpu"}
    manifests.write_manifest(
        path, identity, status="completed", environment=environment
    )
    written = json.loads(manifests.manifest_path(path).read_text())
    assert written["environment"] == environment
    assert manifests.validate_artifact(path) == identity
    assert manifests.matching_manifest(path, identity, status="completed")
    manifests.write_manifest(path, identity, status="completed")
    assert "environment" not in json.loads(manifests.manifest_path(path).read_text())
    assert manifests.validate_artifact(path) == identity


def _counts_table(path, the=8, total=10):
    path.write_text(f"# total_tokens={total}\nthe\t{the}\ndog\t2\n")
    return path


def test_a_file_backed_table_is_identified_by_its_contents(tmp_path, monkeypatch):
    """A changed count table must not pass as the cell already computed.

    The table is the model for these sources, so the identity carries its
    checksum and file name and ignores its directory.
    """
    table = _counts_table(tmp_path / "counts.tsv")
    monkeypatch.setenv("PSYCHO_OWT_COUNTS", str(table))
    context = replace(_context(), model_id="owt")

    model_path, identity = artifacts.resolve_model_snapshot(
        "unigram_surprisal", context
    )
    assert model_path == "owt"
    assert identity["table"] == "counts.tsv"
    assert len(identity["sha256"]) == 64

    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.setenv(
        "PSYCHO_OWT_COUNTS", str(_counts_table(elsewhere / "counts.tsv"))
    )
    _, same = artifacts.resolve_model_snapshot("unigram_surprisal", context)
    assert same == identity

    monkeypatch.setenv("PSYCHO_OWT_COUNTS", str(table))
    _counts_table(table, the=7)
    _, changed = artifacts.resolve_model_snapshot("unigram_surprisal", context)
    assert changed["sha256"] != identity["sha256"]


def test_a_package_backed_table_reports_its_package_version(monkeypatch):
    context = replace(_context(), model_id="wordfreq")
    _, identity = artifacts.resolve_model_snapshot("unigram_surprisal", context)
    assert identity == {"source": "wordfreq", "revision": "package-version"}


def test_the_shared_unigram_method_builds_every_standalone_table(tmp_path, monkeypatch):
    from replica.compute.methods import build_model

    monkeypatch.setenv("PSYCHO_OWT_COUNTS", str(_counts_table(tmp_path / "counts.tsv")))
    model = build_model("unigram_surprisal", replace(_context(), model_id="owt"))
    assert model.name == "unigram_surprisal_en"

    with pytest.raises(ValueError, match="'wordfreq'"):
        build_model("unigram_surprisal", replace(_context(), model_id="gpt2"))


def test_model_device_reads_the_loaded_model_before_the_wrapper():
    from types import SimpleNamespace

    from replica.compute.runner import _model_device

    loaded = SimpleNamespace(_model=SimpleNamespace(device="cuda:0"), device="cpu")
    assert _model_device(loaded) == "cuda:0"
    unloaded = SimpleNamespace(_model=None, device="cuda")
    assert _model_device(unloaded) == "cuda"
    assert _model_device(SimpleNamespace()) is None


def test_units_enter_the_identity_and_window_rides_beside_it(tmp_path):
    context = _context()
    plain = artifacts.artifact_identity("surprisal", context, [_stim()], {})
    assert "units" not in plain
    units = {"log_p_observed": "nats", "surprisal": "bits"}
    with_units = artifacts.artifact_identity("surprisal", context, [_stim()], {}, units)
    assert with_units["units"] == units
    assert with_units != plain
    path = tmp_path / "cell.pkl"
    atomic_pickle({"s": [FeatureModelOutput(values={"x": 1})]}, path)
    window = {"rule": "sliding", "max_positions": 8, "stride": 4}
    derivation = {"source": "cell_tokens.pkl", "rule": "sum"}
    manifests.write_manifest(
        path, with_units, status="completed", window=window, derivation=derivation
    )
    written = json.loads(manifests.manifest_path(path).read_text())
    assert written["window"] == window and written["derivation"] == derivation
    assert manifests.validate_artifact(path) == with_units
    assert not manifests.matching_manifest(path, plain, status="completed")
