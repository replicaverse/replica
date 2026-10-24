"""Cache reuse and manifest integrity under settings the cell name does not carry.

Several run options change the identity without changing the file name. A
table computed under one of them must never be returned as the cell computed
under another, a payload changed by one byte must fail its sidecar, and a
word table must follow the token table it was derived from.
"""

from dataclasses import replace
from pathlib import Path

import pytest

from replica import artifacts as manifests
from replica.artifacts import manifest_path, write_manifest
from replica.compute import artifacts, runner
from replica.compute.methods import ComputeContext
from replica.core import AnnotationLayer, Span, Stimulus
from replica.feature import FeatureModelOutput
from replica.utils.atomic import atomic_pickle
from replica.utils.digest import file_sha256
from replica.utils.git import git_output
from tests.feature.core.annotation.test_token_tables import _install, _load, _manifest

MODEL = {"source": "gpt2", "revision": "a" * 40}


def _context(**overrides) -> ComputeContext:
    base = ComputeContext("gpt2", 5, 64, 10, 1.0, None, True, "en", None)
    return replace(base, **overrides)


def _stimulus() -> Stimulus:
    spans = [Span(0, 2), Span(3, 5)]
    stimulus = Stimulus(
        text="ab cd",
        stimulus_id="s",
        layers={"words": AnnotationLayer("words", spans, ["ab", "cd"])},
    )
    stimulus.add_layer(AnnotationLayer("sentences", list(spans), [1, 1]))
    return stimulus


def _completed_table(path: Path, identity: dict) -> None:
    atomic_pickle(
        {"s": [FeatureModelOutput(values={"log_p_observed": -1.0})] * 2}, path
    )
    write_manifest(path, identity, status="completed")


@pytest.mark.parametrize(
    ("method", "change"),
    [
        ("surprisal", {"chunk_size": 8}),
        ("surprisal", {"start_ids": (0,)}),
        ("surprisal", {"context_window": 512}),
        ("surprisal", {"seed": 1}),
        ("surprisal", {"model_revision": "b" * 40}),
        ("surprisal", {"segmentation": "layer"}),
        ("surprisal", {"window_policy": "cap"}),
        ("first_token_entropy", {"store_distribution": True}),
        ("first_token_entropy", {"context_scope": "sentence"}),
        ("char_beam", {"add_bos": False}),
    ],
)
def test_options_outside_the_cell_name_still_change_the_identity(
    tmp_path, method, change
):
    base, changed = _context(), _context(**change)
    assert runner.cache_stem(method, base, "provo") == runner.cache_stem(
        method, changed, "provo"
    )
    stimuli = [_stimulus()]
    identity = artifacts.artifact_identity(method, base, stimuli, MODEL)
    other = artifacts.artifact_identity(method, changed, stimuli, MODEL)
    assert identity != other
    path = tmp_path / f"{runner.cache_stem(method, base, 'provo')}.pkl"
    _completed_table(path, identity)
    assert runner._should_skip_existing(path, force=False, identity=identity)
    assert not runner._should_skip_existing(path, force=False, identity=other)


@pytest.mark.parametrize(
    ("method", "default_rule", "other_rule"),
    [
        ("mc_word_entropy", "whitespace_continuation", "corpus_units"),
        ("unigram_surprisal", "corpus_segment_lookup", "whitespace_continuation"),
        ("unigram_surprisal", "corpus_segment_lookup", "corpus_units"),
    ],
)
def test_a_boundary_rule_changes_the_identity_of_a_unit_bound_cell(
    tmp_path, method, default_rule, other_rule
):
    context, stimuli = _context(), [_stimulus()]
    default = artifacts.artifact_identity(method, context, stimuli, MODEL, None, None)
    named = artifacts.artifact_identity(
        method, context, stimuli, MODEL, None, other_rule
    )
    assert default["unit_rule"] == default_rule
    assert named["unit_rule"] == other_rule
    path = tmp_path / f"{runner.cache_stem(method, context, 'provo')}.pkl"
    _completed_table(path, named)
    before = manifest_path(path).read_bytes()
    assert not runner._should_skip_existing(path, force=False, identity=default)
    assert manifest_path(path).read_bytes() == before


def test_one_changed_payload_byte_fails_the_sidecar(tmp_path):
    identity = artifacts.artifact_identity(
        "surprisal", _context(), [_stimulus()], MODEL
    )
    path = tmp_path / "cell.pkl"
    _completed_table(path, identity)
    payload = bytearray(path.read_bytes())
    payload[len(payload) // 2] ^= 0xFF
    path.write_bytes(bytes(payload))
    assert not manifests.matching_manifest(path, identity, status="completed")
    with pytest.raises(ValueError, match="invalid artifact manifest"):
        manifests.validate_artifact(path)
    assert not runner._should_skip_existing(path, force=False, identity=identity)


def test_environment_record_names_the_tree_that_ran():
    root = Path(artifacts.__file__).resolve().parents[3]
    record = artifacts.environment_record()
    assert record["code_commit"] == git_output(root, "rev-parse", "HEAD")
    status = git_output(root, "status", "--porcelain", "--", "src/replica")
    assert record["code_dirty"] == (None if status is None else bool(status))


def test_word_table_is_rederived_when_its_token_source_changes(tmp_path, monkeypatch):
    common = _install(monkeypatch, tmp_path)
    first = runner.compute_for("corpus", method="surprisal", **common)
    assert first.status == "completed", first.error
    before = _load(first.path)["s0"][0].values["log_p_observed"]
    token_path = tmp_path / _manifest(first.path)["derivation"]["source"]
    table = _load(token_path)
    table["s0"][0].values["log_p_observed"] -= 1.0
    atomic_pickle(table, token_path)
    sidecar = _manifest(token_path)
    write_manifest(
        token_path,
        sidecar["identity"],
        status="completed",
        environment=sidecar["environment"],
        window=sidecar["window"],
    )

    again = runner.compute_for("corpus", method="surprisal", **common)

    assert again.status == "completed", again.error
    assert _manifest(again.path)["derivation"]["source_sha256"] == file_sha256(
        token_path
    )
    assert _load(again.path)["s0"][0].values["log_p_observed"] == before - 1.0


def test_publishing_a_cell_removes_its_superseded_quarantine(tmp_path):
    path, partial = tmp_path / "cell.pkl", tmp_path / "cell.pkl.partial"
    outputs = {"s0": [FeatureModelOutput(values={"log_p_observed": -1.0})]}
    suspect = runner._write_and_audit_outputs(
        path, partial, outputs, identity={"k": 1}, expected=2
    )
    assert suspect.name == "cell.suspect.pkl" and manifest_path(suspect).exists()

    outputs["s1"] = [FeatureModelOutput(values={"log_p_observed": -2.0})]
    published = runner._write_and_audit_outputs(
        path, partial, outputs, identity={"k": 1}, expected=2
    )

    assert published == path
    assert not suspect.exists() and not manifest_path(suspect).exists()
