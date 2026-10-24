"""Which checkpoints a run resumes.

A checkpoint is resumed only under the identity that wrote it and only when
its payload reads back and passes the audit. Any other checkpoint starts the
run fresh and never raises.
"""

import pickle

import pytest

from replica.artifacts import write_manifest
from replica.compute import runner
from replica.compute.artifacts import artifact_identity
from replica.compute.methods import ComputeContext
from replica.compute.validation import predictor_pickle_issues
from replica.feature import FeatureModelOutput
from replica.utils.atomic import atomic_pickle

from .test_compute_lm_features_checkpoint import (
    _FakeStim,
    _install_fake_dataset,
    _install_fake_extract_keys,
    _install_fake_lm,
    _install_fake_tag,
)

#: Payloads a kill mid-write, a disk fault or a moved class can leave behind.
CORRUPT_PAYLOADS = {
    "truncated": pickle.dumps({"s0": [1]})[:3],
    "protocol": b"\x80\xff",
    "missing_module": b"\x80\x04cno_such_module_x\nThing\n.",
    "reduce_on_string": b"\x80\x04\x8c\x03abc\x94)\x94R\x94.",
    "invalid_utf8": b"\x80\x04\x8c\x01\xff\x94.",
}

IDENTITY = {"schema_version": 1, "method": "surprisal"}


@pytest.mark.parametrize("name", sorted(CORRUPT_PAYLOADS))
def test_an_unreadable_checkpoint_with_its_own_sidecar_starts_fresh(tmp_path, name):
    partial = tmp_path / "cell.pkl.partial"
    partial.write_bytes(CORRUPT_PAYLOADS[name])
    write_manifest(partial, IDENTITY, status="partial")

    assert runner._load_partial_outputs(partial, force=False, identity=IDENTITY) == {}
    assert not runner._valid_cached_pickle(partial)
    report = predictor_pickle_issues(partial)
    assert "unreadable" in report.issues and report.has_hard_issues


def test_a_checkpoint_the_audit_rejects_is_not_resumed(tmp_path):
    partial = tmp_path / "cell.pkl.partial"
    dead = [FeatureModelOutput(values={"log_p_observed": float("nan")})] * 2
    atomic_pickle({"s0": dead}, partial)
    write_manifest(partial, IDENTITY, status="partial")

    assert runner._load_partial_outputs(partial, force=False, identity=IDENTITY) == {}


def test_a_checkpoint_of_another_identity_is_not_merged(tmp_path, monkeypatch):
    monkeypatch.setattr(
        runner,
        "resolve_model_snapshot",
        lambda method, context: (context.model_id, {"revision": "test-commit"}),
    )
    stim_ids = ["s0", "s1", "s2"]
    partial = tmp_path / "fake_cell.pkl.partial"
    seeded = FeatureModelOutput(
        values={"log_p_observed": -2.0}, metadata={"from_seed": True}
    )
    atomic_pickle({"s0": [seeded]}, partial)
    unchunked = ComputeContext("gpt2", 5, 64, 10, 1.0, None, True, "en", None)
    write_manifest(
        partial,
        artifact_identity(
            "modelblocks_surprisal",
            unchunked,
            [_FakeStim(s) for s in stim_ids],
            {"revision": "test-commit"},
        ),
        status="partial",
    )
    invoked: list[str] = []

    with (
        _install_fake_dataset(runner, stim_ids),
        _install_fake_lm(runner, invoked=invoked),
        _install_fake_extract_keys(runner),
        _install_fake_tag(runner),
    ):
        outcome = runner.compute_for(
            "fake",
            method="modelblocks_surprisal",
            model_id="gpt2",
            K=5,
            n_samples=64,
            max_tokens=10,
            alpha=1.0,
            seed=None,
            out_dir=tmp_path,
            add_bos=True,
            force=False,
            chunk_size=8,
        )

    assert outcome.status == "completed", outcome.error
    assert invoked == stim_ids
    with outcome.path.open("rb") as stream:
        table = pickle.load(stream)
    assert "from_seed" not in table["s0"][0].metadata
