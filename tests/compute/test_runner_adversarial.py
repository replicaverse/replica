"""Resume, force, quarantine, incomplete-batch and shard handling of the driver.

A checkpoint is resumed only under the identity that wrote it; ``force``
ignores it; a payload the audit rejects is quarantined and never resumed;
the command exits non-zero on an unsuccessful cell unless told otherwise;
a stimulus slice is clamped to the corpus and named by the clamped bounds.
"""

import json
import pickle
from copy import deepcopy
from dataclasses import replace

import pytest

from replica import artifacts as manifests
from replica.compute import cli, methods, runner
from replica.compute.cache import TOKEN_SENSITIVE, split_cache_filename
from replica.compute.runner import ComputeOutcome
from replica.compute.units import TOKEN_UNIT
from tests.feature.core.annotation.test_token_tables import _lm

from .._stimuli import word_stimulus

STIMULI = [
    word_stimulus("The cats sat on a mat.", "s0"),
    word_stimulus("A second, shorter one.", "s1"),
    word_stimulus("And a third sentence here.", "s2"),
]


class _Scorer:
    """Token scorer double that counts calls and can fail or corrupt a stimulus."""

    def __init__(self, fail_on=(), inf_on=(), add_bos=True):
        self.lm = _lm(bos=add_bos)
        self.fail_on = set(fail_on)
        self.inf_on = set(inf_on)
        self.calls: list[str] = []

    def __getattr__(self, name):
        return getattr(self.lm, name)

    def score_tokens(self, stimulus, extract=None):
        self.calls.append(stimulus.stimulus_id)
        if stimulus.stimulus_id in self.fail_on:
            raise RuntimeError("scorer failed")
        rows = self.lm.score_tokens(stimulus, extract=extract)
        if stimulus.stimulus_id in self.inf_on:
            rows[1].values["log_p_observed"] = float("inf")
        return rows


def _install(monkeypatch, tmp_path, scorer):
    monkeypatch.setattr(
        runner,
        "_load_dataset",
        lambda name: runner._LoadedDataset(deepcopy(STIMULI), len(STIMULI), "en"),
    )
    monkeypatch.setattr(
        runner,
        "resolve_model_snapshot",
        lambda method, context: (
            context.model_id,
            {"source": "test-tokens", "revision": "r" * 40},
        ),
    )
    for method in TOKEN_SENSITIVE:
        monkeypatch.setitem(
            methods.METHOD_SPECS,
            method,
            replace(methods.METHOD_SPECS[method], build_model=lambda _c: scorer),
        )
    return dict(
        model_id="test-tokens",
        K=5,
        n_samples=64,
        max_tokens=10,
        alpha=1.0,
        seed=None,
        out_dir=tmp_path,
        add_bos=True,
        force=False,
        checkpoint_every=1,
    )


def _manifest(path):
    return json.loads(manifests.manifest_path(path).read_text())


TOKEN_CELL = "test-tokens_surprisal_tokens_corpus.pkl"


def _partial_run(monkeypatch, tmp_path):
    """One token-cell run that fails on the last stimulus, leaving a checkpoint."""
    scorer = _Scorer(fail_on={"s2"})
    common = _install(monkeypatch, tmp_path, scorer)
    outcome = runner.compute_for(
        "corpus", method="surprisal", unit=TOKEN_UNIT, **common
    )
    assert outcome.status == "quarantined"
    partial = tmp_path / (TOKEN_CELL + ".partial")
    assert set(pickle.loads(partial.read_bytes())) == {"s0", "s1"}
    assert _manifest(partial)["status"] == "partial"
    return common


def test_matching_partial_resumes_only_the_missing_stimulus(tmp_path, monkeypatch):
    common = _partial_run(monkeypatch, tmp_path)
    scorer = _Scorer()
    _install(monkeypatch, tmp_path, scorer)
    outcome = runner.compute_for(
        "corpus", method="surprisal", unit=TOKEN_UNIT, **common
    )
    assert outcome.status == "completed", outcome.error
    assert scorer.calls == ["s2"]
    assert not (tmp_path / (TOKEN_CELL + ".partial")).exists()
    assert not manifests.manifest_path(tmp_path / (TOKEN_CELL + ".partial")).exists()


def test_partial_under_another_identity_starts_fresh(tmp_path, monkeypatch):
    common = _partial_run(monkeypatch, tmp_path)
    scorer = _Scorer()
    _install(monkeypatch, tmp_path, scorer)
    # chunk_size reaches the identity but not the file name.
    outcome = runner.compute_for(
        "corpus", method="surprisal", unit=TOKEN_UNIT, chunk_size=7, **common
    )
    assert outcome.status == "completed", outcome.error
    assert outcome.path.name == TOKEN_CELL
    assert scorer.calls == ["s0", "s1", "s2"]


def test_force_recomputes_every_stimulus_of_a_partial(tmp_path, monkeypatch):
    common = _partial_run(monkeypatch, tmp_path)
    scorer = _Scorer()
    _install(monkeypatch, tmp_path, scorer)
    outcome = runner.compute_for(
        "corpus", method="surprisal", unit=TOKEN_UNIT, **(common | {"force": True})
    )
    assert outcome.status == "completed", outcome.error
    assert scorer.calls == ["s0", "s1", "s2"]
    assert not (tmp_path / (TOKEN_CELL + ".partial")).exists()


def test_hard_audit_issue_is_quarantined_and_never_resumed(tmp_path, monkeypatch):
    scorer = _Scorer(inf_on={"s1"})
    common = _install(monkeypatch, tmp_path, scorer)
    outcome = runner.compute_for(
        "corpus",
        method="surprisal",
        unit=TOKEN_UNIT,
        **(common | {"checkpoint_every": 100}),
    )
    assert outcome.status == "quarantined"
    assert outcome.error == "Incomplete or invalid predictor output"
    suspect = tmp_path / "test-tokens_surprisal_tokens_corpus.suspect.pkl"
    assert outcome.path == suspect and suspect.exists()
    manifest = _manifest(suspect)
    assert manifest["status"] == "quarantined"
    audit = manifest["audit"]
    assert audit["hard_issues"] and audit["soft_issues"] == []
    assert set(audit["examples"]) == set(audit["hard_issues"])
    assert audit["failed_stimuli"] == 0 and audit["missing_stimuli"] == 0
    assert not (tmp_path / TOKEN_CELL).exists()
    assert not (tmp_path / (TOKEN_CELL + ".partial")).exists()
    assert not (tmp_path / "test-tokens_surprisal_tokens_corpus.failed.log").exists()
    # The quarantined payload does not seed the next run.
    scorer = _Scorer()
    _install(monkeypatch, tmp_path, scorer)
    again = runner.compute_for("corpus", method="surprisal", unit=TOKEN_UNIT, **common)
    assert again.status == "completed", again.error
    assert scorer.calls == ["s0", "s1", "s2"]


def test_checkpointed_hard_payload_is_not_resumed(tmp_path, monkeypatch):
    scorer = _Scorer(inf_on={"s0"}, fail_on={"s2"})
    common = _install(monkeypatch, tmp_path, scorer)
    outcome = runner.compute_for(
        "corpus", method="surprisal", unit=TOKEN_UNIT, **common
    )
    assert outcome.status == "quarantined"
    partial = tmp_path / (TOKEN_CELL + ".partial")
    # The checkpoint written after s1 holds the corrupt s0 row.
    assert set(pickle.loads(partial.read_bytes())) == {"s0", "s1"}
    scorer = _Scorer()
    _install(monkeypatch, tmp_path, scorer)
    again = runner.compute_for("corpus", method="surprisal", unit=TOKEN_UNIT, **common)
    assert again.status == "completed", again.error
    assert scorer.calls == ["s0", "s1", "s2"]


def test_derived_word_cell_does_not_publish_when_the_token_cell_fails(
    tmp_path, monkeypatch
):
    scorer = _Scorer(fail_on={"s1"})
    common = _install(monkeypatch, tmp_path, scorer)
    outcome = runner.compute_for("corpus", method="surprisal", **common)
    assert outcome.status == "quarantined" and outcome.method == "surprisal"
    assert outcome.path.name == "test-tokens_surprisal_tokens_corpus.suspect.pkl"
    assert not list(tmp_path.glob("test-tokens_surprisal_corpus*"))
    audit = _manifest(outcome.path)["audit"]
    assert audit["failed_stimuli"] == 1 and audit["missing_stimuli"] == 1
    assert audit["hard_issues"] == []


# ---- --allow-incomplete ------------------------------------------------------


def _cli(monkeypatch, tmp_path, status):
    manifest = tmp_path / "run.json"
    monkeypatch.setattr(
        cli,
        "compute_for",
        lambda dataset_name, **kw: ComputeOutcome(dataset_name, kw["method"], status),
    )
    monkeypatch.setattr(cli, "expand_datasets", lambda name: [name])
    argv = [
        "--dataset",
        "corpus",
        "--lm",
        "test-tokens",
        "--out-dir",
        str(tmp_path),
        "--run-manifest",
        str(manifest),
    ]
    return argv, manifest


@pytest.mark.parametrize("status", ["quarantined", "busy", "failed"])
def test_unsuccessful_cell_exits_nonzero_unless_incomplete_is_allowed(
    tmp_path, monkeypatch, status
):
    argv, manifest = _cli(monkeypatch, tmp_path, status)
    with pytest.raises(SystemExit) as exit_info:
        cli.main(argv)
    assert exit_info.value.code == 1
    record = json.loads(manifest.read_text())
    assert record["allow_incomplete"] is False
    assert [o["status"] for o in record["outcomes"]] == [status]
    assert cli.main([*argv, "--allow-incomplete"]) is None
    record = json.loads(manifest.read_text())
    assert record["allow_incomplete"] is True
    assert [o["status"] for o in record["outcomes"]] == [status]


def test_successful_cells_exit_zero(tmp_path, monkeypatch):
    argv, manifest = _cli(monkeypatch, tmp_path, "cached")
    assert cli.main(argv) is None
    assert json.loads(manifest.read_text())["requested"] == 1


# ---- --stim-slice -------------------------------------------------------------


def test_slice_is_clamped_to_the_corpus_and_named_by_the_clamp():
    stimuli = list("abcdefgh")
    assert runner._apply_stimulus_slice(stimuli, 8, None) == (stimuli, "")
    sliced, suffix = runner._apply_stimulus_slice(stimuli, 8, (6, 50))
    assert (sliced, suffix) == (["g", "h"], ".shard00006_00008")
    sliced, suffix = runner._apply_stimulus_slice(stimuli, 8, (-3, 2))
    assert (sliced, suffix) == (["a", "b"], ".shard00000_00002")
    assert runner._apply_stimulus_slice(stimuli, 8, (8, 50))[0] == []
    assert runner._apply_stimulus_slice(stimuli, 8, (5, 3))[0] == []
    model, method, config = split_cache_filename(
        "test-tokens_surprisal_tokens_corpus.shard00006_00008", "corpus"
    )
    assert (model, method, config) == ("test-tokens", "surprisal", {"unit": "tokens"})


def test_slice_beyond_the_corpus_is_refused_and_a_partial_slice_is_a_shard(
    tmp_path, monkeypatch
):
    scorer = _Scorer()
    common = _install(monkeypatch, tmp_path, scorer)
    with pytest.raises(ValueError, match="No stimuli selected"):
        runner.compute_for(
            "corpus", method="surprisal", unit=TOKEN_UNIT, stim_slice=(3, 9), **common
        )
    outcome = runner.compute_for(
        "corpus", method="surprisal", unit=TOKEN_UNIT, stim_slice=(2, 9), **common
    )
    assert outcome.status == "completed", outcome.error
    assert (
        outcome.path.name == "test-tokens_surprisal_tokens_corpus.shard00002_00003.pkl"
    )
    assert scorer.calls == ["s2"]
    assert _manifest(outcome.path)["identity"]["n_stimuli"] == 1
    # The derived word shard carries the same suffix and covers the same slice.
    words = runner.compute_for(
        "corpus", method="surprisal", stim_slice=(2, 9), **common
    )
    assert words.status == "completed", words.error
    assert words.path.name == "test-tokens_surprisal_corpus.shard00002_00003.pkl"
    assert set(pickle.loads(words.path.read_bytes())) == {"s2"}
    assert _manifest(words.path)["derivation"]["source"] == outcome.path.name
