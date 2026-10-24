"""Tests for the per-stimulus checkpoint/resume in ``replica.compute``.

Checkpointing exists so that long-running cells (char_beam on multilingual
MECO, mGPT MC entropy on natural_stories) survive a SIGTERM / OOM /
walltime-cancel without losing all of their per-stimulus work. A re-run
of the same cell picks up where the partial left off via ``stimulus_id``
membership.

These tests exercise the packaged producer with a fake LM and fake stimuli.
No GPU or Hugging Face model is required. The method is one whose word cell
the producer scores directly, since the token-derived methods would first
write a token cell under the same patched stem.
"""

import json
import pickle
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

from replica.artifacts import manifest_path, validate_artifact, write_manifest
from replica.compute.artifacts import artifact_identity
from replica.compute.methods import ComputeContext
from replica.core import AnnotationLayer, Span, Stimulus
from replica.feature import FeatureModelOutput
from replica.feature.frequency import sampled
from replica.feature.frequency.model import UnigramSurprisalLM
from replica.utils.atomic import atomic_pickle, temporary_path
from tests.feature.frequency._priming_stub import (
    C1,
    C2,
    Killed,
    refuse_sampling,
    stub_source,
)


@pytest.fixture(scope="module")
def cs():
    from replica.compute import runner

    return runner


# ── Fake dataset + LM ───────────────────────────────────────────────────────


class _FakeStim(Stimulus):
    def __init__(self, sid: str):
        super().__init__(
            text=sid,
            stimulus_id=sid,
            layers={"words": AnnotationLayer("words", [Span(0, len(sid))], [sid])},
        )


@pytest.fixture(autouse=True)
def _snapshot(monkeypatch, cs):
    monkeypatch.setattr(
        cs,
        "resolve_model_snapshot",
        lambda method, context: (context.model_id, {"revision": "test-commit"}),
    )


def _seed_manifest(path, stim_ids):
    context = ComputeContext(
        model_id="gpt2",
        beam_width=5,
        n_samples=64,
        max_tokens=10,
        alpha=1.0,
        seed=None,
        add_bos=True,
        dataset_language="en",
        language_override=None,
    )
    identity = artifact_identity(
        "modelblocks_surprisal",
        context,
        [_FakeStim(s) for s in stim_ids],
        {"revision": "test-commit"},
    )
    write_manifest(path, identity, status="partial")


class _FakeDataset:
    def __init__(self, stim_ids: list[str], *, language: str = "en"):
        self._stims = [_FakeStim(s) for s in stim_ids]
        self.info = SimpleNamespace(language=language)

    def stimuli(self):
        return list(self._stims)


def _install_fake_dataset(cs, stim_ids: list[str]):
    """Patch the registry so dataset 'fake' returns the test's stimuli."""
    fake_ds = _FakeDataset(stim_ids)
    fake_spec = SimpleNamespace(factory=lambda **_: fake_ds)
    return patch.dict(cs.REGISTRY, {"fake": fake_spec}, clear=False)


def _install_fake_lm(
    cs,
    results_per_stim: dict[str, Any] | None = None,
    fail_on: set[str] | None = None,
    invoked: list[str] | None = None,
):
    """Make model construction return deterministic fake outputs."""
    fail_on = fail_on or set()
    results_per_stim = results_per_stim or {}

    def fake_lm(stim, extract=None):
        if invoked is not None:
            invoked.append(stim.stimulus_id)
        if stim.stimulus_id in fail_on:
            raise RuntimeError(f"forced fail on {stim.stimulus_id}")
        return results_per_stim.get(
            stim.stimulus_id,
            [
                FeatureModelOutput(
                    values={"log_p_observed": -2.0}, metadata={"fresh": True}
                )
            ],
        )

    return patch.object(cs, "build_model", lambda method, context: fake_lm)


def _install_fake_extract_keys(cs):
    return patch.object(
        cs,
        "extract_keys",
        lambda method, context: ["log_p_observed"],
    )


def _install_fake_tag(cs, tag: str = "fake_cell"):
    return patch.object(
        cs,
        "cache_stem",
        lambda method, context, dataset: tag,
    )


def _compute_fake_cell(cs, destination, **overrides):
    options = dict(
        method="modelblocks_surprisal",
        model_id="gpt2",
        K=5,
        n_samples=64,
        max_tokens=10,
        alpha=1.0,
        seed=17,
        out_dir=destination,
        add_bos=True,
        force=False,
        checkpoint_every=1,
    )
    return cs.compute_for("fake", **(options | overrides))


def test_interrupted_run_resumes_to_the_uninterrupted_payload(
    cs, tmp_path, monkeypatch
):
    ids = ["s0", "s1", "s2", "s3"]
    seen = []
    interrupt = True

    def score(stim, extract=None):
        seen.append(stim.stimulus_id)
        if interrupt and stim.stimulus_id == "s2":
            raise KeyboardInterrupt("injected interruption")
        return [
            FeatureModelOutput(
                values={"log_p_observed": -float(ids.index(stim.stimulus_id) + 1)}
            )
        ]

    monkeypatch.setattr(cs, "build_model", lambda *args: score)

    def run(destination):
        return _compute_fake_cell(cs, destination)

    with (
        _install_fake_dataset(cs, ids),
        _install_fake_extract_keys(cs),
        _install_fake_tag(cs),
    ):
        resumed_dir, clean_dir = tmp_path / "resumed", tmp_path / "clean"
        with pytest.raises(KeyboardInterrupt, match="injected interruption"):
            run(resumed_dir)
        partial = resumed_dir / "fake_cell.pkl.partial"
        assert set(pickle.loads(partial.read_bytes())) == {"s0", "s1"}
        assert not (resumed_dir / "fake_cell.pkl").exists()
        interrupt = False
        seen.clear()
        assert run(resumed_dir).status == "completed"
        assert seen == ["s2", "s3"]
        assert not partial.exists()
        assert run(clean_dir).status == "completed"
        resumed, clean = resumed_dir / "fake_cell.pkl", clean_dir / "fake_cell.pkl"
        assert pickle.loads(resumed.read_bytes()) == pickle.loads(clean.read_bytes())
        assert validate_artifact(resumed) == validate_artifact(clean)


def test_cli_missing_input_preserves_the_successful_cell(cs, tmp_path, monkeypatch):
    from replica.cli import main
    from replica.compute import cli

    events = []
    load = cs._load_dataset
    publish = cs.write_manifest

    def loading(name):
        events.append(f"load:{name}")
        if name == "missing":
            raise FileNotFoundError("synthetic input is unavailable")
        return load(name)

    def build(*args):
        events.append("build")

        def score(stim, extract=None):
            events.append(f"score:{stim.stimulus_id}")
            return [FeatureModelOutput(values={"log_p_observed": -2.0})]

        return score

    def publishing(path, identity, **kwargs):
        events.append(f"publish:{kwargs['status']}")
        return publish(path, identity, **kwargs)

    monkeypatch.setattr(cli, "expand_datasets", lambda name: ["missing", "fake"])
    monkeypatch.setattr(cs, "_load_dataset", loading)
    monkeypatch.setattr(cs, "build_model", build)
    monkeypatch.setattr(cs, "write_manifest", publishing)
    report = tmp_path / "run.json"
    args = [
        "compute",
        "--dataset",
        "synthetic",
        "--lm",
        "gpt2",
        "--method",
        "modelblocks_surprisal",
        "--out-dir",
        str(tmp_path),
        "--run-manifest",
        str(report),
    ]
    with (
        _install_fake_dataset(cs, ["s0"]),
        _install_fake_extract_keys(cs),
        _install_fake_tag(cs),
    ):
        with pytest.raises(SystemExit) as exc:
            main(args)
    assert exc.value.code == 1
    assert events == [
        "load:missing",
        "load:fake",
        "build",
        "score:s0",
        "publish:completed",
    ]
    assert [item["status"] for item in json.loads(report.read_text())["outcomes"]] == [
        "failed",
        "completed",
    ]
    assert validate_artifact(tmp_path / "fake_cell.pkl") is not None


def test_invalid_cli_options_never_load_or_build(cs, tmp_path, monkeypatch):
    from replica.cli import main

    monkeypatch.setattr(cs, "_load_dataset", lambda *args: pytest.fail("loaded input"))
    monkeypatch.setattr(cs, "build_model", lambda *args: pytest.fail("built model"))
    output = tmp_path / "output"
    with pytest.raises(SystemExit) as exc:
        main(
            [
                "compute",
                "--dataset",
                "fake",
                "--lm",
                "gpt2",
                "--unit",
                "invalid",
                "--out-dir",
                str(output),
            ]
        )
    assert exc.value.code == 2
    assert not output.exists()


def test_import_registers_the_unit_before_it_is_checked(cs, tmp_path, monkeypatch):
    """A module named with --import may register the segmentation --unit names."""
    from replica import segmentation
    from replica.cli import main

    monkeypatch.setattr(segmentation, "SEGMENTERS", dict(segmentation.SEGMENTERS))
    module = tmp_path / "myseg.py"
    module.write_text(
        "from replica.core import Span\n"
        "from replica.segmentation import Segmenter, register_segmenter\n"
        "@register_segmenter('whole')\n"
        "class Whole(Segmenter):\n"
        "    def __init__(self):\n"
        "        super().__init__('whole')\n"
        "    def segment(self, text):\n"
        "        return [Span(0, len(text))]\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    seen = {}
    monkeypatch.setattr(
        cs,
        "_load_dataset",
        lambda *a: seen.setdefault("loaded", True) and pytest.fail("loaded input"),
    )
    args = [
        "compute",
        "--dataset",
        "fake",
        "--lm",
        "gpt2",
        "--unit",
        "whole",
        "--out-dir",
        str(tmp_path / "o"),
    ]
    with pytest.raises(SystemExit) as exc:
        main(args)
    assert exc.value.code == 2 and "whole" not in segmentation.SEGMENTERS
    with pytest.raises(SystemExit) as exc:
        main(["compute", "--import", "myseg", *args[1:]])
    # The import registered the unit; the run then stopped at the unknown dataset, past the check.
    assert "whole" in segmentation.SEGMENTERS
    assert exc.value.code != 2 and "Unknown dataset" in str(exc.value.code)


def test_a_unit_bound_method_refuses_segments_with_whitespace(monkeypatch):
    from replica.compute.units import applicability
    from tests._stimuli import word_stimulus

    stimuli = [word_stimulus("ab cd ef", "s1")]
    from replica.core import AnnotationLayer, Span

    stimuli[0].add_layer(
        AnnotationLayer("pairs", [Span(0, 5), Span(6, 8)], ["ab cd", "ef"])
    )
    fit = applicability("similarity_surprisal_word", "pairs", stimuli)
    assert not fit.defined and "1 of 2 pairs units contain whitespace" in fit.reason
    # A method with a multi-word rule combines the two-word segment.
    fit = applicability("unigram_surprisal", "pairs", stimuli)
    assert fit.status == "aggregated" and fit.multiword_units == 1
    stimuli[0].add_layer(
        AnnotationLayer("halves", [Span(0, 2), Span(3, 5)], ["ab", "cd"])
    )
    assert applicability("unigram_surprisal", "halves", stimuli).status == "defined"
    stimuli[0].add_layer(AnnotationLayer("gap", [Span(0, 2), Span(2, 2)], ["ab", ""]))
    fit = applicability("unigram_surprisal", "gap", stimuli)
    assert (
        not fit.defined and "1 of 2 gap units are empty (for example '')" in fit.reason
    )


# ── Tests ───────────────────────────────────────────────────────────────────


def test_checkpoint_partial_is_written_during_run(cs, tmp_path):
    """After processing ``checkpoint_every`` stims, ``{tag}.pkl.partial`` exists
    with at least that many entries (atomic write, never half-formed)."""
    stim_ids = [f"s{i}" for i in range(8)]
    with (
        _install_fake_dataset(cs, stim_ids),
        _install_fake_lm(cs),
        _install_fake_extract_keys(cs),
        _install_fake_tag(cs),
    ):
        cs.compute_for(
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
            checkpoint_every=3,
        )

    # Successful completion deletes the partial; final pickle holds all stims.
    final = tmp_path / "fake_cell.pkl"
    partial = tmp_path / "fake_cell.pkl.partial"
    assert final.exists()
    assert not partial.exists()
    with final.open("rb") as f:
        obj = pickle.load(f)
    assert set(obj.keys()) == set(stim_ids)


def test_resume_from_partial_skips_done_stims(cs, tmp_path):
    """Pre-populate a partial pickle with results for stims 0-2;
    a fresh ``compute_for`` call must not re-invoke the LM for those."""
    stim_ids = [f"s{i}" for i in range(6)]
    partial = tmp_path / "fake_cell.pkl.partial"
    seeded = {
        "s0": [
            FeatureModelOutput(
                values={"log_p_observed": -2.0}, metadata={"from_seed": True}
            )
        ],
        "s1": [
            FeatureModelOutput(
                values={"log_p_observed": -2.0}, metadata={"from_seed": True}
            )
        ],
        "s2": [
            FeatureModelOutput(
                values={"log_p_observed": -2.0}, metadata={"from_seed": True}
            )
        ],
    }
    with partial.open("wb") as f:
        pickle.dump(seeded, f)
    _seed_manifest(partial, stim_ids)

    invoked: list[str] = []

    with (
        _install_fake_dataset(cs, stim_ids),
        _install_fake_lm(cs, invoked=invoked),
        _install_fake_extract_keys(cs),
        _install_fake_tag(cs),
    ):
        cs.compute_for(
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
            checkpoint_every=2,
        )

    # LM was only invoked for the *unfinished* stims (3, 4, 5).
    assert invoked == ["s3", "s4", "s5"]

    # Final pickle merges seeded + fresh.
    final = tmp_path / "fake_cell.pkl"
    with final.open("rb") as f:
        obj = pickle.load(f)
    assert set(obj.keys()) == set(stim_ids)
    assert obj["s0"][0].metadata["from_seed"] is True
    assert obj["s5"][0].metadata["fresh"] is True


def test_force_ignores_partial(cs, tmp_path):
    """``--force`` discards the partial and recomputes everything."""
    stim_ids = ["s0", "s1"]
    partial = tmp_path / "fake_cell.pkl.partial"
    with partial.open("wb") as f:
        pickle.dump({"s0": [{"stimulus_id": "s0", "from_seed": True}]}, f)

    invoked: list[str] = []

    with (
        _install_fake_dataset(cs, stim_ids),
        _install_fake_lm(cs, invoked=invoked),
        _install_fake_extract_keys(cs),
        _install_fake_tag(cs),
    ):
        cs.compute_for(
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
            force=True,
            checkpoint_every=2,
        )

    assert invoked == ["s0", "s1"]


def test_corrupt_partial_is_recovered_from(cs, tmp_path):
    """An unreadable ``.partial`` (e.g. truncated by a kill mid-write) must
    not break the next run, the script logs and starts fresh."""
    partial = tmp_path / "fake_cell.pkl.partial"
    partial.write_bytes(b"\x80\x05\x00")  # truncated pickle

    stim_ids = ["s0", "s1"]
    with (
        _install_fake_dataset(cs, stim_ids),
        _install_fake_lm(cs),
        _install_fake_extract_keys(cs),
        _install_fake_tag(cs),
    ):
        cs.compute_for(
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
            checkpoint_every=2,
        )

    final = tmp_path / "fake_cell.pkl"
    with final.open("rb") as f:
        obj = pickle.load(f)
    assert set(obj.keys()) == set(stim_ids)


def test_parallel_invocations_bail_on_lock(cs, tmp_path):
    """A concurrent request returns busy while the first producer holds the lock."""
    stim_ids = [f"s{i}" for i in range(6)]
    invoked: list[str] = []
    model_entered = Event()
    release_model = Event()

    def fake_lm(stim, extract=None):
        invoked.append(stim.stimulus_id)
        model_entered.set()
        if not release_model.wait(timeout=60):
            raise RuntimeError("Test did not release the model")
        return [FeatureModelOutput(values={"log_p_observed": -2.0})]

    def run_compute_for():
        return cs.compute_for(
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
            checkpoint_every=2,
        )

    with (
        _install_fake_dataset(cs, stim_ids),
        patch.object(cs, "build_model", lambda *_: fake_lm),
        _install_fake_extract_keys(cs),
        _install_fake_tag(cs),
        patch.object(cs, "environment_record", return_value={}),
        ThreadPoolExecutor(max_workers=2) as pool,
    ):
        first = pool.submit(run_compute_for)
        try:
            assert model_entered.wait(timeout=30), "Producer did not enter the model"
            second = pool.submit(run_compute_for)
            assert second.result(timeout=30).status == "busy"
            assert invoked == [stim_ids[0]]
        finally:
            release_model.set()
            outcome = first.result(timeout=30)
        assert outcome.status == "completed"

    # The first producer processes each stimulus once after being released.
    assert sorted(invoked) == stim_ids, f"got duplicate work: {invoked}"
    # The stable lock inode persists after its owner releases it.
    assert (tmp_path / "fake_cell.pkl.lock").exists()
    # The final pickle exists with all stims.
    final = tmp_path / "fake_cell.pkl"
    with final.open("rb") as f:
        obj = pickle.load(f)
    assert set(obj.keys()) == set(stim_ids)


def test_partial_survives_per_stim_failure(cs, tmp_path):
    """If one stim raises, it's added to ``failed`` and the cell is set aside,
    but the partial still captures the successful stims around it, so a re-run
    finishes the rest."""
    stim_ids = [f"s{i}" for i in range(5)]
    with (
        _install_fake_dataset(cs, stim_ids),
        _install_fake_lm(cs, fail_on={"s2"}),
        _install_fake_extract_keys(cs),
        _install_fake_tag(cs),
    ):
        cs.compute_for(
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
            checkpoint_every=2,
        )

    # Missing stimuli require quarantine.
    assert not (tmp_path / "fake_cell.pkl").exists()
    final = tmp_path / "fake_cell.suspect.pkl"
    with final.open("rb") as f:
        obj = pickle.load(f)
    # s2 raised → omitted; the others present.
    assert set(obj.keys()) == {"s0", "s1", "s3", "s4"}


@pytest.mark.parametrize("revision_changed", [False, True])
def test_entropy_resume_uses_logical_snapshot_not_relocated_cache(
    cs, tmp_path, monkeypatch, revision_changed
):
    from dataclasses import replace

    original_model = {"source": "org/model", "revision": "a" * 40}
    requested_model = dict(original_model)
    if revision_changed:
        requested_model["revision"] = "b" * 40
    context = ComputeContext("org/model", 5, 64, 10, 1.0, 7, True, "en", None)
    stimuli = [_FakeStim(s) for s in ("s0", "s1")]
    identity = artifact_identity(
        "forward_marginal_entropy", context, stimuli, original_model
    )
    partial = tmp_path / "fake_cell.pkl.partial"
    cached = FeatureModelOutput(values={"log_p_observed": -3.0})
    atomic_pickle({"s0": [cached]}, partial)
    write_manifest(partial, identity, status="partial")
    relocated = str(tmp_path / "relocated-snapshot")
    monkeypatch.setattr(
        cs, "resolve_model_snapshot", lambda *_: (relocated, requested_model)
    )
    invoked = []

    def build(method, received):
        assert method == "forward_marginal_entropy"
        assert received == replace(
            context, model_id=relocated, model_identity=requested_model
        )

        def score(stim, extract=None):
            invoked.append(stim.stimulus_id)
            return [FeatureModelOutput(values={"log_p_observed": -2.0})]

        return score

    monkeypatch.setattr(cs, "build_model", build)
    with (
        _install_fake_dataset(cs, [s.stimulus_id for s in stimuli]),
        _install_fake_extract_keys(cs),
        _install_fake_tag(cs),
    ):
        outcome = cs.compute_for(
            "fake",
            method="forward_marginal_entropy",
            model_id=context.model_id,
            K=5,
            n_samples=64,
            max_tokens=10,
            alpha=1.0,
            seed=7,
            out_dir=tmp_path,
            add_bos=True,
            force=False,
        )
    assert outcome.successful
    assert invoked == (["s0", "s1"] if revision_changed else ["s1"])
    with outcome.path.open("rb") as stream:
        outputs = pickle.load(stream)
    assert outputs["s0"][0].values["log_p_observed"] == (
        -2.0 if revision_changed else -3.0
    )


@pytest.mark.parametrize("slices", [[(0, 2), (2, 4)], [(2, 4), (0, 2)]])
def test_sharded_compute_matches_the_unsplit_payload(cs, tmp_path, slices):
    from replica.compute.shards import merge_shards

    ids = ["s0", "s1", "s2", "s3"]
    values = {
        sid: [FeatureModelOutput(values={"log_p_observed": -float(i + 1)})]
        for i, sid in enumerate(ids)
    }
    with (
        _install_fake_dataset(cs, ids),
        _install_fake_lm(cs, results_per_stim=values),
        _install_fake_extract_keys(cs),
        _install_fake_tag(cs),
    ):
        whole, shards = tmp_path / "whole", tmp_path / "shards"
        assert _compute_fake_cell(cs, whole).status == "completed"
        for selection in slices:
            assert (
                _compute_fake_cell(cs, shards, stim_slice=selection).status
                == "completed"
            )
        merged = shards / "merged.pkl"
        assert merge_shards(
            str(shards / "fake_cell.shard*.pkl"), merged, expect_stimuli=4
        ) == len(ids)
        assert pickle.loads(merged.read_bytes()) == pickle.loads(
            (whole / "fake_cell.pkl").read_bytes()
        )
        assert validate_artifact(merged)["n_stimuli"] == len(ids)


# ── Priming checkpoint of the sampled unigram source ────────────────────────

_PRIMED_STIMS = ["a", "b", "the", "dog", "cat"]
_PRIMED_STEM = "gpt2_mmc_unigram_surprisal_fake"


def _unigram_lm(contexts=C1, **source):
    """The mmc cell's model over a stub source whose sampler returns *contexts*."""
    return UnigramSurprisalLM(stub_source(contexts, **source), lookup="lm_surface")


class _RaisingLM:
    """Forward priming to *lm*; raise *error* when scoring stimulus *on*."""

    def __init__(self, lm, on: str | None, error: BaseException):
        self._lm = lm
        self._on = on
        self._error = error

    def prepopulate_cache(self, stimuli, target_layer="words", **checkpoint):
        self._lm.prepopulate_cache(stimuli, target_layer, **checkpoint)

    def __call__(self, stimulus, extract=None):
        if stimulus.stimulus_id == self._on:
            raise self._error
        return self._lm(stimulus, extract=extract)


def _run_mmc(cs, out_dir, lm, *, force=False, contexts=None, checkpoint_every=5):
    """Compute the mmc cell with *lm*, writing the priming file after each word."""

    def build(method, context):
        assert method == "mmc_unigram_surprisal"
        if contexts is not None:
            contexts.append(context)
        return lm

    with (
        _install_fake_dataset(cs, _PRIMED_STIMS),
        patch.object(cs, "build_model", build),
        patch.object(sampled, "PRIMING_WRITE_INTERVAL", 0.0),
    ):
        return cs.compute_for(
            "fake",
            method="mmc_unigram_surprisal",
            model_id="gpt2",
            K=5,
            n_samples=64,
            max_tokens=10,
            alpha=1.0,
            seed=None,
            out_dir=out_dir,
            add_bos=True,
            force=force,
            checkpoint_every=checkpoint_every,
        )


def _cell_values(path) -> dict[str, list[dict]]:
    with open(path, "rb") as stream:
        cell = pickle.load(stream)
    return {sid: [output.values for output in outputs] for sid, outputs in cell.items()}


@pytest.mark.parametrize("ending", ["completed", "quarantined", "cached"])
def test_the_primed_file_is_kept_until_the_cell_is_published(cs, tmp_path, ending):
    """A job killed after priming leaves the primed words for the next attempt,
    which scores from them without sampling. Publishing the cell, or finding it
    cached, removes the file and its writer's temporary file; a quarantined
    cell keeps it, as it keeps the partial checkpoint.
    """
    primed = tmp_path / f"{_PRIMED_STEM}.pkl.primed"
    leftover = temporary_path(primed)
    if ending == "cached":
        assert _run_mmc(cs, tmp_path, _unigram_lm()).status == "completed"
        primed.write_bytes(b"left by a job killed after publishing")
    else:
        with pytest.raises(Killed):
            _run_mmc(cs, tmp_path, _RaisingLM(_unigram_lm(), "the", Killed()))
    assert primed.exists()
    leftover.write_bytes(b"a write cut short")

    failing = "dog" if ending == "quarantined" else None
    lm = _RaisingLM(_unigram_lm(generate=refuse_sampling), failing, RuntimeError())
    assert _run_mmc(cs, tmp_path, lm).status == ending
    if ending == "quarantined":
        assert primed.exists()
    else:
        assert not primed.exists()
        assert not leftover.exists()


def test_priming_checkpoint_leaves_identity_and_values_unchanged(cs, tmp_path):
    """A cell resumed from a priming checkpoint publishes the bytes, name and
    manifest identity of an uninterrupted cell, and the checkpoint reaches
    neither the model's settings nor the manifest.
    """
    contexts: list[ComputeContext] = []
    whole = _run_mmc(cs, tmp_path / "whole", _unigram_lm(), contexts=contexts)
    with pytest.raises(Killed):
        _run_mmc(
            cs, tmp_path / "resumed", _unigram_lm(kill_after=20), contexts=contexts
        )
    with open(tmp_path / "resumed" / f"{_PRIMED_STEM}.pkl.primed", "rb") as stream:
        assert pickle.load(stream)["estimates"]
    resumed = _run_mmc(cs, tmp_path / "resumed", _unigram_lm(C2), contexts=contexts)

    assert whole.status == resumed.status == "completed"
    assert whole.path is not None and resumed.path is not None
    assert resumed.path.name == whole.path.name == f"{_PRIMED_STEM}.pkl"
    manifests = [
        json.loads(manifest_path(outcome.path).read_text())
        for outcome in (whole, resumed)
    ]
    assert manifests[0]["identity"] == manifests[1]["identity"]
    assert manifests[0]["payload_sha256"] == manifests[1]["payload_sha256"]
    for manifest in manifests:
        text = json.dumps(manifest)
        assert "primed" not in text
        assert str(tmp_path) not in text
    assert len(contexts) == 3
    assert contexts[0] == contexts[1] == contexts[2]


@pytest.mark.parametrize("loss", ["deleted", "truncated", "forced", "other_model"])
def test_primed_cell_is_scored_against_one_context_set(cs, tmp_path, monkeypatch, loss):
    """A partial checkpoint whose priming checkpoint is lost, replaced or
    foreign holds stimuli scored against contexts the next attempt cannot use.
    The resumed cell equals an uninterrupted run over the contexts it primes with.
    """
    first = _cell_values(_run_mmc(cs, tmp_path / "c1", _unigram_lm()).path)
    second = _cell_values(_run_mmc(cs, tmp_path / "c2", _unigram_lm(C2)).path)
    assert first["a"] != second["a"]

    cell = tmp_path / "cell"
    killed = _RaisingLM(_unigram_lm(), "the", Killed())
    with pytest.raises(Killed):
        _run_mmc(cs, cell, killed, checkpoint_every=1)
    partial = cell / f"{_PRIMED_STEM}.pkl.partial"
    with open(partial, "rb") as stream:
        assert sorted(pickle.load(stream)) == ["a", "b"]
    primed = cell / f"{_PRIMED_STEM}.pkl.primed"
    lm = _unigram_lm(C2)
    if loss == "deleted":
        primed.unlink()
    elif loss == "truncated":
        primed.write_bytes(primed.read_bytes()[:-40])
    elif loss == "forced":
        # --force replaces the primed file with one over the contexts it draws.
        with pytest.raises(Killed):
            _run_mmc(cs, cell, _unigram_lm(C2, kill_after=1), force=True)
        lm = _unigram_lm(generate=refuse_sampling)
    else:
        # A changed model or tokenizer file changes the content digest in the
        # cell identity, which owns the primed file.
        monkeypatch.setattr(
            cs,
            "resolve_model_snapshot",
            lambda method, context: (
                context.model_id,
                {"revision": "test-commit", "sha256": "1" * 64},
            ),
        )
    assert partial.exists()

    resumed = _run_mmc(cs, cell, lm, checkpoint_every=1)
    assert resumed.status == "completed"
    assert _cell_values(resumed.path) == second
