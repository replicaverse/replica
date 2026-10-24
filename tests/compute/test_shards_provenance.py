"""Shard inputs the merge must refuse: gaps, overlapping slices, quarantined
and unfinished payloads."""

import hashlib
import pickle

import pytest

from replica.artifacts import write_manifest
from replica.compute.shards import merge_shards


def _write(path, payload):
    with open(path, "wb") as stream:
        pickle.dump(payload, stream)
    _sidecar(path, hashlib.sha256(path.name.encode()).hexdigest())
    return path


def _sidecar(path, corpus, *, status="completed"):
    identity = {
        "schema_version": 1,
        "method": "surprisal",
        "options": {"model_id": "gpt2"},
        "model": {"revision": "r"},
        "corpus_sha256": corpus,
        "n_stimuli": len(pickle.loads(path.read_bytes())),
        "source_sha256": "s" * 64,
        "versions": {"torch": "2.13"},
    }
    write_manifest(path, identity, status=status)


def test_a_gap_between_shards_aborts(tmp_path):
    _write(tmp_path / "tag.shard00000_00002.pkl", {"s0": [1], "s1": [2]})
    _write(tmp_path / "tag.shard00003_00005.pkl", {"s3": [4], "s4": [5]})

    with pytest.raises(SystemExit, match="gap"):
        merge_shards(str(tmp_path / "tag.shard*.pkl"), tmp_path / "tag.pkl")
    assert not (tmp_path / "tag.pkl").exists()


def test_shards_that_start_after_the_first_stimulus_abort(tmp_path):
    _write(tmp_path / "tag.shard00002_00004.pkl", {"s2": [3], "s3": [4]})

    with pytest.raises(SystemExit, match="stimulus 0"):
        merge_shards(str(tmp_path / "tag.shard*.pkl"), tmp_path / "tag.pkl")


def test_overlapping_slices_with_distinct_ids_abort(tmp_path):
    _write(tmp_path / "tag.shard00000_00002.pkl", {"s0": [1], "s1": [2]})
    _write(tmp_path / "tag.shard00001_00003.pkl", {"x": [8], "y": [9]})

    with pytest.raises(SystemExit, match="overlap"):
        merge_shards(str(tmp_path / "tag.shard*.pkl"), tmp_path / "tag.pkl")


def test_a_quarantined_shard_matched_by_the_pattern_aborts(tmp_path):
    good = _write(tmp_path / "tag.shard00000_00001.pkl", {"s0": [1]})
    _sidecar(good, "a" * 64)
    suspect = _write(tmp_path / "tag.shard00001_00002.suspect.pkl", {"s1": [2]})
    _sidecar(suspect, "b" * 64, status="quarantined")

    with pytest.raises(SystemExit, match="suspect"):
        merge_shards(str(tmp_path / "tag.shard*.pkl"), tmp_path / "tag.pkl")
    assert not (tmp_path / "tag.pkl").exists()


@pytest.mark.parametrize("status", ["partial", "quarantined"])
def test_a_shard_whose_sidecar_is_not_completed_aborts(tmp_path, status):
    first = _write(tmp_path / "tag.shard00000_00001.pkl", {"s0": [1]})
    _sidecar(first, "a" * 64)
    second = _write(tmp_path / "tag.shard00001_00002.pkl", {"s1": [2]})
    _sidecar(second, "b" * 64, status=status)

    with pytest.raises(ValueError, match="incomplete"):
        merge_shards(str(tmp_path / "tag.shard*.pkl"), tmp_path / "tag.pkl")
    assert not (tmp_path / "tag.pkl").exists()


def test_a_short_shard_set_is_refused(tmp_path):
    """Slices that stop before the corpus ends do not merge into a complete cell."""
    _write(tmp_path / "tag.shard00000_00002.pkl", {"s0": [1], "s1": [2]})
    _sidecar(tmp_path / "tag.shard00000_00002.pkl", "c" * 64)
    _write(tmp_path / "tag.shard00002_00004.pkl", {"s2": [3], "s3": [4]})
    _sidecar(tmp_path / "tag.shard00002_00004.pkl", "c" * 64)

    with pytest.raises(SystemExit, match="5 stimuli are missing"):
        merge_shards(
            str(tmp_path / "tag.shard*.pkl"), tmp_path / "tag.pkl", expect_stimuli=9
        )
    assert not (tmp_path / "tag.pkl").exists()


def test_a_complete_shard_set_merges(tmp_path):
    """Slices reaching the corpus end merge and keep every stimulus."""
    _write(tmp_path / "tag.shard00000_00002.pkl", {"s0": [1], "s1": [2]})
    _sidecar(tmp_path / "tag.shard00000_00002.pkl", "c" * 64)
    _write(tmp_path / "tag.shard00002_00004.pkl", {"s2": [3], "s3": [4]})
    _sidecar(tmp_path / "tag.shard00002_00004.pkl", "c" * 64)

    assert (
        merge_shards(
            str(tmp_path / "tag.shard*.pkl"), tmp_path / "tag.pkl", expect_stimuli=4
        )
        == 4
    )
    assert (tmp_path / "tag.pkl").exists()


def test_unregistered_corpus_requires_explicit_size(tmp_path):
    """Completed shards alone cannot establish the full corpus length."""
    _write(tmp_path / "tag.shard00000_00002.pkl", {"s0": [1], "s1": [2]})
    _sidecar(tmp_path / "tag.shard00000_00002.pkl", "c" * 64)

    with pytest.raises(ValueError, match="pass expect_stimuli"):
        merge_shards(str(tmp_path / "tag.shard*.pkl"), tmp_path / "tag.pkl")
    assert not (tmp_path / "tag.pkl").exists()
    for invalid in (True, 2.0, 0, -1):
        with pytest.raises(ValueError, match="positive integer"):
            merge_shards(
                str(tmp_path / "tag.shard*.pkl"),
                tmp_path / "tag.pkl",
                expect_stimuli=invalid,
            )


def test_a_corpus_that_cannot_be_loaded_aborts(tmp_path, monkeypatch):
    """A registered corpus that fails to load stops the merge."""
    _write(
        tmp_path / "gpt2_surprisal_provo.shard00000_00002.pkl", {"s0": [1], "s1": [2]}
    )
    _sidecar(tmp_path / "gpt2_surprisal_provo.shard00000_00002.pkl", "c" * 64)
    monkeypatch.setattr(
        "replica.compute.runner._load_dataset",
        lambda name: (_ for _ in ()).throw(RuntimeError("no data")),
    )

    with pytest.raises(SystemExit, match="cannot load provo"):
        merge_shards(
            str(tmp_path / "gpt2_surprisal_provo.shard*.pkl"),
            tmp_path / "gpt2_surprisal_provo.pkl",
        )
