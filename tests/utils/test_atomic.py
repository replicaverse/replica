"""Atomic pickling through a temporary file beside the target."""

import os
import pickle

import pytest

from replica.utils.atomic import atomic_pickle, temporary_path


class _Unpicklable:
    def __reduce__(self):
        raise RuntimeError("cannot pickle")


def test_the_target_is_replaced_and_no_temporary_file_is_left(tmp_path):
    target = tmp_path / "atom.pkl"
    atomic_pickle({"a": 1}, target)
    atomic_pickle({"b": 2}, target)
    assert pickle.loads(target.read_bytes()) == {"b": 2}
    assert list(tmp_path.iterdir()) == [target]


def test_a_failed_write_leaves_the_previous_content(tmp_path):
    target = tmp_path / "atom.pkl"
    atomic_pickle({"a": 1}, target)
    with pytest.raises(RuntimeError):
        atomic_pickle({"b": _Unpicklable()}, target)
    assert pickle.loads(target.read_bytes()) == {"a": 1}


@pytest.mark.parametrize("fsync", [False, True])
def test_fsync_flushes_the_temporary_file_before_the_rename(
    tmp_path, monkeypatch, fsync
):
    target = tmp_path / "atom.pkl"
    synced: list[tuple[bool, bool]] = []
    monkeypatch.setattr(
        os,
        "fsync",
        lambda _fd: synced.append((temporary_path(target).exists(), target.exists())),
    )
    atomic_pickle({"a": 1}, target, fsync=fsync)
    assert synced == ([(True, False)] if fsync else [])
