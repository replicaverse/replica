"""Shared assertion for study replay commands."""

import importlib

import pytest


def assert_replay_rejects_stale_destination(tmp_path, module, arguments):
    """A stale accepted file cannot survive as a new command's apparent result.

    Args:
        tmp_path: Isolated pytest filesystem fixture.
        module: Study replay entry point, relative to the studies package.
        arguments: Valid command options with deliberately unavailable inputs.
    """
    output = tmp_path / "previous-result"
    output.mkdir()
    stale = output / "decisions.tsv"
    stale.write_bytes(b"accepted\n1\n")
    main = importlib.import_module("replica.studies." + module).main
    with pytest.raises(FileExistsError):
        main([*arguments, "--out", str(output)])
    assert stale.read_bytes() == b"accepted\n1\n"
    assert list(output.iterdir()) == [stale]
