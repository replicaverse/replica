"""The ``replica`` console CLI (replica.cli).

Pure-Python dispatch tests, no network: every path either prints or exits
before any download is attempted. The fetch core lives in
``replica.dataset.fetch`` (exercised by tests/dataset/fetch/test_fetch.py); these
tests pin the argument parsing and subcommand routing of the console entry.
"""

import os
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path
from subprocess import CompletedProcess

import pytest

import replica.cli as cli_module
from replica.cli import main


@pytest.mark.parametrize(
    "command", [[], ["compute"], ["compute", "merge-shards"], ["results"], ["fetch"]]
)
def test_help_needs_no_models_network_or_writes(tmp_path, command):
    probe = """
import importlib.abc, os, sys

class NoModels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'transformers', 'rpy2', 'tuned_lens'}:
            raise AssertionError('help imported a model backend: ' + fullname)

def audit(event, args):
    if event in {'socket.connect', 'socket.bind', 'subprocess.Popen', 'os.mkdir',
                 'os.remove', 'os.rename', 'os.rmdir'}:
        raise AssertionError('help attempted side effect: ' + event)
    if event == 'open':
        flags = args[2]
        if flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
            raise AssertionError('help attempted a file write')

sys.meta_path.insert(0, NoModels())
sys.addaudithook(audit)
from replica.cli import main
main(sys.argv[1:])
"""
    environment = {
        **os.environ,
        "PYTHONPATH": str(Path(cli_module.__file__).resolve().parents[1]),
        "PYTHONDONTWRITEBYTECODE": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    result = subprocess.run(
        [sys.executable, "-B", "-c", probe, *command, "--help"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("argv", [["datasets"], ["fetch", "--list"]])
def test_dataset_listing_routes_to_shared_implementation(monkeypatch, argv):
    calls = []
    monkeypatch.setattr(
        "replica.dataset.fetch.list_datasets",
        lambda: calls.append("list"),
    )
    main(argv)
    assert calls == ["list"]


def test_dataset_describe_prints_loader_options(capsys):
    main(["datasets", "--describe", "meco"])
    output = capsys.readouterr().out
    assert "Family: meco" in output
    assert "l1_w2" in output
    assert "en_uk" in output


def test_dataset_loadable_lists_catalog_keys(capsys):
    main(["datasets", "--loadable"])
    output = capsys.readouterr().out
    assert "Loadable dataset keys" in output
    assert "meco_l1_w2_en_uk" in output
    assert "provo" in output


def test_dataset_facts_summarizes_the_data_free_tier(capsys):
    main(["datasets", "--facts"])
    output = capsys.readouterr().out
    assert "provo" in output
    assert "meco_l1_w2_en_uk" in output


def test_dataset_facts_forwards_measure_and_snapshot_flags(monkeypatch):
    forwarded = []
    monkeypatch.setattr(
        "replica.dataset.catalog.facts.main",
        lambda argv: forwarded.append(argv) or 0,
    )
    main(["datasets", "--facts", "--measure", "--write-snapshot"])
    assert forwarded == [["--measure", "--write-snapshot"]]


def test_dataset_facts_refuses_a_snapshot_write_without_measuring(monkeypatch, capsys):
    written = []
    monkeypatch.setattr(
        "replica.dataset.catalog.facts.write_snapshot",
        lambda *args, **kwargs: written.append(args),
    )
    with pytest.raises(SystemExit) as exc:
        main(["datasets", "--facts", "--write-snapshot"])
    assert exc.value.code == 2
    assert written == []
    assert "--write-snapshot requires --measure" in capsys.readouterr().err


def test_dataset_facts_propagates_a_nonzero_status(monkeypatch):
    monkeypatch.setattr("replica.dataset.catalog.facts.main", lambda argv: 3)
    with pytest.raises(SystemExit) as exc:
        main(["datasets", "--facts"])
    assert exc.value.code == 3


def test_dataset_describe_unknown_key_exits_nonzero(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["datasets", "--describe", "not_a_dataset"])
    assert exc.value.code == 2
    assert "unknown dataset" in capsys.readouterr().err.lower()


def test_fetch_unknown_dataset_exits_nonzero():
    with pytest.raises(SystemExit) as exc:
        main(["fetch", "definitely_not_a_dataset"])
    # run_fetch passes the message string to sys.exit(), so it is exc.value.code
    # (a non-empty string is also a non-zero exit). Pin the user-facing wording.
    assert "unknown dataset" in str(exc.value.code).lower()


def test_fetch_without_names_prints_help_and_exits(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["fetch"])
    assert exc.value.code == 1
    assert "usage" in capsys.readouterr().out.lower()


def test_no_subcommand_prints_help_and_exits(capsys):
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code == 1
    assert "usage" in capsys.readouterr().out.lower()


def test_version_prints_installed_package_version(capsys):
    main(["version"])
    out = capsys.readouterr().out.strip()
    assert out == version("replicaverse")


def test_prepare_fixations_routes_to_shared_implementation(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "replica.dataset.eye_tracking.meco.prepare.run_prepare",
        lambda names, force=False: calls.append((names, force)),
    )
    main(["prepare-fixations", "meco_l2_w1", "--force"])
    assert calls == [(["meco_l2_w1"], True)]


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (
            ["check"],
            [
                ["python", "-m", "ruff", "check", "."],
                ["python", "-m", "ruff", "format", "--check", "."],
            ],
        ),
        (
            ["check", "--fix"],
            [
                ["python", "-m", "ruff", "check", ".", "--fix"],
                ["python", "-m", "ruff", "format", "."],
            ],
        ),
    ],
)
def test_check_runs_canonical_ruff_commands(monkeypatch, argv, expected):
    calls = []

    def fake_run(command, *, check, cwd):
        assert check is True
        calls.append((command, cwd))
        return CompletedProcess(command, 0)

    monkeypatch.setattr("replica.cli.sys.executable", "python")
    monkeypatch.setattr("replica.cli.subprocess.run", fake_run)

    main(argv)

    repo_root = Path(cli_module.__file__).resolve().parents[2]
    quality_commands = [
        ["python", "scripts/checks/source/references.py"],
        ["python", "scripts/checks/source/test_policy.py"],
        ["python", "scripts/checks/source/import_boundaries.py"],
        ["python", "scripts/checks/source/docstrings.py"],
        ["python", "-m", "mypy"],
        ["python", "scripts/checks/distribution/dataset_metadata.py"],
    ]
    assert calls == [(command, repo_root) for command in [*expected, *quality_commands]]


def test_compute_forwards_argv_verbatim(monkeypatch):
    seen = []
    monkeypatch.setattr("replica.compute.cli.main", seen.append)
    main(["compute", "--dataset", "provo", "--lm", "gpt2"])
    assert seen == [["--dataset", "provo", "--lm", "gpt2"]]


def test_sweep_is_not_a_library_command(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["sweep", "--dry-run"])
    assert exc.value.code == 2
    assert "invalid choice: 'sweep'" in capsys.readouterr().err


@pytest.mark.parametrize("status", [0, 1])
def test_results_forwards_argv_and_exit_status(monkeypatch, status):
    seen = []

    def results_main(argv):
        seen.append(argv)
        return status

    monkeypatch.setattr("replica.studies.records.cli.main", results_main)
    with pytest.raises(SystemExit) as exc:
        main(["results", "stale", "example"])
    assert exc.value.code == status
    assert seen == [["stale", "example"]]


def test_results_help_reaches_delegate_parser(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["results", "--help"])
    assert exc.value.code == 0
    output = capsys.readouterr().out
    assert "replica results" in output
    assert "stamp" in output and "publish" in output


def test_compute_help_reaches_delegate_parser(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["compute", "--help"])
    assert exc.value.code == 0
    assert "--dataset" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("command", "prog"),
    [
        (["compute"], "replica compute"),
        (["compute", "merge-shards"], "replica compute merge-shards"),
    ],
)
def test_delegated_help_names_the_subcommand(capsys, command, prog):
    with pytest.raises(SystemExit) as exc:
        main([*command, "--help"])
    assert exc.value.code == 0
    assert capsys.readouterr().out.startswith(f"usage: {prog} ")


@pytest.mark.parametrize(
    "flag", ["--measure", "--json", "--write-snapshot", "--only-missing"]
)
def test_dataset_facts_flags_without_facts_exit_nonzero(monkeypatch, capsys, flag):
    listed = []
    monkeypatch.setattr(
        "replica.dataset.fetch.list_datasets", lambda: listed.append("list")
    )
    with pytest.raises(SystemExit) as exc:
        main(["datasets", flag])
    assert exc.value.code == 2
    assert listed == []
    err = capsys.readouterr().err
    assert "usage: replica datasets" in err
    assert f"{flag}: only valid with --facts" in err
