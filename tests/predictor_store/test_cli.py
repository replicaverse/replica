"""The ``replica predictors`` command line and the wave labels it derives."""

from pathlib import Path

import pytest

from replica.predictor_store import cli
from replica.predictor_store import export as export_module
from replica.predictor_store import verify as verify_module
from replica.predictor_store.cells import parse_source, wave_label
from replica.predictor_store.readme import render_readme
from tests.predictor_store._cells import DATASET, ToyCorpora, word_payload, write_cell


def test_wave_label_is_the_first_ancestor_outside_the_run_nesting(tmp_path):
    out = tmp_path / "recompute" / "wave2" / "out"
    out.mkdir(parents=True)
    verify = tmp_path / "recompute" / "wave2" / "verify" / "verification.json"
    verify.parent.mkdir()
    verify.write_text("[]")
    assert wave_label(out) == "wave2"
    assert wave_label(verify) == "wave2"
    assert parse_source(str(out)) == ("wave2", out)
    assert parse_source(f"tokens={out}") == ("tokens", out)


def test_export_verify_and_publish_run_from_the_command_line(
    tmp_path, monkeypatch, capsys
):
    out = tmp_path / "wave7" / "out"
    write_cell(out, f"gpt2_surprisal_{DATASET}", word_payload(), method="surprisal")
    monkeypatch.setattr(cli, "Hub", lambda repo, revision: pytest.fail("no hub"))
    monkeypatch.setattr(export_module, "RegistryCorpora", ToyCorpora)
    monkeypatch.setattr(verify_module, "RegistryCorpora", ToyCorpora)
    stage = tmp_path / "stage"
    assert cli.main(["export", "--out", str(stage), "--from", str(out)]) == 0
    assert "cells exported: 1" in capsys.readouterr().out
    assert cli.main(["verify", "--stage", str(stage), "--from", str(out)]) == 0
    assert "OK" in capsys.readouterr().out.splitlines()[-1]
    calls = []

    class Api:
        def upload_folder(self, **kwargs):
            calls.append(kwargs)

        def upload_file(self, **kwargs):
            calls.append(kwargs)

        def list_repo_commits(self, *args, **kwargs):
            return []

        def list_repo_tree(self, *args, **kwargs):
            return []

    monkeypatch.setattr("huggingface_hub.HfApi", Api)
    publish = [
        "publish",
        "--stage",
        str(stage),
        "--repo",
        "example/predictors",
        "--create-pr",
        "--message",
        "Release",
    ]
    with pytest.raises(ValueError, match="without --merge-hub"):
        cli.main(publish)
    assert calls == []
    assert cli.main([*publish, "--initial-release"]) == 0
    assert calls[0]["create_pr"] is True and calls[0]["commit_message"] == "Release"
    assert f"{DATASET}.parquet" in calls[0]["allow_patterns"]
    assert calls[0]["delete_patterns"] is None
    readme = render_readme(stage)
    assert "—" not in readme
    assert "| wave7 | 1 |" in readme
    assert "| positional | words | 1 |" in readme
    assert "verification/wave2.json" not in readme


@pytest.mark.parametrize(
    "option", [["--repo", "example/predictors"], ["--revision", "a" * 40]]
)
def test_export_refuses_a_hub_option_without_merge_hub(tmp_path, capsys, option):
    """--repo and --revision name the store to merge, so they need --merge-hub."""
    out = tmp_path / "out"
    write_cell(out, f"gpt2_surprisal_{DATASET}", word_payload(), method="surprisal")
    stage = tmp_path / "stage"
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["export", "--out", str(stage), "--from", str(out), *option])
    assert exit_info.value.code == 2
    assert "--merge-hub" in capsys.readouterr().err
    assert not stage.exists()


def test_verify_exits_nonzero_on_a_difference(tmp_path, monkeypatch):
    out = tmp_path / "out"
    write_cell(out, f"gpt2_surprisal_{DATASET}", word_payload(), method="surprisal")
    monkeypatch.setattr(export_module, "RegistryCorpora", ToyCorpora)
    monkeypatch.setattr(verify_module, "RegistryCorpora", ToyCorpora)
    stage = tmp_path / "stage"
    assert cli.main(["export", "--out", str(stage), "--from", str(out)]) == 0
    Path(stage / f"{DATASET}.parquet").unlink()
    assert cli.main(["verify", "--stage", str(stage), "--from", str(out)]) == 1
