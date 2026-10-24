"""Exercise the process boundary with real child processes and zipped resources."""

import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from replica.utils.r import RScriptError, run_r_script


def test_resource_is_available_until_process_finishes(tmp_path):
    archive = tmp_path / "package.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr(
            "new_study/model.R",
            "import json, pathlib, sys\n"
            "print(json.dumps([pathlib.Path(__file__).exists(), sys.argv[1:], str(pathlib.Path.cwd())]))\n",
        )
    argument = 'a path with spaces; $(touch unwanted) "quoted"'
    with zipfile.ZipFile(archive) as bundle:
        resource = zipfile.Path(bundle, "new_study/model.R")
        result = run_r_script(
            resource,
            [argument, tmp_path / "some input.tsv"],
            executable=sys.executable,
            r_options=(),
            cwd=tmp_path,
        )
    exists, args, cwd = json.loads(result.stdout)
    assert exists
    assert args == [argument, str(tmp_path / "some input.tsv")]
    assert cwd == str(tmp_path)
    assert not (tmp_path / "unwanted").exists()
    assert not Path(result.args[1]).exists(), "extracted script was not cleaned up"


def test_relative_script_path_is_independent_of_child_working_directory(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    script = Path("model.R")
    script.write_text("from pathlib import Path\nprint(Path.cwd())\n")
    child_directory = tmp_path / "working"
    child_directory.mkdir()
    result = run_r_script(
        script, executable=sys.executable, r_options=(), cwd=child_directory
    )
    assert result.stdout.strip() == str(child_directory)


def test_failure_preserves_exit_status_and_both_output_streams(tmp_path):
    script = tmp_path / "failure.R"
    script.write_text(
        "import sys\nprint('input context')\n"
        "print('model failed', file=sys.stderr)\nsys.exit(7)\n"
    )
    with pytest.raises(RScriptError, match="status 7") as caught:
        run_r_script(script, executable=sys.executable, r_options=())
    assert caught.value.returncode == 7
    assert caught.value.stdout == "input context\n"
    assert caught.value.stderr == "model failed\n"
    assert str(script) in str(caught.value)


def test_long_running_references_can_keep_streaming_diagnostics(tmp_path, capfd):
    script = tmp_path / "reference.R"
    script.write_text(
        "import sys\nprint('fit progress')\nprint('warning', file=sys.stderr)\nsys.exit(3)\n"
    )
    with pytest.raises(RScriptError, match="status 3") as caught:
        run_r_script(
            script, executable=sys.executable, r_options=(), capture_output=False
        )
    output = capfd.readouterr()
    assert output.out == "fit progress\n"
    assert output.err == "warning\n"
    assert caught.value.stdout == caught.value.stderr == ""


def test_timeout_is_explicit_and_terminates_the_child(tmp_path):
    script = tmp_path / "slow.R"
    script.write_text("import time\ntime.sleep(30)\n")
    with pytest.raises(subprocess.TimeoutExpired):
        run_r_script(script, executable=sys.executable, r_options=(), timeout=0.1)


def test_missing_resource_fails_before_launching_the_process(tmp_path):
    with pytest.raises(FileNotFoundError, match="resource does not exist"):
        run_r_script(tmp_path / "absent.R", executable="also-absent")


def test_missing_executable_is_distinct_from_an_r_fit_failure(tmp_path):
    script = tmp_path / "model.R"
    script.write_text("print(1)\n")
    with pytest.raises(FileNotFoundError):
        run_r_script(script, executable=tmp_path / "absent-Rscript")


@pytest.mark.needs_r
def test_real_r_receives_options_and_arguments(tmp_path):
    script = tmp_path / "echo.R"
    script.write_text('cat(commandArgs(trailingOnly=TRUE), sep="\\n")\n')
    result = run_r_script(script, ["a b", "--some-flag"])
    assert result.stdout.splitlines() == ["a b", "--some-flag"]
