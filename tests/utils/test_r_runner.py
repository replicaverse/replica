"""Process execution tests use Python as a stand-in executable, without R."""

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from replica.utils.r import RScriptError, run_r_script


@pytest.mark.parametrize("isolated", [False, True])
def test_success_and_failure_capture_streams(tmp_path, isolated):
    script = tmp_path / "runner.py"
    script.write_text(
        'import sys;print("output");print("diagnostic",file=sys.stderr);sys.exit(int(sys.argv[1]))'
    )
    kw = dict(executable=sys.executable, r_options=(), isolate_process_group=isolated)
    result = run_r_script(script, ["0"], **kw)
    assert result.stdout.strip() == "output"
    with pytest.raises(RScriptError) as exc:
        run_r_script(script, ["4"], **kw)
    assert exc.value.returncode == 4
    assert exc.value.stderr.strip() == "diagnostic"


@pytest.mark.skipif(
    os.name != "posix" or not Path("/proc").is_dir(),
    reason="POSIX process-group lifecycle check",
)
def test_timeout_terminates_children_and_retains_output(tmp_path):
    script = tmp_path / "parent.py"
    pid = tmp_path / "child.pid"
    script.write_text("""import subprocess,sys,time
from pathlib import Path
child=subprocess.Popen([sys.executable,"-c","import time; time.sleep(60)"])
Path(sys.argv[1]).write_text(str(child.pid))
print("started",flush=True)
time.sleep(60)
""")
    with pytest.raises(subprocess.TimeoutExpired) as exc:
        run_r_script(
            script,
            [pid],
            executable=sys.executable,
            r_options=(),
            timeout=1,
            isolate_process_group=True,
        )
    assert "started" in exc.value.stdout
    state = Path("/proc") / pid.read_text() / "stat"
    # A killed child can briefly remain as a zombie until init reaps it; once
    # reaped, reading its /proc entry fails.
    deadline = time.monotonic() + 2
    while True:
        try:
            if state.read_text().split()[2] == "Z":
                break
        except (FileNotFoundError, ProcessLookupError):
            break
        assert time.monotonic() < deadline, "child survived process-group cancellation"
        time.sleep(0.01)
