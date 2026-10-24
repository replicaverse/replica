"""Execute packaged R scripts without depending on a statistical implementation.

Adapters own their input tables, formulas and output schemas. This module owns
only resource lifetime, process execution and diagnostics, so reference scripts
can use it without sharing the numerical code they validate.
"""

import os
import shlex
import signal
import subprocess
from collections.abc import Sequence
from contextlib import suppress
from importlib.resources import as_file
from importlib.resources.abc import Traversable
from pathlib import Path


class RScriptError(RuntimeError):
    """An R process failed; full captured output remains available for inspection.

    Args:
        result: Completed process with a nonzero exit status.
    """

    def __init__(self, result: subprocess.CompletedProcess[str]):
        """Capture diagnostics from the failed R process."""
        self.command = result.args
        self.returncode = result.returncode
        self.stdout = result.stdout or ""
        self.stderr = result.stderr or ""
        super().__init__(
            f"R script exited with status {result.returncode}: "
            f"{shlex.join(result.args)}\n"
            f"stdout:\n{self.stdout[-4000:]}\n"
            f"stderr:\n{self.stderr[-8000:]}"
        )


def _run_process_group(command, *, cwd, timeout, capture_output):
    # Stan can start compiler workers and parallel chains. Kill the POSIX group
    # on cancellation, not just the R parent. Other platforms kill the parent.
    with subprocess.Popen(
        command,
        cwd=cwd,
        text=True,
        start_new_session=(os.name == "posix"),
        stdout=subprocess.PIPE if capture_output else None,
        stderr=subprocess.PIPE if capture_output else None,
    ) as process:
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except BaseException as error:
            with suppress(ProcessLookupError):
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
            stdout, stderr = process.communicate()
            if isinstance(error, subprocess.TimeoutExpired):
                raise subprocess.TimeoutExpired(
                    command, timeout, output=stdout, stderr=stderr
                ) from error
            raise
        return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)


def run_r_script(
    script: Traversable,
    args: Sequence[str | os.PathLike[str]] = (),
    *,
    executable: str | os.PathLike[str] = "Rscript",
    r_options: Sequence[str] = ("--vanilla",),
    cwd: Path | None = None,
    timeout: float | None = None,
    capture_output: bool = True,
    isolate_process_group: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Run an R resource synchronously and return its process result.

    Resource extraction remains alive until the process finishes, including
    for packages imported from a zip file. Arguments are passed directly,
    without shell interpretation. Environment variables are inherited; adapters
    decide whether R startup files are allowed through ``r_options``.

    Args:
        script: Packaged resource from ``importlib.resources.files`` or a Path.
        args: Script arguments, following the script path.
        executable: Rscript executable name or path.
        r_options: R interpreter options; an empty tuple permits startup files.
        cwd: Child process working directory; ``None`` inherits the current one.
        timeout: Maximum execution time in seconds; ``None`` imposes no limit.
        capture_output: Capture stdout and stderr as text. Set False to stream
            them to the parent; the returned stdout and stderr are then None.
        isolate_process_group: On POSIX, isolate R, compiler workers and chains
            together and terminate the whole group on timeout or cancellation.
            On other platforms only the parent process is terminated.

    Raises:
        FileNotFoundError: The script or executable is unavailable.
        RScriptError: R exits with a nonzero status.
        subprocess.TimeoutExpired: The process exceeds ``timeout``.
    """
    if not script.is_file():
        raise FileNotFoundError(f"R script resource does not exist: {script}")
    with as_file(script) as path:
        command = [os.fspath(executable), *r_options, str(path.resolve())]
        command.extend(os.fspath(arg) for arg in args)
        if isolate_process_group:
            result = _run_process_group(
                command, cwd=cwd, timeout=timeout, capture_output=capture_output
            )
        else:
            result = subprocess.run(
                command,
                cwd=cwd,
                timeout=timeout,
                capture_output=capture_output,
                text=True,
                check=False,
            )
    if result.returncode:
        raise RScriptError(result)
    return result
