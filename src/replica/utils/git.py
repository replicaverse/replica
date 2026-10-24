"""Read git state without raising: a missing binary or repository yields ``None``."""

import subprocess
from pathlib import Path


def git_output(root: Path, *args: str) -> str | None:
    """Stripped stdout of ``git -C root args``, or ``None`` when git fails.

    Args:
        root: Directory the command runs in.
        args: Arguments after ``git -C root``.
    """
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *args], capture_output=True, text=True, check=False
        )
    except OSError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None
