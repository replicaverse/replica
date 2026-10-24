"""Regenerate ModelBlocks-output fixtures from a checked-out ModelBlocks.

Run this when:
* upstream ModelBlocks changes the ``.itemmeasures`` format, or
* you add a new ``.sentitems`` input here and need its companion output.

Usage::

    python tests/feature/fixtures/modelblocks/regenerate.py [<modelblocks_dir>]

If ``<modelblocks_dir>`` is omitted, falls back to ``$MODELBLOCKS_DIR``,
then to the modelblocks/modelblocks-release checkout the registry resolves.

The script writes each ``<base>.sentitems`` to ``<base>.itemmeasures``
in this directory by piping it through
``resource-rt/scripts/lineitems2itemmeasures.py``. The corpus name
(passed as argv[1] to that script) is taken from ``<base>``.

This script does not run automatically as part of pytest, the fixtures
are committed to the repo so the test suite is fully self-contained.
"""

import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]


def _find_modelblocks(arg: str | None) -> Path:
    if arg:
        return Path(arg).resolve()
    env = os.environ.get("MODELBLOCKS_DIR")
    if env:
        return Path(env).resolve()
    from replica.external import repo_path

    default = repo_path("modelblocks-release", fetch=False)
    if not default.exists():
        raise SystemExit(
            f"ModelBlocks checkout not found at {default}. "
            "Pass its path or set $MODELBLOCKS_DIR."
        )
    return default


def main(argv: list[str]) -> int:
    mb_dir = _find_modelblocks(argv[1] if len(argv) > 1 else None)
    script = mb_dir / "resource-rt" / "scripts" / "lineitems2itemmeasures.py"
    if not script.exists():
        raise SystemExit(f"missing {script}")

    sentitems = sorted(HERE.glob("*.sentitems"))
    if not sentitems:
        raise SystemExit(f"no .sentitems inputs in {HERE}")

    for src in sentitems:
        corpus = src.stem
        out = src.with_suffix(".itemmeasures")
        with src.open() as stdin, out.open("w") as stdout:
            subprocess.run(
                [sys.executable, str(script), corpus],
                stdin=stdin,
                stdout=stdout,
                check=True,
            )
        print(f"regenerated {out.relative_to(REPO_ROOT)} ({out.stat().st_size}B)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
