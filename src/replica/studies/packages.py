"""Which subpackages of ``replica.studies`` are study packages.

A study is a subpackage that ships a ``README.md``; the shared packages
(``claims``, ``grouped``, ``records``, ``shain``) have none. This module
imports nothing heavy, so readers that only need the study names stay light.
"""

from pathlib import Path

STUDIES_DIR = Path(__file__).resolve().parent


def study_ids() -> tuple[str, ...]:
    """Study package names in alphabetical order."""
    return tuple(
        sorted(
            path.name
            for path in STUDIES_DIR.iterdir()
            if (path / "__init__.py").is_file() and (path / "README.md").is_file()
        )
    )
