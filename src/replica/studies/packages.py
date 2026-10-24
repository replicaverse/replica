"""Discover study packages by their ``README.md`` without importing them."""

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
