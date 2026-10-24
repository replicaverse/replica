"""Input paths for the context-reading-time tables used by Hopton and Opedal.

Source: Opedal et al. (2024), "On the Role of Context in Reading Time Prediction"
(https://aclanthology.org/2024.emnlp-main.179/).
The upstream tables are at https://github.com/rycolab/context-reading-time.
"""

from pathlib import Path

from ..dependency import resolve_data_root
from ..repositories import repo_path


def data_dir(data_root: Path | str | None = None) -> Path:
    """Locate the released per-unit CSVs, preferring the pinned checkout.

    Args:
        data_root: Dependency root; ``None`` uses the standard data-root lookup.
    """
    cloned = (
        repo_path("context-reading-time", data_root, fetch=False)
        / "merged_data_no_zero"
    )
    if cloned.exists():
        return cloned
    root = resolve_data_root(data_root)
    return root / "datasets" / "eye-tracking" / "processed" / "meco_opedal2024context"
