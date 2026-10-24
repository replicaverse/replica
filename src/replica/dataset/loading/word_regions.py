"""Word boundaries for releases that measure multi-word interest areas."""

from collections.abc import Hashable, Iterable
from typing import TypeVar

SourceID = TypeVar("SourceID", bound=Hashable)


def preserve_interest_areas(
    regions: Iterable[tuple[SourceID, str]],
) -> tuple[list[str], list[SourceID], dict[SourceID, int]]:
    """Keep one response unit per measured area, including multiword areas.

    Internal whitespace becomes a natural space in the stimulus. The whole
    area occupies one character span, so predictors can cover every constituent
    token without splitting or duplicating the area's observed duration.
    Empty labels and repeated IDs are ambiguous and raise ``ValueError``.

    Args:
        regions: Ordered source area IDs and displayed text labels.

    Returns:
        Area labels, source IDs, and their one-to-one positional mapping.
    """
    labels: list[str] = []
    source_ids: list[SourceID] = []
    mapping: dict[SourceID, int] = {}
    for source_id, label in regions:
        label = " ".join(label.split())
        if not label or source_id in mapping:
            raise ValueError(
                f"Empty label or duplicate interest-area ID: {source_id!r}"
            )
        mapping[source_id] = len(labels)
        labels.append(label)
        source_ids.append(source_id)
    return labels, source_ids, mapping


def normalize_c1_controls(label: str) -> str:
    """Repair CP1252 punctuation decoded as Latin-1 in EyeLink labels.

    Args:
        label: Interest-area label as the release spells it.
    """
    return label.translate(
        {
            value: bytes([value]).decode("cp1252")
            for value in range(0x80, 0xA0)
            if value not in {0x81, 0x8D, 0x8F, 0x90, 0x9D}
        }
    )
