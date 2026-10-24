"""Unified public entry point for the MECO dataset family.

Release-specific implementations live together in this package. Users should
normally construct datasets with :func:`MECO`.
"""

from typing import Literal, TypeAlias

from .character import MECO_char as _MECOCharacter
from .l1 import LANGUAGES
from .l1 import MECO_L1 as _MECOL1
from .releases import L1W2_SITE_LANGUAGES
from .waves import (
    MECOL1W2 as _MECOL1W2,
)
from .waves import (
    MECOL2W1 as _MECOL2W1,
)
from .waves import (
    MECOL2W2 as _MECOL2W2,
)

MECORelease: TypeAlias = Literal["l1_w1", "l1_w2", "l2_w1", "l2_w2"]
MECORepresentation: TypeAlias = Literal["word", "character"]
MECODataset: TypeAlias = _MECOL1 | _MECOCharacter | _MECOL1W2 | _MECOL2W1 | _MECOL2W2

MECO_RELEASES: tuple[MECORelease, ...] = (
    "l1_w1",
    "l1_w2",
    "l2_w1",
    "l2_w2",
)

__all__ = [
    "LANGUAGES",
    "L1W2_SITE_LANGUAGES",
    "MECO",
    "MECODataset",
    "MECORelease",
    "MECORepresentation",
    "MECO_RELEASES",
]


def MECO(
    release: MECORelease = "l1_w1",
    *,
    language: str | None = None,
    site: str | None = None,
    representation: MECORepresentation = "word",
) -> MECODataset:
    """Construct one release/view of the Multilingual Eye-Movement Corpus.

    This is the stable user-facing entry point. The returned object is one of
    the focused release loaders and supports the ordinary ``load()``,
    ``data()``, and ``fixations()`` APIs, and works with the top-level
    ``evaluate()``.

    Args:
        release: MECO reader group and collection wave.
        language: L1 language code for ``"l1_w1"`` (defaults to ``"en"``).
        site: Collection-site code for ``"l1_w2"``. Required because some
            languages have multiple independent Wave 2 samples.
        representation: ``"word"`` for source interest areas with standard
            fixation-derived first-run and separate released measures, or
            ``"character"`` for the L1 Wave 1 character/fixation-derived view.

    Examples:
        ``MECO("l1_w1", language="de")``
        ``MECO("l1_w1", language="en", representation="character")``
        ``MECO("l1_w2", site="en_uk")``
        ``MECO("l2_w2")``
    """
    if release not in MECO_RELEASES:
        raise ValueError(
            f"Unknown MECO release {release!r}; expected one of {MECO_RELEASES}"
        )
    if representation not in ("word", "character"):
        raise ValueError(
            "Unknown MECO representation "
            f"{representation!r}; expected 'word' or 'character'"
        )
    if representation == "character" and release != "l1_w1":
        raise ValueError(
            "MECO's character representation is available only for release='l1_w1'"
        )

    if release == "l1_w1":
        if site is not None:
            raise ValueError("release='l1_w1' uses language=, not site=")
        selected_language = language or "en"
        if representation == "character":
            return _MECOCharacter(language=selected_language)
        return _MECOL1(language=selected_language)

    if release == "l1_w2":
        if language is not None:
            raise ValueError(
                "release='l1_w2' is organized by collection site; pass site= "
                "instead of language="
            )
        if site is None:
            raise ValueError(
                f"release='l1_w2' requires site=; valid sites are {_MECOL1W2.SITES}"
            )
        return _MECOL1W2(site=site)

    if language is not None or site is not None:
        raise ValueError(
            f"release={release!r} has fixed English stimuli and does not accept "
            "language= or site="
        )
    return _MECOL2W1() if release == "l2_w1" else _MECOL2W2()
