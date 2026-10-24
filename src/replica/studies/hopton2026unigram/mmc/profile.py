"""The fixed MECO language and allocation profile used by the Hopton study."""

import functools
import hashlib
import json
import math
from collections.abc import Mapping
from typing import Final

import numpy as np

from replica.feature.frequency.allocation import AllocationSpec
from replica.feature.frequency.strata import Stratification

#: Level-1 classes: dominant Unicode script of the prefix window. Latin is
#: absent because it is refined into :data:`LATIN_CLASSES`.
SCRIPT_CLASSES: Final[tuple[str, ...]] = (
    "hebrew",
    "greek",
    "arabic",
    "cyrillic",
    "georgian",
    "indic",
    "hangul",
    "thai",
    "cjk",
    "armenian",
    "none",
)
#: Level-2 classes: langdetect over the Latin run containing the prefix end.
LATIN_CLASSES: Final[tuple[str, ...]] = (
    "lat_en",
    "lat_de",
    "lat_nl",
    "lat_it",
    "lat_es",
    "lat_fi",
    "lat_et",
    "lat_no",
    "lat_tr",
    "lat_other",
)
STRATA: Final[tuple[str, ...]] = SCRIPT_CLASSES + LATIN_CLASSES
N_STRATA: Final[int] = len(STRATA)
STRATUM_INDEX: Final[dict[str, int]] = {s: i for i, s in enumerate(STRATA)}
LATIN_BACKSTOP: Final[str] = "lat_other"

#: MECO dataset name -> the stratum its surfaces are drawn from.
MECO_HOME_STRATUM: Final[dict[str, str]] = {
    "meco_de": "lat_de",
    "meco_du": "lat_nl",
    "meco_ee": "lat_et",
    "meco_en": "lat_en",
    "meco_fi": "lat_fi",
    "meco_gr": "greek",
    "meco_he": "hebrew",
    "meco_it": "lat_it",
    "meco_ko": "hangul",
    "meco_no": "lat_no",
    "meco_ru": "cyrillic",
    "meco_sp": "lat_es",
    "meco_tr": "lat_tr",
}
#: MECO release codes are not ISO 639-1: ``du`` is Dutch, ``de`` is German.
MECO_ISO639_1: Final[dict[str, str]] = {
    "de": "de",
    "du": "nl",
    "ee": "et",
    "en": "en",
    "fi": "fi",
    "gr": "el",
    "he": "he",
    "it": "it",
    "ko": "ko",
    "no": "no",
    "ru": "ru",
    "sp": "es",
    "tr": "tr",
}
#: Characters of the decoded prefix that decide its stratum.
PREFIX_WINDOW_CHARS: Final[int] = 40

# --- strata ----------------------------------------------------------------

_SCRIPT_RANGES: Final[tuple[tuple[str, int, int], ...]] = (
    ("hebrew", 0x0590, 0x05FF),
    ("hebrew", 0xFB1D, 0xFB4F),
    ("greek", 0x0370, 0x03FF),
    ("greek", 0x1F00, 0x1FFF),
    ("arabic", 0x0600, 0x06FF),
    ("arabic", 0x0750, 0x077F),
    ("arabic", 0x08A0, 0x08FF),
    ("arabic", 0xFB50, 0xFDFF),
    ("arabic", 0xFE70, 0xFEFF),
    ("cyrillic", 0x0400, 0x052F),
    ("cyrillic", 0x2DE0, 0x2DFF),
    ("cyrillic", 0xA640, 0xA69F),
    ("georgian", 0x10A0, 0x10FF),
    ("georgian", 0x1C90, 0x1CBF),
    ("georgian", 0x2D00, 0x2D2F),
    ("indic", 0x0900, 0x0DFF),
    ("hangul", 0x1100, 0x11FF),
    ("hangul", 0x3130, 0x318F),
    ("hangul", 0xA960, 0xA97F),
    ("hangul", 0xAC00, 0xD7FF),
    ("thai", 0x0E00, 0x0E7F),
    ("cjk", 0x2E80, 0x2FDF),
    ("cjk", 0x3040, 0x30FF),
    ("cjk", 0x3400, 0x4DBF),
    ("cjk", 0x4E00, 0x9FFF),
    ("cjk", 0xF900, 0xFAFF),
    ("cjk", 0x20000, 0x2FA1F),
    ("armenian", 0x0530, 0x058F),
    ("armenian", 0xFB13, 0xFB17),
    ("latin", 0x0041, 0x005A),
    ("latin", 0x0061, 0x007A),
    ("latin", 0x00C0, 0x024F),
    ("latin", 0x1E00, 0x1EFF),
    ("latin", 0x2C60, 0x2C7F),
    ("latin", 0xA720, 0xA7FF),
)
_SCRIPT_ORDER: Final[tuple[str, ...]] = (*SCRIPT_CLASSES[:-1], "latin")
#: langdetect codes that map onto a Latin sub-stratum.
_LANG_TO_LATIN: Final[dict[str, str]] = {
    "en": "lat_en",
    "de": "lat_de",
    "nl": "lat_nl",
    "it": "lat_it",
    "es": "lat_es",
    "fi": "lat_fi",
    "et": "lat_et",
    "no": "lat_no",
    "tr": "lat_tr",
}
#: Characters that continue a Latin run without being Latin letters.
_LATIN_EXTRA: Final[frozenset[str]] = frozenset(" '-.,")


def _char_script(ch: str) -> str | None:
    """Script name of one character, or None when it carries no script."""
    if not ch.isalpha():
        return None
    code = ord(ch)
    for name, low, high in _SCRIPT_RANGES:
        if low <= code <= high:
            return name
    return None


def _dominant_script(text: str) -> str:
    """The most frequent script among *text*'s letters, ties by class order."""
    counts: dict[str, int] = {}
    for ch in text:
        name = _char_script(ch)
        if name is not None:
            counts[name] = counts.get(name, 0) + 1
    if not counts:
        return "none"
    best = max(counts.values())
    for name in _SCRIPT_ORDER:
        if counts.get(name, 0) == best:
            return name
    return "none"


def _latin_run(text: str) -> str:
    """The maximal Latin-script run containing the last Latin letter."""
    last = -1
    for i in range(len(text) - 1, -1, -1):
        if _char_script(text[i]) == "latin":
            last = i
            break
    if last < 0:
        return ""
    joins = [
        _char_script(ch) == "latin" or ch in _LATIN_EXTRA or ch.isdigit() for ch in text
    ]
    start, end = last, last + 1
    while start > 0 and joins[start - 1]:
        start -= 1
    while end < len(text) and joins[end]:
        end += 1
    return text[start:end]


@functools.cache
def _detector():
    """langdetect's ``detect`` with its factory pinned to a fixed seed."""
    try:
        from langdetect import DetectorFactory, LangDetectException, detect
    except ImportError as error:  # pragma: no cover - environment guard
        raise ImportError(
            "the Latin sub-strata need langdetect; install replicaverse[mmc]"
        ) from error
    DetectorFactory.seed = 0
    return detect, LangDetectException


def _detect_language(text: str) -> str:
    """langdetect's code for *text*, or an empty string when it declines."""
    detect, failure = _detector()
    try:
        return str(detect(text))
    except failure:
        return ""


def stratum_label(prefix_text: str) -> str:
    """Stratum of a boundary whose decoded prefix window is *prefix_text*.

    Args:
        prefix_text: Text ending at the boundary. Only its last
            :data:`PREFIX_WINDOW_CHARS` characters decide the label.
    """
    window = prefix_text[-PREFIX_WINDOW_CHARS:]
    script = _dominant_script(window)
    if script != "latin":
        return script
    run = _latin_run(window)
    if not run.strip():
        return LATIN_BACKSTOP
    return _LANG_TO_LATIN.get(_detect_language(run), LATIN_BACKSTOP)


STRATIFICATION = Stratification(
    name="hopton-meco/1",
    labels=STRATA,
    window_chars=PREFIX_WINDOW_CHARS,
    classify=stratum_label,
)
LATIN_BACKSTOP_WEIGHT = 0.5


def allocation_spec(
    *,
    homes: Mapping[str, tuple[str, ...]],
    k_eval: int = 4096,
    k_floor: int = 64,
    k_home: int = 2048,
    replicates: int = 8,
    remainder_cap_factor: float = 1.0,
) -> AllocationSpec:
    """Translate MECO dataset membership into the study's stratum weights.

    Args:
        homes: Surface to MECO dataset names; unknown datasets are refused.
        k_eval: Total draws per word.
        k_floor: Initial draws per eligible stratum.
        k_home: Draws shared by the word's home strata.
        replicates: Independent frame partitions.
        remainder_cap_factor: Cap on the extra draws relative to the floor.
    """
    weights = {}
    for word, datasets in homes.items():
        unknown = set(datasets) - MECO_HOME_STRATUM.keys()
        if unknown:
            raise ValueError(f"unknown MECO home datasets: {sorted(unknown)}")
        row = {MECO_HOME_STRATUM[d]: 1.0 for d in datasets}
        if row.keys() & set(LATIN_CLASSES):
            row.setdefault(LATIN_BACKSTOP, LATIN_BACKSTOP_WEIGHT)
        weights[word] = row
    return AllocationSpec(
        strata=STRATA,
        k_eval=k_eval,
        k_floor=k_floor,
        k_home=k_home,
        replicates=replicates,
        remainder_cap_factor=remainder_cap_factor,
        home_weights=weights,
    )


def legacy_allocation_sha(allocation: AllocationSpec, homes) -> str:
    """Authenticate plans written before explicit stratum weights were stored.

    Args:
        allocation: Rebuilt study allocation.
        homes: Original surface to dataset membership, including empty entries.
    """
    payload = {
        "k_eval": allocation.k_eval,
        "k_floor": allocation.k_floor,
        "k_home": allocation.k_home,
        "latin_backstop_weight": LATIN_BACKSTOP_WEIGHT,
        "replicates": allocation.replicates,
        "remainder_cap_factor": allocation.remainder_cap_factor,
        "homes": {w: list(d) for w, d in sorted(homes.items())},
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def max_cell_budget(
    allocation: AllocationSpec, first_id_homes: Mapping[int, frozenset[int]]
) -> np.ndarray:
    """The study's nominal buffer allowance, not a worst-case draw bound.

    Rows follow the key order of *first_id_homes*. Estimation reallocates
    draws to respect the stored capacities.

    Args:
        allocation: The study's explicit budget and stratum weights.
        first_id_homes: First-token id -> the stratum indices that are home
            to some word starting with it.

    Returns:
        Shape ``(len(first_id_homes), N_STRATA)``, each entry capped at
        ``k_eval``.
    """
    rows: np.ndarray = np.zeros((len(first_id_homes), N_STRATA), dtype=np.int64)
    backstop = STRATUM_INDEX[LATIN_BACKSTOP]
    latin_idx = {STRATUM_INDEX[name] for name in LATIN_CLASSES}
    share = 1.0 / (1.0 + LATIN_BACKSTOP_WEIGHT)
    for row, homes in enumerate(first_id_homes.values()):
        for stratum in range(N_STRATA):
            home = 0.0
            if stratum in homes:
                home = allocation.k_home * (share if stratum in latin_idx else 1.0)
            elif stratum == backstop and homes & latin_idx:
                home = allocation.k_home * LATIN_BACKSTOP_WEIGHT * share
            nominal = (1.0 + allocation.remainder_cap_factor) * (
                allocation.k_floor + home
            )
            rows[row, stratum] = min(int(math.ceil(nominal)), allocation.k_eval)
    return rows
