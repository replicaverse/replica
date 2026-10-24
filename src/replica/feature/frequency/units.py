"""Token-defined unit boundaries and exact occurrence counts."""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np

from .sampled import _word_boundary_marker


@dataclass(frozen=True, slots=True)
class BoundarySet:
    """The two token sets a unit boundary is read against.

    ``opens`` are the tokens that start a new unit: they decide frame
    membership and are what :math:`W_b` sums. ``ends`` adds the end symbol,
    since a unit also closes when nothing follows it.

    A token opens a unit when it carries the tokenizer's word-initial marker
    *or* decodes to leading whitespace. The second clause is what catches the
    newline and tab tokens, which carry no marker; without it the frame misses
    every unit that follows a line break and stops agreeing with ``str.split``.
    """

    opens: np.ndarray
    ends: np.ndarray
    open_ids: frozenset[int]

    def __post_init__(self) -> None:
        if self.opens.size == 0:
            raise ValueError("no unit-initial token ids; check the marker")
        if self.ends.size <= self.opens.size:
            raise ValueError(
                "the end symbol already opens a unit, which would put every "
                "terminal prefix in the frame conditionally"
            )

    def opens_unit(self, ids: Sequence[int], p: int) -> bool:
        """True when a unit opens just after position *p* of *ids*.

        The first and last positions always do: a unit starts at the first
        token, and termination closes whatever is open at the end.

        Args:
            ids: Token ids of one sample, in order.
            p: Position in *ids* the prefix ends at.
        """
        return p == 0 or p == len(ids) - 1 or int(ids[p + 1]) in self.open_ids


def boundary_set(tokenizer) -> BoundarySet:
    """Current unit-opening ids and those plus the tokenizer's end symbol.

    Args:
        tokenizer: Tokenizer whose vocabulary and decoding rules are scanned.
    """
    marker = _word_boundary_marker(tokenizer)
    size = len(tokenizer)
    ids = list(range(size))
    pieces = tokenizer.convert_ids_to_tokens(ids)
    decoded = _decode_singletons(tokenizer, ids)
    special = set(getattr(tokenizer, "all_special_ids", None) or ())
    eos = tokenizer.eos_token_id
    opens = np.array(
        [
            i
            for i, (piece, text) in enumerate(zip(pieces, decoded, strict=True))
            if i not in special
            and (
                (isinstance(piece, str) and piece.startswith(marker))
                or (text and text[0].isspace())
            )
        ],
        dtype=np.int64,
    )
    ends = opens if eos is None else np.union1d(opens, np.array([eos], dtype=np.int64))
    return BoundarySet(
        opens=opens, ends=ends, open_ids=frozenset(int(i) for i in opens)
    )


def _decode_singletons(tokenizer, ids: Sequence[int]) -> list[str]:
    """Each id decoded on its own, so leading whitespace survives."""
    kwargs = {"skip_special_tokens": False, "clean_up_tokenization_spaces": False}
    try:
        return list(tokenizer.batch_decode([[i] for i in ids], **kwargs))
    except (AttributeError, TypeError):
        return [tokenizer.decode([i]) for i in ids]


def unit_counts(words: Sequence[str], samples: Iterable[str]) -> dict[str, int]:
    """Exact pooled occurrence count of each word, split as units are split.

    Model-free, so a shard of the pool can be counted on a cpu and the tallies
    added. The surface carries its separator, which whitespace splitting has
    already consumed, so the key is the surface with the separator dropped.

    A few corpus surfaces span a space (``' 19. Jahrhunderts'``). The estimator
    scores those as one longer token sequence closed by the same boundary mass,
    so the tally has to agree. They are matched on the literal surface rather
    than on split units, because the separator is part of the sequence: units
    held apart by a newline tokenize differently and are not an occurrence.

    Args:
        words: Surfaces to count. A surface with no whitespace left after
            splitting is an error.
        samples: The pool's sentences, walked once.

    Returns:
        Occurrences over the whole pool, keyed by the input surfaces.
    """
    from collections import Counter

    keys = {}
    for word in words:
        parts = word.split()
        if not parts:
            raise ValueError(f"{word!r} has no whitespace-delimited unit")
        keys[word] = parts
    single = {w: p[0] for w, p in keys.items() if len(p) == 1}
    multi = [w for w, p in keys.items() if len(p) > 1]

    tally: Counter[str] = Counter()
    spans: Counter[str] = Counter()
    for sample in samples:
        tally.update(sample.split())
        # A leading separator lets a sample-initial occurrence match too; the
        # bos boundary carries those on the estimator's side.
        padded = " " + sample if multi else ""
        for word in multi:
            probe = word if word[:1].isspace() else " " + word
            start = 0
            while (hit := padded.find(probe, start)) != -1:
                end = hit + len(probe)
                if end == len(padded) or padded[end].isspace():
                    spans[word] += 1
                start = hit + 1
    out = {word: int(tally[key]) for word, key in single.items()}
    out.update({word: int(spans[word]) for word in multi})
    return {word: out[word] for word in keys}


def _special_ids(tokenizer) -> frozenset[int]:
    """Ids that carry no text. ``all_special_ids`` alone misses the stubs."""
    out = {int(i) for i in (getattr(tokenizer, "all_special_ids", ()) or ())}
    for name in ("bos_token_id", "eos_token_id", "pad_token_id"):
        got = getattr(tokenizer, name, None)
        if got is not None:
            out.add(int(got))
    return frozenset(out)


#: Whitespace as ``str.split`` reads it, so a span's leading separator is
#: dropped the same way whether it arrived as a space or as U+00A0.
_WS_BYTES: Final[tuple[bytes, ...]] = tuple(
    sorted(
        {chr(c).encode("utf-8") for c in range(0x3001) if chr(c).isspace()},
        key=len,
        reverse=True,
    )
)


def _lstrip_ws(raw: bytes) -> bytes:
    """Leading whitespace dropped, including the non-ASCII kinds."""
    out = raw.lstrip()
    while out:
        for ws in _WS_BYTES:
            if out.startswith(ws):
                out = out[len(ws) :].lstrip()
                break
        else:
            break
    return out


def priced_tokens(
    tokenizer,
    word: str,
    *,
    bset: "BoundarySet | None" = None,
    columns=None,
) -> tuple[int, ...]:
    """The tokenization the sampled channel scores for *word*.

    A separator-less surface names the same estimand as its spaced form, and its
    bare first token cannot open a unit at an interior boundary, so the channel
    runs through the spaced tokenization. Anything that classifies occurrences as
    canonical or residual has to make the same choice, or the two disagree about
    which span is canonical and the residual count double-counts.

    ``columns`` restricts the fallback to first tokens the sweep holds a column
    for, matching what
    :func:`~replica.feature.frequency.pooled.pps_estimate_batch` can actually score.

    Args:
        tokenizer: Tokenizer the pool was encoded with.
        word: The surface, with or without its leading separator.
        bset: The tokenizer's unit-opening and unit-closing token sets;
            ``None`` derives them.
        columns: First tokens the sweep holds a column for; ``None`` accepts
            any unit-opening spaced first token.

    Returns:
        The token ids the sampled channel scores for the surface.
    """
    bset = boundary_set(tokenizer) if bset is None else bset
    toks = tuple(tokenizer.encode(word, add_special_tokens=False))
    if toks and int(toks[0]) in bset.open_ids:
        return toks
    spaced = tuple(tokenizer.encode(f" {word}", add_special_tokens=False))
    if spaced and int(spaced[0]) in bset.open_ids:
        if columns is None or int(spaced[0]) in columns:
            return spaced
    return toks


def _byte_decoder(tokenizer) -> "dict[str, int] | None":
    """Byte-level vocabulary's char-to-byte map, or ``None`` if it has none."""
    table = getattr(tokenizer, "byte_decoder", None)
    if table:
        return dict(table)
    try:
        from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode
    except ImportError:
        return None
    return {char: byte for byte, char in bytes_to_unicode().items()}


def _piece_bytes(tokenizer) -> list[bytes]:
    """Per-id raw bytes, so a span's surface is a join over it.

    A byte-level vocabulary does not survive swapping the marker for a space:
    every non-ASCII piece comes back as mojibake, which silently drops the
    occurrences in exactly the languages the pool is richest in.
    """
    marker = _word_boundary_marker(tokenizer)
    ids = list(range(len(tokenizer)))
    pieces = tokenizer.convert_ids_to_tokens(ids)
    specials = _special_ids(tokenizer)
    table = _byte_decoder(tokenizer)
    out: list[bytes] = []
    for i, piece in zip(ids, pieces, strict=True):
        if i in specials:
            out.append(b"")
            continue
        text = str(piece)
        if table is not None:
            try:
                out.append(bytes(table[c] for c in text))
                continue
            except KeyError:
                pass
        out.append(text.replace(marker, " ").encode("utf-8"))
    return out


def _unit_starts(row, bset, specials) -> list[int]:
    """Positions where a unit opens.

    A marker-carrying token opens one, and so does the sample's first token
    whether or not it carries a marker, which is what ``str.split`` sees.
    """
    first = next((p for p in range(len(row)) if row[p] not in specials), len(row))
    starts = [p for p in range(len(row)) if row[p] in bset.open_ids and p >= first]
    if first < len(row) and (not starts or starts[0] != first):
        starts.insert(0, first)
    return starts


def unit_total(
    encoded: Iterable[Sequence[int]],
    tokenizer,
    *,
    bset: "BoundarySet | None" = None,
) -> int:
    """Units over the pool, segmented exactly as :func:`unit_counts_split` does.

    The normalizer has to count the same corpus the numerator counts: the
    decoder drops the space before punctuation, so a tally taken on the decoded
    text splits a few units differently from the ids the estimator reads.

    Args:
        encoded: The pool's sampled token ids, one row per sample.
        tokenizer: Tokenizer the pool was encoded with.
        bset: The tokenizer's unit-opening and unit-closing token sets;
            ``None`` derives them.

    Returns:
        The number of units over the pool.
    """
    bset = boundary_set(tokenizer) if bset is None else bset
    specials = _special_ids(tokenizer)
    total = 0
    for ids in encoded:
        row = [int(t) for t in ids if int(t) not in specials]
        total += len(_unit_starts(row, bset, specials))
    return total


def unit_counts_split(
    words: Sequence[str],
    encoded: Iterable[Sequence[int]],
    tokenizer,
    *,
    bset: "BoundarySet | None" = None,
    pieces: "Sequence[bytes] | None" = None,
    columns=None,
) -> dict[str, tuple[int, int]]:
    """Each word's exact count, partitioned into canonical and residual parts.

    Takes the pool's *sampled* token ids. Re-encoding the text would return the
    canonical tokenization of every unit and report no residual mass at all.

    The partition is over the units :func:`unit_counts` finds, so the two parts
    sum to that count and the residual part cannot come out negative.

    Surfaces are matched as bytes, which keeps a byte-level vocabulary exact
    and needs no decode per span.

    Args:
        words: Surfaces to count, keyed as :func:`unit_counts` keys them.
        encoded: The pool's sampled token ids, one row per sample.
        tokenizer: Tokenizer the pool was encoded with.
        bset: The tokenizer's unit-opening and unit-closing token sets;
            ``None`` derives them.
        pieces: Raw bytes per token id; ``None`` derives them from the
            tokenizer's vocabulary.
        columns: First tokens the sweep holds a column for, so the split calls
            canonical the spelling the sampled channel scores; ``None`` accepts
            any unit-opening spaced first token.

    Returns:
        Per word, its ``(canonical, residual)`` occurrence counts.
    """
    bset = boundary_set(tokenizer) if bset is None else bset
    piece = _piece_bytes(tokenizer) if pieces is None else list(pieces)
    # The same choice the sampled channel makes, or the two disagree on which
    # span is canonical and every occurrence of a separator-less surface would be
    # charged twice: once in the canonical expectation, once as residual.
    canon = {
        w: priced_tokens(tokenizer, w, bset=bset, columns=columns)
        for w in dict.fromkeys(words)
    }
    # Keyed like unit_counts: whitespace splitting has consumed the leading
    # separator, so it is dropped here too, while internal ones stay exact.
    want: dict[bytes, list[str]] = {}
    for w in canon:
        want.setdefault(w.lstrip().encode("utf-8"), []).append(w)
    widths = sorted({len(text.split()) for text in canon})
    out = {w: [0, 0] for w in canon}
    specials = _special_ids(tokenizer)

    for ids in encoded:
        # A stored special id would ride along in the last span and turn a
        # canonical occurrence residual, so it is dropped before segmenting.
        row = [int(t) for t in ids if int(t) not in specials]
        starts = _unit_starts(row, bset, specials)
        if not starts:
            continue
        edges = [*starts, len(row)]
        for i in range(len(starts)):
            for u in widths:
                if i + u >= len(edges):
                    continue
                span = row[starts[i] : edges[i + u]]
                text = _lstrip_ws(b"".join(piece[t] for t in span))
                for w in want.get(text, ()):
                    out[w][tuple(span) != canon[w]] += 1
    return {w: (v[0], v[1]) for w, v in out.items()}
