r"""Pooled marginal Monte Carlo unigram estimation with priority sampling.

The marginal Monte Carlo estimator of :mod:`.sampled` sums a word's
autoregressive joint over every word-boundary position of every sample. That
cost is linear in the pool size, so a pool large enough to cover rare languages
cannot be swept exhaustively per word.

This module keeps the estimand and replaces the exhaustive inner sum by a
design-based one:

* one shared sweep over the pool records, per first token, the exact float64
  column mass per stratum and a per-cell top-``K`` priority buffer,
* a word's continuation passes run on a bounded set of boundaries drawn by
  priority sampling (Duffield-Lund-Thorup), weighted ``max(size, tau)``.

A boundary's first-token probability is divided by its unit-open mass
:math:`W_b`, since a prefix joins the frame only when the token that followed
it opened a unit, and multiplied by the closing mass :math:`\beta_b`, so the
estimand is the probability that the next unit *is* the word rather than
merely starts with it. Termination is forced at the end of a sample, which
leaves the terminal prefix contributing nothing.

A caller-supplied classifier partitions the boundary frame using each
boundary's decoded prefix window. Labels drive allocation and diagnostics;
with every positive-mass stratum funded, a different deterministic partition
changes sampling efficiency without changing the estimand.

:func:`exhaustive_estimate_batch` is the oracle: it walks the samples itself
and uses independent estimator arithmetic, so a budget that covers the frame
has to reproduce it.

Two numerators share the sweep and the selection. The pathwise one
(:data:`ESTIMATOR_VERSION`, source ``mmc_pool``) pairs the canonical sum
against the take-store estimate of itself and adds the exact occurrence count,
so it is exact per string before marginalization but can refuse a word whose
draw noise outweighs its count. The expectation-equivalent one
(:data:`EE_ESTIMATOR_VERSION`, source ``mmc_pool_ee``) prices the canonical
sum against the sampler's horizon and adds the exact residual count, so every
term is nonnegative and no draw can leave a unit without an estimate. Each
publishes under its own tag family, ``mmc_pps`` and ``mmc_pps_ee``, through
``method_cache_tag`` with ``n_samples``, ``k_eval`` and ``seed``; reproducing a
cell takes the ``code_git_sha`` and pool fingerprint in its ``table.meta.json``
plus the published pool.
"""

import hashlib
import math
import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Final

import numpy as np

from ..backends.start_state import resolve_start_state
from .allocation import AllocationSpec
from .estimated import EstimatedUnigramSource
from .sampled import (
    _encode_samples,
    _left_pad_batch,
    _padded_logits,
    start_prefix,
)
from .strata import Stratification, _validate_labels
from .units import (
    BoundarySet,  # noqa: F401 - public pooled API
    boundary_set,
    priced_tokens,
    unit_counts,
    unit_counts_split,
    unit_total,  # noqa: F401 - public pooled API
)

#: The pathwise numerator (``mmc_pool``): the canonical sum paired against the
#: take-store estimate of itself, plus the exact occurrence count.
ESTIMATOR_VERSION: Final[str] = "mmc_pps/3"
#: The expectation-equivalent numerator (``mmc_pool_ee``): the canonical sum
#: priced against the sampler's horizon, plus the exact residual count. It
#: shares the ``mmc_pps`` sweep and selection, hence the shared stem; the tag
#: family ``mmc_pps_ee`` keeps its cells apart.
EE_ESTIMATOR_VERSION: Final[str] = "mmc_pps/4"
#: The sweep and selection layout this module validates. The estimate stage
#: reads any selection swept under this layout; :data:`ESTIMATOR_VERSION`
#: names only the estimate-time arithmetic.
SELECTION_VERSION: Final[str] = "mmc_pps_selection/1"

#: Smallest positive float64. A word whose numerator underflows reports this as
#: ``p_hat`` and keeps its accurate value in ``bits``.
TINY: Final[float] = float(np.nextafter(0.0, 1.0))
#: The float32 counterpart. Clamping a float32 tensor with ``TINY`` is a no-op,
#: since it rounds to zero there, so a division by it still yields infinity.
_F32_TINY: Final[float] = float(np.finfo(np.float32).tiny)

_UINT64 = np.uint64
_GOLDEN = _UINT64(0x9E3779B97F4A7C15)
_MIX1 = _UINT64(0xBF58476D1CE4E5B9)
_MIX2 = _UINT64(0x94D049BB133111EB)
_POSITION_ODD = _UINT64(0xD1B54A32D192ED03)
_MASK64 = (1 << 64) - 1
_TWO_POW_53 = float(1 << 53)


# --- counter-based uniforms ------------------------------------------------


def splitmix64(x: "np.ndarray | int") -> "np.ndarray | np.uint64":
    """The splitmix64 finalizer: a uint64 bijection with no carried state.

    Args:
        x: Counter value(s) to mix, read as uint64. A scalar returns a scalar,
            an array returns an array of the same shape.
    """
    z = np.array(x, dtype=np.uint64, copy=True)
    scalar = z.ndim == 0
    z = np.atleast_1d(z)
    z ^= z >> _UINT64(30)
    z *= _MIX1
    z ^= z >> _UINT64(27)
    z *= _MIX2
    z ^= z >> _UINT64(31)
    return z[0] if scalar else z


def boundary_uniform(
    seed: int, sample_index: "np.ndarray | int", position: "np.ndarray | int"
) -> "np.ndarray | float":
    """Uniform draw in ``(0, 1]`` keyed on ``(seed, sample_index, position)``.

    Stateless and order-free: a boundary gets the same draw whichever shard,
    batch or iteration order reaches it, which is what lets per-shard top-``K``
    buffers merge into the pool's top-``K`` exactly.

    Args:
        seed: Run-level seed, so two runs over one pool draw different frames.
        sample_index: Pool-global index of the sample holding the boundary.
        position: Token position of the boundary within that sample.

    Returns:
        A float when both keys are scalars, else an array of draws.
    """
    j = np.atleast_1d(np.array(sample_index, dtype=np.uint64, copy=True))
    p = np.atleast_1d(np.array(position, dtype=np.uint64, copy=True))
    scalar = np.ndim(sample_index) == 0 and np.ndim(position) == 0
    base = np.uint64((int(seed) * int(_GOLDEN)) & _MASK64)
    key = np.asarray(splitmix64(base + j), dtype=np.uint64)
    key = key + p * _POSITION_ODD
    h = np.asarray(splitmix64(key), dtype=np.uint64)
    u = (h >> _UINT64(11)).astype(np.float64) + 1.0
    u /= _TWO_POW_53
    return float(u[0]) if scalar else u


# --- boundary index --------------------------------------------------------


def _read_stratification(blob, expected: Stratification | None) -> dict[str, Any]:
    """Read partition metadata; legacy archives require an explicit policy."""
    keys = {"strata", "stratification_id"} & set(blob.files)
    if not keys:
        if expected is None:
            raise ValueError("legacy archive needs an explicit stratification")
        return {"strata": expected.labels, "stratification_id": expected.identity}
    if len(keys) != 2:
        raise ValueError("incomplete stratification metadata")
    labels = tuple(str(label) for label in blob["strata"])
    identity = str(blob["stratification_id"])
    if expected is not None and (labels, identity) != (
        expected.labels,
        expected.identity,
    ):
        raise ValueError("archive stratification does not match the requested policy")
    return {"strata": labels, "stratification_id": identity}


def _validate_stratification(artifact) -> None:
    object.__setattr__(artifact, "strata", tuple(artifact.strata))
    _validate_labels(artifact.strata)
    if (
        not isinstance(artifact.stratification_id, str)
        or not artifact.stratification_id
    ):
        raise ValueError("stratification_id must be nonempty")


def _same_stratification(parts) -> None:
    if not parts:
        raise ValueError("at least one partitioned artifact is required")
    first = parts[0]
    if any(
        (p.strata, p.stratification_id) != (first.strata, first.stratification_id)
        for p in parts[1:]
    ):
        raise ValueError("parts disagree on the stratification")


def _validate_capacity(capacity, n_strata, n_columns) -> None:
    cap = np.asarray(capacity)
    if cap.ndim != 3 or cap.shape[:2] != (n_columns, n_strata) or cap.shape[2] < 1:
        raise ValueError("capacity shape disagrees with the columns or strata")
    if not np.issubdtype(cap.dtype, np.integer) or (cap < 0).any():
        raise ValueError("capacity must contain non-negative integers")


def _validate_stratum_arrays(artifact, capacity) -> None:
    n = len(artifact.strata)
    if artifact.colsum.ndim != 2 or artifact.colsum.shape[0] != n:
        raise ValueError("column mass shape disagrees with the strata")
    if artifact.stratum_boundary_counts.shape != (n,):
        raise ValueError("boundary counts shape disagrees with the strata")
    if np.ndim(capacity) != 3:
        raise ValueError("capacity shape must be (columns, strata, replicates)")
    _validate_capacity(capacity, n, capacity.shape[0])


@dataclass(frozen=True, slots=True)
class BoundaryIndex:
    """Where the boundaries are, which stratum each is in, and how many.

    ``n_boundaries`` counts every boundary of every sample, the BOS ones
    included, because it is the term the denominator needs. The frame arrays
    describe only the boundaries the sampler may draw.
    """

    strata: tuple[str, ...]
    stratification_id: str
    n_samples: int
    boundary_counts: np.ndarray
    n_boundaries: int
    sample_index: np.ndarray
    position: np.ndarray
    prefix_len: np.ndarray
    stratum: np.ndarray
    multiplicity: np.ndarray
    sample_offset: int = 0
    short_hash: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), np.uint64))
    short_row: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))

    def __post_init__(self) -> None:
        _validate_stratification(self)
        if self.stratum.ndim != 1 or not np.issubdtype(self.stratum.dtype, np.integer):
            raise ValueError("stratum must be a vector of integer labels")
        if ((self.stratum < 0) | (self.stratum >= len(self.strata))).any():
            raise ValueError("boundary stratum index is outside the partition")

    def frame_size(self) -> int:
        """Boundaries the sampler may draw from."""
        return int(self.sample_index.size)

    def shard_view(
        self,
        sample_lo: int,
        sample_hi: int,
        boundary_counts: np.ndarray,
        n_boundaries: int,
    ) -> "BoundaryIndex":
        """The frame rows for samples ``[lo, hi)``, with that shard's counts.

        A sweep task reads the pooled frame, which carries the collapsed
        multiplicities, restricted to the samples it forwards.

        Args:
            sample_lo: First pool-global sample index the shard covers.
            sample_hi: One past the last sample it covers.
            boundary_counts: Boundaries per sample for this shard alone.
            n_boundaries: Their total, which the shard's denominator needs.
        """
        mask = (self.sample_index >= sample_lo) & (self.sample_index < sample_hi)
        return replace(
            self,
            n_samples=sample_hi - sample_lo,
            boundary_counts=np.asarray(boundary_counts),
            n_boundaries=int(n_boundaries),
            sample_index=self.sample_index[mask],
            position=self.position[mask],
            prefix_len=self.prefix_len[mask],
            stratum=self.stratum[mask],
            multiplicity=self.multiplicity[mask],
            sample_offset=int(sample_lo),
            short_hash=np.zeros((0, 2), np.uint64),
            short_row=np.zeros(0, np.int64),
        )

    def save(self, path: "str | Path") -> None:
        """Serialize every field to one ``.npz`` file.

        Args:
            path: Destination file; numpy appends ``.npz`` when it is absent.
        """
        np.savez(
            path,
            strata=np.asarray(self.strata),
            stratification_id=np.array(self.stratification_id),
            n_samples=np.int64(self.n_samples),
            boundary_counts=self.boundary_counts,
            n_boundaries=np.int64(self.n_boundaries),
            sample_index=self.sample_index,
            position=self.position,
            prefix_len=self.prefix_len,
            stratum=self.stratum,
            multiplicity=self.multiplicity,
            sample_offset=np.int64(self.sample_offset),
            short_hash=self.short_hash,
            short_row=self.short_row,
        )

    @classmethod
    def load(
        cls, path: "str | Path", *, stratification: Stratification | None = None
    ) -> "BoundaryIndex":
        """Restore an index written by :meth:`save`.

        Args:
            path: The ``.npz`` file to read.
            stratification: Expected policy; required for legacy archives
                without partition metadata.
        """
        with np.load(path, allow_pickle=False) as blob:
            return cls(
                **_read_stratification(blob, stratification),
                n_samples=int(blob["n_samples"]),
                boundary_counts=blob["boundary_counts"],
                n_boundaries=int(blob["n_boundaries"]),
                sample_index=blob["sample_index"],
                position=blob["position"],
                prefix_len=blob["prefix_len"],
                stratum=blob["stratum"],
                multiplicity=blob["multiplicity"],
                sample_offset=int(blob["sample_offset"]),
                short_hash=blob["short_hash"],
                short_row=blob["short_row"],
            )


def _prefix_hash(ids: Sequence[int]) -> tuple[int, int]:
    """A 128-bit digest of a prefix token sequence, as two 64-bit halves."""
    blob = np.asarray(ids, dtype=np.int32).tobytes()
    digest = hashlib.blake2b(blob, digest_size=16).digest()
    return (
        int.from_bytes(digest[:8], "little"),
        int.from_bytes(digest[8:], "little"),
    )


def _prefix_window(tokenizer, ids: Sequence[int], p: int, window_chars: int) -> str:
    """The decoded tail of ``ids[:p+1]``, long enough to fill the window."""
    start = max(1, p + 1 - window_chars)
    text = tokenizer.decode(list(ids[start : p + 1]), skip_special_tokens=True)
    return text[-window_chars:]


def boundary_index(
    samples: Sequence[str],
    tokenizer,
    *,
    stratification: Stratification,
    sample_offset: int = 0,
    dup_collapse_max_len: int = 0,
    collapse_bos: bool = True,
    drop_terminal: bool = True,
    encoded: "Sequence[Sequence[int]] | None" = None,
    start_ids=None,
) -> BoundaryIndex:
    """Tokenize *samples* and describe their word boundaries.

    Args:
        samples: Sentences of one shard, in pool order.
        tokenizer: Tokenizer whose unit-opening ids define the boundaries.
        stratification: Caller-defined prefix classifier and ordered labels.
        sample_offset: Pool-global index of the first entry of *samples*, so a
            shard's rows carry pool-global sample indices.
        dup_collapse_max_len: Longest prefix eligible for duplicate folding.
            Keep it at zero: boundaries sharing a prefix share their
            probabilities but not their successors, and the remainder needs to
            know which boundaries the word's first token was taken at.
        collapse_bos: Drop the ``[bos]``-only boundary of every sample from the
            frame. It stays in ``n_boundaries`` and re-enters the estimator as
            one collapsed term of multiplicity ``n_samples``.
        drop_terminal: Drop the boundary whose prefix is the whole sample.
            Termination is forced there, so no unit can open after it and its
            contribution is zero; it stays in ``n_boundaries`` because it is
            still a slot the denominator counts.
        encoded: Token ids of *samples*. Pass the pool's own store: re-encoding
            the decoded text can segment it differently from the way it was
            sampled, and every later stage reads prefixes out of the store, so
            the two have to be the same sequence. ``None`` encodes here.
        start_ids: The one-token prefix *samples* were drawn from, used when
            encoding here; ``None`` is the tokenizer's convention.
    """
    bset = boundary_set(tokenizer)
    encoded = (
        _encode_samples(tokenizer, samples, start_ids) if encoded is None else encoded
    )
    if len(encoded) != len(samples):
        raise ValueError(f"{len(encoded)} encodings for {len(samples)} samples")
    counts: np.ndarray = np.zeros(len(samples), dtype=np.int32)
    rows: list[tuple[int, int, int]] = []
    windows: list[str] = []
    hashes: list[tuple[int, int]] = []
    hash_rows: list[int] = []
    for j, ids in enumerate(encoded):
        total = 0
        for p in range(len(ids)):
            if not bset.opens_unit(ids, p):
                continue
            total += 1
            if collapse_bos and p == 0:
                continue
            if drop_terminal and p == len(ids) - 1:
                continue
            if p + 1 > 255:
                raise ValueError("prefix length exceeds the uint8 index width")
            if p + 1 <= dup_collapse_max_len:
                hashes.append(_prefix_hash(ids[: p + 1]))
                hash_rows.append(len(rows))
            rows.append((sample_offset + j, p, p + 1))
            windows.append(
                _prefix_window(tokenizer, ids, p, stratification.window_chars)
            )
        counts[j] = total
    table = np.array(rows, dtype=np.int64).reshape(len(rows), 3)
    return BoundaryIndex(
        strata=stratification.labels,
        stratification_id=stratification.identity,
        n_samples=len(samples),
        boundary_counts=counts,
        n_boundaries=int(counts.sum()),
        sample_index=table[:, 0].astype(np.int32),
        position=table[:, 1].astype(np.uint8),
        prefix_len=table[:, 2].astype(np.uint8),
        stratum=stratification.label(windows),
        multiplicity=np.ones(len(rows), dtype=np.uint32),
        sample_offset=sample_offset,
        short_hash=np.array(hashes, dtype=np.uint64).reshape(len(hashes), 2),
        short_row=np.array(hash_rows, dtype=np.int64),
    )


def merge_index(
    parts: Sequence[BoundaryIndex],
    *,
    expected_shards: int,
    dup_collapse_max_len: int = 0,
) -> BoundaryIndex:
    """Concatenate per-shard indices, refusing to fold duplicate prefixes.

    Folding leaves the canonical sum unchanged but would bias the remainder,
    which subtracts a sum over the boundaries where the word's first token was
    taken: a folded group's members can have different successors.

    Args:
        parts: Per-shard indices; together they have to tile the pool's samples
            without a gap or an overlap.
        expected_shards: How many parts a complete pool has, checked against
            *parts*.
        dup_collapse_max_len: Longest prefix to fold; any positive value is
            refused.
    """
    if dup_collapse_max_len:
        raise ValueError(
            "duplicate-prefix folding is incompatible with the remainder; "
            "pass dup_collapse_max_len=0"
        )
    if len(parts) != expected_shards:
        raise ValueError(f"expected {expected_shards} index parts, got {len(parts)}")
    _same_stratification(parts)
    ordered = sorted(parts, key=lambda part: part.sample_offset)
    running = 0
    for part in ordered:
        if part.sample_offset != running:
            raise ValueError("index parts do not tile the pool without gaps")
        running += part.n_samples
    counts = np.concatenate([p.boundary_counts for p in ordered])
    frame = {
        name: np.concatenate([getattr(p, name) for p in ordered])
        for name in ("sample_index", "position", "prefix_len", "stratum")
    }
    mult = np.concatenate([p.multiplicity for p in ordered]).astype(np.int64)
    keep = _collapse_duplicates(ordered, mult, dup_collapse_max_len)
    return BoundaryIndex(
        strata=ordered[0].strata,
        stratification_id=ordered[0].stratification_id,
        n_samples=running,
        boundary_counts=counts,
        n_boundaries=int(counts.sum()),
        sample_index=frame["sample_index"][keep],
        position=frame["position"][keep],
        prefix_len=frame["prefix_len"][keep],
        stratum=frame["stratum"][keep],
        multiplicity=mult[keep].astype(np.uint32),
        sample_offset=0,
    )


def _collapse_duplicates(
    ordered: Sequence[BoundaryIndex],
    mult: np.ndarray,
    dup_collapse_max_len: int,
) -> np.ndarray:
    """Rows to keep after folding duplicate short prefixes into one row."""
    offsets, running = [], 0
    for part in ordered:
        offsets.append(running)
        running += part.frame_size()
    keep: np.ndarray = np.ones(running, dtype=bool)
    picked = [
        (p.short_hash[p.prefix_len[p.short_row] <= dup_collapse_max_len], off, p)
        for p, off in zip(ordered, offsets, strict=True)
    ]
    hashes = np.concatenate([h for h, _, _ in picked])
    rows = np.concatenate(
        [
            p.short_row[p.prefix_len[p.short_row] <= dup_collapse_max_len] + off
            for _, off, p in picked
        ]
    ).astype(np.int64)
    if rows.size == 0:
        return keep
    order = np.lexsort((rows, hashes[:, 1], hashes[:, 0]))
    hashes, rows = hashes[order], rows[order]
    fresh = np.r_[True, (hashes[1:] != hashes[:-1]).any(axis=1)]
    starts = np.flatnonzero(fresh)
    ends = np.r_[starts[1:], hashes.size]
    for start, end in zip(starts, ends, strict=True):
        if end - start < 2:
            continue
        group = rows[start:end]
        mult[group[0]] = int(mult[group].sum())
        keep[group[1:]] = False
    return keep


# --- sweep -----------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SweepPartial:
    """One shard's contribution to the pooled selection."""

    strata: tuple[str, ...]
    stratification_id: str
    shard_index: int
    n_samples: int
    n_boundaries: int
    colsum: np.ndarray
    stratum_boundary_counts: np.ndarray
    buf_p0: np.ndarray
    buf_sample: np.ndarray
    buf_pos: np.ndarray
    buf_mult: np.ndarray
    buf_count: np.ndarray
    cell_offsets: np.ndarray
    capacity: np.ndarray

    def __post_init__(self) -> None:
        _validate_stratification(self)
        _validate_stratum_arrays(self, self.capacity)

    def save(self, path: "str | Path") -> None:
        """Serialize every field to one ``.npz`` file.

        Args:
            path: Destination file; numpy appends ``.npz`` when it is absent.
        """
        np.savez(
            path,
            strata=np.asarray(self.strata),
            stratification_id=np.array(self.stratification_id),
            shard_index=np.int64(self.shard_index),
            n_samples=np.int64(self.n_samples),
            n_boundaries=np.int64(self.n_boundaries),
            colsum=self.colsum,
            stratum_boundary_counts=self.stratum_boundary_counts,
            buf_p0=self.buf_p0,
            buf_sample=self.buf_sample,
            buf_pos=self.buf_pos,
            buf_mult=self.buf_mult,
            buf_count=self.buf_count,
            cell_offsets=self.cell_offsets,
            capacity=self.capacity,
        )

    @classmethod
    def load(
        cls, path: "str | Path", *, stratification: Stratification | None = None
    ) -> "SweepPartial":
        """Restore a partial written by :meth:`save`.

        Args:
            path: The ``.npz`` file to read.
            stratification: Expected policy; required for legacy archives
                without partition metadata.
        """
        with np.load(path, allow_pickle=False) as blob:
            return cls(
                **_read_stratification(blob, stratification),
                shard_index=int(blob["shard_index"]),
                n_samples=int(blob["n_samples"]),
                n_boundaries=int(blob["n_boundaries"]),
                colsum=blob["colsum"],
                stratum_boundary_counts=blob["stratum_boundary_counts"],
                buf_p0=blob["buf_p0"],
                buf_sample=blob["buf_sample"],
                buf_pos=blob["buf_pos"],
                buf_mult=blob["buf_mult"],
                buf_count=blob["buf_count"],
                cell_offsets=blob["cell_offsets"],
                capacity=blob["capacity"],
            )


def _cell_id(
    col: int, stratum: int, replicate: int, replicates: int, n_strata: int
) -> int:
    return (col * n_strata + stratum) * replicates + replicate


def _priority_order(q: np.ndarray, sample: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """Indices by descending priority, ties by ``(sample, position)``."""
    return np.lexsort((pos.astype(np.int64), sample.astype(np.int64), -q))


def _priority(p0, mult, sample, pos, rng_seed) -> np.ndarray:
    """``size / u`` for boundaries, with ``size = p0 * multiplicity``."""
    u = np.asarray(boundary_uniform(rng_seed, sample, pos), dtype=np.float64)
    sizes = np.asarray(p0, dtype=np.float64) * np.asarray(mult, dtype=np.float64)
    return sizes / u


def bos_first_probs(
    model, tokenizer, first_ids: np.ndarray, start_ids=None
) -> np.ndarray:
    """``P(t | prefix)`` for every first token, from one forward pass.

    Args:
        model: Causal language model the pool was sampled from.
        tokenizer: Tokenizer whose model's start token the pool was drawn from.
        first_ids: Token ids to keep, in the sweep's column order.
        start_ids: The one-token prefix the pool was drawn from; ``None`` is
            the tokenizer's convention, see :func:`start_prefix`.
    """
    import torch

    device = next(model.parameters()).device
    bos = start_prefix(tokenizer, start_ids)[-1]
    ids = torch.tensor([[bos]], device=device)
    with torch.no_grad():
        probs = torch.softmax(model(ids).logits[0, -1, :], dim=-1)
    cols = torch.tensor(np.asarray(first_ids, dtype=np.int64), device=device)
    return probs.index_select(0, cols).double().cpu().numpy()


def _gather_rows(
    model,
    tokenizer,
    batch,
    positions,
    first_ids_t,
    opens_t,
    device,
    encoded=None,
    start_ids=None,
):
    """Softmax rows at *positions* of every sample, and their unit-open mass.

    Returns ``(rows, opens)``: the first-token probabilities of interest and
    :math:`W_b`, both read off the same softmax and kept on device.
    """
    import torch

    encoded = (
        _encode_samples(tokenizer, batch, start_ids) if encoded is None else encoded
    )
    padded, masks, max_len = _left_pad_batch(encoded, tokenizer.pad_token_id)
    with torch.no_grad():
        logits = _padded_logits(model, padded, masks, device)
        probs = torch.softmax(logits, dim=-1)
        gathered = probs.index_select(-1, first_ids_t)
        opened = probs.index_select(-1, opens_t).sum(-1)
    rows, cols = [], []
    for j, ids in enumerate(encoded):
        start = max_len - len(ids)
        for p in positions[j]:
            rows.append(j)
            cols.append(start + int(p))
    if not rows:
        return (
            gathered.new_zeros((0, gathered.shape[-1])),
            opened.new_zeros((0,)),
        )
    idx_r = torch.tensor(rows, device=device, dtype=torch.long)
    idx_c = torch.tensor(cols, device=device, dtype=torch.long)
    return gathered[idx_r, idx_c], opened[idx_r, idx_c]


class _CellBuffers:
    """Per-``(stratum, replicate)`` top-``K`` priority buffers over all columns.

    A group holds one rectangular buffer sized by the widest column budget in
    it; a column's own budget is applied when the buffers are flattened.
    """

    def __init__(self, capacity: np.ndarray, store_device=None):
        """Buffers sized by *capacity*, resting on *store_device* between merges.

        Args:
            capacity: Per-cell budgets, shape ``(n_cols, n_strata, replicates)``.
            store_device: Where a group's buffer rests between merges; ``None``
                keeps it on the device the candidates arrive on. A merge
                always runs on the candidates' device, so the selection is
                the same wherever the buffers rest.
        """
        self.capacity = np.asarray(capacity, dtype=np.int64)
        self.n_cols, self.n_strata, self.replicates = self.capacity.shape
        self.group_k = self.capacity.max(axis=0)
        self.store: dict[tuple[int, int], dict[str, Any]] = {}
        self.store_device = store_device

    def offer(self, key, q, p0, sample, pos, mult) -> None:
        """Merge one chunk's candidates for a ``(stratum, replicate)`` group."""
        import torch

        keep = int(self.group_k[key])
        if keep <= 0 or q.shape[1] == 0:
            return
        old = self.store.get(key)
        if old is not None:
            q = torch.cat([old["q"].to(q.device), q], dim=1)
            p0 = torch.cat([old["p0"].to(p0.device), p0], dim=1)
            sample = torch.cat([old["sample"].to(sample.device), sample], dim=1)
            pos = torch.cat([old["pos"].to(pos.device), pos], dim=1)
            mult = torch.cat([old["mult"].to(mult.device), mult], dim=1)
        order = _stable_priority_order(q, sample, pos)[:, :keep]
        rest = self.store_device if self.store_device is not None else q.device
        self.store[key] = {
            "q": torch.gather(q, 1, order).to(rest),
            "p0": torch.gather(p0, 1, order).to(rest),
            "sample": torch.gather(sample, 1, order).to(rest),
            "pos": torch.gather(pos, 1, order).to(rest),
            "mult": torch.gather(mult, 1, order).to(rest),
        }

    def flatten(self) -> dict[str, np.ndarray]:
        """Ragged per-cell arrays, each column trimmed to its own budget."""
        p0: list[np.ndarray] = []
        sample: list[np.ndarray] = []
        pos: list[np.ndarray] = []
        mult: list[np.ndarray] = []
        counts: np.ndarray = np.zeros(
            self.n_cols * self.n_strata * self.replicates, dtype=np.int32
        )
        for col in range(self.n_cols):
            for s in range(self.n_strata):
                for r in range(self.replicates):
                    cell = _cell_id(col, s, r, self.replicates, self.n_strata)
                    counts[cell] = self._emit(
                        (s, r),
                        col,
                        int(self.capacity[col, s, r]),
                        p0,
                        sample,
                        pos,
                        mult,
                    )
        offsets: np.ndarray = np.zeros(counts.size + 1, dtype=np.int64)
        np.cumsum(counts, out=offsets[1:])
        return {
            "buf_p0": _concat(p0, np.float32),
            "buf_sample": _concat(sample, np.int32),
            "buf_pos": _concat(pos, np.uint8),
            "buf_mult": _concat(mult, np.uint32),
            "buf_count": counts,
            "cell_offsets": offsets,
            "capacity": self.capacity.copy(),
        }

    def _emit(self, key, col, budget, p0, sample, pos, mult) -> int:
        held = self.store.get(key)
        if held is None or budget <= 0:
            return 0
        row_q = held["q"][col].cpu().numpy()
        live = int(np.count_nonzero(np.isfinite(row_q) & (row_q > 0.0)))
        take = min(live, budget)
        if take == 0:
            return 0
        p0.append(held["p0"][col, :take].cpu().numpy())
        sample.append(held["sample"][col, :take].cpu().numpy())
        pos.append(held["pos"][col, :take].cpu().numpy())
        mult.append(held["mult"][col, :take].cpu().numpy())
        return take


def _stable_priority_order(q, sample, pos):
    """Row-wise descending-``q`` order, ties by ``(sample, position)``."""
    import torch

    tie = sample.to(torch.int64) * 256 + pos.to(torch.int64)
    by_tie = torch.argsort(tie, dim=1, stable=True)
    q_sorted = torch.gather(q, 1, by_tie)
    by_q = torch.argsort(q_sorted, dim=1, descending=True, stable=True)
    return torch.gather(by_tie, 1, by_q)


def _concat(parts: Sequence[np.ndarray], dtype) -> np.ndarray:
    if not parts:
        return np.zeros(0, dtype=dtype)
    return np.concatenate(parts).astype(dtype, copy=False)


def sweep_shard(
    samples: Sequence[str],
    index: BoundaryIndex,
    model,
    tokenizer,
    *,
    first_ids: np.ndarray,
    mt_first_ids: np.ndarray,
    k_store: np.ndarray,
    k_shard: np.ndarray,
    rng_seed: int,
    shard_index: int,
    batch_size: int = 64,
    reduce_chunk: int = 8192,
    encoded: "Sequence[Sequence[int]] | None" = None,
    store_device=None,
    start_ids=None,
) -> SweepPartial:
    """Score one shard's boundaries and keep what the pooled estimator needs.

    Emits the exact float64 column mass per stratum plus, for every
    ``(multi-token first id, stratum, replicate)`` cell, this shard's top
    candidates by priority.

    Every first-token probability is divided by the boundary's unit-open mass
    :math:`W_b` before it is stored. A prefix is in the frame because the token
    that followed it opened a unit, so the frame is a size-biased sample of the
    prefixes and :math:`1/W_b` is what removes that bias. Sizes, column masses
    and priorities are therefore all in units of :math:`s_b/W_b`.

    Args:
        samples: This shard's sentences, in pool order.
        index: Frame rows restricted to *samples*, from
            :meth:`BoundaryIndex.shard_view`.
        model: Causal language model scoring the boundaries.
        tokenizer: Tokenizer paired with *model*.
        first_ids: Sorted distinct first tokens to record column mass for.
        mt_first_ids: The subset of *first_ids* that gets per-cell priority
            buffers; it fixes the row order of *k_store* and *k_shard*.
        k_store: Cell sizes the merged pool keeps, shape
            ``(len(mt_first_ids), n_strata, replicates)``.
        k_shard: Cell sizes this shard keeps, same shape, never above
            *k_store*.
        rng_seed: Seed of the priority draw, shared across the pool's shards.
        shard_index: Position of this shard, checked for gaps at merge time.
        batch_size: Samples per forward pass.
        reduce_chunk: Boundaries per reduction block. Fixed, so the column sums
            do not depend on *batch_size*.
        encoded: Token ids of *samples* from the pool's store; ``None`` encodes
            them here.
        store_device: Where the priority buffers rest between merges, see
            :class:`_CellBuffers`; ``None`` keeps them on the model's device.
        start_ids: The one-token prefix *samples* were drawn from, used when
            encoding here; ``None`` is the tokenizer's convention.
    """
    import torch

    first_ids = np.asarray(first_ids, dtype=np.int64)
    mt_first_ids = np.asarray(mt_first_ids, dtype=np.int64)
    if np.any(np.diff(first_ids) <= 0):
        raise ValueError("first_ids must be sorted and distinct")
    _validate_capacity(k_store, len(index.strata), mt_first_ids.size)
    _validate_capacity(k_shard, len(index.strata), mt_first_ids.size)
    if np.shape(k_shard) != np.shape(k_store):
        raise ValueError("k_shard and k_store disagree on the replicate count")
    if np.any(np.asarray(k_shard) > np.asarray(k_store)):
        raise ValueError("k_shard exceeds k_store; the merged buffer is smaller")
    device = next(model.parameters()).device
    mt_cols = np.searchsorted(first_ids, mt_first_ids)
    first_ids_t = torch.tensor(first_ids, device=device, dtype=torch.long)
    opens_t = torch.tensor(
        boundary_set(tokenizer).opens,
        device=device,
        dtype=torch.long,
    )
    buffers = _CellBuffers(np.asarray(k_shard, dtype=np.int64), store_device)
    state = _SweepState(index, reduce_chunk, rng_seed, buffers.replicates, first_ids)
    local: np.ndarray = index.sample_index.astype(np.int64) - index.sample_offset
    starts = np.asarray(np.searchsorted(local, np.arange(len(samples) + 1)))
    for i in range(0, len(samples), batch_size):
        batch = list(samples[i : i + batch_size])
        positions = [
            index.position[starts[i + j] : starts[i + j + 1]] for j in range(len(batch))
        ]
        rows, opened = _gather_rows(
            model,
            tokenizer,
            batch,
            positions,
            first_ids_t,
            opens_t,
            device,
            None if encoded is None else encoded[i : i + batch_size],
            start_ids=start_ids,
        )
        rows = rows / opened.clamp(min=_F32_TINY).unsqueeze(-1)
        state.push(rows, mt_cols, buffers)
    state.flush(mt_cols, buffers)
    return SweepPartial(
        strata=index.strata,
        stratification_id=index.stratification_id,
        shard_index=shard_index,
        n_samples=index.n_samples,
        n_boundaries=index.n_boundaries,
        colsum=state.colsum,
        stratum_boundary_counts=state.stratum_counts,
        **buffers.flatten(),
    )


class _SweepState:
    """Fixed-size chunking of the sweep, so reductions ignore the batch size."""

    def __init__(self, index, reduce_chunk, rng_seed, replicates, first_ids):
        self.index = index
        self.reduce_chunk = int(reduce_chunk)
        self.rng_seed = int(rng_seed)
        self.replicates = int(replicates)
        self.colsum = np.zeros((len(index.strata), first_ids.size), dtype=np.float64)
        self.stratum_counts = np.zeros(len(index.strata), dtype=np.int64)
        self._pending: list[Any] = []
        self._pending_n = 0
        self._start = 0

    def push(self, rows, mt_cols, buffers) -> None:
        if rows.shape[0] == 0:
            return
        self._pending.append(rows)
        self._pending_n += int(rows.shape[0])
        while self._pending_n >= self.reduce_chunk:
            self._consume(self.reduce_chunk, mt_cols, buffers)

    def flush(self, mt_cols, buffers) -> None:
        while self._pending_n > 0:
            self._consume(self._pending_n, mt_cols, buffers)

    def _consume(self, size, mt_cols, buffers) -> None:
        import torch

        block = (
            torch.cat(self._pending, dim=0)
            if len(self._pending) > 1
            else self._pending[0]
        )
        chunk, rest = block[:size], block[size:]
        self._pending = [rest] if rest.shape[0] else []
        self._pending_n -= size
        lo = self._start
        self._start += size
        self._reduce(chunk, lo, lo + size)
        self._select(chunk, lo, lo + size, mt_cols, buffers)

    def _reduce(self, chunk, lo, hi) -> None:
        """Exact per-stratum column mass, in float64, over one fixed chunk."""
        import torch

        stratum = self.index.stratum[lo:hi]
        mult = self.index.multiplicity[lo:hi].astype(np.float64)
        weights = torch.tensor(mult, dtype=torch.float64, device=chunk.device)
        for s in np.unique(stratum):
            mask = torch.tensor(stratum == s, device=chunk.device)
            block = chunk[mask].double() * weights[mask][:, None]
            self.colsum[int(s)] += block.sum(0).cpu().numpy()
            self.stratum_counts[int(s)] += int(mask.sum())

    def _select(self, chunk, lo, hi, mt_cols, buffers) -> None:
        """Offer this chunk's boundaries to the per-cell priority buffers."""
        import torch

        idx = self.index
        sample = idx.sample_index[lo:hi].astype(np.int64)
        pos = idx.position[lo:hi].astype(np.int64)
        stratum = idx.stratum[lo:hi]
        mult = idx.multiplicity[lo:hi].astype(np.int64)
        u = np.asarray(boundary_uniform(self.rng_seed, sample, pos), dtype=np.float64)
        replicate = sample % self.replicates
        cols = torch.tensor(mt_cols, device=chunk.device, dtype=torch.long)
        p0 = chunk.index_select(1, cols)
        for s in np.unique(stratum):
            for r in range(self.replicates):
                rows = np.flatnonzero((stratum == s) & (replicate == r))
                if rows.size == 0:
                    continue
                self._offer(buffers, (int(s), r), rows, p0, mult, u, sample, pos)

    def _offer(self, buffers, key, rows, p0, mult, u, sample, pos) -> None:
        import torch

        take = torch.tensor(rows, device=p0.device, dtype=torch.long)
        block = p0.index_select(0, take).double()
        w = torch.tensor(mult[rows], dtype=torch.float64, device=p0.device)
        sizes = block * w[:, None]
        q = (
            sizes
            / torch.tensor(u[rows], dtype=torch.float64, device=p0.device)[:, None]
        )
        q = torch.where(sizes > 0.0, q, torch.full_like(q, -math.inf))
        n = rows.size
        buffers.offer(
            key,
            q.T.contiguous(),
            block.T.contiguous().float(),
            torch.tensor(sample[rows], dtype=torch.int64, device=p0.device)
            .expand(1, n)
            .repeat(q.shape[1], 1),
            torch.tensor(pos[rows], dtype=torch.int64, device=p0.device)
            .expand(1, n)
            .repeat(q.shape[1], 1),
            torch.tensor(mult[rows], dtype=torch.int64, device=p0.device)
            .expand(1, n)
            .repeat(q.shape[1], 1),
        )


# --- pooled selection ------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PoolSelection:
    """The pool-wide inputs a continuation stage needs, and nothing else."""

    strata: tuple[str, ...]
    stratification_id: str
    pool_fingerprint: str
    n_pool_samples: int
    n_boundaries: int
    denom: int
    first_ids: np.ndarray
    mt_first_ids: np.ndarray
    bos_p0: np.ndarray
    colsum: np.ndarray
    stratum_boundary_counts: np.ndarray
    k_store: np.ndarray
    sel_p0: np.ndarray
    sel_sample: np.ndarray
    sel_pos: np.ndarray
    sel_mult: np.ndarray
    cell_offsets: np.ndarray
    rng_seed: int
    estimator_version: str = SELECTION_VERSION
    sel_prefix: np.ndarray | None = None
    sel_prefix_off: np.ndarray | None = None
    #: Token that followed each selected boundary, and how many samples open
    #: with each of ``first_ids``. Both are filled by :func:`attach_prefixes`.
    sel_next: np.ndarray | None = None
    first_token_counts: np.ndarray | None = None
    first_col: Mapping[int, int] = field(default_factory=dict)
    mt_col: Mapping[int, int] = field(default_factory=dict)
    #: The one-token prefix the pool was drawn and scored from; ``None`` for an
    #: archive written before it was recorded, which keeps the tokenizer's
    #: convention.
    start: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if self.start is not None:
            object.__setattr__(self, "start", _archive_start(self.start))
        _validate_stratification(self)
        _validate_stratum_arrays(self, self.k_store)
        if self.colsum.shape[1] != self.first_ids.size:
            raise ValueError("column mass shape disagrees with first_ids")
        _validate_capacity(self.k_store, len(self.strata), self.mt_first_ids.size)
        object.__setattr__(
            self, "first_col", {int(t): i for i, t in enumerate(self.first_ids)}
        )
        object.__setattr__(
            self, "mt_col", {int(t): i for i, t in enumerate(self.mt_first_ids)}
        )
        if self.denom != self.n_pool_samples + self.n_boundaries:
            raise ValueError("denom must be n_pool_samples + n_boundaries")
        for name in ("colsum", "bos_p0", "sel_p0"):
            values = np.asarray(getattr(self, name))
            if values.size and not bool(np.isfinite(values).all()):
                raise ValueError(f"{name} carries a non-finite entry")

    @property
    def replicates(self) -> int:
        """Independent replicates the selection was drawn in."""
        return int(self.k_store.shape[-1])

    def cell(self, first_col: int, stratum: int, replicate: int) -> slice:
        """Slice of the ``sel_*`` arrays holding one cell's kept boundaries.

        Args:
            first_col: Position of the first token within ``mt_first_ids``.
            stratum: Index into the selection's ordered ``strata``.
            replicate: Which replicate of the frame partition, below
                :attr:`replicates`.
        """
        cell = _cell_id(
            first_col, stratum, replicate, self.replicates, len(self.strata)
        )
        return slice(int(self.cell_offsets[cell]), int(self.cell_offsets[cell + 1]))

    def prefix(self, entry: int) -> list[int]:
        """Token ids of the prefix of one selected boundary.

        Args:
            entry: Row of the ``sel_*`` arrays, as :meth:`cell` slices them.
        """
        if self.sel_prefix is None or self.sel_prefix_off is None:
            raise ValueError(
                "this selection carries no prefixes; attach them from the pool "
                "token store before running continuations"
            )
        lo = int(self.sel_prefix_off[entry])
        hi = int(self.sel_prefix_off[entry + 1])
        return [int(t) for t in self.sel_prefix[lo:hi]]

    def slice_for(self, first_cols: Sequence[int]) -> "PoolSelection":
        """A selection restricted to some multi-token first-token columns.

        Args:
            first_cols: Positions within ``mt_first_ids`` to keep; duplicates
                collapse and the result is in ascending column order.
        """
        cols = sorted({int(c) for c in first_cols})
        offsets, entries = [0], []
        for col in cols:
            for s in range(len(self.strata)):
                for rep in range(self.replicates):
                    sl = self.cell(col, s, rep)
                    entries.append(np.arange(sl.start, sl.stop, dtype=np.int64))
                    offsets.append(offsets[-1] + (sl.stop - sl.start))
        rows: np.ndarray = (
            np.concatenate(entries) if entries else np.zeros(0, dtype=np.int64)
        ).astype(np.int64)
        sub = replace(
            self,
            mt_first_ids=self.mt_first_ids[cols],
            k_store=self.k_store[cols],
            sel_p0=self.sel_p0[rows],
            sel_sample=self.sel_sample[rows],
            sel_pos=self.sel_pos[rows],
            sel_mult=self.sel_mult[rows],
            cell_offsets=np.array(offsets, dtype=np.int64),
            sel_prefix=None,
            sel_prefix_off=None,
            sel_next=None if self.sel_next is None else self.sel_next[rows],
        )
        if self.sel_prefix is None or self.sel_prefix_off is None:
            return sub
        return _carry_prefixes(sub, self, rows)

    def save(self, path: "str | Path") -> None:
        """Serialize every field to one ``.npz`` file.

        Optional fields are written only when they are present, so a selection
        without prefixes attached round-trips as one.

        Args:
            path: Destination file; numpy appends ``.npz`` when it is absent.
        """
        arrays: dict[str, Any] = {
            "strata": np.asarray(self.strata),
            "stratification_id": np.array(self.stratification_id),
            "pool_fingerprint": np.array(self.pool_fingerprint),
            "n_pool_samples": np.int64(self.n_pool_samples),
            "n_boundaries": np.int64(self.n_boundaries),
            "denom": np.int64(self.denom),
            "first_ids": self.first_ids,
            "mt_first_ids": self.mt_first_ids,
            "bos_p0": self.bos_p0,
            "colsum": self.colsum,
            "stratum_boundary_counts": self.stratum_boundary_counts,
            "k_store": self.k_store,
            "sel_p0": self.sel_p0,
            "sel_sample": self.sel_sample,
            "sel_pos": self.sel_pos,
            "sel_mult": self.sel_mult,
            "cell_offsets": self.cell_offsets,
            "rng_seed": np.int64(self.rng_seed),
            "estimator_version": np.array(self.estimator_version),
        }
        if self.sel_prefix is not None and self.sel_prefix_off is not None:
            arrays["sel_prefix"] = self.sel_prefix
            arrays["sel_prefix_off"] = self.sel_prefix_off
        if self.sel_next is not None:
            arrays["sel_next"] = self.sel_next
        if self.first_token_counts is not None:
            arrays["first_token_counts"] = self.first_token_counts
        if self.start is not None:
            arrays["start"] = np.asarray(self.start, dtype=np.int64)
        np.savez(path, **arrays)

    @classmethod
    def load(
        cls, path: "str | Path", *, stratification: Stratification | None = None
    ) -> "PoolSelection":
        """Restore a selection written by :meth:`save`.

        Args:
            path: The ``.npz`` file to read.
            stratification: Expected policy; required for legacy archives
                without partition metadata.
        """
        with np.load(path, allow_pickle=False) as blob:
            keys = set(blob.files)
            metadata = _read_stratification(blob, stratification)
            version = str(blob["estimator_version"])
            legacy = not {"strata", "stratification_id"} & keys
            if legacy and version == "mmc_pps/2":
                version = SELECTION_VERSION
            elif legacy or version != SELECTION_VERSION:
                raise ValueError(
                    "unsupported selection layout or missing partition metadata"
                )
            return cls(
                **metadata,
                pool_fingerprint=str(blob["pool_fingerprint"]),
                n_pool_samples=int(blob["n_pool_samples"]),
                n_boundaries=int(blob["n_boundaries"]),
                denom=int(blob["denom"]),
                first_ids=blob["first_ids"],
                mt_first_ids=blob["mt_first_ids"],
                bos_p0=blob["bos_p0"],
                colsum=blob["colsum"],
                stratum_boundary_counts=blob["stratum_boundary_counts"],
                k_store=blob["k_store"],
                sel_p0=blob["sel_p0"],
                sel_sample=blob["sel_sample"],
                sel_pos=blob["sel_pos"],
                sel_mult=blob["sel_mult"],
                cell_offsets=blob["cell_offsets"],
                rng_seed=int(blob["rng_seed"]),
                estimator_version=version,
                sel_prefix=blob["sel_prefix"] if "sel_prefix" in keys else None,
                sel_prefix_off=(
                    blob["sel_prefix_off"] if "sel_prefix_off" in keys else None
                ),
                sel_next=blob["sel_next"] if "sel_next" in keys else None,
                first_token_counts=(
                    blob["first_token_counts"] if "first_token_counts" in keys else None
                ),
                start=_archive_start(blob["start"]) if "start" in keys else None,
            )


def _archive_start(value) -> tuple[int, ...]:
    """The prefix of a selection, as built or as read from its archive: one
    non-negative integer id.

    Raises:
        ValueError: The value is not one integer id (a float, a bool, a
            string, a negative number, nothing or several ids).
    """
    array = np.asarray(value)
    if (
        array.ndim != 1
        or array.size != 1
        or array.dtype.kind not in "iu"
        or int(array[0]) < 0
    ):
        raise ValueError(
            "the selection archive's start must be one non-negative integer id"
        )
    return (int(array[0]),)


def _ragged_gather(source: np.ndarray, starts: np.ndarray, lengths: np.ndarray):
    """Concatenate ``source[s : s + n]`` over every ``(s, n)`` pair.

    Vectorized because the selection holds one row per drawn boundary and a
    Python loop over them dominates the merge at pool scale.
    """
    offsets: np.ndarray = np.zeros(lengths.size + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])
    total = int(offsets[-1])
    if total == 0:
        return np.zeros(0, dtype=np.int32), offsets
    within = np.arange(total, dtype=np.int64) - np.repeat(offsets[:-1], lengths)
    index = np.repeat(starts.astype(np.int64), lengths) + within
    return np.asarray(source)[index].astype(np.int32, copy=False), offsets


def attach_prefixes(
    selection: PoolSelection, tokens: np.ndarray, offsets: np.ndarray
) -> PoolSelection:
    """Inline each boundary's prefix, its successor, and the pool's openers.

    ``sel_next`` and the opener counts belong to the stored selection layout;
    the opener counts drive the collapsed bos channel of the estimate. Both
    are read off the token store, so neither costs a model pass.

    Args:
        selection: Merged selection whose drawn boundaries lack their prefixes.
        tokens: The pool's token ids, every sample concatenated in pool order.
        offsets: Where each sample starts in *tokens*, with a trailing end, so
            it has one entry more than the pool has samples.
    """
    off = np.asarray(offsets, dtype=np.int64)
    lengths: np.ndarray = selection.sel_pos.astype(np.int64) + 1
    starts = off[selection.sel_sample.astype(np.int64)]
    flat, prefix_off = _ragged_gather(tokens, starts, lengths)
    # A terminal boundary is out of the frame, so a successor exists inside the
    # sample. Reading past its end would silently pick up the next sample's bos,
    # which only happens if the store and the index disagree on the encoding.
    ends = off[selection.sel_sample.astype(np.int64) + 1]
    if np.any(starts + lengths >= ends):
        raise ValueError(
            "a boundary's successor falls outside its sample; the token store "
            "and the boundary index were built from different encodings"
        )
    nxt = np.asarray(tokens, dtype=np.int64)[starts + lengths]
    return replace(
        selection,
        sel_prefix=flat,
        sel_prefix_off=prefix_off,
        sel_next=nxt.astype(np.int32),
        first_token_counts=_opener_counts(tokens, off, selection.first_ids),
    )


def _opener_counts(
    tokens: np.ndarray, offsets: np.ndarray, first_ids: np.ndarray
) -> np.ndarray:
    """How many samples open with each of *first_ids*.

    Every sample's bos boundary is in the frame, so the collapsed bos term
    stands in for one boundary per sample; this is how many of them the
    remainder has to take back.
    """
    lengths = np.diff(offsets)
    keep = lengths > 1
    opener = np.asarray(tokens, dtype=np.int64)[offsets[:-1][keep] + 1]
    order = np.asarray(first_ids, dtype=np.int64)
    slot = np.searchsorted(order, opener)
    slot = np.where(slot < order.size, slot, 0)
    hit = order[slot] == opener
    return np.bincount(slot[hit], minlength=order.size).astype(np.int64)


def _carry_prefixes(target, source, rows):
    """Copy *source*'s prefixes for *rows* onto *target*."""
    lengths = np.diff(source.sel_prefix_off)[rows]
    flat, off = _ragged_gather(source.sel_prefix, source.sel_prefix_off[rows], lengths)
    return replace(target, sel_prefix=flat, sel_prefix_off=off)


def merge_sweep(
    partials: "Sequence[Path | str | SweepPartial]",
    *,
    expected_shards: int,
    n_pool_samples: int,
    n_boundaries: int,
    bos_p0: np.ndarray,
    k_store: np.ndarray,
    first_ids: np.ndarray,
    mt_first_ids: np.ndarray,
    pool_fingerprint: str,
    rng_seed: int,
    wave: int = 20,
    delete_on_consume: bool = False,
    stratification: Stratification | None = None,
    start=None,
) -> PoolSelection:
    """Fold per-shard sweeps into one pooled selection, refusing a gap.

    The per-shard buffers merge exactly because a boundary's priority depends
    only on ``(rng_seed, sample, position)``, so the union of the shards' top
    candidates holds the pool's top candidates unless a shard's own buffer was
    truncated above the merged threshold, which is refused.

    Args:
        partials: One entry per shard, either the object or the path
            :meth:`SweepPartial.save` wrote. Shard indices have to cover
            ``0..expected_shards-1`` exactly once.
        expected_shards: How many partials a complete sweep has.
        n_pool_samples: Samples in the pool, checked against the partials.
        n_boundaries: Boundaries in the pool, checked the same way.
        bos_p0: ``P(t | prefix)`` per first token, from :func:`bos_first_probs`.
        k_store: Cell sizes the merged buffers keep, shape
            ``(len(mt_first_ids), n_strata, replicates)``.
        first_ids: Sorted distinct first tokens the sweep recorded mass for.
        mt_first_ids: The subset with per-cell buffers, in *k_store* row order.
        pool_fingerprint: Identity of the pool, carried into the selection and
            checked at estimate time.
        start: The one-token prefix the pool was drawn from, carried into the
            selection; ``None`` keeps the tokenizer's convention.
        rng_seed: Seed the shards drew their priorities under.
        wave: Validated positive; the fold reads every partial in one pass.
        delete_on_consume: Unlink partial files after a successful merge.
        stratification: Expected policy, required when loading legacy partials.
    """
    loaded = _load_partials(partials, expected_shards, wave, stratification)
    _same_stratification(loaded)
    _validate_capacity(k_store, len(loaded[0].strata), len(mt_first_ids))
    if any(p.capacity.shape != np.shape(k_store) for p in loaded):
        raise ValueError("sweep partials disagree on the cell layout")
    total_samples = int(np.array([p.n_samples for p in loaded], np.int64).sum())
    total_bounds = int(np.array([p.n_boundaries for p in loaded], np.int64).sum())
    if total_samples != n_pool_samples:
        raise ValueError(
            f"sweep covers {total_samples} samples, the pool has {n_pool_samples}"
        )
    if total_bounds != n_boundaries:
        raise ValueError(
            f"sweep counted {total_bounds} boundaries, the index has {n_boundaries}"
        )
    colsum = np.add.reduce(np.stack([p.colsum for p in loaded]), axis=0)
    counts = np.add.reduce(
        np.stack([p.stratum_boundary_counts for p in loaded]), axis=0
    )
    merged = _merge_cells(loaded, k_store, rng_seed)
    selection = PoolSelection(
        strata=loaded[0].strata,
        stratification_id=loaded[0].stratification_id,
        pool_fingerprint=pool_fingerprint,
        n_pool_samples=n_pool_samples,
        n_boundaries=n_boundaries,
        denom=n_pool_samples + n_boundaries,
        first_ids=np.asarray(first_ids, dtype=np.int32),
        mt_first_ids=np.asarray(mt_first_ids, dtype=np.int32),
        bos_p0=np.asarray(bos_p0, dtype=np.float64),
        colsum=colsum,
        stratum_boundary_counts=counts,
        k_store=np.asarray(k_store, dtype=np.int64),
        rng_seed=int(rng_seed),
        start=None if start is None else tuple(int(i) for i in start),
        **merged,
    )
    if delete_on_consume:
        for item in partials:
            if not isinstance(item, SweepPartial):
                Path(item).unlink()
    return selection


def build_pool_selection(
    shards: "Sequence[Sequence[str]]",
    model,
    tokenizer,
    *,
    stratification: Stratification,
    first_ids: np.ndarray,
    mt_first_ids: np.ndarray,
    k_store: np.ndarray,
    k_shard: np.ndarray | None = None,
    rng_seed: int,
    pool_fingerprint: str = "local",
    batch_size: int = 64,
    reduce_chunk: int = 8192,
    dup_collapse_max_len: int = 0,
    start_ids=None,
) -> PoolSelection:
    """Run index, sweep and merge over *shards* in one process.

    The sharded pipeline runs the same stages across tasks; this is the
    single-process form used for small pools and rehearsals.

    Args:
        shards: The pool split into groups of samples, one group per shard.
        model: Causal language model scoring the boundaries.
        tokenizer: Tokenizer paired with *model*.
        stratification: Caller-defined prefix classifier and ordered labels.
        first_ids: Sorted distinct first tokens to record column mass for.
        mt_first_ids: The subset that gets per-cell buffers, in *k_store* row
            order.
        k_store: Cell sizes the merged buffers keep, shape
            ``(len(mt_first_ids), n_strata, replicates)``.
        k_shard: Cell sizes each shard keeps; ``None`` gives every shard the
            merged size.
        rng_seed: Seed of the priority draw.
        pool_fingerprint: Identity of the pool, carried into the selection.
        batch_size: Samples per forward pass.
        reduce_chunk: Boundaries per reduction block.
        dup_collapse_max_len: Longest prefix to fold; any positive value is
            refused by the merge.
        start_ids: The one-token prefix *shards* were drawn from; ``None`` is
            the tokenizer's convention. The selection records the prefix
            either way.
    """
    start_ids = start_prefix(tokenizer, start_ids)
    parts, offset = [], 0
    for chunk in shards:
        parts.append(
            boundary_index(
                chunk,
                tokenizer,
                stratification=stratification,
                sample_offset=offset,
                dup_collapse_max_len=dup_collapse_max_len,
                start_ids=start_ids,
            )
        )
        offset += len(chunk)
    index = merge_index(
        parts,
        expected_shards=len(shards),
        dup_collapse_max_len=dup_collapse_max_len,
    )
    partials, lo = [], 0
    for shard, (chunk, part) in enumerate(zip(shards, parts, strict=True)):
        view = index.shard_view(
            lo, lo + len(chunk), part.boundary_counts, part.n_boundaries
        )
        lo += len(chunk)
        partials.append(
            sweep_shard(
                chunk,
                view,
                model,
                tokenizer,
                first_ids=first_ids,
                mt_first_ids=mt_first_ids,
                k_store=k_store,
                k_shard=k_store if k_shard is None else k_shard,
                rng_seed=rng_seed,
                shard_index=shard,
                batch_size=batch_size,
                reduce_chunk=reduce_chunk,
                start_ids=start_ids,
            )
        )
    selection = merge_sweep(
        partials,
        expected_shards=len(shards),
        n_pool_samples=index.n_samples,
        n_boundaries=index.n_boundaries,
        bos_p0=bos_first_probs(model, tokenizer, first_ids, start_ids),
        k_store=k_store,
        first_ids=first_ids,
        mt_first_ids=mt_first_ids,
        pool_fingerprint=pool_fingerprint,
        rng_seed=rng_seed,
        start=start_ids,
    )
    flat = [t for chunk in shards for t in chunk]
    tokens, offsets = _token_store(flat, tokenizer, start_ids)
    return attach_prefixes(selection, tokens, offsets)


def _token_store(samples, tokenizer, start_ids=None):
    """Ragged token ids of the pool, the layout ``attach_prefixes`` reads."""
    encoded = _encode_samples(tokenizer, samples, start_ids)
    offsets = np.zeros(len(encoded) + 1, dtype=np.int64)
    np.cumsum([len(e) for e in encoded], out=offsets[1:])
    if not encoded:
        return np.zeros(0, dtype=np.int32), offsets
    flat = np.concatenate([np.asarray(e, dtype=np.int32) for e in encoded])
    return flat, offsets


def _load_partials(partials, expected_shards, wave, stratification):
    """Read every partial, refusing a missing or repeated shard index."""
    if len(partials) != expected_shards:
        raise ValueError(
            f"expected {expected_shards} sweep partials, got {len(partials)}"
        )
    if wave < 1:
        raise ValueError("wave must be positive")
    loaded: list[SweepPartial] = []
    for item in partials:
        if isinstance(item, SweepPartial):
            loaded.append(item)
            continue
        loaded.append(SweepPartial.load(item, stratification=stratification))
    if stratification is not None and any(
        (p.strata, p.stratification_id)
        != (stratification.labels, stratification.identity)
        for p in loaded
    ):
        raise ValueError("sweep stratification does not match the requested policy")
    seen = sorted(p.shard_index for p in loaded)
    if seen != list(range(expected_shards)):
        raise ValueError(f"sweep shard indices are not 0..{expected_shards - 1}")
    return sorted(loaded, key=lambda p: p.shard_index)


def _merge_cells(loaded, k_store, rng_seed):
    """Top-``k_store`` union of the shards' buffers, per cell."""
    keep = np.asarray(k_store, dtype=np.int64).reshape(-1)
    n_cells = keep.size
    for part in loaded:
        if part.buf_count.size != n_cells:
            raise ValueError("sweep partials disagree on the cell layout")
    p0, sample, pos, mult = [], [], [], []
    offsets = np.zeros(n_cells + 1, dtype=np.int64)
    for cell in range(n_cells):
        cols = _cell_union(loaded, cell)
        q = _priority(cols["p0"], cols["mult"], cols["sample"], cols["pos"], rng_seed)
        order = _priority_order(q, cols["sample"], cols["pos"])
        _refuse_truncation(loaded, cell, q, order, int(keep[cell]))
        order = order[: int(keep[cell])]
        p0.append(cols["p0"][order])
        sample.append(cols["sample"][order])
        pos.append(cols["pos"][order])
        mult.append(cols["mult"][order])
        offsets[cell + 1] = offsets[cell] + order.size
    return {
        "sel_p0": _concat(p0, np.float32),
        "sel_sample": _concat(sample, np.int32),
        "sel_pos": _concat(pos, np.uint8),
        "sel_mult": _concat(mult, np.uint32),
        "cell_offsets": offsets,
    }


def _cell_union(loaded, cell):
    """One cell's candidates from every shard, in shard order."""
    out = {}
    for name, source, dtype in (
        ("p0", "buf_p0", np.float32),
        ("sample", "buf_sample", np.int32),
        ("pos", "buf_pos", np.uint8),
        ("mult", "buf_mult", np.uint32),
    ):
        parts = [
            getattr(p, source)[p.cell_offsets[cell] : p.cell_offsets[cell + 1]]
            for p in loaded
        ]
        out[name] = _concat(parts, dtype)
    return out


def _refuse_truncation(loaded, cell, q, order, keep) -> None:
    """Refuse a merge whose exactness a shard's own buffer cap destroyed."""
    if order.size <= keep or keep <= 0:
        return
    threshold = float(q[order[keep - 1]])
    base = 0
    for part in loaded:
        size = int(part.cell_offsets[cell + 1] - part.cell_offsets[cell])
        capacity = int(part.capacity.reshape(-1)[cell])
        if size and size == capacity and float(q[base : base + size].min()) > threshold:
            raise ValueError(
                f"shard {part.shard_index} filled cell {cell} above the merged "
                "threshold; raise k_shard and re-sweep"
            )
        base += size


# --- certificate -----------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SupportThresholds:
    """When a sampled estimate is fit to publish without a caveat."""

    cv_max: float = 0.10
    home_share_min: float = 0.9
    bos_share_max: float = 0.25
    top1_share_max: float = 0.25
    residual_ratio_max: float = 0.02
    n_nonzero_min: int = 50


def design_se(replicate_totals: np.ndarray) -> float:
    """Replicate standard error of the sum of *replicate_totals*.

    Args:
        replicate_totals: One total per independent replicate of the draw.
            Their spread scales into the error of their sum, so fewer than two
            replicates leave nothing to read and give zero.
    """
    totals = np.asarray(replicate_totals, dtype=np.float64)
    if totals.size < 2:
        return 0.0
    return float(math.sqrt(totals.size) * totals.std(ddof=1))


def dlt_variance(size: np.ndarray, tau: float, c: np.ndarray) -> float:
    """Duffield-Lund-Thorup variance term of one cell's selected items.

    An item of size ``w`` enters with probability ``min(1, w/tau)`` carrying
    weight ``max(w, tau)``, so its variance is ``w*(tau - w)`` below the
    threshold and zero above it. Charging ``(max(w,tau) - w) * max(w,tau)`` to
    the selected items alone recovers that in expectation.

    Args:
        size: Sizes ``w`` of one cell's selected items, each a boundary's
            first-token probability times its multiplicity.
        tau: That cell's priority threshold, which fixes both the inclusion
            probability and the inflated weight.
        c: Per-item continuation factor the weight is multiplied by, so an
            item is charged in units of ``c`` squared.
    """
    s = np.asarray(size, dtype=np.float64)
    cc = np.asarray(c, dtype=np.float64)
    inflated = np.maximum(s, tau)
    return float(((inflated - s) * inflated) @ (cc**2))


def certify(record_parts: Mapping[str, Any], thresholds: SupportThresholds) -> bool:
    """True when a word's certificate clears every support threshold.

    Args:
        record_parts: One word's certificate fields: ``p_hat``, ``cv``, the
            home, BOS and top-1 shares, the residual bound, the count of
            nonzero contributions, and the ``underflow_fallback`` and
            ``nonpositive`` flags, either of which refuses the word outright.
        thresholds: The bars each of those fields is held to.
    """
    p_hat = float(record_parts["p_hat"])
    if not math.isfinite(p_hat) or p_hat <= 0.0:
        return False
    if record_parts.get("nonpositive") or record_parts.get("underflow_fallback"):
        return False
    checks = (
        float(record_parts["cv"]) <= thresholds.cv_max,
        float(record_parts["home_share"]) >= thresholds.home_share_min,
        float(record_parts["bos_share"]) <= thresholds.bos_share_max,
        float(record_parts["top1_share"]) <= thresholds.top1_share_max,
        float(record_parts["residual_bound"]) / p_hat <= thresholds.residual_ratio_max,
        int(record_parts["n_nonzero_contrib"]) >= thresholds.n_nonzero_min,
    )
    return bool(np.all(checks))


@dataclass(frozen=True, slots=True)
class WordRecord:
    """One word's estimate plus the certificate that qualifies it."""

    word: str
    token_ids: tuple[int, ...]
    n_tokens: int
    p_hat: float
    p_hat_no_bos: float
    bits: float
    k_eval: int
    #: Combined draw error: canonical replicate SE and the take-draw FPC SE.
    se_replicate: float
    se_dlt: float
    cv: float
    home_share: float
    bos_share: float
    top1_share: float
    residual_bound: float
    n_nonzero_contrib: int
    n_strata_contributing: int
    underflow_fallback: bool
    #: True when the remainder outweighed the canonical sum, so the estimate is
    #: at or below zero. Distinct from an underflow and never publishable.
    nonpositive: bool
    supported: bool
    #: The word's exact pooled occurrence count and the canonical sub-frame sum
    #: it pairs against. Their difference is the remainder.
    unit_count: int
    h_hat: float
    home_strata: tuple[str, ...]
    bound_share: tuple[float, ...] = ()

    def to_json(self) -> dict[str, object]:
        """A JSON-safe view, the unit the shard files are written in."""
        out: dict[str, object] = {
            "word": self.word,
            "token_ids": list(self.token_ids),
            "home_strata": list(self.home_strata),
            "bound_share": list(self.bound_share),
            "unit_count": self.unit_count,
        }
        for name in _RECORD_SCALARS:
            out[name] = getattr(self, name)
        return out

    @classmethod
    def from_json(cls, obj: Mapping[str, object]) -> "WordRecord":
        """Inverse of :meth:`to_json`.

        Args:
            obj: A decoded ``to_json`` mapping. ``bound_share`` may be absent,
                in which case the record carries an empty one.
        """
        kwargs: dict[str, Any] = {
            "word": str(obj["word"]),
            "token_ids": tuple(int(t) for t in _as_list(obj["token_ids"])),
            "home_strata": tuple(str(s) for s in _as_list(obj["home_strata"])),
            "bound_share": tuple(
                float(v) for v in _as_list(obj.get("bound_share", ()))
            ),
            "unit_count": int(obj["unit_count"]),
        }
        for name, cast in _RECORD_SCALARS.items():
            kwargs[name] = cast(obj[name])
        return cls(**kwargs)


def _as_list(value: Any) -> list:
    """A JSON array field as a list, whatever static type it carries."""
    return list(value)


#: Scalar record fields and how to read them back from JSON.
_RECORD_SCALARS: Final[dict[str, Callable[[Any], Any]]] = {
    "n_tokens": int,
    "p_hat": float,
    "p_hat_no_bos": float,
    "bits": float,
    "k_eval": int,
    "se_replicate": float,
    "se_dlt": float,
    "cv": float,
    "home_share": float,
    "bos_share": float,
    "top1_share": float,
    "residual_bound": float,
    "n_nonzero_contrib": int,
    "n_strata_contributing": int,
    "underflow_fallback": bool,
    "nonpositive": bool,
    "supported": bool,
    "h_hat": float,
}


# --- continuation ----------------------------------------------------------


def _continuations(
    prefixes, toks, model, tokenizer, batch_size, sort_by_prefix_len, *, bset=None
):
    """Per prefix, ``(cont, log_cont, beta, log_beta)`` for ``toks``.

    ``cont`` is the joint of the tokens after the first; ``beta`` is the mass
    on tokens that close the unit after all of them. They are returned apart
    because whether the closing factor applies depends on how much room the
    sample has left, which only the caller knows.
    """
    import torch

    device = next(model.parameters()).device
    n = len(prefixes)
    order = np.arange(n)
    if sort_by_prefix_len and n > 1:
        order = np.argsort([len(p) for p in prefixes], kind="stable")
    ordered = [prefixes[i] for i in order]
    joint = torch.ones(n, dtype=torch.float64)
    log_joint = torch.zeros(n, dtype=torch.float64)
    for t_idx in range(1, len(toks)):
        tid = toks[t_idx]
        head = list(toks[:t_idx])
        step, log_step = [], []
        for k in range(0, n, batch_size):
            seqs = [pre + head for pre in ordered[k : k + batch_size]]
            padded, masks, _ = _left_pad_batch(seqs, tokenizer.pad_token_id)
            with torch.no_grad():
                logits = _padded_logits(model, padded, masks, device)
                last = logits[:, -1, :]
                step.append(torch.softmax(last, dim=-1)[:, tid].cpu().double())
                log_step.append(torch.log_softmax(last, dim=-1)[:, tid].cpu().double())
        joint = joint * torch.cat(step)
        log_joint = log_joint + torch.cat(log_step)
    bset = boundary_set(tokenizer) if bset is None else bset
    ends = torch.tensor(
        bset.ends,
        device=device,
        dtype=torch.long,
    )
    beta, log_beta = [], []
    for k in range(0, n, batch_size):
        seqs = [pre + list(toks) for pre in ordered[k : k + batch_size]]
        padded, masks, _ = _left_pad_batch(seqs, tokenizer.pad_token_id)
        with torch.no_grad():
            last = _padded_logits(model, padded, masks, device)[:, -1, :]
            closed = last.index_select(-1, ends)
            beta.append(torch.softmax(last, dim=-1).index_select(-1, ends).sum(-1))
            log_beta.append(
                torch.logsumexp(closed, dim=-1) - torch.logsumexp(last, dim=-1)
            )
    inverse = np.argsort(order, kind="stable")
    return (
        joint.numpy()[inverse],
        log_joint.numpy()[inverse],
        torch.cat(beta).cpu().double().numpy()[inverse],
        torch.cat(log_beta).cpu().double().numpy()[inverse],
    )


# --- pooled estimator ------------------------------------------------------


def _validate_estimate_call(selection, allocation, k_eval, pool_fingerprint):
    """Refuse a call whose selection, allocation and budget disagree."""
    if selection.estimator_version != SELECTION_VERSION:
        raise ValueError(
            f"selection was built by {selection.estimator_version}, this module "
            f"reads {SELECTION_VERSION} selections"
        )
    if pool_fingerprint is not None and pool_fingerprint != selection.pool_fingerprint:
        raise ValueError("pool fingerprint does not match the selection")
    if selection.strata != allocation.strata:
        raise ValueError("allocation and selection disagree on the strata")
    if k_eval != allocation.k_eval:
        raise ValueError(
            f"k_eval={k_eval} contradicts the allocation's {allocation.k_eval}"
        )
    if allocation.replicates != selection.replicates:
        raise ValueError("allocation and selection disagree on the replicate count")


def _validate_numerator_inputs(
    words, selection, pool_fingerprint, take_store, k_take, cap, splits
) -> None:
    """Refuse inputs that mix the two numerators or leave one half-specified."""
    if splits is not None and cap is None:
        raise ValueError(
            "the residual remainder pairs the canonical sum against an exact "
            "count, so it needs cap to price the same horizon"
        )
    if cap is not None and splits is None:
        raise ValueError(
            "cap without splits prices the canonical sum against the horizon "
            "while the take estimate it is subtracted from ignores it, which "
            "breaks the cancellation the pathwise numerator rests on"
        )
    if splits is not None and take_store is not None:
        raise ValueError(
            "splits and take_store together: the residual remainder replaces the "
            "take channel, so the store would be validated and never read"
        )
    if splits is not None:
        bad = [w for w in dict.fromkeys(words) if min(splits[w]) < 0]
        if bad:
            raise ValueError(f"negative canonical or residual count for {bad[:3]}")
    if take_store is None and splits is None:
        raise ValueError(
            "the sub-frame sum is estimated from the take store; build it with "
            "build_take_store and pass it as take_store="
        )
    if take_store is None:
        return
    if pool_fingerprint is not None and take_store.pool_fingerprint not in (
        "local",
        pool_fingerprint,
    ):
        raise ValueError(
            f"take store is from pool {take_store.pool_fingerprint[:12]}, the "
            f"selection is {pool_fingerprint[:12]}"
        )
    if k_take < 1:
        raise ValueError(f"k_take={k_take}; the take draw needs at least one row")
    absent = np.setdiff1d(selection.first_ids, take_store.first_ids)
    if absent.size:
        raise ValueError(
            f"take store lacks {absent.size} tracked first tokens (e.g. "
            f"{absent[:3].tolist()}); it was built from a different allocation"
        )


def pps_estimate_batch(
    words: Sequence[str],
    selection: PoolSelection,
    model,
    tokenizer,
    *,
    allocation: AllocationSpec,
    k_eval: int,
    batch_size: int = 64,
    sort_by_prefix_len: bool = True,
    emit_per_stratum: bool = False,
    unit_counts: Mapping[str, int],
    take_store: "TakeStore | None" = None,
    k_take: int = 2048,
    thresholds: SupportThresholds | None = None,
    pool_fingerprint: str | None = None,
    progress: "Callable[[WordRecord], None] | None" = None,
    cap: int | None = None,
    splits: Mapping[str, tuple[int, int]] | None = None,
    denom: int | None = None,
) -> dict[str, WordRecord]:
    r"""Estimate p(w) for every word against a pooled selection.

    Every word runs its continuation on the boundaries its allocation draws,
    weighted ``max(size, tau)``, and carries a certificate of what the draw
    covered. A one-token word is no cheaper than any other: its closing mass
    :math:`\\beta_b` depends on the word, so no column sum can stand in for it.

    ``unit_counts`` holds each word's exact occurrence count over the pool.
    With ``take_store`` given, the numerator is the pathwise one: the canonical
    sum and the sub-frame sum it is paired against cancel in expectation, so
    what survives is that count plus mean-zero noise, and the non-canonical
    tokenizations it includes come along without being enumerated.

    ``splits`` holds each word's ``(canonical, residual)`` counts from
    :func:`unit_counts_split` and selects the expectation-equivalent numerator:
    the remainder is the residual count itself, never a subtraction, so the
    numerator is a sum of nonnegative terms and no draw can make it
    non-positive. It needs ``cap``, the sampler's limit on generated tokens:
    without the cancelling channel the continuation has to be priced against
    the room each boundary actually had. Both counts then come off the sampled
    ids, so the record's ``unit_count`` does too.

    Args:
        words: Surfaces to score; duplicates collapse into one estimate.
        selection: Pooled sweep the draws are taken from, carrying the column
            masses, the stored cells, and the boundary prefixes.
        model: Causal language model scoring the continuation passes.
        tokenizer: Tokenizer the pool was encoded with; its unit-opening and
            unit-closing token sets decide which prefixes are eligible.
        allocation: How a word's draws split over strata and replicates.
        k_eval: Draws per word, which has to equal the allocation's own.
            Raising it narrows the interval and costs forward passes.
        batch_size: Prefixes scored per forward pass.
        sort_by_prefix_len: Order a batch's prefixes by length so padding
            stays bounded; results come back in the caller's order either way.
        emit_per_stratum: Record each stratum's share of the first-token
            column mass on the certificate, which is otherwise left empty.
        unit_counts: Exact pooled occurrence count per word. A word with a
            swept column and no entry is refused.
        take_store: Boundaries whose realized successor is a tracked first
            token, the frame the pathwise sub-frame sum is drawn over. One of
            ``take_store`` and ``splits`` is required, never both.
        k_take: Rows drawn per column for that sum; a column holding no more
            than this many takes is covered in full and its sum is exact.
        thresholds: Support bars the certificates are judged against;
            ``None`` takes the defaults.
        pool_fingerprint: Pool the selection has to come from, and the take
            store too unless it is marked ``local``; ``None`` skips the check.
        progress: Called with each finished record, for reporting as the batch
            runs; ``None`` reports nothing.
        cap: The sampler's limit on generated tokens, which the
            expectation-equivalent numerator prices the continuation against.
            Required with ``splits`` and refused without it.
        splits: ``(canonical, residual)`` occurrence counts per word off the
            pool's sampled ids, selecting the expectation-equivalent numerator.
        denom: Normalizer in place of the selection's slot count
            (``n_pool_samples + n_boundaries``); with ``splits`` the pool's unit
            count over the same segmentation as the counts.

    Returns:
        One record per word, keyed by the surface as it was passed, holding
        the estimate and the certificate that qualifies it.

    Raises:
        ValueError: The selection, allocation, budget, take store, or pool
            disagree, the two numerators' inputs are mixed or half-specified,
            or a word tokenizes to nothing.
        KeyError: A word has a swept column but no occurrence count, or a
            first token outside the sweep.
    """
    _validate_estimate_call(selection, allocation, k_eval, pool_fingerprint)
    _validate_numerator_inputs(
        words, selection, pool_fingerprint, take_store, k_take, cap, splits
    )
    bars = thresholds or SupportThresholds()
    bset = boundary_set(tokenizer)
    opens = bset.open_ids
    out: dict[str, WordRecord] = {}
    for word in dict.fromkeys(words):
        toks = tokenizer.encode(word, add_special_tokens=False)
        if not toks:
            raise ValueError(f"{word!r} tokenizes to nothing")
        if int(toks[0]) not in opens:
            # A separator-less surface names the same estimand as its spaced
            # form: the exact count strips the separator before tallying, so
            # both keys anchor to the type-level expected unit count. The bare
            # first token itself has no valid column, because at an interior
            # boundary it would extend the previous unit rather than begin one,
            # so the sampled channel runs through the spaced tokenization.
            resolved = priced_tokens(
                tokenizer, word, bset=bset, columns=selection.first_col
            )
            if int(resolved[0]) in opens:
                toks = list(resolved)
            elif unit_counts.get(word, 0) > 0:
                out[word] = _count_only_record(
                    word, toks, k_eval, unit_counts, denom or selection.denom
                )
                if progress is not None:
                    progress(out[word])
                continue
            else:
                out[word] = _no_channel_record(word, toks, k_eval, unit_counts)
                if progress is not None:
                    progress(out[word])
                continue
        if word not in unit_counts:
            raise KeyError(f"{word!r} has no pooled occurrence count")
        col = selection.first_col.get(int(toks[0]))
        if col is None:
            raise KeyError(f"{word!r} has a first token outside the pooled sweep")
        record = _sampled_record(
            word,
            toks,
            col,
            selection,
            model,
            tokenizer,
            allocation=allocation,
            k_eval=k_eval,
            take_store=take_store,
            k_take=k_take,
            batch_size=batch_size,
            sort_by_prefix_len=sort_by_prefix_len,
            emit_per_stratum=emit_per_stratum,
            bars=bars,
            unit_count=unit_counts[word],
            bset=bset,
            cap=cap,
            split=None if splits is None else tuple(splits[word]),
            denom=denom,
        )
        out[word] = record
        if progress is not None:
            progress(record)
    return out


def _home_names(allocation: AllocationSpec, word: str) -> tuple[str, ...]:
    return tuple(allocation.strata[i] for i, _ in allocation.home_strata(word))


def _bound_share(colsum: np.ndarray) -> np.ndarray:
    total = float(colsum.sum())
    if total <= 0.0:
        return np.zeros_like(colsum, dtype=np.float64)
    return colsum / total


def _cell_take(selection, sl, want):
    """Rows of one stored cell to keep, and the threshold priority."""
    have = sl.stop - sl.start
    if want >= have:
        return np.arange(have), 0.0
    q = _priority(
        selection.sel_p0[sl],
        selection.sel_mult[sl],
        selection.sel_sample[sl],
        selection.sel_pos[sl],
        selection.rng_seed,
    )
    # Taking the leading rows is the top-k only while the cell stays sorted.
    if q.size > 1 and bool((np.diff(q) > 0.0).any()):
        raise ValueError("stored cell is not in descending priority order")
    return np.arange(want), float(q[want])


@dataclass(frozen=True, slots=True)
class TakeStore:
    """Every frame boundary whose realized successor is a tracked first token.

    The take positions are a property of the pool alone, so they are
    enumerated once; a word's sub-frame sum is then estimated by a seeded
    uniform draw over its column, every weight ``n/m`` and every summand in
    ``[0, 1]``, so the estimate concentrates around the pathwise sum.
    """

    first_ids: np.ndarray
    offsets: np.ndarray
    sample: np.ndarray
    pos: np.ndarray
    pool_fingerprint: str
    tokens: "np.ndarray | None" = None
    token_offsets: "np.ndarray | None" = None

    def column(self, first_id: int) -> int:
        """The column index of *first_id*; ``KeyError`` when untracked.

        Args:
            first_id: Token id to locate among the sorted ``first_ids``.
        """
        idx = int(np.searchsorted(self.first_ids, int(first_id)))
        if idx >= self.first_ids.size or int(self.first_ids[idx]) != int(first_id):
            raise KeyError(f"first token {first_id} has no take column")
        return idx

    def count(self, first_id: int) -> int:
        """How many takes the pool holds for *first_id*.

        Args:
            first_id: Tracked token whose column is measured; its takes are
                the frame boundaries that token actually followed.
        """
        col = self.column(first_id)
        return int(self.offsets[col + 1] - self.offsets[col])

    def takes(self, first_id: int) -> "tuple[np.ndarray, np.ndarray]":
        """The column's take rows as ``(sample, position)`` arrays.

        Args:
            first_id: Tracked token whose column is read.
        """
        col = self.column(first_id)
        lo, hi = int(self.offsets[col]), int(self.offsets[col + 1])
        return self.sample[lo:hi], self.pos[lo:hi]

    def with_pool(self, tokens, token_offsets) -> "TakeStore":
        """A copy that reads prefixes from *tokens* and *token_offsets*.

        Args:
            tokens: Flat token ids of every pool sample, in sample order.
            token_offsets: Where each sample starts in *tokens*, with a final
                entry at its end.
        """
        return replace(
            self,
            tokens=np.asarray(tokens),
            token_offsets=np.asarray(token_offsets, dtype=np.int64),
        )

    def prefixes(self, sample: np.ndarray, pos: np.ndarray) -> list[list[int]]:
        """The boundary prefixes of the given take rows, read from the pool.

        Args:
            sample: Pool sample each row sits in.
            pos: Position of the row's last prefix token inside that sample;
                the prefix runs from the sample's start up to and including
                it.
        """
        if self.tokens is None or self.token_offsets is None:
            raise ValueError("attach the pool token store with with_pool first")
        out = []
        for s_i, p_i in zip(sample.tolist(), pos.tolist(), strict=True):
            start = int(self.token_offsets[s_i])
            out.append(self.tokens[start : start + int(p_i) + 1].tolist())
        return out

    def save(self, path: "str | Path") -> None:
        """Write the store as ``.npz`` with narrow row dtypes.

        An attached pool token store is left out, so a store read back needs
        :meth:`with_pool` again before it can serve prefixes.

        Args:
            path: Destination file; numpy appends ``.npz`` when it is absent.
        """
        np.savez(
            path,
            first_ids=self.first_ids,
            offsets=self.offsets,
            sample=self.sample.astype(np.int32),
            pos=self.pos.astype(np.int16),
            pool_fingerprint=np.array(self.pool_fingerprint),
        )

    @classmethod
    def load(cls, path: "str | Path") -> "TakeStore":
        """Keeps the stored narrow dtypes; rows widen only once drawn.

        Args:
            path: The ``.npz`` file to read.
        """
        with np.load(path, allow_pickle=False) as blob:
            return cls(
                first_ids=blob["first_ids"].astype(np.int64),
                offsets=blob["offsets"].astype(np.int64),
                sample=blob["sample"],
                pos=blob["pos"],
                pool_fingerprint=str(blob["pool_fingerprint"]),
            )


def build_take_store(
    sample_index,
    position,
    tokens,
    token_offsets,
    first_ids,
    pool_fingerprint: str = "local",
) -> TakeStore:
    """Scan the frame for boundaries whose successor is a tracked first token.

    Args:
        sample_index: Which pool sample each frame boundary belongs to.
        position: Position of each boundary's last prefix token inside its
            sample, so the token at the next position is its successor.
        tokens: Flat token ids of every pool sample, in sample order.
        token_offsets: Where each sample starts in *tokens*, with a final
            entry at its end.
        first_ids: Tracked first tokens, one column each; duplicates collapse
            and the columns come out in ascending token order.
        pool_fingerprint: Identity of the pool the rows come from, carried on
            the store so an estimate cannot silently mix pools.

    Raises:
        ValueError: A boundary's successor falls outside its sample, which
            means the token store and the boundary index were built from
            different encodings.
    """
    off = np.asarray(token_offsets, dtype=np.int64)
    toks = np.asarray(tokens)
    samp = np.asarray(sample_index, dtype=np.int64)
    pos = np.asarray(position, dtype=np.int64)
    firsts = np.unique(np.asarray(first_ids, dtype=np.int64))
    if samp.size == 0 or firsts.size == 0:
        return TakeStore(
            first_ids=firsts,
            offsets=np.zeros(firsts.size + 1, dtype=np.int64),
            sample=np.zeros(0, dtype=np.int64),
            pos=np.zeros(0, dtype=np.int64),
            pool_fingerprint=pool_fingerprint,
        ).with_pool(toks, off)
    succ_at = off[samp] + pos + 1
    if bool((succ_at >= off[samp + 1]).any()):
        raise ValueError(
            "a frame boundary's successor falls outside its sample; the token "
            "store and the boundary index were built from different encodings"
        )
    nxt = toks[succ_at].astype(np.int64)
    slot = np.clip(np.searchsorted(firsts, nxt), 0, firsts.size - 1)
    mask = firsts[slot] == nxt
    cols = slot[mask]
    order = np.argsort(cols, kind="stable")
    counts = np.bincount(cols, minlength=firsts.size).astype(np.int64)
    offsets = np.zeros(firsts.size + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    return TakeStore(
        first_ids=firsts,
        offsets=offsets,
        sample=samp[mask][order],
        pos=pos[mask][order],
        pool_fingerprint=pool_fingerprint,
    ).with_pool(toks, off)


def take_store_from_shards(
    shards, tokenizer, first_ids, pool_fingerprint="local", start_ids=None
) -> TakeStore:
    """A take store over in-memory sample shards, for tests and small pools.

    Args:
        shards: Chunks of sample text; their concatenation is the pool.
        tokenizer: Tokenizer to encode the samples with; its boundary tokens
            decide where the frame's boundaries fall.
        first_ids: Tracked first tokens, one take column each.
        pool_fingerprint: Identity carried on the store. The default marks a
            pool no published fingerprint covers, which the estimator accepts
            against any selection.
        start_ids: The one-token prefix *shards* were drawn from; ``None`` is
            the tokenizer's convention.
    """
    samples = [s for shard in shards for s in shard]
    encoded = _encode_samples(tokenizer, samples, start_ids)
    flat, offsets = [], [0]
    for ids in encoded:
        flat.append(np.asarray(ids, dtype=np.int64))
        offsets.append(offsets[-1] + len(ids))
    tokens = np.concatenate(flat) if flat else np.zeros(0, np.int64)
    index = boundary_index(
        samples,
        tokenizer,
        encoded=encoded,
        stratification=Stratification("unstratified/1", ("all",), 1, lambda _: "all"),
    )
    return build_take_store(
        index.sample_index,
        index.position,
        tokens,
        np.asarray(offsets, dtype=np.int64),
        first_ids,
        pool_fingerprint,
    )


def _take_indices(rng_seed: int, first_id: int, n: int, m: int) -> np.ndarray:
    """A seeded uniform draw of ``min(m, n)`` take rows, stateless per column."""
    if n <= m:
        return np.arange(n, dtype=np.int64)
    rng = np.random.default_rng([int(rng_seed), int(first_id), 0x7A6B])
    return np.sort(rng.choice(n, size=m, replace=False)).astype(np.int64)


def _take_h(
    take_store,
    toks,
    col,
    selection,
    model,
    tokenizer,
    batch_size,
    sort_by_prefix_len,
    k_take,
    bos_c,
    *,
    bset=None,
):
    """The sub-frame sum from a uniform draw over the identifiable takes.

    Every weight is ``n/m`` and every summand sits in ``[0, 1]``, so the
    estimate concentrates around the pathwise sum, which full coverage
    (``n <= k_take``) reproduces exactly. The bos channel is an exact count
    and needs no draw.
    """
    if selection.first_token_counts is None:
        raise ValueError(
            "the remainder needs first_token_counts; attach the pool token "
            "store before estimating"
        )
    first_id = int(toks[0])
    bos_total = float(selection.first_token_counts[col]) * bos_c
    n = take_store.count(first_id)
    if n == 0:
        return bos_total, 0.0
    idx = _take_indices(selection.rng_seed, first_id, n, int(k_take))
    samp, pos = take_store.takes(first_id)
    prefixes = take_store.prefixes(samp[idx], pos[idx])
    cont, _, beta, _ = _continuations(
        prefixes, toks, model, tokenizer, batch_size, sort_by_prefix_len, bset=bset
    )
    c_vals = np.asarray(cont * beta, dtype=np.float64)
    m_eff = int(idx.size)
    interior = float(n) * float(c_vals.mean())
    se = 0.0
    if 1 < m_eff < n:
        fpc = 1.0 - m_eff / n
        se = float(n) * float(c_vals.std(ddof=1)) / math.sqrt(m_eff) * math.sqrt(fpc)
    return bos_total + interior, se


def _count_only_record(word, toks, k_eval, unit_counts, denom) -> "WordRecord":
    """The plain count estimate ``C/denom``, for a word without a swept column.

    Unbiased for the same estimand with no variance reduction; the draw-side
    certificate fields are non-finite because no draw exists, which is what
    keeps ``certify`` from vouching for it.
    """
    count = int(unit_counts.get(word, 0))
    p_hat = count / float(denom)
    return WordRecord(
        word=word,
        token_ids=tuple(int(t) for t in toks),
        n_tokens=len(toks),
        p_hat=p_hat,
        p_hat_no_bos=p_hat,
        bits=-math.log2(p_hat),
        k_eval=int(k_eval),
        se_replicate=math.nan,
        se_dlt=math.nan,
        cv=math.nan,
        home_share=0.0,
        bos_share=0.0,
        top1_share=0.0,
        residual_bound=math.inf,
        n_nonzero_contrib=0,
        n_strata_contributing=0,
        underflow_fallback=False,
        nonpositive=False,
        supported=False,
        unit_count=count,
        h_hat=0.0,
        home_strata=(),
        bound_share=(),
    )


def _no_channel_record(word, toks, k_eval, unit_counts) -> "WordRecord":
    """A word with neither a swept column nor a pooled occurrence.

    Both channels are empty, so the record carries the floor probability the
    merge accepts and a certificate that refuses it.
    """
    return WordRecord(
        word=word,
        token_ids=tuple(int(t) for t in toks),
        n_tokens=len(toks),
        p_hat=TINY,
        p_hat_no_bos=TINY,
        bits=math.inf,
        k_eval=int(k_eval),
        se_replicate=math.nan,
        se_dlt=math.nan,
        cv=math.nan,
        home_share=0.0,
        bos_share=0.0,
        top1_share=0.0,
        residual_bound=math.inf,
        n_nonzero_contrib=0,
        n_strata_contributing=0,
        underflow_fallback=False,
        nonpositive=True,
        supported=False,
        unit_count=int(unit_counts.get(word, 0)),
        h_hat=0.0,
        home_strata=(),
        bound_share=(),
    )


def _draw_cells(selection, allocation, word, col, k_eval):
    """Priority-sampled boundaries, weights and cell labels for one word."""
    colsum = selection.colsum[:, col]
    first_id = int(selection.first_ids[col])
    mt = selection.mt_col.get(first_id)
    # A cell serves at most k_store-1 draws and still leaves a threshold, so
    # cap the split at what the buffers hold rather than refusing later.
    capacity = None
    if mt is not None:
        store = np.asarray(selection.k_store[mt], dtype=np.int64)
        capacity = (
            allocation.replicates * np.maximum(store.min(axis=1) - 1, 0)
        ).astype(np.int64)
    k = allocation.budget(word, colsum, capacity)
    total = int(k.sum())
    if total > k_eval or total < 0:
        raise ValueError(f"allocation for {word!r} sums to {total}, not {k_eval}")
    per_replicate = allocation.replicate_budget(k)
    if mt is None and total > 0:
        raise KeyError(
            f"{word!r} is multi-token but first token {first_id} has no sweep "
            "column in this selection; estimating it would return the BOS term "
            "alone"
        )
    parts: list[dict[str, np.ndarray]] = []
    covered = np.zeros(len(selection.strata), dtype=np.float64)
    for s in range(len(selection.strata)):
        for r in range(allocation.replicates):
            want = int(per_replicate[s, r])
            if want <= 0 or mt is None:
                continue
            sl = selection.cell(mt, s, r)
            if sl.stop == sl.start:
                continue
            _refuse_shallow_cell(selection, mt, s, r, want, sl, k_eval)
            take, tau = _cell_take(selection, sl, want)
            sizes = (
                selection.sel_p0[sl].astype(np.float64)
                * selection.sel_mult[sl].astype(np.float64)
            )[take]
            covered[s] += float(sizes.sum())
            parts.append(
                {
                    "rows": sl.start + take,
                    "sizes": sizes,
                    "weights": np.maximum(sizes, tau),
                    "strata": np.full(take.size, s, dtype=np.int64),
                    "replicate": np.full(take.size, r, dtype=np.int64),
                    "tau": np.full(take.size, tau, dtype=np.float64),
                }
            )
    return _stack_parts(parts, colsum, covered, total)


def _stack_parts(parts, colsum, covered, total):
    names = ("rows", "sizes", "weights", "strata", "replicate", "tau")
    dtypes = (np.int64, np.float64, np.float64, np.int64, np.int64, np.float64)
    out: dict[str, Any] = {"colsum": colsum, "covered": covered, "k_total": total}
    for name, dtype in zip(names, dtypes, strict=True):
        out[name] = (
            np.concatenate([p[name] for p in parts])
            if parts
            else np.zeros(0, dtype=dtype)
        ).astype(dtype, copy=False)
    return out


def _refuse_shallow_cell(selection, mt, s, r, want, sl, k_eval) -> None:
    """Refuse a draw the stored buffer cannot serve without truncation."""
    have = sl.stop - sl.start
    stored = int(selection.k_store[mt, s, r])
    if want >= have and stored > 0 and have >= stored:
        raise ValueError(
            f"k_eval={k_eval} asks for {want} of the {have} stored boundaries in "
            f"cell (stratum {s}, replicate {r}); re-sweep with a larger k_store"
        )


def _sampled_record(
    word,
    toks,
    col,
    selection,
    model,
    tokenizer,
    *,
    allocation,
    k_eval,
    take_store,
    k_take,
    batch_size,
    sort_by_prefix_len,
    emit_per_stratum,
    bars,
    unit_count,
    bset,
    cap=None,
    split=None,
    denom=None,
) -> WordRecord:
    """Priority-sampled boundaries, the BOS term, and the take-store remainder.

    With ``split`` given as ``(canonical, residual)`` the remainder is the
    residual count itself, so the record's ``h_hat`` holds the exact canonical
    count, ``se_h`` is zero, and nothing is reconstructed by subtraction.
    """
    drawn = _draw_cells(selection, allocation, word, col, k_eval)
    rows = drawn["rows"]
    bos = start_prefix(tokenizer, selection.start)[-1]
    prefixes = [selection.prefix(int(e)) for e in rows] + [[bos]]
    cont, log_cont, beta, log_beta = _continuations(
        prefixes, toks, model, tokenizer, batch_size, sort_by_prefix_len, bset=bset
    )
    c_all, log_c_all = _closed(cont, log_cont, beta, log_beta, prefixes, len(toks), cap)
    c = c_all[: rows.size]
    contrib = drawn["weights"] * c
    bos_size = selection.n_pool_samples * float(selection.bos_p0[col])
    bos_c = float(c_all[-1])
    bos_term = bos_size * bos_c
    canonical = bos_term + float(np.asarray(contrib, dtype=np.float64).sum())
    if split is None:
        h_hat, se_h = _take_h(
            take_store,
            toks,
            col,
            selection,
            model,
            tokenizer,
            batch_size,
            sort_by_prefix_len,
            k_take,
            bos_c,
            bset=bset,
        )
    else:
        h_hat, se_h = float(split[0]), 0.0
        unit_count = int(split[0]) + int(split[1])
    remainder = float(split[1]) if split is not None else unit_count - h_hat
    numerator = canonical + remainder
    log_terms = np.concatenate(
        [
            np.log(np.maximum(drawn["weights"], TINY)) + log_c_all[: rows.size],
            [math.log(max(bos_size, TINY)) + float(log_c_all[-1])],
        ]
    )
    return _build_record(
        word=word,
        toks=toks,
        selection=selection,
        denom=denom,
        allocation=allocation,
        drawn=drawn,
        c=c,
        contrib=contrib,
        bos_term=bos_term,
        numerator=numerator,
        log_terms=log_terms,
        emit_per_stratum=emit_per_stratum,
        bars=bars,
        unit_count=unit_count,
        h_hat=h_hat,
        se_h=se_h,
    )


def _closed(cont, log_cont, beta, log_beta, prefixes, n_toks, cap):
    """``(c, log c)``, closing the unit only where the sample had the room.

    A unit flush against the cap needs no closing token, since the string ends
    there and the count reads that as a boundary; one that cannot finish before
    the cap is not an occurrence the sampler could have produced. Left with
    ``cap=None`` every boundary is priced as if the sample ran on forever, which
    only cancels when the same factor sits in a subtracted channel.
    """
    c = np.asarray(cont, dtype=np.float64) * np.asarray(beta, dtype=np.float64)
    log_c = np.asarray(log_cont, dtype=np.float64) + np.asarray(
        log_beta, dtype=np.float64
    )
    if cap is None:
        return c, log_c
    room = np.array([int(cap) - (len(p) - 1) for p in prefixes], dtype=np.int64)
    flush = room == int(n_toks)
    c = np.where(flush, cont, c)
    log_c = np.where(flush, log_cont, log_c)
    short = room < int(n_toks)
    c = np.where(short, 0.0, c)
    log_c = np.where(short, -np.inf, log_c)
    return c, log_c


def _replicate_totals(contrib, replicate, replicates):
    totals = np.zeros(replicates, dtype=np.float64)
    for r in range(replicates):
        totals[r] = float(np.asarray(contrib[replicate == r], dtype=np.float64).sum())
    return totals


def _dlt_total(drawn, c, replicates):
    """Summed DLT variance over the cells the draw touched."""
    var = 0.0
    for s in np.unique(drawn["strata"]):
        for r in range(replicates):
            mask = (drawn["strata"] == s) & (drawn["replicate"] == r)
            if not bool(mask.any()):
                continue
            var += dlt_variance(
                drawn["sizes"][mask], float(drawn["tau"][mask][0]), c[mask]
            )
    return var


def _build_record(
    *,
    word,
    toks,
    selection,
    allocation,
    drawn,
    c,
    contrib,
    bos_term,
    numerator,
    log_terms,
    emit_per_stratum,
    bars,
    unit_count,
    h_hat,
    se_h,
    denom=None,
) -> WordRecord:
    """Assemble the record, taking the log domain when the numerator vanishes."""
    denom = selection.denom if denom is None else int(denom)
    remainder = float(unit_count) - h_hat
    # A negative numerator means the remainder outweighed the canonical sum.
    # The log domain holds only the canonical terms, so taking it here would
    # silently publish the canonical sum instead of the estimate. A numerator of
    # exactly zero is the underflow the log domain exists to rescue, and its log
    # terms are still finite, so the test is strictly negative.
    nonpositive = not math.isfinite(numerator) or numerator < 0.0
    # The ratio is what has to stay positive: a numerator far above zero can
    # still underflow once it is divided by a denominator of pool size.
    fallback = not nonpositive and numerator / denom <= 0.0 and remainder >= 0.0
    if nonpositive:
        p_hat, bits = TINY, math.inf
    elif fallback:
        terms = log_terms
        if remainder > 0.0:
            terms = np.concatenate([log_terms, [math.log(remainder)]])
        log_num = _logsumexp(terms)
        bits = -(log_num - math.log(denom)) / math.log(2.0)
        p_hat = max(math.exp(log_num - math.log(denom)), TINY)
    else:
        p_hat = max(numerator / denom, TINY)
        bits = -math.log2(p_hat)
    scale = numerator if numerator > 0 else 1.0
    home_idx = [i for i, _ in allocation.home_strata(word)]
    home_mask = np.isin(drawn["strata"], home_idx)
    # The canonical sum and the sub-frame sum come from independent draws, the
    # priority selection and the uniform take draw, so their errors add in
    # quadrature: the replicate SE of the canonical terms and the finite-
    # population SE of the take estimate.
    se_canonical = design_se(
        _replicate_totals(
            np.asarray(contrib, dtype=np.float64),
            drawn["replicate"],
            allocation.replicates,
        )
    )
    se = math.hypot(se_canonical, float(se_h)) / denom
    se_dlt = math.sqrt(_dlt_total(drawn, c, allocation.replicates)) / denom
    cv = max(se, se_dlt) / p_hat if p_hat > 0 else math.inf
    # colsum and covered are first-token mass, p_hat is whole-word mass, so the
    # unvisited mass is carried through the continuation before the two compare.
    # The factor is the mass-weighted mean continuation seen in each stratum.
    unvisited = np.maximum(drawn["colsum"] - drawn["covered"], 0.0)
    # An empty draw makes bincount return integers, which the divide cannot
    # write back into, so both sides are pinned to float64.
    seen_first = np.bincount(
        drawn["strata"], weights=drawn["sizes"], minlength=unvisited.size
    ).astype(np.float64)
    seen_word = np.bincount(
        drawn["strata"], weights=drawn["sizes"] * c, minlength=unvisited.size
    ).astype(np.float64)
    mean_cont = np.divide(
        seen_word, seen_first, out=np.zeros_like(seen_word), where=seen_first > 0.0
    )
    residual = float((unvisited * mean_cont[: unvisited.size]).sum()) / denom
    parts = {
        "p_hat": p_hat,
        "cv": cv,
        "home_share": float(contrib[home_mask].sum()) / scale,
        "bos_share": bos_term / scale,
        "top1_share": float(contrib.max()) / scale if contrib.size else 0.0,
        "residual_bound": residual,
        "n_nonzero_contrib": int((contrib > 0.0).sum()),
        "underflow_fallback": fallback,
        "nonpositive": nonpositive,
    }
    return WordRecord(
        word=word,
        token_ids=tuple(int(t) for t in toks),
        n_tokens=len(toks),
        p_hat=p_hat,
        p_hat_no_bos=max(numerator - bos_term, 0.0) / denom,
        bits=bits,
        k_eval=int(drawn["k_total"]),
        se_replicate=se,
        se_dlt=se_dlt,
        cv=cv,
        home_share=float(parts["home_share"]),
        bos_share=float(parts["bos_share"]),
        top1_share=float(parts["top1_share"]),
        residual_bound=residual,
        n_nonzero_contrib=int(parts["n_nonzero_contrib"]),
        n_strata_contributing=int(np.unique(drawn["strata"][contrib > 0.0]).size),
        underflow_fallback=fallback,
        nonpositive=nonpositive,
        supported=certify(parts, bars),
        unit_count=unit_count,
        h_hat=h_hat,
        home_strata=_home_names(allocation, word),
        bound_share=tuple(_bound_share(drawn["colsum"])) if emit_per_stratum else (),
    )


def _logsumexp(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return -math.inf
    top = float(finite.max())
    return top + math.log(float(np.exp(finite - top).sum()))


# --- exhaustive regression path --------------------------------------------


def _check_exhaustive_numerator(numerator: str, cap, encoded) -> None:
    if numerator not in ("pathwise", "expectation"):
        raise ValueError(f"unknown numerator {numerator!r}")
    if numerator != "expectation":
        return
    if cap is None:
        raise ValueError(
            "the expectation numerator pairs the canonical channel against "
            "an exact count, so it needs cap to price the same horizon"
        )
    if encoded is None:
        raise ValueError(
            "the expectation numerator needs the pool's sampled ids: "
            "re-encoding reports no residual mass"
        )


def _frame_prefixes(rows, bset):
    """Every interior frame boundary's prefix, its successor, each sample's
    opener, and the pool's boundary count, as the definition walks them.
    """
    prefixes: list[list[int]] = []
    successors: list[int] = []
    openers: list[int] = []
    n_boundaries = 0
    for ids in rows:
        openers.append(int(ids[1]) if len(ids) > 1 else -1)
        for p in range(len(ids)):
            if not bset.opens_unit(ids, p):
                continue
            n_boundaries += 1
            if p == 0 or p == len(ids) - 1:
                continue
            prefixes.append(list(ids[: p + 1]))
            successors.append(int(ids[p + 1]))
    return prefixes, successors, openers, n_boundaries


def _last_rows(seqs, model, tokenizer, device, batch_size):
    """Next-token distributions after each sequence, batched, float64."""
    import torch

    rows = []
    for k in range(0, len(seqs), batch_size):
        padded, masks, _ = _left_pad_batch(
            seqs[k : k + batch_size], tokenizer.pad_token_id
        )
        with torch.no_grad():
            rows.append(
                torch.softmax(
                    _padded_logits(model, padded, masks, device)[:, -1, :], dim=-1
                ).double()
            )
    return torch.cat(rows) if rows else torch.zeros((0, 0), dtype=torch.float64)


def exhaustive_estimate_batch(
    words: Sequence[str],
    samples: Sequence[str],
    model,
    tokenizer,
    batch_size: int = 64,
    *,
    cap: int | None = None,
    numerator: str = "pathwise",
    encoded: "Sequence[Sequence[int]] | None" = None,
    start_ids=None,
) -> dict[str, float]:
    r"""Every boundary summed directly, as the definition reads.

    Deliberately independent of the sampled path: it walks the samples itself
    and scores prefixes one word at a time, so it can stand as the oracle that
    path is checked against.

    ``cap`` is the sampler's limit on generated tokens. Given, the continuation
    is priced against the room a boundary actually has: a unit that cannot
    finish before the cap contributes nothing, and one that ends flush against
    it needs no boundary token, since the string ends there. Left ``None`` the
    continuation is priced as if every boundary had unlimited room, which is
    what the two sampled channels of the ``pathwise`` numerator cancel.

    ``numerator`` picks the decomposition: ``pathwise`` is
    :math:`\widehat S + C - \widehat H`, exact per string; ``expectation`` is
    :math:`\widehat S + C^{\mathrm{res}}`, which drops the cancelling channel
    and is therefore nonnegative, and needs ``cap`` and ``encoded`` to be right.

    ``encoded`` supplies the pool's sampled token ids. Without it the samples
    are re-encoded, which returns the canonical tokenization of every unit and
    reports no residual mass.

    Args:
        words: Surfaces to score; one that tokenizes to nothing gets zero.
        samples: The pool as text, every boundary of which is summed.
        model: Causal language model scoring each prefix.
        tokenizer: Tokenizer whose unit-opening and unit-closing token sets
            decide which positions are boundaries and what closes a unit.
        batch_size: Prefixes scored per forward pass.
        cap: The sampler's limit on generated tokens; ``None`` prices every
            boundary as though it had unlimited room.
        numerator: ``pathwise`` or ``expectation``, the decomposition summed.
        encoded: The pool's sampled token ids, bos first, one row per sample;
            ``None`` re-encodes ``samples``.
        start_ids: The one-token prefix the pool was drawn from; ``None`` is
            the tokenizer's convention.

    Returns:
        p(w) per word, over a denominator of samples plus boundaries.

    Raises:
        ValueError: An unknown numerator, or the expectation numerator without
            ``cap`` or ``encoded``.
    """
    import torch

    _check_exhaustive_numerator(numerator, cap, encoded)
    device = next(model.parameters()).device
    bset = boundary_set(tokenizer)
    opens_t = torch.tensor(bset.opens, device=device, dtype=torch.long)
    ends_t = torch.tensor(bset.ends, device=device, dtype=torch.long)
    bos = start_prefix(tokenizer, start_ids)[-1]

    rows = (
        _encode_samples(tokenizer, samples, start_ids)
        if encoded is None
        else [list(r) for r in encoded]
    )
    prefixes, successors, openers, n_boundaries = _frame_prefixes(rows, bset)
    denom = len(samples) + n_boundaries
    if denom == 0:
        return {w: 0.0 for w in words}

    # The empty prefix is in the frame for every sample whatever followed it,
    # so unlike an interior boundary it is not a size-biased draw: no 1/W, and
    # a weight of N.
    prefixes.append([bos])
    weights = torch.ones(len(prefixes), dtype=torch.float64)
    weights[-1] = float(len(samples))
    heads = _last_rows(prefixes, model, tokenizer, device, batch_size)
    opened = heads.index_select(-1, opens_t).sum(-1)
    opened[-1] = 1.0

    counts = unit_counts(words, samples)
    split = (
        unit_counts_split(words, rows, tokenizer, bset=bset)
        if numerator == "expectation"
        else {}
    )
    succ = np.array([*successors, -1], dtype=np.int64)
    open_tally = np.array(openers, dtype=np.int64)
    # Generated tokens already spent, so cap minus this is the room a unit has.
    spent = torch.tensor([len(pre) - 1 for pre in prefixes], dtype=torch.int64)

    out: dict[str, float] = {}
    for word in words:
        toks = tokenizer.encode(word, add_special_tokens=False)
        if not toks:
            out[word] = 0.0
            continue
        head = heads[:, toks[0]] / opened.clamp(min=TINY)
        cont = torch.ones(len(prefixes), dtype=torch.float64)
        for i in range(1, len(toks)):
            step = _last_rows(
                [pre + list(toks[:i]) for pre in prefixes],
                model,
                tokenizer,
                device,
                batch_size,
            )
            cont = cont * step[:, toks[i]]
        closing = _last_rows(
            [pre + list(toks) for pre in prefixes], model, tokenizer, device, batch_size
        )
        closed = cont * closing.index_select(-1, ends_t).sum(-1)
        if cap is None:
            cont = closed
        else:
            room = cap - spent
            # Flush against the cap the string ends, which closes the unit; past
            # it the unit cannot be realized at all.
            cont = torch.where(room == len(toks), cont, closed)
            cont = torch.where(room < len(toks), torch.zeros_like(cont), cont)
        canonical = float((head * cont * weights).sum().item())
        if numerator == "expectation":
            out[word] = (canonical + split[word][1]) / denom
            continue
        # The sub-frame the remainder takes back: boundaries whose successor was
        # in fact the word's first token, plus the samples that open with it.
        taken = np.asarray(cont.numpy())[succ == int(toks[0])].sum()
        taken += float(cont[-1].item()) * float((open_tally == int(toks[0])).sum())
        out[word] = (canonical + counts[word] - float(taken)) / denom
    return out


# --- FrequencySource -------------------------------------------------------


class PooledUnigramSource(EstimatedUnigramSource):
    """:class:`FrequencySource` serving :func:`pps_estimate_batch` estimates.

    Takes a finished :class:`PoolSelection` rather than a sample list, so it
    cannot be pointed at a generated pool by accident. ``probability`` returns
    p(w), and NaN for a zero, matching the other frequency sources.
    """

    def __init__(
        self,
        *,
        selection: PoolSelection,
        allocation: AllocationSpec,
        k_eval: int,
        model_name: str = "ai-forever/mGPT",
        language: str = "en",
        device: str = "cpu",
        batch_size: int = 64,
        thresholds: SupportThresholds | None = None,
        pool_fingerprint: str | None = None,
        unit_counts: Mapping[str, int] | None = None,
        samples: Iterable[str] | None = None,
        context_window: int | None = None,
    ):
        """Record the pool and scoring settings; the model loads on first use.

        Args:
            selection: Finished pooled sweep every estimate is scored against.
            allocation: How a word's ``k_eval`` draws split over strata and
                replicates.
            k_eval: Draws per word, which has to equal the allocation's own.
            model_name: Causal language model loaded for the continuation
                passes.
            language: Stored on the source for the FrequencySource protocol.
            device: Torch device the model scores on.
            batch_size: Prefixes scored per forward pass.
            thresholds: Support bars the certificates are judged against;
                ``None`` takes the defaults.
            pool_fingerprint: Pool the selection has to come from; ``None``
                skips the check.
            unit_counts: Exact pooled occurrence count per word; estimating
                without it raises rather than dropping the count channel.
            samples: Refused, so a caller meaning to sample a fresh pool gets
                an error instead of a different estimand.
            context_window: Unsupported: the selection fixes the sampled contexts.

        Raises:
            ValueError: Unsupported context window, supplied samples, or missing selection.
        """
        if context_window is not None:
            raise ValueError(
                "PooledUnigramSource does not support context_window: "
                "the selection fixes the sampled contexts"
            )
        if samples is not None:
            raise ValueError(
                "a pooled source scores against a PoolSelection; passing samples "
                "would estimate from a different pool"
            )
        if selection is None:
            raise ValueError("a pooled source needs a PoolSelection")
        self.language = language
        self.case_sensitive = True
        self.model_name = model_name
        self.device = device
        self.batch_size = int(batch_size)
        self.k_eval = int(k_eval)
        self.prob_floor = 0.0
        self.max_continuation_boundaries: int | None = None
        self._selection = selection
        self._allocation = allocation
        self._thresholds = thresholds or SupportThresholds()
        self._pool_fingerprint = pool_fingerprint
        self._unit_counts = unit_counts
        self._model: Any = None
        self._tokenizer: Any = None
        self._start_checked = False
        self._cache: dict[str, float] = {}
        self._records: dict[str, WordRecord] = {}

    def _ensure(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self._tokenizer.pad_token_id is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
        if not self._start_checked:
            default = resolve_start_state(self._tokenizer, self.model_name)
            prefix = start_prefix(self._tokenizer, self._selection.start)
            if default.basis == "override" and tuple(prefix) != default.ids:
                warnings.warn(
                    f"{self.model_name}: pool selection uses start prefix {prefix}, "
                    f"overriding REPLICA's default {list(default.ids)} "
                    f"({', '.join(default.tokens)}). The pool's prefix controls "
                    "these estimates.",
                    UserWarning,
                    stacklevel=2,
                )
            self._start_checked = True
        if self._model is None:
            self._model = AutoModelForCausalLM.from_pretrained(self.model_name)
            self._model.to(self.device).eval()

    def estimate_words(self, words: Sequence[str]) -> dict[str, float]:
        """Batched estimates; the certificates land on ``self._records``.

        Args:
            words: Surface forms to score; duplicates collapse into one
                estimate and words already cached are skipped.

        Returns:
            p(word) keyed by every entry of *words*.
        """
        if self._unit_counts is None:
            raise ValueError(
                "the pooled source needs unit_counts: the estimator pairs each "
                "word's canonical sum against its exact pooled occurrence count"
            )
        todo = [w for w in dict.fromkeys(words) if w not in self._cache]
        if todo:
            self._ensure()
            records = pps_estimate_batch(
                todo,
                self._selection,
                self._model,
                self._tokenizer,
                allocation=self._allocation,
                k_eval=self.k_eval,
                batch_size=self.batch_size,
                thresholds=self._thresholds,
                unit_counts=self._unit_counts,
                pool_fingerprint=self._pool_fingerprint,
            )
            self._records.update(records)
            self._cache.update({w: r.p_hat for w, r in records.items()})
        return {w: self._cache[w] for w in words}

    def estimate(self, word: str) -> float:
        """Estimated unigram probability p(*word*).

        Args:
            word: Surface form to score, as the tokenizer sees it; the
                estimate is cached under this exact string.
        """
        return self.estimate_words([word])[word]
