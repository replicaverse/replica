"""Label the pool's word boundaries and enumerate the realized takes."""

import argparse

import numpy as np

from replica.feature.frequency import pooled as pooled_mod

from . import common
from .common import _done, _write_npz
from .pool import _load_manifest, _shard_encoded, _shard_samples
from .profile import STRATIFICATION


def cmd_index(args: argparse.Namespace) -> int:
    """Tokenize one sample shard and label its boundaries.

    Args:
        args: Parsed `index` arguments: the `pool_manifest`, an optional
            `pool_dir` token store (without it the text is re-encoded), the
            `shard_index` to label, `dup_collapse_max_len` for prefix
            collapsing, and the `out` path for the index.
    """
    if _done(args.out, "index"):
        return 0
    manifest = _load_manifest(args.pool_manifest)
    samples, offset = _shard_samples(manifest, args.shard_index)
    tok = common._tokenizer(manifest["model_name"])
    index = pooled_mod.boundary_index(
        samples,
        tok,
        stratification=STRATIFICATION,
        sample_offset=offset,
        dup_collapse_max_len=args.dup_collapse_max_len,
        encoded=_shard_encoded(args.pool_dir, manifest, args.shard_index),
    )
    _write_npz(args.out, index.save)
    print(
        f"[index {args.shard_index}] {index.n_samples} samples, "
        f"{index.n_boundaries} boundaries, frame {index.frame_size()}",
        flush=True,
    )
    return 0


def cmd_index_merge(args: argparse.Namespace) -> int:
    """Tile the shard indices, collapse duplicate prefixes, split them back.

    Args:
        args: Parsed `index-merge` arguments: the index `parts` to tile, the
            `expected_shards` count they must cover, `dup_collapse_max_len`
            for prefix collapsing, and the `out` path for the merged index.
    """
    if _done(args.out, "index-merge"):
        return 0
    parts = [
        pooled_mod.BoundaryIndex.load(p, stratification=STRATIFICATION)
        for p in sorted(args.parts)
    ]
    merged = pooled_mod.merge_index(
        parts,
        expected_shards=args.expected_shards,
        dup_collapse_max_len=args.dup_collapse_max_len,
    )
    frame_dir = args.out.parent
    running = 0
    for shard, part in enumerate(sorted(parts, key=lambda p: p.sample_offset)):
        lo, hi = running, running + part.n_samples
        running = hi
        piece = merged.shard_view(lo, hi, part.boundary_counts, part.n_boundaries)
        _write_npz(frame_dir / f"frame_{shard:04d}.npz", piece.save)
    _write_npz(args.out, merged.save)
    print(
        f"[index-merge] {merged.n_samples} samples, {merged.n_boundaries} "
        f"boundaries, frame {merged.frame_size()} after collapse",
        flush=True,
    )
    return 0


def cmd_takes(args: argparse.Namespace) -> int:
    """Enumerate the frame boundaries whose successor opens a tracked word.

    The take positions are a property of the pool and the frame alone, so one
    CPU pass writes them; every shard then estimates its sub-frame sums by a
    seeded uniform draw over them.

    Args:
        args: Parsed `takes` arguments: `model_name`, the `pool_dir` token
            store, the boundary `index`, the plan's `allocation`, and the
            `out` path for the take store.
    """
    if _done(args.out, "takes"):
        return 0
    if args.out.suffix != ".npz":
        raise SystemExit(
            "[takes] --out must end in .npz; np.savez appends the suffix and "
            "the done-check would miss the written file"
        )
    manifest = _load_manifest(args.pool_dir / "manifest.json")
    index = pooled_mod.BoundaryIndex.load(args.index, stratification=STRATIFICATION)
    if index.multiplicity.size and int(index.multiplicity.max()) != 1:
        raise SystemExit(
            "[takes] the merged index collapses duplicate prefixes; the take "
            "scan needs one row per boundary"
        )
    with np.load(args.allocation, allow_pickle=False) as blob:
        first_ids = blob["first_ids"].astype(np.int64)
    tokens = np.fromfile(args.pool_dir / "tokens.i32", dtype=np.int32)
    offsets = np.fromfile(args.pool_dir / "tokens.off.i64", dtype=np.int64)
    if index.n_samples != offsets.size - 1:
        raise SystemExit(
            f"[takes] the index covers {index.n_samples} samples, the pool "
            f"stores {offsets.size - 1}"
        )
    open_ids = pooled_mod.boundary_set(common._tokenizer(args.model_name)).open_ids
    store = pooled_mod.build_take_store(
        index.sample_index,
        index.position,
        tokens,
        offsets,
        first_ids,
        manifest["pool_fingerprint"],
    )
    _check_take_counts(store, tokens, offsets, first_ids, open_ids)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    store.save(args.out)
    print(
        f"[takes] {store.sample.size} takes over {first_ids.size} columns "
        f"fingerprint={store.pool_fingerprint[:12]}",
        flush=True,
    )
    return 0


def _check_take_counts(store, tokens, offsets, first_ids, open_ids) -> None:
    """Cross-check the scan against a direct tally of the token store.

    A unit-opening first token sits behind exactly one frame boundary at every
    interior occurrence, so its column count must equal the token's occurrences
    past each sample's opener slot. A first token that does not open a unit has
    no frame boundary with that successor at all, so its column must be empty
    and its sub-frame sum is the exact bos channel alone.
    """
    interior = np.ones(tokens.size, dtype=bool)
    interior[offsets[:-1]] = False  # the bos slot
    starts = offsets[:-1] + 1
    interior[starts[starts < tokens.size]] = False  # the opener, the bos channel
    tally = np.bincount(tokens[interior], minlength=int(first_ids.max()) + 1)
    empties = 0
    for fid in first_ids.tolist():
        expected = int(tally[fid]) if fid in open_ids else 0
        if fid not in open_ids:
            empties += 1
        got = store.count(int(fid))
        if got != expected:
            raise SystemExit(
                f"[takes] column {fid} holds {got} takes but the pool holds "
                f"{expected} identifiable ones; the frame and the store disagree"
            )
    print(f"[takes] {empties} non-opening columns verified empty", flush=True)
