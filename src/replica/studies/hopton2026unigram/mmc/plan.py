"""Collect the union of surfaces and split it into equal-cost shards."""

import argparse
import json
from pathlib import Path

import numpy as np

from replica.feature.frequency import pooled as pooled_mod

from . import common, profile
from .common import (
    _POOLED,
    _done,
    _surfaces_for_dataset,
    _write_npz,
    _write_text,
)


def cmd_words(args: argparse.Namespace) -> int:
    """Collect the union of surfaces across datasets, with their home sets.

    Args:
        args: Parsed `words` arguments: the `source` whose lookup segments
            each stimulus, the `datasets` to scan, the `out` path for the
            union, and an optional `homes_out` path for the per-surface
            dataset membership.
    """
    if _done(args.out, "words"):
        return 0
    homes: dict[str, list[str]] = {}
    for name in args.datasets:
        surfaces = _surfaces_for_dataset(name, args.source)
        for surface in surfaces:
            homes.setdefault(surface, []).append(name)
        print(f"  {name:<30} {len(surfaces):>7} surfaces", flush=True)
    ordered = sorted(homes)
    _write_text(args.out, json.dumps(ordered, ensure_ascii=False))
    if args.homes_out is not None:
        _write_text(args.homes_out, json.dumps(homes, ensure_ascii=False))
    print(f"[words] union {len(ordered)} -> {args.out}", flush=True)
    return 0


def cmd_plan(args: argparse.Namespace) -> int:
    """Split the union into equal-cost slices and size the sweep buffers.

    Cost is ``max(k-1, 0)`` per word for the exhaustive source and
    ``max(k-1, 0) * k_eval`` for the pooled one: a single-subword surface is a
    column lookup, each extra subword a pass over the drawn boundaries.

    Args:
        args: Parsed `plan` arguments: `source`, `model_name`, the `words`
            union and optional `homes`, the `shards` count to split into, the
            `out` bounds path, an optional `alloc_out` allocation path, and
            the allocation sizing knobs `k_eval`, `k_floor`, `k_home`,
            `replicates`, `escalation`, `expected_shards`, and `shard_slack`.
    """
    if args.out.exists() and (args.alloc_out is None or args.alloc_out.exists()):
        print(f"[plan] {args.out} exists, skipping", flush=True)
        return 0
    words = json.loads(args.words.read_text(encoding="utf-8"))
    tok = common._tokenizer(args.model_name)
    # The priced tokenization, so a separator-less surface's spaced first token
    # gets a column: its bare first token cannot open a unit and has no channel.
    bset = pooled_mod.boundary_set(tok)
    token_ids = [list(pooled_mod.priced_tokens(tok, w, bset=bset)) for w in words]
    unit = args.k_eval if args.source in _POOLED else 1
    weights = [max(len(ids) - 1, 0) * unit for ids in token_ids]
    bounds = _equal_cost_bounds(weights, args.shards)
    lines = "".join(f"{start}\t{end}\n" for start, end, _ in bounds)
    _write_text(args.out, lines)
    total = int(np.array(weights, dtype=np.int64).sum())
    print(
        f"[plan] {len(words)} words, cost {total}, {len(bounds)} shards",
        flush=True,
    )
    if args.alloc_out is not None:
        _write_allocation(args, words, token_ids)
    return 0


def _equal_cost_bounds(weights, shards):
    total = int(np.array(weights, dtype=np.int64).sum())
    per_shard = max(1, -(-total // shards))
    bounds: list[tuple[int, int, int]] = []
    start, running = 0, 0
    for index, weight in enumerate(weights):
        running += weight
        if running >= per_shard and len(bounds) < shards - 1:
            bounds.append((start, index + 1, running))
            start, running = index + 1, 0
    bounds.append((start, len(weights), running))
    return bounds


def _write_allocation(args, words, token_ids) -> None:
    """Freeze the per-word budget inputs and the sweep buffer sizes."""
    homes = json.loads(args.homes.read_text(encoding="utf-8")) if args.homes else {}
    allocation = profile.allocation_spec(
        k_eval=args.k_eval,
        k_floor=args.k_floor,
        k_home=args.k_home,
        replicates=args.replicates,
        homes={w: tuple(homes.get(w, ())) for w in words},
    )
    first_ids = np.array(sorted({t[0] for t in token_ids if t}), dtype=np.int32)
    # Every word needs stored boundaries, single-token words included:
    # mmc_pps/2 charges the closing mass beta, which is a forward pass on each
    # boundary's own prefix and so cannot be folded into a column sum.
    mt_ids = first_ids.copy()
    mt_homes = _first_id_homes(words, token_ids, mt_ids, allocation)
    per_stratum = profile.max_cell_budget(allocation, mt_homes)
    k_store = np.repeat(
        (per_stratum * args.escalation)[:, :, None], args.replicates, axis=2
    )
    k_store = np.maximum(np.ceil(k_store / args.replicates).astype(np.int64), 1)
    k_shard = np.maximum(
        np.ceil(k_store / max(args.expected_shards, 1)).astype(np.int64)
        + args.shard_slack,
        1,
    )
    k_shard = np.minimum(k_shard, k_store)
    params = {
        "k_eval": args.k_eval,
        "k_floor": args.k_floor,
        "k_home": args.k_home,
        "replicates": args.replicates,
        "escalation": args.escalation,
        "alloc_sha": allocation.sha(),
        "stratification_id": profile.STRATIFICATION.identity,
    }
    _write_npz(
        args.alloc_out,
        lambda tmp: np.savez(
            tmp,
            first_ids=first_ids,
            mt_first_ids=mt_ids,
            word_ntok=np.array([len(t) for t in token_ids], dtype=np.uint8),
            k_store=k_store,
            k_shard=k_shard,
            params=np.array(json.dumps(params)),
        ),
    )
    print(f"[plan] allocation {params} -> {args.alloc_out}", flush=True)


def _first_id_homes(words, token_ids, mt_ids, allocation):
    """Union of home strata over the words sharing each multi-token first id."""
    lookup = {int(t): i for i, t in enumerate(mt_ids)}
    out: dict[int, set[int]] = {int(t): set() for t in mt_ids}
    for word, ids in zip(words, token_ids, strict=True):
        if len(ids) < 2:
            continue
        row = lookup.get(int(ids[0]))
        if row is None:
            continue
        out[int(ids[0])].update(i for i, _ in allocation.home_strata(word))
    return {t: frozenset(v) for t, v in out.items()}


def _load_allocation(path: Path, homes_path: Path | None, words):
    """Rebuild the allocation spec and buffer sizes written by ``plan``."""
    with np.load(path, allow_pickle=False) as blob:
        params = json.loads(str(blob["params"]))
        arrays = {
            k: blob[k] for k in ("first_ids", "mt_first_ids", "k_store", "k_shard")
        }
    homes = json.loads(homes_path.read_text(encoding="utf-8")) if homes_path else {}
    spec = profile.allocation_spec(
        k_eval=params["k_eval"],
        k_floor=params["k_floor"],
        k_home=params["k_home"],
        replicates=params["replicates"],
        homes={w: tuple(homes.get(w, ())) for w in words},
    )
    if "stratification_id" in params:
        if params["stratification_id"] != profile.STRATIFICATION.identity:
            raise SystemExit(
                "[alloc] plan stratification does not match the study profile"
            )
        digest = spec.sha()
    else:
        digest = profile.legacy_allocation_sha(
            spec, {w: tuple(homes.get(w, ())) for w in words}
        )
    if digest != params["alloc_sha"]:
        raise SystemExit(
            "[alloc] the rebuilt allocation does not match the recorded digest; "
            "the words or homes files differ from the ones plan saw."
        )
    return spec, arrays, params
