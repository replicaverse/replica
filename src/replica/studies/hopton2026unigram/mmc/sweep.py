"""Score boundaries against the first-token columns and fold the selection."""

import argparse

import numpy as np

from replica.feature.frequency import pooled as pooled_mod

from . import common
from .common import _done, _write_npz
from .pool import _load_manifest, _shard_encoded, _shard_samples, _store_arrays_problem
from .profile import STRATIFICATION


def cmd_sweep(args: argparse.Namespace) -> int:
    """Score one shard's boundaries against the pooled first-token columns.

    Args:
        args: Parsed `sweep` arguments: the `pool_manifest` and `pool_dir`
            token store, the boundary `index`, the plan's `alloc`, the
            `shard_index` to score, the `rng_seed` for the boundary draw,
            `batch_size` and `reduce_chunk` for the forward pass, the
            `device` to run on, and the `out` path for the partial sweep.
    """
    if _done(args.out, "sweep"):
        return 0
    manifest = _load_manifest(args.pool_manifest)
    samples, _ = _shard_samples(manifest, args.shard_index)
    index = pooled_mod.BoundaryIndex.load(args.index, stratification=STRATIFICATION)
    with np.load(args.alloc, allow_pickle=False) as blob:
        first_ids = blob["first_ids"]
        mt_first_ids = blob["mt_first_ids"]
        k_store = blob["k_store"]
        k_shard = blob["k_shard"]
    model, tok = _load_model(manifest["model_name"], args.device)
    partial = pooled_mod.sweep_shard(
        samples,
        index,
        model,
        tok,
        first_ids=first_ids,
        mt_first_ids=mt_first_ids,
        k_store=k_store,
        k_shard=k_shard,
        rng_seed=args.rng_seed,
        shard_index=args.shard_index,
        batch_size=args.batch_size,
        reduce_chunk=args.reduce_chunk,
        encoded=_shard_encoded(args.pool_dir, manifest, args.shard_index),
        store_device=args.store_device,
    )
    if partial.n_boundaries != index.n_boundaries:
        raise SystemExit(
            f"[sweep {args.shard_index}] boundary count {partial.n_boundaries} "
            f"differs from the index count {index.n_boundaries}"
        )
    _write_npz(args.out, partial.save)
    print(f"[sweep {args.shard_index}] {partial.n_samples} samples", flush=True)
    return 0


def _load_model(model_name: str, device: str):
    import torch
    from transformers import AutoModelForCausalLM

    tok = common._tokenizer(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)
    model.to(device).eval()
    seen = next(model.parameters()).dtype
    if seen is not torch.float32:
        raise SystemExit(f"{model_name} loaded in {seen}, not float32")
    return model, tok


def cmd_sweep_merge(args: argparse.Namespace) -> int:
    """Fold the shard sweeps into one pooled selection.

    Args:
        args: Parsed `sweep-merge` arguments: the shard `partials` to fold in,
            the `expected_shards` count they must cover, the `pool_manifest`
            and `pool_dir` token store, the boundary `index` and plan's
            `alloc`, the `rng_seed` and `wave` size for the selection draw,
            `delete_on_consume` to remove partials once folded, the `device`
            to run on, and the `out` path for the selection.
    """
    if _done(args.out, "sweep-merge"):
        return 0
    manifest = _load_manifest(args.pool_manifest)
    index = pooled_mod.BoundaryIndex.load(args.index, stratification=STRATIFICATION)
    with np.load(args.alloc, allow_pickle=False) as blob:
        first_ids = blob["first_ids"]
        mt_first_ids = blob["mt_first_ids"]
        k_store = blob["k_store"]
    model, tok = _load_model(manifest["model_name"], args.device)
    start = _pool_start(manifest, tok)
    selection = pooled_mod.merge_sweep(
        sorted(args.partials),
        expected_shards=args.expected_shards,
        n_pool_samples=index.n_samples,
        n_boundaries=index.n_boundaries,
        bos_p0=pooled_mod.bos_first_probs(model, tok, first_ids, start),
        k_store=k_store,
        first_ids=first_ids,
        mt_first_ids=mt_first_ids,
        pool_fingerprint=manifest["pool_fingerprint"],
        rng_seed=args.rng_seed,
        wave=args.wave,
        delete_on_consume=args.delete_on_consume,
        stratification=STRATIFICATION,
        start=start,
    )
    _check_mass_identity(selection, index)
    tokens = np.fromfile(args.pool_dir / "tokens.i32", dtype=np.int32)
    offsets = np.fromfile(args.pool_dir / "tokens.off.i64", dtype=np.int64)
    _check_store_start(tokens, offsets, start)
    selection = pooled_mod.attach_prefixes(selection, tokens, offsets)
    _write_npz(args.out, selection.save)
    print(
        f"[sweep-merge] denom {selection.denom}, "
        f"{selection.sel_p0.size} selected boundaries",
        flush=True,
    )
    return 0


def _pool_start(manifest: dict, tok) -> list[int]:
    """The one-token prefix the pool's rows start with.

    A manifest that records an encoding names it; one written before encodings
    were recorded was built under the tokenizer's own convention, which
    :func:`_check_store_start` then verifies against the store itself.

    Args:
        manifest: The pool manifest.
        tok: The pool model's tokenizer.
    """
    encoding = manifest.get("encoding") or {}
    if "start" in encoding:
        return [int(i) for i in encoding["start"]]
    return pooled_mod.start_prefix(tok)


def _check_store_start(tokens, offsets, start) -> None:
    """Every row of the token store starts with the prefix the pool is read under.

    Raises:
        SystemExit: Some row starts with another token, so the store was built
            under a prefix the manifest does not record.
    """
    problem = _store_arrays_problem(tokens, offsets, offsets.size - 1, start)
    if problem:
        raise SystemExit(
            f"[sweep-merge] the token store {problem}; the pool is read under the "
            f"prefix {start}; rebuild it with `pool --start-ids` so the prefix is "
            "recorded"
        )


def _check_mass_identity(selection, index) -> None:
    """Column mass cannot exceed the frame it was summed over.

    Probabilities over the whole vocabulary sum to one at every boundary, so
    the mass in any column subset is bounded by the frame's total
    multiplicity, which is the pool's non-BOS boundary count.
    """
    mass = float(selection.colsum.sum())
    frame_mass = float(index.n_boundaries - index.n_samples)
    if mass > frame_mass + 1e-6:
        raise SystemExit(
            f"[sweep-merge] column mass {mass} exceeds the frame mass "
            f"{frame_mass}; the sweep is inconsistent."
        )
    if int(selection.stratum_boundary_counts.sum()) > frame_mass:
        raise SystemExit(
            "[sweep-merge] the swept frame is larger than the indexed pool."
        )
