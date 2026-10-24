"""Estimate one shard of the union, exhaustively or by priority sampling."""

import argparse
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np

from replica.feature.frequency import pooled as pooled_mod
from replica.feature.frequency.registry import SOURCES

if TYPE_CHECKING:
    from replica.feature.frequency.pooled import PooledUnigramSource
    from replica.feature.frequency.sampled import SampledUnigramSource

from .common import (
    _EE,
    _PATHWISE,
    _POOLED,
    _configure_sampler,
    _done,
    _git_sha,
    _version_for,
    _write_text,
)
from .plan import _load_allocation
from .profile import STRATIFICATION


def cmd_shard(args: argparse.Namespace) -> int:
    """Estimate one contiguous slice of the word union.

    Args:
        args: Parsed `shard` arguments: `source`, the `words` union and
            `start`/`end` slice bounds, `model_name` and `device`, and either
            the exhaustive source's `samples` with the optional `start_ids`
            they were drawn after, or the pooled source's
            `selection`, `allocation`, `homes`, `pool_fingerprint`, `counts`,
            `take_store`, `pool_dir`, `k_eval`, and `k_take`. Writes to `out`.
    """
    if _done(args.out, "shard"):
        return 0
    words = json.loads(args.words.read_text(encoding="utf-8"))
    mine = words[args.start : args.end]
    if not mine:
        print(f"[shard {args.start}:{args.end}] empty", flush=True)
        return 0
    if args.source in _POOLED:
        return _shard_pooled(args, words, mine)
    return _shard_exhaustive(args, mine)


def _shard_exhaustive(args, mine) -> int:
    samples = json.loads(args.samples.read_text(encoding="utf-8"))
    source = cast(
        "SampledUnigramSource",
        SOURCES["mmc"].build(
            model_name=args.model_name,
            samples=samples,
            device=args.device,
            start_ids=getattr(args, "start_ids", None),
        ),
    )
    _configure_sampler(source)
    if source.max_continuation_boundaries is not None:
        raise SystemExit("mmc must be uncapped; registry default changed")
    estimates = source.estimate_words(mine)
    _reject_nonfinite(estimates, f"shard {args.start}:{args.end}")
    _write_text(args.out, json.dumps(estimates, ensure_ascii=False))
    print(f"[shard {args.start}:{args.end}] {len(estimates)} words", flush=True)
    return 0


def _load_counts(path: Path, pool_fingerprint: str) -> dict:
    """Merged occurrence counts, refusing a tally taken over another pool."""
    blob = json.loads(path.read_text(encoding="utf-8"))
    if blob.get("pool_fingerprint") != pool_fingerprint:
        raise SystemExit(
            f"[counts] {path.name} was taken over pool "
            f"{str(blob.get('pool_fingerprint'))[:12]}, the selection is "
            f"{pool_fingerprint[:12]}"
        )
    return blob["counts"]


def _load_splits(path: Path, pool_fingerprint: str) -> tuple[dict, int]:
    """Canonical/residual counts and the pool's word count.

    Both come off the same segmentation of the same ids, so the numerator and
    the normalizer cannot end up counting different corpora.
    """
    blob = json.loads(path.read_text(encoding="utf-8"))
    if blob.get("pool_fingerprint") != pool_fingerprint:
        raise SystemExit(
            f"[splits] {path.name} was taken over pool "
            f"{str(blob.get('pool_fingerprint'))[:12]}, the selection is "
            f"{pool_fingerprint[:12]}"
        )
    units = int(blob["n_units"])
    if units <= 0:
        raise SystemExit(f"[splits] {path.name} reports {units} units")
    counts = {w: (int(can), int(res)) for w, (can, res) in blob["counts"].items()}
    return counts, units


def _load_take_store(path, pool_dir, pool_fingerprint: str):
    """The take store, with the token store its prefixes are read from."""
    if path is None or pool_dir is None:
        raise SystemExit("[shard] the pooled estimator needs --take-store/--pool-dir")
    store = pooled_mod.TakeStore.load(path)
    if store.pool_fingerprint != pool_fingerprint:
        raise SystemExit(
            f"[takes] {Path(path).name} is from pool "
            f"{store.pool_fingerprint[:12]}, the selection is "
            f"{pool_fingerprint[:12]}"
        )
    tokens = np.memmap(Path(pool_dir) / "tokens.i32", dtype=np.int32, mode="r")
    offsets = np.fromfile(Path(pool_dir) / "tokens.off.i64", dtype=np.int64)
    return store.with_pool(tokens, offsets)


def _pooled_inputs(args, stage: str) -> str:
    """The numerator the source names, refusing the other numerator's inputs.

    Given both kinds of input the run would be one estimator wearing the
    other's name, and nothing downstream could tell.
    """
    if args.source == _EE:
        for flag, name in (("split_counts", "--split-counts"), ("cap", "--cap")):
            if not getattr(args, flag, None):
                raise SystemExit(f"[{stage}] --source {_EE} needs {name}")
        for flag, name in (("take_store", "--take-store"), ("counts", "--counts")):
            if getattr(args, flag, None) is not None:
                raise SystemExit(
                    f"[{stage}] --source {_EE} takes its counts from "
                    f"--split-counts; drop {name}"
                )
        return _EE
    if args.source == _PATHWISE:
        if args.counts is None:
            raise SystemExit(f"[{stage}] --source {_PATHWISE} needs --counts")
        if args.take_store is None or args.pool_dir is None:
            raise SystemExit(
                f"[{stage}] --source {_PATHWISE} needs --take-store and --pool-dir"
            )
        for flag, name in (("split_counts", "--split-counts"), ("cap", "--cap")):
            if getattr(args, flag, None):
                raise SystemExit(
                    f"[{stage}] {name} belongs to --source {_EE}; the pathwise "
                    "numerator does not price the horizon"
                )
        return _PATHWISE
    raise SystemExit(f"[{stage}] {args.source!r} is not a pooled source")


def _shard_pooled(args, words, mine) -> int:
    mode = _pooled_inputs(args, "shard")
    selection = pooled_mod.PoolSelection.load(
        args.selection, stratification=STRATIFICATION
    )
    allocation, _, params = _load_allocation(args.allocation, args.homes, words)
    if args.pool_fingerprint and args.pool_fingerprint != selection.pool_fingerprint:
        raise SystemExit("[shard] the selection is from a different pool")
    if mode == _EE:
        splits, n_units = _load_splits(args.split_counts, selection.pool_fingerprint)
        # The split is the authority on both counts: a tally taken off the text
        # disagrees wherever the decoder dropped a space.
        counts = {w: sum(splits[w]) for w in splits}
        take_store = None
    else:
        counts = _load_counts(args.counts, selection.pool_fingerprint)
        splits, n_units = None, None
        take_store = _load_take_store(
            args.take_store, args.pool_dir, selection.pool_fingerprint
        )
    missing = [w for w in mine if w not in counts]
    if missing:
        raise SystemExit(
            f"[shard] {len(missing)} words have no pooled count, e.g. {missing[:3]}"
        )
    source = cast(
        "PooledUnigramSource",
        SOURCES[args.source].build(
            model_name=args.model_name,
            selection=selection,
            allocation=allocation,
            k_eval=args.k_eval,
            device=args.device,
            pool_fingerprint=selection.pool_fingerprint,
            unit_counts=counts,
        ),
    )
    if source._selection is not selection or source.k_eval != args.k_eval:
        raise SystemExit("[shard] the builder dropped the selection or k_eval")
    if source._pool_fingerprint != selection.pool_fingerprint:
        raise SystemExit("[shard] the builder dropped the pool fingerprint")
    if source.prob_floor != 0.0 or source.max_continuation_boundaries is not None:
        raise SystemExit("[shard] the pooled source must be uncapped and unfloored")
    version = _version_for(mode)
    normalizer = n_units if mode == _EE else selection.denom
    done = _resume(args.out.with_suffix(".jsonl"), args.k_eval, version, normalizer)
    todo = [w for w in mine if w not in done]
    records = _estimate_with_resume(
        source,
        todo,
        args.out.with_suffix(".jsonl"),
        take_store,
        args.k_take,
        cap=args.cap,
        splits=splits,
        version=version,
        denom=n_units,
    )
    records.update(done)
    payload = {
        "pool_fingerprint": selection.pool_fingerprint,
        "n_pool_samples": selection.n_pool_samples,
        "denom": selection.denom,
        "k_eval": args.k_eval,
        "allocation_sha": params["alloc_sha"],
        "estimator_version": version,
        "normalizer": normalizer,
        "cap": args.cap,
        "selection_version": selection.estimator_version,
        "k_take": args.k_take if mode == _PATHWISE else None,
        "code_git_sha": _git_sha(),
        "records": {w: r.to_json() for w, r in records.items()},
    }
    _reject_nonfinite(_probabilities(payload), f"shard {args.start}:{args.end}")
    _write_text(args.out, json.dumps(payload, ensure_ascii=False))
    print(f"[shard {args.start}:{args.end}] {len(records)} words", flush=True)
    return 0


def _resume(path: Path, k_eval: int, version: str, normalizer) -> dict:
    """Records already written by an earlier attempt of this slice.

    A record drawn at another budget is dropped, since ``k_eval`` is part of
    the published cell id; one from another estimator version, or divided by
    another normalizer, is recomputed. The normalizer is checked separately
    because the estimator version alone cannot tell two normalizers apart.
    """
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        blob = json.loads(line)
        if blob.get("estimator_version") != version:
            continue
        if blob.get("normalizer") != normalizer:
            continue
        record = pooled_mod.WordRecord.from_json(blob)
        if record.k_eval not in (0, k_eval):
            continue
        out[record.word] = record
    return out


def _estimate_with_resume(
    source,
    todo,
    jsonl: Path,
    take_store,
    k_take,
    *,
    cap=None,
    splits=None,
    version=None,
    denom=None,
) -> dict:
    """Estimate the remaining words, appending each record as it lands."""
    if not todo:
        return {}
    jsonl.parent.mkdir(parents=True, exist_ok=True)
    out: dict = {}
    with open(jsonl, "a", encoding="utf-8") as handle:

        def _emit(record):
            out[record.word] = record
            line = {
                **record.to_json(),
                "estimator_version": version or pooled_mod.ESTIMATOR_VERSION,
                "normalizer": denom if denom is not None else source._selection.denom,
            }
            handle.write(json.dumps(line, ensure_ascii=False) + "\n")
            handle.flush()

        source._ensure()
        pooled_mod.pps_estimate_batch(
            todo,
            source._selection,
            source._model,
            source._tokenizer,
            allocation=source._allocation,
            k_eval=source.k_eval,
            batch_size=source.batch_size,
            thresholds=source._thresholds,
            unit_counts=source._unit_counts,
            take_store=take_store,
            k_take=k_take,
            pool_fingerprint=source._pool_fingerprint,
            progress=_emit,
            cap=cap,
            splits=splits,
            denom=denom,
        )
    return out


def _probabilities(part: dict) -> dict:
    """The word -> p view of a shard file, whichever source wrote it."""
    if "records" in part:
        return {w: r["p_hat"] for w, r in part["records"].items()}
    return part


def _reject_nonfinite(estimates: dict, where: str) -> None:
    """Raise if any estimate is non-finite or not strictly positive.

    Estimates are sums of strictly positive softmax values, so a NaN means the
    forward pass produced one. A zero survives the merge but turns into a NaN
    at materialize time, so it is rejected here.
    """
    bad = [
        w
        for w, p in estimates.items()
        if p is None or not isinstance(p, (int, float)) or not math.isfinite(p)
    ]
    if bad:
        raise SystemExit(
            f"[{where}] {len(bad)}/{len(estimates)} non-finite estimates "
            f"(e.g. {bad[:5]}). Re-run this slice; do not merge this table."
        )
    zeros = [w for w, p in estimates.items() if p <= 0.0]
    if zeros:
        raise SystemExit(
            f"[{where}] {len(zeros)}/{len(estimates)} estimates are not strictly "
            f"positive (e.g. {zeros[:5]}). A zero publishes as NaN; re-run this "
            "slice."
        )
