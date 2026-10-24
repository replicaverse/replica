"""Combine shard tables into one type probability table and audit it."""

import argparse
import json
from pathlib import Path

import numpy as np

from replica.feature.frequency import pooled as pooled_mod

from .common import _EE, _POOLED, _done, _git_sha, _version_for, _write_text
from .estimate import (
    _load_counts,
    _load_splits,
    _load_take_store,
    _pooled_inputs,
    _probabilities,
    _reject_nonfinite,
)
from .plan import _load_allocation
from .profile import STRATIFICATION
from .sweep import _load_model


def cmd_merge(args: argparse.Namespace) -> int:
    """Combine shard tables into one type -> p table.

    ``--words`` checks the merged table covers the union exactly, and for the
    pooled source every shard must carry the same pool fingerprint.

    Args:
        args: Parsed `merge` arguments: `source`, the `shards` tables to
            combine, the `words` union they must cover, an optional
            `pool_fingerprint` to enforce, an optional `audited` count for the
            metadata sidecar, and the `out` path for the merged table.
    """
    table: dict[str, float] = {}
    records: dict[str, dict] = {}
    meta: dict = {}
    for path in sorted(args.shards):
        part = json.loads(Path(path).read_text(encoding="utf-8"))
        _check_fingerprint(part, Path(path).name, args, meta)
        estimates = _probabilities(part)
        _reject_nonfinite(estimates, Path(path).name)
        _reject_clamped(part.get("records", {}), Path(path).name)
        records.update(part.get("records", {}))
        for word, value in estimates.items():
            if word in table and table[word] != value:
                raise SystemExit(
                    f"shards disagree on {word!r}: {table[word]} vs {value}. "
                    "Shards must share one sample set; regenerate with a single "
                    "`samples` output."
                )
            table[word] = value
    _alias_bare_twins(table, records)
    _check_coverage(table, args.words)
    _write_text(args.out, json.dumps(table, ensure_ascii=False))
    if records:
        _write_records(args, records, meta, len(args.shards))
    print(f"[merge] {len(table)} words <- {len(args.shards)} shards", flush=True)
    return 0


def _alias_bare_twins(table: dict, records: dict) -> None:
    """Publish one value per type when a key appears with and without its lead.

    A separator-less key names the same estimand as its spaced twin, and the
    two are estimated from different draws, so left alone the table would carry
    two slightly different numbers for one quantity. The twin's record is
    copied verbatim under the bare key.
    """
    for word in list(records):
        twin = " " + word
        if word[:1].isspace() or twin not in records:
            continue
        aliased = dict(records[twin])
        aliased["word"] = word
        records[word] = aliased
        if twin in table:
            table[word] = table[twin]


def _reject_clamped(records: dict, where: str) -> None:
    """Refuse a value that sits on the float64 floor rather than on its estimate.

    The record keeps the log-domain surprisal; the table publishes only the
    probability, which at the floor understates it.
    """
    floored = [
        w
        for w, r in records.items()
        if r.get("underflow_fallback") and float(r["p_hat"]) <= pooled_mod.TINY
    ]
    if floored:
        raise SystemExit(
            f"[{where}] {len(floored)} estimates are clamped to the float64 floor "
            f"(e.g. {floored[:5]}); the published probability would understate "
            "their surprisal"
        )


def _check_fingerprint(part, name, args, meta) -> None:
    """Every pooled shard must name the pool the merge was asked for."""
    if not isinstance(part, dict):
        raise SystemExit(f"[merge] {name} is not a shard table")
    if "records" not in part:
        if args.source in _POOLED:
            raise SystemExit(f"[merge] {name} has no records block")
        return
    built = part.get("estimator_version")
    if built != _version_for(args.source):
        raise SystemExit(
            f"[merge] {name} was estimated by {built!r}; --source {args.source} "
            f"merges {_version_for(args.source)!r} shard tables"
        )
    got = part.get("pool_fingerprint")
    if args.pool_fingerprint and got != args.pool_fingerprint:
        raise SystemExit(
            f"[merge] {name} was computed against pool {got}, not "
            f"{args.pool_fingerprint}"
        )
    if meta and meta.get("pool_fingerprint") != got:
        raise SystemExit(f"[merge] {name} disagrees with the earlier shards' pool")
    for key in (
        "k_eval",
        "allocation_sha",
        "rng_seed",
        "estimator_version",
        "selection_version",
        "k_take",
        "cap",
    ):
        if key in meta and key in part and meta[key] != part[key]:
            raise SystemExit(
                f"[merge] {name} has {key}={part[key]!r} but earlier shards have "
                f"{meta[key]!r}; one published cell carries one budget"
            )
    # A shard that predates the normalizer, or carries another one, divides its
    # numerators by a different count: the values would not be comparable and
    # nothing downstream could tell. Absent on one side counts as a difference.
    for key in ("normalizer", "denom"):
        if meta and meta.get(key) != part.get(key):
            raise SystemExit(
                f"[merge] {name} has {key}={part.get(key)!r} but earlier shards "
                f"have {meta.get(key)!r}; every shard must divide by one count"
            )
    meta.update({k: v for k, v in part.items() if k != "records"})


def _check_coverage(table, words_path: Path) -> None:
    expected = set(json.loads(words_path.read_text(encoding="utf-8")))
    missing = expected - set(table)
    if missing:
        raise SystemExit(
            f"[merge] {len(missing)}/{len(expected)} words absent "
            f"(e.g. {sorted(missing)[:5]}). Re-run the shards covering them."
        )
    extra = set(table) - expected
    if extra:
        raise SystemExit(
            f"[merge] {len(extra)} words not in the union (e.g. "
            f"{sorted(extra)[:5]}); the shard tables do not match this plan."
        )


def _write_records(args, records, meta, n_shards) -> None:
    """The per-word certificate sidecars that qualify a pooled table."""
    lines = "".join(
        json.dumps(records[w], ensure_ascii=False) + "\n" for w in sorted(records)
    )
    _write_text(args.out.with_name(args.out.stem + ".records.jsonl"), lines)
    supported = [w for w, r in records.items() if r["supported"]]
    audited = 0
    if args.audited is not None and args.audited.exists():
        audited = len(json.loads(args.audited.read_text(encoding="utf-8")))
    meta = dict(meta)
    meta.update(
        {
            "n_words": len(records),
            "n_shards": n_shards,
            "n_supported": len(supported),
            "n_audited": audited,
            "n_underflow": len(
                [w for w, r in records.items() if r["underflow_fallback"]]
            ),
            # the shards' agreed value, which _check_fingerprint has verified is
            # uniform; the module constant names the pathwise numerator only
            "estimator_version": meta.get(
                "estimator_version", pooled_mod.ESTIMATOR_VERSION
            ),
            "code_git_sha": _git_sha(),
        }
    )
    _write_text(
        args.out.with_name(args.out.stem + ".meta.json"), json.dumps(meta, indent=2)
    )


def cmd_audit(args: argparse.Namespace) -> int:
    """Re-estimate a fraction of the union under a different shard layout.

    Args:
        args: Parsed `audit` arguments: `source`, `model_name` and `device`,
            the `words` union and optional `homes`, the `selection` and
            `allocation` to re-score against, the `fraction` of words to draw
            and its `layout_seed`, `k_eval`, `batch_size`, `k_take`, the
            `counts`, `take_store`, and `pool_dir` the pooled estimate reads,
            the merged `table` to check agreement against, and the `out` path
            for the re-estimated records.
    """
    if _done(args.out, "audit"):
        return 0
    mode = _pooled_inputs(args, "audit")
    words = json.loads(args.words.read_text(encoding="utf-8"))
    rng = np.random.default_rng(args.layout_seed)
    take = max(1, int(round(args.fraction * len(words))))
    picked = sorted(rng.choice(len(words), size=take, replace=False).tolist())
    subset = _audit_subset(words, picked)
    selection = pooled_mod.PoolSelection.load(
        args.selection, stratification=STRATIFICATION
    )
    allocation, _, _ = _load_allocation(args.allocation, args.homes, words)
    if mode == _EE:
        splits, n_units = _load_splits(args.split_counts, selection.pool_fingerprint)
        counts = {w: sum(splits[w]) for w in splits}
        take_store = None
    else:
        counts = _load_counts(args.counts, selection.pool_fingerprint)
        splits, n_units = None, None
        take_store = _load_take_store(
            args.take_store, args.pool_dir, selection.pool_fingerprint
        )
    model, tok = _load_model(args.model_name, args.device)
    records = pooled_mod.pps_estimate_batch(
        subset,
        selection,
        model,
        tok,
        allocation=allocation,
        k_eval=args.k_eval,
        batch_size=args.batch_size,
        unit_counts=counts,
        take_store=take_store,
        k_take=args.k_take,
        cap=args.cap,
        splits=splits,
        denom=n_units if splits is not None else None,
    )
    _compare_audit(records, args.table)
    _write_text(
        args.out,
        json.dumps({w: r.to_json() for w, r in records.items()}, ensure_ascii=False),
    )
    print(f"[audit] {len(records)} words re-estimated and matched", flush=True)
    return 0


def _audit_subset(words: list, picked: list) -> list:
    """The audited surfaces, with aliased keys resolved to their source.

    A separator-less key carries its spaced twin's record verbatim, so the
    twin is what a re-run can reproduce; a fresh draw under the bare key uses
    that word's own home strata and can differ. The twin is audited instead,
    and it covers the bare key because the merge copies its record.
    """
    union = set(words)
    subset = [
        " " + words[i]
        if not words[i][:1].isspace() and " " + words[i] in union
        else words[i]
        for i in picked
    ]
    return sorted(dict.fromkeys(subset))


def _compare_audit(records, table_path: Path) -> None:
    """The selection is fixed in the sweep, so a re-run must reproduce it.

    A difference here comes from the continuation stage or the hardware, and it
    disqualifies the merged table.
    """
    table = json.loads(table_path.read_text(encoding="utf-8"))
    missing = sorted(w for w in records if w not in table)
    if missing:
        raise SystemExit(
            f"[audit] {len(missing)} audited words absent from the merged table "
            f"(e.g. {missing[:5]})"
        )
    bad = [
        (w, float(table[w]), r.p_hat)
        for w, r in records.items()
        if float(table[w]) != r.p_hat
    ]
    if bad:
        raise SystemExit(
            f"[audit] {len(bad)}/{len(records)} words differ from the merged "
            f"table (e.g. {bad[:3]}); do not publish this table"
        )
