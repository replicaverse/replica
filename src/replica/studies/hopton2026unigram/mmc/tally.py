"""Count each tracked surface on the pool text, exactly and by spelling."""

import argparse
import json

import numpy as np

from replica.feature.frequency import pooled as pooled_mod
from replica.utils.digest import text_sha256

from . import common
from .common import _done, _write_text
from .pool import _load_manifest, _shard_encoded, _shard_samples


def cmd_counts(args: argparse.Namespace) -> int:
    """Count one shard's occurrences of every target surface.

    Model-free, so the whole pool is countable on cpus while the sweep runs on
    gpus. The remainder pairs this exact count against a sum over the frame, so
    it has to split the way the frame does: on whitespace.

    Args:
        args: Parsed `counts` arguments: the `pool_manifest`, the `shard_index`
            to tally, the `words` union to count occurrences of, and the `out`
            path for the tally.
    """
    if _done(args.out, "counts"):
        return 0
    manifest = _load_manifest(args.pool_manifest)
    samples, _ = _shard_samples(manifest, args.shard_index)
    blob = args.words.read_text(encoding="utf-8")
    words = json.loads(blob)
    counts = pooled_mod.unit_counts(words, samples)
    # A tally is only meaningful for the word list and the pool it was taken
    # over, and the merge has to know which shard it is, so all three ride with
    # it rather than being assumed at merge time.
    _write_text(
        args.out,
        json.dumps(
            {
                "shard": int(args.shard_index),
                "words_sha": text_sha256(blob),
                "pool_fingerprint": manifest["pool_fingerprint"],
                "counts": counts,
            }
        ),
    )
    print(
        f"[counts {args.shard_index}] {len(samples)} samples, "
        f"{sum(counts.values())} occurrences over {len(words)} surfaces",
        flush=True,
    )
    return 0


def cmd_counts_merge(args: argparse.Namespace) -> int:
    """Add the per-shard tallies into one count per surface.

    Args:
        args: Parsed `counts-merge` arguments: the tally `parts` to sum, the
            `expected_shards` count they must tile, the `words` union they
            were counted against, and the `out` path for the combined tally.
    """
    if _done(args.out, "counts-merge"):
        return 0
    parts = sorted(args.parts)
    if len(parts) != args.expected_shards:
        raise ValueError(f"expected {args.expected_shards} tallies, got {len(parts)}")
    blob = args.words.read_text(encoding="utf-8")
    words = json.loads(blob)
    words_sha = text_sha256(blob)
    total = dict.fromkeys(words, 0)
    seen: dict[int, str] = {}
    fingerprints = set()
    for path in parts:
        tally = json.loads(path.read_text(encoding="utf-8"))
        shard = int(tally["shard"])
        if shard in seen:
            raise ValueError(f"shard {shard} tallied twice: {seen[shard]}, {path.name}")
        seen[shard] = path.name
        if tally["words_sha"] != words_sha:
            raise ValueError(f"{path.name} was counted over a different word list")
        fingerprints.add(tally["pool_fingerprint"])
        for word, n in tally["counts"].items():
            if word not in total:
                raise KeyError(f"{path.name} counts {word!r}, which is not a target")
            total[word] += int(n)
    if sorted(seen) != list(range(args.expected_shards)):
        raise ValueError(f"tallies do not tile 0..{args.expected_shards - 1}")
    if len(fingerprints) != 1:
        raise ValueError(f"tallies span {len(fingerprints)} pools")
    _write_text(
        args.out,
        json.dumps(
            {
                "words_sha": words_sha,
                "pool_fingerprint": fingerprints.pop(),
                "counts": total,
            }
        ),
    )
    covered = sum(1 for n in total.values() if n)
    print(
        f"[counts-merge] {sum(total.values())} occurrences; "
        f"{covered}/{len(total)} surfaces occur at least once",
        flush=True,
    )
    return 0


def cmd_splits(args: argparse.Namespace) -> int:
    """Split one shard's occurrences of every target surface by spelling.

    Each occurrence is classed by whether the span that realized it is the
    word's canonical tokenization, read off the sampled ids: the decoded text
    carries one spelling per unit, so re-encoding it would report no residual
    mass. ``--source mmc_pool_ee`` charges that residual count directly.

    Args:
        args: Parsed `splits` arguments: the `pool_manifest`, the `pool_dir`
            token store the split reads, the `shard_index` to split, the
            `words` union, the `alloc` file naming the swept columns, and the
            `out` path for the split, which also records the shard's unit total
            so the normalizer counts the same segmentation as the numerator.
    """
    if _done(args.out, "splits"):
        return 0
    manifest = _load_manifest(args.pool_manifest)
    tok = common._tokenizer(manifest["model_name"])
    blob = args.words.read_text(encoding="utf-8")
    words = json.loads(blob)
    encoded = _shard_encoded(args.pool_dir, manifest, args.shard_index)
    # The swept columns decide which spelling of a separator-less surface the
    # sampled channel scores, so the split classifies against the same set.
    with np.load(args.alloc, allow_pickle=False) as alloc:
        columns = set(alloc["first_ids"].astype(np.int64).tolist())
    bset = pooled_mod.boundary_set(tok)
    split = pooled_mod.unit_counts_split(
        words, encoded, tok, bset=bset, columns=columns
    )
    units = pooled_mod.unit_total(encoded, tok, bset=bset)
    _write_text(
        args.out,
        json.dumps(
            {
                "shard": int(args.shard_index),
                "words_sha": text_sha256(blob),
                "pool_fingerprint": manifest["pool_fingerprint"],
                "n_units": int(units),
                "counts": {w: [int(c), int(r)] for w, (c, r) in split.items()},
            }
        ),
    )
    canonical = sum(c for c, _ in split.values())
    residual = sum(r for _, r in split.values())
    print(
        f"[splits {args.shard_index}] {len(encoded)} samples, {units} units, "
        f"{canonical} canonical + {residual} residual occurrences over "
        f"{len(words)} surfaces",
        flush=True,
    )
    return 0


def cmd_splits_merge(args: argparse.Namespace) -> int:
    """Add the per-shard splits into one pair per surface, plus the unit total.

    Args:
        args: Parsed `splits-merge` arguments: the shard `parts`, the
            `expected_shards` they must tile, the `words` union, and the `out`
            path for the merged split.
    """
    if _done(args.out, "splits-merge"):
        return 0
    parts = sorted(args.parts)
    if len(parts) != args.expected_shards:
        raise ValueError(f"expected {args.expected_shards} splits, got {len(parts)}")
    blob = args.words.read_text(encoding="utf-8")
    words = json.loads(blob)
    words_sha = text_sha256(blob)
    total = {w: [0, 0] for w in words}
    units = 0
    seen: dict[int, str] = {}
    fingerprints = set()
    for path in parts:
        tally = json.loads(path.read_text(encoding="utf-8"))
        shard = int(tally["shard"])
        if shard in seen:
            raise ValueError(f"shard {shard} split twice: {seen[shard]}, {path.name}")
        seen[shard] = path.name
        if tally["words_sha"] != words_sha:
            raise ValueError(f"{path.name} was split over a different word list")
        fingerprints.add(tally["pool_fingerprint"])
        units += int(tally["n_units"])
        for word, (canonical, residual) in tally["counts"].items():
            if word not in total:
                raise KeyError(f"{path.name} counts {word!r}, which is not a target")
            total[word][0] += int(canonical)
            total[word][1] += int(residual)
    if sorted(seen) != list(range(args.expected_shards)):
        raise ValueError(f"splits do not tile 0..{args.expected_shards - 1}")
    if len(fingerprints) != 1:
        raise ValueError(f"splits span {len(fingerprints)} pools")
    _write_text(
        args.out,
        json.dumps(
            {
                "words_sha": words_sha,
                "pool_fingerprint": fingerprints.pop(),
                "n_units": units,
                "counts": total,
            }
        ),
    )
    covered = sum(1 for c, r in total.values() if c + r)
    residual = sum(r for _, r in total.values())
    print(
        f"[splits-merge] {units} units; {covered}/{len(total)} surfaces occur at "
        f"least once, {residual} occurrences spelled off-canon",
        flush=True,
    )
    return 0
