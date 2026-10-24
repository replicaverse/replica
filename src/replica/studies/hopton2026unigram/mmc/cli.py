"""Argument parsing and stage dispatch for the marginal Monte Carlo pipeline."""

import argparse
import os
from pathlib import Path

from .common import _EE, _PATHWISE, start_ids_arg
from .estimate import cmd_shard
from .index import cmd_index, cmd_index_merge, cmd_takes
from .materialize import cmd_materialize
from .plan import cmd_plan, cmd_words
from .pool import cmd_pool, cmd_samples
from .sweep import cmd_sweep, cmd_sweep_merge
from .table import cmd_audit, cmd_merge
from .tally import cmd_counts, cmd_counts_merge, cmd_splits, cmd_splits_merge

_DESCRIPTION = (
    "Estimate marginal Monte Carlo unigram surprisal, one stage per invocation."
)


def _add_source(parser) -> None:
    parser.add_argument("--source", choices=("mmc", _PATHWISE, _EE), default="mmc")


def _add_samples(sub) -> None:
    p = sub.add_parser("samples", help="draw one sample shard for one LM")
    _add_source(p)
    p.add_argument("--model-name", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--ids-out",
        type=Path,
        default=None,
        help="sidecar of the drawn token ids (ids, offsets), required by `pool --ids-dir`",
    )
    p.add_argument("--n-samples", type=int, default=500)
    p.add_argument("--shard-samples", type=int, default=None)
    p.add_argument("--shard-index", type=int, default=None)
    p.add_argument("--seed-base", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--start-ids",
        type=start_ids_arg,
        default=None,
        help="the one token id samples are drawn after; the tokenizer's bos, "
        "else its eos, by default",
    )
    p.set_defaults(func=cmd_samples)

    p = sub.add_parser("pool", help="merge sample shards into a token store")
    _add_source(p)
    p.add_argument("--model-name", required=True)
    p.add_argument("--shards-dir", type=Path, required=True)
    p.add_argument("--expected-shards", type=int, required=True)
    p.add_argument(
        "--ids-dir",
        type=Path,
        default=None,
        help="sampler-written token ids, one sidecar per shard; the pool reads "
        "them and does not re-encode the text. Omitted, the decoded text is "
        "re-encoded, the historical layout, and the manifest records that",
    )
    p.add_argument(
        "--start-ids",
        type=start_ids_arg,
        default=None,
        help="the one token id the shards were drawn after; the tokenizer's bos, "
        "else its eos, by default; a sidecar that records its prefix must agree",
    )
    p.add_argument("--seed-base", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=cmd_pool)


def _add_counts(sub) -> None:
    p = sub.add_parser("counts", help="count one shard's surface occurrences")
    _add_source(p)
    p.add_argument("--pool-manifest", type=Path, required=True)
    p.add_argument("--shard-index", type=int, required=True)
    p.add_argument("--words", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=cmd_counts)

    p = sub.add_parser("counts-merge", help="add the per-shard tallies")
    _add_source(p)
    p.add_argument("--parts", nargs="+", type=Path, required=True)
    p.add_argument("--expected-shards", type=int, required=True)
    p.add_argument("--words", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=cmd_counts_merge)

    p = sub.add_parser("splits", help="split one shard's occurrences by spelling")
    _add_source(p)
    p.add_argument("--pool-manifest", type=Path, required=True)
    # The split reads the token store because it runs over the sampled ids.
    p.add_argument("--pool-dir", type=Path, required=True)
    p.add_argument("--shard-index", type=int, required=True)
    p.add_argument("--words", type=Path, required=True)
    p.add_argument("--alloc", type=Path, required=True, help="alloc.npz from plan")
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=cmd_splits)

    p = sub.add_parser("splits-merge", help="add the per-shard splits")
    _add_source(p)
    p.add_argument("--parts", nargs="+", type=Path, required=True)
    p.add_argument("--expected-shards", type=int, required=True)
    p.add_argument("--words", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=cmd_splits_merge)


def _add_index(sub) -> None:
    p = sub.add_parser("index", help="label one shard's word boundaries")
    _add_source(p)
    p.add_argument("--pool-manifest", type=Path, required=True)
    p.add_argument(
        "--pool-dir",
        type=Path,
        default=None,
        help="token store to index; without it the text is re-encoded",
    )
    p.add_argument("--shard-index", type=int, required=True)
    p.add_argument("--dup-collapse-max-len", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=cmd_index)

    p = sub.add_parser("index-merge", help="tile and deduplicate the indices")
    _add_source(p)
    p.add_argument("--parts", nargs="+", type=Path, required=True)
    p.add_argument("--expected-shards", type=int, required=True)
    p.add_argument("--dup-collapse-max-len", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=cmd_index_merge)


def _add_sweep(sub) -> None:
    p = sub.add_parser("sweep", help="score one shard's boundaries")
    _add_source(p)
    p.add_argument("--pool-manifest", type=Path, required=True)
    # Required: without the store the sweep re-encodes the text
    # and scores the canonical segmentation instead of the one the model drew,
    # which is the mismatch the recorded ids exist to prevent.
    p.add_argument("--pool-dir", type=Path, required=True)
    p.add_argument("--index", type=Path, required=True)
    p.add_argument("--alloc", type=Path, required=True)
    p.add_argument("--shard-index", type=int, required=True)
    p.add_argument("--rng-seed", type=int, required=True)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--reduce-chunk", type=int, default=8192)
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--store-device",
        default=None,
        help="where the per-cell priority buffers rest between merges "
        "(cpu to keep them in host memory); the model device by default",
    )
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=cmd_sweep)

    p = sub.add_parser("sweep-merge", help="fold shard sweeps into a selection")
    _add_source(p)
    p.add_argument("--partials", nargs="+", type=Path, required=True)
    p.add_argument("--expected-shards", type=int, required=True)
    p.add_argument("--pool-manifest", type=Path, required=True)
    p.add_argument("--pool-dir", type=Path, required=True)
    p.add_argument("--index", type=Path, required=True)
    p.add_argument("--alloc", type=Path, required=True)
    p.add_argument("--rng-seed", type=int, required=True)
    p.add_argument("--wave", type=int, default=20)
    p.add_argument("--delete-on-consume", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=cmd_sweep_merge)


def _add_estimation(sub) -> None:
    p = sub.add_parser("takes", help="enumerate realized takes for every column")
    p.add_argument("--model-name", required=True)
    p.add_argument("--index", type=Path, required=True, help="index/merged.npz")
    p.add_argument("--pool-dir", type=Path, required=True)
    p.add_argument("--allocation", type=Path, required=True, help="alloc.npz")
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=cmd_takes)

    p = sub.add_parser("shard", help="estimate one slice of the union")
    _add_source(p)
    p.add_argument("--model-name", required=True)
    p.add_argument("--words", type=Path, required=True)
    p.add_argument("--homes", type=Path, default=None)
    p.add_argument("--samples", type=Path, default=None)
    p.add_argument("--selection", type=Path, default=None)
    p.add_argument("--allocation", type=Path, default=None)
    p.add_argument("--pool-fingerprint", default=None)
    p.add_argument("--k-eval", type=int, default=4096)
    p.add_argument("--start", type=int, required=True)
    p.add_argument("--end", type=int, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--counts", type=Path, default=None, help="merged per-surface occurrence counts"
    )
    p.add_argument(
        "--take-store", type=Path, default=None, help="takes.npz from the takes stage"
    )
    p.add_argument(
        "--pool-dir", type=Path, default=None, help="token store the prefixes read"
    )
    p.add_argument("--k-take", type=int, default=2048)
    p.add_argument(
        "--start-ids",
        type=start_ids_arg,
        default=None,
        help="for --source mmc: the one token id the samples were drawn after; "
        "the tokenizer's bos, else its eos, by default",
    )
    p.add_argument(
        "--split-counts",
        type=Path,
        default=None,
        help="merged splits from splits-merge, for --source mmc_pool_ee",
    )
    p.add_argument(
        "--cap",
        type=int,
        default=None,
        help="the sampler's limit on generated tokens (SampledUnigramSource.max_len), "
        "for --source mmc_pool_ee",
    )
    p.set_defaults(func=cmd_shard)

    p = sub.add_parser("merge", help="combine shard tables")
    _add_source(p)
    p.add_argument("--shards", nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--words",
        type=Path,
        required=True,
        help="the union the shards were planned from; checks coverage",
    )
    p.add_argument("--pool-fingerprint", default=None)
    p.add_argument("--audited", type=Path, default=None)
    p.set_defaults(func=cmd_merge)

    p = sub.add_parser("audit", help="re-estimate a fraction of the union")
    _add_source(p)
    p.add_argument("--model-name", required=True)
    p.add_argument("--words", type=Path, required=True)
    p.add_argument("--homes", type=Path, default=None)
    p.add_argument("--selection", type=Path, required=True)
    p.add_argument("--allocation", type=Path, required=True)
    p.add_argument("--k-eval", type=int, default=4096)
    p.add_argument("--fraction", type=float, default=0.02)
    p.add_argument("--layout-seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="cuda")
    p.add_argument("--table", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--counts", type=Path, default=None, help="merged per-surface occurrence counts"
    )
    p.add_argument(
        "--take-store", type=Path, default=None, help="takes.npz from the takes stage"
    )
    p.add_argument(
        "--pool-dir", type=Path, default=None, help="token store the prefixes read"
    )
    p.add_argument("--k-take", type=int, default=2048)
    p.add_argument(
        "--split-counts",
        type=Path,
        default=None,
        help="merged splits from splits-merge, for --source mmc_pool_ee",
    )
    p.add_argument(
        "--cap",
        type=int,
        default=None,
        help="the sampler's limit on generated tokens, for --source mmc_pool_ee",
    )
    p.set_defaults(func=cmd_audit)


def build_parser() -> argparse.ArgumentParser:
    """Assemble the subcommand parser for every pipeline stage.

    Returns:
        A parser whose `cmd` subcommand dispatches to each stage's `cmd_*`
        handler through `set_defaults(func=...)`.
    """
    parser = argparse.ArgumentParser(description=_DESCRIPTION)
    sub = parser.add_subparsers(dest="cmd", required=True)
    _add_samples(sub)

    p = sub.add_parser("words", help="collect the union of surfaces")
    _add_source(p)
    p.add_argument("--datasets", nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--homes-out", type=Path, default=None)
    p.set_defaults(func=cmd_words)

    _add_counts(sub)
    _add_index(sub)

    p = sub.add_parser("plan", help="split the union into equal-cost shards")
    _add_source(p)
    p.add_argument("--model-name", required=True)
    p.add_argument("--words", type=Path, required=True)
    p.add_argument("--homes", type=Path, default=None)
    p.add_argument("--shards", type=int, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--alloc-out", type=Path, default=None)
    p.add_argument("--k-eval", type=int, default=4096)
    p.add_argument("--k-floor", type=int, default=64)
    p.add_argument("--k-home", type=int, default=2048)
    p.add_argument("--replicates", type=int, default=8)
    p.add_argument("--escalation", type=int, default=4)
    p.add_argument("--expected-shards", type=int, default=1)
    p.add_argument("--shard-slack", type=int, default=8)
    p.set_defaults(func=cmd_plan)

    _add_sweep(sub)
    _add_estimation(sub)

    p = sub.add_parser("materialize", help="write per-dataset pickles")
    _add_source(p)
    p.add_argument("--model-name", required=True)
    p.add_argument("--table", type=Path, required=True)
    p.add_argument("--datasets", nargs="+", required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--expect-fingerprint", default=None)
    p.add_argument("--n-samples", type=int, default=500)
    p.add_argument("--k-eval", type=int, default=4096)
    p.add_argument("--seed", type=int, default=None)
    p.set_defaults(func=cmd_materialize)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse CLI arguments and run the selected pipeline stage.

    Args:
        argv: Command-line arguments, defaults to `sys.argv[1:]`.

    Returns:
        The selected stage's exit status.
    """
    args = build_parser().parse_args(argv)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    return args.func(args)
