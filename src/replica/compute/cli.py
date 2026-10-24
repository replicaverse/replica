"""Command-line interface for cached language-model feature computation."""

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

from replica.dataset.catalog.registry import GROUPS, REGISTRY, compute_dir
from replica.feature.core.reproduction import check_reproduction_opt_in

from .methods import METHODS
from .runner import ComputeOutcome, compute_for, expand_datasets

log = logging.getLogger("replica.compute")


def _check_unit(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Import ``--import`` modules, then require a known ``--unit``.

    Runs before any corpus or model loads, so an unknown unit fails at once.

    Args:
        parser: The parser, which reports the error.
        args: The parsed arguments.
    """
    import importlib

    from replica.segmentation import SEGMENTERS

    for module in args.imports:
        importlib.import_module(module)
    if (
        args.unit not in ("words", "characters", "tokens")
        and args.unit not in SEGMENTERS
    ):
        parser.error(
            f"unknown unit {args.unit!r}; registered segmentations: {sorted(SEGMENTERS)}"
        )


def build_parser() -> argparse.ArgumentParser:
    """Build the feature-computation command-line parser."""
    parser = argparse.ArgumentParser(
        prog="replica compute",
        description=__doc__.split("\n", 1)[0],
        epilog="Merge completed slices with: replica compute merge-shards --help",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        action="append",
        required=True,
        help=(
            "Dataset name or group. Repeatable. Datasets: "
            f"{sorted(REGISTRY)}. Groups: {sorted(GROUPS)}."
        ),
    )
    parser.add_argument(
        "--lm",
        required=True,
        help="Hugging Face model id, such as 'gpt2' or 'ai-forever/mGPT'.",
    )
    parser.add_argument(
        "--method",
        action="append",
        choices=METHODS,
        help=(
            "Feature method. Repeatable; defaults to surprisal. Choices: "
            + ", ".join(METHODS)
            + "."
        ),
    )
    parser.add_argument(
        "--allow-reproduction",
        action="store_true",
        help="Explicitly enable entropy variants with known errors for reproduction only.",
    )
    parser.add_argument(
        "--K",
        type=int,
        default=5,
        help="Character-beam width.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=64,
        help="Monte Carlo sample count.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=10,
        help="Maximum tokens per Monte Carlo sample.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Rényi parameter; 1.0 selects Shannon entropy.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Nucleus sampling mass for methods that sample utterances.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional Monte Carlo random seed.",
    )
    parser.add_argument(
        "--word-starters",
        default="default",
        choices=["default", "g-only"],
        help="Monte Carlo end-of-word token set; g-only matches Clark et al.'s "
        "released sampler, the default also ends a word on newline prefixes.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Optional Monte Carlo batch size used to limit peak memory.",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Override the dataset language for language-sensitive methods.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory; defaults to PSYCHO_COMPUTE_DIR or compute/.",
    )
    bos_group = parser.add_mutually_exclusive_group()
    bos_group.add_argument(
        "--add-bos",
        dest="add_bos",
        action="store_true",
        default=True,
        help="Prepend a beginning-of-sequence token where supported.",
    )
    bos_group.add_argument(
        "--no-bos",
        dest="add_bos",
        action="store_false",
        help="Disable beginning-of-sequence token prepending.",
    )
    parser.add_argument(
        "--start-ids",
        type=int,
        nargs="+",
        help="Explicit single-token prefix for supported methods; requires exactly one ID.",
    )
    parser.add_argument(
        "--context-window",
        type=int,
        help="Maximum context tokens for supported methods; defaults to the model policy.",
    )
    parser.add_argument(
        "--dtype",
        choices=["float32", "float16", "bfloat16"],
        help="Parameter precision the weights load in; default single precision.",
    )
    parser.add_argument(
        "--device-map",
        choices=["auto"],
        help="Shard the weights over every visible GPU through Accelerate.",
    )
    parser.add_argument(
        "--model-revision",
        help="Model/tokenizer Hub revision; resolved to an immutable snapshot.",
    )
    parser.add_argument(
        "--store-distribution",
        action="store_true",
        help="Store each word's full next-token distribution where the method offers one.",
    )
    parser.add_argument(
        "--run-manifest",
        type=Path,
        help="JSON outcome manifest (default: output directory/run-<id>.json).",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Accept incomplete batches; outcomes are still recorded.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute an existing cache.",
    )
    parser.add_argument(
        "--context-scope",
        choices=["auto", "stimulus", "sentence", "story"],
        default="auto",
        help="auto uses the dataset's declared context for supported methods and "
        "stimulus context elsewhere; stimulus retains page-only context; "
        "sentence uses the authors' segmentation; story requires verified "
        "preceding pages.",
    )
    parser.add_argument(
        "--window-policy",
        choices=["reference", "cap"],
        default="reference",
        help="Context policy for word_information_value: cap trims to the "
        "longest whole-word suffix fitting the model window (inert when "
        "everything fits; the reference errors on longer documents).",
    )
    parser.add_argument(
        "--segmentation",
        choices=["whitespace", "layer"],
        default="whitespace",
        help="Word iteration for word_information_value: layer uses the "
        "corpus's own word layer with no inserted spaces (unspaced scripts).",
    )
    parser.add_argument(
        "--import",
        dest="imports",
        action="append",
        default=[],
        metavar="MODULE",
        help="Import a module before anything else runs, so the segmenters it "
        "registers with replica.segmentation.register_segmenter are available "
        "to --unit; repeatable.",
    )
    parser.add_argument(
        "--unit",
        default="words",
        help="Aggregation unit: words, characters, tokens, or the name of a "
        "registered segmentation (replica.segmentation.SEGMENTERS), whose "
        "layer is attached before scoring and named in the cell id as "
        "_seg-<name>. For the unit-sensitive methods, characters "
        "attaches a per-character layer and emits one value per character; "
        "tokens stores one value per model token for the positional methods "
        "(surprisal, the first-token entropies, the lens surprisals, "
        "continuation entropy, responsivity and the subword similarity "
        "forms), whose word tables are then derived from the token table. A "
        "token table stores log_p_observed in nats; a derived word table "
        "keeps that sum in nats beside surprisal in bits; entropies are in "
        "bits. The manifest identity names the unit of every key.",
    )
    parser.add_argument(
        "--unit-rule",
        default=None,
        metavar="NAME",
        help="Override a unit-bound method's boundary rule, recorded in the "
        "manifest. Defaults are whitespace_continuation for whitespace "
        "estimators, corpus_segment_lookup for unigram lookup, and "
        "spacy_sentence for sentence information value.",
    )
    parser.add_argument(
        "--stim-slice",
        default=None,
        metavar="START:END",
        help="Compute a half-open stimulus slice and write a shard cache.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=5,
        help="Checkpoint after this many newly processed stimuli.",
    )
    return parser


def _parse_stimulus_slice(
    parser: argparse.ArgumentParser,
    value: str | None,
) -> tuple[int, int] | None:
    if value is None:
        return None
    try:
        start_text, end_text = value.split(":", 1)
        return int(start_text), int(end_text)
    except ValueError:
        parser.error(f"--stim-slice must be START:END integers; got {value!r}")


def main(argv: list[str] | None = None) -> None:
    """Compute requested dataset-method cells or merge their completed shards.

    Args:
        argv: Command-line arguments; defaults to ``sys.argv[1:]``. A cell that
            raises is logged and execution continues with the next one.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "merge-shards":
        from .shards import main as merge_main

        merge_main(argv[1:])
        return
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)
    _check_unit(parser, args)
    methods = args.method or ["surprisal"]
    for method in methods:
        try:
            check_reproduction_opt_in(
                method, allow_reproduction=args.allow_reproduction
            )
        except ValueError as error:
            parser.error(str(error))
    output_directory = Path(args.out_dir) if args.out_dir else compute_dir()

    datasets: list[str] = []
    for requested in args.dataset:
        datasets.extend(expand_datasets(requested))
    datasets = list(dict.fromkeys(datasets))
    log.info(
        "[plan] %d dataset(s) x %d method(s) = %d cache files",
        len(datasets),
        len(methods),
        len(datasets) * len(methods),
    )

    stimulus_slice = _parse_stimulus_slice(parser, args.stim_slice)
    output_directory.mkdir(parents=True, exist_ok=True)
    manifest = args.run_manifest or output_directory / f"run-{uuid4().hex}.json"
    outcomes: list[ComputeOutcome] = []

    def write_run_manifest() -> None:
        manifest.parent.mkdir(parents=True, exist_ok=True)
        temp = manifest.with_suffix(".tmp")
        temp.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "requested": len(methods) * len(datasets),
                    "allow_incomplete": args.allow_incomplete,
                    "outcomes": [asdict(outcome) for outcome in outcomes],
                },
                default=str,
                indent=2,
            )
            + "\n"
        )
        temp.replace(manifest)

    write_run_manifest()
    for method in methods:
        for dataset_name in datasets:
            try:
                outcome = compute_for(
                    dataset_name,
                    method=method,
                    model_id=args.lm,
                    K=args.K,
                    n_samples=args.n_samples,
                    max_tokens=args.max_tokens,
                    alpha=args.alpha,
                    seed=args.seed,
                    out_dir=output_directory,
                    add_bos=args.add_bos,
                    force=args.force,
                    language_override=args.language,
                    chunk_size=args.chunk_size,
                    word_starters=(
                        ("\u0120",) if args.word_starters == "g-only" else ()
                    ),
                    checkpoint_every=max(1, args.checkpoint_every),
                    stim_slice=stimulus_slice,
                    top_p=args.top_p,
                    context_scope=args.context_scope,
                    unit=args.unit,
                    unit_rule=args.unit_rule,
                    window_policy=args.window_policy,
                    segmentation=args.segmentation,
                    model_revision=args.model_revision,
                    store_distribution=args.store_distribution,
                    allow_reproduction=args.allow_reproduction,
                    context_window=args.context_window,
                    start_ids=tuple(args.start_ids)
                    if args.start_ids is not None
                    else None,
                    dtype=args.dtype,
                    device_map=args.device_map,
                )
            except Exception as exc:
                log.exception("[fail] %s / %s", dataset_name, method)
                outcome = ComputeOutcome(
                    dataset_name, method, "failed", error=f"{type(exc).__name__}: {exc}"
                )
            outcomes.append(outcome)
            write_run_manifest()
    if not args.allow_incomplete and any(
        not outcome.successful for outcome in outcomes
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
