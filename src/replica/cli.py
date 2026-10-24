"""Console entry point for replica: ``replica <command>``.

Subcommands:
  * ``datasets``        list datasets, describe their loading options, or emit
    machine-readable facts for a docs build (``--facts``)
  * ``fetch <name>...`` download one or more datasets (``--all`` / ``--force``)
  * ``prepare-fixations`` convert downloaded MECO fixation RDAs to lazy Parquet
  * ``check``           run the repository's lint, architecture, typing and
    dataset-metadata checks
  * ``compute ...``     compute LM predictor pickles (forwards to
    :mod:`replica.compute`)
  * ``results ...``     validate, stamp and check the tracked study results
    (forwards to :mod:`replica.studies.records.cli`)
  * ``predictors ...``  export, verify and publish the predictor store
    (forwards to :mod:`replica.predictor_store.cli`)
  * ``start-state M --texts PATH``   compare candidate start tokens
    (forwards to :mod:`replica.feature.diagnostics.start_state`)
  * ``version``         print the installed version

Wired as the ``replica`` console script in ``pyproject.toml``. The fetch
logic lives in :mod:`replica.dataset.fetch` so it works for both a
source checkout and a ``pip install``.
"""

import argparse
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version

from .paths import checkout_root

# Flags ``datasets --facts`` forwards to the facts entry point.
_FACTS_FLAGS = ("measure", "write_snapshot", "json", "only_missing")


def _run_dataset_facts(args: argparse.Namespace) -> None:
    """Forward the requested ``--facts`` flags to the dataset facts CLI."""
    from replica.dataset.catalog.facts import main as facts_main

    forwarded = [
        f"--{name.replace('_', '-')}"
        for name in _FACTS_FLAGS
        if getattr(args, name, False)
    ]
    status = facts_main(forwarded)
    if status:
        raise SystemExit(status)


def _refuse_facts_flags_without_facts(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> None:
    """Usage error for a flag that modifies ``--facts`` passed without it."""
    stray = [
        f"--{name.replace('_', '-')}" for name in _FACTS_FLAGS if getattr(args, name)
    ]
    if stray and not args.facts:
        parser.error(f"{', '.join(stray)}: only valid with --facts")


def _run_developer_checks(*, fix: bool) -> None:
    """Run the same source-quality gates enforced in CI."""
    try:
        repo_root = checkout_root()
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from None
    lint_command = [sys.executable, "-m", "ruff", "check", "."]
    format_command = [sys.executable, "-m", "ruff", "format"]
    if fix:
        lint_command.append("--fix")
        format_command.append(".")
    else:
        format_command.extend(["--check", "."])

    commands = [
        lint_command,
        format_command,
        [sys.executable, "scripts/checks/source/references.py"],
        [sys.executable, "scripts/checks/source/test_policy.py"],
        [sys.executable, "scripts/checks/source/import_boundaries.py"],
        [sys.executable, "scripts/checks/source/docstrings.py"],
        [sys.executable, "-m", "mypy"],
        [sys.executable, "scripts/checks/distribution/dataset_metadata.py"],
    ]
    try:
        for command in commands:
            subprocess.run(command, check=True, cwd=repo_root)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode) from None


def _build_parser() -> tuple[
    argparse.ArgumentParser, argparse.ArgumentParser, argparse.ArgumentParser
]:
    """Return ``(top_parser, fetch_subparser, datasets_subparser)``."""
    parser = argparse.ArgumentParser(
        prog="replica",
        description="REPLICA command-line interface.",
    )
    sub = parser.add_subparsers(dest="command")

    p_datasets = sub.add_parser(
        "datasets",
        help="List datasets, or describe how to load one with --describe NAME",
    )
    dataset_mode = p_datasets.add_mutually_exclusive_group()
    dataset_mode.add_argument(
        "--describe",
        metavar="NAME",
        help="Show loader options and variants for a dataset key or group",
    )
    dataset_mode.add_argument(
        "--loadable",
        action="store_true",
        help="List all loadable catalog keys (rather than download status)",
    )
    dataset_mode.add_argument(
        "--facts",
        action="store_true",
        help="Emit machine-readable facts about every dataset (for docs builds)",
    )
    p_datasets.add_argument(
        "--measure",
        action="store_true",
        help="With --facts: load each corpus and count it (needs the data)",
    )
    p_datasets.add_argument(
        "--write-snapshot",
        action="store_true",
        help="With --facts --measure: refresh the committed counts snapshot",
    )
    p_datasets.add_argument(
        "--json",
        action="store_true",
        help="With --facts: emit the full facts as JSON instead of a summary",
    )
    p_datasets.add_argument(
        "--only-missing",
        action="store_true",
        help="With --facts --measure: skip corpora the snapshot already counts",
    )

    p_fetch = sub.add_parser("fetch", help="Download one or more datasets")
    p_fetch.add_argument(
        "datasets",
        nargs="*",
        help="Dataset(s) to fetch (omit and pass --all to fetch every dataset)",
    )

    p_prepare = sub.add_parser(
        "prepare-fixations",
        help="Prepare downloaded MECO fixation data for lazy access",
    )
    p_prepare.add_argument(
        "datasets",
        nargs="+",
        help="MECO release(s): meco_l1_w1, meco_l1_w2, meco_l2_w1, meco_l2_w2",
    )
    p_prepare.add_argument(
        "--force", action="store_true", help="Rebuild an existing Parquet cache"
    )
    p_fetch.add_argument(
        "--all", action="store_true", help="Fetch all downloadable datasets"
    )
    p_fetch.add_argument(
        "--list",
        action="store_true",
        dest="list_",
        help="List datasets and their status (alias for `replica datasets`)",
    )
    p_fetch.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if files already exist",
    )
    p_fetch.add_argument(
        "--regenerate-yaml",
        action="store_true",
        dest="regenerate_yaml",
        help="(maintainers) recompute the shipped checksum baseline from the "
        "download instead of verifying the download against it",
    )

    p_check = sub.add_parser(
        "check",
        help="Run repository lint and formatting checks",
    )
    p_check.add_argument(
        "--fix",
        action="store_true",
        help="Apply safe lint fixes and formatting instead of checking",
    )

    # Help-listing stubs only: ``main`` forwards these commands verbatim
    # before argparse runs, so flags like ``--help`` reach the delegate parser.
    sub.add_parser(
        "compute",
        help="Compute LM predictor pickles (replica.compute)",
        add_help=False,
    )
    sub.add_parser(
        "results",
        help="Validate, stamp and check tracked study results (studies.records.cli)",
        add_help=False,
    )
    sub.add_parser(
        "predictors",
        help="Export, verify and publish the predictor store (predictor_store)",
        add_help=False,
    )

    sub.add_parser(
        "start-state",
        help="Compare candidate start tokens on supplied texts",
        add_help=False,
    )
    sub.add_parser("version", help="Print the installed version")
    return parser, p_fetch, p_datasets


#: Subcommands with their own parsers: module, entry point, and whether the
#: entry point's return value is the exit status.
_FORWARDED: dict[str, tuple[str, str, bool]] = {
    "compute": ("replica.compute.cli", "main", False),
    "results": ("replica.studies.records.cli", "main", True),
    "predictors": ("replica.predictor_store.cli", "main", True),
    "start-state": ("replica.feature.diagnostics.start_state", "main", True),
}


def _forward(command: str, argv: list[str]) -> None:
    """Run a forwarded subcommand's own entry point on *argv*."""
    import importlib

    module, entry, returns_status = _FORWARDED[command]
    result = getattr(importlib.import_module(module), entry)(argv)
    if returns_status:
        raise SystemExit(result)


def main(argv: list[str] | None = None) -> None:
    """Parse and dispatch a ``replica`` subcommand.

    Args:
        argv: Command-line arguments excluding the program name, defaulting to
            ``sys.argv[1:]``.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in _FORWARDED:
        _forward(argv[0], argv[1:])
        return

    parser, p_fetch, p_datasets = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "datasets":
        _refuse_facts_flags_without_facts(args, p_datasets)
        if args.describe:
            from replica.dataset.catalog.description import print_dataset_description

            try:
                print_dataset_description(args.describe)
            except KeyError as exc:
                parser.error(str(exc))
        elif args.loadable:
            from replica.dataset.catalog.description import available_datasets

            print("\nLoadable dataset keys:\n")
            for name in available_datasets():
                print(f"  {name}")
            print()
        elif args.facts:
            _run_dataset_facts(args)
        else:
            from .dataset.fetch import list_datasets

            list_datasets()
    elif args.command == "fetch":
        from .dataset.fetch import run_fetch

        run_fetch(
            names=args.datasets,
            fetch_all=args.all,
            list_only=args.list_,
            force=args.force,
            regenerate_yaml=args.regenerate_yaml,
            on_empty=p_fetch.print_help,
        )
    elif args.command == "version":
        try:
            print(version("replicaverse"))
        except PackageNotFoundError:
            print("unknown (not installed)")
    elif args.command == "check":
        _run_developer_checks(fix=args.fix)
    elif args.command == "prepare-fixations":
        from .dataset.eye_tracking.meco.prepare import run_prepare

        run_prepare(args.datasets, force=args.force)
    else:
        parser.print_help()
        raise SystemExit(1)


if __name__ == "__main__":
    main()
