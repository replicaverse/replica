"""Validate, locate and manage indexed study results through ``replica results``."""

import argparse
import json
import sys
from pathlib import Path

from .files import (
    _DIGEST,
    produced_by_now,
    read_index,
    results_file,
    run_dir,
    stale_runs,
    stamp,
    studies_with_results,
    unstamped_runs,
    validate_index,
)
from .layout import ROLES


def _parse_inputs(pairs: list[str]) -> dict[str, str]:
    inputs = {}
    for pair in pairs:
        label, _, digest = pair.partition("=")
        if not label or not _DIGEST.match(digest):
            raise SystemExit(f"--input needs LABEL=SHA256, got {pair!r}")
        inputs[label] = digest
    return inputs


def main(argv: list[str] | None = None) -> int:
    """``replica results``: check, stamp and report the tracked runs.

    Args:
        argv: Arguments after ``results``; ``sys.argv[1:]`` when ``None``.
    """
    parser = argparse.ArgumentParser(prog="replica results", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p_stale = sub.add_parser("stale", help="list stamped runs the tree has outgrown")
    p_stale.add_argument("study", nargs="*", help="study packages; default all")
    p_validate = sub.add_parser("validate", help="check every index.json")
    p_validate.add_argument("study", nargs="*", help="study packages; default all")
    p_stamp = sub.add_parser("stamp", help="record what produced a run")
    p_stamp.add_argument("study")
    p_stamp.add_argument("run", help="run path relative to the study's results/")
    p_stamp.add_argument(
        "--manifest", type=Path, help="JSON file with a produced_by block to copy"
    )
    p_stamp.add_argument("--input", action="append", default=[], metavar="LABEL=SHA256")
    p_stamp.add_argument("--driver-commit", help="commit of an external driver repo")
    p_stamp.add_argument(
        "--reason", help="restamp unchanged files after a code change that keeps them"
    )
    p_stamp.add_argument(
        "--replace-producer",
        action="store_true",
        help="replace a recorded producer; without it only a null producer is set",
    )
    p_stamp.add_argument(
        "--depends-on",
        action="append",
        default=None,
        metavar="PATH",
        help="package-relative file the run depends on; default: the study's footprint",
    )
    p_save = sub.add_parser("save", help="copy a run's outputs into the store")
    p_save.add_argument("run_id")
    p_save.add_argument("--study", required=True)
    p_save.add_argument(
        "--from", dest="sources", type=Path, action="append", required=True
    )
    p_save.add_argument(
        "--command", dest="command_text", help="what produced the run, as text"
    )
    p_save.add_argument("--input", action="append", default=[], metavar="LABEL=SHA256")
    p_save.add_argument("--driver-commit", help="commit of an external driver repo")
    p_save.add_argument(
        "--depends-on",
        action="append",
        default=None,
        metavar="PATH",
        help="package-relative file the run depends on; default: the study's footprint",
    )
    p_verify = sub.add_parser("verify", help="check store runs against their manifests")
    p_verify.add_argument("run_id", nargs="+")
    sub.add_parser("list", help="summarize every store run")
    p_publish = sub.add_parser("publish", help="upload a store run and register it")
    p_publish.add_argument("run_id")
    p_publish.add_argument("--study", required=True)
    p_publish.add_argument(
        "--run", required=True, help="axis/stage/analysis/record in results/"
    )
    p_publish.add_argument("--role", required=True, choices=ROLES)
    p_publish.add_argument(
        "--description", required=True, help="calculation and input configuration"
    )
    p_publish.add_argument("--note", help="text for the index entry")
    p_fetch = sub.add_parser("fetch", help="download a run's hub files into the cache")
    p_fetch.add_argument("study", nargs="?", help="study package; default all")
    p_fetch.add_argument("run", nargs="?", help="one run; default every run")
    p_fetch.add_argument(
        "--verification",
        action="store_true",
        help="also fetch archives needed to verify the study's indexed results",
    )
    p_path = sub.add_parser(
        "path", help="resolve an indexed result file or run directory"
    )
    p_path.add_argument("study")
    p_path.add_argument("run")
    p_path.add_argument(
        "name", nargs="?", help="file within the run; default whole run"
    )
    args = parser.parse_args(argv)
    if args.command == "path":
        print(
            results_file(args.study, args.run, args.name)
            if args.name
            else run_dir(args.study, args.run)
        )
        return 0

    if args.command in {"save", "verify", "list", "publish", "fetch"}:
        try:
            return _store_command(args)
        except (RuntimeError, ValueError, OSError) as exc:
            raise SystemExit(f"replica results {args.command}: {exc}") from None
    if args.command == "stamp":
        try:
            return _stamp_command(args)
        except (ValueError, FileNotFoundError) as exc:
            raise SystemExit(f"replica results stamp: {exc}") from None

    studies = args.study or studies_with_results()
    digests: dict[Path, str] = {}
    failed = False
    for study in studies:
        if args.command == "validate":
            problems = validate_index(study)
            failed |= bool(problems)
            print(f"{study}: {'ok' if not problems else len(problems)}")
            for problem in problems:
                print(f"  {problem}")
            continue
        total = len(read_index(study)["runs"])
        unstamped = len(unstamped_runs(study))
        stale = stale_runs(study, digests)
        failed |= bool(stale)
        print(
            f"{study}: {total} runs, {total - unstamped} stamped, "
            f"{unstamped} unstamped, {len(stale)} stale"
        )
        for run, message in stale:
            print(f"  stale: {run} ({message})")
    return 1 if failed else 0


def _stamp_command(args: argparse.Namespace) -> int:
    if args.manifest:
        produced_by = json.loads(args.manifest.read_text()).get("produced_by")
        if not isinstance(produced_by, dict):
            raise ValueError(f"{args.manifest} has no produced_by block")
    else:
        produced_by = produced_by_now(
            args.study,
            _parse_inputs(args.input),
            args.driver_commit,
            depends_on=args.depends_on,
        )
    stamp(
        args.study,
        args.run,
        produced_by,
        reason=args.reason,
        replace=args.replace_producer,
    )
    print(f"stamped {args.study}/{args.run} at {produced_by['footprint'][:12]}")
    return 0


def _store_command(args: argparse.Namespace) -> int:
    from . import store

    if args.command == "save":
        run_dir = store.save(
            args.run_id,
            args.sources,
            study=args.study,
            command=args.command_text,
            inputs=_parse_inputs(args.input),
            driver_commit=args.driver_commit,
            depends_on=args.depends_on,
        )
        print(f"saved {run_dir}")
        return 0
    if args.command == "verify":
        failed = False
        for run_id in args.run_id:
            problems = store.verify(run_id)
            failed |= bool(problems)
            print(f"{run_id}: {'ok' if not problems else len(problems)}")
            for problem in problems:
                print(f"  {problem}")
        return 1 if failed else 0
    if args.command == "list":
        for summary in store.list_runs():
            stamp = "stamped" if summary["stamped"] else "unstamped"
            print(
                f"{summary['run_id']}  {summary['study']}  {summary['saved_utc']}  "
                f"{summary['files']} files  {stamp}"
            )
        return 0
    if args.command == "publish":
        entry = store.publish(
            args.run_id,
            study=args.study,
            run=args.run,
            role=args.role,
            note=args.note,
            description=args.description,
        )
        print(f"published {args.study}/{args.run}: {len(entry['hub'])} hub files")
        return 0
    for study in [args.study] if args.study else studies_with_results():
        for path in store.fetch(study, args.run, verification=args.verification):
            print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
