"""Command-line interface of the predictor store: ``replica predictors``.

Subcommands:
  * ``export``   stage a release from compute output directories
  * ``verify``   read a staged tree back against the pickles
  * ``publish``  upload a staged tree as a pull request
  * ``accept``   publish a quarantined cell after its investigation
"""

from __future__ import annotations

import argparse
import importlib
import logging
import sys
from pathlib import Path

from .cells import parse_source
from .hub import DEFAULT_REPO, Hub


def build_parser() -> argparse.ArgumentParser:
    """Build the ``replica predictors`` parser."""
    parser = argparse.ArgumentParser(
        prog="replica predictors",
        description=__doc__.split("\n", 1)[0],
    )
    parser.add_argument(
        "--import",
        dest="imports",
        action="append",
        default=[],
        metavar="MODULE",
        help="Import a module first, so the segmentations it registers are "
        "attached when cells at their grain are checked; repeatable.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_export = sub.add_parser("export", help="stage a release from output directories")
    p_export.add_argument("--out", type=Path, required=True, help="stage directory")
    p_export.add_argument(
        "--from",
        dest="sources",
        action="append",
        required=True,
        metavar="[LABEL=]DIR",
        help="output directory of one run; repeatable. The label names the "
        "wave; without one, the first ancestor outside out/, out_mila/, "
        "verify/, config/ and logs/ names it.",
    )
    p_export.add_argument(
        "--verification",
        action="append",
        default=[],
        metavar="[LABEL=]JSON",
        help="advisory verification labels from one run; repeatable, not a readback receipt",
    )
    p_export.add_argument(
        "--coverage",
        action="append",
        default=[],
        type=Path,
        metavar="TSV",
        help="coverage table of planned commands; repeatable",
    )
    p_export.add_argument(
        "--merge-hub",
        action="store_true",
        help="carry over the published cells this run does not replace",
    )
    p_export.add_argument(
        "--repo",
        default=None,
        help=f"hub repository to merge, with --merge-hub; {DEFAULT_REPO} by default",
    )
    p_export.add_argument(
        "--drop-misaligned",
        action="store_true",
        help="retire the carried cells whose rows index other units than the "
        "corpus loader produces, instead of failing the export on them",
    )
    p_export.add_argument(
        "--retire",
        type=Path,
        default=None,
        metavar="TSV",
        help="retire the carried cells this file lists, one tag and its reason "
        "per line separated by a tab; the reasons are recorded in the report",
    )
    p_export.add_argument(
        "--revision",
        default=None,
        help="hub commit to merge, with --merge-hub; the head by default",
    )

    p_verify = sub.add_parser(
        "verify", help="compare every fresh source pickle and write a readback receipt"
    )
    p_verify.add_argument("--stage", type=Path, required=True)
    p_verify.add_argument(
        "--from",
        dest="sources",
        action="append",
        required=True,
        metavar="[LABEL=]DIR",
        help="output directory the stage was exported from; repeatable",
    )

    p_publish = sub.add_parser(
        "publish", help="upload a stage with a matching readback receipt"
    )
    p_publish.add_argument("--stage", type=Path, required=True)
    p_publish.add_argument("--repo", default=DEFAULT_REPO)
    mode = p_publish.add_mutually_exclusive_group(required=True)
    mode.add_argument("--create-pr", action="store_true", help="open a pull request")
    mode.add_argument("--direct", action="store_true", help="commit to main")
    p_publish.add_argument("--message", default=None, help="commit message")
    p_publish.add_argument(
        "--allow-failures",
        action="store_true",
        help="allow export failures; a matching successful readback is still required",
    )
    p_publish.add_argument(
        "--initial-release",
        action="store_true",
        help="upload a stage exported without --merge-hub; its index and shared "
        "tables replace the repository's",
    )

    p_accept = sub.add_parser(
        "accept", help="publish a quarantined cell after its investigation"
    )
    p_accept.add_argument("path", type=Path, help="the cell's <tag>.suspect.pkl")
    p_accept.add_argument(
        "--reason",
        required=True,
        help="what the investigation found; recorded under audit.accepted",
    )
    p_stamp = sub.add_parser(
        "stamp", help="write the manifest of a cell another pipeline produced"
    )
    p_stamp.add_argument("path", type=Path, help="the cell's <tag>.pkl")
    p_stamp.add_argument(
        "--model-id", required=True, help="exact model or frequency-source identity"
    )
    p_stamp.add_argument(
        "--input",
        action="append",
        default=[],
        type=Path,
        help="a file the payload was derived from; recorded with its digest",
    )
    p_stamp.add_argument(
        "--reason",
        required=True,
        help="what produced the payload; recorded under external",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run one ``replica predictors`` subcommand.

    Args:
        argv: Arguments after the subcommand name; ``sys.argv`` when ``None``.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    for module in args.imports:
        importlib.import_module(module)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.command == "export":
        from .export import export, read_retirements

        if not args.merge_hub and (args.repo or args.revision):
            parser.error(
                "--repo and --revision name the store to merge; add --merge-hub"
            )
        if args.drop_misaligned and not args.merge_hub:
            parser.error("--drop-misaligned retires carried cells; add --merge-hub")
        if args.retire is not None and not args.merge_hub:
            parser.error("--retire retires carried cells; add --merge-hub")
        hub = Hub(args.repo or DEFAULT_REPO, args.revision) if args.merge_hub else None
        report = export(
            args.out,
            [parse_source(s) for s in args.sources],
            verifications=[parse_source(v) for v in args.verification],
            coverage=list(args.coverage),
            hub=hub,
            drop_misaligned=args.drop_misaligned,
            retire=read_retirements(args.retire) if args.retire else None,
        )
        print(report.summary())
        return 0 if report.ok else 1
    if args.command == "verify":
        from .verify import verify

        readback = verify(args.stage, [parse_source(s) for s in args.sources])
        print(readback.summary())
        return 0 if readback.ok else 1
    if args.command == "accept":
        from .accept import accept

        try:
            print(f"published {accept(args.path, args.reason)}")
        except (ValueError, FileExistsError, FileNotFoundError) as exc:
            print(f"refused: {exc}", file=sys.stderr)
            return 1
        return 0
    if args.command == "stamp":
        from .stamp import SUSPECT, stamp

        try:
            bound = stamp(
                args.path,
                model_id=args.model_id,
                inputs=list(args.input),
                reason=args.reason,
            )
        except (ValueError, FileExistsError, FileNotFoundError) as exc:
            print(f"refused: {exc}", file=sys.stderr)
            return 1
        state = "quarantined" if bound.name.endswith(SUSPECT) else "completed"
        print(f"{state} {bound}")
        return 0
    from .publish import publish

    result = publish(
        args.stage,
        args.repo,
        create_pr=bool(args.create_pr),
        message=args.message,
        allow_failures=args.allow_failures,
        initial_release=args.initial_release,
    )
    for key, value in result.items():
        print(f"{key}: {value}")
    return 0
