"""Audit and re-plan the compute pickles written by ``replica.compute``.

Two operations:

* :func:`audit_pickles`, walk a compute directory and classify every
  ``*.pkl`` file as ok / empty / truncated, without loading the full
  pickle. Detects leftovers from interrupted or OOM-killed array tasks
  before re-running them.

* :func:`build_cleanup_plan`, for a given cell manifest (the TSV
  produced by a predictor campaign) and an on-disk
  compute directory, compute which cells still need work (missing,
  empty, or truncated pickle).

Both operations are pure functions of their inputs and have no scheduler /
HF dependencies.

Pickle integrity heuristic
--------------------------
A non-empty Python pickle ends in the STOP opcode ``0x2e`` (``.``). Reading the
last byte avoids materialising a multi-GB pickle in RAM to find it is corrupt.

A corrupt file whose last byte happens to be ``0x2e`` passes. ``deep=True``
calls :func:`pickle.load` on each candidate for full validation.
"""

import os
import pickle
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------
# Pickle integrity
# --------------------------------------------------------------------------

_STOP_OPCODE = b"."
_MIN_NONEMPTY_BYTES = 6  # pickle.dumps({}) is 5 bytes, anything shorter is empty


def is_valid_pickle(path: os.PathLike, *, deep: bool = False) -> bool:
    """Return True if *path* looks like a valid, non-empty pickle.

    By default uses a fast STOP-opcode check on the last byte of the
    file. Pass ``deep=True`` to additionally call :func:`pickle.load`
    and require the deserialised object to be a non-empty container.

    Args:
        path: File to inspect; a missing file counts as invalid.
        deep: Also load the pickle and reject one that raises or that
            deserialises to an empty container, at the cost of reading the
            whole file into memory.
    """
    p = Path(path)
    try:
        size = p.stat().st_size
    except FileNotFoundError:
        return False
    if size < _MIN_NONEMPTY_BYTES:
        return False
    with open(p, "rb") as f:
        f.seek(-1, os.SEEK_END)
        last = f.read(1)
    if last != _STOP_OPCODE:
        return False
    if not deep:
        return True
    try:
        with open(p, "rb") as f:
            obj = pickle.load(f)
    except Exception:
        # Truncated/corrupt pickles raise varied types (ValueError, KeyError,
        # AttributeError, ModuleNotFoundError, ...); classify as invalid.
        return False
    if isinstance(obj, (dict, list, tuple, set)):
        return len(obj) > 0
    return True


# --------------------------------------------------------------------------
# Audit
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class AuditReport:
    """Classification of a set of pickle files.

    Attributes:
        ok: Paths whose last byte is the pickle STOP opcode.
        empty: Paths whose size is below the empty-dict threshold
            (``pickle.dumps({})`` is 5 bytes).
        truncated: Paths large enough to look populated but whose last
            byte is not the STOP opcode, almost certainly an
            interrupted ``pickle.dump`` from a killed process.
    """

    ok: tuple[Path, ...] = field(default_factory=tuple)
    empty: tuple[Path, ...] = field(default_factory=tuple)
    truncated: tuple[Path, ...] = field(default_factory=tuple)

    @property
    def total(self) -> int:
        """Number of files classified, across all three outcomes."""
        return len(self.ok) + len(self.empty) + len(self.truncated)


def audit_pickles(
    compute_dir: os.PathLike,
    *,
    deep: bool = False,
    pattern: str = "*.pkl",
) -> AuditReport:
    """Classify every pickle in *compute_dir* as ok / empty / truncated.

    Args:
        compute_dir: Directory containing ``.pkl`` files (e.g. the
            result of ``compute_dir()``).
        deep: If True, do a full :func:`pickle.load` on candidates that
            pass the STOP-byte check. Slower but rejects pickles that
            happen to end with ``0x2e`` but are otherwise corrupt.
        pattern: Glob pattern relative to ``compute_dir``.
    """
    root = Path(compute_dir)
    ok: list[Path] = []
    empty: list[Path] = []
    truncated: list[Path] = []
    for p in sorted(root.glob(pattern)):
        try:
            sz = p.stat().st_size
        except FileNotFoundError:
            continue
        if sz < _MIN_NONEMPTY_BYTES:
            empty.append(p)
            continue
        if is_valid_pickle(p, deep=deep):
            ok.append(p)
        else:
            truncated.append(p)
    return AuditReport(ok=tuple(ok), empty=tuple(empty), truncated=tuple(truncated))


# --------------------------------------------------------------------------
# Manifest parsing + tag derivation
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Cell:
    r"""One cell from a manifest line.

    Fields use the campaign TSV format:
    ``tier=<n>\\tdataset=<x>\\tlm=<y>\\tmethod=<z>\\textra=<args>``.
    """

    tier: str
    dataset: str
    lm: str
    method: str
    extra: str = ""


def parse_manifest(lines: Iterable[str]) -> Iterator[Cell]:
    """Parse a predictor campaign manifest.

    Each non-empty line is split on tabs; each field is ``key=value``.
    The ``tier``, ``dataset``, ``lm``, and ``method`` fields are
    required. ``extra`` defaults to the empty string if absent.

    Args:
        lines: Manifest lines, as read from the file; blank lines and
            tab-separated fields without an ``=`` are skipped.

    Returns:
        One :class:`Cell` per manifest line, in file order.

    Raises:
        ValueError: if a required field is missing.
    """
    for raw in lines:
        line = raw.rstrip("\n")
        if not line:
            continue
        kv = {}
        for field_str in line.split("\t"):
            if "=" not in field_str:
                continue
            k, _, v = field_str.partition("=")
            kv[k] = v
        required = ("tier", "dataset", "lm", "method")
        missing = [k for k in required if k not in kv]
        if missing:
            raise ValueError(
                f"Manifest line missing required fields {missing!r}: {line!r}"
            )
        yield Cell(
            tier=kv["tier"],
            dataset=kv["dataset"],
            lm=kv["lm"],
            method=kv["method"],
            extra=kv.get("extra", ""),
        )


def pickle_path_for(cell: Cell, compute_dir: os.PathLike) -> Path:
    """Return the canonical on-disk pickle path for ``cell``.

    The cell's ``extra`` string is parsed by the compute CLI's own parser and
    the name comes from the runner's :func:`~replica.compute.runner.cache_stem`,
    so a flag the cell omits takes the CLI default and ``--context-scope auto``
    resolves as the runner resolves it.

    Args:
        cell: Manifest cell naming the dataset, model, method, and extra flags.
        compute_dir: Root the cache path is built under.

    Raises:
        ValueError: When ``extra`` holds a flag or value the compute CLI rejects.
    """
    import argparse

    from .cache import qualified_model_id, resolve_context_scope
    from .cli import build_parser
    from .methods import ComputeContext
    from .runner import cache_stem

    parser = build_parser()
    parser.exit_on_error = False
    try:
        args, unknown = parser.parse_known_args(
            ["--dataset", cell.dataset, "--lm", cell.lm, *cell.extra.split()]
        )
    except argparse.ArgumentError as error:
        raise ValueError(f"{cell.extra!r}: {error}") from None
    if unknown:
        raise ValueError(f"{cell.extra!r}: not compute flags: {unknown}")
    context = ComputeContext(
        model_id=qualified_model_id(cell.lm, args.language),
        beam_width=args.K,
        n_samples=args.n_samples,
        max_tokens=args.max_tokens,
        alpha=args.alpha,
        seed=args.seed,
        add_bos=args.add_bos,
        dataset_language=None,
        language_override=args.language,
        word_starters=("\u0120",) if args.word_starters == "g-only" else (),
        top_p=args.top_p,
        context_scope=resolve_context_scope(
            cell.dataset, cell.method, args.context_scope
        ),
        unit=args.unit,
        window_policy=args.window_policy,
        segmentation=args.segmentation,
    )
    return Path(compute_dir) / f"{cache_stem(cell.method, context, cell.dataset)}.pkl"


# --------------------------------------------------------------------------
# Cleanup manifest
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CleanupPlan:
    """Result of comparing a manifest to an on-disk compute directory.

    ``need_redo`` lists the cells whose pickle is missing, empty, or
    truncated, the focused set that should be resubmitted with
    ``--force``. ``ok`` is the cells whose pickle is already valid.
    """

    need_redo: tuple[Cell, ...]
    ok: tuple[Cell, ...]
    reasons: tuple[str, ...]  # parallel to need_redo: "missing"/"empty"/"truncated"

    @property
    def total(self) -> int:
        """Number of manifest cells examined, redo and ok together."""
        return len(self.need_redo) + len(self.ok)


def build_cleanup_plan(
    cells: Iterable[Cell],
    compute_dir: os.PathLike,
    *,
    deep: bool = False,
) -> CleanupPlan:
    """For each cell, check its expected pickle on disk and classify it.

    Args:
        cells: Iterable of :class:`Cell` (e.g. ``parse_manifest(...)``).
        compute_dir: Where to look for output pickles.
        deep: Forwarded to :func:`is_valid_pickle`.
    """
    need: list[Cell] = []
    reasons: list[str] = []
    ok: list[Cell] = []
    for c in cells:
        p = pickle_path_for(c, compute_dir)
        try:
            sz = p.stat().st_size
        except FileNotFoundError:
            need.append(c)
            reasons.append("missing")
            continue
        if sz < _MIN_NONEMPTY_BYTES:
            need.append(c)
            reasons.append("empty")
            continue
        if not is_valid_pickle(p, deep=deep):
            need.append(c)
            reasons.append("truncated")
            continue
        ok.append(c)
    return CleanupPlan(need_redo=tuple(need), ok=tuple(ok), reasons=tuple(reasons))


def write_manifest(cells: Iterable[Cell], path: os.PathLike) -> int:
    """Write ``cells`` in the feature-sweep TSV format.

    Args:
        cells: Cells to write, one ``key=value`` tab-separated line each.
        path: File to write, truncated if it exists.

    Returns:
        The number of rows written, so a resubmission script can size its
        array job.
    """
    n = 0
    with open(path, "w") as f:
        for c in cells:
            f.write(
                f"tier={c.tier}\tdataset={c.dataset}\tlm={c.lm}\t"
                f"method={c.method}\textra={c.extra}\n"
            )
            n += 1
    return n
