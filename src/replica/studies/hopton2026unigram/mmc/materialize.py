"""Write each dataset's pickle as a lookup against the merged table."""

import argparse
import json
import pickle
from pathlib import Path

from replica.compute import cache
from replica.dataset.catalog.registry import REGISTRY
from replica.feature.frequency import UnigramSurprisalLM
from replica.feature.frequency.registry import SOURCES, pickle_prefix

from .common import _POOLED, _version_for


class _TableSource:
    """Serves a finished type -> p table; every lookup must be present."""

    language = "en"
    case_sensitive = True

    def __init__(self, table: dict[str, float]):
        self._table = table

    def probability(self, unit: str) -> float:
        try:
            probability = self._table[unit]
        except KeyError:
            raise KeyError(
                f"{unit!r} missing from the type table; the `words` stage did not "
                "cover this dataset"
            ) from None
        return float(probability) if probability > 0 else float("nan")


def cmd_materialize(args: argparse.Namespace) -> int:
    """Write each dataset's pickle as a lookup against the merged table.

    Args:
        args: Parsed `materialize` arguments: `source`, `model_name`, the
            merged `table`, the `datasets` to write pickles for, the
            `out_dir` to write them to, an optional `expect_fingerprint` to
            enforce, and the pooled tag's `n_samples`, `k_eval`, and `seed`.
    """
    table = json.loads(args.table.read_text(encoding="utf-8"))
    _check_expected_fingerprint(args)
    _null_refused(table, args.table)
    model = UnigramSurprisalLM(_TableSource(table), lookup=SOURCES[args.source].lookup)
    prefix = _pickle_prefix(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    status = 0
    for name in args.datasets:
        try:
            dataset = REGISTRY[name].factory()
            outputs = {s.stimulus_id: model(s) for s in dataset.stimuli()}
            path = args.out_dir / f"{prefix}_{name}.pkl"
            with open(path, "wb") as handle:
                pickle.dump(outputs, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"  {name:<30} {len(outputs):>6} stim -> {path.name}", flush=True)
        except Exception as error:  # one bad dataset must not sink the rest
            print(f"  {name:<30} FAILED: {type(error).__name__}: {error}", flush=True)
            status = 1
    return status


def _null_refused(table: dict, table_path: Path) -> None:
    """Serve a refused word as missing rather than as its floor probability.

    A record the certificate refuses outright carries the float64 floor so the
    merge accepts it; as a covariate that floor is ~1074 bits of surprisal on
    every passage-initial row that looks the word up. NaN routes those rows
    through the missing-value path instead, which is what ``_TableSource``
    already does for a non-positive probability.
    """
    sidecar = table_path.with_name(table_path.stem + ".records.jsonl")
    if not sidecar.exists():
        return
    nulled = 0
    for line in sidecar.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        record = json.loads(line)
        if record.get("nonpositive") and record["word"] in table:
            table[record["word"]] = float("nan")
            nulled += 1
    if nulled:
        print(f"[materialize] {nulled} refused words served as missing", flush=True)


def _pickle_prefix(args) -> str:
    """The filename prefix; the pooled source encodes N, k and the seed base."""
    if args.source not in _POOLED:
        return pickle_prefix(args.source, args.model_name)
    from replica.utils.naming import safe_lm

    method_key = SOURCES[args.source].method_key
    if method_key is None:
        raise SystemExit(f"[materialize] {args.source} declares no compute method")
    tag = cache.method_cache_tag(
        method_key,
        n_samples=args.n_samples,
        k_eval=args.k_eval,
        seed=args.seed,
    )
    return f"{safe_lm(args.model_name)}_{tag}"


def _check_expected_fingerprint(args) -> None:
    """The published tag encodes N, k and the seed, so they must be the run's.

    The sidecar the merge wrote carries what was actually estimated; the tag is
    built from the command line, and a mismatch would name the wrong cell.
    """
    if args.source not in _POOLED and not args.expect_fingerprint:
        return
    sidecar = args.table.with_name(args.table.stem + ".meta.json")
    if not sidecar.exists():
        raise SystemExit(f"[materialize] no sidecar at {sidecar} to check the pool")
    meta = json.loads(sidecar.read_text(encoding="utf-8"))
    got = meta.get("pool_fingerprint")
    if args.expect_fingerprint and got != args.expect_fingerprint:
        raise SystemExit(
            f"[materialize] the table names pool {got}, not {args.expect_fingerprint}"
        )
    if args.source not in _POOLED:
        return
    for key, seen in (("n_pool_samples", args.n_samples), ("k_eval", args.k_eval)):
        if meta.get(key) is not None and meta[key] != seen:
            raise SystemExit(
                f"[materialize] the table was built with {key}={meta[key]} but the "
                f"tag would say {seen}; fix the command line, not the tag"
            )
    # The tag names the estimator family, N, k and the seed, so a table has to
    # publish under the source whose numerator produced it.
    built = meta.get("estimator_version")
    expected = _version_for(args.source)
    if built is not None and built != expected:
        raise SystemExit(
            f"[materialize] the table was estimated by {built} but --source "
            f"{args.source} publishes {expected} tables; they do not share a cell id"
        )
