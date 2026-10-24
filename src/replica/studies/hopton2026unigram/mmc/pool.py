"""Draw the sample pool and merge the shards into one token store."""

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple, cast

import numpy as np

from replica.feature.frequency import pooled as pooled_mod
from replica.feature.frequency import sampled as sampled_mod
from replica.feature.frequency.registry import SOURCES

if TYPE_CHECKING:
    from replica.feature.frequency.sampled import SampledUnigramSource

from replica.utils.digest import file_sha256, text_sha256

from . import common
from .common import _write_npz, _write_text

#: Identity of the token store's encoding, bound into the pool fingerprint.
POOL_FORMAT = "replica/mmc-pool/2"
#: The layout of a pool whose rows are re-encoded text, kept for replays.
POOL_FORMAT_REENCODED = "replica/mmc-pool/1"


def cmd_samples(args: argparse.Namespace) -> int:
    """Draw and persist one sample shard for one LM.

    Args:
        args: Parsed `samples` arguments: `model_name`, `device`, `seed_base`,
            the shard's `shard_index` and `shard_samples` count, the `out`
            path to write the shard to, and an optional `ids_out` path for the
            sidecar of the token ids the model drew (`ids` flat, `offsets`
            per sample, `start` the one-token prefix, `request` the sampling
            identity with its generation contract), which the pool reads
            instead of re-encoding the text;
            `start_ids` names that prefix explicitly, `None` keeps the
            tokenizer's convention.
    """
    start_ids = getattr(args, "start_ids", None)
    if args.ids_out is None and start_ids is None and args.out.exists():
        # The invocation older pools were drawn with: the text alone completes
        # the shard, and nothing is resolved to check it.
        print(f"[samples] {args.out} exists, skipping", flush=True)
        return 0
    seed = args.seed_base + (args.shard_index or 0)
    source = cast(
        "SampledUnigramSource",
        SOURCES["mmc"].build(
            model_name=args.model_name,
            device=args.device,
            n_samples=args.shard_samples or args.n_samples,
            seed=seed,
            start_ids=start_ids,
        ),
    )
    common._configure_sampler(source)
    request = _sampling_request(source)
    if _samples_done(args.out, args.ids_out, request, start_ids is not None):
        return 0
    source._ensure()
    samples = source._samples
    if samples is None:
        raise SystemExit("[samples] the sampler produced no samples")
    text = json.dumps(samples, ensure_ascii=False)
    # The ids go first: the text file is the completion marker, and a shard
    # whose text exists without its ids cannot be completed, since the drawn
    # ids are not recoverable from the decoded text.
    if args.ids_out is not None:
        rows = source._sample_ids
        if rows is None or len(rows) != len(samples):
            raise SystemExit("[samples] the sampler kept no ids for its samples")
        offsets = np.zeros(len(rows) + 1, dtype=np.int64)
        offsets[1:] = np.cumsum([len(r) for r in rows])
        flat = np.array([i for r in rows for i in r], dtype=np.int32)
        digest = np.array(text_sha256(text), dtype="U64")
        start = np.array(request["start"], dtype=np.int64)
        record = np.array(json.dumps(request, sort_keys=True))
        _write_npz(
            args.ids_out,
            lambda tmp: np.savez(
                tmp,
                ids=flat,
                offsets=offsets,
                text_sha256=digest,
                start=start,
                request=record,
            ),
        )
    _write_text(args.out, text)
    print(f"[samples] {len(samples)} seed={seed} -> {args.out}", flush=True)
    return 0


#: The sampler settings a sidecar records, so a resume is matched against the
#: request that produced the shard and not only against its text.
REQUEST_FIELDS = ("model_name", "seed", "n_samples", "max_len", "gen_batch_size")


def _sampling_request(source) -> dict:
    """The identity of a sampling request: the producer's settings and prefix.

    Args:
        source: The sampled source as built for the request; only its
            tokenizer is loaded here.
    """
    request = {field: getattr(source, field) for field in REQUEST_FIELDS}
    request["start"] = pooled_mod.start_prefix(
        source._ensure_tokenizer(), source.start_ids
    )
    request["generation"] = sampled_mod.GENERATION_CONTRACT
    return request


def _samples_done(
    out: Path, ids_out: "Path | None", request: dict, explicit_prefix: bool
) -> bool:
    """Whether a shard is complete for this request.

    With a sidecar requested, the shard is complete when the sidecar is bound
    to the text and records the same request. Without one, the text alone
    completes the shard only for a request without an explicit prefix, the
    invocation older pools were drawn with; its identity cannot be verified
    and is not claimed.

    Args:
        out: The shard's text file.
        ids_out: The requested sidecar, or ``None`` when none was asked for.
        request: The sampling identity of this request, see
            :func:`_sampling_request`.
        explicit_prefix: Whether the request names its prefix.

    Raises:
        SystemExit: The text exists without the requested sidecar, the sidecar
            does not describe the text's samples, it records another request,
            or the text alone cannot be matched to an explicit prefix; none of
            these can be repaired from the text, so the shard has to be
            redrawn.
    """
    if not out.exists():
        return False
    if ids_out is None:
        if explicit_prefix:
            raise SystemExit(
                f"[samples] {out} exists without a sidecar, so the prefix it was "
                "drawn after cannot be verified against the explicit one requested; "
                "remove the shard to redraw it, or request --ids-out"
            )
        print(f"[samples] {out} exists, skipping", flush=True)
        return True
    if not ids_out.exists():
        raise SystemExit(
            f"[samples] {out} exists without its ids at {ids_out}; the drawn "
            "ids are not recoverable from the text, so remove the shard to "
            "redraw it"
        )
    text = out.read_text(encoding="utf-8")
    try:
        sidecar = _read_sidecar(ids_out, len(_sample_list(text, out)), text)
    except ValueError as exc:
        raise SystemExit(f"[samples] {exc}; remove the shard to redraw it") from exc
    if sidecar.binding != "digest" or sidecar.request is None:
        raise SystemExit(
            f"[samples] {ids_out} is a legacy sidecar not bound to its text and "
            "request, so the pair cannot be verified; remove the shard to redraw it"
        )
    if sidecar.request != request:
        raise SystemExit(
            f"[samples] {ids_out} records the request {sidecar.request}, this one is "
            f"{request}; remove the shard to redraw it"
        )
    print(f"[samples] {out} exists, skipping", flush=True)
    return True


def _sample_list(text: str, path: Path) -> list[str]:
    """The samples a shard's text serialises, which has to be a list of strings.

    Raises:
        ValueError: The text is not JSON, or decodes to something else.
    """
    try:
        samples = json.loads(text)
    except ValueError as exc:
        raise ValueError(f"{path}: not a JSON shard ({exc})") from exc
    if not isinstance(samples, list) or not all(isinstance(s, str) for s in samples):
        raise ValueError(f"{path}: a shard is a JSON list of strings")
    return samples


class Sidecar(NamedTuple):
    """What a sidecar records: the drawn ids, their row offsets, its bindings."""

    ids: np.ndarray
    offsets: np.ndarray
    #: ``digest`` when the sidecar names the text it was drawn with, ``legacy``
    #: when it predates that.
    binding: str
    #: The one-token prefix the sidecar records, ``None`` when it records none.
    start: "tuple[int, ...] | None"
    #: The sampling request the sidecar records, ``None`` when it records none.
    request: "dict | None"


def _read_sidecar(path: Path, n_samples: int, text: "str | None" = None) -> Sidecar:
    """The arrays and bindings of a sidecar, checked against its samples.

    Args:
        path: The ``.ids.npz`` file.
        n_samples: How many samples the shard's text holds.
        text: The shard's serialised text; a sidecar that carries a text digest
            must match it.

    Raises:
        ValueError: The file is unreadable or lacks an array, the ids are not
            a one-dimensional array of non-negative integers, the offsets are
            not one-dimensional integers partitioning the ids into
            *n_samples* rows, the digest does not match *text*, the recorded
            prefix is not one integer, or the recorded request is not a JSON
            object.
    """
    try:
        with np.load(path) as blob:
            ids, offsets = blob["ids"], blob["offsets"]
            digest = str(blob["text_sha256"]) if "text_sha256" in blob.files else None
            start = blob["start"] if "start" in blob.files else None
            request = str(blob["request"]) if "request" in blob.files else None
    except Exception as exc:
        raise ValueError(f"{path}: unreadable sidecar ({type(exc).__name__})") from exc
    if start is not None and (
        start.ndim != 1
        or start.size != 1
        or start.dtype.kind not in "iu"
        or int(start[0]) < 0
    ):
        raise ValueError(
            f"{path}: the recorded start must be one non-negative integer id"
        )
    if ids.ndim != 1 or ids.dtype.kind not in "iu":
        raise ValueError(f"{path}: ids must be one-dimensional integers")
    if ids.size and ids.min() < 0:
        raise ValueError(f"{path}: ids must be non-negative")
    if offsets.ndim != 1 or offsets.dtype.kind not in "iu":
        raise ValueError(f"{path}: offsets must be one-dimensional integers")
    if ids.size and ids.max() > np.iinfo(np.int32).max:
        raise ValueError(f"{path}: ids exceed the int32 range of the token store")
    if offsets.size != n_samples + 1:
        raise ValueError(
            f"{path}: {max(offsets.size - 1, 0)} id rows for {n_samples} samples"
        )
    # Compared element-wise rather than through a difference, which wraps for
    # unsigned offsets.
    if (
        offsets[0] != 0
        or offsets[-1] != ids.size
        or np.any(offsets < 0)
        or np.any(offsets > ids.size)
        or np.any(offsets[1:] < offsets[:-1])
    ):
        raise ValueError(
            f"{path}: offsets must rise from 0 to the {ids.size} ids within bounds"
        )
    if digest is not None and text is not None and digest != text_sha256(text):
        raise ValueError(f"{path}: the sidecar was drawn with a different text")
    parsed = None
    if request is not None:
        try:
            parsed = json.loads(request)
        except ValueError as exc:
            raise ValueError(f"{path}: the recorded request is not JSON") from exc
        if not isinstance(parsed, dict):
            raise ValueError(f"{path}: the recorded request must be a JSON object")
        _check_request(parsed, start, offsets.size - 1, path)
    return Sidecar(
        ids,
        offsets,
        "digest" if digest is not None else "legacy",
        None if start is None else (int(start[0]),),
        parsed,
    )


def _check_request(request: dict, start, n_rows: int, path: Path) -> None:
    """A recorded request has every field, well typed, and agrees with the arrays.

    Raises:
        ValueError: A field is missing or of the wrong type, the request's
            sample count is not the sidecar's row count, or its start is not
            the recorded start.
    """
    missing = [k for k in (*REQUEST_FIELDS, "start") if k not in request]
    if missing:
        raise ValueError(f"{path}: the recorded request lacks {missing}")

    def is_int(value, minimum=0):
        return (
            isinstance(value, int) and not isinstance(value, bool) and value >= minimum
        )

    if not isinstance(request["model_name"], str) or not request["model_name"]:
        raise ValueError(f"{path}: the recorded request's model_name is not a name")
    if not is_int(request["seed"]):
        raise ValueError(f"{path}: the recorded request's seed is not an integer")
    if not is_int(request["n_samples"]) or not is_int(request["max_len"], 1):
        raise ValueError(f"{path}: the recorded request's counts are not integers")
    if request["gen_batch_size"] is not None and not is_int(
        request["gen_batch_size"], 1
    ):
        raise ValueError(f"{path}: the recorded request's gen_batch_size is invalid")
    if request["n_samples"] != n_rows:
        raise ValueError(
            f"{path}: the recorded request names {request['n_samples']} samples "
            f"for {n_rows} rows"
        )
    # The generation contract is recorded by current producers; an archived
    # sidecar from an earlier producer carries none and is still ingested.
    if "generation" in request and not isinstance(request["generation"], str):
        raise ValueError(f"{path}: the recorded request's generation is not a name")
    try:
        requested = list(sampled_mod.validate_start_ids(request["start"]))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path}: the recorded request's start is invalid") from exc
    if start is None:
        raise ValueError(
            f"{path}: the recorded request names a start but the sidecar records none"
        )
    if requested != [int(start[0])]:
        raise ValueError(
            f"{path}: the recorded request's start {requested} disagrees with the "
            f"recorded start {[int(start[0])]}"
        )


def cmd_pool(args: argparse.Namespace) -> int:
    """Merge sample shards into a token store plus a manifest.

    Args:
        args: Parsed `pool` arguments: `model_name`, the `shards_dir` and
            `expected_shards` count to read, an optional `ids_dir` of
            sampler-written token ids, the run's `seed_base`, and the `out`
            directory for the token store and manifest.
    """
    manifest_path = args.out / "manifest.json"
    shards = _pool_shard_paths(args.shards_dir, args.expected_shards)
    tok = common._tokenizer(args.model_name)
    start_ids = getattr(args, "start_ids", None)
    ids_dir = getattr(args, "ids_dir", None)
    # The encoding names the layout and the prefix every row starts with. A
    # pool without sidecars re-encodes the decoded text, the layout of pools
    # written before sidecars existed, and records that it did.
    encoding = {
        "format": POOL_FORMAT if ids_dir is not None else POOL_FORMAT_REENCODED,
        "model_name": args.model_name,
        "start": pooled_mod.start_prefix(tok, start_ids),
    }
    digests = [text_sha256(path.read_text(encoding="utf-8")) for path in shards]
    ids_digests = (
        []
        if ids_dir is None
        else [file_sha256(ids_dir / f"{path.stem}.ids.npz") for path in shards]
    )
    if manifest_path.exists():
        _check_pool_resume(
            manifest_path,
            encoding,
            digests,
            ids_digests,
            seed_base=args.seed_base,
            n_shards=len(shards),
        )
        return 0
    sizes, bindings, starts, flat, offsets = [], [], [], [], [0]
    for path in shards:
        text = path.read_text(encoding="utf-8")
        samples = _sample_list(text, path)
        sizes.append(len(samples))
        if ids_dir is None:
            rows = sampled_mod._encode_samples(tok, samples, start_ids)
            binding, start_binding = "reencoded", "requested"
        else:
            rows, binding, start_binding = _shard_token_ids(
                path,
                samples,
                tok,
                ids_dir,
                text,
                start_ids,
                producer={
                    "model_name": args.model_name,
                    "seed": args.seed_base + _shard_index(path),
                },
            )
        bindings.append(binding)
        starts.append(start_binding)
        for ids in rows:
            flat.append(np.asarray(ids, dtype=np.int32))
            offsets.append(offsets[-1] + len(ids))
    args.out.mkdir(parents=True, exist_ok=True)
    tokens = np.concatenate(flat) if flat else np.zeros(0, dtype=np.int32)
    _write_bin(args.out / "tokens.i32", tokens)
    _write_bin(args.out / "tokens.off.i64", np.array(offsets, dtype=np.int64))
    store_digests = {name: file_sha256(args.out / name) for name in STORE_FILES}
    # The fingerprint binds the token store the sweep scores, the drawn ids
    # and their encoding identity, beside the decoded text.
    manifest = {
        "n_shards": len(shards),
        "shard_sizes": sizes,
        "shard_paths": [str(p) for p in shards],
        "sha256": digests,
        "ids_sha256": ids_digests,
        "ids_binding": bindings,
        "start_binding": starts,
        "encoding": encoding,
        "store_sha256": store_digests,
        "n_samples": int(np.array(sizes, dtype=np.int64).sum()),
        "seed_base": args.seed_base,
        "model_name": args.model_name,
        "pool_fingerprint": text_sha256(
            "".join(digests)
            + "".join(ids_digests)
            + json.dumps(encoding, sort_keys=True)
        ),
    }
    _write_text(manifest_path, json.dumps(manifest, indent=2))
    print(
        f"[pool] {manifest['n_samples']} samples over {len(shards)} shards "
        f"fingerprint={manifest['pool_fingerprint'][:12]}",
        flush=True,
    )
    return 0


def _check_pool_resume(
    manifest_path: Path, encoding, digests, ids_digests, *, seed_base, n_shards
) -> None:
    """A manifest is reused only for the same request, and only with its token store intact.

    Raises:
        SystemExit: The manifest records other shard digests, sidecar digests,
            encoding, seed base or shard count than this request, or the token
            store beside it is missing or inconsistent with the manifest.
    """
    have = _load_manifest(manifest_path)
    same = (
        have.get("encoding") == encoding
        and list(have.get("sha256", [])) == digests
        and list(have.get("ids_sha256", [])) == ids_digests
        and have.get("seed_base") == seed_base
        and have.get("n_shards") == n_shards
    )
    if not same:
        raise SystemExit(
            f"[pool] {manifest_path} was built from other inputs or under another "
            f"request than this one ({encoding}, seed base {seed_base}, "
            f"{n_shards} shards); point --out at an empty directory"
        )
    problem = _token_store_problem(manifest_path.parent, have)
    if problem:
        raise SystemExit(
            f"[pool] {manifest_path} exists but its token store {problem}; remove "
            "the directory to rebuild the pool"
        )
    verified = (
        "content verified against the recorded digests"
        if "store_sha256" in have
        else "a store written before digests were recorded, structure verified only"
    )
    print(
        f"[pool] {manifest_path} exists for these inputs ({verified}), skipping",
        flush=True,
    )


def _token_store_problem(pool_dir: Path, manifest: dict) -> "str | None":
    """Why the token store beside *manifest* cannot be reused, or ``None``.

    A manifest that records the store's digests binds the files by content; one
    written before that is checked by structure alone.
    """
    tokens_path, offsets_path = pool_dir / "tokens.i32", pool_dir / "tokens.off.i64"
    if not tokens_path.exists() or not offsets_path.exists():
        return "is missing"
    recorded = None
    if "store_sha256" in manifest:
        recorded = manifest["store_sha256"]
        if not _valid_store_digests(recorded):
            return (
                "has a manifest whose store_sha256 is not a mapping of both "
                f"{STORE_FILES} to sha256 digests"
            )
        for path in (tokens_path, offsets_path):
            if file_sha256(path) != recorded[path.name]:
                return f"differs from the digest the manifest records for {path.name}"
    for path, dtype in ((tokens_path, np.int32), (offsets_path, np.int64)):
        if path.stat().st_size % np.dtype(dtype).itemsize:
            return f"has a {path.name} whose size is not a whole number of entries"
    tokens = np.fromfile(tokens_path, dtype=np.int32)
    offsets = np.fromfile(offsets_path, dtype=np.int64)
    start = (manifest.get("encoding") or {}).get("start")
    return _store_arrays_problem(
        tokens, offsets, int(manifest.get("n_samples", -1)), start
    )


#: The token store's files, both of which a manifest's ``store_sha256`` names.
STORE_FILES = ("tokens.i32", "tokens.off.i64")


def _valid_store_digests(value) -> bool:
    """Whether *value* maps exactly the store's files to sha256 hex digests."""
    return (
        isinstance(value, dict)
        and set(value) == set(STORE_FILES)
        and all(
            isinstance(d, str)
            and len(d) == 64
            and all(c in "0123456789abcdef" for c in d)
            for d in value.values()
        )
    )


def _store_arrays_problem(tokens, offsets, n_samples: int, start) -> "str | None":
    """Why a token store's arrays are not a pool of *n_samples* rows, or ``None``.

    Every row holds at least its prefix, so the offsets rise strictly; ids are
    non-negative; with *start* given, every row begins with its last id.
    """
    if offsets.size != n_samples + 1:
        return f"holds {max(offsets.size - 1, 0)} rows for {n_samples} samples"
    if offsets.size and (
        offsets[0] != 0
        or offsets[-1] != tokens.size
        or np.any(offsets[1:] <= offsets[:-1])
    ):
        return "has offsets that do not partition its tokens into non-empty rows"
    if tokens.size and tokens.min() < 0:
        return "holds negative token ids"
    if start and n_samples > 0 and np.any(tokens[offsets[:-1]] != start[-1]):
        return f"has rows that do not start with the recorded prefix {start}"
    return None


def _shard_token_ids(
    path: Path,
    samples,
    tok,
    ids_dir: Path,
    text: "str | None" = None,
    start_ids=None,
    producer: "dict | None" = None,
):
    """Each sample's drawn ids with the pool's one-token prefix prepended.

    The sidecar's ids are read as drawn; re-encoding the decoded text would
    give the canonical segmentation instead. A sidecar that records a prefix
    or a request must agree with *start_ids* and *producer* (model, seed).

    Returns:
        The rows, the sidecar's text binding (``digest`` or ``legacy``) and
        its prefix binding (``sidecar`` when recorded, else ``requested``).

    Raises:
        ValueError: No sidecar, or a recorded prefix or request that disagrees.
    """
    sidecar_path = ids_dir / f"{path.stem}.ids.npz"
    if not sidecar_path.exists():
        raise ValueError(
            f"{path.stem}: no sidecar at {sidecar_path}; the pool reads the drawn "
            "ids and does not re-encode decoded text"
        )
    sidecar = _read_sidecar(sidecar_path, len(samples), text)
    prefix = pooled_mod.start_prefix(tok, start_ids)
    if sidecar.start is not None and list(sidecar.start) != prefix:
        raise ValueError(
            f"{path.stem}: the sidecar was drawn after {list(sidecar.start)}, the "
            f"pool is built under {prefix}"
        )
    if sidecar.request is not None and producer:
        for key, value in producer.items():
            if sidecar.request.get(key) != value:
                raise ValueError(
                    f"{path.stem}: the sidecar records {key}={sidecar.request.get(key)!r}, "
                    f"the pool expects {value!r}"
                )
    ids, offsets = sidecar.ids, sidecar.offsets
    rows = [
        [*prefix, *ids[offsets[i] : offsets[i + 1]].tolist()]
        for i in range(len(samples))
    ]
    return (
        rows,
        sidecar.binding,
        "sidecar" if sidecar.start is not None else "requested",
    )


def _write_bin(path: Path, array: np.ndarray) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    array.tofile(tmp)
    tmp.replace(path)


def _shard_index(path: Path) -> int:
    """The index a shard file name carries (``shard_0007.json`` is 7)."""
    return int(path.stem.rsplit("_", 1)[-1])


def _pool_shard_paths(directory: Path, expected: int) -> list[Path]:
    """The shard files, refusing a missing index."""
    paths = [directory / f"shard_{i:04d}.json" for i in range(expected)]
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        raise SystemExit(
            f"[pool] {len(missing)} sample shards absent (e.g. {missing[:5]}); "
            "re-run those sampling tasks before merging the pool."
        )
    return paths


def _load_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _shard_samples(manifest: dict, shard_index: int) -> tuple[list[str], int]:
    """One shard's samples and the pool index of its first sample."""
    offset = int(np.array(manifest["shard_sizes"][:shard_index], dtype=np.int64).sum())
    path = Path(manifest["shard_paths"][shard_index])
    return json.loads(path.read_text(encoding="utf-8")), offset


def _shard_encoded(pool_dir: "Path | None", manifest: dict, shard_index: int):
    """One shard's token ids, read from the pool store rather than re-encoded.

    The store is the single source of truth: prefixes are read out of it later,
    so the frame has to be built over the same ids. Re-encoding the decoded text
    can segment it differently from the way it was sampled.
    """
    if pool_dir is None:
        return None
    sizes = np.array(manifest["shard_sizes"], dtype=np.int64)
    lo = int(sizes[:shard_index].sum())
    hi = lo + int(sizes[shard_index])
    offsets = np.fromfile(pool_dir / "tokens.off.i64", dtype=np.int64)
    tokens = np.fromfile(pool_dir / "tokens.i32", dtype=np.int32)
    return [tokens[offsets[i] : offsets[i + 1]].tolist() for i in range(lo, hi)]
