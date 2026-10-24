"""Load precomputed predictors from the Hugging Face Parquet store.

Reads the per-dataset Parquet (cached by ``huggingface_hub``), filters to one
cell ``tag``, and reconstructs the ``{stimulus_id: list[FeatureModelOutput]}``
shape the precomputed-feature path expects. A local-pickle fallback keeps it
working offline.

Repo is configurable via ``PSYCHO_PREDICTORS_HF_REPO``.

Parquet imports stay function-local. Repository resolution and auxiliary
asset downloads are owned by :mod:`replica.external.hub`.
"""

import functools
import logging
import os
import pickle
import re
import warnings
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from replica.dataset.catalog.registry import REGISTRY

from .external.hub import (
    HF_REPO,
    HF_REVISION,
    _hub_file,
    fetch_repro_asset,
    predictor_store_identity,
    resolve_store_revision,
)
from .feature.core.protocols import FeatureModelOutput
from .feature.core.reproduction import (
    check_reproduction_opt_in,
    is_reproduction_only,
    warn_reproduction_only,
)
from .utils.naming import (
    canonical_key,
    canonical_method,
    canonical_tag,
    legacy_tags,
    split_cache_filename,
)

__all__ = [
    "HF_REPO",
    "HF_REVISION",
    "assert_not_constant",
    "check_reproduction_consent",
    "declared_method",
    "fetch_repro_asset",
    "file_tag",
    "load_outputs_for_tag",
    "load_samples_for_tag",
    "predictor_store_identity",
    "published_tags",
    "resolve_store_revision",
]

log = logging.getLogger("replica.predictors")
_SHARD_RE = re.compile(r"\.shard\d+_\d+$")


def _rows_for_tag(table, tag: str, filename: str):

    if "tag" in table.column_names:
        return table.filter(_tag_mask(table["tag"], tag))
    owners = [owner for owner, names in _tag_index().items() if filename in names]
    if owners != [tag]:
        raise ValueError(f"{filename}: tagless file has ambiguous ownership: {owners}")
    return table


def _tag_mask(column, tag: str):
    """Rows whose tag is *tag* under its current or an earlier method name."""
    import pyarrow.compute as pc

    mask = pc.call_function("equal", [column, tag])
    for legacy in sorted(legacy_tags(tag)):
        mask = pc.call_function(
            "or", [mask, pc.call_function("equal", [column, legacy])]
        )
    return mask


def _validate_row_key(sid, index) -> None:
    if (
        not isinstance(sid, str)
        or not sid
        or not isinstance(index, int)
        or isinstance(index, bool)
        or index < 0
    ):
        raise ValueError(
            f"Invalid predictor row identity: {sid!r}, unit_index={index!r}"
        )


# Hub failures worth falling back on are all OSError: huggingface_hub's
# EntryNotFoundError / HfHubHTTPError / LocalEntryNotFoundError derive from it
# (HfHubHTTPError via requests' RequestException(IOError)). Any other exception
# propagates.


def assert_not_constant(path: Path, payload: dict, key: str) -> None:
    """Refuse a constant-valued predictor pickle.

    A constant column is collinear with the intercept, so the regression
    drops it and the fit runs uncontrolled.

    Args:
        path: Where the pickle came from, named in the error.
        payload: Stimulus id -> per-unit ``FeatureModelOutput`` records.
        key: Value key of the predictor to check.
    """
    distinct: set[float] = set()
    n_vals = 0
    for outs in payload.values():
        for out in outs:
            v = out.values.get(key)
            if v is None:
                continue
            n_vals += 1
            distinct.add(float(v))
            if len(distinct) > 1:
                return
    if n_vals > 1 and len(distinct) <= 1:
        raise ValueError(
            f"{path}: '{key}' is constant ({distinct or 'all-None'}) across "
            f"all {n_vals} words; degenerate pickle. Rebuild or choose another source."
        )


# Per-word array intermediates live in subfolders, one file per logical cell.
# (marker in tag) -> (subfolder, array column).
_SAMPLE_LAYOUT = (
    ("mcword_entropy", "mcword_samples", "samples"),
    ("cont_entropy", "cont_entropy_samples", "samples"),
    ("ft_entropy", "next_log_probs", "logprobs"),
)

#: Columns a scalar cell carries wherever it lives, so a subfolder file can be
#: told apart from an array one by its schema rather than by its folder name.
_SCALAR_COLUMNS = frozenset({"predictor", "value", "stimulus_id", "unit_index"})


@functools.lru_cache(maxsize=1)
def _tag_index() -> dict[str, tuple[str, ...]]:
    """``{tag: (file, ...)}`` from the store's own index.

    The index records where each cell actually lives, including its subfolder
    and every file a multi-file cell spans, so a cell outside the per-corpus
    scalar table is reachable without a hardcoded folder table. Returns an
    empty mapping when the index is unavailable, leaving the caller's other
    routes intact.
    """
    import pyarrow.parquet as pq

    try:
        path = _hub_file("tags.parquet")
    except OSError as e:
        log.debug("store index unavailable (%s)", e)
        return {}
    table = pq.read_table(path, columns=["tag", "file"]).to_pydict()
    files: dict[str, list[str]] = {}
    for tag, name in zip(table["tag"], table["file"], strict=True):
        if (
            not isinstance(tag, str)
            or not tag
            or not isinstance(name, str)
            or Path(name).is_absolute()
            or ".." in Path(name).parts
        ):
            raise ValueError(f"Invalid predictor index entry: {tag!r}, {name!r}")
        bucket = files.setdefault(canonical_tag(tag), [])
        if name not in bucket:
            bucket.append(name)
    return {tag: tuple(sorted(names)) for tag, names in files.items()}


@functools.lru_cache(maxsize=1)
def _index_methods() -> dict[str, str]:
    """``{tag: method}`` the store's index declares; empty when it declares none."""
    import pyarrow.parquet as pq

    try:
        path = _hub_file("tags.parquet")
    except OSError as e:
        log.debug("store index unavailable (%s)", e)
        return {}
    if "method" not in pq.read_schema(path).names:
        return {}
    table = pq.read_table(path, columns=["tag", "method"]).to_pydict()
    methods: dict[str, str] = {}
    for tag, method in zip(table["tag"], table["method"], strict=True):
        if method is None or not isinstance(tag, str):
            continue
        tag = canonical_tag(tag)
        method = _declared_name(method, f"index row {tag!r}")
        if methods.setdefault(tag, method) != method:
            raise ValueError(f"index declares two methods for {tag!r}")
    return methods


def _index_method(tag: str) -> str | None:
    """The method the store's index declares for *tag*, if the index knows it."""
    tag = canonical_tag(tag)
    if tag not in _tag_index():
        return None
    return _index_methods().get(tag)


def published_tags() -> set[str] | None:
    """Every cell tag in the store's index, or ``None`` when unreachable."""
    index = _tag_index()
    return set(index) if index else None


@functools.lru_cache(maxsize=64)
def _indexed_table(filename: str, columns: tuple[str, ...] | None = None):
    """One stored file, read by the path the index gives."""
    import pyarrow.parquet as pq

    path = _hub_file(filename)
    return pq.read_table(path, columns=list(columns) if columns else None)


def _indexed_scalar_cell(tag: str):
    """Scalar rows for *tag* when it lives outside the per-corpus table.

    Returns ``None`` when the index does not know the tag, or knows it only as
    array-shaped data, which :func:`load_samples_for_tag` serves instead.
    """
    import pyarrow as pa

    parts = []
    for filename in _tag_index().get(tag, ()):
        if "/" not in filename:
            continue  # the per-corpus table, already tried by the caller
        table = _indexed_table(filename)
        if not _SCALAR_COLUMNS.issubset(table.column_names):
            return None
        table = _rows_for_tag(table, tag, filename)
        if "tag" not in table.column_names:
            table = table.append_column("tag", pa.array([tag] * table.num_rows))
        if "text_unit" not in table.column_names:
            table = table.append_column(
                "text_unit", pa.nulls(table.num_rows, type=pa.string())
            )
        parts.append(
            table.select(
                ["tag", "stimulus_id", "unit_index", "predictor", "value", "text_unit"]
            )
        )
    if not parts:
        return None
    return pa.concat_tables(parts)


def _match_dataset(tag: str) -> str | None:
    """Longest registry dataset name that is a '_'-preceded suffix of *tag*."""
    best = None
    for r in REGISTRY:
        if (tag == r or tag.endswith("_" + r)) and (best is None or len(r) > len(best)):
            best = r
    return best


def _tag_method(tag: str) -> str | None:
    """The method a cell name declares; None when the name is outside the grammar."""
    stem = canonical_tag(_SHARD_RE.sub("", tag))
    dataset = _match_dataset(stem)
    if dataset is None:
        return None
    try:
        return split_cache_filename(stem, dataset)[1]
    except ValueError:
        return None


def _declared_name(value: object, source: str) -> str:
    """A declared method under its current name; only a clean string is accepted.

    Args:
        value: The declaration as stored.
        source: What declared it, for the error.
    """
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{source} declares no valid method: {value!r}")
    return canonical_method(value)


def file_tag(path: str | os.PathLike) -> str:
    """The cell name a pickle's or Parquet file's own filename declares.

    Args:
        path: A ``.pkl``, ``.suspect.pkl`` or ``.parquet`` file, sharded or not.
    """
    name = Path(path).name
    for suffix in (".parquet", ".pkl", ".suspect"):
        name = name.removesuffix(suffix)
    return name


def _indexed_file_tags(tag: str) -> list[str]:
    """The cell names the index's files for *tag* declare through their names."""
    return [file_tag(f) for f in _tag_index().get(canonical_tag(tag), ())]


def declared_method(
    tag: str,
    identity: dict | None = None,
    *,
    index_method: str | None = None,
    file_tags: Sequence[str] = (),
) -> str | None:
    """The one method a cell declares through every source in hand.

    Args:
        tag: Requested cell name, with or without a shard suffix.
        identity: The validated manifest identity of a local file, when one
            exists; a present ``method`` must be a clean string.
        index_method: The method the store's index declares for the cell.
        file_tags: The names of the files actually read (indexed Parquet
            files, a local pickle), each declaring through its own name.

    Returns:
        The declared method, or None when no source names one.

    Raises:
        ValueError: Two sources declare different methods, or a declaration
            is not a clean string.
    """
    declarations = {"name": _tag_method(tag)}
    if index_method is not None:
        declarations["index"] = _declared_name(index_method, f"{tag}: the index")
    for file_tag in file_tags:
        declarations[f"file {file_tag}"] = _tag_method(file_tag)
    if identity is not None and "method" in identity:
        declarations["manifest"] = _declared_name(
            identity["method"], f"{tag}: the manifest"
        )
    found = {source: m for source, m in declarations.items() if m is not None}
    if len(set(found.values())) > 1:
        raise ValueError(f"{tag}: method declarations disagree: {found}")
    return next(iter(found.values()), None)


def check_reproduction_consent(
    tag: str,
    identity: dict | None = None,
    *,
    allow_reproduction: bool,
    stacklevel: int = 3,
    index_method: str | None = None,
    file_tags: Sequence[str] = (),
    gated: str | None = None,
) -> str | None:
    """Refuse a reproduction-only cell without consent, warn with it once.

    Args:
        tag: Requested cell name, with or without a shard suffix.
        identity: The validated manifest identity of a local file, when any.
        allow_reproduction: Explicit acknowledgement of the estimator's errors.
        stacklevel: Warning stack depth, counted from this function's caller.
        index_method: The method the store's index declares, when known.
        file_tags: The names of the files actually read, when any.
        gated: The method an earlier call already gated for this load, so the
            warning is not repeated.

    Returns:
        The declared method, or None when the cell declares none.

    Raises:
        ValueError: A declared reproduction-only method without consent, or
            declarations that disagree.
    """
    method = declared_method(
        tag, identity, index_method=index_method, file_tags=file_tags
    )
    if method is not None and is_reproduction_only(method):
        check_reproduction_opt_in(method, allow_reproduction=allow_reproduction)
        if method != gated:
            warn_reproduction_only(
                method, allow_reproduction=allow_reproduction, stacklevel=stacklevel + 1
            )
    return method


@functools.lru_cache(maxsize=4)
def _dataset_table(dataset: str):
    """Download + read one dataset's predictor Parquet (cached in-process)."""
    import pyarrow.parquet as pq

    path = _hub_file(f"{dataset}.parquet")
    return pq.read_table(
        path,
        columns=["tag", "stimulus_id", "unit_index", "predictor", "value", "text_unit"],
    )


@functools.lru_cache(maxsize=256)
def _cell_subtable(dataset: str, tag: str):
    """One cell's rows (cached, so re-loading a tag pays the O(rows) filter once)."""

    t = _dataset_table(dataset)
    return t.filter(_tag_mask(t["tag"], tag))


def _reconstruct(sub) -> dict:
    """Rebuild ``{stim_id: [FeatureModelOutput]}`` from one cell's rows.

    A row's ``text_unit`` is stored in the record's ``metadata``, so the
    precomputed-feature path can check each unit against the corpus.
    """
    sids = sub["stimulus_id"].to_pylist()
    idxs = sub["unit_index"].to_pylist()
    preds = sub["predictor"].to_pylist()
    vals = sub["value"].to_pylist()
    texts = (
        sub["text_unit"].to_pylist()
        if "text_unit" in sub.column_names
        else [None] * len(sids)
    )
    names: dict[str, dict[int, str]] = {}
    per_stim: dict[str, dict[int, dict]] = {}
    for sid, i, k, v, text in zip(sids, idxs, preds, vals, texts, strict=True):
        _validate_row_key(sid, i)
        if text is not None:
            unit_names = names.setdefault(sid, {})
            if i in unit_names and unit_names[i] != text:
                raise ValueError(f"Conflicting predictor text_unit: {(sid, i)}")
            unit_names[i] = text
        if not isinstance(k, str) or not k:
            raise ValueError(f"Invalid predictor name: {k!r}")
        k = canonical_key(k)
        values = per_stim.setdefault(sid, {}).setdefault(i, {})
        if k in values:
            raise ValueError(f"Duplicate predictor row: {(sid, i, k)}")
        values[k] = np.float64(v) if v is not None else np.float64("nan")
    # Length = highest emitted unit + 1; interior gaps are back-filled empty.
    # Relies on every word emitting >=1 scalar, else a trailing empty word is
    # dropped and the reconstructed list comes back short.
    return {
        sid: [
            FeatureModelOutput(
                values=units.get(i, {}),
                metadata=(
                    {"text_unit": names[sid][i]}
                    if sid in names and i in names[sid]
                    else {}
                ),
            )
            for i in range(max(units) + 1)
        ]
        for sid, units in per_stim.items()
    }


@functools.lru_cache(maxsize=2)
def _corpus_stimuli(dataset: str) -> list:
    """The corpus as its loader builds it, once per process."""
    return list(REGISTRY[dataset].factory().stimuli())


def _check_pickle_corpus(path: Path, dataset: str | None) -> dict:
    """Refuse a pickle whose manifest digests another version of its corpus.

    A pickle carries no unit strings, so its manifest's ``corpus_sha256`` is
    the record of the units it was computed on. A pickle without a manifest
    cannot be checked and is refused as well.

    Returns:
        The validated manifest identity.

    Raises:
        ValueError: On a missing or mismatched manifest.
    """
    from replica.artifacts import validate_artifact
    from replica.dataset.catalog.provenance import corpus_fingerprint

    identity = validate_artifact(path)
    if identity is None:
        raise ValueError(
            f"{path}: no provenance manifest, so the units it was computed on "
            "cannot be checked; recompute the cell or read it from the store"
        )
    if dataset is None:
        return identity
    options = identity.get("options", {})
    unit = options.get("unit", "words")
    scope = options.get("context_scope", "stimulus")
    expected = corpus_fingerprint(dataset, _corpus_stimuli(dataset), unit, scope)
    if identity.get("corpus_sha256") != expected:
        raise ValueError(
            f"{path}: computed on another version of {dataset} than the loader "
            "produces; recompute the cell"
        )
    return identity


def load_outputs_for_tag(
    tag: str,
    *,
    local_path: str | os.PathLike | None = None,
    allow_reproduction: bool = False,
) -> dict:
    """Return ``{stim_id: list[FeatureModelOutput]}`` for one scalar predictor cell.

    Reads the Hub Parquet first and falls back to *local_path* when the Hub is
    unreachable or lacks the tag; per-word arrays come from
    :func:`load_samples_for_tag`. The Parquet rows are cached and each call
    builds a fresh dict, so the result is safe to mutate. Stimulus IDs retain
    their exact stored spelling.

    Args:
        tag: Cell identifier to look up, with any ``.shardN_M`` suffix stripped.
        local_path: Pickle to fall back to when the Hub has no scalar rows for
            the tag.
        allow_reproduction: Required when the cell's declared method is kept
            for reproduction only; the cell then loads with a
            ``ReproductionOnlyWarning``.

    Raises:
        FileNotFoundError: Only when the cell is absent, with no scalar rows on
            the Hub and no local pickle; a Hub or dependency failure with no
            local fallback re-raises its own error, so an outage is not read as
            a missing tag.
        ValueError: A reproduction-only cell without consent, or method
            declarations (name, index row, local filename, manifest) that
            disagree, or conflicting text for one stimulus and unit.
    """
    tag = _SHARD_RE.sub("", tag)
    gated = check_reproduction_consent(tag, allow_reproduction=allow_reproduction)
    dataset = _match_dataset(tag)
    hf_error = None
    if dataset is not None:
        try:
            sub = _cell_subtable(dataset, tag)
            if sub.num_rows:
                check_reproduction_consent(
                    tag,
                    allow_reproduction=allow_reproduction,
                    index_method=_index_method(tag),
                    file_tags=_indexed_file_tags(tag),
                    gated=gated,
                )
                return _reconstruct(sub)
            log.debug(
                "cell %s absent from HF table %s; trying local pickle", tag, dataset
            )
        except OSError as e:
            hf_error = e
            log.warning(
                "HF predictor load failed for %s (%s); trying local pickle", tag, e
            )
    # Some scalar cells are published in their own subfolder rather than in the
    # per-corpus table; the index says where, so they need no folder table here.
    try:
        indexed = _indexed_scalar_cell(tag)
    except OSError as e:
        hf_error = hf_error or e
    else:
        if indexed is not None and indexed.num_rows:
            check_reproduction_consent(
                tag,
                allow_reproduction=allow_reproduction,
                index_method=_index_method(tag),
                file_tags=_indexed_file_tags(tag),
                gated=gated,
            )
            return _reconstruct(indexed)
    if local_path is not None and Path(local_path).exists():
        identity = _check_pickle_corpus(Path(local_path), dataset)
        check_reproduction_consent(
            tag,
            identity,
            allow_reproduction=allow_reproduction,
            index_method=_index_method(tag),
            file_tags=[*_indexed_file_tags(tag), file_tag(local_path)],
            gated=gated,
        )
        with open(local_path, "rb") as f:
            payload = pickle.load(f)
        if not isinstance(payload, dict):
            raise ValueError(f"{local_path}: predictor payload must be a mapping")
        return payload
    if hf_error is not None:
        raise hf_error
    raise FileNotFoundError(
        f"predictor cell {tag!r} has no scalar predictors on HF ({HF_REPO}) and no "
        f"local pickle"
        + (f" at {local_path}" if local_path else "")
        + " (per-word array/distribution cells are served by load_samples_for_tag)"
    )


@functools.lru_cache(maxsize=16)
def _samples_table(subdir: str, tag: str, column: str):
    import pyarrow.parquet as pq

    path = _hub_file(f"{subdir}/{tag}.parquet")
    return pq.read_table(path, columns=["stimulus_id", "unit_index", column])


def _indexed_array_cell(tag: str, column: str | None = None):
    """``(table, column)`` for *tag*'s array-shaped subfolder files.

    Returns ``None`` when the index knows the tag in no array-shaped file.
    An array family stores list-typed data columns next to
    ``stimulus_id``/``unit_index`` (plus scalar metadata columns). With one
    list column the choice is implied; a multi-column family needs *column*.
    A cell is indexed at every file that carries its rows, and the scalar
    ones among those hold no array, so they are passed over: a token-grain
    sample cell is indexed at its sample file and at the per-corpus token
    table.
    """
    import pyarrow as pa
    import pyarrow.types as pat

    parts = []
    chosen = column
    for filename in _tag_index().get(tag, ()):
        if "/" not in filename:
            continue
        table = _indexed_table(filename)
        data = [
            field.name
            for field in table.schema
            if pat.is_list(field.type) or pat.is_large_list(field.type)
        ]
        if not data:
            continue
        if chosen is None:
            if len(data) > 1:
                raise ValueError(
                    f"{filename}: cell {tag!r} holds list columns {data}; pass "
                    "column= to pick one"
                )
            chosen = data[0]
        if chosen not in data:
            continue
        table = _rows_for_tag(table, tag, filename)
        parts.append(table.select(["stimulus_id", "unit_index", chosen]))
    if not parts:
        return None
    return pa.concat_tables(parts), chosen


#: The one array family whose published word hashes are 32-bit truncated
#: values; its scalar summaries were computed from the full identities.
_TRUNCATED_HASH_METHOD = "forward_word_lookahead_entropy"


def _warn_truncated_word_hashes(method: str | None, table, column: str) -> None:
    """Warn when a lookahead cell's selected word hashes are the 32-bit ones."""
    if method != _TRUNCATED_HASH_METHOD or column != "id_hash":
        return
    import pyarrow.types as pat

    leaf = table.schema.field(column).type
    while pat.is_list(leaf) or pat.is_large_list(leaf):
        leaf = leaf.value_type
    if pat.is_uint32(leaf):
        warnings.warn(
            f"{column} of this {method} cell holds 32-bit truncated word hashes "
            "that may merge distinct words, so marginal entropy cannot be "
            "reproduced reliably from them; recover the full identities or "
            "recompute the samples",
            UserWarning,
            stacklevel=3,
        )


def _reconstruct_arrays(table, column: str) -> dict:
    """Rebuild ``{stim_id: [np.ndarray per unit]}`` from array-cell rows."""
    import pyarrow as pa

    leaf = table.schema.field(column).type
    depth = 0
    while pa.types.is_list(leaf) or pa.types.is_large_list(leaf):
        depth += 1
        leaf = leaf.value_type
    if not depth or not (pa.types.is_integer(leaf) or pa.types.is_floating(leaf)):
        raise ValueError(f"Predictor column {column!r} must contain numeric arrays")
    dtype = leaf.to_pandas_dtype() if pa.types.is_integer(leaf) else np.float64
    nested = table[column].combine_chunks()
    for _ in range(depth):
        nested = nested.flatten()
        if nested.null_count:
            raise ValueError(f"Predictor column {column!r} contains nested nulls")
    sids = table["stimulus_id"].to_pylist()
    idxs = table["unit_index"].to_pylist()
    arrs = table[column].to_pylist()
    per: dict[str, dict[int, object]] = {}
    for sid, i, a in zip(sids, idxs, arrs, strict=True):
        _validate_row_key(sid, i)
        units = per.setdefault(sid, {})
        if i in units:
            raise ValueError(f"Duplicate predictor array row: {(sid, i)}")
        if a is None:
            units[i] = None
            continue
        try:
            array = np.asarray(a, dtype=dtype)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"Predictor arrays must be rectangular numeric arrays: {(sid, i)}"
            ) from exc
        if array.size == 0 and array.ndim < depth:
            array = array.reshape((*array.shape, *((0,) * (depth - array.ndim))))
        if array.ndim != depth:
            raise ValueError(
                f"Predictor array rank disagrees with its schema: {(sid, i)}"
            )
        units[i] = array
    return {sid: [u.get(i) for i in range(max(u) + 1)] for sid, u in per.items()}


def load_samples_for_tag(
    tag: str, *, column: str | None = None, allow_reproduction: bool = False
) -> dict:
    """Return a cell's per-word array data as ``{stim_id: [np.ndarray per unit]}``.

    Covers array-shaped predictors (MC samples, full ``next_log_probs``,
    per-layer values, ...). The store's index says which subfolder files hold
    the cell; ``_SAMPLE_LAYOUT`` is the offline fallback for the MC
    word-entropy, continuation-entropy and first-token-entropy families.
    Nested arrays retain their rank, and integer arrays retain their stored dtype.
    Stimulus IDs retain their exact stored spelling.
    Empty variable-length lists cannot recover unstored trailing dimensions.

    Args:
        tag: Cell identifier to look up, with any ``.shardN_M`` suffix stripped.
        column: Which list column to read when the cell's file holds more than
            one array column. Inferred from the schema when there is only one.
        allow_reproduction: Required when the cell's declared method is kept
            for reproduction only.

    Selecting the ``id_hash`` column of a ``forward_word_lookahead_entropy``
    cell whose hashes are 32-bit truncated values emits a ``UserWarning``:
    such hashes cannot support marginal reaggregation.

    Raises:
        ValueError: A reproduction-only cell without consent.
    """
    tag = _SHARD_RE.sub("", tag)
    method = check_reproduction_consent(tag, allow_reproduction=allow_reproduction)
    index_error = None
    try:
        indexed = _indexed_array_cell(tag, column)
    except OSError as e:
        index_error = e
    else:
        if indexed is not None:
            method = check_reproduction_consent(
                tag,
                allow_reproduction=allow_reproduction,
                index_method=_index_method(tag),
                file_tags=_indexed_file_tags(tag),
                gated=method,
            )
            table, chosen = indexed
            _warn_truncated_word_hashes(method, table, chosen)
            return _reconstruct_arrays(table, chosen)
    matches = [(sub, col) for marker, sub, col in _SAMPLE_LAYOUT if marker in tag]
    if len(matches) != 1:
        if index_error is not None:
            raise index_error
        raise ValueError(
            f"tag {tag!r} matches {len(matches)} sample-array families "
            f"(need exactly 1): {[m[0] for m in matches]}"
        )
    subdir, column = matches[0]
    table = _samples_table(subdir, tag, column)
    _warn_truncated_word_hashes(method, table, column)
    return _reconstruct_arrays(table, column)
