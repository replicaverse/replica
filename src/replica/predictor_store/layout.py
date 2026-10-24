"""Paths, schemas and array families of the predictor store.

A cell is one compute pickle. The export writes its scalar values into a
long-format table chosen by the cell's unit and method, its array values
into one file per cell under a family folder, and one index row per
(tag, predictor, file). This module fixes where each piece goes and what
columns it carries.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from numbers import Integral

import numpy as np
import pyarrow as pa

from replica.compute.cache import SHARD_RE, split_cache_filename
from replica.compute.units import TOKEN_UNIT, WORD_JOINED
from replica.feature.core.annotation.characters import CHARACTER_LAYER
from replica.feature.core.outputs.units import unit_for
from replica.feature.frequency.registry import SOURCES

INDEX_FILE = "tags.parquet"
COVERAGE_FILE = "coverage.parquet"
HISTORY_FILE = "history.parquet"
PUBLISH_FILE = "publish.json"
READBACK_FILE = "readback.json"
REPORT_FILE = "export_report.json"
README_FILE = "README.md"
MANIFESTS_DIR = "manifests"
VERIFICATION_DIR = "verification"
TOKENS_DIR = "tokens"
CHARACTERS_DIR = "characters"
RESPONSIVITY_DIR = "responsivity"
LENS_DIR = "lens_layers"
LENS_COLUMN = "log_p_layers"
LOOKAHEAD_DIR = "fwd_word_lookahead_entropy_samples"

#: Files at the repository root that are not per-corpus tables.
SIDE_FILES = frozenset({INDEX_FILE, COVERAGE_FILE, HISTORY_FILE})

#: Methods whose word cells are stored dense, one layer vector per word.
LENS_METHODS = frozenset(WORD_JOINED)
LOOKAHEAD_METHOD = "forward_word_lookahead_entropy"
RESPONSIVITY_METHOD = "responsivity"

#: Units that mark a label carried beside the measurements; the value column
#: holds measurements only.
NON_NUMERIC_UNITS = frozenset({"text", "identifier"})

_IDENTITY = [("dataset", pa.string()), ("tag", pa.string())]
_SCALAR = [
    ("predictor", pa.string()),
    ("stimulus_id", pa.string()),
    ("unit_index", pa.int32()),
    ("text_unit", pa.string()),
    ("value", pa.float64()),
    ("value_unit", pa.string()),
]
_STORAGE = [
    ("layer", pa.int32()),
    ("storage_class", pa.string()),
    ("unit_rule", pa.string()),
]
_TOKEN_IDENTITY = [
    ("token_index", pa.int32()),
    ("span_start", pa.int32()),
    ("span_end", pa.int32()),
    ("token_id", pa.int64()),
    ("token", pa.string()),
]

#: Scalar rows of a word or character cell.
WORD_SCHEMA = pa.schema(_IDENTITY + _SCALAR + _STORAGE)
#: Scalar rows of a token cell.
TOKEN_SCHEMA = pa.schema(
    _IDENTITY
    + [("predictor", pa.string()), ("stimulus_id", pa.string())]
    + _TOKEN_IDENTITY
    + [("value", pa.float64()), ("value_unit", pa.string())]
)

INDEX_SCHEMA = pa.schema(
    [
        ("tag", pa.string()),
        ("file", pa.string()),
        ("dataset", pa.string()),
        ("model", pa.string()),
        ("model_kind", pa.string()),
        ("method", pa.string()),
        ("predictor", pa.string()),
        ("value_unit", pa.string()),
        ("n_stimuli", pa.int32()),
        ("n_rows", pa.int64()),
        ("alpha", pa.float64()),
        ("n_samples", pa.int32()),
        ("max_tokens", pa.int32()),
        ("beam_width", pa.int32()),
        ("max_words", pa.int32()),
        ("seed", pa.int32()),
        ("provenance", pa.string()),
        ("chunk_size", pa.int32()),
        ("device", pa.string()),
        ("cuda", pa.string()),
        ("torch", pa.string()),
        ("transformers", pa.string()),
        ("python", pa.string()),
        ("replica", pa.string()),
        ("code_commit", pa.string()),
        ("code_dirty", pa.bool_()),
        ("source_sha256", pa.string()),
        ("model_revision", pa.string()),
        ("computed_at", pa.string()),
        ("layer", pa.int32()),
        ("study", pa.string()),
        ("storage_class", pa.string()),
        ("unit", pa.string()),
        ("unit_rule", pa.string()),
        ("context_scope", pa.string()),
        ("derivation_source", pa.string()),
        ("wave", pa.string()),
        ("verification", pa.string()),
        ("verification_file", pa.string()),
    ]
)

COVERAGE_SCHEMA = pa.schema(
    [
        ("method", pa.string()),
        ("model", pa.string()),
        ("dataset", pa.string()),
        ("unit", pa.string()),
        ("status", pa.string()),
        ("reason_class", pa.string()),
        ("detail", pa.string()),
        ("tag", pa.string()),
    ]
)

HISTORY_SCHEMA = pa.schema(
    [
        ("file", pa.string()),
        ("published_in", pa.string()),
        ("published_at", pa.string()),
        ("last_modified_in", pa.string()),
        ("last_modified_at", pa.string()),
        ("present", pa.bool_()),
    ]
)

#: Coverage statuses of a planned table. ``carried`` is a table the release
#: ships from an earlier one, rather than from this run; ``undefined`` is a
#: table whose measure is not defined on the corpus's units.
EXPORTED, CARRIED, QUARANTINED, ABSENT, UNDEFINED = (
    "exported",
    "carried",
    "quarantined",
    "absent",
    "undefined",
)

#: Tag slots that name a frequency table rather than a language model: the
#: frequency sources without a sampling estimator. A sampled source's tag
#: names the language model it samples.
_FREQUENCY_SOURCES = frozenset(
    name for name, source in SOURCES.items() if source.estimator is None
)
_CONFIG_FIELDS = ("alpha", "n_samples", "max_tokens", "beam_width", "max_words", "seed")
_LENS_KEY_RE = re.compile(r"^(?P<prefix>(?:logit|tuned)_lens_log_p_L)(?P<layer>\d+)$")


@dataclass(frozen=True)
class TagInfo:
    """Filename fields and an optional recorded model identity for one cell."""

    tag: str
    dataset: str
    model_slot: str
    model: str | None
    model_kind: str
    method: str
    config: dict = field(default_factory=dict)

    @property
    def unit(self) -> str:
        """Unit the tag names: ``words`` unless it carries a unit suffix."""
        return str(self.config.get("unit", "words"))

    @property
    def context_scope(self) -> str:
        """Context scope the tag names; ``stimulus`` when it carries none."""
        return str(self.config.get("context_scope", "stimulus"))

    @property
    def segmentation(self) -> str:
        """Word iteration of an information-value tag; ``whitespace`` when it carries none."""
        return str(self.config.get("segmentation", "whitespace"))

    def index_config(self) -> dict[str, float | int | None]:
        """The configuration columns of an index row.

        Returns:
            ``alpha`` as a float and the integer settings as ints; ``None``
            where the tag does not name a setting.
        """
        out: dict[str, float | int | None] = {}
        for name in _CONFIG_FIELDS:
            raw = self.config.get(name)
            if raw is None:
                out[name] = None
            elif name == "alpha":
                out[name] = float(raw)
            else:
                out[name] = int(raw)
        return out


def match_dataset(tag: str, names: Iterable[str]) -> str | None:
    """Longest registry dataset name that is a ``_``-preceded suffix of *tag*.

    Args:
        tag: Cell id, with or without a shard suffix.
        names: Dataset names to try.
    """
    stem = SHARD_RE.sub("", tag)
    best = None
    for name in names:
        if (stem == name or stem.endswith("_" + name)) and (
            best is None or len(name) > len(best)
        ):
            best = name
    return best


def decode_tag(tag: str, dataset: str, *, model_id: str | None = None) -> TagInfo:
    """Split ``<model>_<method+config>_<dataset>`` into its pieces.

    Args:
        tag: Cell id without a shard suffix.
        dataset: The corpus the tag ends in.
        model_id: Recorded model identity, when available. Filename encoding
            loses slash, colon and underscore distinctions, so an LM identity
            cannot be recovered from the tag alone.

    Raises:
        ValueError: The tag carries no known method, or *model_id* is not a
            nonempty string when supplied.
    """
    if model_id is not None and (not isinstance(model_id, str) or not model_id.strip()):
        raise ValueError("model_id must be a nonempty string")
    model_slot, method, config = split_cache_filename(tag, dataset)
    is_freq = model_slot.split("-", 1)[0] in _FREQUENCY_SOURCES
    if model_id is None and is_freq:
        model_id = model_slot
    return TagInfo(
        tag=tag,
        dataset=dataset,
        model_slot=model_slot,
        model=model_id,
        model_kind="frequency_table" if is_freq else "lm",
        method=method,
        config=config,
    )


def token_identity(metadata: dict, index: int) -> tuple[int, int, int, int, str]:
    """Validate a token record's identity against the storage column types.

    Args:
        metadata: Token index, character span, vocabulary ID and decoded label.
        index: Expected token position within its stimulus.
    """
    try:
        start, end = metadata["span"]
        values = (metadata["token_index"], start, end, metadata["token_id"])
        token = metadata["token"]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"token record {index} lacks its identity ({exc})") from exc
    names = ("token_index", "span_start", "span_end", "token_id")
    for name, value in zip(names, values, strict=True):
        kind = TOKEN_SCHEMA.field(name).type
        if (
            not isinstance(value, Integral)
            or isinstance(value, (bool, np.bool_))
            or not 0 <= int(value) < 2 ** (kind.bit_width - 1)
        ):
            raise ValueError(f"token record {index} has invalid {name}: {value!r}")
    if values[0] != index or end <= start or not isinstance(token, str):
        raise ValueError(f"token record {index} has invalid identity: {metadata!r}")
    return (int(values[0]), int(start), int(end), int(values[3]), token)


def check_ids(sids: Iterable[object], where: str) -> None:
    """Require nonempty string stimulus IDs without changing their spelling.

    pyarrow casts silently against the schema, so a numeric key would be
    written as a plausible-looking string instead of failing.

    Args:
        sids: Stimulus ids about to be written.
        where: Source named in the error.

    Raises:
        ValueError: When any ID is not a string or is empty.
    """
    bad = sorted({str(s) for s in sids if not isinstance(s, str) or not s})
    if bad:
        raise ValueError(
            f"{where}: {len(bad)} unusable stimulus id(s), e.g. {bad[:3]}; ids "
            "must be nonempty strings"
        )


def scalar_file(method: str, unit: str, dataset: str, tag: str) -> str | None:
    """Table a cell's scalar rows go to, relative to the store root.

    Args:
        method: Compute method of the cell.
        unit: Grain of the cell's records: ``words``, ``tokens`` or
            ``characters``.
        dataset: Corpus name.
        tag: Cell id.

    Returns:
        ``None`` for a lens word cell, whose per-layer values are stored dense
        under ``lens_layers/`` instead.
    """
    if unit == TOKEN_UNIT:
        return f"{TOKENS_DIR}/{dataset}.parquet"
    if unit == CHARACTER_LAYER:
        return f"{CHARACTERS_DIR}/{dataset}.parquet"
    if method in LENS_METHODS:
        return None
    if method == RESPONSIVITY_METHOD:
        return f"{RESPONSIVITY_DIR}/{tag}.parquet"
    return f"{dataset}.parquet"


def is_root_table(file: str) -> bool:
    """Whether *file* is a per-corpus word table at the repository root.

    Args:
        file: Path relative to the store root.
    """
    return "/" not in file and file.endswith(".parquet") and file not in SIDE_FILES


#: Folders whose files are per-corpus tables rather than per-cell files.
SHARED_FOLDERS = (TOKENS_DIR, CHARACTERS_DIR)


def is_shared_table(file: str) -> bool:
    """Whether *file* is a per-corpus table that holds the rows of many cells.

    A shared table is rewritten wholesale by a publish, so every cell in it
    that a release does not replace has to be carried over.

    Args:
        file: Path relative to the store root.
    """
    if is_root_table(file):
        return True
    folder, _, name = file.partition("/")
    return folder in SHARED_FOLDERS and name.endswith(".parquet") and "/" not in name


def shared_schema(file: str) -> pa.Schema:
    """Schema of a shared per-corpus table.

    Args:
        file: Path relative to the store root.
    """
    return TOKEN_SCHEMA if file.startswith(f"{TOKENS_DIR}/") else WORD_SCHEMA


def dataset_of_file(file: str) -> str:
    """Corpus name of a per-corpus table (root, ``tokens/`` or ``characters/``).

    Args:
        file: Path relative to the store root.
    """
    return file.rsplit("/", 1)[-1][: -len(".parquet")]


@dataclass(frozen=True)
class ArrayFamily:
    """One folder of per-unit array files, one file per cell."""

    folder: str
    column: str
    key: str
    item: pa.DataType
    model_column: bool = False

    def file(self, tag: str) -> str:
        """Path of *tag*'s file in this family, relative to the store root.

        Args:
            tag: Cell id.
        """
        return f"{self.folder}/{tag}.parquet"

    def schema(self, unit: str, depth: int = 1) -> pa.Schema:
        """Schema of a file holding *unit*-grain rows.

        Args:
            unit: Grain of the rows; a token file carries the token identity
                in place of ``text_unit``.
            depth: List nesting of the array column: 1 for a vector per row,
                2 for a matrix per row.
        """
        kind: pa.DataType = self.item
        for _ in range(depth):
            kind = pa.list_(kind)
        head: list[tuple[str, pa.DataType]] = [("dataset", pa.string())]
        if self.model_column:
            head.append(("model", pa.string()))
        head += [("tag", pa.string()), ("stimulus_id", pa.string())]
        head += _unit_columns(unit)
        fields = [pa.field(name, kind_) for name, kind_ in head]
        fields.append(
            pa.field(self.column, kind, metadata={"unit": unit_for(self.key)})
        )
        fields.append(pa.field("value_unit", pa.string()))
        return pa.schema(fields)


def _unit_columns(unit: str) -> list[tuple[str, pa.DataType]]:
    if unit == TOKEN_UNIT:
        return [("unit_index", pa.int32()), *_TOKEN_IDENTITY]
    return [("unit_index", pa.int32()), ("text_unit", pa.string())]


_NEXT_LOG_PROBS = ArrayFamily(
    "next_log_probs", "logprobs", "next_log_probs", pa.float32(), model_column=True
)
_FAMILIES: dict[tuple[str, str], ArrayFamily] = {
    ("mc_word_entropy", "log_p_samples_nats"): ArrayFamily(
        "mcword_samples", "samples", "log_p_samples_nats", pa.float32()
    ),
    ("continuation_entropy", "log_p_samples_nats"): ArrayFamily(
        "cont_entropy_samples", "samples", "log_p_samples_nats", pa.float32()
    ),
    ("similarity_surprisal_noncontextual", "expected_similarity_tokens"): ArrayFamily(
        "simadj_esim",
        "expected_similarity_tokens",
        "expected_similarity_tokens",
        pa.float64(),
    ),
    ("similarity_surprisal_contextual", "expected_similarity_tokens"): ArrayFamily(
        "simadj_esim",
        "expected_similarity_tokens",
        "expected_similarity_tokens",
        pa.float64(),
    ),
}

#: Folders holding array files, with the value key each column carries.
ARRAY_FOLDERS: dict[str, dict[str, str]] = {
    family.folder: {family.column: family.key}
    for family in (*_FAMILIES.values(), _NEXT_LOG_PROBS)
}
ARRAY_FOLDERS[LENS_DIR] = {LENS_COLUMN: LENS_COLUMN}

#: Payload columns of a look-ahead file and the value key each one carries.
LOOKAHEAD_COLUMNS: dict[str, str] = {
    "joint_word_entropy": "joint_word_entropy",
    "conditional_word_entropy": "conditional_word_entropy",
    "marginal_word_entropy": "marginal_word_entropy",
    "marginal_word_entropy_plugin": "marginal_word_entropy_plugin",
    "log_p_step_nats": "fwd_word_log_p_step_nats",
    "id_hash": "fwd_word_id_hash",
}
ARRAY_FOLDERS[LOOKAHEAD_DIR] = dict(LOOKAHEAD_COLUMNS)
#: The key whose unit a look-ahead row's ``value_unit`` reports and whose
#: name the index row carries.
LOOKAHEAD_INDEX_KEY = "joint_word_entropy"
_LOOKAHEAD_TYPES: dict[str, pa.DataType] = {
    "joint_word_entropy": pa.float64(),
    "conditional_word_entropy": pa.float64(),
    "marginal_word_entropy": pa.float64(),
    "marginal_word_entropy_plugin": pa.float64(),
    "log_p_step_nats": pa.float32(),
    "id_hash": pa.uint64(),
}


def lookahead_arrays(values: dict) -> dict[str, np.ndarray]:
    """Validate look-ahead profiles and their sample matrices before storage.

    Args:
        values: One feature record's values.

    Raises:
        ValueError: A column is missing, shapes disagree, or hashes lost precision.
    """
    try:
        arrays = {
            column: np.asarray(values[key]) for column, key in LOOKAHEAD_COLUMNS.items()
        }
    except KeyError as exc:
        raise ValueError(f"Missing look-ahead column: {exc.args[0]}") from exc
    steps = arrays["log_p_step_nats"]
    if steps.ndim != 2:
        raise ValueError("Look-ahead step log probabilities must be a matrix")
    for column, array in arrays.items():
        shape = (
            steps.shape
            if column in {"log_p_step_nats", "id_hash"}
            else (steps.shape[1],)
        )
        if array.shape != shape:
            raise ValueError(
                f"Look-ahead {column} has shape {array.shape}, expected {shape}"
            )
    hashes = arrays["id_hash"]
    if hashes.dtype.kind != "u" or hashes.dtype.itemsize != 8:
        raise ValueError(
            "Look-ahead identities require original uint64 hashes; legacy uint32 "
            "hashes lost bits. Recover the full identities or recompute the samples."
        )
    return arrays


def array_family(method: str, key: str) -> ArrayFamily | None:
    """Family folder of an array-valued key of *method*, or ``None``.

    Args:
        method: Compute method of the cell.
        key: Array-valued key in the cell's records.
    """
    if key == _NEXT_LOG_PROBS.key:
        return _NEXT_LOG_PROBS
    return _FAMILIES.get((method, key))


def is_lens_key(key: str) -> bool:
    """Whether *key* is a per-layer log probability of a lens record.

    Args:
        key: Value key of one record.
    """
    return _LENS_KEY_RE.match(key) is not None


def lens_layer_keys(keys: Sequence[str]) -> list[str]:
    """Per-layer log-probability keys of a lens record, embedding layer first.

    Args:
        keys: Value keys of one record.

    Raises:
        ValueError: When the layer indices do not run ``0..n`` without gaps.
    """
    found = []
    for key in keys:
        match = _LENS_KEY_RE.match(key)
        if match:
            found.append((int(match.group("layer")), key))
    found.sort()
    if [layer for layer, _ in found] != list(range(len(found))):
        raise ValueError(f"lens layer keys do not run 0..n: {[k for _, k in found]}")
    return [key for _, key in found]


def lens_schema() -> pa.Schema:
    """Schema of a dense lens file: one layer vector per word."""
    return pa.schema(
        [
            pa.field("dataset", pa.string()),
            pa.field("tag", pa.string()),
            pa.field("stimulus_id", pa.string()),
            pa.field("unit_index", pa.int32()),
            pa.field("text_unit", pa.string()),
            pa.field(
                LENS_COLUMN,
                pa.list_(pa.float64()),
                metadata={"unit": unit_for(LENS_COLUMN)},
            ),
            pa.field("value_unit", pa.string()),
        ]
    )


def lookahead_schema() -> pa.Schema:
    """Schema of a look-ahead entropy file: six arrays per word."""
    fields = [
        pa.field("dataset", pa.string()),
        pa.field("model", pa.string()),
        pa.field("tag", pa.string()),
        pa.field("stimulus_id", pa.string()),
        pa.field("unit_index", pa.int32()),
        pa.field("text_unit", pa.string()),
        pa.field("n_samples", pa.int32()),
        pa.field("k_words", pa.int32()),
    ]
    for column, key in LOOKAHEAD_COLUMNS.items():
        fields.append(
            pa.field(
                column,
                pa.list_(_LOOKAHEAD_TYPES[column]),
                metadata={"unit": unit_for(key)},
            )
        )
    fields.append(pa.field("value_unit", pa.string()))
    return pa.schema(fields)


def index_key_of_file(file: str) -> str | None:
    """Value key an array file's index row names, or ``None`` for a scalar file.

    Args:
        file: Path relative to the store root.
    """
    folder = file.split("/", 1)[0] if "/" in file else ""
    if folder == LOOKAHEAD_DIR:
        return LOOKAHEAD_INDEX_KEY
    payload = ARRAY_FOLDERS.get(folder)
    if payload is None:
        return None
    return next(iter(payload.values()))
