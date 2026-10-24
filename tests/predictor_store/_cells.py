"""Synthetic compute cells for the predictor store tests.

One toy corpus, ``toyset``, with two stimuli; helpers write pickles with the
manifests the compute driver writes, so the export sees the same files a run
leaves behind.
"""

import pickle
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from replica.artifacts import write_manifest
from replica.compute.cache import split_cache_filename
from replica.compute.units import METHOD_STORAGE
from replica.core import Stimulus
from replica.dataset.catalog.provenance import corpus_fingerprint
from replica.feature import FeatureModelOutput
from replica.utils.naming import canonical_tag
from tests._stimuli import word_stimulus

DATASET = "toyset"
TEXTS = {"toy_1": "ab cd", "toy_2": "ef gh ij"}
# Subword tokens of each text: (start, end, id, string).
TOKENS = {
    "toy_1": [(0, 2, 11, "ab"), (2, 5, 12, " cd")],
    "toy_2": [(0, 1, 13, "e"), (1, 2, 14, "f"), (2, 5, 15, " gh"), (5, 8, 16, " ij")],
}


class ToyCorpora:
    """The toy corpus, standing in for the dataset registry."""

    def names(self) -> list[str]:
        return [DATASET]

    def stimuli(self, dataset: str) -> list[Stimulus]:
        """The toy corpus with a whitespace ``words`` layer."""
        assert dataset == DATASET
        return [word_stimulus(text, sid) for sid, text in TEXTS.items()]

    def word_spacing(self, dataset: str) -> bool:
        """The toy corpus separates its words with spaces."""
        assert dataset == DATASET
        return True


def record(values: dict, **metadata) -> FeatureModelOutput:
    return FeatureModelOutput(
        values=dict(values), metadata={"model_name": "gpt2", **metadata}
    )


def word_payload(key: str = "log_p_observed", scale: float = -1.0, extra=None) -> dict:
    """One record per word, ``scale * (position + 1)`` under *key*."""
    payload = {}
    for sid, text in TEXTS.items():
        rows = []
        for i, _ in enumerate(text.split()):
            values = {key: scale * (i + 1)}
            if extra is not None:
                values.update(extra(sid, i))
            rows.append(record(values))
        payload[sid] = rows
    return payload


def token_payload(key: str = "log_p_observed", extra=None) -> dict:
    """One record per subword token with the token identity in its metadata."""
    payload = {}
    for sid, tokens in TOKENS.items():
        rows = []
        for i, (start, end, token_id, token) in enumerate(tokens):
            values = {key: -0.5 * (i + 1)}
            if extra is not None:
                values.update(extra(sid, i))
            rows.append(
                record(
                    values,
                    token_index=i,
                    span=(start, end),
                    token_id=token_id,
                    token=token,
                )
            )
        payload[sid] = rows
    return payload


def character_payload() -> dict:
    """One record per character, whitespace included."""
    return {
        sid: [record({"log_p_observed": -0.1 * (i + 1)}) for i in range(len(text))]
        for sid, text in TEXTS.items()
    }


def write_cell(
    out: Path,
    tag: str,
    payload: dict,
    *,
    method: str,
    unit: str = "words",
    model_id: str = "gpt2",
    status: str = "completed",
    unit_rule: str | None = None,
    derivation: dict | None = None,
    context_scope: str = "stimulus",
    failed: list[tuple[str, str]] | None = None,
    units: dict | None = None,
) -> Path:
    """Write ``<tag>.pkl`` (or ``.suspect.pkl``) and its manifest under *out*."""
    out.mkdir(parents=True, exist_ok=True)
    name = f"{tag}.pkl" if status == "completed" else f"{tag}.suspect.pkl"
    path = out / name
    with path.open("wb") as stream:
        pickle.dump(payload, stream)
    storage = METHOD_STORAGE[method]
    identity = {
        "schema_version": 1,
        "method": method,
        "storage_class": storage.storage_class,
        "layer": storage.layer,
        "options": {
            "model_id": model_id,
            "unit": unit,
            "context_scope": context_scope,
            "chunk_size": None,
            "alpha": 1.0,
            "seed": 0,
            **split_cache_filename(canonical_tag(tag), DATASET)[2],
        },
        "model": {"source": model_id, "revision": None, "sha256": "0" * 64},
        "corpus_sha256": corpus_fingerprint(
            DATASET, ToyCorpora().stimuli(DATASET), unit, context_scope
        ),
        "n_stimuli": len(payload),
        "source_sha256": "2" * 64,
        "versions": {
            "python": "3.12.13",
            "replica": "0.0.1",
            "torch": "2.11.0",
            "transformers": "4.57.6",
        },
    }
    if units is not None:
        identity["units"] = units
    if storage.study is not None:
        identity["study"] = storage.study
    if unit_rule is not None:
        identity["unit_rule"] = unit_rule
        identity["unit_layer"] = "words"
    write_manifest(
        path,
        identity,
        status="completed" if status == "completed" else "quarantined",
        environment={
            "code_commit": "f" * 40,
            "code_dirty": False,
            "device": "test-gpu",
            "model_device": None,
            "cuda": "13.0",
            "computed_at": "2026-09-20T00:00:00+00:00",
        },
        derivation=derivation,
    )
    if failed:
        (out / f"{tag}.failed.log").write_text(
            "".join(f"{sid}\t{message}\n" for sid, message in failed)
        )
    return path


def read(stage: Path, file: str) -> pa.Table:
    return pq.read_table(stage / file)


def rows(stage: Path, file: str) -> list[dict]:
    return read(stage, file).to_pylist()


class FakeHub:
    """A published store on disk: an index and the files it names."""

    repo = "example/predictors"

    def __init__(
        self,
        root: Path,
        index_rows: list[dict],
        tables: dict[str, pa.Table],
        keep: tuple[str, ...] = (),
    ):
        """Write the index in the legacy columns, plus the newer columns in *keep*."""
        from replica.predictor_store.layout import INDEX_SCHEMA

        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        legacy = [
            name
            for name in INDEX_SCHEMA.names
            if name not in _NEW_INDEX_COLUMNS or name in keep
        ]
        pq.write_table(
            pa.Table.from_pylist(
                [{name: row.get(name) for name in legacy} for row in index_rows],
                schema=pa.schema([INDEX_SCHEMA.field(name) for name in legacy]),
            ),
            root / "tags.parquet",
        )
        for file, table in tables.items():
            (root / file).parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, root / file)

    def revision(self) -> str:
        return "a" * 40

    def download(self, filename: str) -> Path:
        path = self.root / filename
        if not path.exists():
            raise OSError(f"{filename} is not published")
        return path

    def index(self) -> pa.Table:
        return pq.read_table(self.root / "tags.parquet")

    def files(self) -> set[str]:
        """Every file the repository holds, which the index need not name."""
        return {
            path.relative_to(self.root).as_posix()
            for path in self.root.rglob("*")
            if path.is_file()
        }


_NEW_INDEX_COLUMNS = {
    "layer",
    "study",
    "storage_class",
    "unit",
    "unit_rule",
    "context_scope",
    "derivation_source",
    "wave",
    "verification",
    "verification_file",
}


def token_table(dataset: str, cells: dict[str, dict[str, list[float]]]) -> pa.Table:
    """A published token table in the token schema.

    Args:
        dataset: Corpus name.
        cells: ``{tag: {predictor: [value per token of toy_1]}}``.
    """
    from replica.predictor_store.layout import TOKEN_SCHEMA

    columns: dict[str, list] = {name: [] for name in TOKEN_SCHEMA.names}
    for tag, predictors in cells.items():
        for predictor, values in predictors.items():
            for i, value in enumerate(values):
                start, end, token_id, token = TOKENS["toy_1"][i]
                columns["dataset"].append(dataset)
                columns["tag"].append(tag)
                columns["predictor"].append(predictor)
                columns["stimulus_id"].append("toy_1")
                columns["token_index"].append(i)
                columns["span_start"].append(start)
                columns["span_end"].append(end)
                columns["token_id"].append(token_id)
                columns["token"].append(token)
                columns["value"].append(value)
                columns["value_unit"].append("nats")
    return pa.table(
        {
            name: pa.array(columns[name], type=TOKEN_SCHEMA.field(name).type)
            for name in TOKEN_SCHEMA.names
        },
        schema=TOKEN_SCHEMA,
    )


def legacy_word_table(
    dataset: str, cells: dict[str, dict[str, list[float]]]
) -> pa.Table:
    """A published per-corpus table in the eight-column schema.

    Args:
        dataset: Corpus name.
        cells: ``{tag: {predictor: [value per word of toy_1]}}``.
    """
    columns = {
        "dataset": [],
        "tag": [],
        "predictor": [],
        "stimulus_id": [],
        "unit_index": [],
        "text_unit": [],
        "value": [],
        "value_unit": [],
    }
    words = TEXTS["toy_1"].split()
    for tag, predictors in cells.items():
        for predictor, values in predictors.items():
            for i, value in enumerate(values):
                columns["dataset"].append(dataset)
                columns["tag"].append(tag)
                columns["predictor"].append(predictor)
                columns["stimulus_id"].append("toy_1")
                columns["unit_index"].append(i)
                columns["text_unit"].append(words[i])
                columns["value"].append(value)
                columns["value_unit"].append("nats")
    return pa.table(
        {
            "dataset": pa.array(columns["dataset"], pa.string()),
            "tag": pa.array(columns["tag"], pa.string()),
            "predictor": pa.array(columns["predictor"], pa.string()),
            "stimulus_id": pa.array(columns["stimulus_id"], pa.string()),
            "unit_index": pa.array(columns["unit_index"], pa.int32()),
            "text_unit": pa.array(columns["text_unit"], pa.string()),
            "value": pa.array(columns["value"], pa.float64()),
            "value_unit": pa.array(columns["value_unit"], pa.string()),
        }
    )


def write_cell_under_old_name(out: Path, tag_old: str, key_old: str) -> Path:
    """A pickle and manifest as a run before the rename wrote them."""
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{tag_old}.pkl"
    with path.open("wb") as stream:
        pickle.dump(word_payload(key_old, 2.0), stream)
    identity = {
        "schema_version": 1,
        "method": "next_word_entropy",
        "storage_class": "unit_bound",
        "layer": 1,
        "options": {"model_id": "gpt2", "unit": "words", "context_scope": "stimulus"},
        "model": {"source": "gpt2", "revision": None, "sha256": "0" * 64},
        "corpus_sha256": corpus_fingerprint(
            DATASET, ToyCorpora().stimuli(DATASET), "words", "stimulus"
        ),
        "n_stimuli": 2,
        "source_sha256": "2" * 64,
        "versions": {"python": "3.12.13", "replica": "0.0.1"},
        "unit_rule": "whitespace_continuation",
        "unit_layer": "words",
    }
    write_manifest(
        path, identity, status="completed", environment={"code_commit": "f" * 40}
    )
    return path
