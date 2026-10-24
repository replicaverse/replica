"""Readback verification of a staged tree against its pickles."""

import json
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest

from replica.artifacts import manifest_path
from replica.predictor_store import layout
from replica.predictor_store.export import export
from replica.predictor_store.verify import VerifyReport, _compare_arrays, verify
from tests.predictor_store._cells import (
    DATASET,
    ToyCorpora,
    token_payload,
    word_payload,
    write_cell,
)


@pytest.mark.parametrize(
    ("source", "stored", "leaf", "failure"),
    [
        (np.array([2**32], dtype=np.uint64), [0], pa.uint32(), "integers"),
        (np.array([-1], dtype=np.int64), [2**64 - 1], pa.uint64(), "integers"),
        (
            np.array([2**53 + 1], dtype=np.uint64),
            [float(2**53)],
            pa.float64(),
            "integers",
        ),
        (np.arange(1, 7).reshape(2, 3), [[1, 2], [3, 4, 5, 6]], pa.int64(), "shapes"),
        (
            np.array([[1.0, np.nan], [3.0, 4.0]]),
            [[1.0, None], [3.0, 4.0]],
            pa.float64(),
            "nulls",
        ),
        (np.array([1e100]), [np.inf], pa.float32(), "overflows"),
        (np.array([0.1]), [np.float32(0.1)], pa.float32(), None),
        (
            np.array([0.1]),
            [np.nextafter(np.float32(0.1), np.float32(np.inf))],
            pa.float32(),
            "differs",
        ),
        (
            np.arange(24).reshape(2, 3, 4),
            np.arange(24).reshape(2, 3, 4).tolist(),
            pa.int64(),
            None,
        ),
    ],
)
def test_array_verifier_preserves_shape_integer_identity_and_float_storage(
    source, stored, leaf, failure
):
    kind = leaf
    for _ in range(source.ndim):
        kind = pa.list_(kind)
    table = pa.table(
        {
            "stimulus_id": ["s"],
            "unit_index": [0],
            "samples": pa.array([stored], type=kind),
        }
    )
    report = VerifyReport()
    _compare_arrays(
        SimpleNamespace(tag="t"),
        "arrays/t.parquet",
        "samples",
        table,
        {("s", 0): source},
        report,
    )
    if failure is None:
        assert report.ok, report.summary()
    else:
        assert any(failure in message for message in report.failures)


def _lookahead_values():
    return {
        **{key: np.array([1.0, 2.0]) for key in list(layout.LOOKAHEAD_COLUMNS)[:4]},
        "fwd_word_log_p_step_nats": np.array(
            [[-0.1, -0.2], [-0.3, -0.4]], dtype=np.float32
        ),
        "fwd_word_id_hash": np.array(
            [[1, 1 + 2**32], [2**63 + 1, 2**64 - 1]], dtype=np.uint64
        ),
    }


@pytest.mark.parametrize(
    "malformed", [None, "legacy_hash", "hash_shape", "profile_shape"]
)
def test_lookahead_export_and_readback_require_full_identities_and_consistent_shapes(
    tmp_path, corpora, malformed
):
    values = _lookahead_values()
    if malformed == "legacy_hash":
        values["fwd_word_id_hash"] = values["fwd_word_id_hash"].astype(np.uint32)
    elif malformed == "hash_shape":
        values["fwd_word_id_hash"] = np.ones((3, 2), dtype=np.uint64)
    elif malformed == "profile_shape":
        values["conditional_word_entropy"] = np.ones(3)
    tag = f"gpt2_fwd_word_lookahead_entropy_N2_M3_K2_s0_{DATASET}"
    out, stage = tmp_path / "out", tmp_path / "stage"
    write_cell(
        out,
        tag,
        word_payload(extra=lambda s, i: values),
        method=layout.LOOKAHEAD_METHOD,
    )
    result = export(stage, [("w", out)], corpora=corpora)
    if malformed:
        assert not result.ok
        return
    assert result.ok
    table = pq.read_table(stage / layout.LOOKAHEAD_DIR / f"{tag}.parquet")
    assert table.schema.field("id_hash").type.value_type == pa.uint64()
    assert (
        table["id_hash"][0].as_py() == values["fwd_word_id_hash"].reshape(-1).tolist()
    )
    report = verify(stage, [("w", out)], names=[DATASET], corpora=ToyCorpora())
    assert report.ok, report.summary()
    for name in ("n_samples", "k_words"):
        table = table.set_column(
            table.schema.get_field_index(name),
            name,
            pa.array([999] * table.num_rows, type=pa.int32()),
        )
    pq.write_table(table, stage / layout.LOOKAHEAD_DIR / f"{tag}.parquet")
    report = verify(stage, [("w", out)], names=[DATASET], corpora=ToyCorpora())
    assert any("n_samples differs" in message for message in report.failures)
    assert any("k_words differs" in message for message in report.failures)


@pytest.fixture
def staged(tmp_path, corpora, samples):
    out = tmp_path / "out"
    write_cell(
        out,
        f"gpt2_surprisal_tokens_{DATASET}",
        token_payload(),
        method="surprisal",
        unit="tokens",
    )
    write_cell(out, f"gpt2_surprisal_{DATASET}", word_payload(), method="surprisal")
    write_cell(
        out,
        f"gpt2_mcword_entropy_N4_M3_a1_s0_{DATASET}",
        word_payload("mc_word_entropy", 1.0, extra=samples),
        method="mc_word_entropy",
    )
    write_cell(
        out,
        f"gpt2_logit_lens_surprisal_{DATASET}",
        word_payload(
            "logit_lens_log_p_L0",
            -1.0,
            extra=lambda s, i: {"logit_lens_log_p_L1": -2.0},
        ),
        method="logit_lens_surprisal",
    )
    stage = tmp_path / "stage"
    assert export(stage, [("w", out)], corpora=corpora).ok
    return stage, out


def _rewrite(path, table):
    pq.write_table(table, path, compression="zstd")


def test_an_untouched_stage_verifies(staged):
    stage, out = staged
    report = verify(stage, [("w", out)], names=[DATASET], corpora=ToyCorpora())
    assert report.ok, report.summary()
    assert report.cells == 4
    assert report.rows_compared == 6 + 5 + 5
    assert report.arrays_compared == 5 + 5
    assert report.files_indexed == 4


def test_a_changed_scalar_value_fails(staged):
    stage, out = staged
    path = stage / f"{DATASET}.parquet"
    table = pq.read_table(path)
    values = table["value"].to_pylist()
    values[3] = values[3] + 1e-9
    table = table.set_column(
        table.schema.get_field_index("value"), "value", pa.array(values, pa.float64())
    )
    _rewrite(path, table)
    report = verify(stage, [("w", out)], names=[DATASET], corpora=ToyCorpora())
    assert not report.ok
    assert any("staged" in f and "pickle" in f for f in report.failures)


def test_a_changed_token_identity_fails(staged):
    stage, out = staged
    path = stage / layout.TOKENS_DIR / f"{DATASET}.parquet"
    table = pq.read_table(path)
    tokens = table["token"].to_pylist()
    tokens[0] = "xx"
    table = table.set_column(
        table.schema.get_field_index("token"), "token", pa.array(tokens, pa.string())
    )
    _rewrite(path, table)
    report = verify(stage, [("w", out)], names=[DATASET], corpora=ToyCorpora())
    assert any("token identity" in f for f in report.failures)


def test_a_changed_sample_and_a_dropped_row_fail(staged):
    stage, out = staged
    tag = f"gpt2_mcword_entropy_N4_M3_a1_s0_{DATASET}"
    path = stage / "mcword_samples" / f"{tag}.parquet"
    table = pq.read_table(path)
    samples = table["samples"].to_pylist()
    samples[1][2] = samples[1][2] * 2
    table = table.set_column(
        table.schema.get_field_index("samples"),
        table.schema.field("samples"),
        pa.array(samples, pa.list_(pa.float32())),
    )
    _rewrite(path, table)
    report = verify(stage, [("w", out)], names=[DATASET], corpora=ToyCorpora())
    assert any("column samples differs" in f for f in report.failures)
    root = stage / f"{DATASET}.parquet"
    rows = pq.read_table(root)
    _rewrite(root, rows.slice(1))
    report = verify(stage, [("w", out)], names=[DATASET], corpora=ToyCorpora())
    assert any("holds" in f and "pickle has" in f for f in report.failures)
    assert any(f.startswith("index:") for f in report.failures)


def test_an_index_that_disagrees_with_its_files_fails(staged):
    stage, out = staged
    path = stage / layout.INDEX_FILE
    table = pq.read_table(path)
    n_rows = table["n_rows"].to_pylist()
    n_rows[0] += 1
    table = table.set_column(
        table.schema.get_field_index("n_rows"), "n_rows", pa.array(n_rows, pa.int64())
    )
    _rewrite(path, table)
    report = verify(stage, [("w", out)], names=[DATASET], corpora=ToyCorpora())
    assert [f for f in report.failures if f.startswith("index:")]


def test_a_source_added_after_export_fails_complete_coverage(staged):
    stage, out = staged
    write_cell(
        out,
        f"gpt2_restricted_first_token_entropy_{DATASET}",
        word_payload("restricted_first_token_entropy", 1.0),
        method="restricted_first_token_entropy",
    )
    report = verify(stage, [("w", out)], names=[DATASET], corpora=ToyCorpora())
    assert any("complete fresh_cells" in f for f in report.failures)
    assert not (stage / layout.READBACK_FILE).exists()


def test_export_and_verify_refuse_a_prefix_that_contradicts_disabled_bos(
    tmp_path, corpora
):
    out, stage = tmp_path / "out", tmp_path / "stage"
    tag = f"gpt2_surprisal_nobos_{DATASET}"
    path = write_cell(out, tag, word_payload(), method="surprisal")
    sidecar = manifest_path(path)
    manifest = json.loads(sidecar.read_text())
    manifest["identity"]["model"]["start"] = {"ids": []}
    sidecar.write_text(json.dumps(manifest))
    assert export(stage, [("w", out)], corpora=corpora).ok
    assert verify(stage, [("w", out)], names=[DATASET], corpora=corpora).ok

    manifest["identity"]["model"]["start"]["ids"] = [5]
    sidecar.write_text(json.dumps(manifest))
    before = sidecar.read_bytes(), path.read_bytes()
    report = verify(stage, [("w", out)], names=[DATASET], corpora=corpora)
    assert not report.ok
    assert any("contradicts add_bos=False" in error for error in report.failures)
    assert not (stage / layout.READBACK_FILE).exists()
    rejected = export(tmp_path / "rejected", [("w", out)], corpora=corpora)
    assert not rejected.ok and rejected.cells == 0
    assert "contradicts add_bos=False" in rejected.discovery_problems[0]
    assert (sidecar.read_bytes(), path.read_bytes()) == before


def test_a_lens_vector_that_differs_fails(staged):
    stage, out = staged
    tag = f"gpt2_logit_lens_surprisal_{DATASET}"
    path = stage / "lens_layers" / f"{tag}.parquet"
    table = pq.read_table(path)
    vectors = table["log_p_layers"].to_pylist()
    vectors[0] = vectors[0][::-1]
    table = table.set_column(
        table.schema.get_field_index("log_p_layers"),
        table.schema.field("log_p_layers"),
        pa.array(vectors, pa.list_(pa.float64())),
    )
    _rewrite(path, table)
    report = verify(stage, [("w", out)], names=[DATASET], corpora=ToyCorpora())
    assert any("log_p_layers differs" in f for f in report.failures)
    assert pc.count(table["tag"]).as_py() == 5


def test_a_quarantined_cell_the_store_carries_verifies_as_carried(tmp_path, corpora):
    """Coverage may mark a quarantined cell carried when the release ships it."""
    from replica.predictor_store.export import export
    from tests.predictor_store._cells import FakeHub, legacy_word_table

    tag = f"gpt2_restricted_first_token_entropy_{DATASET}"
    hub = FakeHub(
        tmp_path / "hub",
        [
            {
                "tag": tag,
                "file": f"{DATASET}.parquet",
                "dataset": DATASET,
                "model": "gpt2",
                "model_kind": "lm",
                "method": "restricted_first_token_entropy",
                "predictor": "restricted_first_token_entropy",
                "value_unit": "bits",
                "n_stimuli": 1,
                "n_rows": 2,
            }
        ],
        {
            f"{DATASET}.parquet": legacy_word_table(
                DATASET, {tag: {"restricted_first_token_entropy": [3.0, 4.0]}}
            )
        },
    )
    out = tmp_path / "out"
    write_cell(out, f"gpt2_surprisal_{DATASET}", word_payload(), method="surprisal")
    write_cell(
        out,
        tag,
        word_payload("restricted_first_token_entropy", 1.0),
        method="restricted_first_token_entropy",
        status="quarantined",
        failed=[("toy_2", "RuntimeError: out of memory")],
    )
    stage = tmp_path / "stage"
    assert export(stage, [("w", out)], hub=hub, corpora=corpora).ok
    report = verify(stage, [("w", out)], names=[DATASET], corpora=ToyCorpora())
    assert report.ok, report.summary()


@pytest.mark.parametrize(
    ("column", "replacement"),
    [
        ("dataset", "other"),
        ("text_unit", "other"),
        ("value_unit", "bits"),
        ("unit_rule", "invented"),
        ("storage_class", None),
    ],
)
def test_scalar_semantic_columns_are_compared(staged, column, replacement):
    stage, out = staged
    path = stage / f"{DATASET}.parquet"
    table = pq.read_table(path)
    values = table[column].to_pylist()
    index = next(i for i, value in enumerate(values) if value != replacement)
    values[index] = replacement
    table = table.set_column(
        table.schema.get_field_index(column),
        table.schema.field(column),
        pa.array(values, type=table[column].type),
    )
    _rewrite(path, table)
    report = verify(stage, [("w", out)], names=[DATASET], corpora=ToyCorpora())
    assert not report.ok and any(column in error for error in report.failures), (
        report.summary()
    )


@pytest.mark.parametrize("column", ["dataset", "text_unit", "value_unit", "field_unit"])
def test_array_semantic_columns_and_field_units_are_compared(staged, column):
    stage, out = staged
    tag = f"gpt2_mcword_entropy_N4_M3_a1_s0_{DATASET}"
    path = stage / "mcword_samples" / f"{tag}.parquet"
    table = pq.read_table(path)
    if column == "field_unit":
        index = table.schema.get_field_index("samples")
        field = table.schema.field(index).with_metadata({b"unit": b"bits"})
        table = table.set_column(index, field, table[index])
    else:
        values = table[column].to_pylist()
        values[0] = "wrong"
        table = table.set_column(
            table.schema.get_field_index(column),
            table.schema.field(column),
            pa.array(values, type=table[column].type),
        )
    _rewrite(path, table)
    report = verify(stage, [("w", out)], names=[DATASET], corpora=ToyCorpora())
    assert not report.ok, report.summary()
    assert any(
        ("unit metadata" if column == "field_unit" else column) in error
        for error in report.failures
    )


@pytest.mark.parametrize(
    "column", ["alpha", "model_revision", "value_unit", "source_sha256"]
)
def test_index_semantics_are_compared_to_the_source(staged, column):
    stage, out = staged
    path = stage / layout.INDEX_FILE
    table = pq.read_table(path)
    values = table[column].to_pylist()
    values[0] = 0.5 if column == "alpha" else "wrong"
    table = table.set_column(
        table.schema.get_field_index(column),
        table.schema.field(column),
        pa.array(values, type=table[column].type),
    )
    _rewrite(path, table)
    report = verify(stage, [("w", out)], names=[DATASET], corpora=ToyCorpora())
    assert any(
        error.startswith("index:") and column in error for error in report.failures
    ), report.summary()


def test_scalar_nan_cannot_be_replaced_by_arrow_null(tmp_path, corpora):
    payload = word_payload()
    payload["toy_1"][0].values["log_p_observed"] = np.nan
    out, stage = tmp_path / "out", tmp_path / "stage"
    write_cell(out, f"gpt2_surprisal_{DATASET}", payload, method="surprisal")
    assert export(stage, [("w", out)], corpora=corpora).ok
    assert verify(stage, [("w", out)], names=[DATASET], corpora=corpora).ok
    path = stage / f"{DATASET}.parquet"
    table = pq.read_table(path)
    values = table["value"].to_pylist()
    values[next(i for i, value in enumerate(values) if np.isnan(value))] = None
    table = table.set_column(
        table.schema.get_field_index("value"),
        table.schema.field("value"),
        pa.array(values, type=pa.float64()),
    )
    _rewrite(path, table)
    report = verify(stage, [("w", out)], names=[DATASET], corpora=corpora)
    assert not report.ok and any("value" in error for error in report.failures), (
        report.summary()
    )


@pytest.mark.parametrize(
    ("column", "replacement"),
    [("span_start", 1), ("token_id", 999), ("token", "wrong"), ("token_index", 999)],
)
def test_array_token_identity_is_compared(tmp_path, corpora, column, replacement):
    out, stage = tmp_path / "out", tmp_path / "stage"
    tag = f"gpt2_cont_entropy_N2_L3_s0_tokens_{DATASET}"
    payload = token_payload(
        extra=lambda _sid, _index: {"log_p_samples_nats": np.array([-0.1, -0.2])}
    )
    write_cell(out, tag, payload, method="continuation_entropy", unit="tokens")
    assert export(stage, [("w", out)], corpora=corpora).ok
    path = stage / "cont_entropy_samples" / f"{tag}.parquet"
    table = pq.read_table(path)
    values = table[column].to_pylist()
    values[0] = replacement
    table = table.set_column(
        table.schema.get_field_index(column),
        table.schema.field(column),
        pa.array(values, type=table[column].type),
    )
    _rewrite(path, table)
    report = verify(stage, [("w", out)], names=[DATASET], corpora=corpora)
    assert not report.ok and any(column in error for error in report.failures), (
        report.summary()
    )


@pytest.mark.parametrize("kind", ["scalar", "array", "array_disjoint"])
def test_extra_fresh_cell_files_cannot_bypass_source_comparison(
    staged, kind, monkeypatch
):
    stage, out = staged
    tag = f"gpt2_mcword_entropy_N4_M3_a1_s0_{DATASET}"
    source = (
        f"{DATASET}.parquet" if kind == "scalar" else f"mcword_samples/{tag}.parquet"
    )
    extra = "extra.parquet" if kind == "scalar" else "mcword_samples/extra.parquet"
    table = pq.read_table(stage / source, filters=[("tag", "==", tag)])
    column = "value" if kind == "scalar" else "samples"
    values = table[column].to_pylist()
    if kind == "scalar":
        values[0] += 1.0
    else:
        values[0][0] = 12345.0
    if kind == "array_disjoint":
        ids = [
            f"not-in-authenticated-corpus-{sid}"
            for sid in table["stimulus_id"].to_pylist()
        ]
        table = table.set_column(
            table.schema.get_field_index("stimulus_id"),
            table.schema.field("stimulus_id"),
            pa.array(ids),
        )
    field = table.schema.field(column)
    table = table.set_column(
        table.schema.get_field_index(column), field, pa.array(values, type=field.type)
    )
    _rewrite(stage / extra, table)
    path = stage / layout.INDEX_FILE
    index = pq.read_table(path)
    row = next(
        row for row in index.to_pylist() if row["tag"] == tag and row["file"] == source
    )
    row["file"] = extra
    _rewrite(
        path,
        pa.concat_tables([index, pa.Table.from_pylist([row], schema=index.schema)]),
    )
    report = verify(stage, [("w", out)], names=[DATASET], corpora=ToyCorpora())
    assert not report.ok, report.summary()
    assert any(
        extra in error and "source comparison" in error for error in report.failures
    )
    if kind == "array_disjoint":
        from replica import predictors

        monkeypatch.setattr(predictors, "_hub_file", lambda name: str(stage / name))
        for name in ("_tag_index", "_index_methods", "_indexed_table"):
            monkeypatch.setattr(predictors, name, getattr(predictors, name).__wrapped__)
        loaded = predictors.load_samples_for_tag(tag)
        assert loaded["not-in-authenticated-corpus-toy_1"][0][0] == 12345.0


def test_extra_array_column_cannot_bypass_source_comparison(staged):
    stage, out = staged
    tag = f"gpt2_mcword_entropy_N4_M3_a1_s0_{DATASET}"
    path = stage / "mcword_samples" / f"{tag}.parquet"
    table = pq.read_table(path)
    field = table.schema.field("samples").with_name("unverified_draws")
    table = table.append_column(
        field, pa.array([[12345.0]] * table.num_rows, type=field.type)
    )
    _rewrite(path, table)
    report = verify(stage, [("w", out)], names=[DATASET], corpora=ToyCorpora())
    assert not report.ok, report.summary()
    assert any("array columns differ from source" in error for error in report.failures)


@pytest.mark.parametrize(
    "kind",
    [pa.list_(pa.float16()), pa.list_(pa.float64()), pa.large_list(pa.float32())],
)
def test_array_storage_type_must_match_its_declared_family(staged, kind):
    stage, out = staged
    tag = f"gpt2_mcword_entropy_N4_M3_a1_s0_{DATASET}"
    path = stage / "mcword_samples" / f"{tag}.parquet"
    table = pq.read_table(path)
    field = table.schema.field("samples").with_type(kind)
    values = table["samples"].to_pylist()
    if pa.types.is_float16(kind.value_type):
        values = [np.asarray(row, dtype=np.float16) for row in values]
    table = table.set_column(
        table.schema.get_field_index("samples"), field, pa.array(values, type=kind)
    )
    _rewrite(path, table)
    report = verify(stage, [("w", out)], names=[DATASET], corpora=ToyCorpora())
    assert not report.ok, report.summary()
    assert any(
        "type differs from declared storage" in error for error in report.failures
    )


@pytest.mark.parametrize("same_file", [False, True])
def test_alias_rows_cannot_escape_fresh_source_comparison(tmp_path, corpora, same_file):
    out, stage = tmp_path / "out", tmp_path / "stage"
    tag = f"gpt2_restricted_first_token_entropy_{DATASET}"
    alias = tag.replace("restricted_first_token_entropy", "next_word_entropy")
    payload = word_payload(
        "restricted_first_token_entropy",
        extra=lambda _sid, _i: {"next_log_probs": np.array([-0.5, -0.75])},
    )
    write_cell(out, tag, payload, method="restricted_first_token_entropy")
    assert export(stage, [("w", out)], corpora=corpora).ok
    source = f"next_log_probs/{tag}.parquet"
    target = source if same_file else "next_log_probs/extra.parquet"
    original = pq.read_table(stage / source)
    injected = original
    for name, values in {
        "tag": [alias] * original.num_rows,
        "stimulus_id": [
            "unknown-" + sid for sid in original["stimulus_id"].to_pylist()
        ],
        "logprobs": [[12345.0, 12345.0]] * original.num_rows,
    }.items():
        field = original.schema.field(name)
        injected = injected.set_column(
            original.schema.get_field_index(name),
            field,
            pa.array(values, type=field.type),
        )
    _rewrite(
        stage / target,
        pa.concat_tables([original, injected]) if same_file else injected,
    )
    path = stage / layout.INDEX_FILE
    index = pq.read_table(path)
    row = next(row for row in index.to_pylist() if row["file"] == source)
    row.update(tag=alias, file=target)
    _rewrite(
        path,
        pa.concat_tables([index, pa.Table.from_pylist([row], schema=index.schema)]),
    )
    report = verify(stage, [("w", out)], names=[DATASET], corpora=corpora)
    assert not report.ok, report.summary()
    assert any("tag differs from source identity" in error for error in report.failures)
