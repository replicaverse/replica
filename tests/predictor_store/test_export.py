"""Export of synthetic cells into the staged store tree."""

import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from replica.predictor_store import export as export_module
from replica.predictor_store import layout
from replica.predictor_store.export import export
from replica.utils.naming import safe_lm
from tests.predictor_store._cells import (
    DATASET,
    TEXTS,
    TOKENS,
    FakeHub,
    ToyCorpora,
    character_payload,
    legacy_word_table,
    rows,
    token_payload,
    token_table,
    word_payload,
    write_cell,
    write_cell_under_old_name,
)


def _index(stage, tag=None):
    out = rows(stage, layout.INDEX_FILE)
    return [r for r in out if tag is None or r["tag"] == tag]


@pytest.mark.parametrize(
    ("model", "method", "unit_rule"),
    [
        ("gpt2", "restricted_first_token_entropy", "whitespace_continuation"),
        ("wordfreq-zh", "unigram_surprisal", "corpus_segment_lookup"),
    ],
)
def test_word_table_carries_text_units_and_storage_columns(
    tmp_path, corpora, model, method, unit_rule
):
    out = tmp_path / "wave" / "out"
    tag = f"{model}_{method}_{DATASET}"
    write_cell(
        out,
        tag,
        word_payload(method, 2.0),
        method=method,
        model_id=model,
        unit_rule=unit_rule,
    )
    stage = tmp_path / "stage"
    report = export(stage, [("wave", out)], corpora=corpora)
    assert report.ok and report.cells == 1, report.summary()
    table = pq.read_table(stage / f"{DATASET}.parquet")
    assert table.schema.names == layout.WORD_SCHEMA.names
    got = sorted(table.to_pylist(), key=lambda r: (r["stimulus_id"], r["unit_index"]))
    words = [w for text in TEXTS.values() for w in text.split()]
    assert [r["text_unit"] for r in got] == words
    assert [r["value"] for r in got] == [2.0, 4.0, 2.0, 4.0, 6.0]
    assert {r["value_unit"] for r in got} == {"bits"}
    assert {(r["layer"], r["storage_class"], r["unit_rule"]) for r in got} == {
        (1, "unit_bound", unit_rule)
    }
    (index,) = _index(stage, tag)
    assert index["file"] == f"{DATASET}.parquet"
    assert index["predictor"] == method
    assert (index["n_stimuli"], index["n_rows"]) == (2, 5)
    assert (index["unit"], index["unit_rule"], index["wave"]) == (
        "words",
        unit_rule,
        "wave",
    )
    assert index["provenance"] == "manifest" and index["code_commit"] == "f" * 40
    assert (stage / layout.MANIFESTS_DIR / f"{tag}.json").exists()


def test_export_and_read_preserve_opaque_stimulus_ids(tmp_path, monkeypatch, corpora):
    from replica.predictors import _reconstruct
    from tests.predictor_store import _cells

    texts = {"trial_1": "ab cd", "trial_1.0": "ef gh", "trial_².0": "ij kl"}
    monkeypatch.setattr(_cells, "TEXTS", texts)
    payload = word_payload()
    out = tmp_path / "out"
    tag = f"gpt2_surprisal_{DATASET}"
    write_cell(out, tag, payload, method="surprisal")
    stage = tmp_path / "stage"
    assert export(stage, [("w", out)], corpora=corpora).ok
    actual = _reconstruct(pq.read_table(stage / f"{DATASET}.parquet"))
    assert set(actual) == set(texts)
    for sid, records in actual.items():
        assert [r.values for r in records] == [r.values for r in payload[sid]]
        assert [r.metadata["text_unit"] for r in records] == texts[sid].split()


@pytest.mark.parametrize("unit", ["words", "tokens"])
def test_export_rejects_same_length_corpus_changes(tmp_path, unit):
    class ChangedCorpora(ToyCorpora):
        def stimuli(self, dataset):
            stimuli = super().stimuli(dataset)
            stimuli[0].text = "xy cd"
            return stimuli

    tag = f"gpt2_surprisal{'_tokens' if unit == 'tokens' else ''}_{DATASET}"
    out = tmp_path / "out"
    write_cell(
        out,
        tag,
        token_payload() if unit == "tokens" else word_payload(),
        method="surprisal",
        unit=unit,
    )
    stage = tmp_path / "stage"
    report = export(stage, [("w", out)], corpora=ChangedCorpora())
    assert not report.ok and report.cells == 0
    assert "corpus_sha256" in report.misaligned_cells[tag]
    assert not (stage / layout.MANIFESTS_DIR / f"{tag}.json").exists()


def test_export_rejects_a_missing_corpus_digest(tmp_path, corpora):
    tag = f"gpt2_surprisal_{DATASET}"
    out = tmp_path / "out"
    path = write_cell(out, tag, word_payload(), method="surprisal")
    sidecar = path.with_name(path.name + ".manifest.json")
    manifest = json.loads(sidecar.read_text())
    del manifest["identity"]["corpus_sha256"]
    sidecar.write_text(json.dumps(manifest))
    report = export(tmp_path / "stage", [("w", out)], corpora=corpora)
    assert not report.ok and report.cells == 0
    assert "corpus_sha256" in report.misaligned_cells[tag]


@pytest.mark.parametrize(
    "model_id", ["org_team/model_name", "plain_model", "org/model:revision"]
)
def test_export_preserves_the_recorded_model_identity(tmp_path, corpora, model_id):
    out = tmp_path / "out"
    tag = f"{safe_lm(model_id)}_surprisal_{DATASET}"
    path = write_cell(out, tag, word_payload(), method="surprisal", model_id=model_id)
    before = path.read_bytes()
    stage = tmp_path / "stage"
    assert export(stage, [("w", out)], corpora=corpora).ok
    assert {row["model"] for row in _index(stage)} == {model_id}
    assert {row["model"] for row in rows(stage, layout.COVERAGE_FILE)} == {model_id}
    assert path.read_bytes() == before


def test_token_table_carries_the_token_identity(tmp_path, corpora):
    out = tmp_path / "out"
    tag = f"gpt2_surprisal_tokens_{DATASET}"
    write_cell(out, tag, token_payload(), method="surprisal", unit="tokens")
    stage = tmp_path / "stage"
    assert export(stage, [("w", out)], corpora=corpora).ok
    table = pq.read_table(stage / layout.TOKENS_DIR / f"{DATASET}.parquet")
    assert table.schema.names == layout.TOKEN_SCHEMA.names
    got = sorted(table.to_pylist(), key=lambda r: (r["stimulus_id"], r["token_index"]))
    expected = [
        (sid, i, start, end, token_id, token)
        for sid, tokens in TOKENS.items()
        for i, (start, end, token_id, token) in enumerate(tokens)
    ]
    assert [
        (
            r["stimulus_id"],
            r["token_index"],
            r["span_start"],
            r["span_end"],
            r["token_id"],
            r["token"],
        )
        for r in got
    ] == expected
    assert [r["value"] for r in got] == [-0.5, -1.0, -0.5, -1.0, -1.5, -2.0]
    assert not (stage / f"{DATASET}.parquet").exists()
    (index,) = _index(stage, tag)
    assert index["unit"] == "tokens" and index["storage_class"] == "positional"
    assert index["file"] == f"{layout.TOKENS_DIR}/{DATASET}.parquet"


def test_character_table_uses_the_word_schema(tmp_path, corpora):
    out = tmp_path / "out"
    tag = f"gpt2_charbeam_surprisal_K5_chars_{DATASET}"
    write_cell(out, tag, character_payload(), method="char_beam", unit="characters")
    stage = tmp_path / "stage"
    assert export(stage, [("w", out)], corpora=corpora).ok
    table = pq.read_table(stage / layout.CHARACTERS_DIR / f"{DATASET}.parquet")
    assert table.schema.names == layout.WORD_SCHEMA.names
    got = sorted(table.to_pylist(), key=lambda r: (r["stimulus_id"], r["unit_index"]))
    assert [r["text_unit"] for r in got] == list(TEXTS["toy_1"]) + list(TEXTS["toy_2"])
    assert got[2]["text_unit"] == " "
    (index,) = _index(stage, tag)
    assert index["unit"] == "characters" and index["beam_width"] == 5


def test_derived_word_table_names_its_source(tmp_path, corpora):
    out = tmp_path / "out"
    source = f"gpt2_surprisal_tokens_{DATASET}"
    write_cell(out, source, token_payload(), method="surprisal", unit="tokens")
    tag = f"gpt2_surprisal_{DATASET}"
    write_cell(
        out,
        tag,
        word_payload(extra=lambda sid, i: {"surprisal": 1.5 * (i + 1)}),
        method="surprisal",
        derivation={
            "source": f"{source}.pkl",
            "source_sha256": "3" * 64,
            "unit": "tokens",
            "rule": "sum",
            "keys": {"log_p_observed": "sum"},
            "finisher": "surprisal_bits",
            "code_commit": "f" * 40,
        },
    )
    stage = tmp_path / "stage"
    assert export(stage, [("w", out)], corpora=corpora).ok
    by_predictor = {r["predictor"]: r for r in _index(stage, tag)}
    assert set(by_predictor) == {"log_p_observed", "surprisal"}
    for row in by_predictor.values():
        assert row["derivation_source"] == source
        assert row["storage_class"] == "positional" and row["unit_rule"] is None
    assert by_predictor["surprisal"]["value_unit"] == "bits"
    (source_row,) = _index(stage, source)
    assert source_row["derivation_source"] is None
    # The cell carries two predictors; the README counts it as one cell.
    readme = (stage / "README.md").read_text()
    assert "1 word cells are aggregated from a token table" in readme
    assert "| `surprisal` | sum | 1 |" in readme


def test_layer_two_cell_names_its_study(tmp_path, corpora):
    out = tmp_path / "out"
    tag = f"gpt2_modelblocks_surprisal_{DATASET}"
    write_cell(out, tag, word_payload(), method="modelblocks_surprisal")
    stage = tmp_path / "stage"
    report = export(stage, [("w", out)], corpora=corpora)
    assert report.by_unit_layer == {"words/layer2": 1}
    (index,) = _index(stage, tag)
    assert (index["layer"], index["study"]) == (2, "clark2025wordentropy")
    assert {r["layer"] for r in rows(stage, f"{DATASET}.parquet")} == {2}


def test_quarantined_cells_and_plans_fill_the_coverage_table(tmp_path, corpora):
    out = tmp_path / "wave2" / "out"
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
        f"gpt2_modelblocks_surprisal_{DATASET}",
        word_payload(),
        method="modelblocks_surprisal",
    )
    write_cell(
        out,
        f"gpt2_restricted_first_token_renyi_entropy_a0.5_{DATASET}",
        word_payload("restricted_first_token_renyi_entropy_alpha0.5"),
        method="restricted_first_token_renyi_entropy",
        status="quarantined",
        failed=[("toy_2", "ValueError: beam is empty")],
    )
    plan = tmp_path / "plan.tsv"
    plan.write_text(
        "\t".join(
            [
                "complete",
                "surprisal",
                DATASET,
                "gpt2",
                f"gpt2_surprisal_tokens_{DATASET}.pkl;gpt2_surprisal_{DATASET}.pkl",
                "python -m replica.compute --method surprisal",
            ]
        )
        + "\n"
        + "\t".join(
            [
                "running",
                "restricted_first_token_renyi_entropy",
                DATASET,
                "gpt2",
                f"gpt2_restricted_first_token_renyi_entropy_a0.5_{DATASET}.pkl",
                "python -m replica.compute --method restricted_first_token_renyi_entropy",
            ]
        )
        + "\n"
        + "\t".join(
            [
                "missing",
                "first_token_entropy",
                DATASET,
                "gpt2",
                f"gpt2_ft_entropy_{DATASET}.pkl",
                "python -m replica.compute --method first_token_entropy",
            ]
        )
        + "\n"
    )
    verification = tmp_path / "wave2" / "verify" / "verification.json"
    verification.parent.mkdir(parents=True)
    verification.write_text(
        json.dumps(
            [
                {"tag": f"gpt2_surprisal_{DATASET}", "verdict": "pass", "flags": []},
                {
                    "tag": f"gpt2_surprisal_tokens_{DATASET}",
                    "verdict": "check",
                    "flags": ["1 words empty or containing whitespace"],
                },
            ]
        )
    )
    stage = tmp_path / "stage"
    report = export(
        stage,
        [("wave2", out)],
        verifications=[("wave2", verification)],
        coverage=[plan],
        corpora=corpora,
    )
    assert report.ok and report.cells == 3 and report.quarantined == 1
    coverage = {r["tag"]: r for r in rows(stage, layout.COVERAGE_FILE)}
    assert report.coverage_rows == len(coverage) == 5
    assert coverage[f"gpt2_surprisal_tokens_{DATASET}"]["status"] == "exported"
    assert coverage[f"gpt2_surprisal_tokens_{DATASET}"]["unit"] == "tokens"
    assert coverage[f"gpt2_surprisal_{DATASET}"]["reason_class"] == "complete"
    quarantined = coverage[f"gpt2_restricted_first_token_renyi_entropy_a0.5_{DATASET}"]
    assert quarantined["status"] == "quarantined"
    assert quarantined["reason_class"] == "running"
    assert "beam is empty" in quarantined["detail"]
    absent = coverage[f"gpt2_ft_entropy_{DATASET}"]
    assert absent["status"] == "absent" and absent["detail"].startswith("python -m")
    unplanned = coverage[f"gpt2_modelblocks_surprisal_{DATASET}"]
    assert unplanned["status"] == "exported" and unplanned["reason_class"] is None
    assert unplanned["model"] == "gpt2"
    assert not (
        stage / f"gpt2_restricted_first_token_renyi_entropy_a0.5_{DATASET}.parquet"
    ).exists()
    assert "restricted_first_token_renyi_entropy_alpha0.5" not in {
        r["predictor"] for r in rows(stage, f"{DATASET}.parquet")
    }
    verdicts = {
        r["tag"]: (r["verification"], r["verification_file"]) for r in _index(stage)
    }
    assert verdicts[f"gpt2_surprisal_{DATASET}"] == ("pass", "verification/wave2.json")
    assert verdicts[f"gpt2_surprisal_tokens_{DATASET}"] == (
        "check",
        "verification/wave2.json",
    )
    assert verdicts[f"gpt2_modelblocks_surprisal_{DATASET}"] == (None, None)
    assert (
        stage / "verification" / "wave2.json"
    ).read_text() == verification.read_text()
    plan_json = json.loads((stage / layout.PUBLISH_FILE).read_text())
    assert (
        "README.md" in plan_json["written"]
        and layout.PUBLISH_FILE not in plan_json["written"]
    )
    readme = (stage / "README.md").read_text()
    assert "## Coverage and gaps" in readme and "quarantined" in readme
    assert "—" not in readme


def test_sample_arrays_and_lens_vectors_keep_their_folders(tmp_path, corpora, samples):
    out = tmp_path / "out"
    mc = f"gpt2_mcword_entropy_N4_M3_a1_s0_{DATASET}"
    write_cell(
        out,
        mc,
        word_payload("mc_word_entropy", 1.0, extra=samples),
        method="mc_word_entropy",
    )
    lens = f"gpt2_logit_lens_surprisal_{DATASET}"
    write_cell(
        out,
        lens,
        word_payload(
            "logit_lens_log_p_L0",
            -1.0,
            extra=lambda sid, i: {
                "logit_lens_log_p_L1": -2.0,
                "logit_lens_log_p_L2": -3.0,
            },
        ),
        method="logit_lens_surprisal",
    )
    stage = tmp_path / "stage"
    report = export(stage, [("w", out)], corpora=corpora)
    assert report.ok and report.dropped_arrays == {}
    table = pq.read_table(stage / "mcword_samples" / f"{mc}.parquet")
    assert table.schema.names == [
        "dataset",
        "tag",
        "stimulus_id",
        "unit_index",
        "text_unit",
        "samples",
        "value_unit",
    ]
    assert table.schema.field("samples").metadata == {b"unit": b"nats"}
    assert table.schema.field("samples").type.value_type == "float"
    first = table.slice(0, 1).to_pylist()[0]
    assert first["text_unit"] == "ab" and len(first["samples"]) == 4
    assert {
        r["predictor"] for r in rows(stage, f"{DATASET}.parquet") if r["tag"] == mc
    } == {"mc_word_entropy"}
    files = {(r["predictor"], r["file"]) for r in _index(stage, mc)}
    assert files == {
        ("mc_word_entropy", f"{DATASET}.parquet"),
        ("log_p_samples_nats", f"mcword_samples/{mc}.parquet"),
    }
    dense = pq.read_table(stage / "lens_layers" / f"{lens}.parquet")
    assert dense.schema.names == [
        "dataset",
        "tag",
        "stimulus_id",
        "unit_index",
        "text_unit",
        "log_p_layers",
        "value_unit",
    ]
    vectors = sorted(
        dense.to_pylist(), key=lambda r: (r["stimulus_id"], r["unit_index"])
    )
    assert vectors[0]["log_p_layers"] == [-1.0, -2.0, -3.0]
    assert vectors[1]["log_p_layers"] == [-2.0, -2.0, -3.0]
    assert lens not in {r["tag"] for r in rows(stage, f"{DATASET}.parquet")}
    assert [(r["predictor"], r["file"]) for r in _index(stage, lens)] == [
        ("log_p_layers", f"lens_layers/{lens}.parquet")
    ]


def test_merge_hub_carries_untouched_cells_and_lists_stale_files(tmp_path, corpora):
    fresh_surprisal = f"gpt2_surprisal_{DATASET}"
    fresh_cont = f"gpt2_cont_entropy_N4_L2_s0_{DATASET}"
    untouched = f"gpt2_restricted_first_token_entropy_{DATASET}"
    lens = f"gpt2_logit_lens_surprisal_{DATASET}"
    cont_samples = f"cont_entropy_samples/{fresh_cont}.parquet"
    hub = FakeHub(
        tmp_path / "hub",
        [
            {
                "tag": fresh_surprisal,
                "file": f"{DATASET}.parquet",
                "dataset": DATASET,
                "model": "gpt2",
                "model_kind": "lm",
                "method": "surprisal",
                "predictor": "log_p_observed",
                "value_unit": "nats",
                "n_stimuli": 1,
                "n_rows": 2,
                "provenance": "records",
            },
            {
                "tag": untouched,
                "file": f"{DATASET}.parquet",
                "dataset": DATASET,
                "model": "gpt2",
                "model_kind": "lm",
                "method": "restricted_first_token_entropy",
                "predictor": "restricted_first_token_entropy",
                "value_unit": "bits",
                "n_stimuli": 1,
                "n_rows": 2,
                "provenance": "records",
                "computed_at": "2026-08-01T00:00:00+00:00",
            },
            {
                "tag": fresh_cont,
                "file": f"{DATASET}.parquet",
                "dataset": DATASET,
                "model": "gpt2",
                "model_kind": "lm",
                "method": "continuation_entropy",
                "predictor": "continuation_entropy",
                "value_unit": "bits",
                "n_stimuli": 1,
                "n_rows": 2,
                "provenance": "none",
            },
            {
                "tag": fresh_cont,
                "file": cont_samples,
                "dataset": DATASET,
                "model": "gpt2",
                "model_kind": "lm",
                "method": "continuation_entropy",
                "predictor": "log_p_samples_nats",
                "value_unit": "nats",
                "n_stimuli": 1,
                "n_rows": 2,
                "provenance": "none",
            },
            {
                "tag": lens,
                "file": f"lens_layers/{lens}.parquet",
                "dataset": DATASET,
                "model": "gpt2",
                "model_kind": "lm",
                "method": "logit_lens_surprisal",
                "predictor": "log_p_layers",
                "value_unit": "nats",
                "n_stimuli": 1,
                "n_rows": 2,
                "provenance": "none",
            },
        ],
        {
            f"{DATASET}.parquet": legacy_word_table(
                DATASET,
                {
                    fresh_surprisal: {"log_p_observed": [-9.0, -9.5]},
                    untouched: {"restricted_first_token_entropy": [3.0, 4.0]},
                    fresh_cont: {"continuation_entropy": [7.0, 8.0]},
                },
            )
        },
    )
    out = tmp_path / "out"
    write_cell(out, fresh_surprisal, word_payload(), method="surprisal")
    write_cell(
        out,
        fresh_cont,
        word_payload("continuation_entropy", 5.0),
        method="continuation_entropy",
    )
    stage = tmp_path / "stage"
    report = export(stage, [("w", out)], hub=hub, corpora=corpora)
    assert report.ok
    assert report.carried_count_mismatches == []
    table = rows(stage, f"{DATASET}.parquet")
    by_tag = {}
    for row in table:
        by_tag.setdefault(row["tag"], []).append(row)
    assert set(by_tag) == {fresh_surprisal, fresh_cont, untouched}
    carried = sorted(by_tag[untouched], key=lambda r: r["unit_index"])
    assert [r["value"] for r in carried] == [3.0, 4.0]
    assert [r["text_unit"] for r in carried] == ["ab", "cd"]
    assert {(r["layer"], r["storage_class"], r["unit_rule"]) for r in carried} == {
        (1, "unit_bound", None)
    }
    assert sorted(r["value"] for r in by_tag[fresh_surprisal]) == [
        -3.0,
        -2.0,
        -2.0,
        -1.0,
        -1.0,
    ]
    assert all(r["value"] >= 5.0 for r in by_tag[fresh_cont])
    index = {(r["tag"], r["file"]): r for r in _index(stage)}
    assert set(index) == {
        (fresh_surprisal, f"{DATASET}.parquet"),
        (fresh_cont, f"{DATASET}.parquet"),
        (untouched, f"{DATASET}.parquet"),
        (lens, f"lens_layers/{lens}.parquet"),
    }
    kept = index[untouched, f"{DATASET}.parquet"]
    assert kept["wave"] is None and kept["provenance"] == "records"
    assert kept["computed_at"] == "2026-08-01T00:00:00+00:00"
    assert (kept["layer"], kept["storage_class"], kept["unit"]) == (
        1,
        "unit_bound",
        "words",
    )
    assert index[lens, f"lens_layers/{lens}.parquet"]["layer"] == 1
    assert index[fresh_surprisal, f"{DATASET}.parquet"]["wave"] == "w"
    plan = json.loads((stage / layout.PUBLISH_FILE).read_text())
    assert plan["deleted"] == [cont_samples]
    assert plan["replaced_tags"] == sorted([fresh_cont, fresh_surprisal])
    assert plan["base_revision"] == "a" * 40 and plan["repo"] == hub.repo
    assert report.carried_index_rows == 2
    assert f"{DATASET}.parquet" in plan["written"]
    readme = (stage / "README.md").read_text()
    assert "a" * 40 in readme
    assert f'REPO = "hf://datasets/{hub.repo}"' in readme


def test_a_carried_cell_keeps_its_wave_and_verdict(tmp_path, corpora):
    """The published index row's provenance travels with the carried cell."""
    carried = f"gpt2_restricted_first_token_entropy_{DATASET}"
    hub = FakeHub(
        tmp_path / "hub",
        [
            {
                "tag": carried,
                "file": f"{DATASET}.parquet",
                "dataset": DATASET,
                "model": "gpt2",
                "model_kind": "lm",
                "method": "restricted_first_token_entropy",
                "predictor": "restricted_first_token_entropy",
                "value_unit": "bits",
                "n_stimuli": 1,
                "n_rows": 2,
                "unit": "words",
                "wave": "wave2",
                "verification": "pass",
                "verification_file": "verification/wave2.json",
            }
        ],
        {
            f"{DATASET}.parquet": legacy_word_table(
                DATASET, {carried: {"restricted_first_token_entropy": [3.0, 4.0]}}
            )
        },
        keep=("unit", "wave", "verification", "verification_file"),
    )
    out = tmp_path / "out"
    fresh = f"gpt2_surprisal_{DATASET}"
    write_cell(out, fresh, word_payload(), method="surprisal")
    stage = tmp_path / "stage"
    report = export(stage, [("w", out)], hub=hub, corpora=corpora)
    assert report.ok and report.exported_tags == [fresh]
    row = _index(stage, carried)[0]
    assert (row["wave"], row["verification"], row["verification_file"]) == (
        "wave2",
        "pass",
        "verification/wave2.json",
    )
    readme = (stage / "README.md").read_text()
    assert "| w | 1 |" in readme and "| wave2 |" not in readme
    from replica.predictor_store.verify import verify

    readback = verify(stage, [("w", out)], names=[DATASET], corpora=ToyCorpora())
    assert readback.ok, readback.failures
    receipt = json.loads((stage / layout.READBACK_FILE).read_text())
    assert set(receipt["fresh_cells"]) == {fresh}
    assert carried not in receipt["fresh_cells"]


def test_a_carried_token_table_with_a_wave_passes_the_readback(tmp_path, corpora):
    """An unstaged file whose row has a wave is carried, not a missing fresh cell."""
    carried = f"gpt2_surprisal_tokens_{DATASET}"
    hub = FakeHub(
        tmp_path / "hub",
        [
            {
                "tag": carried,
                "file": f"{layout.TOKENS_DIR}/{DATASET}.parquet",
                "dataset": DATASET,
                "model": "gpt2",
                "model_kind": "lm",
                "method": "surprisal",
                "predictor": "log_p_observed",
                "value_unit": "nats",
                "n_stimuli": 1,
                "n_rows": 2,
                "unit": "tokens",
                "wave": "wave1-tokens",
            }
        ],
        {
            f"{layout.TOKENS_DIR}/{DATASET}.parquet": token_table(
                DATASET, {carried: {"log_p_observed": [-1.0, -2.0]}}
            )
        },
        keep=("unit", "wave"),
    )
    out = tmp_path / "out"
    write_cell(
        out,
        f"gpt2_restricted_first_token_entropy_{DATASET}",
        word_payload("restricted_first_token_entropy", 1.0),
        method="restricted_first_token_entropy",
    )
    stage = tmp_path / "stage"
    report = export(stage, [("w", out)], hub=hub, corpora=corpora)
    assert report.ok and not (stage / layout.TOKENS_DIR / f"{DATASET}.parquet").exists()
    assert _index(stage, carried)[0]["wave"] == "wave1-tokens"
    from replica.predictor_store.verify import verify

    readback = verify(stage, [("w", out)], names=[DATASET], corpora=ToyCorpora())
    assert readback.ok and readback.carried_index_rows == 1, readback.failures


def test_a_failing_cell_is_reported_and_keeps_its_published_rows(tmp_path, corpora):
    tag = f"gpt2_surprisal_{DATASET}"
    hub = FakeHub(
        tmp_path / "hub",
        [
            {
                "tag": tag,
                "file": f"{DATASET}.parquet",
                "dataset": DATASET,
                "model": "gpt2",
                "model_kind": "lm",
                "method": "surprisal",
                "predictor": "log_p_observed",
                "value_unit": "nats",
                "n_stimuli": 1,
                "n_rows": 2,
            }
        ],
        {
            f"{DATASET}.parquet": legacy_word_table(
                DATASET, {tag: {"log_p_observed": [-1.0, -2.0]}}
            )
        },
    )
    out = tmp_path / "out"
    broken = word_payload()
    broken["toy_1"][0].values["log_p_observed"] = np.array([1.0, 2.0])
    broken["toy_1"][0].values["unregistered_key"] = 1.0
    write_cell(out, tag, broken, method="surprisal")
    other = f"gpt2_restricted_first_token_entropy_{DATASET}"
    write_cell(
        out,
        other,
        word_payload("restricted_first_token_entropy", 1.0),
        method="restricted_first_token_entropy",
    )
    stage = tmp_path / "stage"
    report = export(stage, [("w", out)], hub=hub, corpora=corpora)
    assert not report.ok and len(report.failures) == 1 and tag in report.failures[0]
    assert report.cells == 1
    kept = [r for r in rows(stage, f"{DATASET}.parquet") if r["tag"] == tag]
    assert sorted(r["value"] for r in kept) == [-2.0, -1.0]
    assert report.replaced_tags == [] and report.files_deleted == []
    plan = json.loads((stage / layout.PUBLISH_FILE).read_text())
    assert plan["failures"] == report.failures


def test_a_pickle_rewritten_during_the_export_is_dropped(tmp_path, corpora):
    """A cell the compute replaces mid-run is taken back out of the stage.

    The corpus loads once, between the first cell and the second, so
    rewriting the second cell's pickle from the loader reproduces the race:
    the export read it, then the compute driver replaced it.
    """
    out = tmp_path / "out"
    kept = f"gpt2_restricted_first_token_entropy_{DATASET}"
    racing = f"gpt2_surprisal_{DATASET}"
    write_cell(
        out,
        kept,
        word_payload("restricted_first_token_entropy", 3.0),
        method="restricted_first_token_entropy",
    )
    write_cell(out, racing, word_payload(), method="surprisal")
    published = legacy_word_table(DATASET, {racing: {"log_p_observed": [-7.0, -8.0]}})
    hub = FakeHub(
        tmp_path / "hub",
        [
            {
                "tag": racing,
                "file": f"{DATASET}.parquet",
                "dataset": DATASET,
                "model": "gpt2",
                "model_kind": "lm",
                "method": "surprisal",
                "predictor": "log_p_observed",
                "value_unit": "nats",
                "n_stimuli": 1,
                "n_rows": 2,
            }
        ],
        {f"{DATASET}.parquet": published},
    )

    class Racing:
        """The toy corpus, rewriting the racing cell as the loader is called."""

        def names(self):
            return corpora.names()

        def word_spacing(self, dataset):
            return True

        def stimuli(self, dataset):
            write_cell(out, racing, word_payload(scale=-5.0), method="surprisal")
            return corpora.stimuli(dataset)

    stage = tmp_path / "stage"
    report = export(stage, [("w", out)], hub=hub, corpora=Racing())
    assert report.changed_during_export == [racing]
    assert report.cells == 1 and report.ok
    assert report.replaced_tags == [] and report.by_unit_layer == {"words/layer1": 1}
    by_tag = {}
    for row in rows(stage, f"{DATASET}.parquet"):
        by_tag.setdefault(row["tag"], []).append(row)
    assert set(by_tag) == {kept, racing}
    assert sorted(r["value"] for r in by_tag[racing]) == [-8.0, -7.0]
    assert not (stage / layout.MANIFESTS_DIR / f"{racing}.json").exists()
    assert (stage / layout.MANIFESTS_DIR / f"{kept}.json").exists()
    index = {(r["tag"], r["predictor"]): r for r in _index(stage)}
    restored = index[racing, "log_p_observed"]
    assert restored["wave"] is None
    assert (restored["n_stimuli"], restored["n_rows"]) == (1, 2)
    assert index[kept, "restricted_first_token_entropy"]["wave"] == "w"
    coverage = {r["tag"]: r for r in rows(stage, layout.COVERAGE_FILE)}
    # The store's own rows came back, so the release still ships the cell.
    assert coverage[racing]["status"] == "carried"
    assert "the pickle changed during the export" in coverage[racing]["detail"]
    assert coverage[kept]["status"] == "exported"
    plan = json.loads((stage / layout.PUBLISH_FILE).read_text())
    assert f"{layout.MANIFESTS_DIR}/{racing}.json" not in plan["written"]
    assert "changed during the export" in report.summary()


def test_a_cell_that_appears_during_the_export_is_reported(tmp_path, corpora):
    out = tmp_path / "out"
    write_cell(out, f"gpt2_surprisal_{DATASET}", word_payload(), method="surprisal")
    late = f"gpt2_restricted_first_token_entropy_{DATASET}"

    class Late:
        """The toy corpus, writing a new cell as the loader is called."""

        def names(self):
            return corpora.names()

        def word_spacing(self, dataset):
            return True

        def stimuli(self, dataset):
            write_cell(
                out,
                late,
                word_payload("restricted_first_token_entropy", 1.0),
                method="restricted_first_token_entropy",
            )
            return corpora.stimuli(dataset)

    stage = tmp_path / "stage"
    report = export(stage, [("w", out)], corpora=Late())
    assert report.appeared_during_export == [late]
    assert report.changed_during_export == []
    assert report.cells == 1
    assert late not in {r["tag"] for r in rows(stage, f"{DATASET}.parquet")}
    assert "appeared during the export" in report.summary()


def test_every_staged_cell_has_its_manifest(tmp_path, corpora):
    out = tmp_path / "out"
    write_cell(
        out,
        f"gpt2_surprisal_tokens_{DATASET}",
        token_payload(),
        method="surprisal",
        unit="tokens",
    )
    write_cell(
        out,
        f"gpt2_restricted_first_token_entropy_{DATASET}",
        word_payload("restricted_first_token_entropy", 1.0),
        method="restricted_first_token_entropy",
    )
    stage = tmp_path / "stage"
    report = export(stage, [("w", out)], corpora=corpora)
    staged = {p.stem for p in (stage / layout.MANIFESTS_DIR).glob("*.json")}
    assert staged == set(report.exported_tags)
    assert len(staged) == report.cells


def _published_token_cell(root, tag, dataset, values, extra_rows=()):
    """A hub holding tokens/<dataset>.parquet with *tag* and any extra cells."""
    cells = {tag: {"log_p_observed": values}}
    index = [
        {
            "tag": tag,
            "file": f"{layout.TOKENS_DIR}/{dataset}.parquet",
            "dataset": dataset,
            "model": "gpt2",
            "model_kind": "lm",
            "method": "surprisal",
            "predictor": "log_p_observed",
            "value_unit": "nats",
            "n_stimuli": 1,
            "n_rows": len(values),
        }
    ]
    for other, other_values in extra_rows:
        cells[other] = {"log_p_observed": other_values}
        index.append({**index[0], "tag": other, "n_rows": len(other_values)})
    return FakeHub(
        root,
        index,
        {f"{layout.TOKENS_DIR}/{dataset}.parquet": token_table(dataset, cells)},
    )


def test_a_rewritten_token_table_carries_the_cells_it_is_not_replacing(
    tmp_path, corpora
):
    """tokens/ and characters/ are shared tables, so a rewrite carries the rest."""
    recomputed = f"gpt2_surprisal_tokens_{DATASET}"
    untouched = f"gpt2_ft_entropy_tokens_{DATASET}"
    hub = _published_token_cell(
        tmp_path / "hub",
        recomputed,
        DATASET,
        [-9.0, -9.5],
        extra_rows=[(untouched, [-4.0, -4.5])],
    )
    out = tmp_path / "out"
    write_cell(out, recomputed, token_payload(), method="surprisal", unit="tokens")
    stage = tmp_path / "stage"
    report = export(stage, [("w", out)], hub=hub, corpora=corpora)
    assert report.ok
    table = rows(stage, f"{layout.TOKENS_DIR}/{DATASET}.parquet")
    by_tag = {}
    for row in table:
        by_tag.setdefault(row["tag"], []).append(row)
    assert set(by_tag) == {recomputed, untouched}
    carried = sorted(by_tag[untouched], key=lambda r: r["token_index"])
    assert [r["value"] for r in carried] == [-4.0, -4.5]
    assert [r["token"] for r in carried] == [t[3] for t in TOKENS["toy_1"][:2]]
    index = {(r["tag"], r["file"]): r for r in _index(stage)}
    kept = index[untouched, f"{layout.TOKENS_DIR}/{DATASET}.parquet"]
    assert kept["wave"] is None and (kept["n_stimuli"], kept["n_rows"]) == (1, 2)
    assert index[recomputed, f"{layout.TOKENS_DIR}/{DATASET}.parquet"]["wave"] == "w"
    plan = json.loads((stage / layout.PUBLISH_FILE).read_text())
    assert plan["deleted"] == []
    assert report.carried_index_rows == 1


def test_a_fresh_token_table_carries_the_published_cells_of_that_corpus(
    tmp_path, corpora
):
    """A run whose token cell is new still rewrites the corpus token table."""
    published = f"gpt2_ft_entropy_tokens_{DATASET}"
    fresh = f"gpt2_surprisal_tokens_{DATASET}"
    hub = _published_token_cell(tmp_path / "hub", published, DATASET, [-4.0, -4.5])
    out = tmp_path / "out"
    write_cell(out, fresh, token_payload(), method="surprisal", unit="tokens")
    stage = tmp_path / "stage"
    assert export(stage, [("w", out)], hub=hub, corpora=corpora).ok
    table = rows(stage, f"{layout.TOKENS_DIR}/{DATASET}.parquet")
    assert {r["tag"] for r in table} == {published, fresh}
    assert sorted(r["value"] for r in table if r["tag"] == published) == [-4.5, -4.0]
    assert {(r["tag"], r["file"]) for r in _index(stage)} == {
        (published, f"{layout.TOKENS_DIR}/{DATASET}.parquet"),
        (fresh, f"{layout.TOKENS_DIR}/{DATASET}.parquet"),
    }


def test_a_partial_file_listing_still_carries_every_indexed_table(tmp_path, corpora):
    """A table the index names is carried although the listing omits it."""
    fresh = f"gpt2_surprisal_{DATASET}"
    untouched = f"gpt2_restricted_first_token_entropy_{DATASET}"

    class PartialHub(FakeHub):
        def files(self):
            return {layout.INDEX_FILE}

    hub = PartialHub(
        tmp_path / "hub",
        [
            {
                "tag": untouched,
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
                DATASET, {untouched: {"restricted_first_token_entropy": [3.0, 4.0]}}
            )
        },
    )
    out = tmp_path / "out"
    write_cell(out, fresh, word_payload(), method="surprisal")
    stage = tmp_path / "stage"
    report = export(stage, [("w", out)], hub=hub, corpora=corpora)
    assert report.ok
    carried = [r for r in rows(stage, f"{DATASET}.parquet") if r["tag"] == untouched]
    assert sorted(r["value"] for r in carried) == [3.0, 4.0]
    assert {r["tag"] for r in _index(stage)} == {fresh, untouched}


def test_a_missing_indexed_table_stops_the_export(tmp_path, corpora):
    """A table the index names but the hub cannot serve fails the export."""
    fresh = f"gpt2_surprisal_{DATASET}"
    hub = FakeHub(
        tmp_path / "hub",
        [
            {
                "tag": f"gpt2_restricted_first_token_entropy_{DATASET}",
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
        {},
    )
    out = tmp_path / "out"
    write_cell(out, fresh, word_payload(), method="surprisal")
    with pytest.raises(OSError, match="is not published"):
        export(tmp_path / "stage", [("w", out)], hub=hub, corpora=corpora)


def test_a_published_table_with_no_index_row_stops_the_export(tmp_path, corpora):
    """A table the store holds but does not index would lose its cells."""
    hub = FakeHub(
        tmp_path / "hub",
        [],
        {
            f"{DATASET}.parquet": legacy_word_table(
                DATASET,
                {
                    f"gpt2_charbeam_surprisal_K5_{DATASET}": {
                        "log_p_observed": [-1.0, -2.0]
                    }
                },
            )
        },
    )
    # The table is on the hub; the index, which says what it holds, is not.
    pq.write_table(
        pq.read_table(hub.root / f"{DATASET}.parquet"), hub.root / f"{DATASET}.parquet"
    )
    pq.write_table(hub.index().slice(0, 0), hub.root / "tags.parquet")
    out = tmp_path / "out"
    write_cell(out, f"gpt2_surprisal_{DATASET}", word_payload(), method="surprisal")
    with pytest.raises(ValueError, match="without an index row"):
        export(tmp_path / "stage", [("w", out)], hub=hub, corpora=corpora)


def test_coverage_marks_a_carried_table_rather_than_missing(tmp_path, corpora):
    """A planned cell the release ships from the store is marked carried."""
    carried = f"gpt2_restricted_first_token_entropy_{DATASET}"
    fresh = f"gpt2_surprisal_{DATASET}"
    hub = FakeHub(
        tmp_path / "hub",
        [
            {
                "tag": carried,
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
                DATASET, {carried: {"restricted_first_token_entropy": [3.0, 4.0]}}
            )
        },
    )
    out = tmp_path / "out"
    write_cell(out, fresh, word_payload(), method="surprisal")
    quarantined = f"gpt2_ft_entropy_{DATASET}"
    write_cell(
        out,
        quarantined,
        word_payload("token_entropy", 1.0),
        method="first_token_entropy",
        status="quarantined",
        failed=[("toy_2", "RuntimeError: out of memory")],
    )
    plan = tmp_path / "plan.tsv"
    plan.write_text(
        "\t".join(
            [
                "running",
                "restricted_first_token_entropy",
                DATASET,
                "gpt2",
                f"{carried}.pkl",
                "python -m replica.compute --method restricted_first_token_entropy",
            ]
        )
        + "\n"
    )
    stage = tmp_path / "stage"
    hub_index_rows = len(hub.index().to_pylist())
    assert export(stage, [("w", out)], coverage=[plan], hub=hub, corpora=corpora).ok
    coverage = {r["tag"]: r for r in rows(stage, layout.COVERAGE_FILE)}
    assert coverage[carried]["status"] == "carried"
    assert coverage[carried]["detail"] == "carried from an earlier release"
    assert coverage[carried]["reason_class"] == "running"
    assert coverage[fresh]["status"] == "exported"
    assert coverage[quarantined]["status"] == "quarantined"
    assert "out of memory" in coverage[quarantined]["detail"]
    assert hub_index_rows == 1


def test_a_filesystem_out_of_room_stops_the_export(tmp_path, corpora, monkeypatch):
    """Quota and disk-capacity errors stop the export."""
    import errno

    out = tmp_path / "out"
    for index in range(3):
        write_cell(
            out,
            f"gpt2_token_renyi_entropy_a0.{index}_{DATASET}",
            word_payload(f"token_renyi_entropy_alpha0.{index}", 1.0),
            method="token_renyi_entropy",
        )
    seen = []
    real = export_module._export_cell

    def failing(cell, *args, **kwargs):
        seen.append(cell.stem)
        raise OSError(errno.EDQUOT, "Disk quota exceeded")

    monkeypatch.setattr(export_module, "_export_cell", failing)
    with pytest.raises(OSError, match="the export stops"):
        export(tmp_path / "stage", [("w", out)], corpora=corpora)
    assert len(seen) == 1, "the run continued past the first quota error"
    assert real is not failing


def test_a_single_unreadable_cell_still_only_fails_itself(tmp_path, corpora):
    out = tmp_path / "out"
    good = f"gpt2_restricted_first_token_entropy_{DATASET}"
    broken = f"gpt2_surprisal_{DATASET}"
    write_cell(
        out,
        good,
        word_payload("restricted_first_token_entropy", 1.0),
        method="restricted_first_token_entropy",
    )
    write_cell(out, broken, word_payload(), method="surprisal")
    (out / f"{broken}.pkl").write_bytes(b"not a pickle")
    stage = tmp_path / "stage"
    report = export(stage, [("w", out)], corpora=corpora)
    assert report.cells == 1
    assert report.discovery_problems and broken in report.discovery_problems[0]
    assert good in {r["tag"] for r in rows(stage, f"{DATASET}.parquet")}


def test_a_shard_set_without_a_canonical_pickle_is_not_a_cell(tmp_path, corpora):
    """Slices of a corpus are not a cell until merge_shards.py has run.

    A sharded compute writes one pickle per stimulus slice. Staging them
    under the shared tag would merge them without checking that they tile the
    corpus from stimulus zero, so a half-landed set would look complete.
    """
    out = tmp_path / "out"
    tag = f"gpt2_surprisal_{DATASET}"
    payload = word_payload()
    first = {"toy_1": payload["toy_1"]}
    second = {"toy_2": payload["toy_2"]}
    write_cell(out, f"{tag}.shard00000_00001", first, method="surprisal")
    write_cell(out, f"{tag}.shard00001_00002", second, method="surprisal")
    other = f"gpt2_restricted_first_token_entropy_{DATASET}"
    write_cell(
        out,
        other,
        word_payload("restricted_first_token_entropy", 2.0),
        method="restricted_first_token_entropy",
    )
    plan = tmp_path / "plan.tsv"
    plan.write_text(
        "\t".join(
            [
                "running",
                "surprisal",
                DATASET,
                "gpt2",
                f"{tag}.pkl",
                "python -m replica.compute --method surprisal",
            ]
        )
        + "\n"
    )
    stage = tmp_path / "stage"
    report = export(stage, [("w", out)], coverage=[plan], corpora=corpora)
    assert report.cells == 1
    assert report.unmerged_shard_sets == {tag: 2}
    assert "unmerged shard sets" in report.summary()
    assert tag not in {r["tag"] for r in rows(stage, f"{DATASET}.parquet")}
    assert tag not in {r["tag"] for r in _index(stage)}
    assert not (stage / layout.MANIFESTS_DIR / f"{tag}.json").exists()
    assert not list((stage / layout.MANIFESTS_DIR).glob(f"{tag}.shard*.json"))
    coverage = {r["tag"]: r for r in rows(stage, layout.COVERAGE_FILE)}
    assert coverage[tag]["status"] == "absent"
    assert "unmerged shard set of 2 files" in coverage[tag]["detail"]
    assert coverage[tag]["reason_class"] == "running"
    assert coverage[other]["status"] == "exported"
    assert report.ok


def test_a_canonical_pickle_beside_leftover_shards_exports_alone(tmp_path, corpora):
    """The merged pickle is the cell; slices left beside it are ignored."""
    out = tmp_path / "out"
    tag = f"gpt2_surprisal_{DATASET}"
    write_cell(out, tag, word_payload(), method="surprisal")
    write_cell(
        out,
        f"{tag}.shard00000_00001",
        {"toy_1": word_payload(scale=-99.0)["toy_1"]},
        method="surprisal",
    )
    stage = tmp_path / "stage"
    report = export(stage, [("w", out)], corpora=corpora)
    assert report.cells == 1 and report.unmerged_shard_sets == {}
    staged = sorted(
        (r["stimulus_id"], r["unit_index"], r["value"])
        for r in rows(stage, f"{DATASET}.parquet")
        if r["predictor"] == "log_p_observed"
    )
    assert [value for _, _, value in staged] == [-1.0, -2.0, -1.0, -2.0, -3.0]
    assert (stage / layout.MANIFESTS_DIR / f"{tag}.json").exists()
    index = {r["tag"] for r in _index(stage)}
    assert index == {tag}


def test_a_shard_set_the_store_already_publishes_reads_as_carried(tmp_path, corpora):
    """A shard set whose merged cell is already published stays shipped."""
    tag = f"gpt2_surprisal_{DATASET}"
    hub = FakeHub(
        tmp_path / "hub",
        [
            {
                "tag": tag,
                "file": f"{DATASET}.parquet",
                "dataset": DATASET,
                "model": "gpt2",
                "model_kind": "lm",
                "method": "surprisal",
                "predictor": "log_p_observed",
                "value_unit": "nats",
                "n_stimuli": 1,
                "n_rows": 2,
            }
        ],
        {
            f"{DATASET}.parquet": legacy_word_table(
                DATASET, {tag: {"log_p_observed": [-7.0, -8.0]}}
            )
        },
    )
    out = tmp_path / "out"
    write_cell(
        out,
        f"{tag}.shard00000_00001",
        {"toy_1": word_payload()["toy_1"]},
        method="surprisal",
    )
    write_cell(
        out,
        f"gpt2_restricted_first_token_entropy_{DATASET}",
        word_payload("restricted_first_token_entropy", 2.0),
        method="restricted_first_token_entropy",
    )
    stage = tmp_path / "stage"
    report = export(stage, [("w", out)], hub=hub, corpora=corpora)
    assert report.unmerged_shard_sets == {tag: 1}
    assert report.replaced_tags == []
    carried = [r for r in rows(stage, f"{DATASET}.parquet") if r["tag"] == tag]
    assert sorted(r["value"] for r in carried) == [-8.0, -7.0]
    coverage = {r["tag"]: r for r in rows(stage, layout.COVERAGE_FILE)}
    assert coverage[tag]["status"] == "carried"
    assert coverage[tag]["model"] is None


def test_a_shard_verification_record_does_not_become_the_cells_verdict(
    tmp_path, corpora
):
    out = tmp_path / "out"
    tag = f"gpt2_surprisal_{DATASET}"
    write_cell(out, tag, word_payload(), method="surprisal")
    verification = tmp_path / "verification.json"
    verification.write_text(
        json.dumps(
            [
                {"tag": tag, "verdict": "pass", "flags": []},
                {
                    "tag": f"{tag}.shard00000_00001",
                    "verdict": "check",
                    "flags": ["stimulus ids differ from loader"],
                },
                {
                    "tag": f"{tag}.shard00001_00002",
                    "verdict": "check",
                    "flags": ["stimulus ids differ from loader"],
                },
            ]
        )
    )
    stage = tmp_path / "stage"
    report = export(
        stage,
        [("w", out)],
        verifications=[("w", verification)],
        corpora=corpora,
    )
    assert report.shard_verification_records == {"w": 2}
    verdicts = {r["tag"]: r["verification"] for r in _index(stage)}
    assert verdicts[tag] == "pass"
    staged = json.loads((stage / "verification" / "w.json").read_text())
    assert [r["tag"] for r in staged] == [tag]


def test_a_plan_naming_one_cell_twice_yields_one_coverage_row(tmp_path, corpora):
    out = tmp_path / "out"
    tag = f"gpt2_surprisal_{DATASET}"
    write_cell(out, tag, word_payload(), method="surprisal")
    plan = tmp_path / "plan.tsv"
    line = "\t".join(
        [
            "complete",
            "surprisal",
            DATASET,
            "gpt2",
            f"{tag}.pkl",
            "python -m replica.compute --method surprisal",
        ]
    )
    plan.write_text(line + "\n" + line + " --word-starters g-only\n")
    stage = tmp_path / "stage"
    report = export(stage, [("w", out)], coverage=[plan], corpora=corpora)
    coverage = rows(stage, layout.COVERAGE_FILE)
    assert len(coverage) == report.coverage_rows == 1
    assert coverage[0]["tag"] == tag and coverage[0]["reason_class"] == "complete"


def test_the_readme_names_the_reason_classes_the_coverage_table_uses(tmp_path, corpora):
    out = tmp_path / "out"
    tag = f"gpt2_surprisal_{DATASET}"
    write_cell(out, tag, word_payload(), method="surprisal")
    plan = tmp_path / "plan.tsv"
    plan.write_text(
        "\t".join(
            [
                "excluded_by_decision",
                "surprisal",
                DATASET,
                "gpt2",
                f"{tag}.pkl",
                "python -m replica.compute --method surprisal",
            ]
        )
        + "\n"
    )
    stage = tmp_path / "stage"
    assert export(stage, [("w", out)], coverage=[plan], corpora=corpora).ok
    readme = (stage / "README.md").read_text()
    assert "`excluded_by_decision`" in readme
    assert "skipped_by_decision" not in readme


def _stale_table(tag: str, texts: list[str]) -> "pa.Table":
    """A published toy_1 cell with text-level rows."""
    table = legacy_word_table(
        DATASET, {tag: {"restricted_first_token_entropy": [3.0] * len(texts)}}
    )
    return table.set_column(
        table.schema.get_field_index("text_unit"), "text_unit", pa.array(texts)
    )


def _hub_with_stale_cell(root, stale: str, texts: list[str], manifest: bool = True):
    hub = FakeHub(
        root,
        [
            {
                "tag": stale,
                "file": f"{DATASET}.parquet",
                "dataset": DATASET,
                "model": "gpt2",
                "model_kind": "lm",
                "method": "restricted_first_token_entropy",
                "predictor": "restricted_first_token_entropy",
                "value_unit": "bits",
                "n_stimuli": 1,
                "n_rows": len(texts),
            }
        ],
        {f"{DATASET}.parquet": _stale_table(stale, texts)},
    )
    if manifest:
        (root / layout.MANIFESTS_DIR).mkdir()
        (root / layout.MANIFESTS_DIR / f"{stale}.json").write_text("{}")
    return hub


def _plan(path, tag):
    path.write_text(
        f"missing\trestricted_first_token_entropy\t{DATASET}\tgpt2\t{tag}.pkl\tpython -m x\n"
    )
    return path


def test_a_carried_cell_off_the_loaders_grid_fails_the_export(tmp_path, corpora):
    """A published cell whose unit strings differ from the loader's is refused."""
    stale = f"gpt2_restricted_first_token_entropy_{DATASET}"
    hub = _hub_with_stale_cell(tmp_path / "hub", stale, ["cd", "ab"])
    out = tmp_path / "out"
    write_cell(out, f"gpt2_surprisal_{DATASET}", word_payload(), method="surprisal")
    stage = tmp_path / "stage"
    report = export(stage, [("w", out)], hub=hub, corpora=corpora)
    assert not report.ok and len(report.failures) == 1
    assert (
        "text_mismatch on 1 stimuli (toy_1 unit 0: stored 'cd', loader 'ab')"
        in report.failures[0]
    )
    assert report.misaligned_carried[stale].startswith(f"{DATASET}: text_mismatch")
    assert report.retired_tags == [] and report.files_deleted == []
    assert "pass --drop-misaligned" in report.summary()
    # The cell stays in the store until it is retired or recomputed.
    assert {r["tag"] for r in rows(stage, f"{DATASET}.parquet")} == {
        stale,
        f"gpt2_surprisal_{DATASET}",
    }
    assert _index(stage, stale) and _index(stage, stale)[0]["wave"] is None


def test_drop_misaligned_retires_a_carried_cell(tmp_path, corpora):
    """With the flag, the cell's rows, index row and manifest leave the store."""
    stale = f"gpt2_restricted_first_token_entropy_{DATASET}"
    hub = _hub_with_stale_cell(tmp_path / "hub", stale, ["ab"])
    out = tmp_path / "out"
    fresh = f"gpt2_surprisal_{DATASET}"
    write_cell(out, fresh, word_payload(), method="surprisal")
    stage = tmp_path / "stage"
    plan = _plan(tmp_path / "plan.tsv", stale)
    report = export(
        stage,
        [("w", out)],
        coverage=[plan],
        hub=hub,
        corpora=corpora,
        drop_misaligned=True,
    )
    assert report.ok
    assert report.retired_tags == [stale] and report.replaced_tags == []
    assert (
        report.misaligned_carried[stale]
        == f"{DATASET}: unit_count on 1 stimuli (toy_1: 1 units stored, 2 in the loader)"
    )
    assert {r["tag"] for r in rows(stage, f"{DATASET}.parquet")} == {fresh}
    assert {r["tag"] for r in _index(stage)} == {fresh}
    publish = json.loads((stage / layout.PUBLISH_FILE).read_text())
    assert publish["retired_tags"] == [stale]
    assert publish["deleted"] == [f"{layout.MANIFESTS_DIR}/{stale}.json"]
    coverage = {r["tag"]: r for r in rows(stage, layout.COVERAGE_FILE)}
    assert (
        coverage[stale]["status"] == "absent"
        and coverage[stale]["detail"] == "python -m x"
    )
    readme = (stage / "README.md").read_text()
    assert "### Cells retired from the store" in readme
    assert f"| `{DATASET}` | 1 |" in readme
    assert "- 1 published cells retired" in readme
    assert "—" not in readme


def test_a_character_cell_in_a_root_table_is_checked_at_its_own_grain(
    tmp_path, corpora
):
    """The index's unit, not the file, says which units a carried cell indexes."""
    chars = f"gpt2_charbeam_surprisal_K5_chars_{DATASET}"
    text = TEXTS["toy_1"]
    n = len(text)
    table = pa.table(
        {
            "dataset": pa.array([DATASET] * n),
            "tag": pa.array([chars] * n),
            "predictor": pa.array(["log_p_observed"] * n),
            "stimulus_id": pa.array(["toy_1"] * n),
            "unit_index": pa.array(list(range(n)), pa.int32()),
            "text_unit": pa.array(list(text)),
            "value": pa.array([-1.0] * n),
            "value_unit": pa.array(["nats"] * n),
        }
    )
    hub = FakeHub(
        tmp_path / "hub",
        [
            {
                "tag": chars,
                "file": f"{DATASET}.parquet",
                "dataset": DATASET,
                "model": "gpt2",
                "model_kind": "lm",
                "method": "char_beam",
                "predictor": "log_p_observed",
                "value_unit": "nats",
                "unit": "characters",
                "n_stimuli": 1,
                "n_rows": len(text),
            }
        ],
        {f"{DATASET}.parquet": table},
    )
    out = tmp_path / "out"
    write_cell(out, f"gpt2_surprisal_{DATASET}", word_payload(), method="surprisal")
    stage = tmp_path / "stage"
    report = export(stage, [("w", out)], hub=hub, corpora=corpora, drop_misaligned=True)
    assert report.ok and report.retired_tags == [] and report.misaligned_carried == {}
    assert chars in {r["tag"] for r in rows(stage, f"{DATASET}.parquet")}


class _PairCorpora:
    """The toy corpus with a second segmentation: one ``pairs`` unit per text."""

    def names(self):
        return [DATASET]

    def word_spacing(self, dataset):
        return True

    def stimuli(self, dataset):
        from replica.core import AnnotationLayer, Span

        stimuli = ToyCorpora().stimuli(dataset)
        for s in stimuli:
            s.add_layer(AnnotationLayer("pairs", [Span(0, len(s.text))], [s.text]))
        return stimuli


def _grain_hub(root, tag, unit, texts):
    n = len(texts)
    table = pa.table(
        {
            "dataset": pa.array([DATASET] * n),
            "tag": pa.array([tag] * n),
            "predictor": pa.array(["log_p_observed"] * n),
            "stimulus_id": pa.array(["toy_1"] * n),
            "unit_index": pa.array(list(range(n)), pa.int32()),
            "text_unit": pa.array(texts),
            "value": pa.array([-1.0] * n),
            "value_unit": pa.array(["nats"] * n),
        }
    )
    return FakeHub(
        root,
        [
            {
                "tag": tag,
                "file": f"{DATASET}.parquet",
                "dataset": DATASET,
                "model": "gpt2",
                "model_kind": "lm",
                "method": "surprisal",
                "predictor": "log_p_observed",
                "value_unit": "nats",
                "unit": unit,
                "n_stimuli": 1,
                "n_rows": n,
            }
        ],
        {f"{DATASET}.parquet": table},
        keep=("unit",),
    )


def test_a_cell_on_another_segmentation_is_checked_against_that_layer(tmp_path):
    """The index's unit names the layer; the loader's words are not the measure."""
    tag = f"gpt2_surprisal_{DATASET}"
    hub = _grain_hub(tmp_path / "hub", tag, "pairs", [TEXTS["toy_1"]])
    (tmp_path / "out").mkdir()
    report = export(
        tmp_path / "s1",
        [("w", tmp_path / "out")],
        hub=hub,
        corpora=_PairCorpora(),
        drop_misaligned=True,
    )
    assert report.ok and report.retired_tags == []
    hub = _grain_hub(tmp_path / "hub2", tag, "pairs", ["ab", "cd"])
    report = export(
        tmp_path / "s2",
        [("w", tmp_path / "out")],
        hub=hub,
        corpora=_PairCorpora(),
        drop_misaligned=True,
    )
    assert (
        report.retired_tags == [tag] and "unit_count" in report.misaligned_carried[tag]
    )
    # A grain the loader does not provide is reported, not compared with the words.
    hub = _grain_hub(tmp_path / "hub3", tag, "pairs", [TEXTS["toy_1"]])
    report = export(
        tmp_path / "s3",
        [("w", tmp_path / "out")],
        hub=hub,
        corpora=ToyCorpora(),
        drop_misaligned=True,
    )
    assert report.retired_tags == [] and not report.ok
    assert report.failures[0].startswith(
        f"{tag}: the loader has no 'pairs' layer on 2 stimuli (toy_1), so cells at that grain cannot be checked; a segmentation registers"
    )


def test_a_corpus_that_fails_to_load_fails_its_carried_cells_only(tmp_path, corpora):
    """The export reports the loader failure and still checks the other corpora."""
    stale = f"gpt2_restricted_first_token_entropy_{DATASET}"
    hub = _hub_with_stale_cell(tmp_path / "hub", stale, ["cd", "ab"], manifest=False)

    class Broken:
        def names(self):
            return corpora.names()

        def word_spacing(self, dataset):
            return True

        def stimuli(self, dataset):
            raise OSError(f"{dataset} is not downloaded")

    (tmp_path / "out").mkdir()
    stage = tmp_path / "stage"
    report = export(stage, [("w", tmp_path / "out")], hub=hub, corpora=Broken())
    assert not report.ok
    assert report.failures == [
        f"{DATASET}.parquet: the {DATASET} loader failed, so its 1 carried cells "
        f"could not be checked: OSError: {DATASET} is not downloaded"
    ]
    assert report.retired_tags == [] and report.misaligned_carried == {}


def test_a_fresh_cell_off_the_loaders_grid_is_not_staged(tmp_path, corpora):
    """A fresh cell with one record too many fails and keeps the published rows."""
    tag = f"gpt2_surprisal_{DATASET}"
    hub = FakeHub(
        tmp_path / "hub",
        [
            {
                "tag": tag,
                "file": f"{DATASET}.parquet",
                "dataset": DATASET,
                "model": "gpt2",
                "model_kind": "lm",
                "method": "surprisal",
                "predictor": "log_p_observed",
                "value_unit": "nats",
                "n_stimuli": 1,
                "n_rows": 2,
            }
        ],
        {
            f"{DATASET}.parquet": legacy_word_table(
                DATASET, {tag: {"log_p_observed": [-1.0, -2.0]}}
            )
        },
    )
    out = tmp_path / "out"
    shifted = word_payload()
    shifted["toy_1"].append(shifted["toy_1"][0])
    write_cell(out, tag, shifted, method="surprisal")
    stage = tmp_path / "stage"
    plan = _plan(tmp_path / "plan.tsv", tag)
    report = export(stage, [("w", out)], coverage=[plan], hub=hub, corpora=corpora)
    assert not report.ok and report.cells == 0
    assert report.appeared_during_export == [] and report.changed_during_export == []
    assert report.misaligned_cells == {
        tag: "unit_count on 1 stimuli (toy_1: 3 units stored, 2 in the loader)"
    }
    assert report.failures == [
        f"{tag}: indexes other units than the {DATASET} loader produces: unit_count on 1 stimuli (toy_1: 3 units stored, 2 in the loader)"
    ]
    # Nothing of the cell reaches the stage; the published table is carried as it is.
    assert not (stage / f"{DATASET}.parquet").exists()
    assert not (stage / layout.MANIFESTS_DIR / f"{tag}.json").exists()
    assert _index(stage, tag) and _index(stage, tag)[0]["wave"] is None
    assert report.files_deleted == [] and report.replaced_tags == []
    coverage = {r["tag"]: r for r in rows(stage, layout.COVERAGE_FILE)}
    assert coverage[tag]["status"] == "carried"
    assert (
        coverage[tag]["detail"]
        == "this run's cell indexes other units than the loader produces"
    )


def test_a_planned_cell_whose_measure_is_undefined_on_the_corpus_is_marked(
    tmp_path, corpora, monkeypatch
):
    monkeypatch.setattr(type(corpora), "word_spacing", lambda self, dataset: False)
    out = tmp_path / "wave" / "out"
    out.mkdir(parents=True)
    plan = tmp_path / "plan.tsv"
    plan.write_text(
        "\n".join(
            "\t".join(row)
            for row in [
                [
                    "missing",
                    "restricted_first_token_entropy",
                    DATASET,
                    "gpt2",
                    f"gpt2_restricted_first_token_entropy_{DATASET}.pkl",
                    "python -m replica.compute --method restricted_first_token_entropy",
                ],
                [
                    "missing",
                    "surprisal",
                    DATASET,
                    "gpt2",
                    f"gpt2_surprisal_{DATASET}.pkl",
                    "python -m replica.compute --method surprisal",
                ],
            ]
        )
        + "\n"
    )
    stage = tmp_path / "stage"
    report = export(stage, [("wave", out)], coverage=[plan], corpora=corpora)
    assert report.ok
    coverage = {r["tag"]: r for r in rows(stage, layout.COVERAGE_FILE)}
    undefined = coverage[f"gpt2_restricted_first_token_entropy_{DATASET}"]
    assert undefined["status"] == "undefined"
    assert undefined["detail"].startswith(
        f"{DATASET}: restricted_first_token_entropy is bound to the units of the words layer"
    )
    assert "its loader declares no word spacing" in undefined["detail"]
    # Word surprisal derives from characters without word spacing, so it is absent, not undefined.
    absent = coverage[f"gpt2_surprisal_{DATASET}"]
    assert absent["status"] == "absent" and absent["detail"].startswith("python -m")
    readme = (stage / layout.README_FILE).read_text()
    assert "`undefined`" in readme


def test_a_corpus_that_fails_to_load_leaves_its_planned_cells_absent(
    tmp_path, corpora, monkeypatch, caplog
):
    loads = []

    def failing(self, dataset):
        loads.append(dataset)
        raise FileNotFoundError("no data")

    monkeypatch.setattr(type(corpora), "stimuli", failing)
    out = tmp_path / "wave" / "out"
    out.mkdir(parents=True)
    plan = tmp_path / "plan.tsv"
    plan.write_text(
        "\n".join(
            "\t".join(
                [
                    "missing",
                    method,
                    DATASET,
                    "gpt2",
                    f"gpt2_{method}_{DATASET}.pkl",
                    f"python -m replica.compute --method {method}",
                ]
            )
            for method in ("restricted_first_token_entropy", "surprisal")
        )
        + "\n"
    )
    stage = tmp_path / "stage"
    report = export(stage, [("wave", out)], coverage=[plan], corpora=corpora)
    assert report.ok
    coverage = {r["tag"]: r for r in rows(stage, layout.COVERAGE_FILE)}
    assert {r["status"] for r in coverage.values()} == {"absent"}
    assert all(r["detail"].startswith("python -m") for r in coverage.values())
    # The failure is kept, so the second planned row triggers no second load.
    assert loads == [DATASET]
    assert "not decided" in caplog.text


def test_a_fresh_cell_under_a_renamed_method_stages_under_the_current_name(
    tmp_path, corpora, monkeypatch
):
    from replica.predictor_store import verify as verify_module
    from replica.predictor_store.verify import verify

    monkeypatch.setattr(verify_module, "RegistryCorpora", type(corpora))
    out = tmp_path / "wave" / "out"
    write_cell_under_old_name(
        out, f"gpt2_next_word_entropy_{DATASET}", "next_word_entropy"
    )
    stage = tmp_path / "stage"
    report = export(stage, [("wave", out)], corpora=corpora)
    assert report.ok and report.cells == 1
    new = f"gpt2_restricted_first_token_entropy_{DATASET}"
    assert report.exported_tags == [new]
    (index,) = _index(stage, new)
    assert index["method"] == "restricted_first_token_entropy"
    assert index["predictor"] == "restricted_first_token_entropy"
    table = rows(stage, f"{DATASET}.parquet")
    assert {r["tag"] for r in table} == {new}
    assert {r["predictor"] for r in table} == {"restricted_first_token_entropy"}
    manifest = json.loads((stage / layout.MANIFESTS_DIR / f"{new}.json").read_text())
    assert manifest["identity"]["method"] == "restricted_first_token_entropy"
    assert not (
        stage / layout.MANIFESTS_DIR / f"gpt2_next_word_entropy_{DATASET}.json"
    ).exists()
    assert report.appeared_during_export == []
    assert verify(stage, [("wave", out)]).ok


def _hub_under_old_name(root, old: str, texts: list[str] | None = None):
    """A store holding one cell published before the rename, with its manifest."""
    values = [3.0] * (len(texts) if texts else 2)
    table = legacy_word_table(DATASET, {old: {"next_word_entropy": values}})
    if texts is not None:
        table = table.set_column(
            table.schema.get_field_index("text_unit"), "text_unit", pa.array(texts)
        )
    hub = FakeHub(
        root,
        [
            {
                "tag": old,
                "file": f"{DATASET}.parquet",
                "dataset": DATASET,
                "model": "gpt2",
                "model_kind": "lm",
                "method": "next_word_entropy",
                "predictor": "next_word_entropy",
                "value_unit": "bits",
                "n_stimuli": 1,
                "n_rows": len(values),
            }
        ],
        {f"{DATASET}.parquet": table},
    )
    (root / layout.MANIFESTS_DIR).mkdir()
    (root / layout.MANIFESTS_DIR / f"{old}.json").write_text(
        json.dumps({"identity": {"method": "next_word_entropy"}, "status": "completed"})
    )
    return hub


def test_a_carried_cell_under_a_renamed_method_is_restaged(tmp_path, corpora):
    old = f"gpt2_next_word_entropy_{DATASET}"
    new = f"gpt2_restricted_first_token_entropy_{DATASET}"
    hub = _hub_under_old_name(tmp_path / "hub", old)
    out = tmp_path / "out"
    write_cell(out, f"gpt2_surprisal_{DATASET}", word_payload(), method="surprisal")
    stage = tmp_path / "stage"
    report = export(stage, [("w", out)], hub=hub, corpora=corpora)
    assert report.ok
    (index,) = _index(stage, new)
    assert index["method"] == "restricted_first_token_entropy"
    assert index["predictor"] == "restricted_first_token_entropy"
    assert not _index(stage, old)
    table = rows(stage, f"{DATASET}.parquet")
    assert {r["tag"] for r in table} == {new, f"gpt2_surprisal_{DATASET}"}
    manifest = json.loads((stage / layout.MANIFESTS_DIR / f"{new}.json").read_text())
    assert manifest["identity"]["method"] == "restricted_first_token_entropy"
    assert f"{layout.MANIFESTS_DIR}/{old}.json" in report.files_deleted
    assert f"{layout.MANIFESTS_DIR}/{new}.json" in report.files_written


def test_a_replaced_cell_under_a_renamed_method_deletes_the_old_manifest(
    tmp_path, corpora
):
    old = f"gpt2_next_word_entropy_{DATASET}"
    new = f"gpt2_restricted_first_token_entropy_{DATASET}"
    hub = _hub_under_old_name(tmp_path / "hub", old)
    out = tmp_path / "out"
    write_cell(
        out,
        new,
        word_payload("restricted_first_token_entropy"),
        method="restricted_first_token_entropy",
    )
    stage = tmp_path / "stage"
    report = export(stage, [("w", out)], hub=hub, corpora=corpora)
    assert report.ok and report.replaced_tags == [new]
    assert f"{layout.MANIFESTS_DIR}/{old}.json" in report.files_deleted
    # The fresh cell's own manifest is staged, with its payload digest.
    manifest = json.loads((stage / layout.MANIFESTS_DIR / f"{new}.json").read_text())
    assert manifest["identity"]["method"] == "restricted_first_token_entropy"
    assert "payload_sha256" in manifest
    assert {r["tag"] for r in rows(stage, f"{DATASET}.parquet")} == {new}
    assert [r["tag"] for r in _index(stage)] == [new]


def test_a_retired_cell_under_a_renamed_method_deletes_the_old_manifest(
    tmp_path, corpora
):
    old = f"gpt2_next_word_entropy_{DATASET}"
    new = f"gpt2_restricted_first_token_entropy_{DATASET}"
    hub = _hub_under_old_name(tmp_path / "hub", old, texts=["ab"])
    out = tmp_path / "out"
    fresh = f"gpt2_surprisal_{DATASET}"
    write_cell(out, fresh, word_payload(), method="surprisal")
    stage = tmp_path / "stage"
    report = export(stage, [("w", out)], hub=hub, corpora=corpora, drop_misaligned=True)
    assert report.ok and report.retired_tags == [new]
    assert report.files_deleted == [f"{layout.MANIFESTS_DIR}/{old}.json"]
    assert not (stage / layout.MANIFESTS_DIR / f"{new}.json").exists()
    assert {r["tag"] for r in _index(stage)} == {fresh}


def test_a_verification_record_under_an_earlier_name_follows_the_cell(
    tmp_path, corpora
):
    old = f"gpt2_next_word_entropy_{DATASET}"
    new = f"gpt2_restricted_first_token_entropy_{DATASET}"
    out = tmp_path / "out"
    write_cell(
        out,
        new,
        word_payload("restricted_first_token_entropy"),
        method="restricted_first_token_entropy",
    )
    verification = tmp_path / "verification.json"
    verification.write_text(json.dumps([{"tag": old, "verdict": "pass", "flags": []}]))
    stage = tmp_path / "stage"
    report = export(
        stage, [("w", out)], verifications=[("w", verification)], corpora=corpora
    )
    assert report.ok and report.shard_verification_records == {}
    (index,) = _index(stage, new)
    assert (index["verification"], index["verification_file"]) == (
        "pass",
        "verification/w.json",
    )
    staged = json.loads((stage / "verification" / "w.json").read_text())
    assert staged == [{"tag": new, "verdict": "pass", "flags": []}]


def test_a_carried_index_row_without_a_method_keeps_it_null():
    tag = f"gpt2_surprisal_{DATASET}"
    row = export_module._carried_index_row({"tag": tag, "method": None}, [DATASET])
    assert row["tag"] == tag and row["method"] is None


def _hub_with_carried_word_cell(root, tag, method, predictor, unit_rule=None):
    """A published toy_1 word cell aligned to the loader, under *method* and *unit_rule*."""
    decoded = layout.decode_tag(tag, DATASET)
    table = legacy_word_table(DATASET, {tag: {predictor: [3.0, 4.0]}})
    hub = FakeHub(
        root,
        [
            {
                "tag": tag,
                "file": f"{DATASET}.parquet",
                "dataset": DATASET,
                "model": decoded.model,
                "model_kind": decoded.model_kind,
                "method": method,
                "predictor": predictor,
                "value_unit": "bits",
                "n_stimuli": 1,
                "n_rows": 2,
                "unit_rule": unit_rule,
            }
        ],
        {f"{DATASET}.parquet": table},
        keep=("unit_rule",) if unit_rule is not None else (),
    )
    (root / layout.MANIFESTS_DIR).mkdir()
    (root / layout.MANIFESTS_DIR / f"{tag}.json").write_text("{}")
    return hub


def test_a_carried_cell_the_contract_finds_undefined_is_reported_and_kept(
    tmp_path, corpora, monkeypatch
):
    """A published unit-bound cell on an unspaced corpus under the default rule is carried with the finding recorded."""
    monkeypatch.setattr(type(corpora), "word_spacing", lambda self, dataset: False)
    tag = f"gpt2_restricted_first_token_entropy_{DATASET}"
    hub = _hub_with_carried_word_cell(
        tmp_path / "hub",
        tag,
        "restricted_first_token_entropy",
        "restricted_first_token_entropy",
        unit_rule="whitespace_continuation",
    )
    out = tmp_path / "out"
    fresh = f"gpt2_charbeam_surprisal_K5_{DATASET}"
    write_cell(out, fresh, word_payload(), method="char_beam")
    stage = tmp_path / "stage"
    report = export(stage, [("w", out)], hub=hub, corpora=corpora)
    assert report.ok and report.retired_tags == []
    assert "declares no word spacing" in report.undefined_carried[tag]
    assert {r["tag"] for r in _index(stage)} == {fresh, tag}
    assert "carried although the unit contract finds them undefined" in report.summary()


@pytest.mark.parametrize(
    ("model", "method", "unit_rule"),
    [
        ("gpt2", "restricted_first_token_entropy", "corpus_segmentation"),
        ("wordfreq-zh", "unigram_surprisal", None),
        ("wordfreq-zh", "unigram_surprisal", "whitespace_continuation"),
        ("wordfreq-zh", "unigram_surprisal", "corpus_segment_lookup"),
    ],
)
def test_unspaced_lookup_and_custom_rules_keep_carried_cells_and_provenance(
    tmp_path, corpora, monkeypatch, model, method, unit_rule
):
    monkeypatch.setattr(type(corpora), "word_spacing", lambda self, dataset: False)
    tag = f"{model}_{method}_{DATASET}"
    hub = _hub_with_carried_word_cell(
        tmp_path / "hub",
        tag,
        method,
        method,
        unit_rule=unit_rule,
    )
    original = hub.root / layout.MANIFESTS_DIR / f"{tag}.json"
    original.write_text(
        json.dumps({"identity": {"method": method, "unit_rule": unit_rule}})
    )
    before = original.read_bytes()
    out = tmp_path / "out"
    write_cell(
        out, f"gpt2_charbeam_surprisal_K5_{DATASET}", word_payload(), method="char_beam"
    )
    stage = tmp_path / "stage"
    report = export(stage, [("w", out)], hub=hub, corpora=corpora)
    assert report.ok and report.undefined_carried == {} and report.retired_tags == []
    assert _index(stage, tag)[0]["unit_rule"] == unit_rule
    assert original.read_bytes() == before
    manifest = f"{layout.MANIFESTS_DIR}/{tag}.json"
    assert manifest not in report.files_written + report.files_deleted
    assert not (stage / manifest).exists()


def test_a_retirement_list_retires_the_listed_carried_cells_with_their_reasons(
    tmp_path, corpora, monkeypatch
):
    """The listed tag leaves the store with the caller's reason recorded; the README names it."""
    monkeypatch.setattr(type(corpora), "word_spacing", lambda self, dataset: False)
    tag = f"gpt2_surprisal_{DATASET}"
    hub = _hub_with_carried_word_cell(tmp_path / "hub", tag, "surprisal", "surprisal")
    out = tmp_path / "out"
    fresh = f"gpt2_charbeam_surprisal_K5_{DATASET}"
    write_cell(out, fresh, word_payload(), method="char_beam")
    stage = tmp_path / "stage"
    report = export(
        stage,
        [("w", out)],
        hub=hub,
        corpora=corpora,
        retire={tag: "word surprisal on toy comes from the char_beam table"},
    )
    assert report.ok and report.retired_tags == [tag]
    assert (
        report.retired_by_request[tag]
        == f"{DATASET}: word surprisal on toy comes from the char_beam table"
    )
    assert tag not in report.undefined_carried
    assert {r["tag"] for r in _index(stage)} == {fresh}
    publish = json.loads((stage / layout.PUBLISH_FILE).read_text())
    assert publish["retired_tags"] == [tag]
    assert publish["deleted"] == [f"{layout.MANIFESTS_DIR}/{tag}.json"]
    readme = (stage / "README.md").read_text()
    assert "### Cells retired from the store" in readme
    assert f"| `{DATASET}` | 1 |" in readme and "—" not in readme


def test_a_retirement_list_refuses_an_unknown_or_fresh_tag(tmp_path, corpora):
    """A listed tag the store does not hold, or that this run stages, fails the export."""
    tag = f"gpt2_restricted_first_token_entropy_{DATASET}"
    hub = _hub_with_carried_word_cell(
        tmp_path / "hub",
        tag,
        "restricted_first_token_entropy",
        "restricted_first_token_entropy",
    )
    out = tmp_path / "out"
    fresh = f"gpt2_surprisal_{DATASET}"
    write_cell(out, fresh, word_payload(), method="surprisal")
    report = export(
        tmp_path / "stage",
        [("w", out)],
        hub=hub,
        corpora=corpora,
        retire={
            fresh: "listed by mistake",
            f"gpt2_nothing_{DATASET}": "never published",
        },
    )
    assert not report.ok and report.retired_tags == []
    assert any("stages it fresh" in f for f in report.failures)
    assert any("not published" in f for f in report.failures)


def test_read_retirements_parses_tags_and_reasons(tmp_path):
    from replica.predictor_store.export import read_retirements

    path = tmp_path / "retire.tsv"
    path.write_text(
        "# comment\n\ngpt2_a_toy\tcontract refuses it\ngpt2_b_toy\tno producer\n"
    )
    assert read_retirements(path) == {
        "gpt2_a_toy": "contract refuses it",
        "gpt2_b_toy": "no producer",
    }
    path.write_text("gpt2_a_toy\n")
    with pytest.raises(ValueError, match="tag<TAB>reason"):
        read_retirements(path)


def test_a_listed_misaligned_cell_retires_without_the_drop_flag(tmp_path, corpora):
    """Retiring one bad cell by name leaves the loader check for the others intact."""
    stale = f"gpt2_restricted_first_token_entropy_{DATASET}"
    hub = _hub_with_stale_cell(tmp_path / "hub", stale, ["ab"])
    out = tmp_path / "out"
    fresh = f"gpt2_surprisal_{DATASET}"
    write_cell(out, fresh, word_payload(), method="surprisal")
    stage = tmp_path / "stage"
    report = export(
        stage,
        [("w", out)],
        hub=hub,
        corpora=corpora,
        retire={stale: "computed against an earlier corpus version"},
    )
    assert report.ok and report.retired_tags == [stale]
    assert report.misaligned_carried == {}
    assert {r["tag"] for r in _index(stage)} == {fresh}


def test_a_named_rule_does_not_lift_an_unsupported_grain(tmp_path, corpora):
    """A custom rule answers the word-spacing condition only; a grain the method lacks stays undefined."""
    from replica.predictor_store.layout import INDEX_SCHEMA

    tag = f"gpt2_restricted_first_token_entropy_chars_{DATASET}"
    hub = _grain_hub(tmp_path / "hub", tag, "characters", list(TEXTS["toy_1"]))
    rows = [
        {**row, "method": "restricted_first_token_entropy", "unit_rule": "custom"}
        for row in hub.index().to_pylist()
    ]
    names = [name for name in INDEX_SCHEMA.names if name in rows[0]]
    pq.write_table(
        pa.Table.from_pylist(
            [{name: r.get(name) for name in names} for r in rows],
            schema=pa.schema([INDEX_SCHEMA.field(n) for n in names]),
        ),
        hub.root / "tags.parquet",
    )
    out = tmp_path / "out"
    write_cell(out, f"gpt2_surprisal_{DATASET}", word_payload(), method="surprisal")
    report = export(tmp_path / "stage", [("w", out)], hub=hub, corpora=corpora)
    assert tag in report.undefined_carried
    assert "not supported for method" in report.undefined_carried[tag]


@pytest.mark.parametrize(
    ("metadata", "value"),
    [
        ("token_index", 0.9),
        ("span", (0.9, 2.9)),
        ("token_id", 11.9),
        ("token_id", -1),
        ("token_id", 2**63),
        ("token_index", 2**31),
        ("token_id", True),
        ("token_index", "0"),
        ("token", None),
        ("span", (0, 99)),
        ("span", (1, 2)),
    ],
)
def test_token_identity_is_validated_before_arrow_conversion(
    tmp_path, corpora, metadata, value
):
    payload = token_payload()
    payload["toy_1"][0].metadata[metadata] = value
    tag = f"gpt2_surprisal_tokens_{DATASET}"
    out, stage = tmp_path / "out", tmp_path / "stage"
    write_cell(out, tag, payload, method="surprisal", unit="tokens")
    report = export(stage, [("w", out)], corpora=corpora)
    assert not report.ok and report.cells == 0, report.summary()
    assert not (stage / layout.MANIFESTS_DIR / f"{tag}.json").exists()


@pytest.mark.parametrize("change", ["unknown", "partial_stimulus", "partial_corpus"])
def test_token_coverage_is_checked_within_present_known_stimuli(
    tmp_path, corpora, change
):
    payload = token_payload()
    if change == "unknown":
        payload["ghost"] = payload.pop("toy_1")
    elif change == "partial_stimulus":
        payload["toy_1"].pop()
    else:
        del payload["toy_2"]
    out = tmp_path / "out"
    write_cell(
        out,
        f"gpt2_surprisal_tokens_{DATASET}",
        payload,
        method="surprisal",
        unit="tokens",
    )
    report = export(tmp_path / "stage", [("w", out)], corpora=corpora)
    assert report.ok == (change == "partial_corpus"), report.summary()


def test_supported_token_identity_types_are_preserved(tmp_path, corpora):
    payload = token_payload()
    record = payload["toy_1"][0]
    record.metadata.update(
        token_index=np.int32(0),
        token_id=np.int64(11),
        span=(np.int32(0), np.int64(2)),
        token="",
    )
    out, stage = tmp_path / "out", tmp_path / "stage"
    write_cell(
        out,
        f"gpt2_surprisal_tokens_{DATASET}",
        payload,
        method="surprisal",
        unit="tokens",
    )
    report = export(stage, [("w", out)], corpora=corpora)
    assert report.ok, report.summary()
    first = pq.read_table(stage / layout.TOKENS_DIR / f"{DATASET}.parquet").to_pylist()[
        0
    ]
    assert (first["token_index"], first["token_id"], first["token"]) == (0, 11, "")


def test_index_keeps_authenticated_parameter_precision(tmp_path, corpora):
    from replica.artifacts import manifest_path
    from replica.compute.cache import method_cache_tag
    from replica.predictor_store.verify import verify

    alpha = 0.123456789
    tag = f"gpt2_{method_cache_tag('token_renyi_entropy', alpha=alpha)}_{DATASET}"
    out, stage = tmp_path / "out", tmp_path / "stage"
    path = write_cell(
        out,
        tag,
        word_payload(f"token_renyi_entropy_alpha{alpha}"),
        method="token_renyi_entropy",
    )
    sidecar = manifest_path(path)
    manifest = json.loads(sidecar.read_text())
    manifest["identity"]["options"]["alpha"] = alpha
    sidecar.write_text(json.dumps(manifest))
    before = sidecar.read_bytes()
    assert export(stage, [("w", out)], corpora=corpora).ok
    assert _index(stage)[0]["alpha"] == alpha
    assert sidecar.read_bytes() == before
    report = verify(stage, [("w", out)], names=[DATASET], corpora=corpora)
    assert report.ok, report.summary()
