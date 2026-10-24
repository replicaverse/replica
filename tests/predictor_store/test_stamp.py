"""Binding a pickle another pipeline produced to a manifest."""

import json
import pickle

import pytest

from replica.artifacts import manifest_path, validate_artifact
from replica.predictor_store import cli
from replica.predictor_store.accept import accept
from replica.predictor_store.export import export
from replica.predictor_store.stamp import stamp
from replica.utils.naming import safe_lm
from tests.predictor_store._cells import DATASET, TEXTS, rows, word_payload

POOLED = f"ai-forever_mGPT_mmc_pps_unigram_surprisal_N40000000_k8192_s900000_{DATASET}"
REGISTERED = f"gpt2_surprisal_{DATASET}"


def _pickle(out, tag, payload):
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{tag}.pkl"
    with path.open("wb") as stream:
        pickle.dump(payload, stream)
    return path


def _table(tmp_path, text="{}"):
    table = tmp_path / "table3.json"
    table.write_text(text)
    return table


def test_a_complete_payload_is_stamped_completed(tmp_path, corpora):
    path = _pickle(tmp_path / "out", POOLED, word_payload("unigram_surprisal", 2.0))
    table = _table(tmp_path)
    bound = stamp(
        path,
        model_id="ai-forever/mGPT",
        inputs=[table],
        reason="materialised from the pooled table",
        corpora=corpora,
    )
    assert bound == path
    identity = validate_artifact(path)
    assert identity["method"] == "mmc_unigram_surprisal_pool"
    assert identity["storage_class"] is None and identity["layer"] is None
    assert identity["options"]["n_samples"] == 40000000
    assert identity["options"]["k_eval"] == 8192
    assert identity["options"]["unit"] == "words"
    assert identity["model"] == {"source": "ai-forever/mGPT"}
    assert identity["n_stimuli"] == len(TEXTS)
    assert len(identity["corpus_sha256"]) == 64
    manifest = json.loads(manifest_path(path).read_text())
    assert manifest["status"] == "completed"
    assert manifest["external"]["reason"] == "materialised from the pooled table"
    assert manifest["external"]["inputs"][0]["path"] == "table3.json"
    assert len(manifest["external"]["inputs"][0]["sha256"]) == 64
    assert "audit" not in manifest
    assert manifest["environment"]["computed_at"]


def test_a_registered_method_takes_its_storage(tmp_path, corpora):
    path = _pickle(tmp_path / "out", REGISTERED, word_payload())
    stamp(path, model_id="gpt2", inputs=[], reason="scored elsewhere", corpora=corpora)
    identity = validate_artifact(path)
    assert identity["method"] == "surprisal"
    assert identity["storage_class"] == "positional" and identity["layer"] == 1


def test_a_stamped_cell_exports(tmp_path, corpora):
    out = tmp_path / "wave" / "out"
    path = _pickle(out, POOLED, word_payload("unigram_surprisal", 2.0))
    stamp(
        path,
        model_id="ai-forever/mGPT",
        inputs=[_table(tmp_path)],
        reason="materialised",
        corpora=corpora,
    )
    stage = tmp_path / "stage"
    report = export(stage, [("wave", out)], corpora=corpora)
    assert report.ok and report.exported_tags == [POOLED]
    index = [r for r in rows(stage, "tags.parquet") if r["tag"] == POOLED]
    assert index and index[0]["storage_class"] is None
    assert {r["tag"] for r in rows(stage, f"{DATASET}.parquet")} == {POOLED}


def test_a_payload_short_of_the_corpus_is_quarantined(tmp_path, corpora):
    payload = word_payload("unigram_surprisal", 2.0)
    del payload[next(iter(payload))]
    path = _pickle(tmp_path / "out", POOLED, payload)
    bound = stamp(
        path,
        model_id="ai-forever/mGPT",
        inputs=[],
        reason="materialised",
        corpora=corpora,
    )
    assert bound.name == f"{POOLED}.suspect.pkl" and not path.exists()
    manifest = json.loads(manifest_path(bound).read_text())
    assert manifest["status"] == "quarantined"
    assert manifest["audit"]["missing_stimuli"] == 1
    assert manifest["external"]["reason"] == "materialised"
    with pytest.raises(ValueError, match="recompute the cell"):
        accept(bound, "the missing stimulus is fine")


@pytest.mark.parametrize("occupied", ["payload", "manifest"])
def test_quarantine_preserves_an_occupied_destination(tmp_path, corpora, occupied):
    payload = word_payload()
    del payload[next(iter(payload))]
    path = _pickle(tmp_path, REGISTERED, payload)
    target = path.with_suffix(".suspect.pkl")
    existing = target if occupied == "payload" else manifest_path(target)
    existing.write_bytes(b"retained artifact")
    before = {item: item.read_bytes() for item in tmp_path.iterdir()}
    with pytest.raises(FileExistsError, match="quarantine destination is occupied"):
        stamp(path, model_id="gpt2", inputs=[], reason="x", corpora=corpora)
    assert {item: item.read_bytes() for item in tmp_path.iterdir()} == before


@pytest.mark.parametrize(
    ("make", "match"),
    [
        (
            lambda tmp, c: _pickle(
                tmp / "out", "gpt2_surprisal_nowhere", word_payload()
            ),
            "no registry dataset",
        ),
        (
            lambda tmp, c: _pickle(tmp / "out", REGISTERED, [1, 2]),
            "not a stimulus mapping",
        ),
    ],
)
def test_refusals(tmp_path, corpora, make, match):
    path = make(tmp_path, corpora)
    with pytest.raises(ValueError, match=match):
        stamp(path, model_id="gpt2", inputs=[], reason="x", corpora=corpora)


def test_an_empty_reason_a_missing_input_and_a_stamped_cell_are_refused(
    tmp_path, corpora
):
    path = _pickle(tmp_path / "out", REGISTERED, word_payload())
    with pytest.raises(ValueError, match="needs a reason"):
        stamp(path, model_id="gpt2", inputs=[], reason="  ", corpora=corpora)
    with pytest.raises(ValueError, match="input not found"):
        stamp(
            path,
            model_id="gpt2",
            inputs=[tmp_path / "absent.json"],
            reason="x",
            corpora=corpora,
        )
    stamp(path, model_id="gpt2", inputs=[], reason="x", corpora=corpora)
    with pytest.raises(ValueError, match="carries a manifest"):
        stamp(path, model_id="gpt2", inputs=[], reason="x", corpora=corpora)


def test_a_stimulus_the_loader_lacks_is_refused(tmp_path, corpora):
    payload = word_payload()
    payload["toy_9"] = payload[next(iter(payload))]
    path = _pickle(tmp_path / "out", REGISTERED, payload)
    with pytest.raises(ValueError, match="does not produce"):
        stamp(path, model_id="gpt2", inputs=[], reason="x", corpora=corpora)


def test_the_command_reports_the_state_and_refusals(
    tmp_path, corpora, monkeypatch, capsys
):
    from replica.predictor_store import stamp as stamp_module

    monkeypatch.setattr(stamp_module, "RegistryCorpora", lambda: corpora)
    path = _pickle(tmp_path / "out", REGISTERED, word_payload())
    assert (
        cli.main(
            ["stamp", str(path), "--model-id", "gpt2", "--reason", "scored elsewhere"]
        )
        == 0
    )
    assert capsys.readouterr().out.startswith("completed ")
    assert (
        cli.main(["stamp", str(path), "--model-id", "gpt2", "--reason", "again"]) == 1
    )
    assert "refused:" in capsys.readouterr().err


@pytest.mark.parametrize("model_id", ["org_team/model_name", "org/model:revision"])
def test_stamp_records_the_explicit_identity(tmp_path, corpora, model_id):
    tag = f"{safe_lm(model_id)}_surprisal_{DATASET}"
    path = _pickle(tmp_path, tag, word_payload())
    before = path.read_bytes()
    stamp(
        path, model_id=model_id, inputs=[], reason="scored elsewhere", corpora=corpora
    )
    identity = validate_artifact(path)
    assert identity["options"]["model_id"] == model_id
    assert identity["model"]["source"] == model_id
    assert path.read_bytes() == before


def test_stamp_requires_an_explicit_matching_identity_before_writing(tmp_path, corpora):
    path = _pickle(tmp_path, REGISTERED, word_payload())
    before = path.read_bytes()
    with pytest.raises(TypeError, match="model_id"):
        stamp(path, inputs=[], reason="x", corpora=corpora)
    with pytest.raises(SystemExit, match="2"):
        cli.main(["stamp", str(path), "--reason", "x"])
    for model_id in (None, "", "another/model"):
        with pytest.raises(ValueError, match="model_id"):
            stamp(path, model_id=model_id, inputs=[], reason="x", corpora=corpora)
    assert path.read_bytes() == before
    assert not manifest_path(path).exists()
    assert not path.with_suffix(".suspect.pkl").exists()
