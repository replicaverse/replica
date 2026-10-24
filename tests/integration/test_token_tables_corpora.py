"""Token tables on real corpora with the models the store uses.

The word tables derived from the token tables must equal the backend's own
word-level call value for value, and a script without word spacing must
refuse token-to-word aggregation while its word surprisal comes from the
character-level scorer. The lens and the exact similarity path derive their
word tables from per-token tables the same way. One model copy lives on the
device at a time: the driver's copy is released before the backend's is
built.
"""

import gc
import json
import pickle

import numpy as np
import pytest

from replica.artifacts import manifest_path
from replica.compute.runner import compute_for
from replica.feature import HuggingFaceLM, SimilarityAdjustedSurprisalEmbeddingLM
from replica.feature.backends.hub import immutable_revision
from replica.feature.core.annotation.alignment import (
    TokenBoundaryError,
    aggregate_nested,
    tokenize_spans,
)
from replica.feature.core.outputs.information import LN2
from replica.utils.digest import file_sha256

pytestmark = [pytest.mark.needs_data, pytest.mark.needs_model, pytest.mark.slow]

CORPORA = [("provo", "gpt2", 1024), ("meco_l1_w2_bp", "ai-forever/mGPT", 512)]
METHODS = [
    ("surprisal", "log_p_observed"),
    ("first_token_entropy", "token_entropy"),
    ("token_renyi_entropy", "token_renyi_entropy_alpha0.5"),
]
CELLS = [
    (dataset, model_id, max_positions, method, key)
    for dataset, model_id, max_positions in CORPORA
    for method, key in METHODS
]
CHINESE = ["meco_l1_w2_ch_s", "meco_l1_w2_ch_t"]


def _stimuli(dataset):
    from replica.dataset.catalog.registry import REGISTRY

    return REGISTRY[dataset].factory().stimuli()


def _release():
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _cell(dataset, model_id, method, out_dir, **overrides):
    outcome = compute_for(
        dataset,
        method=method,
        model_id=model_id,
        K=5,
        n_samples=64,
        max_tokens=10,
        alpha=0.5,
        seed=None,
        out_dir=out_dir,
        add_bos=True,
        force=False,
        **overrides,
    )
    _release()
    return outcome


def _load(path):
    with path.open("rb") as stream:
        return pickle.load(stream)


def _values(outputs, key):
    return np.array([o.values[key] for o in outputs], dtype=float)


@pytest.mark.parametrize("dataset,model_id,max_positions,method,key", CELLS)
def test_derived_word_table_equals_the_backend(
    dataset, model_id, max_positions, method, key, tmp_path
):
    outcome = _cell(dataset, model_id, method, tmp_path)
    assert outcome.status == "completed", outcome.error
    manifest = json.loads(manifest_path(outcome.path).read_text())
    derivation = manifest["derivation"]
    token_path = tmp_path / derivation["source"]
    assert derivation["source_sha256"] == file_sha256(token_path)
    assert derivation["unit"] == "tokens"
    token_manifest = json.loads(manifest_path(token_path).read_text())
    assert token_manifest["window"] == {
        "rule": "sliding",
        "max_positions": max_positions,
        "stride": max_positions // 2,
    }
    unit = "nats" if key == "log_p_observed" else "bits"
    assert token_manifest["identity"]["units"] == {key: unit}
    assert manifest["identity"]["units"] == (
        {"log_p_observed": "nats", "surprisal": "bits"}
        if key == "log_p_observed"
        else {key: "bits"}
    )
    assert token_manifest["identity"]["options"]["add_bos"] is True
    assert token_manifest["identity"]["options"]["context_scope"] == "stimulus"
    # The model identity names the checkpoint by the content digest of its
    # files; the Hub revision is recorded when the resolver could verify one
    # online and stays None for an offline load from the cache.
    model = token_manifest["identity"]["model"]
    assert model["source"] == model_id
    assert len(model["sha256"]) == 64 and model["files"]
    assert model["revision"] == immutable_revision(model["revision"])
    assert token_manifest["environment"]["model_device"]

    backend = HuggingFaceLM(
        model_id,
        add_bos=True,
        provide_word_dist=False,
        token_level_entropy=method == "first_token_entropy",
        token_renyi_alphas=[0.5] if method == "token_renyi_entropy" else None,
    )
    words = _load(outcome.path)
    stimuli = _stimuli(dataset)
    assert set(words) == {s.stimulus_id for s in stimuli}
    try:
        for stimulus in stimuli:
            expected = backend(stimulus, extract=[key])
            stored = words[stimulus.stimulus_id]
            np.testing.assert_allclose(
                _values(stored, key),
                _values(expected, key),
                rtol=1e-9,
                atol=0,
                equal_nan=True,
            )
            if key == "log_p_observed":
                np.testing.assert_allclose(
                    _values(stored, "surprisal"),
                    -_values(expected, key) / LN2,
                    rtol=1e-9,
                    atol=0,
                    equal_nan=True,
                )
    finally:
        del backend
        _release()


@pytest.mark.parametrize("dataset", CHINESE)
def test_chinese_sites_refuse_token_to_word_aggregation(dataset):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("ai-forever/mGPT")
    refused = []
    for stimulus in _stimuli(dataset):
        ids, spans = tokenize_spans(stimulus.text, tokenizer)
        try:
            aggregate_nested(stimulus, "words", spans, np.zeros(len(ids)), "sum")
        except TokenBoundaryError as error:
            assert "crosses a words boundary" in str(error)
            refused.append(stimulus.stimulus_id)
    assert refused, f"no token crossed a word boundary on {dataset}"


@pytest.mark.parametrize("dataset", CHINESE)
def test_chinese_word_surprisal_takes_the_character_route(dataset, tmp_path):
    outcome = _cell(
        dataset, "ai-forever/mGPT", "surprisal", tmp_path, stim_slice=(0, 1)
    )
    assert outcome.status == "completed", outcome.error
    assert outcome.method == "char_beam"
    manifest = json.loads(manifest_path(outcome.path).read_text())
    assert manifest["derivation"]["unit"] == "characters"
    assert manifest["derivation"]["rule"] == "sum"
    assert manifest["identity"]["units"] == {
        "log_p_observed": "nats",
        "surprisal": "bits",
    }
    words = _load(outcome.path)
    for stimulus in _stimuli(dataset)[:1]:
        values = _values(words[stimulus.stimulus_id], "log_p_observed")
        assert len(values) == len(stimulus.layers["words"])
        assert np.isfinite(values).all()
        assert (values < 0).all()
        np.testing.assert_allclose(
            _values(words[stimulus.stimulus_id], "surprisal"), -values / LN2
        )


EXACT_CELLS = [
    ("provo", "gpt2", "logit_lens_surprisal", "sum", None),
    (
        "provo",
        "gpt2",
        "similarity_surprisal_noncontextual",
        "sum",
        "similarity_unit_values",
    ),
]


def _direct_model(model_id, method):
    if method == "logit_lens_surprisal":
        from replica.feature.surprisal.logit_lens import LogitLensLM

        return LogitLensLM(model_id)
    return SimilarityAdjustedSurprisalEmbeddingLM(
        model_id, contextual=False, add_bos=True
    )


@pytest.mark.parametrize("dataset,model_id,method,rule,finisher", EXACT_CELLS)
def test_positional_exact_methods_derive_the_unit_call(
    dataset, model_id, method, rule, finisher, tmp_path
):
    outcome = _cell(dataset, model_id, method, tmp_path)
    assert outcome.status == "completed", outcome.error
    manifest = json.loads(manifest_path(outcome.path).read_text())
    assert manifest["identity"]["storage_class"] == "positional"
    assert manifest["identity"]["layer"] == 1
    derivation = manifest["derivation"]
    assert derivation["rule"] == rule and derivation["finisher"] == finisher
    token_path = tmp_path / derivation["source"]
    assert derivation["source_sha256"] == file_sha256(token_path)
    token_manifest = json.loads(manifest_path(token_path).read_text())
    assert token_manifest["window"]["rule"] == "sliding"
    assert token_manifest["identity"]["options"]["unit"] == "tokens"
    backend = _direct_model(model_id, method)
    words = _load(outcome.path)
    stimuli = _stimuli(dataset)
    assert set(words) == {s.stimulus_id for s in stimuli}
    try:
        for stimulus in stimuli:
            expected = backend(stimulus)
            stored = words[stimulus.stimulus_id]
            keys = [k for k in stored[0].values if np.ndim(stored[0].values[k]) == 0]
            assert keys
            for key in keys:
                np.testing.assert_allclose(
                    _values(stored, key),
                    _values(expected, key),
                    rtol=1e-9,
                    atol=0,
                    equal_nan=True,
                )
    finally:
        del backend
        _release()
