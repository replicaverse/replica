"""WordsProbabilityLM builds its upstream model once; upstream's entry point
rebuilds it per call, which exhausts a GPU over a corpus."""

import pytest

# The tests below stub the upstream factory and never load weights, but the
# marker guard keys on the constructor reference, which is the discipline.
pytestmark = [pytest.mark.needs_model]

wordsprobability = pytest.importorskip("wordsprobability")

from replica.feature.surprisal.wordsprob import WordsProbabilityLM


def _counting_factory(monkeypatch):
    calls = []

    def fake_get_model(model_id):
        calls.append(model_id)
        return object()

    monkeypatch.setattr("wordsprobability.models.get_model", fake_get_model)
    return calls


def test_ensure_model_constructs_once(monkeypatch):
    lm = WordsProbabilityLM("gpt2-small")
    calls = _counting_factory(monkeypatch)
    first = lm._ensure_model()
    second = lm._ensure_model()
    assert first is second
    assert calls == ["gpt2-small"]


def test_compute_per_word_never_reconstructs_the_model(monkeypatch):
    lm = WordsProbabilityLM("gpt2-small")
    used = []

    class _Stub:
        def get_predictions(self, text):
            used.append(text)
            raise _Sentinel

    class _Sentinel(Exception):
        pass

    lm._wp_model = _Stub()

    def forbidden(*a, **k):
        raise AssertionError("_compute_per_word must reuse the cached model")

    monkeypatch.setattr("wordsprobability.models.get_model", forbidden)
    with pytest.raises(_Sentinel):
        lm._compute_per_word("The cat sat.")
    assert used == ["The cat sat."]


@pytest.mark.parametrize(
    "prefix,source", [(None, None), (5, None), (5, "/pinned/model")]
)
def test_explicit_source_and_prefix_reach_the_upstream_backend(
    monkeypatch, prefix, source
):
    from types import SimpleNamespace

    import torch
    from wordsprobability.models import MODELS

    received, loaded = [], []

    class StopAtForward(Exception):
        pass

    class Tokenizer:
        bos_token_id, eos_token_id, vocab_size = 0, 5, 6

        def get_vocab(self):
            return {"<s>": 0, "x": 1, "Ġx": 2, "!": 3, "y": 4, "<eos>": 5}

        def __call__(self, text, **kwargs):
            return {"input_ids": [1], "offset_mapping": [(0, len(text))]}

    class Backend:
        device = torch.device("cpu")
        lm_head = SimpleNamespace(out_features=6)

        def __call__(self, ids, **kwargs):
            received.append(ids.tolist())
            raise StopAtForward

    upstream = MODELS["mgpt"]
    original_source = upstream.model_name

    def load(model):
        loaded.append(model.model_name)
        return Backend(), Tokenizer()

    monkeypatch.setattr(upstream, "_initialise_model", load)
    adapter = WordsProbabilityLM("mgpt", model_source=source, start_token_id=prefix)
    with pytest.raises(StopAtForward):
        adapter._compute_per_word("x")
    assert loaded == [source or original_source]
    assert received == [[[0 if prefix is None else prefix, 1, 5]]]
    assert upstream.model_name == original_source
    assert MODELS["mgpt"] is upstream


def test_compute_binds_the_snapshot_and_only_overrides_the_corrected_prefix(tmp_path):
    from replica.compute.methods import ComputeContext, build_model

    snapshot = tmp_path / "models--ai-forever--mGPT" / "snapshots" / ("a" * 40)
    snapshot.mkdir(parents=True)
    context = ComputeContext(
        str(snapshot),
        5,
        4,
        2,
        1.0,
        0,
        True,
        "en",
        None,
        model_identity={"start": {"ids": [5]}},
    )
    corrected = build_model("wordsprobability_surprisal", context)
    upstream = build_model("wordsprobability_surprisal_buggy", context)
    assert corrected.model_source == upstream.model_source == str(snapshot)
    assert corrected.start_token_id == 5
    assert upstream.start_token_id is None
