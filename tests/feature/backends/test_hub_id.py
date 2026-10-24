"""Hub cache snapshot paths: names report the Hub id, loading uses the path."""

from types import SimpleNamespace

import pytest
import transformers

from replica.feature import HuggingFaceLM
from replica.feature.backends.hub import hub_id
from replica.feature.embedding.huggingface import HuggingFaceEmbedding

SNAPSHOT = "/hub/models--EleutherAI--pythia-70m/snapshots/0123abcd"


def test_snapshot_path_maps_back_to_the_hub_id():
    path = "~/.cache/huggingface/hub/models--facebook--opt-125m/snapshots/27dcfa74/config.json"
    assert hub_id(path) == "facebook/opt-125m"


def test_hub_ids_and_plain_names_pass_through():
    assert hub_id("facebook/opt-125m") == "facebook/opt-125m"
    assert hub_id("gpt2") == "gpt2"
    assert hub_id("/models/local-finetune") == "/models/local-finetune"


class _Model:
    def to(self, device):
        return self

    def eval(self):
        return self


@pytest.fixture
def loads(monkeypatch):
    """Record the target of every ``from_pretrained`` call."""
    calls: list[tuple[str, str]] = []

    def recorder(kind, result):
        def from_pretrained(name, *args, **kwargs):
            calls.append((kind, name))
            return result

        return from_pretrained

    tokenizer = SimpleNamespace()
    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", recorder("tokenizer", tokenizer)
    )
    for auto in (transformers.AutoModelForCausalLM, transformers.AutoModel):
        monkeypatch.setattr(auto, "from_pretrained", recorder("model", _Model()))
    return calls


@pytest.mark.resource_free("from_pretrained is replaced by a recorder")
def test_causal_lm_loads_the_snapshot_it_was_given(loads):
    lm = HuggingFaceLM(SNAPSHOT, device="cpu", provide_word_dist=False)
    lm._ensure_loaded()
    assert loads == [("tokenizer", SNAPSHOT), ("model", SNAPSHOT)]
    assert lm.name == "EleutherAI/pythia-70m"


@pytest.mark.resource_free("from_pretrained is replaced by a recorder")
def test_embedding_loads_the_snapshot_it_was_given(loads):
    embedding = HuggingFaceEmbedding(SNAPSHOT)
    embedding._ensure_loaded()
    assert loads == [("tokenizer", SNAPSHOT), ("model", SNAPSHOT)]
    assert embedding.name == "EleutherAI/pythia-70m"
