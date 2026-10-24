"""Precision and placement options reach the loaders and are validated."""

from types import SimpleNamespace

import pytest
import torch

from replica.feature.backends import device as device_module
from replica.feature.backends.huggingface import HuggingFaceLM


class _FakeModel:
    def __init__(self):
        self.device = torch.device("cpu")
        self.moved_to = None
        self.config = SimpleNamespace(n_positions=1024)

    def to(self, device):
        self.moved_to = device
        return self

    def eval(self):
        return self


def _capture_loading(monkeypatch):
    calls = {}
    model = _FakeModel()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    def fake_model(checkpoint, **kwargs):
        calls.update(kwargs)
        return model

    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", fake_model)
    monkeypatch.setattr(
        AutoTokenizer,
        "from_pretrained",
        lambda *a, **k: SimpleNamespace(bos_token_id=0),
    )
    return calls, model


@pytest.mark.resource_free(
    "from_pretrained is replaced by a fake that records its arguments"
)
def test_default_loading_keeps_single_precision_on_one_device(monkeypatch):
    calls, model = _capture_loading(monkeypatch)
    lm = HuggingFaceLM("gpt2", provide_word_dist=False)
    lm._ensure_loaded()
    assert "torch_dtype" not in calls and "device_map" not in calls
    assert model.moved_to == lm.device


@pytest.mark.resource_free(
    "from_pretrained is replaced by a fake that records its arguments"
)
def test_dtype_and_device_map_reach_the_loader(monkeypatch):
    calls, model = _capture_loading(monkeypatch)
    model.device = torch.device("cpu")
    lm = HuggingFaceLM(
        "gpt2", provide_word_dist=False, dtype="bfloat16", device_map="auto"
    )
    lm._ensure_loaded()
    assert calls["torch_dtype"] is torch.bfloat16 and calls["device_map"] == "auto"
    # Sharded weights are left in place; inputs follow the model's device.
    assert model.moved_to is None and lm.device == model.device


@pytest.mark.resource_free("the constructor refuses the option before any loading")
@pytest.mark.parametrize("kwargs", [{"dtype": "half"}, {"device_map": "balanced"}])
def test_unknown_precision_or_placement_is_refused(kwargs):
    with pytest.raises(ValueError):
        HuggingFaceLM("gpt2", provide_word_dist=False, **kwargs)


def test_loading_options_name_torch_dtypes():
    assert device_module.loading_options("float16", None) == {
        "torch_dtype": torch.float16
    }
    assert device_module.loading_options(None, "auto") == {"device_map": "auto"}
    assert device_module.loading_options(None, None) == {}
