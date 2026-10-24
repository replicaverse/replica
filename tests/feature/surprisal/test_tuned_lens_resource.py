"""Snapshot lookup must retain independently pinned probe resources."""

import sys
from types import SimpleNamespace

import pytest

from replica.feature.surprisal.logit_lens import LogitLensLM
from replica.feature.surprisal.tuned_lens import TunedLensLM


@pytest.mark.resource_free("Stubbed model and lens; no checkpoint loads")
@pytest.mark.parametrize("resource", [None, "EleutherAI/pythia-1b-deduped-v0"])
def test_snapshot_lookup_and_explicit_probe_pin(monkeypatch, resource):
    snapshot = "~/.cache/huggingface/hub/models--facebook--opt-125m/snapshots/model-pin"
    calls = []

    class Lens:
        @staticmethod
        def from_model_and_pretrained(model, **kwargs):
            calls.append(kwargs)
            return Lens()

        def to(self, device):
            return self

    monkeypatch.setitem(sys.modules, "tuned_lens", SimpleNamespace(TunedLens=Lens))
    monkeypatch.setattr(
        LogitLensLM,
        "_ensure_loaded",
        lambda self: setattr(
            self,
            "_model",
            SimpleNamespace(config=SimpleNamespace(name_or_path=snapshot)),
        ),
    )
    model = TunedLensLM(
        snapshot, device="cpu", lens_resource_id=resource, lens_revision="probe-pin"
    )
    model._ensure_loaded()
    model._ensure_loaded()
    assert calls == [
        dict(
            lens_resource_id=resource or "facebook/opt-125m",
            revision="probe-pin",
            repo_id="AlignmentResearch/tuned-lens",
            repo_type="space",
        )
    ]
