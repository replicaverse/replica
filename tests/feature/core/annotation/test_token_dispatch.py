"""Both token backends preserve context boundaries, offsets and output identity."""

import pytest

from replica.core import AnnotationLayer, Span, Stimulus
from replica.feature.backends.huggingface import HuggingFaceLM
from replica.feature.core.annotation.alignment import TEXT_OFFSET
from replica.feature.core.protocols import FeatureModelOutput
from replica.feature.surprisal.logit_lens import LogitLensLM


@pytest.fixture(params=[HuggingFaceLM, LogitLensLM])
def model(request):
    return request.param("unused", device="cpu")


@pytest.mark.parametrize("scope", ["sentence", "stimulus", "story"])
def test_context_dispatch_preserves_backend_outputs(model, scope, monkeypatch):
    stimulus = Stimulus(
        "aa bb cc",
        {
            "words": AnnotationLayer(
                "words", [Span(0, 2), Span(3, 5), Span(6, 8)], ["aa", "bb", "cc"]
            )
        },
        "sample",
    )
    stimulus.add_layer(
        AnnotationLayer("sentences", stimulus.layers["words"].spans, [1, 1, 2])
    )
    model.context_scope = scope
    keys = ["selected-key"]
    calls, produced = [], []

    def score(sub, extract):
        assert extract is keys
        calls.append(sub)
        outputs = [
            FeatureModelOutput(
                values={"selected-key": -3.5},
                metadata={"token_index": 0, "offset": sub.metadata.get(TEXT_OFFSET, 0)},
            )
            for _ in sub.layers["words"].spans
        ]
        produced.extend(outputs)
        return outputs

    monkeypatch.setattr(model, "_score_tokens", score)
    outputs = model.score_tokens(stimulus, keys)
    assert all(
        actual is expected for actual, expected in zip(outputs, produced, strict=True)
    )
    assert [o.metadata["token_index"] for o in outputs] == [0, 1, 2]
    assert [o.values for o in outputs] == [{"selected-key": -3.5}] * 3
    if scope == "sentence":
        assert [s.text for s in calls] == ["aa bb", "cc"]
        assert [o.metadata["offset"] for o in outputs] == [0, 0, 6]
    else:
        assert calls == [stimulus]
        assert calls[0] is stimulus


def test_missing_sentence_boundaries_fail_before_scoring(model, monkeypatch):
    model.context_scope = "sentence"
    monkeypatch.setattr(model, "_score_tokens", lambda *args: pytest.fail("scored"))
    with pytest.raises(ValueError, match="attach the corpus boundaries"):
        model.score_tokens(Stimulus("text", stimulus_id="sample"))


def test_empty_backend_output_stays_empty(model, monkeypatch):
    monkeypatch.setattr(model, "_score_tokens", lambda *args: [])
    assert model.score_tokens(Stimulus("", stimulus_id="sample")) == []
