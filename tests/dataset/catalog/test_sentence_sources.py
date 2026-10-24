"""Corpus sentence sources preserve recorded membership and reject missing sources."""

import pytest

from replica.context import SENTENCE_LAYER
from replica.dataset.catalog.sentences import attach_sentences


def test_unregistered_dataset_raises():
    with pytest.raises(ValueError, match="no sentence-boundary source"):
        attach_sentences("provo", [])


@pytest.mark.needs_data
def test_meco_sentnum_attaches_to_meco_en():
    from replica.dataset.catalog.registry import REGISTRY

    rda = REGISTRY["meco_en"].default_path() / "joint_l1_data_trimmed_version2.1.rda"
    if not rda.exists():
        pytest.skip("MECO L1 joint rda not available")
    stimuli = REGISTRY["meco_en"].factory().stimuli()

    attach_sentences("meco_en", stimuli)
    assert len(stimuli) == 12
    for stimulus in stimuli:
        ids = stimulus.layers[SENTENCE_LAYER].values
        assert len(ids) == len(stimulus.layers["words"].spans)
        # The source sentnum is monotone within a trial and starts at 1.
        assert ids[0] == 1
        assert all(b >= a for a, b in zip(ids, ids[1:], strict=False))
        assert max(ids) > 1


def test_meco_sentence_boundaries_skip_verified_nonword_areas(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import pandas as pd

    from replica.dataset.catalog.registry import REGISTRY
    from replica.dataset.catalog.sentences import _meco_l1_sentnum
    from replica.dataset.eye_tracking.meco import word_utils
    from replica.dataset.eye_tracking.meco.l1 import MECO_L1

    (tmp_path / "joint_l1_data_trimmed_version2.1.rda").touch()
    monkeypatch.setitem(
        REGISTRY, "meco_en", SimpleNamespace(default_path=lambda: tmp_path)
    )
    frame = pd.DataFrame(
        {
            "lang": ["en"] * 4,
            "itemid": [1, 1, 1, 2],
            "wordnum": [1, 2, 3, 1],
            "sentnum": [1] * 4,
            "word": ["one", "", "two", ""],
        }
    )
    monkeypatch.setattr(word_utils, "read_r_dataframe", lambda _: frame)
    dataset = MECO_L1("en")
    stimulus = dataset._make_stimulus(
        ["one", "two"],
        "s1",
        {
            "source_item_id": 1,
            "source_word_numbers": [1, 3],
            "source_nonword_numbers": [2],
        },
    )
    _meco_l1_sentnum("meco_en", [stimulus])
    assert stimulus.layers["sentences"].values == [1, 1]


def test_a_custom_corpus_keeps_the_sentence_layer_its_loader_attached():
    from replica.core import AnnotationLayer, Span, Stimulus

    def stimulus(sid, with_layer):
        s = Stimulus(text="a b", stimulus_id=sid)
        words = AnnotationLayer("words", [Span(0, 1), Span(2, 3)], ["a", "b"])
        s.add_layer(words)
        if with_layer:
            s.add_layer(AnnotationLayer(SENTENCE_LAYER, list(words.spans), [1, 1]))
        return s

    stimuli = [stimulus("s1", True), stimulus("s2", True)]
    attach_sentences("my_corpus", stimuli)
    assert [s.layers[SENTENCE_LAYER].values for s in stimuli] == [[1, 1], [1, 1]]
    with pytest.raises(ValueError, match="attached to 1 of 2 stimuli"):
        attach_sentences("my_corpus", [stimulus("s1", True), stimulus("s2", False)])


@pytest.mark.parametrize(
    ("defect", "message"),
    [
        ("shifted_spans", "sentence spans must equal the word spans"),
        ("short_values", "1 sentence values for 2 words"),
        ("missing_words", "needs a words layer"),
    ],
)
def test_a_malformed_loader_sentence_layer_is_refused(defect, message):
    from replica.core import AnnotationLayer, Span, Stimulus

    stimulus = Stimulus(text="a b", stimulus_id="s1")
    word_spans = [Span(0, 1), Span(2, 3)]
    if defect != "missing_words":
        stimulus.add_layer(AnnotationLayer("words", word_spans, ["a", "b"]))
    spans = [Span(2, 3), Span(0, 1)] if defect == "shifted_spans" else list(word_spans)
    layer = AnnotationLayer(SENTENCE_LAYER, spans, [1, 1])
    if defect == "short_values":
        layer.values = [1]  # the constructor refuses this; a later edit can do it
    stimulus.add_layer(layer)
    with pytest.raises(ValueError, match=message):
        attach_sentences("my_corpus", [stimulus])
