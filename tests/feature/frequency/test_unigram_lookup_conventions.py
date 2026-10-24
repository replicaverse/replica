"""Frequency models preserve source surfaces and distinguish lookup conventions."""

import numpy as np
import pytest

from replica.core import AnnotationLayer, Span, Stimulus
from replica.feature.frequency.model import UnigramSurprisalLM


class _RecordingSource:
    """Record requested surfaces with a constant probability."""

    language = "en"

    def __init__(self):
        self.asked: list[str] = []
        self.warmed: list[str] = []

    def probability(self, unit: str) -> float:
        self.asked.append(unit)
        return 0.5

    def estimate_words(self, words):
        self.warmed.extend(words)
        return dict.fromkeys(words, 0.5)


def _stimulus(text: str, spans: list[tuple[int, int]]) -> Stimulus:
    stimulus = Stimulus(text=text, stimulus_id="s0")
    stimulus.add_layer(
        AnnotationLayer(
            name="words",
            spans=[Span(a, b) for a, b in spans],
            values=[None] * len(spans),
        )
    )
    return stimulus


#: Space-joined text, the shape ``DatasetBase._make_stimulus`` produces.
_SPACED = _stimulus(
    "In ancient Rome The dog", [(0, 2), (3, 10), (11, 15), (16, 19), (20, 23)]
)


def test_table_lookup_preserves_surface_for_source_normalization():
    source = _RecordingSource()
    UnigramSurprisalLM(source, lookup="table")(_SPACED)
    assert source.asked == ["In", "ancient", "Rome", "The", "dog"]


def test_user_table_keeps_case_distinctions_through_the_model():
    from replica.feature.frequency.sources import UserTable

    model = UnigramSurprisalLM(UserTable({"The": 0.2, "the": 0.1}))
    outputs = model(_stimulus("The the", [(0, 3), (4, 7)]))
    np.testing.assert_allclose(
        [out.values["log_p_observed"] for out in outputs], np.log([0.2, 0.1])
    )


def test_lm_surface_preserves_case_and_prepends_after_a_space():
    source = _RecordingSource()
    UnigramSurprisalLM(source, lookup="lm_surface")(_SPACED)
    # Unit 0 is passage-initial, so it stays bare; the rest are word-initial.
    assert source.asked == ["In", " ancient", " Rome", " The", " dog"]


@pytest.mark.parametrize("lookup", ["table", "lm_surface"])
def test_unspaced_segments_are_individual_lookup_keys(lookup):
    source = _RecordingSource()
    source.language = "zh"
    outputs = UnigramSurprisalLM(source, lookup=lookup)(
        _stimulus("北京大学", [(0, 2), (2, 4)])
    )
    assert source.asked == ["北京", "大学"]
    np.testing.assert_allclose([o.values["unigram_surprisal"] for o in outputs], 1.0)
    np.testing.assert_allclose(
        [o.values["log_p_observed"] for o in outputs], np.log(0.5)
    )


def test_lm_surface_keeps_a_fused_interest_area_as_one_surface():
    # MECO stores "5. Juni" as one interest area, collapsed to "5._Juni".
    stimulus = _stimulus("a 5._Juni b", [(0, 1), (2, 9), (10, 11)])

    lm_surface = _RecordingSource()
    UnigramSurprisalLM(lm_surface, lookup="lm_surface")(stimulus)
    assert lm_surface.asked == ["a", " 5. Juni", " b"]

    table = _RecordingSource()
    UnigramSurprisalLM(table, lookup="table")(stimulus)
    assert table.asked == ["a", "5.", "Juni", "b"]


def test_empty_interest_area_sentinel_is_nan_under_both_conventions():
    # ``clean_word`` maps an empty interest area to a literal "_".
    stimulus = _stimulus("a _ b", [(0, 1), (2, 3), (4, 5)])
    for lookup in ("table", "lm_surface"):
        outputs = UnigramSurprisalLM(_RecordingSource(), lookup=lookup)(stimulus)
        assert np.isnan(outputs[1].values["unigram_surprisal"])
        assert np.isnan(outputs[1].values["log_p_observed"])


@pytest.mark.needs_model
def test_samples_are_scored_with_the_bos_they_were_generated_under():
    """``_encode_samples`` prepends BOS explicitly.

    ``add_special_tokens=True`` adds nothing on gpt2 or mGPT.
    """
    transformers = pytest.importorskip("transformers")
    from replica.feature.frequency.sampled import _encode_samples

    tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
    bos = tokenizer.bos_token_id

    # The tokenizer itself adds nothing; the prefix has to be explicit.
    assert tokenizer.encode("the dog", add_special_tokens=True) == tokenizer.encode(
        "the dog", add_special_tokens=False
    )

    encoded = _encode_samples(tokenizer, ["the dog", "", "a cat"])
    assert all(ids[0] == bos for ids in encoded)
    assert encoded[0][1:] == tokenizer.encode("the dog", add_special_tokens=False)
    assert encoded[1] == [bos]  # an empty sample encodes to a single BOS


def test_prepopulate_warms_exactly_the_surfaces_call_requests():
    # The batch pass must warm exactly the keys the per-stimulus pass requests.
    for lookup in ("table", "lm_surface"):
        source = _RecordingSource()
        model = UnigramSurprisalLM(source, lookup=lookup)
        model.prepopulate_cache([_SPACED])
        model(_SPACED)
        assert set(source.warmed) == set(source.asked)
