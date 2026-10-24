"""Per-token tables: scoring, storage and guarded aggregation to words.

The backend's token path is driven with local doubles that produce byte-level
style offsets (a leading space inside the token, standalone spaces as their
own tokens) and a deterministic causal model, so every check runs without a
pretrained checkpoint. The derived word values must equal the unit-level call
of the same backend value for value.
"""

import json
import pickle
import re
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from replica import artifacts as manifests
from replica.compute import methods, runner
from replica.compute.cache import TOKEN_SENSITIVE, cache_filename
from replica.compute.units import TOKEN_UNIT, derive_units, verify_token_rows
from replica.context import CONTEXT_SEQUENCE, SENTENCE_LAYER, attach_preceding_context
from replica.core import AnnotationLayer, Span, Stimulus
from replica.feature import FeatureModelOutput, HuggingFaceLM, PrecedingContextModel
from replica.feature.core.annotation.alignment import TokenBoundaryError, tokenize_spans
from replica.feature.core.outputs.information import LN2
from replica.feature.distance.responsivity import ResponsivityLM
from replica.utils.digest import file_sha256
from tests._stimuli import word_stimulus

KEYS = ("log_p_observed", "token_entropy", "token_renyi_entropy_alpha0.5")
RULES = {
    "log_p_observed": "sum",
    "token_entropy": "first",
    "token_renyi_entropy_alpha0.5": "first",
}


class _Tokenizer:
    """Pieces of up to three characters; a word's first piece keeps its space."""

    bos_token_id = 0
    vocab_size = 64
    name_or_path = "test-tokens"

    def __init__(self, crossing=False):
        self.pieces: list[str] = ["<bos>"]
        self.crossing = crossing

    def _id(self, piece):
        if piece not in self.pieces:
            self.pieces.append(piece)
        assert len(self.pieces) <= self.vocab_size
        return self.pieces.index(piece)

    def __call__(self, text, **kwargs):
        ids, offsets = [], []
        for match in re.finditer(r" ?\S+| ", text):
            start, end = match.span()
            first = start + (text[start] == " ")
            if self.crossing and text[start:end].strip():
                # One token running into the first character of the next word.
                end = min(len(text), end + 2)
                ids.append(self._id(text[start:end]))
                offsets.append((start, end))
                continue
            cuts = (
                [start, *range(first + 3, end, 3), end] if end > first else [start, end]
            )
            for a, b in zip(cuts, cuts[1:], strict=False):
                ids.append(self._id(text[a:b]))
                offsets.append((a, b))
        return {"input_ids": ids, "offset_mapping": offsets}

    def decode(self, ids):
        return "".join(self.pieces[i] for i in ids)


class _CausalModel:
    def __init__(self, limit=1024):
        self.config = SimpleNamespace(max_position_embeddings=limit, vocab_size=64)
        self.device = "cpu"

    def __call__(self, input_ids):
        assert input_ids.shape[-1] <= self.config.max_position_embeddings
        prefix = input_ids.float().cumsum(-1)
        vocabulary = torch.arange(64, dtype=torch.float32)
        logits = -(vocabulary - (prefix.remainder(11) + 1).unsqueeze(-1)).square() / 9
        return SimpleNamespace(logits=logits)


class _DoubleLM(HuggingFaceLM):
    def _ensure_loaded(self):
        assert isinstance(self._model, _CausalModel)


def _lm(*, bos=True, limit=1024, scope="stimulus", crossing=False):
    lm = _DoubleLM(
        "test-tokens",
        add_bos=bos,
        provide_word_dist=False,
        token_level_entropy=True,
        token_renyi_alphas=[0.5],
        device="cpu",
        context_scope=scope,
    )
    lm._tokenizer = _Tokenizer(crossing=crossing)
    lm._model = _CausalModel(limit)
    return lm


def _values(outputs, key):
    return np.array([o.values[key] for o in outputs])


def _tiles(text, rows):
    """Text rebuilt from consecutive distinct spans, byte-split pieces merged."""
    pieces, previous = [], None
    for row in rows:
        span = tuple(row.metadata["span"])
        if span != previous:
            pieces.append(text[span[0] : span[1]])
        previous = span
    return "".join(pieces)


TEXT = "The cats sat on a  mat, unhappily."


@pytest.mark.parametrize("bos", [False, True])
def test_token_rows_tile_the_text_and_name_their_tokens(bos):
    lm = _lm(bos=bos)
    stimulus = word_stimulus(TEXT)
    rows = lm.score_tokens(stimulus)
    ids, spans = tokenize_spans(TEXT, lm._tokenizer)
    assert [r.metadata["token_index"] for r in rows] == list(range(len(ids)))
    assert [r.metadata["token_id"] for r in rows] == ids
    assert [tuple(r.metadata["span"]) for r in rows] == [
        (s.start, s.end) for s in spans
    ]
    assert [r.metadata["token"] for r in rows] == [TEXT[s.start : s.end] for s in spans]
    assert _tiles(TEXT, rows) == TEXT
    verify_token_rows(stimulus, rows)
    first = rows[0].values
    assert all(np.isnan(first[k]) for k in KEYS) == (not bos)
    assert all(np.isfinite(_values(rows[1:], k)).all() for k in KEYS)


@pytest.mark.parametrize("bos", [False, True])
@pytest.mark.parametrize("limit", [8, 1024])
def test_derived_words_equal_the_unit_level_call(bos, limit):
    lm = _lm(bos=bos, limit=limit)
    stimulus = word_stimulus(TEXT)
    rows = lm.score_tokens(stimulus)
    expected = lm(stimulus)
    for key in KEYS:
        derived = derive_units(
            stimulus, rows, unit=TOKEN_UNIT, layer="words", key=key, rule=RULES[key]
        )
        np.testing.assert_array_equal(_values(derived, key), _values(expected, key))
        assert derived[0].metadata == {"model_name": lm.name}
        if key == "log_p_observed":
            np.testing.assert_array_equal(
                _values(derived, "surprisal"), -_values(derived, key) / LN2
            )
        assert set(derived[0].values) == (
            {key, "surprisal"} if key == "log_p_observed" else {key}
        )
    assert len(derived) == len(stimulus.layers["words"])


def test_mean_subtoken_score_has_a_distinct_quantity_and_metadata():
    lm = _lm()
    stimulus = word_stimulus("unhappily cat")
    tokens = lm.score_tokens(stimulus, extract=["log_p_observed"])
    sums = lm(stimulus, extract=["log_p_observed"])
    lm.aggregation = "mean"
    means = lm(stimulus, extract=["mean_token_log_p"])
    for unit, total, mean in zip(
        stimulus.layers["words"].spans, sums, means, strict=True
    ):
        parts = [
            row.values["log_p_observed"]
            for row in tokens
            if row.metadata["span"][0] < unit.end
            and row.metadata["span"][1] > unit.start
        ]
        assert mean.values == {"mean_token_log_p": pytest.approx(np.mean(parts))}
        assert total.values["log_p_observed"] == pytest.approx(np.sum(parts))
        assert mean.metadata["aggregation"] == "mean"
        assert total.metadata["aggregation"] == "sum"
    with pytest.raises(ValueError, match="mean_token_log_p"):
        lm(stimulus, extract=["log_p_observed"])
    np.testing.assert_array_equal(
        _values(lm.score_tokens(stimulus), "log_p_observed"),
        _values(tokens, "log_p_observed"),
    )


def test_decoded_distribution_keeps_zero_mass_types():
    lm = _lm()
    lm._vocab = ["a", "b"]
    lm._word_to_token_ids = {"a": [0, 1], "b": [2]}
    result = lm._marginalise_to_word_dist(np.array([-np.inf, -np.inf, 0.0]))
    np.testing.assert_array_equal(result, [-np.inf, 0.0])
    with pytest.raises(ValueError, match="no finite probability mass"):
        lm._marginalise_to_word_dist(np.full(3, -np.inf))


def test_sentence_scope_offsets_spans_to_the_full_text():
    lm = _lm(scope="sentence")
    text = "The cat sat. A dog barked loudly."
    stimulus = word_stimulus(text)
    words = stimulus.layers["words"]
    stimulus.add_layer(
        AnnotationLayer(SENTENCE_LAYER, list(words.spans), [1] * 3 + [2] * 4)
    )
    rows = lm.score_tokens(stimulus)
    verify_token_rows(stimulus, rows)
    assert [r.metadata["token_index"] for r in rows] == list(range(len(rows)))
    # Each sentence is tokenized alone, so the second starts without a space.
    starts = [r.metadata["span"][0] for r in rows]
    assert 13 in starts and rows[starts.index(13)].metadata["token"] == "A"
    assert _tiles(text, rows) == text.replace(". A", ".A")
    expected = lm(stimulus)
    for key in KEYS:
        derived = derive_units(
            stimulus, rows, unit=TOKEN_UNIT, layer="words", key=key, rule=RULES[key]
        )
        np.testing.assert_array_equal(_values(derived, key), _values(expected, key))


def _page(text, page=1, book="dickens"):
    stimulus = word_stimulus(text, f"sbsat_{book}_{page}")
    stimulus.metadata[CONTEXT_SEQUENCE] = {"group": book, "position": page}
    return stimulus


def test_story_scope_keeps_the_page_tokens_after_the_context():
    pages = [_page("The first page."), _page("Then more text here.", 2)]
    attach_preceding_context(pages)
    lm = _lm()
    model = PrecedingContextModel(lm)
    rows = model.score_tokens(pages[1])
    verify_token_rows(pages[1], rows)
    assert rows[0].metadata["span"] == (0, 3)
    assert rows[0].metadata["token"] == " The"
    assert rows[0].metadata["context_scope"] == "story"
    assert _tiles(pages[1].text, rows) == pages[1].text
    expected = model(pages[1])
    for key in KEYS:
        derived = derive_units(
            pages[1], rows, unit=TOKEN_UNIT, layer="words", key=key, rule=RULES[key]
        )
        np.testing.assert_array_equal(_values(derived, key), _values(expected, key))
    # The page's own tokens depend on the preceding text.
    alone = lm.score_tokens(pages[1])
    assert not np.allclose(
        _values(rows, "log_p_observed"), _values(alone, "log_p_observed")
    )


def test_window_and_unsupported_keys():
    lm = _lm(limit=8)
    assert lm.window() == {"rule": "sliding", "max_positions": 8, "stride": 4}
    with pytest.raises(ValueError, match="next_log_probs"):
        lm.score_tokens(word_stimulus("a b"), extract=["next_log_probs"])
    assert lm.score_tokens(word_stimulus("a b"), extract=["log_p_observed"])[
        0
    ].values.keys() == {"log_p_observed"}


class _WindowTaggingModel:
    """Logits holding the call number and the local position of each row."""

    def __init__(self, limit):
        self.config = SimpleNamespace(max_position_embeddings=limit, vocab_size=2)
        self.calls = 0

    def __call__(self, input_ids):
        n = input_ids.shape[-1]
        assert n <= self.config.max_position_embeddings
        logits = torch.zeros(1, n, 2)
        logits[0, :, 0] = self.calls
        logits[0, :, 1] = torch.arange(n)
        self.calls += 1
        return SimpleNamespace(logits=logits)


def _window_logits(backend, limit, n_tokens):
    model = _WindowTaggingModel(limit)
    ids = torch.zeros(1, n_tokens, dtype=torch.long)
    if backend == "responsivity":
        lm = ResponsivityLM.__new__(ResponsivityLM)
        lm.model_name, lm.device, lm._model = "test-tokens", "cpu", model
        return lm._all_logits(ids, n_tokens)
    lm = _lm(limit=limit)
    lm._model = model
    return lm._sliding_window_logits(ids, n_tokens, limit)[0]


@pytest.mark.parametrize("backend", ["surprisal", "responsivity"])
@pytest.mark.parametrize("limit", [8, 7])
def test_sliding_window_scores_a_position_in_the_first_window_covering_it(
    backend, limit
):
    # A later window starts at k * stride and scores from the previous
    # window's end, start + max_positions - stride.
    n_tokens, stride = 30, limit // 2
    logits = _window_logits(backend, limit, n_tokens)
    for position in range(n_tokens):
        window = next(k for k in range(n_tokens) if position < k * stride + limit)
        assert logits[position].tolist() == [window, position - window * stride]


def test_crossing_token_is_refused_not_split():
    lm = _lm(crossing=True)
    stimulus = word_stimulus("ab cd ef")
    rows = lm.score_tokens(stimulus)
    assert [r.metadata["token"] for r in rows] == ["ab c", " cd e", " ef"]
    with pytest.raises(
        TokenBoundaryError, match="token 0 'ab c' at \\[0, 4\\) crosses"
    ):
        derive_units(
            stimulus,
            rows,
            unit=TOKEN_UNIT,
            layer="words",
            key="log_p_observed",
            rule="sum",
        )


# ---- the compute driver ---------------------------------------------------

STIMULI = [word_stimulus(TEXT, "s0"), word_stimulus("A second, shorter one.", "s1")]


def _install(
    monkeypatch, tmp_path, *, word_spacing=True, stimuli=STIMULI, builder=None
):
    monkeypatch.setattr(
        runner,
        "_load_dataset",
        lambda name: runner._LoadedDataset(
            deepcopy(stimuli), len(stimuli), "en", word_spacing
        ),
    )
    monkeypatch.setattr(
        runner,
        "resolve_model_snapshot",
        lambda method, context: (
            context.model_id,
            {"source": "test-tokens", "revision": "r" * 40},
        ),
    )
    for method in TOKEN_SENSITIVE:
        monkeypatch.setitem(
            methods.METHOD_SPECS,
            method,
            replace(
                methods.METHOD_SPECS[method],
                build_model=builder or (lambda context: _lm(bos=context.add_bos)),
            ),
        )
    return dict(
        model_id="test-tokens",
        K=5,
        n_samples=64,
        max_tokens=10,
        alpha=0.5,
        seed=None,
        out_dir=tmp_path,
        add_bos=True,
        force=False,
    )


def _load(path):
    with path.open("rb") as stream:
        return pickle.load(stream)


def _manifest(path):
    return json.loads(manifests.manifest_path(path).read_text())


def test_token_cell_stores_rows_and_records_its_scoring(tmp_path, monkeypatch):
    common = _install(monkeypatch, tmp_path)
    outcome = runner.compute_for(
        "corpus", method="surprisal", unit=TOKEN_UNIT, **common
    )
    assert outcome.status == "completed", outcome.error
    assert outcome.path.name == "test-tokens_surprisal_tokens_corpus.pkl"
    table = _load(outcome.path)
    assert set(table) == {"s0", "s1"}
    rows = table["s0"]
    assert _tiles(TEXT, rows) == TEXT
    assert set(rows[0].metadata) == {
        "model_name",
        "start_ids",
        "context_window",
        "token_index",
        "span",
        "token_id",
        "token",
    }
    assert rows[0].metadata["start_ids"] == [0]
    assert set(rows[0].values) == {"log_p_observed"}
    manifest = _manifest(outcome.path)
    assert manifest["status"] == "completed"
    assert manifest["identity"]["units"] == {"log_p_observed": "nats"}
    assert manifest["window"] == {
        "rule": "sliding",
        "max_positions": 1024,
        "stride": 512,
    }
    options = manifest["identity"]["options"]
    assert options["unit"] == "tokens" and options["add_bos"] is True
    assert options["context_scope"] == "stimulus"
    assert manifest["identity"]["model"]["revision"] == "r" * 40
    assert (
        "device" in manifest["environment"]
        and "model_device" in manifest["environment"]
    )
    entropy = runner.compute_for(
        "corpus", method="token_renyi_entropy", unit=TOKEN_UNIT, **common
    )
    assert entropy.path.name == "test-tokens_token_renyi_entropy_a0.5_tokens_corpus.pkl"
    assert _manifest(entropy.path)["identity"]["units"] == {
        "token_renyi_entropy_alpha0.5": "bits"
    }


@pytest.mark.parametrize(
    "method,key",
    [
        ("surprisal", "log_p_observed"),
        ("first_token_entropy", "token_entropy"),
        ("token_renyi_entropy", "token_renyi_entropy_alpha0.5"),
    ],
)
def test_word_cell_is_derived_from_the_token_table(tmp_path, monkeypatch, method, key):
    common = _install(monkeypatch, tmp_path)
    outcome = runner.compute_for("corpus", method=method, **common)
    assert outcome.status == "completed", outcome.error
    token_path = tmp_path / cache_filename(
        model_id="test-tokens",
        method=method,
        dataset="corpus",
        alpha=0.5,
        unit=TOKEN_UNIT,
    )
    assert outcome.path == token_path.with_name(
        token_path.name.replace("_tokens_", "_")
    )
    words = _load(outcome.path)
    # One direct instance in corpus order: the tokenizer double assigns ids by
    # first sight, as the driver's own instance did.
    direct = _lm()
    for stimulus in STIMULI:
        expected = direct(stimulus)
        np.testing.assert_array_equal(
            _values(words[stimulus.stimulus_id], key), _values(expected, key)
        )
        assert len(words[stimulus.stimulus_id]) == len(stimulus.layers["words"])
        stored = words[stimulus.stimulus_id]
        if key == "log_p_observed":
            assert set(stored[0].values) == {key, "surprisal"}
            np.testing.assert_array_equal(
                _values(stored, "surprisal"), -_values(stored, key) / LN2
            )
        else:
            assert set(stored[0].values) == {key}
    manifest = _manifest(outcome.path)
    assert manifest["identity"]["options"]["unit"] == "words"
    assert manifest["identity"]["units"] == (
        {"log_p_observed": "nats", "surprisal": "bits"}
        if key == "log_p_observed"
        else {key: "bits"}
    )
    assert manifest["derivation"] == {
        "source": token_path.name,
        "source_sha256": file_sha256(token_path),
        "unit": "tokens",
        "rule": RULES[key],
        "keys": {key: RULES[key]},
        "finisher": "surprisal_bits" if key == "log_p_observed" else None,
        "code_commit": manifest["environment"]["code_commit"],
    }
    assert "window" not in manifest
    # A second request reuses both cells.
    again = runner.compute_for("corpus", method=method, **common)
    assert again.status == "cached" and again.path == outcome.path


def test_crossing_token_quarantines_the_word_cell(tmp_path, monkeypatch):
    common = _install(
        monkeypatch,
        tmp_path,
        builder=lambda context: _lm(bos=context.add_bos, crossing=True),
    )
    outcome = runner.compute_for("corpus", method="surprisal", **common)
    assert outcome.status == "quarantined"
    assert outcome.path.name == "test-tokens_surprisal_corpus.suspect.pkl"
    assert not (tmp_path / "test-tokens_surprisal_corpus.pkl").exists()
    failures = (tmp_path / "test-tokens_surprisal_corpus.failed.log").read_text()
    assert "TokenBoundaryError" in failures and "crosses a words boundary" in failures
    # The token table itself is complete and published.
    assert (tmp_path / "test-tokens_surprisal_tokens_corpus.pkl").exists()


class _SpanModel:
    name = "charbeam_test-tokens_K5"
    vocab = None

    def __call__(self, stimulus, extract=None, target_layer="words"):
        return [
            FeatureModelOutput(
                values={"log_p_observed": -float(index + 1)},
                metadata={"model_name": self.name, "K": 5},
            )
            for index, _ in enumerate(stimulus.layers[target_layer].spans)
        ]


def _unspaced():
    stimulus = Stimulus(text="abcdef", stimulus_id="zh")
    stimulus.add_layer(
        AnnotationLayer(
            "words", [Span(0, 2), Span(2, 3), Span(3, 6)], ["ab", "c", "def"]
        )
    )
    return stimulus


def test_character_route_sums_a_blank_inside_the_unit():
    from replica.feature.core.annotation.characters import attach_characters

    stimulus = Stimulus(text="ab cd ef", stimulus_id="zh")
    stimulus.add_layer(
        AnnotationLayer("words", [Span(0, 5), Span(6, 8)], ["ab cd", "ef"])
    )
    attach_characters([stimulus])
    rows = [
        FeatureModelOutput(
            values={"log_p_observed": -float(i + 1)}, metadata={"model_name": "m"}
        )
        for i in range(len(stimulus.text))
    ]
    derived = derive_units(
        stimulus,
        rows,
        unit="characters",
        layer="words",
        key="log_p_observed",
        rule="sum",
    )
    np.testing.assert_array_equal(_values(derived, "log_p_observed"), [-15.0, -15.0])
    np.testing.assert_array_equal(
        _values(derived, "surprisal"), [15.0 / LN2, 15.0 / LN2]
    )


def test_unspaced_corpus_derives_word_surprisal_from_characters(tmp_path, monkeypatch):
    common = _install(monkeypatch, tmp_path, word_spacing=False, stimuli=[_unspaced()])
    monkeypatch.setitem(
        methods.METHOD_SPECS,
        "char_beam",
        replace(
            methods.METHOD_SPECS["char_beam"], build_model=lambda context: _SpanModel()
        ),
    )
    outcome = runner.compute_for("corpus", method="surprisal", **common)
    assert outcome.status == "completed", outcome.error
    assert outcome.method == "char_beam"
    assert outcome.path.name == "test-tokens_charbeam_surprisal_K5_corpus.pkl"
    words = _load(outcome.path)["zh"]
    np.testing.assert_array_equal(_values(words, "log_p_observed"), [-3.0, -3.0, -15.0])
    np.testing.assert_array_equal(
        _values(words, "surprisal"), np.array([3.0, 3.0, 15.0]) / LN2
    )
    assert words[0].metadata == {"model_name": "charbeam_test-tokens_K5", "K": 5}
    manifest = _manifest(outcome.path)
    assert manifest["identity"]["units"] == {
        "log_p_observed": "nats",
        "surprisal": "bits",
    }
    assert manifest["derivation"]["unit"] == "characters"
    assert manifest["derivation"]["rule"] == "sum"
    assert (
        manifest["derivation"]["source"]
        == "test-tokens_charbeam_surprisal_K5_chars_corpus.pkl"
    )
    assert (tmp_path / manifest["derivation"]["source"]).exists()
    with pytest.raises(ValueError, match="no word spacing"):
        runner.compute_for("corpus", method="first_token_entropy", **common)


def test_token_unit_is_refused_where_it_has_no_meaning(tmp_path, monkeypatch):
    common = _install(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="token-unit methods"):
        runner.compute_for("corpus", method="char_beam", unit=TOKEN_UNIT, **common)
    with pytest.raises(ValueError, match="store-distribution"):
        runner.compute_for(
            "corpus",
            method="first_token_entropy",
            unit=TOKEN_UNIT,
            store_distribution=True,
            **common,
        )


def test_distribution_cells_keep_the_direct_word_path(tmp_path, monkeypatch):
    common = _install(monkeypatch, tmp_path)
    outcome = runner.compute_for(
        "corpus", method="first_token_entropy", store_distribution=True, **common
    )
    assert outcome.status == "completed", outcome.error
    assert "derivation" not in _manifest(outcome.path)
    assert not (tmp_path / "test-tokens_ft_entropy_tokens_corpus.pkl").exists()


def test_verify_token_rows_rejects_gaps_and_disorder():
    stimulus = word_stimulus("ab cd")
    rows = _lm().score_tokens(stimulus)
    verify_token_rows(stimulus, rows)
    with pytest.raises(ValueError, match="covered by no token"):
        verify_token_rows(stimulus, rows[:-1])
    swapped = [rows[1], rows[0]]
    for index, row in enumerate(swapped):
        row.metadata["token_index"] = index
    with pytest.raises(ValueError, match="out of order"):
        verify_token_rows(stimulus, swapped)


# ---- real tokenizers and a small checkpoint --------------------------------

PROBES = [
    "The cat sat on the mat.",
    "Ta uksed on lahti",
    "café naïve résumé",
    "我们今天去公园散步。",
    "a  b\tc",
]


@pytest.mark.needs_model
@pytest.mark.parametrize("model_id", ["gpt2", "ai-forever/mGPT"])
def test_real_tokenizer_spans_reassemble_the_text(model_id):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    for text in PROBES:
        ids, spans = tokenize_spans(text, tokenizer)
        assert tokenizer.decode(ids) == text
        rows = [
            FeatureModelOutput(
                values={}, metadata={"token_index": i, "span": (s.start, s.end)}
            )
            for i, s in enumerate(spans)
        ]
        assert _tiles(text, rows) == text
        verify_token_rows(Stimulus(text=text, stimulus_id="p"), rows)


@pytest.mark.needs_model
@pytest.mark.parametrize("bos", [False, True])
def test_tiny_gpt2_derived_words_equal_the_unit_call(bos):
    lm = HuggingFaceLM(
        "sshleifer/tiny-gpt2",
        add_bos=bos,
        provide_word_dist=False,
        token_level_entropy=True,
        token_renyi_alphas=[0.5],
    )
    for text in PROBES[:3]:
        stimulus = word_stimulus(text)
        rows = lm.score_tokens(stimulus)
        verify_token_rows(stimulus, rows)
        expected = lm(stimulus)
        for key in KEYS:
            derived = derive_units(
                stimulus, rows, unit=TOKEN_UNIT, layer="words", key=key, rule=RULES[key]
            )
            np.testing.assert_array_equal(_values(derived, key), _values(expected, key))
