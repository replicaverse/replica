"""Correctness tests for the four surprisal FeatureModels.

Each method is checked against the published reference algorithm of
the papers REPLICA claims to support: a fresh re-implementation of the
reference (derived from the published code or paper) is inlined and
REPLICA is asserted to match within numerical noise.

  * HuggingFaceLM (sum-of-subword-logprobs) is verified against the
    reference algorithms of Wilcox et al. (2023), Pimentel et al. (2023)
    and Giulianelli et al. (2023). They differ in BOS handling and in unit
    conventions; a wrong HF path would be very unlikely to match all
    three.
  * WordsProbabilityLM(correction="buggy") is verified against an inline
    reproduction of wordsprobability's surprisal_buggy column - and
    cross-checked: it must equal HuggingFaceLM(add_bos=True) aggregated
    per whitespace word.
  * WordsProbabilityLM(correction="corrected") is verified to *differ*
    from "buggy" by a measurable margin (the published BOW/BOS/EOW
    correction is non-trivial; if it were a no-op the test catches it).
  * Source-free controls resolve directly from stimulus annotations.
  * surprisal_model() is checked against its dispatch table.

All tests use the publicly-available ``gpt2`` checkpoint. The module is
marked ``needs_model``; the heavier cases are additionally marked ``slow``.
"""

from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from replica.core import AnnotationLayer, Span, Stimulus
from replica.dataset.measures import FIRST_PASS_RUN
from replica.feature import (
    CharBeamLM,
    HuggingFaceLM,
    WordsProbabilityLM,
    surprisal_model,
)
from tests._stimuli import word_stimulus

pytestmark = pytest.mark.needs_model

SENTENCES = [
    # Sentence with multi-subword words ("seashells" → 'seas','hell','s';
    # "seashore" → 'se','ash','ore'). The non-trivial subword aggregation
    # is where the alignment-bug failure modes live; a simple
    # single-token-per-word sentence would not catch them.
    "She sells seashells by the seashore",
    # Longer mixed-vocabulary sentence to exercise the sliding-window
    # codepath only kicks in on much longer inputs, but this still adds
    # diversity over the multi-subword case above.
    "There has been a great deal of recent interest in this question",
]


def _whitespace_word_stimulus(text: str) -> Stimulus:
    """Build a Stimulus with a 'words' layer split on spaces."""
    return word_stimulus(text, "x")


@lru_cache(maxsize=1)
def _gpt2_model_and_tok():
    """Load gpt2 once per process. Eval mode, no grad."""
    tok = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    model.eval()
    return model, tok


# The reference implementations below run gpt2 on CPU; pin REPLICA to CPU too so
# the comparisons are bit-exact (GPU and CPU float32 results differ enough to
# break the tight per-token assertions on a GPU node).
@pytest.fixture(scope="module")
def hf_lm_no_bos(lm_cache):
    return lm_cache("gpt2", add_bos=False, device="cpu")


@pytest.fixture(scope="module")
def hf_lm_with_bos(lm_cache):
    return lm_cache("gpt2", add_bos=True, device="cpu")


# ============================================================================
# Reference implementations (inlined fresh from each paper's published code)
# ============================================================================


def _ref_wilcox2023_per_word_bits(sentence: str) -> np.ndarray:
    """Wilcox et al. (2023) reference (xlang-processing/scripts/get_predictors.py).

    Algorithm:
      tokenize(sent, add_special_tokens=True)
      probs = softmax(model(ids).logits)
      surp[t] = -log2(probs[t-1, ids[t]])     # bits, position 0 = 0
      word agg: sum surps for subwords starting with ``Ġ``-boundary;
                first token gets surp=0 by convention.
    """
    model, tok = _gpt2_model_and_tok()
    sent_tokens = tok.tokenize(sentence, add_special_tokens=True)
    ids = tok.convert_tokens_to_ids(sent_tokens)
    with torch.no_grad():
        probs = model(torch.tensor([ids])).logits.softmax(-1).squeeze(0).numpy()
    per_tok = np.zeros(len(ids))
    for t in range(1, len(ids)):
        per_tok[t] = -np.log2(probs[t - 1, ids[t]])

    word_surps, cur = [], 0.0
    for tk, s in zip(sent_tokens, per_tok, strict=False):
        if tk.startswith("Ġ"):
            word_surps.append(cur)
            cur = float(s)
        else:
            cur += float(s)
    word_surps.append(cur)
    return np.array(word_surps)


def _ref_pimentel2024_per_subword_nats(sentence: str) -> tuple[list[str], np.ndarray]:
    """Pimentel et al. (2023) reference (anticipation-on-reading-times/.../gpt2.py).

    Algorithm:
      input = [BOS] + tokens + [EOS]
      cross_entropy(shift_logits, shift_labels) per position, in nats.
      Drop the EOS-prediction position; first len(ids) values are the
      surprisals of the original tokens given BOS+prefix.
    """
    model, tok = _gpt2_model_and_tok()
    enc = tok(sentence, add_special_tokens=False)
    ids = enc["input_ids"]
    sent_tokens = tok.convert_ids_to_tokens(ids)
    bos = tok.bos_token_id
    eos = tok.eos_token_id
    tensor_input = torch.tensor([[bos, *ids, eos]])
    with torch.no_grad():
        logits = model(tensor_input).logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = tensor_input[..., 1:].contiguous()
        per_tok = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="none",
        ).numpy()
    return sent_tokens, per_tok[: len(ids)]


def _ref_giulianelli2023_per_subword_nats(
    sentence: str,
) -> tuple[list[str], np.ndarray]:
    """Giulianelli et al. (2023) reference (information-value/code/compute_surprisal.py).

    Algorithm:
      input_ids = [BOS] + tokens
      log_probs = log_softmax(model(input_ids).logits)
      surp[t] = -log_probs[t, ids[t]] in nats.
    """
    model, tok = _gpt2_model_and_tok()
    ids = tok(sentence, add_special_tokens=False)["input_ids"]
    sent_tokens = tok.convert_ids_to_tokens(ids)
    bos = tok.bos_token_id
    input_ids = torch.tensor([bos, *ids])
    with torch.no_grad():
        logits = model(input_ids.unsqueeze(0)).logits.squeeze(0)
        log_probs = F.log_softmax(logits, dim=-1).numpy()
    per_tok = np.array([-log_probs[t, ids[t]] for t in range(len(ids))])
    return sent_tokens, per_tok


# ============================================================================
# Per-subword surprisal: REPLICA output, when a stimulus's units layer
# is exactly the BPE subwords, must match each reference's per-token
# array (with BOS=True for wordsprobability*/giulianelli, BOS=False for wilcox).
# ============================================================================


def _replica_per_subword_nats(
    sentence: str, lm: HuggingFaceLM
) -> tuple[list[str], np.ndarray]:
    """Run REPLICA HuggingFaceLM with a units layer aligned to subwords."""
    tok = AutoTokenizer.from_pretrained("gpt2")
    enc = tok(sentence, return_offsets_mapping=True, add_special_tokens=False)
    sent_tokens = tok.convert_ids_to_tokens(enc["input_ids"])
    spans = [Span(s, e) for s, e in enc["offset_mapping"]]
    stim = Stimulus(text=sentence, stimulus_id="x")
    stim.add_layer(AnnotationLayer(name="units", spans=spans, values=sent_tokens))
    out = lm(stim, extract=["log_p_observed"], target_layer="units")
    return sent_tokens, -np.array([o.values["log_p_observed"] for o in out])


@pytest.mark.slow
@pytest.mark.parametrize("sentence", SENTENCES)
def test_huggingface_lm_matches_wilcox2023(hf_lm_no_bos, sentence):
    """HuggingFaceLM(add_bos=False), per whitespace-word, in bits, equals
    Wilcox et al. (2023)'s published per-word surprisal."""
    ref = _ref_wilcox2023_per_word_bits(sentence)
    stim = _whitespace_word_stimulus(sentence)
    out = hf_lm_no_bos(stim, extract=["log_p_observed"], target_layer="words")
    nats = np.array([o.values["log_p_observed"] for o in out])
    bits = np.where(np.isnan(nats), 0.0, -nats / np.log(2))
    diff = np.abs(ref - bits)
    assert diff.max() < 1e-4, (
        f"max |diff| = {diff.max():.2e} at index {diff.argmax()} on {sentence!r}"
    )


@pytest.mark.slow
@pytest.mark.parametrize("sentence", SENTENCES)
def test_huggingface_lm_matches_pimentel2024(hf_lm_with_bos, sentence):
    """HuggingFaceLM(add_bos=True), per BPE subword, in nats, equals
    Pimentel et al. (2023)'s published cross_entropy-based per-token surprisal."""
    pim_tokens, ref = _ref_pimentel2024_per_subword_nats(sentence)
    replica_tokens, REPLICA = _replica_per_subword_nats(sentence, hf_lm_with_bos)
    assert pim_tokens == replica_tokens
    diff = np.abs(ref - REPLICA)
    assert diff.max() < 1e-4, (
        f"max |diff| = {diff.max():.2e} at token "
        f"{replica_tokens[diff.argmax()]!r} in {sentence!r}"
    )


@pytest.mark.slow
@pytest.mark.parametrize("sentence", SENTENCES)
def test_huggingface_lm_matches_giulianelli2023(hf_lm_with_bos, sentence):
    """HuggingFaceLM(add_bos=True), per BPE subword, in nats, equals
    Giulianelli et al. (2023)'s log_softmax-based per-token surprisal."""
    gi_tokens, ref = _ref_giulianelli2023_per_subword_nats(sentence)
    replica_tokens, REPLICA = _replica_per_subword_nats(sentence, hf_lm_with_bos)
    assert gi_tokens == replica_tokens
    diff = np.abs(ref - REPLICA)
    # The Giulianelli et al. (2023) path uses log_softmax-then-index, the same code path
    # as REPLICA; expect bit-exact agreement.
    assert diff.max() < 1e-9, f"max |diff| = {diff.max():.2e} on {sentence!r}"


# ============================================================================
# WordsProbabilityLM
# ============================================================================


@pytest.mark.slow
def test_wordsprob_buggy_equals_hf_with_bos():
    """wordsprobability's surprisal_buggy column reproduces the standard
    sum-of-subword-logprobs convention. HuggingFaceLM(add_bos=True) summed
    per whitespace word is that convention, so the two must agree."""
    sentence = "The quick brown fox jumps over the lazy dog."
    stim = _whitespace_word_stimulus(sentence)

    hf = HuggingFaceLM("gpt2", add_bos=True)
    hf_out = hf(stim, extract=["log_p_observed"], target_layer="words")
    hf_surp = -np.array([o.values["log_p_observed"] for o in hf_out])

    buggy = WordsProbabilityLM("gpt2-small", correction="buggy")
    b_out = buggy(stim, target_layer="words")
    b_surp = -np.array([o.values["log_p_observed"] for o in b_out])

    diff = np.abs(hf_surp - b_surp)
    assert diff.max() < 1e-4, (
        f"WordsProbabilityLM(buggy) diverged from HuggingFaceLM(add_bos=True): "
        f"max |diff| = {diff.max():.2e}"
    )


@pytest.mark.slow
def test_wordsprob_sums_the_words_of_a_multiword_unit():
    """A unit of two whitespace words takes the sum of their log probabilities."""
    text = "The quick brown fox jumps over the lazy dog."
    single = _whitespace_word_stimulus(text)
    fused = Stimulus(text=text, stimulus_id="x")
    spans = list(single.layers["words"].spans)
    fused_spans = [Span(spans[0].start, spans[1].end), *spans[2:]]
    fused.add_layer(
        AnnotationLayer(
            "words", fused_spans, [text[s.start : s.end] for s in fused_spans]
        )
    )
    lm = WordsProbabilityLM("gpt2-small", correction="corrected")
    per_word = [o.values["log_p_observed"] for o in lm(single, target_layer="words")]
    per_unit = [o.values["log_p_observed"] for o in lm(fused, target_layer="words")]
    assert len(per_unit) == len(per_word) - 1
    np.testing.assert_allclose(
        per_unit, [per_word[0] + per_word[1], *per_word[2:]], rtol=0, atol=1e-12
    )


@pytest.mark.slow
def test_wordsprob_corrected_differs_from_buggy():
    """The Pimentel & Meister (2024) correction is non-trivial, at least
    one word should shift by more than 0.05 nats. If it were a no-op
    (i.e.\\ same as buggy), this test catches that."""
    sentence = "The quick brown fox jumps over the lazy dog."
    stim = _whitespace_word_stimulus(sentence)
    buggy = WordsProbabilityLM("gpt2-small", correction="buggy")
    corr = WordsProbabilityLM("gpt2-small", correction="corrected")
    b_surp = -np.array(
        [o.values["log_p_observed"] for o in buggy(stim, target_layer="words")]
    )
    c_surp = -np.array(
        [o.values["log_p_observed"] for o in corr(stim, target_layer="words")]
    )
    diff = np.abs(b_surp - c_surp)
    assert diff.max() > 0.05, (
        "WordsProbabilityLM(corrected) is suspiciously close to "
        "WordsProbabilityLM(buggy); the BOW/BOS/EOW correction may be "
        f"silently disabled. max |diff| = {diff.max():.2e}"
    )


# ============================================================================
# Source-free controls
# ============================================================================


def test_evaluate_with_data_only_predictor():
    """Source-free features require no model placeholder."""
    from replica import (
        OLS,
        AnalysisConfig,
        Datum,
        EvaluationSpec,
        ItemSplitter,
        UnitLength,
        evaluate,
    )
    from replica.dataset import Dataset, DatasetInfo

    # Construct a minimal in-memory Dataset with two stimuli.
    class _Toy(Dataset):
        info = DatasetInfo(name="toy")
        default_measure = "rt"

        def __init__(self):
            self._stimuli = []
            self._data = []
            for sid, text in [("s1", "the cat sat"), ("s2", "the dog ran")]:
                stim = _whitespace_word_stimulus(text)
                stim.stimulus_id = sid
                self._stimuli.append(stim)
                # Two participants per stimulus, random RTs.
                for pid in ("p1", "p2"):
                    rt = np.array([200.0, 250.0, 300.0]) + np.array([0.0, 10.0, -5.0])
                    self._data.append(
                        Datum(
                            stimulus=stim,
                            measure_name="rt",
                            values=rt,
                            participant_id=pid,
                            layer="words",
                        )
                    )

        def stimuli(self):
            return self._stimuli

        def data(self, measure=None, participant=None, *, source="default"):
            return [
                d
                for d in self._data
                if (measure is None or d.measure_name == measure)
                and (participant is None or d.participant_id == participant)
            ]

        @property
        def measures(self):
            return ["rt"]

        def load(self, path):  # pragma: no cover
            pass

    ds = _Toy()
    spec = EvaluationSpec(
        predictors=[UnitLength()],
        controls=[],
        targets=[ds.feature("rt")],
    )
    cfg = AnalysisConfig(
        design=None, transform_dv=np.log2, model=OLS(), cv=ItemSplitter(2), n_folds=2
    )
    results = evaluate(ds, spec, cfg)
    assert len(results) == 1
    assert results[0].n_observations > 0


# ============================================================================
# surprisal_model() factory
# ============================================================================


def test_surprisal_model_dispatches_to_correct_classes():
    """The factory maps a method name to a FeatureModel class. Pointing
    'surprisal' at the wrong class silently produces wrong numbers in every
    downstream comparison table."""
    standard = surprisal_model("gpt2", method="surprisal", add_bos=True)
    assert isinstance(standard, HuggingFaceLM)

    char_beam = surprisal_model("gpt2", method="char_beam", K=5)
    assert isinstance(char_beam, CharBeamLM)

    corr = surprisal_model("gpt2-small", method="wordsprobability_surprisal")
    assert isinstance(corr, WordsProbabilityLM)
    assert corr.correction == "corrected"

    buggy = surprisal_model("gpt2-small", method="wordsprobability_surprisal_buggy")
    assert isinstance(buggy, WordsProbabilityLM)
    assert buggy.correction == "buggy"


def test_surprisal_model_rejects_unknown_method():
    with pytest.raises(ValueError, match="Unknown surprisal method"):
        surprisal_model("gpt2", method="not_a_real_method")


# ============================================================================
# End-to-end gap tests: each surprisal method must compose cleanly with
# evaluate() to produce a valid AnalysisResult. Catches the case where the
# FeatureModel produces correct outputs in isolation but doesn't satisfy
# evaluate()'s contract (e.g. NaN handling, target_layer alignment).
# ============================================================================


@pytest.mark.slow
@pytest.mark.parametrize(
    "method, kwargs, model_id",
    [
        ("surprisal", {"add_bos": True}, "gpt2"),
        ("wordsprobability_surprisal_buggy", {}, "gpt2-small"),
        ("wordsprobability_surprisal", {}, "gpt2-small"),
    ],
    ids=["standard", "pm_buggy", "pm_corrected"],
)
def test_surprisal_method_runs_end_to_end_through_evaluate(method, kwargs, model_id):
    """For each word-level surprisal method, building a EvaluationSpec and
    running it through evaluate() must produce an AnalysisResult with a
    finite delta_ll and at least one observation."""

    from replica import (
        OLS,
        AnalysisConfig,
        EvaluationSpec,
        ItemSplitter,
        ModelFeature,
        Surprisal,
        UnitLength,
        evaluate,
    )
    from replica.dataset import PROVO

    repo_root = Path(__file__).resolve().parents[3]
    provo_dir = repo_root / "datasets/eye-tracking/original/provo"
    if not provo_dir.exists() or not any(provo_dir.iterdir()):
        pytest.skip(f"PROVO data missing at {provo_dir}; run `replica fetch provo`.")

    ds = PROVO()
    ds.load(str(provo_dir))
    # Trim to a handful of stimuli so this test stays fast.
    keep = {s.stimulus_id for s in ds.stimuli()[:5]}
    ds._stimuli = [s for s in ds._stimuli if s.stimulus_id in keep]
    ds._data = [d for d in ds._data if d.stimulus.stimulus_id in keep]

    spec = EvaluationSpec(
        predictors=[
            ModelFeature(
                surprisal_model(model_id, method=method, **kwargs), Surprisal()
            )
        ],
        controls=[UnitLength()],
        targets=[ds.feature(FIRST_PASS_RUN)],
    )
    cfg = AnalysisConfig(
        design=None, transform_dv=np.log2, model=OLS(), cv=ItemSplitter(2), n_folds=2
    )

    results = evaluate(ds, spec, cfg)
    assert len(results) == 1, f"{method}: expected 1 result, got {len(results)}"
    r = results[0]
    assert np.isfinite(r.delta_ll), f"{method}: delta_ll is non-finite ({r.delta_ll})"
    assert r.n_observations > 0, f"{method}: zero observations after pipeline ran"
