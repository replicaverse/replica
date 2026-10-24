"""Verification cases for REPLICA's public predictor exports.

One :class:`PredictorEntry` per public predictor export, tagged with how it is
verified. This is the single source the CI completeness gate derives from
(``tests/feature/diagnostics/test_catalog.py``), so the catalog cannot drift from the
shipped surface without failing a test.

Status taxonomy:

- ``FINGERPRINTED``, a predictor that reads the reference model's outputs and
  fingerprints *exactly* to a battery spec; ``make`` + ``variants`` let it be
  run live. The expensive backends that only run on real LMs are not here.
- ``VERIFIED``, a real predictor whose target is pinned by a *specific* test
  (a real-LM referee, an exact-limit derivation, or an external-port anchor).
  ``detail`` names it.
- ``NEGATIVE_CONTROL``, data-derived feature that must match *no* spec.
- ``NOT_A_PREDICTOR``, protocol, container, meta-feature, dependent variable,
  data-source adapter, or helper. ``detail`` says which.

"""

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum

from replica.feature.extractors import (
    ContinuationEntropy,
    Entropy,
    ExpectedProbability,
    MCWordEntropy,
    PrevUnitLength,
    Rank,
    RenyiEntropy,
    Surprisal,
    SurprisalReduction,
    TokenEntropy,
    TokenRenyiEntropy,
    UnitLength,
    UnitPosition,
)

__all__ = ["CATALOG", "PredictorEntry", "PredictorStatus", "entry_by_name"]


class PredictorStatus(StrEnum):
    """How a catalogued export is verified, one member per taxonomy tier."""

    FINGERPRINTED = "fingerprinted"
    VERIFIED = "verified"
    NEGATIVE_CONTROL = "negative_control"
    NOT_A_PREDICTOR = "not_a_predictor"


@dataclass
class PredictorEntry:
    """One catalogued predictor export.

    ``variants`` (FINGERPRINTED only) is a tuple of ``(kwargs, expected_spec)``
    pairs: each constructs ``make(**kwargs)`` and must fingerprint exactly to
    ``expected_spec``.
    """

    name: str
    surface: str  # "extractor" | "feature"
    status: PredictorStatus
    category: str
    blurb: str
    detail: str = ""  # verification pointer (VERIFIED) / reason (NOT_A_PREDICTOR)
    equation: str = ""  # inner LaTeX, no delimiters
    paper: tuple[str, ...] = ()
    make: Callable | None = None
    variants: tuple[tuple[dict, str], ...] = field(default_factory=tuple)


def _fp(name, category, blurb, expected, *, make, equation="", paper=(), variants=None):
    return PredictorEntry(
        name=name,
        surface="extractor",
        status=PredictorStatus.FINGERPRINTED,
        category=category,
        blurb=blurb,
        equation=equation,
        paper=paper,
        make=make,
        variants=variants or (({}, expected),),
    )


_P = r"p(w \mid \mathbf{w}_{<t})"

# Paper references: (bibtex-registry key, url). An entry's ``paper`` holds
# keys into this table, so no citation text is stored here.
_REFS = {
    "hale2001": ("hale2001probabilistic", "https://aclanthology.org/N01-1021"),
    "levy2008": (
        "levy2008expectation",
        "https://doi.org/10.1016/j.cognition.2007.05.006",
    ),
    "smith2013": (
        "dataset:smith2013-log-reading-time",
        "https://doi.org/10.1016/j.cognition.2013.02.013",
    ),
    "goodkind2018": ("goodkind2018predictive", "https://aclanthology.org/W18-0102"),
    "wilcox2023": ("wilcox2023surprisal", "https://aclanthology.org/2023.tacl-1.82/"),
    "shain2024": ("shain2024logarithmic", "https://doi.org/10.1073/pnas.2307876121"),
    "pimentel2023": (
        "pimentel2023anticipation",
        "https://doi.org/10.1162/tacl_a_00603",
    ),
    "giulianelli2024": (
        "giulianelli2024generalized",
        "https://aclanthology.org/2024.findings-emnlp.682/",
    ),
    "clark2025": (
        "clark2025wordentropy",
        "https://aclanthology.org/2025.ijcnlp-short.4/",
    ),
    "kuribayashi2025": ("kuribayashi2025internal", "https://doi.org/10.1162/TACL.a.58"),
    "opedal2024": (
        "opedal2024context",
        "https://aclanthology.org/2024.emnlp-main.179/",
    ),
    "pimentel_meister2024": (
        "pimentel2024probability",
        "https://aclanthology.org/2024.emnlp-main.1020/",
    ),
    "giulianelli_tok2024": (
        "giulianelli2024tokenization",
        "https://aclanthology.org/2024.emnlp-main.1032/",
    ),
    "giulianelli2023": (
        "giulianelli2023infovalue",
        "https://openreview.net/forum?id=fkAKjbRvxj",
    ),
    "hopton2026": ("hopton2026unigram", ""),
}


# ── Extractor surface ────────────────────────────────────────────────────

_EXTRACTORS = [
    _fp(
        "Surprisal",
        "surprisal",
        "Information content of the observed word.",
        "surprisal",
        make=Surprisal,
        paper=(
            "hale2001",
            "levy2008",
            "smith2013",
            "goodkind2018",
            "wilcox2023",
            "shain2024",
        ),
        equation=rf"s(w_t) = -\log_2 {_P}\big|_{{w=w_t}}",
    ),
    _fp(
        "Entropy",
        "entropy",
        "Shannon entropy of the pooled first-subword distribution "
        "(next_log_probs; anticipated uncertainty).",
        "entropy",
        make=Entropy,
        equation=rf"H(W_t \mid \mathbf{{w}}_{{<t}}) = -\sum_w {_P}\,\log_2 {_P}",
    ),
    _fp(
        "RenyiEntropy",
        "entropy",
        "Rényi-α entropy of the pooled first-subword distribution (next_log_probs).",
        "renyi_entropy_alpha0.5",
        make=RenyiEntropy,
        equation=rf"H_\alpha = \tfrac{{1}}{{1-\alpha}}\log_2 \sum_w {_P}^{{\,\alpha}}",
        variants=(
            ({"alpha": 0.5}, "renyi_entropy_alpha0.5"),
            ({"alpha": 1.0}, "entropy"),
            ({"alpha": 2.0}, "renyi_entropy_alpha2.0"),
        ),
    ),
    _fp(
        "ExpectedProbability",
        "anticipation",
        "Model's expectation of its own next-word probability (peakedness).",
        "expected_probability",
        make=ExpectedProbability,
        paper=("giulianelli2024",),
        equation=rf"\mathbb{{E}}_{{w\sim p}}[\,p(w)\,] = \sum_w {_P}^{{\,2}} = 2^{{-H_2}}",
    ),
    _fp(
        "Rank",
        "rank",
        "Rank of the observed word in the next-word distribution (ties share "
        "the best rank).",
        "rank",
        make=Rank,
        equation=rf"r(w_t) = 1 + \big|\{{w : {_P} > p(w_t \mid \mathbf{{w}}_{{<t}})\}}\big|",
    ),
    _fp(
        "SurprisalReduction",
        "anticipation",
        "Uncertainty resolved on seeing the word: entropy − surprisal.",
        "surprisal_reduction",
        make=SurprisalReduction,
        equation=r"H(W_t \mid \mathbf{w}_{<t}) - s(w_t)",
    ),
    _fp(
        "TokenEntropy",
        "entropy",
        "Shannon entropy over the (sub)token vocabulary at the first token.",
        "entropy",
        make=TokenEntropy,
        paper=("pimentel2023",),
    ),
    _fp(
        "TokenRenyiEntropy",
        "entropy",
        "Rényi-α entropy over the subword distribution at the first token; "
        "Pimentel et al. (2023) report α=½ as a stronger RT predictor than "
        "Shannon.",
        "renyi_entropy_alpha0.5",
        make=TokenRenyiEntropy,
        paper=("pimentel2023",),
        variants=(
            ({"alpha": 0.5}, "renyi_entropy_alpha0.5"),
            ({"alpha": 2.0}, "renyi_entropy_alpha2.0"),
        ),
    ),
    _fp(
        "ContinuationEntropy",
        "anticipation",
        "Entropy of the distribution over length-L continuations (reads "
        "precomputed MC estimates).",
        "continuation_entropy_L2",
        make=ContinuationEntropy,
        paper=("giulianelli2024",),
        equation=r"H_L = -\mathbb{E}_{c \sim p}\big[\log_2 p(c \mid \mathbf{w}_{<t})\big]",
    ),
    _fp(
        "MCWordEntropy",
        "anticipation",
        "Entropy over whole next words via word-boundary-stopped MC sampling.",
        "entropy",
        make=MCWordEntropy,
        paper=("clark2025",),
    ),
    PredictorEntry(
        "SampledEntropy",
        "extractor",
        PredictorStatus.VERIFIED,
        "entropy",
        "Rényi entropy estimated from retained sample log probabilities.",
        detail="closed-form distribution and stored-output checks in "
        "tests/feature/entropy/test_renyi_estimators.py",
    ),
    PredictorEntry(
        "LayerSurprisal",
        "extractor",
        PredictorStatus.VERIFIED,
        "surprisal",
        "Per-layer logit-lens surprisal from an internal hidden state.",
        detail="real-LM referee in test_real_lm.py (final layer ≡ output, "
        "intermediate differs) + anchored vs released surprisal.json in the "
        "Kuribayashi et al. (2025) reproduction",
        paper=("kuribayashi2025",),
        equation=r"-\log_2 \mathrm{softmax}(W_U\,\mathrm{LN}(h_{l,t}))[w_t]",
    ),
    PredictorEntry(
        "ResponsivityDistance",
        "extractor",
        PredictorStatus.VERIFIED,
        "responsivity",
        "Embedding-space distance from the observed word to LM-sampled "
        "alternatives (no closed-form spec in the battery).",
        detail="exact embedding-space limit tests in test_responsivity_mock.py "
        "(incl. the importance-sampling mixed-limit)",
        paper=("giulianelli2024",),
    ),
    PredictorEntry(
        "UnitLength",
        "extractor",
        PredictorStatus.NEGATIVE_CONTROL,
        "lexical",
        "Character count of the word (data-derived control).",
        detail="equals each unit's character length (verified); matches no "
        "distributional spec",
        make=UnitLength,
    ),
    PredictorEntry(
        "UnitPosition",
        "extractor",
        PredictorStatus.NEGATIVE_CONTROL,
        "lexical",
        "0-indexed position of the word (data-derived control).",
        detail="equals the 0-indexed unit position (verified); matches no "
        "distributional spec",
        make=UnitPosition,
    ),
    PredictorEntry(
        "PrevUnitLength",
        "extractor",
        PredictorStatus.NEGATIVE_CONTROL,
        "lexical",
        "Character count of the previous word (data-derived control).",
        detail="equals the previous unit's character length (verified); matches "
        "no distributional spec",
        make=PrevUnitLength,
    ),
    PredictorEntry(
        "Feature",
        "extractor",
        PredictorStatus.NOT_A_PREDICTOR,
        "infrastructure",
        "The Feature protocol.",
        detail="protocol",
    ),
    PredictorEntry(
        "FeatureNotAvailable",
        "extractor",
        PredictorStatus.NOT_A_PREDICTOR,
        "infrastructure",
        "Raised when a feature needs absent model outputs.",
        detail="exception type",
    ),
    PredictorEntry(
        "InUnit",
        "extractor",
        PredictorStatus.NOT_A_PREDICTOR,
        "meta-feature",
        "Converts a declared logarithmic measurement unit.",
        detail="closed forms and composition contracts in test_measurement.py",
    ),
    PredictorEntry(
        "Lag",
        "extractor",
        PredictorStatus.NOT_A_PREDICTOR,
        "meta-feature",
        "Shifts another feature's values by n positions (spillover).",
        detail="meta-feature over another feature",
    ),
    PredictorEntry(
        "DerivedFeature",
        "extractor",
        PredictorStatus.NOT_A_PREDICTOR,
        "meta-feature",
        "Arithmetic combination of other features.",
        detail="meta-feature over other features",
    ),
    PredictorEntry(
        "Interaction",
        "extractor",
        PredictorStatus.NOT_A_PREDICTOR,
        "meta-feature",
        "Element-wise product of two features.",
        detail="meta-feature over two features",
    ),
    PredictorEntry(
        "PrecomputedFeature",
        "extractor",
        PredictorStatus.NOT_A_PREDICTOR,
        "infrastructure",
        "Reads a named key from stored model outputs.",
        detail="reader for arbitrary stored values; fingerprint the pickle's "
        "column instead",
    ),
]

# ── Feature (backend) surface ────────────────────────────────────────────


def _verified(name, category, blurb, detail, *, equation="", paper=()):
    return PredictorEntry(
        name=name,
        surface="feature",
        status=PredictorStatus.VERIFIED,
        category=category,
        blurb=blurb,
        detail=detail,
        equation=equation,
        paper=paper,
    )


def _infra(name, category, blurb, detail):
    return PredictorEntry(
        name=name,
        surface="feature",
        status=PredictorStatus.NOT_A_PREDICTOR,
        category=category,
        blurb=blurb,
        detail=detail,
    )


_BACKENDS = [
    _infra(
        "FeatureExtractor",
        "infrastructure",
        "Per-unit model-output calculation protocol.",
        "protocol; bound extraction tested in test_resolution.py",
    ),
    _infra(
        "FeatureIndex",
        "infrastructure",
        "Named observation coordinates.",
        "alignment and immutable coordinate tests in test_resolution.py",
    ),
    _infra(
        "FeatureBlock",
        "infrastructure",
        "Indexed values with native axes and contributors.",
        "immutable value and alignment tests in test_resolution.py",
    ),
    _infra(
        "FeatureValues",
        "infrastructure",
        "Resolved blocks and declared provenance.",
        "resolved-input immutability tested in test_resolution.py",
    ),
    _infra(
        "ModelFeature",
        "infrastructure",
        "Model source bound to an extractor.",
        "shared requests, source identity and bounded raw storage in test_resolution.py",
    ),
    _infra(
        "resolve",
        "infrastructure",
        "Resolve a bound quantity without fitting.",
        "numerical composition tests in test_measurement.py",
    ),
    _infra(
        "resolve_features",
        "infrastructure",
        "Resolve a shared feature dependency graph.",
        "source planning and contributor tests in test_resolution.py",
    ),
    _infra(
        "aggregate",
        "infrastructure",
        "Explicit observation reduction.",
        "group means and contributor identities in test_resolution.py",
    ),
    _infra(
        "named",
        "infrastructure",
        "Column alias preserving quantity identity.",
        "alias/source distinction in test_resolution.py",
    ),
    _infra(
        "FeatureMeasurement",
        "infrastructure",
        "Declared feature quantity and unit.",
        "optional measurement description",
    ),
    _infra(
        "describe_feature",
        "infrastructure",
        "Serializes declared feature measurements.",
        "metadata helper; unknown measurements stay unknown",
    ),
    _verified(
        "HuggingFaceLM",
        "surprisal",
        "Causal-LM backend: surprisal, token entropy, Rényi, pooled "
        "first-subword distribution.",
        "surprisal / token-entropy / Rényi reproduced vs an independent "
        "raw-transformers referee on gpt2 (test_real_lm.py)",
    ),
    _verified(
        "ContinuationEntropyLM",
        "anticipation",
        "MC estimator of length-L continuation entropy (full-vocab sampling).",
        "production __call__ vs closed form on the token chain "
        "(test_mc_backends_mock.py, test_torch_reference.py); L=1 unbiased vs "
        "exact token entropy on gpt2 (test_real_lm.py)",
        paper=("giulianelli2024",),
    ),
    _verified(
        "MCWordEntropyLM",
        "anticipation",
        "MC estimator of word entropy with word-boundary stopping.",
        "production __call__ vs enumerated outcome distribution (EOW "
        "marginalization + truncation), Shannon and Rényi-½ "
        "(test_mc_backends_mock.py)",
        paper=("clark2025",),
    ),
    _verified(
        "ForwardMarginalEntropyLM",
        "anticipation",
        "MC forward marginal look-ahead entropy H(c_{i+k}|ctx_i) over depths "
        "k=1..L (entropy of the mixture of N sampled continuations), plus the "
        "free per-step conditional and joint cumulative profiles.",
        "production __call__'s marginal, free, and joint profiles match the exact "
        "matrix-algebra closed form on the token chain "
        "(test_mc_backends_mock.py): marginal=H(mu_k), free=path-averaged "
        "conditional, joint=continuation entropy at each depth",
    ),
    _verified(
        "ForwardWordLookaheadEntropyLM",
        "anticipation",
        "Word-level K-word forward look-ahead via word-boundary MC: joint, "
        "per-word conditional, and Miller-Madow marginal profiles.",
        "production __call__ pinned by its exact by-construction invariants "
        "(joint==cumsum(conditional), monotone joint, Miller-Madow >= plug-in), "
        "seeded determinism, output contract, and a truncation-rule anti-trap "
        "guard (its per-word conditional differs from ``mc_word_entropy`` by "
        "design) on the token chain (test_mc_backends_mock.py); an exact "
        "multi-word value referee is absent",
    ),
    _verified(
        "LogitLensLM",
        "surprisal",
        "Per-layer logit-lens surprisal backend.",
        "final layer ≡ output referee, intermediate differs (test_real_lm.py)",
        paper=("kuribayashi2025",),
    ),
    _verified(
        "ResponsivityLM",
        "responsivity",
        "Decontextualised responsivity distances over sampled alternatives.",
        "exact embedding-space limits incl. the importance-sampling mixed "
        "limit (test_responsivity_mock.py)",
        paper=("giulianelli2024",),
    ),
    _verified(
        "ClozeLM",
        "surprisal",
        "Human cloze probabilities as a next-word predictor (PROVO).",
        "true probabilities fingerprint as contextual surprisal "
        "(test_frequency_and_cloze.py)",
    ),
    _verified(
        "UnigramSurprisalLM",
        "frequency",
        "Context-free unigram surprisal as a FeatureModel.",
        "nats round-trip through Surprisal + bits key, exact vs "
        "unigram_surprisal spec (test_frequency_and_cloze.py)",
    ),
    _verified(
        "UnigramSurprisal",
        "frequency",
        "−log₂ of a word's unigram probability (frequency control).",
        "exact vs unigram_surprisal spec (test_frequency_and_cloze.py)",
    ),
    _verified(
        "LogFrequency",
        "frequency",
        "log₁₀ unigram probability from a frequency source.",
        "fingerprints affine with slope −log₁₀2 (raw log₁₀ probability) "
        "(test_frequency_and_cloze.py)",
    ),
    _verified(
        "SampledUnigramSource",
        "frequency",
        "Sampling-based unigram-probability estimators (Monte Carlo / marginal MC).",
        "mc estimator fingerprints as unigram_surprisal on chain samples "
        "(test_frequency_and_cloze.py); the mmc port anchored vs the authors' "
        "util.py in test_sampled_unigram.py",
        paper=("hopton2026",),
    ),
    _verified(
        "CharBeamLM",
        "surprisal",
        "Character-level surprisal via top-K tokenisation marginalisation.",
        "no closed form on tractable mocks; cross-checked vs HuggingFaceLM in "
        "test_surprisal_methods.py / test_char_beam.py",
        paper=("giulianelli_tok2024",),
    ),
    _verified(
        "WordsProbabilityLM",
        "surprisal",
        "Word-level surprisal with BOW/BOS/EOW correction.",
        "'buggy' mode verified ≡ HuggingFaceLM in test_surprisal_methods.py",
        paper=("pimentel_meister2024",),
    ),
    _verified(
        "InformationValueLM",
        "information value",
        "Distance of an utterance from LM-sampled alternatives.",
        "port verified rtol=1e-6 vs the authors' scorer in test_information_value.py",
        paper=("giulianelli2023",),
    ),
    _verified(
        "InformationValue",
        "information value",
        "Reader for InformationValueLM outputs.",
        "same port anchor as InformationValueLM (test_information_value.py)",
    ),
    _verified(
        "InformationValueScorer",
        "information value",
        "Scores information value from a target and sampled alternatives.",
        "port verified rtol=1e-6 vs the authors' scorer (test_information_value.py)",
    ),
    _verified(
        "ModelBlocksFeature",
        "adaptor",
        "Reads predictor columns from ModelBlocks .evmeasures files.",
        "covered by test_modelblocks*.py",
    ),
    _verified(
        "SimilarityAdjustedSurprisalEmbeddingLM",
        "distance",
        "Similarity-adjusted surprisal via input-embedding (exact) or MC "
        "final-layer cosine similarity.",
        "non-contextual path anchored vs a from-scratch reference at rtol=1e-6 "
        "(test_similarity_surprisal.py); the MC contextual path is covered by "
        "seeded-determinism tests only",
    ),
    _verified(
        "SimilarityAdjustedSurprisalWordLM",
        "distance",
        "Word-level similarity-adjusted surprisal via flair POS match + "
        "Levenshtein orthographic similarity over MC word samples.",
        "seeded determinism, word-starter sampling, and the POS variant "
        "(test_similarity_surprisal.py); no from-scratch value anchor",
    ),
    _verified(
        "SimilarityAdjustedSurprisal",
        "distance",
        "Surprisal discounted by expected cosine similarity to alternatives "
        "(Meister et al. (2024)).",
        "REPLICA evaluate() vs direct numpy at rtol=1e-12 (test_meister2024similarity.py)",
    ),
    _verified(
        "NextWordInformationValue",
        "distance",
        "Expected similarity-weighted surprisal, the information-value companion "
        "to similarity-adjusted surprisal.",
        "REPLICA evaluate() vs direct numpy at rtol=1e-12 (test_meister2024similarity.py)",
    ),
    _infra(
        "UserTable",
        "data source",
        "User-supplied unigram-probability table (a FrequencySource).",
        "data adapter; the predictors built on it (UnigramSurprisal / "
        "LogFrequency) are the verified entries",
    ),
    _infra(
        "WordfreqSource",
        "data source",
        "Frequencies from the wordfreq package.",
        "frequency-source lookup",
    ),
    _infra(
        "KenLMSource",
        "data source",
        "Frequencies from a KenLM model file.",
        "frequency-source lookup",
    ),
    _infra(
        "OpenWebTextSource",
        "data source",
        "Unigram counts over OpenWebText.",
        "frequency-source lookup",
    ),
    _infra(
        "SubtlexSource",
        "data source",
        "Frequencies from a SUBTLEX table.",
        "frequency-source lookup",
    ),
    _infra(
        "PrecedingContextModel",
        "infrastructure",
        "Scores target units using preceding stimuli as context.",
        "context wrapper; the wrapped model supplies the predictor",
    ),
    _infra(
        "FeatureModel",
        "infrastructure",
        "The FeatureModel protocol.",
        "protocol",
    ),
    _infra(
        "FeatureModelOutput",
        "infrastructure",
        "Per-unit model output container.",
        "dataclass",
    ),
    _infra(
        "FrequencySource",
        "infrastructure",
        "The FrequencySource protocol (base-free unigram probability).",
        "protocol; arithmetic certified via UserTable consumers",
    ),
    _infra(
        "TokenAlignment",
        "infrastructure",
        "Word↔token alignment mapping.",
        "dataclass",
    ),
    _infra(
        "align_by_spans",
        "infrastructure",
        "Span-midpoint token alignment helper.",
        "deterministic helper, exercised by every backend test",
    ),
    _infra(
        "aggregate_nested",
        "infrastructure",
        "Guarded token-to-unit aggregation that raises on any crossing token.",
        "deterministic helper, pinned by the alignment and token-table tests",
    ),
    _infra(
        "nested_members",
        "infrastructure",
        "Token indices of every unit, raising on any crossing token.",
        "deterministic helper, pinned by the alignment and token-table tests",
    ),
    _infra(
        "tokenize_spans",
        "infrastructure",
        "Token ids and character spans of a text under a tokenizer.",
        "deterministic helper, pinned by the alignment and token-table tests",
    ),
    _infra(
        "TokenBoundaryError",
        "infrastructure",
        "Raised when a token crosses a unit boundary or a unit has no token.",
        "exception type",
    ),
    _infra(
        "align_from_tokenizer",
        "infrastructure",
        "Tokenize + align helper.",
        "deterministic helper, exercised by the token-chain and real-LM tests",
    ),
    _infra(
        "PrecomputedFeatureModel",
        "infrastructure",
        "Replays stored outputs per stimulus.",
        "replay container; fingerprint the stored columns instead",
    ),
    _infra(
        "MergedFeatureModel",
        "infrastructure",
        "Unions the per-unit values of several models.",
        "combinator; fingerprint its members instead",
    ),
    _infra(
        "HuggingFaceEmbedding",
        "representation",
        "Hidden-state embeddings from a causal LM.",
        "representation extractor (no scalar target)",
    ),
    _infra(
        "HiddenState",
        "representation",
        "Selects a hidden-state layer.",
        "representation extractor (no scalar target)",
    ),
    _infra(
        "ModelBlocksRunner",
        "adaptor",
        "Drives the external ModelBlocks make chain.",
        "external toolchain driver",
    ),
    _infra(
        "surprisal_model",
        "adaptor",
        "Builds the surprisal backend a method name selects.",
        "factory helper",
    ),
    _infra(
        "write_sentitems",
        "adaptor",
        "ModelBlocks sentitems writer.",
        "adaptor helper",
    ),
    _infra(
        "ItemMeasuresTable",
        "adaptor",
        "ModelBlocks measures-table reader.",
        "adaptor table reader",
    ),
    _verified(
        "TunedLensLM",
        "surprisal",
        "Per-layer next-word surprisal via the tuned lens (vs the logit lens).",
        "matches the released tuned-lens outputs (test_kuribayashi2025internal.py)",
        paper=("kuribayashi2025",),
    ),
    _verified(
        "ContinuationEntropyLMBuggy",
        "anticipation",
        "MC continuation-entropy backend reproducing the released measures.py "
        "(reshape reinterpretation and inherited top_k sampling).",
        "tested against the corrected estimate (test_buggy_entropy_predictors.py)",
        paper=("giulianelli2024",),
    ),
    _verified(
        "ContinuationEntropyBuggy",
        "anticipation",
        "Buggy sequence-level continuation entropy (the paper's measures.py "
        "implementation).",
        "tested to land far above the corrected estimate "
        "(test_buggy_entropy_predictors.py)",
        paper=("giulianelli2024",),
    ),
    _verified(
        "TokenEntropySum",
        "entropy",
        "Buggy Pimentel et al. (2023) word entropy: sum of per-subtoken token entropies.",
        "tested to be at least the first-token value and larger on a "
        "multi-subtoken word (test_buggy_entropy_predictors.py)",
        paper=("pimentel2023",),
    ),
    _verified(
        "TokenRenyiEntropySum",
        "entropy",
        "Buggy Pimentel et al. (2023) word Rényi entropy: sum of per-subtoken Rényi values.",
        "tested to be at least the first-token value and larger on a "
        "multi-subtoken word (test_buggy_entropy_predictors.py)",
        paper=("pimentel2023",),
    ),
]

CATALOG: list[PredictorEntry] = [*_EXTRACTORS, *_BACKENDS]

_BY_NAME = {e.name: e for e in CATALOG}


def entry_by_name(name: str) -> PredictorEntry:
    """Look a catalog entry up by export name, raising KeyError if absent.

    Args:
        name: the exported symbol's name, as it appears in ``PredictorEntry.name``.
    """
    return _BY_NAME[name]
