"""Feature-model construction for the cached feature-computation pipeline."""

import logging
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Literal, cast

from replica.feature import (
    ContinuationEntropyLM,
    ContinuationEntropyLMBuggy,
    FeatureModel,
    ForwardMarginalEntropyLM,
    ForwardWordLookaheadEntropyLM,
    HuggingFaceLM,
    InformationValueLM,
    MCWordEntropyLM,
    ResponsivityLM,
    SimilarityAdjustedSurprisalEmbeddingLM,
    SimilarityAdjustedSurprisalWordLM,
    UnigramSurprisalLM,
    surprisal_model,
)
from replica.feature.backends.device import resolve_device
from replica.feature.core.outputs.units import (
    INFORMATION_VALUE_DISTANCES,
    INFORMATION_VALUE_FAMILIES,
)
from replica.feature.core.reproduction import check_reproduction_opt_in
from replica.feature.frequency.registry import BITS_KEY, NATS_KEY, SOURCES

log = logging.getLogger("replica.compute")


@dataclass(frozen=True)
class ComputeContext:
    """Parameters needed to construct and name one feature model."""

    model_id: str
    beam_width: int
    n_samples: int
    max_tokens: int
    alpha: float
    seed: int | None
    add_bos: bool
    dataset_language: str | None
    language_override: str | None
    chunk_size: int | None = None
    word_starters: tuple[str, ...] = ()
    top_p: float = 0.9
    context_scope: str = "stimulus"
    unit: str = "words"
    window_policy: str = "reference"
    segmentation: str = "whitespace"
    model_revision: str | None = None
    store_distribution: bool = False
    model_identity: dict | None = None
    allow_reproduction: bool = False
    start_ids: tuple[int, ...] | None = None
    context_window: int | None = None
    dtype: str | None = None
    device_map: str | None = None


def precision(context: "ComputeContext") -> dict:
    """Loading options a Hugging Face backend takes from the context, only when set.

    Args:
        context: Parameters of the cell.
    """
    options: dict = {}
    if context.dtype is not None:
        options["dtype"] = context.dtype
    if context.device_map is not None:
        options["device_map"] = context.device_map
    return options


@dataclass(frozen=True)
class MethodSpec:
    """Construction and extraction rules for one computation method."""

    description: str
    build_model: Callable[[ComputeContext], object]
    extract_keys: Callable[[ComputeContext], list[str]]
    start_policy: Literal[
        "resolved",
        "sampling",
        "tokenizer_bos",
        "model_bos",
        "decoded_model_bos",
        "space_context",
        "unprefixed_embeddings",
        "table",
    ] = "resolved"
    supports_context_window: bool = False


#: Unigram tables that resolve their own file and report their own language.
_STANDALONE_TABLES = frozenset({"wt103", "owt"})


def _resolve_wordfreq_language(context: ComputeContext) -> str:
    """Resolve a wordfreq language from CLI, model, dataset, then fallback."""
    suffix = context.model_id.split("-", 1)[1] if "-" in context.model_id else None
    if (
        suffix is not None
        and context.dataset_language is not None
        and suffix != context.dataset_language
    ):
        log.warning(
            "[lang] LM language %r differs from dataset language %r, "
            "using LM language. Drop the '-%s' suffix from --lm to "
            "use the dataset's language.",
            suffix,
            context.dataset_language,
            suffix,
        )
    return context.language_override or suffix or context.dataset_language or "en"


def _layer_count(model_id: str) -> int:
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_id)
    return getattr(config, "n_layer", None) or config.num_hidden_layers


def _build_word_infovalue(context: ComputeContext):
    from replica.feature.distance.word_infovalue import WordInfoValueLM

    return WordInfoValueLM(
        model_id=context.model_id,
        k=context.n_samples,
        max_new_tokens=context.max_tokens,
        seed=context.seed or 0,
        window_policy=context.window_policy,
        segmentation=context.segmentation,
    )


def _build_unigram_surprisal(context: ComputeContext):
    if context.model_id in _STANDALONE_TABLES:
        return UnigramSurprisalLM(SOURCES[context.model_id].build())
    if not context.model_id.startswith("wordfreq"):
        raise ValueError(
            "unigram_surprisal requires an LM id starting with 'wordfreq' (or "
            f"one of {sorted(_STANDALONE_TABLES)}); got {context.model_id!r}"
        )
    source = SOURCES["wordfreq"].build(language=_resolve_wordfreq_language(context))
    return UnigramSurprisalLM(source)


def _build_sampled_unigram(context: ComputeContext, *, source: str):
    """Sample and score under the resolved start state recorded in the identity."""
    spec = SOURCES[source]
    return UnigramSurprisalLM(
        spec.build(
            model_name=context.model_id,
            device=resolve_device(),
            batch_size=context.chunk_size or 64,
            language=context.dataset_language or "en",
            start_ids=_resolved_start_ids(context, sampling=True),
            context_window=context.context_window,
        ),
        lookup=spec.lookup,
    )


def _recorded_start_ids(context: ComputeContext) -> tuple[int, ...] | None:
    """Validate a resolved identity prefix, distinguishing absence from empty."""
    from replica.feature.backends.start_state import validate_start_ids

    identity = context.model_identity or {}
    if "start" not in identity:
        return None
    record = identity["start"]
    if not isinstance(record, dict) or not isinstance(record.get("ids"), (list, tuple)):
        raise ValueError("The model identity requires a start.ids sequence")
    ids = record["ids"]
    return validate_start_ids(ids) if ids else ()


def _resolved_start_ids(
    context: ComputeContext, *, sampling: bool = False
) -> list[int]:
    """The identity's start ids, resolved from the tokenizer when absent.

    Raises:
        ValueError: The resolved start state is empty; a sampler cannot draw
            from it and recording a substitute would contradict the identity.
    """
    recorded = _recorded_start_ids(context)
    if recorded is None:
        from transformers import AutoTokenizer

        from replica.feature.backends.start_state import start_ids

        tokenizer = AutoTokenizer.from_pretrained(context.model_id)
        ids = start_ids(tokenizer, context.model_id, override=context.start_ids)
        if not ids and sampling:
            # The sampler owns its convention for a model without a prefix.
            from replica.feature.frequency.sampled import start_prefix

            ids = start_prefix(tokenizer, None)
    else:
        ids = list(recorded)
        if context.start_ids is not None and ids != list(context.start_ids):
            raise ValueError("Requested start_ids disagree with the model identity")
    if not ids:
        raise ValueError(
            f"{context.model_id}: the resolved start state is empty, so there is "
            "no prefix to sample from"
        )
    return ids


def _validate_start_request(method: str, context: ComputeContext) -> None:
    """Reject overrides that a method cannot apply to its actual inputs.

    A method outside ``BOS_SENSITIVE`` ignores ``add_bos``, so its explicit
    token stays enabled under ``--no-bos``; a BOS-sensitive method disables
    the prefix with ``add_bos=False`` and then accepts no explicit token.
    """
    from replica.feature.backends.start_state import validate_start_ids

    from .cache import BOS_SENSITIVE

    prefix_enabled = method not in BOS_SENSITIVE or context.add_bos
    validate_start_ids(context.start_ids, add_bos=prefix_enabled)
    accepts_prefix = METHOD_SPECS[method].start_policy in {"resolved", "sampling"}
    if context.start_ids is not None and not accepts_prefix:
        raise ValueError(
            f"{method} uses a fixed input convention and does not accept start_ids"
        )
    if accepts_prefix:
        recorded = _recorded_start_ids(context)
        if context.start_ids is not None and recorded is not None:
            if recorded != tuple(context.start_ids):
                raise ValueError("Requested start_ids disagree with the model identity")
        if recorded and method in BOS_SENSITIVE and not context.add_bos:
            raise ValueError("The model identity prefix contradicts add_bos=False")


def _validate_context_request(method: str, context: ComputeContext) -> None:
    """Reject context settings that the method or recorded identity cannot apply."""
    from replica.feature.backends.generation import validate_context_window

    validate_context_window(context.context_window)
    identity = context.model_identity or {}
    if (
        context.context_window is not None or "context_window" in identity
    ) and not METHOD_SPECS[method].supports_context_window:
        raise ValueError(f"{method} does not support context_window")
    if "context_window" in identity:
        recorded = validate_context_window(identity["context_window"])
        if recorded is None:
            raise ValueError("The model identity requires a resolved context_window")
        if context.context_window is not None and recorded != context.context_window:
            raise ValueError(
                "Requested context_window disagrees with the model identity"
            )


def _apply_start_record(
    method: str, context: ComputeContext, *, tokenizer_path: str | None = None
) -> ComputeContext:
    """Build from the recorded input; verify an active empty prefix before loading weights."""
    if method == "unigram_surprisal" or METHOD_SPECS[method].start_policy not in {
        "resolved",
        "sampling",
    }:
        return context
    recorded = _recorded_start_ids(context)
    if recorded is None:
        return context
    if recorded:
        return replace(context, start_ids=recorded)
    if METHOD_SPECS[method].start_policy == "sampling":
        raise ValueError(
            "The resolved start state is empty; sampling requires a prefix"
        )
    from .cache import BOS_SENSITIVE

    if method not in BOS_SENSITIVE or context.add_bos:
        from transformers import AutoTokenizer

        from replica.feature.backends.start_state import start_ids

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path or context.model_id)
        if start_ids(tokenizer, context.model_id):
            raise ValueError(
                "The empty model identity prefix disagrees with the tokenizer"
            )
    return context


def _build_buggy_token_entropy(
    context: ComputeContext,
    *,
    renyi: bool,
):
    return HuggingFaceLM(
        context.model_id,
        add_bos=context.add_bos,
        start_ids=context.start_ids,
        context_window=context.context_window,
        **precision(context),
        provide_word_dist=False,
        token_level_entropy=not renyi,
        token_entropy_sum=not renyi,
        token_renyi_alphas=[0.5] if renyi else None,
        token_renyi_entropy_sum=renyi,
        allow_reproduction=context.allow_reproduction,
    )


def _build_lens(context: ComputeContext, *, tuned: bool):
    from replica.feature.surprisal.logit_lens import LogitLensLM
    from replica.feature.surprisal.tuned_lens import TunedLensLM

    cls = TunedLensLM if tuned else LogitLensLM
    return cls(context.model_id, context_scope=context.context_scope)


def _build_corrected_wordsprobability(context: ComputeContext):
    """Apply the resolved prefix while retaining the upstream word correction."""
    ids = _resolved_start_ids(context)
    if len(ids) != 1:
        raise ValueError("wordsprobability requires exactly one start token")
    return surprisal_model(
        context.model_id,
        method="wordsprobability_surprisal",
        start_token_id=ids[0],
    )


def _lens_layer_keys(lens: str, context: ComputeContext) -> list[str]:
    """Per-layer storage keys, with the layer count from the config alone.

    Excludes the mirrored ``log_p_observed``: its convention (leading-space
    words, BOS) disagrees with the ``surprisal`` cell of the same model, so
    it must never be stored under that name.
    """
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(context.model_id)
    n_blocks = getattr(config, "num_hidden_layers", None) or config.n_layer
    return [f"{lens}_log_p_L{layer}" for layer in range(n_blocks + 1)]


METHOD_SPECS: dict[str, MethodSpec] = {
    "surprisal": MethodSpec(
        description="summed subword surprisal",
        build_model=lambda context: surprisal_model(
            context.model_id,
            method="surprisal",
            add_bos=context.add_bos,
            start_ids=context.start_ids,
            context_window=context.context_window,
            **precision(context),
            context_scope=context.context_scope,
        ),
        extract_keys=lambda _context: ["log_p_observed"],
        supports_context_window=True,
    ),
    "char_beam": MethodSpec(
        description="character-beam surprisal",
        build_model=lambda context: surprisal_model(
            context.model_id,
            method="char_beam",
            K=context.beam_width,
            start_ids=context.start_ids,
            context_window=context.context_window,
            **precision(context),
        ),
        extract_keys=lambda _context: ["log_p_observed"],
        supports_context_window=True,
    ),
    "modelblocks_surprisal": MethodSpec(
        description="ModelBlocks-convention word surprisal",
        build_model=lambda context: surprisal_model(
            context.model_id,
            method="modelblocks_surprisal",
        ),
        extract_keys=lambda _context: ["log_p_observed"],
        start_policy="model_bos",
    ),
    "wordsprobability_surprisal": MethodSpec(
        description="corrected word-level surprisal",
        build_model=_build_corrected_wordsprobability,
        extract_keys=lambda _context: ["log_p_observed"],
    ),
    "wordsprobability_surprisal_buggy": MethodSpec(
        description="uncorrected sum-of-subwords word surprisal",
        build_model=lambda context: surprisal_model(
            context.model_id,
            method="wordsprobability_surprisal_buggy",
        ),
        extract_keys=lambda _context: ["log_p_observed"],
        start_policy="tokenizer_bos",
    ),
    "first_token_entropy": MethodSpec(
        description="closed-form first-token Shannon entropy",
        build_model=lambda context: HuggingFaceLM(
            context.model_id,
            add_bos=context.add_bos,
            start_ids=context.start_ids,
            context_window=context.context_window,
            **precision(context),
            provide_word_dist=True,
            token_level_entropy=True,
        ),
        # The per-word predictive distribution is vocabulary-sized; only a
        # cell that asks for it stores it.
        extract_keys=lambda context: (
            ["token_entropy", "next_log_probs"]
            if context.store_distribution
            else ["token_entropy"]
        ),
        supports_context_window=True,
    ),
    "first_token_entropy_buggy": MethodSpec(
        description="paper-compatible summed first-token Shannon entropy",
        build_model=lambda context: _build_buggy_token_entropy(
            context,
            renyi=False,
        ),
        extract_keys=lambda _context: ["token_entropy_sum"],
        supports_context_window=True,
    ),
    "token_renyi_entropy": MethodSpec(
        description="closed-form first-token Rényi entropy",
        build_model=lambda context: HuggingFaceLM(
            context.model_id,
            add_bos=context.add_bos,
            start_ids=context.start_ids,
            context_window=context.context_window,
            **precision(context),
            provide_word_dist=False,
            token_level_entropy=False,
            token_renyi_alphas=[context.alpha],
        ),
        extract_keys=lambda context: [f"token_renyi_entropy_alpha{context.alpha}"],
        supports_context_window=True,
    ),
    "renyi_entropy_buggy": MethodSpec(
        description="paper-compatible summed first-token Rényi entropy",
        build_model=lambda context: _build_buggy_token_entropy(
            context,
            renyi=True,
        ),
        extract_keys=lambda _context: ["token_renyi_entropy_sum"],
        supports_context_window=True,
    ),
    "logit_lens_surprisal": MethodSpec(
        description="per-layer logit-lens surprisal",
        build_model=lambda context: _build_lens(context, tuned=False),
        extract_keys=lambda context: _lens_layer_keys("logit_lens", context),
        start_policy="decoded_model_bos",
    ),
    "tuned_lens_surprisal": MethodSpec(
        description="per-layer tuned-lens surprisal",
        build_model=lambda context: _build_lens(context, tuned=True),
        extract_keys=lambda context: _lens_layer_keys("tuned_lens", context),
        start_policy="decoded_model_bos",
    ),
    "restricted_first_token_entropy": MethodSpec(
        description="closed-form restricted first-token Shannon entropy",
        build_model=lambda context: HuggingFaceLM(
            context.model_id,
            add_bos=context.add_bos,
            start_ids=context.start_ids,
            context_window=context.context_window,
            **precision(context),
            provide_word_dist=False,
            token_level_entropy=False,
            restricted_first_token_entropy=True,
            context_scope=context.context_scope,
        ),
        extract_keys=lambda _context: ["restricted_first_token_entropy"],
        supports_context_window=True,
    ),
    "restricted_first_token_renyi_entropy": MethodSpec(
        description="closed-form restricted first-token Rényi entropy",
        build_model=lambda context: HuggingFaceLM(
            context.model_id,
            add_bos=context.add_bos,
            start_ids=context.start_ids,
            context_window=context.context_window,
            **precision(context),
            provide_word_dist=False,
            token_level_entropy=False,
            restricted_first_token_renyi_alphas=[context.alpha],
        ),
        extract_keys=lambda context: [
            f"restricted_first_token_renyi_entropy_alpha{context.alpha}"
        ],
        supports_context_window=True,
    ),
    "continuation_entropy": MethodSpec(
        description="Monte Carlo continuation entropy",
        build_model=lambda context: ContinuationEntropyLM(
            context.model_id,
            n_samples=context.n_samples,
            max_tokens=context.max_tokens,
            add_bos=context.add_bos,
            start_ids=context.start_ids,
            context_window=context.context_window,
            seed=context.seed,
            model_identity=context.model_identity,
            chunk_size=context.chunk_size,
        ),
        extract_keys=lambda _context: [
            "continuation_entropy",
            "log_p_samples_nats",
        ],
        supports_context_window=True,
    ),
    "continuation_entropy_buggy": MethodSpec(
        description="paper-compatible Monte Carlo continuation entropy",
        build_model=lambda context: ContinuationEntropyLMBuggy(
            context.model_id,
            allow_reproduction=context.allow_reproduction,
            n_samples=context.n_samples,
            max_tokens=context.max_tokens,
            add_bos=context.add_bos,
            start_ids=context.start_ids,
            context_window=context.context_window,
            seed=context.seed,
            model_identity=context.model_identity,
            chunk_size=context.chunk_size,
        ),
        extract_keys=lambda _context: [
            "continuation_entropy_buggy",
            "log_p_samples_nats",
        ],
        supports_context_window=True,
    ),
    "forward_marginal_entropy": MethodSpec(
        description="forward marginal look-ahead entropy",
        build_model=lambda context: ForwardMarginalEntropyLM(
            context.model_id,
            n_samples=context.n_samples,
            max_tokens=context.max_tokens,
            add_bos=context.add_bos,
            start_ids=context.start_ids,
            context_window=context.context_window,
            seed=context.seed,
            model_identity=context.model_identity,
            chunk_size=context.chunk_size,
        ),
        extract_keys=lambda _context: [
            "forward_marginal_entropy",
            "free_step_entropy",
            "joint_entropy",
        ],
        supports_context_window=True,
    ),
    "forward_word_lookahead_entropy": MethodSpec(
        description="word-level forward look-ahead entropy",
        build_model=lambda context: ForwardWordLookaheadEntropyLM(
            context.model_id,
            n_samples=context.n_samples,
            max_words=5,
            max_iter=context.max_tokens,
            add_bos=context.add_bos,
            start_ids=context.start_ids,
            context_window=context.context_window,
            seed=context.seed,
            model_identity=context.model_identity,
            chunk_size=context.chunk_size,
        ),
        extract_keys=lambda _context: [
            "joint_word_entropy",
            "conditional_word_entropy",
            "marginal_word_entropy",
            "marginal_word_entropy_plugin",
            "fwd_word_log_p_step_nats",
            "fwd_word_id_hash",
        ],
        supports_context_window=True,
    ),
    "similarity_surprisal_noncontextual": MethodSpec(
        description="similarity-adjusted surprisal, exact input-embedding cosine",
        build_model=lambda context: SimilarityAdjustedSurprisalEmbeddingLM(
            context.model_id,
            contextual=False,
            add_bos=context.add_bos,
            start_ids=context.start_ids,
            context_window=context.context_window,
            seed=context.seed or 0,
        ),
        extract_keys=lambda _context: [
            "similarity_adjusted_surprisal",
            "next_word_information_value",
            "similarity_adjusted_surprisal_first",
            "next_word_information_value_first",
            "expected_similarity_tokens",
        ],
        supports_context_window=True,
    ),
    "similarity_surprisal_contextual": MethodSpec(
        description="similarity-adjusted surprisal, MC final-layer cosine",
        build_model=lambda context: SimilarityAdjustedSurprisalEmbeddingLM(
            context.model_id,
            contextual=True,
            n_samples=context.n_samples,
            add_bos=context.add_bos,
            start_ids=context.start_ids,
            context_window=context.context_window,
            seed=context.seed or 0,
        ),
        extract_keys=lambda _context: [
            "similarity_adjusted_surprisal",
            "next_word_information_value",
            "similarity_adjusted_surprisal_first",
            "next_word_information_value_first",
            "expected_similarity_tokens",
            "similarity_samples_tokens",
        ],
        supports_context_window=True,
    ),
    "similarity_surprisal_word": MethodSpec(
        description="similarity-adjusted surprisal, MC full-word POS + Levenshtein",
        build_model=lambda context: SimilarityAdjustedSurprisalWordLM(
            context.model_id,
            n_samples=context.n_samples,
            max_iter=context.max_tokens,
            add_bos=context.add_bos,
            start_ids=context.start_ids,
            context_window=context.context_window,
            seed=context.seed or 0,
        ),
        extract_keys=lambda _context: [
            "similarity_adjusted_surprisal_pos",
            "next_word_information_value_pos",
            "similarity_adjusted_surprisal_orth",
            "next_word_information_value_orth",
            "pos_match_samples",
            "orth_similarity_samples",
            "sampled_words",
            "observed_pos",
        ],
        supports_context_window=True,
    ),
    "word_information_value": MethodSpec(
        description="per-word, per-layer information value (Tsipidi et al. 2026)",
        build_model=_build_word_infovalue,
        extract_keys=lambda context: [
            f"infovalue_layer{i}" for i in range(_layer_count(context.model_id) + 1)
        ],
        start_policy="unprefixed_embeddings",
    ),
    "information_value": MethodSpec(
        description="information value against sampled alternative utterances",
        build_model=lambda context: InformationValueLM(
            context.model_id,
            n_samples=context.n_samples,
            top_p=context.top_p,
            max_new_tokens=context.max_tokens,
            seed=context.seed or 0,
            device=resolve_device(),
            sample_batch=context.chunk_size,
        ),
        extract_keys=lambda _context: [
            f"information_value_{family}_{distance}"
            for family in INFORMATION_VALUE_FAMILIES
            for distance in INFORMATION_VALUE_DISTANCES
        ],
        start_policy="space_context",
    ),
    "responsivity": MethodSpec(
        description="decontextualised responsivity distance (unbiased IS)",
        build_model=lambda context: ResponsivityLM(
            context.model_id,
            n_samples=context.n_samples,
            expected_is_correction="unbiased",
            add_bos=context.add_bos,
            start_ids=context.start_ids,
            context_window=context.context_window,
            seed=context.seed or 0,
            device=resolve_device(),
        ),
        extract_keys=lambda _context: [
            "decontextualised_distance",
            "expected_decontextualised_distance",
        ],
        supports_context_window=True,
    ),
    "mc_word_entropy": MethodSpec(
        description="Monte Carlo word entropy",
        build_model=lambda context: MCWordEntropyLM(
            context.model_id,
            n_samples=context.n_samples,
            max_iter=context.max_tokens,
            alpha=context.alpha,
            add_bos=context.add_bos,
            start_ids=context.start_ids,
            context_window=context.context_window,
            seed=context.seed,
            model_identity=context.model_identity,
            chunk_size=context.chunk_size,
            **(
                {"word_starter_chars": context.word_starters}
                if context.word_starters
                else {}
            ),
        ),
        extract_keys=lambda _context: [
            "mc_word_entropy",
            "log_p_samples_nats",
        ],
        supports_context_window=True,
    ),
    "unigram_surprisal": MethodSpec(
        description="frequency-table unigram surprisal",
        build_model=_build_unigram_surprisal,
        extract_keys=lambda _context: [BITS_KEY, NATS_KEY],
        start_policy="table",
    ),
    "mmc_unigram_surprisal": MethodSpec(
        description="marginal Monte Carlo unigram surprisal",
        build_model=lambda context: _build_sampled_unigram(
            context,
            source="mmc",
        ),
        extract_keys=lambda _context: [BITS_KEY, NATS_KEY],
        start_policy="sampling",
        supports_context_window=True,
    ),
}

METHODS = tuple(METHOD_SPECS)


def build_model(method: str, context: ComputeContext):
    """Construct the feature model registered for ``method``.

    Args:
        method: Key of :data:`METHOD_SPECS`.
        context: Model id, sampling budget, and language settings the builder
            reads.

    Raises:
        ValueError: When *method* is not registered.
    """
    try:
        specification = METHOD_SPECS[method]
    except KeyError:
        raise ValueError(f"Unknown method {method!r}. Valid: {METHODS}") from None
    check_reproduction_opt_in(method, allow_reproduction=context.allow_reproduction)
    _validate_start_request(method, context)
    _validate_context_request(method, context)
    context = _apply_start_record(method, context)
    identity = context.model_identity or {}
    if "context_window" in identity:
        context = replace(context, context_window=identity["context_window"])
    if context.context_scope == "story":
        from replica.feature.core.composition.context import PrecedingContextModel

        from .cache import resolve_context_scope

        resolve_context_scope("", method, "story")
        model = specification.build_model(replace(context, context_scope="stimulus"))
        return PrecedingContextModel(cast(FeatureModel, model))
    return specification.build_model(context)


def extract_keys(method: str, context: ComputeContext) -> list[str]:
    """Return the output keys requested for ``method``.

    Args:
        method: Key of :data:`METHOD_SPECS`.
        context: Settings the key list depends on, such as the Rényi order or
            the model whose layer count fixes how many per-layer keys there are.
    """
    return METHOD_SPECS[method].extract_keys(context)
