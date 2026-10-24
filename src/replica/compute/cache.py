"""Canonical names for cached language-model feature outputs.

:func:`method_cache_tag` and :func:`cache_filename` build a cell id;
:func:`parse_cache_tag` and :func:`split_cache_filename` read one back. The two
directions are pinned to each other by a round-trip test, so a consumer can
decode a stored tag without re-deriving the grammar.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from replica.utils.naming import (
    SHARD_RE,
    canonical_keys,
    canonical_method,
    canonical_tag,
    parse_cache_tag,
    safe_lm,
    split_cache_filename,
)

__all__ = [
    "BOS_SENSITIVE",
    "SCOPE_SENSITIVE",
    "SHARD_RE",
    "STORY_SENSITIVE",
    "TOKEN_SENSITIVE",
    "UNIT_SENSITIVE",
    "cache_filename",
    "cache_path",
    "canonical_manifest",
    "method_cache_tag",
    "parse_cache_tag",
    "qualified_model_id",
    "resolve_context_scope",
    "split_cache_filename",
]


def _number_tag(value: float) -> str:
    return "1" if value == 1.0 else f"{value:g}"


def _seed_suffix(seed: int | None) -> str:
    return f"_s{seed}" if seed is not None else ""


#: Methods whose model construction consumes ``add_bos``. Running any other
#: method without a beginning-of-sequence token produces identical numbers, so
#: its tag stays unqualified. ``test_cache_tag_roundtrip`` pins this against the
#: builders, so a new method cannot quietly join or leave the set.
BOS_SENSITIVE = frozenset(
    {
        "surprisal",
        "first_token_entropy",
        "first_token_entropy_buggy",
        "token_renyi_entropy",
        "renyi_entropy_buggy",
        "restricted_first_token_entropy",
        "restricted_first_token_renyi_entropy",
        "continuation_entropy",
        "continuation_entropy_buggy",
        "forward_marginal_entropy",
        "forward_word_lookahead_entropy",
        "similarity_surprisal_noncontextual",
        "similarity_surprisal_contextual",
        "similarity_surprisal_word",
        "responsivity",
        "mc_word_entropy",
    }
)


def _bos_suffix(method: str, add_bos: bool) -> str:
    """``_nobos`` for a no-BOS run of a method the flag reaches."""
    return "" if add_bos or method not in BOS_SENSITIVE else "_nobos"


#: Methods whose model construction consumes ``context_scope``. Only these
#: append the ``_sent`` suffix; the roundtrip test keeps the set honest.
SCOPE_SENSITIVE = frozenset(
    {
        "surprisal",
        "restricted_first_token_entropy",
        "logit_lens_surprisal",
        "tuned_lens_surprisal",
    }
)

# Live word predictors for which preceding pages form a causal input prefix.
# Frequency-only and utterance-level methods keep their existing protocols.
STORY_SENSITIVE = BOS_SENSITIVE | frozenset(
    {
        "char_beam",
        "modelblocks_surprisal",
        "wordsprobability_surprisal",
        "wordsprobability_surprisal_buggy",
        "logit_lens_surprisal",
        "tuned_lens_surprisal",
        "word_information_value",
    }
)


def resolve_context_scope(dataset: str, method: str, scope: str = "auto") -> str:
    """Resolve a caller's context scope against dataset and method capabilities.

    Args:
        dataset: Registry key of the corpus.
        method: Predictor method name.
        scope: ``"auto"`` uses the dataset's declared context when supported
            by the method, otherwise ``"stimulus"``. Explicit choices are
            ``"stimulus"``, ``"sentence"`` and ``"story"``.

    Raises:
        ValueError: An unknown scope, ``story`` for a method outside
            :data:`STORY_SENSITIVE`, or ``sentence`` for a method outside
            :data:`SCOPE_SENSITIVE`, whose cell would otherwise share the
            stimulus-scope name while its manifest recorded a scope the model
            never applied.
    """
    if scope == "auto":
        from replica.dataset.catalog.registry import REGISTRY

        scope = getattr(REGISTRY.get(dataset), "default_context_scope", "stimulus")
        if scope == "story" and method not in STORY_SENSITIVE:
            return "stimulus"
        if scope == "sentence" and method not in SCOPE_SENSITIVE:
            return "stimulus"
    if scope not in {"stimulus", "sentence", "story"}:
        raise ValueError(f"unknown context_scope {scope!r}")
    if scope == "story" and method not in STORY_SENSITIVE:
        raise ValueError(f"Story context is not supported for method {method!r}")
    if scope == "sentence" and method not in SCOPE_SENSITIVE:
        raise ValueError(
            f"Sentence context is not supported for method {method!r}; its model "
            "scores the whole stimulus and its cell name carries no scope"
        )
    return scope


#: Methods that can aggregate to a ``characters`` target layer. Only these
#: append the ``_chars`` suffix; any other method with ``unit="characters"``
#: is refused at run time.
UNIT_SENSITIVE = frozenset({"char_beam"})

#: Positional methods whose estimator yields one value per model token, so
#: their cells can be stored per token (``_tokens``) and their word cells are
#: derived from that table. Any other method with ``unit="tokens"`` is refused
#: at run time.
TOKEN_SENSITIVE = frozenset(
    {
        "surprisal",
        "first_token_entropy",
        "token_renyi_entropy",
        "logit_lens_surprisal",
        "tuned_lens_surprisal",
        "continuation_entropy",
        "responsivity",
        "similarity_surprisal_noncontextual",
        "similarity_surprisal_contextual",
    }
)


def _unit_suffix(method: str, unit: str) -> str:
    """``_chars``, ``_tokens`` or ``_seg-<name>`` for a run at another unit than words."""
    if unit == "words":
        return ""
    if unit == "characters":
        return "_chars" if method in UNIT_SENSITIVE else ""
    if unit == "tokens":
        return "_tokens" if method in TOKEN_SENSITIVE else ""
    from replica.segmentation import SEGMENTERS

    if unit in SEGMENTERS:
        return f"_seg-{unit}"
    raise ValueError(f"unknown unit {unit!r}")


def _scope_suffix(method: str, context_scope: str) -> str:
    """Name a supported sentence or story context without aliasing page caches."""
    if context_scope not in {"stimulus", "sentence", "story"}:
        raise ValueError(f"unknown or unresolved context_scope {context_scope!r}")
    if context_scope == "story":
        resolve_context_scope("", method, context_scope)
        return "_story"
    if context_scope == "stimulus" or method not in SCOPE_SENSITIVE:
        return ""
    return "_sent"


def qualified_model_id(model_id: str, language_override: str | None) -> str:
    """Fold a ``--language`` override into the model slot of the cell id.

    ``wordfreq`` with ``--language de`` names the cell ``wordfreq-de``; the bare
    ``wordfreq`` name holds runs whose language came from the dataset. Other
    models ignore the override, so their ids pass through.

    Args:
        model_id: Model or frequency-source id, optionally already carrying a
            ``-<language>`` suffix.
        language_override: Language code the run was asked for; ``None`` leaves
            the id untouched.

    Raises:
        ValueError: When the id already names a different language.
    """
    if language_override is None or not model_id.startswith("wordfreq"):
        return model_id
    suffix = model_id.split("-", 1)[1] if "-" in model_id else None
    if suffix is not None and suffix != language_override:
        raise ValueError(
            f"--language {language_override!r} conflicts with the language "
            f"already named by {model_id!r}"
        )
    return f"wordfreq-{language_override}"


@dataclass(frozen=True)
class _TagParams:
    """The naming parameters a parameterised tag may read."""

    beam_width: int
    n_samples: int
    max_tokens: int
    max_words: int
    alpha: float
    seed: int | None
    top_p: float
    k_eval: int | None
    word_starters: tuple[str, ...] = ()
    """End-of-word token prefixes for Monte Carlo word sampling. Empty means
    the default set; a non-default set is a different predictor, so it is
    named."""


def _word_starter_suffix(word_starters: tuple[str, ...]) -> str:
    """``_wsg`` for the release's leading-space-only set, empty for the default.

    Which tokens end a word decides when a sampled word stops, so it changes
    the estimate rather than only its noise: on matched seeds the two sets give
    Renyi-1/2 means that differ far beyond the seed spread. Two cells that
    differ here must therefore not share a name. ASCII only, since the tag
    reaches file names on the predictor hub.
    """
    if not word_starters or tuple(word_starters) == ("\u0120", "\u010a", "\u0109"):
        return ""
    if tuple(word_starters) == ("\u0120",):
        return "_wsg"
    raise ValueError(f"unnamed word-starter set: {word_starters!r}")


def _pooled_mmc_tag(method: str, family: str) -> Callable[[_TagParams], str]:
    """Pooled MMC names its evaluation budget, so an absent one is an error
    rather than a tag that collides with a different budget's cells. The two
    numerators are two estimators of one quantity, so each has its own family.
    """

    def build(p: _TagParams) -> str:
        if p.k_eval is None:
            raise ValueError(f"{method} needs an explicit k_eval")
        return f"{family}_unigram_surprisal_N{p.n_samples}_k{p.k_eval}{_seed_suffix(p.seed)}"

    return build


#: Methods whose configuration belongs in the tag, one builder each. A table
#: keeps the grammar declarative next to ``constant_tags``, so adding a method
#: costs one entry instead of another branch.
_PARAMETERISED_TAGS: dict[str, Callable[[_TagParams], str]] = {
    "char_beam": lambda p: f"charbeam_surprisal_K{p.beam_width}",
    "token_renyi_entropy": lambda p: f"token_renyi_entropy_a{_number_tag(p.alpha)}",
    "restricted_first_token_renyi_entropy": lambda p: (
        f"restricted_first_token_renyi_entropy_a{_number_tag(p.alpha)}"
    ),
    "continuation_entropy": lambda p: (
        f"cont_entropy_N{p.n_samples}_L{p.max_tokens}{_seed_suffix(p.seed)}"
    ),
    "continuation_entropy_buggy": lambda p: (
        f"cont_entropy_buggy_N{p.n_samples}_L{p.max_tokens}{_seed_suffix(p.seed)}"
    ),
    "forward_marginal_entropy": lambda p: (
        f"fwd_marg_entropy_N{p.n_samples}_L{p.max_tokens}{_seed_suffix(p.seed)}"
    ),
    "forward_word_lookahead_entropy": lambda p: (
        f"fwd_word_lookahead_entropy_N{p.n_samples}_M{p.max_tokens}_K{p.max_words}"
        f"{_seed_suffix(p.seed)}"
    ),
    "mc_word_entropy": lambda p: (
        f"mcword_entropy_N{p.n_samples}_M{p.max_tokens}_a{_number_tag(p.alpha)}"
        f"{_seed_suffix(p.seed)}{_word_starter_suffix(p.word_starters)}"
    ),
    "responsivity": lambda p: f"responsivity_N{p.n_samples}{_seed_suffix(p.seed)}",
    "word_information_value": lambda p: (
        f"word_infoval_N{p.n_samples}_M{p.max_tokens}{_seed_suffix(p.seed)}"
    ),
    "information_value": lambda p: (
        f"infoval_N{p.n_samples}_M{p.max_tokens}_p{_number_tag(p.top_p)}"
        f"{_seed_suffix(p.seed)}"
    ),
    "similarity_surprisal_noncontextual": lambda _p: "simadj_surprisal_noncontextual",
    "similarity_surprisal_contextual": lambda p: (
        f"simadj_surprisal_contextual_N{p.n_samples}{_seed_suffix(p.seed)}"
    ),
    "similarity_surprisal_word": lambda p: (
        f"simadj_surprisal_word_N{p.n_samples}_M{p.max_tokens}{_seed_suffix(p.seed)}"
    ),
    "mmc_unigram_surprisal_pool": _pooled_mmc_tag(
        "mmc_unigram_surprisal_pool", "mmc_pps"
    ),
    "mmc_unigram_surprisal_pool_ee": _pooled_mmc_tag(
        "mmc_unigram_surprisal_pool_ee", "mmc_pps_ee"
    ),
}


def method_cache_tag(
    method: str,
    *,
    beam_width: int = 5,
    n_samples: int = 64,
    max_tokens: int = 10,
    max_words: int = 5,
    alpha: float = 1.0,
    seed: int | None = None,
    add_bos: bool = True,
    top_p: float = 0.9,
    context_scope: str = "stimulus",
    unit: str = "words",
    k_eval: int | None = None,
    word_starters: tuple[str, ...] = (),
    window_policy: str = "reference",
    segmentation: str = "whitespace",
) -> str:
    """Return the stable cache tag for one feature-computation method.

    Args:
        method: Feature-computation method the tag names.
        beam_width: Character-beam width, read by ``char_beam`` alone.
        n_samples: Monte Carlo draws per word.
        max_tokens: Token budget for one sampled continuation.
        max_words: Word budget for one sampled look-ahead continuation.
        alpha: Rényi order; 1.0 is Shannon entropy.
        seed: Monte Carlo seed; ``None`` leaves the tag unseeded.
        add_bos: Whether a beginning-of-sequence token was prepended; a false
            value appends ``_nobos`` for the methods in :data:`BOS_SENSITIVE`.
        top_p: Nucleus-sampling mass for methods that sample whole utterances.
        context_scope: ``stimulus``, ``sentence`` (``_sent`` for methods in
            :data:`SCOPE_SENSITIVE`), or ``story`` (``_story`` for methods in
            :data:`STORY_SENSITIVE`). ``auto`` must be resolved with the dataset
            before naming a cache.
        unit: ``words``, ``characters`` (``_chars`` for the methods in
            :data:`UNIT_SENSITIVE`) or ``tokens`` (``_tokens`` for the methods
            in :data:`TOKEN_SENSITIVE`).
            A registered segmentation name appends ``_seg-<name>``.
        k_eval: Evaluation budget of the pooled marginal Monte Carlo estimator,
            which requires it.
        window_policy: Word information value context policy, reference or cap.
        segmentation: Word information value units, whitespace or layer.
        word_starters: End-of-word token prefixes for Monte Carlo word sampling;
            empty selects the default set, which is not written into the tag.

    Raises:
        ValueError: When *method* has no tag, or an option is outside the values
            the grammar can name.
    """
    constant_tags = {
        "surprisal": "surprisal",
        "modelblocks_surprisal": "modelblocks_surprisal",
        "wordsprobability_surprisal": "wordsprobability_surprisal",
        "wordsprobability_surprisal_buggy": "wordsprobability_surprisal_buggy",
        "first_token_entropy": "ft_entropy",
        "first_token_entropy_buggy": "ft_entropy_buggy",
        "renyi_entropy_buggy": "renyi05_buggy",
        "restricted_first_token_entropy": "restricted_first_token_entropy",
        "logit_lens_surprisal": "logit_lens_surprisal",
        "tuned_lens_surprisal": "tuned_lens_surprisal",
        "unigram_surprisal": "unigram_surprisal",
        "mmc_unigram_surprisal": "mmc_unigram_surprisal",
    }
    if method in constant_tags:
        tag = constant_tags[method]
    elif method in _PARAMETERISED_TAGS:
        tag = _PARAMETERISED_TAGS[method](
            _TagParams(
                beam_width=beam_width,
                n_samples=n_samples,
                max_tokens=max_tokens,
                max_words=max_words,
                alpha=alpha,
                seed=seed,
                top_p=top_p,
                k_eval=k_eval,
                word_starters=word_starters,
            )
        )
    else:
        raise ValueError(f"Unknown method {method!r} for feature computation")
    if window_policy not in {"reference", "cap"} or segmentation not in {
        "whitespace",
        "layer",
    }:
        raise ValueError("Invalid window_policy or segmentation")
    if method == "word_information_value":
        tag += "_cap" if window_policy == "cap" else ""
        tag += "_layer" if segmentation == "layer" else ""
    return (
        tag
        + _unit_suffix(method, unit)
        + _bos_suffix(method, add_bos)
        + _scope_suffix(method, context_scope)
    )


# Inverse of method_cache_tag, one pattern per method. Named groups use the
# keyword names of that function, so a match is directly its config.
def canonical_manifest(manifest: dict) -> dict:
    """*manifest* with its method, value keys and source table under current names.

    Args:
        manifest: A compute manifest, as written or as published.
    """
    identity = manifest.get("identity")
    if not isinstance(identity, dict) or "method" not in identity:
        return manifest
    identity = {**identity, "method": canonical_method(str(identity["method"]))}
    if isinstance(identity.get("units"), dict):
        identity["units"] = canonical_keys(identity["units"])
    out = {**manifest, "identity": identity}
    derivation = manifest.get("derivation")
    if isinstance(derivation, dict) and isinstance(derivation.get("source"), str):
        source = derivation["source"]
        stem, suffix = (
            (source[:-4], ".pkl") if source.endswith(".pkl") else (source, "")
        )
        out["derivation"] = {**derivation, "source": canonical_tag(stem) + suffix}
    return out


def cache_filename(
    *,
    model_id: str,
    method: str,
    dataset: str,
    beam_width: int = 5,
    n_samples: int = 64,
    max_tokens: int = 10,
    max_words: int = 5,
    alpha: float = 1.0,
    seed: int | None = None,
    add_bos: bool = True,
    top_p: float = 0.9,
    context_scope: str = "stimulus",
    unit: str = "words",
    word_starters: tuple[str, ...] = (),
    window_policy: str = "reference",
    segmentation: str = "whitespace",
) -> str:
    """Return the canonical pickle filename for one compute cell.

    Args:
        model_id: Model or frequency-source id; ``safe_lm`` makes it path-safe.
        method: Feature-computation method the cell holds.
        dataset: Registry name of the corpus the cell was computed on.
        beam_width: Character-beam width, read by ``char_beam`` alone.
        n_samples: Monte Carlo draws per word.
        max_tokens: Token budget for one sampled continuation.
        max_words: Word budget for one sampled look-ahead continuation.
        alpha: Rényi order; 1.0 is Shannon entropy.
        seed: Monte Carlo seed; ``None`` leaves the name unseeded.
        add_bos: Whether a beginning-of-sequence token was prepended.
        top_p: Nucleus-sampling mass for methods that sample whole utterances.
        context_scope: Resolved ``stimulus``, ``sentence``, or ``story`` scope.
        unit: ``words``, ``characters`` or ``tokens``.
        window_policy: Word information value context policy, reference or cap.
        segmentation: Word information value units, whitespace or layer.
        word_starters: End-of-word token prefixes for Monte Carlo word sampling;
            empty selects the default set.
    """
    tag = method_cache_tag(
        method,
        beam_width=beam_width,
        n_samples=n_samples,
        max_tokens=max_tokens,
        max_words=max_words,
        alpha=alpha,
        seed=seed,
        add_bos=add_bos,
        top_p=top_p,
        context_scope=context_scope,
        unit=unit,
        word_starters=word_starters,
        window_policy=window_policy,
        segmentation=segmentation,
    )

    return f"{safe_lm(model_id)}_{tag}_{dataset}.pkl"


def cache_path(
    directory: str | Path,
    *,
    model_id: str,
    method: str,
    dataset: str,
    beam_width: int = 5,
    n_samples: int = 64,
    max_tokens: int = 10,
    max_words: int = 5,
    alpha: float = 1.0,
    seed: int | None = None,
    add_bos: bool = True,
    top_p: float = 0.9,
    context_scope: str = "stimulus",
    unit: str = "words",
    word_starters: tuple[str, ...] = (),
    window_policy: str = "reference",
    segmentation: str = "whitespace",
) -> Path:
    """Return a canonical cache path rooted at ``directory``.

    Args:
        directory: Directory the pickle is placed in.
        model_id: Model or frequency-source id; ``safe_lm`` makes it path-safe.
        method: Feature-computation method the cell holds.
        dataset: Registry name of the corpus the cell was computed on.
        beam_width: Character-beam width, read by ``char_beam`` alone.
        n_samples: Monte Carlo draws per word.
        max_tokens: Token budget for one sampled continuation.
        max_words: Word budget for one sampled look-ahead continuation.
        alpha: Rényi order; 1.0 is Shannon entropy.
        seed: Monte Carlo seed; ``None`` leaves the name unseeded.
        add_bos: Whether a beginning-of-sequence token was prepended.
        top_p: Nucleus-sampling mass for methods that sample whole utterances.
        context_scope: Resolved ``stimulus``, ``sentence``, or ``story`` scope.
        unit: ``words``, ``characters`` or ``tokens``.
        window_policy: Word information value context policy, reference or cap.
        segmentation: Word information value units, whitespace or layer.
        word_starters: End-of-word token prefixes for Monte Carlo word sampling;
            empty selects the default set.
    """
    return Path(directory) / cache_filename(
        model_id=model_id,
        method=method,
        dataset=dataset,
        beam_width=beam_width,
        n_samples=n_samples,
        max_tokens=max_tokens,
        max_words=max_words,
        alpha=alpha,
        seed=seed,
        add_bos=add_bos,
        top_p=top_p,
        context_scope=context_scope,
        unit=unit,
        word_starters=word_starters,
        window_policy=window_policy,
        segmentation=segmentation,
    )
