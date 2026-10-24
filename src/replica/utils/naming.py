"""Cell names: model slots, canonical tags and the tag decoding grammar.

The decoder here is the authority for reading a stored cell name back into
``(model_slot, method, config)``; ``replica.compute.cache`` builds the names and
re-exports the decoder. Nothing here depends on the producers, so readers of
the predictor store can resolve a cell's declared method offline.
"""

import re
from math import isfinite


def safe_lm(model_id: str) -> str:
    """Filesystem-friendly form of a HF model id (e.g. ``ai-forever/mGPT``).

    Args:
        model_id: Hub identifier, or any name whose ``/`` and ``:`` would
            otherwise open a directory level or break a path.

    Returns:
        The id with both characters replaced by underscores, so it can be a
        single path component or a filename tag.
    """
    return model_id.replace("/", "_").replace(":", "_")


def alpha_tag(alpha: float) -> str:
    """Compact filename tag for a Rényi alpha.

    Args:
        alpha: Order of the Rényi entropy the file holds; 1.0 is the Shannon
            case.

    Returns:
        ``"1"`` for alpha 1.0, otherwise ``%g`` with six significant digits, so
        0.5 tags as ``"0.5"`` and no trailing zeros reach the filename.
    """
    return "1" if alpha == 1.0 else f"{alpha:g}"


#: Methods renamed since cells were published: old name to current. The
#: store reads a cell under either name and stages it under the current one.
RENAMED_METHODS: dict[str, str] = {
    "next_word_entropy": "restricted_first_token_entropy",
    "next_word_renyi_entropy": "restricted_first_token_renyi_entropy",
}


def canonical_method(method: str) -> str:
    """*method* under its current name.

    Args:
        method: Compute method name, current or renamed.
    """
    return RENAMED_METHODS.get(method, method)


def canonical_tag(tag: str) -> str:
    """*tag* with a renamed method piece under its current name.

    Args:
        tag: Cell id, current or renamed.
    """
    for old, new in RENAMED_METHODS.items():
        tag = re.sub(rf"(?<=_){re.escape(old)}(?=_)", new, tag)
    return tag


def canonical_key(key: str) -> str:
    """A record's value key under its current name.

    Args:
        key: Value key, current or renamed.
    """
    for old, new in RENAMED_METHODS.items():
        if key == old or key.startswith(old + "_alpha"):
            return new + key[len(old) :]
    return key


def canonical_keys(values: dict) -> dict:
    """Rename a mapping's keys, rejecting aliases that collide.

    Args:
        values: Mapping with original value-key spellings.
    """
    renamed = {}
    originals = {}
    for key, value in values.items():
        canonical = canonical_key(key)
        if canonical in renamed:
            raise ValueError(
                f"keys {originals[canonical]!r} and {key!r} both name {canonical!r}"
            )
        renamed[canonical] = value
        originals[canonical] = key
    return renamed


def legacy_tags(tag: str) -> set[str]:
    """Spellings of *tag* under earlier method names.

    Args:
        tag: Cell id under its current name.
    """
    spellings = set()
    for old, new in RENAMED_METHODS.items():
        legacy = re.sub(rf"(?<=_){re.escape(new)}(?=_)", old, tag)
        if legacy != tag:
            spellings.add(legacy)
    return spellings


_NUMBER = r"\d+(?:\.\d*)?(?:[eE][+-]?\d+)?"
_TAG_PATTERNS: tuple[tuple[str, str], ...] = (
    ("surprisal", r"surprisal"),
    ("modelblocks_surprisal", r"modelblocks_surprisal"),
    ("wordsprobability_surprisal", r"wordsprobability_surprisal"),
    ("wordsprobability_surprisal_buggy", r"wordsprobability_surprisal_buggy"),
    ("first_token_entropy", r"ft_entropy"),
    ("first_token_entropy_buggy", r"ft_entropy_buggy"),
    ("renyi_entropy_buggy", r"renyi05_buggy"),
    ("restricted_first_token_entropy", r"restricted_first_token_entropy"),
    ("logit_lens_surprisal", r"logit_lens_surprisal"),
    ("tuned_lens_surprisal", r"tuned_lens_surprisal"),
    ("unigram_surprisal", r"unigram_surprisal"),
    ("mmc_unigram_surprisal", r"mmc_unigram_surprisal"),
    ("char_beam", r"charbeam_surprisal_K(?P<beam_width>\d+)"),
    ("token_renyi_entropy", rf"token_renyi_entropy_a(?P<alpha>{_NUMBER})"),
    (
        "restricted_first_token_renyi_entropy",
        rf"restricted_first_token_renyi_entropy_a(?P<alpha>{_NUMBER})",
    ),
    (
        "continuation_entropy",
        r"cont_entropy_N(?P<n_samples>\d+)_L(?P<max_tokens>\d+)(?:_s(?P<seed>\d+))?",
    ),
    (
        "continuation_entropy_buggy",
        r"cont_entropy_buggy_N(?P<n_samples>\d+)_L(?P<max_tokens>\d+)"
        r"(?:_s(?P<seed>\d+))?",
    ),
    (
        "forward_marginal_entropy",
        r"fwd_marg_entropy_N(?P<n_samples>\d+)_L(?P<max_tokens>\d+)"
        r"(?:_s(?P<seed>\d+))?",
    ),
    (
        "forward_word_lookahead_entropy",
        r"fwd_word_lookahead_entropy_N(?P<n_samples>\d+)_M(?P<max_tokens>\d+)"
        r"_K(?P<max_words>\d+)(?:_s(?P<seed>\d+))?",
    ),
    (
        "mc_word_entropy",
        r"mcword_entropy_N(?P<n_samples>\d+)_M(?P<max_tokens>\d+)"
        rf"_a(?P<alpha>{_NUMBER})(?:_s(?P<seed>\d+))?(?:_ws(?P<word_starters>g))?",
    ),
    ("responsivity", r"responsivity_N(?P<n_samples>\d+)(?:_s(?P<seed>\d+))?"),
    (
        "word_information_value",
        r"word_infoval_N(?P<n_samples>\d+)_M(?P<max_tokens>\d+)(?:_s(?P<seed>\d+))?"
        r"(?P<cap>_cap)?(?P<layer>_layer)?",
    ),
    (
        "information_value",
        r"infoval_N(?P<n_samples>\d+)_M(?P<max_tokens>\d+)"
        rf"_p(?P<top_p>{_NUMBER})(?:_s(?P<seed>\d+))?",
    ),
    ("similarity_surprisal_noncontextual", r"simadj_surprisal_noncontextual"),
    (
        "similarity_surprisal_contextual",
        r"simadj_surprisal_contextual_N(?P<n_samples>\d+)(?:_s(?P<seed>\d+))?",
    ),
    (
        "similarity_surprisal_word",
        r"simadj_surprisal_word_N(?P<n_samples>\d+)_M(?P<max_tokens>\d+)(?:_s(?P<seed>\d+))?",
    ),
    (
        "mmc_unigram_surprisal_pool",
        r"mmc_pps_unigram_surprisal_N(?P<n_samples>\d+)_k(?P<k_eval>\d+)"
        r"(?:_s(?P<seed>\d+))?",
    ),
    (
        "mmc_unigram_surprisal_pool_ee",
        r"mmc_pps_ee_unigram_surprisal_N(?P<n_samples>\d+)_k(?P<k_eval>\d+)"
        r"(?:_s(?P<seed>\d+))?",
    ),
)
_NOBOS = r"(?P<nobos>_nobos)?"
_SENT = r"(?:(?P<sent>_sent)|(?P<story>_story))?"
_CHARS = r"(?:(?P<chars>_chars)|(?P<tokens>_tokens)|_seg-(?P<seg>[a-z][a-z0-9]*(?:-[a-z0-9]+)*))?"
_ANCHORED = [
    (m, re.compile(p + _CHARS + _NOBOS + _SENT + "$")) for m, p in _TAG_PATTERNS
]
_FLOAT_PARAMS = frozenset({"alpha", "top_p"})
# Sharded cells append this suffix to the tag to name their stimulus slice.
SHARD_RE = re.compile(r"\.shard\d+_\d+$")


def _typed(
    config: dict[str, str],
) -> dict[str, float | int | bool | str | tuple[str, ...]]:
    out: dict[str, float | int | bool | str | tuple[str, ...]] = {}
    for key, value in config.items():
        if value is None:
            continue
        if key == "nobos":
            out["add_bos"] = False
        elif key == "sent":
            out["context_scope"] = "sentence"
        elif key == "story":
            out["context_scope"] = "story"
        elif key == "cap":
            out["window_policy"] = "cap"
        elif key == "layer":
            out["segmentation"] = "layer"
        elif key == "chars":
            out["unit"] = "characters"
        elif key == "tokens":
            out["unit"] = "tokens"
        elif key == "seg":
            out["unit"] = value
        elif key == "word_starters":
            # `_wsg` is the release's leading-space-only set; the default set is
            # not written into the tag, so its absence means the default.
            out["word_starters"] = ("\u0120",)
        elif key in _FLOAT_PARAMS:
            number = float(value)
            if not isfinite(number):
                raise ValueError(f"nonfinite {key} in cache tag: {value!r}")
            out[key] = number
        else:
            out[key] = int(value)
    return out


def parse_cache_tag(
    tag: str,
) -> tuple[str, dict[str, float | int | bool | str | tuple[str, ...]]]:
    """Recover ``(method, config)`` from a tag piece built by
    :func:`replica.compute.cache.method_cache_tag`.

    The config holds only the keywords the method's tag actually encodes, so a
    parameter left at its default and not written into the tag is absent.

    Args:
        tag: The method piece alone, without the model or dataset slots.

    Raises:
        ValueError: When no method's grammar matches *tag* end to end.
    """
    for method, pattern in _ANCHORED:
        match = pattern.fullmatch(tag)
        if match:
            return method, _typed(match.groupdict())
    raise ValueError(f"not a method cache tag: {tag!r}")


def split_cache_filename(
    stem: str, dataset: str
) -> tuple[str, str, dict[str, float | int | bool | str | tuple[str, ...]]]:
    """Split a cell id into ``(model_slot, method, config)``.

    *stem* is a :func:`replica.compute.cache.cache_filename` name without its
    extension, or the
    equivalent tag; a ``.shard<a>_<b>`` suffix is dropped. The model slot is
    returned as it appears in the name (``safe_lm`` form), since the caller
    knows whether it is a model id or a frequency-table name.

    Args:
        stem: The cell id to split.
        dataset: Registry name of the corpus, stripped off the end of *stem*
            where it is present.

    Raises:
        ValueError: When *stem* carries no method tag.
    """
    stem = SHARD_RE.sub("", stem)
    if stem.endswith("_" + dataset):
        stem = stem[: -(len(dataset) + 1)]
    best: tuple[str, re.Match] | None = None
    for method, pattern in _ANCHORED:
        match = pattern.search(stem)
        # Longest end-anchored match wins: `restricted_first_token_renyi_entropy_a0.5` must
        # not be read as the shorter `restricted_first_token_entropy`.
        if match and (best is None or len(match.group(0)) > len(best[1].group(0))):
            best = (method, match)
    if best is None:
        raise ValueError(f"no method cache tag in {stem!r}")
    method, match = best
    return stem[: match.start()].rstrip("_"), method, _typed(match.groupdict())
