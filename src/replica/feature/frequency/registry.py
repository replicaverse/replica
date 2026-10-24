"""Single source of truth for unigram / frequency baselines.

Maps a canonical source name to (a) how to build its
:class:`~replica.feature.frequency.FrequencySource` and (b) how its
on-disk predictor pickles are named. The producer
(``replica.compute``, through each source's ``method_key``) and the consumers
(``clark2025wordentropy.wordentropy._UNIGRAM_SUFFIXES``,
study feature builders) all read this one
vocabulary instead of each carrying its own parameter name for "which unigram
baseline".

The pickle prefixes are the stable identifiers of computed predictor pickles, so
changing a tag string makes existing pickles unreachable. The literals are
pinned by ``tests/feature/test_frequency_registry.py``.
"""

from collections.abc import Callable
from dataclasses import dataclass

from ...external import repo_path
from ...utils.naming import safe_lm
from ..core.protocols import FrequencySource

# Canonical value keys every UnigramSurprisalLM writes into FeatureModelOutput;
# exposed as constants so consumers stop re-typing the literals.
BITS_KEY = "unigram_surprisal"  # -log2 p(word), bits
NATS_KEY = "log_p_observed"  # log p(word), nats (surprisal-shaped slot)


@dataclass(frozen=True)
class FreqSource:
    """One frequency-baseline source.

    ``tag`` is the pickle-name piece. For ``lm_prefixed`` sources the on-disk
    prefix is ``f"{safe_lm(model_id)}_{tag}"`` (e.g. ``gpt2_mmc_unigram_surprisal``);
    for the standalone count-based sources (owt, kenlm) ``tag`` is already the
    full bare prefix (e.g. ``owt_unigram_surprisal``).
    """

    name: str
    tag: str
    lm_prefixed: bool
    build: Callable[..., FrequencySource]
    method_key: str | None = None  # replica compute --method, or None (standalone)
    estimator: str | None = None  # set for sampled sources
    default_model_id: str | None = None
    #: Which surface form to look up per unit. ``"table"`` splits joined units
    #: and lets the source normalize each lookup key;
    #: ``"lm_surface"`` keeps the passage's own surface, which is what a
    #: subword LM scores. See :class:`~replica.feature.frequency.model.UnigramSurprisalLM`.
    lookup: str = "table"


# --- lazy builders (no torch/kenlm at module import) -----------------------


def _build_wordfreq(*, language: str = "en", **_):
    from .sources import WordfreqSource

    return WordfreqSource(language=language)


def default_owt_counts() -> str | None:
    """Resolve the OpenWebText unigram-count TSV backing the count-MLE baseline.

    ``$PSYCHO_OWT_COUNTS`` if set, else the copy published in the predictor
    dataset (``owt/unigram_counts.tsv``, cached by ``huggingface_hub``).
    ``None`` when neither is reachable.
    """
    from ...external.hub import fetch_repro_asset

    return fetch_repro_asset("owt/unigram_counts.tsv", env_var="PSYCHO_OWT_COUNTS")


def _build_owt(*, path=None, language: str = "en", **_):
    from .sources import OpenWebTextSource

    if path is None:
        path = default_owt_counts()
        if path is None:
            raise ValueError(
                "the owt source needs its count table: set PSYCHO_OWT_COUNTS to "
                "a word-count TSV, or run where the predictor dataset is reachable"
            )
    return OpenWebTextSource(str(path), language=language)


def _build_kenlm(*, path, language: str = "en", **_):
    from .sources import KenLMSource

    return KenLMSource(str(path), language=language)


def default_wt103_counts() -> str:
    """The unigram csv Pimentel et al. (2023) release with their corpora."""
    return str(
        repo_path("anticipation-on-reading-times", fetch=False)
        / "corpora"
        / "unigrams.csv"
    )


def _build_wt103(*, path=None, language: str = "en", **_):
    from .sources import WikiText103Source

    if path is None:
        path = default_wt103_counts()
    return WikiText103Source(str(path), language=language)


#: Sources whose values come from one file on disk, by source name. A run
#: identifies the file by content, so a changed table cannot pass as a hit.
TABLE_RESOLVERS: dict[str, Callable[[], str | None]] = {
    "wt103": default_wt103_counts,
    "owt": default_owt_counts,
}


def table_path(source: str) -> str | None:
    """The file backing *source*, or ``None`` when it carries no single file.

    Args:
        source: Key of :data:`SOURCES`.
    """
    resolver = TABLE_RESOLVERS.get(source)
    return None if resolver is None else resolver()


def _build_sampled(
    *,
    model_name,
    n_samples=500,
    max_len=50,
    seed=0,
    device="cpu",
    batch_size=64,
    gen_batch_size=128,
    language="en",
    max_continuation_boundaries=None,
    prob_floor=0.0,
    samples=None,
    start_ids=None,
    context_window=None,
    **_,
):
    """Build the sampled marginal Monte Carlo source."""
    from .sampled import SampledUnigramSource

    return SampledUnigramSource(
        samples=samples,
        model_name=model_name,
        n_samples=n_samples,
        max_len=max_len,
        seed=seed,
        device=device,
        batch_size=batch_size,
        gen_batch_size=gen_batch_size,
        language=language,
        max_continuation_boundaries=max_continuation_boundaries,
        prob_floor=prob_floor,
        start_ids=start_ids,
        context_window=context_window,
    )


def _build_pooled(
    *,
    model_name,
    selection=None,
    allocation=None,
    k_eval=None,
    device="cpu",
    batch_size=64,
    language="en",
    samples=None,
    thresholds=None,
    pool_fingerprint=None,
    unit_counts=None,
    context_window=None,
    **_,
):
    """Build the pooled source with explicit selection, allocation and sample budget."""
    from .pooled import PooledUnigramSource

    if selection is None:
        raise ValueError("the mmc_pool source needs a PoolSelection")
    if k_eval is None:
        raise ValueError("the mmc_pool source needs an explicit k_eval")
    if allocation is None:
        raise ValueError("the mmc_pool source needs an explicit allocation")
    return PooledUnigramSource(
        selection=selection,
        allocation=allocation,
        k_eval=k_eval,
        model_name=model_name,
        language=language,
        device=device,
        batch_size=batch_size,
        thresholds=thresholds,
        pool_fingerprint=pool_fingerprint,
        unit_counts=unit_counts,
        samples=samples,
        context_window=context_window,
    )


SOURCES: dict[str, FreqSource] = {
    "wordfreq": FreqSource(
        "wordfreq",
        "unigram_surprisal",
        lm_prefixed=True,
        build=_build_wordfreq,
        method_key="unigram_surprisal",
        default_model_id="wordfreq",
    ),
    # Both estimate unigram probabilities over OpenWebText: owt is the raw
    # whitespace-token MLE; kenlm is the modified-Kneser-Ney unigram tier of a
    # KenLM model trained on the same corpus.
    "owt": FreqSource(
        "owt",
        "owt_unigram_surprisal",
        lm_prefixed=False,
        build=_build_owt,
        method_key="unigram_surprisal",
        default_model_id="owt",
    ),
    "kenlm": FreqSource(
        "kenlm", "kenlm_unigram_surprisal", lm_prefixed=False, build=_build_kenlm
    ),
    # Pimentel et al. (2023)'s WikiText-103 unigram table, scored with their
    # Moses tokenize-and-sum recipe.
    "wt103": FreqSource(
        "wt103",
        "wt103_unigram_surprisal",
        lm_prefixed=False,
        build=_build_wt103,
        method_key="unigram_surprisal",
        default_model_id="wt103",
    ),
    # Marginal Monte Carlo over samples from the LM itself, scored on
    # word-initial surfaces at word-boundary positions.
    "mmc": FreqSource(
        "mmc",
        "mmc_unigram_surprisal",
        lm_prefixed=True,
        build=_build_sampled,
        method_key="mmc_unigram_surprisal",
        estimator="mmc",
        lookup="lm_surface",
    ),
    # The same estimator over one pooled sweep, with the per-word boundary set
    # drawn by priority sampling. Its own tag keeps its pickles distinct.
    "mmc_pool": FreqSource(
        "mmc_pool",
        "mmc_pps_unigram_surprisal",
        lm_prefixed=True,
        build=_build_pooled,
        method_key="mmc_unigram_surprisal_pool",
        estimator="mmc_pps",
        lookup="lm_surface",
    ),
    # The same sweep and selection with the expectation-equivalent numerator:
    # the canonical channel priced against the sampler's horizon plus the exact
    # residual count, so no draw can leave a unit without an estimate. A
    # different estimator of the same quantity, so its own tag family.
    "mmc_pool_ee": FreqSource(
        "mmc_pool_ee",
        "mmc_pps_ee_unigram_surprisal",
        lm_prefixed=True,
        build=_build_pooled,
        method_key="mmc_unigram_surprisal_pool_ee",
        estimator="mmc_pps_ee",
        lookup="lm_surface",
    ),
}


def pickle_prefix(source: str, model_id: str | None = None) -> str:
    """Filename prefix (everything before ``_<dataset>.pkl``) for *source*'s
    predictor pickles.

    Pass ``model_id`` (the full HF id, e.g. ``"ai-forever/mGPT"``) for
    ``lm_prefixed`` sources; the bare count sources ignore it.

    Args:
        source: Key into :data:`SOURCES`, the canonical name of the baseline.
        model_id: Full Hugging Face id of the LM the estimates came from,
            slugified into the prefix. ``None`` falls back to the source's
            ``default_model_id``, and an LM-prefixed source with neither is an
            error.
    """
    spec = SOURCES[source]
    if not spec.lm_prefixed:
        return spec.tag
    mid = model_id or spec.default_model_id
    if mid is None:
        raise ValueError(
            f"frequency source {source!r} is LM-prefixed; pass a model_id "
            "for its pickle prefix."
        )
    return f"{safe_lm(mid)}_{spec.tag}"
