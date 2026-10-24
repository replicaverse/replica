"""A parameter that changes a method's computed values must change its cell id.

``chunk_size`` stays out of the cell id: the default sampler keys each sample's
uniforms by its index, so the batch size changes values only through the
batch-dependent forward pass.
"""

import pytest

from replica.compute.cache import (
    BOS_SENSITIVE,
    SCOPE_SENSITIVE,
    TOKEN_SENSITIVE,
    UNIT_SENSITIVE,
    cache_filename,
    qualified_model_id,
)
from replica.compute.methods import METHOD_SPECS

# Per method: the non-BOS parameters of ``cache_filename`` whose change changes
# the computed values. ``model_id`` and ``dataset`` are implicit for every
# method; ``add_bos`` and ``context_scope`` derive from the sensitivity sets.
VALUE_CHANGING: dict[str, set[str]] = {
    "surprisal": set(),
    "modelblocks_surprisal": set(),
    "char_beam": {"beam_width"},
    "wordsprobability_surprisal": set(),
    "wordsprobability_surprisal_buggy": set(),
    "first_token_entropy": set(),
    "first_token_entropy_buggy": set(),
    "renyi_entropy_buggy": set(),
    "restricted_first_token_entropy": set(),
    "logit_lens_surprisal": set(),
    "tuned_lens_surprisal": set(),
    "token_renyi_entropy": {"alpha"},
    "restricted_first_token_renyi_entropy": {"alpha"},
    "unigram_surprisal": set(),
    "mmc_unigram_surprisal": set(),
    "continuation_entropy": {"n_samples", "max_tokens", "seed"},
    "continuation_entropy_buggy": {"n_samples", "max_tokens", "seed"},
    "forward_marginal_entropy": {"n_samples", "max_tokens", "seed"},
    "forward_word_lookahead_entropy": {
        "n_samples",
        "max_tokens",
        "max_words",
        "seed",
    },
    "mc_word_entropy": {"n_samples", "max_tokens", "alpha", "seed"},
    "responsivity": {"n_samples", "seed"},
    "information_value": {"n_samples", "max_tokens", "top_p", "seed"},
    "word_information_value": {"n_samples", "max_tokens", "seed"},
    "similarity_surprisal_noncontextual": set(),
    "similarity_surprisal_contextual": {"n_samples", "seed"},
    "similarity_surprisal_word": {"n_samples", "max_tokens", "seed"},
}

_DEFAULTS = dict(
    model_id="gpt2",
    dataset="provo",
    beam_width=5,
    n_samples=64,
    max_tokens=10,
    max_words=5,
    alpha=1.0,
    seed=0,
    add_bos=True,
    top_p=0.9,
    context_scope="stimulus",
)

_VARIED = dict(
    model_id="gpt2-medium",
    dataset="dundee",
    beam_width=7,
    n_samples=128,
    max_tokens=11,
    max_words=6,
    alpha=0.5,
    seed=3,
    add_bos=False,
    top_p=0.8,
    context_scope="sentence",
)


def test_every_method_declares_its_value_changing_parameters():
    assert set(VALUE_CHANGING) == set(METHOD_SPECS)


@pytest.mark.parametrize("method", sorted(METHOD_SPECS))
def test_value_changing_parameters_reach_the_filename(method):
    params = VALUE_CHANGING[method] | {"model_id", "dataset"}
    if method in BOS_SENSITIVE:
        params = params | {"add_bos"}
    if method in SCOPE_SENSITIVE:
        params = params | {"context_scope"}
    base = cache_filename(method=method, **_DEFAULTS)
    for param in sorted(params):
        varied = {**_DEFAULTS, param: _VARIED[param]}
        assert cache_filename(method=method, **varied) != base, (
            f"{method}: changing {param} does not change the cell id"
        )
    # The unit axis reaches the name exactly for the methods that can emit
    # per-character or per-token cells.
    for unit, methods in (("characters", UNIT_SENSITIVE), ("tokens", TOKEN_SENSITIVE)):
        varied = cache_filename(method=method, **{**_DEFAULTS, "unit": unit})
        assert (varied != base) == (method in methods), (method, unit)


class TestQualifiedModelId:
    """``--language`` is part of a frequency source's identity."""

    def test_language_reaches_the_wordfreq_model_slot(self):
        assert qualified_model_id("wordfreq", "de") == "wordfreq-de"

    def test_agreeing_suffix_is_kept(self):
        assert qualified_model_id("wordfreq-de", "de") == "wordfreq-de"

    def test_conflicting_suffix_raises(self):
        with pytest.raises(ValueError, match="conflicts"):
            qualified_model_id("wordfreq-en", "de")

    def test_non_wordfreq_models_pass_through(self):
        assert qualified_model_id("gpt2", "de") == "gpt2"

    def test_no_override_passes_through(self):
        assert qualified_model_id("wordfreq", None) == "wordfreq"

    def test_qualified_ids_name_distinct_cells(self):
        names = {
            cache_filename(
                method="unigram_surprisal",
                **{**_DEFAULTS, "model_id": model_id},
            )
            for model_id in ("wordfreq", "wordfreq-de", "wordfreq-en")
        }
        assert len(names) == 3
