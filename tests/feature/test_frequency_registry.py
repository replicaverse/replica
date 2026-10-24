"""Drift guard for the frequency-source registry.

Pins ``frequency.registry.pickle_prefix`` to the on-disk pickle-name literals
the producers and consumers use. A drifting prefix makes already-computed
predictor pickles unreachable.
"""

import pytest

from replica.feature.frequency.registry import (
    BITS_KEY,
    NATS_KEY,
    SOURCES,
    pickle_prefix,
)
from replica.utils.naming import safe_lm


def test_values_keys_unchanged():
    assert BITS_KEY == "unigram_surprisal"
    assert NATS_KEY == "log_p_observed"


def test_registered_sources():
    assert set(SOURCES) == {
        "wordfreq",
        "owt",
        "kenlm",
        "mmc",
        "mmc_pool",
        "mmc_pool_ee",
        "wt103",
    }


def test_matches_compute_lm_features_producer_tags():
    # The feature-cache service combines the safe model id and method tag.
    assert (
        pickle_prefix("wordfreq", "wordfreq")
        == f"{safe_lm('wordfreq')}_unigram_surprisal"
    )
    assert (
        pickle_prefix("wordfreq", "wordfreq-en")
        == f"{safe_lm('wordfreq-en')}_unigram_surprisal"
    )
    assert pickle_prefix("mmc", "gpt2") == "gpt2_mmc_unigram_surprisal"
    assert (
        pickle_prefix("mmc", "ai-forever/mGPT")
        == f"{safe_lm('ai-forever/mGPT')}_mmc_unigram_surprisal"
    )


def test_prefix_families():
    # Bare (standalone count) vs LM-prefixed (composed), must not be conflated.
    assert SOURCES["owt"].lm_prefixed is False
    assert SOURCES["kenlm"].lm_prefixed is False
    assert SOURCES["wordfreq"].lm_prefixed is True
    assert SOURCES["mmc"].lm_prefixed is True


def test_lm_prefixed_source_requires_model_id():
    with pytest.raises(ValueError):
        pickle_prefix("mmc")  # no model_id, no default


def test_method_keys_match_compute_lm_features():
    # The sources exposed as compute_lm_features --method choices.
    method_keys = {s.method_key for s in SOURCES.values() if s.method_key}
    assert method_keys == {
        "unigram_surprisal",
        "mmc_unigram_surprisal",
        "mmc_unigram_surprisal_pool",
        "mmc_unigram_surprisal_pool_ee",
    }


def test_table_sources_use_the_table_lookup():
    # A frequency table is keyed on bare lowercase forms; only the sampled
    # source scores the passage's own surface through a subword LM.
    for name in ("wordfreq", "owt", "kenlm"):
        assert SOURCES[name].lookup == "table"
    assert SOURCES["mmc"].lookup == "lm_surface"
    assert SOURCES["mmc_pool"].lookup == "lm_surface"


def test_mmc_sums_every_boundary_by_default():
    # Capping the continuation passes would make the estimate an approximation;
    # the default must stay exact so a cap is always a deliberate opt-in.
    source = SOURCES["mmc"].build(model_name="gpt2")
    assert source.max_continuation_boundaries is None
    assert source.prob_floor == 0.0
    assert source.n_samples == 500


def test_wordfreq_resolves_serbo_croatian_codes():
    from replica.feature.frequency.sources import WordfreqSource

    assert WordfreqSource(language="sr").language == "sh"
    assert WordfreqSource(language="hr").language == "sh"
    assert WordfreqSource(language="en").language == "en"


def test_the_pooled_source_has_its_own_pickle_family():
    # Distinct tags are what keep the pooled pickles from overwriting the
    # exhaustive ones for the same model.
    assert pickle_prefix("mmc_pool", "ai-forever/mGPT") != pickle_prefix(
        "mmc", "ai-forever/mGPT"
    )
    assert (
        pickle_prefix("mmc_pool", "ai-forever/mGPT")
        == f"{safe_lm('ai-forever/mGPT')}_mmc_pps_unigram_surprisal"
    )
    assert SOURCES["mmc_pool"].estimator == "mmc_pps"
    assert SOURCES["mmc_pool"].lm_prefixed is True


def test_the_two_pooled_numerators_are_two_sources_with_two_pickle_families():
    """The expectation-equivalent numerator estimates the same quantity from
    the same sweep, so only its tag keeps its cells from overwriting the
    pathwise ones."""
    pathwise = SOURCES["mmc_pool"]
    expectation = SOURCES["mmc_pool_ee"]
    assert expectation.build is not None
    assert expectation.estimator == "mmc_pps_ee"
    assert expectation.lookup == pathwise.lookup == "lm_surface"
    assert expectation.lm_prefixed is True
    assert (
        pickle_prefix("mmc_pool_ee", "ai-forever/mGPT")
        == f"{safe_lm('ai-forever/mGPT')}_mmc_pps_ee_unigram_surprisal"
    )
    assert pickle_prefix("mmc_pool_ee", "ai-forever/mGPT") != pickle_prefix(
        "mmc_pool", "ai-forever/mGPT"
    )


def test_the_pooled_builder_forwards_its_selection_and_budget():
    # The builder swallows unknown keywords, so the ones that decide which
    # pool is scored have to be named in its signature.
    sentinel = object()
    allocation = object()
    counts = {" the": 3}
    source = SOURCES["mmc_pool"].build(
        model_name="gpt2",
        selection=sentinel,
        k_eval=1000,
        allocation=allocation,
        unit_counts=counts,
    )
    assert source._selection is sentinel
    assert source._allocation is allocation
    assert source._unit_counts is counts
    assert source.k_eval == 1000
    assert source.prob_floor == 0.0
    assert source.max_continuation_boundaries is None
    # Lazy: no model or tokenizer is loaded until an estimate is asked for.
    assert source._model is None and source._tokenizer is None


def test_the_multiverse_consumer_stays_on_the_bare_mmc_tag():
    from replica.multiverse.run.features import pickle_template_for

    assert pickle_template_for("ai-forever/mGPT", "mmc_unigram_surprisal").endswith(
        "mmc_unigram_surprisal_{ds}.pkl"
    )
    assert pickle_prefix("mmc", "ai-forever/mGPT") + "_{ds}.pkl" == pickle_template_for(
        "ai-forever/mGPT", SOURCES["mmc"].tag
    )


def test_every_standalone_table_computes_through_the_shared_method():
    """A source with a file behind it is a cell of ``unigram_surprisal``."""
    for name in ("owt", "wt103"):
        assert SOURCES[name].method_key == "unigram_surprisal"
        assert SOURCES[name].default_model_id == name
        assert not SOURCES[name].lm_prefixed


def test_the_runner_names_a_standalone_table_cell_by_its_prefix():
    from replica.compute.cache import cache_filename

    for name in ("owt", "wt103"):
        assert (
            cache_filename(model_id=name, method="unigram_surprisal", dataset="provo")
            == f"{pickle_prefix(name)}_provo.pkl"
        )


def test_only_file_backed_sources_resolve_a_table():
    from replica.feature.frequency.registry import TABLE_RESOLVERS

    assert set(TABLE_RESOLVERS) == {"owt", "wt103"}


def test_owt_reads_the_override_table_and_reports_its_own_language(
    tmp_path, monkeypatch
):
    """``$PSYCHO_OWT_COUNTS`` stands in for the published count table."""
    counts = tmp_path / "counts.tsv"
    counts.write_text("# total_tokens=10\nthe\t8\ndog\t2\n")
    monkeypatch.setenv("PSYCHO_OWT_COUNTS", str(counts))

    from replica.feature.frequency.registry import table_path

    assert table_path("owt") == str(counts)
    assert table_path("wordfreq") is None

    source = SOURCES["owt"].build()
    assert source.language == "en"
    assert source.probability("the") == pytest.approx(0.8)


def test_owt_refuses_to_build_without_a_table(monkeypatch):
    monkeypatch.setenv("PSYCHO_OWT_COUNTS", "")
    monkeypatch.setattr(
        "replica.external.hub.fetch_repro_asset", lambda *args, **kwargs: None
    )
    with pytest.raises(ValueError, match="needs its count table"):
        SOURCES["owt"].build()
