"""Round-trip guard for the cell-id grammar.

``method_cache_tag`` names a compute cell and ``parse_cache_tag`` reads that
name back. Anything that decodes a stored tag (a published predictor table, a
cache sweep) relies on the two agreeing, so every method is round-tripped here
over the parameter values the tags actually encode.
"""

import pytest

from replica.compute import methods as methods_mod
from replica.compute.cache import (
    cache_filename,
    method_cache_tag,
    parse_cache_tag,
    split_cache_filename,
)
from replica.compute.methods import METHODS

# The keyword names method_cache_tag accepts, and values that differ from its
# defaults so a dropped parameter shows up as a mismatch.
PARAMS = {
    "beam_width": 7,
    "n_samples": 512,
    "max_tokens": 30,
    "max_words": 4,
    "alpha": 0.5,
    "seed": 3,
    "add_bos": False,
    "top_p": 0.85,
    "context_scope": "sentence",
    "unit": "characters",
}


# Expected names and complete decoded configurations are independent of the parser.
CASES = {
    "char_beam": ("charbeam_surprisal_K7_chars", dict(beam_width=7, unit="characters")),
    "surprisal": (
        "surprisal_nobos_sent",
        dict(add_bos=False, context_scope="sentence"),
    ),
    "modelblocks_surprisal": ("modelblocks_surprisal", {}),
    "wordsprobability_surprisal": ("wordsprobability_surprisal", {}),
    "wordsprobability_surprisal_buggy": ("wordsprobability_surprisal_buggy", {}),
    "first_token_entropy": ("ft_entropy_nobos", dict(add_bos=False)),
    "first_token_entropy_buggy": ("ft_entropy_buggy_nobos", dict(add_bos=False)),
    "renyi_entropy_buggy": ("renyi05_buggy_nobos", dict(add_bos=False)),
    "token_renyi_entropy": (
        "token_renyi_entropy_a0.5_nobos",
        dict(alpha=0.5, add_bos=False),
    ),
    "restricted_first_token_entropy": (
        "restricted_first_token_entropy_nobos_sent",
        dict(add_bos=False, context_scope="sentence"),
    ),
    "restricted_first_token_renyi_entropy": (
        "restricted_first_token_renyi_entropy_a0.5_nobos",
        dict(alpha=0.5, add_bos=False),
    ),
    "logit_lens_surprisal": (
        "logit_lens_surprisal_sent",
        dict(context_scope="sentence"),
    ),
    "tuned_lens_surprisal": (
        "tuned_lens_surprisal_sent",
        dict(context_scope="sentence"),
    ),
    "unigram_surprisal": ("unigram_surprisal", {}),
    "mmc_unigram_surprisal": ("mmc_unigram_surprisal", {}),
    "continuation_entropy": (
        "cont_entropy_N512_L30_s3_nobos",
        dict(n_samples=512, max_tokens=30, seed=3, add_bos=False),
    ),
    "continuation_entropy_buggy": (
        "cont_entropy_buggy_N512_L30_s3_nobos",
        dict(n_samples=512, max_tokens=30, seed=3, add_bos=False),
    ),
    "forward_marginal_entropy": (
        "fwd_marg_entropy_N512_L30_s3_nobos",
        dict(n_samples=512, max_tokens=30, seed=3, add_bos=False),
    ),
    "forward_word_lookahead_entropy": (
        "fwd_word_lookahead_entropy_N512_M30_K4_s3_nobos",
        dict(n_samples=512, max_tokens=30, max_words=4, seed=3, add_bos=False),
    ),
    "mc_word_entropy": (
        "mcword_entropy_N512_M30_a0.5_s3_nobos",
        dict(n_samples=512, max_tokens=30, alpha=0.5, seed=3, add_bos=False),
    ),
    "responsivity": (
        "responsivity_N512_s3_nobos",
        dict(n_samples=512, seed=3, add_bos=False),
    ),
    "word_information_value": (
        "word_infoval_N512_M30_s3",
        dict(n_samples=512, max_tokens=30, seed=3),
    ),
    "information_value": (
        "infoval_N512_M30_p0.85_s3",
        dict(n_samples=512, max_tokens=30, top_p=0.85, seed=3),
    ),
    "similarity_surprisal_noncontextual": (
        "simadj_surprisal_noncontextual_nobos",
        dict(add_bos=False),
    ),
    "similarity_surprisal_contextual": (
        "simadj_surprisal_contextual_N512_s3_nobos",
        dict(n_samples=512, seed=3, add_bos=False),
    ),
    "similarity_surprisal_word": (
        "simadj_surprisal_word_N512_M30_s3_nobos",
        dict(n_samples=512, max_tokens=30, seed=3, add_bos=False),
    ),
}


def test_every_compute_method_has_an_independent_name_contract():
    assert set(CASES) == set(METHODS)


@pytest.mark.parametrize("method", sorted(CASES))
def test_tag_round_trips(method):
    expected_tag, expected_config = CASES[method]
    assert method_cache_tag(method, **PARAMS) == expected_tag
    parsed_method, config = parse_cache_tag(expected_tag)
    assert parsed_method == method
    assert config == expected_config
    assert method_cache_tag(parsed_method, **config) == expected_tag


@pytest.mark.parametrize("method", sorted(CASES))
def test_filename_contains_the_complete_method_identity(method):
    tag, expected_config = CASES[method]
    filename = cache_filename(
        model_id="model", method=method, dataset="corpus", **PARAMS
    )
    assert filename == f"model_{tag}_corpus.pkl"
    assert split_cache_filename(filename.removesuffix(".pkl"), "corpus") == (
        "model",
        method,
        expected_config,
    )


@pytest.mark.parametrize(
    "model_id,dataset,model_slot",
    [
        ("gpt2", "provo", "gpt2"),
        ("gpt2-medium", "meco_en", "gpt2-medium"),
        ("ai-forever/mGPT", "natural_stories_maze", "ai-forever_mGPT"),
        ("Qwen/Qwen3-0.6B", "meco_l1_w2_en_uk", "Qwen_Qwen3-0.6B"),
    ],
)
def test_filename_separates_model_and_dataset_names(model_id, dataset, model_slot):
    filename = cache_filename(
        model_id=model_id, method="mc_word_entropy", dataset=dataset, **PARAMS
    )
    tag, config = CASES["mc_word_entropy"]
    assert filename == f"{model_slot}_{tag}_{dataset}.pkl"
    assert split_cache_filename(filename.removesuffix(".pkl"), dataset) == (
        model_slot,
        "mc_word_entropy",
        config,
    )


@pytest.mark.parametrize(
    "method", [m for m, (_, config) in CASES.items() if "seed" in config]
)
@pytest.mark.parametrize("seed", [None, 0])
def test_absent_and_zero_seeds_have_distinct_identities(method, seed):
    tag, config = CASES[method]
    expected_config = {k: v for k, v in config.items() if k != "seed"}
    if seed is not None:
        expected_config["seed"] = seed
    expected_tag = tag.replace("_s3", "" if seed is None else "_s0")
    assert method_cache_tag(method, **(PARAMS | {"seed": seed})) == expected_tag
    assert parse_cache_tag(expected_tag) == (method, expected_config)


@pytest.mark.parametrize(
    "method,option",
    [(method, option) for method, (_, config) in CASES.items() for option in config],
)
def test_every_named_option_changes_the_cell_identity(method, option):
    alternative = {
        "beam_width": 8,
        "n_samples": 513,
        "max_tokens": 31,
        "max_words": 5,
        "alpha": 1.0,
        "seed": 4,
        "top_p": 0.9,
        "add_bos": True,
        "context_scope": "stimulus",
        "unit": "words",
    }[option]
    changed = method_cache_tag(method, **(PARAMS | {option: alternative}))
    assert changed != CASES[method][0]


def test_shard_suffix_is_dropped():
    stem = "gpt2_mcword_entropy_N512_M30_a1_s0_celer.shard0_500"
    model, method, config = split_cache_filename(stem, "celer")
    assert (model, method) == ("gpt2", "mc_word_entropy")
    assert config == {"n_samples": 512, "max_tokens": 30, "alpha": 1.0, "seed": 0}


def test_longest_match_wins():
    # The shorter `restricted_first_token_entropy` is a prefix of the Renyi tag's name.
    _, method, config = split_cache_filename(
        "gpt2_restricted_first_token_renyi_entropy_a0.5_provo", "provo"
    )
    assert method == "restricted_first_token_renyi_entropy"
    assert config == {"alpha": 0.5}
    # And `unigram_surprisal` is a suffix of both RB variants.
    for tag, expected in [
        ("wordfreq_unigram_surprisal_provo", "unigram_surprisal"),
        ("gpt2_mmc_unigram_surprisal_provo", "mmc_unigram_surprisal"),
    ]:
        assert split_cache_filename(tag, "provo")[1] == expected


@pytest.mark.parametrize(
    "tag", ["", "surprisal_extra", "charbeam_surprisal_KX", "cont_entropy_N"]
)
def test_unknown_tag_raises(tag):
    with pytest.raises(ValueError):
        parse_cache_tag(tag)


def test_wordsprobability_pair_is_written_and_distinguished():
    assert (
        method_cache_tag("wordsprobability_surprisal") == "wordsprobability_surprisal"
    )
    assert (
        method_cache_tag("wordsprobability_surprisal_buggy")
        == "wordsprobability_surprisal_buggy"
    )
    # The shared prefix must not collapse the two predictor identities.
    for tag, expected in [
        ("gpt2_wordsprobability_surprisal_provo", "wordsprobability_surprisal"),
        (
            "gpt2_wordsprobability_surprisal_buggy_provo",
            "wordsprobability_surprisal_buggy",
        ),
    ]:
        assert split_cache_filename(tag, "provo")[1] == expected


def _builders_reading(marker: str) -> set[str]:
    """Method names whose builder (or a ``_build_`` helper it calls) contains
    ``marker``. Helper bodies stop at the next module-level statement so a
    trailing helper does not swallow ``METHOD_SPECS`` itself."""
    import re
    from pathlib import Path

    source = Path(methods_mod.__file__).read_text(encoding="utf-8")
    helpers = {
        name
        for name in re.findall(r"def (_build_\w+)\(", source)
        if marker
        in re.split(
            r"\ndef |\nMETHOD_SPECS", source.split(f"def {name}(")[1], maxsplit=1
        )[0]
    }
    blocks = re.split(r'\n    "([a-z_0-9]+)": MethodSpec\(', source)
    return {
        name
        for name, body in zip(blocks[1::2], blocks[2::2], strict=True)
        if marker in body or any(h in body for h in helpers)
    }


def test_bos_sensitive_matches_the_builders():
    """``BOS_SENSITIVE`` is the grammar's copy of which builders read add_bos.

    The two live in different modules on purpose: importing ``methods`` from
    ``cache`` would drag every feature model into the naming layer. This scans
    the builder source instead, so a method that starts or stops honouring
    add_bos cannot leave the tag behind.
    """
    from replica.compute.cache import BOS_SENSITIVE

    reads_add_bos = _builders_reading("add_bos=context.add_bos")
    assert reads_add_bos == set(BOS_SENSITIVE), (
        "BOS_SENSITIVE has drifted from the builders: "
        f"only in builders {reads_add_bos - set(BOS_SENSITIVE)!r}, "
        f"only in the set {set(BOS_SENSITIVE) - reads_add_bos!r}"
    )


def test_scope_sensitive_matches_the_builders():
    """Same drift guard as BOS, for the ``context_scope`` axis."""
    from replica.compute.cache import SCOPE_SENSITIVE

    reads_scope = _builders_reading("context_scope=context.context_scope")
    assert reads_scope == set(SCOPE_SENSITIVE), (
        "SCOPE_SENSITIVE has drifted from the builders: "
        f"only in builders {reads_scope - set(SCOPE_SENSITIVE)!r}, "
        f"only in the set {set(SCOPE_SENSITIVE) - reads_scope!r}"
    )


# Pooled MMC estimators have separate cache tags. Check both directions for
# numerators: they are two estimators of one quantity and must never share a
# cell id.
POOLED = "mmc_unigram_surprisal_pool"
POOLED_EE = "mmc_unigram_surprisal_pool_ee"
_FAMILY = {POOLED: "mmc_pps", POOLED_EE: "mmc_pps_ee"}


@pytest.mark.parametrize("method", [POOLED, POOLED_EE])
@pytest.mark.parametrize("seed", [None, 100000])
def test_pooled_mmc_tag_round_trips(method, seed):
    tag = method_cache_tag(method, n_samples=40000000, k_eval=4096, seed=seed)
    expected = f"{_FAMILY[method]}_unigram_surprisal_N40000000_k4096" + (
        f"_s{seed}" if seed is not None else ""
    )
    assert tag == expected
    parsed, config = parse_cache_tag(tag)
    assert parsed == method
    assert config == {"n_samples": 40000000, "k_eval": 4096} | (
        {"seed": seed} if seed is not None else {}
    )
    assert method_cache_tag(parsed, **config) == tag


@pytest.mark.parametrize("method", [POOLED, POOLED_EE])
def test_pooled_mmc_cell_id_splits_back(method):
    """The store holds these cells lm-prefixed, so the publish chain has to
    recover the model slot and re-emit the same id."""
    stem = (
        f"ai-forever_mGPT_{_FAMILY[method]}_unigram_surprisal_N40000000_k4096"
        "_s100000_meco_tr"
    )
    model_slot, parsed, config = split_cache_filename(stem, "meco_tr")
    assert model_slot == "ai-forever_mGPT"
    assert parsed == method
    assert config["k_eval"] == 4096
    assert method_cache_tag(parsed, **config) == stem.removeprefix(
        "ai-forever_mGPT_"
    ).removesuffix("_meco_tr")


def test_the_two_pooled_families_do_not_parse_into_each_other():
    pathwise = method_cache_tag(POOLED, n_samples=8, k_eval=4, seed=1)
    expectation = method_cache_tag(POOLED_EE, n_samples=8, k_eval=4, seed=1)
    assert pathwise != expectation
    assert parse_cache_tag(pathwise)[0] == POOLED
    assert parse_cache_tag(expectation)[0] == POOLED_EE


@pytest.mark.parametrize("method", [POOLED, POOLED_EE])
def test_pooled_mmc_tag_needs_its_budget(method):
    with pytest.raises(ValueError, match="k_eval"):
        method_cache_tag(method, n_samples=40000000)


def test_word_starter_set_is_named_and_does_not_rename_default_cells():
    """The end-of-word token set changes the estimate, so it belongs in the name.

    Which tokens close a word decides when a sampled word stops, so each set
    defines its own predictor. The default set stays unwritten, or every
    published cell would be orphaned by its own rename.
    """
    from replica.compute.cache import (
        cache_filename,
        method_cache_tag,
        parse_cache_tag,
    )

    kw = dict(n_samples=512, max_tokens=20, alpha=1.0, seed=0)
    default = method_cache_tag("mc_word_entropy", **kw)
    gonly = method_cache_tag("mc_word_entropy", word_starters=("Ġ",), **kw)
    assert default == "mcword_entropy_N512_M20_a1_s0"
    assert gonly == default + "_wsg"
    # The explicit default set must name itself the same as leaving it out.
    assert (
        method_cache_tag("mc_word_entropy", word_starters=("Ġ", "Ċ", "ĉ"), **kw)
        == default
    )

    assert parse_cache_tag(gonly)[1]["word_starters"] == ("Ġ",)
    assert "word_starters" not in parse_cache_tag(default)[1]

    stem = dict(model_id="gpt2", method="mc_word_entropy", dataset="dundee")
    assert (
        cache_filename(**stem, **kw) == "gpt2_mcword_entropy_N512_M20_a1_s0_dundee.pkl"
    )
    assert (
        cache_filename(**stem, word_starters=("Ġ",), **kw)
        == "gpt2_mcword_entropy_N512_M20_a1_s0_wsg_dundee.pkl"
    )


@pytest.mark.parametrize(
    "tag",
    [
        "standard",
        "standard_nobos",
        "modelblocks",
        "wordsprobability",
        "wordsprobability_buggy",
        "pim_corrected",
        "pim_buggy",
        "charbeam_K5",
        "simadj_contextual_N50_s0",
        "simadj_noncontextual",
        "simadj_word_N50_M10_s0",
        "fwd_word_lookahead_N512_M30_K5_s0",
    ],
)
def test_obsolete_cache_names_are_rejected(tag):
    with pytest.raises(ValueError):
        parse_cache_tag(tag)
    with pytest.raises(ValueError):
        split_cache_filename(f"gpt2_{tag}_provo", "provo")


@pytest.mark.parametrize("method", sorted(CASES))
def test_token_unit_names_only_the_token_methods(method):
    from replica.compute.cache import TOKEN_SENSITIVE

    plain = method_cache_tag(method, **(PARAMS | {"unit": "words"}))
    tagged = method_cache_tag(method, **(PARAMS | {"unit": "tokens"}))
    if method not in TOKEN_SENSITIVE:
        assert tagged == plain
        return
    # The unit suffix precedes the BOS and scope suffixes.
    cut = min([len(plain), *(plain.find(s) for s in ("_nobos", "_sent") if s in plain)])
    assert tagged == plain[:cut] + "_tokens" + plain[cut:]
    parsed_method, config = parse_cache_tag(tagged)
    assert parsed_method == method and config["unit"] == "tokens"
    assert method_cache_tag(parsed_method, **config) == tagged
    filename = cache_filename(
        model_id="gpt2", method=method, dataset="provo", **(PARAMS | {"unit": "tokens"})
    )
    assert split_cache_filename(filename.removesuffix(".pkl"), "provo") == (
        "gpt2",
        method,
        config,
    )


def test_a_registered_segmentation_names_the_unit_in_the_cell_id(monkeypatch):
    from replica import segmentation
    from replica.core import Span

    class Whole(segmentation.Segmenter):
        def __init__(self):
            super().__init__("ptb-words")

        def segment(self, text):
            return [Span(0, len(text))]

    monkeypatch.setattr(segmentation, "SEGMENTERS", dict(segmentation.SEGMENTERS))
    segmentation.register_segmenter("ptb-words")(Whole)
    name = cache_filename(
        model_id="gpt2", method="surprisal", dataset="provo", unit="ptb-words"
    )
    assert name == "gpt2_surprisal_seg-ptb-words_provo.pkl"
    slot, method, config = split_cache_filename(name[: -len(".pkl")], "provo")
    assert (slot, method, config) == ("gpt2", "surprisal", {"unit": "ptb-words"})
    with pytest.raises(ValueError, match="unknown unit"):
        cache_filename(
            model_id="gpt2", method="surprisal", dataset="provo", unit="morphemes"
        )


def test_renamed_methods_read_under_either_name():
    from replica.compute.cache import canonical_manifest
    from replica.utils.naming import (
        RENAMED_METHODS,
        canonical_key,
        canonical_method,
        canonical_tag,
    )

    assert RENAMED_METHODS["next_word_entropy"] == "restricted_first_token_entropy"
    assert canonical_method("next_word_entropy") == "restricted_first_token_entropy"
    assert canonical_method("surprisal") == "surprisal"
    assert (
        canonical_tag("gpt2_next_word_entropy_provo")
        == "gpt2_restricted_first_token_entropy_provo"
    )
    assert (
        canonical_tag("ai-forever_mGPT_next_word_renyi_entropy_a0.5_meco_de")
        == "ai-forever_mGPT_restricted_first_token_renyi_entropy_a0.5_meco_de"
    )
    assert canonical_tag("gpt2_surprisal_next_word_entropy_x") == (
        "gpt2_surprisal_restricted_first_token_entropy_x"
    )
    assert canonical_key("next_word_entropy") == "restricted_first_token_entropy"
    assert canonical_key("next_word_renyi_entropy_alpha0.5") == (
        "restricted_first_token_renyi_entropy_alpha0.5"
    )
    assert canonical_key("next_word_information_value") == "next_word_information_value"
    manifest = {
        "identity": {
            "method": "next_word_entropy",
            "units": {"next_word_entropy": "bits"},
        },
        "status": "completed",
    }
    assert canonical_manifest(manifest) == {
        "identity": {
            "method": "restricted_first_token_entropy",
            "units": {"restricted_first_token_entropy": "bits"},
        },
        "status": "completed",
    }
    assert canonical_manifest({"status": "x"}) == {"status": "x"}


def test_legacy_spellings_and_manifest_sources_follow_the_rename():
    from replica.compute.cache import canonical_manifest
    from replica.utils.naming import legacy_tags

    assert legacy_tags("gpt2_restricted_first_token_entropy_provo") == {
        "gpt2_next_word_entropy_provo"
    }
    assert legacy_tags("gpt2_surprisal_provo") == set()
    manifest = canonical_manifest(
        {
            "identity": {
                "method": "next_word_renyi_entropy",
                "units": {"next_word_renyi_entropy_alpha0.5": "bits"},
            },
            "derivation": {"source": "gpt2_next_word_renyi_entropy_a0.5_provo.pkl"},
        }
    )
    assert manifest["identity"]["method"] == "restricted_first_token_renyi_entropy"
    assert manifest["identity"]["units"] == {
        "restricted_first_token_renyi_entropy_alpha0.5": "bits"
    }
    assert manifest["derivation"]["source"] == (
        "gpt2_restricted_first_token_renyi_entropy_a0.5_provo.pkl"
    )


def test_decoder_loads_without_the_producers():
    """A store reader resolves a cell's method from ``replica.utils.naming`` alone."""
    import subprocess
    import sys

    probe = (
        "import sys, replica.utils.naming as n\n"
        "assert n.split_cache_filename('gpt2_renyi05_buggy_provo', 'provo')[1]"
        " == 'renyi_entropy_buggy'\n"
        "print(sorted(m for m in sys.modules if m.startswith("
        "('replica.compute', 'replica.predictor_store', 'replica.feature'))))"
    )
    out = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "[]"


@pytest.mark.parametrize(
    ("method", "parameter", "value", "spelling"),
    [
        ("token_renyi_entropy", "alpha", 1e-5, "token_renyi_entropy_a1e-05"),
        (
            "restricted_first_token_renyi_entropy",
            "alpha",
            1e6,
            "restricted_first_token_renyi_entropy_a1e+06",
        ),
        ("mc_word_entropy", "alpha", 1e-5, "mcword_entropy_N64_M10_a1e-05"),
        ("information_value", "top_p", 1e-5, "infoval_N64_M10_p1e-05"),
    ],
)
def test_encoder_scientific_notation_is_readable(method, parameter, value, spelling):
    assert method_cache_tag(method, **{parameter: value}) == spelling
    decoded_method, config = parse_cache_tag(spelling)
    assert decoded_method == method
    assert config[parameter] == value


@pytest.mark.parametrize("suffix", ["nan", "inf", "1e999", "1e", "1e+", "1e-"])
def test_nonfinite_or_malformed_tag_numbers_are_rejected(suffix):
    with pytest.raises(ValueError):
        parse_cache_tag(f"token_renyi_entropy_a{suffix}")
