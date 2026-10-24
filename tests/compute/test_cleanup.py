"""Tests for ``replica.compute.cleanup``: pickle audit + manifest planning."""

import pickle
from types import SimpleNamespace

import pytest

from replica.compute.cleanup import (
    Cell,
    audit_pickles,
    build_cleanup_plan,
    is_valid_pickle,
    parse_manifest,
    pickle_path_for,
    write_manifest,
)

# ── 1. Pickle integrity ──────────────────────────────────────────────────


class TestIsValidPickle:
    def test_recognises_real_pickle(self, tmp_path):
        p = tmp_path / "real.pkl"
        with open(p, "wb") as f:
            pickle.dump({"hello": "world"}, f)
        assert is_valid_pickle(p)

    def test_rejects_missing_file(self, tmp_path):
        assert not is_valid_pickle(tmp_path / "ghost.pkl")

    def test_rejects_empty_dict_pickle_when_deep(self, tmp_path):
        """``pickle.dumps({})`` is 5 bytes, below ``_MIN_NONEMPTY_BYTES`` (6),
        so the shallow check rejects it on the size guard. ``deep=True`` would
        also reject it because the dict is empty."""
        p = tmp_path / "empty_dict.pkl"
        with open(p, "wb") as f:
            pickle.dump({}, f)
        # Shallow: size below threshold, rejected before the STOP-byte check.
        assert is_valid_pickle(p) is False  # size < _MIN_NONEMPTY_BYTES (5 < 6)
        # Even with deep, it's rejected because the dict is empty.
        assert is_valid_pickle(p, deep=True) is False

    def test_rejects_truncated(self, tmp_path):
        """Truncate a real pickle mid-stream, last byte will not be 0x2e."""
        p = tmp_path / "truncated.pkl"
        with open(p, "wb") as f:
            pickle.dump({"a": list(range(1000))}, f)  # produces a multi-byte pickle
        full = p.read_bytes()
        # Drop the last 5 bytes (definitely removes the STOP opcode).
        p.write_bytes(full[:-5])
        assert not is_valid_pickle(p)

    def test_rejects_size_below_threshold(self, tmp_path):
        p = tmp_path / "tiny.pkl"
        p.write_bytes(b".")  # one byte, last byte is STOP, but size < threshold
        assert not is_valid_pickle(p)

    def test_accepts_non_dict_root_when_deep(self, tmp_path):
        """``deep=True`` accepts non-dict/list pickles when they're non-empty
        in a meaningful sense, primitives are accepted as long as they parsed."""
        p = tmp_path / "scalar.pkl"
        with open(p, "wb") as f:
            pickle.dump(3.14, f)
        assert is_valid_pickle(p, deep=True)

    def test_deep_rejects_garbage_with_stop_byte(self, tmp_path):
        """File ends with 0x2e but is otherwise garbage, shallow accepts,
        deep rejects via pickle.UnpicklingError."""
        p = tmp_path / "bogus.pkl"
        p.write_bytes(b"\x00" * 200 + b".")
        assert is_valid_pickle(p) is True  # shallow false positive
        assert is_valid_pickle(p, deep=True) is False  # deep catches it


# ── 2. audit_pickles ─────────────────────────────────────────────────────


class TestAuditPickles:
    def test_empty_dir(self, tmp_path):
        report = audit_pickles(tmp_path)
        assert report.total == 0
        assert report.ok == ()

    def test_mixed_dir(self, tmp_path):
        # Valid pickle
        with open(tmp_path / "good.pkl", "wb") as f:
            pickle.dump({"k": [1, 2, 3]}, f)
        # Empty (sub-threshold)
        (tmp_path / "empty.pkl").write_bytes(b"\x80\x04")  # 2 bytes
        # Truncated (ends in non-STOP byte)
        with open(tmp_path / "trunc.pkl", "wb") as f:
            pickle.dump({"big": list(range(1000))}, f)
        bs = (tmp_path / "trunc.pkl").read_bytes()
        (tmp_path / "trunc.pkl").write_bytes(bs[:-3])

        report = audit_pickles(tmp_path)

        def names(paths):
            return sorted(p.name for p in paths)

        assert names(report.ok) == ["good.pkl"]
        assert names(report.empty) == ["empty.pkl"]
        assert names(report.truncated) == ["trunc.pkl"]
        assert report.total == 3

    def test_pattern_filter(self, tmp_path):
        with open(tmp_path / "want.pkl", "wb") as f:
            pickle.dump({"a": 1}, f)
        with open(tmp_path / "skip.dat", "wb") as f:
            pickle.dump({"a": 1}, f)
        report = audit_pickles(tmp_path, pattern="*.pkl")
        assert len(report.ok) == 1
        assert report.ok[0].name == "want.pkl"


# ── 3. parse_manifest ────────────────────────────────────────────────────


class TestParseManifest:
    def test_basic_line(self):
        lines = ["tier=2\tdataset=provo\tlm=gpt2\tmethod=surprisal\textra="]
        cells = list(parse_manifest(lines))
        assert len(cells) == 1
        c = cells[0]
        assert c.tier == "2" and c.dataset == "provo"
        assert c.lm == "gpt2" and c.method == "surprisal"
        assert c.extra == ""

    def test_extras_preserved_verbatim(self):
        line = (
            "tier=5\tdataset=dundee\tlm=ai-forever/mGPT\t"
            "method=continuation_entropy\textra=--n-samples 64 --max-tokens 10 --seed 0"
        )
        c = next(parse_manifest([line]))
        assert c.extra == "--n-samples 64 --max-tokens 10 --seed 0"

    def test_blank_lines_skipped(self):
        lines = ["", "tier=2\tdataset=provo\tlm=gpt2\tmethod=surprisal\textra=", ""]
        assert len(list(parse_manifest(lines))) == 1

    def test_missing_required_field_raises(self):
        with pytest.raises(ValueError, match="missing required fields"):
            list(parse_manifest(["tier=2\tdataset=provo\tlm=gpt2\textra="]))

    def test_extra_field_defaults_to_empty(self):
        line = "tier=2\tdataset=provo\tlm=gpt2\tmethod=surprisal"
        c = next(parse_manifest([line]))
        assert c.extra == ""


# ── 4. pickle_path_for ───────────────────────────────────────────────────


class TestPicklePathFor:
    """Each branch of the tag-derivation must round-trip to the exact filename
    that ``compute_lm_features.py`` writes. Tags are checked against the
    method registry, so any new method must define a canonical cache name."""

    @pytest.fixture
    def compute_dir(self, tmp_path):
        return tmp_path

    def test_standard(self, compute_dir):
        c = Cell(tier="2", dataset="provo", lm="gpt2", method="surprisal")
        assert pickle_path_for(c, compute_dir).name == "gpt2_surprisal_provo.pkl"

    def test_mgpt_safe_name(self, compute_dir):
        c = Cell(tier="2", dataset="provo", lm="ai-forever/mGPT", method="surprisal")
        assert (
            pickle_path_for(c, compute_dir).name
            == "ai-forever_mGPT_surprisal_provo.pkl"
        )

    def test_word_starter_set_reaches_the_name(self, compute_dir):
        """A g-only cell must resolve to the file the compute CLI writes.

        The path is what a finished cell is verified by and what the cleanup
        tooling reads, so a name that drops the suffix reports a finished cell
        as missing.
        """
        base = "--alpha 1.0 --n-samples 512 --max-tokens 20 --seed 0"
        default = Cell(
            tier="4",
            dataset="dundee",
            lm="gpt2",
            method="mc_word_entropy",
            extra=base,
        )
        g_only = Cell(
            tier="4",
            dataset="dundee",
            lm="gpt2",
            method="mc_word_entropy",
            extra=base + " --word-starters g-only",
        )
        assert (
            pickle_path_for(default, compute_dir).name
            == "gpt2_mcword_entropy_N512_M20_a1_s0_dundee.pkl"
        )
        assert (
            pickle_path_for(g_only, compute_dir).name
            == "gpt2_mcword_entropy_N512_M20_a1_s0_wsg_dundee.pkl"
        )

    def test_char_beam_with_K(self, compute_dir):
        c = Cell(
            tier="4", dataset="provo", lm="gpt2", method="char_beam", extra="--K 5"
        )
        assert (
            pickle_path_for(c, compute_dir).name
            == "gpt2_charbeam_surprisal_K5_provo.pkl"
        )

    def test_char_beam_default_K(self, compute_dir):
        c = Cell(tier="4", dataset="provo", lm="gpt2", method="char_beam", extra="")
        assert (
            pickle_path_for(c, compute_dir).name
            == "gpt2_charbeam_surprisal_K5_provo.pkl"
        )

    def test_wordsprobability(self, compute_dir):
        c = Cell(
            tier="5", dataset="provo", lm="gpt2", method="wordsprobability_surprisal"
        )
        assert (
            pickle_path_for(c, compute_dir).name
            == "gpt2_wordsprobability_surprisal_provo.pkl"
        )

    def test_wordsprobability_buggy(self, compute_dir):
        c = Cell(
            tier="5",
            dataset="provo",
            lm="gpt2",
            method="wordsprobability_surprisal_buggy",
        )
        assert (
            pickle_path_for(c, compute_dir).name
            == "gpt2_wordsprobability_surprisal_buggy_provo.pkl"
        )

    def test_ft_entropy(self, compute_dir):
        c = Cell(tier="3", dataset="provo", lm="gpt2", method="first_token_entropy")
        assert pickle_path_for(c, compute_dir).name == "gpt2_ft_entropy_provo.pkl"

    def test_token_renyi_alpha_0_5(self, compute_dir):
        c = Cell(
            tier="3",
            dataset="provo",
            lm="gpt2",
            method="token_renyi_entropy",
            extra="--alpha 0.5",
        )
        assert (
            pickle_path_for(c, compute_dir).name
            == "gpt2_token_renyi_entropy_a0.5_provo.pkl"
        )

    def test_token_renyi_alpha_1_renders_as_1(self, compute_dir):
        c = Cell(
            tier="3",
            dataset="provo",
            lm="gpt2",
            method="token_renyi_entropy",
            extra="--alpha 1.0",
        )
        assert (
            pickle_path_for(c, compute_dir).name
            == "gpt2_token_renyi_entropy_a1_provo.pkl"
        )

    def test_continuation_entropy(self, compute_dir):
        c = Cell(
            tier="5",
            dataset="provo",
            lm="gpt2",
            method="continuation_entropy",
            extra="--n-samples 64 --max-tokens 10 --seed 0",
        )
        assert (
            pickle_path_for(c, compute_dir).name
            == "gpt2_cont_entropy_N64_L10_s0_provo.pkl"
        )

    def test_mc_word_entropy(self, compute_dir):
        c = Cell(
            tier="5",
            dataset="provo",
            lm="gpt2",
            method="mc_word_entropy",
            extra="--n-samples 64 --max-tokens 30 --alpha 0.5 --seed 0",
        )
        assert (
            pickle_path_for(c, compute_dir).name
            == "gpt2_mcword_entropy_N64_M30_a0.5_s0_provo.pkl"
        )

    def test_unigram_surprisal(self, compute_dir):
        # An explicit override always reaches the model slot, so a cell run
        # with ``--language`` never shares a name with the bare-wordfreq cell
        # whose language came from the dataset.
        c = Cell(
            tier="1",
            dataset="provo",
            lm="wordfreq",
            method="unigram_surprisal",
            extra="--language en",
        )
        assert (
            pickle_path_for(c, compute_dir).name
            == "wordfreq-en_unigram_surprisal_provo.pkl"
        )
        bare = Cell(
            tier="1",
            dataset="provo",
            lm="wordfreq",
            method="unigram_surprisal",
            extra="",
        )
        assert (
            pickle_path_for(bare, compute_dir).name
            == "wordfreq_unigram_surprisal_provo.pkl"
        )

    def test_unknown_method_raises(self, compute_dir):
        c = Cell(tier="?", dataset="provo", lm="gpt2", method="bogus")
        with pytest.raises(ValueError, match="Unknown method"):
            pickle_path_for(c, compute_dir)

    @pytest.mark.parametrize(
        ("dataset", "lm", "method", "extra", "expected"),
        [
            # SB-SAT word predictors default to story context, so their files
            # carry ``_story``; a frequency cell has no context and does not.
            ("sbsat", "gpt2", "surprisal", "", "gpt2_surprisal_story_sbsat.pkl"),
            (
                "sbsat",
                "ai-forever/mGPT",
                "char_beam",
                "--K 5",
                "ai-forever_mGPT_charbeam_surprisal_K5_story_sbsat.pkl",
            ),
            (
                "sbsat",
                "gpt2",
                "continuation_entropy",
                "--n-samples 64 --max-tokens 10 --seed 0",
                "gpt2_cont_entropy_N64_L10_s0_story_sbsat.pkl",
            ),
            (
                "sbsat",
                "wordfreq",
                "unigram_surprisal",
                "--language en",
                "wordfreq-en_unigram_surprisal_sbsat.pkl",
            ),
            (
                "sbsat",
                "gpt2",
                "surprisal",
                "--context-scope stimulus",
                "gpt2_surprisal_sbsat.pkl",
            ),
            (
                "provo",
                "gpt2",
                "surprisal",
                "--context-scope auto",
                "gpt2_surprisal_provo.pkl",
            ),
            (
                "provo",
                "gpt2",
                "word_information_value",
                "--n-samples 5 --max-tokens 3 --seed 0 --window-policy cap "
                "--segmentation layer",
                "gpt2_word_infoval_N5_M3_s0_cap_layer_provo.pkl",
            ),
        ],
    )
    def test_name_is_the_one_compute_writes(
        self, compute_dir, dataset, lm, method, extra, expected
    ):
        """Scope, window policy and segmentation resolve as the compute CLI
        resolves them, so the plan finds a complete cell where compute wrote it."""
        c = Cell(tier="?", dataset=dataset, lm=lm, method=method, extra=extra)
        assert pickle_path_for(c, compute_dir).name == expected

    @pytest.mark.parametrize("extra", ["--K five", "--no-such-flag 1"])
    def test_extras_the_compute_cli_rejects_raise(self, compute_dir, extra):
        c = Cell(tier="?", dataset="provo", lm="gpt2", method="char_beam", extra=extra)
        with pytest.raises(ValueError):
            pickle_path_for(c, compute_dir)


# ── 5. build_cleanup_plan ────────────────────────────────────────────────


class TestBuildCleanupPlan:
    def _put_valid(self, path):
        with open(path, "wb") as f:
            pickle.dump({"stim_1": [1, 2, 3]}, f)

    def test_all_present(self, tmp_path):
        cells = [
            Cell(tier="2", dataset="provo", lm="gpt2", method="surprisal"),
            Cell(tier="2", dataset="dundee", lm="gpt2", method="surprisal"),
        ]
        for c in cells:
            self._put_valid(pickle_path_for(c, tmp_path))
        plan = build_cleanup_plan(cells, tmp_path)
        assert plan.need_redo == ()
        assert plan.ok == tuple(cells)
        assert plan.total == 2

    def test_partial(self, tmp_path):
        c1 = Cell(tier="2", dataset="provo", lm="gpt2", method="surprisal")
        c2 = Cell(tier="2", dataset="dundee", lm="gpt2", method="surprisal")
        c3 = Cell(tier="2", dataset="geco", lm="gpt2", method="surprisal")
        self._put_valid(pickle_path_for(c1, tmp_path))
        # c2: empty pickle on disk
        pickle_path_for(c2, tmp_path).write_bytes(b"")
        # c3: missing entirely
        plan = build_cleanup_plan([c1, c2, c3], tmp_path)
        assert plan.ok == (c1,)
        assert plan.need_redo == (c2, c3)
        assert plan.reasons == ("empty", "missing")

    def test_truncated_classified(self, tmp_path):
        c = Cell(tier="2", dataset="provo", lm="gpt2", method="surprisal")
        p = pickle_path_for(c, tmp_path)
        with open(p, "wb") as f:
            pickle.dump({"k": list(range(1000))}, f)
        # Chop off the STOP opcode
        p.write_bytes(p.read_bytes()[:-3])
        plan = build_cleanup_plan([c], tmp_path)
        assert plan.reasons == ("truncated",)


# ── 6. write_manifest round-trip ─────────────────────────────────────────


class TestWriteManifest:
    def test_round_trip(self, tmp_path):
        cells = [
            Cell(tier="2", dataset="provo", lm="gpt2", method="surprisal"),
            Cell(
                tier="5",
                dataset="dundee",
                lm="ai-forever/mGPT",
                method="continuation_entropy",
                extra="--n-samples 64 --max-tokens 10 --seed 0",
            ),
        ]
        path = tmp_path / "m.tsv"
        n = write_manifest(cells, path)
        assert n == 2
        round_tripped = list(parse_manifest(path.read_text().splitlines()))
        assert round_tripped == cells


# cleanup ↔ compute_lm_features tag round-trip. Both sides independently encode
# the tag convention; this asserts they produce identical filenames per method.


class TestCleanupComputeSurprisalRoundTrip:
    @pytest.fixture(scope="class")
    @classmethod
    def cs_mod(cls):
        """Use the producer's actual method registry."""
        from replica.compute import methods as compute_methods
        from replica.compute import runner as compute

        return SimpleNamespace(
            ComputeContext=compute_methods.ComputeContext,
            METHOD_SPECS=compute_methods.METHOD_SPECS,
            build_model=compute_methods.build_model,
            cache_stem=compute.cache_stem,
            methods_module=compute_methods,
        )

    @pytest.mark.parametrize(
        "method, lm, ctx_overrides, cell_extra",
        [
            ("surprisal", "gpt2", {}, ""),
            ("modelblocks_surprisal", "gpt2", {}, ""),
            ("char_beam", "gpt2", {"beam_width": 5}, "--K 5"),
            ("char_beam", "gpt2", {"beam_width": 3}, "--K 3"),
            ("wordsprobability_surprisal", "gpt2", {}, ""),
            ("wordsprobability_surprisal_buggy", "gpt2", {}, ""),
            ("first_token_entropy", "gpt2", {}, ""),
            ("first_token_entropy_buggy", "gpt2", {}, ""),
            ("token_renyi_entropy", "gpt2", {"alpha": 0.5}, "--alpha 0.5"),
            ("token_renyi_entropy", "gpt2", {"alpha": 1.0}, "--alpha 1.0"),
            ("renyi_entropy_buggy", "gpt2", {}, ""),
            ("restricted_first_token_entropy", "gpt2", {}, ""),
            (
                "restricted_first_token_renyi_entropy",
                "gpt2",
                {"alpha": 0.5},
                "--alpha 0.5",
            ),
            (
                "restricted_first_token_renyi_entropy",
                "gpt2",
                {"alpha": 1.0},
                "--alpha 1.0",
            ),
            (
                "continuation_entropy",
                "gpt2",
                {"n_samples": 64, "max_tokens": 10, "seed": 0},
                "--n-samples 64 --max-tokens 10 --seed 0",
            ),
            (
                "continuation_entropy_buggy",
                "gpt2",
                {"n_samples": 64, "max_tokens": 10, "seed": 0},
                "--n-samples 64 --max-tokens 10 --seed 0",
            ),
            (
                "mc_word_entropy",
                "gpt2",
                {"n_samples": 64, "max_tokens": 30, "alpha": 0.5, "seed": 0},
                "--n-samples 64 --max-tokens 30 --alpha 0.5 --seed 0",
            ),
            (
                "mc_word_entropy",
                "gpt2",
                {"n_samples": 64, "max_tokens": 30, "alpha": 1.0, "seed": 0},
                "--n-samples 64 --max-tokens 30 --alpha 1.0 --seed 0",
            ),
            (
                "forward_marginal_entropy",
                "gpt2",
                {"n_samples": 64, "max_tokens": 10, "seed": 0},
                "--n-samples 64 --max-tokens 10 --seed 0",
            ),
            (
                "forward_word_lookahead_entropy",
                "gpt2",
                {"n_samples": 64, "max_tokens": 10, "seed": 0},
                "--n-samples 64 --max-tokens 10 --seed 0",
            ),
            ("unigram_surprisal", "wordfreq", {}, ""),
            ("mmc_unigram_surprisal", "gpt2", {}, ""),
            ("logit_lens_surprisal", "gpt2", {}, ""),
            ("tuned_lens_surprisal", "gpt2", {}, ""),
            # Convention axes: both derivations must agree on the suffixed ids.
            ("surprisal", "gpt2", {"add_bos": False}, "--no-bos"),
            (
                "surprisal",
                "gpt2",
                {"context_scope": "sentence"},
                "--context-scope sentence",
            ),
            (
                "logit_lens_surprisal",
                "gpt2",
                {"context_scope": "sentence"},
                "--context-scope sentence",
            ),
            (
                "information_value",
                "gpt2",
                {"n_samples": 100, "max_tokens": 60, "top_p": 0.8, "seed": 0},
                "--n-samples 100 --max-tokens 60 --top-p 0.8 --seed 0",
            ),
            (
                "responsivity",
                "gpt2",
                {"n_samples": 64, "seed": 0},
                "--n-samples 64 --seed 0",
            ),
            (
                "information_value",
                "gpt2",
                {"n_samples": 100, "max_tokens": 60, "top_p": 0.9, "seed": 0},
                "--n-samples 100 --max-tokens 60 --top-p 0.9 --seed 0",
            ),
            (
                "word_information_value",
                "ai-forever/mGPT",
                {"n_samples": 5, "max_tokens": 3, "seed": 0},
                "--n-samples 5 --max-tokens 3 --seed 0",
            ),
            ("similarity_surprisal_noncontextual", "gpt2", {}, ""),
            (
                "similarity_surprisal_contextual",
                "gpt2",
                {"n_samples": 64, "seed": 0},
                "--n-samples 64 --seed 0",
            ),
            (
                "similarity_surprisal_word",
                "gpt2",
                {"n_samples": 64, "max_tokens": 10, "seed": 0},
                "--n-samples 64 --max-tokens 10 --seed 0",
            ),
            # mGPT picks up the LM-safe-name conversion (slash → underscore).
            ("surprisal", "ai-forever/mGPT", {}, ""),
        ],
    )
    def test_filename_matches(
        self, cs_mod, method, lm, ctx_overrides, cell_extra, tmp_path
    ):
        # Defaults match the feature sweep.
        ctx_kwargs = dict(
            model_id=lm,
            beam_width=5,
            n_samples=64,
            max_tokens=10,
            alpha=1.0,
            seed=None,
            add_bos=True,
            dataset_language=None,
            language_override=None,
        )
        ctx_kwargs.update(ctx_overrides)
        ctx = cs_mod.ComputeContext(**ctx_kwargs)

        producer_filename = f"{cs_mod.cache_stem(method, ctx, 'provo')}.pkl"

        # cleanup.py's independent derivation must agree.
        cell = Cell(
            tier="?",
            dataset="provo",
            lm=lm,
            method=method,
            extra=cell_extra,
        )
        cleanup_filename = pickle_path_for(cell, tmp_path).name

        assert cleanup_filename == producer_filename, (
            f"\nfeature producer: {producer_filename}\n"
            f"cleanup.py:        {cleanup_filename}\n"
            "The tag-format rule has drifted between the two modules. "
            "Update both at once."
        )

    def test_language_override_matches_the_producer_slot(self, cs_mod, tmp_path):
        """``--language de`` and the producer's normalised ``wordfreq-de``
        name the same cell; a bare ``wordfreq`` run with an override must
        never collide with the unqualified cell."""
        ctx = cs_mod.ComputeContext(
            model_id="wordfreq-de",
            beam_width=5,
            n_samples=64,
            max_tokens=10,
            alpha=1.0,
            seed=None,
            add_bos=True,
            dataset_language=None,
            language_override=None,
        )
        producer_filename = (
            f"{cs_mod.cache_stem('unigram_surprisal', ctx, 'provo')}.pkl"
        )
        cell = Cell(
            tier="?",
            dataset="provo",
            lm="wordfreq",
            method="unigram_surprisal",
            extra="--language de",
        )
        assert pickle_path_for(cell, tmp_path).name == producer_filename

    def test_every_method_in_specs_is_covered(self, cs_mod):
        """Every registered method has a cache-naming round-trip case.

        The covered set is read off ``test_filename_matches``'s parametrize
        list, so only a real case can cover a method.
        """
        [cases] = [
            mark.args[1]
            for mark in type(self).test_filename_matches.pytestmark
            if mark.name == "parametrize"
        ]
        covered = {method for method, *_ in cases}
        missing = set(cs_mod.METHOD_SPECS) - covered
        assert not missing, (
            f"METHOD_SPECS contains {missing!r} but no round-trip case "
            "covers them. Add to the parametrize list above."
        )

    @pytest.mark.parametrize(
        "method", ["first_token_entropy_buggy", "renyi_entropy_buggy"]
    )
    def test_buggy_token_compute_methods_warn(self, cs_mod, method):
        """Selecting a reproduction-only compute method warns before LM work."""
        from replica.feature.entropy import ReproductionOnlyWarning

        ctx = cs_mod.ComputeContext(
            model_id="gpt2",
            beam_width=5,
            n_samples=64,
            max_tokens=10,
            alpha=0.5,
            seed=0,
            add_bos=True,
            dataset_language="en",
            language_override=None,
            allow_reproduction=True,
        )
        with pytest.warns(ReproductionOnlyWarning, match="only for reproduction"):
            model = cs_mod.build_model(method, ctx)
        assert model.token_entropy_sum is (method == "first_token_entropy_buggy")
        assert model.token_renyi_entropy_sum is (method == "renyi_entropy_buggy")
