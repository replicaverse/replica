"""Study input counts, coverage, folds and GAMM means against paper fixtures."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path

import pandas as pd
import pytest

from replica.studies.hopton2026speech import Hopton2026Speech, word_coverage
from replica.studies.hopton2026speech import reference as ref
from replica.studies.records.resources import resource_file

_FIXTURES = Path(__file__).parent / "fixtures"


def _delta_ll_fixture() -> pd.DataFrame:
    return pd.read_csv(_FIXTURES / "hopton2026speech_delta_ll.tsv", sep="\t")


def _fixture_label(model: str, granularity: str, variant: str) -> str:
    """The ``Model`` column value in ``delta_ll_summary`` for one cell."""
    if model == "gpt2":
        return "ME GPT-2"
    lab = {
        "gslm": "GSLM",
        "spirit": "SpiRit",
        "llamamimi": "Llama-Mimi",
        "llamamimi8b": "Llama-Mimi-8B",
    }[model]
    v = {"marg": "Marg.", "unmarg": "Unmarg."}[variant]
    g = {"word": "Word", "phone": "Phone", "frame": "Frame"}[granularity]
    return f"ME {lab} {v} {g}"


# ---------------------------------------------------------------------------
# Data layer (no R): input tables reproduce Table 6 counts.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "options",
    [
        {"model": "unknown"},
        {"granularity": "syllable"},
        {"variant": "mrag"},
        {"model": "gpt2", "variant": "marg"},
        {"model": "gpt2", "granularity": "phone"},
        {"granularity": "phone", "variant": "marg"},
        {"granularity": "frame", "variant": "marg"},
    ],
)
def test_invalid_conditions_fail_before_loading_inputs(options, monkeypatch):
    def unexpected_input(*args):
        pytest.fail("Invalid conditions must be rejected before loading inputs")

    monkeypatch.setattr(
        import_module("replica.studies.hopton2026speech"),
        "_input_table",
        unexpected_input,
    )
    study = Hopton2026Speech()
    for method in (study.dataset, study.evaluation_spec, study.analysis_config):
        kwargs = {"dataset": None} if method == study.evaluation_spec else {}
        with pytest.raises(ValueError):
            method(**options, **kwargs)


@pytest.mark.artifact
@pytest.mark.parametrize(
    ("granularity", "model", "n_obs"), [(g, m, n) for (g, m), n in ref.N_OBS.items()]
)
def test_model_data_observation_counts(granularity, model, n_obs):
    """Each cell's model-data row count matches Table 6 N obs (at omega=1000)."""
    study = Hopton2026Speech()
    df = study._model_data(
        model=model, granularity=granularity, variant="unmarg", offset=1000
    )
    assert len(df) == n_obs


@pytest.mark.artifact
def test_word_coverage():
    """Word-level marginalized coverage matches Table 6 (1229; 959/809/1010/1009)."""
    cov = word_coverage().set_index("model")
    for model, (est, total) in ref.WORD_COVERAGE.items():
        assert int(cov.loc[model, "n_tokens_est"]) == est
        assert int(cov.loc[model, "n_tokens_total"]) == total


@pytest.mark.artifact
def test_fold_map_covers_every_cell():
    """folds.parquet holds a fold for every item of every cell and offset."""
    folds = pd.read_parquet(resource_file("hopton2026speech", "data/folds.parquet"))
    stats = pd.read_csv(_FIXTURES / "hopton2026speech_fold_stats.tsv", sep="\t")
    for _, r in stats.iterrows():
        cell = (
            f"{str(r['Granularity']).lower()}_"
            + {
                "GPT-2": "gpt2",
                "GSLM": "gslm",
                "SpiRit": "spirit",
                "Llama-Mimi": "llamamimi",
                "Llama-Mimi-8B": "llamamimi8b",
            }[r["LM"]]
        )
        m = folds[(folds["cell"] == cell) & (folds["offset_ms"] == int(r["offset_ms"]))]
        assert len(m) == int(r["n_items"]), (cell, r["offset_ms"])


# ---------------------------------------------------------------------------
# Reproduction (needs R / mgcv): evaluate() delta-LL vs the frozen fixture.
# ---------------------------------------------------------------------------

# All 17 published delta-LL cells x 3 offsets = 51 GAMMs. Word carries GPT-2
# plus each speech LM's marginalized and unmarginalized surprisal; phone and
# frame are unmarginalized per speech LM. Fitting every fold of every cell is
# heavy, so the test carries the ``slow`` mark alongside ``needs_r``.
_REPRO_CELLS = [
    # word (9): GPT-2 baseline + marg/unmarg per speech LM
    ("gpt2", "word", "unmarg"),
    ("gslm", "word", "marg"),
    ("gslm", "word", "unmarg"),
    ("spirit", "word", "marg"),
    ("spirit", "word", "unmarg"),
    ("llamamimi", "word", "marg"),
    ("llamamimi", "word", "unmarg"),
    ("llamamimi8b", "word", "marg"),
    ("llamamimi8b", "word", "unmarg"),
    # phone (4): unmarginalized per speech LM
    ("gslm", "phone", "unmarg"),
    ("spirit", "phone", "unmarg"),
    ("llamamimi", "phone", "unmarg"),
    ("llamamimi8b", "phone", "unmarg"),
    # frame (4): unmarginalized per speech LM
    ("gslm", "frame", "unmarg"),
    ("spirit", "frame", "unmarg"),
    ("llamamimi", "frame", "unmarg"),
    ("llamamimi8b", "frame", "unmarg"),
]

#: Anchored at ``rtol=1e-6, atol=5e-5``. The study's ``analysis_fn`` reproduces
#: the paper's exact scoring (``shrink_unseen`` held-out RE prediction, one pooled
#: held-out-residual sigma per model, per-observation mean, R's replayed folds),
#: so ``delta_ll_per_unit`` matches the frozen ``mean_DeltaLL``. ``atol`` covers
#: the near-zero cells, where the residual is a fit-side floor (mgcv
#: ``discrete=TRUE`` binning plus small fREML numerical differences) below the
#: paper's reported 1e-4 precision and the bootstrap CI width. Same anchor shape
#: as ``hopton2026unigram``.
_REPRO_RTOL = 1e-6
_REPRO_ATOL = 5e-5


@pytest.mark.artifact
@pytest.mark.slow
@pytest.mark.needs_r
@pytest.mark.parametrize("offset", ref.OFFSETS)
@pytest.mark.parametrize(("model", "granularity", "variant"), _REPRO_CELLS)
def test_reproduction_delta_ll(model, granularity, variant, offset):
    from replica.analysis.evaluation.evaluate import evaluate

    study = Hopton2026Speech()
    dataset = study.dataset(
        model=model, granularity=granularity, variant=variant, offset=offset
    )
    results = evaluate(
        dataset,
        study.evaluation_spec(
            model=model,
            granularity=granularity,
            variant=variant,
            offset=offset,
            dataset=dataset,
        ),
        study.analysis_config(
            model=model, granularity=granularity, variant=variant, offset=offset
        ),
    )
    got = results[0].delta_ll_per_unit

    fx = _delta_ll_fixture()
    row = fx[
        (fx["Model"] == _fixture_label(model, granularity, variant))
        & (fx["offset_ms"] == offset)
    ]
    assert len(row) == 1, (model, granularity, variant, offset)
    expected = float(row["mean_DeltaLL"].iloc[0])
    assert got == pytest.approx(expected, rel=_REPRO_RTOL, abs=_REPRO_ATOL)


def _delta_ll_timesmooth_fixture() -> pd.DataFrame:
    """Frozen delta-LL from the R time-smooth run (``plots_timeSmooth/``)."""
    return pd.read_csv(_FIXTURES / "hopton2026speech_delta_ll_timesmooth.tsv", sep="\t")


@pytest.mark.artifact
@pytest.mark.slow
@pytest.mark.needs_r
@pytest.mark.parametrize("offset", [ref.MAIN_OFFSET])
@pytest.mark.parametrize(("model", "granularity", "variant"), _REPRO_CELLS)
def test_reproduction_delta_ll_timesmooth(model, granularity, variant, offset):
    """Appendix time-smooth baseline: a normalized trial-time nuisance smooth
    ``s(norm_time, k=4)`` added to the full model and the baseline (symmetric,
    as in the R under ``INCLUDE_TIME_SMOOTH``), scored the same held-out way and
    anchored to the frozen ``plots_timeSmooth`` delta-LL. The appendix reports
    this robustness check at the 1000 ms main offset only, so it is 17 cells."""
    from replica.analysis.evaluation.evaluate import evaluate

    study = Hopton2026Speech()
    dataset = study.dataset(
        model=model, granularity=granularity, variant=variant, offset=offset
    )
    results = evaluate(
        dataset,
        study.evaluation_spec(
            model=model,
            granularity=granularity,
            variant=variant,
            offset=offset,
            time_smooth=True,
            dataset=dataset,
        ),
        study.analysis_config(
            model=model,
            granularity=granularity,
            variant=variant,
            offset=offset,
            time_smooth=True,
        ),
    )
    got = results[0].delta_ll_per_unit

    fx = _delta_ll_timesmooth_fixture()
    row = fx[
        (fx["Model"] == _fixture_label(model, granularity, variant))
        & (fx["offset_ms"] == offset)
    ]
    assert len(row) == 1, (model, granularity, variant, offset)
    expected = float(row["mean_DeltaLL"].iloc[0])
    assert got == pytest.approx(expected, rel=_REPRO_RTOL, abs=_REPRO_ATOL)
