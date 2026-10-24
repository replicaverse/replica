"""Tests for replica.multiverse.curve.report."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from replica.dataset.measures import FIRST_PASS_RUN
from replica.multiverse.curve import report as scr

_DELTAS = {}


@pytest.fixture(autouse=True)
def _clear_deltas():
    _DELTAS.clear()


def _load_deltas(row):
    return _DELTAS.get(tuple(row[column] for column in scr.CELL_KEY))


_REPORT = scr.ReportSpec(
    swap_fields=(
        "word_len_source",
        "freq_source",
        "baseline_lags",
        "log_prob_source",
        "entropy_source",
        "renyi_source",
        "transform_dv",
        "outlier_filter",
        "cv_splitter",
        "sig_test",
    ),
    joint_criteria={
        "cv_splitter": "paper",
        "transform_dv": "paper",
        "outlier_filter": "None",
    },
    load_deltas=_load_deltas,
    assume_independent_observations=True,
    attribution={
        "word_len_source": ("word_len_source",),
        "freq_source": ("freq_source",),
        "baseline_lags": ("baseline_lags",),
        "lm_source": ("log_prob_source", "entropy_source", "renyi_source"),
        "transform_dv": ("transform_dv",),
        "outlier_filter": ("outlier_filter",),
        "cv_splitter": ("cv_splitter",),
        "sig_test": ("sig_test",),
    },
    likelihood_scale_fields=("transform_dv",),
    exclude_atomic_cell=lambda label: label == "bugfix" or "+bugfix" in label,
)


_DECISIONS = (
    ("controls", {"word_len_source": "replica", "freq_source": "replica"}),
    ("lags", {"baseline_lags": "replica"}),
    (
        "predictors",
        {
            "log_prob_source": "replica",
            "entropy_source": "replica",
            "renyi_source": "replica",
        },
    ),
    ("log2", {"transform_dv": "log2"}),
    ("outlier", {"outlier_filter": {"floor": 80.0, "method": "mad", "k": 2.5}}),
    ("cv", {"cv_splitter": "replica"}),
    ("sig", {"sig_test": "replica"}),
)


def _spec(**overrides):
    values = dict.fromkeys(_REPORT.swap_fields, "paper")
    values["outlier_filter"] = None
    return SimpleNamespace(**{**values, **overrides})


def _label(position, mode):
    if position == 0:
        return "anchor"
    names = [name for name, _ in _DECISIONS]
    return names[position - 1] if mode == "ablation" else "+".join(names[:position])


def _recipe_spec(recipe, paper, position, *, mode):
    overrides = {}
    selected = (
        _DECISIONS[:position]
        if mode != "ablation"
        else _DECISIONS[max(0, position - 1) : position]
    )
    for _, values in selected:
        overrides.update(values)
    return _spec(**overrides)


_RECIPE = SimpleNamespace(label=_label)
_REGISTRY = SimpleNamespace(recipe=lambda name: _RECIPE, recipe_spec=_recipe_spec)


def _add_swap(df, mode):
    return scr.add_swap_columns(
        df, mode, registry=_REGISTRY, report=_REPORT, recipe="bundled"
    )


def _cell_row(
    paper="paper_a",
    position=0,
    corpus="provo",
    dv=FIRST_PASS_RUN,
    predictor="renyi05",
    status="ok",
    delta_ll=10.0,
    n_obs=100,
    pvalue=0.01,
    cohens_f2=0.02,
    delta_r2=np.nan,
    supported=True,
):
    return {
        "paper": paper,
        "position": position,
        "cell": _RECIPE.label(position, "ablation") if position >= 0 else "",
        "corpus": corpus,
        "dv": dv,
        "predictor": predictor,
        "status": status,
        "delta_ll": delta_ll,
        "n_obs": n_obs,
        "pvalue": pvalue,
        "cohens_f2": cohens_f2,
        "delta_r2": delta_r2,
        "supported": supported,
    }


# ---------------------------------------------------------------------------
# load_grid_cells
# ---------------------------------------------------------------------------


class TestLoadGridCells:
    def test_concats_and_keeps_last_duplicate(self, tmp_path):
        cols = list(_cell_row())
        first = pd.DataFrame([_cell_row(delta_ll=1.0), _cell_row(position=1)])
        # Same (paper, position, cell, corpus, dv, predictor) appears again with a new
        # value, the re-run/append case; the later row must win.
        second = pd.DataFrame([_cell_row(delta_ll=99.0)])
        first.to_csv(tmp_path / "a_position0.tsv", sep="\t", index=False, columns=cols)
        second.to_csv(tmp_path / "b_position0.tsv", sep="\t", index=False, columns=cols)

        df = scr.load_grid_cells(tmp_path)

        assert len(df) == 2
        dup = df[(df["position"] == 0)]
        assert float(dup["delta_ll"].iloc[0]) == 99.0

    def test_raises_on_empty_dir(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            scr.load_grid_cells(tmp_path)

    def test_rejects_non_replication_cell_schema(self, tmp_path):
        pd.DataFrame([{"dataset": "provo", "delta_ll": 1.0}]).to_csv(
            tmp_path / "benchmark.tsv", sep="\t", index=False
        )
        with pytest.raises(ValueError, match="missing required grid columns"):
            scr.load_grid_cells(tmp_path)


# ---------------------------------------------------------------------------
# add_swap_columns
# ---------------------------------------------------------------------------


class TestAddSwapColumns:
    def test_ablation_flips_exactly_one_dimension_per_step(self):
        df = pd.DataFrame([_cell_row(position=s) for s in range(8)])

        out = _add_swap(df, "ablation")

        anchor = out[out["position"] == 0].iloc[0]
        for s in range(1, 7):
            row = out[out["position"] == s].iloc[0]
            flipped = [f for f in _REPORT.swap_fields if row[f] != anchor[f]]
            assert len(flipped) >= 1, f"position {s} flips nothing"
            # The predictor-swap flips the three LM-source dims together and
            # controls flips two; every other single flip is one field.
            assert len(flipped) <= 3, f"position {s} flips {flipped}"

    def test_cv_splitter_column_marks_the_splitter_swap(self):
        df = pd.DataFrame([_cell_row(position=0), _cell_row(position=6)])

        out = _add_swap(df, "ablation")

        assert out[out["position"] == 0]["cv_splitter"].iloc[0] == "paper"
        assert out[out["position"] == 6]["cv_splitter"].iloc[0] != "paper"


# ---------------------------------------------------------------------------
# summarize_claims (incl. the joint test's alignment gates)
# ---------------------------------------------------------------------------


class TestSummarizeClaims:
    def _save_sidecar(self, paper, corpus, predictor, position, dv, arr):
        key = (
            paper,
            corpus,
            dv,
            predictor,
            position,
            _RECIPE.label(position, "ablation"),
        )
        _DELTAS[key] = scr.CellDeltas(key, tuple(range(len(arr))), arr)

    def test_joint_explicitly_excludes_incomparable_swaps(self):
        rng = np.random.default_rng(7)
        # Clearly positive per-obs ΔLL, large enough that the seed-7 draw keeps
        # the joint p-value below alpha.
        base = rng.normal(0.15, 0.1, size=50)
        rows = [_cell_row(position=position) for position in (0, 1, 2, 4, 5, 6)]
        # Positions 0/1/2: aligned (length 50) → in the joint test.
        # Position 4 (log2-DV swap): aligned length but another ΔLL scale → out.
        # Position 5 (outlier swap): length 40 → out via the explicit filter rule.
        # Position 6 (splitter swap): aligned length but reordered rows → out.
        for position in (0, 1, 2, 4, 6):
            self._save_sidecar(
                "paper_a",
                "provo",
                "renyi05",
                position,
                FIRST_PASS_RUN,
                base + rng.normal(0, 0.01, 50),
            )
        self._save_sidecar("paper_a", "provo", "renyi05", 5, FIRST_PASS_RUN, base[:40])
        df = _add_swap(pd.DataFrame(rows), "ablation")

        claims = scr.summarize_claims(
            df, report=_REPORT, n_resamples=500, rng=np.random.default_rng(0)
        )

        assert len(claims) == 1
        c = claims.iloc[0]
        assert c["n_specs"] == 6
        assert c["n_specs_joint"] == 3
        assert c["joint_positions"] == "0,1,2"
        assert c["joint_median_dll"] == pytest.approx(np.mean(base), abs=0.01)
        assert c["joint_p"] < 0.05

    def test_no_sidecars_yields_nan_joint(self):
        df = _add_swap(pd.DataFrame([_cell_row(position=0)]), "ablation")

        claims = scr.summarize_claims(df, report=_REPORT, n_resamples=100)

        c = claims.iloc[0]
        assert c["n_specs_joint"] == 0
        assert np.isnan(c["joint_p"])

    def test_descriptives_over_both_effect_columns(self):
        rows = [
            _cell_row(position=0, delta_ll=10.0, n_obs=100, cohens_f2=0.02),
            _cell_row(position=1, delta_ll=-5.0, n_obs=100, cohens_f2=np.nan),
        ]
        df = _add_swap(pd.DataFrame(rows), "ablation")

        c = scr.summarize_claims(df, report=_REPORT, n_resamples=100).iloc[0]

        assert c["n_specs"] == 2
        assert c["frac_in_direction"] == 0.5
        assert c["dll_per_obs_median"] == pytest.approx(0.025)  # (0.1 - 0.05)/2
        assert c["n_specs_f2"] == 1  # NaN f² spec drops from the f² curve
        assert c["cohens_f2_median"] == pytest.approx(0.02)

    def test_r2_native_runner_fills_the_delta_r2_curve(self):
        """ΔR²-only cells report ΔR² but no ΔLL, the ΔR² block must
        carry the curve while the ΔLL block stays NaN."""
        rows = [
            _cell_row(
                paper="paper_b",
                position=s,
                delta_ll=np.nan,
                cohens_f2=np.nan,
                delta_r2=0.01 * (s + 1),
            )
            for s in range(3)
        ]
        df = _add_swap(pd.DataFrame(rows), "ablation")

        c = scr.summarize_claims(df, report=_REPORT, n_resamples=100).iloc[0]

        assert np.isnan(c["dll_per_obs_median"])
        assert c["n_specs_f2"] == 0
        assert c["n_specs_dr2"] == 3
        assert c["delta_r2_median"] == pytest.approx(0.02)
        assert c["frac_in_direction_dr2"] == 1.0

    def test_non_ok_rows_are_excluded(self):
        rows = [
            _cell_row(position=0),
            _cell_row(position=1, status="error", delta_ll=np.nan),
        ]
        df = _add_swap(pd.DataFrame(rows), "ablation")

        c = scr.summarize_claims(df, report=_REPORT, n_resamples=100).iloc[0]

        assert c["n_specs"] == 1


# ---------------------------------------------------------------------------
# paired_swaps / swap_summary_tables
# ---------------------------------------------------------------------------


class TestPairedSwaps:
    def _df(self, rows):
        return _add_swap(pd.DataFrame(rows), "ablation")

    def test_pairs_effect_and_flip_against_anchor(self):
        df = self._df(
            [
                _cell_row(position=0, delta_r2=0.010, supported=True),
                _cell_row(position=1, delta_r2=0.005, supported=False),
            ]
        )

        long, metrics = scr.paired_swaps(df[df["status"] == "ok"], report=_REPORT)

        assert metrics["paper_a"] == "delta_r2"
        row = long.iloc[0]
        assert row["rel"] == pytest.approx(-0.5)
        assert row["flip"] == 1.0

    def test_missing_anchor_yields_nan_not_a_flip(self):
        """A claim whose anchor cell is absent (errored anchor / partial run)
        must not count as a conclusion reversal."""
        df = self._df([_cell_row(position=1, delta_r2=0.005, supported=False)])

        long, _ = scr.paired_swaps(df[df["status"] == "ok"], report=_REPORT)

        assert np.isnan(long["rel"].iloc[0])
        assert np.isnan(long["flip"].iloc[0])

    def test_summary_n_counts_only_paired_claims(self):
        df = self._df(
            [
                _cell_row(position=0, corpus="provo", delta_r2=0.01, supported=True),
                _cell_row(position=1, corpus="provo", delta_r2=0.01, supported=True),
                # dundee has no anchor → its swapped row must not enter n.
                _cell_row(position=1, corpus="dundee", delta_r2=0.01, supported=True),
            ]
        )

        pct, flips, n, _ = scr.swap_summary_tables(
            df[df["status"] == "ok"], report=_REPORT
        )

        assert int(n.iloc[0, 0]) == 1
        assert flips.iloc[0, 0] == 0.0

    def test_dll_metric_masks_rel_across_dv_transform(self):
        """ΔLL/N papers (no ΔR²): the log2-DV pairing crosses likelihood
        scales and must be masked, while the flip stays valid."""
        rows = [
            _cell_row(position=0, delta_r2=np.nan, supported=True),
            _cell_row(position=4, delta_r2=np.nan, supported=True),  # log2 swap
        ]
        df = self._df(rows)

        long, metrics = scr.paired_swaps(df[df["status"] == "ok"], report=_REPORT)

        assert metrics["paper_a"] == "dll_per_obs"
        assert np.isnan(long["rel"].iloc[0])
        assert long["flip"].iloc[0] == 0.0


class TestGridRevisionGuard:
    def test_stale_cell_labels_raise(self):
        df = pd.DataFrame([{**_cell_row(position=2), "cell": "predictors"}])

        with pytest.raises(ValueError, match="grid revision"):
            _add_swap(df, "ablation")

    def test_current_labels_pass(self):
        df = pd.DataFrame([_cell_row(position=2)])

        out = _add_swap(df, "ablation")

        assert out["baseline_lags"].iloc[0] == "replica"


# ---------------------------------------------------------------------------
# attribution_by_paper
# ---------------------------------------------------------------------------


class TestAttributionByPaper:
    def test_attributes_variance_to_the_moving_swap(self):
        rows = []
        # Two corpora x 7 single flips; the entropy-source swap doubles the
        # effect, everything else leaves it flat.
        for corpus in ("provo", "dundee"):
            for position in range(8):
                rows.append(_cell_row(position=position, corpus=corpus, cohens_f2=0.01))
        df = _add_swap(pd.DataFrame(rows), "ablation")
        moved = (
            df["entropy_source"] != df[df["position"] == 0]["entropy_source"].iloc[0]
        )
        df.loc[moved, "cohens_f2"] = 0.05

        att = scr.attribution_by_paper(df, "cohens_f2", report=_REPORT)

        assert len(att) == 1
        row = att.iloc[0]
        assert row["n_cells"] == 16
        eta_cols = [c for c in att.columns if c.startswith("eta2_")]
        top = max(eta_cols, key=lambda c: row[c])
        # log_prob/entropy/renyi flip together (collinear), so attribution
        # collapses them into the composite ``lm_source`` dimension.
        assert top == "eta2_lm_source"
        assert row["eta2_lm_source"] > 0.5

    def test_dll_per_obs_effect_is_derived_when_missing(self):
        df = _add_swap(
            pd.DataFrame([_cell_row(position=s) for s in range(3)]), "ablation"
        )

        att = scr.attribution_by_paper(df, "dll_per_obs", report=_REPORT)

        assert att.iloc[0]["effect"] == "dll_per_obs"
        assert att.iloc[0]["n_cells"] == 3


# ---------------------------------------------------------------------------
# add_swap_columns, atomic mode
# ---------------------------------------------------------------------------


_ATOMIC_REGISTRY = SimpleNamespace(
    cells_for_mode=lambda mode: (
        lambda paper: (
            ("anchor", _spec()),
            ("freq", _spec(freq_source="replica")),
            ("log2", _spec(transform_dv="log2")),
        )
    )
)


def _atomic_add_swap(df):
    return scr.add_swap_columns(
        df, "atomic-ablation", registry=_ATOMIC_REGISTRY, report=_REPORT
    )


class TestAddSwapColumnsAtomic:
    def _row(self, cell):
        return {**_cell_row(position=-1), "cell": cell}

    def test_swap_columns_come_from_the_decision_specs(self):
        df = pd.DataFrame([self._row("anchor"), self._row("freq"), self._row("log2")])

        out = _atomic_add_swap(df)

        by_cell = out.set_index("cell")
        assert by_cell.loc["anchor", "freq_source"] == "paper"
        assert by_cell.loc["freq", "freq_source"] == "replica"
        assert by_cell.loc["freq", "transform_dv"] == "paper"
        assert by_cell.loc["log2", "transform_dv"] == "log2"

    def test_bugfix_variant_rows_are_dropped(self):
        df = pd.DataFrame(
            [self._row("anchor"), self._row("bugfix"), self._row("freq+bugfix")]
        )

        out = _atomic_add_swap(df)

        assert list(out["cell"]) == ["anchor"]

    def test_missing_cell_column_is_rejected(self):
        df = pd.DataFrame([_cell_row(position=-1)]).drop(columns=["cell"])

        with pytest.raises(ValueError, match="'cell' column"):
            _atomic_add_swap(df)

    def test_unknown_cell_label_is_rejected(self):
        df = pd.DataFrame([self._row("outlier")])

        with pytest.raises(ValueError, match="unknown atomic cell"):
            _atomic_add_swap(df)


def test_ablation_mode_pins_the_flipped_fields_per_step():
    expected = {
        1: {"word_len_source", "freq_source"},
        2: {"baseline_lags"},
        3: {"log_prob_source", "entropy_source", "renyi_source"},
        4: {"transform_dv"},
        5: {"outlier_filter"},
        6: {"cv_splitter"},
        7: {"sig_test"},
    }
    df = pd.DataFrame([_cell_row(position=s) for s in range(8)])

    out = _add_swap(df, "ablation")

    anchor = out[out["position"] == 0].iloc[0]
    for position, fields in expected.items():
        row = out[out["position"] == position].iloc[0]
        flipped = {f for f in _REPORT.swap_fields if row[f] != anchor[f]}
        assert flipped == fields, f"position {position}"


def test_load_grid_cells_keeps_atomic_cells_distinct_by_cell_label(tmp_path):
    rows = [
        {**_cell_row(position=-1), "cell": "anchor", "delta_ll": 1.0},
        {**_cell_row(position=-1), "cell": "freq", "delta_ll": 2.0},
    ]
    cols = list(rows[0])
    pd.DataFrame(rows).to_csv(
        tmp_path / "cells.tsv", sep="\t", index=False, columns=cols
    )

    df = scr.load_grid_cells(tmp_path)

    # Both position=-1 rows survive: identity includes the cell label, so
    # atomic cells do not collapse onto the shared numeric position.
    assert len(df) == 2


def test_report_accepts_unrelated_dimensions_and_comparability_rules():
    from types import SimpleNamespace

    report = scr.ReportSpec(
        swap_fields=("smoothing", "optimizer"),
        joint_criteria={"partition": "shared", "response_units": "counts"},
        load_deltas=_load_deltas,
        assume_independent_observations=True,
        attribution={"estimator": ("smoothing", "optimizer")},
        likelihood_scale_fields=("response_units",),
    )
    registry = SimpleNamespace(
        recipe=lambda name: SimpleNamespace(
            label=lambda position, mode: f"case{position}"
        ),
        recipe_spec=lambda *args, **kwargs: SimpleNamespace(
            smoothing=args[2] % 2, optimizer="solver|a"
        ),
    )
    df = pd.DataFrame(
        [
            {
                **_cell_row(position=i, cohens_f2=0.01 + (i % 2) * 0.1),
                "cell": f"case{i}",
                "partition": "shared",
                "response_units": "counts" if i < 2 else "log-counts",
            }
            for i in range(4)
        ]
    )
    annotated = scr.add_swap_columns(
        df, "ablation", registry=registry, recipe="custom", report=report
    )
    assert annotated.smoothing.tolist() == ["0", "1", "0", "1"]
    assert "freq_source" not in annotated
    for _, row in df.iterrows():
        key = tuple(row[column] for column in scr.CELL_KEY)
        _DELTAS[key] = scr.CellDeltas(key, ("obs1", "obs2", "obs3"), [0.01, 0.02, 0.03])
    claims = scr.summarize_claims(annotated, report=report, n_resamples=10)
    assert claims.iloc[0].joint_positions == "0,1"
    paired, _ = scr.paired_swaps(annotated, report=report)
    assert paired.rel.notna().tolist() == [True, False, False]
    attributed = scr.attribution_by_paper(annotated, report=report, extra_dims=())
    assert attributed.iloc[0].eta2_estimator == pytest.approx(1)


def test_atomic_exclusion_is_opt_in():
    from dataclasses import replace

    rows = pd.DataFrame([{**_cell_row(position=-1), "cell": "bugfix"}])
    with pytest.raises(ValueError, match="unknown atomic cell"):
        scr.add_swap_columns(
            rows,
            "atomic-ablation",
            registry=_ATOMIC_REGISTRY,
            report=replace(_REPORT, exclude_atomic_cell=None),
        )


@pytest.mark.parametrize(
    "ids,values,message",
    [
        (("b", "a", "c"), [0.1, 0.2, 0.3], "IDs or order differ"),
        (("a", "b", "other"), [0.1, 0.2, 0.3], "IDs or order differ"),
        (("a", "b"), [0.1, 0.2], "IDs or order differ"),
        (("a", "a", "c"), [0.1, 0.2, 0.3], "IDs must be unique"),
        (("a", "b", "c"), [[0.1, 0.2, 0.3]], "1D delta array"),
    ],
)
def test_joint_rejects_unverified_observation_alignment(ids, values, message):
    from dataclasses import replace

    df = _add_swap(
        pd.DataFrame([_cell_row(position=0), _cell_row(position=1)]), "ablation"
    )
    for i, row in df.iterrows():
        key = tuple(row[column] for column in scr.CELL_KEY)
        _DELTAS[key] = scr.CellDeltas(
            key,
            ("a", "b", "c") if i == 0 else ids,
            [0.1, 0.2, 0.3] if i == 0 else values,
        )
    with pytest.raises(ValueError, match=message):
        scr.summarize_claims(
            df, report=replace(_REPORT, joint_criteria={}), n_resamples=10
        )


def test_atomic_cells_load_distinct_identified_vectors():
    import json

    rows = pd.DataFrame(
        [
            {**_cell_row(position=-1), "cell": "anchor"},
            {**_cell_row(position=-1), "cell": "freq"},
        ]
    )
    calls = []
    expected_keys = []
    for i, row in rows.iterrows():
        key = tuple(row[column] for column in scr.CELL_KEY)
        expected_keys.append(key)
        _DELTAS[key] = scr.CellDeltas(
            key, ("a", "b", "c"), np.array([0.1, 0.2, 0.3]) + i
        )

    def load(row):
        key = tuple(row[column] for column in scr.CELL_KEY)
        calls.append(key)
        return _DELTAS[key]

    report = scr.ReportSpec(
        swap_fields=(),
        joint_criteria={},
        load_deltas=load,
        assume_independent_observations=True,
    )
    summary = scr.summarize_claims(rows, report=report, n_resamples=10).iloc[0]
    assert calls == expected_keys and calls[0] != calls[1]
    assert summary.n_specs_joint == 2
    assert summary.joint_median_dll == pytest.approx(0.7)
    assert json.loads(summary.joint_cells) == ["anchor", "freq"]
    assert summary.joint_positions == "-1,-1"

    # A loader accidentally opening the same sidecar twice cannot label it as
    # the other cell: identity comes from the retained source metadata.
    from dataclasses import replace

    def wrong_loader(row):
        return _DELTAS[expected_keys[0]]

    with pytest.raises(ValueError, match="cell identity does not match"):
        scr.summarize_claims(
            rows, report=replace(report, load_deltas=wrong_loader), n_resamples=10
        )


def test_joint_rejects_duplicate_cell_rows():
    report = scr.ReportSpec(
        swap_fields=(), joint_criteria={}, load_deltas=lambda row: None
    )
    rows = pd.DataFrame([_cell_row(), _cell_row()])
    with pytest.raises(ValueError, match="duplicate cell identity"):
        scr.summarize_claims(rows, report=report, n_resamples=10)


@pytest.mark.parametrize("identity", [None, np.nan, ("subject", np.nan), pd.NA])
def test_joint_rejects_missing_source_identities(identity):
    from dataclasses import replace

    row = _cell_row()
    key = tuple(row[column] for column in scr.CELL_KEY)
    deltas = scr.CellDeltas(key, (identity, "other"), [0.1, 0.2])
    report = scr.ReportSpec(
        swap_fields=(),
        joint_criteria={},
        load_deltas=lambda row: deltas,
        assume_independent_observations=True,
    )
    with pytest.raises(ValueError, match="observation IDs must be non-missing"):
        scr.summarize_claims(pd.DataFrame([row]), report=report, n_resamples=10)
    deltas = replace(deltas, cell_key=(*key[:-1], identity), observation_ids=("a", "b"))
    with pytest.raises(ValueError, match="loaded cell identity must be non-missing"):
        scr.summarize_claims(pd.DataFrame([row]), report=report, n_resamples=10)
    with pytest.raises(ValueError, match="cell identity fields must be non-missing"):
        scr.summarize_claims(
            pd.DataFrame([{**row, "cell": identity}]), report=report, n_resamples=10
        )


def test_joint_requires_explicit_independent_observations():
    from dataclasses import replace

    row = _cell_row()
    key = tuple(row[column] for column in scr.CELL_KEY)
    report = scr.ReportSpec(swap_fields=(), joint_criteria={})
    summary = scr.summarize_claims(pd.DataFrame([row]), report=report).iloc[0]
    assert summary.n_specs == 1 and summary.n_specs_joint == 0
    assert np.isnan(summary.joint_p)
    report = replace(
        report, load_deltas=lambda row: scr.CellDeltas(key, ("a", "b"), [0.1, 0.2])
    )
    with pytest.raises(ValueError, match="assume_independent_observations=True"):
        scr.summarize_claims(pd.DataFrame([row]), report=report, n_resamples=10)
