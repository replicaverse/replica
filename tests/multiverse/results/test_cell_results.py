"""Tests for grid-cell TSV row formatting."""

from types import SimpleNamespace

import pytest

from replica.dataset.measures import FIRST_PASS_RUN
from replica.multiverse.results.cell_results import (
    GRID_TSV_COLUMNS,
    check_append_header,
    default_supported,
    error_row,
    format_tsv_row,
    skip_row,
    standardized_effects,
    success_row,
)


def _manifest_row() -> SimpleNamespace:
    return SimpleNamespace(
        paper="paper_a",
        position=7,
        cell="controls+lags+predictors+log2+outlier+cv+sig",
        corpus="meco_en",
        dv=FIRST_PASS_RUN,
        lm="gpt2",
        predictor_key="standard",
        pickle_path=None,
        present=True,
    )


def test_default_supported_uses_positive_delta_ll_or_r2():
    assert default_supported({"delta_ll": 0.1, "delta_r2": -1.0})
    assert default_supported({"delta_ll": -1.0, "delta_r2": 0.1})
    assert not default_supported({"delta_ll": -1.0, "delta_r2": 0.0})


def test_success_row_uses_schema_defaults():
    row = success_row(
        _manifest_row(),
        {"delta_ll": 1.2, "delta_ll_se": 0.3, "pvalue": 0.04},
        paper_tag="paper_a_surp",
    )

    assert row["paper"] == "paper_a_surp"
    assert row["cell"] == "controls+lags+predictors+log2+outlier+cv+sig"
    assert row["position"] == 7
    assert row["status"] == "ok"
    assert row["n_folds"] == 10
    assert row["supported"] == "True"
    assert row["fold_delta_ll"] == ""
    assert format_tsv_row(row).split("\t")[0] == "paper_a_surp"


def test_success_row_respects_explicit_supported_false():
    row = success_row(_manifest_row(), {"delta_ll": 1.0, "supported": False})

    assert row["supported"] == "False"


def test_error_row_records_explicit_error_fields():
    row = error_row(_manifest_row(), RuntimeError("boom"))
    parts = format_tsv_row(row).split("\t")

    assert len(parts) == len(GRID_TSV_COLUMNS)
    assert row["status"] == "error"
    assert row["error_type"] == "RuntimeError"
    assert row["error_message"] == "boom"
    assert row["supported"] == "False"
    assert row["fold_delta_r2"] == ""


def test_skip_row_records_explicit_skip_reason():
    row = skip_row(_manifest_row(), "missing predictor pickle")

    assert row["status"] == "skip"
    assert row["skip_reason"] == "missing predictor pickle"
    assert row["supported"] == "False"


# ---------------------------------------------------------------------------
# Scale-free effect-size column (Cohen's f²-style)
# ---------------------------------------------------------------------------


def test_standardized_effects_blank_without_r2_full():
    # ΔLL-only cell carries no r2_full → f² unavailable.
    assert standardized_effects({"delta_ll_per_unit": 0.5})["cohens_f2"] is None
    assert standardized_effects({})["cohens_f2"] is None


def test_success_row_includes_cohens_f2():
    row = success_row(
        _manifest_row(),
        {
            "delta_ll": 1.2,
            "delta_ll_per_unit": 0.5,
            "delta_r2": 0.02,
            "r2_full": 0.2,
            "pvalue": 0.04,
        },
    )
    assert row["cohens_f2"] == (0.02 / 0.80)
    # The serialized row still has exactly one field per column.
    assert len(format_tsv_row(row).split("\t")) == len(GRID_TSV_COLUMNS)


def test_error_and_skip_rows_leave_cohens_f2_blank():
    err = error_row(_manifest_row(), RuntimeError("boom"))
    skip = skip_row(_manifest_row(), "missing")
    for row in (err, skip):
        assert row["cohens_f2"] == ""


# ---------------------------------------------------------------------------
# check_append_header
# ---------------------------------------------------------------------------


def test_check_append_header_passes_on_missing_and_matching(tmp_path):
    p = tmp_path / "grid.tsv"
    check_append_header(p, GRID_TSV_COLUMNS)  # missing file: ok
    p.write_text("\t".join(GRID_TSV_COLUMNS) + "\nrow1\n")
    check_append_header(p, GRID_TSV_COLUMNS)  # matching header: ok


def test_check_append_header_rejects_stale_schema(tmp_path):
    p = tmp_path / "grid.tsv"
    p.write_text("\t".join(GRID_TSV_COLUMNS[:-1]) + "\nrow1\n")  # old, narrower
    with pytest.raises(ValueError, match="ragged"):
        check_append_header(p, GRID_TSV_COLUMNS)


def test_error_row_serializes_to_one_line_with_stable_columns():
    # Multi-line R subprocess errors must not inject TSV lines or columns.
    row = error_row(_manifest_row(), RuntimeError("bad\tvalue\r\nsecond line"))

    line = format_tsv_row(row)

    assert "\n" not in line and "\r" not in line
    assert len(line.split("\t")) == len(GRID_TSV_COLUMNS)
    assert "bad value  second line" in line


def test_format_tsv_row_maps_none_to_blank():
    row = {column: None for column in GRID_TSV_COLUMNS}
    assert format_tsv_row(row) == "\t" * (len(GRID_TSV_COLUMNS) - 1)


def test_check_append_header_accepts_an_empty_existing_file(tmp_path):
    empty = tmp_path / "cells.tsv"
    empty.touch()
    check_append_header(empty, GRID_TSV_COLUMNS)
