"""Loading the replication results, and the protocol reviews applied before anything is counted."""

from copy import deepcopy

import pytest

from replica.studies.claims import replication_results
from replica.studies.claims.replication_results import claim_index, reviewed_results


@pytest.mark.artifact
def test_the_shipped_results_parse_and_every_claim_is_registered(
    claim_registry, replication_results
):
    """A summary for an unregistered claim would render a dangling anchor."""
    results = replication_results
    known = set(claim_index(claim_registry))
    assert set(results.get("claims", {})) <= known


def test_released_transfer_cannot_inherit_corrected_results_or_unverified_provenance():
    from copy import deepcopy

    corrected = {"total": {"supporting_cells": 9, "cells": 10}}
    results = {
        "claims": {
            "test_claim": {**corrected, "replication_repetition": deepcopy(corrected)}
        }
    }
    with pytest.raises(ValueError, match="fresh-run scope"):
        reviewed_results(results, {"claims": [{"id": "test_claim"}]})
    results["claims"]["test_claim"]["replication_repetition"].update(
        stage="replication_repetition",
        analysis_method="released",
        provenance="author_code_compute",
        review={"status": "verified"},
        source_run="independent-released-run",
        scope="Released analysis on frozen predictors.",
        artifact={"sha256": "a" * 64},
    )
    reviewed = reviewed_results(results, {"claims": [{"id": "test_claim"}]})
    assert reviewed["claims"]["test_claim"]["total"] == corrected["total"]
    assert (
        reviewed["claims"]["test_claim"]["replication_repetition"]["source_run"]
        == "independent-released-run"
    )


def _claim():
    return {
        "comparand": {
            "cells": ["effect"],
            "significance": {"positive_values": [True]},
        },
        "evidence": {
            "paper": {"values": {"effect": True}},
            "replica": {"values": {"effect": True}},
        },
    }


def _numerical_transfer():
    claim = _claim()
    claim["id"] = "numerical"
    claim["comparand"]["significance"] = False
    entry = {
        "total": None,
        "numerical": {
            "datasets": 1,
            "significance_test": None,
            "rows": [
                {
                    "corpus": "new",
                    "source_arm": "original",
                    "model": "joint",
                    "delta_ll": 3.0,
                    "n_obs": 10,
                }
            ],
        },
    }
    return claim, entry


@pytest.mark.parametrize("problem", ["empty", "count", "nonfinite", "significance"])
def test_numerical_transfer_rejects_invalid_or_mislabeled_evidence(problem):
    claim, entry = _numerical_transfer()
    if problem == "empty":
        entry["numerical"]["rows"] = []
    elif problem == "count":
        entry["numerical"]["datasets"] = 2
    elif problem == "nonfinite":
        entry["numerical"]["rows"][0]["delta_ll"] = float("nan")
    else:
        claim["comparand"]["significance"] = {"positive_values": [True]}
    with pytest.raises(ValueError, match="numerical replication"):
        reviewed_results({"claims": {claim["id"]: entry}}, {"claims": [claim]})


def test_numerical_transfer_still_requires_completed_corrections():
    claim, entry = _numerical_transfer()
    claim["method_defect"] = {"changes_registered_decisions": None}
    results = reviewed_results({"claims": {claim["id"]: entry}}, {"claims": [claim]})
    assert "numerical" not in results["claims"][claim["id"]]
    assert (
        "not available"
        in results["claims"][claim["id"]]["excluded_numerical"]["reason"]
    )


def test_exploratory_recipe_verdicts_are_preserved_without_replication_counts():
    entry = {
        "total": {
            "cells": 24,
            "supporting_cells": 23,
            "supported_when": "transplant_verdict_holds",
        },
        "by_model": {"gpt2": {"cells": 12}},
    }
    original = {"claims": {"claim": entry}}
    before = deepcopy(original)
    reviewed = reviewed_results(original, {"claims": [{"id": "claim"}]})
    assert original == before
    assert reviewed["claims"]["claim"]["total"] is None
    assert reviewed["claims"]["claim"]["excluded"]["summary"] == entry["total"]
    assert reviewed_results(reviewed, {"claims": [{"id": "claim"}]}) == reviewed


def _descriptive_claim():
    return {
        "id": "descriptive",
        "comparand": {
            "cells": ["a", "b"],
            "unit": "decision",
            "significance": False,
            "cell_logic": "enumerated",
        },
        "evidence": {
            "paper": {"values": {"a": True, "b": True}},
            "official_code": {"values": {"a": True, "b": True}},
            "replica": {
                "provenance": "independent_compute_from_author_predictors",
                "consumes_official_results": False,
                "values": {"a": True, "b": False},
            },
        },
        "expected_matches": {},
    }


def test_a_descriptive_label_does_not_verify_a_transplant_protocol():
    entry = {
        "total": {
            "cells": 4,
            "supporting_cells": 4,
            "supported_when": "transplant_verdict_holds",
        },
        "by_model": {},
    }
    results = {"claims": {"descriptive": deepcopy(entry), "tested": deepcopy(entry)}}
    tested = {
        "id": "tested",
        "comparand": {"unit": "decision", "significance": {"positive_values": [True]}},
    }
    reviewed = reviewed_results(results, {"claims": [_descriptive_claim(), tested]})
    assert reviewed["claims"]["descriptive"]["total"] is None
    assert reviewed["claims"]["descriptive"]["excluded"]["summary"] == entry["total"]
    assert reviewed["claims"]["tested"]["total"] is None


def test_a_descriptive_recipe_replication_is_scoped_from_the_claim_tags():
    """Own corpora are listed and not counted; other models move to robustness."""
    claim = _descriptive_claim()
    claim["comparand"]["cell_tags"] = {
        "a": {"corpus": "meco_en", "lm": "mgpt"},
        "b": {"corpus": "meco_ge", "lm": "mgpt"},
    }
    points = [
        {"dataset": "provo", "lm": "mgpt", "significant": True},
        {"dataset": "meco_de", "lm": "mgpt", "significant": True},
        {"dataset": "meco_no", "lm": "mgpt", "significant": True},
        {"dataset": "provo", "lm": "gpt2", "significant": True},
    ]
    entry = {
        "total": {
            "cells": 4,
            "supporting_cells": 4,
            "supported_when": "transplant_verdict_holds",
            "points": points,
        }
    }
    reviewed = reviewed_results({"claims": {"descriptive": entry}}, {"claims": [claim]})
    assert reviewed["claims"]["descriptive"]["total"] is None
    total = reviewed["claims"]["descriptive"]["excluded"]["summary"]
    assert (total["cells"], total["supporting_cells"], total["own_corpus_cells"]) == (
        2,
        2,
        1,
    )
    assert reviewed["claims"]["descriptive"]["robustness"]["gpt2"]["cells"] == 1
    assert reviewed["claims"]["descriptive"]["paper_lms"] == ["mgpt"]


def test_load_results_assembles_study_files_and_honors_explicit_registry(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(replication_results, "STUDIES_DIR", tmp_path)
    monkeypatch.setattr(replication_results, "STUDY_IDS", ("alpha2020x", "beta2021y"))
    registry = {"claims": []}
    calls = []
    monkeypatch.setattr(
        replication_results, "load_registry", lambda: calls.append(True) or registry
    )
    (tmp_path / "alpha2020x").mkdir()
    (tmp_path / "alpha2020x" / "replication_results.yaml").write_text(
        "schema: 4\nclaims: {}\nsources: [first]\n"
    )
    assert replication_results.load_results(registry)["sources"] == ["first"]
    assert calls == []
    (tmp_path / "beta2021y").mkdir()
    (tmp_path / "beta2021y" / "replication_results.yaml").write_text(
        "schema: 4\nclaims: {}\nsources: [second]\n"
    )
    assert replication_results.load_results()["sources"] == ["first", "second"]
    assert calls == [True]
    for study in ("alpha2020x", "beta2021y"):
        (tmp_path / study / "replication_results.yaml").unlink()
    assert replication_results.load_results(registry) == {
        "claims": {},
        "sources": [],
        "unmapped_contrasts": [],
    }


def test_save_results_files_claims_and_sources_by_study(tmp_path, monkeypatch):
    monkeypatch.setattr(replication_results, "STUDIES_DIR", tmp_path)
    monkeypatch.setattr(replication_results, "STUDY_IDS", ("alpha2020x", "beta2021y"))
    for study in ("alpha2020x", "beta2021y"):
        (tmp_path / study).mkdir()
    results = {
        "claims": {
            "alpha2020x_claim": {"total": {"cells": 1}, "source_run": "alpha_run"},
            "beta2021y_claim": {"total": {"cells": 2}},
        },
        "sources": [
            "alpha_run",
            "beta2021y_native/table.tsv",
            "shared/beta2021y_claim",
        ],
    }
    written = replication_results.save_results(results)
    assert sorted(p.parent.name for p in written) == ["alpha2020x", "beta2021y"]
    assembled = replication_results.load_raw_results()
    assert assembled["claims"] == results["claims"]
    assert assembled["sources"] == sorted(results["sources"])
    alpha = replication_results.load_raw_results.__globals__["yaml"].safe_load(
        (tmp_path / "alpha2020x" / "replication_results.yaml").read_text()
    )
    assert alpha["sources"] == ["alpha_run"]
    with pytest.raises(ValueError, match="belongs to no study"):
        replication_results.save_results({"claims": {}, "sources": ["orphan_run"]})
