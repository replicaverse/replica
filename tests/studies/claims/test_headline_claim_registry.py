import re
import subprocess
from copy import deepcopy
from pathlib import Path

import pytest

from replica.dataset.measures import canonical_measure
from replica.studies.claims.registry import (
    BLOCKED_REASONS,
    CAVEAT_LABELS,
    CELL_TAG_KEYS,
    COMPARAND_LABELS,
    CORRECTION_KEYS,
    COVERAGE_LABELS,
    INDEPENDENT_REPLICA_PROVENANCE,
    LABEL_CONTRACT_VERSION,
    PROVENANCE_LABELS,
    REPLICA_PROVENANCE,
    RESULT_LABELS,
    VERDICT_LABELS,
    VERDICT_ORDER,
    VERDICT_SHORT,
    agreement_by_tag,
    cell_agreements,
    cell_tags,
    cells_table,
    claim_status,
    claims_for_study,
    compare_sources,
    comparison_summary,
    corrected_values,
    covered_subset,
    render_readme_claims,
    specification_line,
    verdict,
)
from replica.studies.packages import study_ids


def test_covered_subset_preserves_missing_cells_in_each_source():
    claim = {
        "comparand": {"cells": ["a", "b", "c"]},
        "evidence": {
            "paper": {"values": {"a": 1, "b": 2, "c": 3}},
            "replica": {"status": "partial", "values": {"a": 1, "b": 2}},
            "repetition": {"status": "partial", "values": {"b": 20, "c": 30}},
            "official_code": {"status": "missing", "values": None},
        },
    }
    original = deepcopy(claim)
    covered = covered_subset(claim)
    assert covered["comparand"]["cells"] == ["a", "b"]
    assert covered["evidence"] == {
        "paper": {"values": {"a": 1, "b": 2}},
        "replica": {"status": "available", "values": {"a": 1, "b": 2}},
        "repetition": {"status": "partial", "values": {"b": 20}},
        "official_code": {"status": "missing", "values": None},
    }
    assert claim == original


@pytest.mark.artifact
def test_every_active_study_has_a_headline_claim(claim_registry):
    registry = claim_registry
    active = set(registry["scope"]["active_studies"])
    covered = {claim["study"] for claim in registry["claims"]}
    assert covered == active
    deferred = set(registry["scope"]["deferred"])
    assert deferred == set(study_ids()) - active
    # The integration tree must not silently lose an existing study registry.
    if (Path(__file__).resolve().parents[3] / "release/public.toml").is_file():
        assert deferred == {"hopton2026speech", "hopton2026unigram"}
    assert active.isdisjoint(deferred)


@pytest.mark.artifact
def test_every_active_study_has_auditable_sources(claim_registry):
    registry = claim_registry
    active = set(registry["scope"]["active_studies"])
    assert set(registry["study_sources"]) == active
    repo = Path(__file__).resolve().parents[3]
    for study, sources in registry["study_sources"].items():
        assert sources["paper"]["url"].startswith("https://"), study
        document = sources["paper"]["audit_document"]
        assert set(document) == {"url", "version", "format", "sha256"}, study
        assert document["url"].startswith("https://"), study
        assert document["version"].strip(), study
        assert document["format"] in {"pdf", "xml"}, study
        assert re.fullmatch(r"[0-9a-f]{64}", document["sha256"]), study
        assert sources["official_code"]["url"].startswith("https://"), study
        assert (repo / sources["replica"]["implementation"]).is_file(), study
        assert (repo / sources["replica"]["test"]).is_file(), study


@pytest.mark.artifact
def test_every_claim_uses_a_renderable_coverage_status(claim_registry):
    for claim in claim_registry["claims"]:
        assert claim["coverage"] in COVERAGE_LABELS, claim["id"]


def test_sign_match_kind_compares_directions_of_stored_numbers():
    """``kind: sign`` lets a direction claim carry numbers, not booleans."""
    claim = {
        "id": "synthetic",
        "comparand": {"cells": ["a", "b"], "match": {"kind": "sign"}},
        "evidence": {
            "paper": {"values": {"a": 0.31, "b": -2.0}},
            "replica": {"values": {"a": 0.0007, "b": -0.4}},
            "official_code": {"values": {"a": -0.31, "b": -2.0}},
        },
    }
    assert compare_sources(claim, "paper", "replica") is True
    assert compare_sources(claim, "paper", "official_code") is False


@pytest.mark.artifact
def test_claim_ids_and_declared_cells_are_unique(claim_registry):
    registry = claim_registry
    ids = [claim["id"] for claim in registry["claims"]]
    assert len(ids) == len(set(ids))
    for claim in registry["claims"]:
        cells = claim["comparand"]["cells"]
        assert cells
        assert len(cells) == len(set(cells)), claim["id"]


@pytest.mark.artifact
def test_value_mappings_cover_declared_cells_unless_partial(claim_registry):
    """A lone source cannot evade cell validation for lack of a comparison."""
    for claim in claim_registry["claims"]:
        expected = set(claim["comparand"]["cells"])
        for source, evidence in claim["evidence"].items():
            if "values" in evidence:
                supplied = set(evidence["values"])
                assert supplied <= expected, (claim["id"], source)
                if evidence.get("status") != "partial":
                    assert supplied == expected, (claim["id"], source)
                compare_sources(claim, source, source)


@pytest.mark.artifact
def test_evidence_schema_has_no_yaml_flow_mapping_fragments(claim_registry):
    """An unquoted comma in a YAML flow scalar must not become a fake key."""
    allowed = {
        "status",
        "source",
        "values",
        "provenance",
        "consumes_official_results",
        "link",
        "artifact",
        "decisions",
        "decision_extract",
    }
    for claim in claim_registry["claims"]:
        assert (
            {
                "paper",
                "official_code",
                "replica",
            }
            <= set(claim["evidence"])
            <= {"paper", "official_code", "replica", "repetition"}
        ), claim["id"]
        for source, evidence in claim["evidence"].items():
            assert set(evidence) <= allowed, (
                claim["id"],
                source,
                set(evidence) - allowed,
            )


@pytest.mark.artifact
def test_declared_pairwise_matches_require_direct_paper_replica_agreement(
    claim_registry,
):
    """Chained agreements never replace a direct paper-to-REPLICA check."""
    for claim in claim_registry["claims"]:
        expected = claim["expected_matches"]
        observed = {
            "paper_official_code": compare_sources(claim, "paper", "official_code"),
            "official_code_replica": compare_sources(claim, "official_code", "replica"),
            "paper_replica": compare_sources(claim, "paper", "replica"),
        }
        assert observed == expected, claim["id"]
        if observed["paper_official_code"] and observed["official_code_replica"]:
            assert observed["paper_replica"], (
                f"{claim['id']}: paper and official code agree, and official code "
                "and REPLICA agree, but the direct paper-to-REPLICA check does not"
            )


@pytest.mark.artifact
def test_replica_provenance_cannot_hide_a_result_reader(claim_registry):
    for claim in claim_registry["claims"]:
        replica = claim["evidence"]["replica"]
        provenance = replica["provenance"]
        assert provenance in REPLICA_PROVENANCE, claim["id"]
        consumes_official = replica["consumes_official_results"]
        if consumes_official:
            assert provenance == "official_result_reader", claim["id"]
        if provenance in INDEPENDENT_REPLICA_PROVENANCE:
            assert not consumes_official, claim["id"]


@pytest.mark.artifact
def test_available_repetition_declares_computation_provenance(claim_registry):
    """Repetition counts apply the same independent-computation contract as replica."""
    for claim in claim_registry["claims"]:
        evidence = claim["evidence"].get("repetition")
        if evidence is None or evidence["status"] not in {"available", "partial"}:
            continue
        assert evidence["provenance"] in REPLICA_PROVENANCE, claim["id"]
        assert type(evidence["consumes_official_results"]) is bool, claim["id"]
        if evidence["consumes_official_results"]:
            assert evidence["provenance"] == "official_result_reader", claim["id"]
        if evidence["provenance"] in INDEPENDENT_REPLICA_PROVENANCE:
            assert not evidence["consumes_official_results"], claim["id"]
        if evidence["status"] == "available":
            assert evidence.get("values"), claim["id"]


@pytest.mark.artifact
def test_only_independent_computations_may_be_called_independent_paper_replica(
    claim_registry,
):
    for claim in claim_registry["claims"]:
        replica = claim["evidence"]["replica"]
        independent = replica["provenance"] in INDEPENDENT_REPLICA_PROVENANCE
        if (
            claim["coverage"] == "complete"
            and claim["expected_matches"]["paper_replica"]
        ):
            assert independent, claim["id"]


@pytest.mark.artifact
def test_coverage_label_is_compatible_with_the_evidence_state(claim_registry):
    """Each label's necessary conditions on the evidence statuses.

    Coverage also depends on how much of the paper's claim the registered cells
    span, which the registry does not encode, so these are one-sided checks: a
    given evidence state can still admit more than one label.
    """
    for claim in claim_registry["claims"]:
        coverage = claim["coverage"]
        official = claim["evidence"]["official_code"]["status"]
        replica = claim["evidence"]["replica"]
        status, provenance = replica["status"], replica["provenance"]
        where = f"{claim['id']}: official={official} replica={status}/{provenance}"
        if official == "missing":
            assert provenance != "official_result_reader", (
                f"{where}: reads official results that are declared missing"
            )
        if official == "missing" and status == "missing":
            assert coverage == "missing", where
        if coverage == "missing":
            assert status == "missing", where
        elif coverage == "complete":
            assert status == "available", where
            assert provenance in INDEPENDENT_REPLICA_PROVENANCE, where
        elif coverage == "complete_official_result_reader":
            assert status == "available", where
            assert provenance == "official_result_reader", where
        elif coverage == "partial":
            assert status in {"available", "partial"}, where
        elif coverage == "incomplete_replica":
            assert official in {"available", "partial"} and status == "partial", where
        elif coverage == "incomplete_official_and_replica":
            assert official in {"partial", "missing"}, where
            assert status in {"partial", "missing"}, where
            assert "partial" in {official, status}, where


@pytest.mark.artifact
def test_evidence_links_resolve(claim_registry):
    """A linked artifact is reachable: a tracked repo file or a pinned URL."""
    repo = Path(__file__).resolve().parents[3]
    tracked = set(
        subprocess.run(
            ["git", "-C", str(repo), "ls-files"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.split()
    )
    for claim in claim_registry["claims"]:
        for source, evidence in claim["evidence"].items():
            link = evidence.get("link")
            if link is None:
                continue
            assert set(link) == {"label", "url"}, (claim["id"], source)
            assert link["label"], (claim["id"], source)
            url = link["url"]
            where = f"{claim['id']} [{source}] {url}"
            if url.startswith("http"):
                assert url.startswith("https://"), where
                if "github.com" in url and "/blob/" in url:
                    assert re.search(r"/blob/[0-9a-f]{40}/", url), where
                    assert re.search(r"#L\d+(?:-L\d+)?$", url), where
                if "github.com" in url and "/tree/" in url:
                    # A repository-root tree ends at the commit, with no path.
                    assert re.search(r"/tree/[0-9a-f]{40}(/|$)", url), where
            else:
                from replica.studies.records.files import hub_url

                parts = Path(url).parts
                if (
                    len(parts) > 5
                    and parts[:3] == ("src", "replica", "studies")
                    and parts[4] == "results"
                ):
                    remote = hub_url(parts[3], "/".join(parts[5:]))
                    assert remote and re.search(
                        r"/(?:blob|tree)/[0-9a-f]{40}/", remote
                    ), where
                else:
                    assert url in tracked, f"{where}: not a tracked repo file"


def test_provenance_labels_cover_every_provenance():
    assert set(PROVENANCE_LABELS) == REPLICA_PROVENANCE


def test_verdict_vocabularies_agree():
    assert set(VERDICT_LABELS) == set(VERDICT_SHORT) == set(VERDICT_ORDER)
    assert len(VERDICT_ORDER) == len(set(VERDICT_ORDER))
    for reason in BLOCKED_REASONS:
        assert f"blocked_{reason}" in VERDICT_LABELS, reason


@pytest.mark.artifact
def test_public_label_contract_v1_is_stable_and_orthogonal(claim_registry):
    """Public wording changes require an explicit contract-version bump."""
    assert LABEL_CONTRACT_VERSION == 9
    assert RESULT_LABELS == {
        "reproduced": "Reproduced",
        "reobtained": "Matches released results",
        "reproduced_as_released": "Repeated as released",
        "repeated": "Repeated; correction pending",
        "reference_only": "Corrected reference only",
        "partial": "Partially reproduced",
        "not_reproduced": "Not reproduced",
        "not_evaluated": "Not yet evaluated",
    }
    assert COMPARAND_LABELS == {
        "numeric": "Numeric values",
        "sign": "Effect directions",
        "exact": "Decisions",
    }
    assert COVERAGE_LABELS == {
        "complete": "whole claim covered",
        "complete_official_result_reader": "whole claim, from released results",
        "partial": "narrowed scope",
        "incomplete_official_and_replica": "evidence incomplete on both sides",
        "incomplete_replica": "REPLICA evidence incomplete",
        "missing": "no REPLICA evidence",
    }
    assert PROVENANCE_LABELS == {
        "independent_compute": "independent recompute, REPLICA predictors",
        "independent_compute_from_author_predictors": (
            "independent refit, authors' predictors"
        ),
        "independent_compute_from_retained_predictors": (
            "independent refit, retained predictors"
        ),
        "from_spec_compute": "from-spec recompute, REPLICA predictors",
        "official_result_reader": "reads official results",
        "implemented_no_result": "implemented, result pending",
        "implemented_unavailable_inputs": "implementation only",
        "missing": "not implemented",
    }
    assert CAVEAT_LABELS == {
        "none": "None recorded",
        "mismatch_root_caused": "Paper/REPLICA mismatch, cause identified",
        "mismatch_cause_unknown": "Paper/REPLICA mismatch, cause unknown",
        "released_defect_changes_decision": (
            "Correcting a released-code divergence changes decisions"
        ),
        "released_divergence_no_change": (
            "Released-code divergence found; registered decisions unchanged"
        ),
        "correction_not_fully_evaluated": "Correction not fully evaluated",
        "blocked_missing_data": "Blocked: a required input has no released source",
        "blocked_gated_data": (
            "Blocked: a required input is access-controlled and no copy is held"
        ),
        "blocked_underspecified_method": (
            "Blocked: the paper or release omits data or parameter details the "
            "reproduction needs"
        ),
        "blocked_missing_official_analysis": (
            "Blocked: the released code lacks the analysis that produces the "
            "claim's cells"
        ),
        "not_yet_tested": "No REPLICA result yet",
        "evidence_incomplete": "Paper-to-REPLICA comparison incomplete",
        "partial_coverage": "REPLICA covers a subset of the registered cells",
    }
    for claim in claim_registry["claims"]:
        status = claim_status(claim)
        assert status["contract_version"] == LABEL_CONTRACT_VERSION
        assert status["result"]["id"] in RESULT_LABELS
        assert status["scope"]["id"] in COMPARAND_LABELS
        assert status["scope"]["coverage_id"] in COVERAGE_LABELS
        assert status["evidence"]["id"] in PROVENANCE_LABELS
        assert status["caveats"]
        assert all(item["id"] in CAVEAT_LABELS for item in status["caveats"])


@pytest.mark.artifact
def test_public_result_never_combines_reproduced_and_failed_flags(claim_registry):
    """Released-result agreement cannot create a contradictory public Result."""
    for claim in claim_registry["claims"]:
        flags = set(verdict(claim))
        result = claim_status(claim)["result"]["id"]
        fixed = corrected_values(claim)
        effective = (
            {
                **claim,
                "evidence": {
                    **claim["evidence"],
                    "replica": {**claim["evidence"]["replica"], "values": fixed},
                },
            }
            if fixed is not None
            else claim
        )
        if (claim.get("reproduction_review") or {}).get(
            "status"
        ) == "fidelity_unresolved":
            assert result == "reference_only", claim["id"]
        elif fixed is None and any(claim.get(key) for key in CORRECTION_KEYS):
            expected = (
                "repeated"
                if compare_sources(claim, "paper", "replica") is True
                else "not_evaluated"
            )
            assert result == expected, claim["id"]
        elif (
            fixed is not None and compare_sources(effective, "paper", "replica") is True
        ):
            assert result == "reproduced", claim["id"]
        elif flags & {"failed_mismatch", "failed_untraced", "failed_defect"}:
            assert result in {
                "not_reproduced",
                "partial",
                "reproduced_as_released",
            }, claim["id"]


@pytest.mark.artifact
def test_reading_released_results_is_never_labelled_reproduced(claim_registry):
    """Re-obtaining the paper's numbers from released files is not a reproduction."""
    for claim in claim_registry["claims"]:
        result = claim_status(claim)["result"]["id"]
        reader = claim["evidence"]["replica"]["provenance"] == "official_result_reader"
        if reader:
            assert result != "reproduced", claim["id"]
        elif result == "reobtained":
            raise AssertionError(f"{claim['id']}: reobtained without a result reader")


@pytest.mark.artifact
def test_every_claim_has_a_computable_verdict(claim_registry):
    """A non-empty, known, weakest-first label set for every claim."""
    for claim in claim_registry["claims"]:
        labels = verdict(claim)
        assert labels, claim["id"]
        assert all(label in VERDICT_LABELS for label in labels), (
            claim["id"],
            labels,
        )
        assert labels == sorted(labels, key=VERDICT_ORDER.index), claim["id"]


@pytest.mark.artifact
def test_every_claim_has_a_plain_summary_that_explains_combined_verdicts(
    claim_registry,
):
    """The site must explain evidence that is simultaneously positive and negative."""
    for claim in claim_registry["claims"]:
        summary = comparison_summary(claim)
        assert summary.endswith("."), claim["id"]
        assert "paper-to-REPLICA" in summary or "paper" in summary, claim["id"]
        labels = set(verdict(claim))
        if {"failed_defect", "reproduced_direction"} <= labels:
            assert "recovers" in summary and "does not reproduce" in summary, (
                claim["id"],
                summary,
            )


@pytest.mark.artifact
def test_failure_records_accompany_every_disagreement(claim_registry):
    """A paper-to-REPLICA disagreement is only reportable with its record.

    ``traced: false`` marks a stable mismatch with no defect identified on
    either side, so the note has to say what was checked and ruled out; a
    claim without a disagreement must not carry a failure record that would
    render as one.
    """
    for claim in claim_registry["claims"]:
        compared = covered_subset(claim) or claim
        disagrees = compare_sources(compared, "paper", "replica") is False
        failure = claim.get("failure")
        if disagrees:
            assert failure is not None, claim["id"]
            assert set(failure) == {"traced", "note"}, claim["id"]
            assert isinstance(failure["traced"], bool), claim["id"]
            assert failure["note"].strip(), claim["id"]
        else:
            assert failure is None, claim["id"]


@pytest.mark.artifact
def test_blocked_markers_use_the_enum_and_stay_current(claim_registry):
    """Structured blockers: an enum reason, prose detail, and no stale marker.

    A completed paper-to-REPLICA agreement on the declared cells contradicts
    an open blocker at the same comparand, so the two cannot coexist.
    """
    for claim in claim_registry["claims"]:
        blocked = claim.get("blocked")
        if blocked is None:
            continue
        assert set(blocked) == {"reason", "detail"}, claim["id"]
        assert blocked["reason"] in BLOCKED_REASONS, claim["id"]
        assert blocked["detail"].strip(), claim["id"]
        assert compare_sources(claim, "paper", "replica") is not True, claim["id"]


@pytest.mark.artifact
def test_not_attempted_claims_carry_no_replica_evidence(claim_registry):
    for claim in claim_registry["claims"]:
        if "not_attempted" not in verdict(claim):
            continue
        replica = claim["evidence"]["replica"]
        assert replica["status"] == "missing", claim["id"]
        assert "values" not in replica, claim["id"]


@pytest.mark.artifact
def test_residual_labels_never_share_a_verdict(claim_registry):
    """Pending and not-attempted appear only alone.

    They are residual states; a claim that already carries a settled outcome
    (a decision-changing defect, a blocker, an agreement, a disagreement)
    must not also read as work in flight.
    """
    for claim in claim_registry["claims"]:
        labels = set(verdict(claim))
        if labels & {"pending", "not_attempted"}:
            assert len(labels) == 1, (claim["id"], labels)


@pytest.mark.artifact
def test_failed_defect_tracks_the_decision_changing_annotations(claim_registry):
    """The defect label appears exactly when an annotation changes decisions."""
    for claim in claim_registry["claims"]:
        changing = [
            claim.get(key)
            for key in ("predictor_correction", "test_divergence", "method_defect")
            if (claim.get(key) or {}).get("changes_registered_decisions")
        ]
        assert ("failed_defect" in verdict(claim)) == bool(changing), claim["id"]


@pytest.mark.artifact
def test_official_and_replica_evidence_render_something(claim_registry):
    """Every rendered bullet needs a link, prose, or both."""
    for claim in claim_registry["claims"]:
        for source in ("official_code", "replica"):
            evidence = claim["evidence"][source]
            assert evidence.get("source") or evidence.get("link"), (claim["id"], source)


# Restating the anchor renders the same sentence twice, so a paper source has to
# carry content words the anchor does not. Locators and stopwords do not count.
_FILLER = frozenset(
    """a an and are as at both by for in is it its of on or that the their these
    this those to with published publishes paper papers report reports reported
    section sections figure figures table tables appendix see also""".split()
)


def _content_words(text: str) -> set[str]:
    return {word for word in re.findall(r"[a-z]+", text.lower())} - _FILLER


@pytest.mark.artifact
def test_paper_source_adds_to_the_anchor(claim_registry):
    """A paper source that only restates the anchor is rendered twice."""
    for claim in claim_registry["claims"]:
        source = claim["evidence"]["paper"].get("source")
        if source is None:
            continue
        added = _content_words(source) - _content_words(claim["anchor"])
        assert len(added) >= 2, (
            f"{claim['id']}: paper source adds only {sorted(added)} to the "
            f"anchor {claim['anchor']!r}"
        )


@pytest.mark.artifact
def test_page_locators_link_the_exact_audited_document(claim_registry):
    """Specific PDF pages must never link only to a mutable landing page."""
    registry = claim_registry
    for claim in registry["claims"]:
        if "PDF page" not in claim["anchor"]:
            continue
        document = registry["study_sources"][claim["study"]]["paper"]["audit_document"]
        assert document["format"] == "pdf", claim["id"]
        rendered = render_readme_claims(claim["study"], registry)
        expected = f"]({document['url']})"
        claim_item = rendered.split(f"**{claim['claim']}**", 1)[1]
        claim_item = re.split(r"\n\n\d+\. \*\*", claim_item, maxsplit=1)[0]
        assert expected in claim_item, claim["id"]


@pytest.mark.artifact
def test_every_paper_official_match_tracks_independent_replica_status(claim_registry):
    """Completed disagreements require a recorded failure."""
    for claim in claim_registry["claims"]:
        if compare_sources(claim, "paper", "official_code") is not True:
            continue
        if compare_sources(claim, "paper", "replica") is True:
            continue
        assert claim["gap"], (
            f"{claim['id']}: paper and official code agree, but the missing direct "
            "paper-to-REPLICA check has no tracked implementation gap"
        )
        assert claim["evidence"]["replica"]["status"] in {
            "partial",
            "missing",
        } or claim.get("failure"), claim["id"]


@pytest.mark.artifact
def test_predictor_corrections_state_both_the_bug_and_its_consequence(claim_registry):
    """A confirmed predictor defect requires evidence of the correction's effect."""
    annotated = [c for c in claim_registry["claims"] if "predictor_correction" in c]
    studies = {claim["study"] for claim in annotated}
    assert studies == {
        "giulianelli2023infovalue",
        "kuribayashi2025internal",
        "nair2026clozing",
        "pimentel2023anticipation",
        "wilcox2023surprisal",
        "giulianelli2024generalized",
    } & set(claim_registry["scope"]["active_studies"]), studies
    required = {
        "divergence",
        "effect",
        "confirmed_by_authors",
        "changes_registered_decisions",
    }
    for claim in annotated:
        correction = claim["predictor_correction"]
        assert required <= set(correction) <= required | {"artifacts", "corrected"}, (
            claim["id"],
            set(correction),
        )
        corrected = correction.get("corrected")
        if corrected is not None:
            # The decisions the Reproduction column counts: one per cell, with
            # the table they are read from.
            assert (
                {"values", "artifact"}
                <= set(corrected)
                <= {"values", "artifact", "decision_extract"}
            ), claim["id"]
            assert set(corrected["values"]) == set(claim["comparand"]["cells"]), claim[
                "id"
            ]
            assert correction["changes_registered_decisions"] == (
                corrected["values"] != claim["evidence"]["replica"]["values"]
            ), claim["id"]
        assert correction["divergence"].strip() and correction["effect"].strip()
        assert isinstance(correction["confirmed_by_authors"], bool), claim["id"]
        # True fails the described method, False leaves the decisions
        # standing, None means the corrected rerun is still owed.
        assert correction["changes_registered_decisions"] in (True, False, None)
        # The paper words bare number regeneration as re-obtaining, never as
        # reproducing, for exactly these studies.
        source = claim["evidence"]["replica"].get("source", "")
        assert "eproduc" not in source, (claim["id"], source)


@pytest.mark.artifact
def test_method_defects_state_both_the_defect_and_its_consequence(claim_registry):
    """A defect in the method itself needs the defect and its consequence.

    Unlike a predictor correction, code and paper agree here: the released
    method carries the defect, so the faithful rerun reproduces the published
    numbers while the corrected rerun says whether the decisions survive.
    """
    annotated = [c for c in claim_registry["claims"] if "method_defect" in c]
    assert {claim["study"] for claim in annotated} == {
        "giulianelli2023infovalue",
        "giulianelli2024tokenization",
        "meister2024similarity",
        "pimentel2023anticipation",
        "wilcox2023surprisal",
        "opedal2024context",
        "nair2026clozing",
        "oota2023linguistic",
        "xu2023linearity",
        "kuribayashi2021perplexity",
        "kuribayashi2025internal",
        "shain2024dissociation",
        "shain2024logarithmic",
        "re2025spatiotemporal",
    } & set(claim_registry["scope"]["active_studies"])
    required = {
        "divergence",
        "effect",
        "confirmed_by_authors",
        "changes_registered_decisions",
    }
    for claim in annotated:
        defect = claim["method_defect"]
        assert required <= set(defect) <= required | {"artifacts", "corrected"}, (
            claim["id"],
            set(defect),
        )
        assert defect["divergence"].strip() and defect["effect"].strip()
        assert isinstance(defect["confirmed_by_authors"], bool), claim["id"]
        assert defect["changes_registered_decisions"] in (True, False, None)


@pytest.mark.artifact
def test_test_divergences_state_both_the_defect_and_its_consequence(claim_registry):
    """A released test that is not the test its code names needs both halves.

    The consequence has to say what the claim does under each call, since a
    null result obtained from the weaker test is not the null result the paper
    describes.
    """
    annotated = [c for c in claim_registry["claims"] if "test_divergence" in c]
    assert {claim["study"] for claim in annotated} == {
        "clark2025wordentropy",
        "meister2024similarity",
        "wilcox2023surprisal",
    } & set(claim_registry["scope"]["active_studies"])
    required = {
        "divergence",
        "effect",
        "confirmed_by_authors",
        "changes_registered_decisions",
    }
    for claim in annotated:
        divergence = claim["test_divergence"]
        assert required <= set(divergence) <= required | {"artifacts", "corrected"}, (
            claim["id"],
            set(divergence),
        )
        corrected = divergence.get("corrected")
        if corrected is not None:
            assert (
                {"values", "artifact"}
                <= set(corrected)
                <= {"values", "artifact", "decision_extract"}
            ), claim["id"]
            assert set(corrected["values"]) == set(claim["comparand"]["cells"]), claim[
                "id"
            ]
            comparison = {
                **claim,
                "evidence": {
                    **claim["evidence"],
                    "corrected": {"values": corrected["values"]},
                },
            }
            assert divergence["changes_registered_decisions"] == (
                not compare_sources(comparison, "replica", "corrected")
            ), claim["id"]
        assert divergence["divergence"].strip() and divergence["effect"].strip()
        assert isinstance(divergence["confirmed_by_authors"], bool), claim["id"]
        assert divergence["changes_registered_decisions"] in (True, False, None)
        # The official-code evidence must record what the release's own call
        # yields, not what the corrected call yields, or the comparison
        # silently changes which test the paper is being held to.
        official = claim["evidence"]["official_code"]
        if official["status"] == "missing":
            # A defect can also occur in our reconstruction of an unreleased
            # analysis. Its reference refit must not become an exact repetition.
            assert claim["evidence"]["replica"]["provenance"] == "from_spec_compute"
            assert "REPLICA" in divergence["divergence"]


@pytest.mark.artifact
def test_a_claim_whose_cells_disagree_declares_how_its_cells_combine(claim_registry):
    """Losing one cell falsifies a ranking but not a claim quantified over a population."""
    for claim in claim_registry["claims"]:
        fixed = corrected_values(claim)
        effective = (
            {
                **claim,
                "evidence": {
                    **claim["evidence"],
                    "replica": {**claim["evidence"]["replica"], "values": fixed},
                },
            }
            if fixed is not None
            else claim
        )
        cells = cell_agreements(effective, "paper", "replica")
        if not cells or all(cells) or not any(cells):
            continue
        logic = claim["comparand"].get("cell_logic")
        assert logic in {"conjunctive", "enumerated"}, (
            f"{claim['id']}: agrees on {sum(cells)} of {len(cells)} cells, so it "
            "must declare comparand.cell_logic"
        )
        result = claim_status(claim)["result"]["id"]
        if (claim.get("reproduction_review") or {}).get(
            "status"
        ) == "fidelity_unresolved":
            expected = "reference_only"
        elif fixed is None and any(claim.get(key) for key in CORRECTION_KEYS):
            expected = (
                "repeated"
                if compare_sources(covered_subset(claim) or claim, "paper", "replica")
                is True
                else "not_evaluated"
            )
        elif (
            fixed is not None
            and compare_sources(covered_subset(claim) or claim, "paper", "replica")
            is True
        ):
            expected = "reproduced_as_released"
        else:
            expected = "not_reproduced" if logic == "conjunctive" else "partial"
        assert result == expected, claim["id"]


@pytest.mark.artifact
def test_the_result_carries_its_cell_count_and_an_evidence_marker(claim_registry):
    """Every claim shows what evidence its result rests on, beside the result."""
    for claim in claim_registry["claims"]:
        status = claim_status(claim)
        marker = status["marker"]
        assert marker["field"] == "coverage"
        assert marker["label"] == COVERAGE_LABELS[claim["coverage"]]
        assert marker["reading"]
        result = status["result"]
        expected = claim["comparand"]["cells"]
        if claim["evidence"]["replica"]["status"] == "partial":
            values = corrected_values(claim)
            if values is None:
                values = claim["evidence"]["replica"].get("values", {})
            expected = [cell for cell in expected if cell in values]
        assert result["cells_total"] in {0, len(expected)}, claim["id"]
        assert 0 <= result["cells_agree"] <= result["cells_total"]


@pytest.mark.artifact
def test_every_cell_is_tagged_with_known_dimensions(claim_registry):
    """A cell says which measure, corpus, model and language it stands for,
    or says explicitly that it stands for none, so counts can be split by
    measure without guessing from cell names."""
    for claim in claim_registry["claims"]:
        tags = claim["comparand"]["cell_tags"]
        assert set(tags) == set(claim["comparand"]["cells"]), claim["id"]
        for cell, tag in tags.items():
            assert set(tag) <= CELL_TAG_KEYS, (
                claim["id"],
                cell,
                set(tag) - CELL_TAG_KEYS,
            )
            assert all(isinstance(v, str) and v for v in tag.values()), (
                claim["id"],
                cell,
            )


@pytest.mark.artifact
def test_a_cell_measure_is_a_canonical_name(claim_registry):
    """Tags share the loaders' vocabulary, written in the canonical spelling."""
    for claim in claim_registry["claims"]:
        for cell, tag in claim["comparand"]["cell_tags"].items():
            if "dv" in tag:
                assert canonical_measure(tag["dv"]) == tag["dv"], (claim["id"], cell)


@pytest.mark.artifact
def test_per_measure_agreement_partitions_the_tagged_cells(claim_registry):
    for claim in claim_registry["claims"]:
        status = claim_status(claim)["result"]
        by_dv = status["by_dv"]
        tagged = [t for t in cell_tags(claim).values() if "dv" in t]
        if status["cells_total"] == 0:
            assert by_dv == {}, claim["id"]
            continue
        observed_tags = cell_tags(claim)
        if claim["evidence"]["replica"]["status"] == "partial":
            values = corrected_values(claim)
            if values is None:
                values = claim["evidence"]["replica"].get("values", {})
            observed_tags = {c: t for c, t in observed_tags.items() if c in values}
        observed = [t for t in observed_tags.values() if "dv" in t]
        assert sum(b["cells_total"] for b in by_dv.values()) == len(observed), claim[
            "id"
        ]
        assert sum(b["cells_agree"] for b in by_dv.values()) <= status["cells_agree"]
        official = agreement_by_tag(claim, "official_code", "dv")
        if tagged and compare_sources(claim, "paper", "official_code") is not None:
            assert sum(b["cells_total"] for b in official.values()) == len(tagged)
        else:
            assert official == {}, claim["id"]


def test_cells_table_shows_every_source_and_names_an_absence():
    """A reader sees each cell's three values; a source without one reads n/a, never blank."""
    claim = {
        "comparand": {"cells": ["en_linear", "ppl_delta"]},
        "evidence": {
            "paper": {"values": {"en_linear": "+", "ppl_delta": -0.69}},
            "official_code": {"status": "partial"},
            "replica": {"values": {"en_linear": True}},
        },
    }
    lines = cells_table(claim, indent="")
    assert lines[0] == "| cell | paper | official code | REPLICA |"
    assert lines[2] == "| en linear | + | n/a | yes |"
    assert lines[3] == "| ppl delta | -0.69 | n/a | n/a |"


@pytest.mark.artifact
def test_every_detailed_claim_carries_its_cells_table(claim_registry):
    """The detailed website view retains every cell even with a compact README."""
    registry = claim_registry
    for study in sorted({c["study"] for c in registry["claims"]}):
        block = render_readme_claims(study, registry, detail=True)
        claims = claims_for_study(study, registry)
        cells = sum(len(c["comparand"]["cells"]) for c in claims)
        assert block.count("\n     | ") == cells + len(claims), study


def test_compact_cells_do_not_fill_missing_corrections_with_old_results():
    claim = {
        "comparand": {"cells": ["linear"]},
        "evidence": {
            "paper": {"values": {"linear": "+"}},
            "repetition": {"values": {"linear": "ns"}},
            "replica": {"values": {"linear": "+"}},
        },
        "method_defect": {"divergence": "Uncorrected context"},
    }
    rows = cells_table(claim, indent="", compact=True)
    assert (
        rows[0]
        == "| cell | paper | original-data repetition | original-data reanalysis |"
    )
    assert rows[-1] == "| linear | + | ns | n/a |"
    claim["method_defect"]["corrected"] = {"values": {"linear": "-"}}
    assert cells_table(claim, indent="", compact=True)[-1] == "| linear | + | ns | - |"


def test_specification_line_escapes_its_placeholders():
    """``<response>`` is prose; unescaped it is an HTML tag on GitHub and the site."""
    claim = {
        "study": "wilcox2023surprisal",
        "specification": {
            "evaluate": "study.evaluate(dv=<response>)",
            "analysis": "OLS",
            "test": "p < .05",
            "implemented_in": "src/replica/studies/wilcox2023surprisal/__init__.py",
        },
    }
    line = specification_line(claim)
    assert "dv=\\<response\\>" in line
    assert "p \\< .05" in line
    assert "<response>" not in line


@pytest.mark.artifact
def test_every_claim_specifies_its_computation_and_test(claim_registry):
    """Each claim states the call that produces its numbers, the analysis, the test and the code."""
    for claim in claim_registry["claims"]:
        spec = claim["specification"]
        assert set(spec) == {"evaluate", "analysis", "test", "implemented_in"}, claim[
            "id"
        ]
        for key, value in spec.items():
            assert isinstance(value, str) and value.strip(), (claim["id"], key)
        assert Path(spec["implemented_in"]).exists(), (
            claim["id"],
            spec["implemented_in"],
        )


@pytest.mark.artifact
def test_significance_rules_are_explicit_and_reference_declared_cells(claim_registry):
    for claim in claim_registry["claims"]:
        declaration = claim["comparand"]["significance"]
        rules = (
            declaration.get(
                "by_cell",
                {
                    key: declaration["positive_values"]
                    for key in declaration.get("cells", claim["comparand"]["cells"])
                },
            )
            if declaration
            else {}
        )
        assert set(rules) <= set(claim["comparand"]["cells"]), claim["id"]
        assert all(isinstance(values, list) and values for values in rules.values()), (
            claim["id"]
        )


@pytest.mark.artifact
def test_scoped_reference_evidence_has_its_own_artifact_and_never_claim_cells(
    claim_registry,
):
    from replica.studies.claims.values import validate_artifact_schema

    for claim in claim_registry["claims"]:
        for reference in claim.get("reference_evidence", []):
            assert reference["id"] and reference["scope"] and reference["source"]
            assert reference["values"]
            assert set(reference["stages"]) <= {"repetition", "reproduction"}
            assert set(reference["values"]) == set(reference["artifact"]["extract"])
            assert set(reference["values"]).isdisjoint(claim["comparand"]["cells"])
            assert "run" not in reference["artifact"]
            validate_artifact_schema(reference["artifact"])


def test_partial_values_preserve_known_cells_and_leave_missing_cells_unknown():
    from replica.studies.claims.registry import cells_table

    claim = {
        "id": "partial_japanese",
        "comparand": {
            "cells": ["english", "japanese_low", "japanese_high"],
            "match": {"kind": "numeric", "atol": 0.031},
        },
        "evidence": {
            "paper": {
                "status": "available",
                "values": {
                    "english": -0.87,
                    "japanese_low": 0.53,
                    "japanese_high": -0.68,
                },
            },
            "repetition": {
                "status": "partial",
                "values": {"japanese_low": 0.527392, "japanese_high": -0.668844},
            },
        },
    }
    assert compare_sources(claim, "paper", "repetition") is None
    assert cell_agreements(claim, "paper", "repetition") is None
    table = "\n".join(cells_table(claim))
    assert "| english | -0.87 | n/a | n/a | n/a |" in table
    assert "| japanese low | 0.53 | n/a | 0.527392 | n/a |" in table
    claim["evidence"]["repetition"]["values"]["japanese_low"] = -0.53
    assert compare_sources(claim, "paper", "repetition") is None


@pytest.mark.parametrize(
    "status,values",
    [
        ("available", {"japanese": 0.53}),
        ("partial", {"japanese": 0.53, "unknown": 0.2}),
    ],
)
def test_invalid_partial_or_complete_value_coverage_is_rejected(status, values):
    claim = {
        "id": "invalid_coverage",
        "comparand": {"cells": ["english", "japanese"], "match": {"kind": "numeric"}},
        "evidence": {
            "paper": {"values": {"english": -0.87, "japanese": 0.53}},
            "repetition": {"status": status, "values": values},
        },
    }
    with pytest.raises(ValueError, match="declared cells"):
        compare_sources(claim, "paper", "repetition")
    with pytest.raises(ValueError, match="declared cells"):
        cell_agreements(claim, "paper", "repetition")


def test_partial_replica_values_decide_the_covered_cells_only():
    from replica.studies.claims.registry import claim_status

    claim = {
        "id": "partial_covered",
        "coverage": "partial",
        "comparand": {
            "cells": ["a", "b", "c"],
            "cell_tags": {name: {"dv": "rt"} for name in ("a", "b", "c")},
            "match": {"kind": "exact"},
            "cell_logic": "conjunctive",
        },
        "evidence": {
            "paper": {
                "status": "available",
                "values": {"a": True, "b": True, "c": True},
            },
            "replica": {
                "status": "partial",
                "provenance": "independent_compute_from_author_predictors",
                "consumes_official_results": False,
                "values": {"a": True, "b": True},
            },
        },
    }
    status = claim_status(claim)
    assert status["result"]["id"] == "reproduced"
    assert (status["result"]["cells_agree"], status["result"]["cells_total"]) == (2, 2)
    assert "partial_coverage" in {caveat["id"] for caveat in status["caveats"]}
    claim["evidence"]["replica"]["values"]["b"] = False
    claim["failure"] = {"traced": True, "note": "boundary"}
    assert claim_status(claim)["result"]["id"] == "not_reproduced"
    claim["method_defect"] = {
        "divergence": "Changed analysis",
        "changes_registered_decisions": False,
        "corrected": {"values": {"a": True, "b": False}},
    }
    corrected = claim_status(claim)
    assert corrected["result"]["id"] == "not_reproduced"
    assert corrected["result"]["cells_total"] == 2
    assert corrected["result"]["cells_agree"] == 1
    assert corrected["scope"]["cells"] == 3
    assert "partial_coverage" in {c["id"] for c in corrected["caveats"]}
    claim["method_defect"]["corrected"]["values"]["b"] = True
    corrected = claim_status(claim)
    assert corrected["result"]["id"] == "reproduced"
    assert corrected["result"]["cells_total"] == 2
    assert corrected["result"]["cells_agree"] == 2
    assert corrected["result"]["by_dv"] == {"rt": {"cells_total": 2, "cells_agree": 2}}
    assert corrected["scope"]["cells"] == 3


@pytest.mark.artifact
def test_overview_decision_rules_and_scoped_targets_are_explicit(claim_registry):
    for claim in claim_registry["claims"]:
        rule = claim["comparand"].get("agreement_rule")
        assert rule in {None, "significance", "claim_decision"}
        for source, evidence in claim["evidence"].items():
            if "decisions" in evidence:
                assert source == "paper"
                assert "decision_extract" not in evidence
                assert set(evidence["decisions"]) == set(evidence["values"])
                assert set(evidence["decisions"].values()) <= {"+", "-", "ns"}
        for reference in claim.get("reference_evidence", []):
            comparison = reference.get("comparison")
            if not comparison:
                continue
            assert comparison["label"] and reference["scope"]
            assert reference["provenance"] in INDEPENDENT_REPLICA_PROVENANCE
            if "targets" in comparison:
                assert set(comparison) == {"targets", "label"}
                assert set(comparison["targets"]) == set(reference["values"])
                assert set(comparison["targets"].values()) <= set(
                    claim["comparand"]["cells"]
                )
                assert len(set(comparison["targets"].values())) == len(
                    reference["values"]
                )
            else:
                if "source_measure" in comparison:
                    assert set(comparison) == {
                        "source_measure",
                        "paper_extract",
                        "label",
                    }
                    assert comparison["source_measure"] == "fdurSPsummed"
                    assert all(
                        extract["where"]["response"] == comparison["source_measure"]
                        for extract in reference["artifact"]["extract"].values()
                    )
                else:
                    assert set(comparison) == {"measure", "paper_extract", "label"}
                    assert (
                        canonical_measure(comparison["measure"])
                        == comparison["measure"]
                    )
                assert reference.get("decision_extract")


@pytest.mark.artifact
def test_headline_orderings_pair_values_within_each_corpus_and_response(claim_registry):
    for claim in claim_registry["claims"]:
        ordering = claim["comparand"].get("headline_ordering")
        if not ordering:
            continue
        assert set(ordering) == {"label", "pairs"}
        assert ordering["label"] and ordering["pairs"]
        assert not claim["comparand"]["significance"]
        assert claim["comparand"]["match"]["kind"] == "numeric"
        paper = claim["evidence"]["paper"]["values"]
        used = []
        for corpus, pair in ordering["pairs"].items():
            assert set(pair) == {"higher", "lower"}
            assert pair["higher"] != pair["lower"]
            assert all(
                claim["comparand"]["cell_tags"][cell]["corpus"] == corpus
                for cell in pair.values()
            )
            assert paper[pair["higher"]] > paper[pair["lower"]]
            used.extend(pair.values())
        assert sorted(used) == sorted(claim["comparand"]["cells"])


@pytest.mark.artifact
def test_every_claim_declares_whether_its_cells_are_significance_decisions(
    claim_registry,
):
    for claim in claim_registry["claims"]:
        declaration = claim["comparand"]["significance"]
        assert declaration is False or isinstance(declaration, dict), claim["id"]
        if declaration:
            assert declaration["positive_values"]
            assert set(declaration.get("cells", claim["comparand"]["cells"])) <= set(
                claim["comparand"]["cells"]
            )
