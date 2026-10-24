"""Machine-readable audit of published headline-claim reproductions.

Each study package carries its own ``claims.yaml`` with the study's sources,
predictor needs and claims; :func:`load_registry` assembles them into one
registry in study order, so a tree with fewer study packages registers the
studies it has.

The registry deliberately distinguishes an independently computed REPLICA
result from a REPLICA reader or plotter over an official-code result. Both are
useful, but only the former is an independent paper-to-REPLICA reproduction.
A claim's ``coverage`` describes coverage of the abstract-level claim, not just
of the listed slice. ``anchor`` locates the claim in the paper; an evidence
``source`` says what that source contributes, and a paper source that only
restates the anchor is dropped rather than rendered twice.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

import yaml

from replica.studies.packages import STUDIES_DIR, study_ids

#: Schema version of the assembled registry and the date of its last audit.
REGISTRY_VERSION = 5
LAST_AUDITED = "2026-09-01"
CLAIMS_FILE = "claims.yaml"
STUDY_IDS = study_ids()

REPLICA_PROVENANCE = frozenset(
    {
        "independent_compute",
        "independent_compute_from_author_predictors",
        "independent_compute_from_retained_predictors",
        "official_result_reader",
        "from_spec_compute",
        "implemented_no_result",
        "implemented_unavailable_inputs",
        "missing",
    }
)
INDEPENDENT_REPLICA_PROVENANCE = frozenset(
    {
        "independent_compute",
        "independent_compute_from_author_predictors",
        "independent_compute_from_retained_predictors",
        "from_spec_compute",
    }
)

# Coverage states how much of the claim is covered; whether REPLICA rebuilt the
# predictors or reused the authors' is the provenance label's job, so "complete"
# must not also assert independence.
#: What evidence the comparison rests on, shown beside the result so a narrowed
#: scope is never read as a whole-claim reproduction. These are statements about
#: the evidence, never about the outcome.
COVERAGE_LABELS = {
    "complete": "whole claim covered",
    "complete_official_result_reader": "whole claim, from released results",
    "partial": "narrowed scope",
    "incomplete_official_and_replica": "evidence incomplete on both sides",
    "incomplete_replica": "REPLICA evidence incomplete",
    "missing": "no REPLICA evidence",
}

# Why a claim cannot proceed. Graded on purpose: ``missing_data`` is a whole
# absent input; ``underspecified_method`` is the far more common case of a
# present input whose data or parameter details are incomplete.
BLOCKED_REASONS = {
    "missing_data": "a required input has no released source",
    "gated_data": "a required input is access-controlled and no copy is held",
    "underspecified_method": (
        "the paper or release omits data or parameter details the reproduction needs"
    ),
    "missing_official_analysis": (
        "the released code lacks the analysis that produces the claim's cells"
    ),
}

# The outcome axis. Derived by verdict(), never authored in the YAML; a claim
# holds every label that applies, ordered weakest first like COVERAGE_ORDER.
VERDICT_LABELS = {
    "failed_mismatch": (
        "paper and REPLICA disagree on the declared cells, and the "
        "disagreement is root-caused"
    ),
    "failed_untraced": (
        "paper and REPLICA disagree on the declared cells; the mismatch is "
        "stable but no defect is identified on either side"
    ),
    "failed_defect": (
        "a confirmed defect in the released code changes the registered "
        "decisions, so the described method is not reproduced"
    ),
    **{
        f"blocked_{reason}": f"blocked: {sentence}"
        for reason, sentence in BLOCKED_REASONS.items()
    },
    "not_attempted": "no REPLICA implementation yet, and nothing blocks one",
    "pending": "REPLICA evidence is in flight, and nothing external blocks it",
    "reproduced_direction": (
        "paper and REPLICA agree on the declared decision or sign cells"
    ),
    "reproduced_value": (
        "paper and REPLICA agree on the declared numeric cells within the "
        "declared tolerance"
    ),
}

# Compact names for internal audit reports and compatibility. Public pages use
# the versioned Result/Scope/Evidence/Caveat contract below, not these
# multi-valued flags.
VERDICT_SHORT = {
    "failed_mismatch": "Does not reproduce",
    "failed_untraced": "Does not reproduce (cause unknown)",
    "failed_defect": "Does not reproduce as specified",
    "blocked_missing_data": "Blocked: missing data",
    "blocked_gated_data": "Blocked: gated data",
    "blocked_underspecified_method": "Blocked: missing details",
    "blocked_missing_official_analysis": "Blocked: no official analysis",
    "not_attempted": "Not yet tested",
    "pending": "Evidence incomplete",
    "reproduced_direction": "Reproduced (direction)",
    "reproduced_value": "Reproduced (values)",
}

VERDICT_ORDER = [
    "failed_mismatch",
    "failed_untraced",
    "failed_defect",
    "blocked_missing_data",
    "blocked_gated_data",
    "blocked_underspecified_method",
    "blocked_missing_official_analysis",
    "not_attempted",
    "pending",
    "reproduced_direction",
    "reproduced_value",
]

# Whether REPLICA rebuilt the predictors or reused the authors' columns is the
# question a "fresh refit" phrasing hides, so the label is rendered, not implied.
PROVENANCE_LABELS = {
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

# Public labels are a versioned contract. The internal ``verdict`` flags below
# retain all simultaneously true audit facts; the site renders one Result per
# claim and keeps Scope, Evidence, and Caveats as separate axes.
LABEL_CONTRACT_VERSION = 9

# ``reobtained`` is agreement reached by reading the authors' released result
# files: the paper's numbers come back, but no analysis was rerun, so by the
# project's definition that is not a reproduction and is never labelled one.
RESULT_LABELS = {
    "reproduced": "Reproduced",
    "reobtained": "Matches released results",
    "reproduced_as_released": "Repeated as released",
    "repeated": "Repeated; correction pending",
    "reference_only": "Corrected reference only",
    "partial": "Partially reproduced",
    "not_reproduced": "Not reproduced",
    "not_evaluated": "Not yet evaluated",
}

COMPARAND_LABELS = {
    "numeric": "Numeric values",
    "sign": "Effect directions",
    "exact": "Decisions",
}

CAVEAT_LABELS = {
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
    **{
        f"blocked_{reason}": f"Blocked: {sentence}"
        for reason, sentence in BLOCKED_REASONS.items()
    },
    "not_yet_tested": "No REPLICA result yet",
    "evidence_incomplete": "Paper-to-REPLICA comparison incomplete",
    "partial_coverage": "REPLICA covers a subset of the registered cells",
}


# One-sentence readings of the Scope and Evidence vocabularies for readers
# outside the project. The site shows them as hover readouts beside the short
# labels; the READMEs keep the labels alone.
COVERAGE_EXPLANATIONS = {
    "complete": (
        "Every cell the claim rests on was computed by REPLICA and compared "
        "with the paper."
    ),
    "complete_official_result_reader": (
        "Every cell was checked, by reading the authors' released result files "
        "rather than recomputing them."
    ),
    "partial": (
        "The comparison covers only part of the claim's scope; see the recorded limitations."
    ),
    "incomplete_official_and_replica": (
        "Neither the authors' released outputs nor REPLICA's computation yet "
        "covers the claim's cells."
    ),
    "incomplete_replica": (
        "The authors' released outputs cover the cells; REPLICA's own "
        "computation does not yet."
    ),
    "missing": "No REPLICA computation exists for this claim yet.",
}

PROVENANCE_EXPLANATIONS = {
    "independent_compute": (
        "REPLICA computed the predictors and refit the analysis itself; no "
        "author-produced number enters the result."
    ),
    "independent_compute_from_author_predictors": (
        "REPLICA refit the analysis itself, starting from the predictor values "
        "the authors released."
    ),
    "independent_compute_from_retained_predictors": (
        "REPLICA refit the analysis itself from retained predictor values. The "
        "source record identifies released and generated inputs and any "
        "unverified generation settings."
    ),
    "from_spec_compute": (
        "REPLICA implemented the analysis from the paper's description, with "
        "its own predictors, because no usable official analysis code exists."
    ),
    "official_result_reader": (
        "REPLICA reads the authors' released result files and re-derives the "
        "decision from them; nothing is recomputed."
    ),
    "implemented_no_result": (
        "The analysis is implemented in REPLICA but has not been run for this claim."
    ),
    "implemented_unavailable_inputs": (
        "The analysis is implemented in REPLICA, but an input it needs is not "
        "available."
    ),
    "missing": "REPLICA does not implement this analysis.",
}

COMPARAND_EXPLANATIONS = {
    "numeric": (
        "The paper's reported numbers are compared with REPLICA's within a "
        "stated tolerance."
    ),
    "sign": "Only the direction of each effect is compared.",
    "exact": (
        "Each cell is a yes-or-no decision, such as significant or not, that "
        "must match exactly."
    ),
}


def claims_path(study: str) -> Path:
    """The claims file of one study package.

    Args:
        study: Study package name.
    """
    return STUDIES_DIR / study / CLAIMS_FILE


def load_study_claims(study: str) -> dict[str, Any] | None:
    """One study's ``claims.yaml``, or None for a study without claims.

    Args:
        study: Study package name.
    """
    path = claims_path(study)
    if not path.is_file():
        return None
    with path.open(encoding="utf8") as stream:
        return yaml.safe_load(stream)


def load_registry(path: Path | str | None = None) -> dict[str, Any]:
    """Load the headline-claim registry.

    Args:
        path: One registry file in the assembled shape, read as is; omitted,
            the per-study claims files are assembled in study order.
    """
    if path is not None:
        with Path(path).open(encoding="utf8") as stream:
            return yaml.safe_load(stream)
    registry: dict[str, Any] = {
        "version": REGISTRY_VERSION,
        "last_audited": LAST_AUDITED,
        "scope": {"active_studies": [], "deferred": []},
        "study_sources": {},
        "claims": [],
    }
    for study in STUDY_IDS:
        doc = load_study_claims(study)
        if doc is None:
            registry["scope"]["deferred"].append(study)
            continue
        registry["scope"]["active_studies"].append(study)
        registry["study_sources"][study] = doc["sources"]
        registry["claims"].extend(doc["claims"])
    return registry


def _source_values(claim: dict[str, Any], source: str) -> dict[str, Any] | None:
    """Validate one source's cells, allowing declared partial coverage."""
    evidence = claim["evidence"][source]
    values = evidence.get("values")
    if values is None:
        return None
    expected = set(claim["comparand"]["cells"])
    supplied = set(values)
    if not supplied <= expected or (
        evidence.get("status") != "partial" and supplied != expected
    ):
        raise ValueError(
            f"{claim['id']}: {source} values do not use the declared cells"
        )
    return values


def compare_sources(claim: dict[str, Any], left: str, right: str) -> bool | None:
    """Compare two verification sources on the claim's exact cells.

    ``None`` means that a source lacks values for at least one declared cell.
    Only sources marked partial may omit cells; unknown cells always raise.
    Available partial values remain accessible in the per-cell table.

    Args:
        claim: One registry claim, read for its two evidence blocks and the
            ``comparand`` that fixes the cells and the match rule.
        left: Evidence key on one side, e.g. ``"paper"``.
        right: Evidence key on the other, e.g. ``"replica"``.

    Raises:
        ValueError: The two value sets are not keyed by the declared cells, or
            the comparand names an unsupported match kind.
    """
    left_values = _source_values(claim, left)
    right_values = _source_values(claim, right)
    if left_values is None or right_values is None:
        return None
    cells = claim["comparand"]["cells"]
    expected = set(cells)
    if set(left_values) != expected or set(right_values) != expected:
        return None

    match = claim["comparand"]["match"]
    kind = match["kind"]
    if kind == "exact":
        return all(left_values[cell] == right_values[cell] for cell in cells)
    if kind == "numeric":
        rtol = float(match.get("rtol", 0.0))
        atol = float(match.get("atol", 0.0))
        return all(
            math.isclose(
                float(left_values[cell]),
                float(right_values[cell]),
                rel_tol=rtol,
                abs_tol=atol,
            )
            for cell in cells
        )
    if kind == "sign":
        # A direction claim over stored numbers: the machine compares signs
        # rather than a hand-transcribed boolean.
        def _sign(value: float) -> int:
            return (value > 0) - (value < 0)

        return all(
            _sign(float(left_values[cell])) == _sign(float(right_values[cell]))
            for cell in cells
        )
    raise ValueError(f"{claim['id']}: unsupported match kind {kind!r}")


def cell_agreements(claim: dict[str, Any], left: str, right: str) -> list[bool] | None:
    """Per-cell agreement between two evidence sources, in the declared cell order.

    Args:
        claim: One registry claim.
        left: Evidence source name, e.g. ``"paper"``.
        right: Evidence source name, e.g. ``"replica"``.

    Returns:
        One boolean per declared cell, or ``None`` when either source lacks
        any declared value. Missing observations are not disagreements.
    """
    left_values = _source_values(claim, left)
    right_values = _source_values(claim, right)
    if left_values is None or right_values is None:
        return None
    expected = set(claim["comparand"]["cells"])
    if set(left_values) != expected or set(right_values) != expected:
        return None
    match = claim["comparand"]["match"]
    kind = match["kind"]
    out: list[bool] = []
    for cell in claim["comparand"]["cells"]:
        a, b = left_values[cell], right_values[cell]
        if kind == "exact":
            out.append(a == b)
        elif kind == "numeric":
            out.append(
                math.isclose(
                    float(a),
                    float(b),
                    rel_tol=float(match.get("rtol", 0.0)),
                    abs_tol=float(match.get("atol", 0.0)),
                )
            )
        elif kind == "sign":
            out.append(
                ((float(a) > 0) - (float(a) < 0)) == ((float(b) > 0) - (float(b) < 0))
            )
        else:
            raise ValueError(f"{claim['id']}: unsupported match kind {kind!r}")
    return out


#: The dimensions a comparand cell may be tagged with. ``dv`` is a canonical
#: measure name; ``corpus`` a dataset name; ``lm`` the language model; ``lang``
#: an ISO 639-1 code; ``variant`` whatever else distinguishes the cell, such as
#: an estimator or a skip convention.
CELL_TAG_KEYS = frozenset({"dv", "corpus", "lm", "lang", "variant"})


def cell_tags(claim: dict[str, Any]) -> dict[str, dict[str, str]]:
    """Each declared cell's tags, in the declared cell order.

    Args:
        claim: One registry claim; its ``comparand.cell_tags`` names every cell.
    """
    tags = claim["comparand"]["cell_tags"]
    return {cell: dict(tags[cell]) for cell in claim["comparand"]["cells"]}


def agreement_by_tag(
    claim: dict[str, Any], right: str, key: str = "dv"
) -> dict[str, dict[str, int]]:
    """Paper agreement per value of one cell tag, e.g. per measure.

    Cells without the tag are left out, so the counts here sum to at most the
    claim's total and a pooled cell never masquerades as a measure.

    Args:
        claim: One registry claim.
        right: Evidence source compared against the paper, ``"replica"`` or
            ``"official_code"``.
        key: The tag to group by, one of ``CELL_TAG_KEYS``.

    Returns:
        ``{tag value: {"cells_agree": n, "cells_total": m}}``, empty when the
        source has no values.
    """
    cells = cell_agreements(claim, "paper", right)
    if cells is None:
        return {}
    out: dict[str, dict[str, int]] = {}
    for agrees, tags in zip(cells, cell_tags(claim).values(), strict=True):
        value = tags.get(key)
        if value is None:
            continue
        bucket = out.setdefault(value, {"cells_agree": 0, "cells_total": 0})
        bucket["cells_total"] += 1
        bucket["cells_agree"] += int(agrees)
    return out


def verdict(claim: dict[str, Any]) -> list[str]:
    """The claim's outcome labels, derived, ordered weakest first.

    A claim holds every label that applies: a confirmed released-code defect
    that changes the registered decisions fails the described method even
    while the released artifact's own numbers reproduce, and a structured
    ``blocked`` marker coexists with whatever outcome the evidence already
    carries. A paper-to-REPLICA disagreement requires an authored
    ``failure`` record whose ``traced`` flag decides between the root-caused
    and the explicitly uncertain failure label. Pending and not-attempted
    are residual labels, held only when the claim carries no other outcome:
    a defect that already fails the described method is a settled outcome,
    not work in flight.

    Args:
        claim: One registry claim.

    Returns:
        Keys of ``VERDICT_LABELS``, sorted by ``VERDICT_ORDER``.
    """
    labels: set[str] = set()
    for key in ("predictor_correction", "test_divergence", "method_defect"):
        annotation = claim.get(key)
        if annotation and annotation.get("changes_registered_decisions"):
            labels.add("failed_defect")
    blocked = claim.get("blocked")
    if blocked is not None:
        labels.add(f"blocked_{blocked['reason']}")
    agreement = compare_sources(claim, "paper", "replica")
    if agreement is True:
        kind = claim["comparand"]["match"]["kind"]
        labels.add("reproduced_value" if kind == "numeric" else "reproduced_direction")
    elif agreement is False:
        labels.add(
            "failed_mismatch" if claim["failure"]["traced"] else "failed_untraced"
        )
    elif not labels:
        replica = claim["evidence"]["replica"]
        labels.add("not_attempted" if replica["status"] == "missing" else "pending")
    return sorted(labels, key=VERDICT_ORDER.index)


def has_independent_replica(claim: dict[str, Any]) -> bool:
    """Whether REPLICA independently computes the declared artifact values.

    Args:
        claim: One registry claim; its ``replica`` evidence must be available,
            carry values, and read no official result.
    """
    replica = claim["evidence"]["replica"]
    return (
        replica["status"] == "available"
        and replica["provenance"] in INDEPENDENT_REPLICA_PROVENANCE
        and not replica["consumes_official_results"]
        and replica.get("values") is not None
    )


def _disagreement_result(conjunctive: bool, cells: list[bool]) -> str:
    """Classify disagreement using the registered cell logic."""
    return "not_reproduced" if conjunctive or not any(cells) else "partial"


def _with_replica_values(
    claim: dict[str, Any], values: dict[str, Any]
) -> dict[str, Any]:
    """Copy a claim with replacement REPLICA values for an explicit comparison."""
    return {
        **claim,
        "evidence": {
            **claim["evidence"],
            "replica": {**claim["evidence"]["replica"], "values": values},
        },
    }


def covered_subset(claim: dict[str, Any]) -> dict[str, Any] | None:
    """Restrict a claim to the cells a partial REPLICA source supplies.

    The copy declares the source available over exactly those cells, so the
    paper comparison runs on the covered subset. Other sources retain only
    their recorded values in those cells. Returns None when the REPLICA source
    is complete or carries no values.

    Args:
        claim: One registry claim.
    """
    replica = claim["evidence"]["replica"]
    if replica.get("status") != "partial" or not replica.get("values"):
        return None
    covered = [
        cell for cell in claim["comparand"]["cells"] if cell in replica["values"]
    ]
    evidence = {
        name: (
            {
                **source,
                "values": {
                    cell: source["values"][cell]
                    for cell in covered
                    if cell in source["values"]
                },
            }
            if isinstance(source, dict) and source.get("values")
            else source
        )
        for name, source in claim["evidence"].items()
    }
    evidence["replica"] = {**evidence["replica"], "status": "available"}
    return {
        **claim,
        "comparand": {**claim["comparand"], "cells": covered},
        "evidence": evidence,
    }


def _corrected_result(
    claim: dict[str, Any], fixed: dict[str, Any], anchored_agreement: bool | None
) -> tuple[str, list[bool]]:
    """Use corrected cells for the public result, retaining an anchored-only label."""
    effective = {
        **claim,
        "evidence": {
            **claim["evidence"],
            "replica": {**claim["evidence"]["replica"], "values": fixed},
        },
    }
    effective = covered_subset(effective) or effective
    agreement = compare_sources(effective, "paper", "replica")
    cells = cell_agreements(effective, "paper", "replica") or []
    if agreement is True:
        reader = claim["evidence"]["replica"]["provenance"] == "official_result_reader"
        return ("reobtained" if reader else "reproduced"), cells
    if agreement is False:
        if anchored_agreement is True:
            return "reproduced_as_released", cells
        conjunctive = (
            claim["comparand"].get("cell_logic", "conjunctive") == "conjunctive"
        )
        return _disagreement_result(conjunctive, cells), cells
    return "not_evaluated", cells


def _comparison_inputs(
    claim: dict[str, Any],
) -> tuple[dict[str, Any] | None, set[str], bool | None, list[bool]]:
    """Compare the paper with REPLICA over the cells REPLICA supplies.

    A partial REPLICA source is compared over its covered cells; a complete one
    over every registered cell.

    Args:
        claim: One registry claim.
    """
    covered = covered_subset(claim)
    compared = covered if covered is not None else claim
    return (
        covered,
        set(verdict(compared)),
        compare_sources(compared, "paper", "replica"),
        cell_agreements(compared, "paper", "replica") or [],
    )


def _caveat_ids(
    claim: dict[str, Any],
    agreement: bool | None,
    flags: set[str],
    covered: dict[str, Any] | None,
) -> list[str]:
    """List the public caveats of a claim in vocabulary order without duplicates.

    Args:
        claim: One registry claim.
        agreement: Paper-to-REPLICA agreement over the compared cells.
        flags: Internal verdict flags of the compared claim.
        covered: The covered subset when REPLICA is partial, else None.
    """
    caveat_ids: list[str] = []
    if agreement is False:
        caveat_ids.append(
            "mismatch_root_caused"
            if claim["failure"]["traced"]
            else "mismatch_cause_unknown"
        )

    correction_states = {
        (claim.get(key) or {}).get("changes_registered_decisions")
        for key in ("predictor_correction", "test_divergence", "method_defect")
        if claim.get(key) is not None
    }
    if True in correction_states:
        caveat_ids.append("released_defect_changes_decision")
    if False in correction_states:
        caveat_ids.append("released_divergence_no_change")
    if None in correction_states:
        caveat_ids.append("correction_not_fully_evaluated")

    blocked = claim.get("blocked")
    if blocked is not None:
        caveat_ids.append(f"blocked_{blocked['reason']}")
    if covered is not None:
        caveat_ids.append("partial_coverage")
    if flags == {"not_attempted"}:
        caveat_ids.append("not_yet_tested")
    elif flags == {"pending"}:
        caveat_ids.append("evidence_incomplete")

    # Preserve the vocabulary order and suppress duplicate generic correction
    # labels when a claim carries more than one annotation of the same state.
    caveat_ids = list(dict.fromkeys(caveat_ids))
    return caveat_ids or ["none"]


def claim_status(claim: dict[str, Any]) -> dict[str, Any]:
    """Return the versioned public Result/Scope/Evidence/Caveat contract.

    The Result says whether the claim holds under the setting REPLICA ran, and
    the marker beside it says what evidence that rests on; neither stands in for
    the other. Agreement reached only while a defect in the released computation
    is kept is reported as reproduced as released, not as reproduced. Where the
    registered cells disagree, whether losing one falsifies the claim is a
    property of the claim's own sentence, declared per claim as
    ``comparand.cell_logic``: a conjunction of distinct sub-statements, such as
    a ranking or a uniqueness, fails outright, while cells enumerating a
    population the claim quantifies over hold partly.

    Args:
        claim: One registry claim.
    """
    covered, flags, agreement, cells = _comparison_inputs(claim)
    conjunctive = claim["comparand"].get("cell_logic", "conjunctive") == "conjunctive"
    reader = claim["evidence"]["replica"]["provenance"] == "official_result_reader"
    if agreement is True and reader:
        result_id = "reobtained"
    elif agreement is True and "failed_defect" in flags:
        # The paper's decisions hold only while the released computation's
        # defect is kept: agreement under the release's own procedure.
        result_id = "reproduced_as_released"
    elif flags & {"failed_mismatch", "failed_untraced", "failed_defect"}:
        lost_everything = not any(cells)
        result_id = "not_reproduced" if conjunctive or lost_everything else "partial"
    elif agreement is True:
        result_id = "reproduced"
    else:
        result_id = "not_evaluated"

    fixed = corrected_values(claim)
    if fixed is not None:
        result_id, cells = _corrected_result(claim, fixed, agreement)
    elif any(claim.get(key) for key in CORRECTION_KEYS) and not reader:
        result_id = "repeated" if agreement is True else "not_evaluated"

    if (claim.get("reproduction_review") or {}).get("status") == "fidelity_unresolved":
        result_id = "reference_only"

    effective = _with_replica_values(claim, fixed) if fixed is not None else claim
    effective = covered_subset(effective) or effective
    match_kind = claim["comparand"]["match"]["kind"]
    caveat_ids = _caveat_ids(claim, agreement, flags, covered)
    return {
        "contract_version": LABEL_CONTRACT_VERSION,
        "result": {
            "id": result_id,
            "label": RESULT_LABELS[result_id],
            "cells_agree": sum(cells),
            "cells_total": len(cells),
            "by_dv": agreement_by_tag(
                effective,
                "replica",
                "dv",
            ),
        },
        "marker": {
            "field": "coverage",
            "id": claim["coverage"],
            "label": COVERAGE_LABELS[claim["coverage"]],
            "reading": COVERAGE_EXPLANATIONS[claim["coverage"]],
        },
        "scope": {
            "id": match_kind,
            "label": COMPARAND_LABELS[match_kind],
            "cells": len(claim["comparand"]["cells"]),
            "coverage_id": claim["coverage"],
            "coverage_label": COVERAGE_LABELS[claim["coverage"]],
        },
        # A provenance label beside a claim with no registered REPLICA values
        # would read as evidence that exists; the flag lets renderers say so.
        "evidence": {
            "id": claim["evidence"]["replica"]["provenance"],
            "label": PROVENANCE_LABELS[claim["evidence"]["replica"]["provenance"]],
            "has_values": "values" in claim["evidence"]["replica"],
        },
        "caveats": [
            {"id": caveat_id, "label": CAVEAT_LABELS[caveat_id]}
            for caveat_id in caveat_ids
        ],
    }


def comparison_summary(claim: dict[str, Any]) -> str:
    """Explain a claim verdict without requiring the registry vocabulary.

    A claim can both match the paper under the released computation and fail
    after a defect in that computation is corrected. The internal audit flags
    preserve both facts, but their conjunction is easy to misread. This
    sentence makes the relationship explicit and also distinguishes a
    result-file reader from an independent recomputation.

    Args:
        claim: One registry claim.

    Returns:
        A reader-facing account of what the evidence establishes.
    """
    labels = set(verdict(claim))
    review = claim.get("reproduction_review") or {}
    if review.get("status") == "fidelity_unresolved":
        return (
            review["reason"]
            + " Released-procedure and earlier method-only results remain separately reported."
        )
    agreement = compare_sources(claim, "paper", "replica")
    replica = claim["evidence"]["replica"]

    if agreement is True:
        kind = claim["comparand"]["match"]["kind"]
        object_ = "values" if kind == "numeric" else "decisions or directions"
        if has_independent_replica(claim):
            opening = (
                f"An independent REPLICA analysis recovers the paper's registered "
                f"{object_}."
            )
        elif replica["provenance"] == "official_result_reader":
            opening = (
                f"The released result files recover the paper's registered {object_}; "
                "REPLICA verifies those files but does not independently recompute the "
                "analysis."
            )
        else:
            opening = f"REPLICA recovers the paper's registered {object_}."
    elif agreement is False:
        cells = cell_agreements(claim, "paper", "replica") or []
        where = (
            f"on {sum(cells)} of {len(cells)} registered cells and disagrees on the rest"
            if sum(cells) > len(cells) / 2
            else "on the registered cells"
        )
        verb = (
            "agrees with the paper"
            if sum(cells) > len(cells) / 2
            else "disagrees with the paper"
        )
        if "failed_mismatch" in labels:
            opening = (
                f"REPLICA {verb} {where}, and the difference has been traced to a "
                "specific cause."
            )
        else:
            opening = f"REPLICA {verb} {where}, but the cause has not been established."
    elif "not_attempted" in labels:
        opening = (
            "No REPLICA result is available, so there is no paper-to-REPLICA "
            "comparison for the registered cells."
        )
    else:
        opening = (
            "The implementation or evidence is incomplete, so no direct "
            "paper-to-REPLICA comparison is available for the registered cells."
        )
        if claim.get("reference_evidence"):
            opening = (
                "Scoped reference refits are available and reported below. Their "
                "scope or procedure differs from the registered experiment, so "
                "the paper-wide repetition and corrected reproduction remain incomplete."
            )

    if "failed_defect" in labels:
        if agreement is None:
            opening = (
                "REPLICA has not completed a direct comparison on the registered "
                "cells, but correcting a confirmed defect in the released "
                "computation changes at least one claim-defining decision, so the "
                "method the paper describes does not give the paper's result."
            )
        else:
            opening += (
                " A confirmed defect changes at least one claim-defining decision "
                "when corrected, so the method described by the paper does not "
                "reproduce."
            )

    fixed = corrected_values(claim)
    if fixed is not None:
        fixed_claim = {
            **claim,
            "evidence": {**claim["evidence"], "replica": {**replica, "values": fixed}},
        }
        if compare_sources(fixed_claim, "paper", "replica") is True:
            opening = (
                "With the registered corrections applied, REPLICA recovers the paper's "
                "registered decisions. The released-code refit is reported separately."
            )
    elif any(claim.get(key) for key in CORRECTION_KEYS):
        opening += " The corrected reproduction has not yet been registered."

    blocked = claim.get("blocked")
    if blocked is not None:
        opening += f" Completion is blocked. {blocked['detail'].rstrip('.')}."
    return opening


def audit_rows(registry: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Return compact computed rows for reporting and tests.

    Args:
        registry: Already-parsed registry; ``None`` loads the shipped one.

    Returns:
        One row per claim, carrying its verdict, its coverage, and the three
        pairwise source comparisons.
    """
    registry = load_registry() if registry is None else registry
    rows = []
    for claim in registry["claims"]:
        rows.append(
            {
                "id": claim["id"],
                "study": claim["study"],
                "coverage": claim["coverage"],
                "verdict": verdict(claim),
                "paper_official_code": compare_sources(claim, "paper", "official_code"),
                "official_code_replica": compare_sources(
                    claim, "official_code", "replica"
                ),
                "paper_replica": compare_sources(claim, "paper", "replica"),
                "independent_replica": has_independent_replica(claim),
            }
        )
    return rows


def claims_for_study(
    study: str, registry: dict[str, Any] | None = None
) -> list[dict[str, Any]]:
    """Return the canonical headline claims for one study in registry order.

    Args:
        study: Package name under ``src/replica/studies``.
        registry: Already-parsed registry; ``None`` loads the shipped one.
    """
    registry = load_registry() if registry is None else registry
    return [claim for claim in registry["claims"] if claim["study"] == study]


def _repo_link(path: str, study: str) -> str:
    """Link from a study README to a repository path, climbing only as far as it must.

    Args:
        path: Repository-root-relative path of the target.
        study: Package name of the README's study, which fixes the README's
            directory.
    """
    from ..records.files import hub_url

    parts = path.split("/")
    if (
        len(parts) > 5
        and parts[:3] == ["src", "replica", "studies"]
        and parts[4] == "results"
    ):
        remote = hub_url(parts[3], "/".join(parts[5:]))
        if remote:
            return remote
    return os.path.relpath(path, f"src/replica/studies/{study}")


def _evidence_body(
    evidence: dict[str, Any],
    study: str,
    fallback: tuple[str, str] | None = None,
) -> str:
    """A markdown link to the artifact, then whatever prose it cannot carry.

    A claim that names its own artifact links that. Otherwise the prose keeps
    its own wording and the study-level source follows as a labelled pointer:
    wrapping prose in a URL would assert that the prose describes that file.
    """
    source = evidence.get("source", "")
    link = evidence.get("link")
    if link is not None:
        url = (
            link["url"]
            if link["url"].startswith("http")
            else _repo_link(link["url"], study)
        )
        return f"[{link['label']}]({url}). {source}".strip()
    if fallback and source:
        label, url = fallback
        return f"{source.rstrip('.')} ([{label}]({url}))."
    return source


def specification_line(claim: dict[str, Any], link: bool = False) -> str:
    """The claim's computation and test in one line: the call, the analysis, the test, the code.

    Args:
        claim: One registry claim.
        link: Render the implementing module as a link relative to the study's
            README rather than as a code span.
    """
    spec = claim["specification"]
    path = spec["implemented_in"]
    code = f"[{path}]({_repo_link(path, claim['study'])})" if link else f"`{path}`"
    # Placeholders such as ``<response>`` are prose, and an unescaped angle
    # bracket is an HTML tag to every Markdown renderer the line reaches.
    evaluate, analysis, test = (
        spec[key].replace("<", "\\<").replace(">", "\\>")
        for key in ("evaluate", "analysis", "test")
    )
    return f"Evaluate: {evaluate} Analysis: {analysis} Test: {test} Code: {code}."


def _cell_value(value: Any) -> str:
    """One registered value as a table cell: decisions as yes or no, numbers as written."""
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return format(value, "g")
    return str(value)


CORRECTION_KEYS = ("predictor_correction", "test_divergence", "method_defect")


def corrected_values(claim: dict[str, Any]) -> dict[str, Any] | None:
    """REPLICA's per-cell decisions with every registered defect corrected, or ``None``.

    A correction block may carry a ``corrected`` evidence block (``values`` and
    the ``artifact`` they are read from). Those are the decisions the
    Reproduction column counts: the paper's method, defects fixed.

    Args:
        claim: One registry claim.
    """
    annotations = [claim[key] for key in CORRECTION_KEYS if claim.get(key)]
    if not annotations:
        return None
    candidates = [(a.get("corrected") or {}).get("values") for a in annotations]
    if any(values is None for values in candidates):
        return None
    # Separate fixes are not evidence for a run with every fix applied.
    if any(values != candidates[0] for values in candidates[1:]):
        return None
    artifacts = [a["corrected"].get("artifact") for a in annotations]
    if len(artifacts) > 1 and (
        artifacts[0] is None or any(a != artifacts[0] for a in artifacts[1:])
    ):
        return None
    return candidates[0]


def cells_table(
    claim: dict[str, Any], indent: str = "     ", compact: bool = False
) -> list[str]:
    """The claim's registered cells, paper beside official code beside REPLICA, as Markdown table lines.

    Each source shows the value it registers for the cell and ``n/a`` where it
    registers none, so a reader sees every side of the comparison the Result
    is drawn from without opening the registry. A claim with a registered
    correction gains a column with REPLICA's decisions under the corrected
    computation.

    Args:
        claim: One registry claim.
        indent: Prefix of every line, the list-item indentation of the README block.
        compact: Show the paper and both original-data analysis axes, without
            duplicating corrected values or released-output comparison targets.
    """
    evidence = claim["evidence"]
    columns = [
        ("paper", evidence.get("paper", {}).get("values") or {}),
        ("official code", evidence.get("official_code", {}).get("values") or {}),
        ("REPLICA", evidence.get("replica", {}).get("values") or {}),
    ]
    if evidence.get("repetition", {}).get("values"):
        columns.insert(
            2,
            (
                "REPLICA, released repetition",
                evidence["repetition"].get("values") or {},
            ),
        )
    corrected = corrected_values(claim)
    if compact:
        reanalysis = (
            corrected
            if any(claim.get(key) for key in CORRECTION_KEYS)
            else evidence.get("replica", {}).get("values")
        )
        columns = [
            ("paper", evidence.get("paper", {}).get("values") or {}),
            (
                "original-data repetition",
                evidence.get("repetition", {}).get("values") or {},
            ),
            ("original-data reanalysis", reanalysis or {}),
        ]
    elif corrected is not None:
        columns.append(("REPLICA, defect corrected", corrected))
    else:
        for key in CORRECTION_KEYS:
            partial = (claim.get(key) or {}).get("corrected")
            if partial and partial.get("values"):
                columns.append(
                    (f"REPLICA, {key.replace('_', ' ')} only", partial["values"])
                )
    lines = [
        f"{indent}| cell | " + " | ".join(label for label, _ in columns) + " |",
        f"{indent}|---|" + "---|" * len(columns),
    ]
    for cell in claim["comparand"]["cells"]:
        row = [cell.replace("_", " ")] + [_cell_value(v.get(cell)) for _, v in columns]
        lines.append(f"{indent}| " + " | ".join(row) + " |")
    return lines


def _compact_readme_cells(study, claims, paper_url):
    """Render source-linked decisions and scoped comparisons without repeated methods."""
    lines = []
    for index, claim in enumerate(claims, 1):
        status = claim_status(claim)
        result = status["result"]
        fraction = ""
        if result["id"] == "partial" and result["cells_total"]:
            fraction = f", {result['cells_agree']} of {result['cells_total']} cells"
        result_line = f"{result['label']}{fraction} ({status['marker']['label']})"
        if result["cells_total"] < status["scope"]["cells"]:
            result_line += (
                f"; {result['cells_total']}/{status['scope']['cells']} "
                "original-data cells available"
            )
        caveats = "; ".join(item["label"] for item in status["caveats"])
        anchor = f"[{claim['anchor'].rstrip('.')}]({paper_url})."
        if lines:
            lines.append("")
        lines.extend(
            [
                f"{index}. **{claim['claim']}**",
                f"   - **Result:** {result_line}. {caveats}.",
                f"   - **Paper:** {anchor}",
                "",
                *cells_table(claim, compact=True),
                "",
            ]
        )
        links = []
        for key, label in (
            ("repetition", "Original-data repetition"),
            ("replica", "Original-data reanalysis"),
        ):
            artifact = (claim["evidence"].get(key) or {}).get("artifact")
            if artifact:
                links.append(f"[{label}]({_repo_link(artifact['path'], study)})")
        if links:
            lines.append("   - **Evidence:** " + "; ".join(links) + ".")
        for reference in claim.get("reference_evidence", []):
            label = (
                "Scoped repetition"
                if reference.get("kind") == "scoped_repetition"
                else "Scoped reanalysis"
                if reference.get("kind") == "scoped_reproduction"
                else "Scoped reference"
            )
            values = ", ".join(
                f"{name.replace('_', ' ')}: {_cell_value(value)}"
                for name, value in (reference.get("values") or {}).items()
            )
            link = _evidence_body({**reference, "source": ""}, study)
            lines.append(f"   - **{label}:** {link} {values}.")
    gaps = dict.fromkeys(c["gap"] for c in claims if c.get("gap"))
    if gaps:
        lines.extend(["", "**Scope limits:** " + " ".join(gaps)])
    return "\n".join(lines)


def _compact_readme_claims(claims: list[dict[str, Any]], paper_url: str) -> str:
    """Render claim identities, decisions and locators without duplicating methods."""
    lines = []
    for number, claim in enumerate(claims, 1):
        status = claim_status(claim)
        marker = status["marker"]["label"]
        if claim.get("reference_evidence") and not claim["evidence"]["replica"].get(
            "values"
        ):
            marker = "no whole-claim result; scoped evidence available"
        if lines:
            lines.append("")
        lines.extend(
            [
                f"{number}. **{claim['claim']}**",
                f"   - **Result:** {status['result']['label']} ({marker}).",
                f"   - **Scope:** {status['scope']['label']} ({status['scope']['cells']} "
                f"{'cell' if status['scope']['cells'] == 1 else 'cells'}).",
                f"   - **Paper:** [{claim['anchor'].rstrip('.')}]({paper_url}).",
            ]
        )
    return "\n".join(lines)


def render_readme_claims(
    study: str, registry: dict[str, Any] | None = None, *, detail: bool | None = None
) -> str:
    """Render the audited, canonical claim block for a study README.

    The YAML registry remains the source of truth. Tests require the checked-in
    README block to equal this rendering exactly, so public prose cannot drift
    from the evidence and coverage statuses used by reports.

    Args:
        study: Package name under ``src/replica/studies``.
        registry: Already-parsed registry; ``None`` loads the shipped one.
        detail: Include cell tables and full evidence; defaults to the study's
            README preference. Website evidence always requests full detail.

    Raises:
        KeyError: The registry holds no claim for that study.
    """
    registry = load_registry() if registry is None else registry
    claims = claims_for_study(study, registry)
    if not claims:
        raise KeyError(f"{study}: no registered headline claims")

    sources = registry["study_sources"][study]
    compact_cells = detail is None and sources.get("readme_claims") == "compact"
    if detail is None:
        detail = sources.get("readme_format", "full") != "compact"
    # Claim locators must open the exact document that was audited. A DOI or
    # proceedings landing page can resolve to a later version with different
    # pagination; ``audit_document`` pins both the content hash and the direct
    # full-text URL.
    paper_url = sources["paper"]["audit_document"]["url"]
    if compact_cells:
        return _compact_readme_cells(study, claims, paper_url)
    if not detail:
        return _compact_readme_claims(claims, paper_url)
    code_fallback = ("official code", sources["official_code"]["url"])
    replica_fallback_link = (
        "implementation",
        _repo_link(sources["replica"]["implementation"], study),
    )

    lines: list[str] = []
    for index, claim in enumerate(claims, 1):
        status = claim_status(claim)
        paper_source = claim["evidence"]["paper"].get("source")
        replica_evidence = claim["evidence"]["replica"]
        provenance = status["evidence"]["label"]
        evidence_line = provenance + (
            ""
            if status["evidence"]["has_values"]
            else "; scoped reference results available; registered comparison incomplete"
            if claim.get("reference_evidence")
            else "; no REPLICA result registered yet"
        )
        scope = status["scope"]
        scope_cells = "cell" if scope["cells"] == 1 else "cells"
        caveats = "; ".join(item["label"] for item in status["caveats"])
        anchor = f"[{claim['anchor'].rstrip('.')}]({paper_url})."
        # A bullet that reports an absence gets no pointer: linking the study's
        # entry point beside "not rerun" reads as if that file were the thing.
        replica_fallback = (
            None
            if replica_evidence["provenance"] in {"missing", "implemented_no_result"}
            else replica_fallback_link
        )
        code_evidence = claim["evidence"]["official_code"]
        code_link = None if code_evidence["status"] == "missing" else code_fallback
        result = status["result"]
        fraction = ""
        if result["id"] == "partial" and result["cells_total"]:
            fraction = f", {result['cells_agree']} of {result['cells_total']} cells"
        result_line = f"{result['label']}{fraction} ({status['marker']['label']})"
        if lines:
            lines.append("")
        lines.extend(
            [
                f"{index}. **{claim['claim']}**",
                f"   - **Result:** {result_line}.",
                f"   - **Scope:** {scope['label']} ({scope['cells']} {scope_cells}).",
                f"   - **Specification:** {specification_line(claim, link=True)}",
                f"   - **Evidence:** {evidence_line}.",
                f"   - **Caveats:** {caveats}.",
                "   - **Cells:** the registered values, paper beside official code "
                "beside REPLICA.",
                "",
                *cells_table(claim),
                "",
                f"   - **Paper:** {anchor + ' ' + paper_source if paper_source else anchor}",
                "   - **Official/reference code:** "
                + _evidence_body(code_evidence, study, code_link),
                f"   - **REPLICA** ({provenance}): "
                + _evidence_body(replica_evidence, study, replica_fallback),
            ]
        )
        blocked = claim.get("blocked")
        for reference in claim.get("reference_evidence", []):
            label = {
                "scoped_repetition": "Scoped repetition evidence",
                "scoped_reproduction": "Scoped corrected reproduction evidence",
            }.get(reference.get("kind"), "Scoped reference evidence")
            lines.append(f"   - **{label}:** " + _evidence_body(reference, study))
        if blocked:
            reason = blocked["reason"].replace("_", " ")
            lines.append(f"   - **Blocked ({reason}):** {blocked['detail']}")
        failure = claim.get("failure")
        if failure:
            qualifier = "root-caused" if failure["traced"] else "cause not established"
            lines.append(f"   - **Failure record ({qualifier}):** {failure['note']}")
        correction = claim.get("predictor_correction")
        if correction:
            lines.append(
                "   - **Released predictor diverges from the paper's definition:** "
                + correction["divergence"]
            )
            lines.append(
                "   - **With the predictor corrected:** " + correction["effect"]
            )
        test_divergence = claim.get("test_divergence")
        if test_divergence:
            lines.append(
                "   - **Released test diverges from the one the code names:** "
                + test_divergence["divergence"]
            )
            lines.append(
                "   - **Under the test it names:** " + test_divergence["effect"]
            )
        method_defect = claim.get("method_defect")
        if method_defect:
            # "Confirmed" is reserved for the authors' word; a defect REPLICA
            # established on its own says so.
            label = (
                "Confirmed defect in the released method"
                if method_defect.get("confirmed_by_authors")
                else "Defect found in the released method (not confirmed by the authors)"
            )
            lines.append(f"   - **{label}:** " + method_defect["divergence"])
            lines.append(
                "   - **With the defect corrected:** " + method_defect["effect"]
            )
        if claim.get("gap"):
            lines.append(f"   - **Remaining gap:** {claim['gap']}")
    return "\n".join(lines)


def main() -> None:
    """Print the strict audit in a compact tab-separated form."""
    print(
        "claim\tverdict\tcoverage\tPaper vs official code\t"
        "official code vs REPLICA\tPaper vs REPLICA\tindependent REPLICA"
    )
    for row in audit_rows():
        print(
            row["id"],
            "; ".join(row["verdict"]),
            row["coverage"],
            row["paper_official_code"],
            row["official_code_replica"],
            row["paper_replica"],
            row["independent_replica"],
            sep="\t",
        )


if __name__ == "__main__":
    main()
