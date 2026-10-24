"""Cross-dataset replication results after the claim registry's protocol reviews.

Each study package's ``replication_results.yaml`` holds the archived transfer
summaries of its claims and the runs they came from; :func:`load_raw_results`
assembles them in study order and :func:`save_results` writes an assembled
mapping back, one file per study. :func:`reviewed_results` applies each
claim's registered reviews before anything is counted: evidence a review
excludes is kept under ``excluded``, and a released-method transfer must carry
verified fresh-run provenance.
"""

from __future__ import annotations

import math
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from replica.studies.packages import STUDIES_DIR

from .registry import CORRECTION_KEYS, STUDY_IDS, corrected_values, load_registry
from .values import validate_artifact_schema

RESULTS_FILE = "replication_results.yaml"
SCHEMA = 4


def results_path(study: str) -> Path:
    """The replication results file of one study package.

    Args:
        study: Study package name.
    """
    return STUDIES_DIR / study / RESULTS_FILE


def _empty() -> dict[str, Any]:
    return {"schema": SCHEMA, "sources": [], "claims": {}, "unmapped_contrasts": []}


def load_raw_results() -> dict[str, Any]:
    """The per-study results files assembled, before any protocol review."""
    results = _empty()
    sources: set[str] = set()
    for study in STUDY_IDS:
        path = results_path(study)
        if not path.is_file():
            continue
        doc = yaml.safe_load(path.read_text(encoding="utf8"))
        results["claims"].update(doc.get("claims") or {})
        sources.update(doc.get("sources") or [])
        results["unmapped_contrasts"].extend(doc.get("unmapped_contrasts") or [])
    results["sources"] = sorted(sources)
    return results


def _study_of(
    source: str, claims: dict[str, Any], existing: dict[str, set[str]]
) -> set[str]:
    """The studies a source run belongs to.

    A study named in the run's path owns it; otherwise the studies that
    already file it or whose claim entries cite it.
    """
    parts = source.split("/")
    named = {
        study
        for study in STUDY_IDS
        if any(part == study or part.startswith(study + "_") for part in parts)
    }
    if named:
        return named
    found = {study for study, sources in existing.items() if source in sources}
    for cid, entry in claims.items():
        study = cid.split("_")[0]
        for ref in _strings(entry):
            if (
                ref == source
                or source.startswith(ref + "/")
                or ref.startswith(source + "/")
            ):
                found.add(study)
    return found


def _strings(value: Any):
    if isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)
    elif isinstance(value, str):
        yield value


def save_results(results: dict[str, Any]) -> list[Path]:
    """Write an assembled results mapping back to the per-study files.

    Claims go to the study their id names; a source run goes to every study
    named in its path, that already lists it, or whose claim entries cite it.

    Args:
        results: Assembled mapping as :func:`load_raw_results` returns it.

    Returns:
        The files written.

    Raises:
        ValueError: A source run that no rule attributes to a study.
    """
    existing: dict[str, set[str]] = {}
    for study in STUDY_IDS:
        path = results_path(study)
        if path.is_file():
            existing[study] = set(
                yaml.safe_load(path.read_text(encoding="utf8")).get("sources") or []
            )
    by_study: dict[str, dict[str, Any]] = {}
    for cid, entry in results.get("claims", {}).items():
        by_study.setdefault(cid.split("_")[0], _empty())["claims"][cid] = entry
    for source in results.get("sources", []):
        studies = _study_of(source, results.get("claims", {}), existing)
        if not studies:
            raise ValueError(f"source run {source!r} belongs to no study")
        for study in studies:
            by_study.setdefault(study, _empty())["sources"].append(source)
    for contrast in results.get("unmapped_contrasts", []):
        by_study.setdefault(contrast.get("study", ""), _empty())[
            "unmapped_contrasts"
        ].append(contrast)
    written = []
    for study, doc in by_study.items():
        if study not in STUDY_IDS:
            raise ValueError(f"{study!r} is not a study package")
        doc["sources"] = sorted(set(doc["sources"]))
        if not doc["unmapped_contrasts"]:
            del doc["unmapped_contrasts"]
        path = results_path(study)
        path.write_text(
            yaml.safe_dump(doc, sort_keys=False, width=88, allow_unicode=True),
            encoding="utf8",
        )
        written.append(path)
    return written


def claim_index(registry: dict[str, Any]) -> dict[str, dict]:
    """The registry's claims keyed by id, registry order preserved.

    Args:
        registry: The assembled claim registry.
    """
    return {c["id"]: c for c in registry["claims"]}


def load_results(registry: dict[str, Any] | None = None) -> dict[str, Any]:
    """Cross-dataset outcomes per claim, empty when the results file is absent.

    Args:
        registry: Already parsed claims; omitted, read the current registry.

    Returns:
        The assembled per-study results, keyed by ``claims``, ``sources``
        and ``unmapped_contrasts``.
    """
    raw = load_raw_results()
    if not raw["claims"] and not raw["sources"]:
        return {"claims": {}, "sources": [], "unmapped_contrasts": []}
    return reviewed_results(raw, load_registry() if registry is None else registry)


def _paper_code_datasets(corpus: str) -> set[str]:
    """Dataset names a registry corpus tag stands for (paper codes and registry codes)."""
    aliases = {
        "meco_ge": "meco_de",
        "meco_du": "meco_du",
        "meco_sp": "meco_sp",
        "meco_gr": "meco_gr",
    }
    return {corpus, aliases.get(corpus, corpus)}


def scope_transplant_entry(entry: dict, claim: dict) -> None:
    """Scope a recipe replication like a contrast replication, from the claim's cell tags.

    The recipe verdicts were archived without corpus or model scoping. A
    replication is the paper's model on a corpus the paper never saw, so cells
    on the claim's own corpora are listed and not counted, and cells run with
    another language model move to the ``robustness`` block. Both are read off
    the claim's ``cell_tags`` (``corpus`` and ``lm``); an entry whose claim
    carries no such tags is left as archived. Mutates ``entry`` in place.

    Args:
        entry: A recipe replication's results block (``total`` with ``points``).
        claim: The registry claim the entry belongs to.
    """
    tags = (claim.get("comparand") or {}).get("cell_tags") or {}
    own = set().union(
        *(_paper_code_datasets(t["corpus"]) for t in tags.values() if t.get("corpus"))
    )
    lms = {t["lm"] for t in tags.values() if t.get("lm")}
    total = entry.get("total") or {}
    points = total.get("points") or []
    if not points or not lms:
        return

    def is_own(dataset: str) -> bool:
        return dataset in own or ("meco" in own and dataset.startswith("meco"))

    for point in points:
        point["own_corpus"] = is_own(point.get("dataset", ""))

    def summary(group: list[dict]) -> dict:
        counted = [p for p in group if not p["own_corpus"]]
        holds = sum(1 for p in counted if p.get("significant"))
        return {
            **{k: v for k, v in total.items() if k not in ("points",)},
            "supporting_cells": holds,
            "cells": len(counted),
            "own_corpus_cells": len(group) - len(counted),
            "datasets": len({p["dataset"] for p in counted}),
            "scored": len(counted),
            "positive": holds,
            "positive_significant": holds,
            "negative_significant": 0,
            "points": group,
        }

    paper_points = [p for p in points if p.get("lm") in lms]
    other = [p for p in points if p.get("lm") not in lms]
    entry["total"] = summary(paper_points)
    entry["paper_lms"] = sorted(lms)
    entry["own_corpora"] = sorted({p["dataset"] for p in points if p["own_corpus"]})
    entry["by_model"] = {
        lm: summary([p for p in paper_points if p.get("lm") == lm])
        for lm in sorted(lms)
    }
    if other:
        entry["robustness"] = {
            lm: summary([p for p in other if p.get("lm") == lm])
            for lm in sorted({p.get("lm") for p in other})
        }


def check_released_entry(entry: dict) -> None:
    """Require explicit fresh-run provenance before showing released-method transfers.

    Args:
        entry: A separately registered replication/repetition result.
    """
    if (
        entry.get("stage") != "replication_repetition"
        or entry.get("analysis_method") != "released"
        or entry.get("provenance")
        not in {"author_code_compute", "independent_recomputation"}
        or (entry.get("review") or {}).get("status") != "verified"
        or not entry.get("source_run")
        or not entry.get("scope")
        or not (entry.get("artifact") or {}).get("sha256")
    ):
        raise ValueError(
            "Released-method transfer lacks fresh-run scope and verified provenance"
        )


def _scoped_corrected_baseline(claim: dict, review: dict) -> bool:
    """Accept an explicitly reviewed substitute without filling original cells."""
    reference_id = review.get("corrected_baseline_reference")
    if not reference_id or review.get("status") != "verified":
        return False
    references = [
        ref
        for ref in claim.get("reference_evidence", [])
        if ref.get("id") == reference_id
    ]
    if len(references) != 1:
        return False
    ref = references[0]
    targets = ref.get("comparison", {}).get("targets", {})
    values = ref.get("values") or {}
    artifact = ref.get("artifact") or {}
    try:
        validate_artifact_schema(artifact)
    except (TypeError, ValueError):
        return False
    if not values or any(
        value is None
        or (isinstance(value, (int, float)) and not math.isfinite(value))
        or (isinstance(value, str) and value not in {"+", "-", "ns"})
        or not isinstance(value, (bool, int, float, str))
        for value in values.values()
    ):
        return False
    return bool(
        ref.get("kind") == "scoped_reproduction"
        and "reproduction" in ref.get("stages", [])
        and ref.get("provenance") == "independent_compute"
        and not ref.get("consumes_official_results", False)
        and ref.get("scope")
        and re.fullmatch(r"[0-9a-f]{64}", artifact["sha256"])
        and set(artifact.get("extract", {})) == set(values)
        and set(values) == set(targets)
        and set(targets.values()) == set(claim["comparand"]["cells"])
        and {key for key in CORRECTION_KEYS if claim.get(key)}
        <= set(ref.get("corrected_defects", []))
    )


def reviewed_results(
    results: dict[str, Any], registry: dict[str, Any]
) -> dict[str, Any]:
    """Apply claim-level protocol reviews before any website counts are made.

    Preserve excluded evidence for inspection. A legacy recipe verdict does
    not establish significance or that only the corpus changed. Corrections
    must also have a completed baseline before their transfer can count.

    Args:
        results: Archived per-claim summaries, preserved without mutation.
        registry: Claims carrying correction evidence and protocol reviews.
    """
    reviewed = deepcopy(results)
    claims = claim_index(registry)
    for cid, entry in reviewed.get("claims", {}).items():
        if entry.get("replication_repetition"):
            check_released_entry(entry["replication_repetition"])
        if entry.get("stage") == "replication_repetition":
            check_released_entry(entry)
            continue
        claim = claims.get(cid, {})
        review = claim.get("replication_review") or {}
        reason = (
            review.get("reason") if review.get("status") == "robustness_only" else None
        )
        comparand = claim.get("comparand") or {}
        numerical = entry.get("numerical")
        if numerical:
            rows = numerical.get("rows") or []
            if (
                comparand.get("significance")
                or numerical.get("significance_test") is not None
                or not rows
                or numerical.get("datasets") != len({row["corpus"] for row in rows})
                or any(not math.isfinite(row["delta_ll"]) for row in rows)
            ):
                raise ValueError(f"Invalid numerical replication evidence: {cid}")
        descriptive = (
            comparand.get("unit") == "decision"
            and comparand.get("significance") is False
        )
        if (
            descriptive
            and (entry.get("total") or {}).get("supported_when")
            == "transplant_verdict_holds"
        ):
            scope_transplant_entry(entry, claim)
        # A descriptive label does not verify the archived procedure or turn
        # an aggregate recipe verdict into a statistical test.
        if (
            not reason
            and (entry.get("total") or {}).get("supported_when")
            == "transplant_verdict_holds"
        ):
            reason = (
                "This legacy summary has not been verified against the claim's "
                "dataset-only protocol. Its aggregate verdict does not identify "
                "the underlying statistical tests, so it is excluded from "
                "replication counts pending review of the archived analysis."
            )
        if (
            not reason
            and any(claim.get(key) for key in CORRECTION_KEYS)
            and corrected_values(claim) is None
            and not _scoped_corrected_baseline(claim, review)
        ):
            reason = (
                "A reproduction correcting every registered defect is not available. "
                "The existing cross-corpus runs cannot yet establish replication "
                "of that corrected procedure."
            )
        if reason and entry.get("total") is not None:
            entry["excluded"] = {
                "reason": reason,
                "summary": entry.pop("total"),
                "by_model": entry.pop("by_model", {}),
                "original_corpora": review.get("original_corpora", []),
            }
            if entry.get("source_manifest"):
                entry["excluded"]["source_manifest"] = entry["source_manifest"]
                entry["excluded"]["source_run"] = entry["source_run"]
            entry["total"] = None
        if reason and entry.get("numerical"):
            entry["excluded_numerical"] = {
                "reason": reason,
                "results": entry.pop("numerical"),
            }
    return reviewed
