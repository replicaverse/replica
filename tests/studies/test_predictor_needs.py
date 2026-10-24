"""The predictor-needs sections stay consistent with the claims and the store."""

import pytest

from replica.studies.claims.predictor_needs import (
    load_needs,
    published_cell_patterns,
    validate,
)

pytestmark = pytest.mark.needs_data


def test_needs_registry_validates():
    validate()


def test_every_claim_is_linked_from_some_predictor():
    """Only synthetic or dataset-column-only claims may go unlinked."""
    from replica.studies.claims.registry import load_registry

    needs = load_needs()
    registry = load_registry()
    linked = {
        claim_id
        for entries in needs["studies"].values()
        for entry in entries
        for claim_id in entry.get("claims", ())
    }
    computed_claims = {
        claim["id"]
        for claim in registry["claims"]
        if claim["study"] in needs["studies"]
    }
    unlinked = computed_claims - linked
    # A claim may legitimately consume only dataset columns or synthetic data;
    # those are the only accepted reasons for an unlinked claim.
    accepted = {
        "shain2018dtsr_synthetic_irf_recovery",
        "shain2021cdr_hyperparameter_consistency",
        "shain2021cdr_synthetic_recovery",
        "re2025spatiotemporal_fixation_duration_surprisal",
        "wilcox2023surprisal_linear_link",
    }
    assert unlinked <= accepted, f"claims with no predictor link: {sorted(unlinked)}"


def test_published_patterns_exist_in_the_store():
    patterns = sorted(published_cell_patterns())
    assert patterns, "no published cells declared"
    from replica.predictors import published_tags

    published = published_tags()
    if published is None:
        pytest.skip("predictor store unreachable")
    missing = []
    for pattern in patterns:
        if pattern.endswith("*"):
            prefix = pattern[:-1]
            if not any(tag.startswith(prefix) for tag in published):
                missing.append(pattern)
        elif pattern not in published:
            missing.append(pattern)
    assert not missing, f"declared cells with no store tag: {missing}"
