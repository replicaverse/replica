"""Shared assertions for study claim renderers."""

from replica.studies.claims.registry import (
    claim_status,
    claims_for_study,
    render_readme_claims,
)


def assert_compact_claims(claim_registry, study):
    block = render_readme_claims(study, claim_registry)
    assert "\n     | " not in block
    for claim in claims_for_study(study, claim_registry):
        assert claim["claim"] in block
        assert claim["anchor"].rstrip(".") in block
        assert claim_status(claim)["result"]["label"] in block
