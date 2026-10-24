"""Machine-checkable predictor layer of the claim registry.

The ``predictor_needs`` section of each study's ``claims.yaml`` records the
predictors its claims consume and their status, mirroring the study README's
``## Predictors`` table; ``validate`` ties it to the claims. ``status``
vocabulary:

- ``published``: a faithful store cell family exists; ``cells`` names it (a
  full tag, or a prefix ending in ``*``)
- ``computable``: REPLICA computes the convention; no faithful cell is
  published yet
- ``from_spec``: computed from the stated specification; no released column
  or independent anchor certifies it
- ``blocked``: REPLICA code exists but a pipeline axis is missing;
  ``blocker`` says which
- ``absent``: nothing in ``src`` computes it; ``blocker`` says what is
  missing
- ``dataset_column``: read from the dataset or the authors' release, not
  computed
- ``not_reproducible``: the authors' artifact cannot be rebuilt; ``blocker``
  says why
- ``shared``: documented under another study; ``see`` names it

``claims`` lists the registry claim ids that consume the predictor.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from replica.studies.claims.registry import STUDY_IDS, load_registry, load_study_claims

#: Schema version of the assembled predictor-needs mapping.
NEEDS_VERSION = 1

STATUSES = frozenset(
    {
        "published",
        "computable",
        "from_spec",
        "blocked",
        "absent",
        "dataset_column",
        "not_reproducible",
        "shared",
    }
)
#: Statuses whose entry must say what stands in the way.
NEEDS_BLOCKER = frozenset({"blocked", "absent", "not_reproducible"})


def load_needs(path: Path | str | None = None) -> dict[str, Any]:
    """Load the predictor-needs registry.

    Args:
        path: One needs file in the assembled shape, read as is; omitted, the
            ``predictor_needs`` sections of the per-study claims files.
    """
    if path is not None:
        with Path(path).open(encoding="utf8") as stream:
            return yaml.safe_load(stream)
    needs: dict[str, Any] = {"version": NEEDS_VERSION, "studies": {}}
    for study in STUDY_IDS:
        doc = load_study_claims(study)
        if doc is not None:
            needs["studies"][study] = doc["predictor_needs"]
    return needs


def validate(
    needs: dict[str, Any] | None = None,
    registry: dict[str, Any] | None = None,
) -> None:
    """Raise ``ValueError`` on any inconsistency between the two registries.

    Args:
        needs: Already-parsed predictor needs; ``None`` loads the shipped one.
        registry: Already-parsed claim registry; ``None`` loads the shipped one.

    Raises:
        ValueError: The studies documented differ from the active ones, an
            entry's status or blocker is not usable, or it points at a claim
            that does not exist or belongs to another study.
    """
    needs = needs if needs is not None else load_needs()
    registry = registry if registry is not None else load_registry()

    active = set(registry["scope"]["active_studies"])
    documented = set(needs["studies"])
    if active != documented:
        raise ValueError(
            f"predictor_needs must document exactly the active studies; "
            f"missing {sorted(active - documented)}, "
            f"extra {sorted(documented - active)}"
        )

    claims_by_id = {claim["id"]: claim for claim in registry["claims"]}
    for study, entries in needs["studies"].items():
        if not entries:
            raise ValueError(f"{study}: empty predictor list")
        for entry in entries:
            name = entry.get("predictor")
            if not name:
                raise ValueError(f"{study}: entry without a predictor name")
            status = entry.get("status")
            if status not in STATUSES:
                raise ValueError(f"{study}/{name}: unknown status {status!r}")
            if status == "published" and not entry.get("cells"):
                raise ValueError(f"{study}/{name}: published without cells")
            if status != "published" and entry.get("cells"):
                raise ValueError(f"{study}/{name}: cells on a {status} entry")
            if status in NEEDS_BLOCKER and not entry.get("blocker"):
                raise ValueError(f"{study}/{name}: {status} without a blocker")
            if status == "shared":
                see = entry.get("see")
                if see not in needs["studies"]:
                    raise ValueError(f"{study}/{name}: shared points at {see!r}")
            for claim_id in entry.get("claims", ()):
                claim = claims_by_id.get(claim_id)
                if claim is None:
                    raise ValueError(f"{study}/{name}: unknown claim {claim_id!r}")
                if claim["study"] != study:
                    raise ValueError(
                        f"{study}/{name}: claim {claim_id!r} belongs to "
                        f"{claim['study']!r}"
                    )


def published_cell_patterns(needs: dict[str, Any] | None = None) -> set[str]:
    """Cell tags and ``*``-suffixed prefixes named by published entries.

    Args:
        needs: Already-parsed predictor needs; ``None`` loads the shipped one.
    """
    needs = needs if needs is not None else load_needs()
    return {
        pattern
        for entries in needs["studies"].values()
        for entry in entries
        for pattern in entry.get("cells", ())
    }
