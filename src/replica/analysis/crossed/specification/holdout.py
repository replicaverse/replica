"""Frozen participant/material holdout design and training inputs."""

import hashlib

import numpy as np


def split(frame, *, fraction=0.25, seed=20260927):
    """Freeze the independent holdout using identities only, before fitting.

    Args:
        frame: Observations with participant, material and item identities.
        fraction: Fraction of unique identities held out on each axis.
        seed: Salt for deterministic identity hashes, independent of row order.
    """
    if not 0 < fraction < 1:
        raise ValueError("Holdout fraction must be strictly between zero and one")
    chosen = {}
    for group in ("participant", "material"):
        if frame[group].isna().any():
            raise ValueError("Missing independent sampling identity")
        values = sorted(
            frame[group].astype(str).unique(),
            key=lambda value: hashlib.sha256(
                f"{seed}:{group}:{value}".encode()
            ).hexdigest(),
        )
        count = max(2, int(np.ceil(len(values) * fraction)))
        if len(values) - count < 2:
            raise ValueError("Insufficient independent groups for training and test")
        chosen[group] = values[:count]
    reader = frame.participant.astype(str).isin(chosen["participant"])
    material = frame.material.astype(str).isin(chosen["material"])
    role = np.where(
        reader & material, "test", np.where(~reader & ~material, "train", "unused")
    )
    if not np.any(role == "test") or not np.any(role == "train"):
        raise ValueError("Empty independent training or test rectangle")
    for group in ("participant", "material", "item"):
        if set(frame.loc[role == "test", group]) & set(
            frame.loc[role == "train", group]
        ):
            raise ValueError("Test identity leaks into training")
    return role, {
        "protocol": "independent-readers-materials-v1",
        "seed": seed,
        "fraction_per_axis": fraction,
        "test_identities": chosen,
        "rows": {key: int(np.sum(role == key)) for key in ("train", "test", "unused")},
        "inference": "conditional on the recorded fitted training models",
    }


def prepare(frame, specification, out, *, fraction=0.25, seed=20260927):
    """Write independent training inputs and audit their random slopes anew.

    Args:
        frame: Matched complete observations for every compared model.
        specification: Explicit mixed-model formulas and full-data design audits.
        out: Fresh directory for immutable split, frames and training specification.
        fraction: Fraction of unique identities held out on each axis.
        seed: Salt for reproducible identity-based holdout assignment.
    """
    import json

    from replica.analysis.crossed.specification.random_effects import (
        random_terms,
        slope_audit,
    )
    from replica.utils.digest import file_sha256

    frame = frame.copy()
    frame["role"], receipt = split(frame, fraction=fraction, seed=seed)
    train = frame.loc[frame.role.eq("train")].copy()
    audits = [
        slope_audit(train, audit["group"], audit["requested"])
        for audit in specification["random_effect_audits"]
    ]
    spec = {
        **specification,
        "random_terms": random_terms(audits),
        "random_effect_audits": audits,
        "bootstrap_samples": 0,
        "status": "independent predictive training fit; inference is conditional on this fit",
    }
    out.mkdir(parents=True, exist_ok=False)
    for name, data in (("observations", frame), ("training", train)):
        data.to_csv(
            out / (name + ".tsv.gz"),
            sep="\t",
            index=False,
            compression={"method": "gzip", "mtime": 0},
        )
    (out / "specification.json").write_text(json.dumps(spec, indent=2) + "\n")
    receipt["files"] = {path.name: file_sha256(path) for path in sorted(out.iterdir())}
    (out / "source.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt
