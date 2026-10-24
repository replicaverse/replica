"""Audit crossed random-slope designs before fitting participant observations."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd


def slope_audit(frame: pd.DataFrame, group: str, columns: list[str]) -> dict:
    """Identify estimable within-group slopes without consulting response outcomes.

    Args:
        frame: Complete predictor design and authentic grouping identities.
        group: Participant, item or source-material grouping column.
        columns: Ordered candidate slopes, with focal terms before nuisance terms.

    Returns:
        Retained columns, reasons for omissions and group-variation counts.
        This checks structural identifiability, not statistical power or convergence.
    """
    if frame.empty or frame[group].isna().any():
        raise ValueError("Random effects require nonmissing observation identities")
    if len(columns) != len(set(columns)):
        raise ValueError("Duplicate random-slope term")
    values = frame[columns].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Random-slope audit requires complete finite predictors")
    centered = frame[columns] - frame.groupby(group)[columns].transform("mean")
    spans = frame.groupby(group)[columns].agg(lambda x: x.max() - x.min())
    varying = (spans > 0).sum().to_dict()
    design = np.empty((len(frame), 0))
    retained, omitted = [], {}
    for column in columns:
        if varying[column] < 2:
            omitted[column] = "varies within fewer than two groups"
            continue
        candidate = centered[column].to_numpy(dtype=float)
        scale = np.linalg.norm(candidate)
        candidate = candidate / scale if scale else candidate
        trial = np.column_stack([design, candidate])
        if np.linalg.matrix_rank(trial) == trial.shape[1]:
            retained.append(column)
            design = trial
        else:
            omitted[column] = "aliased with earlier within-group slopes"
    return {
        "group": group,
        "groups": int(frame[group].nunique()),
        "requested": columns,
        "retained": retained,
        "omitted": omitted,
        "varying_groups": {key: int(value) for key, value in varying.items()},
        "partition_sha256": hashlib.sha256(
            pd.factorize(frame[group], sort=False)[0].astype("<i8").tobytes()
        ).hexdigest(),
    }


def random_terms(audits: list[dict]) -> list[str]:
    """Build explicit independent intercept/slope terms from reviewed design audits.

    Args:
        audits: Output of :func:`slope_audit` for each grouping factor.

    All compared fixed-effect models must use the same returned terms.
    """
    terms = []
    partitions = {}
    for audit in audits:
        group = audit["group"]
        if audit["groups"] < 2:
            raise ValueError(f"Random effects need at least two {group} levels")
        names = [group, *audit["retained"]]
        if any(not name.isidentifier() for name in names):
            raise ValueError("Formula column names must be simple identifiers")
        partition = audit.get("partition_sha256")
        if partition in partitions:
            previous = partitions[partition]
            if audit["retained"] != previous["retained"]:
                raise ValueError(
                    "Equivalent grouping factors must request the same slope design"
                )
            audit["alias_of"] = previous["group"]
            audit["group_omission"] = (
                "identical observation partition; duplicate variance components are unidentified"
            )
            continue
        if partition is not None:
            partitions[partition] = audit
        terms.append(f"(1 | {group})")
        terms.extend(f"(0 + {column} | {group})" for column in audit["retained"])
    return terms
