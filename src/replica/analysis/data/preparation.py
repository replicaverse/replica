"""Resolve indexed quantities into one aligned scalar observation table."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np

from replica.analysis.contracts.grouped import GroupedModel
from replica.analysis.contracts.labels import categorical_labels
from replica.analysis.data.arrays import _sentence_ids_for_indices, _transform_dv
from replica.analysis.data.filters import OutlierFilter
from replica.analysis.data.observations import ObservationTable
from replica.analysis.specification.config import AnalysisConfig, _identity_response
from replica.contracts.features import (
    align_values,
    block_origins,
)
from replica.feature.resolution import resolve_features

if TYPE_CHECKING:
    from replica.analysis.specification.evaluation import EvaluationSpec
    from replica.dataset import Dataset


def _all_features(spec, analysis):
    features = {}
    for feature in (*spec.predictors, *spec.controls):
        features[feature.name] = feature
    selected = {f.name: f for f in (*spec.targets, *spec.predictors, *spec.controls)}
    for step in analysis.preprocessing:
        if selected.get(step.feature.name) is not step.feature:
            raise ValueError("Preprocessing must reference a selected feature object")
        if any(step.feature is target for target in spec.targets) and any(
            reference.name in {f.name for f in spec.predictors}
            for reference in step.against
        ):
            raise ValueError("Target preprocessing cannot depend on focal predictors")
        for feature in step.against:
            if feature.name in selected and selected[feature.name] is not feature:
                raise ValueError(f"Ambiguous preprocessing input {feature.name!r}")
            previous = features.get(feature.name)
            if previous is not None and previous is not feature:
                raise ValueError(f"Ambiguous feature name {feature.name!r}")
            features[feature.name] = feature
    return list(features.values())


def _resolved_inputs(dataset, spec, analysis):
    if not dataset.stimuli():
        raise ValueError("Dataset has no stimuli; load its materials before evaluation")
    features = _all_features(spec, analysis)
    requested = [*spec.targets, *features]
    if spec.observations is not None:
        requested.append(spec.observations)
    resolved = resolve_features(
        dataset,
        requested,
        segmentation=spec.segmentation,
        layer_from=(spec.observations,)
        if spec.observations is not None
        else spec.targets,
    )
    values_by_feature = {
        id(feature): values for feature, values in zip(requested, resolved, strict=True)
    }
    focal_identities = {
        values_by_feature[id(feature)].provenance["outcome_identity"]
        for feature in spec.predictors
    }
    for step in analysis.preprocessing:
        if any(step.feature is target for target in spec.targets):
            for reference in step.against:
                value = values_by_feature[id(reference)]
                if any(
                    identity in focal_identities
                    for block in value.blocks
                    for origins in block_origins(value, block)
                    for identity, _ in origins
                ):
                    raise ValueError(
                        "Target preprocessing cannot depend on focal predictors, including aliases"
                    )
    return features, values_by_feature


def _model_name(values):
    names = []

    def collect(provenance):
        name = provenance.get("model_name")
        if name is not None and name not in names:
            names.append(name)
        for source in provenance.get("dependencies", ()):
            if isinstance(source, Mapping):
                collect(source)

    for value in values:
        collect(value.provenance)
    return names[0] if len(names) == 1 else (" + ".join(names) if names else "dataset")


def _measurement(feature, values):
    return {
        "name": feature.name,
        "measurement": None
        if values.measurement is None
        else values.measurement.to_dict(),
    }


def _row_digest(keys):
    return hashlib.sha256(
        json.dumps(keys, default=str, separators=(",", ":")).encode()
    ).hexdigest()


def _scalar(block, name):
    if block.values.ndim != 1 or block.values.dtype.kind not in "biuf":
        raise ValueError(
            f"Scalar analysis requires numeric scalar feature {name!r}; select dimensions or use EncodingConfig"
        )
    return np.asarray(block.values, dtype=float)


def _aligned_mask(response, inputs, block_index, target_name, missing):
    y = _scalar(response, target_name)
    candidate = response.eligible & np.isfinite(y)
    mask = candidate.copy()
    for name, aligned in inputs.items():
        values = _scalar(aligned[block_index], name)
        mask &= aligned[block_index].eligible & np.isfinite(values)
    if missing == "error" and not mask.all():
        raise ValueError(f"Incomplete observations for target {target_name!r}")
    return y, candidate, mask


def _target_policy(raw, participants, config):
    outliers = OutlierFilter(
        floor=config.outlier_floor,
        ceiling=config.outlier_ceiling,
        method=config.outlier_method,
        k=config.outlier_k,
    )
    keep = outliers.keep_mask(raw, participants)
    after_outliers = int(keep.sum())
    transformed = np.full_like(raw, np.nan, dtype=float)
    transformed[keep] = _transform_dv(config.transform_dv, raw[keep])
    keep &= np.isfinite(transformed)
    if config.missing == "error" and int(keep.sum()) != after_outliers:
        raise ValueError("Target transform produced nonfinite values")
    return transformed, keep, after_outliers


def _prepare_target(dataset, spec, config, target, features, resolved):
    target_values = resolved[id(target)]
    if len({block.layer for block in target_values.blocks}) > 1:
        raise ValueError(
            "A scalar target spans multiple annotation layers; select one explicit layer before evaluation"
        )
    anchor = (
        resolved[id(spec.observations)]
        if spec.observations is not None
        else target_values
    )
    stimuli = dataset.stimuli()
    order = {stimulus.stimulus_id: i for i, stimulus in enumerate(stimuli)}
    if not order:
        raise ValueError("Dataset has no stimuli; load its materials before evaluation")
    anchor = replace(
        anchor,
        blocks=tuple(
            sorted(anchor.blocks, key=lambda block: order[block.stimulus.stimulus_id])
        ),
    )
    if not anchor.blocks:
        raise ValueError(f"No observations for target {target.name!r}")
    target_blocks = align_values(target_values, anchor)
    inputs = {
        feature.name: align_values(resolved[id(feature)], anchor)
        for feature in features
    }
    pmap: dict[str, int] = {}
    imap: dict[str, int] = {}
    y_chunks, p_chunks, item_chunks, group_chunks, sentence_chunks = [], [], [], [], []
    pid_chunks, sid_chunks, unit_chunks = [], [], []
    column_chunks: dict[str, list[np.ndarray]] = {
        feature.name: [] for feature in features
    }
    factors: dict[str, list[np.ndarray]] = (
        {} if config.design is None else {f.name: [] for f in config.design.factors}
    )
    target_origins: list[frozenset] = []
    input_origins: dict[str, list[frozenset]] = {f.name: [] for f in features}
    row_keys: list[tuple] = []
    n_input = n_candidate = n_complete = 0
    for bi, block in enumerate(anchor.blocks):
        if (
            "unit" not in block.index.names
            or block.value_dims
            and spec.observations is None
        ):
            raise ValueError(
                "Scalar observation indexes require annotation units; use EncodingConfig for sampled targets"
            )
        response = target_blocks[bi]
        y, candidate, mask = _aligned_mask(
            response, inputs, bi, target.name, config.missing
        )
        n_input += len(y)
        n_candidate += int(candidate.sum())
        n_complete += int(mask.sum())
        selected = np.flatnonzero(mask)
        if not len(selected):
            continue
        indices = np.asarray(
            [key[block.index.names.index("unit")] for key in block.index.keys],
            dtype=int,
        )
        sid = block.stimulus.stimulus_id
        pid = block.participant_id or ""
        if pid not in pmap:
            pmap[pid] = len(pmap)
        if sid not in imap:
            imap[sid] = len(imap)
        y_chunks.append(y[selected])
        p_chunks.append(np.full(len(selected), pmap[pid], dtype=int))
        item_chunks.append(np.full(len(selected), imap[sid], dtype=int))
        group_chunks.append(np.full(len(selected), order[sid], dtype=int))
        sentence_chunks.append(
            _sentence_ids_for_indices(
                block.stimulus, imap[sid], indices[selected], block.layer
            )
        )
        pid_chunks.append(np.full(len(selected), pid, dtype=object))
        sid_chunks.append(np.full(len(selected), sid, dtype=object))
        unit_chunks.append(indices[selected])
        for name, chunks in column_chunks.items():
            chunks.append(inputs[name][bi].values[selected])
            origins = block_origins(
                resolved[id(next(f for f in features if f.name == name))],
                inputs[name][bi],
            )
            input_origins[name].extend(origins[i] for i in selected)
        origins = target_outcome_origins(target_values, response)
        target_origins.extend(origins[i] for i in selected)
        row_keys.extend(block.index.keys[i] for i in selected)
        if config.design is not None:
            for factor in config.design.factors:
                factors[factor.name].append(
                    factor.values_from(
                        stimulus=block.stimulus,
                        metadata=block.metadata,
                        participant_id=block.participant_id,
                        layer=block.layer,
                        indices=indices[selected],
                    )
                )
    if not y_chunks:
        raise ValueError(
            f"No valid observations after filtering for target {target.name!r}"
        )
    raw = np.concatenate(y_chunks)
    participants = np.concatenate(p_chunks)
    columns = {name: np.concatenate(chunks) for name, chunks in column_chunks.items()}
    transformed, keep, after_outliers = _target_policy(raw, participants, config)
    retained = np.flatnonzero(keep)
    if not len(retained):
        raise ValueError(
            f"No valid observations after target policy for {target.name!r}"
        )
    source_values = {f.name: resolved[id(f)] for f in features}
    source_values[target.name] = target_values
    response_record = {
        **dict(target_values.provenance),
        "target": target.name,
        "model_name": _model_name(source_values.values()),
        "configuration": config.metadata(),
        "feature_measurements": [_measurement(f, resolved[id(f)]) for f in features],
        "target_measurement": _measurement(target, target_values),
        "feature_sources": {f.name: dict(resolved[id(f)].provenance) for f in features},
        "observation_anchor": None
        if spec.observations is None
        else spec.observations.name,
        "observation_index_names": list(anchor.blocks[0].index.names),
        "row_exclusions": {
            "input_units_after_loader": n_input,
            "response_or_dataset_mask": n_input - n_candidate,
            "missing_features": n_candidate - n_complete,
            "outlier_filter": n_complete - after_outliers,
            "nonfinite_transformed_response": after_outliers - len(retained),
            "retained": len(retained),
        },
        "cohort_identity": _row_digest([row_keys[i] for i in retained]),
        "target_identity": _row_digest(
            [sorted(target_origins[i], key=str) for i in retained]
        ),
        "score_scale": config.score_scale,
    }
    return ObservationTable(
        transformed[keep],
        {name: values[keep] for name, values in columns.items()},
        participants[keep],
        np.concatenate(item_chunks)[keep],
        np.concatenate(group_chunks)[keep],
        np.concatenate(sentence_chunks)[keep],
        participant_ids=pmap,
        item_ids=imap,
        stimulus_ids=np.concatenate(sid_chunks)[keep],
        unit_indices=np.concatenate(unit_chunks)[keep],
        response=response_record,
        factors={
            name: categorical_labels(np.concatenate(chunks)[keep], name=name)
            for name, chunks in factors.items()
        },
        raw_y=raw[keep],
        row_keys=tuple(row_keys[i] for i in retained),
        target_origins=tuple(target_origins[i] for i in retained),
        input_origins={
            name: tuple(origins[i] for i in retained)
            for name, origins in input_origins.items()
        },
        source_values=source_values,
    )


def prepare_observations(
    dataset: Dataset,
    spec: EvaluationSpec,
    analysis: AnalysisConfig | None = None,
    *,
    target_name: str | None = None,
) -> ObservationTable:
    """Resolve and align scalar quantities without fitting statistical models.

    Args:
        dataset: Loaded material population.
        spec: Targets, input features and optional observation anchor.
        analysis: Target policy, grouping and preprocessing declarations.
        target_name: Selected target name; required with multiple targets.
    """
    spec.__post_init__()
    config = (analysis or AnalysisConfig()).resolve(
        tuple(f.name for f in spec.predictors)
    )
    if target_name is None:
        if len(spec.targets) != 1:
            raise ValueError("Specify target_name when preparing multiple targets")
        target = spec.targets[0]
    else:
        matches = [f for f in spec.targets if f.name == target_name]
        if not matches:
            raise ValueError(f"Unknown target {target_name!r}")
        target = matches[0]
    selected = {f.name for f in (target, *spec.predictors, *spec.controls)}
    config = replace(
        config,
        preprocessing=tuple(
            step for step in config.preprocessing if step.feature.name in selected
        ),
    )
    spec = replace(spec, targets=[target])
    features, resolved = _resolved_inputs(dataset, spec, config)
    return _prepare_target(dataset, spec, config, target, features, resolved)


def target_outcome_origins(values, block):
    """Keep target outcomes and observed contributors, excluding fixed context.

    Args:
        values: Resolved target quantity with declared source scopes.
        block: Aligned target block whose contributor rows are retained.
    """
    identity = values.provenance.get("outcome_identity")
    scopes = values.provenance.get("origin_scopes", {})
    origins = block_origins(values, block)
    excluded = {
        source
        for source, scope in scopes.items()
        if source != identity and scope != "observations"
    }
    if not excluded:
        return origins
    return tuple(
        frozenset(origin for origin in row if origin[0] not in excluded)
        for row in origins
    )


def _input_holdout_origins(observations, declaration):
    """Identify measured contributors carrying a declared held-out coordinate."""
    if declaration is None:
        return {}
    sources = {factor.name: factor.source for factor in declaration.factors}
    heldout = [sources[name] for name in declaration.cv_units]
    coordinates = {
        "participant": "participant",
        "stimulus": "stimulus",
        "occurrence": "unit",
        "word": "unit",
    }
    result = {}
    for name, rows in observations.input_origins.items():
        values = observations.source_values[name]
        scopes = values.provenance.get("origin_scopes", {})
        indexes = values.provenance.get("origin_indexes", {})
        measured = {
            identity for identity, scope in scopes.items() if scope == "observations"
        }
        if not measured:
            continue
        if any(source not in coordinates for source in heldout):
            if any(
                len({key for identity, key in row if identity in measured}) > 1
                for row in rows
            ):
                raise ValueError(
                    "Aggregated observations require explicit contributor alignment for custom held-out factors"
                )
            continue
        relevant = {
            identity
            for identity in measured
            if any(
                coordinates[source] in indexes.get(identity, ()) for source in heldout
            )
        }
        result[name] = tuple(
            frozenset(origin for origin in row if origin[0] in relevant) for row in rows
        )
    return result


def validate_fold_dependencies(
    observations, splits, *, in_sample=False, declaration=None
):
    """Reject outcome leakage and measured contributors pooled across holdouts.

    Args:
        observations: Prepared values and their original contributor identities.
        splits: Boolean training/test masks over the prepared rows.
        in_sample: Explicit whole-population fitting without CV claims.
        declaration: Named holdout factors used to check pooled observations.
    """
    validate_fold_origins(
        observations.target_origins,
        observations.input_origins,
        splits,
        in_sample=in_sample,
        holdout_origins=_input_holdout_origins(observations, declaration),
    )


def validate_fold_origins(
    target_origins, input_origins, splits, *, in_sample=False, holdout_origins=None
):
    """Check outcome lineage across the declared train/test partitions.

    Args:
        target_origins: Original outcome contributors for each target row.
        input_origins: Contributors for each aligned input column.
        splits: Boolean training and test masks over the observation rows.
        in_sample: Explicit fitting on the whole population, without CV claims.
        holdout_origins: Measured input contributors carrying held-out coordinates.
    """
    if in_sample or not target_origins:
        return
    for train, test in splits:
        training = set().union(*(target_origins[i] for i in np.flatnonzero(train)))
        heldout = set().union(*(target_origins[i] for i in np.flatnonzero(test)))
        if training & heldout:
            raise ValueError(
                "A target observation appears in both training and test; split by its original identity"
            )
        for name, origins in (holdout_origins or {}).items():
            left = set().union(*(origins[i] for i in np.flatnonzero(train)))
            right = set().union(*(origins[i] for i in np.flatnonzero(test)))
            if left & right:
                raise ValueError(
                    f"Feature {name!r} pools measured contributors across a held-out factor; aggregate within folds or change the declared holdout"
                )
        for name, origins in input_origins.items():
            consumed = set().union(*(origins[i] for i in np.flatnonzero(train | test)))
            if consumed & heldout:
                raise ValueError(
                    f"Feature {name!r} consumes held-out target observations; use an explicit sequential protocol"
                )


def _validate_named_analysis(analysis, analysis_model, spec):
    if analysis.preprocessing and getattr(analysis_model, "internal_tuning", False):
        raise ValueError(
            "This backend tunes on internal folds without refitting AnalysisConfig.preprocessing; supply a fixed backend or an explicit nested analysis"
        )
    target_steps = [
        step
        for step in analysis.preprocessing
        if any(step.feature is target for target in spec.targets)
    ]
    if target_steps and getattr(analysis_model, "needs_events", False):
        raise ValueError(
            "Continuous-time target preprocessing requires an event-aware analysis; it is not applied by this backend"
        )
    if analysis.score_scale == "original":
        if analysis.analysis_fn is not None or getattr(
            analysis_model, "owns_loglik", False
        ):
            raise ValueError(
                "Original-scale scoring requires the Gaussian prediction interface"
            )
        transform = analysis.transform_dv
        if transform is not _identity_response and not (
            callable(getattr(transform, "inverse", None))
            and callable(getattr(transform, "log_abs_det_jacobian", None))
        ):
            raise ValueError(
                "Original-scale scoring requires transform_dv.inverse and transform_dv.log_abs_det_jacobian"
            )
    if analysis.analysis_fn is not None and analysis.preprocessing:
        raise ValueError(
            "Custom analysis_fn must own its preprocessing; AnalysisConfig.preprocessing is not applied"
        )
    if analysis.design is not None:
        if analysis.analysis_fn is not None:
            raise ValueError("Named Design is not supported by analysis_fn")
        if not isinstance(analysis_model, GroupedModel):
            raise ValueError(
                f"{type(analysis_model).__name__} does not support named Design; use a GroupedModel backend or set design=None"
            )
        analysis_model.validate_design(analysis.design)
        if (
            analysis.design.scoring == "marginal"
            and analysis.sigma_source != "training"
        ):
            raise ValueError("Marginal scoring requires sigma_source='training'")
        numeric_names = {f.name for f in (*spec.controls, *spec.predictors)}
        if numeric_names & set(analysis.design.categorical):
            raise ValueError("Numeric and categorical fixed names must be distinct")
        slopes = {s for r in analysis.design.random_effects for s in r.slopes}
        if slopes - numeric_names:
            raise ValueError(
                f"Random slopes require declared numeric predictors: {sorted(slopes - numeric_names)}"
            )
        analysis.design.for_model(tuple(f.name for f in spec.controls))
