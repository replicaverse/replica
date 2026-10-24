"""Predictor declarations, source loading and alignment for caller-defined designs."""

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from replica.artifacts import manifest_path
from replica.utils.naming import safe_lm


@dataclass(frozen=True)
class Predictor:
    """One predictor in the grid's catalog.

    ``pickle_suffix`` is everything after the LM prefix in the predictor-pickle
    filename. ``key`` is the value read from each ``FeatureModelOutput.values``.
    ``extra_baseline`` entries are ``(label, pickle_suffix, feature_key)``.
    """

    label: str
    pickle_suffix: str
    key: str
    citation: str
    extra_baseline: tuple = ()


def _model_prefix(lm: str, aliases: Mapping[str, str] | None) -> str:
    """Normalize an identity without accepting unsafe or conflicting aliases."""
    model_id = aliases.get(lm, lm) if aliases is not None else lm
    identities = [model_id, *(aliases.values() if aliases is not None else ())]
    prefixes: dict[str, str] = {}
    for identity in identities:
        if (
            not isinstance(identity, str)
            or not identity
            or any(
                character.isspace() or character in "\\{}\x00" for character in identity
            )
            or any(
                part in {"", ".", ".."}
                for part in identity.replace(":", "/").split("/")
            )
        ):
            raise ValueError(f"invalid canonical model identifier: {identity!r}")
        prefix = safe_lm(identity)
        if prefix in prefixes and prefixes[prefix] != identity:
            raise ValueError(
                "model identifiers collide in cache filenames: "
                f"{prefixes[prefix]!r} and {identity!r}"
            )
        prefixes[prefix] = identity
    return safe_lm(model_id)


def pickle_template_for(
    lm: str, suffix: str, *, aliases: Mapping[str, str] | None = None
) -> str:
    """Build a predictor filename template from a canonical model identifier.

    Args:
        lm: Model identifier; slash and colon separators are normalized
            by the same convention as compute output filenames.
        suffix: Predictor tag sitting between the model prefix and the dataset.
        aliases: Optional application aliases mapped to canonical identifiers.
            An unlisted identifier is used directly. Application short keys have
            no implicit meaning: use ``ai-forever/mGPT`` or supply
            ``aliases={"mgpt": "ai-forever/mGPT"}``.

    Returns:
        A filename with a ``{ds}`` placeholder for the dataset name.

    Raises:
        ValueError: An identity is empty or unsafe, or distinct identifiers in
            the supplied mapping normalize to the same filename prefix.
    """
    return f"{_model_prefix(lm, aliases)}_{suffix}_{{ds}}.pkl"


MIN_VALID_PICKLE_BYTES = 100


def load_pickle_outputs(path: Path, *, allow_reproduction: bool = False) -> dict:
    """Read one predictor cell into the expected dict shape.

    Reads the Hugging Face Parquet predictor store by cell tag (``path.stem``),
    falling back to the local pickle at *path* when the Hub is unreachable.

    Args:
        path: Local pickle path; its stem doubles as the store's cell tag.
        allow_reproduction: Required when the cell's declared method is kept
            for reproduction only.

    Returns:
        Stimulus id to list of ``FeatureModelOutput``, one entry per unit.
    """
    from replica.predictors import load_outputs_for_tag

    data = load_outputs_for_tag(
        Path(path).stem, local_path=path, allow_reproduction=allow_reproduction
    )
    if not isinstance(data, dict):
        raise TypeError(
            f"{path}: expected dict[stim_id, list[FeatureModelOutput]], "
            f"got {type(data).__name__}"
        )
    return data


def resolve_existing_pickle(
    cdir: Path,
    lm: str,
    suffix: str,
    ds: str,
    *,
    aliases: Mapping[str, str] | None = None,
) -> Path | None:
    """Resolve a predictor pickle with the exact requested model and method tag.

    Seed-tagged suffixes match that seed alone. Files at or below
    ``MIN_VALID_PICKLE_BYTES`` count as absent. A sidecar's declared model must
    match when present; payload integrity is checked by the reader.

    Args:
        cdir: Compute directory to look in.
        lm: Canonical model identifier, or a caller-supplied alias.
        suffix: Predictor tag, including any seed tag such as ``_s0``.
        ds: Dataset name completing the filename.
        aliases: Optional application aliases mapped to canonical identifiers.

    Returns:
        The path of a sufficiently large pickle, or ``None`` when absent.
    """
    path = cdir / pickle_template_for(lm, suffix, aliases=aliases).format(ds=ds)
    if not path.exists() or path.stat().st_size <= MIN_VALID_PICKLE_BYTES:
        return None
    sidecar = manifest_path(path)
    if sidecar.exists():
        try:
            metadata = json.loads(sidecar.read_text())
            identity = metadata["identity"]
            recorded = identity.get("options", {}).get("model_id")
            if recorded is None:
                recorded = identity.get("model", {}).get("source")
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            raise ValueError(f"{path}: invalid model identity metadata") from exc
        model_id = aliases.get(lm, lm) if aliases is not None else lm
        if recorded is not None and recorded != model_id:
            raise ValueError(
                f"{path}: stored model {recorded!r} differs from {model_id!r}"
            )
    return path


def _load_first_source(
    cdir: Path,
    templates: tuple[str, ...],
    dataset: str,
    loader: Callable[[Path], dict],
) -> dict:
    """Load the first of *templates* the loader holds; the last one's error propagates."""
    *preferred, last = templates
    for template in preferred:
        try:
            return loader(cdir / template.format(ds=dataset))
        except FileNotFoundError:
            continue
    return loader(cdir / last.format(ds=dataset))


def check_source_alignment(
    dataset: str, sources: Mapping[str, Mapping[str, Sequence]]
) -> None:
    """Require every predictor source to match the first on stimuli and units.

    Args:
        dataset: Corpus name, used in the error message.
        sources: Source label to ``{stim_id: per-unit outputs}``; the first
            entry is the reference the others must match.

    Raises:
        ValueError: A source differs from the reference in its stimulus ids,
            or in the unit count of a stimulus, which the message names.
    """
    (ref_label, ref), *others = sources.items()
    for label, payload in others:
        if set(payload) != set(ref):
            extra = set(payload) - set(ref)
            missing = set(ref) - set(payload)
            raise ValueError(
                f"{dataset}: stim_id mismatch in {label} pickle. "
                f"extra={sorted(extra)[:3]} missing={sorted(missing)[:3]}"
            )
    for sid, ref_units in ref.items():
        for label, payload in others:
            if len(payload[sid]) != len(ref_units):
                raise ValueError(
                    f"{dataset}/{sid}: {label} has {len(payload[sid])} "
                    f"units, {ref_label} has {len(ref_units)}"
                )
