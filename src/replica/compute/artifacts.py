"""Producer identities, environments and immutable model resolution."""

import hashlib
import json
import platform
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from replica.artifacts import SCHEMA_VERSION
from replica.context import corpus_digest
from replica.core import Stimulus
from replica.feature.backends.hub import resolve_model_bundle

from ..utils.digest import file_sha256
from ..utils.git import git_output
from .cache import BOS_SENSITIVE
from .methods import (
    METHOD_SPECS,
    ComputeContext,
    _apply_start_record,
    _validate_context_request,
    _validate_start_request,
)
from .units import METHOD_STORAGE, bound_layer, boundary_rule


def artifact_identity(
    method: str,
    context: ComputeContext,
    stimuli: list[Stimulus],
    model: dict,
    units: dict[str, str] | None = None,
    rule: str | None = None,
) -> dict:
    """Identify resolved options, implementation, model and ordered corpus units.

    The identity also names the method's storage class, its layer (and the
    study of a layer-2 method) and, for a unit-bound method, the boundary
    rule the estimator used and the layer whose units that rule binds.

    Args:
        method: Computation method.
        context: All requested computation settings, including resolved language.
        stimuli: Ordered stimuli after segmentation and sharding.
        model: Immutable model and tokenizer snapshot identity.
        units: Unit of every value key the table stores, recorded for token
            tables and the word tables derived from them.
        rule: Boundary rule named by the caller for a unit-bound method;
            ``None`` records the method's default.
    """
    # Permission to reproduce an error does not change the numerical method.
    options = asdict(context)
    options.pop("allow_reproduction")
    corpus = corpus_digest(stimuli, context.unit, context.context_scope)
    storage = METHOD_STORAGE[method]
    identity = {
        "schema_version": SCHEMA_VERSION,
        "method": method,
        "storage_class": storage.storage_class,
        "layer": storage.layer,
        "options": options,
        "model": model,
        "corpus_sha256": corpus,
        "n_stimuli": len(stimuli),
        "source_sha256": source_digest(),
        "versions": package_versions(),
    }
    if storage.study is not None:
        identity["study"] = storage.study
    boundary = boundary_rule(method, rule)
    if boundary is not None:
        identity["unit_rule"] = boundary
        identity["unit_layer"] = bound_layer(method)
    if units is not None:
        identity["units"] = dict(units)
    # JSON normalization also turns tuple-valued options into stable lists.
    return json.loads(json.dumps(identity))


def source_digest() -> str:
    """SHA-256 over the paths and contents of the package's Python and R files."""
    source = hashlib.sha256()
    root = Path(__file__).parents[1]
    for path in sorted(root.rglob("*")):
        if path.suffix in {".py", ".R"}:
            source.update(str(path.relative_to(root)).encode())
            source.update(bytes.fromhex(file_sha256(path)))
    return source.hexdigest()


def package_versions() -> dict[str, str]:
    """Python and the versions of the packages a cell's values depend on."""
    versions = {"python": platform.python_version()}
    for package in (
        "replica",
        "numpy",
        "torch",
        "transformers",
        "tokenizers",
        "wordfreq",
        "wordsprobability",
        "genlm-bytes",
    ):
        try:
            versions[package] = version(
                "replicaverse" if package == "replica" else package
            )
        except PackageNotFoundError:
            versions[package] = "absent"
    return versions


def environment_record(model_device: str | None = None) -> dict:
    """Source commit, dirty state, device, CUDA build and write time of a cell.

    Stored beside the identity and never compared, so it cannot invalidate a
    cache. torch is imported here, so the record is the same whether or not the
    model stack has been loaded.

    Args:
        model_device: Device the scoring model's weights live on, such as
            ``cuda:0``, when the caller knows it.

    Returns:
        ``code_commit``, ``code_dirty``, ``device`` (the CUDA device torch can
        see at call time, or ``cpu``), ``model_device``, ``cuda`` and
        ``computed_at``.
    """
    root = Path(__file__).resolve().parents[3]
    commit = git_output(root, "rev-parse", "HEAD") if (root / ".git").exists() else None
    dirty = None
    if commit is not None:
        status = git_output(root, "status", "--porcelain", "--", "src/replica")
        dirty = None if status is None else bool(status)
    device, cuda = "cpu", None
    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None and torch.cuda.is_available():
        device = torch.cuda.get_device_name(torch.cuda.current_device())
        cuda = torch.version.cuda
    return {
        "code_commit": commit,
        "code_dirty": dirty,
        "device": device,
        "model_device": model_device,
        "cuda": cuda,
        "computed_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def resolve_model_snapshot(method: str, context: ComputeContext) -> tuple[str, dict]:
    """Materialize one immutable snapshot for both model and tokenizer loaders.

    Args:
        method: Frequency tables do not need a model snapshot.
        context: Requested model id and optional Hub revision.
    """
    _validate_start_request(method, context)
    _validate_context_request(method, context)
    if method == "unigram_surprisal":
        from replica.feature.frequency.registry import table_path

        table = table_path(context.model_id)
        if table is None:
            return context.model_id, {
                "source": context.model_id,
                "revision": "package-version",
            }
        return context.model_id, {
            "source": context.model_id,
            "table": Path(table).name,
            "sha256": file_sha256(Path(table)),
        }
    path, identity = resolve_model_bundle(context.model_id, context.model_revision)
    context = _apply_start_record(method, context, tokenizer_path=path)
    if METHOD_SPECS[method].supports_context_window:
        from transformers import AutoConfig

        from replica.feature.backends.generation import usable_position_limit

        limit = usable_position_limit(
            context.model_id,
            AutoConfig.from_pretrained(path),
            context_window=(context.model_identity or {}).get(
                "context_window", context.context_window
            ),
        )
        identity = {**identity, "context_window": limit}
    identity = {**identity, "start": _start_record(path, method, context)}
    if context.dtype is not None or context.device_map is not None:
        identity["precision"] = {
            "dtype": context.dtype,
            "device_map": context.device_map,
        }
    return path, identity


def _start_record(path: str, method: str, context: ComputeContext) -> dict:
    """The method's effective prefix or recipe-specific input policy."""
    from transformers import AutoConfig, AutoTokenizer

    from replica.feature.backends.start_state import StartState, resolve_start_state

    _validate_start_request(method, context)
    tokenizer = AutoTokenizer.from_pretrained(path)
    policy = METHOD_SPECS[method].start_policy
    if policy == "space_context":
        return {
            "basis": "recipe",
            "add_special_tokens": True,
            "empty_context_text": " ",
            "empty_context_ids": tokenizer(" ")["input_ids"],
        }
    if policy == "unprefixed_embeddings":
        return {
            "basis": "recipe",
            "generation_add_special_tokens": True,
            "embedding_add_special_tokens": False,
            "first_unit": "unscored",
        }
    if policy == "resolved":
        return resolve_start_state(
            tokenizer,
            path,
            start_ids=context.start_ids,
            add_bos=method not in BOS_SENSITIVE or context.add_bos,
        ).record()
    if policy == "sampling":
        state = resolve_start_state(tokenizer, path, start_ids=context.start_ids)
        if state.ids:
            return state.record()
        ident, basis = tokenizer.eos_token_id, "eos_fallback"
        if ident is None:
            raise ValueError("Unconditional sampling requires a BOS or EOS token")
    elif policy == "tokenizer_bos":
        ident, basis = tokenizer.bos_token_id, "upstream_tokenizer"
    elif policy in {"model_bos", "decoded_model_bos"}:
        ident, basis = AutoConfig.from_pretrained(path).bos_token_id, "model_config"
        if policy == "decoded_model_bos" and ident is None:
            ident, basis = tokenizer.bos_token_id, "tokenizer"
    else:
        raise ValueError(f"Unknown start policy {policy!r}")
    ids = [] if ident is None else [int(ident)]
    if policy == "decoded_model_bos":
        if ident is None:
            raise ValueError("The lens recipe requires a BOS token")
        ids = tokenizer(tokenizer.decode(int(ident)), add_special_tokens=False)[
            "input_ids"
        ]
    record = StartState(
        tuple(ids), tuple(tokenizer.convert_ids_to_tokens(ids)), basis
    ).record()
    if policy == "model_bos":
        record["text_add_special_tokens"] = True
    return record
