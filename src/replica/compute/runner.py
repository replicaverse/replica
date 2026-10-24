"""Compute and cache LM-derived features for every dataset in REPLICA.

Runs one feature-computing :class:`FeatureModel` (surprisal, entropy, Rényi
entropy, unigram, ... selected by ``--method``) over each stimulus in a
dataset and writes the per-stimulus outputs to ``compute/{tag}.pkl``.
The pickle has the shape :class:`PrecomputedFeatureModel` consumes::

    {stimulus_id: list[FeatureModelOutput]}

The output ``{tag}`` is ``{model}_{method}_{dataset}``, e.g.
``gpt2_surprisal_provo``, ``gpt2_charbeam_surprisal_K5_provo``,
``gpt2_wordsprobability_surprisal_provo``. A per-token cell carries the
``_tokens`` unit suffix and one record per model token; the word cells of
those methods are aggregated from it and record the derivation in their
manifest.
"""

import gc
import json
import logging
import pickle
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

from replica.artifacts import (
    manifest_path,
    matching_manifest,
    validate_artifact,
    write_manifest,
)
from replica.dataset.catalog.registry import REGISTRY
from replica.dataset.catalog.registry import expand as _registry_expand
from replica.feature.core.annotation.alignment import WHITESPACE_UNIT
from replica.feature.core.protocols import FeatureModelOutput
from replica.feature.core.reproduction import check_reproduction_opt_in
from replica.utils.atomic import atomic_pickle, temporary_path
from replica.utils.digest import file_sha256

from .artifacts import artifact_identity, environment_record, resolve_model_snapshot
from .cache import (
    TOKEN_SENSITIVE,
    cache_filename,
    qualified_model_id,
    resolve_context_scope,
)
from .methods import (
    ComputeContext,
    build_model,
    extract_keys,
)
from .units import (
    AGGREGATION_RULES,
    TOKEN_UNIT,
    WORD_JOINED,
    Applicability,
    applicability,
    boundary_rule,
    derivation_plan,
    derive_records,
    derived_keys,
    token_keys,
    value_units,
    verify_token_rows,
)


@dataclass(frozen=True)
class ComputeOutcome:
    """One requested cell's lifecycle result; artifact paths always exist."""

    dataset: str
    method: str
    status: Literal["completed", "cached", "busy", "quarantined", "failed"]
    path: Path | None = None
    error: str | None = None

    @property
    def successful(self) -> bool:
        """Whether this request has a complete, verified artifact."""
        return self.status in {"completed", "cached"}


log = logging.getLogger("replica.compute")

#: Stimuli between full garbage collections. A collect costs O(tracked objects),
#: which grows with the resident corpus, so a collect after every stimulus is
#: slow on a large corpus; one every 50 stimuli still clears the sampling
#: estimators' garbage.
_GC_EVERY = 50

#: Methods whose model primes a word cache it can checkpoint and resume.
_RESUMABLE_PRIMING = frozenset({"mmc_unigram_surprisal"})


@dataclass(frozen=True)
class _LoadedDataset:
    stimuli: list
    n_total: int
    language: str | None
    word_spacing: bool = True


@dataclass(frozen=True)
class _OutputPaths:
    pickle_path: Path
    partial_path: Path
    primed_path: Path
    lock_path: Path
    fail_log: Path


# --------------------------------------------------------------------------
# Compute + cache
# --------------------------------------------------------------------------


def expand_datasets(name: str) -> list[str]:
    """Resolve a single name or group alias into concrete dataset names.

    Args:
        name: A registry key or a group alias.

    Raises:
        SystemExit: When *name* is neither, carrying the registry's message.
    """
    try:
        return _registry_expand(name)
    except KeyError as e:
        raise SystemExit(str(e)) from None


def cache_stem(
    method: str,
    context: ComputeContext,
    dataset_name: str,
) -> str:
    """Return the canonical cache stem for one compute cell.

    Args:
        method: Feature-computation method the cell holds.
        context: Model id and sampling options the cell id names.
        dataset_name: Registry name of the corpus the cell covers.

    Returns:
        The :func:`~replica.compute.cache.cache_filename` name without
        its ``.pkl`` extension.
    """
    filename = cache_filename(
        model_id=context.model_id,
        method=method,
        dataset=dataset_name,
        beam_width=context.beam_width,
        n_samples=context.n_samples,
        max_tokens=context.max_tokens,
        alpha=context.alpha,
        seed=context.seed,
        add_bos=context.add_bos,
        top_p=context.top_p,
        context_scope=context.context_scope,
        unit=context.unit,
        word_starters=context.word_starters,
        window_policy=context.window_policy,
        segmentation=context.segmentation,
    )
    return filename.removesuffix(".pkl")


def compute_for(
    dataset_name: str,
    *,
    method: str,
    model_id: str,
    K: int,
    n_samples: int,
    max_tokens: int,
    alpha: float,
    seed: int | None,
    out_dir: Path,
    add_bos: bool,
    force: bool,
    language_override: str | None = None,
    chunk_size: int | None = None,
    word_starters: tuple[str, ...] = (),
    checkpoint_every: int = 5,
    stim_slice: tuple[int, int] | None = None,
    top_p: float = 0.9,
    context_scope: str = "auto",
    unit: str = "words",
    unit_rule: str | None = None,
    window_policy: str = "reference",
    segmentation: str = "whitespace",
    model_revision: str | None = None,
    store_distribution: bool = False,
    allow_reproduction: bool = False,
    start_ids: tuple[int, ...] | None = None,
    context_window: int | None = None,
    dtype: str | None = None,
    device_map: str | None = None,
) -> ComputeOutcome:
    """Run the LM over every stimulus in *dataset_name* and pickle the outputs.

    Per-stimulus exceptions are caught and logged; the offending stim's
    entry is omitted from the pickle (downstream code will see a missing
    key and fall back to whatever its policy is, typically NaN). A failing
    stimulus does not stop a multi-corpus run.

    Args:
        dataset_name: Registry name of the corpus to run over.
        method: Feature-computation method to compute.
        model_id: Hugging Face model id or frequency-source name.
        K: Character-beam width, read by ``char_beam`` alone.
        n_samples: Monte Carlo draws per word.
        max_tokens: Token budget for one sampled continuation.
        alpha: Rényi order; 1.0 is Shannon entropy.
        seed: Monte Carlo seed; ``None`` leaves the run and cell id unseeded.
        out_dir: Directory the pickle, checkpoint, lock, and failure log go in.
        add_bos: Prepend a beginning-of-sequence token where the method takes
            one.
        force: Recompute even when a readable cell is already there, discarding
            any checkpoint.
        language_override: Language code for language-sensitive methods; folded
            into the model slot of the cell id.
        chunk_size: Monte Carlo batch size, bounding peak memory; ``None`` lets
            the model choose.
        word_starters: End-of-word token prefixes for Monte Carlo word sampling;
            empty selects the default set.
        checkpoint_every: Newly computed stimuli between checkpoint writes.
        stim_slice: Half-open ``(start, end)`` stimulus range for a shard run,
            written to a shard-suffixed pickle; ``None`` covers the corpus.
        top_p: Nucleus-sampling mass for methods that sample whole utterances.
        context_scope: ``auto`` uses the corpus's declared
            ``default_context_scope``; ``stimulus`` and ``story`` force one scope. ``stimulus`` keeps
            page-only context; ``story`` requires verified preceding pages;
            ``sentence`` uses the authors' sentence segmentation.
        unit: ``words``, ``characters`` to attach a per-character layer and
            emit one value per character, or ``tokens`` to emit one value per
            model token with its character span. For the methods in
            :data:`~replica.compute.cache.TOKEN_SENSITIVE`, ``words`` computes
            the token table and derives the word table from it through
            :func:`~replica.feature.core.annotation.alignment.nested_members`; on a
            corpus without word spacing, word surprisal derives from the
            ``char_beam`` character table instead and the other positional
            methods raise.
            A token table stores ``log_p_observed`` in nats; a derived word
            table keeps that sum in nats beside ``surprisal`` in bits;
            entropies are in bits. The manifest identity names each unit.
        unit_rule: Name of the boundary rule a unit-bound method is computed
            under, recorded in the manifest identity. Without it, a corpus
            on which :func:`~replica.compute.units.applicability` finds the
            measure undefined is refused with the reason; a corpus whose units
            are the model's own words records the default rule. A positional
            method takes no rule.
        window_policy: Context policy for ``word_information_value``: ``cap``
            trims to the longest whole-word suffix fitting the model window.
        model_revision: Hub model and tokenizer revision; resolved to one snapshot.
        store_distribution: Also keep each word's full next-token distribution
            beside the scalar, where the method offers one
            (``first_token_entropy``).
        allow_reproduction: Explicitly enable reproduction-only entropy methods.
        start_ids: Optional single-token prefix override for supported methods.
        context_window: Maximum context tokens for supported methods.
        dtype: Parameter precision the weights load in; ``None`` keeps single precision.
        device_map: ``auto`` shards the weights over every visible device.
        segmentation: Word iteration for ``word_information_value``: ``layer``
            uses the corpus's own word layer with no inserted spaces.

    Returns:
        An explicit completed, cached, busy or quarantined outcome. Setup
        errors raise; batch callers record them as failed outcomes.
    """
    check_reproduction_opt_in(method, allow_reproduction=allow_reproduction)
    if checkpoint_every < 1:
        raise ValueError("checkpoint_every must be at least 1")
    # A language override changes unigram values, so it must change the cell
    # id: fold it into the wordfreq model slot before anything is named.
    model_id = qualified_model_id(model_id, language_override)
    rule = boundary_rule(method, unit_rule)

    # Load first, then build the context with the dataset's resolved
    # language (needed for language-sensitive methods like unigram_surprisal).
    loaded = _load_dataset(dataset_name)
    stimuli = loaded.stimuli
    if unit == "characters":
        from replica.feature.core.annotation.characters import attach_characters

        attach_characters(stimuli)
    elif unit == TOKEN_UNIT:
        if store_distribution:
            raise ValueError(
                "--store-distribution stores word-type distributions, which a "
                "token table does not hold"
            )
    elif unit != "words":
        _attach_segment_unit(stimuli, unit, store_distribution)
    # Decided on the whole corpus, before any shard slice, so every shard's
    # manifest records the same multi-word count.
    fit = applicability(
        method,
        unit,
        stimuli,
        word_spacing=loaded.word_spacing,
        unit_rule=unit_rule,
        segmentation=segmentation,
    )
    if not fit.defined:
        raise ValueError(f"{dataset_name}: {fit.reason}")

    context_scope = resolve_context_scope(dataset_name, method, context_scope)
    if context_scope == "story":
        from ..context import attach_preceding_context

        # Resolve preceding pages before slicing: a shard beginning on page 3
        # still needs pages 1 and 2 in its model input and artifact identity.
        attach_preceding_context(stimuli)

    # Optional shard slice, process only ``stimuli[start:end]`` and
    # write to a shard-suffixed pickle. A separate merge step stitches the
    # shards back into the canonical pickle once every shard has landed.
    stimuli, shard_suffix = _apply_stimulus_slice(stimuli, loaded.n_total, stim_slice)
    if not stimuli:
        raise ValueError("No stimuli selected for computation")

    # A sentence-scope run uses the authors' segmentation, attached here so
    # the model consumes a plain layer and never picks a segmenter itself.
    if context_scope == "sentence":
        from replica.dataset.catalog.sentences import attach_sentences

        attach_sentences(dataset_name, stimuli)

    context = ComputeContext(
        model_id=model_id,
        beam_width=K,
        n_samples=n_samples,
        max_tokens=max_tokens,
        alpha=alpha,
        seed=seed,
        add_bos=add_bos,
        dataset_language=loaded.language,
        language_override=language_override,
        chunk_size=chunk_size,
        word_starters=word_starters,
        top_p=top_p,
        context_scope=context_scope,
        unit=unit,
        window_policy=window_policy,
        segmentation=segmentation,
        model_revision=model_revision,
        store_distribution=store_distribution,
        allow_reproduction=allow_reproduction,
        start_ids=start_ids,
        context_window=context_window,
        dtype=dtype,
        device_map=device_map,
    )
    if unit == "words" and method in TOKEN_SENSITIVE and not store_distribution:
        return _derive_word_cell(
            dataset_name,
            method,
            context,
            stimuli,
            loaded.word_spacing,
            out_dir,
            shard_suffix,
            force=force,
            checkpoint_every=checkpoint_every,
        )
    return _compute_cell(
        dataset_name,
        method,
        context,
        stimuli,
        out_dir,
        shard_suffix,
        force=force,
        checkpoint_every=checkpoint_every,
        rule=rule,
        multiword=_multiword_record(fit),
    )


def _multiword_record(fit: Applicability) -> dict | None:
    """Manifest block naming how a cell valued its units of several whitespace words.

    Returns:
        ``None`` for a cell without such units.
    """
    if not fit.multiword_units or fit.rule is None:
        return None
    return {
        "over": WHITESPACE_UNIT,
        "rule": fit.rule,
        "multiword_units": fit.multiword_units,
    }


def _attach_segment_unit(stimuli: list, unit: str, store_distribution: bool) -> None:
    """Attach the registered segmentation *unit* to *stimuli*.

    Raises:
        ValueError: When no segmentation is registered under *unit* or a
            word-type distribution is requested.
    """
    from ..segmentation import SEGMENTERS, attach_segmentation

    if unit not in SEGMENTERS:
        raise ValueError(
            f"unknown unit {unit!r}; registered segmentations: "
            f"{sorted(SEGMENTERS)}, besides words, characters and tokens"
        )
    if store_distribution:
        raise ValueError(
            "--store-distribution stores word-type distributions of word cells"
        )
    attach_segmentation(unit, stimuli)


def _compute_cell(
    dataset_name: str,
    method: str,
    context: ComputeContext,
    stimuli: list,
    out_dir: Path,
    shard_suffix: str,
    *,
    force: bool,
    checkpoint_every: int,
    rule: str | None = None,
    multiword: dict | None = None,
) -> ComputeOutcome:
    """Score *stimuli* with the method's model and publish the cell."""
    tag = cache_stem(method, context, dataset_name)
    paths = _output_paths(out_dir, tag, shard_suffix)

    log.info(
        "[lm  ] method=%s model_id=%s K=%s N=%s L=%s alpha=%s add_bos=%s seed=%s "
        "language=%s unit=%s",
        method,
        context.model_id,
        context.beam_width,
        context.n_samples,
        context.max_tokens,
        context.alpha,
        context.add_bos,
        context.seed,
        context.language_override or context.dataset_language,
        context.unit,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    lock_fp = _acquire_lock(paths.lock_path)
    if lock_fp is None:
        return ComputeOutcome(
            dataset_name, method, "busy", error="Cell is locked by another process"
        )

    try:
        # Resolve and recheck under the stable lock: an earlier writer may have
        # completed after this process loaded the dataset.
        model_path, model_identity = resolve_model_snapshot(method, context)
        per_token = context.unit == TOKEN_UNIT
        units = (
            value_units(token_keys(extract_keys(method, context)))
            if per_token
            else None
        )
        identity = artifact_identity(
            method, context, stimuli, model_identity, units, rule
        )
        primes = method in _RESUMABLE_PRIMING
        if _should_skip_existing(paths.pickle_path, force=force, identity=identity):
            _discard_primed(paths.primed_path)
            return ComputeOutcome(dataset_name, method, "cached", paths.pickle_path)
        # A primed cell rescores every stimulus by cache lookups and skips its
        # partial checkpoint, which may hold stimuli scored against other contexts.
        outputs = (
            {}
            if primes
            else _load_partial_outputs(
                paths.partial_path, force=force, identity=identity
            )
        )
        lm = build_model(
            method, replace(context, model_id=model_path, model_identity=model_identity)
        )
        environment = environment_record(_model_device(lm))
        # Batch-estimate every word up front when the model supports it
        # (sampling-based unigram sources): one vocab-scale forward sweep over
        # the samples instead of a per-word sweep on every __call__. Without
        # this the MMC estimator re-sweeps its sample pool for every word,
        # orders of magnitude slower and it accumulates GPU memory. No-op
        # otherwise.
        if hasattr(lm, "prepopulate_cache"):
            remaining = [s for s in stimuli if s.stimulus_id not in outputs]
            if primes:
                if force:
                    _discard_primed(paths.primed_path)
                lm.prepopulate_cache(
                    remaining,
                    checkpoint=paths.primed_path,
                    owner=identity,
                    target_layer=context.unit,
                )
            else:
                lm.prepopulate_cache(remaining, target_layer=context.unit)
        extract = extract_keys(method, replace(context, model_id=model_path))
        if per_token:
            extract = token_keys(extract)
        t0 = time.time()
        failed = _compute_stimuli(
            stimuli,
            outputs,
            lm,
            extract,
            partial_path=paths.partial_path,
            checkpoint_every=checkpoint_every,
            identity=identity,
            environment=environment,
            target_layer=context.unit if context.unit != "words" else None,
            coverage="words" if method in WORD_JOINED else None,
        )
        # Taken again after scoring: the model stack loads lazily, so the
        # device it ran on is only known once the outputs exist.
        environment = environment_record(_model_device(lm))
        published = _write_and_audit_outputs(
            paths.pickle_path,
            paths.partial_path,
            outputs,
            failed=failed,
            expected=len(stimuli),
            identity=identity,
            environment=environment,
            window=_model_record(lm, "window") if per_token else None,
            sampling=_model_record(lm, "sampling") if per_token else None,
            multiword=multiword,
        )
        if published == paths.pickle_path:
            _discard_primed(paths.primed_path)
        log.info(
            "[done] wrote %s  (%.1f s)  ok=%d  failed=%d",
            paths.pickle_path,
            time.time() - t0,
            len(outputs),
            len(failed),
        )
    finally:
        _release_lock(lock_fp, paths.lock_path)

    return _outcome(dataset_name, method, paths, published, failed)


def _discard_primed(path: Path) -> None:
    """Remove a priming checkpoint and the temporary file its writer uses."""
    path.unlink(missing_ok=True)
    temporary_path(path).unlink(missing_ok=True)


def _model_record(lm, name: str) -> dict | None:
    """A provenance record (``window`` or ``sampling``) of a token-scoring model.

    Read through a context wrapper from the model it wraps; ``None`` when the
    model reports none.
    """
    record = getattr(getattr(lm, "model", lm), name, None)
    return None if record is None else record()


def _outcome(
    dataset_name: str,
    method: str,
    paths: _OutputPaths,
    published: Path,
    failed: list[tuple[str, str]],
) -> ComputeOutcome:
    if failed:
        _write_failure_log(paths.fail_log, failed)
    elif paths.fail_log.exists():
        paths.fail_log.unlink()
    status: Literal["completed", "quarantined"] = (
        "completed" if published == paths.pickle_path else "quarantined"
    )
    return ComputeOutcome(
        dataset_name,
        method,
        status,
        published,
        None if status == "completed" else "Incomplete or invalid predictor output",
    )


def _derive_word_cell(
    dataset_name: str,
    method: str,
    context: ComputeContext,
    stimuli: list,
    word_spacing: bool,
    out_dir: Path,
    shard_suffix: str,
    *,
    force: bool,
    checkpoint_every: int,
) -> ComputeOutcome:
    """Compute the fine-grained table of a token method and aggregate it to words.

    A corpus with word spacing takes the method's own token table. A corpus
    without it takes the ``char_beam`` character table, since a subword token
    there can cover parts of two words while a character never does; the
    applicability contract admits only surprisal there.
    """
    if word_spacing:
        source_method, source_unit = method, TOKEN_UNIT
    else:
        from replica.feature.core.annotation.characters import attach_characters

        source_method, source_unit = "char_beam", "characters"
        attach_characters(stimuli)
        log.info(
            "[route] %s has no word spacing; word surprisal derives from "
            "char_beam K=%d characters",
            dataset_name,
            context.beam_width,
        )
    fine = _compute_cell(
        dataset_name,
        source_method,
        replace(context, unit=source_unit),
        stimuli,
        out_dir,
        shard_suffix,
        force=force,
        checkpoint_every=checkpoint_every,
    )
    if not fine.successful or fine.path is None:
        return ComputeOutcome(
            dataset_name, source_method, fine.status, fine.path, fine.error
        )
    return _aggregate_cell(
        dataset_name,
        source_method,
        context,
        stimuli,
        fine.path,
        source_unit,
        out_dir,
        shard_suffix,
        force=force,
    )


def _aggregate_cell(
    dataset_name: str,
    method: str,
    context: ComputeContext,
    stimuli: list,
    source: Path,
    source_unit: str,
    out_dir: Path,
    shard_suffix: str,
    *,
    force: bool,
) -> ComputeOutcome:
    """Publish the word cell aggregated from a verified fine-grained cell."""
    tag = cache_stem(method, context, dataset_name)
    paths = _output_paths(out_dir, tag, shard_suffix)
    rule = AGGREGATION_RULES[method]
    log.info(
        "[agg ] %s -> %s (%s over %s)",
        source.name,
        paths.pickle_path.name,
        rule,
        source_unit,
    )
    lock_fp = _acquire_lock(paths.lock_path)
    if lock_fp is None:
        return ComputeOutcome(
            dataset_name, method, "busy", error="Cell is locked by another process"
        )
    try:
        source_identity = validate_artifact(source)
        if source_identity is None:
            raise ValueError(f"{source} has no verified manifest")
        keys = token_keys(extract_keys(method, context))
        plan, finisher = derivation_plan(method, keys)
        units = value_units(derived_keys(method, keys))
        identity = artifact_identity(
            method, context, stimuli, source_identity["model"], units
        )
        if _should_skip_existing(
            paths.pickle_path, force=force, identity=identity, source=source
        ):
            return ComputeOutcome(dataset_name, method, "cached", paths.pickle_path)
        with open(source, "rb") as stream:
            table = pickle.load(stream)
        outputs: dict[str, list] = {}
        failed: list[tuple[str, str]] = []
        for stimulus in stimuli:
            try:
                outputs[stimulus.stimulus_id] = derive_records(
                    stimulus,
                    table[stimulus.stimulus_id],
                    unit=source_unit,
                    layer="words",
                    plan=plan,
                    finisher=finisher,
                )
            except (KeyError, ValueError) as e:
                failed.append((stimulus.stimulus_id, f"{type(e).__name__}: {e}"))
                log.warning("  [fail] stim=%s: %s", stimulus.stimulus_id, e)
        environment = environment_record()
        derivation = {
            "source": source.name,
            "source_sha256": file_sha256(source),
            "unit": source_unit,
            "rule": rule,
            "keys": plan,
            "finisher": finisher,
            "code_commit": environment["code_commit"],
        }
        published = _write_and_audit_outputs(
            paths.pickle_path,
            paths.partial_path,
            outputs,
            failed=failed,
            expected=len(stimuli),
            identity=identity,
            environment=environment,
            derivation=derivation,
        )
    finally:
        _release_lock(lock_fp, paths.lock_path)
    return _outcome(dataset_name, method, paths, published, failed)


def _load_dataset(dataset_name: str) -> _LoadedDataset:
    log.info("[load] %s ...", dataset_name)
    spec = REGISTRY[dataset_name]
    ds = spec.factory()
    stimuli = ds.stimuli()
    language = getattr(ds.info, "language", None)
    log.info(
        "[load] %s: %d stimuli (language=%s)", dataset_name, len(stimuli), language
    )
    return _LoadedDataset(
        stimuli=stimuli,
        n_total=len(stimuli),
        language=language,
        word_spacing=bool(getattr(ds.info, "word_spacing", True)),
    )


def _apply_stimulus_slice(
    stimuli: list, n_total: int, stim_slice: tuple[int, int] | None
) -> tuple[list, str]:
    if stim_slice is None:
        return stimuli, ""
    s_start, s_end = stim_slice
    s_start = max(0, min(s_start, n_total))
    s_end = max(s_start, min(s_end, n_total))
    sliced = stimuli[s_start:s_end]
    shard_suffix = f".shard{s_start:05d}_{s_end:05d}"
    log.info(
        "[shard] processing stim slice [%d:%d] of %d (n=%d)",
        s_start,
        s_end,
        n_total,
        len(sliced),
    )
    return sliced, shard_suffix


def _output_paths(out_dir: Path, tag: str, shard_suffix: str) -> _OutputPaths:
    stem = f"{tag}{shard_suffix}"
    return _OutputPaths(
        pickle_path=out_dir / f"{stem}.pkl",
        partial_path=out_dir / f"{stem}.pkl.partial",
        primed_path=out_dir / f"{stem}.pkl.primed",
        lock_path=out_dir / f"{stem}.pkl.lock",
        fail_log=out_dir / f"{stem}.failed.log",
    )


def _model_device(lm) -> str | None:
    """Device string of a loaded scoring model, or ``None`` when it has none yet."""
    model = getattr(lm, "_model", None) or getattr(lm, "model", None)
    device = getattr(model, "device", None)
    if device is None:
        device = getattr(lm, "device", None)
    return None if device is None else str(device)


def _should_skip_existing(
    pickle_path: Path, *, force: bool, identity: dict, source: Path | None = None
) -> bool:
    """Whether a published cell at *pickle_path* answers the current request.

    Args:
        pickle_path: The cell's canonical payload.
        force: Recompute regardless of what is there.
        identity: Identity the current run requires.
        source: For a derived cell, the fine-grained table it must derive
            from; its digest has to be the one the cell's sidecar recorded.
    """
    if force or not pickle_path.exists():
        return False
    if (
        matching_manifest(pickle_path, identity, status="completed")
        and (source is None or _derived_from(pickle_path, source))
        and _valid_cached_pickle(pickle_path)
    ):
        log.info("[skip] %s already exists (use --force to recompute)", pickle_path)
        return True
    log.info("[stale] %s has no matching verified manifest; recomputing.", pickle_path)
    return False


def _derived_from(pickle_path: Path, source: Path) -> bool:
    """Whether the sidecar of *pickle_path* names the current bytes of *source*."""
    try:
        manifest = json.loads(manifest_path(pickle_path).read_text())
        return manifest["derivation"]["source_sha256"] == file_sha256(source)
    except (OSError, ValueError, KeyError, TypeError):
        return False


def _accepted_issues(pickle_path: Path) -> set[str] | None:
    """Hard issue buckets the sidecar of *pickle_path* records as accepted, else None."""
    try:
        audit = json.loads(manifest_path(pickle_path).read_text()).get("audit") or {}
    except (OSError, ValueError, TypeError):
        return None
    if not audit.get("accepted"):
        return None
    return set(audit.get("hard_issues", ()))


def _valid_cached_pickle(pickle_path: Path) -> bool:
    try:
        if pickle_path.stat().st_size <= 0:
            return False
        with open(pickle_path, "rb") as f:
            cached = pickle.load(f)
        if not isinstance(cached, dict) or not cached:
            return False
        from replica.compute.validation import predictor_pickle_issues

        report = predictor_pickle_issues(pickle_path)
        if not report.has_hard_issues:
            return True
        # A cell published by ``replica predictors accept`` keeps the issues
        # its investigation covered; a payload with a new one is recomputed.
        accepted = _accepted_issues(pickle_path)
        return (
            accepted is not None
            and set(report.record(failed=0, missing=0)["hard_issues"]) <= accepted
        )
    except Exception as e:
        # Corrupt bytes can surface as almost any exception from the unpickler.
        log.warning(
            "[skip-check] %s unreadable (%s); recomputing.",
            pickle_path,
            type(e).__name__,
        )
        return False


def _acquire_lock(lock_path: Path):
    import fcntl  # Unix-only; deferred so the module imports on Windows

    lock_fp = open(lock_path, "a")
    try:
        fcntl.flock(lock_fp.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        log.info("[lock] %s held by another process; bailing.", lock_path.name)
        lock_fp.close()
        return None
    return lock_fp


def _release_lock(lock_fp, _lock_path: Path) -> None:
    import fcntl  # Unix-only; deferred so the module imports on Windows

    try:
        fcntl.flock(lock_fp.fileno(), fcntl.LOCK_UN)
    finally:
        lock_fp.close()
        # Never unlink: queued or delayed openers must keep the same inode.


def _load_partial_outputs(
    partial_path: Path, *, force: bool, identity: dict
) -> dict[str, list]:
    outputs: dict[str, list] = {}
    if force or not matching_manifest(partial_path, identity, status="partial"):
        return outputs
    try:
        with open(partial_path, "rb") as f:
            cached = pickle.load(f)
        if isinstance(cached, dict) and _valid_cached_pickle(partial_path):
            outputs = cached
            log.info("[resume] loaded %d stim(s) from %s", len(outputs), partial_path)
    except Exception as e:
        # Corrupt bytes can surface as almost any exception from the unpickler.
        log.warning(
            "[resume] %s unreadable (%s); starting fresh.",
            partial_path,
            type(e).__name__,
        )
    return outputs


def _compute_stimuli(
    stimuli: list,
    outputs: dict[str, list],
    lm,
    extract: list[str],
    *,
    partial_path: Path,
    checkpoint_every: int,
    identity: dict,
    environment: dict | None = None,
    target_layer: str | None = None,
    coverage: str | None = None,
) -> list[tuple[str, str]]:
    failed: list[tuple[str, str]] = []
    t0 = time.time()
    progress_every = max(1, len(stimuli) // 10)
    new_count = 0
    skipped = 0
    for i, stim in enumerate(stimuli, 1):
        if stim.stimulus_id in outputs:
            skipped += 1
            continue
        _compute_one_stimulus(
            stim, outputs, lm, extract, failed, target_layer, coverage
        )
        new_count += 1
        if new_count % _GC_EVERY == 0:
            gc.collect()
        if new_count % checkpoint_every == 0:
            atomic_pickle(outputs, partial_path)
            write_manifest(
                partial_path, identity, status="partial", environment=environment
            )
        if i % progress_every == 0 or i == len(stimuli):
            _log_progress(i, len(stimuli), t0, outputs, failed, skipped)
    return failed


def _compute_one_stimulus(
    stim,
    outputs: dict[str, list],
    lm,
    extract: list[str],
    failed: list[tuple[str, str]],
    target_layer: str | None = None,
    coverage: str | None = None,
) -> None:
    try:
        if target_layer == TOKEN_UNIT:
            result = lm.score_tokens(stim, extract=extract)
            if not isinstance(result, list) or any(
                not isinstance(item, FeatureModelOutput) for item in result
            ):
                raise ValueError(f"Expected token records for {stim.stimulus_id}")
            verify_token_rows(stim, result, coverage)
        else:
            result = (
                lm(stim, extract=extract, target_layer=target_layer)
                if target_layer
                else lm(stim, extract=extract)
            )
            expected = len(stim.layers[target_layer or "words"])
            if (
                not isinstance(result, list)
                or len(result) != expected
                or any(not isinstance(item, FeatureModelOutput) for item in result)
            ):
                raise ValueError(
                    f"Expected {expected} FeatureModelOutput records for "
                    f"{stim.stimulus_id}"
                )
        outputs[stim.stimulus_id] = result
    except Exception as e:
        failed.append((stim.stimulus_id, f"{type(e).__name__}: {e}"))
        log.warning("  [fail] stim=%s: %s", stim.stimulus_id, e)


def _log_progress(
    i: int,
    n_stimuli: int,
    t0: float,
    outputs: dict[str, list],
    failed: list[tuple[str, str]],
    skipped: int,
) -> None:
    log.info(
        "  %d / %d (%.1f s elapsed, %d ok, %d failed, %d resumed)",
        i,
        n_stimuli,
        time.time() - t0,
        len(outputs),
        len(failed),
        skipped,
    )


def _write_and_audit_outputs(
    pickle_path: Path,
    partial_path: Path,
    outputs: dict[str, list],
    *,
    failed: list[tuple[str, str]] | None = None,
    expected: int | None = None,
    identity: dict,
    environment: dict | None = None,
    window: dict | None = None,
    sampling: dict | None = None,
    derivation: dict | None = None,
    multiword: dict | None = None,
) -> Path:
    from replica.compute.validation import predictor_pickle_issues

    candidate = pickle_path.with_suffix(".candidate.pkl")
    atomic_pickle(outputs, candidate)
    report = predictor_pickle_issues(candidate)
    # A stimulus that raised leaves no entry, so a cell short of its requested
    # stimuli is missing whole documents. The per-stimulus handler catches
    # everything, CUDA out-of-memory included, and the audit reads only the
    # rows that are present, so neither notices. Quarantine instead of
    # publishing a cell that looks complete.
    missing = max(0, (expected or 0) - len(outputs))
    incomplete = bool(failed) or missing or not outputs
    # Recorded for a cell the audit flagged or a run short of its stimuli, so
    # the investigation of a quarantined cell starts from the manifest.
    audit = (
        report.record(failed=len(failed or ()), missing=missing)
        if report.issues or incomplete
        else None
    )
    suspect_path = pickle_path.with_suffix(".suspect.pkl")
    if report.has_hard_issues or incomplete:
        candidate.replace(suspect_path)
        write_manifest(
            suspect_path,
            identity,
            status="quarantined",
            environment=environment,
            window=window,
            sampling=sampling,
            derivation=derivation,
            multiword=multiword,
            audit=audit,
        )
        reason = (
            f"{len(failed or ())} stimuli failed, {missing} absent"
            if incomplete
            else "hard issues"
        )
        log.warning(
            "[suspect] %s, moved to %s; %s:",
            pickle_path.name,
            suspect_path.name,
            reason,
        )
    else:
        if report.soft_only:
            log.warning("[soft-issues] %s: advisory issues", pickle_path.name)
        candidate.replace(pickle_path)
        write_manifest(
            pickle_path,
            identity,
            status="completed",
            environment=environment,
            window=window,
            sampling=sampling,
            derivation=derivation,
            multiword=multiword,
            audit=audit,
        )
        # The published cell supersedes the quarantined payload of an earlier
        # attempt, which a shard merge over the directory would otherwise see.
        suspect_path.unlink(missing_ok=True)
        manifest_path(suspect_path).unlink(missing_ok=True)
    _log_audit_issues(report)
    if report.has_hard_issues or incomplete:
        # Keep valid work resumable; a hard-invalid payload is never resumed.
        if not report.has_hard_issues and outputs:
            atomic_pickle(outputs, partial_path)
            write_manifest(
                partial_path, identity, status="partial", environment=environment
            )
        return suspect_path
    partial_path.unlink(missing_ok=True)
    manifest_path(partial_path).unlink(missing_ok=True)
    return pickle_path


def _log_audit_issues(report) -> None:
    if report.is_clean:
        return
    for bucket, examples in sorted(report.issues.items()):
        log.warning("  %s:", bucket)
        for example in examples:
            log.warning("    ↳ %s", example)


def _write_failure_log(fail_log: Path, failed: list[tuple[str, str]]) -> None:
    with open(fail_log, "w") as f:
        for sid, msg in failed:
            f.write(f"{sid}\t{msg}\n")
    log.info("[done] %d failures listed in %s", len(failed), fail_log)
