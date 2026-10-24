"""Shared base class for the MC entropy estimators.

The window-boundary and Rényi-aggregation helpers live in :mod:`.utils`; the
public re-export surface (incl. those helpers) lives in
:mod:`replica.feature.entropy.estimators`.
"""

import hashlib
import re
from pathlib import Path

import numpy as np

from replica.feature.backends.generation import (
    usable_position_limit,
    validate_context_window,
)
from replica.feature.backends.start_state import start_ids as resolve_start_ids
from replica.feature.backends.start_state import validate_start_ids
from replica.feature.core.annotation.alignment import aligned_token_ids
from replica.feature.core.annotation.characters import CHARACTER_LAYER

from ...core import Stimulus
from ..backends.device import resolve_device
from ..backends.hub import immutable_revision, resolve_model_bundle
from ..core.protocols import FeatureModelOutput


class _BaseEntropyEstimatorLM:
    """Shared utilities for the MC entropy estimators.

    Each subclass implements ``_estimate_with_samples(prefix_token_ids)
    -> (value, samples)`` for a prefix of token ids (the context up to but
    not including the word being predicted); ``_estimate`` is the
    scalar-only wrapper over it.
    """

    vocab = None  # not a finite-distribution model

    # Sample as upstream ``mc_word_entropy.py`` does: one advancing stream and
    # ``torch.multinomial``. Used by the bit-identity anchor.
    release_rng_stream = False

    def __init__(
        self,
        model_id: str,
        *,
        add_bos: bool,
        seed: int | None,
        model_identity: dict | None = None,
        start_ids: tuple[int, ...] | None = None,
        context_window: int | None = None,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_id = model_id
        self.add_bos = add_bos
        self.context_window = validate_context_window(context_window)
        self._requested_start_ids = validate_start_ids(start_ids, add_bos=add_bos)
        self._seed = seed
        load_path, self._model_identity = self._resolve_identity(
            model_id, model_identity
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            load_path,
            low_cpu_mem_usage=True,
            use_safetensors=any(
                name.endswith(".safetensors") for name in self._model_identity["files"]
            ),
            local_files_only=True,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(
            load_path, local_files_only=True
        )
        self._model_max_ctx_size()
        self._resolved_start_ids()
        self._validate_settings_identity(model_identity)
        _, loaded_identity = resolve_model_bundle(load_path)
        if any(
            loaded_identity[key] != self._model_identity[key]
            for key in ("files", "sha256")
        ):
            raise ValueError("Model bundle changed during model/tokenizer loading")
        # Use the GPU when available; on CPU the forward pass and KV cache
        # are far slower and OOM on long stimuli.
        self._model = self._model.to(resolve_device())
        self._model.eval()
        self._device = next(self._model.parameters()).device
        # Part of the realization: batched matmul is not associative, so the
        # architecture moves a logit enough to flip a sampled token.
        self._device_name = (
            torch.cuda.get_device_name(self._device)
            if self._device.type == "cuda"
            else "cpu"
        )
        self._torch_version = torch.__version__
        # One generator per LM, re-seeded per stimulus. None uses the global RNG.
        if seed is None:
            self._rng = None
        else:
            self._rng = torch.Generator(device=self._device)
            self._rng.manual_seed(int(seed))

    def _validate_settings_identity(self, supplied: dict | None) -> None:
        """Check recorded scoring settings against the effective configuration."""
        if supplied is None:
            return
        if "start" in supplied:
            record = supplied["start"]
            if not isinstance(record, dict) or not isinstance(
                record.get("ids"), (list, tuple)
            ):
                raise ValueError("The model identity requires a start.ids sequence")
            ids = validate_start_ids(record["ids"]) if record["ids"] else ()
            if ids != tuple(self._resolved_start_ids()):
                raise ValueError("Resolved start_ids disagree with the model identity")
        if "context_window" in supplied:
            window = validate_context_window(supplied["context_window"])
            if window is None or window != self._model_max_ctx_size():
                raise ValueError(
                    "Resolved context_window disagrees with the model identity"
                )

    @staticmethod
    def _resolve_identity(model_id: str, supplied: dict | None) -> tuple[str, dict]:
        """Verify identity assertions against the files that will be loaded."""
        revision = None
        source = model_id
        local = Path(model_id).is_dir()
        if supplied is not None:
            revision = immutable_revision(supplied.get("revision"))
            if supplied.get("revision") is not None and revision is None:
                raise ValueError("model_identity requires a full Hub revision")
            sha256 = supplied.get("sha256")
            if sha256 is not None and not re.fullmatch(r"[0-9a-f]{64}", sha256):
                raise ValueError("model_identity requires a full bundle sha256")
            if revision is None and sha256 is None:
                raise ValueError("model_identity requires a revision or bundle sha256")
            if local and revision is not None:
                source = supplied.get("source", "")
                if not source or Path(source).is_dir() or Path(source).is_absolute():
                    raise ValueError(
                        "A local revision assertion requires a verified Hub source"
                    )
            if not local and sha256 is not None and revision is None:
                raise ValueError(
                    "A remote hash assertion requires a pinned Hub revision"
                )
        load_path, identity = resolve_model_bundle(source, revision)
        if revision is not None and identity["revision"] != revision:
            raise ValueError("Hub revision assertion cannot be verified offline")
        if local and source != model_id:
            _, local_identity = resolve_model_bundle(model_id)
            if local_identity["sha256"] != identity["sha256"]:
                raise ValueError("Local files differ from the pinned Hub model bundle")
            load_path = model_id
        if supplied is not None:
            for key in ("sha256", "files"):
                if key in supplied and supplied[key] != identity[key]:
                    raise ValueError(
                        f"Loaded model bundle differs from model_identity {key}"
                    )
        return load_path, identity

    def _seed_for_stimulus(self, stimulus: Stimulus, target_layer: str | None) -> None:
        """Seed from the seed and the stimulus content, so draws do not depend
        on what the run computed before this stimulus.

        Keyed on text and the spans of ``target_layer``, not ``stimulus_id``:
        the predictor is a function of its input. A per-token estimate passes
        no layer and is keyed on the text alone. ``hashlib`` because ``hash``
        is salted per process.
        """
        if self._rng is None or self.release_rng_stream:
            return
        layer = stimulus.layers.get(target_layer) if target_layer is not None else None
        spans = (
            ";".join(f"{s.start}-{s.end}" for s in layer.spans)
            if layer is not None
            else ""
        )
        digest = hashlib.sha256(
            "\x00".join(
                (
                    str(int(self._seed)),
                    str(self._model_identity["sha256"]),
                    stimulus.text,
                    spans,
                )
            ).encode()
        ).digest()
        # torch seeds are 64-bit; take the low 63 bits to stay non-negative.
        self._stimulus_seed = int.from_bytes(digest[:8], "big") >> 1
        self._rng.manual_seed(self._stimulus_seed)

    def _uniform_block(self, key: object, offset: int, k: int, steps: int):
        """Uniforms for samples ``[offset, offset + k)``, one column per step.

        Indexed by the sample's own position, so the batch size and shard
        layout do not select them. Drawn on CPU so the plan is the same on
        every GPU.
        """
        import torch

        seed_src = getattr(self, "_stimulus_seed", None)
        if seed_src is None:
            material = str(self._model_identity["sha256"])
            digest = hashlib.sha256(
                f"{int(self._seed or 0)}:{material}".encode()
            ).digest()
            seed_src = int.from_bytes(digest[:8], "big") >> 1
        digest = hashlib.sha256(f"{seed_src}:{key}".encode()).digest()
        gen = torch.Generator()
        gen.manual_seed(int.from_bytes(digest[:8], "big") >> 1)
        block = torch.rand(
            int(self.n_samples), int(steps), generator=gen, dtype=torch.float64
        )
        return block[offset : offset + k].to(self._device)

    @staticmethod
    def _inverse_cdf(probs, u):
        """First index whose cumulative mass passes ``u``, row-wise.

        Distributionally equal to ``torch.multinomial(probs, 1)``, but a pure
        function of ``(probs, u)`` rather than of the RNG stream position.
        """
        import torch

        cdf = torch.cumsum(probs, dim=-1)
        cdf = cdf / cdf[..., -1:].clamp_min(1e-300)
        idx = torch.searchsorted(cdf.contiguous(), u.unsqueeze(-1).contiguous())
        return idx.clamp_max(probs.shape[-1] - 1)

    @property
    def _provenance(self) -> dict:
        """The settings that decide which draws were taken.

        The seed and device move the sample, and so does the batch size under
        the advancing-stream sampler; none of them reaches ``name``. Computed
        on access from the current settings.
        """
        md = {
            "model_name": self.name,
            "start_ids": self._resolved_start_ids(),
            "context_window": self._model_max_ctx_size(),
            "seed": getattr(self, "_seed", None),
            "model_identity": getattr(self, "_model_identity", None),
            "seed_scheme": (
                "advancing-stream" if self.release_rng_stream else "model-bundle-v1"
            ),
            "chunk_size": getattr(self, "chunk_size", None),
            "device": getattr(self, "_device_name", None),
            "torch": getattr(self, "_torch_version", None),
            # The two paths give different values at the same seed.
            "sampler": (
                "stream" if getattr(self, "release_rng_stream", False) else "position"
            ),
        }
        for attr in ("n_samples", "max_tokens", "max_iter", "max_words", "alpha"):
            if hasattr(self, attr):
                md[attr] = getattr(self, attr)
        return md

    def _model_max_ctx_size(self) -> int:
        """Resolve and cache the effective token budget."""
        if not hasattr(self, "_context_limit"):
            cfg = getattr(self._model, "config", None)
            name = (
                getattr(self, "model_id", "")
                or getattr(cfg, "_name_or_path", "")
                or getattr(self._model, "name_or_path", "")
            )
            self._context_limit = usable_position_limit(
                name, cfg, context_window=getattr(self, "context_window", None)
            )
        return self._context_limit

    def _resolved_start_ids(self) -> list[int]:
        """Resolve and cache the prefix used for alignment and sampling."""
        if not hasattr(self, "_start_prefix_ids"):
            self._start_prefix_ids = tuple(
                resolve_start_ids(
                    getattr(self, "_tokenizer", None),
                    model_name=getattr(self, "model_id", None),
                    override=getattr(self, "_requested_start_ids", None),
                    add_bos=self.add_bos,
                )
            )
        return list(self._start_prefix_ids)

    def _word_start_token_id(self) -> int | None:
        """Start token recognized by word samplers, even when no prefix is added."""
        if not hasattr(self, "_word_start_id"):
            ids = (
                self._resolved_start_ids()
                if self.add_bos
                else resolve_start_ids(
                    self._tokenizer,
                    model_name=getattr(self, "model_id", None),
                    override=getattr(self, "_requested_start_ids", None),
                )
            )
            self._word_start_id = ids[0] if ids else None
        return self._word_start_id

    def _prepare_word_sampling(
        self,
        stimulus: Stimulus,
        target_layer: str,
    ) -> tuple[list[int], int, list[tuple[int, int]], bool]:
        """Validate, align, and locate the prefix boundary for every unit.

        The one-pass estimators differ in how they sample at those boundaries,
        but their stimulus validation, deterministic seeding, and alignment
        contract are identical.
        """
        if target_layer not in stimulus.layers:
            raise ValueError(
                f"{type(self).__name__} expects a {target_layer!r} layer; "
                f"available: {list(stimulus.layers)}"
            )
        self._check_stimulus_compat(stimulus, target_layer)
        self._seed_for_stimulus(stimulus, target_layer)

        token_ids, alignment, bos_offset, n_units = aligned_token_ids(
            stimulus,
            self._tokenizer,
            target_layer,
            self.add_bos,
            start_ids=self._resolved_start_ids() if self.add_bos else None,
        )
        sample_positions = []
        for unit_index in range(n_units):
            positions = alignment.mapping[unit_index]
            if not positions:
                continue
            first_pos = bos_offset + positions[0]
            if first_pos > 0:
                sample_positions.append((unit_index, first_pos - 1))
        return token_ids, n_units, sample_positions, len(token_ids) == bos_offset

    @staticmethod
    def _unit_word_counts(stimulus: Stimulus, target_layer: str) -> dict[int, int]:
        """Whitespace-word count of every unit holding more than one."""
        return {
            index: count
            for index, text in enumerate(stimulus.unit_strings(target_layer))
            if (count := len(text.split())) > 1
        }

    def _outputs_with_samples(
        self,
        per_unit_entropy: np.ndarray,
        per_unit_samples: list[np.ndarray | None],
    ) -> list[FeatureModelOutput]:
        """Build the common scalar-entropy output schema."""
        outputs = []
        for entropy, samples in zip(per_unit_entropy, per_unit_samples, strict=True):
            values = self._wrap(entropy)
            values["log_p_samples_nats"] = (
                np.empty(0, dtype=np.float64)
                if samples is None
                else np.asarray(samples, dtype=np.float64)
            )
            outputs.append(
                FeatureModelOutput(
                    values=values,
                    metadata=dict(self._provenance),
                )
            )
        return outputs

    def _outputs_from_mapping(
        self,
        n_units: int,
        values_by_unit: dict[int, dict],
    ) -> list[FeatureModelOutput]:
        """Build outputs for estimators whose value schema is already complete."""
        return [
            FeatureModelOutput(
                values=values_by_unit.get(unit_index, self._empty_values()),
                metadata=dict(self._provenance),
            )
            for unit_index in range(n_units)
        ]

    def _collect_sample_chunks(self, dispatch) -> dict[int, np.ndarray]:
        """Run a one-pass sampler in bounded chunks and concatenate by unit."""
        chunks_by_unit: dict[int, list[np.ndarray]] = {}
        chunk_size = self.chunk_size or self.n_samples
        remaining = self.n_samples
        while remaining > 0:
            size = min(chunk_size, remaining)
            offset = self.n_samples - remaining
            for unit_index, samples in dispatch(size, offset).items():
                chunks_by_unit.setdefault(unit_index, []).append(samples)
            remaining -= size
        return {
            unit_index: np.concatenate(chunks)
            for unit_index, chunks in chunks_by_unit.items()
        }

    def _advance_master_cache(
        self,
        token_ids: list[int],
        current_position: int,
        target_position: int,
        batch_size: int,
        master_kv,
        last_logits,
    ):
        """Advance a shared transformer cache through observed input tokens."""
        import torch
        from transformers import DynamicCache

        while current_position <= target_position:
            next_id = torch.full(
                (batch_size, 1),
                token_ids[current_position],
                device=self._device,
                dtype=torch.long,
            )
            fresh = DynamicCache.from_legacy_cache(master_kv)
            with torch.no_grad():
                output = self._model(
                    next_id,
                    past_key_values=fresh,
                    use_cache=True,
                )
            master_kv = output.past_key_values.to_legacy_cache()
            last_logits = output.logits[:, -1, :].double()
            current_position += 1
        return current_position, master_kv, last_logits

    def __call__(
        self,
        stimulus: Stimulus,
        extract: list[str] | None = None,
        target_layer: str = "words",
    ) -> list[FeatureModelOutput]:

        # Validate the unit layer up-front. The estimators are word-level
        # predictors: each unit reads the distribution before its first
        # token, which a per-character layer (e.g. MECO_char) leaves
        # undefined for every character but one per token, so
        # _check_stimulus_compat refuses it.
        if target_layer not in stimulus.layers:
            raise ValueError(
                f"{type(self).__name__}: stimulus has no {target_layer!r} "
                f"layer; available: {list(stimulus.layers)}"
            )
        self._check_stimulus_compat(stimulus, target_layer)

        # Token-align the stimulus the same way HuggingFaceLM does, so
        # the word boundaries match the units.
        token_ids, alignment, bos_offset, n_units = aligned_token_ids(
            stimulus,
            self._tokenizer,
            target_layer,
            self.add_bos,
            start_ids=self._resolved_start_ids() if self.add_bos else None,
        )

        if len(token_ids) == bos_offset:
            return [
                FeatureModelOutput(
                    values=self._empty_values(),
                    metadata=dict(self._provenance),
                )
                for _ in range(n_units)
            ]

        # alignment.mapping[i] is the list of token positions belonging
        # to unit i (in the un-BOS-prefixed token stream). The prefix
        # before unit i ends at the token before unit i's first token.
        # Without BOS, unit 0's prefix is empty (no left context) so
        # entropy is undefined for that position.
        per_unit_entropy = np.full(n_units, np.nan)
        # Per-unit per-sample joint log-probabilities (in nats). Stored
        # alongside the aggregate scalar so an aggregator change can be
        # applied by re-aggregating the pickle without a fresh MC run.
        per_unit_samples: list[np.ndarray | None] = [None] * n_units
        for i in range(n_units):
            tok_positions = alignment.mapping[i]
            if not tok_positions:
                continue
            first_pos = tok_positions[0]
            prefix_ids = token_ids[: bos_offset + first_pos]
            if not prefix_ids:
                continue
            value, samples = self._estimate_with_samples(prefix_ids)
            per_unit_entropy[i] = value
            per_unit_samples[i] = samples

        return self._outputs_with_samples(per_unit_entropy, per_unit_samples)

    # Subclasses customise these.
    def _empty_values(self) -> dict:
        raise NotImplementedError

    def _wrap(self, value: float) -> dict:
        raise NotImplementedError

    def _estimate(self, prefix_ids: list[int]) -> float:
        """Scalar-only wrapper around :meth:`_estimate_with_samples`; unit
        tests call it directly.
        """
        value, _ = self._estimate_with_samples(prefix_ids)
        return value

    def _estimate_with_samples(
        self,
        prefix_ids: list[int],
    ) -> tuple[float, np.ndarray]:
        """Return ``(entropy_estimate, log_p_samples_nats)``.

        Subclasses must override either this or both
        ``_estimate`` and ``_sample_log_p_chunk`` + the chunked driver
        plus the aggregator.
        """
        raise NotImplementedError

    def _check_stimulus_compat(self, stimulus: Stimulus, target_layer: str) -> None:
        """Refuse a per-character unit layer; subclasses add their own checks.

        ``MCWordEntropyLM`` overrides it to also reject no-whitespace text,
        where its word-boundary stopping rule is meaningless.
        """
        if target_layer == CHARACTER_LAYER:
            raise ValueError(
                f"{type(self).__name__} requires a word-level target_layer; got "
                f"{CHARACTER_LAYER!r}. A unit reads the distribution before its "
                "first token, and most characters start no token. CharBeamLM "
                "scores per-character units (surprisal)."
            )
