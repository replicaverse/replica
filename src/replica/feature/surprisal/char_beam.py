"""Character-level surprisal via top-K tokenisation marginalisation.

Reference: Giulianelli et al. (2024b), "On the Proper Treatment of
Tokenization in Psycholinguistics".

Thin REPLICA-side ``FeatureModel`` adapter on top of ``genlm-bytes``, the
reference implementation maintained by the GenLM team
(https://github.com/genlm/genlm-bytes). The byte-level beam search is
delegated to that library; the resulting per-byte log-probs are aggregated
to whatever Span layer the stimulus declares.
"""

import asyncio
import concurrent.futures

import numpy as np

from ...core import Stimulus
from ..backends.device import (
    loading_options,
    resolve_device,
    validate_device_map,
    validate_dtype,
)
from ..backends.generation import usable_position_limit, validate_context_window
from ..backends.hub import hub_id
from ..backends.start_state import start_ids as resolve_start_ids
from ..backends.start_state import validate_start_ids
from ..core.protocols import FeatureModelOutput


def _run_coro(coro):
    """Run *coro* to completion whether or not an event loop is already running.

    ``asyncio.run`` works from a plain script or test, but raises inside a live
    event loop (a Jupyter kernel). In that case the coroutine runs on a
    short-lived worker thread that owns its own loop.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(asyncio.run, coro).result()


class CharBeamLM:
    """Character-level surprisal via top-K tokenisation marginalisation.

    Weights and tokenizer are shared across calls; each call uses a fresh
    beam. Cache memory grows with beam width and text length. Pruning
    discards low-weight candidates.

    Args:
        start_ids: Optional single-token prefix; ``None`` uses the backend default.
        context_window: Optional token limit; ``None`` uses the model default.
            Scoring fails when any evaluated beam prefix exceeds this limit.
        model_id: Any GPT-style byte-level BPE tokenizer (gpt2,
            ai-forever/mGPT, EleutherAI/pythia-*). SentencePiece is not
            supported.
        K: Beam width. The paper uses 5.
        heal: ``genlm-bytes`` token healing at the beam boundary.
            ``False`` keeps the released behavior.
        prune_threshold: Drop beam candidates below this fraction of the
            max probability. ``0.0`` (default) prunes nothing; ``0.05``
            trades a small bias for a lower memory peak.

    Returns one :class:`FeatureModelOutput` per unit. ``log_p_observed``
    sums byte log-probabilities over the unit's UTF-8 bytes.
    """

    vocab = None  # CharBeamLM does not provide a finite vocab distribution

    def __init__(
        self,
        model_id: str = "gpt2",
        K: int = 5,
        heal: bool = False,
        prune_threshold: float = 0.0,
        *,
        start_ids: tuple[int, ...] | None = None,
        context_window: int | None = None,
        dtype: str | None = None,
        device_map: str | None = None,
    ):
        """Load the weights, tokenizer, and byte-vocab trie shared by every call.

        Args:
            model_id: Hub id or local path of the checkpoint.
            K: Beam width.
            heal: Whether to heal token boundaries.
            prune_threshold: Probability mass below which beams are dropped.
            start_ids: Optional single-token prefix.
            context_window: Maximum tokens per forward pass.
            dtype: Parameter precision, ``float32``, ``float16`` or
                ``bfloat16``; ``None`` keeps single precision.
            device_map: ``auto`` shards the weights over every visible device;
                ``None`` places the whole model on one device.
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = f"charbeam_{hub_id(model_id)}_K{K}"
        self.model_id = model_id
        self._requested_start_ids = validate_start_ids(start_ids)
        requested_context = validate_context_window(context_window)
        self.K = K
        self.heal = heal
        self.prune_threshold = prune_threshold

        # AsyncTransformer places its inputs on model.device.
        self._hf_tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.dtype = validate_dtype(dtype)
        self.device_map = validate_device_map(device_map)
        self._hf_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            low_cpu_mem_usage=True,
            **loading_options(self.dtype, self.device_map),
        )
        if self.device_map is None:
            self._hf_model = self._hf_model.to(resolve_device())
        self._hf_model.eval()
        self.context_window = usable_position_limit(
            model_id, self._hf_model.config, context_window=requested_context
        )

        # Cache the tokenizer's byte-vocabulary trie across calls.
        from genlm.backend.tokenization.bytes import get_byte_vocab
        from genlm.bytes.trie import AsyncTokenByteTrie

        self._byte_vocab = get_byte_vocab(self._hf_tokenizer)
        self._async_trie = AsyncTokenByteTrie.from_vocab(self._byte_vocab)

    def __call__(
        self,
        stimulus: Stimulus,
        extract: list[str] | None = None,
        target_layer: str | None = None,
    ) -> list[FeatureModelOutput]:
        """One output per unit, its ``log_p_observed`` summed over the unit's bytes.

        A unit whose span covers a byte the beam could not score is NaN.

        Args:
            stimulus: Text to score, carrying the annotation layers.
            extract: Ignored, since the only value this model can supply is
                ``log_p_observed``.
            target_layer: Name of the annotation layer whose units the outputs
                align to. ``None`` picks ``words`` when the stimulus has that
                layer, else ``characters``.
        """
        if target_layer is None:
            if "words" in stimulus.layers:
                target_layer = "words"
            elif "characters" in stimulus.layers:
                target_layer = "characters"
            else:
                raise ValueError(
                    "CharBeamLM needs a stimulus with a 'words' or "
                    "'characters' layer (or an explicit target_layer=); "
                    f"available: {list(stimulus.layers)}"
                )
        if target_layer not in stimulus.layers:
            raise ValueError(
                f"CharBeamLM: target_layer={target_layer!r} not on stimulus; "
                f"available: {list(stimulus.layers)}"
            )

        text = stimulus.text
        chars_layer = stimulus.layers[target_layer]
        n_chars = len(text)
        text_bytes = text.encode("utf-8")
        n_bytes = len(text_bytes)

        # char_start[i], char_end[i] = UTF-8 byte offsets bracketing
        # text[i]. Used to sum byte log-probs into per-char log-probs.
        char_start = np.zeros(n_chars, dtype=np.int64)
        char_end = np.zeros(n_chars, dtype=np.int64)
        b_off = 0
        for i, ch in enumerate(text):
            char_start[i] = b_off
            b_off += len(ch.encode("utf-8"))
            char_end[i] = b_off

        # A fresh AsyncTransformer wrapper per call keeps the per-stimulus
        # state (TokenTrie KV cache, ByteBeamState) local to the call; only
        # the instance-cached byte-vocab trie persists across stimuli.
        log_p_bytes = _run_coro(self._byte_log_p(text_bytes))
        if log_p_bytes.shape != (n_bytes,):
            raise ValueError(
                f"CharBeamLM: byte beam returned {log_p_bytes.shape} log-probs "
                f"for {n_bytes} bytes of stimulus {stimulus.stimulus_id!r}; "
                "expected one per byte."
            )

        # Sum bytes -> per-char log-probabilities. For a multi-byte
        # UTF-8 char, log P(char | ctx) = log P(byte_0|ctx) + log
        # P(byte_1|ctx,byte_0) + ... by chain rule of probability.
        log_p_text = np.full(n_chars, np.nan)
        for i in range(n_chars):
            vals = log_p_bytes[char_start[i] : char_end[i]]
            if np.isnan(vals).any():
                log_p_text[i] = np.nan
            else:
                log_p_text[i] = float(vals.sum())

        # Aggregate to whatever unit boundaries the requested layer
        # defines (one character per Span for a "characters" layer;
        # multi-character spans for a "words" layer).
        outputs: list[FeatureModelOutput] = []
        for span in chars_layer.spans:
            indices = list(range(span.start, span.end))
            if not indices:
                outputs.append(
                    FeatureModelOutput(
                        values={"log_p_observed": np.nan},
                        metadata={
                            "model_name": self.name,
                            "K": self.K,
                            "start_ids": self._start_prefix(),
                            "context_window": self.context_window,
                        },
                    )
                )
                continue
            vals = log_p_text[indices]
            lp = np.nan if np.isnan(vals).any() else float(vals.sum())
            outputs.append(
                FeatureModelOutput(
                    values={"log_p_observed": lp},
                    metadata={
                        "model_name": self.name,
                        "K": self.K,
                        "start_ids": self._start_prefix(),
                        "context_window": self.context_window,
                    },
                )
            )
        return outputs

    def _start_prefix(self) -> list[int]:
        """The token prefix supplied to the byte model."""
        return resolve_start_ids(
            self._hf_tokenizer,
            self.model_id,
            override=getattr(self, "_requested_start_ids", None),
        )

    async def _byte_log_p(self, text_bytes: bytes) -> np.ndarray:
        """Score each UTF-8 byte under the tokenization beam."""
        from genlm.backend.llm.hf import AsyncTransformer
        from genlm.bytes import BeamParams, ByteBeamState
        from genlm.bytes.byte_lm.lm_state import StatefulTokenizedLM
        from genlm.bytes.byte_lm.trie_state import (
            LazyTrieState,
            TrieMode,
        )

        limit = self.context_window
        pending: set[asyncio.Task] = set()

        class _ContextLimitedTransformer(AsyncTransformer):
            def batch_evaluate_queries(self):
                queries = tuple(self.queries)
                try:
                    super().batch_evaluate_queries()
                except Exception as error:
                    # Timer callbacks must deliver model failures to the callers.
                    for query in queries:
                        if not query.future.done():
                            query.future.set_exception(error)

            async def next_token_logprobs(self, token_ids):
                if len(token_ids) > limit:
                    raise ValueError(
                        f"Character beam requires a prefix of {len(token_ids)} "
                        f"tokens, exceeding context_window={limit}"
                    )
                task = asyncio.current_task()
                if task is not None and task not in pending:
                    pending.add(task)
                    task.add_done_callback(pending.discard)
                return await super().next_token_logprobs(token_ids)

        backend = _ContextLimitedTransformer(self._hf_model, self._hf_tokenizer)
        params = BeamParams(
            K=self.K,
            heal=self.heal,
            prune_threshold=self.prune_threshold,
        )
        # Reuse the cached trie with the same start prefix as token scorers.
        initial_lts = LazyTrieState(
            trie=self._async_trie,
            node=self._async_trie.trie.root,
            lm_state=StatefulTokenizedLM.initial(
                backend,
                initial_context=self._start_prefix(),
            ),
            weight=0.0,
            mode=TrieMode.WITH_EOS,
        )
        try:
            state = ByteBeamState([await initial_lts.materialize()], params)
            out = np.full(len(text_bytes), np.nan)
            for t, byte in enumerate(text_bytes):
                d = (await state.logp_next()).materialize()
                try:
                    out[t] = float(d[byte])
                except (KeyError, TypeError):
                    out[t] = np.nan
                state = await (state.prune() << byte)
            return out
        finally:
            if backend.timer is not None:
                backend.timer.cancel()
                backend.timer = None
            backend.reset_async_queries()
            unfinished = [
                task
                for task in pending
                if task is not asyncio.current_task() and not task.done()
            ]
            for task in unfinished:
                task.cancel()
            if unfinished:
                await asyncio.gather(*unfinished, return_exceptions=True)
            await self._async_trie.cleanup()
