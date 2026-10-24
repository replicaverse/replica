"""ModelBlocks-style word surprisal, the convention used by Clark et al. (2025).

Computes per-sub-token surprisal under document-level context with 50%-overlap
sliding windows, then redistributes the word-boundary log-probability between
adjacent tokens before summing sub-tokens to words. Interior words match
Pimentel & Meister (2024)'s "corrected" word probability by construction
(:class:`WordsProbabilityLM`); the definitions differ at sequence edges
(BOS/EOS handling).

Exposed via ``surprisal_model(model_id, method="modelblocks_surprisal")``; returns
``log_p_observed`` (nats) per word, like the other surprisal backends.

Distinct from the sibling ``feature/modelblocks.py`` (same name, different
layer): that module *reads* ModelBlocks' own precomputed ``.itemmeasures``
predictor tables, whereas this one *computes* surprisal in the ModelBlocks
convention.
"""

import numpy as np

from replica.feature.core.annotation._text import _layer_strings
from replica.feature.core.annotation.alignment import aggregate_whitespace_words
from replica.feature.core.outputs.information import LN2

from ...core import Stimulus
from ..backends.generation import usable_position_limit
from ..core.protocols import FeatureModelOutput


class ModelBlocksSurprisalLM:
    """Word surprisal matching ModelBlocks' ``get_llm_surprisal``/``postprocess_bi``.

    Args:
        model_id: HuggingFace causal-LM id (e.g. ``"gpt2"``).
        device: ``"cpu"`` (default) or ``"cuda"``; an explicit ``"cpu"``
            is upgraded to CUDA when available (unlike
            :class:`HuggingFaceLM`, which honors it).
    """

    vocab = None

    def __init__(self, model_id: str = "gpt2", device: str = "cpu"):
        """Record the model choice; weights load lazily on the first call."""
        self.name = f"modelblocks_surprisal_{model_id}"
        self._model_id = model_id
        self.device = device
        self._model = None
        self._tokenizer = None
        self._space_ids = None  # token ids whose vocab string starts with "Ġ"

    def _ensure(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self._model_id)
        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_id, low_cpu_mem_usage=True
        )
        if self.device == "cpu" and torch.cuda.is_available():
            self.device = "cuda"
        self._model.to(self.device).eval()
        # Word-initial tokens: vocab strings starting with the GPT-2 space byte.
        self._space_ids = torch.tensor(
            [i for t, i in self._tokenizer.vocab.items() if t.startswith("Ġ")],
            device=self.device,
            dtype=torch.long,
        )

    def _subtoken_rows(self, text: str):
        """Per-output-sub-token ``(surprisal_bits, bori, bprob_bits, is_eos)``.

        Mirrors ``get_llm_surprisal.py``: BOS-prefixed, 50%-overlap windows of
        the model's usable position count (:func:`usable_position_limit`,
        which caps mGPT at its trained 512) keeping each window's later half;
        ``bori`` is ``"B"`` for word-initial (Ġ-prefixed) tokens else ``"I"``;
        ``bprob`` is the word-initial log₂-probability mass. A single EOS row
        (its boundary mass for the position past the first window's end) is
        appended after the first window, exactly as upstream does.
        """
        import torch

        tok, model = self._tokenizer, self._model
        ctx = usable_position_limit(self._model_id, model.config)
        bos = model.config.bos_token_id
        ids = ([bos] if bos is not None else []) + tok(text)["input_ids"]

        batches, work, start = [], list(ids), 0
        while len(work) > ctx:
            batches.append((work[:ctx], work[1 : ctx + 1], start))
            work = work[ctx // 2 :]
            start = ctx // 2
        batches.append((work, work[1:], start))

        rows: list[tuple[float, str, float, bool]] = []
        eos_emitted = False
        for inp_ids, out_ids, s_idx in batches:
            with torch.no_grad():
                logits = model(
                    torch.tensor([inp_ids], device=self.device)
                ).logits.squeeze(0)
            probs = torch.softmax(logits, dim=-1)
            out_t = torch.tensor(out_ids, device=self.device)
            obs = (-torch.log2(probs[torch.arange(len(out_ids)), out_t])).tolist()
            bprob = torch.log2(probs[:, self._space_ids].sum(-1)).tolist()
            sub_strs = tok.convert_ids_to_tokens(out_ids)
            for i in range(s_idx, len(out_ids)):
                bori = "B" if sub_strs[i].startswith("Ġ") else "I"
                rows.append((obs[i], bori, bprob[i], False))
            if not eos_emitted:
                eos_b = bprob[len(inp_ids) - 1]
                rows.append((-eos_b, "B", eos_b, True))
                eos_emitted = True
        return rows

    @staticmethod
    def _postprocess_and_aggregate(rows, n_words: int) -> np.ndarray:
        """Apply ModelBlocks' ``postprocess_bi`` redistribution, sum to words (bits).

        When the next token is word-initial (``B``), its word-initial log-prob is
        moved from the current token onto it; words then start at each non-EOS
        ``B`` token. EOS rows take part in the redistribution but carry no word.
        """
        surp = [r[0] for r in rows]
        bori = [r[1] for r in rows]
        bprob = [r[2] for r in rows]
        for i in range(len(rows) - 1):
            if bori[i + 1] == "B":
                surp[i] -= bprob[i + 1]
                surp[i + 1] += bprob[i + 1]
        # A word starts at each "B" token; the document's first token is "I"
        # (no leading space) and opens the first word. EOS rows take part in
        # redistribution but open no word.
        word_surps: list[float] = []
        cur: float | None = None
        for i, row in enumerate(rows):
            if row[3]:  # is_eos
                continue
            if cur is None:
                cur = surp[i]  # first word's opening token
            elif bori[i] == "B":
                word_surps.append(cur)
                cur = surp[i]
            else:
                cur += surp[i]
        if cur is not None:
            word_surps.append(cur)
        out = np.asarray(word_surps, dtype=float)
        if len(out) != n_words:
            raise ValueError(
                f"ModelBlocksSurprisalLM: aggregated {len(out)} words but the "
                f"text has {n_words} whitespace words."
            )
        return out

    def __call__(
        self,
        stimulus: Stimulus,
        extract: list[str] | None = None,
        target_layer: str = "words",
    ) -> list[FeatureModelOutput]:
        """One output per unit, carrying the word's ``log_p_observed`` in nats.

        Args:
            stimulus: Text to score; the model scores its whitespace words, and
                a unit of ``target_layer`` holding several takes the sum of
                their log probabilities.
            extract: Ignored, since the only value this model can supply is
                ``log_p_observed``.
            target_layer: Name of the annotation layer whose units the outputs
                align to, one per unit.
        """
        self._ensure()
        if not _layer_strings(stimulus, target_layer):
            return []
        ws_words = stimulus.text.split()
        rows = self._subtoken_rows(stimulus.text)
        surp_bits = self._postprocess_and_aggregate(rows, len(ws_words))
        log_p_nats = -surp_bits * LN2
        per_word = [
            FeatureModelOutput(
                values={"log_p_observed": float(lp)},
                metadata={"model_name": self.name},
            )
            for lp in log_p_nats
        ]
        # A unit of several words takes the sum of their log probabilities.
        return aggregate_whitespace_words(stimulus, target_layer, per_word, "sum")


__all__ = ["ModelBlocksSurprisalLM"]
