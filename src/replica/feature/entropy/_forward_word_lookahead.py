"""Word-level forward look-ahead entropy (word-boundary MC, K words deep).

From the context up to a fixation, sample N continuations of K whole words with
the word-boundary stopping rule of Clark et al. (2025), recording per sample,
per word-depth d=1..K: the word's joint log-prob (nats) and a stable hash of
its sampled token ids. The saved arrays support shorter look-ahead depths and
offline re-aggregation at their stored precision.

Estimands recovered offline from the saved [N,K] arrays:
  * joint(k)       = -(1/N) Σ_s Σ_{d<=k} logp[s,d] / ln2
  * conditional(k) = joint(k) - joint(k-1) = -(1/N) Σ_s logp[s,k] / ln2
                     = MC estimate of E_path[H(w_{t+k}|w_{<t+k})]
  * marginal(k)    = count plug-in on the depth-k word identities, p̂(w)=n_w/N,
                     H = -Σ_w p̂ log2 p̂, Miller-Madow corrected. Secondary and
                     biased (capped near log2 N, MM-corrected); the depth-k word
                     of each ancestral trajectory is an iid marginal draw, but
                     the single-pass estimator is the biased count plug-in.
Identity: conditional(1) == joint(1). Neither equals ``mc_word_entropy`` at the
same settings: a look-ahead sample runs up to ``max_words * max_iter +
max_words`` token steps in total, so its first word may run past the
``max_iter`` steps at which ``mc_word_entropy`` truncates a word, and the two
draw from separately keyed uniforms.
Word identity uses a rolling hash of the sampled token sequence (fast, on-line);
different token paths to the same surface word count as distinct types, a minor
upward bias on the (already secondary) marginal. Subclasses MCWordEntropyLM
for its subword masks and compat checks, with its own k-word sampler and
one-pass / half-overlap-windowed drivers.
"""

import numpy as np

from replica.feature.backends.hub import hub_id
from replica.feature.core.outputs.information import LN2

from ...core import Stimulus
from ..core.protocols import FeatureModelOutput
from ._mc_word import MCWordEntropyLM
from .utils import _clark_windows

_HASH_PRIME = 1099511628211  # FNV-ish; int64 wrap gives a cheap stable token-seq hash


class ForwardWordLookaheadEntropyLM(MCWordEntropyLM):
    """K-word forward look-ahead via word-boundary MC sampling (see module docstring).

    Args:
        start_ids: Optional single-token prefix; ``None`` uses the backend default.
        context_window: Token budget; ``None`` uses the model default.
        model_identity: Verified Hub source/revision or local bundle ``sha256``.
            Assertions are checked against loaded files; omitted identities resolve
            from the model source.
    """

    @property
    def supports_output_union(self) -> bool:
        """Combining requests preserves the same computed or sampled values."""
        return type(self) is ForwardWordLookaheadEntropyLM

    def __init__(
        self,
        model_id: str = "gpt2",
        *,
        n_samples: int = 512,
        max_words: int = 5,
        max_iter: int = 30,
        add_bos: bool = True,
        seed: int | None = None,
        model_identity: dict | None = None,
        start_ids: tuple[int, ...] | None = None,
        context_window: int | None = None,
        chunk_size: int | None = None,
    ):
        super().__init__(
            model_id,
            n_samples=n_samples,
            max_iter=max_iter,
            alpha=1.0,
            add_bos=add_bos,
            seed=seed,
            model_identity=model_identity,
            start_ids=start_ids,
            context_window=context_window,
            chunk_size=chunk_size,
        )
        if int(max_words) < 1:
            raise ValueError(f"max_words must be >= 1, got {max_words}")
        self.max_words = int(max_words)
        self.name = f"fwd_word_lookahead_{hub_id(model_id)}_N{n_samples}_M{max_iter}_K{max_words}"

    def _empty_values(self) -> dict:
        K = self.max_words
        nan = np.full(K, np.nan)
        return {
            "joint_word_entropy": nan.copy(),
            "conditional_word_entropy": nan.copy(),
            "marginal_word_entropy": nan.copy(),  # MM-corrected count plug-in
            "marginal_word_entropy_plugin": nan.copy(),  # raw (uncorrected)
            "fwd_word_log_p_step_nats": np.empty((0, K), np.float32),
            "fwd_word_id_hash": np.empty((0, K), np.uint64),
        }

    # ---- aggregation: [N,K] per-sample logp + id-hash -> the three profiles ----
    def _aggregate(self, logp: np.ndarray, idh: np.ndarray) -> dict:
        N, K = logp.shape
        cum = np.cumsum(logp, axis=1)
        joint = -cum.mean(axis=0) / LN2
        cond = -logp.mean(axis=0) / LN2  # == joint(k)-joint(k-1)
        marg_plug = np.empty(K)
        marg_mm = np.empty(K)
        for k in range(K):
            _, counts = np.unique(idh[:, k], return_counts=True)
            p = counts / N
            Hplug = float(-(p * np.log2(p)).sum())
            S = len(counts)
            marg_plug[k] = Hplug
            marg_mm[k] = Hplug + (S - 1) / (2 * N * LN2)  # Miller-Madow (bits)
        return {
            "joint_word_entropy": joint,
            "conditional_word_entropy": cond,
            "marginal_word_entropy": marg_mm,
            "marginal_word_entropy_plugin": marg_plug,
            # raw samples for offline re-aggregation (joint/cond/Rényi/marginal);
            # Preserve the full hashes used by the marginal estimator.
            "fwd_word_log_p_step_nats": logp.astype(np.float32),
            "fwd_word_id_hash": idh.astype(np.uint64),
        }

    def __call__(
        self,
        stimulus: Stimulus,
        extract: list[str] | None = None,
        target_layer: str = "words",
    ) -> list[FeatureModelOutput]:
        """Joint, conditional and marginal word look-ahead entropies per unit.

        Args:
            stimulus: Text whose word boundaries are scored.
            extract: Part of the shared FeatureModel signature; every value is
                returned.
            target_layer: Annotation layer whose units are the words scored.
        """
        token_ids, n_units, sample_positions, _ = self._prepare_word_sampling(
            stimulus, target_layer
        )
        per_unit: dict[int, dict] = {}
        if sample_positions:
            unit_words = self._unit_word_counts(stimulus, target_layer)
            ctx_size = self._model_max_ctx_size()
            max_prefix = max(p for _, p in sample_positions)
            # reserve the full _sample_k_words step budget; under-reserving
            # lets sampling run past GPT-2's wpe limit -> position OOB.
            longest = max(unit_words.values(), default=1)
            budget = self._step_budget(longest)
            single = max_prefix + 1 + budget <= ctx_size
            # collect per word: lists of [k,K] logp and idhash chunks
            acc_lp: dict[int, list[np.ndarray]] = {}
            acc_id: dict[int, list[np.ndarray]] = {}
            chunk = self.chunk_size or self.n_samples
            remaining = self.n_samples
            while remaining > 0:
                kk = min(chunk, remaining)
                offset = self.n_samples - remaining
                res = (
                    self._klookahead_one_pass(
                        token_ids, sample_positions, kk, offset, unit_words
                    )
                    if single
                    else self._klookahead_windowed(
                        token_ids, sample_positions, kk, ctx_size, offset, unit_words
                    )
                )
                for wi, (lp, idh) in res.items():
                    acc_lp.setdefault(wi, []).append(lp)
                    acc_id.setdefault(wi, []).append(idh)
                remaining -= kk
            for wi, lp_chunks in acc_lp.items():
                logp = np.concatenate(lp_chunks, axis=0)
                idh = np.concatenate(acc_id[wi], axis=0)
                per_unit[wi] = self._aggregate(logp, idh)

        return self._outputs_from_mapping(n_units, per_unit)

    def _step_budget(self, first_words: int = 1) -> int:
        """Sampling steps the K-word sampler may take when depth 1 holds *first_words* words."""
        words = self.max_words + first_words - 1
        return words * self.max_iter + words

    # ---- one-pass KV-shared master driver (mirrors MCWordEntropyLM) ----
    def _klookahead_one_pass(
        self, token_ids, sample_positions, k, offset=0, unit_words=None
    ):
        import torch

        unit_words = unit_words or {}

        device = self._device
        gen = self._rng
        bos_id = self._word_start_token_id()
        ordered = sorted(sample_positions, key=lambda x: x[1])
        out_map: dict[int, tuple[np.ndarray, np.ndarray]] = {}

        first_pos = ordered[0][1]
        ctx = torch.tensor(
            [token_ids[: first_pos + 1]], device=device, dtype=torch.long
        ).repeat(k, 1)
        with torch.no_grad():
            o = self._model(ctx, use_cache=True)
            master_kv = o.past_key_values.to_legacy_cache()
            last_logits = o.logits[:, -1, :].double()
        out_map[ordered[0][0]] = self._sample_k_words(
            last_logits,
            master_kv,
            k,
            gen,
            token_ids[first_pos],
            bos_id,
            ordered[0][0],
            offset,
            first_words=unit_words.get(ordered[0][0], 1),
        )

        cur_pos = first_pos + 1
        for word_i, target_pos in ordered[1:]:
            cur_pos, master_kv, last_logits = self._advance_master_cache(
                token_ids, cur_pos, target_pos, k, master_kv, last_logits
            )
            out_map[word_i] = self._sample_k_words(
                last_logits,
                master_kv,
                k,
                gen,
                token_ids[target_pos],
                bos_id,
                word_i,
                offset,
                first_words=unit_words.get(word_i, 1),
            )
        return out_map

    def _klookahead_windowed(
        self, token_ids, sample_positions, k, ctx_size, offset=0, unit_words=None
    ):
        longest = max((unit_words or {}).values(), default=1)
        words = self.max_words + longest - 1
        half = ctx_size // 2 - words * self.max_iter
        if half < 1:
            raise ValueError(
                f"ctx_size={ctx_size} too small for {words} words of "
                f"max_iter={self.max_iter} tokens"
            )
        out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for window_tokens, window_positions in _clark_windows(
            token_ids, sample_positions, half
        ):
            out.update(
                self._klookahead_one_pass(
                    window_tokens, window_positions, k, offset, unit_words
                )
            )
        return out

    def _draw_planes(self, word_index: int, offset: int, k: int, max_steps: int):
        """Uniform planes for the subword/end draws and the starter draws, or ``None`` pairs."""
        if self.release_rng_stream:
            return None, None
        return (
            self._uniform_block(("kw", word_index), offset, k, max_steps + 1),
            self._uniform_block(("kws", word_index), offset, k, max_steps),
        )

    # ---- the K-sequential-word sampler (per-stream state machine) ----
    def _sample_k_words(
        self,
        logits_after_prev,
        master_kv,
        k,
        gen,
        prev_tok,
        bos_id,
        word_index=0,
        offset=0,
        first_words=1,
    ):
        """Sample k streams of K whole words. Returns (logp[k,K] nats, idhash[k,K] uint64).

        Reduced-vocab sampling rule: a word = a starter token (Ġ/Ċ/ĉ; subword right
        after BOS) then [subword|EOW] draws until EOW. On EOW the same logits also
        sample the next word's starter (no extra forward), so streams progress
        through K words despite variable word length. Depth 1 is the unit:
        ``first_words`` whole words, for a unit of several whitespace words.
        """
        import torch
        from transformers import DynamicCache

        dev = self._device
        n_sub = self._subword_ids.numel()
        logp = torch.zeros((k, self.max_words), device=dev, dtype=torch.float64)
        idh = torch.zeros((k, self.max_words), device=dev, dtype=torch.int64)
        wd = torch.zeros(k, device=dev, dtype=torch.long)  # current word index 0..K
        running = torch.zeros(
            k, device=dev, dtype=torch.int64
        )  # rolling token hash for current word
        done = torch.zeros(k, dtype=torch.bool, device=dev)

        def fold(h, tok):  # rolling int64 hash, wraps on overflow
            return h * _HASH_PRIME + (tok.to(torch.int64) + 1)

        max_steps = self._step_budget(first_words)
        # Words still to fold into depth 1 before the unit closes.
        unit_left = torch.full((k,), first_words - 1, device=dev, dtype=torch.long)
        u_tok, u_start = self._draw_planes(word_index, offset, k, max_steps)

        # Word 0, token 1: starter from prefix logits (subword if right after BOS).
        first_sub = bos_id is not None and prev_tok == bos_id
        allowed = self._subword_ids if first_sub else self._space_ids
        probs = torch.softmax(logits_after_prev.index_select(1, allowed), dim=-1)
        li = (
            torch.multinomial(probs, 1, generator=gen)
            if u_tok is None
            else self._inverse_cdf(probs, u_tok[:, 0])
        )
        sp = probs.gather(1, li).squeeze(1).clamp_min(1e-40)
        logp[:, 0] += torch.log(sp)
        cur = allowed[li.squeeze(1)]
        running = fold(torch.zeros(k, device=dev, dtype=torch.int64), cur)
        cur = cur.unsqueeze(1)

        kv = DynamicCache.from_legacy_cache(master_kv)
        with torch.no_grad():
            for step in range(max_steps):
                if bool(done.all()):
                    break
                o = self._model(cur, past_key_values=kv, use_cache=True)
                kv = o.past_key_values
                probs = torch.softmax(o.logits[:, -1, :].double(), dim=-1)
                sub_p = probs.index_select(1, self._subword_ids)
                spc_p = probs.index_select(1, self._space_ids)
                eow_p = spc_p.sum(1, keepdim=True)
                local = torch.cat([sub_p, eow_p], dim=1)  # [subword | EOW]
                li = (
                    torch.multinomial(local, 1, generator=gen)
                    if u_tok is None
                    else self._inverse_cdf(local, u_tok[:, step + 1])
                ).squeeze(1)
                pr = local.gather(1, li.unsqueeze(1)).squeeze(1).clamp_min(1e-40)
                lp = torch.log(pr)
                is_sub = li < n_sub
                act = ~done
                # accumulate into the stream's current word
                widx = wd.clamp_max(self.max_words - 1)
                logp[torch.arange(k, device=dev), widx] += torch.where(
                    act, lp, torch.zeros_like(lp)
                )

                # subword continues current word; EOW closes it
                sub_tok = self._subword_ids[li.clamp_max(n_sub - 1)]
                running = torch.where(is_sub & act, fold(running, sub_tok), running)

                eow = (~is_sub) & act
                # close word on EOW; a word inside the depth-1 unit closes without advancing
                if bool(eow.any()):
                    arange = torch.arange(k, device=dev)
                    within = eow & (wd == 0) & (unit_left > 0)
                    unit_left = torch.where(within, unit_left - 1, unit_left)
                    closes = eow & ~within
                    idh[arange[closes], widx[closes]] = running[closes]
                    wd = torch.where(closes, wd + 1, wd)
                    newly_done = closes & (wd >= self.max_words)
                    done = done | newly_done
                    # open next word: sample a starter from the same logits
                    cont = eow & (~done)
                    sp_probs = torch.softmax(
                        o.logits[:, -1, :].double().index_select(1, self._space_ids),
                        dim=-1,
                    )
                    sli = (
                        torch.multinomial(sp_probs, 1, generator=gen)
                        if u_start is None
                        else self._inverse_cdf(sp_probs, u_start[:, step])
                    ).squeeze(1)
                    ssp = (
                        sp_probs.gather(1, sli.unsqueeze(1)).squeeze(1).clamp_min(1e-40)
                    )
                    starter = self._space_ids[sli]
                    nwidx = wd.clamp_max(self.max_words - 1)
                    logp[arange[cont], nwidx[cont]] += torch.log(ssp[cont])
                    opened = torch.where(
                        within,
                        fold(running, starter),
                        fold(torch.zeros(k, device=dev, dtype=torch.int64), starter),
                    )
                    running = torch.where(cont, opened, running)
                    # next token to feed: subword for sub streams, new starter for cont streams
                    nxt = torch.where(
                        is_sub, sub_tok, torch.where(cont, starter, sub_tok)
                    )
                else:
                    nxt = sub_tok
                # no per-word cap: only the global max_steps loop bounds sampling;
                # streams still mid-word at the cap are closed after the loop.
                cur = nxt.unsqueeze(1)

        self._hash_unclosed(idh, wd, running)
        return logp.cpu().numpy(), idh.cpu().numpy().astype(np.uint64)

    def _hash_unclosed(self, idh, wd, running) -> None:
        """Store the running hash of every stream the step cap left mid-word."""
        import torch

        unclosed = wd < self.max_words
        if not bool(unclosed.any()):
            return
        ar = torch.arange(wd.numel(), device=wd.device)
        wi = wd.clamp_max(self.max_words - 1)
        idh[ar[unclosed], wi[unclosed]] = torch.where(
            idh[ar[unclosed], wi[unclosed]] == 0,
            running[unclosed],
            idh[ar[unclosed], wi[unclosed]],
        )
