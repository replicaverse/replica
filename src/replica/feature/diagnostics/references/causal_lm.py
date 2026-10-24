"""Token-level reference quantities from a causal language model.

Raw model logits supply exact token targets. Continuation samples use the
production sampler and define a finite empirical target, not exact entropy.
"""

import numpy as np

from replica.core import Stimulus
from replica.feature.backends.start_state import start_ids
from replica.feature.core.outputs.information import LN2, nats_to_bits

__all__ = ["continuation_log2p_pool", "exact_token_quantities"]


def continuation_log2p_pool(
    model_id: str,
    words: list[str],
    n_samples: int,
    max_tokens: int = 2,
    add_bos: bool = True,
    seed: int = 0,
    position: int = -1,
    chunk_size: int = 512,
) -> np.ndarray:
    """Per-sample log2-probabilities of continuations drawn from a real LM.

    Each value is ``log2 p(c_i | context)`` for one ``max_tokens``-token
    continuation sampled from the model's full next-token distribution by the
    production :class:`ContinuationEntropyLM` sampler. ``-mean(pool)`` is the
    Shannon estimate; ``(1/(1-a)) * log2(mean(2 ** ((a-1) * pool)))`` a Renyi one.

    Args:
        model_id: Hugging Face id of the causal LM to sample from.
        words: context words, in reading order, one probe stimulus.
        n_samples: continuations to draw, which is the size of the pool.
        max_tokens: tokens per continuation.
        add_bos: prepend the model's BOS token to the context.
        seed: seed of the sampler, so the pool is reproducible.
        position: which word's output to take the pool from; the default takes
            the last word, which has the most left context.
        chunk_size: continuations drawn per batch, which bounds peak memory
            and leaves the pool unchanged.

    Returns:
        The pool. Its aggregate defines an empirical target for resampling
        experiments; bootstrap subsamples estimate that target. The finite pool
        does not give the model's exact entropy.
    """
    from replica.feature.diagnostics.references.markov import make_stimulus
    from replica.feature.entropy.estimators import ContinuationEntropyLM

    # chunk_size bounds GPU memory (draw the M samples in batches); the
    # estimate is identical, only the per-batch memory peak changes.
    lm = ContinuationEntropyLM(
        model_id,
        n_samples=int(n_samples),
        max_tokens=int(max_tokens),
        add_bos=add_bos,
        seed=seed,
        chunk_size=int(chunk_size),
    )
    outs = lm(make_stimulus(list(words)))
    samples_nats = np.asarray(
        outs[position].values["log_p_samples_nats"], dtype=np.float64
    )
    if samples_nats.size == 0:
        raise ValueError(
            f"no continuation samples at position {position} "
            "(needs left context; use a multi-word context)"
        )
    return nats_to_bits(samples_nats)


_MODEL_CACHE: dict[tuple[str, str], tuple] = {}


def _load(model_id: str, device: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    key = (model_id, device)
    if key not in _MODEL_CACHE:
        tok = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id).eval().to(device)
        _MODEL_CACHE[key] = (tok, model)
    return _MODEL_CACHE[key]


def exact_token_quantities(
    model_id: str,
    stimulus: Stimulus,
    target_layer: str = "words",
    add_bos: bool = True,
    renyi_alphas: tuple[float, ...] = (0.5, 2.0),
    device: str = "cpu",
) -> dict[str, np.ndarray]:
    """Exact per-word token-level quantities from one full forward pass.

    Adds a ``{model_id}_diagnostics_tokens`` layer to ``stimulus``, overwritten
    on repeat calls; loaded models are cached on the module for the process
    lifetime.

    Args:
        model_id: Hugging Face id of the causal LM to score with.
        stimulus: text to score; the returned arrays run one value per unit of
            ``target_layer``.
        target_layer: annotation layer defining the unit granularity.
        add_bos: prepend the model's BOS token, which gives unit 0 a context.
        renyi_alphas: orders to return a token-level Rényi entropy for.
        device: torch device the model is loaded and run on.

    Returns:
        Per-unit arrays in bits, NaN where undefined: ``surprisal`` (sum over
        the word's subwords), ``token_entropy`` and
        ``token_renyi_entropy_alpha{a}`` (full token vocabulary, at the word's
        first subword, REPLICA's first-token convention).
    """
    import torch

    from replica.feature.core.annotation.alignment import align_from_tokenizer

    tok, model = _load(model_id, device)
    alignment = align_from_tokenizer(
        stimulus,
        tok,
        source_layer=target_layer,
        layer_name=f"{model_id}_diagnostics_tokens",
    )
    token_ids = [v for _, v in stimulus.layers[alignment.target_layer]]
    prefix = start_ids(tok) if add_bos else []
    bos = len(prefix)
    ids = prefix + token_ids

    with torch.no_grad():
        logits = model(torch.tensor([ids], device=device)).logits[0].double()
    log_probs = torch.log_softmax(logits, dim=-1).cpu().numpy()

    n = len(alignment.mapping)
    out = {
        "surprisal": np.full(n, np.nan),
        "token_entropy": np.full(n, np.nan),
    }
    for a in renyi_alphas:
        out[f"token_renyi_entropy_alpha{float(a)}"] = np.full(n, np.nan)

    for i, positions in enumerate(alignment.mapping):
        if not positions:
            continue
        first = bos + positions[0]
        if first == 0:
            continue  # no left context
        d = log_probs[first - 1]
        p = np.exp(d)
        out["token_entropy"][i] = float(-(p * d).sum() / LN2)
        for a in renyi_alphas:
            if float(a) == 1.0:
                # The Shannon limit; the Rényi formula is 0/0 there.
                value = out["token_entropy"][i]
            else:
                scaled = float(a) * d
                m = scaled.max()
                log_sum = m + np.log(np.exp(scaled - m).sum())
                value = float(log_sum / ((1.0 - float(a)) * LN2))
            out[f"token_renyi_entropy_alpha{float(a)}"][i] = value
        lp = 0.0
        for pos in positions:
            t = bos + pos
            if t == 0:
                lp = np.nan
                break
            lp += log_probs[t - 1, ids[t]]
        out["surprisal"][i] = -lp / LN2
    return out
