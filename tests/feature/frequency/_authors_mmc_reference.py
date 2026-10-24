"""Test reference: an independent port of the authors' sampled unigram estimator.

:func:`mmc_estimate` ports ``meco_rep/util.py``'s ``rb_estimator_chain`` and
:func:`boundary_first_token_probs` its ``get_boundary_position_probs_batch``,
with this package's ``position_ids`` and BOS handling. Kept close to the upstream
source so the two can be diffed.

Methodology: ``src/replica/studies/hopton2026unigram/README.md``.
"""

import torch


def _encode_with_bos(tokenizer, samples):
    """Sample token ids, BOS-prefixed."""
    bos = tokenizer.bos_token_id
    if bos is None:
        bos = tokenizer.eos_token_id
    return [
        [bos] + tokenizer.encode(s, add_special_tokens=True) if s else [bos]
        for s in samples
    ]


def _padded_batch(sequences, pad_id, device):
    """Left-pad, giving the real tokens their unpadded positions."""
    max_len = max(len(s) for s in sequences)
    padded, masks = [], []
    for seq in sequences:
        pad_len = max_len - len(seq)
        padded.append([pad_id] * pad_len + list(seq))
        masks.append([0] * pad_len + [1] * len(seq))
    input_ids = torch.tensor(padded, dtype=torch.long, device=device)
    attention_mask = torch.tensor(masks, dtype=torch.long, device=device)
    position_ids = (attention_mask.cumsum(-1) - 1).clamp(min=0)
    return input_ids, attention_mask, position_ids, max_len


def boundary_first_token_probs(
    token_id, samples, model, tokenizer, batch_size, marker="Ġ"
):
    """``P(token_id | prefix)`` at word-boundary positions of every sample."""
    device = next(model.parameters()).device
    out = []
    for i in range(0, len(samples), batch_size):
        batch = samples[i : i + batch_size]
        encoded = _encode_with_bos(tokenizer, batch)
        input_ids, attention_mask, position_ids, max_len = _padded_batch(
            encoded, tokenizer.pad_token_id, device
        )
        with torch.no_grad():
            logits = model(
                input_ids, attention_mask=attention_mask, position_ids=position_ids
            ).logits
            tok_probs = torch.softmax(logits, dim=-1)[:, :, token_id]
        for j, ids in enumerate(encoded):
            valid = tok_probs[j, max_len - len(ids) :].cpu().tolist()
            strs = tokenizer.convert_ids_to_tokens(ids)
            record = []
            for p in range(len(ids)):
                if p == 0 or p == len(strs) - 1 or strs[p + 1].startswith(marker):
                    record.append((p, valid[p]))
            out.append(record)
    return out


def mmc_estimate(unigram, samples, model, tokenizer, batch_size=64, marker="Ġ"):
    """Marginal Monte Carlo estimate over word-boundary positions."""
    device = next(model.parameters()).device
    unigram_token_ids = tokenizer.encode(unigram, add_special_tokens=False)
    n_tokens = len(unigram_token_ids)
    if not n_tokens:
        return 0.0

    total_prob = 0.0
    n_samples = 0
    total_prefixes = 0

    for i in range(0, len(samples), batch_size):
        batch_samples = samples[i : i + batch_size]
        encoded = _encode_with_bos(tokenizer, batch_samples)
        boundary_probs_by_sample = boundary_first_token_probs(
            unigram_token_ids[0], batch_samples, model, tokenizer, batch_size, marker
        )

        position_list, joint_list = [], []
        for j, boundary_probs in enumerate(boundary_probs_by_sample):
            for p, prob in boundary_probs:
                position_list.append(encoded[j][: p + 1])
                joint_list.append(prob)

        joint = torch.tensor(joint_list, dtype=torch.float64)

        for t_idx in range(1, n_tokens):
            token_id = unigram_token_ids[t_idx]
            word_prefix = unigram_token_ids[:t_idx]
            step_probs = []
            for k in range(0, len(position_list), batch_size):
                chunk = position_list[k : k + batch_size]
                sequences = [list(prefix) + word_prefix for prefix in chunk]
                input_ids, attention_mask, position_ids, _ = _padded_batch(
                    sequences, tokenizer.pad_token_id, device
                )
                with torch.no_grad():
                    logits = model(
                        input_ids,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                    ).logits
                    probs = torch.softmax(logits[:, -1, :], dim=-1)
                    step_probs.append(probs[:, token_id].cpu().double())
            joint = joint * torch.cat(step_probs) if step_probs else joint

        total_prob += float(joint.sum().item())
        total_prefixes += len(position_list)
        n_samples += len(batch_samples)

    if n_samples + total_prefixes == 0:
        return 0.0
    return total_prob / (n_samples + total_prefixes)
