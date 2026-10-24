r"""Write the per-sample terms the variance bootstrap resamples.

One pass over mGPT produces everything :mod:`variance` needs, so the bootstrap
itself runs on CPU with no model. Port of ``precompute_rb`` / ``precompute_mc``
in the authors' ``meco_rep/variance_retok.py``, reusing REPLICA's
:func:`~replica.feature.frequency.sampled._boundary_probs_per_sample`,
which is already anchored against their ``util.py`` in
``tests/feature/test_sampled_unigram.py``.

Run::

    python -m replica.studies.hopton2026unigram.dump_variance_inputs \\
        --samples /path/to/meco_rep/samples_maxLen50/N500.json \\
        --words   /path/to/context-reading-time/merged_data_no_zero/en.csv \\
        --out     src/replica/studies/hopton2026unigram/data \\
        --device cuda --batch-size 128 --verify

Writes ``variance_samples.parquet`` (one row per token and sample) and
``variance_boundaries.parquet`` (one row per sample). The boundary counts are
properties of the sampled string rather than the unigram, so they are stored
once instead of per token.

This is a long job: cost per unigram grows with its subword count, so a few
thousand types run for hours, and the script shards, logs progress and marks
completion:

* Each unigram's result lands in its own shard under ``--shard-dir`` as soon
  as it is computed. Re-running skips finished shards, so an interrupted job
  resumes where it stopped.
* Progress is logged to stdout and to ``dump.log`` in the shard directory,
  with a work-weighted estimate of time remaining that accounts for
  multi-token unigrams costing more.
* Completion is explicit: the run assembles the shards, checks every unigram
  is present, and writes ``DONE.json``. Without that file the dump is
  unfinished.

``--finalize-only`` assembles and validates existing shards without touching
the model, for when a job is killed after the last shard was written.
``--verify`` re-derives sample estimates from the assembled arrays and compares
them against :func:`mmc_estimate`, catching a mis-ordered or truncated
dump before it reaches the committed data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ...feature.frequency.sampled import (
    _boundary_probs_per_sample,
    _encode_samples,
    _left_pad_batch,
    _padded_logits,
    _word_boundary_marker,
)

MODEL_NAME = "sberbank-ai/mGPT"

#: Relative cost of one extra subword token, used only to weight the
#: work-remaining estimate that gets logged.
_COST_PER_EXTRA_TOKEN = 20.7

log = logging.getLogger("hopton2026.dump_variance")


def _setup_logging(log_path: Path) -> None:
    """Log to stdout and to a file, both timestamped."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%H:%M:%S")
    stream = logging.StreamHandler()
    stream.setFormatter(fmt)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(fmt)
    log.setLevel(logging.INFO)
    log.handlers = [stream, file_handler]


def _format_duration(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def load_samples(path: Path) -> list[str]:
    """Read one sampled string per line, matching the authors' loader."""
    text = path.read_text(encoding="utf-8")
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError:
        return [line.strip() for line in text.splitlines() if line.strip()]
    return [str(s) for s in loaded]


def token_for_row(word: str, ianum: int, space_lookup: dict | None, trialid) -> str:
    """The surface form estimated for one interest area.

    A leading space is prepended for every unit after the first in a trial.
    ``space_lookup`` overrides that per unit when the character-level data is
    available, as in the authors' ``build_space_lookup``.
    """
    if space_lookup is not None:
        has_space = space_lookup.get((trialid, ianum), ianum > 1)
    else:
        has_space = ianum > 1
    return f" {word}" if has_space else word


def precompute_mc(token: str, samples: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Per-sample NMC counts and denominators.

    Counts overlapping-free substring occurrences of ``token`` rather than
    splitting the sample, so a leading space is part of the match.
    """
    counts, boundaries = [], []
    for sample in samples:
        count, pos = 0, 0
        while True:
            pos = sample.find(token, pos)
            if pos == -1:
                break
            count += 1
            pos += len(token)
        counts.append(count)
        boundaries.append(len(sample.split()) + 1)
    return np.array(counts, dtype=np.int32), np.array(boundaries, dtype=np.int32)


def precompute_rb(
    token: str, samples: list[str], model, tokenizer, batch_size: int, marker: str
) -> tuple[np.ndarray, np.ndarray]:
    """Per-sample MMC numerators and boundary counts.

    The first token's probability at each word boundary comes from one batched
    sweep; the remaining tokens of a multi-token unigram are chained on with
    one forward pass per token over the boundary prefixes.
    """
    import torch

    device = next(model.parameters()).device
    token_ids = tokenizer.encode(token, add_special_tokens=False)
    if not token_ids:
        n = len(samples)
        return np.zeros(n), np.zeros(n, dtype=np.int32)

    per_sample = _boundary_probs_per_sample(
        token_ids[0], samples, model, tokenizer, batch_size, marker
    )
    # Same encoding the boundary sweep indexed, so ``p`` slices the right prefix
    # and keeps the BOS the first-token probability was conditioned on.
    encoded = _encode_samples(tokenizer, samples)

    positions: list[list[int]] = []
    joint_list: list[float] = []
    for j, boundaries in enumerate(per_sample):
        for p, prob in boundaries:
            positions.append(encoded[j][: p + 1])
            joint_list.append(prob)
    joint = torch.tensor(joint_list, dtype=torch.float64)

    for t_idx in range(1, len(token_ids)):
        word_prefix = token_ids[:t_idx]
        step = []
        for k in range(0, len(positions), batch_size):
            chunk = positions[k : k + batch_size]
            sequences = [list(prefix) + word_prefix for prefix in chunk]
            padded, masks, _ = _left_pad_batch(sequences, tokenizer.pad_token_id)
            with torch.no_grad():
                logits = _padded_logits(model, padded, masks, device)
                probs = torch.softmax(logits[:, -1, :], dim=-1)
                step.append(probs[:, token_ids[t_idx]].cpu().double())
        joint *= torch.cat(step)

    prob_sums, counts, ptr = [], [], 0
    for boundaries in per_sample:
        n_bounds = len(boundaries)
        prob_sums.append(float(joint[ptr : ptr + n_bounds].sum()))
        counts.append(n_bounds)
        ptr += n_bounds
    return np.array(prob_sums, dtype=float), np.array(counts, dtype=np.int32)


# ---------------------------------------------------------------------------
# Shard bookkeeping
# ---------------------------------------------------------------------------


def _token_list_fingerprint(tokens: list[str]) -> str:
    """Hash of the ordered token list, so a resume cannot mix two runs.

    Shards are named by position, so a changed word list or space rule would
    otherwise silently pair old shards with new tokens.
    """
    digest = hashlib.sha256()
    for token in tokens:
        digest.update(token.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()[:16]


def _write_manifest(shard_dir: Path, tokens: list[str], samples: int, args) -> None:
    (shard_dir / "manifest.json").write_text(
        json.dumps(
            {
                "fingerprint": _token_list_fingerprint(tokens),
                "n_tokens": len(tokens),
                "n_samples": samples,
                "model_name": args.model_name,
                "batch_size": args.batch_size,
                "samples_path": str(args.samples),
                "words_path": str(args.words),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _check_manifest(shard_dir: Path, tokens: list[str]) -> None:
    path = shard_dir / "manifest.json"
    if not path.exists():
        return
    prior = json.loads(path.read_text(encoding="utf-8"))
    current = _token_list_fingerprint(tokens)
    if prior.get("fingerprint") != current:
        raise ValueError(
            f"{shard_dir} holds shards for a different token list "
            f"({prior.get('fingerprint')} against {current}). Point --shard-dir "
            f"somewhere else or delete the existing shards."
        )


def _completed(shard_dir: Path, n_tokens: int) -> set[int]:
    """Indices whose shard exists and reads back cleanly."""
    done = set()
    for path in shard_dir.glob("token_*.parquet"):
        try:
            index = int(path.stem.split("_")[1])
        except (IndexError, ValueError):
            continue
        if not 0 <= index < n_tokens:
            continue
        try:
            pd.read_parquet(path, columns=["sample_idx"])
        except Exception:
            log.warning("shard %s is unreadable and will be recomputed", path.name)
            path.unlink(missing_ok=True)
            continue
        done.add(index)
    return done


def _assemble(shard_dir: Path, out_dir: Path, tokens: list[str]) -> pd.DataFrame:
    """Concatenate shards in token order, checking none are missing."""
    missing = [
        i
        for i in range(len(tokens))
        if not (shard_dir / f"token_{i:06d}.parquet").exists()
    ]
    if missing:
        raise ValueError(
            f"{len(missing)} of {len(tokens)} shards are missing (first few: "
            f"{missing[:5]}). Re-run without --finalize-only to fill them in."
        )
    frames = [
        pd.read_parquet(shard_dir / f"token_{i:06d}.parquet")
        for i in range(len(tokens))
    ]
    long = pd.concat(frames, ignore_index=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    long.to_parquet(
        out_dir / "variance_samples.parquet", compression="zstd", index=False
    )
    return long


def _write_shard(shard_dir: Path, index: int, token: str, rb_sums, mc_counts) -> None:
    """Write one unigram's shard, renaming into place once complete.

    A crash mid-write leaves only the ``.tmp`` file, so a resume never mistakes
    a torn shard for a finished one.
    """
    tmp = shard_dir / f"token_{index:06d}.parquet.tmp"
    pd.DataFrame(
        {
            "token": token,
            "sample_idx": np.arange(len(rb_sums), dtype=np.int32),
            "rb_prob_sum": rb_sums,
            "mc_count": mc_counts,
        }
    ).to_parquet(tmp, compression="zstd", index=False)
    tmp.rename(shard_dir / f"token_{index:06d}.parquet")


def _store_boundaries(shard_dir: Path, token: str, rb_counts, mc_bounds) -> None:
    """Write the per-sample boundary vectors once, then check they hold."""
    path = shard_dir / "boundaries.parquet"
    if not path.exists():
        pd.DataFrame(
            {
                "sample_idx": np.arange(len(rb_counts), dtype=np.int32),
                "rb_boundary_count": rb_counts,
                "mc_boundary_count": mc_bounds,
            }
        ).to_parquet(path, compression="zstd", index=False)
        return
    stored = pd.read_parquet(path)["rb_boundary_count"].to_numpy()
    if not np.array_equal(stored, rb_counts):
        raise ValueError(
            f"boundary counts for {token!r} differ from the stored vector; "
            f"they are expected to depend on the sampled string alone"
        )


def _compute_shards(
    *,
    tokens: list[str],
    samples: list[str],
    weights: np.ndarray,
    done: set[int],
    shard_dir: Path,
    model,
    tokenizer,
    marker: str,
    batch_size: int,
    log_every: int,
) -> None:
    """Fill in every shard not already present, logging cost-weighted progress."""
    total_weight = float(weights.sum())
    done_weight = float(weights[list(done)].sum()) if done else 0.0
    started = time.monotonic()
    worked = 0.0

    for index, token in enumerate(tokens):
        if index in done:
            continue
        t0 = time.monotonic()
        rb_sums, rb_counts = precompute_rb(
            token, samples, model, tokenizer, batch_size, marker
        )
        mc_counts, mc_bounds = precompute_mc(token, samples)

        _store_boundaries(shard_dir, token, rb_counts, mc_bounds)
        _write_shard(shard_dir, index, token, rb_sums, mc_counts)

        worked += weights[index]
        done_weight += weights[index]
        elapsed = time.monotonic() - started
        if (index + 1) % log_every == 0 or index + 1 == len(tokens):
            rate = worked / elapsed if elapsed > 0 else 0.0
            eta = (total_weight - done_weight) / rate if rate > 0 else float("nan")
            log.info(
                "%5d/%d  %5.1f%% by cost  last %6.1fs  elapsed %s  eta %s",
                index + 1,
                len(tokens),
                100.0 * done_weight / total_weight,
                time.monotonic() - t0,
                _format_duration(elapsed),
                _format_duration(eta),
            )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--samples", type=Path, required=True)
    p.add_argument("--words", type=Path, required=True, help="per-unit CSV with ia")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--shard-dir", type=Path, default=None)
    p.add_argument("--model-name", default=MODEL_NAME)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="cpu")
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--finalize-only", action="store_true")
    p.add_argument("--verify", action="store_true")
    a = p.parse_args(argv)

    shard_dir = a.shard_dir or (a.out / "_variance_shards")
    shard_dir.mkdir(parents=True, exist_ok=True)
    _setup_logging(shard_dir / "dump.log")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(a.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    samples = load_samples(a.samples)
    words = pd.read_csv(a.words).drop_duplicates(["trialid", "ianum"])
    tokens = sorted(
        {
            token_for_row(str(r.ia), int(r.ianum), None, r.trialid)
            for r in words.itertuples()
        }
    )
    _check_manifest(shard_dir, tokens)
    _write_manifest(shard_dir, tokens, len(samples), a)

    # The paper averages variance over corpus units, so a frequent word counts
    # once per occurrence. Estimation happens once per distinct form (about
    # half as many here), and this mapping expands the results back to units
    # at aggregation time.
    a.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "trialid": words["trialid"].to_numpy(),
            "ianum": words["ianum"].to_numpy(),
            "token": [
                token_for_row(str(r.ia), int(r.ianum), None, r.trialid)
                for r in words.itertuples()
            ],
        }
    ).to_parquet(a.out / "variance_units.parquet", compression="zstd", index=False)

    subword_counts = [
        max(len(tokenizer.encode(t, add_special_tokens=False)), 1) for t in tokens
    ]
    weights = np.array(
        [1.0 + (n - 1) * _COST_PER_EXTRA_TOKEN for n in subword_counts], dtype=float
    )
    histogram = pd.Series(subword_counts).value_counts().sort_index()

    log.info("token list      : %d unigrams over %d samples", len(tokens), len(samples))
    log.info("subword counts  : %s", histogram.to_dict())
    log.info(
        "reference cost  : about %s on an H100 at batch 128 (2.1 + 43.5*(T-1) per unigram)",
        _format_duration(float(np.sum(2.1 + (np.array(subword_counts) - 1) * 43.5))),
    )
    log.info("shard directory : %s", shard_dir)

    done = _completed(shard_dir, len(tokens))
    if done:
        log.info(
            "resuming        : %d of %d shards already present", len(done), len(tokens)
        )

    model = None
    if not a.finalize_only:
        model = AutoModelForCausalLM.from_pretrained(a.model_name).to(a.device).eval()
        log.info("model loaded on %s", a.device)
        _compute_shards(
            tokens=tokens,
            samples=samples,
            weights=weights,
            done=done,
            shard_dir=shard_dir,
            model=model,
            tokenizer=tokenizer,
            marker=_word_boundary_marker(tokenizer),
            batch_size=a.batch_size,
            log_every=a.log_every,
        )

    long = _assemble(shard_dir, a.out, tokens)
    bounds = pd.read_parquet(shard_dir / "boundaries.parquet")
    bounds.to_parquet(
        a.out / "variance_boundaries.parquet", compression="zstd", index=False
    )

    n_expected = len(tokens) * len(samples)
    if len(long) != n_expected:
        raise ValueError(f"assembled {len(long)} rows, expected {n_expected}")
    if long["token"].nunique() != len(tokens):
        raise ValueError(
            f"assembled {long['token'].nunique()} distinct tokens, "
            f"expected {len(tokens)}"
        )

    if a.verify:
        from ...feature.backends.generation import declared_position_limit
        from ...feature.frequency.sampled import mmc_estimate, start_prefix

        if model is None:
            model = AutoModelForCausalLM.from_pretrained(a.model_name).to(a.device)
            model = model.eval()
        denom = len(samples) + float(bounds["rb_boundary_count"].sum())
        for token in tokens[:5]:
            sub = long[long.token == token]
            got = float(sub.rb_prob_sum.sum()) / denom
            want = mmc_estimate(
                token,
                samples,
                model,
                tokenizer,
                a.batch_size,
                start_ids=start_prefix(tokenizer),
                context_window=declared_position_limit(model.config),
            )
            if not np.isclose(got, want, rtol=1e-9, atol=0):
                raise ValueError(f"{token!r}: assembled {got!r} against {want!r}")
        log.info("verify ok       : 5 tokens reproduce mmc_estimate")

    (a.out / "DONE.json").write_text(
        json.dumps(
            {
                "n_tokens": len(tokens),
                "n_samples": len(samples),
                "n_rows": int(len(long)),
                "fingerprint": _token_list_fingerprint(tokens),
                "verified": bool(a.verify),
                "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    log.info("=" * 62)
    log.info(
        "DUMP COMPLETE   %d tokens x %d samples = %d rows",
        len(tokens),
        len(samples),
        len(long),
    )
    log.info("wrote           %s", a.out / "variance_samples.parquet")
    log.info("wrote           %s", a.out / "variance_boundaries.parquet")
    log.info("wrote           %s", a.out / "DONE.json")
    log.info("shards kept in  %s (safe to delete once committed)", shard_dir)
    log.info("=" * 62)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
