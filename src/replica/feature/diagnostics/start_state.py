"""Compare candidate start tokens on caller-supplied texts.

The report compares tokenizer prefixes, mean special-token probabilities after
capped probe inputs, first-token surprisal and the scripts of generated samples.
"""

from __future__ import annotations

from replica.feature.backends.start_state import resolve_start_state


def start_state_report(model_name: str, texts: list[str], device: str = "cpu") -> str:
    """Compare candidate start tokens on supplied texts.

    Reports the six largest mean special-token probabilities after the first
    256 tokens of each nonempty input, mean first-token surprisal under each
    candidate prefix, and the scripts of samples drawn from each candidate.

    Args:
        model_name: Hub id or snapshot path.
        texts: Document-initial texts selected by the caller.
        device: Torch device for the model.
    """
    import collections
    import math

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device).eval()
    specials = {
        str(tokenizer.convert_ids_to_tokens(i)): int(i)
        for i in sorted(set(getattr(tokenizer, "all_special_ids", []) or []))
    }
    declared = [
        str(tokenizer.convert_ids_to_tokens(i))
        for i in tokenizer("x", add_special_tokens=True)["input_ids"]
        if i in specials.values()
    ]
    state = resolve_start_state(tokenizer, model_name)
    lines = [
        f"model {model_name}: bos={tokenizer.bos_token!r} eos={tokenizer.eos_token!r}",
        f"tokenizer adds on encode: {declared or 'nothing'}",
        f"resolved prefix: {list(state.tokens) or 'none'} (basis {state.basis})",
    ]
    end_mass: dict[str, list[float]] = collections.defaultdict(list)
    first_ids = []
    with torch.no_grad():
        for text in texts:
            ids = tokenizer(text, add_special_tokens=False)["input_ids"][:256]
            if not ids:
                continue
            first_ids.append(ids[0])
            last = torch.log_softmax(
                model(torch.tensor([ids], device=device)).logits[0, -1], -1
            )
            for name, ident in specials.items():
                end_mass[name].append(math.exp(float(last[ident])))
    ranked = _rank_special_mass(end_mass)
    lines.append(
        "Mean P(special | capped input), top six: "
        + ", ".join(f"{k} {p:.2e}" for p, k in ranked[:6])
    )
    candidates = dict.fromkeys(
        [tokenizer.bos_token, tokenizer.eos_token, *state.tokens, ranked[0][1]]
    )
    for token in candidates:
        if token is None or token not in specials:
            continue
        ident = specials[token]
        with torch.no_grad():
            dist = torch.log_softmax(
                model(torch.tensor([[ident]], device=device)).logits[0, -1], -1
            )
            first = -sum(float(dist[i]) for i in first_ids) / max(1, len(first_ids))
            torch.manual_seed(0)
            drawn = model.generate(
                torch.tensor([[ident]], device=device),
                do_sample=True,
                top_k=0,
                max_new_tokens=24,
                num_return_sequences=32,
                pad_token_id=tokenizer.pad_token_id
                if tokenizer.pad_token_id is not None
                else tokenizer.eos_token_id,
            )
        mix = collections.Counter(
            _script(tokenizer.decode(g[1:], skip_special_tokens=True)) for g in drawn
        )
        lines.append(
            f"after {token}: first-token surprisal {first / math.log(2):.2f} bits; "
            "samples " + ", ".join(f"{k} {v / 32:.2f}" for k, v in mix.most_common(4))
        )
    return "\n".join(lines)


def _rank_special_mass(end_mass: dict[str, list[float]]) -> list[tuple[float, str]]:
    """Rank special tokens by mean probability after the capped probe inputs.

    Raises:
        ValueError: No probe text produced a token, or the tokenizer declares
            no special token, so there is nothing to rank.
    """
    if not end_mass or not all(end_mass.values()):
        raise ValueError(
            "no special-token mass to rank: the probe texts yielded no tokens "
            "or the tokenizer declares no special tokens; pass --texts with "
            "document-initial texts"
        )
    return sorted(((sum(v) / len(v), k) for k, v in end_mass.items()), reverse=True)


def _script(text: str) -> str:
    """The dominant script of *text*'s letters, by Unicode block."""
    counts: dict[str, int] = {}
    for char in text:
        if not char.isalpha():
            continue
        code = ord(char)
        if code < 0x250:
            key = "latin"
        elif 0x400 <= code < 0x530:
            key = "cyrillic"
        elif 0x590 <= code < 0x600:
            key = "hebrew"
        elif 0x600 <= code < 0x700:
            key = "arabic"
        elif 0x3000 <= code < 0xA000 or 0xAC00 <= code < 0xD7B0:
            key = "cjk"
        else:
            key = "other"
        counts[key] = counts.get(key, 0) + 1
    return max(counts, key=counts.get) if counts else "none"


def main(argv: list[str] | None = None) -> int:
    """Compare start tokens using the document-initial texts in ``--texts``.

    Args:
        argv: Command-line arguments excluding the program name, defaulting to
            ``sys.argv[1:]``.
    """
    import argparse
    import sys
    from pathlib import Path

    parser = argparse.ArgumentParser(
        prog="replica start-state",
        description="Compare candidate start tokens on supplied texts.",
    )
    parser.add_argument("model", help="Hub id or snapshot path")
    parser.add_argument(
        "--texts",
        type=Path,
        required=True,
        help="UTF-8 file with one document-initial text per line",
    )
    parser.add_argument("--device", default="cpu", help="torch device")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    texts = [
        t for t in args.texts.read_text(encoding="utf-8").splitlines() if t.strip()
    ]
    if not texts:
        parser.error("--texts must contain at least one nonblank text")
    print(start_state_report(args.model, texts, device=args.device))
    return 0
