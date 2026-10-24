"""Construct surprisal backends from named scoring methods."""

from ..backends import HuggingFaceLM
from ..backends.hub import hub_id
from .char_beam import CharBeamLM
from .wordsprob import WordsProbabilityLM


def surprisal_model(model_id: str = "gpt2", *, method: str = "surprisal", **kwargs):
    """Build a feature model for a named surprisal method.

    Args:
        model_id: Model identifier; word-probability methods also accept
            supported Hugging Face aliases for the upstream model names.
        method: ``surprisal`` selects ``HuggingFaceLM``; ``char_beam`` selects
            ``CharBeamLM``; ``modelblocks_surprisal`` selects
            ``ModelBlocksSurprisalLM``. ``wordsprobability_surprisal`` uses
            ``WordsProbabilityLM`` with the word-boundary correction, while
            ``wordsprobability_surprisal_buggy`` selects its uncorrected convention.
        kwargs: Forwarded to the selected backend, such as ``add_bos`` for
            ``surprisal`` or ``K`` for ``char_beam``.

    Raises:
        ValueError: The method name is unsupported.
    """
    if method == "surprisal":
        return HuggingFaceLM(model_id, **kwargs)
    if method == "char_beam":
        return CharBeamLM(model_id, **kwargs)
    if method == "modelblocks_surprisal":
        from .modelblocks import ModelBlocksSurprisalLM

        return ModelBlocksSurprisalLM(model_id, **kwargs)
    if method in ("wordsprobability_surprisal", "wordsprobability_surprisal_buggy"):
        # The upstream ``wordsprobability`` package uses ``-small`` /
        # ``-medium`` / etc. suffixes that the HuggingFace hub doesn't
        # use ("gpt2" on HF is GPT-2 small). Accept the bare HF name as
        # an alias for the corresponding upstream id so that the same
        # ``model_id`` string round-trips across methods.
        _PIM_ALIASES = {
            "gpt2": "gpt2-small",
            # mGPT support comes from upstream PR #4 (the add_mgpt_support
            # branch); pyproject.toml pins a commit of that branch.
            "ai-forever/mGPT": "mgpt",
            "EleutherAI/pythia-70m": "pythia-70m",
            "EleutherAI/pythia-160m": "pythia-160m",
            "EleutherAI/pythia-410m": "pythia-410m",
            "EleutherAI/pythia-1.4b": "pythia-14b",
            "EleutherAI/pythia-2.8b": "pythia-28b",
            "EleutherAI/pythia-6.9b": "pythia-69b",
            "EleutherAI/pythia-12b": "pythia-120b",
        }
        resolved = _PIM_ALIASES.get(hub_id(model_id), hub_id(model_id))
        correction = "corrected" if method == "wordsprobability_surprisal" else "buggy"
        from pathlib import Path

        if Path(model_id).is_dir():
            kwargs.setdefault("model_source", model_id)
        return WordsProbabilityLM(resolved, correction=correction, **kwargs)
    raise ValueError(
        f"Unknown surprisal method {method!r}. "
        "Valid: 'surprisal', 'char_beam', 'modelblocks_surprisal', "
        "'wordsprobability_surprisal', 'wordsprobability_surprisal_buggy'."
    )
