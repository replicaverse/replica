"""Model-specific defaults shared by scoring and sampling backends."""

from dataclasses import dataclass

from .hub import hub_id


@dataclass(frozen=True)
class ModelDefaults:
    """Model-specific overrides; ``None`` defers to tokenizer or model metadata."""

    start_token: str | None = None
    context_window: int | None = None


# mGPT's model card specifies 512 training tokens. EOS is REPLICA's empirical
# scoring choice, not a claim about the model's training prefix.
_MGPT = ModelDefaults(start_token="<|endoftext|>", context_window=512)
_NO_OVERRIDES = ModelDefaults()


def model_defaults(model_name: str) -> ModelDefaults:
    """Return REPLICA's defaults for a model.

    Args:
        model_name: Hub id or local cache snapshot path.
    """
    if hub_id(model_name) in {"ai-forever/mGPT", "sberbank-ai/mGPT"}:
        return _MGPT
    return _NO_OVERRIDES
