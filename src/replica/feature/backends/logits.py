"""Logits processor guarding the sampling feature models.

Imports transformers, so the samplers import it inside the method that
builds their ``generate()`` call.
"""

from __future__ import annotations

import torch
from transformers import LogitsProcessor

from .generation import NonFiniteLogitsError


class FiniteLogitsProcessor(LogitsProcessor):
    """Mask non-finite logits before sampling.

    Some attention kernels intermittently emit ``inf``/``nan`` logits, which
    ``torch.multinomial`` turns into a device-side assert that leaves the CUDA
    context unusable for the rest of the process. Masking keeps the sampled
    distribution unchanged whenever the logits are sane; a fully non-finite row
    raises a catchable error instead.
    """

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        """Send every non-finite score to ``-inf`` so sampling stays defined.

        Args:
            input_ids: Tokens generated so far; the mask depends only on the
                scores, so this is left unread.
            scores: Next-token logits for the current step, one row per
                sequence being generated.

        Raises:
            NonFiniteLogitsError: When some row holds no finite logit at all.
        """
        bad = ~torch.isfinite(scores)
        if bool(bad.any()):
            if bool(bad.all(dim=-1).any()):
                raise NonFiniteLogitsError("a sampling step produced no finite logits")
            scores = scores.masked_fill(bad, float("-inf"))
        return scores
