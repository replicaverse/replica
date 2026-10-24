"""Entropy Feature extractors.

Closed-form over the LM's predictive distribution (Entropy, RenyiEntropy,
ExpectedProbability), at the subword level (TokenEntropy, TokenRenyiEntropy
and their summed variants), and readers for the Monte-Carlo estimators'
pre-computed values (ContinuationEntropy, ContinuationEntropyBuggy,
MCWordEntropy).
"""

from collections.abc import Sequence

import numpy as np

from replica.feature.core.measurement import FeatureMeasurement
from replica.feature.core.outputs._access import _get_next_log_probs
from replica.feature.core.outputs.information import (
    collision_prob,
    renyi_bits,
    shannon_bits,
)
from replica.feature.core.protocols import ModelOutputRecord

from ...core import Stimulus
from ..core.protocols import FeatureNotAvailable
from ..core.reproduction import warn_reproduction_only
from .utils import _renyi_entropy_bits_from_log_p_nats


class _NextLogProbsFeature:
    """Shared reader for closed-form features over ``next_log_probs``.

    Position 0 is NaN when the distribution is missing there (no BOS); a
    missing distribution at a later position raises FeatureNotAvailable.
    """

    def configuration(self) -> dict:
        """Return computation settings without the column alias."""
        return {}

    name: str
    # Feature name used in the raised error; defaults to ``name``.
    _raise_name: str | None = None

    def _compute(self, log_probs: np.ndarray) -> float:
        raise NotImplementedError

    @property
    def required_outputs(self) -> tuple[str, ...]:
        """Model value keys consumed by this feature."""
        return ("next_log_probs",)

    def __call__(
        self,
        stimulus: Stimulus,
        outputs: Sequence[ModelOutputRecord] | None,
        target_layer: str = "words",
    ) -> np.ndarray:
        if outputs is None:
            raise ValueError(f"{type(self).__name__} requires model outputs")
        n = len(outputs)
        vals = np.full(n, np.nan)
        for t in range(n):
            nlp = _get_next_log_probs(outputs[t])
            if nlp is None:
                if t == 0:
                    continue  # No BOS: position 0 has no distribution, stays NaN.
                model_name = outputs[t].metadata.get("model_name", "")
                raise FeatureNotAvailable(self._raise_name or self.name, model_name)
            vals[t] = self._compute(nlp)
        return vals


class _RequiredValueFeature:
    """Shared reader for a stored per-output value keyed by ``name``.

    A missing key raises FeatureNotAvailable, appending ``_hint`` (the
    constructor flag that enables the value) to the model name.
    ``_nan_at_start = True`` instead treats a missing position 0 as NaN.
    """

    def configuration(self) -> dict:
        """Return computation settings without the column alias."""
        return {"key": self.name}

    name: str
    _hint = ""
    _nan_at_start = False

    @property
    def required_outputs(self) -> tuple[str, ...]:
        """Model value keys consumed by this feature."""
        return (self.name,)

    def __call__(
        self,
        stimulus: Stimulus,
        outputs: Sequence[ModelOutputRecord] | None,
        target_layer: str = "words",
    ) -> np.ndarray:
        if outputs is None:
            raise ValueError(f"{type(self).__name__} requires model outputs")
        n = len(outputs)
        vals = np.full(n, np.nan)
        for t in range(n):
            v = outputs[t].values.get(self.name)
            if v is not None:
                vals[t] = float(v)
            elif t > 0 or not self._nan_at_start:
                # A missing value raises FeatureNotAvailable.
                model_name = outputs[t].metadata.get("model_name", "")
                raise FeatureNotAvailable(
                    self.name, model_name + self._hint, missing=self.name
                )
        return vals


class _StoredEstimateFeature:
    """Shared reader for a pre-computed estimate keyed by ``name``; positions
    without a stored value stay NaN. A key absent from every output raises
    FeatureNotAvailable, appending ``_hint`` (the model that writes it).
    """

    def configuration(self) -> dict:
        """Return computation settings without the column alias."""
        return {"key": self.name}

    name: str
    _hint = ""

    @property
    def required_outputs(self) -> tuple[str, ...]:
        """Model value keys consumed by this feature."""
        return (self.name,)

    def __call__(
        self,
        stimulus: Stimulus,
        outputs: Sequence[ModelOutputRecord] | None,
        target_layer: str = "words",
    ) -> np.ndarray:
        if outputs is None:
            raise ValueError(f"{type(self).__name__} requires model outputs")
        if outputs and not any(self.name in o.values for o in outputs):
            model_name = outputs[0].metadata.get("model_name", "")
            raise FeatureNotAvailable(
                self.name, model_name + self._hint, missing=self.name
            )
        n = len(outputs)
        vals = np.full(n, np.nan)
        for t in range(n):
            v = outputs[t].values.get(self.name)
            if v is not None:
                vals[t] = float(v)
        return vals


class Entropy(_NextLogProbsFeature):
    """Shannon entropy of the model's ``next_log_probs`` distribution in bits.

    ``Entropy[t]`` is read from ``outputs[t].next_log_probs``, the distribution
    at the position that predicts unit t. :class:`HuggingFaceLM` builds it by
    pooling the next-token distribution over token ids with the same
    lower-cased, stripped surface form, so case and leading-space variants
    merge and word pieces such as ``ing`` count as types; for the entropy over
    whole words use :class:`MCWordEntropy`. Position 0 is NaN when no BOS is
    used (the model has no distribution predicting the first unit).

    Raises FeatureNotAvailable if next_log_probs is None at a position
    where it should be available (i.e. positions after the first).
    """

    name = "entropy"
    measurement = FeatureMeasurement(
        "entropy", "bits", {"order": 1, "distribution": "next_log_probs"}
    )

    def _compute(self, log_probs: np.ndarray) -> float:
        return shannon_bits(log_probs)


class ExpectedProbability(_NextLogProbsFeature):
    """Expected probability under the model's own distribution: E_{v~p}[p(v)] = Σ p(v)².

    Equal to 2^(-H₂) for H₂ in bits (Rényi-2 / collision entropy): high when the
    distribution is peaked, low when flat. Anticipation-strength measure
    from Giulianelli et al. (2024a), *Generalized Measures of
    Anticipation and Responsivity*.

    ``ExpectedProbability[t]`` reads ``outputs[t].next_log_probs``, the pooled
    first-subword distribution described on :class:`Entropy`. Position
    0 is NaN when no BOS is used. Raises FeatureNotAvailable if
    next_log_probs is None at a position where it should be available.
    """

    name = "expected_probability"
    measurement = FeatureMeasurement(
        "expected_probability", "dimensionless", {"distribution": "next_log_probs"}
    )

    def _compute(self, log_probs: np.ndarray) -> float:
        return collision_prob(log_probs)


class TokenEntropy(_RequiredValueFeature):
    """Shannon entropy over the full token vocabulary.

    Requires that the FeatureModel stores ``token_entropy`` in its outputs
    (e.g. ``HuggingFaceLM(token_level_entropy=True)``).

    This matches the entropy computation in Pimentel et al. (2023) and
    other papers that compute entropy at the subword token level. For
    multi-token words, uses the *first* subword token's entropy.

    Compare with ``Entropy``, which computes entropy over the marginalised
    word-type distribution (smaller vocabulary, different quantity).
    """

    name = "token_entropy"
    measurement = FeatureMeasurement(
        "entropy", "bits", {"order": 1, "distribution": "first_token"}
    )
    _hint = " (set token_level_entropy=True)"


class TokenEntropySum(_RequiredValueFeature):
    """Buggy Pimentel et al. (2023) word entropy: sum of per-subtoken token entropies.

    Reproduces the paper's released ``agg_entropy`` mis-routed through the
    summing aggregator: for a multi-subtoken word, the word's value is the
    sum of its subtokens' Shannon entropies rather than the first subtoken's
    (the corrected :class:`TokenEntropy`). Since each term is non-negative,
    this is ≥ the first-token value, with strict inequality only when a
    later subtoken has positive entropy.

    Requires ``HuggingFaceLM(..., token_entropy_sum=True,
    allow_reproduction=True)``, which stores the summed value under
    ``token_entropy_sum``.
    """

    name = "token_entropy_sum"
    measurement = FeatureMeasurement(
        "summed_token_entropy", "bits", {"order": 1, "reproduction_only": True}
    )
    _hint = " (set token_entropy_sum=True)"

    def __init__(self, *, allow_reproduction: bool = False) -> None:
        """Require reproduction opt-in and warn about the incorrect quantity.

        Args:
            allow_reproduction: Explicit acknowledgement of the estimator's errors.
        """
        warn_reproduction_only(
            "first_token_entropy_buggy", allow_reproduction=allow_reproduction
        )


class TokenRenyiEntropySum(_RequiredValueFeature):
    """Buggy Pimentel et al. (2023) word Rényi entropy: sum of per-subtoken Rényi values.

    The Rényi-alpha analogue of :class:`TokenEntropySum`: for a multi-subtoken
    word, the word's value is the sum of its subtokens' Rényi-alpha entropies
    rather than the first subtoken's (the corrected :class:`TokenRenyiEntropy`).
    Since each Rényi term is non-negative, this is >= the first-token value,
    with strict inequality only when a later subtoken has positive entropy.

    Requires ``HuggingFaceLM(..., token_renyi_alphas=[alpha],
    token_renyi_entropy_sum=True, allow_reproduction=True)``, which stores the
    summed value under ``token_renyi_entropy_sum`` (using the first configured alpha).
    """

    name = "token_renyi_entropy_sum"
    measurement = FeatureMeasurement(
        "summed_token_entropy",
        "bits",
        {"source_key": "token_renyi_entropy_sum", "reproduction_only": True},
    )
    _hint = " (set token_renyi_alphas=[alpha, ...] and token_renyi_entropy_sum=True)"

    def __init__(self, *, allow_reproduction: bool = False) -> None:
        """Require reproduction opt-in and warn about the incorrect quantity.

        Args:
            allow_reproduction: Explicit acknowledgement of the estimator's errors.
        """
        warn_reproduction_only(
            "renyi_entropy_buggy", allow_reproduction=allow_reproduction
        )


class TokenRenyiEntropy(_RequiredValueFeature):
    """Subword Rényi-alpha entropy at the LM's first-subword distribution.

    This is the form used by Pimentel et al. (2023), on the *raw subword
    distribution* (no word-type marginalisation).

    Reads ``outputs[t].values["token_renyi_entropy_alpha{alpha}"]``,
    populated by :class:`HuggingFaceLM` when constructed with
    ``token_renyi_alphas=[alpha, ...]``. Position 0 is NaN unless the
    LM was built with ``add_bos=True``.

    Compare with:
      * :class:`RenyiEntropy`, same formula but on the *word-type*
        marginalised distribution (smaller vocab; different quantity).
      * :class:`TokenEntropy`, Shannon limit (alpha = 1) of this feature.

    Args:
        alpha: Pimentel et al. (2023) use ``0.5``. ``1.0`` reduces to Shannon
            (≡ :class:`TokenEntropy`); ``2.0`` is subword collision entropy.
    """

    _nan_at_start = True

    def __init__(self, alpha: float = 0.5):
        """Bind the feature to one Rényi order, which names the value it reads."""
        if not np.isfinite(alpha) or alpha < 0:
            raise ValueError(f"Rényi alpha must be finite and nonnegative, got {alpha}")
        self.alpha = float(alpha)
        self.name = f"token_renyi_entropy_alpha{self.alpha}"
        self.measurement = FeatureMeasurement(
            "entropy", "bits", {"order": self.alpha, "distribution": "first_token"}
        )
        self._hint = f" (set token_renyi_alphas=[{self.alpha}, ...])"


class RenyiEntropy(_NextLogProbsFeature):
    """Rényi-alpha entropy of the model's ``next_log_probs`` distribution in bits.

    H_alpha(W_t | ctx) = (1 / (1 - alpha)) · log₂ Σ_w P(w | ctx)^alpha

    Closed-form over the pooled first-subword distribution described on
    :class:`Entropy`. The ``alpha=0.5`` result of Pimentel et al. (2023) is
    for the subword distribution, :class:`TokenRenyiEntropy`. Reads
    ``outputs[t].next_log_probs``. Position 0 is NaN when no BOS is
    used. Equivalent to :class:`Entropy` at alpha = 1 (in the limit via
    L'Hôpital); to ``-log₂ Σ P(w)²`` (collision entropy) at alpha = 2. At
    alpha = 0 it returns the log-base-two number of positive-probability outcomes.

    Args:
        alpha: Finite nonnegative order: 0.5 (Pimentel et al., 2023), 1.0 (Shannon ≡
            :class:`Entropy`), 2.0 (collision entropy = -log of
            :class:`ExpectedProbability`).
    """

    def configuration(self) -> dict:
        """Return computation settings without the column alias."""
        return {"alpha": self.alpha}

    _raise_name = "renyi_entropy"

    def __init__(self, alpha: float = 0.5):
        """Bind the feature to one Rényi order, which its name records."""
        if not np.isfinite(alpha) or alpha < 0:
            raise ValueError(f"Rényi alpha must be finite and nonnegative, got {alpha}")
        # alpha=1 is the Shannon limit; renyi_bits delegates that case to shannon_bits.
        self.alpha = float(alpha)
        self.name = f"renyi_entropy_alpha{self.alpha}"

    @property
    def measurement(self) -> FeatureMeasurement:
        """Report the selected order and predictive distribution."""
        return FeatureMeasurement(
            "entropy", "bits", {"order": self.alpha, "distribution": "next_log_probs"}
        )

    def _compute(self, log_probs: np.ndarray) -> float:
        return renyi_bits(log_probs, self.alpha)


class SampledEntropy:
    """Estimate Rényi entropy in bits from retained Monte Carlo samples.

    Reads ``log_p_samples_nats``: one vector of log p(X) per unit, with X
    drawn from p. Word-boundary and fixed-length continuation samples define
    different distributions. This estimator does not correct proposal or
    truncation mismatches; the source sampler must satisfy that contract.
    Empty vectors denote undefined positions and return NaN. Missing keys
    raise :class:`FeatureNotAvailable`.
    """

    def configuration(self) -> dict:
        """Return computation settings without the column alias."""
        return {"alpha": self.alpha}

    required_outputs = ("log_p_samples_nats",)

    def __init__(self, alpha: float = 1.0):
        """Select the entropy order without modifying the stored samples.

        Args:
            alpha: Finite nonnegative Rényi order; 1.0 estimates Shannon entropy.
        """
        if not np.isfinite(alpha) or alpha < 0:
            raise ValueError(f"Rényi alpha must be finite and nonnegative, got {alpha}")
        self.alpha = float(alpha)
        self.name = f"sampled_entropy_alpha{self.alpha}"

    @property
    def measurement(self) -> FeatureMeasurement:
        """Report the selected order and retained-sample input."""
        return FeatureMeasurement(
            "entropy", "bits", {"order": self.alpha, "source_key": "log_p_samples_nats"}
        )

    def __call__(
        self,
        stimulus: Stimulus,
        outputs: Sequence[ModelOutputRecord] | None,
        target_layer: str = "words",
    ) -> np.ndarray:
        """Aggregate each unit's sample vector using the entropy backend's kernel.

        Args:
            stimulus: Stimulus whose units the outputs describe.
            outputs: Per-unit records containing retained sample log probabilities.
            target_layer: Annotation layer to which the records already align.
        """
        if outputs is None:
            raise ValueError("SampledEntropy requires model outputs")
        values = np.full(len(outputs), np.nan)
        for index, output in enumerate(outputs):
            model_name = output.metadata.get("model_name", "")
            if "continuation_entropy_buggy" in output.values or (
                "cont_entropy_buggy" in model_name
            ):
                raise ValueError(
                    "SampledEntropy cannot use continuation_entropy_buggy samples: "
                    "their sampling and log probabilities do not match. "
                    "Use samples from ContinuationEntropyLM."
                )
            samples = output.values.get("log_p_samples_nats")
            if samples is None:
                raise FeatureNotAvailable(
                    self.name, model_name, missing="log_p_samples_nats"
                )
            try:
                values[index] = _renyi_entropy_bits_from_log_p_nats(samples, self.alpha)
            except ValueError as exc:
                raise ValueError(f"{exc} (unit {index})") from exc
        return values


class ContinuationEntropy(_StoredEstimateFeature):
    """Sequence-level entropy via Monte Carlo over continuations.

    Reference: Giulianelli et al. (2024a), "Generalized Measures of
    Anticipation and Responsivity in Online Language Processing", the
    "Entropy" special case of generalized surprisal:

        H_cont(ctx) = - E_{c ~ p_LM(·|ctx)} [log p_LM(c | ctx)]
                    ≈ -(1/N) Σᵢ log p_LM(c⁽ⁱ⁾ | ctx)

    where each ``c⁽ⁱ⁾`` is a sampled continuation of up to L tokens.
    Reads the pre-computed estimate from
    ``outputs[t].values["continuation_entropy"]`` (produced by
    :class:`ContinuationEntropyLM`). Position 0 is NaN when no BOS is
    used.
    """

    name = "continuation_entropy"
    measurement = FeatureMeasurement(
        "sampled_continuation_entropy", "bits", {"source_key": "continuation_entropy"}
    )
    _hint = " (use ContinuationEntropyLM)"


class ContinuationEntropyBuggy(_StoredEstimateFeature):
    """Sequence-level entropy as the responsivity paper's ``measures.py`` computes it.

    Reads the pre-computed estimate from
    ``outputs[t].values["continuation_entropy_buggy"]`` (produced by
    :class:`ContinuationEntropyLMBuggy`), which keeps the paper's inherited
    ``top_k=50`` sampling and its reshape in place of a transpose, unlike the
    corrected :class:`ContinuationEntropy`. Position 0 is NaN when
    no BOS is used.
    """

    name = "continuation_entropy_buggy"
    measurement = FeatureMeasurement(
        "continuation_entropy_buggy", "bits", {"reproduction_only": True}
    )
    _hint = " (use ContinuationEntropyLMBuggy)"

    def __init__(self, *, allow_reproduction: bool = False) -> None:
        """Require reproduction opt-in and warn about the incorrect quantity.

        Args:
            allow_reproduction: Explicit acknowledgement of the estimator's errors.
        """
        warn_reproduction_only(
            "continuation_entropy_buggy", allow_reproduction=allow_reproduction
        )


class MCWordEntropy(_StoredEstimateFeature):
    """Word-level entropy via Monte Carlo with word-boundary stopping.

    Reference: Clark et al. (2025), "How Well Does First-Token
    Entropy Approximate Word Entropy as a Psycholinguistic Predictor?".
    For each word position, |S| samples are drawn from the LM, each
    extended subword-by-subword until a word boundary (a whitespace-
    prefixed token in GPT-2 BPE) is sampled. The estimate is

        H_word(W_t | ctx) ≈ -(1/|S|) Σ_s log₂ P(s | ctx)

    or for Rényi alpha ≠ 1,

        H_alpha ≈ (1/(1-alpha)) · log₂ ( (1/|S|) Σ_s P(s | ctx)^(alpha-1) ).

    Reads the pre-computed estimate from
    ``outputs[t].values["mc_word_entropy"]`` (produced by
    :class:`MCWordEntropyLM`). Position 0 is NaN when no BOS is used.

    Differs from :class:`ContinuationEntropy`, whose samples have a fixed
    length: each MC sample here is exactly one word, ending at the next
    whitespace.
    """

    name = "mc_word_entropy"
    measurement = FeatureMeasurement(
        "sampled_word_entropy", "bits", {"source_key": "mc_word_entropy"}
    )
    _hint = " (use MCWordEntropyLM)"
