"""Measurement units for named model-output values.

Fixed keys and parameterized prefixes distinguish nats, bits, and dimensionless
values in predictor tables.
"""

import re
from math import isfinite

#: Value-key -> unit, for keys whose name is fixed.
VALUE_UNITS: dict[str, str] = {
    # Log-probabilities, natural log (negated and rescaled by the Surprisal
    # extractor, which is where bits first appear).
    "log_p_observed": "nats",
    "mean_token_log_p": "nats",
    "next_log_probs": "nats",
    "log_p_samples_nats": "nats",
    "fwd_word_log_p_step_nats": "nats",
    # The lens cells' per-layer log-probability vector, packed dense on the
    # store (index = layer, embedding layer first).
    "log_p_layers": "nats",
    # The positive bits form a derived word table keeps beside the summed
    # log probability of its tokens.
    "surprisal": "bits",
    # Closed-form entropies over an LM distribution.
    "token_entropy": "bits",
    "token_entropy_sum": "bits",
    "token_renyi_entropy_sum": "bits",
    "restricted_first_token_entropy": "bits",
    # Monte-Carlo entropy estimates.
    "continuation_entropy": "bits",
    "continuation_entropy_buggy": "bits",
    "mc_word_entropy": "bits",
    "forward_marginal_entropy": "bits",
    "free_step_entropy": "bits",
    "joint_entropy": "bits",
    "joint_word_entropy": "bits",
    "conditional_word_entropy": "bits",
    "marginal_word_entropy": "bits",
    "marginal_word_entropy_plugin": "bits",
    # Unigram baseline: the LM writes both bases from the same probability.
    "unigram_surprisal": "bits",
    # Similarity-adjusted surprisal is -log m for a similarity m in [0, 1].
    "similarity_adjusted_surprisal": "nats",
    "similarity_adjusted_surprisal_first": "nats",
    "similarity_adjusted_surprisal_pos": "nats",
    "similarity_adjusted_surprisal_orth": "nats",
    # Cosine similarities/distances and the 1 - m information values.
    "next_word_information_value": "dimensionless",
    "next_word_information_value_first": "dimensionless",
    "next_word_information_value_pos": "dimensionless",
    "next_word_information_value_orth": "dimensionless",
    "expected_similarity_tokens": "dimensionless",
    "similarity_samples_tokens": "dimensionless",
    "orth_similarity_samples": "dimensionless",
    "pos_match_samples": "dimensionless",
    "decontextualised_distance": "dimensionless",
    "expected_decontextualised_distance": "dimensionless",
    # Not measurements: identifiers and labels carried alongside the estimates.
    "fwd_word_id_hash": "identifier",
    "sampled_words": "text",
    "observed_pos": "text",
}

#: The distances information value is aggregated over. All are dimensionless
#: fractions or embedding distances except ``length``, a count of spaCy tokens.
INFORMATION_VALUE_DISTANCES: tuple[str, ...] = (
    "1gram",
    "2gram",
    "3gram",
    "1gram_pos",
    "2gram_pos",
    "3gram_pos",
    "cosine",
    "euclidean",
    "length",
)

#: One key per (family, distance). ``mean``/``min``/``max`` aggregate the
#: target-to-alternative distances, ``expected`` the alternative-to-alternative
#: ones, and the ``ooc`` families repeat both against alternatives sampled from
#: an empty context.
INFORMATION_VALUE_FAMILIES: tuple[str, ...] = (
    "mean",
    "min",
    "max",
    "expected",
    "deviation",
    "ooc_mean",
    "ooc_min",
    "ooc_max",
    "ooc_expected",
    "ooc_deviation",
    "context_informativeness",
    "expected_context_informativeness",
)

for _family in INFORMATION_VALUE_FAMILIES:
    for _distance in INFORMATION_VALUE_DISTANCES:
        VALUE_UNITS[f"information_value_{_family}_{_distance}"] = (
            "tokens" if _distance == "length" else "dimensionless"
        )
del _family, _distance

#: Prefix -> unit, for keys parameterised by alpha or layer index.
_PREFIX_UNITS: tuple[tuple[str, str], ...] = (
    ("token_renyi_entropy_alpha", "bits"),
    ("restricted_first_token_renyi_entropy_alpha", "bits"),
    ("renyi_entropy_alpha", "bits"),
    ("infovalue_layer", "dimensionless"),
    ("logit_lens_log_p_L", "nats"),
    ("tuned_lens_log_p_L", "nats"),
)

_SUFFIX_RE = re.compile(r"\d+(?:\.\d+)?")
_ALPHA_RE = re.compile(r"\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")


class UnknownValueKey(KeyError):
    """No unit is registered for a value key."""


def unit_for(key: str) -> str:
    """Return the measurement unit of the value stored under *key*.

    Args:
        key: A ``FeatureModelOutput.values`` key, either a fixed name listed in
            :data:`VALUE_UNITS` or an alpha- or layer-parameterised name matched
            on its stable prefix.

    Raises:
        UnknownValueKey: When no unit is registered for *key*; a new predictor
            declares its base here before anything can publish it.
    """
    if key in VALUE_UNITS:
        return VALUE_UNITS[key]
    for prefix, unit in _PREFIX_UNITS:
        if key.startswith(prefix):
            suffix = key[len(prefix) :]
            pattern = _ALPHA_RE if prefix.endswith("alpha") else _SUFFIX_RE
            if pattern.fullmatch(suffix) and isfinite(float(suffix)):
                return unit
    raise UnknownValueKey(key)
