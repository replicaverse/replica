"""Storage classes, per-token predictor tables and their guarded derivation.

Every compute method belongs to one of two storage classes. A ``positional``
estimator conditions on a prefix ending at a token position and returns a
value for that position; its cell is stored per token when the method is
listed in :data:`replica.compute.cache.TOKEN_SENSITIVE`, and any coarser unit
then derives from the token table through
:func:`~replica.feature.core.annotation.alignment.nested_members`, which raises on any
token that would have to be split, dropped or interpolated. A ``unit_bound``
estimator takes the unit as an input, so its cell is stored at the unit it
was computed for, and the manifest names the boundary rule the estimator
used.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from replica.compute.cache import TOKEN_SENSITIVE, UNIT_SENSITIVE
from replica.core import Span, Stimulus
from replica.feature.core.annotation.alignment import nested_members
from replica.feature.core.annotation.characters import CHARACTER_LAYER
from replica.feature.core.outputs.information import LN2
from replica.feature.core.outputs.units import unit_for
from replica.feature.core.protocols import FeatureModelOutput
from replica.feature.distance.similarity_surprisal import similarity_unit_values

TOKEN_UNIT = "tokens"

POSITIONAL = "positional"
UNIT_BOUND = "unit_bound"


@dataclass(frozen=True)
class MethodStorage:
    """How a method's cells are stored.

    Layer 1 uses library conventions. Layer 2 records a study-associated
    computation, including faithful upstream adapters and defect reproductions.
    The layer is provenance, not a correctness classification; known-incorrect
    methods have separate explicit reproduction guards.
    """

    storage_class: str
    layer: int = 1
    study: str | None = None


#: Storage class, layer and (for layer 2) the study of every method the
#: compute driver accepts.
METHOD_STORAGE: dict[str, MethodStorage] = {
    "surprisal": MethodStorage(POSITIONAL),
    "char_beam": MethodStorage(POSITIONAL),
    "modelblocks_surprisal": MethodStorage(UNIT_BOUND, 2, "clark2025wordentropy"),
    "wordsprobability_surprisal": MethodStorage(UNIT_BOUND),
    "wordsprobability_surprisal_buggy": MethodStorage(
        POSITIONAL, 2, "pimentel2024probability"
    ),
    "first_token_entropy": MethodStorage(POSITIONAL),
    "first_token_entropy_buggy": MethodStorage(
        POSITIONAL, 2, "pimentel2023anticipation"
    ),
    "token_renyi_entropy": MethodStorage(POSITIONAL),
    "renyi_entropy_buggy": MethodStorage(POSITIONAL, 2, "pimentel2023anticipation"),
    "logit_lens_surprisal": MethodStorage(POSITIONAL),
    "tuned_lens_surprisal": MethodStorage(POSITIONAL),
    "restricted_first_token_entropy": MethodStorage(UNIT_BOUND),
    "restricted_first_token_renyi_entropy": MethodStorage(UNIT_BOUND),
    "continuation_entropy": MethodStorage(POSITIONAL),
    "continuation_entropy_buggy": MethodStorage(
        POSITIONAL, 2, "giulianelli2024generalized"
    ),
    "forward_marginal_entropy": MethodStorage(POSITIONAL),
    "forward_word_lookahead_entropy": MethodStorage(UNIT_BOUND),
    "similarity_surprisal_noncontextual": MethodStorage(POSITIONAL),
    "similarity_surprisal_contextual": MethodStorage(POSITIONAL),
    "similarity_surprisal_word": MethodStorage(UNIT_BOUND),
    "word_information_value": MethodStorage(UNIT_BOUND),
    "information_value": MethodStorage(UNIT_BOUND),
    "responsivity": MethodStorage(POSITIONAL),
    "mc_word_entropy": MethodStorage(UNIT_BOUND),
    "unigram_surprisal": MethodStorage(UNIT_BOUND),
    "mmc_unigram_surprisal": MethodStorage(UNIT_BOUND),
}

#: Storage class of every method, read off :data:`METHOD_STORAGE`.
STORAGE_CLASSES: dict[str, str] = {
    method: storage.storage_class for method, storage in METHOD_STORAGE.items()
}

#: Model-word boundary: tokens continue until the next whitespace-initial token.
DEFAULT_UNIT_RULE = "whitespace_continuation"

#: Boundaries supplied by corpus segments or a sentence layer.
UNIT_RULES: dict[str, str] = {
    "unigram_surprisal": "corpus_segment_lookup",
    "information_value": "spacy_sentence",
}

#: Layer whose units a unit-bound estimator's rule binds, where it is not the
#: word layer.
BOUND_LAYERS: dict[str, str] = {"information_value": "sentences"}


def storage_class(method: str) -> str:
    """Storage class of *method*.

    Args:
        method: Compute method name.

    Raises:
        KeyError: When the method has no class.
    """
    return STORAGE_CLASSES[method]


def bound_layer(method: str) -> str | None:
    """Layer whose units the rule of a unit-bound *method* binds.

    Args:
        method: Compute method name.

    Returns:
        The layer name, or ``None`` for a positional method.
    """
    if STORAGE_CLASSES[method] == POSITIONAL:
        return None
    return BOUND_LAYERS.get(method, "words")


def boundary_rule(method: str, requested: str | None) -> str | None:
    """Boundary rule recorded for a cell of *method*.

    Args:
        method: Compute method name.
        requested: Rule name the caller passed, or ``None`` for the default.

    Returns:
        The rule for a unit-bound method, ``None`` for a positional one.

    Raises:
        ValueError: When a rule is named for a positional method.
    """
    if STORAGE_CLASSES[method] == POSITIONAL:
        if requested is not None:
            raise ValueError(
                f"{method} is positional and takes no unit rule; "
                "the unit is chosen when the token table is aggregated"
            )
        return None
    return requested or UNIT_RULES.get(method, DEFAULT_UNIT_RULE)


#: How a positional method's per-token values combine into one unit: the
#: log probabilities of the unit's tokens or characters sum; the entropies
#: and distances take the value at the unit's first token.
AGGREGATION_RULES: dict[str, str] = {
    "surprisal": "sum",
    "first_token_entropy": "first",
    "token_renyi_entropy": "first",
    "char_beam": "sum",
    "logit_lens_surprisal": "sum",
    "tuned_lens_surprisal": "sum",
    "continuation_entropy": "first",
    "responsivity": "first",
    "similarity_surprisal_noncontextual": "sum",
    "similarity_surprisal_contextual": "sum",
}

#: Methods that score the words joined by single spaces; their token spans
#: cover the word spans only.
WORD_JOINED = frozenset({"logit_lens_surprisal", "tuned_lens_surprisal"})

#: How a word-level estimator values a unit of several whitespace words.
#: ``sum``: log probabilities add by the chain rule. ``unit``: the estimator
#: takes the unit as its input (sampled to its boundary, priced as one
#: surface, or scored under ``segmentation=layer``). ``first``: a first-token
#: measure, read at the unit's first token. A method absent here is refused
#: on such a corpus unless ``--unit-rule`` names a rule.
MULTIWORD_RULES: dict[str, str] = {
    "unigram_surprisal": "sum",
    "wordsprobability_surprisal": "sum",
    "wordsprobability_surprisal_buggy": "sum",
    "modelblocks_surprisal": "sum",
    "mc_word_entropy": "unit",
    "forward_word_lookahead_entropy": "unit",
    "mmc_unigram_surprisal": "unit",
    "word_information_value": "unit",
    "restricted_first_token_entropy": "first",
    "restricted_first_token_renyi_entropy": "first",
}


#: Statuses of :class:`Applicability`.
DEFINED, AGGREGATED, UNDEFINED = "defined", "aggregated", "undefined"


@dataclass(frozen=True)
class Applicability:
    """Whether a method's measure is defined on a corpus's units, and how.

    ``defined`` or ``aggregated`` (rule ``sum``), with ``rule`` and
    ``multiword_units`` where units of several whitespace words meet a
    :data:`MULTIWORD_RULES` entry; ``undefined`` with the ``reason``.
    """

    status: str
    reason: str | None = None
    rule: str | None = None
    multiword_units: int = 0

    @property
    def defined(self) -> bool:
        """Whether a cell can be computed."""
        return self.status != UNDEFINED


def applicability(
    method: str,
    unit: str,
    stimuli: list[Stimulus],
    *,
    word_spacing: bool = True,
    unit_rule: str | None = None,
    segmentation: str = "whitespace",
) -> Applicability:
    """The contract between *method*'s measure and a corpus's *unit* segmentation.

    A grain needs a method that scores it; a token method's word cell needs
    word spacing (``surprisal`` derives from characters); a whitespace-bound
    method needs word spacing and single-word units, or a multi-word rule;
    lexical lookup uses the supplied corpus segments; a sentence-bound method
    needs the ``sentences`` layer; information value
    needs the segmentation that fits the units. A named *unit_rule* lifts the
    whitespace conditions.

    Args:
        method: Compute method name.
        unit: ``words``, ``characters``, ``tokens`` or an attached segmentation.
        stimuli: The corpus, with the layer *unit* names.
        word_spacing: The loader's word spacing flag.
        unit_rule: Boundary rule the caller names, or ``None`` for the default.
        segmentation: Word iteration of ``word_information_value``.

    Raises:
        KeyError: When the method has no storage class.
    """
    storage_class(method)
    grain = _grain_fit(method, unit, word_spacing)
    if grain is not None:
        return grain
    # A positional method with a multi-word rule scores whitespace words.
    layer = bound_layer(method) or "words"
    if layer != "words" and unit != "words":
        return Applicability(
            UNDEFINED,
            f"{method} is bound to the units of the {layer} layer; --unit {unit} "
            "names another segmentation",
        )
    checked = unit if layer == "words" else layer
    offending, total, multiword = malformed_units(stimuli, checked)
    rule = MULTIWORD_RULES.get(method)
    if rule is not None:
        offending = [text for text in offending if not text.strip()]
    reasons = []
    boundary = UNIT_RULES.get(method, DEFAULT_UNIT_RULE)
    if checked == unit and not word_spacing and boundary == DEFAULT_UNIT_RULE:
        reasons.append("its loader declares no word spacing")
    if offending:
        examples = ", ".join(repr(text) for text in offending[:3])
        reasons.append(
            f"{len(offending)} of {total} {checked} units {_defect(checked, offending)} "
            f"(for example {examples})"
        )
    if reasons and unit_rule is None:
        advice = (
            ""
            if storage_class(method) == POSITIONAL
            else "; pass --unit-rule NAME to name the boundary rule the cell is computed under"
        )
        return Applicability(
            UNDEFINED,
            f"{method} is bound to the units of the {checked} layer under the rule "
            f"{boundary}, but {' and '.join(reasons)}" + advice,
        )
    if method == "word_information_value":
        recipe = _infovalue_fit(multiword, segmentation, word_spacing)
        if recipe is not None:
            return recipe
    return _bound(rule, multiword)


def _grain_fit(method: str, unit: str, word_spacing: bool) -> Applicability | None:
    """The contract for the token and character grains and the positional word cell.

    Returns:
        ``None`` when the caller must check the method's input units.
    """
    scorers = {
        CHARACTER_LAYER: ("character", UNIT_SENSITIVE),
        TOKEN_UNIT: ("token", TOKEN_SENSITIVE),
    }
    if unit in scorers:
        grain, methods = scorers[unit]
        if method in methods:
            return Applicability(DEFINED)
        return Applicability(
            UNDEFINED,
            f"--unit {unit} is not supported for method {method!r}; "
            f"{grain}-unit methods: {sorted(methods)}",
        )
    if bound_layer(method) is not None or method in MULTIWORD_RULES:
        return None
    # A token method's word cell derives from its token table; the other
    # positional methods score the layer they are given.
    if unit != "words" or word_spacing or method not in TOKEN_SENSITIVE:
        return Applicability(DEFINED)
    if method == "surprisal":
        return Applicability(DEFINED)
    return Applicability(
        UNDEFINED,
        f"the corpus has no word spacing, so {method} cannot be aggregated to "
        "words from tokens and has no character-level scorer; store the token "
        f"table with --unit {TOKEN_UNIT}",
    )


def _bound(rule: str | None, multiword: int) -> Applicability:
    """A defined cell, with its rule where units of several words meet one."""
    if rule is None or not multiword:
        return Applicability(DEFINED)
    status = AGGREGATED if rule == "sum" else DEFINED
    return Applicability(status, rule=rule, multiword_units=multiword)


def _infovalue_fit(
    multiword: int, segmentation: str, word_spacing: bool
) -> Applicability | None:
    """Where the information-value segmentation and the corpus's units disagree.

    Returns:
        ``None`` when they agree.
    """
    if multiword and segmentation != "layer":
        return Applicability(
            UNDEFINED,
            "word_information_value iterates whitespace words under the reference "
            f"recipe, but {multiword} units hold several; pass --segmentation "
            "layer to score the units",
        )
    if segmentation == "layer" and word_spacing and not multiword:
        return Applicability(
            UNDEFINED,
            "every unit is one whitespace word, so the reference whitespace "
            "recipe applies; drop --segmentation layer",
        )
    return None


def _defect(layer: str, offending: list[str]) -> str:
    """What is wrong with the *offending* units of *layer*."""
    if layer in BOUND_LAYERS.values():
        return "are malformed"
    empty = [text for text in offending if not text.strip()]
    if len(empty) == len(offending):
        return "are empty"
    if not empty:
        return "contain whitespace"
    return "are empty or contain whitespace"


def malformed_units(stimuli: list[Stimulus], layer: str) -> tuple[list[str], int, int]:
    """Malformed units of *layer* (empty, holding whitespace, or a broken sentence run), the total, and the multi-word count.

    Args:
        stimuli: The corpus, carrying *layer*.
        layer: ``sentences``, or a layer of text units.
    """
    offending: list[str] = []
    total = 0
    multiword = 0
    for stimulus in stimuli:
        if layer not in BOUND_LAYERS.values():
            for text in stimulus.unit_strings(layer):
                total += 1
                if not text or any(character.isspace() for character in text):
                    offending.append(text)
                if len(text.split()) > 1:
                    multiword += 1
            continue
        ids = getattr(stimulus.layers.get(layer), "values", None)
        if ids is None:
            total += 1
            if not stimulus.text.strip():
                offending.append(f"{stimulus.stimulus_id}: blank text")
            continue
        seen: list = []
        for identifier in ids:
            if identifier is None:
                offending.append(f"{stimulus.stimulus_id}: word without a sentence id")
            elif not seen or identifier != seen[-1]:
                if identifier in seen:
                    offending.append(
                        f"{stimulus.stimulus_id}: sentence {identifier!r} is split"
                    )
                seen.append(identifier)
        total += len(seen)
    return offending, total, multiword


#: Array-valued keys derived by stacking the unit's per-token rows.
STACKED_KEYS = frozenset({"expected_similarity_tokens", "similarity_samples_tokens"})


def _surprisal_bits(values: dict, _metadata: dict) -> dict:
    """Add the positive bits form of a summed log probability."""
    return {**values, "surprisal": float(-values["log_p_observed"] / LN2)}


def _similarity_unit(values: dict, metadata: dict) -> dict:
    """The four similarity scalars of a unit from its stacked expectations."""
    scalars = similarity_unit_values(
        values["expected_similarity_tokens"], tuple(metadata["alphas"])
    )
    return {**scalars, **values}


#: Finishing steps of a derived unit record, by name; each takes the carried
#: values and the record metadata.
FINISHERS: dict[str, Callable[[dict, dict], dict]] = {
    "surprisal_bits": _surprisal_bits,
    "similarity_unit_values": _similarity_unit,
}


def token_keys(keys: list[str]) -> list[str]:
    """Value keys a token table holds out of a method's extracted keys.

    The first-subword variants a unit record carries are unit-level keys, so
    a token table leaves them out and the derivation adds them back.

    Args:
        keys: Value keys the method extracts for a unit record.
    """
    return [key for key in keys if not key.endswith("_first")]


def derivation_plan(method: str, keys: list[str]) -> tuple[dict[str, str], str | None]:
    """How the unit record of *method* derives from its per-token keys.

    Args:
        method: Positional method the token table belongs to.
        keys: Value keys the token table stores.

    Returns:
        The rule of every key that is carried into the unit record (``sum``
        or ``first`` for a scalar, ``first`` or ``stack`` for an array) and
        the name of the finisher in :data:`FINISHERS` that computes the
        remaining keys from them, or ``None``.
    """
    rule = AGGREGATION_RULES[method]
    if method in (
        "similarity_surprisal_noncontextual",
        "similarity_surprisal_contextual",
    ):
        return {
            key: "stack" for key in keys if key in STACKED_KEYS
        }, "similarity_unit_values"
    plan = {key: rule for key in keys}
    for key in keys:
        if key == "log_p_samples_nats":
            plan[key] = "first"
    return plan, "surprisal_bits" if keys == ["log_p_observed"] else None


def derived_keys(method: str, keys: list[str]) -> list[str]:
    """Value keys of the unit table derived from a token table with *keys*.

    Args:
        method: Positional method the token table belongs to.
        keys: Value keys the token table stores.
    """
    plan, finisher = derivation_plan(method, keys)
    derived = list(plan)
    if finisher == "surprisal_bits":
        derived.append("surprisal")
    if finisher == "similarity_unit_values":
        derived = [
            "similarity_adjusted_surprisal",
            "next_word_information_value",
            "similarity_adjusted_surprisal_first",
            "next_word_information_value_first",
            *derived,
        ]
    return derived


def value_unit(key: str) -> str:
    """Unit of a stored value key, from the feature-level unit registry.

    Args:
        key: Value key of a token or derived unit record.
    """
    return unit_for(key)


def value_units(keys: list[str] | tuple[str, ...]) -> dict[str, str]:
    """Unit of every key, for the manifest identity.

    Args:
        keys: Value keys the table stores.
    """
    return {key: value_unit(key) for key in keys}


def verify_token_rows(
    stimulus: Stimulus, rows: list[FeatureModelOutput], layer: str | None = None
) -> None:
    """Check that token records tile the scored text of *stimulus* in order.

    Args:
        stimulus: Text the records were scored on.
        rows: Token records in text order.
        layer: When given, only the characters inside this layer's spans have
            to be covered, for a method that scores the units alone.

    A combining mark (Unicode categories Mn, Mc and Me) counts as covered
    when the character it attaches to is covered: tokenizers normalise a
    decomposed letter to its composed form, so the mark has no token of its
    own, and the token of its base letter stands for both.

    Raises:
        ValueError: A record lacks its index or span, a span steps backwards
            or leaves the text, or a non-whitespace character that has to be
            covered is covered by no token.
    """
    text = stimulus.text
    covered = np.zeros(len(text), dtype=bool)
    previous = 0
    for index, row in enumerate(rows):
        span = row.metadata.get("span")
        if span is None or row.metadata.get("token_index") != index:
            raise ValueError(
                f"{stimulus.stimulus_id}: token record {index} lacks its index or span"
            )
        start, end = span
        if start < previous or end <= start or end > len(text):
            raise ValueError(
                f"{stimulus.stimulus_id}: token record {index} span [{start}, {end}) "
                "is out of order or out of range"
            )
        covered[start:end] = True
        previous = start
    exempt = np.array([character.isspace() for character in text], dtype=bool)
    for index, character in enumerate(text):
        if index and unicodedata.category(character)[0] == "M" and covered[index - 1]:
            exempt[index] = True
    if layer is not None:
        inside = np.zeros(len(text), dtype=bool)
        for span in stimulus.layers[layer].spans:
            inside[span.start : span.end] = True
        exempt |= ~inside
    missing = np.flatnonzero(~covered & ~exempt)
    if len(missing):
        raise ValueError(
            f"{stimulus.stimulus_id}: characters at {missing[:5].tolist()} are "
            "covered by no token"
        )


def _record_spans(
    stimulus: Stimulus, rows: list[FeatureModelOutput], unit: str
) -> tuple[list[Span], list[str] | None]:
    if unit == TOKEN_UNIT:
        return (
            [Span(*row.metadata["span"]) for row in rows],
            [str(row.metadata.get("token")) for row in rows],
        )
    if unit == CHARACTER_LAYER:
        spans = list(stimulus.layers[CHARACTER_LAYER].spans)
        if len(spans) != len(rows):
            raise ValueError(
                f"{stimulus.stimulus_id}: {len(rows)} character records for "
                f"{len(spans)} characters"
            )
        return spans, None
    raise ValueError(f"unknown source unit {unit!r}")


def _reduce(rows: list[FeatureModelOutput], indices: list[int], key: str, rule: str):
    if rule == "first":
        value = rows[indices[0]].values[key]
        return float(value) if np.ndim(value) == 0 else np.asarray(value)
    if rule == "stack":
        return np.stack([np.asarray(rows[i].values[key]) for i in indices])
    if rule == "sum":
        picked = np.array([rows[i].values[key] for i in indices], dtype=float)
        return float(np.sum(picked))
    raise ValueError(f"unknown aggregation rule {rule!r}")


def derive_records(
    stimulus: Stimulus,
    rows: list[FeatureModelOutput],
    *,
    unit: str,
    layer: str,
    plan: dict[str, str],
    finisher: str | None = None,
) -> list[FeatureModelOutput]:
    """Aggregate one stimulus's fine-grained records to the units of *layer*.

    Args:
        stimulus: Text carrying the target layer, and the ``characters`` layer
            when the records are per character.
        rows: Per-token records carrying their span, or per-character records
            in the order of the ``characters`` layer.
        unit: ``tokens`` or ``characters``, the grain of *rows*.
        layer: Target layer, such as ``words``.
        plan: Rule per carried key, see :func:`derivation_plan`. A scalar
            sums in text order or takes the unit's first record; an array
            takes the first record or stacks the unit's records.
        finisher: Name in :data:`FINISHERS` of the step that completes each
            unit record from the carried keys.

    Returns:
        One record per unit, with the model name of the source.
    """
    spans, labels = _record_spans(stimulus, rows, unit)
    members = nested_members(stimulus, layer, spans, labels)
    metadata = {
        name: rows[0].metadata[name]
        for name in ("model_name", "K", "context_scope", "alphas")
        if rows and name in rows[0].metadata
    }
    finish = FINISHERS[finisher] if finisher is not None else None
    outputs = []
    for indices in members:
        values = {key: _reduce(rows, indices, key, rule) for key, rule in plan.items()}
        if finish is not None:
            values = finish(values, metadata)
        outputs.append(FeatureModelOutput(values=values, metadata=dict(metadata)))
    return outputs


def derive_units(
    stimulus: Stimulus,
    rows: list[FeatureModelOutput],
    *,
    unit: str,
    layer: str,
    key: str,
    rule: str,
) -> list[FeatureModelOutput]:
    """Aggregate one scalar key of a stimulus's records to the units of *layer*.

    Args:
        stimulus: Text carrying the target layer, and the ``characters`` layer
            when the records are per character.
        rows: Per-token records carrying their span, or per-character records
            in the order of the ``characters`` layer.
        unit: ``tokens`` or ``characters``, the grain of *rows*.
        layer: Target layer, such as ``words``.
        key: Value key aggregated from each record.
        rule: ``sum`` or ``first``, see :data:`AGGREGATION_RULES`.

    Returns:
        One record per unit, with the model name of the source. A summed log
        probability is stored under *key* in nats and again as a positive
        ``surprisal`` in bits; any other key is stored alone.
    """
    return derive_records(
        stimulus,
        rows,
        unit=unit,
        layer=layer,
        plan={key: rule},
        finisher="surprisal_bits" if key == "log_p_observed" else None,
    )
