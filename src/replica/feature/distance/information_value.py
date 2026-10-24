"""Information value from Giulianelli et al. (2023).

Measures how unexpected an observed utterance is via its distance from
plausible alternatives sampled from an LM::

    IV_agg^dist(target | context) = agg_{a ~ p_LM(·|context)} dist(target, a)

``agg`` ∈ {mean, min, max}; ``dist`` is lexical (token n-gram overlap),
syntactic (POS n-gram overlap), or semantic (sentence-embedding cosine /
euclidean), plus absolute length difference.

  * :class:`InformationValueScorer`, port of the authors'
    ``scorer.py`` + ``compute_information_value.py``, checked against their
    code on shared alternatives. The random-context variant is not implemented.
  * :class:`InformationValueLM`, :class:`FeatureModel` that samples
    alternatives natively (no precomputed CSV).

The measure is defined on an utterance, so each stimulus is segmented into
sentences and a sentence's value is broadcast to the units inside it.
"""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral
from typing import TYPE_CHECKING

import numpy as np

from replica.feature.backends.hub import hub_id
from replica.feature.core.measurement import FeatureMeasurement
from replica.feature.core.outputs.units import INFORMATION_VALUE_DISTANCES
from replica.feature.core.protocols import ModelOutputRecord

from ..backends.generation import NonFiniteLogitsError, usable_position_limit
from ..core.protocols import FeatureModelOutput

if TYPE_CHECKING:
    from ...core import Stimulus

_AGGREGATIONS = {"mean": np.mean, "min": np.min, "max": np.max}
_DEFAULT_EMBED_MODEL = "sentence-transformers/all-distilroberta-v1"


def _ngrams(tokens: list[str], n: int) -> list[tuple]:
    """n-grams as tuples (nltk.ngrams semantics, the authors' dependency)."""
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def _inverse_overlap(seq1: list[tuple], seq2: list[tuple]) -> float:
    """Symmetric inverse n-gram overlap = 1 - overlap fraction.

    Port of ``Scorer._inverse_overlap`` (``scorer.py``): counts membership
    both ways and normalises by the combined length. NaN when both empty.
    """
    count_1_in_2 = sum(1 for g in seq2 if g in seq1)
    count_2_in_1 = sum(1 for g in seq1 if g in seq2)
    combined = len(seq1) + len(seq2)
    if combined == 0:
        return float("nan")
    return (combined - (count_1_in_2 + count_2_in_1)) / combined


class InformationValueScorer:
    """Compute information value from a target and sampled alternatives.

    Faithful to ``compute_information_value.py::get_distances_fast`` followed
    by mean/min/max aggregation over the cross distances. ``score`` returns a
    dict keyed ``information_value_{agg}_{dist}`` for ``agg`` in
    {mean, min, max} and ``dist`` in
    {1gram, 2gram, 3gram, 1gram_pos, 2gram_pos, 3gram_pos, cosine,
    euclidean, length}.
    """

    def __init__(
        self,
        embed_model: str = _DEFAULT_EMBED_MODEL,
        spacy_model: str = "en_core_web_sm",
        device: str = "cpu",
        max_ngram: int = 3,
    ):
        """Configure the embedding and parsing models; both load on first use.

        Args:
            embed_model: Hugging Face model whose mean-pooled last hidden state
                gives the sentence embeddings the cosine and euclidean distances
                are taken over.
            spacy_model: spaCy pipeline supplying tokenisation, POS tags, and
                sentence boundaries.
            device: Torch device the embedding model runs on.
            max_ngram: Largest n for the token and POS n-gram overlap distances.
        """
        self.embed_model_name = embed_model
        self.spacy_model_name = spacy_model
        self.device = device
        self.max_ngram = int(max_ngram)
        self._nlp = None
        self._tokenizer = None
        self._model = None
        # Target-independent lexical features per alternative set. Scoring alternates
        # between a sentence's own alternatives and the shared out-of-context
        # ones, so two entries keep the shared set from being evicted every
        # sentence and recomputed for the whole corpus.
        self._feature_cache: dict[tuple[bool, tuple[str, ...]], dict] = {}

    def _ensure_loaded(self):
        if self._nlp is not None:
            return
        import spacy
        from transformers import AutoModel, AutoTokenizer

        self._nlp = spacy.load(self.spacy_model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(self.embed_model_name)
        self._model = (
            AutoModel.from_pretrained(self.embed_model_name).eval().to(self.device)
        )

    def _spacy_tokens(self, s: str) -> list:
        return list(self._nlp(s.strip()))

    def _embeddings(self, strings: Sequence[str]) -> np.ndarray:
        """Mean-pooled, L2-normalised sentence embeddings.

        Port of ``Scorer.compute_embeddings``: spaCy-detokenise each string,
        encode with the embedding tokenizer (padded), mean-pool the last
        hidden state under the attention mask, then L2-normalise.
        """
        import torch

        texts = [self._nlp(s.strip()).text for s in strings]
        enc = self._tokenizer(texts, padding=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self._model(**enc, output_hidden_states=True)
        tok_emb = out.hidden_states[-1]
        mask = enc["attention_mask"].unsqueeze(-1).expand(tok_emb.size()).float()
        pooled = (tok_emb * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        normed = torch.nn.functional.normalize(pooled, p=2, dim=1)
        return normed.cpu().numpy()

    def _features(self, strings: Sequence[str], *, include_self: bool = True) -> dict:
        """Lexical features of an alternative set, independent of the target.

        The out-of-context alternatives are shared by every target in a
        corpus, so their tokenisation and n-gram distances are reused. Semantic
        embeddings follow the authors' joint target-and-alternatives batch.
        """
        key = (include_self, tuple(strings))
        cached = self._feature_cache.get(key)
        if cached is not None:
            return cached

        tokens = [self._spacy_tokens(s) for s in strings]
        grams: dict[str, list] = {}
        pairs: dict[str, list[float]] = {}
        for n in range(1, self.max_ngram + 1):
            for pos, suffix in [(False, ""), (True, "_pos")]:
                per_string = [
                    _ngrams([t.pos_ if pos else t.text.lower() for t in toks], n)
                    for toks in tokens
                ]
                grams[f"{n}gram{suffix}"] = per_string
                if include_self:
                    pairs[f"{n}gram{suffix}"] = [
                        _inverse_overlap(per_string[i], per_string[j])
                        for i in range(len(per_string))
                        for j in range(i + 1, len(per_string))
                    ]

        lengths = [len(toks) for toks in tokens]
        if include_self:
            pairs["length"] = [
                float(abs(lengths[i] - lengths[j]))
                for i in range(len(lengths))
                for j in range(i + 1, len(lengths))
            ]
        features = {
            "grams": grams,
            "lengths": lengths,
            "pairs": pairs,
        }
        if len(self._feature_cache) >= 2:
            self._feature_cache.pop(next(iter(self._feature_cache)))
        self._feature_cache[key] = features
        return features

    def _distances(
        self,
        target: str,
        alternatives: Sequence[str],
        features: dict | None = None,
        *,
        include_self: bool = True,
    ) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
        """Target-to-alternative and alternative-to-alternative distances.

        Port of ``get_distances_fast``. The pair order is the authors' nested
        loop, ``alt1_idx < alt2_idx`` with ``alt1_idx`` outermost, because they
        truncate that list positionally and the order therefore decides which
        pairs survive.
        """
        from scipy.spatial.distance import cosine, euclidean

        if features is None:
            features = self._features(alternatives, include_self=include_self)
        cross: dict[str, list[float]] = {}
        tgt_tokens = self._spacy_tokens(target)

        for n in range(1, self.max_ngram + 1):
            for pos, suffix in [(False, ""), (True, "_pos")]:
                name = f"{n}gram{suffix}"
                tgt_grams = _ngrams(
                    [t.pos_ if pos else t.text.lower() for t in tgt_tokens], n
                )
                cross[name] = [
                    _inverse_overlap(tgt_grams, g) for g in features["grams"][name]
                ]

        tgt_len = len(tgt_tokens)
        cross["length"] = [float(abs(tgt_len - n)) for n in features["lengths"]]

        # Batch composition changes float32 attention arithmetic. Match
        # get_distances_fast's joint batch, including the target first.
        embeddings = self._embeddings([target, *alternatives])
        tgt_emb, alt_embs = embeddings[0], embeddings[1:]
        self_ = dict(features["pairs"])
        for name, fn in (("cosine", cosine), ("euclidean", euclidean)):
            cross[name] = [
                float(fn(alt_embs[i], tgt_emb)) for i in range(len(alt_embs))
            ]
            if include_self:
                self_[name] = [
                    float(fn(alt_embs[i], alt_embs[j]))
                    for i in range(len(alt_embs))
                    for j in range(i + 1, len(alt_embs))
                ]
        return cross, self_

    def score_in_context(
        self, target: str, alternatives: Sequence[str]
    ) -> dict[str, float]:
        """Score target-to-alternative distances without unused pairwise measures.

        The embedding batch and distance arithmetic match :meth:`score`.
        Omitting alternative-to-alternative distances avoids quadratic work
        when only Equation 1's in-context information value is needed.

        Args:
            target: Observed target utterance.
            alternatives: Alternative utterances sampled from the same context.
        """
        self._ensure_loaded()
        if not len(alternatives):
            raise ValueError("Information value requires at least one alternative")
        cross, _ = self._distances(target, alternatives, include_self=False)
        return {
            f"information_value_{agg}_{dist}": float(
                fn(np.asarray(values, dtype=float))
            )
            for dist, values in cross.items()
            for agg, fn in _AGGREGATIONS.items()
        }

    def score(
        self,
        target: str,
        alternatives: Sequence[str],
        ooc_alternatives: Sequence[str] | None = None,
        distance_limit: int | None = None,
    ) -> dict[str, float]:
        """Information value from target-to-alternative and pairwise distances.

        Args:
            target: The observed utterance being scored.
            alternatives: Utterances sampled from the target's own context.
            ooc_alternatives: Utterances sampled from an empty context, shared
                across the corpus; ``None`` omits the ``ooc`` and
                ``context_informativeness`` families.
            distance_limit: Optional positional cut on every distance list.
                None uses all distances. A finite limit reproduces the source
                script's order-dependent cut, including on alternative pairs.

        Raises:
            ValueError: When *alternatives* is empty.
        """
        _validate_distance_limit(distance_limit)
        self._ensure_loaded()
        alternatives = list(alternatives)
        if not alternatives:
            raise ValueError("InformationValueScorer.score needs ≥1 alternative")

        cut = slice(None, distance_limit)
        cross, self_ = self._distances(target, alternatives)

        out: dict[str, float] = {}
        for dist, vals in cross.items():
            arr = np.asarray(vals[cut], dtype=float)
            for agg_name, agg_fn in _AGGREGATIONS.items():
                out[f"information_value_{agg_name}_{dist}"] = float(agg_fn(arr))
        for dist, vals in self_.items():
            expected = float(np.mean(np.asarray(vals[cut], dtype=float)))
            out[f"information_value_expected_{dist}"] = expected
            out[f"information_value_deviation_{dist}"] = abs(
                out[f"information_value_mean_{dist}"] - expected
            )

        if ooc_alternatives is None:
            return out

        ooc_cross, ooc_self = self._distances(target, list(ooc_alternatives))
        for dist, vals in ooc_cross.items():
            arr = np.asarray(vals[cut], dtype=float)
            for agg_name, agg_fn in _AGGREGATIONS.items():
                out[f"information_value_ooc_{agg_name}_{dist}"] = float(agg_fn(arr))
            # The authors subtract the mean over every in-context distance,
            # before truncation, so this term ignores distance_limit.
            out[f"information_value_context_informativeness_{dist}"] = out[
                f"information_value_ooc_mean_{dist}"
            ] - float(np.mean(np.asarray(cross[dist], dtype=float)))
        for dist, vals in ooc_self.items():
            expected = float(np.mean(np.asarray(vals[cut], dtype=float)))
            out[f"information_value_ooc_expected_{dist}"] = expected
            out[f"information_value_ooc_deviation_{dist}"] = abs(
                out[f"information_value_ooc_mean_{dist}"] - expected
            )
            out[f"information_value_expected_context_informativeness_{dist}"] = (
                expected - float(np.mean(np.asarray(self_[dist], dtype=float)))
            )
        return out


def _validate_distance_limit(value: int | None) -> None:
    if value is not None and (
        isinstance(value, bool) or not isinstance(value, Integral) or value < 1
    ):
        raise ValueError("distance_limit must be a positive integer or None")


class InformationValueLM:
    """Native information-value FeatureModel.

    The unit of the measure is a **sentence**: the authors key their released
    columns by ``{text_id}_{sentence_num}`` and take the target to be one
    sentence with every preceding sentence of the same passage as its context.
    So each stimulus is segmented, each sentence is scored against
    ``n_samples`` continuations sampled from that context, and the sentence's
    values are broadcast to the units that fall inside it.

    A second set of alternatives is sampled once from an empty context and
    reused for every sentence, which is what the out-of-context and
    ``context_informativeness`` columns compare against.

    Pairwise estimates use all alternative pairs. Set ``distance_limit=n_samples``
    to reproduce the source script's positional cut on distance lists.
    """

    @property
    def supports_output_union(self) -> bool:
        """Combining requests preserves the same computed or sampled values."""
        return type(self) is InformationValueLM

    vocab = None

    def __init__(
        self,
        model_name: str = "gpt2",
        n_samples: int = 100,
        top_p: float = 0.9,
        top_k: int = 50,
        max_new_tokens: int = 60,
        device: str = "cpu",
        seed: int = 0,
        embed_model: str = _DEFAULT_EMBED_MODEL,
        spacy_model: str = "en_core_web_sm",
        out_of_context: bool = True,
        truncate_to_sentence: bool = True,
        sample_batch: int | None = None,
        temperature: float = 1.0,
        typical_p: float = 1.0,
        *,
        distance_limit: int | None = None,
    ):
        """Configure the sampler and the scorer it feeds.

        Args:
            model_name: Hugging Face causal LM the alternatives are sampled
                from.
            n_samples: Alternatives drawn per sentence.
            top_p: Nucleus-sampling mass.
            top_k: Top-k cut the nucleus is intersected with.
            max_new_tokens: Positive integer token budget for one sampled
                alternative; must leave room for a prompt token in the model window.
            device: Torch device the LM and the scorer's models run on.
            seed: Seed for the sampling draws; the global torch state is
                restored afterwards.
            embed_model: Sentence-embedding model handed to the scorer.
            spacy_model: spaCy pipeline handed to the scorer.
            out_of_context: Also draw one empty-context set, which the ``ooc``
                and ``context_informativeness`` families compare against.
            truncate_to_sentence: Keep only an alternative's first sentence, so
                a sentence target is not compared against a paragraph.
            sample_batch: Alternatives generated per call, bounding the joint KV
                cache; ``None`` draws them in one call.
            temperature: Sampling temperature; one retains unscaled logits.
            typical_p: Locally typical sampling mass; one disables this filter.
            distance_limit: Optional positional distance-list cut, independent
                of the number of sampled alternatives. None uses all distances.
        """
        self.name = f"{hub_id(model_name)}_information_value"
        self.model_name = model_name
        self.n_samples = int(n_samples)
        _validate_distance_limit(distance_limit)
        self.distance_limit = None if distance_limit is None else int(distance_limit)
        # The authors' nucleus_095 config sets top_p 0.9, the same value as
        # nucleus_09, and they leave top_k at the Transformers default of 50
        # rather than disabling it, so nucleus sampling there is intersected
        # with the top 50 tokens.
        self.top_p = float(top_p)
        self.top_k = int(top_k)
        self.temperature = float(temperature)
        self.typical_p = float(typical_p)
        if not np.isfinite(self.typical_p) or not 0 < self.typical_p <= 1:
            raise ValueError("Typical sampling mass must lie in (0, 1]")
        if not np.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("Sampling temperature must be finite and positive")
        if (
            isinstance(max_new_tokens, bool)
            or not isinstance(max_new_tokens, (int, Integral))
            or max_new_tokens < 1
        ):
            raise ValueError("max_new_tokens must be a positive integer")
        self.max_new_tokens: int = int(max_new_tokens)
        # Generating all n_samples sequences at once holds their joint KV
        # cache, which can exceed the GPU near the position cap on
        # billion-parameter models. Chunking draws the same distribution
        # over several rounds and only perturbs the draw stream.
        self.sample_batch = int(sample_batch) if sample_batch else None
        self.device = device
        self.seed = int(seed)
        self.out_of_context = bool(out_of_context)
        self.truncate_to_sentence = bool(truncate_to_sentence)
        self._lm = None
        self._tok = None
        self._ooc_cache: list[str] | None = None
        self._document_context: dict[str, str] = {}
        self.scorer = InformationValueScorer(
            embed_model=embed_model, spacy_model=spacy_model, device=device
        )

    def _ensure_loaded(self):
        if self._lm is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tok = AutoTokenizer.from_pretrained(self.model_name)
        self._lm = (
            AutoModelForCausalLM.from_pretrained(
                self.model_name, attn_implementation="eager"
            )
            .eval()
            .to(self.device)
        )

    def _sample_alternatives(self, context: str) -> list[str]:
        import torch

        self._ensure_loaded()
        limit = usable_position_limit(self.model_name, self._lm.config)
        if self.max_new_tokens >= limit:
            raise ValueError(
                f"max_new_tokens={self.max_new_tokens} must be less than the "
                f"context window of {limit} to leave room for a prompt token"
            )
        # The authors feed a single space for an absent context rather than a
        # BOS token, so an empty context is still a real (if uninformative)
        # prompt and the sampler is never conditioned on nothing.
        input_ids = self._tok(context or " ", return_tensors="pt").input_ids.to(
            self.device
        )
        # Reserve room for generated tokens within the resolved context window.
        from ..backends.logits import FiniteLogitsProcessor

        input_ids = input_ids[:, -(limit - self.max_new_tokens) :]
        # generate() draws from the global torch RNG; seed it locally and
        # restore the prior state so other components in the same process are
        # not perturbed (generate has no generator= arg to seed it directly).
        torch_device = torch.device(self.device)
        fork_devices = [torch_device.index or 0] if torch_device.type == "cuda" else []
        batch = self.sample_batch or self.n_samples
        sampling_options = (
            {"temperature": self.temperature} if self.temperature != 1.0 else {}
        )
        if self.typical_p != 1.0:
            sampling_options["typical_p"] = self.typical_p
        sequences = []
        with torch.random.fork_rng(devices=fork_devices):
            torch.manual_seed(self.seed)
            with torch.no_grad():
                for offset in range(0, self.n_samples, batch):
                    gen = self._lm.generate(
                        input_ids,
                        do_sample=True,
                        logits_processor=[FiniteLogitsProcessor()],
                        top_p=self.top_p,
                        top_k=self.top_k,
                        max_new_tokens=self.max_new_tokens,
                        num_return_sequences=min(batch, self.n_samples - offset),
                        pad_token_id=self._tok.eos_token_id,
                        **sampling_options,
                    )
                    sequences.extend(gen)
        prompt_len = input_ids.shape[1]
        raw = [
            self._tok.decode(g[prompt_len:], skip_special_tokens=True)
            for g in sequences
        ]
        return [self._first_sentence(text) for text in raw]

    def _first_sentence(self, text: str) -> str:
        """Keep an alternative's first sentence, the authors' ``--separator spacy``.

        Generation runs to ``max_new_tokens`` and so overruns the sentence the
        target is, which would compare a sentence against a paragraph. Without
        this the overrun systematically inflates the length distance.
        """
        stripped = text.strip()
        if not self.truncate_to_sentence or not stripped:
            return stripped
        self.scorer._ensure_loaded()
        sentences = self.scorer._nlp(stripped).sents
        first = next(iter(sentences), None)
        return first.text if first is not None else stripped

    def _sentences(self, text: str) -> list[tuple[int, int]]:
        """Character spans of the stimulus's sentences, in order."""
        self.scorer._ensure_loaded()
        doc = self.scorer._nlp(text)
        spans = [(s.start_char, s.end_char) for s in doc.sents if s.text.strip()]
        return spans or [(0, len(text))]

    #: Metadata keys that name the document a stimulus came from, most
    #: specific first. Only a key shared by several stimuli adds anything.
    DOCUMENT_KEYS = ("docid", "document_id", "article_id", "text_id")

    def prepopulate_cache(self, stimuli, target_layer: str = "words") -> None:
        """Record what precedes each stimulus inside its document, in corpus order.

        Without it, a corpus that makes every sentence its own stimulus scores
        each sentence as passage-initial, the measure's out-of-context case.

        Args:
            stimuli: The corpus's stimuli in reading order; those carrying a
                metadata key from :attr:`DOCUMENT_KEYS` group into documents.
            target_layer: Part of the shared FeatureModel signature; the
                document context is read from the stimulus text and metadata.
        """
        self._document_context = {}
        seen: dict[tuple[str, object], list[str]] = {}
        for stimulus in stimuli:
            metadata = stimulus.metadata or {}
            key = next(
                ((k, metadata[k]) for k in self.DOCUMENT_KEYS if k in metadata),
                None,
            )
            if key is None:
                continue
            preceding = seen.setdefault(key, [])
            if preceding:
                self._document_context[stimulus.stimulus_id] = " ".join(preceding)
            preceding.append(stimulus.text)

    def _out_of_context(self) -> list[str] | None:
        if not self.out_of_context:
            return None
        if self._ooc_cache is None:
            self._ooc_cache = self._sample_alternatives("")
        return self._ooc_cache

    def __call__(
        self,
        stimulus: Stimulus,
        extract: list[str] | None = None,
        target_layer: str = "words",
    ) -> list[FeatureModelOutput]:
        """Score each sentence and broadcast its values to the units inside it.

        Args:
            stimulus: Passage to score, segmented into sentences here.
            extract: Part of the shared FeatureModel signature; every scored
                column is returned regardless.
            target_layer: Annotation layer whose spans the sentence values are
                broadcast onto, one output per unit.

        Returns:
            One output per unit. A sentence whose sampling hit non-finite
            logits contributes empty values rather than aborting the stimulus.
        """
        layer = stimulus.layers[target_layer]
        text = stimulus.text
        sentences = self._sentences(text)
        ooc = self._out_of_context()
        # What came before this stimulus in its document, if the corpus says.
        prefix = self._document_context.get(stimulus.stimulus_id, "")

        # One score per sentence; the context is every preceding sentence of
        # this stimulus, matching the authors' per-passage context build-up.
        per_sentence: list[dict[str, float]] = []
        for start, end in sentences:
            context = f"{prefix} {text[:start]}".strip() if prefix else text[:start]
            try:
                alternatives = self._sample_alternatives(context.strip())
            except NonFiniteLogitsError:
                per_sentence.append({})
                continue
            values = self.scorer.score(
                text[start:end].strip(),
                alternatives,
                ooc_alternatives=ooc,
                distance_limit=self.distance_limit,
            )
            per_sentence.append(values)

        outputs: list[FeatureModelOutput] = []
        for span in layer.spans:
            # A unit belongs to the last sentence that starts at or before it,
            # so a unit straddling a boundary follows its first character.
            idx = 0
            for i, (start, _end) in enumerate(sentences):
                if span.start >= start:
                    idx = i
            outputs.append(
                FeatureModelOutput(
                    values=dict(per_sentence[idx]),
                    metadata={
                        "model_name": self.name,
                        "sentence_index": idx,
                        "n_samples": self.n_samples,
                        "distance_limit": self.distance_limit,
                    },
                )
            )
        return outputs


class InformationValue:
    """Feature extractor for one information-value metric.

    Reads ``information_value_{aggregation}_{distance}`` from an
    :class:`InformationValueLM` (or precomputed) output.

    Args:
        distance: one of ``1gram``/``2gram``/``3gram`` (lexical),
            ``1gram_pos``/``2gram_pos``/``3gram_pos`` (syntactic),
            ``cosine``/``euclidean`` (semantic), or ``length``.
        aggregation: ``mean`` (default), ``min``, or ``max``.
    """

    def __init__(self, distance: str = "cosine", aggregation: str = "mean"):
        """Fix which stored column this feature reads.

        The class docstring describes the parameters.

        Raises:
            ValueError: When *aggregation* is not one of mean, min, max.
        """
        if aggregation not in _AGGREGATIONS:
            raise ValueError(f"aggregation must be in {set(_AGGREGATIONS)}")
        self.distance = distance
        self.aggregation = aggregation
        self._key = f"information_value_{aggregation}_{distance}"
        self.name = self._key
        self.measurement = (
            FeatureMeasurement(
                "information_value",
                "tokens" if distance == "length" else "dimensionless",
                {"distance": distance, "aggregation": aggregation},
            )
            if distance in INFORMATION_VALUE_DISTANCES
            else None
        )

    def configuration(self) -> dict:
        """Declare the stored computation key independently of its alias."""
        return {"key": self._key}

    @property
    def required_outputs(self) -> tuple[str, ...]:
        """Model value keys consumed by this feature."""
        return (self._key,)

    def __call__(
        self,
        stimulus: Stimulus,
        outputs: Sequence[ModelOutputRecord] | None,
        target_layer: str = "words",
    ) -> np.ndarray:
        """Read the configured column out of the model outputs.

        Args:
            stimulus: Part of the shared Feature signature; the values come from
                *outputs*.
            outputs: Per-unit model outputs; a unit missing the column is NaN.
            target_layer: Part of the shared Feature signature; *outputs* are
                already unit-aligned.

        Raises:
            ValueError: When *outputs* is ``None``.
        """
        if outputs is None:
            raise ValueError("InformationValue requires model outputs")
        return np.array([o.values.get(self._key, np.nan) for o in outputs])
