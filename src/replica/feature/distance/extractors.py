"""Distance Feature extractors (observed-vs-alternatives distances)."""

from collections.abc import Sequence

import numpy as np

from replica.feature.core.measurement import FeatureMeasurement
from replica.feature.core.protocols import ModelOutputRecord

from ...core import Stimulus


class _StoredScalarFeature:
    """Shared reader for one stored scalar per output (key ``_key``);
    positions without a stored value stay NaN.
    """

    def configuration(self) -> dict:
        """Return computation settings without the column alias."""
        return {"key": self._key}

    _key: str
    _requires = "model outputs"

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
        if outputs is None:
            raise ValueError(f"{type(self).__name__} requires {self._requires}")
        n = len(outputs)
        vals = np.full(n, np.nan)
        for t in range(n):
            v = outputs[t].values.get(self._key)
            if v is None:
                continue
            vals[t] = float(v)
        return vals


class ResponsivityDistance(_StoredScalarFeature):
    """Decontextualised responsivity distance from
    Giulianelli et al. (2024a).

    Pulls one of two scalars per word from a ``ResponsivityLM``'s outputs:

      * ``observed`` (default), ``decontextualised_distance``, the mean
        cosine distance between the actually-observed word's embedding and
        the embeddings of words sampled from the LM's distribution at the
        same context. This is the *responsivity* measure (how much the
        observed word displaced the model from its expectation).
      * ``expected``, ``expected_decontextualised_distance``, the mean
        pairwise cosine distance among the sampled alternatives only. This
        is the *expected responsivity* (the spread of the LM's predictive
        distribution in embedding space, independent of which word was
        actually observed).

    Position 0 is NaN unless the underlying ``ResponsivityLM`` was built
    with ``add_bos=True``.
    """

    _requires = "model outputs from a ResponsivityLM (or compatible)"

    def __init__(self, kind: str = "observed"):
        """Pick which of the two stored distances this feature reads.

        Args:
            kind: ``"observed"`` for the observed-vs-alternatives distance,
                ``"expected"`` for the alternatives' own mean pairwise distance.

        Raises:
            ValueError: When *kind* is neither.
        """
        if kind not in ("observed", "expected"):
            raise ValueError(
                f"ResponsivityDistance kind must be 'observed' or 'expected', "
                f"got {kind!r}"
            )
        self.kind = kind
        self.measurement = FeatureMeasurement(
            "cosine_distance", "dimensionless", {"kind": kind}
        )
        self._key = (
            "decontextualised_distance"
            if kind == "observed"
            else "expected_decontextualised_distance"
        )
        self.name = (
            "responsivity_distance"
            if kind == "observed"
            else "expected_responsivity_distance"
        )


class _SimilarityAdjustedScalar(_StoredScalarFeature):
    """Shared scalar-key extractor for the Meister et al. (2024) predictors.

    ``similarity`` selects the variant: ``"embedding"`` reads the plain key
    produced by :class:`SimilarityAdjustedSurprisalEmbeddingLM` (the key does
    not record whether the model was contextual); ``"pos"`` and
    ``"orthographic"`` read the suffixed keys of
    :class:`SimilarityAdjustedSurprisalWordLM`.
    """

    _base_key: str
    _quantity: str
    _unit: str
    _requires = "model outputs from a similarity-adjusted surprisal FeatureModel"

    _SUFFIX = {"embedding": "", "pos": "_pos", "orthographic": "_orth"}

    def __init__(self, similarity: str = "embedding"):
        if similarity not in self._SUFFIX:
            raise ValueError(
                f"similarity must be one of {sorted(self._SUFFIX)}, got {similarity!r}"
            )
        self.similarity = similarity
        self._key = f"{self._base_key}{self._SUFFIX[similarity]}"
        self.name = self._key
        self.measurement = FeatureMeasurement(
            self._quantity, self._unit, {"similarity": similarity}
        )


class SimilarityAdjustedSurprisal(_SimilarityAdjustedScalar):
    """Similarity-adjusted surprisal ``-log E_{w'~p}[z(w_t, w')]`` in nats, from
    Meister et al. (2024), summed over a word's subwords for the embedding
    variant. ``inf`` marks an all-dissimilar MC draw (expected similarity
    0).
    """

    _base_key = "similarity_adjusted_surprisal"
    _quantity = "similarity_adjusted_surprisal"
    _unit = "nats"


class NextWordInformationValue(_SimilarityAdjustedScalar):
    """Next-word information value ``E_{w'~p}[1 - z(w_t, w')]`` from
    Meister et al. (2024).
    """

    _base_key = "next_word_information_value"
    _quantity = "expected_dissimilarity"
    _unit = "dimensionless"
