"""Shared interface behavior for model-estimated unigram sources."""


class EstimatedUnigramSource:
    """Expose a cached estimator through the probability-source protocol."""

    def probability(self, unit: str) -> float:
        """Return p(*unit*), or NaN when its estimate is zero.

        Args:
            unit: Surface form to score, as the concrete estimator sees it.
        """
        probability = self.estimate(unit)
        return float(probability) if probability > 0 else float("nan")
