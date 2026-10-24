"""Integration-test helpers, plus re-exports of ``data_root`` and
``_ConstantLM`` from the top-level conftest.
"""

import numpy as np

from replica.core import Stimulus
from replica.feature import FeatureModelOutput

from ..conftest import _ConstantLM, constant_lm, data_root


class _VaryingLM:
    """Toy LM that returns position-dependent log-probs.

    Unlike _ConstantLM, this produces varied surprisal values so that
    GAM smooth terms have enough unique values to fit.
    """

    name = "varying_lm"
    vocab = ["a", "b", "c"]

    def __call__(self, stimulus: Stimulus, extract=None, target_layer="words"):
        n = len(stimulus.layers[target_layer])
        nlp = np.log(np.array([1 / 3, 1 / 3, 1 / 3]))
        rng = np.random.RandomState(42)
        outputs = []
        for t in range(n):
            lp = np.nan if t == 0 else -1.0 - rng.uniform(0, 2.0)
            outputs.append(
                FeatureModelOutput(values={"log_p_observed": lp, "next_log_probs": nlp})
            )
        return outputs
