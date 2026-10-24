"""Every method in METHOD_SPECS must resolve to a cache tag.

Guards the METHOD_SPECS <-> cache-naming agreement: a method registered
without a tag rule in ``method_cache_tag`` would only fail at compute time.
"""

import pytest

from replica.compute.cache import cache_filename, method_cache_tag
from replica.compute.methods import METHOD_SPECS, ComputeContext
from replica.compute.runner import cache_stem

# Representative hyperparameters (the sweep defaults); no model is built.
_HYPERPARAMS = {
    "beam_width": 5,
    "n_samples": 64,
    "max_tokens": 10,
    "alpha": 1.0,
    "seed": None,
}


@pytest.mark.parametrize("method", sorted(METHOD_SPECS))
def test_every_method_has_cache_tag(method):
    tag = method_cache_tag(method, **_HYPERPARAMS)
    assert isinstance(tag, str) and tag

    filename = cache_filename(
        model_id="gpt2", method=method, dataset="provo", **_HYPERPARAMS
    )
    assert filename == f"gpt2_{tag}_provo.pkl"

    context = ComputeContext(
        model_id="gpt2",
        add_bos=True,
        dataset_language=None,
        language_override=None,
        chunk_size=None,
        **_HYPERPARAMS,
    )
    assert cache_stem(method, context, "provo") == f"gpt2_{tag}_provo"
