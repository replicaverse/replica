"""A sampled unigram source over the stub LM, with a model that can be killed.

Shared by the priming tests of the source and of the compute runner.
"""

import math

from replica.feature.frequency.sampled import SampledUnigramSource
from tests.feature.frequency._pooled_stub import stub_pair

#: The contexts a first attempt draws.
C1 = [" the dog ran", " a cat sat", " the the dog", " hello mat"]
#: What a later attempt would draw; a resume must never score against these.
C2 = [" cat", " a b c", " the mat ran", " hello"]


class Killed(BaseException):
    """Stands in for the job being killed at its time limit."""


class Killable:
    """Delegate forward calls to *model*, raising :class:`Killed` after *limit*."""

    def __init__(self, model, limit: float = math.inf):
        self._model = model
        self.config = model.config
        self._limit = limit
        self.calls = 0

    def parameters(self):
        return self._model.parameters()

    def __call__(self, *args, **kwargs):
        if self.calls >= self._limit:
            raise Killed
        self.calls += 1
        return self._model(*args, **kwargs)


def refuse_sampling():
    """A sampler that fails the test, for a run that must reuse saved contexts."""
    raise AssertionError("the contexts were drawn again")


def stub_source(contexts=C1, *, kill_after=math.inf, generate=None, **settings):
    """A source whose sampler returns *contexts* and whose model is killed after
    *kill_after* forward calls.
    """
    tokenizer, model = stub_pair()
    settings.setdefault("batch_size", 2)
    settings.setdefault("n_samples", len(C1))
    source = SampledUnigramSource(model_name="stub", **settings)
    source._tokenizer = tokenizer
    source._model = Killable(model, kill_after)
    source._generate_samples = generate or (lambda: list(contexts))
    return source
