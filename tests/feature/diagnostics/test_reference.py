"""MarkovReferenceLM: protocol compliance, probe coverage, exact outputs."""

import numpy as np
import pytest

from replica.feature import FeatureModel
from replica.feature.diagnostics import MarkovReferenceLM, make_stimulus


@pytest.fixture(scope="module")
def ref():
    return MarkovReferenceLM()


class TestProtocol:
    def test_is_feature_model(self, ref):
        assert isinstance(ref, FeatureModel)

    def test_outputs_one_per_unit(self, ref):
        stim = make_stimulus(["ba", "da", "ga"])
        outs = ref(stim)
        assert len(outs) == 3

    def test_next_log_probs_normalized(self, ref):
        stim = ref.probe_stimulus()
        for out in ref(stim):
            p = np.exp(out.values["next_log_probs"])
            assert np.isclose(p.sum(), 1.0, atol=1e-12)

    def test_log_p_observed_matches_chain(self, ref):
        stim = make_stimulus(["da", "ba", "ta"])
        outs = ref(stim)
        i_da, i_ba, i_ta = (ref.vocab.index(w) for w in ["da", "ba", "ta"])
        assert np.isclose(outs[0].values["log_p_observed"], np.log(ref.initial[i_da]))
        assert np.isclose(
            outs[1].values["log_p_observed"], np.log(ref.transition[i_da, i_ba])
        )
        assert np.isclose(
            outs[2].values["log_p_observed"], np.log(ref.transition[i_ba, i_ta])
        )

    def test_vocab_index_metadata(self, ref):
        stim = make_stimulus(["ba", "da"])
        out = ref(stim)[1]
        assert out.metadata["vocab_index"]["da"] == 1


class TestProbe:
    def test_probe_covers_all_bigrams(self, ref):
        idx = ref.state_indices(ref.probe_stimulus())
        v = len(ref.vocab)
        assert len(idx) == v * v + 1
        bigrams = {(int(a), int(b)) for a, b in zip(idx[:-1], idx[1:], strict=True)}
        assert len(bigrams) == v * v

    def test_sample_stimuli_deterministic_and_in_vocab(self, ref):
        a = ref.sample_stimuli(2, 10, seed=3)
        b = ref.sample_stimuli(2, 10, seed=3)
        assert [s.text for s in a] == [s.text for s in b]
        for s in a:
            assert all(w in ref.vocab for w in s.unit_strings())

    def test_oov_word_raises(self, ref):
        with pytest.raises(KeyError, match="not in reference vocab"):
            ref.state_indices(make_stimulus(["ba", "zz"]))


class TestValidation:
    def test_rows_must_be_stochastic(self):
        t = np.full((8, 8), 0.125)
        t[0, 0] = 0.5
        with pytest.raises(ValueError, match="sum to 1"):
            MarkovReferenceLM(transition=t)

    def test_shape_must_match_vocab(self):
        with pytest.raises(ValueError, match="transition must be"):
            MarkovReferenceLM(transition=np.eye(3), vocab=["a", "b"])
