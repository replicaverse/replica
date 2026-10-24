"""Tests for configurable evaluation: PrecomputedFeatureModel, PrecomputedFeature,
Lag, DerivedFeature, TabularDataset, and CVSplitter.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from replica.analysis.contracts.results import FoldResult
from replica.analysis.cv import (
    CrossedGroupSplitter,
    ItemSplitter,
    MinMaxScaler,
    MultiSeedSplitter,
    NoScaler,
    RowSplitter,
    ShuffledKFoldSplitter,
    SubjectSentenceMod10Splitter,
    ZScoreScaler,
)
from replica.analysis.evaluation.evaluate import evaluate
from replica.analysis.fitting.folds import _cv_splits
from replica.analysis.fitting.scoring import _r_squared
from replica.analysis.models import OLS, StatsmodelsOLS
from replica.analysis.specification.config import AnalysisConfig
from replica.analysis.specification.evaluation import EvaluationSpec
from replica.core import Datum
from replica.dataset.base import TabularDataset, TabularDatasetConfig
from replica.feature import (
    FeatureModelOutput,
    ModelFeature,
    PrecomputedFeatureModel,
    resolve,
)
from replica.feature.extractors import (
    DerivedFeature,
    Lag,
    PrecomputedFeature,
    Surprisal,
    UnitLength,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
from tests.analysis._fixtures import make_stimulus  # shared fixture


def make_outputs(values_dicts):
    """Build list[FeatureModelOutput] from a list of value dicts."""
    return [FeatureModelOutput(values=d) for d in values_dicts]


def _bound(stimulus, outputs, extractor):
    return ModelFeature(
        PrecomputedFeatureModel("stored", {stimulus.stimulus_id: outputs}), extractor
    )


def _resolved(stimulus, feature):
    return (
        resolve(SimpleNamespace(stimuli=lambda: [stimulus]), feature).blocks[0].values
    )


def _toy_dataset_and_model(n_stimuli, n_words):
    """One reader's random reading times and a feature model of random log-probs.

    The first word of every stimulus has no log-probability, so it drops out.
    """
    from replica.dataset import DatasetInfo, SPRDataset

    class ToyDS(SPRDataset):
        def load(self, path):
            pass

    ds = ToyDS()
    ds.info = DatasetInfo(name="toy")
    np.random.seed(42)
    for i in range(n_stimuli):
        words = [f"w{j}" for j in range(n_words)]
        stim = make_stimulus(tuple(words), f"stim_{i}")
        ds._stimuli.append(stim)
        ds._data.append(
            Datum(
                stimulus=stim,
                measure_name="spr_rt",
                values=np.random.uniform(200, 500, size=n_words),
                participant_id="p1",
            )
        )

    outputs = {}
    for stim in ds.stimuli():
        n = len(stim.layers["words"])
        outputs[stim.stimulus_id] = [
            FeatureModelOutput(
                values={"log_p_observed": float(-np.random.uniform(0.5, 3.0))}
            )
            for _ in range(n)
        ]
        outputs[stim.stimulus_id][0].values["log_p_observed"] = float("nan")
    return ds, PrecomputedFeatureModel("m", outputs)


# ---------------------------------------------------------------------------
# TestPrecomputedFeature
# ---------------------------------------------------------------------------


class TestPrecomputedFeature:
    def test_extracts_correct_key(self):
        stim = make_stimulus(("a", "b", "c"))
        outputs = make_outputs([{"surp": 1.0}, {"surp": 2.0}, {"surp": 3.0}])
        feat = PrecomputedFeature("surprisal", "surp")
        result = feat(stim, outputs)
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])

    def test_missing_key_returns_nan(self):
        stim = make_stimulus(("a", "b"))
        outputs = make_outputs([{"surp": 1.0}, {}])
        feat = PrecomputedFeature("surprisal", "surp")
        result = feat(stim, outputs)
        assert result[0] == 1.0
        assert np.isnan(result[1])

    def test_requires_outputs(self):
        stim = make_stimulus(("a",))
        feat = PrecomputedFeature("surprisal", "surp")
        with pytest.raises(ValueError, match="requires outputs"):
            feat(stim, None)


# ---------------------------------------------------------------------------
# TestLag
# ---------------------------------------------------------------------------


class TestLag:
    @pytest.mark.parametrize(
        "values, n, expected",
        [
            ([1.0, 2.0, 3.0, 4.0], 1, [np.nan, 1.0, 2.0, 3.0]),
            ([10.0, 20.0, 30.0, 40.0], 2, [np.nan, np.nan, 10.0, 20.0]),
            ([1.0, 2.0, 3.0], -1, [2.0, 3.0, np.nan]),
            ([5.0, 6.0], 0, [5.0, 6.0]),
        ],
        ids=["lag1", "lag2", "lag_neg1", "lag0"],
    )
    def test_shift(self, values, n, expected):
        """Lag(n) shifts the wrapped feature by n, NaN-filling the gap."""
        stim = make_stimulus(tuple("abcdefgh"[: len(values)]))
        outputs = make_outputs([{"x": v} for v in values])
        result = _resolved(
            stim, Lag(_bound(stim, outputs, PrecomputedFeature("x", "x")), n)
        )
        np.testing.assert_array_equal(result, expected)

    def test_name(self):
        inner = ModelFeature(
            PrecomputedFeatureModel("stored", {}),
            PrecomputedFeature("surprisal", "surp"),
        )
        lag = Lag(inner, 1)
        assert lag.name == "lag1_surprisal"

    @pytest.mark.parametrize("n", [1, -1])
    def test_integer_values_leave_a_nan_gap(self, n):
        """An integer-valued feature shifts into a float array with NaN in the gap."""
        stim = make_stimulus(("a", "bb", "ccc"))
        outputs = make_outputs([{"wlen": v} for v in (1, 2, 3)])
        result = _resolved(
            stim, Lag(_bound(stim, outputs, PrecomputedFeature("wlen", "wlen")), n)
        )
        expected = [np.nan, 1.0, 2.0] if n == 1 else [2.0, 3.0, np.nan]
        np.testing.assert_array_equal(result, expected)

    def test_with_data_derived_feature(self):
        """Lag should work with data-derived features like UnitLength."""
        stim = make_stimulus(("I", "am", "here"))
        lag = Lag(UnitLength(), 1)
        result = _resolved(stim, lag)
        assert np.isnan(result[0])
        assert result[1] == 1.0  # len("I")
        assert result[2] == 2.0  # len("am")


# ---------------------------------------------------------------------------
# TestDerivedFeature
# ---------------------------------------------------------------------------


class TestDerivedFeature:
    def test_arithmetic_combination(self):
        stim = make_stimulus(("a", "bb", "ccc"))
        outputs = make_outputs([{"x": 1.0}, {"x": 2.0}, {"x": 3.0}])
        f1 = _bound(stim, outputs, PrecomputedFeature("x", "x"))
        f2 = UnitLength()
        derived = DerivedFeature("sum", lambda a, b: a + b, f1, f2)
        result = _resolved(stim, derived)
        # x + len: [1+1, 2+2, 3+3]
        np.testing.assert_array_equal(result, [2.0, 4.0, 6.0])


# ---------------------------------------------------------------------------
# TestRowBasedCV
# ---------------------------------------------------------------------------


class TestCVSplitter:
    """Test CVSplitter protocol and built-in implementations."""

    def test_row_splitter_runs(self):
        ds, fm = _toy_dataset_and_model(n_stimuli=5, n_words=10)
        results = evaluate(
            dataset=ds,
            spec=EvaluationSpec(
                targets=[ds.feature(ds.default_measure)],
                predictors=[ModelFeature(fm, Surprisal())],
                controls=[UnitLength()],
            ),
            analysis=AnalysisConfig(
                design=None,
                transform_dv=np.log2,
                model=OLS(),
                cv=RowSplitter(n_folds=5),
            ),
        )
        assert len(results) == 1
        r = results[0]
        assert r.n_observations > 0
        assert np.isfinite(r.delta_ll)

    def test_item_splitter_default(self):
        """Default (no cv_splitter) uses ItemSplitter."""
        ds, fm = _toy_dataset_and_model(n_stimuli=5, n_words=10)
        r_default = evaluate(
            dataset=ds,
            spec=EvaluationSpec(
                targets=[ds.feature(ds.default_measure)],
                predictors=[ModelFeature(fm, Surprisal())],
                controls=[UnitLength()],
            ),
            analysis=AnalysisConfig(
                design=None, transform_dv=np.log2, model=OLS(), n_folds=5
            ),
        )[0]
        r_explicit = evaluate(
            dataset=ds,
            spec=EvaluationSpec(
                targets=[ds.feature(ds.default_measure)],
                predictors=[ModelFeature(fm, Surprisal())],
                controls=[UnitLength()],
            ),
            analysis=AnalysisConfig(
                design=None,
                transform_dv=np.log2,
                model=OLS(),
                cv=ItemSplitter(n_folds=5),
            ),
        )[0]
        assert r_default.delta_ll == r_explicit.delta_ll

    def test_custom_splitter(self):
        """A custom CVSplitter works with evaluate()."""

        class LeaveOneItemOut:
            def split(self, y, groups, participants, items):
                unique = np.unique(groups)
                for g in unique:
                    test_mask = groups == g
                    yield ~test_mask, test_mask

        ds, fm = _toy_dataset_and_model(n_stimuli=5, n_words=10)
        results = evaluate(
            dataset=ds,
            spec=EvaluationSpec(
                targets=[ds.feature(ds.default_measure)],
                predictors=[ModelFeature(fm, Surprisal())],
            ),
            analysis=AnalysisConfig(
                design=None, transform_dv=np.log2, model=OLS(), cv=LeaveOneItemOut()
            ),
        )
        assert len(results) == 1
        assert np.isfinite(results[0].delta_ll)

    def test_item_splitter_splits(self):
        """ItemSplitter yields correct number of folds and non-overlapping masks."""
        y = np.arange(20, dtype=float)
        groups = np.array([0] * 4 + [1] * 4 + [2] * 4 + [3] * 4 + [4] * 4)
        participants = np.zeros(20, dtype=int)
        items = groups.copy()

        splitter = ItemSplitter(n_folds=3)
        splits = list(splitter.split(y, groups, participants, items))
        assert len(splits) == 3
        # Every observation appears in exactly one test fold
        all_test = np.zeros(20, dtype=bool)
        for train_mask, test_mask in splits:
            assert np.all(train_mask == ~test_mask)
            all_test |= test_mask
        assert np.all(all_test)

    def test_row_splitter_splits(self):
        """RowSplitter assigns rows sequentially."""
        y = np.arange(10, dtype=float)
        groups = np.zeros(10, dtype=int)
        participants = np.zeros(10, dtype=int)
        items = np.zeros(10, dtype=int)

        splitter = RowSplitter(n_folds=3)
        splits = list(splitter.split(y, groups, participants, items))
        assert len(splits) == 3
        # Row 0 in fold 0, row 1 in fold 1, row 2 in fold 2, row 3 in fold 0, ...
        _, test_0 = splits[0]
        assert test_0[0] and test_0[3] and test_0[6] and test_0[9]
        assert not test_0[1] and not test_0[2]


class TestPopulateSentenceIds:
    def test_numbers_stay_unique_after_a_stimulus_that_carries_its_own(self):
        """A stimulus that carries its own ids must not reset the running
        counter to numbers a preceding stimulus already used."""
        from types import SimpleNamespace

        from replica.analysis.cv import populate_sentence_ids

        first = make_stimulus(("A.", "b.", "c.", "d."), "first")
        carried = make_stimulus(("x",), "carried")
        carried.metadata["per_word_sentence_ids"] = [0]
        last = make_stimulus(("e.", "f."), "last")
        stimuli = [first, carried, last]
        total = populate_sentence_ids(SimpleNamespace(stimuli=lambda: stimuli))

        assert list(first.metadata["per_word_sentence_ids"]) == [0, 1, 2, 3]
        assert list(carried.metadata["per_word_sentence_ids"]) == [0]
        assert list(last.metadata["per_word_sentence_ids"]) == [4, 5]
        assert total == 6


# ---------------------------------------------------------------------------
# TestTabularDataset
# ---------------------------------------------------------------------------


class TestTabularDataset:
    def test_builds_from_dict(self):
        data = {
            "trialid": [1, 1, 1, 2, 2, 2],
            "ianum": [1, 2, 3, 1, 2, 3],
            "ia": ["The", "cat", "sat", "A", "dog", "ran"],
            "gaze_rt": [200.0, 300.0, 250.0, 180.0, 320.0, 270.0],
        }
        config = TabularDatasetConfig(name="test_ds")
        ds = TabularDataset(data, config)
        assert len(ds.stimuli()) == 2
        assert len(ds.data(measure="gaze_rt")) == 2
        assert ds.info.name == "test_ds"
        assert ds.default_measure == "gaze_rt"
        np.testing.assert_array_equal(
            ds.data(measure="gaze_rt")[0].values, [200.0, 300.0, 250.0]
        )
        assert ds.data(measure="first_run") == []

    def test_builds_from_dataframe(self):

        df = pd.DataFrame(
            {
                "trialid": [1, 1, 2, 2],
                "ianum": [1, 2, 1, 2],
                "ia": ["Hello", "world", "Foo", "bar"],
                "gaze_rt": [100.0, 200.0, 150.0, 250.0],
            }
        )
        config = TabularDatasetConfig(name="df_test")
        ds = TabularDataset(df, config)
        assert len(ds.stimuli()) == 2
        words_0 = ds.stimuli()[0].unit_strings("words")
        assert words_0 == ["Hello", "world"]

    def test_stimulus_ids(self):
        data = {
            "trialid": [10, 10, 20, 20],
            "ianum": [1, 2, 1, 2],
            "ia": ["a", "b", "c", "d"],
            "gaze_rt": [1.0, 2.0, 3.0, 4.0],
        }
        config = TabularDatasetConfig(name="id_test")
        ds = TabularDataset(data, config)
        sids = [s.stimulus_id for s in ds.stimuli()]
        assert "trial_10" in sids
        assert "trial_20" in sids

    def test_custom_columns(self):
        data = {
            "sent_id": [1, 1, 1],
            "pos": [0, 1, 2],
            "token": ["I", "like", "cats"],
            "rt": [150.0, 200.0, 180.0],
        }
        config = TabularDatasetConfig(
            name="custom",
            trial_col="sent_id",
            unit_index_col="pos",
            text_col="token",
            measure_col="rt",
        )
        ds = TabularDataset(data, config)
        assert len(ds.stimuli()) == 1
        assert ds.stimuli()[0].unit_strings("words") == ["I", "like", "cats"]
        assert ds.data(measure="rt")[0].values[1] == 200.0

    def test_exclusion_mask_nan_only(self):
        data = {
            "trialid": [1, 1, 1],
            "ianum": [1, 2, 3],
            "ia": ["a", "b", "c"],
            "gaze_rt": [100.0, float("nan"), 200.0],
        }
        config = TabularDatasetConfig(name="excl_test")
        ds = TabularDataset(data, config)
        datum = ds.data(measure="gaze_rt")[0]
        mask = ds.exclusion_mask(datum)
        np.testing.assert_array_equal(mask, [True, False, True])

    def test_with_participants(self):
        data = {
            "trialid": [1, 1, 1, 1],
            "ianum": [1, 2, 1, 2],
            "ia": ["hi", "there", "hi", "there"],
            "gaze_rt": [100.0, 200.0, 150.0, 250.0],
            "subj": ["p1", "p1", "p2", "p2"],
        }
        config = TabularDatasetConfig(name="multi_p", participant_col="subj")
        ds = TabularDataset(data, config)
        assert len(ds.stimuli()) == 1
        assert len(ds.data(measure="gaze_rt")) == 2
        pids = [d.participant_id for d in ds.data()]
        assert "p1" in pids
        assert "p2" in pids

    def test_load_raises(self):
        data = {
            "trialid": [1],
            "ianum": [1],
            "ia": ["x"],
            "gaze_rt": [1.0],
        }
        ds = TabularDataset(data, TabularDatasetConfig(name="t"))
        with pytest.raises(NotImplementedError):
            ds.load("/tmp/nonexistent")


# ---------------------------------------------------------------------------
# TestEndToEnd
# ---------------------------------------------------------------------------


class TestEndToEnd:
    """End-to-end test mirroring the comparison script workflow using only
    built-in REPLICA classes."""

    def test_precomputed_pipeline(self):
        """PrecomputedFeatureModel + PrecomputedFeature + Lag + TabularDataset
        + evaluate with cv_strategy='row'."""
        np.random.seed(123)

        # Build tabular data with 3 trials, 5 words each
        n_trials, n_words = 3, 5
        rows = {
            "trialid": [],
            "ianum": [],
            "ia": [],
            "gaze_rt": [],
        }
        all_surp = {}
        all_freq = {}

        for t in range(n_trials):
            words_t = [f"word{t}_{w}" for w in range(n_words)]
            surps = np.random.uniform(2, 10, n_words)
            freqs = np.random.uniform(0.01, 0.5, n_words)
            for w in range(n_words):
                rows["trialid"].append(t)
                rows["ianum"].append(w)
                rows["ia"].append(words_t[w])
                rows["gaze_rt"].append(float(np.random.uniform(200, 500)))

            sid = f"trial_{t}"
            all_surp[sid] = surps
            all_freq[sid] = freqs

        config = TabularDatasetConfig(name="end2end_test")
        ds = TabularDataset(rows, config)

        # Build PrecomputedFeatureModel with surp and freq
        outputs = {}
        for stim in ds.stimuli():
            sid = stim.stimulus_id
            n = len(stim.layers["words"])
            trial_outputs = []
            for i in range(n):
                trial_outputs.append(
                    FeatureModelOutput(
                        values={
                            "surp": float(all_surp[sid][i]),
                            "freq": float(all_freq[sid][i]),
                        }
                    )
                )
            outputs[sid] = trial_outputs

        fm = PrecomputedFeatureModel("precomputed", outputs)

        # Features
        surp = ModelFeature(fm, PrecomputedFeature("surprisal", "surp"))
        freq = ModelFeature(fm, PrecomputedFeature("frequency", "freq"))
        prev_surp = Lag(surp, 1)

        results = evaluate(
            dataset=ds,
            spec=EvaluationSpec(
                targets=[ds.feature("gaze_rt")],
                predictors=[surp],
                controls=[freq, prev_surp],
            ),
            analysis=AnalysisConfig(
                design=None,
                model=OLS(),
                transform_dv=lambda x: x,
                n_folds=3,
                cv=RowSplitter(n_folds=3),
            ),
        )

        assert len(results) == 1
        r = results[0]
        assert r.dataset_name == "end2end_test"
        assert r.model_name == "precomputed"
        assert r.feature_name == "surprisal"
        assert r.n_observations > 0
        assert np.isfinite(r.delta_ll)

    def test_derived_feature_in_pipeline(self):
        """DerivedFeature works in evaluate()."""
        np.random.seed(42)
        n_trials, n_words = 2, 6
        rows = {"trialid": [], "ianum": [], "ia": [], "rt": []}
        for t in range(n_trials):
            for w in range(n_words):
                rows["trialid"].append(t)
                rows["ianum"].append(w)
                rows["ia"].append(f"w{w}")
                rows["rt"].append(float(np.random.uniform(150, 400)))

        config = TabularDatasetConfig(name="derived_test", measure_col="rt")
        ds = TabularDataset(rows, config)

        outputs = {}
        for stim in ds.stimuli():
            n = len(stim.layers["words"])
            outputs[stim.stimulus_id] = [
                FeatureModelOutput(values={"a": float(i), "b": float(i * 2)})
                for i in range(n)
            ]
        fm = PrecomputedFeatureModel("m", outputs)

        feat_a = ModelFeature(fm, PrecomputedFeature("a", "a"))
        feat_b = ModelFeature(fm, PrecomputedFeature("b", "b"))
        combined = DerivedFeature("a_plus_b", lambda a, b: a + b, feat_a, feat_b)

        results = evaluate(
            dataset=ds,
            spec=EvaluationSpec(targets=[ds.feature("rt")], predictors=[combined]),
            analysis=AnalysisConfig(
                design=None,
                model=OLS(),
                transform_dv=lambda x: x,
                n_folds=2,
            ),
        )
        assert len(results) == 1
        assert np.isfinite(results[0].delta_ll)


# ---------------------------------------------------------------------------
# TestShuffledKFoldSplitter
# ---------------------------------------------------------------------------


class TestShuffledKFoldSplitter:
    """Test ShuffledKFoldSplitter matches sklearn KFold."""

    def test_matches_sklearn(self):
        """Fold assignments must be identical to sklearn KFold."""
        from sklearn.model_selection import KFold

        for n in [17, 23, 100]:
            for seed in [0, 42]:
                kf = KFold(n_splits=10, shuffle=True, random_state=seed)
                sklearn_folds = list(kf.split(np.arange(n)))

                splitter = ShuffledKFoldSplitter(n_folds=10, seed=seed)
                y = np.arange(n, dtype=float)
                dummy = np.zeros(n, dtype=int)
                our_folds = list(splitter.split(y, dummy, dummy, dummy))

                for i, (_sk_train, sk_test) in enumerate(sklearn_folds):
                    _our_train, our_test = our_folds[i]
                    np.testing.assert_array_equal(
                        np.where(our_test)[0],
                        sk_test,
                        err_msg=f"Mismatch at n={n}, seed={seed}, fold={i}",
                    )


# ---------------------------------------------------------------------------
# TestScaler
# ---------------------------------------------------------------------------


class TestScaler:
    """Test Scaler protocol and built-in implementations."""

    @pytest.mark.parametrize("ddof", [0, 1])
    def test_zscore_scaler(self, ddof):
        """Both splits use the declared training standard deviation."""
        train = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]])
        test = np.array([[5.0, 50.0], [0.0, 0.0]])
        scaler = ZScoreScaler(ddof=ddof)
        t_train, t_test = scaler.fit_transform(train, test)
        np.testing.assert_allclose(np.mean(t_train, axis=0), [0.0, 0.0], atol=1e-10)
        np.testing.assert_allclose(
            np.std(t_train, axis=0, ddof=ddof), [1.0, 1.0], atol=1e-10
        )
        np.testing.assert_allclose(
            t_test, (test - train.mean(axis=0)) / train.std(axis=0, ddof=ddof)
        )
        assert scaler.configuration() == {"ddof": ddof}
        assert scaler.last_fit_diagnostics()["ddof"] == ddof

    @pytest.mark.parametrize("ddof", [-1, 0.5, True, np.nan])
    def test_zscore_rejects_invalid_ddof(self, ddof):
        with pytest.raises(ValueError, match="non-negative integer"):
            ZScoreScaler(ddof=ddof)

    def test_zscore_requires_enough_training_rows(self):
        scaler = ZScoreScaler(ddof=1)
        scaler.fit_transform(np.array([2.0, 3.0]), np.array([4.0]))
        with pytest.raises(ValueError, match="number of training rows"):
            scaler.fit_transform(np.array([2.0]), np.array([4.0]))
        assert scaler.last_fit_diagnostics() is None

    def test_zscore_scaler_1d(self):
        train = np.array([1.0, 2.0, 3.0, 4.0])
        test = np.array([5.0])
        t_train, _t_test = ZScoreScaler().fit_transform(train, test)
        np.testing.assert_allclose(np.mean(t_train), 0.0, atol=1e-10)

    def test_minmax_scaler(self):
        """MinMaxScaler scales training data to [0, 1]."""
        train = np.array([[0.0, 10.0], [5.0, 30.0], [10.0, 50.0]])
        test = np.array([[2.5, 20.0], [15.0, 60.0]])
        scaler = MinMaxScaler()
        t_train, t_test = scaler.fit_transform(train, test)
        np.testing.assert_allclose(t_train[:, 0], [0.0, 0.5, 1.0])
        np.testing.assert_allclose(t_train[:, 1], [0.0, 0.5, 1.0])
        np.testing.assert_allclose(t_test[0, 0], 0.25)
        np.testing.assert_allclose(t_test[1, 0], 1.5)  # Extrapolation

    def test_minmax_scaler_1d(self):
        train = np.array([0.0, 5.0, 10.0])
        test = np.array([2.5])
        t_train, t_test = MinMaxScaler().fit_transform(train, test)
        np.testing.assert_allclose(t_train, [0.0, 0.5, 1.0])
        np.testing.assert_allclose(t_test, [0.25])

    def test_minmax_constant_column(self):
        """Constant column should not raise and should give 0."""
        train = np.array([[5.0, 1.0], [5.0, 2.0], [5.0, 3.0]])
        test = np.array([[5.0, 4.0]])
        t_train, _t_test = MinMaxScaler().fit_transform(train, test)
        np.testing.assert_allclose(t_train[:, 0], [0.0, 0.0, 0.0])

    def test_no_scaler(self):
        """NoScaler returns inputs unchanged."""
        train = np.array([[1.0, 2.0], [3.0, 4.0]])
        test = np.array([[5.0, 6.0]])
        t_train, t_test = NoScaler().fit_transform(train, test)
        np.testing.assert_array_equal(t_train, train)
        np.testing.assert_array_equal(t_test, test)

    def test_scaler_in_evaluate(self):
        """MinMaxScaler works as scaler parameter in evaluate()."""
        ds, fm = _toy_dataset_and_model(n_stimuli=3, n_words=6)

        r_zscore = evaluate(
            dataset=ds,
            spec=EvaluationSpec(
                targets=[ds.feature(ds.default_measure)],
                predictors=[ModelFeature(fm, Surprisal())],
                controls=[UnitLength()],
            ),
            analysis=AnalysisConfig(
                design=None, transform_dv=np.log2, model=OLS(), scaler=ZScoreScaler()
            ),
        )[0]
        r_minmax = evaluate(
            dataset=ds,
            spec=EvaluationSpec(
                targets=[ds.feature(ds.default_measure)],
                predictors=[ModelFeature(fm, Surprisal())],
                controls=[UnitLength()],
            ),
            analysis=AnalysisConfig(
                design=None, transform_dv=np.log2, model=OLS(), scaler=MinMaxScaler()
            ),
        )[0]
        assert np.isfinite(r_zscore.delta_ll)
        assert np.isfinite(r_minmax.delta_ll)
        # Wiring probe: the coefficient is scale-dependent (per-std vs
        # per-range units), so it must differ between the two scalers. ΔLL
        # is not a valid probe, OLS predictions are invariant to affine
        # predictor rescaling, so ΔLL matches under both scalers.
        assert r_zscore.coefficient != r_minmax.coefficient


# ---------------------------------------------------------------------------
# TestRSquared
# ---------------------------------------------------------------------------


class TestRSquared:
    """Test R-squared metric."""

    def test_perfect_prediction(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert _r_squared(y, y) == 1.0

    def test_mean_prediction(self):
        """Predicting the mean gives R-squared = 0."""
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.full_like(y, np.mean(y))
        np.testing.assert_allclose(_r_squared(y, y_pred), 0.0, atol=1e-10)

    def test_worse_than_mean(self):
        """R-squared can be negative when predictions are worse than the mean."""
        y = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([10.0, 20.0, 30.0])
        assert _r_squared(y, y_pred) < 0

    def test_known_value(self):
        """Check against hand-computed R-squared."""
        y = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.1, 1.9, 3.2, 3.8])
        # SS_res = 0.01 + 0.01 + 0.04 + 0.04 = 0.10
        # SS_tot = 2.25 + 0.25 + 0.25 + 2.25 = 5.0
        # R-squared = 1 - 0.10/5.0 = 0.98
        np.testing.assert_allclose(_r_squared(y, y_pred), 0.98, atol=1e-10)


# ---------------------------------------------------------------------------
# TestStatsmodelsOLS
# ---------------------------------------------------------------------------


class TestStatsmodelsOLS:
    """Test StatsmodelsOLS matches statsmodels directly and works in evaluate()."""

    def test_default_solves_correlated_predictors(self):
        x = np.linspace(-1.0, 1.0, 100)
        y = np.tile([-1.0, 1.0], 50)
        X = np.column_stack([x, x + 0.1 * y])

        predicted, sigma = StatsmodelsOLS().fit_predict_with_sigma(y, X, X)

        np.testing.assert_allclose(predicted, y, atol=1e-12)
        assert sigma < 1e-12

    def test_nonzero_penalty_selects_regularized_fit(self):
        import statsmodels.api as sm

        X = np.linspace(-1.0, 1.0, 100)[:, None]
        y = 3.0 * X[:, 0]
        expected = sm.OLS(y, sm.add_constant(X)).fit_regularized(alpha=0.2)

        predicted = StatsmodelsOLS(alpha=0.2).fit_predict(y, X, X)

        np.testing.assert_allclose(predicted, expected.predict(sm.add_constant(X)))
        assert not np.allclose(predicted, y)

    def test_least_squares_sigma_uses_design_rank(self):
        import statsmodels.api as sm

        x = np.linspace(-1.0, 1.0, 100)
        y = 3.0 * x + np.sin(np.arange(len(x)))
        X = x[:, None]
        duplicated = np.column_stack([x, x])
        expected = np.sqrt(sm.OLS(y, sm.add_constant(X)).fit().scale)

        _, sigma = StatsmodelsOLS().fit_predict_with_sigma(y, X, X)
        _, duplicated_sigma = StatsmodelsOLS().fit_predict_with_sigma(
            y, duplicated, duplicated
        )

        np.testing.assert_allclose([sigma, duplicated_sigma], expected, rtol=1e-12)

    def test_explicit_iterative_fit_warns_without_changing_predictions(self):
        import statsmodels.api as sm
        from statsmodels.tools.sm_exceptions import ConvergenceWarning

        x = np.linspace(-1.0, 1.0, 100)
        y = np.tile([-1.0, 1.0], 50)
        X = np.column_stack([x, x + 0.1 * y])
        expected = sm.OLS(y, sm.add_constant(X)).fit_regularized(alpha=0.0)
        assert not expected.converged

        with pytest.warns(ConvergenceWarning, match="did not converge"):
            predicted = StatsmodelsOLS(regularized=True).fit_predict(y, X, X)

        np.testing.assert_array_equal(predicted, expected.predict(sm.add_constant(X)))

    def test_matches_statsmodels_directly(self):
        """StatsmodelsOLS.fit_predict matches manual statsmodels call."""
        import statsmodels.api as sm

        np.random.seed(42)
        n = 50
        X = np.random.randn(n, 3)
        y = X @ np.array([1.0, 2.0, -0.5]) + 0.1 * np.random.randn(n)

        X_train, X_test = X[:40], X[40:]
        y_train = y[:40]

        model = StatsmodelsOLS(alpha=0.0, regularized=True)
        pred, sigma = model.fit_predict_with_sigma(y_train, X_train, X_test)

        # Manual statsmodels
        X_tr_c = sm.add_constant(X_train)
        X_te_c = sm.add_constant(X_test)
        reg = sm.OLS(y_train, X_tr_c).fit_regularized(alpha=0.0, L1_wt=1.0)
        expected = reg.predict(X_te_c)

        np.testing.assert_allclose(pred, expected, rtol=1e-12)
        np.testing.assert_array_equal(pred, model.fit_predict(y_train, X_train, X_test))
        residuals = y_train - reg.predict(X_tr_c)
        expected_sigma = np.sqrt(
            np.sum(residuals**2) / (len(y_train) - X_tr_c.shape[1])
        )
        assert sigma == pytest.approx(expected_sigma, rel=1e-12)

    def test_unregularized_matches_ols(self):
        """StatsmodelsOLS(regularized=False) gives similar results to the OLS."""
        np.random.seed(7)
        n = 60
        X = np.random.randn(n, 2)
        y = X @ np.array([3.0, -1.0]) + 0.5 * np.random.randn(n)

        X_train, X_test = X[:50], X[50:]
        y_train = y[:50]

        sm_model = StatsmodelsOLS(regularized=False)
        ols_model = OLS()

        pred_sm = sm_model.fit_predict(y_train, X_train, X_test)
        pred_ols = ols_model.fit_predict(y_train, X_train, X_test)

        np.testing.assert_allclose(pred_sm, pred_ols, rtol=1e-6)

    def test_in_evaluate(self):
        """StatsmodelsOLS works as analysis_model in evaluate()."""
        ds, fm = _toy_dataset_and_model(n_stimuli=10, n_words=8)

        results = evaluate(
            dataset=ds,
            spec=EvaluationSpec(
                targets=[ds.feature(ds.default_measure)],
                predictors=[ModelFeature(fm, Surprisal())],
                controls=[UnitLength()],
            ),
            analysis=AnalysisConfig(
                design=None,
                transform_dv=np.log2,
                model=StatsmodelsOLS(regularized=False),
            ),
        )
        assert len(results) == 1
        assert np.isfinite(results[0].delta_ll)


# ---------------------------------------------------------------------------
# TestFixedSplitter
# ---------------------------------------------------------------------------


class TestFixedSplitter:
    """Test FixedSplitter yields a single fold from a pre-defined mask."""

    def test_single_fold(self):
        from replica.analysis.cv import FixedSplitter

        mask = np.array([True, True, True, False, False])
        splitter = FixedSplitter(train_mask=mask)
        y = np.arange(5, dtype=float)
        groups = np.arange(5)
        folds = list(splitter.split(y, groups, groups, groups))
        assert len(folds) == 1
        train_mask, test_mask = folds[0]
        np.testing.assert_array_equal(train_mask, mask)
        np.testing.assert_array_equal(test_mask, ~mask)

    def test_from_indices(self):
        from replica.analysis.cv import FixedSplitter

        splitter = FixedSplitter.from_indices(np.array([0, 2, 4]), n_total=6)
        y = np.arange(6, dtype=float)
        groups = np.arange(6)
        folds = list(splitter.split(y, groups, groups, groups))
        train_mask, test_mask = folds[0]
        assert train_mask.sum() == 3
        assert test_mask.sum() == 3
        assert train_mask[0] and train_mask[2] and train_mask[4]
        assert test_mask[1] and test_mask[3] and test_mask[5]

    def test_length_mismatch_raises(self):
        from replica.analysis.cv import FixedSplitter

        splitter = FixedSplitter(train_mask=np.array([True, False]))
        y = np.arange(5, dtype=float)
        groups = np.arange(5)
        with pytest.raises(ValueError, match="does not match"):
            list(splitter.split(y, groups, groups, groups))

    def test_all_train(self):
        from replica.analysis.cv import FixedSplitter

        mask = np.ones(10, dtype=bool)
        splitter = FixedSplitter(train_mask=mask)
        y = np.arange(10, dtype=float)
        groups = np.arange(10)
        folds = list(splitter.split(y, groups, groups, groups))
        train_mask, test_mask = folds[0]
        assert train_mask.all()
        assert not test_mask.any()

    def test_mask_fn(self):
        from replica.analysis.cv import FixedSplitter

        # Train on items 0-2, test on items 3-4
        splitter = FixedSplitter(mask_fn=lambda y, g, p, items: items < 3)
        y = np.arange(5, dtype=float)
        groups = np.arange(5)
        items = np.array([0, 1, 2, 3, 4])
        folds = list(splitter.split(y, groups, groups, items))
        train_mask, test_mask = folds[0]
        assert train_mask.sum() == 3
        assert test_mask.sum() == 2

    def test_from_items(self):
        from replica.analysis.cv import FixedSplitter

        # 10 observations, items [0,0,1,1,2,2,3,3,4,4]
        # Train on items {0, 1, 2}
        items = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
        splitter = FixedSplitter.from_items({0, 1, 2})
        y = np.arange(10, dtype=float)
        groups = items.copy()
        folds = list(splitter.split(y, groups, groups, items))
        train_mask, test_mask = folds[0]
        assert train_mask.sum() == 6  # 2 obs x 3 items
        assert test_mask.sum() == 4  # 2 obs x 2 items
        # First 6 observations (items 0,1,2) should be train
        np.testing.assert_array_equal(train_mask[:6], True)
        np.testing.assert_array_equal(test_mask[6:], True)

    def test_both_args_raises(self):
        from replica.analysis.cv import FixedSplitter

        with pytest.raises(ValueError, match="not both"):
            FixedSplitter(
                train_mask=np.array([True]),
                mask_fn=lambda y, g, p, i: np.ones(len(y), dtype=bool),
            )

    def test_no_args_raises(self):
        from replica.analysis.cv import FixedSplitter

        with pytest.raises(ValueError, match="either"):
            FixedSplitter()


# ---------------------------------------------------------------------------
# TestMultiSeedSplitter
# ---------------------------------------------------------------------------


class TestMultiSeedSplitter:
    """Test MultiSeedSplitter yields correct total folds and metadata."""

    def test_total_folds(self):
        """3 seeds x 5 folds = 15 total splits."""
        splitter = MultiSeedSplitter(
            [0, 1, 2],
            lambda s: ShuffledKFoldSplitter(n_folds=5, seed=s),
        )
        n = 50
        y = np.arange(n, dtype=float)
        dummy = np.zeros(n, dtype=int)
        splits = list(splitter.split(y, dummy, dummy, dummy))
        assert len(splits) == 15

    def test_metadata(self):
        """fold_metadata() returns correct seed and inner_fold_index."""
        splitter = MultiSeedSplitter(
            [10, 20],
            lambda s: ShuffledKFoldSplitter(n_folds=3, seed=s),
        )
        n = 30
        y = np.arange(n, dtype=float)
        dummy = np.zeros(n, dtype=int)
        list(splitter.split(y, dummy, dummy, dummy))

        meta = splitter.fold_metadata()
        assert len(meta) == 6
        assert meta[0] == {"seed": 10, "inner_fold_index": 0}
        assert meta[1] == {"seed": 10, "inner_fold_index": 1}
        assert meta[2] == {"seed": 10, "inner_fold_index": 2}
        assert meta[3] == {"seed": 20, "inner_fold_index": 0}
        assert meta[5] == {"seed": 20, "inner_fold_index": 2}

    def test_metadata_before_split_raises(self):
        """fold_metadata() raises before split() is called."""
        splitter = MultiSeedSplitter([0], lambda s: ShuffledKFoldSplitter(3, s))
        with pytest.raises(RuntimeError):
            splitter.fold_metadata()

    def test_sentence_aware_partitions_survive_wrapping(self):
        y = np.arange(12, dtype=float)
        groups = np.repeat([0, 1], 6)
        participants = np.tile([0, 1], 6)
        sentences = np.tile([0, 1, 2], 4)
        raw_readers = np.where(participants == 0, "20", "3")
        splitter = MultiSeedSplitter(
            [5, 9], lambda seed: SubjectSentenceMod10Splitter(3)
        )
        direct = _cv_splits(
            SubjectSentenceMod10Splitter(3),
            y,
            groups,
            participants,
            groups,
            sentences,
            raw_readers,
        )
        wrapped = _cv_splits(
            splitter, y, groups, participants, groups, sentences, raw_readers
        )
        for actual, expected in zip(wrapped, direct * 2, strict=True):
            np.testing.assert_array_equal(actual, expected)

    def test_named_factor_routing_and_metadata_survive_repeated_splits(self):
        created = []

        def factory(seed):
            inner = CrossedGroupSplitter(2, seed=seed + 1)
            created.append(inner)
            return inner

        splitter = MultiSeedSplitter([5, 9], factory)
        assert splitter.uses_declared_factors and splitter.uses_named_factors
        y = np.zeros(12)
        readers, materials = np.indices((3, 4)).reshape(2, -1)
        groups = {"reader": readers, "material": materials}
        for call in range(2):
            iterator = splitter.split(y, groups, readers, materials)
            first = next(iterator)
            with pytest.raises(RuntimeError, match="Complete"):
                splitter.fold_metadata()
            splits = [first, *iterator]
            assert len(created) == 2 * (call + 1)
            for index, metadata in enumerate(splitter.fold_metadata()):
                seed = 5 if index < 4 else 9
                assert metadata["seed"] == seed
                assert metadata["inner"]["seed"] == seed + 1
                assert metadata["inner"]["axis_names"] == ["reader", "material"]
                train, test = splits[index]
                for labels in groups.values():
                    assert not set(labels[train]) & set(labels[test])

    def test_inconsistent_factory_routing_is_rejected(self):
        with pytest.raises(ValueError, match="at least one seed"):
            MultiSeedSplitter([], lambda seed: RowSplitter(2))
        with pytest.raises(ValueError, match="consistent routing"):
            MultiSeedSplitter(
                [0, 1],
                lambda seed: (
                    RowSplitter(2) if seed == 0 else SubjectSentenceMod10Splitter(2)
                ),
            )
        inner = RowSplitter(2)
        splitter = MultiSeedSplitter([0], lambda seed: inner)
        dummy = np.arange(4)
        list(splitter.split(dummy, dummy, dummy, dummy))
        inner = SubjectSentenceMod10Splitter(2)
        with pytest.raises(ValueError, match="changed its routing"):
            list(splitter.split(dummy, dummy, dummy, dummy))


# ---------------------------------------------------------------------------
# TestPerFold
# ---------------------------------------------------------------------------


class TestPerFold:
    @pytest.mark.parametrize(
        "per_fold, retain",
        [(False, False), (True, False), (True, True)],
        ids=["aggregate-only", "folds", "observations"],
    )
    def test_retention_preserves_metrics_and_aggregation(self, per_fold, retain):
        ds, fm = _toy_dataset_and_model(n_stimuli=5, n_words=8)
        result = evaluate(
            ds,
            EvaluationSpec(
                targets=[ds.feature(ds.default_measure)],
                predictors=[ModelFeature(fm, Surprisal())],
                controls=[UnitLength()],
            ),
            AnalysisConfig(
                design=None,
                transform_dv=np.log2,
                model=OLS(),
                cv=ShuffledKFoldSplitter(n_folds=5, seed=0),
                per_fold=per_fold,
                **({"retain_per_obs_delta_ll": True} if retain else {}),
            ),
        )[0]
        assert {"delta_r2", "baseline_r2", "full_r2"} <= result.metrics.keys()
        assert np.isfinite(result.delta_r2)
        if not per_fold:
            assert result.fold_results is None
            return

        folds = result.fold_results
        assert folds is not None and len(folds) == 5
        for fold in folds:
            assert isinstance(fold, FoldResult)
            assert fold.n_test > 0
            assert np.isfinite(
                [
                    fold.delta_ll,
                    fold.delta_r2,
                    fold.baseline_ll,
                    fold.full_ll,
                    fold.baseline_r2,
                    fold.full_r2,
                ]
            ).all()
            assert fold.mean_delta_ll == pytest.approx(fold.delta_ll / fold.n_test)
            if retain:
                assert fold.per_obs_delta_ll.shape == (fold.n_test,)
                assert np.isfinite(fold.per_obs_delta_ll).all()
                np.testing.assert_allclose(
                    fold.per_obs_delta_ll.sum(), fold.delta_ll, rtol=1e-9, atol=1e-9
                )
            else:
                assert fold.per_obs_delta_ll is None
        np.testing.assert_allclose(
            sum(fold.delta_ll for fold in folds), result.delta_ll, rtol=1e-10
        )
        if retain:
            pooled = np.concatenate([fold.per_obs_delta_ll for fold in folds])
            np.testing.assert_allclose(pooled.sum(), result.delta_ll, rtol=1e-9)

    def test_per_fold_with_multi_seed(self):
        ds, fm = _toy_dataset_and_model(n_stimuli=5, n_words=8)
        results = evaluate(
            dataset=ds,
            spec=EvaluationSpec(
                targets=[ds.feature(ds.default_measure)],
                predictors=[ModelFeature(fm, Surprisal())],
                controls=[UnitLength()],
            ),
            analysis=AnalysisConfig(
                design=None,
                transform_dv=np.log2,
                model=OLS(),
                cv=MultiSeedSplitter(
                    range(3), lambda s: ShuffledKFoldSplitter(5, seed=s)
                ),
                per_fold=True,
            ),
        )
        r = results[0]
        assert r.fold_results is not None
        assert len(r.fold_results) == 15  # 3 seeds x 5 folds
        # Check metadata
        assert r.fold_results[0].metadata["seed"] == 0
        assert r.fold_results[0].metadata["inner_fold_index"] == 0
        assert r.fold_results[5].metadata["seed"] == 1
        assert r.fold_results[5].metadata["inner_fold_index"] == 0
