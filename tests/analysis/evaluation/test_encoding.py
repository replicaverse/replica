"""Tests for encoding model signal processing and CV utilities."""

import random

import numpy as np
import pytest

from replica.analysis import EncodingConfig
from replica.analysis.evaluation.encoding import (
    TemporalKFold,
    _alignment_times,
    _columnwise_pearson,
    _fit_fold,
    _ridge_svd,
    _select_alpha_bootstrap,
    _standardize_columns,
    _zscore_columns,
    lanczos_interpolate,
    make_fir_delays,
)
from replica.core import AnnotationLayer, Span, Stimulus


def test_alignment_times_preserve_nonmonotonic_source_order():
    """Lanczos accepts the local timestamp reversals present in Narratives."""
    spans = [Span(0, 1), Span(2, 3), Span(4, 5)]
    stimulus = Stimulus("a b c", stimulus_id="overlapping-speech")
    stimulus.add_layer(
        AnnotationLayer(
            "word_times",
            spans,
            [(0.0, 0.2), (1.0, 1.2), (0.8, 1.0)],
        )
    )

    np.testing.assert_array_equal(
        _alignment_times(stimulus, "word_times"), [0.1, 1.1, 0.9]
    )


class TestColumnwisePearson:
    def test_matches_numpy_corrcoef(self):
        actual = np.array(
            [[1.0, 4.0, 2.0], [2.0, 3.0, 2.0], [3.0, 2.0, 2.0], [4.0, 1.0, 2.0]]
        )
        predicted = np.array(
            [[2.0, 1.0, 9.0], [4.0, 2.0, 8.0], [6.0, 3.0, 7.0], [8.0, 4.0, 6.0]]
        )

        result = _columnwise_pearson(actual, predicted)
        with np.errstate(invalid="ignore", divide="ignore"):
            expected = np.array(
                [
                    np.corrcoef(actual[:, i], predicted[:, i])[0, 1]
                    for i in range(actual.shape[1])
                ]
            )
        np.testing.assert_allclose(result, expected, equal_nan=True)

    def test_constant_or_nonfinite_column_is_nan(self):
        actual = np.array([[1.0, 5.0, 1.0], [2.0, 5.0, np.nan], [3.0, 5.0, 3.0]])
        predicted = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])

        result = _columnwise_pearson(actual, predicted)

        assert result[0] == 1.0
        assert np.isnan(result[1])
        assert np.isnan(result[2])


class TestLanczosInterpolate:
    def test_identity_same_grid(self):
        """When old and new grids are identical, output ≈ input."""
        rng = np.random.default_rng(42)
        n = 50
        times = np.linspace(0, 10, n)
        data = rng.standard_normal((n, 3))

        result = lanczos_interpolate(data, times, times, window=3)

        assert result.shape == data.shape
        # Interior points should be very close (edge effects at boundaries)
        np.testing.assert_allclose(result[5:-5], data[5:-5], atol=0.05)

    def test_sinusoid_correlation(self):
        """Lanczos preserves sinusoidal shape (high correlation)."""
        # Dense sampling at 20 Hz
        old_times = np.linspace(0, 5, 100)
        freq = 0.5  # 0.5 Hz, well below Nyquist for both grids
        data = np.sin(2 * np.pi * freq * old_times)

        # Resample to 4 Hz grid (interior only, avoid edge effects)
        new_times = np.linspace(1, 4, 12)
        result = lanczos_interpolate(data, old_times, new_times, window=3)

        expected = np.sin(2 * np.pi * freq * new_times)
        r = np.corrcoef(result, expected)[0, 1]
        assert r > 0.99, f"Correlation {r:.4f} too low"

    def test_1d_input(self):
        """A 1-D series interpolates to a 1-D result."""
        old_times = np.arange(10, dtype=float)
        data = np.sin(old_times * 0.5)
        new_times = np.arange(0.5, 9, 1.0)

        result = lanczos_interpolate(data, old_times, new_times)
        assert result.ndim == 1
        assert len(result) == len(new_times)

    def test_output_shape(self):
        """Output has shape (n_new, n_features)."""
        data = np.ones((20, 5))
        old_times = np.arange(20, dtype=float)
        new_times = np.arange(0, 19, 1.5)

        result = lanczos_interpolate(data, old_times, new_times)
        assert result.shape == (len(new_times), 5)


class TestMakeFirDelays:
    def test_shape(self):
        """Output shape = (n_trs, n_feat * n_delays)."""
        data = np.ones((100, 10))
        delays = range(1, 9)
        result = make_fir_delays(data, delays)
        assert result.shape == (100, 10 * 8)

    def test_shift(self):
        """Data appears at correct offset for each delay."""
        n = 20
        data = np.arange(n, dtype=float).reshape(-1, 1)
        delays = [1, 3]
        result = make_fir_delays(data, delays)

        # Delay 1: data[0] appears at row 1, data[1] at row 2, etc.
        assert result[0, 0] == 0.0  # zero-filled gap
        assert result[1, 0] == 0.0  # data[0] shifted by 1
        assert result[2, 0] == 1.0  # data[1] at row 2

        # Delay 3: data[0] appears at row 3
        assert result[0, 1] == 0.0
        assert result[1, 1] == 0.0
        assert result[2, 1] == 0.0
        assert result[3, 1] == 0.0  # data[0] shifted by 3
        assert result[4, 1] == 1.0  # data[1] at row 4

    def test_1d_input(self):
        """A 1-D series yields one delayed column per requested delay."""
        data = np.arange(10, dtype=float)
        result = make_fir_delays(data, [1, 2])
        assert result.shape == (10, 2)

    def test_delay_zero_fill(self):
        """First d rows of each delayed column are zero."""
        data = np.ones((10, 3))
        delays = [2, 5]
        result = make_fir_delays(data, delays)

        # First 2 rows of first block should be zero
        np.testing.assert_array_equal(result[:2, :3], 0.0)
        # First 5 rows of second block should be zero
        np.testing.assert_array_equal(result[:5, 3:6], 0.0)


class TestZscoreColumns:
    def test_mean_zero_std_one(self):
        """After z-scoring, columns have mean ≈ 0 and std ≈ 1."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal((100, 5)) * 10 + 50

        result = _zscore_columns(data)
        np.testing.assert_allclose(result.mean(axis=0), 0.0, atol=1e-12)
        np.testing.assert_allclose(result.std(axis=0), 1.0, atol=1e-12)

    def test_zero_std_column(self):
        """Constant columns become zero (no division by zero)."""
        data = np.array([[1.0, 5.0], [1.0, 10.0], [1.0, 15.0]])
        result = _zscore_columns(data)
        np.testing.assert_array_equal(result[:, 0], 0.0)


class TestTemporalKFold:
    def test_all_indices_in_test(self):
        """Every index appears in exactly one test fold."""
        n_trs = 100
        cv = TemporalKFold(n_folds=4, buffer_trs=5)

        test_indices = []
        for _, test_idx in cv.split(n_trs):
            test_indices.extend(test_idx.tolist())

        assert sorted(test_indices) == list(range(n_trs))

    def test_buffer_excluded(self):
        """Buffer TRs around test fold are excluded from training."""
        n_trs = 100
        buffer = 5
        cv = TemporalKFold(n_folds=4, buffer_trs=buffer)

        for train_idx, test_idx in cv.split(n_trs):
            test_start = test_idx[0]
            test_end = test_idx[-1]

            # Training keeps every sample farther than the buffer from the
            # test block, and nothing else.
            expected = [
                i
                for i in range(n_trs)
                if i < test_start - buffer or i > test_end + buffer
            ]
            assert train_idx.tolist() == expected

            # Specifically: indices just before test fold should not be in train
            for b in range(1, buffer + 1):
                before = test_start - b
                after = test_end + b
                if 0 <= before < n_trs:
                    assert before not in set(train_idx)
                if 0 <= after < n_trs:
                    assert after not in set(train_idx)

    def test_number_of_folds(self):
        """Correct number of folds generated."""
        cv = TemporalKFold(n_folds=5, buffer_trs=3)
        folds = list(cv.split(200))
        assert len(folds) == 5


class TestRidgeSVD:
    def test_recovers_linear(self):
        """Ridge regression recovers a known linear relationship."""
        rng = np.random.default_rng(42)
        n = 200
        d = 5
        n_voxels = 3

        X = rng.standard_normal((n, d))
        true_w = rng.standard_normal((d, n_voxels))
        y = X @ true_w + rng.standard_normal((n, n_voxels)) * 0.1

        # Split into train/test
        X_train, X_test = X[:150], X[150:]
        y_train, y_test = y[:150], y[150:]

        y_pred, _wt = _ridge_svd(X_train, y_train, X_test, alpha=0.01)

        # Predictions should correlate well with actual
        for vi in range(n_voxels):
            r = np.corrcoef(y_test[:, vi], y_pred[:, vi])[0, 1]
            assert r > 0.95, f"Voxel {vi}: r = {r:.3f}"


@pytest.mark.parametrize(
    "kwargs, field",
    [
        ({"fir_delays": (-1, 1)}, "fir_delays"),
        ({"fir_delays": (1.5,)}, "fir_delays"),
        ({"lanczos_window": 0}, "lanczos_window"),
        ({"alpha_chunk_length": 0}, "alpha_chunk_length"),
        ({"alpha_nchunks": 0}, "alpha_nchunks"),
        ({"participants": "p12"}, "participants"),
    ],
)
def test_encoding_config_rejects_settings_that_fail_later(kwargs, field):
    """Each of these would otherwise surface as a broadcasting error, NaN
    correlations, a disabled alpha search, or a substring participant match."""
    with pytest.raises(ValueError, match=field):
        EncodingConfig(**kwargs)


class TestAlphaSelection:
    def _data(self, n_rows):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(n_rows, 8))
        y = X @ rng.normal(size=(8, 3)) + 0.1 * rng.normal(size=(n_rows, 3))
        return _standardize_columns(X)[0], _standardize_columns(y)[0]

    def test_too_few_chunks_falls_back_to_the_middle_alpha_and_says_so(self):
        alphas = np.array([1.0, 10.0, 100.0])
        X, y = self._data(700)  # 17 complete chunks of 40, fewer than 20
        with pytest.warns(UserWarning, match="alpha_nchunks"):
            alpha, selection = _select_alpha_bootstrap(X, y, alphas)
        assert (alpha, selection) == (10.0, "fallback")

    def test_enough_chunks_select_on_the_held_out_split(self):
        X, y = self._data(1000)
        alpha, selection = _select_alpha_bootstrap(X, y, np.array([1.0, 10.0, 100.0]))
        assert selection == "held_out"
        assert alpha in (1.0, 10.0, 100.0)

    def test_holdout_covering_every_row_cannot_select_an_alpha(self):
        X, y = self._data(8)
        with pytest.warns(UserWarning, match="remaining training rows"):
            result = _select_alpha_bootstrap(
                X, y, np.array([1.0, 10.0, 100.0]), chunklen=4, nchunks=2
            )
        assert result == (10.0, "fallback")


def test_alpha_selection_fits_normalization_inside_its_training_partition(monkeypatch):
    X = np.arange(24, dtype=float).reshape(12, 2)
    X[:, 1] **= 2
    y = X.sum(axis=1, keepdims=True)
    matrices = []
    original_svd = np.linalg.svd

    def capture(matrix, **kwargs):
        matrices.append(matrix.copy())
        return original_svd(matrix, **kwargs)

    monkeypatch.setattr(np.linalg, "svd", capture)
    rng = random.Random(0)
    monkeypatch.setattr(rng, "shuffle", lambda chunks: None)
    config = EncodingConfig(alphas=(1.0, 3.0), alpha_chunk_length=2, alpha_nchunks=1)
    _fit_fold(X[:10], X[10:] + 100, y[:10], y[10:], config, rng)
    inner_train = X[2:10]
    expected = (inner_train - inner_train.mean(axis=0)) / inner_train.std(axis=0)
    np.testing.assert_array_equal(matrices[0], expected)
