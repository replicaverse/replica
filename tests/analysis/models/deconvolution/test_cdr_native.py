"""Unit tests for the native CDR machinery (kernels, optimizer, plumbing)."""

import warnings
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from replica.analysis.models.deconvolution.data import (
    apply_reference_cutoff,
    build_design,
    build_design_from_events,
    build_multistream_design,
    compute_history_intervals,
    expand_history,
)
from replica.analysis.models.deconvolution.kernels import (
    EPSILON,
    constrain_lower,
    constrain_lower_inverse,
    double_gamma_irf,
    exp_irf,
    normal_irf,
    shifted_gamma_kgt1_irf,
    softplus_inverse,
)
from replica.analysis.models.deconvolution.model import CDRModel, DiracTerm, KernelTerm
from replica.analysis.models.deconvolution.train import (
    EMA,
    ConvergenceDiagnostic,
    TFNadam,
    enable_deterministic_training,
    held_out_loglik,
)

_FIXTURES = Path(__file__).parent / "fixtures"


def _nonfinite_objective(model):
    """Build an objective that preserves a gradient while returning NaNs."""

    def objective(design, idx=None, **_):
        n = len(design.y) if idx is None else len(idx)
        return model.coef.sum() * torch.full((n,), float("nan"))

    return objective


def test_seeded_training_enables_deterministic_torch_operations():
    previous = torch.are_deterministic_algorithms_enabled()
    try:
        enable_deterministic_training()
        assert torch.are_deterministic_algorithms_enabled()
    finally:
        torch.use_deterministic_algorithms(previous)


@pytest.mark.parametrize("enabled, warn_only", [(False, False), (True, True)])
def test_a_seeded_fit_restores_the_callers_determinism_setting(enabled, warn_only):
    """Deterministic kernels apply to the seeded fit only; afterwards the
    process has the setting it had before the fit."""
    from replica.analysis.models.deconvolution.train import fit_cdr

    n = 32
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {
            "subject": ["s"] * n,
            "docid": ["d"] * n,
            "time": np.arange(n) * 0.25,
            "a": rng.standard_normal(n),
        }
    )
    y = X[["subject", "docid", "time"]].copy()
    y["rt"] = rng.standard_normal(n)
    design = build_design(
        X,
        y,
        impulse_names=["a"],
        dv="rt",
        series_ids=["subject", "docid"],
        history_length=4,
    )
    previous = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )
    try:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
        fit_cdr(
            CDRModel([KernelTerm("a", "Exp")], [], ["a"]),
            design,
            n_iter=1,
            minibatch_size=16,
            check_convergence=False,
            seed=0,
        )
        assert torch.are_deterministic_algorithms_enabled() is enabled
        assert torch.is_deterministic_algorithms_warn_only_enabled() is warn_only
    finally:
        torch.use_deterministic_algorithms(previous[0], warn_only=previous[1])


def test_held_out_loglik_uses_bounded_batches():
    class Model:
        def __init__(self):
            self.batch_lengths = []

        def substituted(self, estimate):
            assert estimate is None
            return nullcontext()

        def log_lik(self, design, idx=None):
            length = len(design.y[idx])
            self.batch_lengths.append(length)
            return torch.ones(length)

    class MovingAverage:
        swaps = []

        def swap_in(self):
            self.swaps.append("in")

        def swap_out(self):
            self.swaps.append("out")

    model = Model()
    moving_average = MovingAverage()
    design = SimpleNamespace(y=np.zeros(5))
    assert held_out_loglik(model, design, moving_average, batch_size=2) == 5
    assert model.batch_lengths == [2, 2, 1]
    assert moving_average.swaps == ["in", "out"]


class TestKernels:
    def test_exp_matches_closed_form(self):
        x = torch.linspace(0.01, 5, 50)
        beta = torch.tensor(1.7)
        expected = 1.7 * np.exp(-1.7 * x.numpy())
        np.testing.assert_allclose(exp_irf(x, beta).numpy(), expected, rtol=1e-6)

    def test_shifted_gamma_matches_scipy(self):
        from scipy import stats

        x = torch.linspace(0.01, 5, 50)
        alpha, beta, delta = 2.5, 1.3, -0.4
        eps = float(EPSILON)
        expected = stats.gamma.pdf(x.numpy() - delta + eps, alpha, scale=1 / beta) / (
            1 - stats.gamma.cdf(-delta + eps, alpha, scale=1 / beta) + eps
        )
        got = shifted_gamma_kgt1_irf(
            x, torch.tensor(alpha), torch.tensor(beta), torch.tensor(delta)
        )
        np.testing.assert_allclose(got.numpy(), expected, rtol=1e-5)

    def test_shain2018_shifted_gamma_uses_the_unnormalized_density(self):
        """Shain & Schuler (2018) use the raw shifted density without truncation."""
        from scipy import stats

        x = torch.linspace(0.01, 5, 50)
        alpha, beta, delta = 2.5, 1.3, -0.4
        expected = stats.gamma.pdf(
            x.numpy() - delta + float(EPSILON), alpha, scale=1 / beta
        )
        got = shifted_gamma_kgt1_irf(
            x,
            torch.tensor(alpha),
            torch.tensor(beta),
            torch.tensor(delta),
            normalize=False,
        )
        np.testing.assert_allclose(got.numpy(), expected, rtol=1e-5)

    def test_normal_is_truncated_density(self):
        from scipy import stats

        x = torch.linspace(0.01, 5, 50)
        mu, sd = 0.7, 0.9
        eps = float(EPSILON)
        expected = stats.norm.pdf(x.numpy(), mu, sd) / (
            1 - stats.norm.cdf(0, mu, sd) + eps
        )
        got = normal_irf(x, torch.tensor(mu), torch.tensor(sd))
        np.testing.assert_allclose(got.numpy(), expected, rtol=1e-5)

    def test_constraint_round_trip(self):
        for theta, lb in [(2.0, 1.0), (1.0, 0.0), (0.03, 0.0)]:
            raw = constrain_lower_inverse(theta, lb)
            back = constrain_lower(raw, lb)
            np.testing.assert_allclose(float(back), theta, rtol=1e-6)

    def test_softplus_inverse_inverts(self):
        x = torch.tensor([0.1, 1.0, 5.0])
        np.testing.assert_allclose(
            torch.nn.functional.softplus(softplus_inverse(x)).numpy(),
            x.numpy(),
            rtol=1e-6,
        )

    def test_shifted_gamma_alpha_gradient_skips_normalizer(self):
        alpha = torch.tensor(2.0, requires_grad=True)
        beta = torch.tensor(1.0)
        delta = torch.tensor(-0.5)
        out = shifted_gamma_kgt1_irf(torch.tensor([1.0]), alpha, beta, delta)
        out.sum().backward()
        assert alpha.grad is not None and torch.isfinite(alpha.grad)

    def test_lcg_matches_explicit_mixture(self):
        from scipy import stats

        from replica.analysis.models.deconvolution.kernels import lcg_irf

        n = 10
        locs = np.linspace(0.0, 1.0, n)
        values = np.linspace(1.0, 0.1, n)
        scales = np.arange(1, n + 1) * 0.5
        x = np.linspace(0.01, 3.0, 7)
        numerator = sum(
            values[k] * stats.norm.pdf(x, locs[k], scales[k]) for k in range(n)
        )
        denominator = sum(
            values[k] * (1 - stats.norm.cdf(0.0, locs[k], scales[k])) for k in range(n)
        )
        got = lcg_irf(
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(locs, dtype=torch.float32),
            torch.tensor(values, dtype=torch.float32),
            torch.tensor(scales, dtype=torch.float32),
        )
        np.testing.assert_allclose(got.numpy(), numerator / denominator, rtol=1e-5)

    def test_lcg_family_layout_follows_the_reference(self):
        """Ten bases as locations, then values, then scales; only scales bounded."""
        from replica.analysis.models.deconvolution.model import FAMILIES, LCG_BASES

        names, bounds, defaults = FAMILIES["LCG"]
        assert len(names) == 3 * LCG_BASES
        assert names[0] == "x1" and names[LCG_BASES] == "y1"
        assert names[2 * LCG_BASES] == "s1"
        assert bounds[0] is None and bounds[LCG_BASES] is None
        assert bounds[2 * LCG_BASES] == 0.0
        assert defaults[:LCG_BASES] == (0.0,) * LCG_BASES
        assert defaults[LCG_BASES] == 1.0
        assert defaults[LCG_BASES + 1] == 0.0
        assert defaults[2 * LCG_BASES :] == tuple(
            float(i) for i in range(1, LCG_BASES + 1)
        )

    def test_double_gamma_matches_explicit_scipy_densities(self):
        from scipy import stats

        x = np.linspace(0.0, 30.0, 101, dtype=np.float32)
        alpha_main, alpha_under = 6.0, 16.0
        beta_main, beta_under, c = 1.1, 0.9, 1.0 / 6.0
        expected = (
            stats.gamma.pdf(x + float(EPSILON), alpha_main, scale=1 / beta_main)
            - c * stats.gamma.pdf(x + float(EPSILON), alpha_under, scale=1 / beta_under)
        ) / (1 - c)
        got = double_gamma_irf(
            torch.tensor(x),
            torch.tensor(alpha_main),
            torch.tensor(alpha_under),
            torch.tensor(beta_main),
            torch.tensor(beta_under),
            torch.tensor(c),
        )
        np.testing.assert_allclose(got.numpy(), expected, rtol=5e-5, atol=5e-8)

    @pytest.mark.parametrize(
        "family",
        [
            "HRFDoubleGamma1",
            "HRFDoubleGamma2",
            "HRFDoubleGamma3",
            "HRFDoubleGamma4",
            "HRFDoubleGamma5",
        ],
    )
    def test_released_hrf_families_are_finite_at_default_init(self, family):
        design = TestFit._design(n=12)
        model = CDRModel([KernelTerm("a", family)], [], ["a"])
        got = model.convolved(design).detach()
        assert got.shape == (12, 1)
        assert torch.isfinite(got).all()


class TestTFNadam:
    def test_matches_recorded_tf16_trajectory(self):
        """Five optimizer steps against a trajectory recorded from
        tf.contrib.opt.NadamOptimizer in the pinned cognition21 environment
        (same quadratic-plus-l1 objective, lr 0.001)."""
        ref = np.load(_FIXTURES / "tf16_nadam_trajectory.npy")
        w = torch.nn.Parameter(torch.tensor([1.0, -2.0, 0.5]))
        x = torch.tensor([0.3, -1.2, 2.0])
        opt = TFNadam([w], lr=0.001)
        traj = []
        for _ in range(5):
            opt.zero_grad()
            loss = ((w * x - 1.0) ** 2).sum() + 0.1 * w.abs().sum()
            loss.backward()
            opt.step()
            traj.append(w.detach().numpy().copy())
        np.testing.assert_allclose(np.array(traj), ref, rtol=0, atol=1e-6)


class TestEMA:
    def test_shadow_update_and_swap(self):
        p = torch.nn.Parameter(torch.tensor([2.0]))
        ema = EMA([p], decay=0.9)
        with torch.no_grad():
            p.fill_(4.0)
        ema.update()
        np.testing.assert_allclose(ema.shadow[0].numpy(), [0.9 * 2 + 0.1 * 4])
        ema.swap_in()
        np.testing.assert_allclose(p.detach().numpy(), [2.2])
        ema.swap_out()
        np.testing.assert_allclose(p.detach().numpy(), [4.0])


class TestConvergenceDiagnostic:
    def test_first_update_has_too_few_points_for_a_correlation(self):
        diag = ConvergenceDiagnostic(n_iterates=50)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = diag.update(1.0)
            assert type(result) is bool
            assert not result
            assert not caught

    def test_declining_loss_not_converged(self):
        diag = ConvergenceDiagnostic(n_iterates=50)
        converged = False
        for i in range(120):
            converged = diag.update(100.0 - i)
        assert not converged

    def test_flat_noisy_loss_converges(self):
        rng = np.random.default_rng(0)
        diag = ConvergenceDiagnostic(n_iterates=50)
        converged = False
        for _ in range(200):
            converged = converged or diag.update(1.0 + rng.normal(0, 0.01))
        assert converged

    def test_autocorrelation_diagnostic_does_not_gate_convergence(self):
        class Controlled(ConvergenceDiagnostic):
            calls = 0

            def _corr_p(self, x, y, df):
                del x, y, df
                self.calls += 1
                return (0.0, 1.0) if self.calls % 2 else (0.0, 0.0)

        diag = Controlled(n_iterates=4)
        assert any(diag.update(float(i)) for i in range(12))


class TestReferenceCutoff:
    """The cutoff reaches back the way the reference's pointer reaches back.

    Both fixtures below are the output of the pinned reference's
    ``cdr.data.get_time_windows`` on the same stream, with
    ``window_length=8`` and ``t_delta_cutoff=60``.
    """

    #: One series with a 66-unit gap, then a second series with a 85-unit gap.
    IMPULSE_SERIES = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
    IMPULSE_TIME = [0.0, 1.0, 2.0, 3.0, 4.0, 70.0, 71.0, 72.0, 0.0, 5.0, 90.0, 96.0]
    RESPONSE_SERIES = [0, 0, 0, 0, 0, 1, 1, 1]
    RESPONSE_TIME = [2.5, 4.0, 70.5, 71.5, 72.5, 5.0, 90.0, 95.0]
    REFERENCE_FIRST = [0, 0, 6, 6, 6, 8, 11, 11]
    REFERENCE_LAST = [3, 5, 6, 7, 8, 10, 11, 11]

    def _last_obs(self):
        return np.asarray(self.REFERENCE_LAST, dtype=np.int64)

    def test_pointer_matches_the_reference_windows(self):
        first = np.zeros(len(self.RESPONSE_TIME), dtype=np.int64)
        last = self._last_obs()
        apply_reference_cutoff(
            first_obs=first,
            last_obs=last,
            response_time=np.asarray(self.RESPONSE_TIME, dtype=float),
            response_series=np.asarray(self.RESPONSE_SERIES),
            impulse_time=np.asarray(self.IMPULSE_TIME, dtype=float),
            series_starts={0: 0, 1: 8},
            t_delta_cutoff=60.0,
        )

        assert first.tolist() == self.REFERENCE_FIRST

    def test_an_exact_cutoff_would_disagree(self):
        """The two rules differ, so the anchor above distinguishes them."""
        times = np.asarray(self.IMPULSE_TIME, dtype=float)
        exact = [
            int(np.searchsorted(times[:8] if s == 0 else times[8:], t - 60.0))
            + (0 if s == 0 else 8)
            for s, t in zip(self.RESPONSE_SERIES, self.RESPONSE_TIME, strict=True)
        ]

        assert exact != self.REFERENCE_FIRST

    def test_a_design_build_uses_the_reference_windows(self):
        design = build_design_from_events(
            response_time=np.asarray(self.RESPONSE_TIME, dtype=float),
            response_series=np.asarray(self.RESPONSE_SERIES),
            response_y=np.arange(len(self.RESPONSE_TIME), dtype=float),
            response_group=None,
            impulse_time=np.asarray(self.IMPULSE_TIME, dtype=float),
            impulse_series=np.asarray(self.IMPULSE_SERIES),
            impulse_values=np.ones((len(self.IMPULSE_TIME), 1), dtype=float),
            impulse_names=["a"],
            history_length=8,
            t_delta_cutoff=60.0,
        )

        widths = design.time_mask[:, :, 0].sum(axis=1).astype(int).tolist()
        expected = [
            last - first
            for first, last in zip(
                self.REFERENCE_FIRST, self.REFERENCE_LAST, strict=True
            )
        ]
        assert widths == expected

    def test_no_cutoff_keeps_the_whole_series_history(self):
        design = build_design_from_events(
            response_time=np.asarray(self.RESPONSE_TIME, dtype=float),
            response_series=np.asarray(self.RESPONSE_SERIES),
            response_y=np.arange(len(self.RESPONSE_TIME), dtype=float),
            response_group=None,
            impulse_time=np.asarray(self.IMPULSE_TIME, dtype=float),
            impulse_series=np.asarray(self.IMPULSE_SERIES),
            impulse_values=np.ones((len(self.IMPULSE_TIME), 1), dtype=float),
            impulse_names=["a"],
            history_length=8,
        )

        widths = design.time_mask[:, :, 0].sum(axis=1).astype(int).tolist()
        assert widths == [3, 5, 6, 7, 8, 2, 3, 3]


class TestEarlyStoppingDesign:
    def test_dev_groups_use_the_folds_training_levels(self, monkeypatch):
        """A participant the fold never trained on reads the unknown row.

        The dataset's dense codes and the fold's re-encoded ones only coincide
        when every level is in the training partition, so an unmapped dev
        design silently indexes another participant's offsets.
        """
        from types import SimpleNamespace

        import replica.analysis.models.deconvolution.cdrnn as cdrnn_module
        from replica.analysis.models.deconvolution.cdrnn import CDRNNAnalysis

        captured = {}

        def _capture(**kwargs):
            captured.update(kwargs)
            return "design"

        monkeypatch.setattr(cdrnn_module, "build_design_from_events", _capture)
        monkeypatch.setattr(cdrnn_module, "design_on", lambda design, device: design)

        # Dataset codes 0..3; the fold trains on 1 and 3 only.
        dev = SimpleNamespace(
            time=np.array([0.0, 1.0, 2.0]),
            series=np.array([0, 0, 0]),
            y=np.array([1.0, 2.0, 3.0]),
            group=np.array([1, 3, 2]),
            item=None,
        )
        impulses = SimpleNamespace(
            index=SimpleNamespace(time=np.array([0.0]), series=np.array([0])),
            matrix=lambda names: np.zeros((1, len(names))),
        )
        events = SimpleNamespace(
            y=np.ones(4),
            group=np.array([1, 3, 2, 1]),
            impulses=impulses,
            subset=lambda mask: dev,
        )
        analysis = CDRNNAnalysis(random_item_intercept=False)
        train_design = SimpleNamespace(y_mean=0.0, y_sd=1.0)

        analysis._dev_design(
            events,
            np.array([True, True, False, False]),
            np.array([False, False, True, False]),
            train_design,
            ["a"],
            {},
            0,
            {1: 0, 3: 1},
            2,
        )

        assert captured["response_group"].tolist() == [0, 1, 2]


class TestDataPlumbing:
    def _streams(self):
        X = pd.DataFrame(
            {
                "subject": ["s1"] * 4 + ["s2"] * 2,
                "docid": ["d1"] * 4 + ["d1"] * 2,
                "time": [0.0, 1.0, 2.0, 3.0, 0.0, 1.0],
                "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            }
        )
        y = pd.DataFrame(
            {
                "subject": ["s1", "s1", "s2"],
                "docid": ["d1", "d1", "d1"],
                "time": [1.5, 3.0, 1.0],
                "rt": [10.0, 11.0, 12.0],
            }
        )
        return X, y

    def test_history_intervals(self):
        X, y = self._streams()
        first, last = compute_history_intervals(X, y, ["subject", "docid"])
        np.testing.assert_array_equal(first, [0, 0, 4])
        np.testing.assert_array_equal(last, [2, 4, 6])

    def test_expand_history_right_aligned(self):
        X, y = self._streams()
        first, last = compute_history_intervals(X, y, ["subject", "docid"])
        vals = X[["a"]].to_numpy(np.float32)
        times = X["time"].to_numpy(np.float32)
        X_2d, time_X_2d, mask = expand_history(vals, times, first, last, 3)
        np.testing.assert_array_equal(X_2d[0, :, 0], [0.0, 1.0, 2.0])
        np.testing.assert_array_equal(mask[0, :, 0], [0.0, 1.0, 1.0])
        np.testing.assert_array_equal(X_2d[1, :, 0], [2.0, 3.0, 4.0])
        np.testing.assert_array_equal(time_X_2d[1, :, 0], [1.0, 2.0, 3.0])

    def test_nonfinite_minibatch_loss_does_not_poison_the_epoch(self):
        from replica.analysis.models.deconvolution.train import finite_or_zero

        assert finite_or_zero(float("nan")) == 0.0
        assert finite_or_zero(float("inf")) == 0.0
        assert finite_or_zero(12.5) == 12.5

    def test_an_entire_nonfinite_epoch_fails_fast(self, monkeypatch):
        from replica.analysis.models.deconvolution.train import fit_cdr

        design = TestFit._design(n=16)
        model = CDRModel([KernelTerm("a", "Exp")], [], ["a"])

        monkeypatch.setattr(model, "objective_ll", _nonfinite_objective(model))
        with pytest.raises(
            FloatingPointError,
            match=(
                r"all 2 minibatch losses.*first non-finite update at "
                r"iteration 1, minibatch 1: loss=nan.*non-finite gradients="
                r"\('coef',\)"
            ),
        ):
            fit_cdr(model, design, n_iter=10, minibatch_size=8, seed=0)

    def test_deferred_readback_rejects_an_entire_nonfinite_epoch(self, monkeypatch):
        from replica.analysis.models.deconvolution.train import fit_cdr

        design = TestFit._design(n=16)
        model = CDRModel([KernelTerm("a", "Exp")], [], ["a"])

        monkeypatch.setattr(model, "objective_ll", _nonfinite_objective(model))
        with pytest.raises(
            FloatingPointError,
            match=(
                r"all 2 minibatch losses.*deferred loss readback found a "
                r"non-finite minibatch.*non-finite parameters after the epoch="
            ),
        ):
            fit_cdr(
                model,
                design,
                n_iter=10,
                minibatch_size=8,
                seed=0,
                defer_loss_readback=True,
            )

    def test_dirac_gate_requires_simultaneity(self):
        X, y = self._streams()
        design = build_design(
            X,
            y,
            impulse_names=["a"],
            dv="rt",
            series_ids=["subject", "docid"],
            history_length=3,
        )
        model = CDRModel([], [DiracTerm("a")], ["a"])
        with torch.no_grad():
            model.intercept.zero_()
            model.coef.fill_(1.0)
        out = model.forward(design).detach()
        # Rows 2 and 3 are exactly simultaneous with their newest impulse
        # (row 1 after the y.time clamp is not; 1.5 > impulse time 1.0).
        np.testing.assert_allclose(out.numpy(), [0.0, 4.0, 6.0], rtol=1e-6)

    def test_multistream_and_response_aligned_impulses(self):
        y = pd.DataFrame(
            {
                "subject": ["s", "s"],
                "docid": ["d", "d"],
                "time": [1.0, 2.0],
                "rt": [10.0, 12.0],
                "tr": [3.0, 4.0],
            }
        )
        words = pd.DataFrame(
            {
                "subject": ["s", "s"],
                "docid": ["d", "d"],
                "time": [0.0, 2.0],
                "rate": [1.0, 1.0],
            }
        )
        sound = pd.DataFrame(
            {
                "subject": ["s", "s", "s"],
                "docid": ["d", "d", "d"],
                "time": [0.5, 1.5, 2.0],
                "power": [0.1, 0.2, 0.3],
            }
        )
        design = build_multistream_design(
            [words, sound],
            y,
            impulse_sources={"rate": 0, "power": 1, "tr": None},
            dv="rt",
            series_ids=["subject", "docid"],
            history_length=3,
        )
        np.testing.assert_array_equal(design.X_2d[0, :, 0], [0.0, 0.0, 1.0])
        np.testing.assert_allclose(design.X_2d[1, :, 1], [0.1, 0.2, 0.3])
        np.testing.assert_array_equal(design.X_2d[:, -1, 2], [3.0, 4.0])
        np.testing.assert_array_equal(design.time_X_2d[:, -1, 2], [1.0, 2.0])

    def test_second_group_random_intercept_centers_and_handles_unknown(self):
        X, y = self._streams()
        y["site"] = ["x", "y", "new"]
        design = build_design(
            X,
            y,
            impulse_names=["a"],
            dv="rt",
            series_ids=["subject", "docid"],
            history_length=3,
            rangf=["subject", "site"],
            rangf_levels={"subject": ["s1", "s2"], "site": ["x", "y"]},
        )
        model = CDRModel(
            [],
            [DiracTerm("a")],
            ["a"],
            extra_random_intercept_n_levels=(2,),
        )
        with torch.no_grad():
            model.intercept.zero_()
            model.coef.zero_()
            model.intercept_random_extra[0].copy_(torch.tensor([1.0, 3.0]))
        got = model.forward(design).detach().numpy()
        np.testing.assert_allclose(got, [-1.0, 1.0, 0.0], atol=1e-6)

    def test_shared_irf_uses_one_fixed_and_random_parameter_tensor(self):
        model = CDRModel(
            [
                KernelTerm("a", "HRFDoubleGamma5", ran=True, irf_id="HRF"),
                KernelTerm("b", "HRFDoubleGamma5", ran=True, irf_id="HRF"),
            ],
            [],
            ["a", "b"],
            rangf_n_levels=6,
            random_intercept=True,
            random_coefs=True,
            extra_random_intercept_n_levels=(78,),
        )
        assert model.kernel_parameter_index == [0, 0]
        assert len(model.irf_raw) == len(model.irf_raw_random) == 1
        assert (
            model.constrained_params(0).data_ptr()
            != model.constrained_params(1).data_ptr()
        )
        np.testing.assert_allclose(
            model.constrained_params(0).detach().numpy(),
            model.constrained_params(1).detach().numpy(),
        )
        # 1 fixed intercept + 2 slopes + 1 noise scale + 5 HRF parameters,
        # 6 fROI intercepts + 6*2 fROI slopes + 6*5 fROI HRF offsets, and
        # 78 subject intercepts.
        assert sum(p.numel() for p in model.parameters()) == 135

    def test_random_effects_center_to_zero_mean(self):
        X, y = self._streams()
        design = build_design(
            X,
            y,
            impulse_names=["a"],
            dv="rt",
            series_ids=["subject", "docid"],
            history_length=3,
            rangf=["subject"],
        )
        model = CDRModel(
            [KernelTerm("a", "Exp", ran=True)],
            [],
            ["a"],
            rangf_n_levels=2,
            random_intercept=True,
            random_coefs=True,
        )
        with torch.no_grad():
            model.intercept_random.copy_(torch.tensor([1.0, 3.0]))
        out_known = model.forward(design).detach()
        with torch.no_grad():
            model.intercept_random.add_(5.0)
        out_shifted = model.forward(design).detach()
        np.testing.assert_allclose(out_known.numpy(), out_shifted.numpy(), atol=1e-6)


class TestPenalty:
    def test_random_offsets_are_seeded_at_construction(self):
        def offsets(seed):
            model = CDRModel(
                [KernelTerm("a", "Exp", ran=True)],
                [],
                ["a"],
                rangf_n_levels=3,
                seed=seed,
            )
            return model.irf_raw_random[0].detach().numpy()

        np.testing.assert_array_equal(offsets(4), offsets(4))
        assert not np.array_equal(offsets(4), offsets(5))

    def test_ranef_penalty_is_scaled_half_sum_of_squares(self):
        model = CDRModel(
            [KernelTerm("a", "Exp")],
            [],
            ["a"],
            rangf_n_levels=3,
            random_intercept=True,
        )
        with torch.no_grad():
            model.intercept_random.copy_(torch.tensor([1.0, -1.0, 0.0]))
        penalty = model.ranef_penalty(2.0)
        np.testing.assert_allclose(float(penalty), 2.0 * 0.5 * 2.0, rtol=1e-6)


@pytest.mark.parametrize("family", ["Exp", "Normal", "ShiftedGammaShapeGT1", "LCG"])
def test_default_inits_round_trip(family):
    model = CDRModel([KernelTerm("a", family)], [], ["a"])
    from replica.analysis.models.deconvolution.model import FAMILIES

    _, _, defaults = FAMILIES[family]
    got = model.constrained_params(0).detach().numpy()
    np.testing.assert_allclose(got, defaults, rtol=1e-5, atol=1e-6)


class TestFit:
    """End-to-end trainer behavior on a small, self-generated stream."""

    @staticmethod
    def _design(beta_true=1.5, coef_true=2.0, n=500, seed=0):
        from replica.analysis.models.deconvolution.data import build_design

        rng = np.random.default_rng(seed)
        times = np.arange(n, dtype=float) * 0.25
        a = rng.standard_normal(n)
        X = pd.DataFrame(
            {"subject": ["s"] * n, "docid": ["d"] * n, "time": times, "a": a}
        )
        # Response at each impulse: the true mean convolves the impulse
        # history with an exponential kernel of rate beta_true.
        conv = np.zeros(n)
        for i in range(n):
            lo = max(0, i - 15)
            dt = times[i] - times[lo : i + 1]
            conv[i] = np.sum(a[lo : i + 1] * beta_true * np.exp(-beta_true * dt))
        y = pd.DataFrame(
            {
                "subject": ["s"] * n,
                "docid": ["d"] * n,
                "time": times,
                "rt": 5.0 + coef_true * conv + 0.05 * rng.standard_normal(n),
            }
        )
        return build_design(
            X,
            y,
            impulse_names=["a"],
            dv="rt",
            series_ids=["subject", "docid"],
            history_length=16,
        )

    def test_deferred_loss_readback_preserves_variational_updates(self):
        from replica.analysis.models.deconvolution.train import fit_cdr
        from replica.analysis.models.deconvolution.variational import (
            VariationalPosterior,
        )

        design = self._design(n=96)

        def run(defer):
            model = CDRModel([KernelTerm("a", "Exp")], [], ["a"])
            posterior = VariationalPosterior(model, seed=3)
            result, ema = fit_cdr(
                model,
                design,
                n_iter=3,
                minibatch_size=32,
                check_convergence=False,
                seed=7,
                posterior=posterior,
                defer_loss_readback=defer,
            )
            parameters = {
                name: value.detach().clone()
                for name, value in posterior.named_parameters()
            }
            shadows = [value.detach().clone() for value in ema.shadow]
            return result, parameters, shadows

        eager, eager_parameters, eager_shadows = run(False)
        deferred, deferred_parameters, deferred_shadows = run(True)
        assert deferred == eager
        assert deferred_parameters.keys() == eager_parameters.keys()
        assert all(
            torch.equal(deferred_parameters[name], eager_parameters[name])
            for name in eager_parameters
        )
        assert all(
            torch.equal(deferred_value, eager_value)
            for deferred_value, eager_value in zip(
                deferred_shadows, eager_shadows, strict=True
            )
        )

    @pytest.mark.slow
    def test_fit_recovers_known_exp_kernel(self):
        """Recovery of a known kernel from a generated stream.

        Runs at the reference's own learning-rate scale: a much larger rate
        makes the reference's loss-slope rule stop on a noisy plateau before
        the noise scale has settled.
        """
        from replica.analysis.models.deconvolution.train import fit_cdr

        design = self._design()
        model = CDRModel([KernelTerm("a", "Exp")], [], ["a"])
        start = -float(model.objective_ll(design).sum().detach())
        result, ema = fit_cdr(
            model,
            design,
            n_iter=600,
            learning_rate=0.005,
            minibatch_size=128,
            convergence_n_iterates=100,
            seed=0,
        )
        assert result.n_iterations > 0
        ema.swap_in()
        end = -float(model.objective_ll(design).sum().detach())
        assert end < start / 2, "training did not substantially reduce the loss"
        # The response scale is standardized, so the fitted coefficient is
        # compared after rescaling by the training response SD.
        coef = float(model.coef[0].detach()) * design.y_sd
        beta = float(model.constrained_params(0)[0].detach())
        assert abs(coef - 2.0) < 0.5, f"coefficient {coef} far from truth"
        assert abs(beta - 1.5) < 0.5, f"kernel rate {beta} far from truth"


class TestParameterSubstitution:
    """The forward pass can run on values other than the stored parameters."""

    @staticmethod
    def _model():
        return CDRModel(
            [KernelTerm("a", "Exp", ran=True)],
            [DiracTerm("b")],
            ["a", "b"],
            rangf_n_levels=3,
            random_intercept=True,
            random_coefs=True,
        )

    def test_substituted_values_are_read_instead(self):
        model = self._model()
        replacement = torch.full_like(model.coef, 7.0)
        with model.substituted({"coef": replacement}):
            assert torch.equal(model.parameter("coef"), replacement)
            assert torch.equal(model.parameter("intercept"), model.intercept)
        assert torch.equal(model.parameter("coef"), model.coef)

    def test_substitution_does_not_write_the_parameters(self):
        model = self._model()
        before = model.coef.detach().clone()
        with model.substituted({"coef": torch.full_like(model.coef, 7.0)}):
            pass
        assert torch.equal(model.coef.detach(), before)

    def test_every_trainable_parameter_is_addressable(self):
        model = self._model()
        for name, param in model.named_parameters():
            assert model.parameter(name) is param


class TestVariationalPosterior:
    """The reference's Bayesian estimator: normal posteriors, analytic divergence."""

    @staticmethod
    def _model():
        return CDRModel(
            [KernelTerm("a", "Exp", ran=True)],
            [],
            ["a"],
            rangf_n_levels=4,
            random_intercept=True,
            random_coefs=True,
        )

    def test_starts_at_the_prior_so_the_divergence_is_zero(self):
        """The later reference vintage initializes exactly at its prior."""
        from replica.analysis.models.deconvolution.variational import (
            VariationalPosterior,
        )

        posterior = VariationalPosterior(self._model())
        assert float(posterior.kl()) == pytest.approx(0.0, abs=1e-5)

    def test_2018_initialization_jitter_is_seeded_and_leaves_priors_fixed(self):
        from replica.analysis.models.deconvolution.variational import (
            VariationalPosterior,
        )

        model = self._model()
        first = VariationalPosterior(model, init_sd=0.1, seed=3)
        again = VariationalPosterior(model, init_sd=0.1, seed=3)
        other = VariationalPosterior(model, init_sd=0.1, seed=4)
        assert torch.equal(first.loc["coef"], again.loc["coef"])
        assert torch.equal(first.scale_raw["coef"], again.scale_raw["coef"])
        assert not torch.equal(first.loc["coef"], other.loc["coef"])
        assert torch.equal(first.get_buffer("prior_loc_coef"), model.coef.detach())
        assert float(first.kl()) > 0

    def test_random_effect_priors_are_centred_at_zero(self):
        """Offsets are initialized away from zero to break symmetry; their prior
        is centred at zero."""
        from replica.analysis.models.deconvolution.variational import (
            VariationalPosterior,
        )

        model = self._model()
        with torch.no_grad():
            model.irf_raw_random[0].add_(0.3)
        posterior = VariationalPosterior(model)
        prior_loc = posterior.get_buffer("prior_loc_irf_raw_random__0")
        assert torch.count_nonzero(prior_loc) == 0
        assert torch.count_nonzero(posterior.loc["irf_raw_random__0"]) == 0

    def test_population_priors_are_centred_where_fitting_starts(self):
        from replica.analysis.models.deconvolution.variational import (
            VariationalPosterior,
        )

        model = self._model()
        posterior = VariationalPosterior(model)
        assert torch.equal(
            posterior.get_buffer("prior_loc_y_sd_raw"), model.y_sd_raw.detach()
        )

    def test_scales_follow_the_configured_ratios(self):
        from replica.analysis.models.deconvolution.variational import (
            CDRPriors,
            VariationalPosterior,
        )

        priors = CDRPriors(
            coef_sd=2.0, ranef_to_fixef_ratio=0.5, posterior_to_prior_ratio=0.25
        )
        posterior = VariationalPosterior(self._model(), priors)
        assert float(posterior.scale("coef")[0]) == pytest.approx(2.0 * 0.25, rel=1e-5)
        # The random/fixed ratio defines the random-effect prior; the
        # posterior/prior ratio is then applied once to that prior.
        assert float(posterior.scale("coef_random")[0, 0]) == pytest.approx(
            2.0 * 0.5 * 0.25, rel=1e-5
        )

    def test_a_selective_posterior_has_only_the_requested_parameters(self):
        from replica.analysis.models.deconvolution.variational import (
            VariationalPosterior,
        )

        model = self._model()
        posterior = VariationalPosterior(model, names=("intercept", "coef"))
        assert posterior.names == ["intercept", "coef"]
        assert set(posterior.loc) == {"intercept", "coef"}
        assert set(posterior.scale_raw) == {"intercept", "coef"}
        with pytest.raises(ValueError, match="unknown variational parameter"):
            VariationalPosterior(model, names=("missing",))
        with pytest.raises(ValueError, match="must be unique"):
            VariationalPosterior(model, names=("coef", "coef"))

    def test_item_intercepts_use_the_random_effect_prior_ratio(self):
        from replica.analysis.models.deconvolution.cdrnn import CDRNNModel
        from replica.analysis.models.deconvolution.variational import (
            CDRPriors,
            VariationalPosterior,
        )

        model = CDRNNModel(
            ["rate"],
            nn_inputs=[],
            item_n_levels=3,
            history_length=2,
        )
        priors = CDRPriors(
            intercept_sd=2.0,
            ranef_to_fixef_ratio=0.1,
            posterior_to_prior_ratio=0.01,
        )
        posterior = VariationalPosterior(
            model, priors, names=("item_intercept_random",), scale_epsilon=1e-5
        )
        prior_sd = posterior.get_buffer("prior_sd_item_intercept_random")
        assert float(prior_sd[0, 0]) == pytest.approx(0.2)
        scale = posterior.scale("item_intercept_random")[0, 0].detach()
        assert float(scale) == pytest.approx(0.00201, rel=1e-5)

    def test_divergence_matches_the_closed_form(self):
        from replica.analysis.models.deconvolution.variational import (
            VariationalPosterior,
        )

        model = CDRModel([KernelTerm("a", "Exp")], [], ["a"])
        posterior = VariationalPosterior(model)
        with torch.no_grad():
            posterior.loc["coef"].fill_(0.4)
        expected = 0.0
        for name in posterior.names:
            loc = posterior.loc[name.replace(".", "__")]
            scale = posterior.scale(name)
            prior_loc = posterior.get_buffer(f"prior_loc_{name.replace('.', '__')}")
            prior_sd = posterior.get_buffer(f"prior_sd_{name.replace('.', '__')}")
            term = (
                torch.log(prior_sd / scale)
                + (scale**2 + (loc - prior_loc) ** 2) / (2 * prior_sd**2)
                - 0.5
            )
            expected += float(term.sum())
        assert float(posterior.kl()) == pytest.approx(expected, rel=1e-5)

    def test_draws_are_reproducible_and_spread_by_the_scale(self):
        from replica.analysis.models.deconvolution.variational import (
            VariationalPosterior,
        )

        posterior = VariationalPosterior(self._model())
        first = posterior.sample(torch.Generator().manual_seed(3))
        again = posterior.sample(torch.Generator().manual_seed(3))
        for name in posterior.names:
            assert torch.equal(first[name], again[name])
        other = posterior.sample(torch.Generator().manual_seed(4))
        assert not torch.equal(first["coef"], other["coef"])

    def test_zero_scale_draws_the_mean(self):
        from replica.analysis.models.deconvolution.variational import (
            CDRPriors,
            VariationalPosterior,
        )

        posterior = VariationalPosterior(
            self._model(), CDRPriors(posterior_to_prior_ratio=0.0)
        )
        draw = posterior.sample(torch.Generator().manual_seed(0))
        for name, value in draw.items():
            assert torch.allclose(value, posterior.means()[name], atol=1e-6)

    def test_objective_is_the_scaled_likelihood_less_the_divergence(self):
        """The reference scales the likelihood to full-dataset units and leaves
        the divergence terms unscaled."""
        from replica.analysis.models.deconvolution.train import fit_cdr
        from replica.analysis.models.deconvolution.variational import (
            VariationalPosterior,
        )

        design = TestFit._design(n=64)
        model = CDRModel([KernelTerm("a", "Exp")], [], ["a"])
        posterior = VariationalPosterior(model)
        result, _ = fit_cdr(
            model,
            design,
            n_iter=1,
            minibatch_size=len(design.y),
            convergence_n_iterates=10,
            seed=0,
            posterior=posterior,
        )

        replay = VariationalPosterior(CDRModel([KernelTerm("a", "Exp")], [], ["a"]))
        draw = replay.sample(torch.Generator().manual_seed(0))
        with model.substituted(draw):
            ll = model.objective_ll(design)
        expected = float(-ll.sum().detach() + replay.kl().detach())
        assert result.final_mean_loss == pytest.approx(expected, rel=1e-5)

    def test_improper_objective_omits_the_divergence(self):
        """The released no-prior BBVI variant optimizes sampled likelihood."""
        from replica.analysis.models.deconvolution.train import fit_cdr
        from replica.analysis.models.deconvolution.variational import (
            VariationalPosterior,
        )

        design = TestFit._design(n=64)
        model = CDRModel([KernelTerm("a", "Exp")], [], ["a"])
        posterior = VariationalPosterior(model)
        with torch.no_grad():
            posterior.loc["coef"].add_(0.4)
        initial = {
            name: value.detach().clone()
            for name, value in posterior.state_dict().items()
        }
        result, _ = fit_cdr(
            model,
            design,
            n_iter=1,
            learning_rate=0.0,
            minibatch_size=len(design.y),
            convergence_n_iterates=10,
            seed=0,
            posterior=posterior,
            include_kl=False,
        )

        replay = VariationalPosterior(CDRModel([KernelTerm("a", "Exp")], [], ["a"]))
        replay.load_state_dict(initial)
        draw = replay.sample(torch.Generator().manual_seed(0))
        with model.substituted(draw):
            expected = float(-model.objective_ll(design).sum().detach())
        assert float(replay.kl().detach()) > 0
        assert result.final_mean_loss == pytest.approx(expected, rel=1e-5)

    def test_fitting_leaves_the_model_parameters_alone(self):
        from replica.analysis.models.deconvolution.train import fit_cdr
        from replica.analysis.models.deconvolution.variational import (
            VariationalPosterior,
        )

        design = TestFit._design(n=64)
        model = CDRModel([KernelTerm("a", "Exp")], [], ["a"])
        posterior = VariationalPosterior(model)
        before = model.coef.detach().clone()
        fit_cdr(
            model,
            design,
            n_iter=3,
            minibatch_size=32,
            convergence_n_iterates=10,
            seed=0,
            posterior=posterior,
        )
        assert torch.equal(model.coef.detach(), before)
        assert not torch.equal(posterior.loc["coef"].detach(), before)

    @pytest.mark.slow
    def test_variational_fit_recovers_known_exp_kernel(self):
        """The Bayesian estimator recovers the generating kernel of a stream
        the maximum-likelihood estimator also recovers.

        Runs at the reference's own convergence window and at the learning
        rate its Bayesian configurations use. A single-draw lower bound is a
        noisier objective than a likelihood, and the shorter window the
        maximum-likelihood test can afford stops this fit on a plateau its
        posterior scales have not yet left.
        """
        from replica.analysis.models.deconvolution.train import fit_cdr
        from replica.analysis.models.deconvolution.variational import (
            VariationalPosterior,
        )

        design = TestFit._design()
        model = CDRModel([KernelTerm("a", "Exp")], [], ["a"])
        posterior = VariationalPosterior(model)
        result, ema = fit_cdr(
            model,
            design,
            n_iter=4000,
            learning_rate=0.01,
            minibatch_size=128,
            convergence_n_iterates=500,
            seed=0,
            posterior=posterior,
        )
        assert result.converged
        ema.swap_in()
        with model.substituted(posterior.means()):
            coef = float(model.parameter("coef")[0].detach()) * design.y_sd
            beta = float(model.constrained_params(0)[0].detach())
        assert abs(coef - 2.0) < 0.1, f"coefficient {coef} far from truth"
        assert abs(beta - 1.5) < 0.1, f"kernel rate {beta} far from truth"
        # An informative likelihood concentrates the posterior far inside its
        # prior, which the divergence term alone would not do.
        assert float(posterior.scale("coef").mean()) < 0.01
