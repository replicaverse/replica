"""Tests for neural continuous-time deconvolutional regression."""

import numpy as np
import pytest
import torch

from replica.analysis.models.deconvolution.cdrnn import (
    PRED_DIST_EPSILON,
    CDRNNConfig,
    CDRNNModel,
    IRFLayer,
    cdrnn_trainable_parameter_count,
    cdrnn_variational_names,
    centered_offsets,
    exgaussian_log_prob,
    gelu,
    glorot_normal,
)


def _design(n=40, history=8, n_impulse=3, n_levels=4, seed=0):
    from replica.analysis.models.deconvolution.data import CDRDesign

    rng = np.random.default_rng(seed)
    times = np.arange(n, dtype=np.float32) * 0.3
    X = rng.standard_normal((n, history, n_impulse)).astype(np.float32)
    time_X = np.repeat(
        (times[:, None] - np.arange(history, 0, -1)[None, :] * 0.3).astype(np.float32)[
            :, :, None
        ],
        n_impulse,
        axis=2,
    )
    mask = np.ones((n, history, n_impulse), dtype=np.float32)
    mask[:, :2, :] = 0.0
    return CDRDesign(
        X_2d=X,
        time_X_2d=time_X,
        time_mask=mask,
        time_y=times,
        y=(300.0 + 50.0 * rng.standard_normal(n)).astype(np.float64),
        gf_y=rng.integers(0, n_levels, size=(n, 1)),
        impulse_names=[f"i{k}" for k in range(n_impulse)],
        y_mean=300.0,
        y_sd=50.0,
    )


def _model(**kwargs):
    names = ["i0", "i1", "i2"]
    return CDRNNModel(
        names,
        nn_inputs=kwargs.pop("nn_inputs", ["i1", "i2"]),
        rangf_n_levels=kwargs.pop("rangf_n_levels", 4),
        history_length=kwargs.pop("history_length", 8),
        **kwargs,
    )


class TestActivationAndInit:
    def test_activation_is_the_sigmoid_approximation(self):
        """The reference approximates the unit; the exact form is a different
        function and would move every fitted weight."""
        x = torch.linspace(-4, 4, 33)
        torch.testing.assert_close(gelu(x), x * torch.sigmoid(1.702 * x))
        assert not torch.allclose(gelu(x), torch.nn.functional.gelu(x), atol=1e-3)

    def test_weights_are_glorot_normal_not_uniform(self):
        weight = glorot_normal(100, 200, torch.Generator().manual_seed(0))
        expected = (2.0 / 300.0) ** 0.5
        assert weight.shape == (100, 200)
        assert float(weight.std()) == pytest.approx(expected, rel=0.05)
        # A uniform initializer of the same scale would have a hard bound.
        assert float(weight.abs().max()) > expected * 3

    def test_biases_start_at_zero_and_output_layer_has_none(self):
        model = _model()
        for layer in model.layers:
            assert torch.count_nonzero(layer.bias) == 0
        assert model.output_layer.bias is None
        assert model.output_layer.bias_random is None

    def test_coefficients_start_at_zero_and_scales_at_one(self):
        """The fit starts at an intercept-only response of unit scale."""
        model = _model()
        assert torch.count_nonzero(model.coef) == 0
        params = model._linked(model.intercept.detach().unsqueeze(0))
        assert float(params["mu"]) == pytest.approx(0.0)
        for name in ("sigma", "beta"):
            assert float(params[name]) == pytest.approx(
                1.0 + PRED_DIST_EPSILON, rel=1e-6
            )


class TestRandomEffects:
    def test_offsets_are_centered_with_a_zero_row_appended(self):
        raw = torch.tensor([[1.0, 2.0], [3.0, 6.0], [5.0, 4.0]])
        out = centered_offsets(raw)
        assert out.shape == (4, 2)
        assert torch.allclose(out[:3].mean(dim=0), torch.zeros(2), atol=1e-6)
        assert torch.count_nonzero(out[3]) == 0

    def test_an_unseen_level_gets_no_offset(self):
        model = _model(random_intercept=True, random_coefs=True)
        with torch.no_grad():
            model.intercept_random.normal_()
        design = _design()
        design.gf_y[:] = model.rangf_n_levels
        theta = model.response_params(design)
        torch.testing.assert_close(
            theta, model.intercept.unsqueeze(0).expand_as(theta), atol=1e-5, rtol=1e-5
        )

    def test_the_word_gets_an_intercept_of_its_own(self):
        """The papers' formulas cross a by-participant effect with a by-word
        one, so a word carries an offset no participant term can express."""
        from dataclasses import replace

        model = _model(item_n_levels=5)
        assert model.item_intercept_random.shape == (5, len(model.param_names))
        design = _design()
        with torch.no_grad():
            model.item_intercept_random.normal_()
        codes = torch.zeros(len(design.y), dtype=torch.int64)
        first = model.response_params(replace(design, item=codes))
        assert not torch.allclose(
            first, model.response_params(replace(design, item=codes + 1))
        )

    def test_a_word_the_fit_never_saw_gets_no_offset(self):
        from dataclasses import replace

        model = _model(item_n_levels=3)
        design = _design()
        with torch.no_grad():
            model.item_intercept_random.normal_()
        without = model.response_params(design)
        unseen = torch.full((len(design.y),), 3, dtype=torch.int64)
        torch.testing.assert_close(
            model.response_params(replace(design, item=unseen)), without
        )

    def test_the_network_carries_bias_offsets_on_hidden_layers_only(self):
        model = _model()
        assert all(layer.bias_random is not None for layer in model.layers)
        assert len(model.layers) == model.config.n_layers

    def test_level_dropout_sends_groups_to_the_unseen_level(self):
        """Active while fitting and never at prediction, as the reference has it."""
        model = _model(config=CDRNNConfig(ranef_dropout_rate=1.0))
        gf = torch.zeros(16, dtype=torch.int64)
        torch.manual_seed(0)
        dropped = model._dropped_groups(gf, training=True)
        assert torch.all(dropped == model.rangf_n_levels)
        kept = model._dropped_groups(gf, training=False)
        assert torch.all(kept == 0)


class TestForwardPass:
    def test_hidden_dropout_is_independent_by_row_and_history_slot(self):
        layer = IRFLayer(1, 4, activation=False, dropout=0.5)
        with torch.no_grad():
            layer.kernel.fill_(1.0)
        x = torch.ones(16, 8, 1)
        torch.manual_seed(0)
        out = layer(x, None, training=True)
        # A fixed mask would make every row and slot identical. Ordinary
        # DenseLayer dropout varies along both dimensions.
        assert not torch.equal(out[0], out[1])
        assert not torch.equal(out[:, 0], out[:, 1])

    def test_masked_history_slots_contribute_nothing(self):
        model = _model()
        with torch.no_grad():
            for layer in model.layers:
                layer.bias.fill_(0.5)
            model.coef.fill_(1.0)
        design = _design()
        expected = model.response_params(design)
        design.X_2d[:, :2, :] = 1e6
        got = model.response_params(design)
        torch.testing.assert_close(got, expected, atol=1e-4, rtol=1e-4)

    def test_output_carries_the_stabilizing_divide(self):
        """The reference divides its network output by the window depth times
        the impulse count, which sets the scale every coefficient works at."""
        model = _model()
        design = _design()
        with torch.no_grad():
            model.coef.fill_(1.0)
        theta = model.response_params(design) - model.intercept

        X = torch.as_tensor(design.X_2d)
        mask = torch.as_tensor(design.time_mask[:, :, 0])
        slot_time = torch.as_tensor(design.time_X_2d[:, :, 0])
        t_delta = torch.clamp(
            torch.as_tensor(design.time_y)[:, None] - slot_time, min=0
        )
        columns = [t_delta[..., None], (slot_time / model.x_time_sd)[..., None]]
        columns += [X[:, :, model.col[n], None] for n in model.nn_inputs]
        raw = model._network(torch.cat(columns, -1), None, training=False)
        raw = raw.reshape(*raw.shape[:2], 3, len(model.param_names))
        raw = raw / (model.history_length * 3) * mask[:, :, None, None]
        manual = (X[:, :, :, None] * raw).sum(dim=(1, 2))
        torch.testing.assert_close(theta, manual, atol=1e-5, rtol=1e-5)

    def test_dropping_an_input_leaves_its_channel_and_coefficient(self):
        """The papers' linearity null removes a predictor from the network's
        input while it keeps its own output channel and coefficient."""
        full = _model(nn_inputs=["i1", "i2"])
        null = _model(nn_inputs=["i1"])
        assert full.coef.shape == null.coef.shape
        assert full.output_layer.kernel.shape == null.output_layer.kernel.shape
        assert full.layers[0].kernel.shape[0] == null.layers[0].kernel.shape[0] + 1

    def test_a_predictor_independent_network_reads_time_only(self):
        model = _model(config=CDRNNConfig(input_dependent_irf=False))
        assert model.nn_inputs == []
        assert model.layers[0].kernel.shape[0] == 2

    def test_additive_groups_are_independent_networks(self):
        model = _model(
            nn_inputs=["i1", "i2"],
            network_groups=(("i0", "i1"), ("i2",)),
        )
        assert model.term_impulses == ("i0", "i1", "i2")
        assert model.coef.shape == (3, len(model.param_names))
        assert len(model.group_layers) == 2
        assert model.group_inputs == (("i1",), ("i2",))
        assert model.group_layers[0][0].kernel is not model.group_layers[1][0].kernel
        assert model.group_output_layers[0].kernel.shape[1] == 2 * len(
            model.param_names
        )

    def test_additive_group_forward_matches_independent_manual_sum(self):
        model = _model(
            nn_inputs=["i1", "i2"],
            network_groups=(("i0", "i1"), ("i2",)),
        )
        design = _design()
        with torch.no_grad():
            model.coef.fill_(1.0)
        theta = model.response_params(design) - model.intercept

        X = torch.as_tensor(design.X_2d)
        mask = torch.as_tensor(design.time_mask[:, :, 0])
        slot_time = torch.as_tensor(design.time_X_2d[:, :, 0])
        t_delta = torch.clamp(
            torch.as_tensor(design.time_y)[:, None] - slot_time, min=0
        )
        manual = torch.zeros_like(theta)
        for i, (group, inputs) in enumerate(
            zip(model.network_groups, model.group_inputs, strict=True)
        ):
            columns = [t_delta[..., None], slot_time[..., None]]
            columns += [X[:, :, model.col[name], None] for name in inputs]
            raw = model._group_network(i, torch.cat(columns, -1), None, training=False)
            raw = raw.reshape(*raw.shape[:2], len(group), len(model.param_names))
            raw = raw / (model.history_length * len(group))
            raw = raw * mask[:, :, None, None]
            values = X[:, :, [model.col[name] for name in group]]
            manual += (values[:, :, :, None] * raw).sum(dim=(1, 2))
        torch.testing.assert_close(theta, manual, atol=1e-5, rtol=1e-5)

    def test_additive_groups_reject_unknown_impulses(self):
        with pytest.raises(ValueError, match="unknown impulses"):
            _model(network_groups=(("i0", "missing"),))

    def test_additive_groups_reject_overlapping_terminals(self):
        with pytest.raises(ValueError, match="must partition"):
            _model(network_groups=(("i0", "i1"), ("i0", "i2")))

    def test_additive_groups_must_cover_every_terminal(self):
        with pytest.raises(ValueError, match=r"missing \['i2'\]"):
            _model(network_groups=(("i0", "i1"),))


class TestJointResponses:
    @staticmethod
    def _joint_design():
        from dataclasses import replace

        design = _design(n=12, history=8, n_impulse=3, seed=4)
        y = np.column_stack((design.y, design.y + 25.0))
        y[1, 0] = np.nan
        y[3, 1] = np.nan
        return replace(
            design,
            y=y,
            y_mean=np.nanmean(y, axis=0),
            y_sd=np.nanstd(y, axis=0),
        )

    def test_response_specific_parameters_wrap_one_shared_network(self):
        model = _model(
            n_responses=2,
            random_intercept=True,
            random_coefs=True,
            item_n_levels=5,
        )
        n_param = len(model.param_names)
        assert model.intercept.shape == (2, n_param)
        assert model.coef.shape == (2, 3, n_param)
        assert model.intercept_random.shape == (4, 2, n_param)
        assert model.coef_random.shape == (4, 2, 3, n_param)
        assert model.item_intercept_random.shape == (5, 2, n_param)
        # The hidden stack remains one shared object; only its output has one
        # contiguous block per response.
        assert len(model.layers) == model.config.n_layers
        assert model.output_layer.kernel.shape[1] == 2 * 3 * n_param

    def test_one_response_column_does_not_broadcast_across_rows(self):
        from dataclasses import replace

        design = _design(n=12)
        design = replace(
            design,
            y=design.y[:, None],
            y_mean=np.asarray([design.y_mean]),
            y_sd=np.asarray([design.y_sd]),
        )
        model = _model(n_responses=1)
        assert model.objective_ll(design).shape == (12,)
        assert model.log_lik_by_response(design).shape == (12,)

    def test_missing_cells_are_masked_and_objective_averages_declared_dvs(self):
        model = _model(n_responses=2, seed=3)
        design = self._joint_design()
        by_response = model.log_lik_by_response(design)
        assert by_response.shape == (12, 2)
        assert torch.isfinite(by_response).all()
        assert float(by_response[1, 0].detach()) == 0.0
        assert float(by_response[3, 1].detach()) == 0.0
        torch.testing.assert_close(
            model.objective_ll(design, training=False),
            by_response.sum(dim=-1) / 2,
        )
        torch.testing.assert_close(model.log_lik(design), by_response.sum(dim=-1))
        assert model.predict_by_response(design).shape == (12, 2)

    def test_gradients_reach_every_response_specific_parameter(self):
        model = _model(
            n_responses=2,
            random_intercept=True,
            random_coefs=True,
            seed=3,
        )
        loss = -model.objective_ll(self._joint_design()).sum()
        loss.backward()
        for name, parameter in model.named_parameters():
            assert parameter.grad is not None, f"{name} received no gradient"
            assert torch.isfinite(parameter.grad).all()


class TestResponseDistribution:
    def test_log_density_matches_the_closed_form(self):
        """Checked against the textbook expression written through the
        complementary error function, whose own accuracy is what sets the
        tolerance here; the implementation uses the stabler scaled form."""
        y = torch.linspace(-2.0, 4.0, 25).double()
        mu, sigma, beta = 0.3, 0.8, 1.5
        got = exgaussian_log_prob(
            y,
            torch.tensor(mu).double(),
            torch.tensor(sigma).double(),
            torch.tensor(beta).double(),
        )
        rate = 1.0 / beta
        expected = (
            np.log(rate / 2.0)
            + rate / 2.0 * (2 * mu + rate * sigma**2 - 2 * y.numpy())
            + np.log(
                torch.special.erfc(
                    torch.tensor(
                        (mu + rate * sigma**2 - y.numpy()) / (np.sqrt(2) * sigma)
                    )
                ).numpy()
            )
        )
        np.testing.assert_allclose(got.numpy(), expected, rtol=1e-7)

    def test_density_integrates_to_one(self):
        grid = torch.linspace(-6.0, 30.0, 200001).double()
        density = torch.exp(
            exgaussian_log_prob(
                grid,
                torch.tensor(0.5).double(),
                torch.tensor(1.2).double(),
                torch.tensor(2.0).double(),
            )
        )
        mass = torch.trapezoid(density, grid)
        assert float(mass) == pytest.approx(1.0, abs=1e-6)

    def test_density_stays_finite_when_the_gaussian_scale_is_broad(self):
        """The equivalent exponent-plus-log-CDF form cancels to nan here."""
        y = torch.tensor(0.0)
        mu = torch.tensor(0.0, requires_grad=True)
        sigma = torch.tensor(1e5, requires_grad=True)
        beta = torch.tensor(1e-2, requires_grad=True)
        got = exgaussian_log_prob(y, mu, sigma, beta)
        expected = torch.distributions.Normal(mu, sigma).log_prob(y)
        torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-6)
        (-got).backward()
        assert all(torch.isfinite(value.grad) for value in (mu, sigma, beta))

    def test_mean_is_the_location_plus_the_exponential_mean(self):
        torch.manual_seed(0)
        mu, sigma, beta = 0.5, 1.2, 2.0
        draws = (
            torch.randn(400000).double() * sigma
            + mu
            + torch.distributions.Exponential(1.0 / beta).sample((400000,)).double()
        )
        assert float(draws.mean()) == pytest.approx(mu + beta, abs=0.02)

    def test_positive_parameters_carry_the_reference_floor(self):
        model = _model()
        params = model._linked(torch.full((1, 3), -50.0))
        for name in ("sigma", "beta"):
            assert float(params[name]) == pytest.approx(PRED_DIST_EPSILON, rel=1e-6)

    def test_likelihood_is_on_the_response_scale(self):
        """The distribution is fit standardized and then rescaled, so its
        density on the original scale carries the logarithm of that scale."""
        model = _model()
        design = _design()
        ll = model.log_lik(design)
        theta = model.response_params(design)
        params = model._linked(theta)
        z = (
            torch.as_tensor(design.y, dtype=torch.float32) - design.y_mean
        ) / design.y_sd
        standardized = exgaussian_log_prob(
            z, params["mu"], params["sigma"], params["beta"]
        )
        torch.testing.assert_close(
            ll, standardized - float(np.log(design.y_sd)), atol=1e-5, rtol=1e-5
        )

    @pytest.mark.parametrize("distribution", ["normal", "exgaussianinvrate"])
    def test_a_location_only_response_is_refused(self, distribution):
        """The reference estimates every other response parameter as a
        constant when only the location is predicted; this model would fix
        the Gaussian scale at one or fail on the missing skew."""
        with pytest.raises(NotImplementedError, match="distributional_regression"):
            CDRNNConfig(distribution=distribution, distributional_regression=False)

    @pytest.mark.parametrize(
        "distribution, names",
        [("normal", ("mu", "sigma")), ("exgaussianinvrate", ("mu", "sigma", "beta"))],
    )
    def test_every_response_parameter_is_predicted(self, distribution, names):
        model = _model(config=CDRNNConfig(distribution=distribution))
        assert model.param_names == names
        assert model.coef.shape[1] == len(names)
        design = _design()
        assert torch.isfinite(model.log_lik(design)).all()


class TestPenalties:
    def test_random_effect_penalty_has_no_half_factor(self):
        """The parametric model's penalty halves the sum of squares; this one
        does not, so the same nominal scale is twice the pressure."""
        model = _model(random_intercept=True)
        with torch.no_grad():
            model.intercept_random.copy_(
                torch.tensor(
                    [
                        [1.0, 0.0, 0.0],
                        [-1.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                    ]
                )
            )
            for layer in model.layers:
                layer.bias_random.zero_()
        assert float(model.ranef_penalty(1.0)) == pytest.approx(2.0, rel=1e-5)

    def test_only_hidden_weights_are_regularized(self):
        model = _model()
        with torch.no_grad():
            for layer in model.layers:
                layer.kernel.zero_()
            model.output_layer.kernel.fill_(10.0)
        assert float(model.regularization()) == 0.0

    def test_regularization_uses_the_configured_scale(self):
        model = _model(config=CDRNNConfig(nn_regularizer_scale=3.0))
        with torch.no_grad():
            for layer in model.layers:
                layer.kernel.zero_()
            model.layers[0].kernel[0, 0] = 2.0
        assert float(model.regularization()) == pytest.approx(12.0, rel=1e-6)

    def test_additive_networks_are_both_regularized(self):
        model = _model(
            network_groups=(("i0", "i1"), ("i2",)),
            config=CDRNNConfig(nn_regularizer_scale=2.0),
        )
        with torch.no_grad():
            for group in model.group_layers:
                for layer in group:
                    layer.kernel.zero_()
            model.group_layers[0][0].kernel[0, 0] = 2.0
            model.group_layers[1][0].kernel[0, 0] = 3.0
        assert float(model.regularization()) == pytest.approx(26.0, rel=1e-6)


class TestFitting:
    @pytest.mark.parametrize(
        ("n_impulses", "n_responses", "n_participants", "n_items", "expected"),
        [
            (7, 3, 84, 2413, 64_410),
            (6, 2, 14, 43_117, 522_024),
        ],
    )
    def test_hybrid_parameter_counts_match_the_author_checkpoints(
        self, n_impulses, n_responses, n_participants, n_items, expected
    ):
        from replica.analysis.models.deconvolution.variational import (
            CDRPriors,
            VariationalPosterior,
        )

        names = ["rate", *(f"i{i}" for i in range(1, n_impulses))]
        model = CDRNNModel(
            names,
            nn_inputs=names[1:],
            n_responses=n_responses,
            rangf_n_levels=n_participants,
            random_intercept=True,
            random_coefs=True,
            item_n_levels=n_items,
            history_length=32,
        )
        posterior = VariationalPosterior(
            model,
            CDRPriors(
                ranef_to_fixef_ratio=0.1,
                posterior_to_prior_ratio=0.01,
            ),
            names=cdrnn_variational_names(model),
            scale_epsilon=1e-5,
        )
        assert cdrnn_trainable_parameter_count(model, posterior) == expected

    def test_default_hybrid_posterior_leaves_only_neural_weights_as_points(self):
        from replica.analysis.models.deconvolution.variational import (
            CDRPriors,
            VariationalPosterior,
        )

        model = _model(
            random_intercept=True,
            random_coefs=True,
            item_n_levels=5,
        )
        names = cdrnn_variational_names(model)
        assert names == (
            "intercept",
            "coef",
            "intercept_random",
            "coef_random",
            "item_intercept_random",
        )
        posterior = VariationalPosterior(
            model,
            CDRPriors(
                ranef_to_fixef_ratio=0.1,
                posterior_to_prior_ratio=0.01,
            ),
            names=names,
            scale_epsilon=1e-5,
        )
        selected = sum(model.parameter(name).numel() * 2 for name in names)
        point = sum(
            parameter.numel()
            for name, parameter in model.named_parameters()
            if name not in names
        )
        assert cdrnn_trainable_parameter_count(model, posterior) == selected + point
        assert all(not name.startswith(("layers.", "output_layer.")) for name in names)

    def test_hybrid_fit_updates_posterior_and_point_parameters_together(self):
        from replica.analysis.models.deconvolution.train import fit_cdr
        from replica.analysis.models.deconvolution.variational import (
            CDRPriors,
            VariationalPosterior,
        )

        design = _design(n=48, seed=2)
        model = _model(random_intercept=True, random_coefs=True, seed=4)
        posterior = VariationalPosterior(
            model,
            CDRPriors(
                ranef_to_fixef_ratio=0.1,
                posterior_to_prior_ratio=0.01,
            ),
            names=cdrnn_variational_names(model),
            scale_epsilon=1e-5,
        )
        base_coef = model.coef.detach().clone()
        posterior_coef = posterior.loc["coef"].detach().clone()
        neural_kernel = model.layers[0].kernel.detach().clone()
        _, ema = fit_cdr(
            model,
            design,
            n_iter=2,
            learning_rate=0.001,
            minibatch_size=16,
            ema_updates_per_batch=1,
            optimizer="adam",
            max_global_norm=1.0,
            check_convergence=False,
            seed=7,
            posterior=posterior,
        )
        # Substituted conventional tensors are not optimized in place, while
        # their posterior and the point-estimated neural stack both are.
        assert torch.equal(model.coef.detach(), base_coef)
        assert not torch.equal(posterior.loc["coef"].detach(), posterior_coef)
        assert not torch.equal(model.layers[0].kernel.detach(), neural_kernel)
        assert len(ema.shadow) == sum(
            1 for parameter in posterior.parameters() if parameter.requires_grad
        ) + sum(
            1
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and name not in posterior.names
        )

    def test_seed_reproduces_dropout_training(self):
        from replica.analysis.models.deconvolution.train import fit_cdr

        design = _design(n=48, seed=2)
        fitted = []
        for _ in range(2):
            model = _model(seed=4)
            _, ema = fit_cdr(
                model,
                design,
                n_iter=2,
                minibatch_size=16,
                optimizer="adam",
                check_convergence=False,
                seed=7,
            )
            fitted.append(torch.cat([value.ravel() for value in ema.shadow]))
        torch.testing.assert_close(fitted[0], fitted[1], rtol=0, atol=0)

    @pytest.mark.slow
    def test_a_fit_improves_the_held_out_density(self):
        from replica.analysis.models.deconvolution.train import fit_cdr

        design = _design(n=600, seed=1)
        # A second draw from the same generator, which the fit never sees.
        held_out = _design(n=600, seed=2)
        model = _model(random_intercept=True, random_coefs=True, seed=0)
        torch.manual_seed(0)
        with torch.no_grad():
            start = float(model.log_lik(held_out).sum())
        _, ema = fit_cdr(
            model,
            design,
            n_iter=40,
            learning_rate=0.001,
            minibatch_size=128,
            optimizer="adam",
            max_global_norm=1.0,
            ranef_regularizer_scale=model.config.ranef_regularizer_scale,
            check_convergence=False,
            seed=0,
        )
        ema.swap_in()
        with torch.no_grad():
            assert float(model.log_lik(held_out).sum()) > start

    def test_gradients_reach_every_trainable_part(self):
        design = _design()
        model = _model(random_intercept=True, random_coefs=True, seed=0)
        torch.manual_seed(0)
        loss = -model.objective_ll(design).sum() + model.regularization()
        loss.backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"{name} received no gradient"
            assert torch.isfinite(param.grad).all(), f"{name} has a non-finite gradient"

    def test_gradients_reach_both_additive_networks(self):
        design = _design()
        model = _model(
            network_groups=(("i0", "i1"), ("i2",)),
            random_intercept=True,
            random_coefs=True,
            seed=0,
        )
        torch.manual_seed(0)
        loss = -model.objective_ll(design).sum() + model.regularization()
        loss.backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"{name} received no gradient"
            assert torch.isfinite(param.grad).all(), f"{name} has a non-finite gradient"
