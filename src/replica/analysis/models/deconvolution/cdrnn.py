"""Neural continuous-time deconvolutional regression."""

from dataclasses import asdict, dataclass, field, replace

import numpy as np
import torch

from replica.analysis.contracts.models import LikelihoodModel
from replica.analysis.models.deconvolution.data import (
    CDRDesign,
    build_design_from_events,
    transform_impulses,
)
from replica.analysis.models.deconvolution.kernels import softplus, softplus_inverse
from replica.analysis.models.deconvolution.model import (
    _SubstitutableParameters,
    design_on,
)
from replica.analysis.models.deconvolution.variational import (
    CDRPriors,
    VariationalPosterior,
)
from replica.analysis.models.support.diagnostics import _FitDiagnostics

#: The reference's sigmoid approximation to the Gaussian error linear unit.
GELU_SCALE = 1.702

#: Floor added to every positive response parameter after its link.
PRED_DIST_EPSILON = 1e-2

#: Response distributions, with their parameters in the order they are emitted.
DISTRIBUTIONS = {
    "normal": ("mu", "sigma"),
    "exgaussianinvrate": ("mu", "sigma", "beta"),
}

#: Parameters constrained positive by a softplus link.
POSITIVE_PARAMS = ("sigma", "beta")


def gelu(x: torch.Tensor) -> torch.Tensor:
    """The reference's activation, which approximates the exact form.

    Args:
        x: Pre-activations, gated element-wise by a sigmoid of ``x`` scaled by
            :data:`GELU_SCALE`.
    """
    return x * torch.sigmoid(GELU_SCALE * x)


def glorot_normal(
    fan_in: int, fan_out: int, generator: torch.Generator | None = None
) -> torch.Tensor:
    """Normal Glorot weights with the reference's standard deviation.

    Args:
        fan_in: Input width, the first axis of the returned matrix.
        fan_out: Output width, the second axis.
        generator: Torch generator drawn from, so a seeded fit initializes
            identically. ``None`` draws from the global stream.

    Returns:
        A ``fan_in`` by ``fan_out`` matrix, normal with standard deviation
        ``sqrt(2 / (fan_in + fan_out))``.
    """
    sd = (2.0 / (fan_in + fan_out)) ** 0.5
    weight = torch.empty(fan_in, fan_out, dtype=torch.float32)
    return torch.nn.init.normal_(weight, std=sd, generator=generator)


def centered_offsets(raw: torch.Tensor) -> torch.Tensor:
    """By-level offsets, mean-centered, with a zero row for an unseen level.

    Args:
        raw: Stored offsets, one row per training level of the grouping factor.

    Returns:
        The offsets centered over the level axis, with one zero row appended,
        so indexing the next code up contributes nothing.
    """
    centered = raw - raw.mean(dim=0, keepdim=True)
    pad = torch.zeros(
        (1, *centered.shape[1:]), dtype=centered.dtype, device=centered.device
    )
    return torch.cat([centered, pad], dim=0)


def exgaussian_log_prob(
    y: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor, beta: torch.Tensor
) -> torch.Tensor:
    """Log density of an exponentially modified Gaussian of mean ``mu + beta``.

    ``beta`` is the exponential component's mean, which the reference obtains
    by inverting the rate its distribution takes. The positive-tail branch is
    written through scaled complementary error to avoid both underflow and
    cancellation.

    Args:
        y: Observations to score, on the same scale as *mu*.
        mu: Mean of the Gaussian component.
        sigma: Standard deviation of the Gaussian component, positive.
        beta: Mean of the exponential component, positive, which is what
            skews the density to the right.
    """
    # Writing this as ``large positive exponent + large negative log-CDF``
    # catastrophically cancels once a trajectory predicts a broad Gaussian.
    # For a = (sigma / beta - z) / sqrt(2), factor erfc(a) into
    # exp(-a**2) * erfcx(a); its exponent cancels symbolically and leaves the
    # finite expression below. erfcx overflows on the opposite tail, where the
    # ordinary erfc expression has no cancellation, so use that branch there.
    z = (y - mu) / sigma
    v = sigma / beta
    a = (v - z) / np.sqrt(2.0)
    right = a >= 0

    # Keep the inactive branch finite as well. Autograd differentiates both
    # sides of torch.where, where a masked infinity could otherwise yield nan.
    z_right = torch.where(right, z, torch.zeros_like(z))
    a_right = torch.clamp_min(a, 0.0)
    log_right = (
        -torch.log(2.0 * beta)
        - 0.5 * z_right**2
        + torch.log(torch.special.erfcx(a_right))
    )

    v_left = torch.where(right, torch.zeros_like(v), v)
    z_left = torch.where(right, torch.zeros_like(z), z)
    a_left = torch.clamp_max(a, 0.0)
    log_left = (
        -torch.log(2.0 * beta)
        + 0.5 * v_left**2
        - v_left * z_left
        + torch.log(torch.special.erfc(a_left))
    )
    return torch.where(right, log_right, log_left)


@dataclass
class CDRNNConfig:
    """The reference's settings for the neural component and its response.

    Defaults are the ones every configuration of both reading papers sets or
    inherits. Every response parameter is predicted, the reference's neural
    default; ``distributional_regression=False`` raises. Input rescaling is
    set per impulse through ``CDRNNAnalysis.impulse_ops``.
    """

    n_layers: int = 2
    n_units: int = 32
    dropout_rate: float = 0.1
    ranef_dropout_rate: float = 0.1
    nn_regularizer_scale: float = 5.0
    ranef_regularizer_scale: float = 10.0
    #: When false the network sees only time, giving a predictor-independent
    #: shape that still differs per impulse.
    input_dependent_irf: bool = True
    distribution: str = "exgaussianinvrate"
    distributional_regression: bool = True

    def __post_init__(self) -> None:
        if not self.distributional_regression:
            raise NotImplementedError(
                "distributional_regression=False is not implemented: the "
                "reference then predicts the location only and estimates the "
                "other response parameters as constants, which this model does "
                "not do. Keep distributional_regression=True."
            )

    def params(self) -> tuple[str, ...]:
        """Response parameters the network predicts, in the order it emits them.

        Returns:
            Every parameter of the chosen distribution.
        """
        return DISTRIBUTIONS[self.distribution]


class IRFLayer(torch.nn.Module):
    """One layer of the network, with the reference's by-level bias offsets.

    Random effects enter the network as additive offsets on the bias vectors
    of its hidden layers; the output layer carries no bias and so no offset.
    """

    def __init__(
        self,
        n_in: int,
        n_out: int,
        *,
        bias: bool = True,
        activation: bool = True,
        dropout: float = 0.0,
        n_levels: int = 0,
        generator: torch.Generator | None = None,
    ):
        """Allocate the layer's weight matrix and its optional bias terms.

        Args:
            n_in: Input width.
            n_out: Output width.
            bias: Give the layer a bias vector. The output layer carries none,
                and so carries no by-level offsets either.
            activation: Pass the result through :func:`gelu`; the output layer
                stays linear.
            dropout: Element-wise drop probability applied during training
                only; zero disables it.
            n_levels: Grouping levels the bias offsets cover, one row short of
                this count so the last code is the unseen level. Below two, the
                layer carries no offsets.
            generator: Torch generator the weights are drawn from, so a seeded
                fit initializes identically. ``None`` draws from the global
                stream.
        """
        super().__init__()
        self.activation = activation
        self.dropout = dropout
        self.kernel = torch.nn.Parameter(glorot_normal(n_in, n_out, generator))
        self.bias = (
            torch.nn.Parameter(torch.zeros(n_out, dtype=torch.float32))
            if bias
            else None
        )
        self.bias_random = (
            torch.nn.Parameter(torch.zeros(n_levels - 1, n_out, dtype=torch.float32))
            if bias and n_levels > 1
            else None
        )

    def forward(
        self,
        x: torch.Tensor,
        gf: torch.Tensor | None,
        *,
        training: bool,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Project one history window, offsetting the bias by the row's level.

        Args:
            x: Inputs, response by history slot by feature.
            gf: Grouping code per response, selecting a bias offset row.
                ``None`` applies the population bias alone.
            training: Apply dropout, rescaling the units that survive it.
                False leaves the projection alone.
            generator: Torch generator the dropout mask is drawn from; ``None``
                draws from the global stream.

        Returns:
            Activations, response by history slot by output width.
        """
        out = x @ self.kernel
        if self.bias is not None:
            out = out + self.bias
            if self.bias_random is not None and gf is not None:
                out = out + centered_offsets(self.bias_random)[gf][:, None, :]
        if self.activation:
            out = gelu(out)
        if training and self.dropout:
            # The reference's IRF DenseLayer uses ordinary element-wise
            # dropout. Its fixed-over-time option applies only to the recurrent
            # and input dropout layers.
            keep = (
                torch.rand(
                    out.shape,
                    device=out.device,
                    generator=generator,
                )
                >= self.dropout
            ).to(out.dtype)
            out = out * keep / (1.0 - self.dropout)
        return out


class CDRNNModel(_SubstitutableParameters, torch.nn.Module):
    """Deconvolutional regression whose impulse response is a network.

    ``nn_inputs`` names the impulses the network reads. The reference drops
    the constant-one rate impulse and whatever a configuration names, so an
    impulse may keep its own output channel, coefficient and random slope
    while losing its place in the input, which is how the papers state a
    strictly linear null.
    """

    def __init__(
        self,
        impulse_names: list[str],
        *,
        nn_inputs: list[str],
        n_responses: int = 1,
        network_groups: tuple[tuple[str, ...], ...] | None = None,
        config: CDRNNConfig | None = None,
        rangf_n_levels: int | None = None,
        random_intercept: bool = False,
        random_coefs: bool = False,
        item_n_levels: int = 0,
        x_time_sd: float = 1.0,
        history_length: int = 32,
        seed: int | None = None,
    ):
        """Build the response network and the conventional regression terms.

        Args:
            impulse_names: Impulse channels in the design's column order, which
                also fixes the coefficient order.
            nn_inputs: Impulses the network reads. One left out keeps its own
                channel, coefficient, and random slope, and loses only its
                influence on the response shape.
            n_responses: Dependent variables predicted jointly. Above one, the
                network emits a contiguous output slice per response.
            network_groups: Disjoint impulse groups, each given its own network.
                They have to partition *impulse_names*; ``None`` runs a single
                network over all of them.
            config: Network shape, regularization, and response distribution.
                ``None`` takes the reference's defaults.
            rangf_n_levels: Grouping levels that receive fitted offsets. The
                next code up indexes an unseen level, contributing nothing.
            random_intercept: Give each grouping level its own intercept offset.
            random_coefs: Give each grouping level its own coefficient offsets.
            item_n_levels: Word levels given an intercept alongside the
                participant's. Zero leaves the term out.
            x_time_sd: Divisor for the network's absolute-time input; the lag
                input is left raw.
            history_length: Impulse slots per response, which also scales the
                network's per-slot weights down by the window size.
            seed: Seed for the weight initialization; ``None`` is unseeded.
        """
        super().__init__()
        self.config = config or CDRNNConfig()
        self.impulse_names = list(impulse_names)
        self.col = {name: i for i, name in enumerate(impulse_names)}
        self.nn_inputs = list(nn_inputs) if self.config.input_dependent_irf else []
        self.network_groups = (
            tuple(tuple(group) for group in network_groups)
            if network_groups is not None
            else None
        )
        if self.network_groups is not None:
            if not self.network_groups or any(
                not group for group in self.network_groups
            ):
                raise ValueError("network_groups must contain non-empty groups")
            unknown = {
                name
                for group in self.network_groups
                for name in group
                if name not in self.col
            }
            if unknown:
                raise ValueError(
                    f"unknown impulses in network_groups: {sorted(unknown)!r}"
                )
            if any(len(set(group)) != len(group) for group in self.network_groups):
                raise ValueError(
                    "an impulse may occur only once within a network group"
                )
            flattened = [name for group in self.network_groups for name in group]
            if len(set(flattened)) != len(flattened):
                raise ValueError(
                    "network groups must partition their terminal impulses"
                )
            missing = set(self.impulse_names) - set(flattened)
            if missing:
                raise ValueError(
                    "network groups must partition all terminal impulses; "
                    f"missing {sorted(missing)!r}"
                )
        self.param_names = self.config.params()
        if n_responses < 1:
            raise ValueError("n_responses must be positive")
        self.n_responses = int(n_responses)
        self.x_time_sd = float(x_time_sd)
        self.history_length = int(history_length)
        self.rangf_n_levels = rangf_n_levels
        self.random_intercept = random_intercept
        self.random_coefs = random_coefs
        self.item_n_levels = item_n_levels
        self._substitute: dict[str, torch.Tensor] | None = None

        # Separate networks have separate terminal terms. The reference's
        # formula canonicalizer assigns an overlapping terminal to the first
        # CDRNN term, so the groups partition the terminal terms.
        self.term_impulses = (
            tuple(name for group in self.network_groups for name in group)
            if self.network_groups is not None
            else tuple(self.impulse_names)
        )
        n_impulse = len(self.term_impulses)
        n_param = len(self.param_names)
        generator = None
        if seed is not None:
            generator = torch.Generator().manual_seed(seed)

        intercept_init = torch.tensor(
            [
                float(softplus_inverse(torch.tensor(1.0)))
                if name in POSITIVE_PARAMS
                else 0.0
                for name in self.param_names
            ],
            dtype=torch.float32,
        )
        if self.n_responses == 1:
            self.intercept = torch.nn.Parameter(intercept_init)
            self.coef = torch.nn.Parameter(
                torch.zeros(n_impulse, n_param, dtype=torch.float32)
            )
        else:
            self.intercept = torch.nn.Parameter(
                intercept_init[None, :].repeat(self.n_responses, 1)
            )
            self.coef = torch.nn.Parameter(
                torch.zeros(self.n_responses, n_impulse, n_param, dtype=torch.float32)
            )

        levels = rangf_n_levels or 0
        if random_intercept:
            shape = (
                (levels, n_param)
                if self.n_responses == 1
                else (levels, self.n_responses, n_param)
            )
            self.intercept_random = torch.nn.Parameter(torch.zeros(*shape))
        if random_coefs:
            shape = (
                (levels, n_impulse, n_param)
                if self.n_responses == 1
                else (levels, self.n_responses, n_impulse, n_param)
            )
            self.coef_random = torch.nn.Parameter(torch.zeros(*shape))
        # The papers' formulas give the word an intercept of its own, on top
        # of the by-participant effects.
        if item_n_levels:
            shape = (
                (item_n_levels, n_param)
                if self.n_responses == 1
                else (item_n_levels, self.n_responses, n_param)
            )
            self.item_intercept_random = torch.nn.Parameter(torch.zeros(*shape))

        if self.network_groups is None:
            n_in = 2 + len(self.nn_inputs)
            widths = [n_in] + [self.config.n_units] * self.config.n_layers
            self.layers = self._make_hidden_layers(widths, levels, generator)
            self.output_layer = IRFLayer(
                widths[-1],
                n_impulse * n_param * self.n_responses,
                bias=False,
                activation=False,
                generator=generator,
            )
        else:
            self.group_inputs = tuple(
                tuple(name for name in group if name in self.nn_inputs)
                for group in self.network_groups
            )
            group_layers = []
            group_outputs = []
            for group, inputs in zip(
                self.network_groups, self.group_inputs, strict=True
            ):
                widths = [2 + len(inputs)] + [
                    self.config.n_units
                ] * self.config.n_layers
                group_layers.append(self._make_hidden_layers(widths, levels, generator))
                group_outputs.append(
                    IRFLayer(
                        widths[-1],
                        len(group) * n_param * self.n_responses,
                        bias=False,
                        activation=False,
                        generator=generator,
                    )
                )
            self.group_layers = torch.nn.ModuleList(group_layers)
            self.group_output_layers = torch.nn.ModuleList(group_outputs)

    def _make_hidden_layers(self, widths, levels, generator):
        return torch.nn.ModuleList(
            [
                IRFLayer(
                    widths[i],
                    widths[i + 1],
                    dropout=self.config.dropout_rate,
                    n_levels=levels + 1,
                    generator=generator,
                )
                for i in range(self.config.n_layers)
            ]
        )

    @property
    def device(self) -> torch.device:
        """Device the model's parameters sit on."""
        return self.intercept.device

    def _network(
        self, x: torch.Tensor, gf: torch.Tensor | None, *, training: bool
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, gf, training=training)
        return self.output_layer(x, None, training=False)

    def _group_network(
        self, group: int, x: torch.Tensor, gf: torch.Tensor | None, *, training: bool
    ) -> torch.Tensor:
        for layer in self.group_layers[group]:
            x = layer(x, gf, training=training)
        return self.group_output_layers[group](x, None, training=False)

    def _dropped_groups(self, gf: torch.Tensor | None, training: bool):
        """Group codes with the reference's by-level dropout to the unseen level.

        Only the network reads these; the random intercepts and slopes read
        the undropped codes, and the dropout is never active at prediction.
        """
        rate = self.config.ranef_dropout_rate
        if gf is None or not training or not rate:
            return gf
        unknown = self.rangf_n_levels or 0
        drop = torch.rand(gf.shape, device=gf.device) < rate
        return torch.where(drop, torch.full_like(gf, unknown), gf)

    def response_params(
        self,
        design: CDRDesign,
        idx: np.ndarray | torch.Tensor | slice | None = None,
        *,
        training: bool = False,
    ) -> torch.Tensor:
        """Return per-observation response parameters.

        The network turns each history slot's lag, timestamp, and impulse
        values into a weight, and the weighted impulses sum into the
        conventional intercept and coefficients.

        Args:
            design: Windowed impulse history, response timestamps, and the
                grouping codes the random effects are indexed by.
            idx: Rows of the design to evaluate; ``None`` takes all of them.
            training: Apply the network's dropout and the by-level dropout that
                sends a grouping code to the unseen level. Prediction leaves
                both off.

        Returns:
            Unlinked response parameters, one row per evaluated response, with
            a response axis before the parameter axis when more than one
            dependent variable is predicted.
        """
        sl = slice(None) if idx is None else idx
        device = self.device
        X = torch.as_tensor(design.X_2d[sl], device=device)
        time_X = torch.as_tensor(design.time_X_2d[sl], device=device)
        time_y = torch.as_tensor(design.time_y[sl], device=device)
        mask = torch.as_tensor(design.time_mask[sl, :, 0], device=device).to(X.dtype)
        gf = (
            torch.as_tensor(design.gf_y[sl, 0], device=device)
            if len(design.gf_y) and design.gf_y.shape[1]
            else None
        )
        # Impulse timestamps are per impulse in the windowed arrays; the
        # network reads one per history slot, and the reference clamps the
        # time offset at zero rather than moving the response.
        slot_time = time_X[:, :, 0]
        t_delta = torch.clamp(time_y[:, None] - slot_time, min=0.0)

        dropped_gf = self._dropped_groups(gf, training)
        coef = self.parameter("coef").unsqueeze(0)
        if self.random_coefs and gf is not None:
            coef = coef + centered_offsets(self.parameter("coef_random"))[gf]
        n_param = len(self.param_names)
        if self.network_groups is None:
            columns = [t_delta[..., None], (slot_time / self.x_time_sd)[..., None]]
            columns += [X[:, :, self.col[name], None] for name in self.nn_inputs]
            irf = self._network(
                torch.cat(columns, dim=-1), dropped_gf, training=training
            )
            n_impulse = len(self.impulse_names)
            if self.n_responses == 1:
                irf = irf.reshape(*irf.shape[:2], n_impulse, n_param)
                irf = irf / (self.history_length * n_impulse)
                irf = irf * mask[:, :, None, None]
                weighted = X[:, :, :, None] * irf * coef[:, None]
                out = weighted.sum(dim=(1, 2))
            else:
                # The reference allocates one contiguous output slice per
                # response, and reshapes each slice as terminal x parameter.
                # Keep that response-major output order even though the
                # tensor used below is terminal x response x parameter.
                irf = irf.reshape(
                    *irf.shape[:2], self.n_responses, n_impulse, n_param
                ).permute(0, 1, 3, 2, 4)
                irf = irf / (self.history_length * n_impulse)
                irf = irf * mask[:, :, None, None, None]
                # Coefficients are stored response-first but the IRF carries
                # terminal before response, matching the reference's output
                # slice layout.
                coef_by_terminal = coef.permute(0, 2, 1, 3)
                weighted = X[:, :, :, None, None] * irf * coef_by_terminal[:, None]
                out = weighted.sum(dim=(1, 2))
        else:
            out_shape = (
                (len(X), n_param)
                if self.n_responses == 1
                else (len(X), self.n_responses, n_param)
            )
            out = torch.zeros(out_shape, dtype=X.dtype, device=X.device)
            first_term = 0
            for group_index, (group, inputs) in enumerate(
                zip(self.network_groups, self.group_inputs, strict=True)
            ):
                columns = [
                    t_delta[..., None],
                    (slot_time / self.x_time_sd)[..., None],
                ]
                columns += [X[:, :, self.col[name], None] for name in inputs]
                irf = self._group_network(
                    group_index,
                    torch.cat(columns, dim=-1),
                    dropped_gf,
                    training=training,
                )
                if self.n_responses == 1:
                    irf = irf.reshape(*irf.shape[:2], len(group), n_param)
                    irf = irf / (self.history_length * len(group))
                    irf = irf * mask[:, :, None, None]
                else:
                    irf = irf.reshape(
                        *irf.shape[:2], self.n_responses, len(group), n_param
                    ).permute(0, 1, 3, 2, 4)
                    irf = irf / (self.history_length * len(group))
                    irf = irf * mask[:, :, None, None, None]
                group_X = X[:, :, [self.col[name] for name in group]]
                last_term = first_term + len(group)
                if self.n_responses == 1:
                    weighted = (
                        group_X[:, :, :, None]
                        * irf
                        * coef[:, None, first_term:last_term]
                    )
                else:
                    coef_by_terminal = coef[:, :, first_term:last_term].permute(
                        0, 2, 1, 3
                    )
                    weighted = (
                        group_X[:, :, :, None, None] * irf * coef_by_terminal[:, None]
                    )
                out = out + weighted.sum(dim=(1, 2))
                first_term = last_term

        intercept = self.parameter("intercept").unsqueeze(0)
        if self.random_intercept and gf is not None:
            intercept = (
                intercept + centered_offsets(self.parameter("intercept_random"))[gf]
            )
        if self.item_n_levels and design.item is not None:
            intercept = (
                intercept
                + centered_offsets(self.parameter("item_intercept_random"))[
                    design.item[sl]
                ]
            )
        return out + intercept

    def _linked(self, theta: torch.Tensor) -> dict[str, torch.Tensor]:
        """Response parameters after their links, on the standardized scale."""
        out = {}
        for i, name in enumerate(self.param_names):
            value = theta[..., i]
            if name in POSITIVE_PARAMS:
                value = softplus(value) + PRED_DIST_EPSILON
            out[name] = value
        return out

    def _log_prob(
        self, y: torch.Tensor, theta: torch.Tensor, design: CDRDesign
    ) -> torch.Tensor:
        """Log density on the response's own scale.

        The distribution is built on the standardized scale and then shifted
        and scaled, so its density on the original scale carries the
        logarithm of that scale.
        """
        params = self._linked(theta)
        mean = torch.as_tensor(design.y_mean, dtype=y.dtype, device=y.device)
        sd = torch.as_tensor(design.y_sd, dtype=y.dtype, device=y.device)
        z = (y - mean) / sd
        if self.config.distribution == "normal":
            log_p = torch.distributions.Normal(params["mu"], params["sigma"]).log_prob(
                z
            )
        else:
            log_p = exgaussian_log_prob(
                z, params["mu"], params["sigma"], params["beta"]
            )
        return log_p - torch.log(sd)

    def _response_values(self, design: CDRDesign, index) -> torch.Tensor:
        """Targets with the shape declared by ``n_responses``."""
        values = torch.as_tensor(
            design.y[index], dtype=torch.float32, device=self.device
        )
        if self.n_responses == 1:
            if values.ndim == 2 and values.shape[1] == 1:
                return values[:, 0]
            if values.ndim == 1:
                return values
        elif values.ndim == 2 and values.shape[1] == self.n_responses:
            return values
        raise ValueError(
            f"response shape {tuple(values.shape)!r} does not match "
            f"n_responses={self.n_responses}"
        )

    def log_lik_by_response(
        self,
        design: CDRDesign,
        idx: np.ndarray | torch.Tensor | slice | None = None,
    ) -> torch.Tensor:
        """Raw held-out log likelihood for each response, with missing cells zeroed.

        Args:
            design: Windowed impulse history plus the observed responses and
                the constants that put them back on their own scale.
            idx: Rows of the design to score; ``None`` takes all of them.

        Returns:
            Log density per response column, on the response's own scale. A
            column a row does not observe contributes zero rather than nan.
        """
        sl = slice(None) if idx is None else idx
        theta = self.response_params(design, idx=idx, training=False)
        y = self._response_values(design, sl)
        finite = torch.isfinite(y)
        mean = torch.as_tensor(design.y_mean, dtype=y.dtype, device=y.device)
        safe_y = torch.where(finite, y, mean)
        log_p = self._log_prob(safe_y, theta, design)
        return torch.where(finite, log_p, torch.zeros_like(log_p))

    def objective_ll(
        self,
        design: CDRDesign,
        idx: np.ndarray | torch.Tensor | slice | None = None,
        *,
        training: bool = True,
    ) -> torch.Tensor:
        """Per-observation log density, which is what the reference optimizes.

        Args:
            design: Windowed impulse history plus the observed responses.
            idx: Rows of the design in this minibatch; ``None`` takes all.
            training: Apply dropout, as a training step does. Scoring passes
                false.

        Returns:
            One value per row. Under a joint response it is the row's summed
            log density divided by the count of declared dependent variables,
            including any the row does not observe.
        """
        sl = slice(None) if idx is None else idx
        theta = self.response_params(design, idx=idx, training=training)
        y = self._response_values(design, sl)
        finite = torch.isfinite(y)
        mean = torch.as_tensor(design.y_mean, dtype=y.dtype, device=y.device)
        safe_y = torch.where(finite, y, mean)
        log_p = torch.where(
            finite,
            self._log_prob(safe_y, theta, design),
            torch.zeros_like(y),
        )
        if self.n_responses == 1:
            return log_p
        # The reference averages the row loss over the number of declared
        # dependent variables, including a declared response missing on this
        # row, before applying the minibatch-to-dataset scale.
        return log_p.sum(dim=-1) / self.n_responses

    def log_lik(
        self,
        design: CDRDesign,
        idx: np.ndarray | torch.Tensor | slice | None = None,
    ) -> torch.Tensor:
        """Return held-out log likelihood by observation.

        Args:
            design: Windowed impulse history plus the observed responses.
            idx: Rows of the design to score; ``None`` takes all of them.

        Returns:
            One value per row, summed over the response columns rather than
            averaged, so it is the row's joint held-out likelihood.
        """
        log_p = self.log_lik_by_response(design, idx=idx)
        return log_p if self.n_responses == 1 else log_p.sum(dim=-1)

    def predict_by_response(
        self,
        design: CDRDesign,
        idx: np.ndarray | torch.Tensor | slice | None = None,
    ) -> torch.Tensor:
        """Response means on each response's original scale.

        Args:
            design: Windowed impulse history for the responses to predict.
            idx: Rows of the design to predict; ``None`` takes all of them.

        Returns:
            The fitted distribution's mean, which for the skewed response adds
            the exponential component's mean to the Gaussian one.
        """
        theta = self.response_params(design, idx=idx, training=False)
        params = self._linked(theta)
        z = params["mu"]
        if "beta" in params:
            z = z + params["beta"]
        mean = torch.as_tensor(design.y_mean, dtype=z.dtype, device=z.device)
        sd = torch.as_tensor(design.y_sd, dtype=z.dtype, device=z.device)
        return z * sd + mean

    def predict(self, design: CDRDesign) -> torch.Tensor:
        """Point prediction on the response's own scale.

        The reference reports the fitted distribution's mode, which it obtains
        by sampling. This reports its mean, which is available in closed form
        and is deterministic; held-out likelihood, which is what the papers'
        comparisons use, is unaffected.

        Args:
            design: Windowed impulse history for every response to predict.
        """
        return self.predict_by_response(design)

    def ranef_penalty(
        self, scale: float, *, exclude: set[str] | None = None
    ) -> torch.Tensor:
        """L2 penalty on the centered random offsets, without a half factor.

        Covers the by-level intercepts and slopes and the network's own bias
        offsets, which is the set the reference penalizes at this scale.

        Args:
            scale: Multiplier on the summed penalty.
            exclude: Parameter names left out, so a term a variational
                posterior already penalizes through its KL is not counted
                twice.
        """
        exclude = set() if exclude is None else exclude
        total = torch.zeros((), dtype=torch.float32, device=self.device)
        matrices: list[tuple[str, torch.Tensor]] = []
        if self.random_intercept:
            matrices.append(("intercept_random", self.parameter("intercept_random")))
        if self.random_coefs:
            matrices.append(("coef_random", self.parameter("coef_random")))
        if self.item_n_levels:
            matrices.append(
                (
                    "item_intercept_random",
                    self.parameter("item_intercept_random"),
                )
            )
        if self.network_groups is None:
            for i, layer in enumerate(self.layers):
                if layer.bias_random is not None:
                    matrices.append((f"layers.{i}.bias_random", layer.bias_random))
        else:
            for group_index, group in enumerate(self.group_layers):
                for layer_index, layer in enumerate(group):
                    if layer.bias_random is not None:
                        matrices.append(
                            (
                                f"group_layers.{group_index}.{layer_index}.bias_random",
                                layer.bias_random,
                            )
                        )
        for name, raw in matrices:
            if name in exclude:
                continue
            centered = raw - raw.mean(dim=0, keepdim=True)
            total = total + (centered * centered).sum()
        return scale * total

    def regularization(self) -> torch.Tensor:
        """L2 on the network's hidden weights, which the output layer escapes."""
        total = torch.zeros((), dtype=torch.float32, device=self.device)
        layers = (
            self.layers
            if self.network_groups is None
            else (layer for group in self.group_layers for layer in group)
        )
        for layer in layers:
            total = total + (layer.kernel * layer.kernel).sum()
        return self.config.nn_regularizer_scale * total


CDRNN_VARIATIONAL_STEMS = frozenset(
    {
        "intercept",
        "coef",
        "intercept_random",
        "coef_random",
        "item_intercept_random",
    }
)


def cdrnn_variational_names(model: CDRNNModel) -> tuple[str, ...]:
    """Parameters the reference's ``random_variables=default`` samples.

    CDRNN defaults to every conventional regression component being a random
    variable while leaving the neural component as a point estimate.  This is
    a hybrid BBVI model.

    Args:
        model: Model whose trainable parameters are partitioned.

    Returns:
        Names whose leading segment is a conventional regression component, in
        :meth:`torch.nn.Module.named_parameters` order.
    """
    return tuple(
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and name.split(".", maxsplit=1)[0] in CDRNN_VARIATIONAL_STEMS
    )


def cdrnn_trainable_parameter_count(
    model: CDRNNModel, posterior: VariationalPosterior | None
) -> int:
    """Count optimized scalars, without counting substituted base tensors.

    Args:
        model: Model whose trainable parameters are counted.
        posterior: Variational posterior whose own tensors are counted in place
            of the model parameters it stands in for. ``None`` counts the
            model's trainable parameters alone.
    """
    selected = set() if posterior is None else set(posterior.names)
    return sum(
        parameter.numel()
        for parameter in (() if posterior is None else posterior.parameters())
    ) + sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and name not in selected
    )


@dataclass
class CDRNNFit:
    """CDRNN fitting hyperparameters."""

    n_iter: int = 100000
    learning_rate: float = 0.001
    minibatch_size: int = 1024
    ema_decay: float = 0.999
    max_global_norm: float | None = 1.0
    #: Entries the convergence window holds: held-out evaluations when a
    #: development partition exists, training iterations otherwise.
    convergence_n_iterates: int = 25
    convergence_alpha: float = 0.5
    #: Iterations between held-out evaluations while stopping early.
    eval_every: int = 10
    check_convergence: bool = True
    seed: int | None = None
    #: Training iterations between progress lines; zero disables logging.
    log_every: int = 100
    #: The released reading configurations use variational conventional
    #: effects and point-estimated neural weights.
    inference: str = "hybrid_variational"
    priors: CDRPriors = field(
        default_factory=lambda: CDRPriors(
            ranef_to_fixef_ratio=0.1,
            posterior_to_prior_ratio=0.01,
        )
    )
    posterior_scale_epsilon: float = 1e-5


@dataclass
class CDRNNAnalysis(_FitDiagnostics, LikelihoodModel):
    """The neural deconvolutional model over an event channel.

    ``nn_inputs`` names the impulses the network reads. Leaving it unset gives
    the reference's own rule, every impulse but the constant-one rate;
    naming a subset states the papers' linearity null, where the omitted
    predictor keeps its own response channel and coefficient.
    """

    rate_impulse: str = "rate"
    #: Stable formula order for impulse channels. This matters when the same
    #: full model is reached through different focal-predictor ablations.
    impulse_order: tuple[str, ...] | None = None
    nn_inputs: tuple[str, ...] | None = None
    #: Inputs for the baseline side of ``evaluate()``. When this is set, the
    #: baseline retains the focal impulse and differs only in what its network
    #: reads. This represents a linearity null rather than an ablation.
    baseline_nn_inputs: tuple[str, ...] | None = None
    #: Independent networks and their disjoint terminal impulses.
    network_groups: tuple[tuple[str, ...], ...] | None = None
    baseline_network_groups: tuple[tuple[str, ...], ...] | None = None
    impulse_ops: dict[str, str] = field(default_factory=dict)
    history_length: int = 32
    t_delta_cutoff: float | None = 60.0
    cutoff_mode: str = "reference"
    random_intercept: bool = True
    random_coefs: bool = True
    #: Give the word an intercept of its own, as the papers' formulas do.
    random_item_intercept: bool = True
    config: CDRNNConfig = field(default_factory=CDRNNConfig)
    device: str = "cpu"
    #: Directory for each side's held-out TSV and checkpoint; each fold
    #: overwrites the previous fold's files. ``None`` writes nothing.
    artifact_dir: str | None = None
    fit: CDRNNFit = field(default_factory=CDRNNFit)
    #: Stop on the fit of the rows a fold assigns to neither side, which for
    #: the released splits is the development partition.
    early_stopping: bool = True

    name = "cdrnn"
    owns_loglik = True
    needs_events = True
    needs_grouping = True

    def configuration(self) -> dict:
        """Return declared options without fitted state."""
        return asdict(self)

    @staticmethod
    def _groups(X: np.ndarray) -> np.ndarray:
        return X[:, -2].astype(np.int64)

    def _dev_design(
        self,
        events,
        train_mask,
        test_mask,
        train_design,
        names,
        known,
        unseen,
        group_codes,
        unknown_group,
    ):
        """Design for the rows the fold leaves out of both sides.

        The reference fits on one partition of the release, stops on a second
        and reports on a third; the second is what a two-way fold leaves
        unassigned, so it is read from there.
        """
        if (
            not self.early_stopping
            or train_mask is None
            or test_mask is None
            or events.y is None
            or events.group is None
        ):
            return None
        held = ~(np.asarray(train_mask) | np.asarray(test_mask))
        if not held.any():
            return None
        dev = events.subset(held)
        dev_items = None
        if self.random_item_intercept and dev.item is not None:
            # A development word the fit never saw contributes no offset, as
            # a held-out one does.
            dev_items = torch.as_tensor(
                [known.get(int(code), unseen) for code in dev.item], dtype=torch.int64
            )
        # Random-effect rows are indexed by the fold's training levels, so the
        # dataset-wide codes have to go through the same table the fold built.
        dev_groups = np.asarray(
            [group_codes.get(int(code), unknown_group) for code in dev.group],
            dtype=np.int64,
        )
        design = build_design_from_events(
            response_time=dev.time,
            response_series=dev.series,
            response_y=dev.y,
            response_group=dev_groups,
            impulse_time=events.impulses.index.time,
            impulse_series=events.impulses.index.series,
            impulse_values=transform_impulses(
                events.impulses.matrix(names), names, self.impulse_ops, 0
            ),
            impulse_names=names,
            history_length=self.history_length,
            t_delta_cutoff=self.t_delta_cutoff,
            cutoff_mode=self.cutoff_mode,
            y_mean=train_design.y_mean,
            y_sd=train_design.y_sd,
            item=dev_items,
        )
        return design_on(design, self.device)

    def _item_codes(self, train_events, test_events):
        """Word codes for both partitions, with unseen words sent to a default.

        Levels come from the fitted partition, so a word the fit never saw
        contributes no offset, which is how the reference treats one.
        """
        if not self.random_item_intercept or train_events.item is None:
            return None, None, 0, {}
        known = {int(code): i for i, code in enumerate(np.unique(train_events.item))}
        unseen = len(known)
        train = torch.as_tensor(
            [known[int(c)] for c in train_events.item], dtype=torch.int64
        )
        test = torch.as_tensor(
            [known.get(int(c), unseen) for c in test_events.item], dtype=torch.int64
        )
        return train, test, unseen, known

    def _inputs(self, names: list[str], *, is_full: bool = True) -> list[str]:
        selected = (
            self.baseline_nn_inputs
            if not is_full and self.baseline_nn_inputs is not None
            else self.nn_inputs
        )
        if selected is not None:
            return [n for n in names if n in selected]
        return [n for n in names if n != self.rate_impulse]

    def _network_groups(self, names: list[str], *, is_full: bool):
        groups = self.network_groups if is_full else self.baseline_network_groups
        if groups is None:
            return None
        return tuple(tuple(name for name in group if name in names) for group in groups)

    def fit_score(
        self,
        y_train: np.ndarray,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        *,
        is_full: bool,
        events,
        train_mask: np.ndarray | None = None,
        test_mask: np.ndarray | None = None,
    ):
        """Fit the network on the training responses and score the held-out ones.

        Under ``early_stopping``, rows a fold assigns to neither side become the
        development partition the fit stops on.

        Args:
            y_train: Dependent variable on the training rows.
            X_train: Training design matrix, read only for the participant
                codes in its second-to-last column.
            X_test: The same columns on the held-out rows.
            y_test: Held-out dependent variable, scored under the fitted model.
            is_full: Fit the full term set rather than the baseline. A baseline
                that declares its own network inputs or groups keeps the focal
                impulse and drops only its access to the network, which is the
                linearity null rather than an ablation.
            events: Response timeline and the impulse stream the network
                convolves over, which carries the predictors; supplied by
                ``evaluate()`` because ``needs_events`` is set.
            train_mask: Responses of *events* this fold fits on; ``None`` takes
                all of them and leaves no development partition.
            test_mask: Responses of *events* this fold scores on; ``None`` takes
                all of them and leaves no development partition.

        Returns:
            Summed held-out log likelihood, the point predictions, and the
            per-observation likelihood the paired significance tests need.

        Raises:
            ValueError: When the configured inference mode is neither
                ``hybrid_variational`` nor ``point``.
        """
        self._fit_diagnostics = None
        from replica.analysis.contracts.models import FoldScore
        from replica.analysis.models.deconvolution.analysis import (
            fold_designs,
            write_fold_artifacts,
        )
        from replica.analysis.models.deconvolution.train import (
            fit_cdr,
            reproducibility_metadata,
        )

        # An ordinary baseline ablates the focal impulse. A linearity null
        # keeps that impulse's channel, coefficient, and random slope, while
        # removing only its access to the response network.
        keeps_focal = not is_full and (
            self.baseline_nn_inputs is not None
            or self.baseline_network_groups is not None
        )
        names, train_events, test_events, train_design, test_design = fold_designs(
            events=events,
            is_full=is_full or keeps_focal,
            y_train=y_train,
            y_test=y_test,
            train_groups=self._groups(X_train),
            test_groups=self._groups(X_test),
            train_mask=train_mask,
            test_mask=test_mask,
            impulse_ops=self.impulse_ops,
            history_length=self.history_length,
            t_delta_cutoff=self.t_delta_cutoff,
            cutoff_mode=self.cutoff_mode,
            ddof=0,
            impulse_order=self.impulse_order,
        )
        # ``fold_designs`` reserves the next code for held-out-only groups;
        # only training levels receive neural random-effect parameters.
        n_levels = int(train_design.gf_y.max()) + 1
        train_items, test_items, n_items, item_map = self._item_codes(
            train_events, test_events
        )
        model = CDRNNModel(
            names,
            nn_inputs=self._inputs(names, is_full=is_full),
            network_groups=self._network_groups(names, is_full=is_full),
            config=self.config,
            rangf_n_levels=n_levels,
            random_intercept=self.random_intercept,
            random_coefs=self.random_coefs,
            item_n_levels=n_items,
            # The reference divides impulse timestamps by their spread over
            # the whole impulse table, which every partition shares.
            x_time_sd=float(np.std(events.impulses.index.time)) or 1.0,
            history_length=self.history_length,
            seed=self.fit.seed,
        ).to(self.device)
        if self.fit.inference not in ("hybrid_variational", "point"):
            raise ValueError(f"unknown CDRNN inference mode: {self.fit.inference!r}")
        posterior = (
            VariationalPosterior(
                model,
                self.fit.priors,
                names=cdrnn_variational_names(model),
                scale_epsilon=self.fit.posterior_scale_epsilon,
                seed=self.fit.seed,
            ).to(self.device)
            if self.fit.inference == "hybrid_variational"
            else None
        )
        train_design = design_on(replace(train_design, item=train_items), self.device)
        test_design = design_on(replace(test_design, item=test_items), self.device)
        train_levels = np.unique(self._groups(X_train))
        group_codes = {int(level): index for index, level in enumerate(train_levels)}
        dev_design = self._dev_design(
            events,
            train_mask,
            test_mask,
            train_design,
            names,
            item_map,
            n_items,
            group_codes,
            n_levels,
        )

        result, ema = fit_cdr(
            model,
            train_design,
            n_iter=self.fit.n_iter,
            learning_rate=self.fit.learning_rate,
            minibatch_size=self.fit.minibatch_size,
            ema_decay=self.fit.ema_decay,
            ema_updates_per_batch=1,
            ranef_regularizer_scale=self.config.ranef_regularizer_scale,
            convergence_n_iterates=self.fit.convergence_n_iterates,
            convergence_alpha=self.fit.convergence_alpha,
            check_convergence=self.fit.check_convergence,
            optimizer="adam",
            max_global_norm=self.fit.max_global_norm,
            dev_design=dev_design,
            eval_every=self.fit.eval_every,
            two_tailed_convergence=dev_design is None,
            seed=self.fit.seed,
            log_every=self.fit.log_every,
            posterior=posterior,
        )

        ema.swap_in()
        estimate = None if posterior is None else posterior.means()
        with torch.no_grad(), model.substituted(estimate):
            per_obs = model.log_lik(test_design).cpu().numpy().astype(np.float64)
            y_pred = model.predict(test_design).cpu().numpy().astype(np.float64)
        # A checkpoint is evidence for the state that produced the artifact.
        # Capture the installed best/EMA estimate before restoring the raw
        # optimizer parameters, so the checkpoint rescores to the reported
        # likelihood.
        checkpoint_state = None
        if self.artifact_dir is not None:
            effective = {
                name: parameter
                if estimate is None or name not in estimate
                else estimate[name]
                for name, parameter in model.named_parameters()
            }
            checkpoint_state = {
                f"estimate_{name.replace('.', '__')}": parameter.detach()
                .cpu()
                .numpy()
                .copy()
                for name, parameter in effective.items()
            }
            if posterior is not None:
                checkpoint_state.update(
                    {
                        f"posterior_{name.replace('.', '__')}": parameter.detach()
                        .cpu()
                        .numpy()
                        .copy()
                        for name, parameter in posterior.named_parameters()
                    }
                )
        ema.swap_out()
        if self.artifact_dir is not None:
            write_fold_artifacts(
                self.artifact_dir,
                is_full=is_full,
                events=test_events,
                y_true=y_test,
                y_pred=y_pred,
                loglik=per_obs,
            )
            from pathlib import Path

            assert checkpoint_state is not None
            state = checkpoint_state
            side = "full" if is_full else "baseline"
            state.update(
                n_iterations=np.asarray(result.n_iterations),
                converged=np.asarray(result.converged),
                final_mean_loss=np.asarray(result.final_mean_loss),
                y_mean=np.asarray(train_design.y_mean),
                y_sd=np.asarray(train_design.y_sd),
                seed=np.asarray(-1 if self.fit.seed is None else self.fit.seed),
                side=np.asarray(side),
                model_kind=np.asarray("cdrnn"),
                impulse_names=np.asarray(names),
                # A side's ablation is its inputs plus the group signature:
                # the additive-null contrast keeps every input and differs
                # only in how they split across networks.
                nn_input_names=np.asarray(sorted(self._inputs(names, is_full=is_full))),
                network_group_signature=np.asarray(
                    "|".join(
                        ",".join(group)
                        for group in (
                            self._network_groups(names, is_full=is_full) or ()
                        )
                    )
                ),
                inference=np.asarray(self.fit.inference),
                trainable_parameter_count=np.asarray(
                    cdrnn_trainable_parameter_count(model, posterior)
                ),
            )
            state.update(reproducibility_metadata(model, self.fit.seed))
            np.savez_compressed(
                Path(self.artifact_dir) / f"{side}_checkpoint.npz", **state
            )
        self._fit_diagnostics = {
            "convergence_applicable": True,
            "converged": bool(result.converged),
            "details": {
                "n_iterations": int(result.n_iterations),
                "final_mean_loss": float(result.final_mean_loss),
            },
        }
        return FoldScore(
            loglik=float(per_obs.sum()), y_pred=y_pred, per_obs_loglik=per_obs
        )
