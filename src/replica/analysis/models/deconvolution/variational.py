"""Mean-field variational posterior for CDR models."""

from collections.abc import Iterable
from dataclasses import dataclass

import torch

from replica.analysis.models.deconvolution.kernels import softplus, softplus_inverse
from replica.analysis.models.deconvolution.model import CDRModel

#: Parameters holding by-group offsets rather than population values.
RANDOM_PARAMETERS = (
    "intercept_random",
    "intercept_random_extra",
    "item_intercept_random",
    "coef_random",
    "irf_raw_random",
)


@dataclass(frozen=True)
class CDRPriors:
    """Prior and initial posterior scale configuration."""

    intercept_sd: float = 1.0
    coef_sd: float = 1.0
    irf_param_sd: float = 1.0
    y_sd_sd: float = 1.0
    ranef_to_fixef_ratio: float = 1.0
    posterior_to_prior_ratio: float = 1.0

    def prior_sd(self, name: str) -> float:
        """Prior scale for one parameter, by the role its name gives it.

        Args:
            name: A ``named_parameters`` name; its first dotted segment is the
                role, and an unknown role raises.
        """
        stem = name.split(".", maxsplit=1)[0]
        population = {
            "intercept": self.intercept_sd,
            "intercept_random": self.intercept_sd,
            "intercept_random_extra": self.intercept_sd,
            "item_intercept_random": self.intercept_sd,
            "coef": self.coef_sd,
            "coef_random": self.coef_sd,
            "irf_raw": self.irf_param_sd,
            "irf_raw_random": self.irf_param_sd,
            "y_sd_raw": self.y_sd_sd,
        }
        if stem not in population:
            raise KeyError(f"no prior is defined for parameter {name!r}")
        sd = population[stem]
        if stem in RANDOM_PARAMETERS:
            sd *= self.ranef_to_fixef_ratio
        return sd

    def posterior_sd_init(self, name: str) -> float:
        """Where one posterior's scale starts, relative to its prior.

        Args:
            name: A ``named_parameters`` name, resolved as in :meth:`prior_sd`.
        """
        # The reference first builds the random-effect prior by applying the
        # random/fixed ratio, then initializes its posterior at the same
        # posterior/prior ratio as every other parameter.  ``prior_sd`` has
        # already applied the former, so applying it again here would square
        # the random/fixed ratio.
        return self.prior_sd(name) * self.posterior_to_prior_ratio


def _key(name: str) -> str:
    """Parameter name as a ``ParameterDict`` key, which admits no dots."""
    return name.replace(".", "__")


class VariationalPosterior(torch.nn.Module):
    """Independent normal posterior over selected trainable parameters."""

    def __init__(
        self,
        model: CDRModel,
        priors: CDRPriors | None = None,
        *,
        names: Iterable[str] | None = None,
        init_sd: float = 0.0,
        scale_epsilon: float = 0.0,
        seed: int | None = None,
    ):
        """Build one independent normal per selected parameter.

        A population parameter's prior is centred on the model's current
        value; a by-group offset's prior is centred at zero.

        Args:
            model: Model whose trainable parameters the posterior covers and
                whose current values seed the population priors.
            priors: Prior and initial-posterior scales; ``None`` takes the
                :class:`CDRPriors` defaults.
            names: Parameters to give a posterior, which must be unique and
                trainable in *model*. ``None`` selects every trainable one.
            init_sd: Standard deviation of the noise added to the initial
                locations and raw scales; ``0.0`` starts them deterministically.
            scale_epsilon: Floor added to every posterior scale after its
                softplus link.
            seed: Seed for the initial-jitter generator; ``None`` leaves the
                generator unseeded.
        """
        super().__init__()
        if init_sd < 0:
            raise ValueError("init_sd must be non-negative")
        if scale_epsilon < 0:
            raise ValueError("scale_epsilon must be non-negative")
        self.priors = priors or CDRPriors()
        self.scale_epsilon = float(scale_epsilon)
        generator = torch.Generator(device=model.device)
        if seed is not None:
            generator.manual_seed(seed)

        def jitter(value: torch.Tensor) -> torch.Tensor:
            if init_sd == 0:
                return value.clone()
            return (
                value
                + torch.randn(
                    value.shape,
                    dtype=value.dtype,
                    device=value.device,
                    generator=generator,
                )
                * init_sd
            )

        available = {n: p for n, p in model.named_parameters() if p.requires_grad}
        if names is None:
            self.names = list(available)
        else:
            requested = list(names)
            if len(requested) != len(set(requested)):
                raise ValueError("variational parameter names must be unique")
            missing = set(requested) - set(available)
            if missing:
                raise ValueError(
                    f"unknown variational parameter names: {sorted(missing)!r}"
                )
            self.names = requested
        if not self.names:
            raise ValueError("a variational posterior needs at least one parameter")
        self.loc = torch.nn.ParameterDict()
        self.scale_raw = torch.nn.ParameterDict()
        for name in self.names:
            param = model.parameter(name)
            is_random = name.split(".")[0] in RANDOM_PARAMETERS
            prior_loc = torch.zeros_like(param) if is_random else param.detach().clone()
            self.register_buffer(f"prior_loc_{_key(name)}", prior_loc)
            self.register_buffer(
                f"prior_sd_{_key(name)}",
                torch.full_like(param, self.priors.prior_sd(name)),
            )
            self.loc[_key(name)] = torch.nn.Parameter(jitter(prior_loc))
            scale_init = torch.full_like(
                param,
                float(
                    softplus_inverse(
                        torch.tensor(
                            self.priors.posterior_sd_init(name), dtype=param.dtype
                        )
                    )
                ),
            )
            self.scale_raw[_key(name)] = torch.nn.Parameter(jitter(scale_init))

    @property
    def device(self) -> torch.device:
        """Device the posterior's tensors sit on."""
        return self.loc[_key(self.names[0])].device

    def scale(self, name: str) -> torch.Tensor:
        """One posterior's standard deviation, softplus-linked and floored.

        Args:
            name: A parameter name from :attr:`names`.
        """
        return softplus(self.scale_raw[_key(name)]) + self.scale_epsilon

    def means(self) -> dict[str, torch.Tensor]:
        """Posterior means, which are what prediction substitutes."""
        return {name: self.loc[_key(name)] for name in self.names}

    def sample(
        self, generator: torch.Generator | None = None
    ) -> dict[str, torch.Tensor]:
        """One reparameterized draw from the posterior.

        Args:
            generator: Source of the standard-normal noise; ``None`` draws
                from the ambient torch RNG.
        """
        out = {}
        for name in self.names:
            loc = self.loc[_key(name)]
            noise = torch.randn(
                loc.shape, dtype=loc.dtype, generator=generator, device=loc.device
            )
            out[name] = loc + self.scale(name) * noise
        return out

    def kl(self) -> torch.Tensor:
        """Analytic Kullback-Leibler divergence from the priors, summed."""
        total = torch.zeros((), dtype=torch.float32, device=self.device)
        for name in self.names:
            prior_loc = getattr(self, f"prior_loc_{_key(name)}")
            prior_sd = getattr(self, f"prior_sd_{_key(name)}")
            posterior = torch.distributions.Normal(
                self.loc[_key(name)], self.scale(name)
            )
            prior = torch.distributions.Normal(prior_loc, prior_sd)
            total = total + torch.distributions.kl_divergence(posterior, prior).sum()
        return total
