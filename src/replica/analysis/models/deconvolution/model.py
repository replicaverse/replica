"""The CDR mean function and Gaussian likelihood."""

from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np
import torch

from replica.analysis.models.deconvolution.data import CDRDesign
from replica.analysis.models.deconvolution.kernels import (
    EPSILON,
    constrain_lower,
    constrain_lower_inverse,
    constrain_upper,
    constrain_upper_inverse,
    double_gamma_irf,
    exp_irf,
    lcg_irf,
    normal_irf,
    shifted_gamma_kgt1_irf,
    softplus,
    softplus_inverse,
)

LCG_BASES = 10


def _float_tensor(value, device) -> torch.Tensor:
    """Convert numeric input to a writable float32 tensor."""
    if isinstance(value, np.ndarray) and not value.flags.writeable:
        value = value.copy()
    return torch.as_tensor(value, dtype=torch.float32, device=device)


def design_on(design: CDRDesign, device) -> CDRDesign:
    """The same design with its arrays as tensors on ``device``.

    Fitting reads the impulse history once per minibatch, so the history
    lives wherever the parameters do rather than crossing to it each step.

    Args:
        design: Design whose windowed arrays are moved; it is left untouched.
        device: Torch device, or anything ``torch.as_tensor`` accepts as one.
    """
    from dataclasses import replace

    moved = {
        name: torch.as_tensor(getattr(design, name), device=device)
        for name in ("X_2d", "time_X_2d", "time_mask", "time_y", "y", "gf_y")
    }
    if design.item is not None:
        moved["item"] = torch.as_tensor(design.item, device=device)
    return replace(design, **moved)


def _lcg_family(
    bases: int = LCG_BASES,
) -> tuple[tuple[str, ...], tuple[float | None, ...], tuple[float, ...]]:
    """Summed-Gaussian family: a location, value and scale per basis.

    Parameter order and initialization follow the reference: locations all at
    zero (the released configurations leave this family's spacing power at
    zero), the first value one and the rest zero, scales counting up from one.
    Only the scales are bounded.
    """
    names = tuple(
        [f"x{i}" for i in range(1, bases + 1)]
        + [f"y{i}" for i in range(1, bases + 1)]
        + [f"s{i}" for i in range(1, bases + 1)]
    )
    bounds: tuple[float | None, ...] = (None,) * (2 * bases) + (0.0,) * bases
    defaults = (
        (0.0,) * bases
        + (1.0,)
        + (0.0,) * (bases - 1)
        + tuple(float(i) for i in range(1, bases + 1))
    )
    return names, bounds, defaults


# family -> (ordered param names, per-param lower bound or None, default init)
FAMILIES: dict[
    str, tuple[tuple[str, ...], tuple[float | None, ...], tuple[float, ...]]
] = {
    "Exp": (("beta",), (0.0,), (1.0,)),
    "ExpRateGT1": (("beta",), (1.0,), (2.0,)),
    "Normal": (("mu", "sigma2"), (None, 0.0), (0.0, 1.0)),
    "ShiftedGammaShapeGT1": (
        ("alpha", "beta", "delta"),
        (1.0, 0.0, None),
        (2.0, 1.0, -1.0),
    ),
    # The released configurations write this family as ``G()``; the toolkit
    # names its models LCG.
    "LCG": _lcg_family(),
    "HRFDoubleGamma1": (("beta",), (0.0,), (1.0,)),
    "HRFDoubleGamma2": (
        ("alpha", "beta"),
        (1.0, 0.0),
        (6.0, 1.0),
    ),
    "HRFDoubleGamma3": (
        ("alpha", "beta", "c"),
        (1.0, 0.0, None),
        (6.0, 1.0, 1.0 / 6.0),
    ),
    "HRFDoubleGamma4": (
        ("alpha_main", "alpha_undershoot", "beta", "c"),
        (1.0, 1.0, 0.0, None),
        (6.0, 16.0, 1.0, 1.0 / 6.0),
    ),
    "HRFDoubleGamma5": (
        (
            "alpha_main",
            "alpha_undershoot",
            "beta_main",
            "beta_undershoot",
            "c",
        ),
        (1.0, 1.0, 0.0, 0.0, None),
        (6.0, 16.0, 1.0, 1.0, 1.0 / 6.0),
    ),
}


@dataclass
class KernelTerm:
    """One impulse convolved with one parametric kernel."""

    impulse: str
    family: str
    param_init: dict[str, float] = field(default_factory=dict)
    ran: bool = False
    irf_id: str | None = None


@dataclass
class DiracTerm:
    """A response-simultaneous (unconvolved) impulse."""

    impulse: str


class _SubstitutableParameters:
    """Resolve module parameters through an optional temporary substitution.

    Both native CDR implementations use this indirection when evaluating a
    posterior draw. Keeping the lookup and context-management semantics here
    prevents the linear and neural models from drifting apart.
    """

    _substitute: dict[str, torch.Tensor] | None

    def parameter(self, name: str) -> torch.Tensor:
        """Return a named parameter or its active substituted value.

        Args:
            name: A ``named_parameters`` name; dotted numeric segments index
                into parameter or module lists.
        """
        if self._substitute is not None and name in self._substitute:
            return self._substitute[name]
        value: object = self
        for part in name.split("."):
            value = value[int(part)] if part.isdigit() else getattr(value, part)  # type: ignore[index]
        return value  # type: ignore[return-value]

    @contextmanager
    def substituted(self, values: dict[str, torch.Tensor] | None):
        """Temporarily resolve supplied values in place of stored parameters.

        Args:
            values: Parameter names mapped to tensors that stand in for them.
                ``None`` leaves the stored parameters in force.

        """
        previous = self._substitute
        self._substitute = values
        try:
            yield self
        finally:
            self._substitute = previous


class CDRModel(_SubstitutableParameters, torch.nn.Module):
    """CDR with fixed effects plus optional random effects.

    ``rangf_n_levels`` counts training levels only; an extra final code is
    reserved for unknown levels and always maps to a zero offset. ``delta``
    (upper-bounded at 0) is the one upper-bounded parameter in the wired
    families; its offsets mirror the lower-bounded transform.
    """

    def __init__(
        self,
        kernel_terms: list[KernelTerm],
        dirac_terms: list[DiracTerm],
        impulse_names: list[str],
        *,
        rangf_n_levels: int | None = None,
        random_intercept: bool = False,
        random_coefs: bool = False,
        extra_random_intercept_n_levels: tuple[int, ...] = (),
        intercept_init: float = 0.0,
        y_sd_init: float = 1.0,
        normalize_truncated_kernels: bool = True,
        seed: int | None = None,
    ):
        """Allocate one coefficient per terminal and one parameter set per kernel.

        Args:
            kernel_terms: Convolved terms, each naming an impulse, a kernel
                family, initial kernel parameters, and whether the kernel
                varies by group. Terms sharing an ``irf_id`` share parameters.
            dirac_terms: Impulses entering unconvolved, contributing only at
                the response's own timestamp.
            impulse_names: Column order of the design's impulse axis; every
                term's impulse has to appear in it.
            rangf_n_levels: Training levels of the by-group factor. One extra
                code past the end stands for an unseen level and offsets zero.
            random_intercept: Give the grouping factor a by-level intercept.
            random_coefs: Give the grouping factor by-level coefficients on
                every terminal.
            extra_random_intercept_n_levels: Level counts of further grouping
                factors, read from the design's later grouping columns.
            intercept_init: Starting population intercept, on the scale the
                design's ``y_mean``/``y_sd`` declare.
            y_sd_init: Starting residual standard deviation, on that same scale.
            normalize_truncated_kernels: Renormalize the truncated normal and
                shifted-gamma kernels to unit mass over the positive lags.
            seed: Seed for the random-IRF offset initialisation; ``None``
                leaves the generator unseeded.
        """
        super().__init__()
        self.kernel_terms = kernel_terms
        self.dirac_terms = dirac_terms
        self.impulse_names = list(impulse_names)
        self.col = {name: i for i, name in enumerate(impulse_names)}
        self.n_terminals = len(kernel_terms) + len(dirac_terms)
        self.rangf_n_levels = rangf_n_levels
        self.random_intercept = random_intercept
        self.random_coefs = random_coefs
        self.extra_random_intercept_n_levels = tuple(
            int(x) for x in extra_random_intercept_n_levels
        )
        self.normalize_truncated_kernels = normalize_truncated_kernels
        generator = None if seed is None else torch.Generator().manual_seed(seed)

        self.intercept = torch.nn.Parameter(
            torch.tensor(intercept_init, dtype=torch.float32)
        )
        self.coef = torch.nn.Parameter(
            torch.zeros(self.n_terminals, dtype=torch.float32)
        )
        self.y_sd_raw = torch.nn.Parameter(
            softplus_inverse(torch.tensor(y_sd_init, dtype=torch.float32))
        )

        self.irf_raw = torch.nn.ParameterList()
        self.kernel_parameter_index: list[int] = []
        self._kernel_parameter_terms: list[int] = []
        shared: dict[str, int] = {}
        for term_idx, term in enumerate(kernel_terms):
            if term.irf_id is not None and term.irf_id in shared:
                parameter_idx = shared[term.irf_id]
                owner = kernel_terms[self._kernel_parameter_terms[parameter_idx]]
                if owner.family != term.family or owner.ran != term.ran:
                    raise ValueError(
                        f"shared IRF {term.irf_id!r} has inconsistent family or "
                        "random-effect declarations"
                    )
                self.kernel_parameter_index.append(parameter_idx)
                continue
            parameter_idx = len(self.irf_raw)
            self.kernel_parameter_index.append(parameter_idx)
            self._kernel_parameter_terms.append(term_idx)
            if term.irf_id is not None:
                shared[term.irf_id] = parameter_idx
            names, bounds, defaults = FAMILIES[term.family]
            raws = []
            for name, lb, default in zip(names, bounds, defaults, strict=True):
                init = float(term.param_init.get(name, default))
                if name == "delta":
                    raw = constrain_upper_inverse(init, 0.0)
                elif lb is None:
                    raw = torch.tensor(init, dtype=torch.float32)
                else:
                    raw = constrain_lower_inverse(init, lb)
                raws.append(raw)
            self.irf_raw.append(torch.nn.Parameter(torch.stack(raws)))

        self._substitute: dict[str, torch.Tensor] | None = None

        L = rangf_n_levels or 0
        if random_intercept:
            self.intercept_random = torch.nn.Parameter(
                torch.zeros(L, dtype=torch.float32)
            )
        if random_coefs:
            self.coef_random = torch.nn.Parameter(
                torch.zeros(L, self.n_terminals, dtype=torch.float32)
            )
        self.intercept_random_extra = torch.nn.ParameterList(
            [
                torch.nn.Parameter(torch.zeros(n, dtype=torch.float32))
                for n in self.extra_random_intercept_n_levels
            ]
        )
        self.irf_raw_random = torch.nn.ParameterList()
        for term_idx in self._kernel_parameter_terms:
            term = kernel_terms[term_idx]
            if term.ran:
                n_par = len(FAMILIES[term.family][0])
                self.irf_raw_random.append(
                    torch.nn.Parameter(
                        torch.randn(L, n_par, dtype=torch.float32, generator=generator)
                        * EPSILON
                    )
                )
            else:
                self.irf_raw_random.append(
                    torch.nn.Parameter(torch.zeros(0, 0), requires_grad=False)
                )

    @property
    def device(self) -> torch.device:
        """Device the model's parameters sit on."""
        return self.intercept.device

    def constrained_params(self, term_idx: int) -> torch.Tensor:
        """Fixed (population) kernel parameters of one term, constrained.

        Args:
            term_idx: Index into ``kernel_terms``.
        """
        term = self.kernel_terms[term_idx]
        names, bounds, _ = FAMILIES[term.family]
        parameter_idx = self.kernel_parameter_index[term_idx]
        raw = self.parameter(f"irf_raw.{parameter_idx}")
        out = []
        for j, (name, lb) in enumerate(zip(names, bounds, strict=True)):
            if name == "delta":
                out.append(constrain_upper(raw[j], 0.0))
            elif lb is None:
                out.append(raw[j])
            else:
                out.append(constrain_lower(raw[j], lb))
        return torch.stack(out)

    def _random_offsets(self, term_idx: int) -> torch.Tensor:
        """Centered random offsets for one kernel term."""
        term = self.kernel_terms[term_idx]
        theta = self.constrained_params(term_idx)
        names, bounds, _ = FAMILIES[term.family]
        parameter_idx = self.kernel_parameter_index[term_idx]
        raw = self.parameter(f"irf_raw_random.{parameter_idx}")
        offsets = []
        for j, (name, lb) in enumerate(zip(names, bounds, strict=True)):
            col = raw[:, j]
            if name == "delta":
                offset = -softplus(col)
                mean = offset.mean()
                offset = (offset - mean) * (theta[j] + EPSILON) / (mean - EPSILON)
            elif lb is None:
                offset = col - col.mean()
            else:
                offset = softplus(col)
                mean = offset.mean()
                offset = (offset - mean) * (theta[j] - lb - EPSILON) / (mean + EPSILON)
            offsets.append(offset)
        return torch.stack(offsets, dim=1)

    def _row_params(self, term_idx: int, gf: torch.Tensor | None) -> torch.Tensor:
        """Per-row kernel parameters, including random offsets."""
        term = self.kernel_terms[term_idx]
        theta = self.constrained_params(term_idx)
        if not term.ran or gf is None:
            return theta.unsqueeze(0)
        offset = self._random_offsets(term_idx)
        offset = torch.cat(
            [
                offset,
                torch.zeros(
                    1, offset.shape[1], dtype=offset.dtype, device=offset.device
                ),
            ],
            dim=0,
        )
        return theta.unsqueeze(0) + offset[gf]

    def _apply_kernel(
        self, family: str, t_delta: torch.Tensor, params: torch.Tensor
    ) -> torch.Tensor:
        if family == "LCG":
            # A basis axis is inserted so the (row, history) grid broadcasts
            # against the per-basis locations, values and scales.
            n = params.shape[-1] // 3
            basis = params.unsqueeze(-2)
            return lcg_irf(
                t_delta, basis[..., :n], basis[..., n : 2 * n], basis[..., 2 * n :]
            )
        p = [params[..., j].unsqueeze(-1) for j in range(params.shape[-1])]
        if family in ("Exp", "ExpRateGT1"):
            return exp_irf(t_delta, p[0])
        if family == "Normal":
            return normal_irf(
                t_delta,
                p[0],
                p[1],
                normalize=self.normalize_truncated_kernels,
            )
        if family == "ShiftedGammaShapeGT1":
            return shifted_gamma_kgt1_irf(
                t_delta,
                p[0],
                p[1],
                p[2],
                normalize=self.normalize_truncated_kernels,
            )
        if family.startswith("HRFDoubleGamma"):
            if family == "HRFDoubleGamma1":
                alpha_main, alpha_undershoot = 6.0, 16.0
                beta_main = beta_undershoot = p[0]
                c = torch.full_like(p[0], 1.0 / 6.0)
            elif family == "HRFDoubleGamma2":
                alpha_main = p[0]
                alpha_undershoot = p[0] + 10.0
                beta_main = beta_undershoot = p[1]
                c = torch.full_like(p[0], 1.0 / 6.0)
            elif family == "HRFDoubleGamma3":
                alpha_main = p[0]
                alpha_undershoot = p[0] + 10.0
                beta_main = beta_undershoot = p[1]
                c = p[2]
            elif family == "HRFDoubleGamma4":
                alpha_main, alpha_undershoot = p[0], p[1]
                beta_main = beta_undershoot = p[2]
                c = p[3]
            else:
                alpha_main, alpha_undershoot = p[0], p[1]
                beta_main, beta_undershoot, c = p[2], p[3], p[4]
            return double_gamma_irf(
                t_delta,
                alpha_main,
                alpha_undershoot,
                beta_main,
                beta_undershoot,
                c,
            )
        raise ValueError(f"unknown kernel family: {family}")

    def convolved(
        self, design: CDRDesign, idx: np.ndarray | None = None
    ) -> torch.Tensor:
        """Each terminal's convolution of the impulse history, before its
        coefficient, shape ``(n_responses, n_terminals)``.

        These are the response-aligned predictors the reference writes out to
        carry a fitted model into a second-stage regression, so a coefficient
        it estimates there is a coefficient on a fitted response shape.

        Args:
            design: Windowed impulse history and response timestamps.
            idx: Rows of the design to convolve; ``None`` takes all of them.
        """
        sl = slice(None) if idx is None else idx
        device = self.device
        X = torch.as_tensor(design.X_2d[sl], device=device)
        time_X = torch.as_tensor(design.time_X_2d[sl], device=device)
        time_y = torch.as_tensor(design.time_y[sl], device=device)
        gf = (
            torch.as_tensor(design.gf_y[sl, 0], device=device)
            if len(design.gf_y) and design.gf_y.shape[1]
            else None
        )
        t_delta = time_y[:, None, None] - time_X

        convs = []
        for term_idx, term in enumerate(self.kernel_terms):
            k = self.col[term.impulse]
            params = self._row_params(term_idx, gf)
            irf = self._apply_kernel(term.family, t_delta[:, :, k], params)
            convs.append((X[:, :, k] * irf).sum(dim=1))
        for term in self.dirac_terms:
            k = self.col[term.impulse]
            aligned = (t_delta[:, -1, k] == 0).to(X.dtype)
            convs.append(X[:, -1, k] * aligned)
        return torch.stack(convs, dim=1)

    def forward(
        self,
        design: CDRDesign,
        idx: np.ndarray | torch.Tensor | slice | None = None,
    ) -> torch.Tensor:
        """Standardized-scale conditional mean, shape (n_responses,).

        Args:
            design: Windowed impulse history, response timestamps, and the
                grouping codes the random effects index.
            idx: Rows of the design to evaluate; ``None`` takes all of them.
        """
        sl = slice(None) if idx is None else idx
        X_conv = self.convolved(design, idx)
        gf = (
            torch.as_tensor(design.gf_y[sl, 0], device=self.device)
            if len(design.gf_y) and design.gf_y.shape[1]
            else None
        )

        coef = self.parameter("coef").unsqueeze(0)
        if self.random_coefs and gf is not None:
            stored = self.parameter("coef_random")
            cr = stored - stored.mean(dim=0, keepdim=True)
            cr = torch.cat(
                [cr, torch.zeros(1, cr.shape[1], dtype=cr.dtype, device=cr.device)],
                dim=0,
            )
            coef = coef + cr[gf]
        out = (X_conv * coef).sum(dim=1)

        intercept = self.parameter("intercept")
        if self.random_intercept and gf is not None:
            stored = self.parameter("intercept_random")
            ir = stored - stored.mean()
            ir = torch.cat([ir, torch.zeros(1, dtype=ir.dtype, device=ir.device)])
            out = out + ir[gf]
        for extra_idx in range(len(self.intercept_random_extra)):
            group_idx = extra_idx + 1
            if design.gf_y.shape[1] <= group_idx:
                raise ValueError(
                    "design has no grouping column for extra random intercept "
                    f"{group_idx}"
                )
            code = torch.as_tensor(design.gf_y[sl, group_idx], device=self.device)
            stored = self.parameter(f"intercept_random_extra.{extra_idx}")
            centered = stored - stored.mean()
            centered = torch.cat(
                [
                    centered,
                    torch.zeros(1, dtype=centered.dtype, device=centered.device),
                ]
            )
            out = out + centered[code]
        return out + intercept

    def y_sd(self) -> torch.Tensor:
        """Residual standard deviation on the standardized response scale."""
        return softplus(self.parameter("y_sd_raw"))

    def log_lik(
        self,
        design: CDRDesign,
        idx: np.ndarray | torch.Tensor | slice | None = None,
    ) -> torch.Tensor:
        """Per-observation log-density on the original response scale.

        Args:
            design: Windowed impulse history plus the observed responses and
                the constants that put them back on their own scale.
            idx: Rows of the design to score; ``None`` takes all of them.
        """
        sl = slice(None) if idx is None else idx
        out = self.forward(design, idx=idx)
        loc = out * design.y_sd + design.y_mean
        scale = self.y_sd() * design.y_sd
        y = _float_tensor(design.y[sl], self.device)
        return torch.distributions.Normal(loc, scale).log_prob(y)

    def predict(self, design: CDRDesign) -> torch.Tensor:
        """Point prediction on the original response scale.

        Args:
            design: Windowed impulse history for every response to predict.
        """
        return self.forward(design) * design.y_sd + design.y_mean

    def objective_ll(
        self, design: CDRDesign, idx: np.ndarray | None = None
    ) -> torch.Tensor:
        """Standardized-scale per-observation log-likelihood (training objective).

        Args:
            design: Windowed impulse history plus the observed responses.
            idx: Rows of the design in this minibatch; ``None`` takes all.
        """
        sl = slice(None) if idx is None else idx
        out = self.forward(design, idx=idx)
        y = _float_tensor(design.y[sl], self.device)
        y_std = (y - design.y_mean) / design.y_sd
        return torch.distributions.Normal(out, self.y_sd()).log_prob(y_std)

    def regularization(self) -> torch.Tensor:
        """Penalties the model owns; this one's are all opt-in by the caller."""
        return torch.zeros((), dtype=torch.float32, device=self.device)

    def ranef_penalty(
        self, scale: float, *, exclude: set[str] | None = None
    ) -> torch.Tensor:
        """L2 penalty ``scale * 0.5 * sum(x^2)`` on centered random offsets.

        Matches the reference's l2_regularizer applied to the centered random
        intercept/coefficient matrices and the constrained, centered random
        IRF offsets.

        Args:
            scale: Multiplier on the summed penalty.
            exclude: Parameter names left out, so a term a variational
                posterior already penalizes through its KL is not counted twice.
        """
        exclude = set() if exclude is None else exclude
        total = torch.zeros((), dtype=torch.float32, device=self.device)
        if self.random_intercept and "intercept_random" not in exclude:
            stored = self.parameter("intercept_random")
            ir = stored - stored.mean()
            total = total + 0.5 * (ir * ir).sum()
        if self.random_coefs and "coef_random" not in exclude:
            stored = self.parameter("coef_random")
            cr = stored - stored.mean(dim=0, keepdim=True)
            total = total + 0.5 * (cr * cr).sum()
        for extra_idx in range(len(self.intercept_random_extra)):
            if f"intercept_random_extra.{extra_idx}" in exclude:
                continue
            stored = self.parameter(f"intercept_random_extra.{extra_idx}")
            centered = stored - stored.mean()
            total = total + 0.5 * (centered * centered).sum()
        for parameter_idx, term_idx in enumerate(self._kernel_parameter_terms):
            term = self.kernel_terms[term_idx]
            if not term.ran:
                continue
            if f"irf_raw_random.{parameter_idx}" in exclude:
                continue
            offsets = self._random_offsets(term_idx)
            total = total + 0.5 * (offsets * offsets).sum()
        return scale * total
