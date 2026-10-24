"""CDR impulse-response kernels and parameter constraints."""

import torch

# 4x float32 machine epsilon, the reference's global kernel/constraint guard.
EPSILON = 4.0 * torch.finfo(torch.float32).eps


def softplus(x: torch.Tensor) -> torch.Tensor:
    """Map an unbounded tensor onto the positive reals, elementwise.

    Args:
        x: Unconstrained values, as the optimizer stores them.
    """
    return torch.nn.functional.softplus(x)


def softplus_inverse(x: torch.Tensor) -> torch.Tensor:
    """log(expm1(x)), stabilized like the reference implementation.

    Args:
        x: Positive values to send back to the unconstrained space.
    """
    return x + torch.log(-torch.expm1(-x))


def constrain_lower(raw: torch.Tensor, lb: float) -> torch.Tensor:
    """theta = lb + eps + softplus(raw).

    Args:
        raw: Unconstrained parameter, as the optimizer stores it.
        lb: Lower bound the family imposes, which the result stays above.
    """
    return lb + EPSILON + softplus(raw)


def constrain_lower_inverse(theta: float, lb: float) -> torch.Tensor:
    """The raw value :func:`constrain_lower` maps to ``theta``.

    Args:
        theta: Target value in the constrained space, strictly above ``lb``.
        lb: The same lower bound the forward transform uses.
    """
    return softplus_inverse(torch.tensor(theta - lb - EPSILON, dtype=torch.float32))


def constrain_upper(raw: torch.Tensor, ub: float) -> torch.Tensor:
    """theta = ub - eps - softplus(raw).

    Args:
        raw: Unconstrained parameter, as the optimizer stores it.
        ub: Upper bound the family imposes, which the result stays below.
    """
    return ub - EPSILON - softplus(raw)


def constrain_upper_inverse(theta: float, ub: float) -> torch.Tensor:
    """The raw value :func:`constrain_upper` maps to ``theta``.

    Args:
        theta: Target value in the constrained space, strictly below ``ub``.
        ub: The same upper bound the forward transform uses.
    """
    return softplus_inverse(torch.tensor(-(theta - ub + EPSILON), dtype=torch.float32))


def _normal_pdf(x: torch.Tensor, mu: torch.Tensor, sd: torch.Tensor) -> torch.Tensor:
    z = (x - mu) / sd
    return torch.exp(-0.5 * z * z) / (sd * (2.0 * torch.pi) ** 0.5)


def _normal_cdf(x: torch.Tensor, mu: torch.Tensor, sd: torch.Tensor) -> torch.Tensor:
    return torch.special.ndtr((x - mu) / sd)


def _gamma_pdf(
    x: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor
) -> torch.Tensor:
    return torch.exp(
        alpha * torch.log(beta)
        + (alpha - 1.0) * torch.log(x)
        - beta * x
        - torch.lgamma(alpha)
    )


def _gamma_cdf(
    x: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor
) -> torch.Tensor:
    return torch.special.gammainc(alpha, beta * x)


def exp_irf(x: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """f(x) = beta * exp(-beta x); Exp and ExpRateGT1 share this form.

    Args:
        x: Lag between the response and the impulse, in the event streams' own
            time unit.
        beta: Decay rate, positive; ``ExpRateGT1`` bounds it above one instead.
    """
    return beta * torch.exp(-beta * x)


def shifted_gamma_kgt1_irf(
    x: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    delta: torch.Tensor,
    *,
    normalize: bool = True,
) -> torch.Tensor:
    """Gamma pdf shifted left by delta (< 0), renormalized to unit mass on x > 0.

    f(x) = Gamma(alpha, beta).pdf(x - delta + eps)
           / (1 - Gamma(alpha, beta).cdf(-delta + eps) + eps)

    The normalizer's alpha is detached, so no gradient flows through the
    CDF's shape argument while beta/delta keep theirs, matching the
    reference's training behavior.

    Args:
        x: Lag between the response and the impulse.
        alpha: Gamma shape, bounded above one so the density starts at zero.
        beta: Gamma rate, positive.
        delta: Leftward shift, bounded at or below zero, which moves the peak
            towards lag zero.
        normalize: Divide by the mass the shift pushes below zero. False gives
            the bare shifted density.
    """
    num = _gamma_pdf(x - delta + EPSILON, alpha, beta)
    if not normalize:
        return num
    den = 1.0 - _gamma_cdf(-delta + EPSILON, alpha.detach(), beta) + EPSILON
    return num / den


def normal_irf(
    x: torch.Tensor, mu: torch.Tensor, sd: torch.Tensor, *, normalize: bool = True
) -> torch.Tensor:
    """Truncated-normal density on x > 0.

    The reference names the second parameter ``sigma2`` but uses it directly
    as the scale (SD); this function takes that scale as ``sd``.

    Args:
        x: Lag between the response and the impulse.
        mu: Location of the peak, unbounded.
        sd: Scale, positive; the reference's ``sigma2`` slot.
        normalize: Divide by the mass above zero, giving unit mass on the
            positive lags. False gives the untruncated density.
    """
    density = _normal_pdf(x, mu, sd)
    if not normalize:
        return density
    return density / (1.0 - _normal_cdf(torch.zeros_like(x), mu, sd) + EPSILON)


def double_gamma_irf(
    x: torch.Tensor,
    alpha_main: torch.Tensor,
    alpha_undershoot: torch.Tensor,
    beta_main: torch.Tensor,
    beta_undershoot: torch.Tensor,
    c: torch.Tensor,
) -> torch.Tensor:
    """Normalized double-gamma hemodynamic response function.

    The five-parameter form behind ``HRFDoubleGamma1``--``HRFDoubleGamma5``;
    the lower-dimensional families tie or fix its arguments. The epsilon inside
    the PDF arguments follows the TensorFlow reference and keeps the gamma
    density finite at lag zero.

    Args:
        x: Lag between the response and the impulse.
        alpha_main: Shape of the peak gamma, above one.
        alpha_undershoot: Shape of the gamma subtracted after the peak.
        beta_main: Rate of the peak gamma, positive.
        beta_undershoot: Rate of the subtracted gamma, positive.
        c: Weight of the undershoot; ``1 - c`` is the normalizer, the
            numerator's mass with no finite support bound (as in the Cognition
            configurations).
    """
    alpha_main = torch.as_tensor(alpha_main, dtype=x.dtype, device=x.device)
    alpha_undershoot = torch.as_tensor(alpha_undershoot, dtype=x.dtype, device=x.device)
    beta_main = torch.as_tensor(beta_main, dtype=x.dtype, device=x.device)
    beta_undershoot = torch.as_tensor(beta_undershoot, dtype=x.dtype, device=x.device)
    c = torch.as_tensor(c, dtype=x.dtype, device=x.device)
    x = x + EPSILON
    numerator = _gamma_pdf(x, alpha_main, beta_main) - c * _gamma_pdf(
        x, alpha_undershoot, beta_undershoot
    )
    return numerator / (1.0 - c)


def lcg_irf(
    x: torch.Tensor,
    locs: torch.Tensor,
    values: torch.Tensor,
    scales: torch.Tensor,
) -> torch.Tensor:
    """Linear combination of Gaussians (family ``G``), unit-normalized on x > 0.

    f(x) = sum_k y_k N(x; x_k, s_k).pdf / sum_k y_k (1 - N(.; x_k, s_k).cdf(0))

    Args:
        x: Lag between the response and the impulse.
        locs: Per-basis Gaussian means, the basis running along the last axis.
        values: Per-basis weights, unbounded, so a basis may subtract.
        scales: Per-basis standard deviations, positive.
    """
    xk = x.unsqueeze(-1)
    num = (values * _normal_pdf(xk, locs, scales)).sum(dim=-1)
    zero = torch.zeros((), dtype=x.dtype, device=x.device)
    den = (values * (1.0 - _normal_cdf(zero, locs, scales))).sum(dim=-1)
    return num / den
