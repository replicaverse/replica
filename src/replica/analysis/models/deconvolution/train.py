"""Optimization and convergence utilities for CDR models."""

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import functools
from dataclasses import dataclass

import numpy as np
import scipy.stats
import torch

from replica.analysis.models.deconvolution.data import CDRDesign
from replica.analysis.models.deconvolution.model import CDRModel
from replica.analysis.models.deconvolution.variational import VariationalPosterior


def enable_deterministic_training() -> None:
    """Require deterministic PyTorch operations for seeded fits."""
    torch.use_deterministic_algorithms(True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _restores_determinism(fit):
    """Wrap *fit* so torch's determinism settings end as the caller had them."""

    @functools.wraps(fit)
    def wrapper(*args, **kwargs):
        enabled = torch.are_deterministic_algorithms_enabled()
        warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        cudnn = (torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic)
        try:
            return fit(*args, **kwargs)
        finally:
            torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
            torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = cudnn

    return wrapper


def reproducibility_metadata(model, seed: int | None) -> dict[str, np.ndarray]:
    """Describe the deterministic runtime that produced a fit.

    Args:
        model: The fitted model, read for the device its parameters sit on.
        seed: Seed the fit ran under; ``None`` records that the run was not
            made deterministic.

    Returns:
        Torch version, CUDA version, device type and name, and the
        determinism flag, each wrapped as a zero-dimensional array so they
        can go straight into a checkpoint archive.
    """
    device = model.device
    return {
        "deterministic": np.asarray(seed is not None),
        "torch_version": np.asarray(str(torch.__version__)),
        "cuda_version": np.asarray(str(torch.version.cuda or "")),
        "device_type": np.asarray(device.type),
        "device_name": np.asarray(
            torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
        ),
    }


REPRODUCIBILITY_FIELDS = frozenset(
    {"deterministic", "torch_version", "cuda_version", "device_type", "device_name"}
)


def validate_reproducibility_metadata(payload, source) -> dict[str, str | bool]:
    """Validate deterministic-runtime metadata from a saved fit.

    Args:
        payload: An opened ``.npz`` checkpoint, read through ``.files`` and
            item access.
        source: Label for the checkpoint, quoted back in every error message;
            callers pass the archive's path.

    Returns:
        The runtime fields as plain strings, with ``deterministic`` true.
        Raises when a field is missing, empty, or records a non-deterministic
        run.
    """
    missing = REPRODUCIBILITY_FIELDS - set(payload.files)
    if missing:
        raise ValueError(
            f"checkpoint is missing runtime metadata {sorted(missing)}: {source}"
        )
    if not bool(payload["deterministic"]):
        raise ValueError(f"checkpoint was not produced deterministically: {source}")
    metadata = {
        name: str(payload[name]) for name in REPRODUCIBILITY_FIELDS - {"deterministic"}
    }
    if not metadata["torch_version"] or not metadata["device_type"]:
        raise ValueError(f"checkpoint has invalid runtime metadata: {source}")
    return {"deterministic": True, **metadata}


class TFNadam:
    """tf.contrib.opt.NadamOptimizer semantics (Adam + Nesterov momentum).

    m_t = b1 m + (1-b1) g;  v_t = b2 v + (1-b2) g^2;
    lr_t = lr sqrt(1-b2^t)/(1-b1^t);
    theta -= lr_t (b1 m_t + (1-b1) g) / (sqrt(v_t) + eps).
    No momentum-decay scheduling (unlike torch.optim.NAdam).
    """

    def __init__(
        self,
        params,
        lr: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ):
        """Allocate the first- and second-moment buffers.

        Args:
            params: Tensors to optimize; those with no gradient are dropped.
            lr: Base step size, before the bias correction each step applies.
            beta1: Decay of the first moment, the momentum term.
            beta2: Decay of the second moment, the squared-gradient term.
            eps: Floor added to the second moment's square root in the
                denominator.
        """
        self.params = [p for p in params if p.requires_grad]
        self.lr, self.beta1, self.beta2, self.eps = lr, beta1, beta2, eps
        self.t = 0
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]

    @torch.no_grad()
    def step(self) -> None:
        """Advance one update from the gradients the last backward pass left."""
        self.t += 1
        lr_t = self.lr * (1 - self.beta2**self.t) ** 0.5 / (1 - self.beta1**self.t)
        for p, m, v in zip(self.params, self.m, self.v, strict=True):
            if p.grad is None:
                continue
            g = p.grad
            m.mul_(self.beta1).add_(g, alpha=1 - self.beta1)
            v.mul_(self.beta2).addcmul_(g, g, value=1 - self.beta2)
            m_bar = self.beta1 * m + (1 - self.beta1) * g
            p.addcdiv_(m_bar, v.sqrt() + self.eps, value=-lr_t)

    def zero_grad(self) -> None:
        """Drop every stored gradient, so the next backward pass starts clean."""
        for p in self.params:
            p.grad = None


class EMA:
    """shadow = decay*shadow + (1-decay)*value, no bias correction."""

    def __init__(self, params, decay: float = 0.999):
        """Start the shadow copies at the parameters' current values.

        Args:
            params: Tensors to shadow; the same objects are written back on
                :meth:`swap_in`.
            decay: Weight kept on the existing shadow at each update, so a
                larger value averages over more steps.
        """
        self.decay = decay
        self.params = list(params)
        self.shadow = [p.detach().clone() for p in self.params]
        self.backup: list[torch.Tensor] | None = None

    @torch.no_grad()
    def update(self) -> None:
        """Fold the parameters' current values into the shadow copies."""
        for s, p in zip(self.shadow, self.params, strict=True):
            s.mul_(self.decay).add_(p, alpha=1 - self.decay)

    @torch.no_grad()
    def swap_in(self) -> None:
        """Install the shadow values, keeping the live ones for :meth:`swap_out`."""
        self.backup = [p.detach().clone() for p in self.params]
        for p, s in zip(self.params, self.shadow, strict=True):
            p.copy_(s)

    @torch.no_grad()
    def swap_out(self) -> None:
        """Put the values :meth:`swap_in` set aside back on the parameters."""
        assert self.backup is not None
        for p, b in zip(self.params, self.backup, strict=True):
            p.copy_(b)
        self.backup = None


class ConvergenceDiagnostic:
    """Detect convergence from a rolling loss-slope test."""

    def __init__(
        self, n_iterates: int = 500, alpha: float = 0.5, two_tailed: bool = True
    ):
        """Size the rolling loss window and fix the test it applies.

        Args:
            n_iterates: Iterations the window holds, which is also the minimum
                number that has to elapse before convergence can be declared.
            alpha: Threshold the slope's p-value and the proportion of
                converged iterations both have to exceed.
            two_tailed: Test the slope two-tailed; false takes the left tail
                only, so a still-improving series stays unconverged.
        """
        self.n = n_iterates
        self.alpha = alpha
        self.two_tailed = two_tailed
        self.iterates = np.zeros(n_iterates)
        self.history = np.zeros(n_iterates)
        self.step = 0

    def _corr_p(self, x: np.ndarray, y: np.ndarray, df: int) -> tuple[float, float]:
        if len(x) < 2 or len(y) < 2:
            return 0.0, 0.0
        sx, sy = x.std(), y.std()
        if sx == 0 or sy == 0:
            return 0.0, 0.0
        r = float(np.corrcoef(x, y)[0, 1])
        if abs(r) >= 1.0:
            return r, 0.0
        t = r * np.sqrt(df / (1 - r * r))
        if self.two_tailed:
            p = 1 - (scipy.stats.t.cdf(abs(t), df) - scipy.stats.t.cdf(-abs(t), df))
        else:
            # A still-improving series has a negative slope and so a small
            # left-tail probability; a flat one is already past the threshold.
            p = scipy.stats.t.cdf(t, df)
        return r, (float(p) if np.isfinite(p) else 0.0)

    def update(self, mean_loss: float) -> bool:
        """Record one iteration's loss and report whether the fit has converged.

        Args:
            mean_loss: This iteration's mean minibatch loss, pushed onto the
                rolling window. A held-out criterion enters negated, so that
                falling values still mean improvement.

        Returns:
            True once the window has filled, the loss-versus-iteration slope is
            indistinguishable from flat at ``alpha``, and that has held for more
            than ``alpha`` of the last ``n_iterates`` iterations.
        """
        self.step += 1
        self.iterates[:-1] = self.iterates[1:]
        self.iterates[-1] = mean_loss

        start = max(0, self.n - self.step)
        y = self.iterates[start:]
        x = np.arange(len(y), dtype=float)
        df = self.n - 2
        _, p_slope = self._corr_p(x, y, df)
        # The reference computes this diagnostic without using the result;
        # calling it keeps the step accounting aligned.
        self._corr_p(y[1:], y[:-1], df)

        locally = self.step > self.n and p_slope > self.alpha
        self.history[:-1] = self.history[1:]
        self.history[-1] = float(locally)
        proportion = self.history.mean()

        return bool(
            self.step > self.n and p_slope > self.alpha and proportion > self.alpha
        )


def build_optimizer(name: str, params, learning_rate: float):
    """The optimizer of one reference variant.

    The parametric path uses the Nesterov-momentum Adam of its era; the
    neural one uses plain Adam at the framework's own defaults.

    Args:
        name: ``"nadam"`` for :class:`TFNadam`, ``"adam"`` for
            :class:`torch.optim.Adam`. Anything else raises.
        params: Tensors to update; each optimizer drops those without a
            gradient.
        learning_rate: Base step size, before either optimizer's own bias
            correction.
    """
    if name == "nadam":
        return TFNadam(params, lr=learning_rate)
    if name == "adam":
        return torch.optim.Adam(params, lr=learning_rate, eps=1e-8)
    raise ValueError(f"unknown optimizer: {name!r}")


def held_out_loglik(
    model: CDRModel,
    design: CDRDesign,
    ema: EMA,
    posterior: VariationalPosterior | None = None,
    *,
    batch_size: int = 4096,
) -> float:
    """Evaluate held-out log likelihood at the parameter moving average.

    Args:
        model: Model to score with; its parameters are restored on the way out.
        design: Held-out partition, scored on its own response scale.
        ema: Shadow parameters swapped in for the evaluation, so the score
            reflects the average rather than the latest step.
        posterior: Variational posterior whose means stand in for the sampled
            parameters. ``None`` scores the point parameters.
        batch_size: Responses per forward pass, which bounds peak memory. A
            value below one raises.

    Returns:
        Log likelihood summed over every response in *design*.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    ema.swap_in()
    estimate = None if posterior is None else posterior.means()
    try:
        with torch.no_grad(), model.substituted(estimate):
            total = sum(
                float(model.log_lik(design, idx=slice(start, start + batch_size)).sum())
                for start in range(0, len(design.y), batch_size)
            )
    finally:
        ema.swap_out()
    return total


def finite_or_zero(value: float) -> float:
    """A non-finite minibatch loss contributes zero to the epoch mean.

    Mirrors the reference, so one non-finite batch cannot turn the convergence
    diagnostic into NaN.

    Args:
        value: One minibatch's loss, returned unchanged when it is finite.
    """
    return value if np.isfinite(value) else 0.0


def finite_epoch_losses(
    values: list[float], iteration: int, first_nonfinite_update: str | None = None
) -> list[float]:
    """Apply the reference's zeroing rule, but reject a wholly failed epoch.

    Args:
        values: One epoch's minibatch losses, in the order they were taken.
        iteration: 1-based epoch index, quoted in the error.
        first_nonfinite_update: Description of the update that first went
            non-finite, appended to the error to locate the failure. ``None``
            leaves the error without that detail.

    Returns:
        The losses with every non-finite entry replaced by zero.

    Raises:
        FloatingPointError: When no loss in the epoch is finite.
    """
    if not any(np.isfinite(value) for value in values):
        detail = (
            f"; first non-finite update at {first_nonfinite_update}"
            if first_nonfinite_update is not None
            else ""
        )
        raise FloatingPointError(
            f"all {len(values)} minibatch losses were non-finite at "
            f"iteration {iteration}{detail}"
        )
    return [finite_or_zero(value) for value in values]


def _nonfinite_parameter_names(named_parameters, *, gradients: bool) -> tuple[str, ...]:
    """Names of tensors that first make an optimizer update non-finite."""
    bad = []
    for name, parameter in named_parameters:
        value = parameter.grad if gradients else parameter
        if value is not None and not bool(torch.isfinite(value).all()):
            bad.append(name)
    return tuple(bad)


def _optimizer_step_diagnostic(
    named_parameters,
    opt,
    clipped,
    max_global_norm: float | None,
    loss_value: float,
    iteration: int,
    minibatch: int,
) -> str | None:
    """Take one step and describe it only when it is already non-finite."""
    grad_norm = None
    if max_global_norm is not None:
        grad_norm = float(
            torch.nn.utils.clip_grad_norm_(clipped, max_global_norm).detach()
        )
    suspicious = not np.isfinite(loss_value) or (
        grad_norm is not None and not np.isfinite(grad_norm)
    )
    bad_gradients = (
        _nonfinite_parameter_names(named_parameters, gradients=True)
        if suspicious
        else ()
    )
    opt.step()
    if not suspicious:
        return None
    bad_parameters = _nonfinite_parameter_names(named_parameters, gradients=False)
    return (
        f"iteration {iteration}, minibatch {minibatch}: loss={loss_value!r}, "
        f"gradient_norm={grad_norm!r}, non-finite gradients={bad_gradients!r}, "
        f"non-finite parameters after step={bad_parameters!r}"
    )


def _backward_optimizer_step(
    named_parameters,
    opt,
    clipped,
    max_global_norm: float | None,
    loss: torch.Tensor,
    iteration: int,
    minibatch: int,
    *,
    defer_loss_readback: bool,
) -> tuple[float | torch.Tensor, str | None]:
    """Backpropagate once, optionally leaving the scalar loss on-device."""
    loss_snapshot = loss.detach()
    loss_value = None if defer_loss_readback else float(loss_snapshot)
    loss.backward()
    if defer_loss_readback:
        if max_global_norm is not None:
            torch.nn.utils.clip_grad_norm_(clipped, max_global_norm)
        opt.step()
        return loss_snapshot, None
    assert loss_value is not None
    diagnostic = _optimizer_step_diagnostic(
        named_parameters,
        opt,
        clipped,
        max_global_norm,
        loss_value,
        iteration,
        minibatch,
    )
    return loss_value, diagnostic


def _finish_deferred_loss_readback(
    losses: list[torch.Tensor],
    named_parameters,
    iteration: int,
) -> tuple[list[float], str | None]:
    """Synchronize deferred scalar losses once and describe any failure."""
    values = torch.stack(losses).detach().cpu().numpy().astype(float).tolist()
    if all(np.isfinite(value) for value in values):
        return values, None
    bad_parameters = _nonfinite_parameter_names(named_parameters, gradients=False)
    diagnostic = (
        f"iteration {iteration}: deferred loss readback found a non-finite "
        "minibatch; non-finite parameters after the "
        f"epoch={bad_parameters!r}"
    )
    return values, diagnostic


def _read_epoch_losses(
    losses: list[float | torch.Tensor],
    named_parameters,
    iteration: int,
    *,
    deferred: bool,
) -> tuple[list[float], str | None]:
    """Return host scalars for either eager or deferred collection."""
    if not deferred:
        return losses, None
    return _finish_deferred_loss_readback(losses, named_parameters, iteration)


@dataclass
class FitResult:
    """How a fit ended: iterations run, whether it converged, and its last loss."""

    n_iterations: int
    converged: bool
    final_mean_loss: float


def _trainable_named_parameters(
    model: CDRModel, posterior: VariationalPosterior | None
) -> tuple[list[tuple[str, torch.nn.Parameter]], set[str]]:
    """Posterior tensors plus point parameters in a hybrid model."""
    posterior_names = set() if posterior is None else set(posterior.names)
    named = (
        []
        if posterior is None
        else [
            (f"posterior.{name}", parameter)
            for name, parameter in posterior.named_parameters()
            if parameter.requires_grad
        ]
    )
    named.extend(
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and name not in posterior_names
    )
    return named, posterior_names


@_restores_determinism
def fit_cdr(
    model: CDRModel,
    design: CDRDesign,
    *,
    n_iter: int = 100000,
    learning_rate: float = 0.001,
    minibatch_size: int = 1024,
    ema_decay: float = 0.999,
    ema_updates_per_batch: int | None = None,
    ranef_regularizer_scale: float | None = None,
    convergence_n_iterates: int = 500,
    convergence_alpha: float = 0.5,
    check_convergence: bool = True,
    optimizer: str = "nadam",
    max_global_norm: float | None = None,
    dev_design: CDRDesign | None = None,
    eval_every: int = 1,
    two_tailed_convergence: bool = True,
    seed: int | None = None,
    log_every: int = 0,
    posterior: VariationalPosterior | None = None,
    include_kl: bool = True,
    defer_loss_readback: bool = False,
) -> tuple[FitResult, EMA]:
    """Fit a CDR model until convergence or the iteration limit.

    One iteration is a shuffled pass over the training responses. Parameters
    are updated in place; the returned average is what should be scored with.

    Args:
        model: Model to fit, updated in place.
        design: Training partition, the windowed impulse history and the
            responses to fit it to.
        n_iter: Ceiling on iterations, reached when convergence is not declared
            first.
        learning_rate: Base step size handed to the optimizer.
        minibatch_size: Responses per update. It also sets the factor each
            minibatch likelihood is scaled by, so the loss keeps the full
            training set's magnitude at any batch size.
        ema_decay: Weight kept on the shadow parameters at each average update,
            so a larger value averages over more steps.
        ema_updates_per_batch: Average updates applied after each minibatch.
            ``None`` applies one under a variational posterior and two without.
        ranef_regularizer_scale: Multiplier for the model's L2 penalty on
            centered random offsets. ``None`` or zero adds no penalty.
        convergence_n_iterates: Entries the convergence window holds, and the
            minimum number before convergence can be declared. An entry is one
            iteration, or one held-out evaluation when *dev_design* is given.
        convergence_alpha: Threshold on both the slope's p-value and the
            proportion of converged iterations.
        check_convergence: Consult the diagnostic. False runs every iteration up
            to *n_iter*.
        optimizer: Optimizer name for :func:`build_optimizer`.
        max_global_norm: Clip gradients to this global norm before each step;
            ``None`` leaves them unclipped.
        dev_design: Held-out partition. Given one, convergence tracks held-out
            log likelihood rather than training loss, and the average from the
            best evaluation is the one returned.
        eval_every: Iterations between held-out evaluations, used only alongside
            *dev_design*.
        two_tailed_convergence: Test the loss slope two-tailed; false takes the
            left tail only, so a still-improving series stays unconverged.
        seed: Seeds shuffling, torch, and the posterior sampler, and requires
            deterministic kernels for the fit; the caller's determinism
            settings are restored on return. ``None`` leaves the fit unseeded.
        log_every: Iterations between progress lines; zero prints none.
        posterior: Variational posterior, sampled once per minibatch and
            substituted for the parameters. ``None`` fits point estimates.
        include_kl: Add the posterior's KL divergence to the loss. Applies only
            alongside *posterior*.
        defer_loss_readback: Keep each minibatch loss on device and synchronize
            once per iteration, which trades the per-update non-finite
            diagnostic for fewer device syncs.

    Returns:
        The fit summary and the exponential moving average holding the
        parameter values to score with.
    """
    rng = np.random.default_rng(seed) if seed is not None else np.random
    if seed is not None:
        enable_deterministic_training()
        torch.manual_seed(seed)
    if ema_updates_per_batch is None:
        ema_updates_per_batch = 1 if posterior is not None else 2
    generator = None
    if posterior is not None and seed is not None:
        generator = torch.Generator(device=posterior.device).manual_seed(seed)

    n_train = len(design.y)
    index_device = model.device if model.device.type != "cpu" else None
    minibatch_scale = n_train / minibatch_size
    named_parameters, posterior_names = _trainable_named_parameters(model, posterior)
    parameters = [parameter for _, parameter in named_parameters]
    opt = build_optimizer(optimizer, parameters, learning_rate)
    clipped = parameters
    ema = EMA(parameters, decay=ema_decay)
    diag = ConvergenceDiagnostic(
        convergence_n_iterates, convergence_alpha, two_tailed_convergence
    )
    best_dev = -np.inf
    best_shadow: list[torch.Tensor] | None = None
    first_nonfinite_update: str | None = None

    converged = False
    iteration = 0
    mean_loss = float("nan")
    while not converged and iteration < n_iter:
        perm = rng.permutation(n_train)
        losses = []
        for j in range(0, n_train, minibatch_size):
            idx = perm[j : j + minibatch_size]
            if index_device is not None:
                idx = torch.as_tensor(idx, device=index_device)
            opt.zero_grad()
            draw = None if posterior is None else posterior.sample(generator)
            with model.substituted(draw):
                ll = model.objective_ll(design, idx=idx)
                loss = -ll.sum() * minibatch_scale
                if ranef_regularizer_scale:
                    loss = loss + model.ranef_penalty(
                        ranef_regularizer_scale, exclude=posterior_names
                    )
                loss = loss + model.regularization()
            if posterior is not None and include_kl:
                loss = loss + posterior.kl()
            loss_value, update_diagnostic = _backward_optimizer_step(
                named_parameters,
                opt,
                clipped,
                max_global_norm,
                loss,
                iteration + 1,
                j // minibatch_size + 1,
                defer_loss_readback=defer_loss_readback,
            )
            if update_diagnostic is not None and first_nonfinite_update is None:
                first_nonfinite_update = update_diagnostic
            for _ in range(ema_updates_per_batch):
                ema.update()
            losses.append(loss_value)
        iteration += 1
        losses, deferred_diagnostic = _read_epoch_losses(
            losses,
            named_parameters,
            iteration,
            deferred=defer_loss_readback,
        )
        first_nonfinite_update = first_nonfinite_update or deferred_diagnostic
        losses = finite_epoch_losses(losses, iteration, first_nonfinite_update)
        mean_loss = float(np.mean(losses))
        if dev_design is None:
            converged = check_convergence and diag.update(mean_loss)
        elif iteration % eval_every == 0:
            dev_ll = held_out_loglik(model, dev_design, ema, posterior)
            if dev_ll > best_dev:
                best_dev = dev_ll
                best_shadow = [s.clone() for s in ema.shadow]
            converged = check_convergence and diag.update(-dev_ll)
        if log_every and iteration % log_every == 0:
            print(f"iter {iteration}: mean loss {mean_loss:.4f}", flush=True)

    if best_shadow is not None:
        ema.shadow = best_shadow
    return FitResult(iteration, bool(converged), mean_loss), ema
