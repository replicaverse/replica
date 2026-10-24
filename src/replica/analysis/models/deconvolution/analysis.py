"""CDR analysis-model integration."""

from dataclasses import asdict, dataclass, field

import numpy as np

from replica.analysis.contracts.events import ResponseEvents
from replica.analysis.contracts.models import FoldScore, LikelihoodModel
from replica.analysis.models.deconvolution.data import (
    CDRDesign,
    build_design_from_events,
    transform_impulses,
)
from replica.analysis.models.deconvolution.model import (
    CDRModel,
    DiracTerm,
    KernelTerm,
    design_on,
)
from replica.analysis.models.deconvolution.train import (
    fit_cdr,
    reproducibility_metadata,
)
from replica.analysis.models.deconvolution.variational import (
    CDRPriors,
    VariationalPosterior,
)
from replica.analysis.models.support.diagnostics import _FitDiagnostics


def fold_group_codes(
    train_groups: np.ndarray, *held_out_groups: np.ndarray
) -> tuple[np.ndarray, ...]:
    """Encode training levels and map unseen held-out levels to one code.

    Further positional arrays are encoded against the training table, and a
    level absent from it takes the next code up, which the models carry as a
    fixed-zero unknown row rather than a trainable random effect.

    Args:
        train_groups: Grouping levels of the training rows, whose sorted unique
            values define the codes.
        held_out_groups: Further level arrays, each encoded against that same
            table; every level missing from it shares the one unknown code.

    Returns:
        One integer code array per array passed in, the training one first.
    """
    train = np.asarray(train_groups).reshape(-1)
    levels = np.unique(train)
    table = {level: index for index, level in enumerate(levels)}
    unknown = len(levels)
    train_codes = np.asarray([table[level] for level in train], dtype=np.int64)
    return (
        train_codes,
        *(
            np.asarray(
                [table.get(level, unknown) for level in np.asarray(groups).reshape(-1)],
                dtype=np.int64,
            )
            for groups in held_out_groups
        ),
    )


@dataclass(frozen=True)
class CDRFit:
    """CDR fitting hyperparameters.

    Args:
        n_iter: Ceiling on training iterations, reached when convergence is not
            declared first.
        learning_rate: Base step size of the optimizer.
        minibatch_size: Responses per update.
        ema_decay: Weight kept on the parameter moving average at each update,
            which is the state scored.
        ranef_regularizer_scale: Multiplier of the L2 penalty on the centred
            random offsets; ``None`` adds no penalty.
        convergence_n_iterates: Iterations the convergence window holds, and
            the minimum run before convergence can be declared.
        convergence_alpha: Threshold on the loss slope's p-value and on the
            share of converged iterations in the window.
        check_convergence: Stop on the convergence diagnostic; false runs
            all ``n_iter`` iterations.
        seed: Seeds initialisation, shuffling and posterior draws and requires
            deterministic kernels; ``None`` leaves the fit unseeded.
        inference: ``"mle"`` fits point estimates, ``"variational"`` a
            mean-field normal posterior.
        priors: Prior scales under ``"variational"``; ``None`` takes the
            :class:`CDRPriors` defaults.
        posterior_init_sd: Standard deviation of the jitter on the initial
            posterior; ``0.0`` starts it deterministically.
    """

    n_iter: int = 100000
    learning_rate: float = 0.001
    minibatch_size: int = 1024
    ema_decay: float = 0.999
    ranef_regularizer_scale: float | None = None
    convergence_n_iterates: int = 500
    convergence_alpha: float = 0.5
    check_convergence: bool = True
    seed: int | None = None
    inference: str = "mle"
    priors: CDRPriors | None = None
    posterior_init_sd: float = 0.0

    def __post_init__(self):
        if self.inference not in ("mle", "variational"):
            raise ValueError(f"unknown inference: {self.inference!r}")
        if self.posterior_init_sd < 0:
            raise ValueError("posterior_init_sd must be non-negative")


def write_fold_artifacts(
    directory: str,
    *,
    is_full: bool,
    events: ResponseEvents,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    loglik: np.ndarray,
    response_name: str | None = None,
) -> None:
    """Record held-out fit per observation, keyed on the event timeline.

    The papers' comparisons are over per-observation held-out fit and they
    pair observations across models, so each value is written with the
    timestamp and series that identify its response in the release.

    Args:
        directory: Where the TSV is written; created when it does not exist.
        is_full: Which side of the contrast produced these values, which names
            the file ``full`` or ``baseline``.
        events: Held-out responses, supplying each row's timestamp and the
            impulse index that resolves its series to a participant and text.
        y_true: Observed responses, differenced against *y_pred* for the
            squared-error column.
        y_pred: Point predictions on the response's own scale.
        loglik: Held-out log likelihood per response.
        response_name: Suffix distinguishing one response of a joint-response
            model; ``None`` writes the single-response filename.
    """
    from pathlib import Path

    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    index = events.impulses.index
    first: dict[int, tuple[str, str]] = {}
    for series, participant, stimulus in zip(
        index.series, index.participant_id, index.stimulus_id, strict=True
    ):
        first.setdefault(int(series), (str(participant), str(stimulus)))
    squared = (np.asarray(y_true, dtype=float) - y_pred) ** 2
    name = "full" if is_full else "baseline"
    response_suffix = "" if response_name is None else f"_{response_name}"
    with open(path / f"{name}{response_suffix}_test.tsv", "w") as handle:
        handle.write(
            "participant\tstimulus\tseries\ttime\tprediction\tsquared_error\tloglik\n"
        )
        for i, series in enumerate(events.series):
            participant, stimulus = first[int(series)]
            handle.write(
                f"{participant}\t{stimulus}\t{int(series)}\t"
                f"{float(events.time[i])!r}\t"
                f"{float(y_pred[i])!r}\t{float(squared[i])!r}\t"
                f"{float(loglik[i])!r}\n"
            )


def fold_designs(
    *,
    events: ResponseEvents,
    is_full: bool,
    y_train: np.ndarray,
    y_test: np.ndarray,
    train_groups: np.ndarray,
    test_groups: np.ndarray,
    train_mask: np.ndarray | None,
    test_mask: np.ndarray | None,
    impulse_ops: dict[str, str],
    history_length: int,
    t_delta_cutoff: float | None = None,
    cutoff_mode: str = "reference",
    ddof: int = 1,
    impulse_order: tuple[str, ...] | None = None,
    standardize_response: bool = True,
) -> tuple[list[str], ResponseEvents, ResponseEvents, CDRDesign, CDRDesign]:
    """Windowed designs for one fold, shared by both deconvolutional models.

    The impulse table is whole in every fold, so only the response side is
    split, and the held-out design is standardized by the training partition's
    own centre and scale.

    Args:
        events: Response timeline and the impulse stream it is windowed
            against, both covering the whole dataset.
        is_full: Take the full term set, controls plus focal impulses, rather
            than the baseline's controls alone.
        y_train: Dependent variable on the training rows.
        y_test: Dependent variable on the held-out rows.
        train_groups: Grouping level per training row, the participant codes
            that define the random-effect levels.
        test_groups: The same levels on the held-out rows, encoded against the
            training table.
        train_mask: Responses of *events* the fold fits on; ``None`` takes all
            of them.
        test_mask: Responses of *events* the fold scores on; ``None`` takes all
            of them.
        impulse_ops: Impulse name to the transform applied to its column, over
            the whole impulse table rather than per partition.
        history_length: Impulse slots kept per response.
        t_delta_cutoff: Lag bound on how far back a window reaches; ``None``
            lets it reach the start of the series.
        cutoff_mode: Released pointer behavior or a corrected sliding window.
        ddof: Delta degrees of freedom of the standard deviation the ``z`` and
            ``s`` transforms divide by.
        impulse_order: Names to lead the impulse columns, which pins the column
            order when the same model is reached through different ablations.
            Names it omits follow in the event channel's own order.
        standardize_response: Centre and scale the response by the training
            partition. False leaves it on its own scale.

    Returns:
        The impulse names, the training and held-out response subsets, and the
        design built from each.
    """
    names = events.impulse_names(is_full=is_full)
    if impulse_order is not None:
        available = set(names)
        names = [name for name in impulse_order if name in available]
        names += [
            name for name in events.impulse_names(is_full=is_full) if name not in names
        ]
    index = events.impulses.index
    common = {
        "impulse_time": index.time,
        "impulse_series": index.series,
        "impulse_values": transform_impulses(
            events.impulses.matrix(names), names, impulse_ops, ddof
        ),
        "impulse_names": names,
        "history_length": history_length,
        "t_delta_cutoff": t_delta_cutoff,
        "cutoff_mode": cutoff_mode,
    }
    train_events = events if train_mask is None else events.subset(train_mask)
    test_events = events if test_mask is None else events.subset(test_mask)
    train_groups, test_groups = fold_group_codes(train_groups, test_groups)
    response_scale = {} if standardize_response else {"y_mean": 0.0, "y_sd": 1.0}
    train_design = build_design_from_events(
        response_time=train_events.time,
        response_series=train_events.series,
        response_y=y_train,
        response_group=train_groups,
        **response_scale,
        **common,
    )
    test_design = build_design_from_events(
        response_time=test_events.time,
        response_series=test_events.series,
        response_y=y_test,
        response_group=test_groups,
        y_mean=train_design.y_mean,
        y_sd=train_design.y_sd,
        **common,
    )
    return names, train_events, test_events, train_design, test_design


@dataclass
class CDRAnalysis(_FitDiagnostics, LikelihoodModel):
    """Continuous-time deconvolutional regression over an event channel.

    Args:
        kernel_family: Impulse response family of every convolved term, a key
            of ``model.FAMILIES`` such as ``"ShiftedGammaShapeGT1"``.
        param_init: Initial kernel parameters by name, shared by every
            convolved term; a parameter left out takes the family default.
        dirac_impulses: Impulses entering unconvolved, at the response's own
            timestamp.
        impulse_ops: Impulse name to the transform applied to its whole
            column, as :func:`impulse_op` names them.
        history_length: Impulse slots kept per response.
        t_delta_cutoff: Lag bound on how far back a window reaches; ``None``
            lets it reach the start of the series.
        random_intercept: Give each participant an intercept offset.
        random_coefs: Give each participant coefficient offsets.
        random_irf: Give each participant kernel-parameter offsets.
        normalize_truncated_kernels: Renormalize the truncated normal and
            shifted-gamma kernels to unit mass over the positive lags.
        standardize_response: Fit the response centred and scaled by the
            training partition; false fits it on its own scale.
        device: Torch device the fit runs on.
        artifact_dir: Directory for ``{side}_test.tsv`` and
            ``{side}_checkpoint.npz``, where ``side`` is ``baseline`` or
            ``full``; each fold overwrites the previous fold's files.
            ``None`` writes nothing.
        fit: Optimizer, convergence and inference settings.
    """

    kernel_family: str = "ShiftedGammaShapeGT1"
    param_init: dict[str, float] = field(default_factory=dict)
    dirac_impulses: tuple[str, ...] = ()
    impulse_ops: dict[str, str] = field(default_factory=dict)
    history_length: int = 128
    t_delta_cutoff: float | None = None
    random_intercept: bool = False
    random_coefs: bool = False
    random_irf: bool = False
    normalize_truncated_kernels: bool = True
    standardize_response: bool = True
    device: str = "cpu"
    artifact_dir: str | None = None
    fit: CDRFit = field(default_factory=CDRFit)

    name = "cdr"
    owns_loglik = True
    needs_events = True
    needs_grouping = True

    def configuration(self) -> dict:
        """Return declared options without fitted state."""
        return asdict(self)

    def _terms(
        self, impulse_names: list[str]
    ) -> tuple[list[KernelTerm], list[DiracTerm]]:
        kernels = [
            KernelTerm(name, self.kernel_family, dict(self.param_init), self.random_irf)
            for name in impulse_names
            if name not in self.dirac_impulses
        ]
        diracs = [DiracTerm(n) for n in impulse_names if n in self.dirac_impulses]
        return kernels, diracs

    @staticmethod
    def _groups(X: np.ndarray) -> np.ndarray:
        """Participant codes, which ``needs_grouping`` appends to the design."""
        return X[:, -2].astype(np.int64)

    def fit_score(
        self,
        y_train: np.ndarray,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        *,
        is_full: bool,
        events: ResponseEvents,
        train_mask: np.ndarray | None = None,
        test_mask: np.ndarray | None = None,
    ) -> FoldScore:
        """Fit on the training responses and score the held-out ones.

        The design matrices carry the impulses only as their trailing grouping
        codes: the predictors themselves come from *events*, which is the
        stream the convolution runs over.

        Args:
            y_train: Dependent variable on the training rows.
            X_train: Training design matrix, read only for the participant
                codes in its second-to-last column.
            X_test: The same columns on the held-out rows.
            y_test: Held-out dependent variable, scored under the fitted model.
            is_full: Fit the full term set rather than the baseline's controls
                alone, which also names the artifacts this fold writes.
            events: Response timeline and impulse stream, supplied by
                ``evaluate()`` because ``needs_events`` is set.
            train_mask: Responses of *events* this fold fits on; ``None`` takes
                all of them.
            test_mask: Responses of *events* this fold scores on; ``None``
                takes all of them.

        Returns:
            Summed held-out log likelihood, the point predictions, and the
            per-observation likelihood the paired significance tests need.
        """
        self._fit_diagnostics = None
        names, train_events, test_events, train_design, test_design = fold_designs(
            events=events,
            is_full=is_full,
            y_train=y_train,
            y_test=y_test,
            train_groups=self._groups(X_train),
            test_groups=self._groups(X_test),
            train_mask=train_mask,
            test_mask=test_mask,
            impulse_ops=self.impulse_ops,
            history_length=self.history_length,
            t_delta_cutoff=self.t_delta_cutoff,
            standardize_response=self.standardize_response,
        )
        # Group codes were built from the training partition.  Held-out-only
        # levels use the next code, which CDRModel supplies as a fixed-zero
        # unknown row rather than a trainable random effect.
        n_levels = int(train_design.gf_y.max()) + 1

        kernels, diracs = self._terms(names)
        model = CDRModel(
            kernels,
            diracs,
            names,
            rangf_n_levels=n_levels,
            random_intercept=self.random_intercept,
            random_coefs=self.random_coefs,
            intercept_init=(
                0.0 if self.standardize_response else float(y_train.mean())
            ),
            y_sd_init=(
                1.0 if self.standardize_response else float(y_train.std(ddof=1))
            ),
            normalize_truncated_kernels=self.normalize_truncated_kernels,
            seed=self.fit.seed,
        )
        model = model.to(self.device)
        train_design = design_on(train_design, self.device)
        test_design = design_on(test_design, self.device)
        posterior = (
            VariationalPosterior(
                model,
                self.fit.priors,
                init_sd=self.fit.posterior_init_sd,
                seed=self.fit.seed,
            ).to(self.device)
            if self.fit.inference == "variational"
            else None
        )
        result, ema = fit_cdr(
            model,
            train_design,
            n_iter=self.fit.n_iter,
            learning_rate=self.fit.learning_rate,
            minibatch_size=self.fit.minibatch_size,
            ema_decay=self.fit.ema_decay,
            ranef_regularizer_scale=self.fit.ranef_regularizer_scale,
            convergence_n_iterates=self.fit.convergence_n_iterates,
            convergence_alpha=self.fit.convergence_alpha,
            check_convergence=self.fit.check_convergence,
            seed=self.fit.seed,
            posterior=posterior,
        )

        import torch

        ema.swap_in()
        estimate = None if posterior is None else posterior.means()
        with torch.no_grad(), model.substituted(estimate):
            per_obs = model.log_lik(test_design).cpu().numpy().astype(np.float64)
            y_pred = model.predict(test_design).cpu().numpy().astype(np.float64)
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

            state = {
                f"estimate_{name.replace('.', '__')}": value.detach().cpu().numpy()
                for name, value in (
                    posterior.named_parameters()
                    if posterior is not None
                    else model.named_parameters()
                )
            }
            state.update(
                n_iterations=np.asarray(result.n_iterations),
                converged=np.asarray(result.converged),
                final_mean_loss=np.asarray(result.final_mean_loss),
                y_mean=np.asarray(train_design.y_mean),
                y_sd=np.asarray(train_design.y_sd),
                seed=np.asarray(-1 if self.fit.seed is None else self.fit.seed),
                inference=np.asarray(self.fit.inference),
                standardize_response=np.asarray(self.standardize_response),
            )
            state.update(reproducibility_metadata(model, self.fit.seed))
            side = "full" if is_full else "baseline"
            np.savez_compressed(
                Path(self.artifact_dir) / f"{side}_checkpoint.npz", **state
            )
        ema.swap_out()
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
