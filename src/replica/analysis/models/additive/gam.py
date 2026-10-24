"""Generalized additive models with mgcv."""

import os
import re as _re
import tempfile
from importlib.resources import as_file, files

import numpy as np

from replica.analysis.models.support.prediction import _FitPredictWithSigma
from replica.analysis.models.support.r import _importr, _r_context
from replica.utils.arrays import real_array_with_missing
from replica.utils.r import run_r_script


class RGamModel(_FitPredictWithSigma):
    """R's mgcv::gam() or mgcv::bam(), generalised additive model.

    Supports three modes of increasing flexibility:

    Auto-formula (default)::

        RGamModel(smooth_cols=[3, 4])
        # Builds: y ~ x0 + x1 + x2 + s(x3, bs='cr', k=-1) + s(x4, bs='cr', k=-1)

    Auto-formula with random effects (``random_smooths`` sets
    ``needs_grouping``)::

        RGamModel(smooth_cols=[0, 1], random_smooths=True)
        # Builds: y ~ s(x0, bs='cr', k=-1) + s(x1, bs='cr', k=-1)
        #         + s(x0, participant_id, bs='re') + ...
        #         + s(participant_id, bs='re')

    Custom R formula (full control; naming ``participant_id`` or ``item_id``
    sets ``needs_grouping``)::

        RGamModel(
            formula="y ~ s(surp) + s(wlen) + s(surp, participant_id, bs='re')",
            col_names={"x0": "surp", "x1": "wlen"},
        )

    Args:
        smooth_cols: X-column indices that get ``s(col, bs='cr', k=-1)``
            smooth terms. If None and no formula given, applies a smooth
            to the last fixed-effect column.
        formula: Raw R formula. When provided, ``smooth_cols`` is ignored.
            Column names must match the auto-generated ``x0``/``x1``/...
            or the names given in ``col_names``.
        baseline_formula: Raw R formula for fits whose design matrix lacks
            columns that ``formula`` names. ``evaluate()`` hands one model
            both the controls-only and the full design, distinguished only
            by column count, so a fixed ``formula`` naming the focal columns
            cannot fit the baseline; this one is chosen for it instead.
            Requires ``formula`` and ``col_names``.
        col_names: Mapping from auto-generated to custom column names
            (e.g. ``{"x0": "surp"}``).
        use_bam: Use ``mgcv::bam()``, mgcv's large-n fitter, instead of
            ``mgcv::gam()``.
        random_smooths: If True (with ``formula`` None), add by-subject
            random-effect smooths ``s(col, participant_id, bs='re')`` for
            each smooth column plus a random intercept. Implies
            ``needs_grouping=True``.
        n_threads: Threads for ``bam()`` (ignored for ``gam()``).
        smooth_all: Apply an ``s(...)`` smooth to every fixed-effect column,
            not only ``smooth_cols``.
        smooth_k: Basis dimension ``k`` for the smooths; ``< 0`` uses mgcv's
            default (``k=-1``).
        discrete: Fit ``bam()`` with covariate discretisation, which is faster.
            Requires ``use_bam=True`` and ``method`` ``"fREML"`` or ``"NCV"``,
            the methods mgcv discretises under; ``fit_loglik`` fits without it.
        prediction_discrete: Discretise prediction covariates with mgcv's
            grid approximation. Requires ``use_bam=True`` and ``discrete=True``.
            The grid depends on each prediction batch; ``shrink_unseen`` uses
            one batch per pattern of unseen groups. Defaults to False.
        method: mgcv smoothing-parameter selection method (default "REML").
        re_subject_intercept: Add a by-subject random intercept
            ``s(participant_id, bs='re')`` (implies ``needs_grouping=True``).
        re_subject_slope_cols: By-subject random slopes for these X columns;
            a list of indices, the string "all" (every fixed column), or None.
        re_item_intercept: Add a by-item random intercept.
        re_item_slope_cols: By-item random slopes; same forms as the subject side.
        re_predict: "exclude" (default) drops ``bs='re'`` smooths and every
            smooth involving ``participant_id`` or ``item_id``, including
            ``bs='fs'``, factor-``by`` smooths and tensor interactions, from
            held-out predictions. Grouping variables in parametric fixed
            effects keep their true values and require seen test levels.
            "conditional" keeps the smooths at the true
            grouping levels (test levels of groups used in the selected
            formula must occur in training rows retained by mgcv);
            "shrink_unseen" (``use_bam=True`` only) fits using training levels
            alone and drops a smooth for a test row if any grouping variable
            it uses is unseen among the training rows retained by mgcv.
            Effects of known groups remain, including when the other group is
            unseen. Grouping variables in parametric fixed effects still
            require seen test levels.
        sigma_mode: "sig2" (default) reports mgcv's estimated scale;
            "train_mse" reports sqrt(mean(train residuals^2)) from the
            fit's own training residuals.
        smooth_basis: basis string for auto-formula smooths ("cr" default);
            ``None`` emits bare ``s(col)`` (mgcv's thin-plate default), in
            which case ``smooth_k`` is not emitted either.
    """

    name = "gam"

    def configuration(self) -> dict:
        """Return declared options without fitted state."""
        return {
            "smooth_cols": self.smooth_cols,
            "formula": self.formula,
            "baseline_formula": self.baseline_formula,
            "col_names": self.col_names,
            "use_bam": self.use_bam,
            "random_smooths": self.random_smooths,
            "smooth_all": self.smooth_all,
            "smooth_k": self.smooth_k,
            "discrete": self.discrete,
            "prediction_discrete": self.prediction_discrete,
            "method": self.method,
            "n_threads": self.n_threads,
            "re_subject_intercept": self.re_subject_intercept,
            "re_subject_slope_cols": self.re_subject_slope_cols,
            "re_item_intercept": self.re_item_intercept,
            "re_item_slope_cols": self.re_item_slope_cols,
            "re_predict": self.re_predict,
            "sigma_mode": self.sigma_mode,
            "smooth_basis": self.smooth_basis,
        }

    def __init__(
        self,
        smooth_cols: list[int] | None = None,
        *,
        formula: str | None = None,
        baseline_formula: str | None = None,
        col_names: dict[str, str] | None = None,
        use_bam: bool = False,
        random_smooths: bool = False,
        smooth_all: bool = False,
        smooth_k: int = -1,
        discrete: bool = False,
        prediction_discrete: bool = False,
        method: str = "REML",
        n_threads: int = 1,
        # Explicit random-effects controls (independent of smooth_cols).
        # Each *_slope_cols can be a list of indices, the literal string
        # "all" (every fixed column), or None.
        re_subject_intercept: bool = False,
        re_subject_slope_cols: list[int] | str | None = None,
        re_item_intercept: bool = False,
        re_item_slope_cols: list[int] | str | None = None,
        re_predict: str = "exclude",
        sigma_mode: str = "sig2",
        smooth_basis: str | None = "cr",
    ):
        """Fix the formula, the fitter, and how held-out rows are predicted.

        ``needs_grouping`` is derived here: it turns on when any random-effect
        option is set, or when a supplied ``formula`` names a grouping column.
        """
        self.smooth_basis = smooth_basis
        if re_predict not in ("exclude", "conditional", "shrink_unseen"):
            raise ValueError(
                "re_predict must be 'exclude', 'conditional', or "
                f"'shrink_unseen', got {re_predict!r}"
            )
        if sigma_mode not in ("sig2", "train_mse"):
            raise ValueError(
                f"sigma_mode must be 'sig2' or 'train_mse', got {sigma_mode!r}"
            )
        if re_predict == "shrink_unseen" and not use_bam:
            raise ValueError("re_predict='shrink_unseen' requires use_bam=True")
        if discrete and not (use_bam and method in ("fREML", "NCV")):
            raise ValueError(
                "discrete=True requires use_bam=True with method 'fREML' or "
                f"'NCV', which mgcv discretises under; got use_bam={use_bam}, "
                f"method={method!r}"
            )
        if prediction_discrete and not (use_bam and discrete):
            raise ValueError(
                "prediction_discrete=True requires use_bam=True and discrete=True"
            )
        self.re_predict = re_predict
        self.sigma_mode = sigma_mode
        self.smooth_cols = smooth_cols
        self.formula = formula
        self.baseline_formula = baseline_formula
        self.col_names = col_names or {}
        if baseline_formula is not None and (formula is None or not self.col_names):
            raise ValueError(
                "baseline_formula requires formula and col_names: the choice "
                "between the two is made by which named columns the design "
                "matrix carries"
            )
        self.use_bam = use_bam
        self.random_smooths = random_smooths
        self.smooth_all = smooth_all
        self.smooth_k = smooth_k
        self.discrete = discrete
        self.prediction_discrete = prediction_discrete
        self.method = method
        self.n_threads = n_threads
        self.re_subject_intercept = re_subject_intercept
        self.re_subject_slope_cols = re_subject_slope_cols
        self.re_item_intercept = re_item_intercept
        self.re_item_slope_cols = re_item_slope_cols

        # ``random_smooths=True`` is a shortcut for "RE-slope on every
        # smoothed column + RE-intercept on participant" (no item
        # RE-effect). Translate to the explicit fields when none of the
        # explicit flags are set.
        if random_smooths and not (
            re_subject_intercept
            or re_subject_slope_cols
            or re_item_intercept
            or re_item_slope_cols
        ):
            self.re_subject_intercept = True
            self.re_subject_slope_cols = "smoothed"

        self.needs_grouping = (
            self.re_subject_intercept
            or self.re_subject_slope_cols is not None
            or self.re_item_intercept
            or self.re_item_slope_cols is not None
            or (
                formula is not None
                and _re.search(r"\b(participant_id|item_id)\b", formula) is not None
            )
        )

    def _smooth_set(self, n_fixed: int) -> set[int]:
        if n_fixed == 0:
            return set()
        if self.smooth_all:
            return set(range(n_fixed))
        if self.smooth_cols is not None:
            return set(self.smooth_cols)
        return {n_fixed - 1}

    def _build_formula(self, n_cols: int) -> str:
        if self.formula is not None:
            return self._select_formula(n_cols)

        n_fixed = n_cols - 2 if self.needs_grouping else n_cols

        # A random-effects-only model is a valid specification: the RE smooths
        # parameterise it on their own. This is the usual baseline for a GAMM
        # comparison, where the target model adds the focal smooths on top of
        # by-participant and by-item random effects, so a caller passing no
        # controls gets ``y ~ s(participant_id, bs='re') + s(item_id, bs='re')``
        # rather than an error. Random slopes need a fixed column to attach to,
        # so at least one random intercept is required.
        re_intercept = self.re_subject_intercept or self.re_item_intercept
        if n_fixed < 0 or (n_fixed == 0 and not re_intercept):
            raise ValueError(
                f"RGamModel: no fixed-effect columns (got {n_cols} total, "
                f"{'2 reserved for grouping' if self.needs_grouping else 'none reserved'})"
            )

        smooth_set = self._smooth_set(n_fixed)

        def _resolve_re_cols(spec):
            # Every branch clamps to the columns actually present. The baseline
            # fit of an ablation drops the target predictor, so a configured
            # index can point past the end; emitting a random slope for it puts
            # a column in the formula that is not in the data.
            if spec is None:
                return set()
            if spec == "all":
                return set(range(n_fixed))
            if spec == "smoothed":
                return {i for i in smooth_set if 0 <= i < n_fixed}
            return {i for i in spec if 0 <= i < n_fixed}

        subject_slope_set = _resolve_re_cols(self.re_subject_slope_cols)
        item_slope_set = _resolve_re_cols(self.re_item_slope_cols)

        k_arg = "k=-1" if self.smooth_k < 0 else f"k={self.smooth_k}"

        def _smooth_term(col: str) -> str:
            if self.smooth_basis is None:
                # mgcv defaults: thin-plate spline, default basis size.
                return f"s({col})"
            return f"s({col}, bs='{self.smooth_basis}', {k_arg})"

        terms = []
        for i in range(n_fixed):
            col = self.col_names.get(f"x{i}", f"x{i}")
            terms.append(_smooth_term(col) if i in smooth_set else col)

        for i in sorted(subject_slope_set):
            col = self.col_names.get(f"x{i}", f"x{i}")
            terms.append(f"s({col}, participant_id, bs='re')")
        if self.re_subject_intercept:
            terms.append("s(participant_id, bs='re')")
        for i in sorted(item_slope_set):
            col = self.col_names.get(f"x{i}", f"x{i}")
            terms.append(f"s({col}, item_id, bs='re')")
        if self.re_item_intercept:
            terms.append("s(item_id, bs='re')")

        return "y ~ " + " + ".join(terms)

    def _select_formula(self, n_cols: int) -> str:
        """The full formula when the design carries every column it names,
        else ``baseline_formula``. The baseline design is the full one with
        the focal columns removed from the end, so presence is decided by
        position through ``col_names``.
        """
        if self.baseline_formula is None:
            return self.formula
        n_fixed = n_cols - 2 if self.needs_grouping else n_cols
        available = {self.col_names.get(f"x{i}", f"x{i}") for i in range(n_fixed)}
        needed = {
            name
            for name in self.col_names.values()
            if _re.search(
                rf"(?<![A-Za-z0-9_.]){_re.escape(name)}(?![A-Za-z0-9_.])",
                self.formula,
            )
        }
        return self.formula if needed <= available else self.baseline_formula

    def _to_dataframe(self, ro, X, y=None):
        """Convert numpy arrays to R data.frame, handling grouping columns.

        Grouping columns are passed as integer vectors. Factor conversion
        happens in ``_assign_data()`` after assignment to R globalenv,
        avoiding rpy2 FactorVector issues with mgcv.
        """
        data = {}
        if y is not None:
            data["y"] = ro.FloatVector(y)

        if self.needs_grouping:
            n_fixed = X.shape[1] - 2
            for i in range(n_fixed):
                col = self.col_names.get(f"x{i}", f"x{i}")
                data[col] = ro.FloatVector(X[:, i])
            data["participant_id"] = ro.IntVector(X[:, -2].astype(int))
            data["item_id"] = ro.IntVector(X[:, -1].astype(int))
        else:
            for i in range(X.shape[1]):
                col = self.col_names.get(f"x{i}", f"x{i}")
                data[col] = ro.FloatVector(X[:, i])

        return ro.r["data.frame"](**data)

    def _fit_cmd(self, *, discrete: bool | None = None) -> str:
        # ``discrete`` defaults to ``self.discrete``. Callers fitting through
        # rpy2 in-process (fit_loglik) pass ``discrete=False``: bam(discrete=
        # TRUE) crashes through rpy2 (SEXP encoding bug), the approximation is
        # honored only via the fit_bam.R subprocess (fit_predict_with_sigma).
        if discrete is None:
            discrete = self.discrete
        fn = "mgcv::bam" if self.use_bam else "mgcv::gam"
        cmd = (
            f".gam_model <- {fn}(as.formula(.formula_str), "
            f'data=.train_data, method="{self.method}"'
        )
        if self.use_bam and self.n_threads > 1:
            cmd += f", nthreads={self.n_threads}"
        if self.use_bam and discrete:
            cmd += ", discrete=TRUE"
        cmd += ")"
        return cmd

    def _fit_predict_subprocess(
        self,
        y_train: np.ndarray,
        X_train: np.ndarray,
        X_test: np.ndarray,
        *,
        want_coefs: bool = False,
    ) -> tuple[np.ndarray, float, np.ndarray | None, np.ndarray | None]:
        """Fit ``mgcv::bam`` via an external Rscript subprocess.

        Writes train + test TSVs, invokes ``fit_bam.R``, reads
        predictions back. Avoids an rpy2 SEXP encoding bug that makes
        ``bam(..., discrete=TRUE)`` fail with "Error in 1:dk$nr[i] :
        NA/NaN argument" when invoked through rpy2, pure R fits the
        same data + formula cleanly.

        Returns ``(preds, sigma, beta, se)``. ``beta`` and ``se`` are
        full-length numpy arrays aligned to ``[intercept, x0, x1, ...]``
        when ``want_coefs=True`` and the model has parametric (linear)
        fixed effects, else ``(None, None)``. Smoothed columns read NaN
        at their position, since they have no single slope.
        """

        y_train = real_array_with_missing(y_train, name="y_train")
        X_train = real_array_with_missing(X_train, name="X_train")
        X_test = real_array_with_missing(X_test, name="X_test")
        n_fixed = (X_train.shape[1] - 2) if self.needs_grouping else X_train.shape[1]
        col_names = [self.col_names.get(f"x{i}", f"x{i}") for i in range(n_fixed)]
        header = ["y", *col_names]
        if self.needs_grouping:
            header.extend(["participant_id", "item_id"])

        def _write_tsv(X, y, path):
            arrs = [y] + [np.asarray(X[:, i], dtype=np.float64) for i in range(n_fixed)]
            if self.needs_grouping:
                arrs.append(np.asarray(X[:, -2], dtype=np.int64))
                arrs.append(np.asarray(X[:, -1], dtype=np.int64))
            np.savetxt(
                path,
                np.column_stack(arrs),
                delimiter="\t",
                header="\t".join(header),
                comments="",
                fmt="%.17g",
            )

        with tempfile.TemporaryDirectory(prefix="psycho_bam_") as td:
            train_p = os.path.join(td, "train.tsv")
            test_p = os.path.join(td, "test.tsv")
            out_p = os.path.join(td, "preds.tsv")
            coefs_p = os.path.join(td, "coefs.tsv") if want_coefs else None
            _write_tsv(X_train, y_train, train_p)
            _write_tsv(X_test, np.zeros(X_test.shape[0]), test_p)

            args = [
                "--train",
                train_p,
                "--test",
                test_p,
                "--formula",
                self._build_formula(X_train.shape[1]),
                "--out",
                out_p,
                "--method",
                self.method,
                "--discrete",
                "TRUE" if self.discrete else "FALSE",
                "--prediction_discrete",
                "TRUE" if self.prediction_discrete else "FALSE",
                "--nthreads",
                str(max(self.n_threads, 1)),
                "--re_predict",
                self.re_predict,
                "--sigma_mode",
                self.sigma_mode,
            ]
            if coefs_p is not None:
                args.extend(["--coefs", coefs_p])

            with as_file(files(__package__).joinpath("gam_prediction.R")) as helper:
                args.extend(["--prediction_helpers", str(helper)])
                res = run_r_script(files(__package__).joinpath("fit_bam.R"), args)
            sigma = None
            for line in res.stdout.splitlines():
                if line.startswith("SIGMA "):
                    sigma = float(line.split()[1])
                    break
            if sigma is None:
                raise RuntimeError(
                    f"no SIGMA line in fit_bam.R output: {res.stdout[-500:]}"
                )
            # atleast_1d: a one-row test fold collapses loadtxt to a 0-d
            # scalar, breaking the (n_test,) contract the rpy2 paths flatten.
            preds = np.atleast_1d(np.loadtxt(out_p, delimiter="\t", skiprows=1))

            beta = se = None
            if want_coefs and os.path.exists(coefs_p):
                # Build full-length beta/se aligned to [intercept, x0, ..., xN].
                # Smoothed terms are not in summary(m)$p.table, so their
                # position stays NaN.
                beta = np.full(1 + n_fixed, np.nan)
                se = np.full(1 + n_fixed, np.nan)
                with open(coefs_p) as fh:
                    next(fh, None)  # header
                    name_to_idx = {"(Intercept)": 0}
                    for j, nm in enumerate(col_names, start=1):
                        name_to_idx[nm] = j
                    for line in fh:
                        parts = line.rstrip("\n").split("\t")
                        if len(parts) < 3:
                            continue
                        term, est, sd = parts[0], parts[1], parts[2]
                        if term in name_to_idx:
                            i = name_to_idx[term]
                            beta[i] = float(est)
                            se[i] = float(sd)
            return preds, sigma, beta, se

    def _fit_predict_with_sigma_subprocess(
        self,
        y_train: np.ndarray,
        X_train: np.ndarray,
        X_test: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        preds, sigma, _, _ = self._fit_predict_subprocess(
            y_train, X_train, X_test, want_coefs=False
        )
        return preds, sigma

    @property
    def provides_stats(self) -> bool:
        """True only on the bam-subprocess path; the rpy2 path returns no
        parametric coefficient table. ``evaluate()`` reads this to choose
        ``fit_predict_with_stats`` over the sigma/fit_predict fallback.
        """
        return self.use_bam

    def fit_predict_with_stats(
        self,
        y_train: np.ndarray,
        X_train: np.ndarray,
        X_test: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Fit + predict + return parametric coefficient table.

        For bam-as-LMER (no spline smooths in fixed effects), every
        fixed column has a parametric coefficient that's reported here.
        A smoothed column has no single slope, so beta and se are NaN at its
        position; with every fixed column smoothed only the intercept is set.

        Args:
            y_train: Dependent variable on the training rows.
            X_train: Training design matrix; when ``needs_grouping`` holds, its
                last two columns are the participant and item codes.
            X_test: The same columns on the held-out rows.
        """
        if not self.use_bam:
            raise NotImplementedError(
                "fit_predict_with_stats is implemented only for the "
                "bam-subprocess path. Use fit_predict_with_sigma for the "
                "rpy2 path or extend it."
            )
        preds, sigma, beta, se = self._fit_predict_subprocess(
            y_train, X_train, X_test, want_coefs=True
        )
        if beta is None:
            n_fixed = (
                (X_train.shape[1] - 2) if self.needs_grouping else X_train.shape[1]
            )
            beta = np.full(1 + n_fixed, np.nan)
            se = np.full(1 + n_fixed, np.nan)
        return preds, beta, se, sigma

    def _assign_data(self, ro, name, r_df):
        """Assign a data.frame to R globalenv, factorizing grouping columns."""
        ro.globalenv[name] = r_df
        if self.needs_grouping:
            ro.r(
                f"{name}$participant_id <- factor({name}$participant_id); "
                f"{name}$item_id <- factor({name}$item_id)"
            )

    def fit_predict_with_sigma(
        self,
        y_train: np.ndarray,
        X_train: np.ndarray,
        X_test: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """Predict the test rows and report the fit's residual scale.

        Args:
            y_train: Dependent variable on the training rows.
            X_train: Training design matrix; when ``needs_grouping`` holds, its
                last two columns are the participant and item codes.
            X_test: The same columns on the held-out rows; ``re_predict``
                decides whether the random-effect smooths reach them.

        Returns:
            The test predictions and the scale selected by ``sigma_mode``.
        """
        if self.use_bam:
            # bam is fitted via a subprocess to ``fit_bam.R``. See
            # ``_fit_predict_with_sigma_subprocess`` for the rationale.
            return self._fit_predict_with_sigma_subprocess(y_train, X_train, X_test)
        y_train = real_array_with_missing(y_train, name="y_train")
        X_train = real_array_with_missing(X_train, name="X_train")
        X_test = real_array_with_missing(X_test, name="X_test")
        with _r_context() as ro:
            r_train = self._to_dataframe(ro, X_train, y=y_train)
            r_test = self._to_dataframe(ro, X_test)

            formula_str = self._build_formula(X_train.shape[1])

            _importr("mgcv")
            self._assign_data(ro, ".train_data", r_train)
            self._assign_data(ro, ".test_data", r_test)
            ro.globalenv[".formula_str"] = formula_str
            ro.r(self._fit_cmd())

            ro.r(files(__package__).joinpath("gam_prediction.R").read_text())

            if self.re_predict == "conditional":
                preds = np.array(
                    ro.r(
                        "local({"
                        "test <- replica_gam_prediction_data(.gam_model, "
                        ".train_data, .test_data, 'conditional'); "
                        "predict(.gam_model, newdata=test, type='response')})"
                    )
                ).flatten()
            else:
                guaranteed = "TRUE" if self.needs_grouping else "FALSE"
                preds = np.array(
                    ro.r(
                        "local({"
                        "test <- replica_gam_prediction_data(.gam_model, "
                        ".train_data, .test_data, 'exclude'); "
                        "predict(.gam_model, newdata=test, "
                        "exclude=replica_gam_excluded_smooths(.gam_model), "
                        f"newdata.guaranteed={guaranteed}, type='response')}})"
                    )
                ).flatten()
            if self.sigma_mode == "train_mse":
                sigma = float(
                    np.array(
                        ro.r("sqrt(mean(residuals(.gam_model, type='response')^2))")
                    )[0]
                )
            else:
                sig2 = float(np.array(ro.r(".gam_model$sig2"))[0])
                sigma = np.sqrt(sig2)

        return preds, sigma

    def fit_loglik(
        self,
        y: np.ndarray,
        X: np.ndarray,
    ) -> float:
        """Fit GAM on all data and return the model's log-likelihood.

        Returns R's ``logLik(model)`` directly.

        Args:
            y: Response variable, shape (n,).
            X: Predictor matrix, optionally with grouping columns appended
                (last 2 columns = participant_id, item_id when
                ``needs_grouping=True``).

        Returns:
            Log-likelihood as a float.

        Raises:
            ValueError: ``prediction_discrete=True`` requires the prediction path.
        """
        if self.prediction_discrete:
            raise ValueError(
                "prediction_discrete=True is unavailable for fit_loglik; "
                "use fit_predict_with_sigma for discrete BAM prediction"
            )
        y = real_array_with_missing(y, name="y")
        X = real_array_with_missing(X, name="X")
        with _r_context() as ro:
            r_data = self._to_dataframe(ro, X, y=y)
            formula_str = self._build_formula(X.shape[1])

            _importr("mgcv")
            self._assign_data(ro, ".train_data", r_data)
            ro.globalenv[".formula_str"] = formula_str
            # In-process rpy2 fit: discrete=FALSE so bam does not hit the
            # rpy2 SEXP crash (discrete is subprocess-only, see _fit_cmd).
            ro.r(self._fit_cmd(discrete=False))
            ll = float(np.array(ro.r("as.numeric(logLik(.gam_model))"))[0])

        return ll
