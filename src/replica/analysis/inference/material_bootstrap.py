"""Paired OLS comparisons with material-separated CV and full-refit resampling.

The bootstrap samples whole materials, refitting both regressions in every
fold in every draw. Copies of a material always retain the same fold. It
estimates sampling uncertainty conditional on the recorded predictor tables,
analysis population, and fold allocation; it does not regenerate LM samples.

The centered bootstrap test is approximate.
The decision p-value is the maximum of this test and the exact-enumerated
fold sign-flip diagnostic. This conservative intersection cannot improve on
either test's p-value and retains the finite fold-sign-flip resolution. The
fold diagnostic alone does not account for overlapping training sets.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from ..cv.materials import material_folds
from ..models.fixed_effects._least_squares import least_squares, residual_variance
from .significance import paired_permutation_pvalue

PROTOCOL = "material-cv-full-refit-bootstrap-v1"


def bootstrap_summary(effect: float, draws: np.ndarray) -> dict:
    """Centered bootstrap tail p-value and basic 95% interval of one effect.

    A non-finite draw is a failed refit: it counts as extreme and widens the
    interval on both sides.

    Args:
        effect: The effect on the observed sample.
        draws: The effect refitted on each bootstrap sample, one per draw.
    """
    errors = draws - effect
    failed = ~np.isfinite(errors)
    extreme = (
        failed
        | (np.abs(errors) >= abs(effect))
        | np.isclose(np.abs(errors), abs(effect), rtol=1e-12, atol=1e-15)
    )
    return {
        "p_bootstrap": (int(extreme.sum()) + 1) / (len(draws) + 1),
        "ci_low": float(
            effect
            - np.quantile(
                np.where(failed, np.inf, errors), 0.975, method="inverted_cdf"
            )
        ),
        "ci_high": float(
            effect
            - np.quantile(
                np.where(failed, -np.inf, errors), 0.025, method="inverted_cdf"
            )
        ),
        "bootstrap_failures": int(failed.sum()),
    }


@dataclass(frozen=True)
class Contrast:
    """A paired comparison on one common set of complete observations.

    Args:
        name: Stable name for the comparison.
        baseline: Control-model columns, excluding the automatic intercept.
        target: Full target-model columns, excluding the intercept.
    """

    name: str
    baseline: tuple[str, ...]
    target: tuple[str, ...]


class GroupedOLS:
    """OLS and training-residual Gaussian scores from per-material QR factors.

    Multiplicity weights give exactly the fits obtained by copying every row
    of a resampled material. QR compression preserves residual norms without
    forming normal equations or copying the full tables for every draw.
    An active test fold with an unestimable or perfect training fit invalidates
    its draw.

    Args:
        y: Finite response vector.
        predictors: Finite predictor columns aligned with the response.
        groups: Material ID per observation, the unit the bootstrap resamples.
        n_folds: Maximum ItemSplitter folds.
        sigma_correction: Training-variance denominator, ``mle`` or ``df``.
    """

    def __init__(
        self,
        y: Sequence[float],
        predictors: Mapping[str, Sequence[float]],
        groups: Sequence[str],
        *,
        n_folds: int = 10,
        sigma_correction: str = "mle",
    ):
        """Validate observations and compress each material's augmented design."""
        if any(
            np.ma.isMaskedArray(values) or np.iscomplexobj(values)
            for values in (y, *predictors.values())
        ):
            raise ValueError("OLS requires unmasked real responses and predictors")
        self.y = np.asarray(y, dtype=float)
        self.names = tuple(predictors)
        if self.y.ndim != 1 or len(groups) != len(self.y):
            raise ValueError("Response, predictors and materials must align")
        columns = [np.asarray(values, dtype=float) for values in predictors.values()]
        if any(column.shape != self.y.shape for column in columns):
            raise ValueError("Each predictor must be a vector matching the response")
        self.X = np.column_stack([np.ones(len(self.y)), *columns])
        if not np.isfinite(self.y).all() or not np.isfinite(self.X).all():
            raise ValueError("Select the common complete rows before fitting")
        if sigma_correction not in ("mle", "df"):
            raise ValueError("Unknown training sigma convention")
        self.sigma_correction = sigma_correction
        self.row_folds = material_folds(groups, n_folds)
        self.groups, codes = np.unique(
            np.asarray(groups, dtype=str), return_inverse=True
        )
        self.group_codes = codes
        self.n_groups = len(self.groups)
        self.n_folds = int(self.row_folds.max()) + 1
        self.counts = np.bincount(codes).astype(float)
        order = np.argsort(codes, kind="stable")
        cuts = np.r_[0, np.cumsum(self.counts).astype(int)]
        self.group_folds = self.row_folds[order[cuts[:-1]]]
        # The intercept absorbs this origin; QR never sees a large common offset.
        augmented = np.column_stack([self.X, self.y - self.y[0]])[order]
        slices = [slice(a, b) for a, b in zip(cuts[:-1], cuts[1:], strict=True)]
        factors = [np.linalg.qr(augmented[s], mode="r") for s in slices]
        self._qr = np.vstack(factors)
        self._qr_groups = np.repeat(
            np.arange(self.n_groups), [len(factor) for factor in factors]
        )
        self._qr_folds = self.group_folds[self._qr_groups]

    def scores(
        self, columns: Sequence[str], weights: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Refit each weighted CV sample and return total and fold likelihoods.

        Args:
            columns: Model terms, excluding intercept.
            weights: Material multiplicities, shape ``(draws, materials)``.
        """
        return self._score(columns, *self._weighted_data(weights))

    def _weighted_data(
        self, weights: np.ndarray
    ) -> tuple[list[np.ndarray], np.ndarray]:
        """Reduce weighted material factors to one QR factor per fold and draw."""
        if np.ma.isMaskedArray(weights) or np.iscomplexobj(weights):
            raise ValueError("Material multiplicities must be unmasked real values")
        w = np.asarray(weights, dtype=float)
        if (
            w.ndim != 2
            or w.shape[1] != self.n_groups
            or np.any(w < 0)
            or not np.isfinite(w).all()
        ):
            raise ValueError("Invalid material multiplicities")
        folds = []
        for fold in range(self.n_folds):
            selected = self._qr_folds == fold
            factors = self._qr[selected]
            groups = self._qr_groups[selected]
            reduced = np.empty((len(w), 0, factors.shape[1]))
            # Bound temporary memory independently of the corpus's row count.
            for start in range(0, len(factors), 1024):
                block = slice(start, start + 1024)
                weighted = np.sqrt(w[:, groups[block]])[..., None] * factors[block]
                reduced = np.asarray(
                    np.linalg.qr(np.concatenate([reduced, weighted], axis=1), mode="r")
                )
            folds.append(reduced)
        return folds, w * self.counts

    def _score(
        self, columns: Sequence[str], weighted: list[np.ndarray], counts: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fit selected columns and evaluate each held-out material fold."""
        if len(set(columns)) != len(columns):
            raise ValueError("Duplicate regression term")
        cols = [0, *(self.names.index(c) + 1 for c in columns)]
        size = len(counts)
        fold_ll = np.zeros((size, self.n_folds))
        valid = np.ones(size, dtype=bool)
        for fold in range(self.n_folds):
            test = weighted[fold]
            train = np.concatenate(
                [rows for index, rows in enumerate(weighted) if index != fold], axis=1
            )
            group_test = self.group_folds == fold
            n_train = counts[:, ~group_test].sum(axis=1)
            n_test = counts[:, group_test].sum(axis=1)
            design = train[:, :, cols]
            response = train[:, :, -1]
            beta, rank, _ = least_squares(design, response, n_observations=n_train)
            rss = np.sum(
                (response - np.einsum("bij,bj->bi", design, beta)) ** 2, axis=1
            )
            variance = residual_variance(rss, n_train, rank, self.sigma_correction)
            active = n_test > 0
            # A perfect fit's roundoff residuals cannot define a Gaussian scale.
            roundoff = np.finfo(float).eps * max(design.shape[-2:])
            fitted_magnitude = np.einsum("bij,bj->bi", np.abs(design), np.abs(beta))
            residual_scale = np.maximum(np.abs(response), fitted_magnitude)
            resolved = rss > np.sum((roundoff * residual_scale) ** 2, axis=1)
            good = (n_train > rank) & np.isfinite(variance) & resolved
            valid &= ~active | good
            variance = np.where(good, variance, 1.0)
            residuals = test[:, :, -1] - np.einsum("bij,bj->bi", test[:, :, cols], beta)
            rss_test = np.sum(residuals**2, axis=1)
            fold_ll[:, fold] = -0.5 * (
                n_test * np.log(2 * np.pi * variance) + rss_test / variance
            )
        total_n = counts.sum(axis=1)
        scores = np.divide(
            fold_ll.sum(axis=1), total_n, out=np.full(size, np.nan), where=total_n > 0
        )
        scores[~valid] = np.nan
        fold_ll[~valid] = np.nan
        return scores, fold_ll

    def compare(
        self,
        contrasts: Sequence[Contrast],
        *,
        n_resamples: int = 9999,
        seed: int = 20260909,
        batch_size: int = 64,
    ) -> tuple[list[dict], np.ndarray]:
        """Score paired contrasts and retain every full-refit bootstrap effect.

        Failed resamples are counted in the tail. Confidence limits
        conservatively place failed draws in both tails. All contrasts use
        identical draws and complete rows.

        Args:
            contrasts: Baseline and target models with unique comparison names.
            n_resamples: Number of whole-material bootstrap draws.
            seed: Reproducible NumPy generator seed.
            batch_size: Memory bound for simultaneous refits.
        """
        if n_resamples < 1 or batch_size < 1:
            raise ValueError("Positive resampling and batch counts are required")
        if not contrasts or len({c.name for c in contrasts}) != len(contrasts):
            raise ValueError("Contrasts need unique names")
        models = list(dict.fromkeys(tuple(c.baseline) for c in contrasts))
        models += [
            m
            for m in dict.fromkeys(tuple(c.target) for c in contrasts)
            if m not in models
        ]
        weighted, counts = self._weighted_data(np.ones((1, self.n_groups)))
        observed = {m: self._score(m, weighted, counts) for m in models}
        draws = np.empty((n_resamples, len(contrasts)))
        rng = np.random.default_rng(seed)
        for start in range(0, n_resamples, batch_size):
            size = min(batch_size, n_resamples - start)
            weights = rng.multinomial(
                self.n_groups, np.full(self.n_groups, 1 / self.n_groups), size=size
            )
            weighted, counts = self._weighted_data(weights)
            fitted = {m: self._score(m, weighted, counts)[0] for m in models}
            for j, c in enumerate(contrasts):
                draws[start : start + size, j] = fitted[c.target] - fitted[c.baseline]
        rows = []
        fold_n = np.bincount(self.row_folds)
        for j, c in enumerate(contrasts):
            effect = float(observed[c.target][0][0] - observed[c.baseline][0][0])
            if not np.isfinite(effect):
                raise ValueError(f"Observed CV comparison is not estimable: {c.name}")
            fold_delta = observed[c.target][1][0] - observed[c.baseline][1][0]
            # Signed sums have the same test statistic (up to a fixed scale)
            # as the overall held-out mean, including unequal fold sizes.
            p_fold = paired_permutation_pvalue(fold_delta)
            summary = bootstrap_summary(effect, draws[:, j])
            rows.append(
                {
                    "contrast": c.name,
                    "protocol": PROTOCOL,
                    "mean_delta_ll": effect,
                    "ci_low": summary["ci_low"],
                    "ci_high": summary["ci_high"],
                    "p_fold_exact": p_fold,
                    "p_bootstrap": summary["p_bootstrap"],
                    "pvalue": max(p_fold, summary["p_bootstrap"]),
                    "n_obs": len(self.y),
                    "n_groups": self.n_groups,
                    "n_folds": self.n_folds,
                    "fold_p_floor": 2 / 2**self.n_folds,
                    "bootstrap_resolution": 1 / (n_resamples + 1),
                    "n_resamples": n_resamples,
                    "bootstrap_failures": summary["bootstrap_failures"],
                    "seed": seed,
                    "sigma_correction": self.sigma_correction,
                    "fold_delta_ll": fold_delta.tolist(),
                    "fold_n_test": fold_n.tolist(),
                }
            )
        return rows, draws
