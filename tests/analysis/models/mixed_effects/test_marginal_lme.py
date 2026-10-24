"""MarginalLMEModel: the per-subject marginal likelihood and its supported designs."""

import numpy as np
import pytest

from replica.analysis.models.mixed_effects.marginal_lme import MarginalLMEModel


@pytest.mark.parametrize(
    "kwargs",
    [
        {"random_slope_control_cols": [0]},
        {"focal_random_slope": True},
    ],
)
def test_uncorrelated_slopes_are_refused(kwargs):
    """lme4 orders ``||`` random effects term by term, which the per-subject
    slicing of Zt and Lambdat does not follow."""
    with pytest.raises(NotImplementedError, match="uncorrelated"):
        MarginalLMEModel(uncorrelated_slopes=True, **kwargs)


def test_an_intercept_only_structure_accepts_either_separator():
    """With no slopes ``(1 || subject)`` is a single term, so nothing is refused."""
    model = MarginalLMEModel(uncorrelated_slopes=True)
    assert model._formula(1, is_full=True) == "y ~ z.(x0) + (1 || subject)"


def _subjects_with_slopes(seed=0):
    rng = np.random.default_rng(seed)
    subject = np.repeat(np.arange(15), 30)
    x = rng.normal(size=subject.size)
    y = (
        1.0
        + rng.normal(0, 1, 15)[subject]
        + (0.3 + rng.normal(0, 0.5, 15)[subject]) * x
        + rng.normal(0, 0.5, subject.size)
    )
    return y, np.column_stack([x, subject, np.zeros(subject.size)])


@pytest.mark.needs_r
@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"random_slope_control_cols": [0]},
        {"random_slope_control_cols": [0], "scaling": "training"},
    ],
)
def test_marginal_loglik_equals_lme4_loglik_on_the_training_rows(kwargs):
    """Scored on the rows it was fit to, the marginal MVN likelihood is the
    ML fit's own ``logLik``, so the identity checks the per-subject Z and
    covariance."""
    from replica.analysis.models.support.r import _r_context

    y, X = _subjects_with_slopes()
    model = MarginalLMEModel(**kwargs)
    loglik, _ = model.fit_score(y, X, X, y, is_full=False)
    diagnostics = model.last_fit_diagnostics()
    assert diagnostics["convergence_applicable"] is True
    assert diagnostics["details"]["optimizer_convergence"] == [
        model.fit_diagnostics[-1]["optimizer_code"]
    ]
    assert diagnostics["singular"] == model.fit_diagnostics[-1]["singular"]
    with _r_context() as ro:
        reference = float(ro.r("as.numeric(logLik(.m))")[0])
    assert loglik == pytest.approx(reference, rel=1e-8)
