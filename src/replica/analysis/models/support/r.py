"""R conversion, package loading and coefficient extraction."""

from contextlib import contextmanager

import numpy as np


@contextmanager
def _r_context():
    """Context manager for rpy2 operations with thread-safe numpy conversion.

    Yields the ``rpy2.robjects`` module.  All numpy ↔ R conversions happen
    inside the ``localconverter`` block, avoiding global-state mutation from
    ``numpy2ri.activate()``/``deactivate()``.
    """
    try:
        import rpy2.robjects as ro
        from rpy2.robjects import numpy2ri
        from rpy2.robjects.conversion import localconverter
    except ImportError as error:
        raise ImportError(
            "R-based analysis requires a working rpy2 and R installation. "
            "Install the r extra (from a checkout: "
            "python -m pip install -e '.[r]') and the R packages "
            "required by your backend (lme4 for mixed models, mgcv for GAMs)."
        ) from error
    with localconverter(ro.default_converter + numpy2ri.converter):
        yield ro


def _to_r_dataframe(ro, X, y=None, col_prefix="x"):
    """Convert numpy arrays to an R data.frame.

    Must be called within an ``_r_context()`` block.
    """
    data = {}
    if y is not None:
        data["y"] = ro.FloatVector(y)
    for i in range(X.shape[1]):
        data[f"{col_prefix}{i}"] = ro.FloatVector(X[:, i])
    return ro.r["data.frame"](**data)


def _importr(name: str):
    """Import an R package using rpy2.robjects.packages.importr."""
    from rpy2.robjects.packages import importr

    return importr(name)


def _coefficients_by_term(
    ro, model_var: str, n_fixed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Estimates and standard errors ordered ``[intercept, x0, x1, ...]``.

    R leaves a linearly dependent column out of the coefficient table, so rows
    are placed by term name and a dropped column reads NaN. Must be called
    within an ``_r_context()`` block.
    """
    table = f"summary({model_var})$coefficients"
    rows = int(np.asarray(ro.r(f"nrow({table})"))[0])
    # lmer's table has three columns and lm's four; the shape comes from R.
    values = np.asarray(ro.r(table), dtype=float).reshape(rows, -1, order="F")
    position = {"(Intercept)": 0, **{f"x{i}": i + 1 for i in range(n_fixed)}}
    beta = np.full(1 + n_fixed, np.nan)
    se = np.full(1 + n_fixed, np.nan)
    for row, term in enumerate(ro.r(f"rownames({table})")):
        beta[position[str(term)]] = values[row, 0]
        se[position[str(term)]] = values[row, 1]
    return beta, se
