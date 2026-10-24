"""Skip tests with an explanatory reason when external resources are missing."""

import functools
import logging
import os

import pytest

from replica.external import DependencyMismatchError


def require_data(rep, predicate=None, reason: str | None = None):
    """Fetch-if-missing data gate: run ``rep.ensure_data()`` (clone a ``git:``
    dep or download+extract a ``release:``/``url:`` archive into the configured
    root -- ``$PSYCHO_REPL_DATA`` or the repo), then skip if the data is still
    absent. ``predicate`` overrides the default ``rep.verify()`` check.
    """
    name = getattr(rep, "name", "replication")
    base_reason = reason or f"could not provision data for {name}"

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            fetch_error = None
            try:
                rep.ensure_data()
            except DependencyMismatchError:
                raise
            except Exception as exc:
                fetch_error = exc
                logging.getLogger("replica").info(
                    "ensure_data for %s raised: %s", name, exc
                )
            ok = predicate() if predicate is not None else rep.verify()
            if not ok:
                msg = base_reason
                if fetch_error is not None:
                    msg += f" (ensure_data raised: {fetch_error!r})"
                pytest.skip(msg)
            return fn(*args, **kwargs)

        return wrapper

    return decorator


def require(predicate, reason: str):
    """Return a decorator that skips when ``predicate()`` is False.

    ``predicate`` is a zero-arg callable returning True when the resource
    the test needs is present. It runs at call time, so file probes add
    nothing to pytest collection.
    """

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if not predicate():
                pytest.skip(reason)
            return fn(*args, **kwargs)

        return wrapper

    return decorator


def load_or_skip(ds, path, *, load=None):
    """Load a dataset, skipping the test only when its data path is absent.

    An exception raised by the load itself propagates, so a loader regression
    fails the test. ``path`` may be a str or ``Path``. ``load`` overrides the
    call (default ``ds.load(str(path))``) for loaders with extra kwargs.
    Returns the loaded ``ds``.
    """
    p = str(path)
    if not os.path.exists(p):
        pytest.skip(f"data not present: {p}")
    if load is not None:
        load()
    else:
        ds.load(p)
    return ds
