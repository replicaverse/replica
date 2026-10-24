"""Per-paper study recipes, each provides ``dataset()``, ``evaluation_spec()``, ``analysis_config()``.

Each module packages one paper's recipe, and the reproduction tests anchor
on it. Components are injectable, so a recipe can be re-run with one
analysis decision swapped (a robustness check) or on other datasets
(a replication).

Example::

    from replica.studies import wilcox2023surprisal

    ds = wilcox2023surprisal.dataset(language="en")
    spec = wilcox2023surprisal.evaluation_spec(language="en", dataset=ds)
    config = wilcox2023surprisal.analysis_config()
    results = evaluate(ds, spec, config)

Naming: ``lastname{year}{slug}``, where the slug disambiguates papers
by the same first author + year (e.g. ``giulianelli2024tokenization``
vs ``giulianelli2024generalized``).

Importing a study reaches the model and analysis stacks (torch, rpy2), which
the docs build does not install, so the studies resolve on first use rather
than at package import; light submodules such as ``headline_claims`` stay
importable anywhere.

To add a study, see ``docs/tutorials/extending_replica.ipynb``.
"""

import importlib
import inspect
import sys
import types

from .base import DataDependency, Study
from .packages import study_ids

# Registered study ids without constructing anything: reading ``ALL_STUDIES``
# imports and instantiates every study package, which callers that only need
# names must not pay.
STUDY_IDS: tuple[str, ...] = study_ids()

_instances: dict[str, Study] = {}


def _study_class(name: str) -> type[Study]:
    """The Study subclass a package exports: its class name lowercases to the package name."""
    module = importlib.import_module(f".{name}", __name__)
    for attr, value in vars(module).items():
        if (
            inspect.isclass(value)
            and issubclass(value, Study)
            and value is not Study
            and attr.lower() == name
        ):
            return value
    raise AttributeError(
        f"{module.__name__} exports no Study subclass named like the package"
    )


def _instance(name: str) -> Study:
    """The shared Study instance for one package, built on first use."""
    inst = _instances.get(name)
    if inst is None:
        inst = _study_class(name)()
        _instances[name] = inst
    return inst


class _LazyStudies(types.ModuleType):
    """Resolves each study name to its Study instance on first access.

    The import system rebinds a package attribute to the submodule whenever
    that submodule is imported directly, so an ordinary module ``__getattr__``
    would answer with the module or the instance depending on import order.
    Resolving at ``__getattribute__`` keeps ``replica.studies.<name>`` the
    instance whatever the import order. A study class is reachable from this
    package under its own name as well, resolved through the package whose
    name it lowercases to: ``Wilcox2023Surprisal`` through ``wilcox2023surprisal``.
    """

    def __getattribute__(self, name: str):
        if name in STUDY_IDS:
            return _instance(name)
        return super().__getattribute__(name)

    def __getattr__(self, name: str):
        if name != name.lower() and name.lower() in STUDY_IDS:
            return _study_class(name.lower())
        if name == "ALL_STUDIES":
            value = {pkg: _instance(pkg) for pkg in STUDY_IDS}
            self.__dict__[name] = value
            return value
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    def __dir__(self):
        return sorted(set(super().__dir__()) | set(STUDY_IDS) | {"ALL_STUDIES"})


sys.modules[__name__].__class__ = _LazyStudies

__all__ = ["ALL_STUDIES", "STUDY_IDS", "DataDependency", "Study", *STUDY_IDS]
