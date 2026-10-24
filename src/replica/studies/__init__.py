"""Study recipes with dataset, feature specification and analysis settings.

Study packages expose ``dataset()``, ``evaluation_spec()`` and
``analysis_config()``. Components can be replaced for replication and robustness
analyses. Study instances load on first access; ``STUDY_IDS`` lists the available
packages without importing their optional dependencies.

Example::

    from replica import evaluate
    from replica.studies import wilcox2023surprisal as study

    dataset = study.dataset(language="en")
    spec = study.evaluation_spec(language="en", dataset=dataset)
    results = evaluate(dataset, spec, study.analysis_config())

Name study packages ``lastname{year}{slug}``. See
``docs/tutorials/extending_replica.ipynb`` for the extension guide.
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
