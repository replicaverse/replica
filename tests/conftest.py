import os
from copy import deepcopy
from functools import wraps

import numpy as np
import pytest

from replica.core import Stimulus
from replica.feature import FeatureModelOutput

from ._resource_policy import resource_errors

_OFFLINE_TIERS = {
    "fast": "not artifact and not statistical and not dependency",
    "regression": "artifact or statistical or dependency",
    "all": "",
}
_OFFLINE_EXCLUSIONS = (
    "not slow and not needs_model and not needs_r and not needs_data "
    "and not needs_network and not needs_lmer_rs"
)


@pytest.fixture(autouse=True)
def seed_rng():
    """Fix numpy RNG seed for reproducible tests."""
    np.random.seed(42)


@pytest.fixture(scope="session")
def _published_evidence():
    """Read the committed registries once per worker."""
    from replica.studies.claims.registry import load_registry
    from replica.studies.claims.replication_results import load_results

    registry = load_registry()
    return registry, load_results(registry)


@pytest.fixture
def claim_registry(_published_evidence):
    """Give each test its own mutable claims."""
    return deepcopy(_published_evidence[0])


@pytest.fixture
def replication_results(_published_evidence):
    """Give each test its own mutable result summaries."""
    return deepcopy(_published_evidence[1])


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(items):
    published = {"claim_registry", "replication_results"}
    errors = []
    for item in items:
        errors.extend(resource_errors(item))
        if published.intersection(item.fixturenames) and not item.get_closest_marker(
            "artifact"
        ):
            raise pytest.UsageError(
                f"{item.nodeid} uses published evidence; add @pytest.mark.artifact"
            )
    if errors:
        raise pytest.UsageError("\n".join(errors))


def pytest_addoption(parser):
    parser.addoption(
        "--offline-tier",
        choices=tuple(_OFFLINE_TIERS),
        default=None,
        help="Select the fast, regression or combined offline suite used by CI.",
    )
    parser.addoption(
        "--require-external",
        action="store_true",
        help="Fail if any selected external test is skipped.",
    )
    parser.addoption(
        "--data-root",
        action="store",
        default=None,
        help="Root directory containing downloaded datasets",
    )


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    result = yield
    report = result.get_result()
    if report.skipped and item.config.getoption("--require-external"):
        report.outcome = "failed"
        report.longrepr = f"Required test was skipped: {item.nodeid}\n{report.longrepr}"


@pytest.hookimpl(hookwrapper=True)
def pytest_make_collect_report(collector):
    """Fail a module skipped at import, such as by ``pytest.importorskip``.

    Rewriting the report before ``pytest_collectreport`` runs lets the session
    count it as a collection error, so the run exits non-zero.
    """
    result = yield
    report = result.get_result()
    if report.skipped and collector.config.getoption("--require-external"):
        report.outcome = "failed"
        report.longrepr = (
            f"Required module was skipped: {report.nodeid}\n{report.longrepr}"
        )


@pytest.fixture(scope="session")
def data_root():
    """Root directory for dataset files (used by tests that hit real data).

    The CLI override is applied in pytest_configure, so fixtures and registry
    factories use the same configured root, including in installed-wheel tests.
    """
    from replica.dataset.catalog.registry import data_dir

    return os.fspath(data_dir())


def pytest_configure(config):
    """Apply the data root and avoid rehashing source files on every test load."""
    tier = config.getoption("--offline-tier")
    if tier is not None:
        selections = [_OFFLINE_TIERS[tier], _OFFLINE_EXCLUSIONS, config.option.markexpr]
        config.option.markexpr = " and ".join(f"({s})" for s in selections if s)
    root = config.getoption("--data-root")
    if root is not None:
        os.environ["PSYCHO_DATA_DIR"] = root
    os.environ.setdefault("PSYCHO_SKIP_CHECKSUMS", "1")
    if config.getoption("disable_socket", default=False):
        # Git runs in a child process, beyond pytest-socket's Python boundary.
        os.environ["GIT_ALLOW_PROTOCOL"] = "file"


def _all_subclasses(cls):
    for sub in cls.__subclasses__():
        yield sub
        yield from _all_subclasses(sub)


@pytest.fixture(scope="session", autouse=True)
def _cache_dataset_loads():
    """Load each corpus at most once per session.

    Per-corpus test classes load via function-scoped ``autouse`` fixtures, so
    without this a corpus re-parses before every test method, and again in each
    file touching it. Wraps each loader's ``load`` to snapshot the populated
    instance state on the first call and restore it on later ones.
    ``PSYCHO_CACHE_DATASETS=1`` explicitly enables it. Cached state is copied
    both when saved and restored so mutations do not leak between tests.

    A failed load is not cached.
    """
    from replica.dataset.base import Dataset

    if os.environ.get("PSYCHO_CACHE_DATASETS") != "1":
        yield
        return

    cache: dict = {}
    originals: dict = {}

    def make_cached(orig):
        @wraps(orig)
        def cached_load(self, *args, **kwargs):
            path = args[0] if args else kwargs.get("path")
            # Cache only real on-disk corpus loads (the slow ones). Skip
            # synthetic/no-path loaders: toy test datasets reuse the same
            # (class, no-path) signature with different in-memory data, so
            # caching them would share state. Also keeps the wrapper
            # signature-agnostic for loaders with a non-(path) signature.
            if not path or not os.path.exists(str(path)):
                return orig(self, *args, **kwargs)
            # Key includes the pre-load instance state (constructor args like
            # GECO's ``group`` / MECO's ``language``), same (class, path) can
            # load different content depending on those, so keying on path
            # alone would collide across variants.
            ctor = repr(sorted((k, repr(v)) for k, v in self.__dict__.items()))
            key = (
                type(self).__name__,
                str(path),
                repr(args[1:]),
                repr(sorted(kwargs.items())),
                ctor,
            )
            if key in cache:
                self.__dict__.update(deepcopy(cache[key]))
                return None
            result = orig(self, *args, **kwargs)
            cache[key] = deepcopy(self.__dict__)
            return result

        return cached_load

    for cls in set(_all_subclasses(Dataset)):
        if "load" in cls.__dict__:
            originals[cls] = cls.__dict__["load"]
            cls.load = make_cached(originals[cls])
    try:
        yield
    finally:
        for cls, orig in originals.items():
            cls.load = orig


class _ConstantLM:
    """Toy language model that returns constant log-probs for pipeline testing."""

    name = "constant_lm"
    vocab = ["a", "b", "c"]

    def __call__(self, stimulus: Stimulus, extract=None, target_layer="words"):
        n = len(stimulus.layers[target_layer])
        nlp = np.log(np.array([1 / 3, 1 / 3, 1 / 3]))
        outputs = []
        for t in range(n):
            lp = np.nan if t == 0 else -1.0
            outputs.append(
                FeatureModelOutput(values={"log_p_observed": lp, "next_log_probs": nlp})
            )
        return outputs


@pytest.fixture
def constant_lm():
    """Shared constant LM instance for pipeline testing."""
    return _ConstantLM()


@pytest.fixture(scope="session")
def lm_cache():
    """Session-scoped cache of ``HuggingFaceLM`` instances.

    Loading GPT-2-class weights takes seconds per instantiation, so
    ``lm_cache(model_name, **kwargs)`` returns one shared instance per
    construction signature for the whole session (per xdist worker). Instances
    are stateless after init, so sharing across modules is safe.
    """
    cache: dict = {}

    def get(model_name: str = "gpt2", **kwargs):
        key = (model_name, tuple(sorted(kwargs.items())))
        if key not in cache:
            from replica.feature import HuggingFaceLM

            cache[key] = HuggingFaceLM(model_name, **kwargs)
        return cache[key]

    yield get
    cache.clear()
