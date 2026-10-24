"""Resource requirements belong to each collected case, including fixtures."""

import pytest

pytest_plugins = ["pytester"]

_HOOK = """
import pytest
from tests._resource_policy import resource_errors

@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(items):
    errors = [error for item in items for error in resource_errors(item)]
    if errors:
        raise pytest.UsageError("\\n".join(errors))
"""


def _collect(pytester, source, fixtures="", module=None):
    pytester.makeini("""
[pytest]
markers =
    needs_model: pretrained models
    needs_data: corpus data
    needs_r: R backends
    needs_lmer_rs: Rust mixed models
    resource_free(reason): reviewed preflight or stub
""")
    pytester.makeconftest(_HOOK + fixtures)
    if module is None:
        pytester.makepyfile(source)
    else:
        pytester.makepyfile(**{module: source})
    return pytester.runpytest("--collect-only", "--strict-markers", "-q")


def test_a_marked_sibling_cannot_hide_an_unmarked_model_test(pytester):
    result = _collect(
        pytester,
        """
import pytest
@pytest.mark.needs_model
def test_marked():
    HuggingFaceLM('model')
def test_unmarked():
    HuggingFaceLM('model')
""",
    )
    assert result.ret == pytest.ExitCode.USAGE_ERROR
    result.stderr.fnmatch_lines(["*test_unmarked: model loading requires*"])
    assert "test_marked: model loading requires" not in result.stderr.str()


@pytest.mark.parametrize("marker", ["needs_r", "needs_lmer_rs"])
def test_r_tiers_cannot_load_a_model(pytester, marker):
    result = _collect(
        pytester,
        f"""
import pytest
@pytest.mark.{marker}
def test_model():
    ResponsivityLM('gpt2')
""",
    )
    assert result.ret == pytest.ExitCode.USAGE_ERROR
    message = "test_model: model loading requires needs_model or needs_data"
    assert message in result.stderr.str()


@pytest.mark.parametrize(
    "fixture,marker", [("lm_cache", "needs_model"), ("data_root", "needs_data")]
)
def test_transitive_fixture_requirements_use_each_items_markers(
    pytester, fixture, marker
):
    fixtures = f"""
@pytest.fixture
def {fixture}():
    return object()
@pytest.fixture
def indirect({fixture}):
    return {fixture}
"""
    result = _collect(
        pytester,
        f"""
import pytest
@pytest.mark.{marker}
def test_marked(indirect): pass
def test_unmarked(indirect): pass
""",
        fixtures,
    )
    assert result.ret == pytest.ExitCode.USAGE_ERROR
    assert "test_unmarked:" in result.stderr.str()
    assert "test_marked:" not in result.stderr.str()


def test_fixture_body_constructor_requires_marker(pytester):
    result = _collect(
        pytester,
        "def test_model(model): pass",
        """
@pytest.fixture
def model():
    return HuggingFaceLM('model')
""",
    )
    assert result.ret == pytest.ExitCode.USAGE_ERROR
    result.stderr.fnmatch_lines(["*test_model: model loading requires*"])


def test_class_inheritance_and_parameter_marks_are_respected(pytester):
    result = _collect(
        pytester,
        """
import pytest
class Base:
    def test_model(self):
        HuggingFaceLM('model')
@pytest.mark.needs_model
class TestInherited(Base): pass
@pytest.mark.parametrize('value', [
    pytest.param(1, marks=pytest.mark.needs_model, id='marked'),
    pytest.param(2, id='unmarked'),
])
def test_parameter(value):
    HuggingFaceLM('model')
""",
    )
    assert result.ret == pytest.ExitCode.USAGE_ERROR
    assert "test_parameter[unmarked]:" in result.stderr.str()
    assert "test_parameter[marked]:" not in result.stderr.str()
    assert "TestInherited::test_model:" not in result.stderr.str()


def test_module_markers_cover_every_item(pytester):
    result = _collect(
        pytester,
        """
import pytest
pytestmark = pytest.mark.needs_model
def test_one(): HuggingFaceLM('one')
def test_two(): HuggingFaceLM('two')
""",
    )
    assert result.ret == pytest.ExitCode.OK


def test_unused_resource_fixture_does_not_affect_an_item(pytester):
    result = _collect(
        pytester,
        "def test_plain(): pass",
        """
@pytest.fixture
def unused():
    return HuggingFaceLM('model')
""",
    )
    assert result.ret == pytest.ExitCode.OK


@pytest.mark.parametrize("uses_parent", [False, True])
def test_fixture_overrides_only_include_reachable_parent_bodies(pytester, uses_parent):
    argument = "model" if uses_parent else ""
    result = _collect(
        pytester,
        f"""
import pytest
@pytest.fixture
def model({argument}):
    return object()
def test_model(model): pass
""",
        """
@pytest.fixture
def model():
    return HuggingFaceLM('model')
""",
    )
    assert result.ret == (
        pytest.ExitCode.USAGE_ERROR if uses_parent else pytest.ExitCode.OK
    )


@pytest.mark.parametrize(
    "reason,body,message",
    [
        ("", "HuggingFaceLM('model')", "resource_free requires a reason"),
        ("stub", "pass", "stale resource_free marker"),
    ],
)
def test_resource_free_exemptions_need_live_reasoned_uses(
    pytester, reason, body, message
):
    result = _collect(
        pytester,
        f"""
import pytest
@pytest.mark.resource_free({reason!r})
def test_stub(): {body}
""",
    )
    assert result.ret == pytest.ExitCode.USAGE_ERROR
    assert message in result.stderr.str()


def test_resource_free_exemption_cannot_hide_real_resource_fixtures(pytester):
    result = _collect(
        pytester,
        """
import pytest
@pytest.mark.resource_free('stub')
def test_stub(lm_cache):
    HuggingFaceLM('model')
""",
        """
@pytest.fixture
def lm_cache(): pass
""",
    )
    assert result.ret == pytest.ExitCode.USAGE_ERROR
    assert "resource_free cannot exempt resource fixtures" in result.stderr.str()


def test_reviewed_preflight_stays_in_resource_free_tier(pytester):
    result = _collect(
        pytester,
        """
import pytest
@pytest.mark.resource_free('Invalid scope fails before resolution')
def test_preflight():
    HuggingFaceLM('model', context_scope='invalid')
""",
    )
    assert result.ret == pytest.ExitCode.OK


@pytest.mark.parametrize(
    "module,marker,call,rejected",
    [
        ("tests/dataset/test_corpus", "", "REGISTRY['provo'].factory()", True),
        ("tests/studies/test_corpus", "", "repo_path('checkout')", True),
        (
            "tests/dataset/test_corpus",
            "@pytest.mark.needs_data",
            "REGISTRY['provo'].factory()",
            False,
        ),
        ("tests/feature/test_corpus", "", "REGISTRY['provo'].factory()", False),
        ("tests/dataset/test_corpus", "", "REGISTRY['provo'].default_path()", False),
        ("tests/studies/test_corpus", "", "repo_path('checkout', fetch=False)", False),
    ],
)
def test_corpus_directories_require_needs_data_for_data_access(
    pytester, module, marker, call, rejected
):
    result = _collect(
        pytester,
        f"""
import pytest
{marker}
def test_corpus():
    {call}
""",
        module=module,
    )
    expected = pytest.ExitCode.USAGE_ERROR if rejected else pytest.ExitCode.OK
    assert result.ret == expected
    message = "test_corpus: on-disk data access requires needs_data"
    assert (message in result.stderr.str()) is rejected
