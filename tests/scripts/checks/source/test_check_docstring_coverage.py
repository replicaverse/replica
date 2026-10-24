"""Docstring gate (scripts/checks/source/docstrings.py)."""

import ast
import textwrap

import pytest

from scripts.checks.source import docstrings as checker


def write(tmp_path, source: str, name: str = "mod.py"):
    """Write a dedented module and return its path."""
    path = tmp_path / name
    path.write_text(textwrap.dedent(source).lstrip(), encoding="utf-8")
    return path


def func(source: str):
    """Parse a single function definition out of *source*."""
    return ast.parse(textwrap.dedent(source).lstrip()).body[0]


def test_args_section_names_are_parsed_with_or_without_types():
    doc = """
    Summary.

    Args:
        plain: One.
        typed (int): Two.
        *extras: Three.
        **rest: Four.
    """
    assert checker.documented_params(textwrap.dedent(doc)) == {
        "plain",
        "typed",
        "extras",
        "rest",
    }


def test_parsing_stops_at_the_next_section():
    doc = """
    Summary.

    Args:
        real: One.

    Returns:
        fake: not a parameter.
    """
    assert checker.documented_params(textwrap.dedent(doc)) == {"real"}


def test_a_wrapped_sphinx_role_is_description_not_a_parameter():
    """A continuation line starting with a role must not read as an entry."""
    doc = """
    Summary.

    Args:
        window: One of the choices
            see :data:`WINDOWS` for the list.
        mode: Resolved through
            :func:`resolve` before use.
    """
    assert checker.documented_params(textwrap.dedent(doc)) == {"window", "mode"}


def test_a_docstring_without_an_args_section_describes_nothing():
    assert checker.documented_params("Summary only.\n") == set()


def test_required_params_keep_varargs_and_drop_self():
    node = func("""
        def f(self, a, b=1, *args, c=2, **kwargs):
            pass
    """)
    assert checker.required_params(node) == ["a", "b", "c", "args", "kwargs"]


def test_an_underscore_name_is_allowed_but_not_required():
    """A signature marks a parameter unused by naming it with an underscore."""
    node = func("""
        def f(a, _unused, **_):
            pass
    """)
    assert checker.required_params(node) == ["a"]
    assert checker.signature_params(node) == ["a", "_unused", "_"]


def test_describing_an_underscore_parameter_is_not_a_phantom(tmp_path):
    path = write(
        tmp_path,
        '''
        """Module."""


        def f(a, _tags):
            """Summary.

            Args:
                a: The input.
                _tags: Accepted for protocol parity, unread.
            """
    ''',
    )
    assert checker.violations(path) == []


def test_a_run_on_module_summary_is_reported(tmp_path):
    path = write(
        tmp_path,
        '''
        """A summary that keeps going
        onto a second line."""
    ''',
    )
    assert [(rule, detail) for _, rule, detail in checker.violations(path)] == [
        ("summary", "summary line runs into the next line")
    ]


def test_a_module_summary_followed_by_a_blank_line_passes(tmp_path):
    path = write(
        tmp_path,
        '''
        """A standalone summary.

        Then the body.
        """
    ''',
    )
    assert checker.violations(path) == []


def test_an_undescribed_parameter_is_reported(tmp_path):
    path = write(
        tmp_path,
        '''
        """Module."""


        def f(a, b):
            """Summary.

            Args:
                a: The first one.
            """
    ''',
    )
    assert [(rule, detail) for _, rule, detail in checker.violations(path)] == [
        ("params", "f: b")
    ]


@pytest.mark.parametrize("signature", ["a", "", "_unused"])
def test_a_parameter_the_signature_lacks_is_reported(tmp_path, signature):
    path = write(
        tmp_path,
        '''
        """Module."""


        def f(SIGNATURE):
            """Summary.

            Args:
                gone: Removed from the signature.
            """
    '''.replace("SIGNATURE", signature),
    )
    assert ("phantom", "f: gone") in [
        (rule, detail) for _, rule, detail in checker.violations(path)
    ]


def test_a_class_docstring_may_describe_the_constructor(tmp_path):
    path = write(
        tmp_path,
        '''
        """Module."""


        class C:
            """Summary.

            Args:
                a: Held as state.
            """

            def __init__(self, a):
                """Store *a*."""
    ''',
    )
    assert checker.violations(path) == []


def test_a_section_the_renderer_cannot_render_is_reported(tmp_path):
    """quartodoc raises NotImplementedError on Yields and Warns, failing the build."""
    path = write(
        tmp_path,
        '''
        """Module."""


        def gen(n):
            """Make squares.

            Args:
                n: How many.

            Yields:
                One square per step.
            """
    ''',
    )
    assert ("section", "Yields: does not render") in [
        (rule, detail) for _, rule, detail in checker.violations(path)
    ]


def test_returns_and_raises_sections_are_accepted(tmp_path):
    path = write(
        tmp_path,
        '''
        """Module."""


        def f(n):
            """Make squares.

            Args:
                n: How many.

            Returns:
                An iterator of squares.

            Raises:
                ValueError: When *n* is negative.
            """
    ''',
    )
    assert checker.violations(path) == []


def test_a_missing_module_docstring_is_reported(tmp_path):
    path = write(tmp_path, "x = 1\n")
    assert [(rule, detail) for _, rule, detail in checker.violations(path)] == [
        ("module", "module has no docstring")
    ]


def test_nested_definitions_and_private_symbols_are_out_of_scope(tmp_path):
    path = write(
        tmp_path,
        '''
        """Module."""


        def outer():
            """Documented."""

            def inner(a, b):
                return a + b

            return inner


        def _private(a):
            return a


        class _Hidden:
            def method(self, a):
                return a
    ''',
    )
    assert checker.violations(path) == []


def test_dunders_other_than_init_and_call_are_out_of_scope(tmp_path):
    path = write(
        tmp_path,
        '''
        """Module."""


        class C:
            """Documented."""

            def __repr__(self):
                return "C()"

            def __call__(self, a):
                """Run it.

                Args:
                    a: The input.
                """
    ''',
    )
    assert checker.violations(path) == []


def test_overload_stubs_are_out_of_scope(tmp_path):
    path = write(
        tmp_path,
        '''
        """Module."""

        from typing import overload


        @overload
        def f(a: int) -> int: ...


        def f(a):
            """Double it.

            Args:
                a: The input.
            """
            return a * 2
    ''',
    )
    assert checker.violations(path) == []


def test_targets_skip_private_modules_and_the_excluded_study(tmp_path, monkeypatch):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "keep.py").write_text("", encoding="utf-8")
    (tmp_path / "pkg" / "_skip.py").write_text("", encoding="utf-8")
    monkeypatch.setattr(checker, "REPO", tmp_path)
    monkeypatch.setattr(checker, "EXCLUDE", ("pkg/frozen",))
    monkeypatch.setattr(checker, "IN_SCOPE", ("pkg/frozen/lib/",))
    (tmp_path / "pkg" / "frozen" / "lib").mkdir(parents=True)
    (tmp_path / "pkg" / "frozen" / "mod.py").write_text("", encoding="utf-8")
    (tmp_path / "pkg" / "frozen" / "lib" / "api.py").write_text("", encoding="utf-8")
    assert [p.name for p in checker.targets(["pkg"])] == ["api.py", "keep.py"]


def test_the_excluded_study_path_exists():
    """A stale exclusion would silently drop a package from the gate."""
    for rel in (*checker.EXCLUDE, *checker.IN_SCOPE):
        assert (checker.REPO / rel).is_dir(), rel


def test_report_mode_lists_every_rule(tmp_path, monkeypatch, capsys):
    write(tmp_path, "def f(a):\n    return a\n")
    monkeypatch.setattr(checker, "REPO", tmp_path)
    monkeypatch.setattr(checker, "ROOTS", ("",))
    assert checker.main(["--report"]) == 1
    out = capsys.readouterr()
    assert "module (1)" in out.out and "missing (1)" in out.out
    assert "docstring violation(s)" in out.err


def test_a_checkout_under_an_underscore_directory_is_scanned(
    tmp_path, monkeypatch, capsys
):
    """Only path parts inside the repository mark a module private."""
    repo = tmp_path / "_work" / "repo"
    (repo / "pkg").mkdir(parents=True)
    (repo / "pkg" / "keep.py").write_text('"""Kept."""\n', encoding="utf-8")
    (repo / "pkg" / "_skip.py").write_text("", encoding="utf-8")
    monkeypatch.setattr(checker, "REPO", repo)
    assert [p.name for p in checker.targets(["pkg"])] == ["keep.py"]
    assert checker.main(["pkg"]) == 0
    assert "1 files, no violations" in capsys.readouterr().out


def test_a_missing_path_fails(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(checker, "REPO", tmp_path)
    assert checker.main(["no_such_dir"]) == 1
    assert "no such path in the repository: no_such_dir" in capsys.readouterr().err


def test_an_empty_scope_fails(tmp_path, monkeypatch, capsys):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "_private.py").write_text("x = 1\n", encoding="utf-8")
    monkeypatch.setattr(checker, "REPO", tmp_path)
    assert checker.main(["pkg"]) == 1
    assert "no Python files in scope under pkg" in capsys.readouterr().err
