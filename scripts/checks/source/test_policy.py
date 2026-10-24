#!/usr/bin/env python3
"""Check test markers and tracked references without collecting pytest cases."""

import ast
import re
import subprocess
from pathlib import Path

# FeatureModel constructors that load a real pretrained checkpoint, so a test
# using them must be excluded from the model-free tier. Model-free models
# (ClozeLM reads a stimulus layer; UnigramSurprisalLM uses a FrequencySource;
# _ConstantLM / MarkovReferenceLM are toys) are intentionally absent.
_MODEL_LOADERS = frozenset(
    {
        "HuggingFaceLM",
        "CharBeamLM",
        "LogitLensLM",
        "TunedLensLM",
        "ModelBlocksSurprisalLM",
        "WordsProbabilityLM",
        "ContinuationEntropyLM",
        "ContinuationEntropyLMBuggy",
        "MCWordEntropyLM",
        "ForwardMarginalEntropyLM",
        "ForwardWordLookaheadEntropyLM",
        "HuggingFaceEmbedding",
        "InformationValueLM",
        "WordInfoValueLM",
        "ResponsivityLM",
        "SimilarityAdjustedSurprisalEmbeddingLM",
        "SimilarityAdjustedSurprisalWordLM",
    }
)

# Directories whose tests read corpora, pinned checkouts and releases; a module
# there that resolves one of those on disk carries ``needs_data`` on the module
# or on the tests that do. Elsewhere those calls run against temporary paths
# or monkeypatched resolvers and stay in the fast tier.
_DATA_DIRS = ("studies", "integration", "dataset")

# Calls that fetch, verify or load a corpus, a pinned checkout or a release.
# Path builders stay out of the set: ``DatasetRegistration.default_path`` joins the data
# root, and ``repo_path(..., fetch=False)`` resolves a directory without a fetch.
_DATA_ACCESSORS = frozenset({"require_data", "repo_path", "factory", "ensure_data"})

# Markers a model-loading test must carry. Runs selected by needs_model or
# needs_data provide checkpoints; runs selected by needs_r or needs_lmer_rs have
# no model cache.
_MODEL_MARKERS = frozenset({"needs_model", "needs_data"})


#: What may not appear, in the order violations are reported.
_FORBIDDEN = (
    re.compile(r"/cluster/(?:scratch|home|project)/"),
    re.compile(r"\blocal/(?:scripts|reports|data|tests)\b"),
    # Both spellings of a path: a literal, and Path composition on segments.
    # "paper/REPLICA" and "paper/official-code" are prose, so a path segment
    # has to follow before this counts as a reference.
    re.compile(r"(?<![\w/])paper/(?=[a-z0-9_]+(?:/|\.[a-z]))"),
    re.compile(r"""["']paper["']\s*/"""),
    re.compile(r"\bdocs/replications\b"),
    re.compile(r"\bexternal_repos\b"),
    re.compile(r"(?i)\b(?:slurm|sbatch|squeue|euler)\b"),
    # Machine identity: node hostnames and site paths.
    re.compile(r"\beu-(?:login|lo-g)[\w-]*"),
    re.compile(r"(?i)\bskiegeland\b|cotterell/|\bmila\b"),
)

#: Files whose subject *is* the rule, so they have to name what they exclude.
_EXEMPT = {
    ".gitignore",
    "release/public.toml",
    "scripts/checks/source/test_policy.py",
    "tests/scripts/checks/source/test_check_test_policy.py",
    "tests/studies/test_study_readmes.py",
    # Owns the checkout layout: the one place allowed to spell the directory.
    "src/replica/external/repositories.py",
    "tests/studies/test_external_repo_pins.py",
}

_TEXT_SUFFIXES = {
    ".py",
    ".md",
    ".qmd",
    ".toml",
    ".cfg",
    ".yaml",
    ".yml",
    ".R",
    ".sh",
    ".ipynb",
    ".json",
    ".tsv",
    ".html",
    ".js",
    ".mjs",
    ".scss",
    ".patch",
}


def _marker_names_of(value: ast.expr) -> set[str]:
    """Marker names in a ``pytestmark`` value (single, list, or tuple)."""
    items = value.elts if isinstance(value, (ast.List, ast.Tuple)) else [value]
    out: set[str] = set()
    for it in items:
        attr = it.func if isinstance(it, ast.Call) else it
        if isinstance(attr, ast.Attribute):  # pytest.mark.NAME[(...)]
            out.add(attr.attr)
    return out


def _pytestmark_assignments(tree: ast.Module) -> list[ast.Assign]:
    return [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "pytestmark" for t in node.targets)
    ]


def _module_markers(tree: ast.Module) -> set[str]:
    """Collect the last module assignment and all function or class markers."""
    markers: set[str] = set()
    assigns = _pytestmark_assignments(tree)
    if assigns:
        markers |= _marker_names_of(assigns[-1].value)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            for dec in node.decorator_list:
                attr = dec.func if isinstance(dec, ast.Call) else dec
                if isinstance(attr, ast.Attribute):
                    markers.add(attr.attr)
    return markers


def _model_use(tree: ast.Module) -> tuple[set[str], bool]:
    """(set of model-loading constructors called, whether lm_cache fixture used)."""
    loaders: set[str] = set()
    uses_lm_cache = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in _MODEL_LOADERS:
                loaders.add(node.func.id)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if any(a.arg == "lm_cache" for a in node.args.args):
                uses_lm_cache = True
    return loaders, uses_lm_cache


def _data_access(call: ast.Call) -> str | None:
    """The accessor *call* names, or None when it only builds a path."""
    f = call.func
    name = (
        f.id
        if isinstance(f, ast.Name)
        else f.attr
        if isinstance(f, ast.Attribute)
        else None
    )
    if name not in _DATA_ACCESSORS:
        return None
    path_only = any(
        k.arg == "fetch"
        and isinstance(k.value, ast.Constant)
        and k.value.value is False
        for k in call.keywords
    )
    return None if name == "repo_path" and path_only else name


def _data_accessors(tree: ast.Module) -> set[str]:
    """Data-resolving calls made anywhere in the module."""
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = _data_access(node)
            if name is not None:
                found.add(name)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and any(
            a.arg == "data_root" for a in node.args.args
        ):
            found.add("data_root fixture")
    return found


def _marker_errors(relative: str, text: str) -> list[str]:
    tree = ast.parse(text, filename=relative)
    markers = _module_markers(tree)
    errors = []
    loaders, cache = _model_use(tree)
    if (loaders or cache) and not markers.intersection(
        _MODEL_MARKERS | {"resource_free"}
    ):
        errors.append(f"{relative}: model loading requires needs_model or needs_data")
    parts = Path(relative).parts
    if len(parts) > 2 and parts[1] in _DATA_DIRS:
        if _data_accessors(tree) and not markers.intersection(
            {"needs_data", "resource_free"}
        ):
            errors.append(f"{relative}: on-disk data access requires needs_data")
    if len(_pytestmark_assignments(tree)) > 1:
        errors.append(
            f"{relative}: repeated pytestmark assignment; combine markers in one list"
        )
    return errors


def _path_errors(relative: str, text: str) -> list[str]:
    if relative in _EXEMPT or relative.startswith(("local/", "docs/replications/")):
        return []
    return [
        f"{relative}:{number}: developer-only reference: {line.strip()[:90]}"
        for number, line in enumerate(text.splitlines(), 1)
        if any(pattern.search(line) for pattern in _FORBIDDEN)
    ]


def _check(root: Path) -> list[str]:
    listing = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z"],
        capture_output=True,
        text=True,
        check=True,
    )
    tracked = {name for name in listing.stdout.split("\0") if name}
    # Include new test modules in the marker scan before they are staged.
    tests = {
        p.relative_to(root).as_posix() for p in (root / "tests").rglob("test_*.py")
    }
    errors = [
        f"{name}: result payloads belong on the results hub"
        for name in sorted(tracked)
        if re.match(r"src/replica/studies/[^/]+/results/", name)
        and not re.fullmatch(
            r"src/replica/studies/[^/]+/results/(?:index\.json|README\.md)", name
        )
    ]
    for relative in sorted(tracked | tests):
        path = root / relative
        if not path.is_file() or path.suffix not in _TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if relative in tracked:
            errors.extend(_path_errors(relative, text))
        if relative in tests:
            errors.extend(_marker_errors(relative, text))
    return errors


if __name__ == "__main__":
    errors = _check(Path(__file__).resolve().parents[3])
    if errors:
        print("\n".join(errors))
    else:
        print("Test marker and tracked-reference policies passed.")
    raise SystemExit(bool(errors))
