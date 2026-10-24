"""Keep reference anchors independent and distinguish them from comparison tools."""

from pathlib import Path

import pytest

from replica.utils.imports import import_names
from scripts.checks.source.import_boundaries import _rule_findings, _scan

_SRC = Path(__file__).resolve().parents[2] / "src"
_STUDIES = _SRC / "replica" / "studies"
_R_BACKENDS = _SRC / "replica" / "analysis" / "models"
_ANALYSIS = ("replica.analysis", "replica.feature", "replica.multiverse")
# These modules supply fixed anchors, read retained author results or execute
# an independent engine. Helpers they import must obey the same boundary.
_ANCHORS = {
    "clark2025wordentropy/reference.py",
    "giulianelli2024generalized/swap_reference.py",
    "hopton2026speech/reference.py",
    "hopton2026unigram/reference.py",
    "nair2026clozing/reference.py",
    "shain2021cdr/synthetic/regenerate_reference.py",
    "shain2021cdr/synthetic/tf_reference.py",
    "shain2024logarithmic/authors/authors_gam_reference.py",
}
# A filename containing "reference" does not make the whole module an
# independent numerical implementation. Keep the direct-import guard on these
# comparison/reporting tools, without claiming their entire import graph is independent.
_COMPARISONS = {
    "kuribayashi2021perplexity/analysis/syntax_reference.py": "Executes an isolated author R cell using the notebook hash from refits.",
    "kuribayashi2025internal/reanalysis/complete_reference.py": "Aggregates checked fits and production reanalysis sensitivity tables.",
    "oota2023linguistic/authors_reference.py": "Compares the author implementation with the production encoding arm.",
    "shain2021cdr/authors_recovery_reference.py": "Compares author-fit parameters using the production parameter transform.",
    "shain2024logarithmic/tables/reference_results.py": "Reports archived native refits; it is not an independent generator.",
}
_R_ANCHORS = {
    "giulianelli2023infovalue/ml_reference.R",
    "giulianelli2023infovalue/transfer_association_reference.R",
    "giulianelli2023infovalue/transfer_correlation_reference.R",
    "giulianelli2023infovalue/transfer_reference.R",
    "hopton2026unigram/hopton2026_r_reference.R",
    "meister2024similarity/transfer_reference.R",
    "opedal2024context/meco_reference.R",
    "opedal2024context/opedal2024context_r_reference.R",
    "pimentel2023anticipation/pimentel2023anticipation_r_reference.R",
    "wilcox2023surprisal/reproduction/wilcox2023surprisal_r_link_reference.R",
    "wilcox2023surprisal/reproduction/wilcox2023surprisal_r_reference.R",
    "xu2023linearity/xu2023linearity_r_reference.R",
}
#: Reference module -> (module it may import, why its anchor stays independent).
_ALLOWED = {
    "shain2021cdr/authors_recovery_reference.py": (
        "replica.analysis.models.deconvolution.model",
        "maps the authors-engine raw kernel tensors to constrained values with "
        "CDRModel.constrained_params, the transform TestForwardAnchor matches "
        "against the authors' engine; no REPLICA fit enters the anchor",
    ),
}


def _within(name: str, module: str) -> bool:
    return name == module or name.startswith(module + ".")


def _present(paths):
    return {p for p in paths if (_STUDIES / p).parent.is_dir()}


def _reference_errors(root, origins):
    modules, edges, unresolved = _scan(root)
    owners = {path.relative_to(root).as_posix(): name for name, path in modules.items()}
    # Unknown imports are forbidden edges, including when reached through a helper.
    for location in unresolved:
        path, _, line = location.rpartition(":")
        edges.append(
            dict(
                source=owners[path],
                target="<computed-import>",
                owner=None,
                path=path,
                line=int(line),
            )
        )
    return _rule_findings(
        {
            "name": "independent-reference",
            "sources": list(origins),
            # Root re-exports are lazy; require imports from explicit owners.
            "forbidden": ["replica", "<computed-import>"]
            + [p for layer in _ANALYSIS for p in (layer, layer + ".*")],
            "transitive": True,
        },
        modules,
        edges,
    )


def test_reference_inventory_covers_present_study_packages():
    actual = {
        p.relative_to(_STUDIES).as_posix()
        for pattern in ("*reference*.py", "*reference*.R")
        for p in _STUDIES.rglob(pattern)
        if p.parent != _STUDIES
    }
    assert not (_ANCHORS & _COMPARISONS.keys())
    assert actual == _present(_ANCHORS | _COMPARISONS.keys() | _R_ANCHORS)


def test_python_anchors_have_no_transitive_analysis_dependency():
    origins = [
        "replica.studies." + p.removesuffix(".py").replace("/", ".")
        for p in _present(_ANCHORS)
    ]
    assert origins, "no independent Python anchors found"
    assert not _reference_errors(_SRC.parent, origins)


@pytest.mark.parametrize(
    ("source", "target"),
    [
        ("from .analysis import fit\n", "replica.analysis.fit"),
        ("from replica import AnalysisResult as fit\n", "replica.AnalysisResult"),
        ("import replica as pg\nfit = pg.AnalysisResult\n", "replica"),
        (
            "from importlib import import_module\n"
            "name = 'replica.analysis'\nfit = import_module(name).fit\n",
            "<computed-import>",
        ),
    ],
)
@pytest.mark.parametrize("via_helper", [False, True])
def test_anchor_guard_rejects_direct_and_indirect_analysis_imports(
    tmp_path, via_helper, source, target
):
    package = tmp_path / "src/replica"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "anchor.py").write_text(
        "from .helper import fit\n" if via_helper else source
    )
    (package / "helper.py").write_text(source)
    (package / "analysis.py").write_text("def fit(): pass\n")
    errors = _reference_errors(tmp_path, ["replica.anchor"])
    assert len(errors) == 1
    assert "replica.anchor" in errors[0] and target in errors[0]
    if via_helper:
        assert "replica.helper" in errors[0]
    (package / "analysis.py").unlink()
    (package / "unrelated.py").write_text(source)
    (package / "helper.py").write_text("def fit(): pass\n")
    (package / "anchor.py").write_text("from .helper import fit\n")
    assert not _reference_errors(tmp_path, ["replica.anchor"])


def test_r_anchor_generators_do_not_source_replica_r_code():
    """Comments may point at the registry; sourcing REPLICA's R backends
    (``replica/analysis/models/**/*.R``) would make the anchor circular."""
    r_refs = sorted(_STUDIES.rglob("*reference*.R"))
    assert r_refs, "no R anchor generators found"
    backends = sorted(path.name for path in _R_BACKENDS.rglob("*.R"))
    assert backends, f"expected REPLICA's R backends under {_R_BACKENDS}"
    for path in r_refs:
        text = path.read_text()
        named = [name for name in backends if name in text]
        assert "replica/analysis" not in text and "analysis/models/" not in text, (
            f"{path.name} sources REPLICA R code"
        )
        assert not named, f"{path.name} names REPLICA R backends {named}"


def test_python_anchor_generators_do_not_import_replica_analysis_code():
    modules = {
        path.relative_to(_STUDIES).as_posix(): path
        for path in sorted(_STUDIES.rglob("*reference*.py"))
    }
    assert modules, "no Python reference modules found"
    assert _present(_ALLOWED) <= set(modules)
    offenders = {}
    for key, path in modules.items():
        package = ".".join(path.relative_to(_SRC).with_suffix("").parts[:-1])
        allowed = _ALLOWED[key][0] if key in _ALLOWED else None
        imported = sorted(
            name
            for name in import_names(path.read_text(), package)
            if any(_within(name, layer) for layer in _ANALYSIS)
            and not (allowed and _within(name, allowed))
        )
        if imported:
            offenders[key] = imported
    assert not offenders, f"reference modules import REPLICA analysis code: {offenders}"
