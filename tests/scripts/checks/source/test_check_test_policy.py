"""The repository scan reports missing markers and nonportable references."""

import subprocess

import pytest

from scripts.checks.source.test_policy import _check, _marker_errors, _path_errors

CLUSTER_PATH = "/cluster/" + "home/person/data"


@pytest.mark.parametrize(
    "body",
    ["HuggingFaceLM('model')", "def test_model(lm_cache): pass"],
)
def test_model_loading_requires_an_excluding_marker(body):
    assert _marker_errors("tests/feature/test_sample.py", body)
    assert not _marker_errors(
        "tests/feature/test_sample.py", "pytestmark = pytest.mark.needs_model\n" + body
    )


@pytest.mark.parametrize(
    "loader",
    [
        "InformationValueLM",
        "ModelBlocksSurprisalLM",
        "ResponsivityLM",
        "SimilarityAdjustedSurprisalEmbeddingLM",
        "SimilarityAdjustedSurprisalWordLM",
        "TunedLensLM",
        "WordInfoValueLM",
    ],
)
def test_every_pretrained_loader_requires_a_model_marker(loader):
    assert _marker_errors("tests/feature/test_sample.py", f"{loader}('gpt2')")


@pytest.mark.parametrize(
    "marker,accepted",
    [
        ("needs_model", True),
        ("needs_data", True),
        ("needs_r", False),
        ("needs_lmer_rs", False),
    ],
)
def test_only_model_and_data_tiers_may_load_a_model(marker, accepted):
    body = f"pytestmark = pytest.mark.{marker}\nHuggingFaceLM('model')"
    assert (not _marker_errors("tests/feature/test_sample.py", body)) is accepted


@pytest.mark.parametrize("marker", ["needs_data", "resource_free('stub')"])
def test_data_markers_follow_the_test_location(marker):
    body = "def test_load(data_root): pass"
    assert _marker_errors("tests/dataset/test_sample.py", body)
    assert not _marker_errors("tests/feature/test_sample.py", body)
    assert not _marker_errors(
        "tests/dataset/test_sample.py", f"@pytest.mark.{marker}\n" + body
    )


@pytest.mark.parametrize(
    "call,reads_data",
    [
        ("get('provo').default_path()", False),
        ("repo_path('checkout', fetch=False)", False),
        ("repo_path('checkout')", True),
        ("repo_path('checkout', fetch=True)", True),
        ("get('provo').factory()", True),
    ],
)
def test_building_a_path_is_not_data_access(call, reads_data):
    body = f"def test_path():\n    {call}\n"
    assert bool(_marker_errors("tests/dataset/test_sample.py", body)) is reads_data


def test_reassigning_pytestmark_reports_the_file():
    errors = _marker_errors(
        "tests/test_sample.py",
        "pytestmark = pytest.mark.needs_data\npytestmark = pytest.mark.slow",
    )
    assert len(errors) == 1
    assert "tests/test_sample.py: repeated pytestmark" in errors[0]


@pytest.mark.parametrize(
    "reference",
    [CLUSTER_PATH, "external_repos/corpus", "local/reports/run"],
)
def test_nonportable_references_report_file_and_line(reference):
    errors = _path_errors("src/sample.py", f"# heading\npath = {reference!r}\n")
    assert len(errors) == 1
    assert errors[0].startswith("src/sample.py:2:")
    assert not _path_errors("src/replica/external/repositories.py", reference)


def test_scan_checks_tracked_paths_and_unstaged_test_markers(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "tracked file.md").write_text("external_repos/corpus")
    (tmp_path / "untracked.md").write_text("external_repos/corpus")
    subprocess.run(["git", "-C", str(tmp_path), "add", "tracked file.md"], check=True)
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_new.py").write_text("HuggingFaceLM('model')")
    errors = _check(tmp_path)
    assert len(errors) == 2
    assert any(e.startswith("tracked file.md:1:") for e in errors)
    assert any(e.startswith("tests/test_new.py: model loading") for e in errors)


@pytest.mark.parametrize(
    "name",
    [
        "docs/page.qmd",
        "docs/page.html",
        "docs/script.js",
        "docs/check.mjs",
        "docs/theme.scss",
        "tests/fixtures/table.tsv",
        "src/replica/dataset/fix.patch",
    ],
)
def test_web_pages_and_data_tables_are_scanned(tmp_path, name):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    target = tmp_path / name
    target.parent.mkdir(parents=True)
    target.write_text(f"see {CLUSTER_PATH}\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True)
    assert _check(tmp_path) == [
        f"{name}:1: developer-only reference: see {CLUSTER_PATH}"
    ]


@pytest.mark.parametrize(
    "name",
    [
        "reproduction/repetition/reading_time/r0123456789ab/cells.tsv",
        "indexXjson",
        "READMEamd",
    ],
)
def test_tracked_result_payloads_are_rejected_even_with_an_index(tmp_path, name):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    results = tmp_path / "src/replica/studies/example2026/results"
    results.mkdir(parents=True)
    (results / "index.json").write_text("{}")
    (results / "README.md").write_text("# Results\n")
    payload = results / name
    payload.parent.mkdir(parents=True, exist_ok=True)
    payload.write_text("value\n1\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True)
    assert _check(tmp_path) == [
        f"{payload.relative_to(tmp_path)}: result payloads belong on the results hub"
    ]
