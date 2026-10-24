"""Extension and failure behavior at the study-to-sweep boundary."""

import csv
from dataclasses import replace
from types import ModuleType

import pytest

from replica.multiverse.run.replication.contracts import ReplicationSpec
from replica.multiverse.run.replication.runner import run_sweep
from replica.multiverse.spec import DatasetCell


def _module():
    module = ModuleType("replica.studies.new_study.replication")
    module.OUTPUT = "new_study.tsv"
    module.COLUMNS = ["dataset", "lm", "value"]
    module.CORPORA = [DatasetCell("new_corpus", "spr_rt", "spr")]
    module.LMS = ["new_model"]
    module.run_cell = lambda ds, lm, tags, **kwargs: [
        {"dataset": ds.name, "lm": lm, "value": 42}
    ]
    return module


def _spec(module=None):
    return ReplicationSpec.from_module(
        module or _module(), default_corpora=(), default_lms=("gpt2",)
    )


@pytest.mark.parametrize(
    "attribute,value,message",
    [
        ("OUTPUT", "../outside.tsv", "TSV basename"),
        ("COLUMNS", "dataset", "sequence of names"),
        ("COLUMNS", ["dataset", "dataset"], "duplicate"),
        ("COLUMNS", ["typo"], "include dataset"),
        ("CORPORA", ["a_corpus"], "DatasetCell"),
        ("LMS", "gpt2", "sequence of names"),
        ("run_cell", lambda ds: [], "invalid signature"),
        ("run_cell", None, "callable"),
    ],
)
def test_bad_extension_declarations_fail_at_adaptation(attribute, value, message):
    module = _module()
    setattr(module, attribute, value)
    with pytest.raises(ValueError, match=message):
        _spec(module)


def test_declaration_lists_are_snapshotted():
    module = _module()
    spec = _spec(module)
    module.COLUMNS.append("another_value")
    module.CORPORA.clear()
    module.LMS.clear()
    assert spec.columns == ("dataset", "lm", "value")
    assert len(spec.corpora) == len(spec.lms) == 1


def test_failure_is_distinct_from_skip_and_other_cells_still_finish(tmp_path):
    def run_cell(ds, lm, tags, **kwargs):
        if lm == "broken":
            raise RuntimeError("bad fit")
        return [] if lm == "inapplicable" else [{"dataset": ds.name, "lm": lm}]

    spec = replace(
        _spec(), lms=("broken", "inapplicable", "working"), run_cell=run_cell
    )
    result = run_sweep({"new": spec}, cdir=tmp_path, tags=set(), output_dir=tmp_path)
    assert result.n_rows == result.n_skipped == len(result.failures) == 1
    failure = result.failures[0]
    assert (failure.study, failure.cell.name, failure.lm) == (
        "new",
        "new_corpus",
        "broken",
    )
    assert "bad fit" in failure.error
    assert "working" in (tmp_path / spec.output).read_text()


def test_invalid_second_row_does_not_leave_a_partial_cell(tmp_path):
    def run_cell(ds, lm, tags, **kwargs):
        return [{"dataset": ds.name}, {"dataset": ds.name, "typo": 123}]

    spec = replace(_spec(), run_cell=run_cell)
    result = run_sweep({"new": spec}, cdir=tmp_path, tags=None, output_dir=tmp_path)
    assert result.n_rows == 0
    assert "undeclared columns" in result.failures[0].error
    assert (tmp_path / spec.output).read_text() == "dataset\tlm\tvalue\n"


def test_output_schema_conflicts_are_detected_before_computing_or_writing(tmp_path):
    def must_not_run(*args, **kwargs):
        pytest.fail("schema preflight must precede computation")

    first = replace(_spec(), output="first.tsv", run_cell=must_not_run)
    second = replace(first, output="second.tsv")
    existing = tmp_path / "second.tsv"
    existing.write_text("incompatible\n")
    with pytest.raises(ValueError, match="existing header"):
        run_sweep(
            {"first": first, "second": second},
            cdir=tmp_path,
            tags=None,
            output_dir=tmp_path,
        )
    assert not (tmp_path / "first.tsv").exists()
    assert existing.read_text() == "incompatible\n"


def test_shared_destinations_must_have_the_same_schema(tmp_path):
    first = _spec()
    second = replace(first, columns=("dataset", "other"))
    with pytest.raises(ValueError, match="different output schemas"):
        run_sweep(
            {"first": first, "second": second},
            cdir=tmp_path,
            tags=None,
            output_dir=tmp_path,
        )
    assert not list(tmp_path.iterdir())


def test_tsv_values_round_trip_delimiters_and_newlines(tmp_path):
    value = 'a\tvalue\nwith "quotes"'
    spec = replace(
        _spec(),
        run_cell=lambda ds, *args, **kwargs: [{"dataset": ds.name, "value": value}],
    )
    run_sweep({"new": spec}, cdir=tmp_path, tags=None, output_dir=tmp_path)
    with (tmp_path / spec.output).open(newline="") as handle:
        assert list(csv.DictReader(handle, delimiter="\t"))[0]["value"] == value


def _contrast_module(monkeypatch, tmp_path, *, summarize=None):
    """A contrast study whose fit crashes for one focal and succeeds for another."""
    from replica.multiverse.run.replication import contrast

    aliases = {"custom": "lab/custom-model"}

    def resolve(*args, **kwargs):
        assert kwargs["aliases"] == aliases
        return tmp_path / "cell.pkl"

    monkeypatch.setattr(contrast, "resolve_predictor_source", resolve)
    monkeypatch.setattr(contrast, "_prepared_dataset", lambda ds, aggregate: object())
    monkeypatch.setattr(contrast, "evaluate", lambda dataset, spec, config: [spec])
    logged = ("delta_ll", "pvalue", "pvalue_per_datapoint", "n_obs")
    monkeypatch.setattr(
        contrast,
        "row_from_analysis",
        lambda ds, key, stage, analysis, dataset, lm: {
            "dataset": ds.name,
            "lm": lm,
            "value": analysis,
            **dict.fromkeys(logged, 1),
        },
    )

    def build_spec(model, focal, ds, *, dataset):
        if focal.key == "custom_entropy":
            raise RuntimeError("R backend crashed")
        return focal.key

    from replica.multiverse.run.features import Predictor

    recipe = contrast.ContrastRecipe(
        "stage",
        ("custom_entropy", "surprisal"),
        build_spec,
        lambda: None,
        build_model=lambda *a, **k: object(),
        model_aliases=aliases,
        summarize=summarize,
        predictors={
            "custom_entropy": Predictor(
                "custom_entropy", "synthetic_entropy", "entropy", "synthetic"
            ),
            "surprisal": Predictor(
                "surprisal", "surprisal", "log_p_observed", "synthetic"
            ),
        },
    )
    module = _module()
    module.COLUMNS = [*module.COLUMNS, *logged]
    module.run_cell = lambda ds, lm, tags, *, select, cdir: contrast.run_contrast_cells(
        recipe, ds, lm, cdir=cdir, select=select
    )
    return module


def test_failed_contrast_fit_is_a_failure_that_keeps_sibling_rows(
    tmp_path, monkeypatch
):
    spec = _spec(_contrast_module(monkeypatch, tmp_path))
    result = run_sweep({"new": spec}, cdir=tmp_path, tags=None, output_dir=tmp_path)
    assert (result.n_rows, result.n_skipped, len(result.failures)) == (1, 0, 1)
    assert "custom_entropy: RuntimeError: R backend crashed" in result.failures[0].error
    with (tmp_path / spec.output).open() as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert [row["value"] for row in rows] == ["surprisal"]


def test_contrast_recipe_uses_the_callers_summary(tmp_path, monkeypatch):
    seen = []

    def summarize(ds, key, stage, analysis, dataset, *, lm):
        seen.append((ds.name, key, stage, analysis, lm))
        return {
            "dataset": ds.name,
            "lm": lm,
            "value": "caller summary",
            "delta_ll": 2.0,
            "pvalue": 0.25,
            "pvalue_per_datapoint": float("nan"),
            "n_obs": 4,
        }

    class Summary:
        __call__ = staticmethod(summarize)

        def __bool__(self):
            return False

    spec = _spec(_contrast_module(monkeypatch, tmp_path, summarize=Summary()))
    result = run_sweep({"new": spec}, cdir=tmp_path, tags=None, output_dir=tmp_path)
    assert result.n_rows == 1
    assert seen == [("new_corpus", "surprisal", "stage", "surprisal", "new_model")]
    with (tmp_path / spec.output).open() as handle:
        assert (
            list(csv.DictReader(handle, delimiter="\t"))[0]["value"] == "caller summary"
        )
