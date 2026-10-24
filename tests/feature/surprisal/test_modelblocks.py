"""Tests for the ModelBlocks bridge.

Two layers:

1. :class:`ItemMeasuresTable` + :class:`ModelBlocksFeature`, pure
   file-format readers. Tested against synthetic ``.itemmeasures``
   fixtures so no ModelBlocks install is needed.
2. :class:`ModelBlocksRunner`, invokes ``make``. Tested against a
   tiny stub ``make`` shell script that mimics ModelBlocks's behavior
   (writes an ``.itemmeasures`` file). This pins the subprocess wiring
   without depending on the real ModelBlocks toolchain.
"""

import stat
from pathlib import Path

import numpy as np
import pytest

from replica.core import AnnotationLayer, Span, Stimulus
from replica.dataset.catalog.sentences import attach_sentences
from replica.feature import (
    ItemMeasuresTable,
    ModelBlocksFeature,
    ModelBlocksRunner,
    write_sentitems,
)
from tests._stimuli import word_stimulus

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _stim(text: str, sid: str, **metadata) -> Stimulus:
    """Build a Stimulus with a whitespace-tokenized ``words`` layer."""
    return word_stimulus(text, sid, metadata=metadata)


def _write_table(path: Path, header: list[str], rows: list[list[str]]) -> Path:
    lines = [" ".join(header)] + [" ".join(r) for r in rows]
    path.write_text("\n".join(lines) + "\n")
    return path


# ---------------------------------------------------------------------------
# Layer 1: ItemMeasuresTable parsing
# ---------------------------------------------------------------------------


def test_parses_minimal_table(tmp_path):
    p = _write_table(
        tmp_path / "t.itemmeasures",
        ["word", "sentid", "sentpos", "dlt"],
        [
            ["the", "1", "1", "0.0"],
            ["cat", "1", "2", "0.5"],
            ["sat", "1", "3", "1.2"],
        ],
    )
    t = ItemMeasuresTable.from_file(p)
    assert t.columns == ["word", "sentid", "sentpos", "dlt"]
    assert t.word_col == "word"
    assert t.sentid_col == "sentid"
    assert t.sentpos_col == "sentpos"
    assert t.docid_col is None
    assert t.subject_col is None
    assert t.has_docid is False
    assert len(t.rows) == 3
    assert t.lookup(1, 2) == {
        "word": "cat",
        "sentid": "1",
        "sentpos": "2",
        "dlt": "0.5",
    }
    assert t.word_at(1, 3) == "sat"


def test_all_trailing_underscore_aliases_resolve_in_one_file(tmp_path):
    # `.all-itemmeasures` files use trailing-underscore column names for
    # every key role. One file with all five exercises every alias path.
    p = _write_table(
        tmp_path / "t.itemmeasures",
        ["subject_", "word_", "sentid_", "sentpos_", "docid_", "totsurp_"],
        [["s1", "the", "1", "1", "D", "3.4"]],
    )
    t = ItemMeasuresTable.from_file(p)
    assert t.subject_col == "subject_"
    assert t.word_col == "word_"
    assert t.sentid_col == "sentid_"
    assert t.sentpos_col == "sentpos_"
    assert t.docid_col == "docid_"
    assert t.has_docid is True
    assert t.lookup(sentid=1, sentpos=1, docid="D")["totsurp_"] == "3.4"
    assert t.word_at(sentid=1, sentpos=1, docid="D") == "the"


def test_table_without_word_column_parses(tmp_path):
    p = _write_table(
        tmp_path / "t.itemmeasures",
        ["sentid", "sentpos", "dlt"],
        [["1", "1", "0.0"]],
    )
    t = ItemMeasuresTable.from_file(p)
    assert t.word_col is None
    assert t.word_at(1, 1) is None


def test_subject_level_lookup_returns_first_row_for_key(tmp_path):
    p = _write_table(
        tmp_path / "all.itemmeasures",
        ["subject", "word", "sentid", "sentpos", "wlen"],
        [
            ["s1", "the", "1", "1", "3"],
            ["s2", "the", "1", "1", "3"],
            ["s3", "the", "1", "1", "3"],
            ["s1", "cat", "1", "2", "3"],
            ["s2", "cat", "1", "2", "3"],
        ],
    )
    t = ItemMeasuresTable.from_file(p)
    assert len(t.rows) == 5  # all rows preserved
    assert t.word_at(1, 1) == "the"
    assert t.word_at(1, 2) == "cat"
    assert t.lookup(1, 1)["subject"] == "s1"  # first-row-wins


def test_verify_stimulus_level_passes_for_consistent_column(tmp_path):
    p = _write_table(
        tmp_path / "all.itemmeasures",
        ["subject", "word", "sentid", "sentpos", "wlen"],
        [
            ["s1", "the", "1", "1", "3"],
            ["s2", "the", "1", "1", "3"],
        ],
    )
    t = ItemMeasuresTable.from_file(p)
    t.verify_stimulus_level("wlen")  # does not raise


def test_verify_stimulus_level_raises_for_subject_varying_column(tmp_path):
    p = _write_table(
        tmp_path / "bad.itemmeasures",
        ["subject", "word", "sentid", "sentpos", "fdur"],
        [
            ["s1", "the", "1", "1", "200"],
            ["s2", "the", "1", "1", "250"],
        ],
    )
    t = ItemMeasuresTable.from_file(p)
    with pytest.raises(ValueError, match="not stimulus-level"):
        t.verify_stimulus_level("fdur")


def test_unknown_column_raises_keyerror_at_every_entry_point(tmp_path):
    # The "column does not exist" invariant surfaces at four APIs.
    # Each must reject with KeyError naming the bad column.
    p = _write_table(
        tmp_path / "t.itemmeasures",
        ["word", "sentid", "sentpos", "dlt"],
        [["a", "1", "1", "0"]],
    )
    t = ItemMeasuresTable.from_file(p)
    for op in (
        lambda: t.verify_stimulus_level("nope"),
        lambda: t.column("nope"),
        lambda: t.column_array("nope"),
        lambda: ModelBlocksFeature(t, column="nope"),
    ):
        with pytest.raises(KeyError, match="'nope'.*not in"):
            op()


def test_duplicate_key_without_subject_raises(tmp_path):
    p = _write_table(
        tmp_path / "dup.itemmeasures",
        ["word", "sentid", "sentpos", "dlt"],
        [
            ["the", "1", "1", "0.0"],
            ["the", "1", "1", "0.1"],
        ],
    )
    with pytest.raises(ValueError, match="duplicate key"):
        ItemMeasuresTable.from_file(p)


def test_missing_sentid_or_sentpos_raises(tmp_path):
    p = _write_table(
        tmp_path / "bad.itemmeasures",
        ["word", "sentid", "dlt"],
        [["the", "1", "0.0"]],
    )
    with pytest.raises(ValueError, match="sentid \\+ sentpos"):
        ItemMeasuresTable.from_file(p)


def test_column_count_mismatch_raises(tmp_path):
    p = tmp_path / "ragged.itemmeasures"
    p.write_text("word sentid sentpos dlt\nthe 1 1\n")
    with pytest.raises(ValueError, match="expected 4 columns"):
        ItemMeasuresTable.from_file(p)


def test_duplicate_column_name_in_header_raises(tmp_path):
    p = tmp_path / "dup.itemmeasures"
    p.write_text("word sentid sentpos dlt dlt\nthe 1 1 0 0\n")
    with pytest.raises(ValueError, match="duplicate column names"):
        ItemMeasuresTable.from_file(p)


def test_empty_file_raises(tmp_path):
    p = tmp_path / "empty.itemmeasures"
    p.write_text("")
    with pytest.raises(ValueError, match="empty"):
        ItemMeasuresTable.from_file(p)


def test_non_integer_sentid_raises(tmp_path):
    p = _write_table(
        tmp_path / "bad.itemmeasures",
        ["word", "sentid", "sentpos", "dlt"],
        [["a", "X", "1", "0"]],
    )
    with pytest.raises(ValueError, match="non-integer sentid/sentpos"):
        ItemMeasuresTable.from_file(p)


def test_skips_blank_lines(tmp_path):
    p = tmp_path / "blanks.itemmeasures"
    p.write_text("word sentid sentpos dlt\nthe 1 1 0.0\n\ncat 1 2 0.5\n\n")
    t = ItemMeasuresTable.from_file(p)
    assert len(t.rows) == 2


def test_column_array_returns_floats_with_nan(tmp_path):
    p = _write_table(
        tmp_path / "t.itemmeasures",
        ["word", "sentid", "sentpos", "dlt"],
        [
            ["a", "1", "1", "0.0"],
            ["b", "1", "2", "NA"],
            ["c", "1", "3", "1.5"],
            ["d", "1", "4", "nan"],
        ],
    )
    t = ItemMeasuresTable.from_file(p)
    arr = t.column_array("dlt")
    assert arr[0] == 0.0
    assert np.isnan(arr[1])
    assert arr[2] == 1.5
    assert np.isnan(arr[3])


def test_column_array_raises_on_non_numeric(tmp_path):
    p = _write_table(
        tmp_path / "t.itemmeasures",
        ["word", "sentid", "sentpos", "dlt"],
        [["a", "1", "1", "abc"]],
    )
    t = ItemMeasuresTable.from_file(p)
    with pytest.raises(ValueError, match="is not numeric"):
        t.column_array("dlt")


# ---------------------------------------------------------------------------
# Layer 1: docid (multi-doc) support
# ---------------------------------------------------------------------------


def test_docid_changes_key_tuple_and_disambiguates_rows(tmp_path):
    p = _write_table(
        tmp_path / "t.itemmeasures",
        ["docid", "word", "sentid", "sentpos", "dlt"],
        [
            ["d1", "a", "1", "1", "0.0"],
            ["d2", "x", "1", "1", "9.0"],
        ],
    )
    t = ItemMeasuresTable.from_file(p)
    assert t.lookup(1, 1, docid="d1")["dlt"] == "0.0"
    assert t.lookup(1, 1, docid="d2")["dlt"] == "9.0"


def test_docid_required_when_present(tmp_path):
    p = _write_table(
        tmp_path / "t.itemmeasures",
        ["docid", "word", "sentid", "sentpos", "dlt"],
        [["d1", "a", "1", "1", "0"]],
    )
    t = ItemMeasuresTable.from_file(p)
    with pytest.raises(ValueError, match="pass docid="):
        t.lookup(1, 1)


def test_docid_forbidden_when_absent(tmp_path):
    p = _write_table(
        tmp_path / "t.itemmeasures",
        ["word", "sentid", "sentpos", "dlt"],
        [["a", "1", "1", "0"]],
    )
    t = ItemMeasuresTable.from_file(p)
    with pytest.raises(ValueError, match="no docid column"):
        t.lookup(1, 1, docid="d1")


# ---------------------------------------------------------------------------
# Layer 1: ModelBlocksFeature
# ---------------------------------------------------------------------------


def test_feature_returns_column_values_in_word_order(tmp_path):
    p = _write_table(
        tmp_path / "t.itemmeasures",
        ["word", "sentid", "sentpos", "dlt"],
        [
            ["The", "7", "1", "0.0"],
            ["cat", "7", "2", "1.5"],
            ["sat", "7", "3", "2.0"],
        ],
    )
    feat = ModelBlocksFeature(ItemMeasuresTable.from_file(p), column="dlt")
    s = _stim("The cat sat", "s1", sentid=7)
    vals = feat(s, outputs=None)
    np.testing.assert_array_equal(vals, [0.0, 1.5, 2.0])
    assert feat.name == "mb_dlt"


def test_feature_emits_nan_for_NA_or_blank(tmp_path):
    p = _write_table(
        tmp_path / "t.itemmeasures",
        ["word", "sentid", "sentpos", "dlt"],
        [
            ["a", "1", "1", "0.0"],
            ["b", "1", "2", "NA"],
            ["c", "1", "3", "nan"],
        ],
    )
    feat = ModelBlocksFeature(ItemMeasuresTable.from_file(p), column="dlt")
    vals = feat(_stim("a b c", "s", sentid=1), outputs=None)
    assert vals[0] == 0.0
    assert np.isnan(vals[1])
    assert np.isnan(vals[2])


def test_feature_raises_when_stimulus_has_no_sentid(tmp_path):
    p = _write_table(
        tmp_path / "t.itemmeasures",
        ["word", "sentid", "sentpos", "dlt"],
        [["a", "1", "1", "0"]],
    )
    feat = ModelBlocksFeature(ItemMeasuresTable.from_file(p), column="dlt")
    s = Stimulus(text="a", stimulus_id="x")
    s.add_layer(AnnotationLayer("words", [Span(0, 1)], ["a"]))
    with pytest.raises(KeyError, match="'sentid'"):
        feat(s, outputs=None)


def test_feature_raises_on_non_int_sentid_metadata(tmp_path):
    p = _write_table(
        tmp_path / "t.itemmeasures",
        ["word", "sentid", "sentpos", "dlt"],
        [["a", "1", "1", "0"]],
    )
    feat = ModelBlocksFeature(ItemMeasuresTable.from_file(p), column="dlt")
    s = _stim("a", "x", sentid="not-a-number")
    with pytest.raises(ValueError, match="not int-coercible"):
        feat(s, outputs=None)


def test_feature_raises_on_missing_row(tmp_path):
    p = _write_table(
        tmp_path / "t.itemmeasures",
        ["word", "sentid", "sentpos", "dlt"],
        [["a", "1", "1", "0"]],
    )
    feat = ModelBlocksFeature(ItemMeasuresTable.from_file(p), column="dlt")
    s = _stim("a b", "s", sentid=1)
    with pytest.raises(KeyError, match="no row for"):
        feat(s, outputs=None)


def test_feature_sentpos_base_zero(tmp_path):
    p = _write_table(
        tmp_path / "t.itemmeasures",
        ["word", "sentid", "sentpos", "dlt"],
        [
            ["a", "1", "0", "9.0"],
            ["b", "1", "1", "8.0"],
        ],
    )
    feat = ModelBlocksFeature(
        ItemMeasuresTable.from_file(p), column="dlt", sentpos_base=0
    )
    vals = feat(_stim("a b", "s", sentid=1), outputs=None)
    np.testing.assert_array_equal(vals, [9.0, 8.0])


def test_feature_with_docid_in_metadata(tmp_path):
    p = _write_table(
        tmp_path / "t.itemmeasures",
        ["docid", "word", "sentid", "sentpos", "dlt"],
        [
            ["d1", "a", "1", "1", "0.5"],
            ["d2", "a", "1", "1", "9.5"],
        ],
    )
    feat = ModelBlocksFeature(ItemMeasuresTable.from_file(p), column="dlt")
    s1 = _stim("a", "s", sentid=1, docid="d1")
    s2 = _stim("a", "s", sentid=1, docid="d2")
    np.testing.assert_array_equal(feat(s1, outputs=None), [0.5])
    np.testing.assert_array_equal(feat(s2, outputs=None), [9.5])


def test_feature_raises_when_table_needs_docid_but_stimulus_lacks_it(tmp_path):
    p = _write_table(
        tmp_path / "t.itemmeasures",
        ["docid", "word", "sentid", "sentpos", "dlt"],
        [["d1", "a", "1", "1", "0"]],
    )
    feat = ModelBlocksFeature(ItemMeasuresTable.from_file(p), column="dlt")
    s = _stim("a", "s", sentid=1)
    with pytest.raises(KeyError, match="no 'docid' in metadata"):
        feat(s, outputs=None)


def test_feature_raises_on_non_numeric_column_value(tmp_path):
    p = _write_table(
        tmp_path / "t.itemmeasures",
        ["word", "sentid", "sentpos", "dlt"],
        [["a", "1", "1", "garbage"]],
    )
    feat = ModelBlocksFeature(ItemMeasuresTable.from_file(p), column="dlt")
    with pytest.raises(ValueError, match="non-numeric"):
        feat(_stim("a", "s", sentid=1), outputs=None)


# ---------------------------------------------------------------------------
# Helper: .sentitems writer
# ---------------------------------------------------------------------------


def test_write_sentitems_emits_one_article_per_stimulus(tmp_path):
    s1 = _stim("the cat sat", "s1", sentid=1)
    s2 = _stim("a dog ran", "s2", sentid=2)
    out = write_sentitems([s1, s2], tmp_path / "out.sentitems")
    text = out.read_text()
    assert text == "!ARTICLE\nthe cat sat\n!ARTICLE\na dog ran\n"


def test_write_sentitems_uses_sentence_layer_when_given(tmp_path):
    text = "I am here. You are there."
    # Word layer: 7 words
    words = ["I", "am", "here.", "You", "are", "there."]
    spans, off = [], 0
    for w in words:
        spans.append(Span(off, off + len(w)))
        off += len(w) + 1
    s = Stimulus(text=text, stimulus_id="s", metadata={"sentid": 1})
    s.add_layer(AnnotationLayer("words", spans, words))
    # Sentence layer: 2 sentences split on character boundaries
    sent_spans = [Span(0, 10), Span(11, len(text))]
    s.add_layer(AnnotationLayer("sentences", sent_spans, ["s1", "s2"]))

    out = write_sentitems([s], tmp_path / "ss.sentitems", sentences_layer="sentences")
    text_out = out.read_text()
    assert text_out == "!ARTICLE\nI am here.\nYou are there.\n"


def test_write_sentitems_groups_word_parallel_sentence_ids(tmp_path):
    # The layout attach_sentences builds: word spans valued by sentence id.
    s = _stim("The cat sat. It purred loudly.", "s")
    s.metadata["source_sentence_numbers"] = [1, 1, 1, 2, 2, 2]
    attach_sentences("potec", [s])
    out = write_sentitems([s], tmp_path / "ids.sentitems", sentences_layer="sentences")
    assert out.read_text() == "!ARTICLE\nThe cat sat.\nIt purred loudly.\n"


def test_write_sentitems_falls_back_when_sentence_layer_absent(tmp_path):
    s = _stim("just one line", "s", sentid=1)
    out = write_sentitems([s], tmp_path / "x.sentitems", sentences_layer="not_present")
    assert out.read_text() == "!ARTICLE\njust one line\n"


# ---------------------------------------------------------------------------
# Layer 2: ModelBlocksRunner against a stub `make`
# ---------------------------------------------------------------------------


def _fake_modelblocks(
    tmp_path: Path, stub_body: str | None = None
) -> tuple[Path, Path]:
    """Lay out a directory tree the runner accepts and a stub make in PATH.

    Returns (mb_dir, stub_path). The default stub writes a canned
    ``.itemmeasures`` file at the target argument's path.
    """
    mb_dir = tmp_path / "mb"
    (mb_dir / "resource-general").mkdir(parents=True)
    (mb_dir / "workspace" / "genmodel").mkdir(parents=True)

    bin_dir = tmp_path / "stub_bin"
    bin_dir.mkdir()
    stub = bin_dir / "fakemake"
    if stub_body is None:
        stub_body = (
            "#!/bin/sh\n"
            'target="$1"\n'
            'mkdir -p "$(dirname "$target")"\n'
            'cat > "$target" <<EOF\n'
            "word sentid sentpos dlt\n"
            "the 1 1 0.0\n"
            "cat 1 2 1.5\n"
            "sat 1 3 2.0\n"
            "EOF\n"
        )
    stub.write_text(stub_body)
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return mb_dir, stub


def test_runner_construction_validates_checkout_layout(tmp_path):
    # No resource-general subdir → reject.
    bad = tmp_path / "not-mb"
    bad.mkdir()
    with pytest.raises(FileNotFoundError, match="resource-general"):
        ModelBlocksRunner(bad)
    # resource-general exists but no workspace/ → reject.
    mb = tmp_path / "mb"
    (mb / "resource-general").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="workspace"):
        ModelBlocksRunner(mb)


def test_runner_feature_round_trips_through_stub(tmp_path):
    mb_dir, stub = _fake_modelblocks(tmp_path)
    runner = ModelBlocksRunner(mb_dir, make_executable=str(stub))
    feat = runner.feature("genmodel/foo.dlt.itemmeasures", column="dlt")
    s = _stim("the cat sat", "s", sentid=1)
    np.testing.assert_array_equal(feat(s, outputs=None), [0.0, 1.5, 2.0])
    assert feat.name == "mb_dlt"


def test_runner_make_propagates_subprocess_failure(tmp_path):
    mb_dir, stub = _fake_modelblocks(
        tmp_path,
        stub_body="#!/bin/sh\necho 'no recipe' 1>&2\nexit 2\n",
    )
    runner = ModelBlocksRunner(mb_dir, make_executable=str(stub))
    with pytest.raises(RuntimeError, match="failed \\(exit 2\\)"):
        runner.make("genmodel/foo.dlt.itemmeasures")


def test_runner_make_propagates_subprocess_stderr_tail(tmp_path):
    mb_dir, stub = _fake_modelblocks(
        tmp_path,
        stub_body="#!/bin/sh\necho 'BERKELEY-PARSER MISSING' 1>&2\nexit 1\n",
    )
    runner = ModelBlocksRunner(mb_dir, make_executable=str(stub))
    with pytest.raises(RuntimeError, match="BERKELEY-PARSER MISSING"):
        runner.make("genmodel/foo.dlt.itemmeasures")


def test_runner_make_raises_if_target_missing_after_success(tmp_path):
    mb_dir, stub = _fake_modelblocks(tmp_path, stub_body="#!/bin/sh\nexit 0\n")
    runner = ModelBlocksRunner(mb_dir, make_executable=str(stub))
    with pytest.raises(FileNotFoundError, match="reported success"):
        runner.make("genmodel/foo.dlt.itemmeasures")


def test_runner_make_wraps_timeout(tmp_path):
    # Stub sleeps longer than the timeout so subprocess.TimeoutExpired fires.
    mb_dir, stub = _fake_modelblocks(tmp_path, stub_body="#!/bin/sh\nsleep 5\nexit 0\n")
    runner = ModelBlocksRunner(mb_dir, make_executable=str(stub))
    with pytest.raises(RuntimeError, match="timed out after 0.5s"):
        runner.make("genmodel/foo.dlt.itemmeasures", timeout=0.5)


def test_runner_passes_env_to_subprocess(tmp_path):
    # Stub interpolates $REPLICA_TEST_VAR into the produced file.
    mb_dir, stub = _fake_modelblocks(
        tmp_path,
        stub_body=(
            "#!/bin/sh\n"
            'target="$1"\n'
            'mkdir -p "$(dirname "$target")"\n'
            'cat > "$target" <<EOF\n'
            "word sentid sentpos val\n"
            "tok 1 1 ${REPLICA_TEST_VAR}\n"
            "EOF\n"
        ),
    )
    runner = ModelBlocksRunner(
        mb_dir,
        make_executable=str(stub),
        env={"REPLICA_TEST_VAR": "42.0"},
    )
    t = runner.itemmeasures("genmodel/foo.val.itemmeasures")
    assert t.column("val") == ["42.0"]


def test_runner_make_executable_string_is_shell_split(tmp_path):
    # Stub asserts argv[1] == "--special-flag" and argv[2] is the target.
    mb_dir, stub = _fake_modelblocks(
        tmp_path,
        stub_body=(
            "#!/bin/sh\n"
            'if [ "$1" != "--special-flag" ]; then exit 99; fi\n'
            'target="$2"\n'
            'mkdir -p "$(dirname "$target")"\n'
            'cat > "$target" <<EOF\n'
            "word sentid sentpos dlt\n"
            "a 1 1 0\n"
            "EOF\n"
        ),
    )
    runner = ModelBlocksRunner(mb_dir, make_executable=f"{stub} --special-flag")
    t = runner.itemmeasures("genmodel/foo.itemmeasures")
    assert t.word_at(1, 1) == "a"


def test_runner_empty_make_executable_raises(tmp_path):
    mb_dir, _ = _fake_modelblocks(tmp_path)
    with pytest.raises(ValueError, match="empty argv"):
        ModelBlocksRunner(mb_dir, make_executable="")


def test_runner_reads_modelblocks_dir_from_env(tmp_path, monkeypatch):
    mb_dir, _ = _fake_modelblocks(tmp_path)
    monkeypatch.setenv("MODELBLOCKS_DIR", str(mb_dir))
    runner = ModelBlocksRunner()
    assert runner.modelblocks_dir == mb_dir.resolve()


def test_runner_raises_without_dir_or_env(monkeypatch):
    monkeypatch.delenv("MODELBLOCKS_DIR", raising=False)
    with pytest.raises(ValueError, match="MODELBLOCKS_DIR"):
        ModelBlocksRunner()


def test_runner_capture_false_still_produces_output(tmp_path, capfd):
    # Stub writes a recognisable banner to stdout; with capture=False the
    # subprocess inherits the test process's stdout so capfd sees it.
    mb_dir, stub = _fake_modelblocks(
        tmp_path,
        stub_body=(
            "#!/bin/sh\n"
            'target="$1"\n'
            "echo REPLICA_BANNER\n"
            'mkdir -p "$(dirname "$target")"\n'
            'cat > "$target" <<EOF\n'
            "word sentid sentpos dlt\n"
            "a 1 1 0\n"
            "EOF\n"
        ),
    )
    runner = ModelBlocksRunner(mb_dir, make_executable=str(stub))
    out = runner.make("genmodel/foo.itemmeasures", capture=False)
    assert out.exists()
    captured = capfd.readouterr()
    assert "REPLICA_BANNER" in captured.out


# ---------------------------------------------------------------------------
# Real-output fixture: a hand-built sample that mimics ModelBlocks's actual
# `.itemmeasures` shape (subject_/word_/sentid_/sentpos_ + a couple of
# predictor columns). Lives inline to keep the test self-contained.
# ---------------------------------------------------------------------------


REALISTIC_ALL_ITEMMEASURES = """subject_ word_ sentid_ sentpos_ wlen_ totsurp_ fdur_
s1 The 1 1 3 4.50 250
s1 cat 1 2 3 9.12 305
s1 sat 1 3 3 6.71 280
s2 The 1 1 3 4.50 240
s2 cat 1 2 3 9.12 290
s2 sat 1 3 3 6.71 275
"""


def test_realistic_all_itemmeasures_fixture_parses_and_round_trips(tmp_path):
    p = tmp_path / "x.all-itemmeasures"
    p.write_text(REALISTIC_ALL_ITEMMEASURES)
    t = ItemMeasuresTable.from_file(p)
    assert t.subject_col == "subject_"
    assert t.word_col == "word_"
    assert len(t.rows) == 6  # long format, all rows preserved

    s = _stim("The cat sat", "s", sentid=1)
    # Stimulus-level columns import cleanly with the safety check on.
    surp = ModelBlocksFeature(
        t, column="totsurp_", name="mb_totsurp", verify_stimulus_level=True
    )
    np.testing.assert_array_equal(surp(s, outputs=None), [4.50, 9.12, 6.71])

    wlen = ModelBlocksFeature(
        t, column="wlen_", name="mb_wlen", verify_stimulus_level=True
    )
    np.testing.assert_array_equal(wlen(s, outputs=None), [3.0, 3.0, 3.0])

    # The subject-level column trips the safety check.
    with pytest.raises(ValueError, match="not stimulus-level"):
        ModelBlocksFeature(t, column="fdur_", verify_stimulus_level=True)


# ---------------------------------------------------------------------------
# Real-output fixture: produced by ModelBlocks's actual
# `resource-rt/scripts/lineitems2itemmeasures.py`. Regenerate via
# `tests/feature/fixtures/modelblocks/regenerate.py`. The fixture is
# checked in so this test runs without any ModelBlocks install.
# ---------------------------------------------------------------------------


FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "modelblocks"


def test_real_modelblocks_fixture_has_expected_shape():
    p = FIXTURES / "sample.itemmeasures"
    assert p.exists(), (
        f"missing real-output fixture at {p}; "
        f"run tests/feature/fixtures/modelblocks/regenerate.py"
    )
    # The exact bytes produced by ModelBlocks's converter. If this
    # equality ever breaks, the upstream format changed; rerun
    # regenerate.py and review the diff before updating.
    assert p.read_text() == (
        "word sentid sentpos discid discpos corpus\n"
        "The 0 1 0 1 sample\n"
        "cat 0 2 0 2 sample\n"
        "sat 0 3 0 3 sample\n"
        "on 0 4 0 4 sample\n"
        "the 0 5 0 5 sample\n"
        "mat. 0 6 0 6 sample\n"
        "A 1 1 0 7 sample\n"
        "dog 1 2 0 8 sample\n"
        "ran 1 3 0 9 sample\n"
        "fast. 1 4 0 10 sample\n"
        "She 2 1 1 1 sample\n"
        "reads 2 2 1 2 sample\n"
        "books 2 3 1 3 sample\n"
        "quietly. 2 4 1 4 sample\n"
    )


def test_real_fixture_parses_with_discid_as_docid_alias():
    t = ItemMeasuresTable.from_file(FIXTURES / "sample.itemmeasures")
    assert t.columns == ["word", "sentid", "sentpos", "discid", "discpos", "corpus"]
    assert t.word_col == "word"
    assert t.sentid_col == "sentid"
    assert t.sentpos_col == "sentpos"
    # ModelBlocks's converter writes `discid`; exposed as the
    # document-grouping column under the canonical name `docid`.
    assert t.docid_col == "discid"
    assert t.has_docid is True
    # 14 words total across three sentences in two articles.
    assert len(t.rows) == 14


def test_real_fixture_lookup_uses_zero_based_sentid_and_one_based_sentpos():
    t = ItemMeasuresTable.from_file(FIXTURES / "sample.itemmeasures")
    # `lineitems2itemmeasures.py`: "ids start from 0, poses start from 1"
    row = t.lookup(sentid=0, sentpos=1, docid="0")
    assert row is not None
    assert row["word"] == "The"
    # Second article's only sentence starts at sentid=2 (sentid does not
    # reset across articles in this converter), discid="1".
    row = t.lookup(sentid=2, sentpos=1, docid="1")
    assert row is not None
    assert row["word"] == "She"


def test_real_fixture_feature_round_trip_second_article():
    t = ItemMeasuresTable.from_file(FIXTURES / "sample.itemmeasures")
    feat = ModelBlocksFeature(t, column="discpos")
    # discid="1", sentid=2 (continues across articles), 4 words.
    s = _stim("She reads books quietly.", "s2", sentid=2, docid="1")
    vals = feat(s, outputs=None)
    np.testing.assert_array_equal(vals, [1, 2, 3, 4])


def test_real_fixture_word_check_passes_in_strict_mode():
    t = ItemMeasuresTable.from_file(FIXTURES / "sample.itemmeasures")
    feat = ModelBlocksFeature(t, column="sentpos", strict_word_check=True)
    s = _stim("A dog ran fast.", "s", sentid=1, docid="0")
    # Should not raise: words match the fixture row-for-row.
    vals = feat(s, outputs=None)
    np.testing.assert_array_equal(vals, [1, 2, 3, 4])


def test_real_fixture_word_check_catches_tokenization_drift():
    t = ItemMeasuresTable.from_file(FIXTURES / "sample.itemmeasures")
    feat = ModelBlocksFeature(t, column="sentpos", strict_word_check=True)
    # PTB-style split off the period, the fixture has "mat." as one
    # token, so this should trip the check.
    s = _stim("The cat sat on the mat", "s", sentid=0, docid="0")
    with pytest.raises(ValueError, match="word mismatch"):
        feat(s, outputs=None)


# ---------------------------------------------------------------------------
# Discid alias unit tests (the canonical converter emits `discid`, not `docid`)
# ---------------------------------------------------------------------------


def test_discid_resolves_as_docid_alias_on_synthetic_input(tmp_path):
    p = _write_table(
        tmp_path / "t.itemmeasures",
        ["word", "sentid", "sentpos", "discid", "dlt"],
        [
            ["a", "0", "1", "0", "0.5"],
            ["a", "0", "1", "1", "9.5"],
        ],
    )
    t = ItemMeasuresTable.from_file(p)
    assert t.docid_col == "discid"
    assert t.has_docid is True
    assert t.lookup(sentid=0, sentpos=1, docid="0")["dlt"] == "0.5"
    assert t.lookup(sentid=0, sentpos=1, docid="1")["dlt"] == "9.5"


def test_docid_wins_over_discid_when_both_present(tmp_path):
    p = _write_table(
        tmp_path / "t.itemmeasures",
        ["word", "sentid", "sentpos", "docid", "discid", "dlt"],
        [["a", "0", "1", "A", "0", "0.5"]],
    )
    t = ItemMeasuresTable.from_file(p)
    # `docid` precedes `discid` in the alias tuple, so `docid` wins.
    assert t.docid_col == "docid"
    assert t.lookup(sentid=0, sentpos=1, docid="A")["dlt"] == "0.5"


# ---------------------------------------------------------------------------
# Multi-sentence stimulus support (sentence_layer)
# ---------------------------------------------------------------------------


def _multi_sentence_stim(text: str, sid: str, sentences: list[dict]) -> Stimulus:
    """Build a stimulus with `words` and `sentences` layers.

    `sentences` is a list of dicts ``{"text": "...", "sentid": int,
    "docid": str | None}`` giving each sentence's surface form and
    metadata. The sentence span is computed by locating the text inside
    `text`. The words layer is a whitespace tokenization of each
    sentence.
    """
    word_spans, word_strs = [], []
    sent_spans, sent_values = [], []
    cursor = 0
    for sent in sentences:
        s_text = sent["text"]
        start = text.index(s_text, cursor)
        end = start + len(s_text)
        sent_spans.append(Span(start, end))
        sent_meta = {"sentid": sent["sentid"]}
        if sent.get("docid") is not None:
            sent_meta["docid"] = sent["docid"]
        sent_values.append(sent_meta)
        off = start
        for w in s_text.split():
            idx = text.index(w, off)
            word_spans.append(Span(idx, idx + len(w)))
            word_strs.append(w)
            off = idx + len(w)
        cursor = end
    stim = Stimulus(text=text, stimulus_id=sid, metadata={})
    stim.add_layer(AnnotationLayer("words", word_spans, word_strs))
    stim.add_layer(AnnotationLayer("sentences", sent_spans, sent_values))
    return stim


def test_feature_with_sentence_layer_aligns_across_multiple_sentences():
    # The real fixture's article 0 has two sentences (sentid 0, sentid 1).
    # Build a single multi-sentence stimulus that spans both and import the
    # `sentpos` column: per-sentence position should restart at 1.
    t = ItemMeasuresTable.from_file(FIXTURES / "sample.itemmeasures")
    text = "The cat sat on the mat. A dog ran fast."
    stim = _multi_sentence_stim(
        text,
        "story_0",
        [
            {"text": "The cat sat on the mat.", "sentid": 0, "docid": "0"},
            {"text": "A dog ran fast.", "sentid": 1, "docid": "0"},
        ],
    )
    feat = ModelBlocksFeature(t, column="sentpos", sentence_layer="sentences")
    vals = feat(stim, outputs=None)
    np.testing.assert_array_equal(vals, [1, 2, 3, 4, 5, 6, 1, 2, 3, 4])


def test_feature_with_sentence_layer_word_check_succeeds(tmp_path):
    p = _write_table(
        tmp_path / "t.itemmeasures",
        ["word", "sentid", "sentpos", "dlt"],
        [
            ["The", "0", "1", "0.0"],
            ["cat", "0", "2", "1.0"],
            ["A", "1", "1", "2.0"],
            ["dog", "1", "2", "3.0"],
        ],
    )
    text = "The cat. A dog."
    stim = _multi_sentence_stim(
        text,
        "s",
        [
            {"text": "The cat.", "sentid": 0},
            {"text": "A dog.", "sentid": 1},
        ],
    )
    # The fixture words are "The", "cat", "A", "dog" but the stimulus
    # words are "The", "cat.", "A", "dog.", sentence-final punctuation
    # is attached. With strict checking off, lookup happens by position
    # and finds the right rows.
    feat = ModelBlocksFeature(
        ItemMeasuresTable.from_file(p),
        column="dlt",
        sentence_layer="sentences",
    )
    vals = feat(stim, outputs=None)
    np.testing.assert_array_equal(vals, [0.0, 1.0, 2.0, 3.0])


def test_feature_sentence_layer_missing_on_stimulus_raises(tmp_path):
    p = _write_table(
        tmp_path / "t.itemmeasures",
        ["word", "sentid", "sentpos", "dlt"],
        [["a", "0", "1", "0"]],
    )
    feat = ModelBlocksFeature(
        ItemMeasuresTable.from_file(p),
        column="dlt",
        sentence_layer="sentences",
    )
    s = _stim("a", "s", sentid=0)  # no sentences layer
    with pytest.raises(KeyError, match="sentence_layer='sentences' is not present"):
        feat(s, outputs=None)


@pytest.mark.parametrize(
    "value,error,match",
    [
        # A plain string where the layer holds one mapping per sentence.
        ("a sentence", TypeError, "not a mapping"),
        ({"docid": "0"}, KeyError, "no 'sentid' key"),
        ({"sentid": "x"}, ValueError, "not int-coercible"),
    ],
    ids=["value_must_be_mapping", "value_missing_sentid", "non_int_sentid"],
)
def test_feature_sentence_layer_value_errors(tmp_path, value, error, match):
    p = _write_table(
        tmp_path / "t.itemmeasures",
        ["word", "sentid", "sentpos", "dlt"],
        [["a", "0", "1", "0"]],
    )
    feat = ModelBlocksFeature(
        ItemMeasuresTable.from_file(p),
        column="dlt",
        sentence_layer="sentences",
    )
    s = Stimulus(text="a", stimulus_id="s")
    s.add_layer(AnnotationLayer("words", [Span(0, 1)], ["a"]))
    s.add_layer(AnnotationLayer("sentences", [Span(0, 1)], [value]))
    with pytest.raises(error, match=match):
        feat(s, outputs=None)


def test_feature_sentence_layer_refuses_word_parallel_sentence_ids(tmp_path):
    p = _write_table(
        tmp_path / "t.itemmeasures",
        ["word", "sentid", "sentpos", "dlt"],
        [["a", "1", "1", "0"]],
    )
    feat = ModelBlocksFeature(
        ItemMeasuresTable.from_file(p),
        column="dlt",
        sentence_layer="sentences",
    )
    s = _stim("a", "s", source_sentence_numbers=[1])
    attach_sentences("potec", [s])
    with pytest.raises(TypeError, match="one span per sentence"):
        feat(s, outputs=None)


def test_feature_sentence_layer_word_outside_all_sentences_raises(tmp_path):
    p = _write_table(
        tmp_path / "t.itemmeasures",
        ["word", "sentid", "sentpos", "dlt"],
        [["a", "0", "1", "0"]],
    )
    feat = ModelBlocksFeature(
        ItemMeasuresTable.from_file(p),
        column="dlt",
        sentence_layer="sentences",
    )
    s = Stimulus(text="a b", stimulus_id="s")
    s.add_layer(AnnotationLayer("words", [Span(0, 1), Span(2, 3)], ["a", "b"]))
    # Sentence covers only the first word; "b" at position 2..3 is outside.
    s.add_layer(AnnotationLayer("sentences", [Span(0, 1)], [{"sentid": 0}]))
    with pytest.raises(KeyError, match="not contained in any span"):
        feat(s, outputs=None)


def test_feature_sentence_layer_coerces_int_docid_to_string(tmp_path):
    p = _write_table(
        tmp_path / "t.itemmeasures",
        ["word", "sentid", "sentpos", "docid", "dlt"],
        [["a", "0", "1", "0", "9.0"]],
    )
    feat = ModelBlocksFeature(
        ItemMeasuresTable.from_file(p),
        column="dlt",
        sentence_layer="sentences",
    )
    s = Stimulus(text="a", stimulus_id="s")
    s.add_layer(AnnotationLayer("words", [Span(0, 1)], ["a"]))
    # docid is an integer here; lookup must coerce to str to match the
    # table's string-typed docid column.
    s.add_layer(AnnotationLayer("sentences", [Span(0, 1)], [{"sentid": 0, "docid": 0}]))
    np.testing.assert_array_equal(feat(s, outputs=None), [9.0])


def test_feature_sentence_layer_requires_docid_when_table_does(tmp_path):
    p = _write_table(
        tmp_path / "t.itemmeasures",
        ["word", "sentid", "sentpos", "docid", "dlt"],
        [["a", "0", "1", "D1", "0"]],
    )
    feat = ModelBlocksFeature(
        ItemMeasuresTable.from_file(p),
        column="dlt",
        sentence_layer="sentences",
    )
    s = Stimulus(text="a", stimulus_id="s")
    s.add_layer(AnnotationLayer("words", [Span(0, 1)], ["a"]))
    s.add_layer(AnnotationLayer("sentences", [Span(0, 1)], [{"sentid": 0}]))
    with pytest.raises(KeyError, match="entry 0 has no 'docid'"):
        feat(s, outputs=None)


# ---------------------------------------------------------------------------
# Empty-stimulus short-circuit
# ---------------------------------------------------------------------------


def test_feature_empty_stimulus_returns_empty_array_without_reading_metadata(tmp_path):
    # An empty stimulus must not require sentid in metadata: there's
    # nothing to look up, so the right answer is an empty array.
    p = _write_table(
        tmp_path / "t.itemmeasures",
        ["word", "sentid", "sentpos", "dlt"],
        [["a", "0", "1", "0"]],
    )
    feat = ModelBlocksFeature(ItemMeasuresTable.from_file(p), column="dlt")
    empty = Stimulus(text="", stimulus_id="empty")
    empty.add_layer(AnnotationLayer("words", [], []))
    vals = feat(empty, outputs=None)
    assert vals.shape == (0,)
    assert vals.dtype == np.float64


# ---------------------------------------------------------------------------
# Runner: sentence_layer is forwarded through
# ---------------------------------------------------------------------------


def test_runner_feature_forwards_sentence_layer(tmp_path):
    mb_dir, stub = _fake_modelblocks(tmp_path)
    runner = ModelBlocksRunner(mb_dir, make_executable=str(stub))
    feat = runner.feature(
        "genmodel/foo.dlt.itemmeasures",
        column="dlt",
        sentence_layer="sentences",
    )
    # The default stub yields rows for sentid=1, sentpos=1..3 ("the cat sat").
    text = "the cat sat"
    stim = _multi_sentence_stim(
        text,
        "s",
        [{"text": "the cat sat", "sentid": 1}],
    )
    np.testing.assert_array_equal(feat(stim, outputs=None), [0.0, 1.5, 2.0])
