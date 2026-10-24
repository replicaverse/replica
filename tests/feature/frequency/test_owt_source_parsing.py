"""Only the tab-less header line is a comment, so hashtag vocabulary survives."""

import math

from replica.feature.frequency.sources import OpenWebTextSource


def _write_table(tmp_path, lines):
    path = tmp_path / "owt.tsv"
    path.write_text("\n".join(lines) + "\n")
    return str(path)


def test_hashtag_vocabulary_survives_the_header(tmp_path):
    path = _write_table(
        tmp_path,
        ["# total_tokens=100", "the\t80", "#MeToo\t15", "#\t5"],
    )
    source = OpenWebTextSource(path)
    assert source.probability("the") == 0.8
    assert source.probability("#MeToo") == 0.15
    assert source.probability("#") == 0.05


def test_header_sets_the_total_rather_than_the_sum(tmp_path):
    path = _write_table(tmp_path, ["# total_tokens=200", "the\t80"])
    assert OpenWebTextSource(path).probability("the") == 0.4


def test_unknown_word_is_nan_without_smoothing(tmp_path):
    path = _write_table(tmp_path, ["# total_tokens=100", "the\t80"])
    source = OpenWebTextSource(path)
    assert math.isnan(source.probability("The"))


def test_lookup_is_case_sensitive(tmp_path):
    path = _write_table(
        tmp_path,
        ["# total_tokens=100", "The\t10", "the\t80"],
    )
    source = OpenWebTextSource(path)
    assert source.case_sensitive
    assert source.probability("The") == 0.1
    assert source.probability("the") == 0.8
