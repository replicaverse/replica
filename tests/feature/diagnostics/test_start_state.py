"""Start-token diagnostics use supplied texts and classify generated samples."""

import pytest

from replica.feature.diagnostics import start_state


@pytest.mark.parametrize(
    "text,expected",
    [
        ("The committee", "latin"),
        ("Городской совет", "cyrillic"),
        ("הוועדה", "hebrew"),
        ("اللجنة", "arabic"),
        ("위원회", "cjk"),
        ("委员会", "cjk"),
        ("Επιτροπή", "other"),
        ("123!", "none"),
    ],
)
def test_sample_script(text, expected):
    assert start_state._script(text) == expected


def test_special_token_ranking_requires_nonempty_mass():
    with pytest.raises(ValueError, match="no special-token mass"):
        start_state._rank_special_mass({})
    with pytest.raises(ValueError, match="no special-token mass"):
        start_state._rank_special_mass({"<s>": []})
    ranked = start_state._rank_special_mass({"<s>": [0.1, 0.3], "<e>": [0.5]})
    assert ranked == [(0.5, "<e>"), (0.2, "<s>")]


def test_cli_uses_only_the_supplied_texts(tmp_path, monkeypatch, capsys):
    path = tmp_path / "texts.txt"
    path.write_text("The committee\n\n委员会\n  \n", encoding="utf-8")
    received = []

    def report(model, texts, device):
        received.append((model, texts, device))
        return "comparison"

    monkeypatch.setattr(start_state, "start_state_report", report)
    assert start_state.main(["model", "--texts", str(path)]) == 0
    assert received == [("model", ["The committee", "委员会"], "cpu")]
    assert capsys.readouterr().out == "comparison\n"


@pytest.mark.parametrize("empty_file", [False, True])
def test_cli_refuses_missing_or_blank_texts_before_loading(
    empty_file, tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(
        start_state, "start_state_report", lambda *a, **kw: pytest.fail("report called")
    )
    argv = ["model"]
    if empty_file:
        path = tmp_path / "texts.txt"
        path.write_text("\n  \n", encoding="utf-8")
        argv += ["--texts", str(path)]
    with pytest.raises(SystemExit) as exc:
        start_state.main(argv)
    assert exc.value.code == 2
    assert "--texts" in capsys.readouterr().err
