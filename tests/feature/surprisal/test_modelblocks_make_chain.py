"""End-to-end integration test for the ModelBlocks bridge.

The format-and-contract tests in ``test_modelblocks.py`` exercise the
reader and Runner against synthetic files and a stub ``make``.
This file complements them by actually invoking the real ModelBlocks
``make`` chain on a real checkout and checking that the Runner +
reader handle the byte-for-byte output.

Preconditions (``tests._require.require`` skips a test whose preconditions
are missing, naming the missing one):

* A modelblocks/modelblocks-release checkout, resolved by the registry
  (or pointed at by ``$MODELBLOCKS_DIR``) with a bootstrapped
  ``workspace/`` directory.
* GNU Make >= 4.0 available somewhere. ModelBlocks's
  ``define VAR :=`` syntax was introduced in Make 3.82, and macOS
  ships 3.81; ``brew install make`` provides 4.x at
  ``/opt/homebrew/opt/make/libexec/gnubin/make``.

The parse-time prerequisite chain for ``%ucl.itemmeasures`` reads the
checkout's UCL data-directory config pointer even though the rule body only
needs ``.sentitems``. Each test points it at an empty stand-in directory and
restores the checkout's own file afterwards.
"""

import os
import shutil
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest

from replica.external import repo_path
from replica.feature import ModelBlocksRunner
from tests._require import require

DEFAULT_MB = repo_path("modelblocks-release", fetch=False)


def _modelblocks_dir() -> Path:
    env = os.environ.get("MODELBLOCKS_DIR")
    return Path(env).resolve() if env else DEFAULT_MB.resolve()


def _find_gmake() -> str | None:
    """Return path to a GNU Make >= 4.0, or None."""
    candidates = [
        os.environ.get("GNUMAKE"),
        shutil.which("gmake"),
        "/opt/homebrew/opt/make/libexec/gnubin/make",
        "/usr/local/opt/make/libexec/gnubin/make",
        shutil.which("make"),
    ]
    for c in candidates:
        if not c or not Path(c).exists():
            continue
        try:
            out = subprocess.run(
                [c, "--version"], capture_output=True, text=True, check=True
            ).stdout
        except (subprocess.CalledProcessError, OSError):
            continue
        first = out.splitlines()[0] if out else ""
        # Parse "GNU Make 4.4.1" → major version.
        for tok in first.split():
            try:
                major = int(tok.split(".")[0])
            except ValueError:
                continue
            if "Make" in first and major >= 4:
                return c
    return None


def _preconditions_met() -> bool:
    mb = _modelblocks_dir()
    if not (mb / "resource-general").is_dir():
        return False
    if not (mb / "workspace").is_dir():
        return False
    return _find_gmake() is not None


needs_modelblocks = require(
    _preconditions_met,
    (
        "Real ModelBlocks make-chain integration test requires (1) a "
        "checkout of modelblocks/modelblocks-release (or "
        "$MODELBLOCKS_DIR) with `workspace/` bootstrapped via `make` in "
        "the checkout root, and (2) GNU Make >= 4.0 available "
        "(`brew install make`)."
    ),
)


@contextmanager
def _fake_ucl_config(mb: Path, fake: Path):
    """Point the checkout's UCL config at *fake* for the block, then restore it.

    Make evaluates the prerequisites of the ``.sentitems`` rule at parse
    time, so the pointer has to name an existing directory holding the empty
    data files that rule references. The original file is written back, or
    removed when the checkout had none, even when the block raises.
    """
    cfg = mb / "config" / "user-ucl-directory.txt"
    original = cfg.read_bytes() if cfg.exists() else None
    (fake / "eye_movement").mkdir(parents=True, exist_ok=True)
    for name in ("stimuli.txt", "eyetracking.RT.txt", "selfpacedreading.RT.txt"):
        (fake / "eye_movement" / name).touch()
    cfg.write_text(f"{fake}\n")
    try:
        yield
    finally:
        if original is None:
            cfg.unlink(missing_ok=True)
        else:
            cfg.write_bytes(original)


def test_fake_ucl_config_restores_the_checkout_pointer(tmp_path):
    checkout = tmp_path / "modelblocks"
    (checkout / "config").mkdir(parents=True)
    pointer = checkout / "config" / "user-ucl-directory.txt"
    fake = tmp_path / "fake_ucl"

    with _fake_ucl_config(checkout, fake):
        assert pointer.read_text() == f"{fake}\n"
        assert (fake / "eye_movement" / "stimuli.txt").is_file()
    assert not pointer.exists()

    pointer.write_bytes(b"/data/ucl\n")
    with pytest.raises(RuntimeError), _fake_ucl_config(checkout, fake):
        assert pointer.read_text() == f"{fake}\n"
        raise RuntimeError("make failed")
    assert pointer.read_bytes() == b"/data/ucl\n"


@pytest.mark.needs_data
@needs_modelblocks
def test_real_make_chain_produces_itemmeasures_via_runner(tmp_path):
    mb = _modelblocks_dir()
    gmake = _find_gmake()
    assert gmake is not None  # gated by `needs_modelblocks`

    workspace = mb / "workspace"
    sentitems = workspace / "replica_smoke_ucl.sentitems"
    target_rel = "replica_smoke_ucl.itemmeasures"
    sentitems.write_text(
        "!ARTICLE\nThe cat sat on the mat.\nA dog ran fast.\n"
        "!ARTICLE\nShe reads books quietly.\n"
    )
    target_path = workspace / target_rel
    if target_path.exists():
        target_path.unlink()

    try:
        runner = ModelBlocksRunner(mb, make_executable=gmake)
        with _fake_ucl_config(mb, tmp_path / "fake_ucl"):
            table = runner.itemmeasures(target_rel, timeout=120)
    finally:
        sentitems.unlink(missing_ok=True)
        target_path.unlink(missing_ok=True)

    # Format invariants from the real `lineitems2itemmeasures.py`:
    assert table.columns == ["word", "sentid", "sentpos", "discid", "discpos", "corpus"]
    assert table.docid_col == "discid"
    assert table.has_docid is True

    # Three sentences across two articles → 14 word rows.
    assert len(table.rows) == 14

    # ModelBlocks's indexing convention, verified against real output:
    # `sentid` is 0-based and does not reset across articles;
    # `sentpos` is 1-based within a sentence;
    # `discid` is 0-based per article.
    first = table.lookup(sentid=0, sentpos=1, docid="0")
    assert first is not None and first["word"] == "The"

    second_article_first = table.lookup(sentid=2, sentpos=1, docid="1")
    assert second_article_first is not None
    assert second_article_first["word"] == "She"

    # The `sentpos` column, read as a numeric Feature column, must
    # restart at 1 per sentence, pinning that the reader sees
    # exactly the values the converter wrote.
    sentpos_col = table.column_array("sentpos")
    np.testing.assert_array_equal(
        sentpos_col,
        [1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 1, 2, 3, 4],
    )


def _have_transformers_and_torch() -> bool:
    try:
        import torch  # noqa: F401 (availability probe)
        import transformers  # noqa: F401 (availability probe)
    except ImportError:
        return False
    return True


needs_modelblocks_and_hf = require(
    lambda: _preconditions_met() and _have_transformers_and_torch(),
    (
        "Real GPT-2 surprisal integration test requires the make-chain "
        "preconditions PLUS importable `torch` and `transformers`. "
        "Install with `pip install torch transformers`."
    ),
)


@pytest.mark.needs_model
@pytest.mark.needs_data
@needs_modelblocks_and_hf
def test_real_make_chain_extracts_gpt2_surprisal_via_runner(tmp_path):
    """Drive the real GPT-2 surprisal pipeline end-to-end through the
    Runner: `make` invokes ModelBlocks's actual `get_llm_surprisal.py`
    on a downloaded GPT-2 model, `roll_toks.py` aggregates subword
    tokens, and the paired-file loader joins the predictor's positional
    `[word, totsurp, rolled]` output onto the gold `.itemmeasures`.
    """
    mb = _modelblocks_dir()
    gmake = _find_gmake()
    assert gmake is not None

    workspace = mb / "workspace"
    # Filename must end with a corpus suffix that has a make rule; `ucl`
    # is used because its rule body needs only the .sentitems provided here.
    base = "replica_gpt2_ucl"
    gold_target = f"{base}.itemmeasures"
    pred_target = f"{base}.gpt2.itemmeasures"
    sentitems = workspace / f"{base}.sentitems"
    sentitems.write_text("!ARTICLE\nThe cat sat on the mat.\nA dog ran fast.\n")
    artifacts = [
        workspace / gold_target,
        workspace / pred_target,
        workspace / f"{base}.gpt2.tokmeasures",
        workspace / f"{base}.gpt2.tokmeasures.log",
    ]
    for a in artifacts:
        a.unlink(missing_ok=True)

    # The HF script invokes `python3`; route it to the same interpreter
    # `transformers` was just imported from, to avoid accidentally
    # picking up a different env.
    py_dir = str(Path(sys.executable).parent)
    extra_path = py_dir + ":" + os.environ.get("PATH", "")

    try:
        runner = ModelBlocksRunner(
            mb,
            make_executable=gmake,
            env={"PATH": extra_path},
        )
        # The first call downloads GPT-2 (~500 MB) and runs inference;
        # generous timeout.
        with _fake_ucl_config(mb, tmp_path / "fake_ucl"):
            table = runner.itemmeasures(gold_target, pred_target, timeout=1800)
    finally:
        sentitems.unlink(missing_ok=True)
        for a in artifacts:
            a.unlink(missing_ok=True)

    # Joined shape: gold's key columns + the predictor's totsurp + rolled.
    assert "totsurp" in table.columns
    assert table.sentid_col == "sentid"
    assert len(table.rows) == 10  # two sentences, 6 + 4 words

    totsurp = table.column_array("totsurp")
    assert totsurp.shape == (10,)
    # GPT-2 surprisal is non-negative (bits) and finite.
    assert np.all(np.isfinite(totsurp))
    assert np.all(totsurp >= 0)

    # Sanity-anchor a few values against the GPT-2 distribution. The
    # function-word "the" after "sat on" is highly predictable; "cat"
    # at sentence start is not.
    assert totsurp[4] < 3.0, f"'the' (pos 5) surprisal {totsurp[4]} unexpectedly high"
    assert totsurp[1] > 8.0, f"'cat' (pos 2) surprisal {totsurp[1]} unexpectedly low"
    assert totsurp[3] < 5.0, f"'on' (pos 4) surprisal {totsurp[3]} unexpectedly high"
