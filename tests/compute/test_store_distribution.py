"""The distribution is stored only for a cell that asks for it."""

from dataclasses import replace

from replica.compute import extract_keys
from replica.compute.methods import ComputeContext


def _context(**overrides):
    base = ComputeContext("gpt2", 5, 64, 10, 1.0, 0, True, "en", None)
    return replace(base, **overrides)


def test_first_token_entropy_keeps_the_scalar_only_by_default():
    assert extract_keys("first_token_entropy", _context()) == ["token_entropy"]


def test_store_distribution_adds_the_next_token_distribution():
    keys = extract_keys("first_token_entropy", _context(store_distribution=True))
    assert keys == ["token_entropy", "next_log_probs"]


def test_the_flag_is_part_of_the_identity():
    from replica.compute.artifacts import artifact_identity
    from replica.core import AnnotationLayer, Span, Stimulus

    stim = Stimulus(
        "ab cd",
        {"words": AnnotationLayer("words", [Span(0, 2), Span(3, 5)], ["ab", "cd"])},
        "s",
    )
    plain = artifact_identity("first_token_entropy", _context(), [stim], {})
    stored = artifact_identity(
        "first_token_entropy", _context(store_distribution=True), [stim], {}
    )
    assert plain["options"]["store_distribution"] is False
    assert stored != plain
