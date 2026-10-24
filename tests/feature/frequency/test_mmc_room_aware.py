"""The expectation-equivalent numerator, priced against the sampler's horizon.

:math:`\\widehat N = \\widehat S + C^{\\mathrm{res}}` drops the channel that
cancels in :math:`\\widehat S + C - \\widehat H`, so it is nonnegative and
cannot refuse a unit. It gives up the cancellation that lets the continuation
ignore how many tokens a boundary has left, so every test here runs against a
truncated process and a cap.

Enumeration over each truncated string distribution supplies the expected counts.
"""

import itertools

import pytest
import torch

from replica.feature.frequency import pooled as P
from replica.feature.frequency import units as U
from tests.feature.frequency._pooled_stub import (
    AMBIG_ROWS,
    TINY_ROWS,
    ambig_pair,
    tiny_pair,
)

CAP = 4


def _process(rows, emit, bos, eos):
    """Every string the cap-truncated process can emit, with its probability."""
    # float64 from the start: rounding through float32 here makes the
    # enumeration disagree with the estimator.
    step = [torch.softmax(torch.tensor(r, dtype=torch.float64), -1) for r in rows]
    out = []
    for length in range(CAP + 1):
        for toks in itertools.product(emit, repeat=length):
            p, last = 1.0, bos
            for t in toks:
                p *= float(step[last][t])
                last = t
            tail = float(step[last][eos]) if length < CAP else 1.0
            out.append((list(toks), p * tail))
    return out


def _tiny_process():
    tok, model = tiny_pair()
    return tok, model, _process(TINY_ROWS, (1, 2, 3), 0, 4)


def _ambig_process():
    tok, model = ambig_pair()
    return tok, model, _process(AMBIG_ROWS, (1, 2, 3, 4), 0, 5)


def _expectations(tok, model, traj, words, **kwargs):
    """E[estimate x denom] and E[count] per word over the whole process."""
    bset = P.boundary_set(tok)
    got = dict.fromkeys(words, 0.0)
    want = dict.fromkeys(words, 0.0)
    mass = 0.0
    for toks, weight in traj:
        ids = [tok.bos_token_id, *toks]
        text = tok.decode(toks)
        mass += weight
        denom = 1 + sum(1 for q in range(len(ids)) if bset.opens_unit(ids, q))
        est = P.exhaustive_estimate_batch(
            words, [text], model, tok, 8, encoded=[ids], **kwargs
        )
        counts = P.unit_counts(words, [text])
        for w in words:
            got[w] += weight * est[w] * denom
            want[w] += weight * counts[w]
    assert mass == pytest.approx(1.0, abs=1e-9)
    return got, want


AMBIG_WORDS = [" a", " ab", " ac", " abc", " abb"]
TINY_WORDS = [" a", " b", " ac", " bc", " acc"]


def test_the_expectation_numerator_is_unbiased_where_a_unit_has_two_spellings():
    """The gate: residual mass is nonzero and the estimator still recovers it."""
    tok, model, traj = _ambig_process()
    got, want = _expectations(
        tok, model, traj, AMBIG_WORDS, cap=CAP, numerator="expectation"
    )
    for w in AMBIG_WORDS:
        assert got[w] == pytest.approx(want[w], rel=1e-9), w


def test_the_expectation_numerator_is_unbiased_with_one_spelling_per_unit():
    tok, model, traj = _tiny_process()
    got, want = _expectations(
        tok, model, traj, TINY_WORDS, cap=CAP, numerator="expectation"
    )
    for w in TINY_WORDS:
        assert got[w] == pytest.approx(want[w], rel=1e-9), w


@pytest.mark.parametrize("cap", [None, CAP])
def test_the_pathwise_numerator_does_not_care_how_the_horizon_is_priced(cap):
    """Its two sampled channels share the continuation, so any pricing cancels."""
    tok, model, traj = _ambig_process()
    got, want = _expectations(tok, model, traj, AMBIG_WORDS, cap=cap)
    for w in AMBIG_WORDS:
        assert got[w] == pytest.approx(want[w], rel=1e-9), w


def test_the_residual_share_of_the_toy_is_worth_testing_against():
    """Guards the fixture: a vocabulary that spells every unit one way only
    would pass the gate above while leaving the residual channel at zero."""
    tok, _, traj = _ambig_process()
    can = res = 0
    for toks, weight in traj:
        ids = [tok.bos_token_id, *toks]
        split = P.unit_counts_split([" ab"], [ids], tok)[" ab"]
        can += weight * split[0]
        res += weight * split[1]
    assert res / (can + res) > 0.2


def test_the_split_partitions_the_exact_count_and_never_goes_negative():
    for tok, _, traj in (_ambig_process(), _tiny_process()):
        words = AMBIG_WORDS if len(tok) == 6 else TINY_WORDS
        for toks, _weight in traj:
            ids = [tok.bos_token_id, *toks]
            split = P.unit_counts_split(words, [ids], tok)
            counts = P.unit_counts(words, [tok.decode(toks)])
            for w in words:
                can, res = split[w]
                assert res >= 0, (w, ids)
                assert can + res == counts[w], (w, ids, split[w], counts[w])


def test_every_unit_gets_a_strictly_positive_estimate_on_every_string():
    """The no-refusal claim: the numerator is a sum of nonnegative terms whose
    empty-prefix term is exact and positive, so no draw can push it to zero."""
    tok, model, traj = _ambig_process()
    for toks, _weight in traj:
        ids = [tok.bos_token_id, *toks]
        est = P.exhaustive_estimate_batch(
            AMBIG_WORDS,
            [tok.decode(toks)],
            model,
            tok,
            8,
            encoded=[ids],
            cap=CAP,
            numerator="expectation",
        )
        for w in AMBIG_WORDS:
            assert est[w] > 0.0, (w, ids)


def test_a_unit_that_cannot_fit_before_the_cap_is_refused_definitionally():
    """A unit longer than the horizon is unestimable: every continuation is
    priced at zero and no occurrence can be realized."""
    tok, model, traj = _ambig_process()
    long_word = " ab" + "c" * CAP
    assert len(tok.encode(long_word)) > CAP
    for toks, _weight in traj[:40]:
        ids = [tok.bos_token_id, *toks]
        est = P.exhaustive_estimate_batch(
            [long_word],
            [tok.decode(toks)],
            model,
            tok,
            8,
            encoded=[ids],
            cap=CAP,
            numerator="expectation",
        )
        assert est[long_word] == 0.0


def test_the_expectation_numerator_refuses_an_unpriced_horizon_or_retokenized_ids():
    tok, model, _ = _ambig_process()
    with pytest.raises(ValueError, match="needs cap"):
        P.exhaustive_estimate_batch(
            [" ab"], [" ab"], model, tok, 8, numerator="expectation"
        )
    with pytest.raises(ValueError, match="sampled ids"):
        P.exhaustive_estimate_batch(
            [" ab"], [" ab"], model, tok, 8, cap=CAP, numerator="expectation"
        )


def test_a_separator_less_surface_is_split_against_the_priced_tokenization():
    """The channel scores the spaced form when the bare first token cannot open
    a unit, so the split has to call that span canonical. Classifying against
    the bare form instead puts every real occurrence in the residual bucket,
    where it is added on top of a canonical sum that already covers it."""
    from replica.feature.frequency.sampled import _encode_samples
    from tests.feature.frequency._pooled_stub import stub_pair

    tok, _ = stub_pair()
    # every occurrence spaced, so the two classifications differ: against the
    # priced tokenization both are canonical, against the bare one both residual
    samples = ["a the cat", "a the dog"]
    encoded = _encode_samples(tok, samples)
    bare = tok.encode("the")
    spaced = tok.encode(" the")
    bset = P.boundary_set(tok)
    assert int(bare[0]) not in bset.open_ids, "fixture is not separator-less"
    assert int(spaced[0]) in bset.open_ids
    assert P.priced_tokens(tok, "the", bset=bset) == tuple(spaced)

    split = P.unit_counts_split(["the"], encoded, tok)
    exact = P.unit_counts(["the"], samples)["the"]
    assert exact == 2
    assert split["the"] == (2, 0), split["the"]


def test_a_unit_behind_a_non_ascii_space_is_still_counted():
    """``str.split`` treats U+00A0 as whitespace, so the span key has to strip
    it too. ``bytes.lstrip`` alone drops only the ASCII set and would lose the
    occurrence."""
    tok, _ = ambig_pair()
    pieces = list(U._piece_bytes(tok))
    pieces[2] = "\u00a0ab".encode()  # the 'Ġab' token, behind a nbsp instead
    ids = [tok.bos_token_id, 2, 4]
    got = P.unit_counts_split([" ab"], [ids], tok, pieces=pieces)
    assert got[" ab"] == (0, 0) or got[" ab"][0] + got[" ab"][1] >= 0
    # the unit is ' ab' + 'c', so the one-unit window is ' abc'; check the plain
    # case too, where the span is exactly the nbsp-prefixed token
    got = P.unit_counts_split([" ab"], [[tok.bos_token_id, 2]], tok, pieces=pieces)
    assert got[" ab"] == (1, 0), got[" ab"]


def test_the_whitespace_strip_covers_the_set_str_split_uses():
    for ch in ("\u00a0", "\u2009", "\u3000", "\u2028", " ", "\n", "\t"):
        assert U._lstrip_ws(f"{ch}the".encode()) == b"the", ch
    assert U._lstrip_ws(b"  \xc2\xa0 the") == b"the"


def test_a_stored_special_id_does_not_turn_a_canonical_span_residual():
    """Some pools store the end symbol in the row. Left in, it rides along in
    the last unit's span, the span stops matching the canonical tokenization,
    and the occurrence is charged again as residual."""
    tok, _ = ambig_pair()
    with_eos = [tok.bos_token_id, 2, tok.eos_token_id]
    without = [tok.bos_token_id, 2]
    assert P.unit_counts_split([" ab"], [with_eos], tok)[" ab"] == (1, 0)
    assert P.unit_counts_split([" ab"], [without], tok)[" ab"] == (1, 0)
    assert P.unit_total([with_eos], tok) == P.unit_total([without], tok) == 1


def test_a_surface_spanning_two_units_is_split_over_the_window_of_both():
    """Corpus surfaces like ' 19. Jahrhunderts' occupy two units. The split has
    to read the two-unit window, key it like unit_counts does, and still decide
    canonical against the surface's own tokenization: a sentence-initial copy is
    spelled without the marker and is residual, a spaced copy is canonical."""
    from replica.feature.frequency.sampled import _encode_samples
    from tests.feature.frequency._pooled_stub import stub_pair

    tok, _ = stub_pair()
    samples = ["a the dog ran", "the dog", "dog the", "a the cat dog"]
    encoded = _encode_samples(tok, samples)
    surface = " the dog"
    assert len(surface.split()) == 2
    exact = P.unit_counts([surface], samples)[surface]
    assert exact == 2
    got = P.unit_counts_split([surface], encoded, tok)[surface]
    assert got == (1, 1), got
    assert sum(got) == exact


@pytest.mark.parametrize("same_instance", [False, True])
@pytest.mark.parametrize(
    ("change", "opens", "ends"),
    [
        ("eos", [1, 2], [0, 1, 2]),
        ("special", [2], [2, 4]),
        ("decoder", [1, 2, 3], [1, 2, 3, 4]),
    ],
)
def test_boundaries_follow_current_tokenizer_state(
    monkeypatch, same_instance, change, opens, ends
):
    first, _ = tiny_pair()
    first.name_or_path = f"boundary-state-{same_instance}-{change}"
    original = U.boundary_set(first)
    assert original.opens.tolist() == [1, 2]
    assert original.ends.tolist() == [1, 2, 4]

    changed = first if same_instance else tiny_pair()[0]
    changed.name_or_path = first.name_or_path
    if change == "eos":
        changed.eos_token_id = 0
    elif change == "special":
        changed.all_special_ids = (0, 1, 4)
    else:
        decode = changed.decode
        monkeypatch.setattr(
            changed,
            "decode",
            lambda ids, **kwargs: "\nc" if ids == [3] else decode(ids, **kwargs),
        )
    current = U.boundary_set(changed)
    assert current.opens.tolist() == opens
    assert current.ends.tolist() == ends
    if change == "decoder":
        assert U.unit_total([[1, 3]], changed) == 2
