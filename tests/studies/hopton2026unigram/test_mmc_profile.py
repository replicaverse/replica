"""The study's frozen language partition and its compatibility contract."""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from replica.feature.frequency import pooled as P
from replica.studies.hopton2026unigram.mmc import profile as S
from tests.feature.frequency._pooled_stub import (
    _MORE_SAMPLES,
    _SAMPLES,
    _STUB_WORDS,
    _counts,
    _take_store,
    stub_selection,
)

_FIXTURES = Path(__file__).parent / "fixtures"


# --- T16, T17: labels and language codes -----------------------------------


def test_stratum_labels_reproduce_the_recorded_fixture():
    fixture = json.loads(
        (_FIXTURES / "mmc_stratum_prefixes.json").read_text(encoding="utf-8")
    )
    for case in fixture:
        assert S.stratum_label(case["prefix"]) == case["stratum"], case["prefix"]


def test_the_stratum_vocabulary_is_frozen():
    assert S.N_STRATA == len(S.STRATA) == 21
    assert len(set(S.STRATA)) == S.N_STRATA
    assert S.STRATA == S.SCRIPT_CLASSES + S.LATIN_CLASSES


def test_the_meco_language_maps_are_total_and_unambiguous():
    assert set(S.MECO_HOME_STRATUM) == {f"meco_{c}" for c in S.MECO_ISO639_1}
    assert S.MECO_ISO639_1["du"] != S.MECO_ISO639_1["de"]
    assert S.MECO_ISO639_1["du"] == "nl"
    for stratum in S.MECO_HOME_STRATUM.values():
        assert stratum in S.STRATA
    from replica.studies.hopton2026unigram import LANGUAGES

    codes = {c if c != "ge" else "de" for c in LANGUAGES}
    assert codes <= set(S.MECO_ISO639_1)


def _stub_selection(shards, **kwargs):
    return stub_selection(shards, stratification=S.STRATIFICATION, **kwargs)


def _allocation(**kwargs):
    return S.allocation_spec(
        homes={w: ("meco_en",) for w in _STUB_WORDS}, replicates=2, **kwargs
    )


#: Pooled estimates at four budgets over one pinned selection.
_LADDER = {
    6: {
        " the": 0.16924587247484593,
        " dog": 0.06779083108896997,
        " cat": 0.050846530853708785,
        " world": 0.03389830501202487,
        " a": 0.04394574538674817,
        " zz": 0.00022187241943449968,
    },
    8: {
        " the": 0.169109426865718,
        " dog": 0.06779084587639972,
        " cat": 0.05084592611948509,
        " world": 0.03389830510960037,
        " a": 0.042218790996022,
        " zz": 0.0002289457597568367,
    },
    32: {
        " the": 0.16903038283806474,
        " dog": 0.0677914513839392,
        " cat": 0.05084596072059581,
        " world": 0.033898306906894805,
        " a": 0.04368194500642361,
        " zz": 0.0003679159231417962,
    },
    400: {
        " the": 0.16903735343496343,
        " dog": 0.06779137516158905,
        " cat": 0.05084595927283376,
        " world": 0.03389830687317581,
        " a": 0.043447448892010035,
        " zz": 0.00036334448691343534,
    },
}


@pytest.mark.statistical
def test_the_selectivity_ladder_reproduces_its_recorded_values():
    """Recorded per-rung values, and error that falls as the budget grows."""
    pool = [_SAMPLES, _MORE_SAMPLES]
    selection, tok, model = _stub_selection(pool, k_store=64)
    exact = P.exhaustive_estimate_batch(
        _STUB_WORDS,
        _SAMPLES + _MORE_SAMPLES,
        model,
        tok,
        3,
    )
    error = {}
    for k_eval, recorded in _LADDER.items():
        allocation = _allocation(k_eval=k_eval, k_floor=2, k_home=4)
        got = P.pps_estimate_batch(
            _STUB_WORDS,
            selection,
            model,
            tok,
            allocation=allocation,
            k_eval=k_eval,
            batch_size=3,
            unit_counts=_counts(),
            take_store=_take_store(),
        )
        for word, value in recorded.items():
            # The stub's logits are float32, so a recorded value reproduces only
            # to float32 precision; the reduction order moves it below that.
            assert got[word].p_hat == pytest.approx(value, rel=1e-6), (k_eval, word)
        error[k_eval] = float(
            np.median(
                [
                    abs(math.log2(got[w].p_hat) - math.log2(exact[w]))
                    for w in _STUB_WORDS
                ]
            )
        )
    # The paths agree to float32 precision at full coverage.
    assert error[400] < 1e-9
    # A single draw is not monotone in the budget: the numerator is a difference
    # of two sums, so one realization can land closer by luck. What falls with
    # the budget is the error averaged over pool seeds.
    averaged = {}
    for budget in (6, 32):
        runs = []
        for seed in range(8):
            sel, tk, mdl = _stub_selection(pool, k_store=64, rng_seed=seed)
            ref = P.exhaustive_estimate_batch(
                _STUB_WORDS, _SAMPLES + _MORE_SAMPLES, mdl, tk, 3
            )
            drawn = P.pps_estimate_batch(
                _STUB_WORDS,
                sel,
                mdl,
                tk,
                allocation=_allocation(k_eval=budget, k_floor=2, k_home=4),
                k_eval=budget,
                unit_counts=_counts(),
                take_store=_take_store(),
            )
            runs.append(
                max(
                    abs(math.log2(drawn[w].p_hat) - math.log2(ref[w]))
                    for w in _STUB_WORDS
                )
            )
        averaged[budget] = sum(runs) / len(runs)
    assert averaged[32] < averaged[6]


# --- T11: refusals ---------------------------------------------------------


def test_legacy_plan_digest_is_checked_before_adapting_to_explicit_weights(tmp_path):
    from replica.studies.hopton2026unigram.mmc import plan

    # Produced by AllocationSpec on main at c727e97c5, before this refactor.
    old_sha = "eebe514193fac7a625ad2e5101cb39ae08348d9082c9b57f84b8640573879f21"
    params = dict(k_eval=4096, k_floor=64, k_home=2048, replicates=8, alloc_sha=old_sha)
    path = tmp_path / "allocation.npz"
    np.savez(
        path,
        params=np.array(json.dumps(params)),
        first_ids=np.array([1]),
        mt_first_ids=np.array([1]),
        k_store=np.ones((1, 21, 8), dtype=int),
        k_shard=np.ones((1, 21, 8), dtype=int),
    )
    homes = tmp_path / "homes.json"
    homes.write_text(json.dumps({"w": ["meco_en"]}))
    spec, _, restored = plan._load_allocation(path, homes, ["w"])
    assert dict(spec.home_weights["w"]) == {"lat_en": 1.0, "lat_other": 0.5}
    assert restored["alloc_sha"] == old_sha
    homes.write_text(json.dumps({"w": ["meco_he"]}))
    with pytest.raises(SystemExit, match="recorded digest"):
        plan._load_allocation(path, homes, ["w"])
    with pytest.raises(ValueError, match="unknown MECO"):
        S.allocation_spec(homes={"w": ("not-a-meco-dataset",)})
