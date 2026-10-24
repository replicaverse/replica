"""Invalid likelihood comparisons must not be turned into significance tests."""

import pytest

from replica.analysis.crossed.inference.contrasts import nested_test


def model(ll):
    return {
        "nobs": 30,
        "loglik": ll,
        "optimizer_code": 0,
        "variances": [{"grp": "reader", "var1": "focal", "vcov": 1.0}],
    }


def test_nested_test_uses_fixed_rank_and_rejects_different_designs():
    result = nested_test(model(-100), model(-95), ["control"], ["control", "focal"])
    assert result["df"] == 1 and result["statistic"] == 10
    assert result["p_value"] == pytest.approx(0.001565402258002549)
    with pytest.raises(ValueError, match="not nested"):
        nested_test(model(-100), model(-95), ["other"], ["focal"])
    with pytest.raises(ValueError, match="worse"):
        nested_test(model(-100), model(-101), [], ["focal"])
    different = model(-95)
    different["variances"][0]["var1"] = "(Intercept)"
    with pytest.raises(ValueError, match="random designs"):
        nested_test(model(-100), different, [], ["focal"])
    assert nested_test(model(-100), model(-100), [], [])["status"] == "unidentifiable"
