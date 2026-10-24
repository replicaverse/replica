"""Model input preflight and actionable optional-dependency failures."""

import builtins

import numpy as np
import pytest

from replica.analysis.models.support.r import _r_context


@pytest.mark.parametrize(
    "failure",
    [
        ModuleNotFoundError("No module named 'rpy2'"),
        ImportError("R shared library failed"),
    ],
)
def test_r_installation_failure_preserves_diagnostics(monkeypatch, failure):
    original = builtins.__import__

    def unavailable(name, *args, **kwargs):
        if name == "rpy2.robjects":
            raise failure
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", unavailable)
    with pytest.raises(ImportError, match=r"pip install -e '\.\[r\]'") as error:
        with _r_context():
            pytest.fail("The missing backend must not run")
    assert error.value.__cause__ is failure


@pytest.mark.parametrize(
    "backend",
    [
        "rust",
        "rust_loglik",
        "gam",
        "gam_stats",
        "gam_r",
        "gam_loglik",
        "marginal",
        "r_lm",
        "r_lm_stats",
        "r_lmer",
        "r_lmer_stats",
        "r_lmer_loglik",
    ],
)
def test_real_backend_inputs_fail_before_optional_dependencies(backend, monkeypatch):
    from replica.analysis.models.additive import gam
    from replica.analysis.models.fixed_effects import r as r_linear
    from replica.analysis.models.mixed_effects import lmer, lmer_rs, marginal_lme

    def forbidden(*args, **kwargs):
        pytest.fail("Invalid real-model input reached the optional backend")

    monkeypatch.setattr(lmer_rs, "import_lme_python", forbidden)
    monkeypatch.setattr(gam, "run_r_script", forbidden)
    for module in (gam, r_linear, lmer, marginal_lme):
        monkeypatch.setattr(module, "_r_context", forbidden)
    models = {
        "rust": lmer_rs.RustLmerModel().fit_predict,
        "rust_loglik": lmer_rs.RustLmerModel().fit_loglik,
        "gam": gam.RGamModel(use_bam=True).fit_predict,
        "gam_stats": gam.RGamModel(use_bam=True).fit_predict_with_stats,
        "gam_r": gam.RGamModel().fit_predict,
        "gam_loglik": gam.RGamModel().fit_loglik,
        "marginal": marginal_lme.MarginalLMEModel().fit_score,
        "r_lm": r_linear.RLinearModel().fit_predict,
        "r_lm_stats": r_linear.RLinearModel().fit_predict_with_stats,
        "r_lmer": lmer.RLmerModel().fit_predict,
        "r_lmer_stats": lmer.RLmerModel().fit_predict_with_stats,
        "r_lmer_loglik": lmer.RLmerModel().fit_loglik,
    }
    y = np.arange(12.0)
    train = np.column_stack([y, np.repeat(np.arange(3), 4), np.tile(np.arange(4), 3)])
    originals = [y, train] if backend.endswith("loglik") else [y, train, train[:2]]
    if backend == "marginal":
        originals.append(y[:2])
    for position in range(len(originals)):
        for dtype in (complex, object):
            arguments = list(originals)
            arguments[position] = arguments[position].astype(dtype)
            arguments[position].flat[0] = np.complex128(
                arguments[position].flat[0] + 2j
            )
            with pytest.raises(ValueError, match="real-valued"):
                models[backend](
                    *arguments, **({"is_full": False} if backend == "marginal" else {})
                )


def test_random_effect_design_requires_real_values():
    from replica.analysis.models.mixed_effects.options import LmerOptions

    re_design = np.arange(6.0)[:, None]
    assert LmerOptions(re_design=re_design).re_design is re_design
    for dtype in (complex, object):
        invalid = re_design.astype(dtype)
        invalid.flat[0] = np.complex128(invalid.flat[0] + 2j)
        with pytest.raises(ValueError, match="re_design.*real-valued"):
            LmerOptions(re_design=invalid)
