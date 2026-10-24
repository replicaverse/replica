"""Values Hopton et al. (2026a) report, for the tests to anchor on.

Kept apart from the computation so a test failure names which published number
moved. Held-out delta log-likelihood is reported as mean with a 95% bootstrap
CI and a significance mark (one-sided paired sign-flip permutation test, B=5000,
Holm-corrected); ``sig`` is the printed mark (``***`` p<0.001, ``**`` p<0.01,
``*`` p<0.05, ``ns`` otherwise). Delta-LL is comparable within a granularity and
(at the word level) between the marginalized and unmarginalized model of one
speech LM, but not across LMs or granularities.

All numbers are transcribed from the manuscript tables:
``tab:delta_ll_word`` / ``tab:delta_ll_phone`` / ``tab:delta_ll_frame`` (the
offset sweep), ``tab:marg_vs_unmarg`` (the paired-bootstrap contrast) and
``tab:data_summary`` (Table 6 coverage and N obs).
"""

from __future__ import annotations

#: Pupillary-lag offsets in ms (van Rij et al. 2019 bracket; 1000 ms is Winn 2016).
OFFSETS = (900, 1000, 1300)
MAIN_OFFSET = 1000

#: The four speech LMs (feather naming) and their manuscript labels.
SPEECH_LMS = ("gslm", "spirit", "llamamimi", "llamamimi8b")
LM_LABEL = {
    "gslm": "GSLM",
    "spirit": "SpiRit",
    "llamamimi": "Llama-Mimi",
    "llamamimi8b": "Llama-Mimi-8B",
    "gpt2": "GPT-2",
}

# ---------------------------------------------------------------------------
# Held-out delta log-likelihood: mean, (ci_lower, ci_upper), sig, per offset.
# Keyed [condition][offset_ms]. Conditions match the manuscript rows.
# ---------------------------------------------------------------------------

#: Table ``tab:delta_ll_word``. Word level; GPT-2 plus each speech LM's
#: marginalized (IS) and unmarginalized surprisal.
WORD_DELTA_LL: dict[str, dict[int, dict]] = {
    "gpt2": {
        900: {"mean": 0.0035, "ci": (0.0020, 0.0049), "sig": "***"},
        1000: {"mean": 0.0031, "ci": (0.0017, 0.0044), "sig": "***"},
        1300: {"mean": 0.0019, "ci": (0.0007, 0.0030), "sig": "**"},
    },
    "gslm_marg": {
        900: {"mean": 0.0045, "ci": (0.0031, 0.0060), "sig": "**"},
        1000: {"mean": 0.0041, "ci": (0.0028, 0.0054), "sig": "**"},
        1300: {"mean": 0.0021, "ci": (0.0011, 0.0030), "sig": "**"},
    },
    "gslm_unmarg": {
        900: {"mean": -0.0005, "ci": (-0.0016, 0.0006), "sig": "ns"},
        1000: {"mean": -0.0004, "ci": (-0.0014, 0.0006), "sig": "ns"},
        1300: {"mean": -0.0003, "ci": (-0.0010, 0.0004), "sig": "ns"},
    },
    "spirit_marg": {
        900: {"mean": 0.0024, "ci": (0.0009, 0.0038), "sig": "**"},
        1000: {"mean": 0.0018, "ci": (0.0006, 0.0030), "sig": "*"},
        1300: {"mean": 0.0015, "ci": (0.0005, 0.0025), "sig": "**"},
    },
    "spirit_unmarg": {
        900: {"mean": -0.0034, "ci": (-0.0050, -0.0019), "sig": "ns"},
        1000: {"mean": -0.0031, "ci": (-0.0044, -0.0018), "sig": "ns"},
        1300: {"mean": -0.0029, "ci": (-0.0040, -0.0018), "sig": "ns"},
    },
    "llamamimi_marg": {
        900: {"mean": 0.0021, "ci": (0.0007, 0.0035), "sig": "**"},
        1000: {"mean": 0.0018, "ci": (0.0004, 0.0031), "sig": "*"},
        1300: {"mean": 0.0014, "ci": (0.0003, 0.0025), "sig": "*"},
    },
    "llamamimi_unmarg": {
        900: {"mean": 0.0006, "ci": (-0.0001, 0.0013), "sig": "ns"},
        1000: {"mean": 0.0006, "ci": (-0.0001, 0.0012), "sig": "ns"},
        1300: {"mean": 0.0001, "ci": (-0.0004, 0.0005), "sig": "ns"},
    },
    "llamamimi8b_marg": {
        900: {"mean": 0.0032, "ci": (0.0019, 0.0045), "sig": "**"},
        1000: {"mean": 0.0029, "ci": (0.0016, 0.0041), "sig": "**"},
        1300: {"mean": 0.0024, "ci": (0.0014, 0.0035), "sig": "**"},
    },
    "llamamimi8b_unmarg": {
        900: {"mean": 0.0010, "ci": (0.0003, 0.0017), "sig": "*"},
        1000: {"mean": 0.0009, "ci": (0.0003, 0.0016), "sig": "*"},
        1300: {"mean": 0.0006, "ci": (0.0001, 0.0010), "sig": "*"},
    },
}

#: Table ``tab:delta_ll_phone``. Phone level, unmarginalized only.
PHONE_DELTA_LL: dict[str, dict[int, dict]] = {
    "gslm_unmarg": {
        900: {"mean": 0.0018, "ci": (0.0011, 0.0025), "sig": "**"},
        1000: {"mean": 0.0016, "ci": (0.0010, 0.0023), "sig": "**"},
        1300: {"mean": 0.0011, "ci": (0.0007, 0.0016), "sig": "**"},
    },
    "spirit_unmarg": {
        900: {"mean": 0.0025, "ci": (0.0017, 0.0033), "sig": "**"},
        1000: {"mean": 0.0024, "ci": (0.0016, 0.0031), "sig": "**"},
        1300: {"mean": 0.0017, "ci": (0.0010, 0.0023), "sig": "**"},
    },
    "llamamimi_unmarg": {
        900: {"mean": 0.0021, "ci": (0.0013, 0.0030), "sig": "**"},
        1000: {"mean": 0.0018, "ci": (0.0011, 0.0026), "sig": "**"},
        1300: {"mean": 0.0012, "ci": (0.0006, 0.0018), "sig": "**"},
    },
    "llamamimi8b_unmarg": {
        900: {"mean": 0.0023, "ci": (0.0014, 0.0032), "sig": "**"},
        1000: {"mean": 0.0020, "ci": (0.0012, 0.0028), "sig": "**"},
        1300: {"mean": 0.0012, "ci": (0.0006, 0.0018), "sig": "**"},
    },
}

#: Table ``tab:delta_ll_frame``. Frame level, unmarginalized only.
FRAME_DELTA_LL: dict[str, dict[int, dict]] = {
    "gslm_unmarg": {
        900: {"mean": 0.0019, "ci": (0.0007, 0.0031), "sig": "**"},
        1000: {"mean": 0.0009, "ci": (0.0001, 0.0017), "sig": "ns"},
        1300: {"mean": 0.0010, "ci": (0.0001, 0.0018), "sig": "ns"},
    },
    "spirit_unmarg": {
        900: {"mean": 0.0027, "ci": (0.0018, 0.0037), "sig": "**"},
        1000: {"mean": 0.0027, "ci": (0.0017, 0.0036), "sig": "**"},
        1300: {"mean": 0.0014, "ci": (0.0007, 0.0021), "sig": "**"},
    },
    "llamamimi_unmarg": {
        900: {"mean": 0.0035, "ci": (0.0025, 0.0044), "sig": "**"},
        1000: {"mean": 0.0030, "ci": (0.0021, 0.0039), "sig": "**"},
        1300: {"mean": 0.0018, "ci": (0.0010, 0.0025), "sig": "**"},
    },
    "llamamimi8b_unmarg": {
        900: {"mean": 0.0023, "ci": (0.0015, 0.0032), "sig": "**"},
        1000: {"mean": 0.0020, "ci": (0.0011, 0.0028), "sig": "**"},
        1300: {"mean": 0.0011, "ci": (0.0004, 0.0018), "sig": "*"},
    },
}

#: Table ``tab:marg_vs_unmarg``. Paired one-sided bootstrap test on the
#: within-LM difference (marginalized minus unmarginalized) in word delta-LL.
#: ``mean`` positive means marginalized surprisal predicts pupil size better.
#: ``p`` is the reported bootstrap p-value.
MARG_VS_UNMARG: dict[str, dict[int, dict]] = {
    "gslm": {
        900: {"mean": 0.0050, "ci": (0.0032, 0.0068), "p": "<0.001"},
        1000: {"mean": 0.0045, "ci": (0.0029, 0.0061), "p": "<0.001"},
        1300: {"mean": 0.0024, "ci": (0.0011, 0.0036), "p": "<0.001"},
    },
    "llamamimi": {
        900: {"mean": 0.0015, "ci": (0.0001, 0.0030), "p": "0.019"},
        1000: {"mean": 0.0012, "ci": (-0.0002, 0.0026), "p": "0.044"},
        1300: {"mean": 0.0014, "ci": (0.0003, 0.0025), "p": "0.007"},
    },
    "llamamimi8b": {
        900: {"mean": 0.0022, "ci": (0.0008, 0.0035), "p": "0.001"},
        1000: {"mean": 0.0019, "ci": (0.0006, 0.0032), "p": "0.002"},
        1300: {"mean": 0.0019, "ci": (0.0008, 0.0029), "p": "<0.001"},
    },
    "spirit": {
        900: {"mean": 0.0058, "ci": (0.0038, 0.0078), "p": "<0.001"},
        1000: {"mean": 0.0049, "ci": (0.0032, 0.0066), "p": "<0.001"},
        1300: {"mean": 0.0044, "ci": (0.0031, 0.0058), "p": "<0.001"},
    },
}

# ---------------------------------------------------------------------------
# Table 6 (tab:data_summary): coverage and N obs.
# ---------------------------------------------------------------------------

#: Word-level marginalized-estimator coverage: (n_tokens_est, n_tokens_total)
#: over items 3-6 with valid timing (offset-independent). Total is 1229.
WORD_COVERAGE: dict[str, tuple[int, int]] = {
    "gslm": (959, 1229),
    "spirit": (809, 1229),
    "llamamimi": (1010, 1229),
    "llamamimi8b": (1009, 1229),
}

#: N pupil-sample observations entering each GAMM (at omega=1000 ms), keyed
#: (granularity, model). Reproduced exactly by ``Hopton2026Speech._model_data``.
#: Word Marg. and Unmarg. of one LM share one design, so one N per LM.
N_OBS: dict[tuple[str, str], int] = {
    ("word", "gpt2"): 29150,
    ("word", "gslm"): 19418,
    ("word", "spirit"): 10841,
    ("word", "llamamimi"): 22652,
    ("word", "llamamimi8b"): 22515,
    ("phone", "gslm"): 30763,
    ("phone", "spirit"): 29295,
    ("phone", "llamamimi"): 29292,
    ("phone", "llamamimi8b"): 29292,
    ("frame", "gslm"): 11571,
    ("frame", "spirit"): 22262,
    ("frame", "llamamimi"): 31582,
    ("frame", "llamamimi8b"): 31582,
}

#: Number of RSpin items (lists 3-6) and normal-hearing listeners in the study.
N_ITEMS = 162
N_LISTENERS = 21

# ---------------------------------------------------------------------------
# Estimator-variance experiment (Appendix, Figure variance_llamamimi).
# ---------------------------------------------------------------------------

#: Nonparametric bootstrap of the IS estimator's sampling variance, target
#: Llama-Mimi-1.3B: for each word, 1000 resamples (with replacement) of each
#: subsample size from the original 500 importance weights. Variance decreases
#: with subsample size. Reported as a figure, so no per-word anchors here.
VARIANCE_TARGET_LM = "llamamimi"
VARIANCE_SUBSAMPLE_SIZES = (100, 200, 300, 400, 500)
VARIANCE_N_BOOTSTRAP = 1000
VARIANCE_N_SAMPLES = 500
