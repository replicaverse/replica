"""Pinned upstream repositories and their dependency declarations.

Each ``external_repos/<name>`` entry records its source and pinned revision.
``patch`` names a package-relative compatibility diff in :mod:`.adapters`,
applied after checkout.

Studies declare checkouts via :func:`repo_dependencies`; scripts and
loaders resolve concrete paths via :func:`repo_path`, which fetches a
missing checkout on demand.
"""

from __future__ import annotations

from pathlib import Path

from .dependency import DataDependency, fetch_dependency, resolve_data_root


def _git(
    name: str, repo: str, commit: str, patch: str | None = None
) -> tuple[DataDependency, ...]:
    return (
        DataDependency(
            path=f"external_repos/{name}",
            source=f"git:{repo}",
            commit=commit,
            patch=patch,
        ),
    )


def _osf_corpora(
    corpora: dict[str, dict[str, tuple[str, str]]],
) -> tuple[DataDependency, ...]:
    """Build checksum-pinned OSF corpus dependencies."""
    return tuple(
        DataDependency(
            path=f"external_repos/shain2024_data/{stem}_{part}.csv",
            source=f"osf:{file_id}",
            checksum=f"sha256:{digest}",
        )
        for stem, parts in corpora.items()
        for part, (file_id, digest) in parts.items()
    )


EXTERNAL_REPOS: dict[str, tuple[DataDependency, ...]] = {
    # Analysis scripts and statistics PDFs only: no corpora, no surprisal
    # estimates, no intermediate frames. Pinned so the official-code evidence
    # resolves and drifts visibly.
    "Linearity-of-surprisal-on-RT": _git(
        "Linearity-of-surprisal-on-RT",
        "weijiexu-charlie/Linearity-of-surprisal-on-RT",
        "93e3651fdca8fcc445666d1d05e28cf296c9a5f7",
    ),
    "anticipation-on-reading-times": _git(
        "anticipation-on-reading-times",
        "rycolab/anticipation-on-reading-times",
        "14291ca18467d4dd1c4955a141de326558b52e1b",
        patch="external/adapters/anticipation-on-reading-times.patch",
    ),
    # The emnlp18 branch of the CDR toolkit, which is the DTSR the 2018 paper
    # ran. Its five compatibility fixes are in the patch; the fullS baselines
    # additionally need spillover columns the release omits, rebuilt by
    # ``shain2018dtsr.build_spillover``.
    "cdr-emnlp18": _git(
        "cdr-emnlp18",
        "coryshain/cdr",
        "0b8f9cf45395fe0bb34cae7bd6c0fcc572d28704",
        patch="external/adapters/cdr-emnlp18.patch",
    ),
    # The paper-locked branch for the CDR-introducing Cognition paper.
    "cdr-cognition21": _git(
        "cdr-cognition21",
        "coryshain/cdr",
        "662f25629dc1facc4344540aa2b72209ea0ad081",
    ),
    # The branch carrying the neural generalization the later reading papers
    # fit, and the configurations they fit it with.
    "cdr": _git(
        "cdr",
        "coryshain/cdr",
        "dbe63cf916d16addd8b0050360e43587df84c3d1",
    ),
    # The Cognition release (OSF hb5w2): Natural Stories regression frames.
    # The synthetic families ship as an OSF folder tree; the synthetic fetcher
    # of the shain2021cdr study package downloads them.
    "shain2021_data": (
        DataDependency(
            path="external_repos/shain2021_data/SPR_ns/natstor_X.csv",
            source="osf:60a097301521de007c178674",
            checksum="sha256:3be6ce2afdba60f7c3f36f1a0fe1962d7b8c4ebfdfd5c34db0535e5d075c6f03",
        ),
        DataDependency(
            path="external_repos/shain2021_data/SPR_ns/natstor_y_train.csv",
            source="osf:60a096a13b0d2c00847325f8",
            checksum="sha256:b0a0eb6a903e739ebbe9442cd9af6d81f357469f7c56582bf26d4818f21cf80e",
        ),
        DataDependency(
            path="external_repos/shain2021_data/SPR_ns/natstor_y_dev.csv",
            source="osf:60a096831521de007d17a944",
            checksum="sha256:de3d6ceacb544d6efb3dd02c6fedff687eada3f552c508a9fb59e13c70c9c680",
        ),
        DataDependency(
            path="external_repos/shain2021_data/SPR_ns/natstor_y_test.csv",
            source="osf:60a09693d137cb005cf6d096",
            checksum="sha256:3a1c91eb44c238ce1a9aab1e0b2161119c87b77d7126cd7ec1c83a2c3c693186",
        ),
    ),
    # Public Natural Stories language-network fMRI frames (OSF eyp8q). The
    # source names the non-sound-power predictor table ``LANG_ts_X.csv``;
    # cognition21's pinned configuration reads it as ``LANG_X.csv``.
    # Kept separate from the much smaller reading/synthetic dependency so
    # ordinary Shain-2021 checks do not silently trigger a ~430 MB download.
    "shain2021_fmri": (
        DataDependency(
            path="external_repos/shain2021_fmri/fMRI_ns/LANG_X.csv",
            source="osf:fwhdz",
            checksum="sha256:01c311111e9297aaa017a17c7f91d43486c3bfe8f733dff2ae6d7db837a457a0",
        ),
        DataDependency(
            path="external_repos/shain2021_fmri/fMRI_ns/LANG_SP.csv",
            source="osf:f5z7c",
            checksum="sha256:7c85815662c1058a00a60cbdf76ff107e6d4719596c80035a4280b2422a49914",
        ),
        DataDependency(
            path="external_repos/shain2021_fmri/fMRI_ns/LANG_y_train.csv",
            source="osf:sj3yg",
            checksum="sha256:0f30aca9846ac5a38c82cc1b7601d50748f69d613f890a3e4075e4daa1c9785d",
        ),
        DataDependency(
            path="external_repos/shain2021_fmri/fMRI_ns/LANG_y_dev.csv",
            source="osf:upf3e",
            checksum="sha256:f0d3e0ef299e31e529e41bef28f955edde7cb6f9ad5a864fe64534070bc5fb00",
        ),
        DataDependency(
            path="external_repos/shain2021_fmri/fMRI_ns/LANG_y_test.csv",
            source="osf:n7us6",
            checksum="sha256:93cf212c2bd1e1f1e2ea7065358afa3f8b909e53051cad26b82ea27d20024292",
        ),
    ),
    # The same Cognition OSF release (hb5w2) publishes the response-aligned
    # frames generated by all four fMRI baseline transforms.  Pinning those
    # outputs is preferable to reconstructing preprocessing whose command-line
    # invocations were not recorded.  Kept separate because the sixteen files
    # add another ~450 MB to the optional fMRI download.
    "shain2021_fmri_baselines": (
        DataDependency(
            path="external_repos/shain2021_fmri_baselines/fMRI_ns/LANG_convolved_X.csv",
            source="osf:fk258",
            checksum="sha256:6f6ee2c27ba4c072289a285a8f98d1b977bc65ddb71c9ae358286228451c6f84",
        ),
        DataDependency(
            path="external_repos/shain2021_fmri_baselines/fMRI_ns/LANG_convolved_y_train.csv",
            source="osf:60a0961a1521de007d17a873",
            checksum="sha256:244147cb02c682a33f67e3a5824031fcb0a87d3238b64c528c1942a9305f30f2",
        ),
        DataDependency(
            path="external_repos/shain2021_fmri_baselines/fMRI_ns/LANG_convolved_y_dev.csv",
            source="osf:2szan",
            checksum="sha256:78fcc4bfb1a741cb270e8b9656545c0d76d30da83ad43476715f46764727007c",
        ),
        DataDependency(
            path="external_repos/shain2021_fmri_baselines/fMRI_ns/LANG_convolved_y_test.csv",
            source="osf:3rktm",
            checksum="sha256:912a85796ecfff5f8e1549b6acb66d6cbb6d1c5987ff282c24900cfb1914140a",
        ),
        DataDependency(
            path="external_repos/shain2021_fmri_baselines/fMRI_ns/LANG_interpolated_X.csv",
            source="osf:4uby5",
            checksum="sha256:0458f1aee6faa1cd7874b5e7584743abe7483c77af0d6acc8b13104488890095",
        ),
        DataDependency(
            path="external_repos/shain2021_fmri_baselines/fMRI_ns/LANG_interpolated_y_train.csv",
            source="osf:jp6hm",
            checksum="sha256:0932b71a4a0d087553a36975bc1ed7b7ea87c805495f85daf93c367911fbb539",
        ),
        DataDependency(
            path="external_repos/shain2021_fmri_baselines/fMRI_ns/LANG_interpolated_y_dev.csv",
            source="osf:60a096223b0d2c008a730754",
            checksum="sha256:88a18c4c6d3d86fd834640127d2cfc4624c8036cdebe4cf343c30d07070269d4",
        ),
        DataDependency(
            path="external_repos/shain2021_fmri_baselines/fMRI_ns/LANG_interpolated_y_test.csv",
            source="osf:60a09623d137cb0056f69e42",
            checksum="sha256:15fe04562fe2bc85503f13d3f53e71240c3bf3de69dbfea8d9412532c87d5a58",
        ),
        DataDependency(
            path="external_repos/shain2021_fmri_baselines/fMRI_ns/LANG_averaged_X.csv",
            source="osf:4gq7u",
            checksum="sha256:175059b7d0e821cd7c4feb648691868a89c8d1fef21546c85dc4c2d7974e4797",
        ),
        DataDependency(
            path="external_repos/shain2021_fmri_baselines/fMRI_ns/LANG_averaged_y_train.csv",
            source="osf:60a0960ad137cb005bf6948e",
            checksum="sha256:1a6ed70f41398f8c32561868672a52e9aba8493a9dd933ff6e770f41beea425b",
        ),
        DataDependency(
            path="external_repos/shain2021_fmri_baselines/fMRI_ns/LANG_averaged_y_dev.csv",
            source="osf:qsfmw",
            checksum="sha256:2859ad794ee16a549977aea0370b510d75f36f51e3628685a51d2835cc4f99d9",
        ),
        DataDependency(
            path="external_repos/shain2021_fmri_baselines/fMRI_ns/LANG_averaged_y_test.csv",
            source="osf:pvay5",
            checksum="sha256:2897c8b3694161140bdc8febd90165f53ef861283bcd4a5d1393c4ecf980b52c",
        ),
        DataDependency(
            path="external_repos/shain2021_fmri_baselines/fMRI_ns/LANG_lanczos_X.csv",
            source="osf:6qg9f",
            checksum="sha256:101217641c2f8e2db15d77ad198013808219a6483290bc8dae87a49bf2fa9935",
        ),
        DataDependency(
            path="external_repos/shain2021_fmri_baselines/fMRI_ns/LANG_lanczos_y_train.csv",
            source="osf:60a096683b0d2c008b731958",
            checksum="sha256:7511cff60abafbde2080bd53a8f94208b29320010756911e9204c56564015467",
        ),
        DataDependency(
            path="external_repos/shain2021_fmri_baselines/fMRI_ns/LANG_lanczos_y_dev.csv",
            source="osf:ugvhx",
            checksum="sha256:ccdf821f078312f8ae4f3a2ef6967db389b285e630a5698bb2a9be0ad303b763",
        ),
        DataDependency(
            path="external_repos/shain2021_fmri_baselines/fMRI_ns/LANG_lanczos_y_test.csv",
            source="osf:60a096423b0d2c008b731910",
            checksum="sha256:98b5c4c636c48d8d4eb502577139a942dd1b1f63c45eb94058af9118b356e60b",
        ),
    ),
    "cloze-surprisal": _git(
        "cloze-surprisal",
        "sathvikn/cloze-surprisal",
        "4fa5a2c6b7c53340f8c6af4910d20d7e2ce8cf92",
    ),
    "context-reading-time": _git(
        "context-reading-time",
        "rycolab/context-reading-time",
        "b2b573e858c5cfc77b2f4babe502a70d9371d9a4",
    ),
    # Supplies `ridge_utils/utils.py`, whose `make_delayed` the Oota et al.
    # (2023) encoding code imports as `util` but does not ship.
    "deep-fMRI-dataset": _git(
        "deep-fMRI-dataset",
        "HuthLab/deep-fMRI-dataset",
        "eaaa5cd186e0222c374f58adf29ed13ab66cc02a",
    ),
    # The similarity pipeline is here, but its scorer internals live in the
    # src/information_val submodule (glnmario/diverse-surprisal @ f558cdb9),
    # whose repository is deleted and whose SHA the fork network does not
    # serve; recovering them needs the authors.
    "diverse-surprisal": _git(
        "diverse-surprisal",
        "cimeister/diverse-surprisal",
        "5c36c9c8dab0b05ebc4ff70d93b40dfb182be1b7",
    ),
    # The pin is the release the paper's analysis ran on. The second entry is
    # the table the authors regenerated after fixing their sampler,
    # https://github.com/rycolab/generalized-surprisal/commit/d52538a11a17d4a5cb93f4983b3647d904960485,
    # landed beside the released one.
    "generalized-surprisal": (
        *_git(
            "generalized-surprisal",
            "rycolab/generalized-surprisal",
            "564a0a03c72b8521e8bcba31ee7ffab82d22b6bd",
        ),
        DataDependency(
            path="external_repos/generalized-surprisal/data/psyling/aligned_with_estimates_v2.csv",
            source=(
                "url:https://raw.githubusercontent.com/rycolab/generalized-surprisal/"
                "d52538a11a17d4a5cb93f4983b3647d904960485/"
                "data/psyling/aligned_with_estimates_v2.csv"
            ),
            checksum="sha256:243b2926f0df04e1ffcdb5f63b9c749517f33609a7edcf0549356ade5097c549",
        ),
    ),
    # The analysis notebook joins its per-participant fixation-count control from
    # data/psychometric/, which the authors publish on their `word-level` branch
    # and not on `main`. The second entry lands that one file, pinned by SHA,
    # inside the checkout at the path the notebook reads.
    "information-value": (
        *_git(
            "information-value",
            "dmg-illc/information-value",
            "505b2141273a002fc199011135f70b3fa3e1495b",
        ),
        DataDependency(
            path="external_repos/information-value/data/psychometric/RTs_5jun/provo_rt.jsonl",
            source=(
                "url:https://raw.githubusercontent.com/dmg-illc/information-value/"
                "6953531e4cb81f8032fffc4395095d5294aa8665/"
                "data/psychometric/RTs_5jun/provo_rt.jsonl"
            ),
            checksum="sha256:f2060b11b9fd8a48b0441919322194f296442e0218b71674b5786c32f36a8a30",
        ),
        DataDependency(
            path="external_repos/information-value/data/psychometric/RTs_5jun/brown_rt.jsonl",
            source=(
                "url:https://raw.githubusercontent.com/dmg-illc/information-value/"
                "6953531e4cb81f8032fffc4395095d5294aa8665/"
                "data/psychometric/RTs_5jun/brown_rt.jsonl"
            ),
            checksum="sha256:58e13e7ba8a576ef5beb0cb3735cce12e66c2f18738a8c9064ec612b16ec4e5c",
        ),
        DataDependency(
            path="external_repos/information-value/data/psychometric/BLL2018/processed_ratings.jsonl",
            source=(
                "url:https://raw.githubusercontent.com/dmg-illc/information-value/"
                "6953531e4cb81f8032fffc4395095d5294aa8665/"
                "data/psychometric/BLL2018/processed_ratings.jsonl"
            ),
            checksum="sha256:b82c760cfc867c8d0d51967f136e280b698cce221b894bdd87e826b1d241406b",
        ),
    ),
    # Tsipidi et al. (2026), the word-level per-layer information value the
    # word_information_value method ports.
    "llm-representations-rt": _git(
        "llm-representations-rt",
        "rycolab/llm-representations-rt",
        "e4f26f597f0f4a83d9f9e5fec1b3914d11bb54e1",
    ),
    "lingprop-brain-alignment": _git(
        "lingprop-brain-alignment",
        "subbareddy248/lingprop-brain-alignment",
        "35c0570e72bc7ad807ca267ddf720797e4ea41ee",
    ),
    "modelblocks-release": _git(
        "modelblocks-release",
        "modelblocks/modelblocks-release",
        "3ae5120327005cde1c02576eda24876997e20550",
        patch="external/adapters/modelblocks-release.patch",
    ),
    "psycho-toke": _git(
        "psycho-toke",
        "rycolab/psycho-toke",
        "c4b60466c909fd87d2186a0b6d12dce8e9a0fe05",
    ),
    "unigram-freq": _git(
        "unigram-freq",
        "zhopto3/unigram-freq",
        "534a4bca16e2b9b2c12ba22ab5eb329e60f017e1",
    ),
    "word-entropy": _git(
        "word-entropy",
        "christian-clark/word-entropy",
        "96947692833e8b212bd8952637cd05071edd199a",
    ),
    "xlang-processing": _git(
        "xlang-processing",
        "wilcoxeg/xlang-processing",
        "73a2cd845ccc69f5706953b02b3f94ff6fcdb139",
    ),
    # The same repository before the authors' 2024 revision (MECO id fix,
    # unigram-surprisal baseline, frequency filters), which the published
    # TACL text predates; the study README records what each state recovers.
    "xlang-processing-prerevision": _git(
        "xlang-processing-prerevision",
        "wilcoxeg/xlang-processing",
        "891fd878383d1a2bc14c3a006ea01f02496a177a",
    ),
    # Release assets, pinned by tag.
    "surprisal_reading_time_en_ja": (
        DataDependency(
            path="external_repos/surprisal_reading_time_en_ja/dundee_surprisals",
            source="release:kuribayashi4/surprisal_reading_time_en_ja:v0.1:dundee_surprisals.zip",
        ),
        DataDependency(
            path="external_repos/surprisal_reading_time_en_ja/bccwj_surprisals",
            source="release:kuribayashi4/surprisal_reading_time_en_ja:v0.1:bccwj_surprisals.zip",
        ),
    ),
    "surprisal_internal_layers": (
        DataDependency(
            path="external_repos/surprisal_internal_layers",
            # 1.2 GB tar.xz: data/, results/{logit-lens,tuned-lens}/ and the
            # aggregate result CSVs, plus the authors' own code.
            source="release:kuribayashi4/surprisal_internal_layers:v0:surprisal_internal_layers_shared.tar.xz",
        ),
    ),
    # The authors' processed regression matrices, OSF project 6wvqe. The same
    # six corpus directories are mirrored by the dissociation paper's OSF
    # project 8v5qb with identical SHA-256 digests.
    "shain2024_data": _osf_corpora(
        {
            "SPR_natstor/natstor": {
                "X": (
                    "4rjyu",
                    "b0de8d39b4dc4cbf83abe33dcc59f2a3e8475b162d4051651ca8d3de01f6e76b",
                ),
                "y_train": (
                    "wcqna",
                    "559abb39776befaa816f20323843ba91d46037d6a6a23f2b20c9dacc3cddd259",
                ),
                "y_dev": (
                    "hq8de",
                    "b2e5afa68fb091f256983fae8102c092f75293af698b65a4e33accd4b22f8bad",
                ),
                "y_test": (
                    "268r3",
                    "7d5e837cc243808b1f99423e18ef28eafc760589503b5e34f490425a70333c96",
                ),
            },
            "ET_provo/provo": {
                "X": (
                    "qy276",
                    "eea5e8e5682fc7fb14311755941c1a2f69f62211255e612c24964be32caef6b2",
                ),
                "y_train": (
                    "zxj5m",
                    "6ffa48f1565fe4da92ea509c7ad2aef980ad8a999c339c4fd95db6b6825cb9dd",
                ),
                "y_dev": (
                    "5e8zw",
                    "14a516ff0f907cc86ae4542a3ac29b4979789e18710da2ea7b0916e7d95eb1d3",
                ),
                "y_test": (
                    "62z4x",
                    "988467765d2a98490508ae7dca3200bab9c818de05538ccf1bb8977f6165c621",
                ),
            },
            "ET_geco/geco": {
                "X": (
                    "3r6km",
                    "3d8a3ba746a441036e66cc5e8991bb5d3c1d2a51d4a1cd099f5481ca09b72217",
                ),
                "y_train": (
                    "h8sf3",
                    "35356d9feecd94cd788d5771664e954af4ae1327605568c1d50754b7d9429f52",
                ),
                "y_dev": (
                    "3xt6y",
                    "8c6a7540df76f63ca8cb90d5e302fa1a1f634b0c6612112f413f7e5549951f97",
                ),
                "y_test": (
                    "rm28h",
                    "ed6705ac9ddbcf4888e5885726b735bf36013b987027c940f0b94f8617c6b42b",
                ),
            },
            "MAZE_natstor/natstormaze": {
                "X": (
                    "eskd2",
                    "273626b2f9ddbbc68ebbefeac410be28a3c82fd70a31d64bfe7ad14dfd7ef5a2",
                ),
                "y_train": (
                    "rza93",
                    "e579d7c7790de7abd5c9e133d44d734b8141d108c9a06c6037e305a9048e9cf5",
                ),
                "y_dev": (
                    "z2yc6",
                    "4f85f874ea7ddb931790efcc599075332570f75da5b237888b78cef70320aa77",
                ),
                "y_test": (
                    "8y7a9",
                    "b739066ab3c5880f27a0e19f8ad7106b6466cd0722681a87bcaca1b31b5a5b1c",
                ),
            },
            "SPR_brown/brown": {
                "X": (
                    "wzgdq",
                    "e0c2d5f7c6c19af360f916898f872e458692249f86803597da105c3e7da18fed",
                ),
                "y_train": (
                    "5ufts",
                    "b4b46916df00eff31aab95a3afb93acb7a69b17c18cdbd8307a490137cd2f506",
                ),
                "y_dev": (
                    "qw937",
                    "7835b6cf3b7520e1ab8c5a70531ed3f117e207d71016a6b5d3a81ff1670967a9",
                ),
                "y_test": (
                    "bu7p4",
                    "ae30a715f33e8cab48fb9b56617df308c7dce601685f64b022d073e2f37efff9",
                ),
            },
            "ET_dundee/dundee": {
                "X": (
                    "qkrb6",
                    "31262f6581c8eaeee8062352bc1393cb813f37cc2f833b347cb8fa3365d93a88",
                ),
                "y_train": (
                    "35qxy",
                    "08da36747bb8eca68ae6eba42009982d0c5f036ff41963609c79567a60aa9a28",
                ),
                "y_dev": (
                    "stvw5",
                    "2a0c13e9270fd779c3b30a5a85408015fd32a52bc6d7c3481fad41440f3adc11",
                ),
                "y_test": (
                    "jkzsw",
                    "7ec7b57ecb472be31b8597de38fb507c67f4b5d3d9358007e0f920233bc79459",
                ),
            },
        }
    ),
    "spatio-temporal-reading": (
        *_git(
            "spatio-temporal-reading",
            "rycolab/spatio-temporal-reading",
            "8d834386b0669164d4927fd1a95c871b01636a29",
        ),
        DataDependency(
            # Derived MECO gaze CSV, provided manually: it lives in the
            # authors' public `data.zip`, linked from their repo README.
            path="external_repos/spatio-temporal-reading/data/meco_df_gaze_duration.csv",
            source="manual",
        ),
    ),
}


def repo_dependencies(*names: str) -> list[DataDependency]:
    """Dependencies materializing each ``external_repos/<name>``, for Study lists.

    Args:
        names: Keys of :data:`EXTERNAL_REPOS`, each expanding to that
            checkout's pinned clone plus whatever asset is registered with it.
            The expansions concatenate in the order given, and a name the
            registry lacks raises ``KeyError``.
    """
    return [dependency for name in names for dependency in EXTERNAL_REPOS[name]]


def repo_path(
    name: str, data_root: Path | str | None = None, *, fetch: bool = True
) -> Path:
    """Resolve ``external_repos/<name>``, fetching missing pieces if *fetch*.

    The root follows the same precedence as ``Study.ensure_data``, through
    :func:`~replica.external.dependency.resolve_data_root`.

    Args:
        name: Key into the pinned-checkout registry.
        data_root: Root the checkout is resolved against; ``None`` falls back
            to ``$PSYCHO_REPL_DATA``, then the source checkout, then
            ``$XDG_DATA_HOME/replica``.
        fetch: Whether to clone or download any dependency of *name* that is
            not already present under the resolved root, then verify all pins and
            declared patches. False only resolves a path; it makes no integrity claim.
    """
    from .verification import verify_dependency

    root = resolve_data_root(data_root)
    if fetch:
        for dep in EXTERNAL_REPOS[name]:
            if not (root / dep.path).exists():
                fetch_dependency(dep, root)
            verify_dependency(dep, root)
    return root / "external_repos" / name
