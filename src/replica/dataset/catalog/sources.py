"""Dataset acquisition catalog and on-disk path policy.

This module is the single authority for upstream source metadata, fetch aliases,
and the destination directory assigned to each downloadable resource.
"""

from replica.dataset.catalog.registry import REGISTRY as _REGISTRY

from ...external import EXTERNAL_REPOS
from ..eye_tracking.meco.releases import (
    FIXATION_RELEASES,
    WORD_RELEASES,
)
from ..eye_tracking.meco.releases import (
    ISO_TO_FILE as _MECO_ISO_TO_FILE,
)
from ..eye_tracking.meco.releases import (
    LANGUAGES as _MECO_LANGS,
)

# Files each ERP corpus pins in its dataset YAML, by its directory in the
# release: the annotation, and for the N400 studies the sentence tokens and
# the per-subject amplitude table.
_RELEASE_MEMBERS = {
    "S_N400": ("all.txt.averaged_rt.annotation", "tokens.json", "szewczyk_2022.tsv"),
    "M_N400": ("all.txt.averaged_rt.annotation", "tokens.json", "michaelov_2024.tsv"),
    "ZuCO": ("all.txt.averaged_rt.annotation",),
}

DATASETS: dict[str, dict] = {
    "provo": {
        "source": "osf",
        "files": [
            ("a32be", "Provo_Corpus-Eyetracking_Data.csv"),
            ("e4a2m", "Provo_Corpus-Predictability_Norms.csv"),
            ("z3eh6", "Provo_Corpus-Additional_Eyetracking_Data-Fixation_Report.csv"),
        ],
    },
    "meco": {
        "source": "osf",
        "dir": "eye-tracking/original/meco",
        "files": [
            ("6996d71d00304e86926568b7", "joint_l1_data_trimmed_version2.1.rda"),
            ("6996d6acecb34f4a8b65628d", "supp_texts.csv"),
        ],
    },
    **{
        key: {
            "source": "osf",
            "files": [(release.osf_id, release.filename)],
            **({"dir": release.data_subdir} if key not in _REGISTRY else {}),
        }
        for key, release in WORD_RELEASES.items()
    },
    **{
        f"{key}_fixations": {
            "source": "osf",
            "dir": release.data_subdir,
            "files": [(release.osf_id, release.rda_filename)],
            "sha256": {release.rda_filename: release.source_sha256},
            "verify_yaml": False,
            "regenerate_yaml": False,
            "include_in_all": False,
        }
        for key, release in FIXATION_RELEASES.items()
    },
    "natural_stories": {
        "source": "github_raw",
        "repo": "languageMIT/naturalstories",
        "branch": "master",
        "revision": "4700daad696e942f5aba23c957a7423d0de66612",
        "files": [
            ("naturalstories_RTS/all_stories.tok", "all_stories.tok"),
            ("naturalstories_RTS/processed_RTs.tsv", "processed_RTs.tsv"),
        ],
    },
    "ucl_spr": {
        "source": "zip",
        "url": "https://static-content.springer.com/esm/art%3A10.3758%2Fs13428-012-0313-y/MediaObjects/13428_2012_313_MOESM1_ESM.zip",
        "files": {
            "stimuli.txt": "stimuli.txt",
            "stimuli_pos.txt": "stimuli_pos.txt",
            "selfpacedreading.subj.txt": "selfpacedreading.subj.txt",
            "selfpacedreading.RT.txt": "selfpacedreading.RT.txt",
        },
    },
    "ucl_et": {
        "source": "zip",
        "url": "https://static-content.springer.com/esm/art%3A10.3758%2Fs13428-012-0313-y/MediaObjects/13428_2012_313_MOESM1_ESM.zip",
        "files": {
            "stimuli.txt": "stimuli.txt",
            "stimuli_pos.txt": "stimuli_pos.txt",
            "eyetracking.subj.txt": "eyetracking.subj.txt",
            "eyetracking.RT.txt": "eyetracking.RT.txt",
            "eyetracking.fix.txt": "eyetracking.fix.txt",
        },
    },
    # Frank et al.'s (2015) data package is the supplementary material of the
    # article's erratum; it holds the ERP release file, its README and one raw
    # EEGLAB file per participant.
    "frank2015_erp": {
        "source": "zip",
        "url": "https://ars.els-cdn.com/content/image/1-s2.0-S0093934X15001182-mmc1.zip",
        "files": {
            name: name
            for name in (
                "stimuli_erp.mat",
                "README.txt",
                *(f"EEG{participant:02d}.mat" for participant in range(1, 25)),
            )
        },
    },
    "geco": {
        "source": "direct_url",
        "files": [
            (
                "https://expsy.ugent.be/downloads/geco/files/MonolingualReadingData.xlsx",
                "MonolingualReadingData.xlsx",
            ),
            (
                "https://expsy.ugent.be/downloads/geco/files/L1ReadingData.xlsx",
                "L1ReadingData.xlsx",
            ),
            (
                "https://expsy.ugent.be/downloads/geco/files/L2ReadingData.xlsx",
                "L2ReadingData.xlsx",
            ),
            (
                "https://expsy.ugent.be/downloads/geco/files/EnglishMaterial.xlsx",
                "EnglishMaterial.xlsx",
            ),
            (
                "https://expsy.ugent.be/downloads/geco/files/DutchMaterials.xlsx",
                "DutchMaterials.xlsx",
            ),
            (
                "https://expsy.ugent.be/downloads/geco/files/SubjectInformation.xlsx",
                "SubjectInformation.xlsx",
            ),
        ],
    },
    # The ERP corpora are extracts from one 1.2 GB release that the
    # Kuribayashi et al. (2025) study already declares as a data dependency, so
    # fetching any of them fetches that release once and shares it.
    **{
        name: {
            "source": "study_release",
            "repo": "surprisal_internal_layers",
            "files": [
                (f"data/{corpus}/{filename}", filename)
                for filename in _RELEASE_MEMBERS[corpus]
            ],
            "regenerate_yaml": False,
        }
        for name, corpus in (
            ("szewczyk_n400", "S_N400"),
            ("federmeier2007_n400", "S_N400"),
            ("hubbard2019_n400", "S_N400"),
            ("szewczyk2022_n400", "S_N400"),
            ("szewczyk_federmeier2022_n400", "S_N400"),
            ("wlotko_federmeier2012_n400", "S_N400"),
            ("michaelov_n400", "M_N400"),
            ("zuco_n400", "ZuCO"),
        )
    },
    "celer": {
        "source": "manual",
        "provenance": {
            "sent_ia.tsv": "data_v2.0.zip#member=data_v2.0/sent_ia.tsv",
            "sent_fix.tsv": "data_v2.0.zip#member=data_v2.0/sent_fix.tsv",
        },
        "instructions": (
            "The CELER corpus (Berzak et al., 2022) requires proof of LDC access\n"
            "to the Penn Treebank (LDC95T7) and BLLIP corpus (LDC2000T43).\n\n"
            "1. Clone: https://github.com/berzak/celer\n"
            "2. Check out commit 163e1635cf399ac7367dd845d1bf7f27d092e93b\n"
            "3. Place PTB README and BLLIP README.1st in ptb_bllip_readmes/\n"
            "4. Run: python obtain_data.py --release v2.0\n"
            "5. Extract data_v2.0.zip. Copy data_v2.0/sent_ia.tsv and, for\n"
            "   fixation access, data_v2.0/sent_fix.tsv to:\n"
            "   datasets/eye-tracking/original/celer/\n"
            "6. Load CELER normally to verify the shipped checksums.\n"
            "   Do not regenerate checksum baselines from your download."
        ),
    },
    "onestop": {
        "source": "osf_zip",
        "files": [
            ("dq935", "fixations_Paragraph.csv.zip"),
            ("4ajc8", "ia_Paragraph.csv.zip"),
        ],
    },
    "copco": {
        "source": "osf",
        # Word measures come from ExtractedFeatures; the loader's fixation
        # stream comes from FixationReports.
        "files": [
            ("6233373a1372e806cccc4c8f", "ExtractedFeatures/P01.csv"),
            ("6233373a1372e806c8cc4bc6", "ExtractedFeatures/P02.csv"),
            ("623337391372e806cccc4c8a", "ExtractedFeatures/P03.csv"),
            ("6233373a33d85406b3f8ab92", "ExtractedFeatures/P04.csv"),
            ("6233373aac980a070294f8e1", "ExtractedFeatures/P05.csv"),
            ("6233373c33d85406b3f8ab9e", "ExtractedFeatures/P06.csv"),
            ("6233373d1372e806cccc4ca2", "ExtractedFeatures/P07.csv"),
            ("6233373d33d85406b0f8ab46", "ExtractedFeatures/P08.csv"),
            ("6233373e6cd037072130e6f4", "ExtractedFeatures/P09.csv"),
            ("6233373ee9194506b11773b3", "ExtractedFeatures/P10.csv"),
            ("623337406cd037072430e75d", "ExtractedFeatures/P11.csv"),
            ("623337406cd037072530e80a", "ExtractedFeatures/P12.csv"),
            ("623337416cd037072130e6fb", "ExtractedFeatures/P13.csv"),
            ("6233374233d85406b3f8abbb", "ExtractedFeatures/P15.csv"),
            ("623337421372e806cccc4cb7", "ExtractedFeatures/P16.csv"),
            ("623337446cd037072530e821", "ExtractedFeatures/P17.csv"),
            ("623db6cf8d53ef0f4e67fce2", "ExtractedFeatures/P18.csv"),
            ("623db6cfe6b58b0dc8d6d777", "ExtractedFeatures/P19.csv"),
            ("623db6cf938b480e5697bb0d", "ExtractedFeatures/P20.csv"),
            ("623db6d1938b480e5797c671", "ExtractedFeatures/P21.csv"),
            ("623db6d18b90a60ea4ae614e", "ExtractedFeatures/P22.csv"),
            ("623db6d2e6b58b0dc3d6d28c", "ExtractedFeatures/P23.csv"),
            ("623db6d48d53ef0f4b67e4e1", "ExtractedFeatures/P24.csv"),
            ("623db6d4e6b58b0dced6bbc1", "ExtractedFeatures/P25.csv"),
            ("623db6d4e6b58b0dced6bbc5", "ExtractedFeatures/P26.csv"),
            ("623db6d4e6b58b0dc8d6d795", "ExtractedFeatures/P27.csv"),
            ("623db6d5938b480e5697bb2a", "ExtractedFeatures/P28.csv"),
            ("62556b46d5121d062a336c87", "ExtractedFeatures/P29.csv"),
            ("62556b46292eac051dd05dc3", "ExtractedFeatures/P30.csv"),
            ("62556b4628f94004f6a24615", "ExtractedFeatures/P31.csv"),
            ("62556b4828f94004f6a2461a", "ExtractedFeatures/P32.csv"),
            ("62556b49292eac051dd05dcf", "ExtractedFeatures/P33.csv"),
            ("62556b4928f94004f8a248e9", "ExtractedFeatures/P34.csv"),
            ("62556b4b28f94004fca24d45", "ExtractedFeatures/P35.csv"),
            ("62556b4a292eac051cd061ba", "ExtractedFeatures/P36.csv"),
            ("62556b4c6937620648db5020", "ExtractedFeatures/P37.csv"),
            ("62556b4dd5121d062a336c97", "ExtractedFeatures/P38.csv"),
            ("62d541301bb7a539561f3665", "ExtractedFeatures/P39.csv"),
            ("62d54130c79a4c32449e59d1", "ExtractedFeatures/P40.csv"),
            ("62d54130c79a4c323c9e59e3", "ExtractedFeatures/P41.csv"),
            ("631edeecbd6028033cbd71a1", "ExtractedFeatures/P42.csv"),
            ("631edeec4ff6c40332714e18", "ExtractedFeatures/P43.csv"),
            ("631edeef753784038ad9b893", "ExtractedFeatures/P44.csv"),
            ("631edeee753784038ad9b88c", "ExtractedFeatures/P45.csv"),
            ("631edef27537840389d9b377", "ExtractedFeatures/P46.csv"),
            ("631edef2bd60280337bd64a9", "ExtractedFeatures/P47.csv"),
            ("631edef37537840389d9b37b", "ExtractedFeatures/P48.csv"),
            ("631edef3d88c0703559d11e6", "ExtractedFeatures/P49.csv"),
            ("631edef4bd6028033cbd71b8", "ExtractedFeatures/P50.csv"),
            ("631edef5d88c0703519d0a88", "ExtractedFeatures/P51.csv"),
            ("631edef7d88c0703569d15d0", "ExtractedFeatures/P52.csv"),
            ("631edef94ff6c40331714a1a", "ExtractedFeatures/P53.csv"),
            ("631edef7753784038ad9b8a8", "ExtractedFeatures/P54.csv"),
            ("631edef8d88c0703559d11f6", "ExtractedFeatures/P55.csv"),
            ("631edef9d88c0703519d0a90", "ExtractedFeatures/P56.csv"),
            ("631edefa7537840386d9ad95", "ExtractedFeatures/P57.csv"),
            ("636b67bca0cd5306e2b7864e", "ExtractedFeatures/P58.csv"),
            ("61e16fabbc925b02bbd4b109", "FixationReports/FIX_report_P01.txt"),
            ("61e16fa9bc925b02b7d4b44f", "FixationReports/FIX_report_P02.txt"),
            ("61e16fab95a03002c8ed3bbc", "FixationReports/FIX_report_P03.txt"),
            ("61e16fabc99ebd0310017dae", "FixationReports/FIX_report_P04.txt"),
            ("61e16fa995a03002d3ed3aee", "FixationReports/FIX_report_P05.txt"),
            ("61e16fad466d1502a4d56909", "FixationReports/FIX_report_P06.txt"),
            ("61e16fad53349e02cb5f9e5d", "FixationReports/FIX_report_P07.txt"),
            ("61e16faf95a03002d5ed2a2a", "FixationReports/FIX_report_P08.txt"),
            ("61e16fafbc925b02b3d4b2c5", "FixationReports/FIX_report_P09.txt"),
            ("61e16faf53349e02cf5fa3db", "FixationReports/FIX_report_P10.txt"),
            ("61e16fb053349e02cf5fa3e2", "FixationReports/FIX_report_P11.txt"),
            ("61e16fb1bc925b02b3d4b2ca", "FixationReports/FIX_report_P12.txt"),
            ("61e16fb153349e02d35f9766", "FixationReports/FIX_report_P13.txt"),
            ("61e16fb2c99ebd0310017db5", "FixationReports/FIX_report_P15.txt"),
            ("61e16fb3466d1502a4d56921", "FixationReports/FIX_report_P16.txt"),
            ("61e16fb353349e02cb5f9e70", "FixationReports/FIX_report_P17.txt"),
            ("62a2f75578fd98020e710aa2", "FixationReports/FIX_report_P18.txt"),
            ("62a2f75378fd98020d71095d", "FixationReports/FIX_report_P19.txt"),
            ("62a2f758b9b3fc01f4db3beb", "FixationReports/FIX_report_P20.txt"),
            ("62a2f758f0123f01bfeadbb7", "FixationReports/FIX_report_P21.txt"),
            ("62a2f752f0123f01c6eadf1e", "FixationReports/FIX_report_P22.txt"),
            ("62a2f75278fd98020d710957", "FixationReports/FIX_report_P23.txt"),
            ("62a2f75678fd98020e710aa7", "FixationReports/FIX_report_P24.txt"),
            ("62a2f755df5d9401b19892ab", "FixationReports/FIX_report_P25.txt"),
            ("62a2f75278fd98020e710a93", "FixationReports/FIX_report_P26.txt"),
            ("62a2f752a3428e0177ae5c2b", "FixationReports/FIX_report_P27.txt"),
            ("62a2f75678fd98020d710967", "FixationReports/FIX_report_P28.txt"),
            ("62a2f74eb9b3fc01f1db38d7", "FixationReports/FIX_report_P29.txt"),
            ("62a2f74878fd98020b7107fa", "FixationReports/FIX_report_P30.txt"),
            ("62a2f746b9b3fc01f1db38cb", "FixationReports/FIX_report_P31.txt"),
            ("62a2f74d78fd98020d71094e", "FixationReports/FIX_report_P32.txt"),
            ("62a2f74da3428e0177ae5c21", "FixationReports/FIX_report_P33.txt"),
            ("62a2f748a3428e0177ae5c15", "FixationReports/FIX_report_P34.txt"),
            ("62a2f74878fd98020a710754", "FixationReports/FIX_report_P35.txt"),
            ("62a2f74c78fd98020a710759", "FixationReports/FIX_report_P36.txt"),
            ("62a2f749f0123f01c3eadcc2", "FixationReports/FIX_report_P37.txt"),
            ("62a2f749df5d9401b99895e3", "FixationReports/FIX_report_P38.txt"),
            ("62c69e63c79a4c01d99e5ba5", "FixationReports/FIX_report_P39.txt"),
            ("62c69e63588bb901ffb86fd7", "FixationReports/FIX_report_P40.txt"),
            ("62c69e631bb7a502121f37ee", "FixationReports/FIX_report_P41.txt"),
            ("6319a4c209a85224f0d9ea16", "FixationReports/FIX_report_P42.txt"),
            ("6319a4c309a85224f1d9ee64", "FixationReports/FIX_report_P43.txt"),
            ("6319a4c4db9397271311e227", "FixationReports/FIX_report_P44.txt"),
            ("6319a4c59a7d51260f902eda", "FixationReports/FIX_report_P45.txt"),
            ("6319a4c509a85224f0d9ea25", "FixationReports/FIX_report_P46.txt"),
            ("6319a4c9e7f1b727caaaf247", "FixationReports/FIX_report_P47.txt"),
            ("6319a4c809a85224f1d9ee72", "FixationReports/FIX_report_P48.txt"),
            ("6319a4c9e7f1b727caaaf24b", "FixationReports/FIX_report_P49.txt"),
            ("6319a4c8e7f1b727c4aae8a1", "FixationReports/FIX_report_P50.txt"),
            ("6319a4c9e7f1b727caaaf24f", "FixationReports/FIX_report_P51.txt"),
            ("6319a4cbdb9397271711e68f", "FixationReports/FIX_report_P52.txt"),
            ("6319a4cc09a85224ead9e581", "FixationReports/FIX_report_P53.txt"),
            ("6319a4cd9a7d51260c902b80", "FixationReports/FIX_report_P54.txt"),
            ("6319a4cd09a85224f0d9ea3c", "FixationReports/FIX_report_P55.txt"),
            ("6319a4cd9a7d51260f902eee", "FixationReports/FIX_report_P56.txt"),
            ("6319a4cfdb9397271411e42c", "FixationReports/FIX_report_P57.txt"),
            ("636a23f09490f601885351cf", "FixationReports/FIX_report_P58.txt"),
        ],
    },
    "potec": {
        "source": "osf_zip",
        "files": [
            ("53zwb", "corrected_fixations.zip"),
            ("3ywhz", "reading_measures.zip"),
        ],
        "extra_urls": [
            (
                "https://raw.githubusercontent.com/DiLi-Lab/PoTeC/46247ca2aea5876311acf6b59338880d0bf5449d/preprocessing_scripts/word_limits.json",
                "word_limits.json",
            ),
        ],
    },
    "emtec": {
        "source": "osf",
        "files": [
            ("2hs8p", "fixations.csv"),
            ("s4ny8", "reading_measures.csv"),
        ],
    },
    "sbsat": {
        "source": "github_raw",
        "repo": "ahnchive/SB-SAT",
        "branch": "master",
        "revision": "39a9b09df3b0423ee527d7860db225949424cd3d",
        "files": [
            ("fixation/18sat_fixfinal.csv", "18sat_fixfinal.csv"),
            ("fixation/18sat_labels.csv", "18sat_labels.csv"),
            ("fixation/18sat_trialfinal.csv", "18sat_trialfinal.csv"),
        ],
    },
    "rastros": {
        "source": "osf",
        "files": [
            ("zr394", "RastrOS_Corpus_Eyetracking_Data.tsv"),
            ("7qcej", "RastrOS_Corpus_Predictability_Norms.tsv"),
        ],
        "regenerate_yaml": False,
    },
    "rsc": {
        # Both releases land in one directory; ``RSC(release=...)`` picks
        # between them, so one fetch key serves rsc and rsc_extended.
        "source": "osf",
        "files": [
            ("b6fms", "data.csv"),
            ("rfneu", "data_103.csv"),
        ],
    },
    "bsc": {
        "source": "osf_zip",
        "files": [
            ("xfe4s", "BSC.EMD.zip"),
        ],
        "extra_osf_files": [
            ("t4vbg", "BSC.Word.Info.v2.xlsx"),
        ],
    },
    "meco_char": {
        "source": "direct_url",
        "dir": "eye-tracking/processed/meco_char",
        "files": [
            (
                "https://huggingface.co/datasets/samuki-hf/meco-char/"
                f"resolve/main/processed/{name}",
                name,
            )
            for lang in (_MECO_ISO_TO_FILE.get(code, code) for code in _MECO_LANGS)
            for name in (f"{lang}.csv", f"stimulus_{lang}.csv")
        ],
        "regenerate_yaml": False,
    },
    "devarda2023": {
        # The author's all_measures.csv is saved under the loader's local
        # filename aligned.csv.
        "source": "direct_url",
        "files": [
            (
                "https://raw.githubusercontent.com/Andrea-de-Varda/"
                "prediction-resource/7162734da00a9a3c81d5ebaec65bdf6d59e931fc/"
                "all_measures.csv",
                "aligned.csv",
            ),
        ],
        "regenerate_yaml": False,
    },
    "dundee_official": {
        "source": "manual",
        "instructions": (
            "The Dundee Corpus V.1 (Kennedy et al., 2003) is distributed by\n"
            "its author on request; there is no public download.\n\n"
            "1. Ask the corpus author for the V.1 distribution\n"
            "2. Place its 'dundee_corpus' directory at:\n"
            "   datasets/eye-tracking/original/dundee/\n"
            "   so that english/ holds the .dat files and english/items/ the\n"
            "   word-object and .itm files\n"
            '3. Load it with load_dataset("dundee_official") to verify the\n'
            "   shipped checksums. Do not regenerate checksum baselines from\n"
            "   your copy."
        ),
    },
    "dundee": {
        "source": "osf",
        "files": [
            ("649f1ed63809110f5a3c2ddb", "dundee_X.csv"),
            ("649f1eeda2a2f41004436c29", "dundee_y_train.csv"),
            ("649f1edc67aff80fbeedf498", "dundee_y_dev.csv"),
            ("649f1ee1a2a2f40ffb436ec6", "dundee_y_test.csv"),
        ],
        "regenerate_yaml": False,
    },
    "brown_spr": {
        "source": "osf",
        "files": [
            ("649f1fd36513ba0ed83a314a", "brown_X.csv"),
            ("649f1fdba2a2f4102a4363b2", "brown_y_train.csv"),
            ("649f1fd56513ba0ed83a314d", "brown_y_dev.csv"),
            ("649f1fd86513ba0ed83a314f", "brown_y_test.csv"),
        ],
        "regenerate_yaml": False,
    },
    "brothers_kuperberg": {
        "source": "osf",
        "files": [
            ("6516ffad3d9bde04d68d7c1c", "SPRT_LogLin_216.csv"),
            ("6516ff90b0a33e04d6591b52", "AppendixB.csv"),
            ("6516ffa143df320512b625a7", "LogLin_meta.csv"),
            ("6516ff9243df320502b628e4", "LogLin_Naming.csv"),
            ("6516ff8b1bc86504d4f37c51", "Stimuli_Appendix_format.docx"),
        ],
        "regenerate_yaml": False,
    },
    "subtlex_en": {
        "source": "manual",
        "dir": "frequency/en",
        "instructions": (
            "Download the SUBTLEX-US text version (all 74,286 words):\n"
            "  https://www.ugent.be/pp/experimentele-psychologie/en/"
            "research/documents/subtlexus/subtlexus2.zip\n"
            "Extract its tab-separated SUBTLEXus74286wordstextversion.txt into\n"
            "``$PSYCHO_DATA_DIR/frequency/en/``. Then in code:\n"
            "    from replica.feature import SubtlexSource\n"
            "    src = SubtlexSource(\n"
            "        '$PSYCHO_DATA_DIR/frequency/en/"
            "SUBTLEXus74286wordstextversion.txt',\n"
            "        language='en')\n"
        ),
        "regenerate_yaml": False,
    },
    "zuco1": {
        # ZuCo 1.0 .mat files (MATLAB v5) from OSF project q3zws, one per subject
        # and reading task: Normal Reading (NR), Sentiment Reading (SR) and
        # Task-Specific Reading (TSR). One download serves both zuco1
        # (eye-tracking) and zuco1_eeg, whose YAML points its data_path here.
        "source": "osf",
        "files": [
            ("5afd6e5393d197000d2ef69e", "resultsZDM_NR.mat"),
            ("5afd6fbe93d19700102eef9f", "resultsZAB_NR.mat"),
            ("5afd6e59ee2b5f000f44dabe", "resultsZMG_NR.mat"),
            ("5afd6f2f93d197000e2f0200", "resultsZKW_NR.mat"),
            ("5afd6fa9ee2b5f001044cc52", "resultsZJM_NR.mat"),
            ("5afd6fd793d197000f2f2cb0", "resultsZPH_NR.mat"),
            ("5afd6eadee2b5f000d4505f6", "resultsZKB_NR.mat"),
            ("5d0b8e1811bd080019a14f6a", "resultsZJS_NR.mat"),
            ("5afd6f2693d19700102eef4e", "resultsZJN_NR.mat"),
            ("5afd6fb293d197000e2f024c", "resultsZDN_NR.mat"),
            ("5afd6f9393d197000f2f2c6c", "resultsZKH_NR.mat"),
            ("5afd6f6bee2b5f000e44d083", "resultsZGW_NR.mat"),
            ("mxvq5", "resultsZKB_SR.mat"),
            ("ap6cr", "resultsZKW_SR.mat"),
            ("gwmxd", "resultsZAB_SR.mat"),
            ("xbhry", "resultsZJS_SR.mat"),
            ("npj7s", "resultsZGW_SR.mat"),
            ("dgfw4", "resultsZPH_SR.mat"),
            ("2p8zc", "resultsZDM_SR.mat"),
            ("d7wk6", "resultsZMG_SR.mat"),
            ("uvx3y", "resultsZJM_SR.mat"),
            ("ax9em", "resultsZJN_SR.mat"),
            ("np5jf", "resultsZJS_TSR.mat"),
            ("dywc2", "resultsZMG_TSR.mat"),
            ("cukde", "resultsZJN_TSR.mat"),
            ("83vjz", "resultsZJM_TSR.mat"),
            ("rp5vc", "resultsZKB_TSR.mat"),
            ("u5mtw", "resultsZDN_TSR.mat"),
            ("58mfb", "resultsZAB_TSR.mat"),
            ("e7agv", "resultsZGW_TSR.mat"),
            ("p2s9g", "resultsZKH_TSR.mat"),
            ("q5kp7", "resultsZKW_TSR.mat"),
        ],
        "regenerate_yaml": False,
    },
    "zuco2": {
        # ZuCo 2.0 .mat files from OSF project 2urht ("Matlab files" folders).
        # One download serves both zuco2 (eye-tracking) and zuco2_eeg, whose
        # YAML points its data_path at this same directory.
        "source": "osf",
        "files": [
            ("5dd7f151ab905e000be2b419", "resultsYDG_NR.mat"),
            ("5dd7df016fc769000cd4df63", "resultsYSL_NR.mat"),
            ("5dd7d4a4ab905e000ae2a42b", "resultsYHS_NR.mat"),
            ("5dd7ee02fbde36000c8f5645", "resultsYAC_NR.mat"),
            ("5dd7da66215a8f000a4d029f", "resultsYMS_NR.mat"),
            ("5dd7dd14fbde36000b8f32bd", "resultsYRP_NR.mat"),
            ("5dd7f163ab905e0009e29f30", "resultsYFR_NR.mat"),
            ("5dd7eeecab905e000be2b16d", "resultsYAK_NR.mat"),
            ("5dd7e0cf215a8f000b4d14de", "resultsYTL_NR.mat"),
            ("5dd7dc78215a8f00094cf8e2", "resultsYRK_NR.mat"),
            ("5dd7dc3cab905e000be29c96", "resultsYSD_NR.mat"),
            ("5dd7d67dfbde36000c8f3cd0", "resultsYIS_NR.mat"),
            ("5dd7d6276fc769000cd4d6d2", "resultsYLS_NR.mat"),
            ("5dd7d722ab905e000ae2a723", "resultsYMD_NR.mat"),
            ("5dd7efc8fbde36000a8f393a", "resultsYAG_NR.mat"),
            ("5dd7f10f6fc7690009d4d85c", "resultsYDR_NR.mat"),
            ("5dd7f14bfbde36000c8f59c3", "resultsYFS_NR.mat"),
            ("5dd7d9df215a8f000c4d2e56", "resultsYRH_NR.mat"),
            ("5dd7ee7dab905e000ce2e5c0", "resultsYSL_TSR.mat"),
            ("5dd7e2c3215a8f000c4d3617", "resultsYAG_TSR.mat"),
            ("5dd7e43d215a8f00094cfe7b", "resultsYFS_TSR.mat"),
            ("5dd7e2546fc769000bd4f34c", "resultsYAK_TSR.mat"),
            ("5dd7e7e66fc769000cd4e7f4", "resultsYRH_TSR.mat"),
            ("5dd7e733ab905e000ae2b97a", "resultsYLS_TSR.mat"),
            ("5dd7ed116fc769000bd4ff7c", "resultsYTL_TSR.mat"),
            ("5dd7e408fbde3600098f259a", "resultsYFR_TSR.mat"),
            ("5dd7ed40fbde36000b8f4379", "resultsYRK_TSR.mat"),
            ("5dd7ee17fbde36000c8f5654", "resultsYRP_TSR.mat"),
            ("5dd7e7bc215a8f000c4d3a21", "resultsYMS_TSR.mat"),
            ("5dd7e4cd215a8f000a4d0a3e", "resultsYIS_TSR.mat"),
            ("5dd7e4effbde36000b8f3abe", "resultsYHS_TSR.mat"),
            ("5dd7eab6215a8f00094d0279", "resultsYSD_TSR.mat"),
            ("5dd7e976ab905e000ce2de13", "resultsYMD_TSR.mat"),
            ("5dd7e1886fc769000ad4d41c", "resultsYDR_TSR.mat"),
            ("5dd7df3efbde36000c8f45ed", "resultsYAC_TSR.mat"),
            ("5dd7e1d46fc769000ad4d45b", "resultsYDG_TSR.mat"),
        ],
        "regenerate_yaml": False,
    },
    "narratives": {
        "source": "manual",
        "instructions": (
            "Narratives (Nastase et al., 2021; OpenNeuro ds002345). The loader\n"
            "reads surface-projected BOLD from the AFNI derivatives plus the word\n"
            "alignment used by Oota et al. (2023).\n\n"
            "1. From ds002345 (https://openneuro.org/datasets/ds002345, DataLad\n"
            "   or the download button), get the story's fsaverage6 GIfTI files\n"
            "   from derivatives/afni-nosmooth.\n"
            "2. From https://github.com/subbareddy248/lingprop-brain-alignment,\n"
            "   get stimulus_resources/{story}_align.csv.\n"
            "3. Lay out under datasets/fmri/original/narratives/ so that\n"
            "   afni-nosmooth/sub-*/func/ holds the GIfTI files and\n"
            "   stimulus_resources/{story}_align.csv exists (default story:\n"
            "   21styear)."
        ),
        "regenerate_yaml": False,
    },
    "natural_stories_maze": {
        "source": "manual",
        "instructions": (
            "Natural Stories Maze (Boyce & Levy, 2023).\n\n"
            "1. Clone https://github.com/vboyce/natural-stories-maze.\n"
            "2. Put Data/cleaned.rds under\n"
            "   datasets/maze/original/natural_stories_maze/.\n\n"
            "The loader reads the release directly and needs no conversion."
        ),
        "regenerate_yaml": False,
    },
}

# Fetch keys follow registry names; these aliases map the Shain et al. (2024)
# processed-package names and the per-language MECO names onto their fetch key.
ALIASES = {
    "dundee_shain2024logarithmic": "dundee",
    "brown_spr_shain2024logarithmic": "brown_spr",
    # Both RSC releases arrive in one download; the loader picks between them.
    "rsc_extended": "rsc",
}
ALIASES.update(
    {name: "meco_l1_w2" for name in _REGISTRY if name.startswith("meco_l1_w2_")}
)


def resolve_dataset_name(name: str) -> str:
    """Map a fetch alias to its canonical key.

    Args:
        name: Alias or catalog key; a key that is not an alias is returned
            unchanged, whether or not the catalog holds it.
    """
    return ALIASES.get(name, name)


def _release_page(repo: str) -> str:
    """GitHub page of the tagged release that a study dependency downloads."""
    (dependency,) = EXTERNAL_REPOS[repo]
    github_repo, tag, _asset = dependency.source.removeprefix("release:").split(":")
    return f"https://github.com/{github_repo}/releases/tag/{tag}"


def provenance_paths_for(name: str) -> dict[str, str]:
    """Return local filename -> stable upstream locator for a fetch entry.

    This derives provenance from the same identifiers used by the downloader,
    so tests can detect drift between acquisition configuration and YAML.
    Archive locators use a URL fragment to identify the extracted member.

    Args:
        name: Catalog key or fetch alias. Sources with no addressable upstream
            file, such as manual downloads, contribute only whatever the entry
            declares under ``provenance``.
    """
    name = resolve_dataset_name(name)
    config = DATASETS[name]
    source = config["source"]
    paths: dict[str, str] = dict(config.get("provenance", {}))

    if source == "osf":
        paths.update(
            {
                filename: f"https://osf.io/download/{file_id}/"
                for file_id, filename in config["files"]
            }
        )
    elif source == "osf_zip":
        paths.update(
            {
                filename.removesuffix(".zip"): (
                    f"https://osf.io/download/{file_id}/"
                    f"#member={filename.removesuffix('.zip')}"
                )
                for file_id, filename in config["files"]
            }
        )
    elif source == "github_raw":
        revision = config.get("revision", config["branch"])
        paths.update(
            {
                filename: (
                    f"https://raw.githubusercontent.com/{config['repo']}/"
                    f"{revision}/{remote_path}"
                )
                for remote_path, filename in config["files"]
            }
        )
    elif source == "direct_url":
        paths.update({filename: url for url, filename in config["files"]})
    elif source == "zip":
        paths.update(
            {
                local_name: f"{config['url']}#member={archive_name}"
                for archive_name, local_name in config["files"].items()
            }
        )
    elif source == "study_release":
        release = _release_page(config["repo"])
        paths.update(
            {
                filename: f"{release}#member={member}"
                for member, filename in config["files"]
            }
        )

    paths.update(
        {
            filename: f"https://osf.io/download/{file_id}/"
            for file_id, filename in config.get("extra_osf_files", [])
        }
    )
    paths.update({filename: url for url, filename in config.get("extra_urls", [])})
    return paths


def dataset_directory(name: str) -> str:
    """Return the on-disk subdirectory for a fetch key.

    The loader registry owns paths for registered datasets. Shared bundles and
    resources not tied to one registry entry use the directory in this catalog.

    Args:
        name: Catalog key, already resolved through :func:`resolve_dataset_name`.

    Returns:
        The path relative to the data root, such as
        ``eye-tracking/original/meco``.
    """
    spec = _REGISTRY.get(name)
    return spec.default_subdir if spec is not None else DATASETS[name]["dir"]
