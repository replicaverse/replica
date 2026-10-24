"""Brothers & Kuperberg (2021): critical-region SPR corpus.

*Word predictability effects are linear, not logarithmic.* Brothers, T.,
& Kuperberg, G. R. (2021). Journal of Memory and Language.
https://doi.org/10.1016/j.jml.2020.104174

Design: 216 items x 3 cloze conditions (LC / MC / HC) presented to 216
participants self-paced word-by-word. Each (item, condition) is a
distinct sentence frame around a paper-fixed critical word; the three
conditions differ in the context before the critical word, which sets its
cloze probability. The released DV ``SUM_3RT`` sums the self-paced RTs of
the critical word and the two words after it, so each (participant, item,
condition) cell carries a single behavioural number. This differs from the other SPR datasets (Natural
Stories, Brown SPR, UCL SPR), which expose per-word RTs.

Loader convention: 216 x 3 = 648 Stimuli, each carrying the full
sentence text from the BK paper's appendix and a per-word ``words``
layer. Per-(subject, item, condition) Datum carries a NaN-padded value
array with the single finite SUM_3RT placed at the critical word's
position in the words layer.

Data files (fetch via
``replica fetch brothers_kuperberg``):

  * ``SPRT_LogLin_216.csv``, main per-(subject, item) table.
  * ``Stimuli_Appendix_format.docx``, full sentence frames (BK paper
    appendix, hosted on the OSF mirror at 6wvqe).
  * ``AppendixB.csv``, supplementary per-word lexical norms.
  * ``LogLin_meta.csv``, ``LogLin_Naming.csv``, supplementary stats.
"""

import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from replica.dataset.base import SPRDataset
from replica.dataset.contracts import DatasetInfo
from replica.dataset.metadata import io as metadata_io

from ...core import Datum, Stimulus

DATA_FILE = "SPRT_LogLin_216.csv"
_ConditionTriplet = dict[str, tuple[str, int]]
_StimulusByItemCondition = dict[tuple[int, str], Stimulus]


class BrothersKuperberg(SPRDataset):
    """Brothers & Kuperberg (2021) critical-region SPR corpus.

    Default measure is ``sum_3rt`` (the published DV: sum of self-paced
    reading times across a released 3-word critical region).

    The released CSV's ``SUM_3RT`` contains integer millisecond sums no higher
    than 4500. The loader exposes these as ``sum_3rt`` and retains the separate
    ``SUM_3RT_trimmed`` column as ``sum_3rt_trimmed``. That column clips values
    to 300 ms and the participant's condition mean plus 2.5 standard deviations. Each
    (item, condition) is a Stimulus carrying the full sentence frame
    (parsed from the paper's appendix docx) and a per-word ``words``
    layer. Metadata fields: ``item``, ``condition`` (LC/MC/HC),
    ``critical_word``, ``critical_position`` (1-indexed in the words
    layer), ``cloze``, ``log_cloze``, ``trigram``, ``log_trigram``.
    """

    info = DatasetInfo(name="brothers_kuperberg", language="en")
    _declared_measures = ("sum_3rt", "sum_3rt_trimmed")
    # SUM_3RT is the paper's summed-RT DV at a single point (the
    # critical word). It is not additive across other word positions in
    # the sentence, those carry NaN, so the dataset declares no
    # additive measures.
    additive_measures: frozenset[str] = frozenset()

    def __init__(self) -> None:
        """Select the paper's summed-RT DV as the default measure."""
        super().__init__()
        self.default_measure = "sum_3rt"

    def exclusion_mask(self, datum: Datum) -> np.ndarray:
        """Keep only the critical word, the one finite entry of each datum.

        The parent mask drops position 0 of every datum (no spillover
        baseline), and a critical word can sit at position 0.

        Args:
            datum: One (subject, item, condition) trial, NaN everywhere but the
                critical word.

        Returns:
            A per-word boolean mask, True only where the value is finite.
        """
        return ~np.isnan(datum.values)

    def load(self, path: str) -> None:
        """Load from the main CSV plus the stimuli docx.

        Required files (fetch with ``replica fetch brothers_kuperberg``):

          * ``SPRT_LogLin_216.csv``, per-(subject, item) summed RT,
            paper predictors (cloze, trigram), critical word + position.
          * ``Stimuli_Appendix_format.docx``, the paper's sentence
            appendix, parsed for the ``H:/M:/L: <sentence> (NN%)`` frames.

        Args:
            path: Directory holding both files.
        """
        self._load_metadata(path, metadata_path=metadata_io.find_dataset_yaml(path))
        csv_path = Path(path) / DATA_FILE
        docx_path = Path(path) / "Stimuli_Appendix_format.docx"
        if not csv_path.exists():
            raise FileNotFoundError(
                f"BrothersKuperberg: {DATA_FILE} not found at {csv_path}.\n"
                "Run: replica fetch brothers_kuperberg"
            )
        if not docx_path.exists():
            raise FileNotFoundError(
                f"BrothersKuperberg: Stimuli_Appendix_format.docx not "
                f"found at {docx_path}. This file (the BK paper's "
                f"sentence appendix) is needed to recover per-trial "
                f"sentence frames for LM-context predictors. Re-fetch: "
                f"replica fetch brothers_kuperberg"
            )
        self._load_from_csv(csv_path, docx_path)

    @staticmethod
    def _parse_stimuli_docx(
        docx_path: Path,
    ) -> list[dict[str, tuple[str, int]]]:
        """Parse the stimuli appendix into a list of 216 triplets, each a
        dict ``{H/M/L → (sentence, cloze%)}`` in docx order.

        The docx body is a flat sequence of paragraphs matching the
        pattern ``[*] {H|M|L}: <sentence>. (NN%)``. Three consecutive
        matches form one (item, H, M, L) triplet.
        """

        with zipfile.ZipFile(docx_path) as z:
            xml = z.read("word/document.xml").decode("utf-8")
        text = re.sub(r"<w:p[^/]*/>", "\n", xml)
        text = re.sub(r"</w:p>", "\n", text)
        text = re.sub(r"<[^>]+>", "", text)

        pat = re.compile(
            r"^\s*\*?\s*([HML])\s*:\s*(.+?)\s*\((\d+)%\)\s*$", re.MULTILINE
        )
        matches = pat.findall(text)
        triplets: list[dict[str, tuple[str, int]]] = []
        for i in range(0, len(matches), 3):
            chunk = matches[i : i + 3]
            if len(chunk) != 3:
                continue
            by_cond = {c: (s, int(p)) for c, s, p in chunk}
            if set(by_cond.keys()) != {"H", "M", "L"}:
                continue
            triplets.append(by_cond)
        return triplets

    @staticmethod
    def _tokenize_sentence(sentence: str) -> list[str]:
        """Tokenise a BK appendix sentence into per-word tokens.

        The docx ships hyphenated words (e.g. ``fifth-graders``,
        ``first-aid``) and uses curly apostrophes (``fiancée's``).
        Whitespace-tokenised with outer punctuation stripped, so each token
        matches the word-level granularity and the main CSV's
        ``position`` field.
        """
        strip_chars = '.,;:?!"()' + "''“”"
        return [t for t in (w.strip(strip_chars) for w in sentence.split()) if t]

    def _load_from_csv(self, csv_path: Path, docx_path: Path) -> None:
        df = pd.read_csv(csv_path)
        triplets = self._parse_stimuli_docx(docx_path)
        per_item = self._match_items_to_triplets(df, triplets)
        stim_by_pair = self._build_stimuli(df, per_item)
        self._append_behavioral_data(df, stim_by_pair)

    def _match_items_to_triplets(
        self, df: pd.DataFrame, triplets: list[_ConditionTriplet]
    ) -> dict[int, _ConditionTriplet]:
        # A triplet matches an item iff its H/M/L sentences each carry
        # the item's critical word at the 1-indexed `position` slot.
        # Some critical words recur and two sentences can share a slot
        # (e.g. "a small igloo": "igloo"@12 fits one triplet, "small"@11
        # fits two), so a greedy first-match lets the looser item take
        # the constrained item's frame. Solve as a bipartite matching so
        # every triplet is consumed by exactly one item.
        item_keys = (
            df[["ITEM", "critical_word", "position"]]
            .drop_duplicates()
            .sort_values("ITEM")
        )

        def matches(triplet: _ConditionTriplet, cw: str, pos: int) -> bool:
            for cl in ("H", "M", "L"):
                words = self._tokenize_sentence(triplet[cl][0])
                if pos - 1 >= len(words) or words[pos - 1].lower() != cw:
                    return False
            return True

        candidates: dict[int, list[int]] = {}
        item_key: dict[int, tuple[str, int]] = {}
        for _, row in item_keys.iterrows():
            iid = int(row["ITEM"])
            cw = str(row["critical_word"]).lower()
            pos = int(row["position"])  # 1-indexed
            item_key[iid] = (cw, pos)
            candidates[iid] = [
                ti for ti, t in enumerate(triplets) if matches(t, cw, pos)
            ]

        # Kuhn's augmenting-path matching: triplet index -> owning item.
        triplet_owner: dict[int, int] = {}

        def assign(iid: int, seen: set[int]) -> bool:
            for ti in candidates[iid]:
                if ti in seen:
                    continue
                seen.add(ti)
                if ti not in triplet_owner or assign(triplet_owner[ti], seen):
                    triplet_owner[ti] = iid
                    return True
            return False

        unmatched_items: list[tuple[int, str, int]] = []
        for iid in candidates:
            if not assign(iid, set()):
                cw, pos = item_key[iid]
                unmatched_items.append((iid, cw, pos))

        if unmatched_items:
            raise RuntimeError(
                "BrothersKuperberg: could not match these items between "
                "SPRT_LogLin_216.csv and Stimuli_Appendix_format.docx: "
                f"{unmatched_items[:10]}"
            )
        return {iid: triplets[ti] for ti, iid in triplet_owner.items()}

    def _build_stimuli(
        self, df: pd.DataFrame, per_item: dict[int, _ConditionTriplet]
    ) -> _StimulusByItemCondition:
        cond_letter = {"HC": "H", "MC": "M", "LC": "L"}
        self._stimuli: list[Stimulus] = []
        self._data: list[Datum] = []
        stim_by_pair: _StimulusByItemCondition = {}
        missing_items: list[int] = []

        for (item_id, cond), grp in df.groupby(["ITEM", "condition"]):
            stimulus = self._stimulus_from_group(
                int(item_id), str(cond), grp, per_item, cond_letter
            )
            if stimulus is None:
                missing_items.append(int(item_id))
                continue
            stim_by_pair[(int(item_id), str(cond))] = stimulus

        if missing_items:
            # Hard fail, silent attrition here would drop observations the
            # released CSV provides.
            raise RuntimeError(
                f"BrothersKuperberg: failed to match {len(missing_items)} "
                f"(item, condition) pair(s) between SPRT_LogLin_216.csv "
                f"and Stimuli_Appendix_format.docx. First few item IDs: "
                f"{missing_items[:5]}"
            )
        return stim_by_pair

    def _stimulus_from_group(
        self,
        item_id: int,
        cond: str,
        grp: pd.DataFrame,
        per_item: dict[int, _ConditionTriplet],
        cond_letter: dict[str, str],
    ) -> Stimulus | None:
        first_row = grp.iloc[0]
        critical_word = str(first_row["critical_word"])
        crit_pos = int(first_row["position"])  # 1-indexed

        triplet = per_item.get(item_id)
        cl = cond_letter.get(cond)
        if triplet is None or cl is None or cl not in triplet:
            return None

        sentence, _docx_cloze = triplet[cl]
        words = self._tokenize_sentence(sentence)
        crit_idx = crit_pos - 1
        if not (0 <= crit_idx < len(words)):
            return None
        if words[crit_idx].lower() != critical_word.lower():
            return None

        return self._make_stimulus(
            words,
            f"bk_{item_id:03d}_{cl}",
            {
                "item": item_id,
                "condition": cond,
                "critical_word": critical_word,
                "critical_position": crit_pos,
                "cloze": float(first_row["cloze"]),
                "log_cloze": float(first_row["log_cloze"]),
                "trigram": float(first_row["trigram"]),
                "log_trigram": float(first_row["log_trigram"]),
            },
        )

    def _append_behavioral_data(
        self, df: pd.DataFrame, stim_by_pair: _StimulusByItemCondition
    ) -> None:
        # Attach behavioural data: one Datum per (subject, item,
        # condition, measure) with SUM_3RT placed at the critical
        # word's index in the words layer; other positions are NaN.
        for _, row in df.iterrows():
            stim = stim_by_pair.get((int(row["ITEM"]), str(row["condition"])))
            if stim is None:
                continue
            crit_idx = int(row["position"]) - 1
            n_words = len(stim.layers["words"])
            if not (0 <= crit_idx < n_words):
                continue
            subj = str(int(row["SUB"]))
            for measure_name, source_col in (
                ("sum_3rt", "SUM_3RT"),
                ("sum_3rt_trimmed", "SUM_3RT_trimmed"),
            ):
                v = row.get(source_col)
                if v is None or pd.isna(v):
                    continue
                values = np.full(n_words, np.nan)
                values[crit_idx] = float(v)
                self._data.append(
                    Datum(
                        stimulus=stim,
                        measure_name=measure_name,
                        values=values,
                        participant_id=subj,
                    )
                )
