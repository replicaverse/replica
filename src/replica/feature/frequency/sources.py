"""FrequencySource implementations backed by tables, count files, and wordfreq.

Concrete classes for the
:class:`~replica.feature.core.protocols.FrequencySource` protocol, which lives
in :mod:`..core.protocols`. The sampled (LM-derived) source lives in
:mod:`.sampled`.
"""

import csv
from pathlib import Path

import numpy as np


class UserTable:
    """FrequencySource backed by a user-supplied dict of unigram probabilities.

    ``table`` maps a unit string to its probability p(unit) in the linear
    domain. Missing units return NaN.

    Args:
        table: Unit string -> p(unit) in the linear domain. Values outside
            ``(0, 1]`` are rejected; NaN marks a unit as known-unknown.
        language: ISO 639-1 code the source reports, which supplies the
            language suffix of features built on it.
    """

    def __init__(self, table: dict[str, float], language: str = "en"):
        """Check every value is a probability, then keep the table."""
        # Validate the contract up front: values are probabilities in (0, 1]
        # (NaN allowed for "unknown"). Catches the common mistake of passing
        # log-frequencies, which would otherwise silently produce all-NaN
        # predictors downstream (log10 of a negative).
        for unit, p in table.items():
            if not (np.isnan(p) or 0.0 < p <= 1.0):
                raise ValueError(
                    f"UserTable values are unigram probabilities p(unit) in (0, 1]; "
                    f"got {p!r} for {unit!r}. A FrequencySource returns a probability, "
                    f"not a log-frequency."
                )
        self.language = language
        self._table = table

    def probability(self, unit: str) -> float:
        """p(unit) as the table gives it, NaN when it holds no entry.

        Args:
            unit: Surface form, matched against the keys exactly as passed.
        """
        return self._table.get(unit, np.nan)


class WikiText103Source:
    """FrequencySource for Pimentel et al. (2023)'s WikiText-103 unigram table.

    The released ``unigrams.csv`` maps a lowercased Moses token to its natural
    log-probability. A unit is scored the authors' way (``unigram.py``):
    strip punctuation, Moses-tokenize, detokenize each token alone, and sum
    the tokens' surprisals; any out-of-vocabulary token makes the unit NaN.
    :meth:`probability` returns ``exp(log p)`` of the summed quantity so the
    feature layer's log conversion reproduces the authors' value exactly.

    Args:
        path: the authors' ``corpora/unigrams.csv``.
        language: ISO 639-1 code; the released table is English, so anything
            other than ``en`` is rejected.
    """

    def __init__(self, path: str, language: str = "en"):
        """Read the released unigram table into memory."""
        if language != "en":
            raise ValueError(
                f"WikiText-103 unigrams are English only; got {language!r}"
            )
        if not Path(path).exists():
            raise FileNotFoundError(
                f"WikiText-103 unigram table not found at {path}; it ships with "
                "the pinned anticipation-on-reading-times checkout."
            )
        self.language = language
        with open(path, encoding="utf8") as fh:
            self._log_p = {row[0]: float(row[1]) for row in csv.reader(fh)}
        self._tokenize = None
        self._detokenize = None

    def _moses(self):
        if self._tokenize is None or self._detokenize is None:
            import mosestokenizer

            self._tokenize = mosestokenizer.MosesTokenizer("en")
            self._detokenize = mosestokenizer.MosesDetokenizer("en")
        return self._tokenize, self._detokenize

    def probability(self, unit: str) -> float:
        """``exp`` of the summed token log-probabilities for one unit.

        Args:
            unit: Surface form; surrounding whitespace and punctuation come off
                before Moses tokenization.

        Returns:
            NaN when the unit is empty after stripping, or when any of its
            tokens is out of the table's vocabulary.
        """
        import string

        word = unit.lower().strip().strip(string.punctuation)
        if not word:
            return float("nan")
        tokenize, detokenize = self._moses()
        total = 0.0
        for token in tokenize(word):
            log_p = self._log_p.get(detokenize([token]))
            if log_p is None:
                return float("nan")
            total += log_p
        return float(np.exp(total))


class WordfreqSource:
    """FrequencySource backed by the ``wordfreq`` PyPI package.

    Multi-language unigram frequencies from web crawls, subtitles,
    Wikipedia, and Twitter for 40+ languages; data ships with the library.

    :meth:`probability` returns ``p(word)``; unknown words return ``NaN``
    (instead of wordfreq's default ``0.0``) for easy filtering.

    Args:
        language: ISO 639-1 code; pass the same code MECO and other
            multilingual corpora use.
        wordlist: ``"best"`` (default) picks the most accurate available
            wordlist per language.
    """

    #: wordfreq publishes one merged Serbo-Croatian wordlist; the standard
    #: language codes resolve to it.
    MACROLANGUAGE = {"sr": "sh", "hr": "sh", "bs": "sh"}

    def __init__(self, language: str = "en", wordlist: str = "best"):
        """Bind the wordfreq wordlist the lookups read from."""
        import wordfreq

        self._wordfreq = wordfreq
        self.language = self.MACROLANGUAGE.get(language, language)
        self.wordlist = wordlist

    def probability(self, unit: str) -> float:
        """p(unit) in the configured language, NaN when the word is unknown.

        Args:
            unit: Word to look up; wordfreq applies its own per-language
                normalization before matching.
        """
        # ``wordfreq.word_frequency`` returns 0.0 for unknown words.
        p = self._wordfreq.word_frequency(unit, self.language, wordlist=self.wordlist)
        return float(p) if p > 0.0 else float("nan")


class KenLMSource:
    """FrequencySource backed by a KenLM model file (``.arpa`` or ``.binary``).

    For a unigram (order 1) model, ``score(word, bos=False, eos=False)``
    returns ``log10 P(word)`` under Kneser-Ney smoothing, the quantity
    Clark et al. (2025) use as the ``unigram_surprisal`` baseline (trained
    on whitespace-tokenized OWT). A higher-order model scores a
    lone word off its unigram tier, which under modified Kneser-Ney is a
    continuation probability (mass proportional to the number of distinct
    contexts the word follows) rather than the marginal, so order 1 is the
    one that yields a frequency.
    :meth:`probability` exponentiates that score back to ``P(word)``.

    Args:
        path: ``.arpa`` or ``.binary`` (binary loads faster; build via
            ``build_binary <arpa> <binary>``).
        language: ISO 639-1 code the source reports, which supplies the
            language suffix of features built on it; scoring ignores it.
        bos_eos: If True, KenLM adds <s>/</s> before scoring; affects only
            orders >= 2.
    """

    # KenLM scores the literal token; OWT was tokenized case-preserving, so
    # ``The`` must not be folded to ``the`` before lookup.
    case_sensitive = True

    def __init__(self, path: str, language: str = "en", bos_eos: bool = False):
        """Load the KenLM model file that answers the lookups."""
        try:
            import kenlm
        except ImportError as e:
            raise ImportError(
                "KenLMSource requires the `kenlm` package. "
                "Install with: pip install kenlm"
            ) from e
        if not Path(path).exists():
            raise FileNotFoundError(
                f"KenLM model not found at {path}. Train a unigram model with "
                "KenLM's lmplz + build_binary, or point this at an existing "
                "model file."
            )
        self.language = language
        self._bos_eos = bool(bos_eos)
        # ``kenlm.Model`` loads .arpa and .binary alike (binary is faster).
        self._model = kenlm.Model(str(path))

    def probability(self, unit: str) -> float:
        """P(unit) read off the model's unigram tier.

        Args:
            unit: Token to score, in the case the model was trained on; an
                out-of-vocabulary token gets the model's ``<unk>`` mass.
        """
        # ``score`` is log10 of the model's unigram-tier probability for this
        # token; exponentiate to the base-free probability the protocol
        # returns. The feature layer logs it again.
        return float(
            10.0 ** self._model.score(unit, bos=self._bos_eos, eos=self._bos_eos)
        )


class OpenWebTextSource:
    r"""FrequencySource backed by a whitespace-tokenized OpenWebText count table.

    Used to reproduce Clark et al. (2025), whose unigram-surprisal baseline
    is a KenLM order-1 model on OWT. For in-vocab words this approximates
    that model with MLE counts; use :class:`KenLMSource` to read the ARPA
    model itself.

    Args:
        path: TSV ``word\\tcount`` rows; a ``# total_tokens=N`` comment
            line is honoured.
        language: ISO 639-1 code the source reports, which supplies the
            language suffix of features built on it.
        unk_smoothing: Return a small-but-finite probability for unknown
            words (``10 ** (-log10(total_tokens) - 1.0)``, i.e. rarer than
            any singleton) instead of NaN.
    """

    # Count keys are the case-preserving whitespace tokens of OWT; do not fold.
    case_sensitive = True

    def __init__(
        self,
        path: str,
        language: str = "en",
        unk_smoothing: bool = False,
    ):
        """Read the count table and fix the corpus total it normalizes by."""
        self.language = language
        self._unk_smoothing = bool(unk_smoothing)
        self._counts: dict[str, int] = {}
        total = None
        with open(Path(path)) as f:
            for line in f:
                # The header carries no count, so the tab is what separates it
                # from a vocabulary entry. Keying on a leading "#" instead
                # would discard every hashtag word, which on the published
                # OpenWebText table is 91,075 of its 18,469,939 types.
                if "\t" not in line:
                    if line.startswith("# total_tokens="):
                        total = int(line.split("=", 1)[1].strip())
                    continue
                w, c = line.rstrip("\n").split("\t", 1)
                try:
                    self._counts[w] = int(c)
                except ValueError:
                    continue
        if total is None:
            total = sum(self._counts.values())
        self._total = int(total)
        # Unknown-word probability: one decade below a singleton.
        self._unk_prob = float(10.0 ** (-np.log10(max(self._total, 1)) - 1.0))

    def probability(self, unit: str) -> float:
        """Relative frequency of *unit* over the corpus token total.

        Args:
            unit: Whitespace token in the case OpenWebText was tokenized in.

        Returns:
            ``count / total_tokens``; for an absent token, the smoothed
            unknown-word probability when *unk_smoothing* is set, else NaN.
        """
        c = self._counts.get(unit, 0)
        if c <= 0:
            return self._unk_prob if self._unk_smoothing else float("nan")
        return float(c / self._total)


# Per-language column conventions for SUBTLEX-* TSV files: an ISO 639-1 code
# maps to (word_column, frequency_column, kind). ``kind`` is what the frequency
# column holds, since the releases differ: SUBTLEX-US publishes a per-million
# column, while the de/nl/es releases publish raw occurrence counts. Reading a
# count as if it were per-million overstates p by the corpus size in millions,
# so each convention records the column's kind beside its name.
_SUBTLEX_COLUMN_CONVENTIONS: dict[str, tuple[str, str, str]] = {
    "en": ("Word", "SUBTLWF", "per_million"),
    "de": ("Word", "WFfreqcount", "count"),
    "nl": ("Word", "FREQcount", "count"),
    "es": ("Word", "Freq.count", "count"),
}


class SubtlexSource:
    """FrequencySource backed by a SUBTLEX-style frequency table.

    SUBTLEX is a family of psycholinguistic frequency norms derived from
    movie/TV subtitles, published per language (US, UK, DE, NL, ES, ...).

    Yields ``p(word)`` from either kind of frequency column the SUBTLEX
    releases publish: a per-million column becomes ``freq · 10⁻⁶``, a raw
    count is divided by the summed count over the table, which is how the
    per-million columns are themselves derived.

    :data:`_SUBTLEX_COLUMN_CONVENTIONS` supplies the per-language column and
    its kind. Pass ``freq_per_million_col`` or ``count_col`` to override, which
    also declares which kind the column is.

    Args:
        path: The SUBTLEX release as a tab-separated table with a header row.
        language: ISO 639-1 code selecting the column convention; a language
            with no convention falls back to the SUBTLEX-US layout.
        word_col: Defaults to the per-language convention.
        freq_per_million_col: A column of occurrences per million words.
        count_col: A column of raw occurrence counts, normalized by their sum.
    """

    def __init__(
        self,
        path: str | Path,
        language: str = "en",
        word_col: str | None = None,
        freq_per_million_col: str | None = None,
        count_col: str | None = None,
    ):
        """Read the release and convert its frequency column to p(word)."""
        if freq_per_million_col and count_col:
            raise ValueError(
                "Pass freq_per_million_col or count_col, not both; they "
                "declare which kind of column to read."
            )
        # Defaults carry the kind, so an unnamed column is never assumed to be
        # per-million. Unknown languages fall back to the SUBTLEX-US layout.
        default_word, default_freq, default_kind = _SUBTLEX_COLUMN_CONVENTIONS.get(
            language, _SUBTLEX_COLUMN_CONVENTIONS["en"]
        )
        if freq_per_million_col:
            freq_col, kind = freq_per_million_col, "per_million"
        elif count_col:
            freq_col, kind = count_col, "count"
        else:
            freq_col, kind = default_freq, default_kind
        word_col = word_col or default_word

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"SUBTLEX table not found at {path}. Download SUBTLEX-"
                f"{language.upper()} from the SUBTLEX project page "
                "and place it in $PSYCHO_DATA_DIR/frequency/."
            )
        self.language = language
        self._table: dict[str, float] = {}
        # SUBTLEX TSV files are typically Latin-1 encoded; some have a
        # UTF-8 BOM. Try utf-8-sig first (handles both) then fall back.
        encoding_tried: list[str] = []
        for encoding in ("utf-8-sig", "latin-1"):
            try:
                with open(path, newline="", encoding=encoding) as f:
                    reader = csv.DictReader(f, delimiter="\t")
                    fieldnames = reader.fieldnames or ()
                    if word_col not in fieldnames or freq_col not in fieldnames:
                        raise ValueError(
                            f"SUBTLEX TSV at {path} is missing required "
                            f"columns. Expected {word_col!r} and "
                            f"{freq_col!r}; got {reader.fieldnames!r}."
                        )
                    raw: dict[str, float] = {}
                    for row in reader:
                        w = row[word_col]
                        v = row[freq_col]
                        if not w or not v:
                            continue
                        try:
                            value = float(v.replace(",", "."))
                        except ValueError:
                            continue
                        if value <= 0:
                            continue
                        raw[w.lower()] = value
                    if kind == "count":
                        total = sum(raw.values())
                        if total <= 0:
                            raise ValueError(
                                f"SUBTLEX column {freq_col!r} at {path} summed "
                                "to zero; it holds no usable counts."
                            )
                        self._table = {w: c / total for w, c in raw.items()}
                    else:
                        self._table = {w: v * 1e-6 for w, v in raw.items()}
                        # A per-million column cannot put p above 1; if it
                        # does, the column is a raw count mislabelled.
                        offenders = [w for w, p in self._table.items() if p > 1.0]
                        if offenders:
                            raise ValueError(
                                f"SUBTLEX column {freq_col!r} yielded p > 1 for "
                                f"{offenders[0]!r}; it is not per-million "
                                "(likely a raw count). Pass it as count_col."
                            )
                break
            except UnicodeDecodeError:
                encoding_tried.append(encoding)
                continue
        if not self._table and encoding_tried:
            raise ValueError(
                f"SUBTLEX table at {path} could not be decoded "
                f"(tried: {encoding_tried})."
            )

    def probability(self, unit: str) -> float:
        """p(unit) from the norms, NaN when the release has no such word.

        Args:
            unit: Word to look up; the table is keyed by its lowercase form.
        """
        return self._table.get(unit.lower(), float("nan"))
