"""CV splitter protocol and built-in implementations."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

from replica.analysis.contracts.configuration import array_identity, component_options

if TYPE_CHECKING:
    from ...dataset import Dataset


@runtime_checkable
class CVSplitter(Protocol):
    """Strategy for splitting observations into train/test folds.

    ``split`` receives the built flat arrays and yields
    ``(train_mask, test_mask)`` boolean-array pairs.

    ``needs_sentence_ids`` is an optional capability, read by ``evaluate()``
    with the default declared here: a splitter that turns it on partitions
    by sentence and is called as ``split(y, groups, participants, items,
    sentence_ids=..., participant_ids=...)``, the per-row sentence ids and
    raw participant ids added. A splitter may also expose
    ``fold_metadata()``, a list with one dict per yielded fold that
    ``evaluate()`` attaches to the matching :class:`FoldResult`.
    """

    def split(
        self,
        y: np.ndarray,
        groups: np.ndarray,
        participants: np.ndarray,
        items: np.ndarray,
    ) -> Iterable[tuple[np.ndarray, np.ndarray]]:
        """Yield one ``(train_mask, test_mask)`` pair per fold.

        Args:
            y: Transformed dependent variable, one entry per observation; its
                length fixes the length of every mask.
            groups: Position of each row's stimulus in the evaluated stimulus
                list, so all rows of one stimulus share a value.
            participants: Integer code of the reader who produced each row.
            items: Integer code of each row's stimulus, numbered in the order
                stimuli first contribute rows.

        Returns:
            An iterator of mask pairs, each array of length ``len(y)``, marking
            the fitted and the held-out rows. A splitter may leave a row out
            of both.
        """
        ...


@runtime_checkable
class SentenceAwareSplitter(Protocol):
    """Optional splitter capability using sentence and raw participant IDs."""

    needs_sentence_ids: bool

    def split(
        self,
        y: np.ndarray,
        groups: np.ndarray,
        participants: np.ndarray,
        items: np.ndarray,
        *,
        sentence_ids: np.ndarray | None = None,
        participant_ids: np.ndarray | None = None,
    ) -> Iterable[tuple[np.ndarray, np.ndarray]]:
        """Yield folds using the additional row identities.

        Args:
            y: Dependent variable.
            groups: Stimulus groups.
            participants: Encoded participant IDs.
            items: Encoded item IDs.
            sentence_ids: Sentence identifier per observation.
            participant_ids: Raw participant identifier per observation.
        """
        ...


class ItemSplitter(CVSplitter):
    """k-fold CV by stimulus group (item).

    All observations for a stimulus land in the same fold.
    This is the standard psycholinguistic evaluation strategy.
    """

    def configuration(self) -> dict:
        """Return declared options without fitted state."""
        return {
            "n_folds": self.n_folds,
        }

    def __init__(self, n_folds: int = 10):
        """Configure the number of folds.

        Args:
            n_folds: Folds to produce. Capped at the number of distinct
                stimuli, so a corpus with fewer stimuli yields fewer folds.
        """
        if n_folds < 2:
            raise ValueError(
                "n_folds must be at least 2; use FullDataSplitter for in-sample scoring"
            )
        self.n_folds = n_folds

    def split(
        self,
        y: np.ndarray,
        groups: np.ndarray,
        participants: np.ndarray,
        items: np.ndarray,
    ) -> Iterable[tuple[np.ndarray, np.ndarray]]:
        """Deal the distinct stimuli round-robin across folds.

        Args:
            y: Dependent variable, unused: fold membership follows the
                stimulus.
            groups: Stimulus index per row; the sorted distinct values are
                dealt to folds and every row of a stimulus follows it.
            participants: Reader code per row, unused.
            items: Stimulus code per row, unused.

        Returns:
            One ``(train_mask, test_mask)`` pair per fold, the test mask
            selecting the rows of that fold's stimuli.
        """
        unique_groups = np.unique(groups)
        n = min(self.n_folds, len(unique_groups))
        assignments = np.array([i % n for i in range(len(unique_groups))])
        for fold in range(n):
            test_groups = unique_groups[assignments == fold]
            test_mask = np.isin(groups, test_groups)
            yield ~test_mask, test_mask


class RowSplitter(CVSplitter):
    """k-fold CV by sequential row assignment.

    Each observation is assigned to fold ``i % n_folds`` based on its
    position in the flat array. Order depends on how the Dataset and
    evaluate() build that array (by stimulus, then by datum within stimulus).
    """

    def configuration(self) -> dict:
        """Return declared options without fitted state."""
        return {
            "n_folds": self.n_folds,
        }

    def __init__(self, n_folds: int = 10):
        """Configure the number of folds.

        Args:
            n_folds: Folds to produce; row ``i`` is held out in fold
                ``i % n_folds``.
        """
        if n_folds < 2:
            raise ValueError(
                "n_folds must be at least 2; use FullDataSplitter for in-sample scoring"
            )
        self.n_folds = n_folds

    def split(
        self,
        y: np.ndarray,
        groups: np.ndarray,
        participants: np.ndarray,
        items: np.ndarray,
    ) -> Iterable[tuple[np.ndarray, np.ndarray]]:
        """Assign every row to fold ``i % n_folds`` by its position.

        Args:
            y: Dependent variable; only its length is read, to number the rows.
            groups: Stimulus index per row, unused: rows of one stimulus are
                spread across folds.
            participants: Reader code per row, unused.
            items: Stimulus code per row, unused.

        Returns:
            One ``(train_mask, test_mask)`` pair per fold.
        """
        folds = np.array([i % self.n_folds for i in range(len(y))])
        for fold in range(self.n_folds):
            test_mask = folds == fold
            yield ~test_mask, test_mask


class ShuffledKFoldSplitter(CVSplitter):
    """k-fold CV with random shuffling, matching sklearn's KFold.

    Shuffles observation indices using the given random seed, then
    splits into ``n_folds`` consecutive chunks of the shuffled order.
    """

    def configuration(self) -> dict:
        """Return declared options without fitted state."""
        return {
            "n_folds": self.n_folds,
            "seed": self.seed,
        }

    def __init__(self, n_folds: int = 10, seed: int = 42):
        """Configure the number of folds and the shuffle.

        Args:
            n_folds: Folds to produce, each one a consecutive chunk of the
                shuffled row order.
            seed: Seed of the ``RandomState`` that permutes the row indices,
                so the same seed reproduces the same fold membership.
        """
        if n_folds < 2:
            raise ValueError(
                "n_folds must be at least 2; use FullDataSplitter for in-sample scoring"
            )
        self.n_folds = n_folds
        self.seed = seed

    def split(
        self,
        y: np.ndarray,
        groups: np.ndarray,
        participants: np.ndarray,
        items: np.ndarray,
    ) -> Iterable[tuple[np.ndarray, np.ndarray]]:
        """Permute the rows, then cut them into consecutive held-out chunks.

        Args:
            y: Dependent variable; only its length is read, to build the
                permuted index.
            groups: Stimulus index per row, unused: rows of one stimulus are
                spread across folds.
            participants: Reader code per row, unused.
            items: Stimulus code per row, unused.

        Returns:
            One ``(train_mask, test_mask)`` pair per fold. The first
            ``n % n_folds`` folds hold out one extra row, as sklearn's
            ``KFold`` does.
        """
        n = len(y)
        indices = np.arange(n)
        np.random.RandomState(self.seed).shuffle(indices)

        # Match sklearn KFold: first (n % k) folds get one extra element
        fold_sizes: np.ndarray = np.full(self.n_folds, n // self.n_folds, dtype=int)
        fold_sizes[: n % self.n_folds] += 1

        current = 0
        for fold_size in fold_sizes:
            test_mask: np.ndarray = np.zeros(n, dtype=bool)
            test_mask[indices[current : current + fold_size]] = True
            yield ~test_mask, test_mask
            current += fold_size


class SequentialBlockSplitter(CVSplitter):
    """k-fold CV with contiguous blocks.

    Divides n observations into k roughly equal contiguous blocks using
    the same algorithm as R's ``cut(seq(1,n), breaks=k, labels=FALSE)``:
    equal-width intervals over the 1-indexed range, extended by 0.1% on
    each side.

    This is the standard approach in many R-based psycholinguistic analyses
    (e.g. Wilcox et al. (2023), Pimentel et al. (2023)).
    """

    def configuration(self) -> dict:
        """Return declared options without fitted state."""
        return {
            "n_folds": self.n_folds,
        }

    def __init__(self, n_folds: int = 10):
        """Configure the number of blocks.

        Args:
            n_folds: Contiguous blocks to cut the row order into, the ``breaks``
                argument of the R ``cut`` call this reproduces.
        """
        if n_folds < 2:
            raise ValueError(
                "n_folds must be at least 2; use FullDataSplitter for in-sample scoring"
            )
        self.n_folds = n_folds

    @staticmethod
    def _r_cut_folds(n: int, k: int) -> np.ndarray:
        """Match R's ``cut(seq(1,n), breaks=k, labels=FALSE)`` exactly.

        R's ``cut.default`` with a scalar ``breaks`` argument constructs:
          ``breaks <- seq.int(rx[1], rx[2], length.out = k + 1)``
          ``breaks[1]   <- rx[1] - dx / 1000``
          ``breaks[k+1] <- rx[2] + dx / 1000``
        where ``rx = range(x)`` and ``dx = diff(rx)``. Only the two outer
        endpoints are widened, every interior break stays on the
        un-widened linear grid.
        """
        data_range = n - 1
        extend = 0.001 * data_range
        breaks = np.linspace(1.0, float(n), k + 1)
        breaks[0] = 1.0 - extend
        breaks[-1] = float(n) + extend

        seq: np.ndarray = np.arange(1, n + 1, dtype=float)
        folds: np.ndarray = np.zeros(n, dtype=int)
        for i in range(n):
            for j in range(k):
                if j == 0:
                    if seq[i] >= breaks[j] and seq[i] <= breaks[j + 1]:
                        folds[i] = j
                        break
                elif seq[i] > breaks[j] and seq[i] <= breaks[j + 1]:
                    folds[i] = j
                    break
        return folds

    def split(
        self,
        y: np.ndarray,
        groups: np.ndarray,
        participants: np.ndarray,
        items: np.ndarray,
    ) -> Iterable[tuple[np.ndarray, np.ndarray]]:
        """Hold out one contiguous block of the row order per fold.

        Args:
            y: Dependent variable; only its length is read, to cut the blocks.
            groups: Stimulus index per row, unused: a block boundary may fall
                inside a stimulus.
            participants: Reader code per row, unused.
            items: Stimulus code per row, unused.

        Returns:
            One ``(train_mask, test_mask)`` pair per block, in row order.
        """
        folds = self._r_cut_folds(len(y), self.n_folds)
        for fold in range(self.n_folds):
            test_mask = folds == fold
            yield ~test_mask, test_mask


class SubjectSentenceMod10Splitter(CVSplitter):
    """Clark et al. (2025) partitioning: ``p = (m + n) mod k``.

    ``m`` is the integer-encoded subject and ``n`` is the per-word
    sentence number. ``n`` is sourced as follows, in order:

    1. ``sentence_ids`` passed to :meth:`split` (built by
       :func:`~replica.analysis.evaluation.evaluate.evaluate` from each
       stimulus's ``stim.metadata['per_word_sentence_ids']``);
    2. otherwise, the ``items`` array, i.e. the per-stimulus index
       supplied by the flat-array builder.
    """

    needs_sentence_ids = True

    def configuration(self) -> dict:
        """Return declared options without fitted state."""
        return {
            "n_folds": self.n_folds,
        }

    def __init__(self, n_folds: int = 10):
        """Configure the modulus of the partition.

        Args:
            n_folds: The ``k`` of ``(m + n) mod k``, so both the modulus and
                the number of folds yielded.
        """
        if n_folds < 2:
            raise ValueError(
                "n_folds must be at least 2; use FullDataSplitter for in-sample scoring"
            )
        self.n_folds = n_folds

    def split(
        self,
        y: np.ndarray,
        groups: np.ndarray,
        participants: np.ndarray,
        items: np.ndarray,
        sentence_ids: np.ndarray | None = None,
        participant_ids: np.ndarray | None = None,
    ) -> Iterable[tuple[np.ndarray, np.ndarray]]:
        """Hold out the rows whose ``(subject + sentence) mod k`` is the fold.

        Both terms are re-coded to dense sorted-unique codes, so the partition
        depends on the rank order of the ids rather than their absolute values.

        Args:
            y: Dependent variable, unused: fold membership is read off the ids.
            groups: Stimulus index per row, unused.
            participants: Reader code per row, the fallback subject term when
                ``participant_ids`` is absent.
            items: Stimulus code per row, the fallback sentence term when
                ``sentence_ids`` is absent.
            sentence_ids: Sentence number per row, the ``n`` term.
            participant_ids: Raw reader ids per row, preferred as the ``m``
                term so numeric subject columns sort numerically.

        Returns:
            One ``(train_mask, test_mask)`` pair per residue class.
        """

        # Dense sorted-unique codes of the (subject, sentence) pair, so the
        # partition is invariant to the absolute id/code values. Raw
        # ``participant_ids`` are preferred over the re-encoded
        # ``participants`` to match the study's np.unique-based fold_ids.
        def _codes(arr):
            arr = np.asarray(arr)
            # Numeric ids must sort numerically (matches a study's np.unique on
            # its raw int subject/sentence column); string ids sort as-is.
            try:
                arr = arr.astype(np.int64)
            except (ValueError, TypeError):
                pass
            return np.unique(arr, return_inverse=True)[1]

        m_src = participant_ids if participant_ids is not None else participants
        n_src = sentence_ids if sentence_ids is not None else items
        fold_ids = (_codes(m_src) + _codes(n_src)) % self.n_folds
        for fold in range(self.n_folds):
            test_mask = fold_ids == fold
            yield ~test_mask, test_mask


class ModulusPartitionSplitter(CVSplitter):
    """Shain & Schuler (2018) partitioning: one held-out block of ``splitID``.

    ``splitID`` is the sum of the integer codes of the split fields, here the
    participant and the sentence, and a partition into ``n`` parts assigns
    ``splitID % modulus <= modulus - n`` to the first part and each remaining
    residue to one part of its own. The first part is fitted and the named one
    held out, so this yields the single split the papers' experiments use
    rather than a cross-validation. Any remaining part belongs to neither,
    which is what leaves the development partition free for an
    early-stopping rule to watch.

    Codes come from tables the caller supplies, because the reference assigns
    them over the unfiltered impulse table: recomputing them from the rows an
    analysis kept would shift every code below a value the filters removed.
    """

    needs_sentence_ids = True

    def configuration(self) -> dict:
        """Return declared options without fitted state."""
        return {
            "participant_codes": self.participant_codes,
            "sentence_codes": self.sentence_codes,
            "modulus": self.modulus,
            "n_parts": self.n_parts,
            "held_out": self.held_out,
        }

    def __init__(
        self,
        participant_codes: dict[str, int],
        sentence_codes: dict[int, int],
        *,
        modulus: int = 4,
        n_parts: int = 3,
        held_out: int = 2,
    ):
        """Configure the code tables and the shape of the partition.

        Args:
            participant_codes: Integer code per raw participant id, keyed by
                its string form, assigned over the unfiltered impulse table.
            sentence_codes: Integer code per raw sentence number, keyed by its
                integer form, assigned over the same unfiltered table.
            modulus: Divisor applied to the summed codes, the ``modulus`` of
                the reference partition.
            n_parts: Parts the residues are cut into; the first part collects
                every residue up to ``modulus - n_parts``, each later part
                takes one residue.
            held_out: Index of the part used as the test split. Part 0 is
                always the fitted split, so ``held_out=0`` makes the two
                splits identical.

        Raises:
            ValueError: When ``held_out`` does not index one of ``n_parts``.
        """
        if not 0 <= held_out < n_parts:
            raise ValueError(f"held_out must index one of {n_parts} parts")
        self.participant_codes = participant_codes
        self.sentence_codes = sentence_codes
        self.modulus = modulus
        self.n_parts = n_parts
        self.held_out = held_out

    def _residues(self, participant_ids, sentence_ids) -> np.ndarray:
        try:
            participant = np.array(
                [self.participant_codes[str(p)] for p in participant_ids]
            )
            sentence = np.array([self.sentence_codes[int(s)] for s in sentence_ids])
        except KeyError as unknown:
            raise KeyError(
                f"{unknown} is not in the code tables the partition was built "
                "from; they must cover every analyzed row"
            ) from None
        return (participant + sentence) % self.modulus

    def split(
        self,
        y: np.ndarray,
        groups: np.ndarray,
        participants: np.ndarray,
        items: np.ndarray,
        sentence_ids: np.ndarray | None = None,
        participant_ids: np.ndarray | None = None,
    ) -> Iterable[tuple[np.ndarray, np.ndarray]]:
        """Yield the single fit/held-out split the partition defines.

        Args:
            y: Dependent variable, unused: membership is read off the ids.
            groups: Stimulus index per row, unused.
            participants: Re-encoded reader code per row, unused: the codes
                come from the tables the splitter was built with.
            items: Stimulus code per row, unused.
            sentence_ids: Raw sentence number per row, looked up in
                ``sentence_codes``.
            participant_ids: Raw reader id per row, looked up in
                ``participant_codes``.

        Returns:
            A one-element iterator holding the ``(train_mask, test_mask)`` pair.
            Rows in neither part belong to the development partition and appear
            in no mask.

        Raises:
            ValueError: When either id array is missing.
            KeyError: When a row's id is absent from the code tables.
        """
        if sentence_ids is None or participant_ids is None:
            raise ValueError(
                "this partition is defined on participant and sentence "
                "identity, which the dataset must carry"
            )
        residue = self._residues(participant_ids, sentence_ids)
        yield self._part(residue, 0), self._part(residue, self.held_out)

    def _part(self, residue: np.ndarray, index: int) -> np.ndarray:
        if index == 0:
            return residue <= self.modulus - self.n_parts
        return residue == self.modulus - (self.n_parts - index)


def category_codes(values) -> dict:
    """Integer codes by sorted unique value, as the reference assigns them.

    Args:
        values: One column of identifiers, read over the unfiltered table so a
            code never shifts because an analysis filter removed a row.

    Returns:
        Each distinct value mapped to its rank in sorted order, keyed by the
        Python ``str`` or ``int`` form of the value.
    """
    unique = np.unique(np.asarray(values))
    return {
        (str(v) if isinstance(v, str | np.str_) else int(v)): i
        for i, v in enumerate(unique)
    }


_SENTENCE_END_CHARS = frozenset(".!?")


def compute_per_word_sentence_ids(words: list[str]) -> np.ndarray:
    """Assign each word a within-stimulus sentence index.

    A sentence ends after any word ending in ``.``, ``!`` or ``?``. This
    heuristic treats abbreviations such as ``Mr.`` as sentence endings and
    can change sentence-based fold assignments; prefer corpus sentence IDs.

    Args:
        words: Surface forms of one stimulus, in reading order; only the
            final character of each decides a boundary.

    Returns:
        An ``int32`` array of length ``len(words)`` holding a *local* index
        (0, 1, 2, ...) within the stimulus. Callers that need a
        globally-unique sentence number across a corpus must offset the
        per-stimulus output by the running sum of per-stimulus sentence counts.
    """
    sids: np.ndarray = np.zeros(len(words), dtype=np.int32)
    s = 0
    for i, w in enumerate(words):
        sids[i] = s
        if w and w[-1] in _SENTENCE_END_CHARS:
            s += 1
    return sids


def populate_sentence_ids(dataset: Dataset) -> int:
    """Write corpus-wide sentence numbers into each stimulus's metadata.

    ``stim.metadata['per_word_sentence_ids']`` numbers the sentences
    ``0..k0-1`` in the first stimulus, ``k0..k0+k1-1`` in the second, and so on.
    A stimulus that already carries the key keeps it, so a rerun changes
    nothing; later stimuli are numbered above every id seen so far.

    Args:
        dataset: Corpus to annotate in place; every stimulus must expose a
            ``words`` unit layer.

    Returns:
        The total number of sentences assigned across the corpus.
    """
    running = 0
    for stim in dataset.stimuli():
        if "per_word_sentence_ids" in stim.metadata:
            # May be a plain list (e.g. ModelBlocksDataset metadata).
            arr = np.asarray(stim.metadata["per_word_sentence_ids"])
            if len(arr):
                running = max(running, int(arr.max()) + 1)
            continue
        words = stim.unit_strings("words")
        local = compute_per_word_sentence_ids(words)
        stim.metadata["per_word_sentence_ids"] = local + running
        if len(local):
            running += int(local.max()) + 1
    return running


class FixedSplitter(CVSplitter):
    """Single pre-defined train/test split, no cross-validation.

    Yields one fold from a boolean mask over the flat observation array:
    ``train_mask`` gives it directly, or the mutually exclusive ``mask_fn``
    (``(y, groups, participants, items) -> bool_array``) builds it lazily at
    split time. ``from_indices`` and ``from_items`` construct the mask from
    row indices or stimulus IDs.
    """

    def configuration(self) -> dict:
        """Return declared options without fitted state."""
        return {
            "train_mask": array_identity(self.train_mask),
            "train_items": self._train_items,
            "mask_function": None
            if self._mask_fn is None
            else {
                "name": getattr(
                    self._mask_fn, "__qualname__", type(self._mask_fn).__qualname__
                ),
                "options": component_options(self._mask_fn),
            },
        }

    def __init__(
        self,
        train_mask: np.ndarray | None = None,
        *,
        mask_fn: Callable | None = None,
    ):
        """Take the training mask directly, or a rule that builds it later.

        Args:
            train_mask: Boolean array over the flat observation rows, true
                where a row is fitted; copied on the way in.
            mask_fn: Called as ``mask_fn(y, groups, participants, items)`` at
                split time and returning that same boolean array, for a rule
                that can only be resolved once the row order is known.

        Raises:
            ValueError: When both arguments are given, or neither.
        """
        if train_mask is not None and mask_fn is not None:
            raise ValueError("Provide train_mask or mask_fn, not both")
        if train_mask is None and mask_fn is None:
            raise ValueError("Provide either train_mask or mask_fn")
        self.train_mask = (
            np.array(train_mask, dtype=bool, copy=True)
            if train_mask is not None
            else None
        )
        self._mask_fn = mask_fn
        self._train_items: list[int] | None = None

    @classmethod
    def from_indices(cls, train_indices: np.ndarray, n_total: int) -> FixedSplitter:
        """Build from integer indices of training observations.

        Args:
            train_indices: Positions in the flat observation array that are
                fitted; every other position is held out.
            n_total: Length of the flat observation array, so the mask covers
                the rows the indices do not name.
        """
        mask: np.ndarray = np.zeros(n_total, dtype=bool)
        mask[train_indices] = True
        return cls(mask)

    @classmethod
    def from_items(cls, train_items: set | list | np.ndarray) -> FixedSplitter:
        """Build from item (stimulus group) IDs.

        All observations whose integer-encoded item ID is in
        ``train_items`` go to train; the rest go to test.
        The mask is resolved lazily at split time.

        Args:
            train_items: Integer-encoded stimulus ids whose rows are fitted.
                Values are compared as ``int``, so any numeric container works.
        """
        train_set = set(int(i) for i in train_items)
        splitter = cls(
            mask_fn=lambda _y, _groups, _participants, items: np.array(
                [int(i) in train_set for i in items], dtype=bool
            )
        )
        splitter._train_items = sorted(train_set)
        return splitter

    def split(
        self,
        y: np.ndarray,
        groups: np.ndarray,
        participants: np.ndarray,
        items: np.ndarray,
    ) -> Iterable[tuple[np.ndarray, np.ndarray]]:
        """Yield the one configured split.

        Args:
            y: Dependent variable; its length is the mask length the split
                is validated against.
            groups: Stimulus index per row, forwarded to ``mask_fn``.
            participants: Reader code per row, forwarded to ``mask_fn``.
            items: Stimulus code per row, forwarded to ``mask_fn``, which is
                what ``from_items`` reads.

        Returns:
            A one-element iterator holding the ``(train_mask, test_mask)`` pair;
            the two are complements.

        Raises:
            ValueError: When the mask is not one-dimensional of length
                ``len(y)``.
        """
        if self._mask_fn is not None:
            mask = np.atleast_1d(
                np.asarray(self._mask_fn(y, groups, participants, items), dtype=bool)
            )
        else:
            if self.train_mask is None:
                raise ValueError("FixedSplitter needs a training mask")
            mask = self.train_mask
        if mask.ndim != 1 or len(mask) != len(y):
            raise ValueError(
                f"FixedSplitter mask shape ({mask.shape}) "
                f"does not match data length ({len(y)})"
            )
        yield mask, ~mask


class FullDataSplitter(CVSplitter):
    """No cross-validation, train and test on all data.

    Yields a single fold where train_mask = test_mask = all True.
    Use this for full-data model comparison (e.g. comparing logLik of
    LME models without held-out evaluation).

    The score is in-sample fit, since every row is both fitted and scored.
    """

    in_sample = True

    def configuration(self) -> dict:
        """Return declared options without fitted state."""
        return {}

    def split(
        self,
        y: np.ndarray,
        groups: np.ndarray,
        participants: np.ndarray,
        items: np.ndarray,
    ) -> Iterable[tuple[np.ndarray, np.ndarray]]:
        """Yield one fold that fits and scores every row.

        Args:
            y: Dependent variable; only its length is read, to size the mask.
            groups: Stimulus index per row, unused.
            participants: Reader code per row, unused.
            items: Stimulus code per row, unused.

        Returns:
            A one-element iterator holding a pair of all-true masks, so the
            score is in-sample fit.
        """
        all_true: np.ndarray = np.ones(len(y), dtype=bool)
        yield all_true, all_true


class MultiSeedSplitter(CVSplitter):
    """Runs multiple seeds of an inner CVSplitter, yielding all folds across all seeds.

    Usage::

        MultiSeedSplitter(range(10), lambda s: ShuffledKFoldSplitter(10, seed=s))

    This yields 10 seeds x 10 folds = 100 total (train_mask, test_mask) pairs.
    Inner splitters must agree on sentence-ID and declared-factor routing.
    """

    _ROUTING_FLAGS = (
        "needs_sentence_ids",
        "uses_declared_factors",
        "uses_named_factors",
    )
    _configuration: list[dict] | None

    def configuration(self) -> dict:
        """Return declared options without fitted state."""
        if self._configuration is None:
            self._configuration = self._snapshot(self._splitters or [])
        return {
            "seeds": list(self.seeds),
            "splitters": self._configuration,
        }

    @staticmethod
    def _snapshot(splitters):
        return [
            {
                "type": f"{type(inner).__module__}.{type(inner).__qualname__}",
                "options": component_options(inner),
            }
            for inner in splitters
        ]

    def __init__(
        self,
        seeds: Sequence[int],
        splitter_factory: Callable[[int], CVSplitter],
    ):
        """Configure the seeds and the inner splitter they parameterise.

        Args:
            seeds: Seeds to run, in the order their folds are yielded.
            splitter_factory: Builds one fresh splitter per seed per split.
                The first set is built here to determine routing capabilities.
        """
        self.seeds = list(seeds)
        if not self.seeds:
            raise ValueError("MultiSeedSplitter requires at least one seed")
        self.splitter_factory = splitter_factory
        self._fold_metadata: list[dict] | None = None
        self._splitters: list[CVSplitter] | None = [
            splitter_factory(seed) for seed in self.seeds
        ]
        self._configuration = None
        self._routing = self._routing_for(self._splitters)
        (
            self.needs_sentence_ids,
            self.uses_declared_factors,
            self.uses_named_factors,
        ) = self._routing

    @classmethod
    def _routing_for(cls, splitters):
        """Require consistent input routing across a factory's splitters."""
        routing = [
            tuple(bool(getattr(inner, name, False)) for name in cls._ROUTING_FLAGS)
            for inner in splitters
        ]
        if any(flags != routing[0] for flags in routing):
            raise ValueError(
                "Inner splitters must have consistent routing capabilities"
            )
        return routing[0]

    def split(
        self,
        y: np.ndarray,
        groups: np.ndarray,
        participants: np.ndarray,
        items: np.ndarray,
        *,
        sentence_ids: np.ndarray | None = None,
        participant_ids: np.ndarray | None = None,
    ) -> Iterable[tuple[np.ndarray, np.ndarray]]:
        """Yield every inner fold of every seed, seeds in order.

        Args:
            y: Dependent variable, forwarded to each inner splitter.
            groups: Stimulus index per row, forwarded unchanged.
            participants: Reader code per row, forwarded unchanged.
            items: Stimulus code per row, forwarded unchanged.
            sentence_ids: Sentence identities forwarded to requesting splitters.
            participant_ids: Raw reader identities forwarded with sentence IDs.

        Returns:
            The inner splitters' ``(train_mask, test_mask)`` pairs. The seed
            and inner fold index behind each pair are recorded for
            :meth:`fold_metadata`, with any inner metadata under ``inner``.
        """
        self._fold_metadata = None
        inners = self._splitters
        self._splitters = None
        if inners is None:
            inners = [self.splitter_factory(seed) for seed in self.seeds]
        self._configuration = self._snapshot(inners)
        if self._routing_for(inners) != self._routing:
            raise ValueError("The splitter factory changed its routing capabilities")
        metadata: list[dict] = []
        for seed, inner in zip(self.seeds, inners, strict=True):
            if isinstance(inner, SentenceAwareSplitter) and inner.needs_sentence_ids:
                splits = inner.split(
                    y,
                    groups,
                    participants,
                    items,
                    sentence_ids=sentence_ids,
                    participant_ids=participant_ids,
                )
            else:
                splits = inner.split(y, groups, participants, items)
            start = len(metadata)
            for inner_fold_index, (train_mask, test_mask) in enumerate(splits):
                metadata.append({"seed": seed, "inner_fold_index": inner_fold_index})
                yield train_mask, test_mask
            if hasattr(inner, "fold_metadata"):
                inner_metadata = inner.fold_metadata()
                if len(inner_metadata) != len(metadata) - start:
                    raise ValueError(
                        "Inner fold metadata must match the yielded splits"
                    )
                for outer, record in zip(metadata[start:], inner_metadata, strict=True):
                    outer["inner"] = dict(record)
        self._fold_metadata = metadata

    def fold_metadata(self) -> list[dict]:
        """Return metadata after fully consuming the most recent split iterator."""
        if self._fold_metadata is None:
            raise RuntimeError("Complete split() before requesting fold_metadata()")
        return self._fold_metadata
