"""Attach sentence membership from corpus-specific source annotations."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from replica.context import SENTENCE_LAYER
from replica.external import repo_path

if TYPE_CHECKING:
    from replica.core import Stimulus


def _meco_l1_sentnum(dataset: str, stimuli: list[Stimulus]) -> None:
    """MECO L1's per-word ``sentnum`` for the wave-1 ``meco_{lang}`` datasets.

    This is the column the Wilcox et al. (2023) pipeline grouped into
    sentence contexts (``clean_data.Rmd``). Source words must match the stimulus.
    """
    from replica.core import AnnotationLayer
    from replica.dataset.catalog.registry import REGISTRY
    from replica.dataset.eye_tracking.meco.releases import ISO_TO_FILE
    from replica.dataset.eye_tracking.meco.word_utils import (
        clean_word,
        read_r_dataframe,
    )

    lang = dataset.removeprefix("meco_")
    path = REGISTRY[dataset].default_path() / "joint_l1_data_trimmed_version2.1.rda"
    if not path.exists():
        raise FileNotFoundError(
            f"no sentence boundaries for {dataset}: {path} is absent"
        )
    frame = read_r_dataframe(path)
    frame = frame[frame["lang"] == ISO_TO_FILE.get(lang, lang)]
    columns = frame[["itemid", "wordnum", "sentnum", "word"]].drop_duplicates()
    selected_areas = {
        (int(stimulus.metadata["source_item_id"]), int(number))
        for stimulus in stimuli
        for number in stimulus.metadata["source_word_numbers"]
    }
    per_item: dict[int, dict[int, tuple[int, str]]] = {}
    for item, wordnum, sentnum, word in columns.itertuples(index=False):
        if (int(item), int(wordnum)) not in selected_areas:
            continue
        entry = (int(sentnum), clean_word(word))
        seen = per_item.setdefault(int(item), {}).setdefault(int(wordnum), entry)
        if seen != entry:
            raise ValueError(
                f"{dataset} item {item} word {wordnum}: conflicting rows "
                f"{seen!r} vs {entry!r}"
            )
    for stimulus in stimuli:
        item = int(stimulus.metadata["source_item_id"])
        words = per_item.get(item)
        if not words:
            raise ValueError(f"{dataset}: no sentnum rows for item {item}")
        layer = stimulus.layers["words"]
        ids = []
        for span, wordnum in zip(
            layer.spans, stimulus.metadata["source_word_numbers"], strict=True
        ):
            surface = stimulus.text[span.start : span.end]
            sentnum, word = words.get(int(wordnum), (None, None))
            if word != surface:
                raise ValueError(
                    f"{dataset} item {item} word {wordnum}: source surface "
                    f"{word!r} does not match the stimulus {surface!r}"
                )
            ids.append(sentnum)
        stimulus.add_layer(
            AnnotationLayer(name=SENTENCE_LAYER, spans=list(layer.spans), values=ids)
        )


def _natural_stories_ud_sentnum(dataset: str, stimuli: list[Stimulus]) -> None:
    """Natural Stories sentence ids from Kuribayashi et al. (2025)'s release.

    The release's ``data/NS/all.txt.averaged_rt.annotation`` carries the
    ``sent_id`` their preprocessing derived per (article, zone) from the
    corpus's UD parse (``parses/ud/stories-aligned.conllx``). Source words must
    match the stimulus.
    """
    import csv

    from replica.core import AnnotationLayer

    path = (
        repo_path("surprisal_internal_layers", fetch=False)
        / "data"
        / "NS"
        / "all.txt.averaged_rt.annotation"
    )
    if not path.exists():
        raise FileNotFoundError(
            f"no sentence boundaries for {dataset}: {path} is absent"
        )
    per_item: dict[int, dict[int, tuple[int, str]]] = {}
    with path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            entry = (int(row["sent_id"]), row["word"])
            article = per_item.setdefault(int(row["article"]), {})
            seen = article.setdefault(int(row["zone"]), entry)
            if seen != entry:
                raise ValueError(
                    f"{dataset} article {row['article']} zone {row['zone']}: "
                    f"conflicting rows {seen!r} vs {entry!r}"
                )
    for stimulus in stimuli:
        item = int(stimulus.metadata["story_id"])
        zones = per_item.get(item)
        if not zones:
            raise ValueError(f"{dataset}: no sent_id rows for article {item}")
        layer = stimulus.layers["words"]
        if len(layer.spans) != len(zones):
            raise ValueError(
                f"{dataset} article {item}: {len(layer.spans)} stimulus words "
                f"vs {len(zones)} annotated zones"
            )
        ids = []
        for index, span in enumerate(layer.spans):
            surface = stimulus.text[span.start : span.end]
            sent_id, word = zones[index + 1]
            if word != surface:
                raise ValueError(
                    f"{dataset} article {item} zone {index + 1}: source "
                    f"surface {word!r} does not match the stimulus {surface!r}"
                )
            ids.append(sent_id)
        stimulus.add_layer(
            AnnotationLayer(name=SENTENCE_LAYER, spans=list(layer.spans), values=ids)
        )


def _potec_sentnum(_dataset: str, stimuli: list[Stimulus]) -> None:
    """Attach the sentence membership carried by PoTeC's source measure rows."""
    from replica.core import AnnotationLayer

    for stimulus in stimuli:
        words = stimulus.layers["words"]
        ids = stimulus.metadata.get("source_sentence_numbers")
        if ids is None or len(ids) != len(words):
            raise ValueError(
                f"Missing source sentence membership for {stimulus.stimulus_id}"
            )
        stimulus.add_layer(
            AnnotationLayer(
                name=SENTENCE_LAYER, spans=list(words.spans), values=list(ids)
            )
        )


#: Dataset-name predicate and source provider; the first match wins.
SENTENCE_PROVIDERS: tuple[tuple[Callable[[str], bool], Callable], ...] = (
    (
        lambda name: name.startswith("meco_") and len(name) == len("meco_xx"),
        _meco_l1_sentnum,
    ),
    (
        lambda name: name == "natural_stories",
        _natural_stories_ud_sentnum,
    ),
    (lambda name: name == "potec", _potec_sentnum),
)


def _check_loader_sentence_layer(stimulus: Stimulus) -> None:
    """Refuse a loader-attached sentence layer that does not mirror the words."""
    sid = stimulus.stimulus_id
    words = stimulus.layers.get("words")
    if words is None:
        raise ValueError(f"{sid}: a sentence layer needs a words layer to mirror")
    layer = stimulus.layers[SENTENCE_LAYER]
    if layer.spans != words.spans:
        raise ValueError(
            f"{sid}: sentence spans must equal the word spans exactly, in order"
        )
    if len(layer.values) != len(words.spans):
        raise ValueError(
            f"{sid}: {len(layer.values)} sentence values for {len(words.spans)} words"
        )


def attach_sentences(dataset: str, stimuli: list[Stimulus]) -> None:
    """Attach the source data's sentence layer, or raise when none is recorded.

    A registered provider always decides for its corpus. Any other corpus
    keeps a ``sentences`` layer its loader attached to every stimulus.

    Args:
        dataset: Corpus name, matched against
            :data:`SENTENCE_PROVIDERS` to pick its boundary source.
        stimuli: Stimuli to annotate or validate in place. Sentence layers use
            the word spans and one sentence ID per word.
    """
    for matches, provider in SENTENCE_PROVIDERS:
        if matches(dataset):
            provider(dataset, stimuli)
            return
    attached = [stimulus for stimulus in stimuli if SENTENCE_LAYER in stimulus.layers]
    if stimuli and len(attached) == len(stimuli):
        for stimulus in stimuli:
            _check_loader_sentence_layer(stimulus)
        return
    if attached:
        raise ValueError(
            f"{dataset}: a sentence layer is attached to {len(attached)} of "
            f"{len(stimuli)} stimuli; attach it to every stimulus"
        )
    raise ValueError(
        f"no sentence-boundary source is registered for {dataset}; a "
        "sentence-scope cell needs the source corpus's segmentation"
    )
