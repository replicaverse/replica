"""Adapters for stored, empty, and combined feature-model outputs."""

from __future__ import annotations

import unicodedata
from functools import partial
from typing import TYPE_CHECKING

from replica.core import Stimulus
from replica.feature.core.protocols import FeatureModelOutput

if TYPE_CHECKING:
    from replica.feature.core.protocols import FeatureModel


class PrecomputedFeatureModel:
    """FeatureModel that returns stored FeatureModelOutput per stimulus.

    Used when model predictions are pre-computed (e.g., loaded from CSV).

    The stored outputs must index the units of the ``target_layer`` used at
    evaluation time: one record per unit, and a record that names its unit in
    ``metadata["text_unit"]`` (as the predictor store's rows do) must name the
    layer's string, compared after NFC normalization. A cell computed against
    another version of the corpus fails here, before any join by position.
    """

    @property
    def supports_output_union(self) -> bool:
        """Stored values are independent of the requested output keys."""
        return type(self) is PrecomputedFeatureModel

    @property
    def supports_stimulus_selection(self) -> bool:
        """Skipping other stimuli preserves stored outputs and source state."""
        return type(self) is PrecomputedFeatureModel

    vocab = None

    def __init__(self, name: str, outputs: dict[str, list[FeatureModelOutput]]):
        """Hold one stored list of per-unit records per stimulus.

        Args:
            name: Model name carried into AnalysisResult results.
            outputs: Per-unit records keyed by ``Stimulus.stimulus_id``.
        """
        self.name = name
        self._outputs = outputs

    def __call__(
        self,
        stimulus: Stimulus,
        extract: list[str] | None = None,
        target_layer: str = "words",
    ) -> list[FeatureModelOutput]:
        """The stored records for *stimulus*, one per unit.

        Args:
            stimulus: Stimulus looked up by its ``stimulus_id``.
            extract: Unused; nothing is computed, so every stored key comes
                back.
            target_layer: Annotation layer whose unit count the stored list
                has to match, else ``ValueError``.
        """
        stored = self._outputs[stimulus.stimulus_id]
        n_expected = len(stimulus.layers[target_layer])
        if len(stored) != n_expected:
            raise ValueError(
                f"PrecomputedFeatureModel {self.name!r}: stored {len(stored)} "
                f"outputs for stimulus {stimulus.stimulus_id!r}, but "
                f"target layer {target_layer!r} has {n_expected} units."
            )
        if any(record.metadata and "text_unit" in record.metadata for record in stored):
            self._check_text(stimulus, stored, target_layer)
        return stored

    def _check_text(
        self, stimulus: Stimulus, stored: list[FeatureModelOutput], target_layer: str
    ) -> None:
        nfc = partial(unicodedata.normalize, "NFC")
        units = stimulus.unit_strings(target_layer)
        for index, (record, unit) in enumerate(zip(stored, units, strict=True)):
            text = record.metadata.get("text_unit")
            if text is not None and text != unit and nfc(text) != nfc(unit):
                raise ValueError(
                    f"PrecomputedFeatureModel {self.name!r}: stored unit {index} of "
                    f"stimulus {stimulus.stimulus_id!r} is {text!r}, but the "
                    f"{target_layer!r} layer has {unit!r}; the cell indexes another "
                    "version of the corpus"
                )


class MergedFeatureModel:
    """FeatureModel whose per-unit values are the union of several models'.

    This is the seam for evaluating your own comprehension model under a
    paper's own recipe: keep the study's released controls, and let a live
    model supply the predictor under test.

    ::

        model = MergedFeatureModel(study_table, HuggingFaceLM("gpt2"))
        EvaluationSpec(
            targets=[dataset.feature("first_fixation")],
            predictors=[ModelFeature(model, Surprisal())],
            controls=[ModelFeature(model, PrecomputedFeature("wlen", "wlen"))],
        )

    Members are called in order and later values win, so a live model placed
    last overrides a same-named column in the table. A member that writes
    ``log_p_observed`` without ``surprisal`` also drops the ``surprisal`` a
    preceding member wrote, which :class:`Surprisal` would otherwise read first.
    ``vocab`` comes from the last member that declares one.
    """

    @property
    def supports_output_union(self) -> bool:
        """Combine requests only when every member preserves its outputs."""
        return type(self) is MergedFeatureModel and all(
            getattr(model, "supports_output_union", False) for model in self.models
        )

    def __init__(self, *models: FeatureModel, name: str | None = None):
        """Union the positional members' per-unit values, in the order given.

        Args:
            models: Members called in this order, at least one, each returning
                one output per unit; a later member's value wins a shared key,
                and ``vocab`` comes from the last member that declares one.
            name: Model name carried into AnalysisResult results; ``None`` joins the
                member names with ``+``.
        """
        if not models:
            raise ValueError("MergedFeatureModel needs at least one model")
        self.models = models
        self.name = name or "+".join(getattr(m, "name", "?") for m in models)
        self.vocab = next(
            (
                v
                for m in reversed(models)
                if (v := getattr(m, "vocab", None)) is not None
            ),
            None,
        )

    def __call__(
        self,
        stimulus: Stimulus,
        extract: list[str] | None = None,
        target_layer: str = "words",
    ) -> list[FeatureModelOutput]:
        """Every member's values merged per unit, later members winning.

        Args:
            stimulus: Text passed to each member in turn.
            extract: Value keys forwarded to each member unchanged.
            target_layer: Annotation layer every member has to return one
                output per unit of, else ``ValueError``.
        """
        n = len(stimulus.layers[target_layer])
        merged = [FeatureModelOutput(values={}, metadata={}) for _ in range(n)]
        for model in self.models:
            outputs = model(stimulus, extract, target_layer)
            if len(outputs) != n:
                raise ValueError(
                    f"MergedFeatureModel {self.name!r}: member "
                    f"{getattr(model, 'name', model)!r} returned {len(outputs)} "
                    f"outputs for stimulus {stimulus.stimulus_id!r}, expected {n}"
                )
            for target, out in zip(merged, outputs, strict=True):
                if "log_p_observed" in out.values and "surprisal" not in out.values:
                    target.values.pop("surprisal", None)
                target.values.update(out.values)
                target.metadata.update(out.metadata)
        return merged
