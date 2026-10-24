"""Feature-model adapter for preceding context and target-unit alignment."""

from replica.context import context_input
from replica.core import Span, Stimulus
from replica.feature.core.annotation.alignment import content_span
from replica.feature.core.protocols import FeatureModel, FeatureModelOutput


class PrecedingContextModel:
    """Score a causal model with preceding context, returning only target units."""

    @property
    def supports_output_union(self) -> bool:
        """Preserve the wrapped producer's output-request guarantee."""
        return type(self) is PrecedingContextModel and bool(
            getattr(self.model, "supports_output_union", False)
        )

    def __init__(self, model: FeatureModel, *, context_scope: str = "story"):
        """Wrap a model configured to score a complete stimulus.

        Args:
            model: Model returning one output per requested annotation unit.
            context_scope: Scope label used in output metadata and model identity.
        """
        if getattr(model, "context_scope", "stimulus") != "stimulus":
            raise ValueError("PrecedingContextModel needs a stimulus-scope model")
        self.model = model
        self.context_scope = context_scope
        self.name = f"{model.name}#{context_scope}"

    @property
    def vocab(self):
        """Return the wrapped model's output vocabulary."""
        return self.model.vocab

    def __call__(
        self,
        stimulus: Stimulus,
        extract: list[str] | None = None,
        target_layer: str = "words",
    ) -> list[FeatureModelOutput]:
        """Return one output per target unit using the assembled preceding context.

        Args:
            stimulus: Target stimulus with attached preceding context.
            extract: Output fields passed to the wrapped model.
            target_layer: Annotation layer used for scoring and output alignment.
        """
        combined, start = context_input(stimulus, target_layer)
        outputs = self.model(combined, extract=extract, target_layer=target_layer)
        if len(outputs) != len(combined.layers[target_layer]):
            raise ValueError("Context model returned an incorrect unit count")
        return [
            FeatureModelOutput(
                values=dict(output.values),
                metadata={
                    **output.metadata,
                    "context_scope": self.context_scope,
                    "context_unit_index": start + index,
                    "target_unit_index": index,
                },
            )
            for index, output in enumerate(outputs[start:])
        ]

    def score_tokens(
        self,
        stimulus: Stimulus,
        extract: list[str] | None = None,
    ) -> list[FeatureModelOutput]:
        """One output per token of the target stimulus, scored after its context.

        The context and the target are tokenized as one sequence, so the first
        target token may carry the space separating it from the context; its
        span is clipped to the target text. Context tokens are left out, and a
        token whose content straddles the context boundary raises.

        Args:
            stimulus: Target stimulus with attached preceding context.
            extract: Value keys passed to the wrapped model.
        """
        combined, _ = context_input(stimulus)
        offset = len(combined.text) - len(stimulus.text)
        outputs: list[FeatureModelOutput] = []
        for output in self.model.score_tokens(combined, extract):
            start, end = output.metadata["span"]
            if end <= offset:
                continue
            content = content_span(combined.text, Span(start, end))
            if content is not None and content.start < offset:
                raise ValueError(
                    f"{stimulus.stimulus_id}: token {output.metadata['token']!r} "
                    "straddles the context boundary"
                )
            outputs.append(
                FeatureModelOutput(
                    values=dict(output.values),
                    metadata={
                        **output.metadata,
                        "token_index": len(outputs),
                        "span": (max(start, offset) - offset, end - offset),
                        "context_scope": self.context_scope,
                    },
                )
            )
        return outputs
