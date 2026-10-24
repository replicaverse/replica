"""Tuned-lens per-layer surprisal (Belrose et al. (2023))."""

from ..backends.hub import hub_id
from .logit_lens import LogitLensLM


class TunedLensLM(LogitLensLM):
    """Per-layer next-word surprisal via the *tuned* lens.

    Identical to :class:`LogitLensLM` except that internal hidden states are
    projected to the vocabulary with the model's pretrained tuned-lens affine
    probes (``TunedLens.from_model_and_pretrained``) rather than the raw
    unembedding; the output layer still uses the model's own head. This
    mirrors the tuned-lens branch of Kuribayashi et al. (2025)'s ``run_gpt2``.

    Per-word surprisals are stored under ``"tuned_lens_log_p_L{l}"``; read them
    with ``LayerSurprisal(layer, lens="tuned_lens")``. Requires the optional
    ``tuned-lens`` package and a model with published tuned-lens probes.
    """

    @property
    def supports_output_union(self) -> bool:
        """Combining requests preserves the same computed or sampled values."""
        return type(self) is TunedLensLM

    _lens = "tuned_lens"

    def __init__(
        self, model_name="gpt2", *, lens_revision=None, lens_resource_id=None, **kwargs
    ):
        """Optionally pin the probe repository independently of the base model.

        Args:
            model_name: Base model identity.
            lens_revision: Immutable revision of AlignmentResearch/tuned-lens.
            lens_resource_id: Probe directory identity, if different from the model.
            **kwargs: Base logit-lens options, including the model revision.
        """
        super().__init__(model_name, **kwargs)
        self.lens_revision = lens_revision
        self.lens_resource_id = lens_resource_id

    def _ensure_loaded(self):
        if self.device_map is not None:
            raise ValueError("Device dispatch is not validated for tuned lenses")
        super()._ensure_loaded()
        if getattr(self, "_tuned", None) is None:
            from tuned_lens import TunedLens

            options = {
                "lens_resource_id": (
                    self.lens_resource_id
                    if self.lens_resource_id is not None
                    else hub_id(self._model.config.name_or_path)
                )
            }
            if self.lens_revision is not None:
                options.update(
                    revision=self.lens_revision,
                    repo_id="AlignmentResearch/tuned-lens",
                    repo_type="space",
                )
            self._tuned = TunedLens.from_model_and_pretrained(
                self._model, **options
            ).to(self.device)

    def _layer_logits(self, hidden_states, layer_id, out_logits):
        h = hidden_states[layer_id]
        if layer_id == self._n_blocks:
            return self._lm_head(h)  # the model's own output
        return self._tuned(h, idx=layer_id)
