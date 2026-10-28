# SPDX-License-Identifier: Apache-2.0
"""Pure-text entry point for an existing outer checkpoint architecture name.

The class name is checkpoint metadata, not a claim of multimodal support. Text
token IDs, embeddings, logits, expert mapping and weight loading use the same
causal model as HYV4ForCausalLM. There is no processor or vision tower here.
"""

import torch

from vllm.model_executor.models.interfaces import SupportsMRoPE

from .model import HYV4ForCausalLM

_MEDIA_INPUTS = frozenset(
    {
        "pixel_values",
        "pixel_values_videos",
        "image_grid_thw",
        "video_grid_thw",
        "image_embeds",
        "video_embeds",
        "multi_modal_data",
        "mm_features",
    }
)


class HYV4VLForConditionalGeneration(HYV4ForCausalLM, SupportsMRoPE):
    """Keep the checkpoint's architecture identifier for text inference only."""

    supports_multimodal = False

    def forward(
        self,
        input_ids,
        positions,
        intermediate_tensors=None,
        inputs_embeds=None,
        **kwargs,
    ):
        if any(kwargs.get(name) is not None for name in _MEDIA_INPUTS):
            raise NotImplementedError(
                "Hy4-preview media inputs are unsupported in this text-only port"
            )
        kwargs = {
            key: value for key, value in kwargs.items() if key not in _MEDIA_INPUTS
        }
        return super().forward(
            input_ids, positions, intermediate_tensors, inputs_embeds, **kwargs
        )

    def get_mrope_input_positions(self, input_tokens, mm_features):
        if mm_features:
            raise NotImplementedError(
                "Hy4-preview media positions are unsupported in this text-only port"
            )
        # The three axes coincide for ordinary text; preserve the checkpoint's
        # rotary input shape without image/video grid or placeholder logic.
        return torch.arange(len(input_tokens), dtype=torch.long).repeat(3, 1), 0

    def embed_multimodal(self, **kwargs):
        raise NotImplementedError(
            "Hy4-preview media embeddings are unsupported in this text-only port"
        )

    @classmethod
    def get_placeholder_str(cls, modality, index):
        raise NotImplementedError(
            "Hy4-preview media placeholders are unsupported in this text-only port"
        )

    def load_weights(self, weights):
        # A checkpoint carrying the outer architecture may also carry encoder
        # weights. They are not allocated or loaded for this text-only entry.
        text_weights = (
            (name.removeprefix("language_model."), tensor)
            for name, tensor in weights
            if not name.startswith(("vit.", "visual."))
        )
        return super().load_weights(text_weights)
