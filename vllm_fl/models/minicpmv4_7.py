# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""vLLM 0.24 adapter for MiniCPM-V 4.7."""

from __future__ import annotations

from typing import Any

from torch import nn

from vllm.config import VllmConfig
from vllm.model_executor.models import ModelRegistry
from vllm.model_executor.models.minicpmv import MiniCPMVDummyInputsBuilder
from vllm.model_executor.models.minicpmv4_6 import (
    Idefics2VisionTransformer,
    MiniCPMV4_6ForConditionalGeneration,
    MiniCPMV4_6Merger,
    MiniCPMV4_6MultiModalProcessor,
    MiniCPMV4_6ProcessingInfo,
    MiniCPMV4_6ViTWindowAttentionMerger,
)
from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5ForCausalLM,
    Qwen3_5MoeForCausalLM,
)
from vllm.model_executor.models.utils import maybe_prefix
from vllm.multimodal import MULTIMODAL_REGISTRY


class MiniCPMV4_7ProcessingInfo(MiniCPMV4_6ProcessingInfo):
    def get_model_version(self):
        return (4, 7)


def _select_language_model_cls(text_config: Any) -> type[nn.Module]:
    model_type = getattr(text_config, "model_type", "")
    if model_type == "qwen3_5_moe_text" or getattr(text_config, "num_experts", 0):
        return Qwen3_5MoeForCausalLM
    return Qwen3_5ForCausalLM


@MULTIMODAL_REGISTRY.register_processor(
    MiniCPMV4_6MultiModalProcessor,
    info=MiniCPMV4_7ProcessingInfo,
    dummy_inputs=MiniCPMVDummyInputsBuilder,
)
class MiniCPMV4_7ForConditionalGeneration(MiniCPMV4_6ForConditionalGeneration):
    """MiniCPM-V 4.7 with a config-selected Qwen3.5 text tower."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        # The 4.6 constructor hard-codes the dense Qwen3.5 class. Reuse its
        # multimodal components and select the text implementation from the
        # checkpoint config so both dense and MoE variants are supported.
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config

        self.config = config
        self.multimodal_config = multimodal_config
        self.use_data_parallel = multimodal_config.mm_encoder_tp_mode == "data"

        with self._mark_tower_model(vllm_config, {"image"}):
            self.vpm = Idefics2VisionTransformer(
                config.vision_config,
                quant_config=quant_config,
                apply_encoder_attention_mask=True,
                prefix=maybe_prefix(prefix, "vpm"),
            )
            if config.drop_vision_last_layer:
                self.vpm.encoder.layers = self.vpm.encoder.layers[:-1]

            self.vit_merger = MiniCPMV4_6ViTWindowAttentionMerger(
                config.vision_config,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "vit_merger"),
            )
            self.merger = MiniCPMV4_6Merger(
                hidden_size=config.vision_config.hidden_size,
                llm_embed_dim=config.text_config.hidden_size,
            )

        text_model_type = getattr(config.text_config, "model_type", "")
        language_model_cls = _select_language_model_cls(config.text_config)

        with self._mark_language_model(vllm_config):
            saved_model_type = config.model_type
            config.model_type = text_model_type
            try:
                self.language_model = language_model_cls(
                    vllm_config=vllm_config,
                    prefix=maybe_prefix(prefix, "language_model"),
                )
            finally:
                config.model_type = saved_model_type

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )


def _install_transformers_video_doc_compat() -> None:
    """Backfill a docstring-only symbol used by the supplied processor."""
    import transformers.video_processing_utils as video_processing_utils

    if not hasattr(video_processing_utils, "BASE_VIDEO_PROCESSOR_DOCSTRING"):
        video_processing_utils.BASE_VIDEO_PROCESSOR_DOCSTRING = ""


def register_minicpmv4_7() -> None:
    _install_transformers_video_doc_compat()
    ModelRegistry.register_model(
        "MiniCPMV4_7ForConditionalGeneration",
        MiniCPMV4_7ForConditionalGeneration,
    )
