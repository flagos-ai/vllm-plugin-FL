# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Native 0.32 vision/processor with the plugin's quantized language model."""

import torch
from torch import nn

from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.utils import maybe_prefix
from vllm.model_executor.models.vision import is_vit_use_data_parallel
from vllm.models.deepseek_v4.common.vision import DeepseekV4Aligner, DeepseekV4ViT
from vllm.models.deepseek_v41.common.mm_preprocess import (
    DeepseekV4VLDummyInputsBuilder,
    DeepseekV4VLMultiModalProcessor,
    DeepseekV4VLProcessingInfo,
)
from vllm.models.deepseek_v41.nvidia.vl_model import DeepseekV41ForCausalLM as NativeVL
from vllm.multimodal import MULTIMODAL_REGISTRY

from .model import DeepseekV41ForCausalLM as LanguageModel


@MULTIMODAL_REGISTRY.register_processor(
    DeepseekV4VLMultiModalProcessor,
    info=DeepseekV4VLProcessingInfo,
    dummy_inputs=DeepseekV4VLDummyInputsBuilder,
)
class DeepseekV41ForCausalLM(NativeVL):
    """Reuse native image encoding/merge; stream original weights once."""

    compressor_state_capacity = 8

    def __init__(self, *, vllm_config, prefix=""):
        nn.Module.__init__(self)
        mc = vllm_config.model_config
        self.config = mc.hf_config
        self.multimodal_config = mc.multimodal_config
        if self.multimodal_config is None:
            raise ValueError("V4.1 vision requires the native multimodal configuration")
        with self._mark_tower_model(vllm_config, {"image"}):
            self.use_data_parallel = is_vit_use_data_parallel(
                self.config.vision_n_heads
            )
            self.vision = DeepseekV4ViT(self.config)
            self.aligner = DeepseekV4Aligner(self.config)
            for name in ("image_start", "image_end", "image_newline"):
                self.register_parameter(
                    name,
                    nn.Parameter(
                        torch.empty(self.config.hidden_size, dtype=torch.float32)
                    ),
                )
            self.vision.to(dtype=mc.dtype)
            self.aligner.to(dtype=mc.dtype)
        with self._mark_language_model(vllm_config):
            self.language_model = LanguageModel(
                vllm_config=vllm_config, prefix=maybe_prefix(prefix, "language_model")
            )
        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    @staticmethod
    def get_model_state_cls():
        from .model_state import DeepseekV41ModelState

        return DeepseekV41ModelState

    @property
    def decoder_replay_layers(self):
        return None

    def reset_compressor_state(self, request_index):
        self.language_model.reset_compressor_state(request_index)

    def get_language_model(self):
        return self.language_model

    def forward(
        self,
        input_ids,
        positions,
        intermediate_tensors=None,
        inputs_embeds=None,
        **kwargs,
    ):
        return self.language_model(
            input_ids, positions, intermediate_tensors, inputs_embeds, **kwargs
        )

    def load_weights(self, weights):
        parameters = dict(self.named_parameters())
        tower = {
            k: v for k, v in parameters.items() if not k.startswith("language_model.")
        }
        loaded = set()

        def language_weights():
            for name, tensor in weights:
                if name.startswith(("vision.", "aligner.", "image_")):
                    if name not in tower:
                        raise KeyError(f"unknown vision parameter: {name}")
                    parameter = tower[name]
                    getattr(parameter, "weight_loader", default_weight_loader)(
                        parameter, tensor
                    )
                    loaded.add(name)
                else:
                    yield name, tensor

        loaded_language = self.language_model.load_weights(language_weights())
        missing = set(tower) - loaded
        if missing:
            raise ValueError(f"missing vision parameters: {sorted(missing)}")
        return loaded | {"language_model." + name for name in loaded_language}

    def process_weights_after_loading(self):
        # Quantization's parameter-level hooks are owned by the native loader.
        pass
