# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""K3 model components selected at construction through native factories."""

from functools import partial

from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.compressed_tensors import (
    compressed_tensors,
)
from vllm.model_executor.layers.quantization.mxfp4 import Mxfp4Config
from vllm.models.kimi_k3.nvidia.model import (
    KimiDecoderLayer,
    KimiK3ForConditionalGeneration,
    KimiLinearForCausalLM,
    KimiLinearModel,
    KimiMLP,
    KimiMoE,
    LatentMoEComponents,
)

from vllm_fl.ops.kimi_k3.activation import K3SituAndMul
from vllm_fl.models.kimi_k3_context import execution_config
from vllm_fl.attention.kimi_k3.kda import K3DeltaAttention
from vllm_fl.ops.kimi_k3.latent_moe import LatentColumnParallelLinear, ShardedLatentMoERunner
from vllm_fl.attention.kimi_k3.mla import K3MultiHeadLatentAttention
from vllm_fl.ops.kimi_k3.projections import Tp8SharedGateUpProjection
from vllm_fl.models.kimi_k3_vision import K3VisionModel


class K3MLP(KimiMLP):
    gate_up_projection_cls = Tp8SharedGateUpProjection
    situ_activation_cls = K3SituAndMul


class K3MoE(KimiMoE):
    shared_expert_cls = K3MLP

    def _get_expert_partition_alignment(self, config, quant_config):
        return getattr(
            quant_config,
            "min_moe_intermediate_per_partition",
            super()._get_expert_partition_alignment(config, quant_config),
        )

    def _get_latent_moe_components(self, quant_config, use_sequence_parallel):
        shard = getattr(quant_config, "shard_latent_projections", False)
        overlap = not use_sequence_parallel
        if not shard:
            return LatentMoEComponents(overlap_down_projection=overlap)
        return LatentMoEComponents(
            down_projection=partial(
                LatentColumnParallelLinear,
                gather_output=True,
                sequence_parallel=use_sequence_parallel,
            ),
            up_projection=partial(
                LatentColumnParallelLinear,
                gather_output=use_sequence_parallel,
                sequence_parallel=use_sequence_parallel,
            ),
            runner_cls=ShardedLatentMoERunner,
            shard_shared_experts=True,
            overlap_down_projection=overlap,
        )


class K3DecoderLayer(KimiDecoderLayer):
    delta_attention_cls = K3DeltaAttention
    mla_cls = K3MultiHeadLatentAttention
    moe_cls = K3MoE
    mlp_cls = K3MLP


class K3LinearModel(KimiLinearModel):
    decoder_layer_cls = K3DecoderLayer


class K3LinearForCausalLM(KimiLinearForCausalLM):
    def __init__(self, *, vllm_config, prefix=""):
        policy = execution_config(vllm_config)
        self.model_cls = K3LinearModel if policy.enabled else KimiLinearModel
        super().__init__(vllm_config=vllm_config, prefix=prefix)


class K3ForConditionalGeneration(KimiK3ForConditionalGeneration):
    vision_tower_cls = K3VisionModel

    def _maybe_ignore_quant_config(
        self, quant_config: QuantizationConfig | None
    ) -> QuantizationConfig | None:
        if isinstance(
            quant_config, (compressed_tensors.CompressedTensorsConfig, Mxfp4Config)
        ):
            return None
        return quant_config
