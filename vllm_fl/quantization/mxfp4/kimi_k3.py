"""MXFP4 loading with bounded BF16 expert execution through vLLM's router."""

import json

from flag_gems import runtime
from vllm_fl.dispatch import CachedOp

prepare_tp8_experts = CachedOp("kimi_k3_prepare_experts")
run_tp8_experts = CachedOp("kimi_k3_tp8_experts")
grouped_experts = CachedOp("kimi_k3_grouped_experts")
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import mxfp4_w4a16_moe_quant_config
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.quantization.mxfp4 import Mxfp4Config, Mxfp4MoEMethod
from vllm.model_executor.utils import replace_parameter

logger = init_logger(__name__)


class K3Mxfp4Config(Mxfp4Config):
    """Use vLLM's canonical MXFP4 config after checkpoint normalization."""

    # The grouped BF16 kernel accepts every complete MXFP4 group.
    model_family = "kimi_k3"
    min_moe_intermediate_per_partition = 32
    shard_latent_projections = True

    def _make_moe_method(self, moe):
        return K3Bf16MoEMethod(moe)


class K3Bf16MoEMethod(Mxfp4MoEMethod):
    """Keep vLLM routing/shared experts/TP reduction and original packed weights.

    Expert token alignment, tile dequantization, both grouped GEMMs and output
    combination execute on GPU. Workspace sizes use shape metadata only.
    """

    def __init__(self, moe):
        FusedMoEMethodBase.__init__(self, moe)
        if moe.activation != MoEActivation.SITU or moe.has_bias:
            raise NotImplementedError("The verified K3 expert path is bias-free SiTU")
        if (
            moe.tp_size != 8
            or moe.ep_size != 1
            or moe.dp_size != 1
            or moe.is_sequence_parallel
        ):
            raise NotImplementedError("K3 experts require TP8/DP1 with EP/SP disabled")
        self.weight_dtype = "mxfp4"
        self.weights_ready = False
        self.tp8_prepared = None
        self.prefill_workspace = None
        self.beta = moe.activation_situ_beta
        self.linear_beta = moe.activation_situ_linear_beta

    @property
    def supports_eplb(self):
        return False

    @property
    def skip_forward_padding(self):
        return False

    @property
    def has_unpadded_output(self):
        return False

    def maybe_roundup_sizes(
        self,
        hidden_size,
        intermediate_size_per_partition,
        act_dtype,
        moe_parallel_config,
    ):
        if hidden_size % 32 or intermediate_size_per_partition % 32:
            raise ValueError(
                "MXFP4 dimensions must align with the checkpoint's group32 scales"
            )
        return hidden_size, intermediate_size_per_partition

    def get_fused_moe_quant_config(self, layer):
        return mxfp4_w4a16_moe_quant_config(
            w1_scale=layer.w13_weight_scale, w2_scale=layer.w2_weight_scale
        )

    def process_weights_after_loading(self, layer):
        if self.weights_ready:
            return
        for scales in (layer.w13_weight_scale, layer.w2_weight_scale):
            if bool((scales == 255).any()):
                raise ValueError("Non-finite E8M0 expert scale in checkpoint")
        if (
            runtime.device.vendor_name == "metax"
            and self.moe.tp_size == 8
            and tuple(layer.w13_weight.shape) == (896, 768, 1792)
            and tuple(layer.w2_weight.shape) == (896, 3584, 192)
            and self.beta == 4.0
            and self.linear_beta == 25.0
        ):
            values, self.tp8_prepared = prepare_tp8_experts(
                layer.w13_weight,
                layer.w13_weight_scale,
                layer.w2_weight,
                layer.w2_weight_scale,
            )
            names = ("w13_weight", "w13_weight_scale", "w2_weight", "w2_weight_scale")
            for name, value in zip(names, values):
                if name.endswith("_scale"):
                    value.quant_method = "block"
                replace_parameter(layer, name, value)
            self.weights_ready = True
            logger.info("K3_TP8_PACKED_PREPARED %s", json.dumps(self.tp8_prepared))
            return
        self.weights_ready = True

    def apply(
        self, layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input
    ):
        if not self.weights_ready:
            raise RuntimeError("Expert weights have not completed loading")
        if layer.apply_router_weight_on_input:
            raise NotImplementedError("K3 weights outputs after G2")
        if layer.expert_map is not None:
            raise ValueError("Unexpected EP map in ordinary TP")
        if self.tp8_prepared is not None:
            return run_tp8_experts(
                x,
                (
                    layer.w13_weight,
                    layer.w13_weight_scale,
                    layer.w2_weight,
                    layer.w2_weight_scale,
                ),
                topk_weights,
                topk_ids,
                self.tp8_prepared,
                workspace=self.prefill_workspace,
            )
        return grouped_experts(
            x,
            layer.w13_weight,
            layer.w13_weight_scale,
            layer.w2_weight,
            layer.w2_weight_scale,
            topk_weights,
            topk_ids,
            beta=self.beta,
            linear_beta=self.linear_beta if self.linear_beta is not None else -1.0,
        )
