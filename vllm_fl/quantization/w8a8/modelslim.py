# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ModelSlim dynamic W8A8 configuration, Linear and modular MoE adapters."""

import json
from functools import cache
from pathlib import Path

import torch

from vllm.model_executor.layers.fused_moe import FusedMoeWeightScaleSupported
from vllm.model_executor.layers.fused_moe.config import int8_w8a8_moe_quant_config
from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonExperts
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.fused_moe.oracle.int8 import make_int8_moe_kernel
from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.utils import set_weight_attrs

from vllm_fl.dispatch import CachedOp

MODELSLIM_CONFIG_FILENAME = "quant_model_description.json"
_EXPERT_SHARDS = (
    "experts.0.gate_proj",
    "experts.0.up_proj",
    "experts.0.down_proj",
)


class AscendModelSlimConfig(QuantizationConfig):
    """Load dynamic/FLOAT metadata without implicitly enabling a model.

    Checkpoint naming belongs to the model's hf_to_vllm_mapper; absent a mapper,
    names are preserved. Quantized attention/cache and static INT8 need their
    own methods. Registration is an explicit action by platform integration.
    """

    def __init__(self, quant_config=None, *, weight_nz_mode=1):
        super().__init__()
        if weight_nz_mode not in (0, 1, 2):
            raise ValueError("weight_nz_mode must be 0, 1 or 2")
        self.weight_nz_mode = weight_nz_mode
        self.packed_modules_mapping = {
            "gate_up_proj": ["gate_proj", "up_proj"],
        }
        self.hf_to_vllm_mapper = None
        self.quant_description = {}
        self._set_description(quant_config or {})

    def _set_description(self, description):
        if not isinstance(description, dict):
            raise ValueError("ModelSlim description must be a JSON object")
        for key, value in description.items():
            if (
                key.endswith((".weight", ".weight_scale", ".weight_offset"))
                or key == "model_quant_type"
            ) and value not in ("FLOAT", "W8A8_DYNAMIC"):
                raise NotImplementedError(
                    f"Unsupported ModelSlim type {value!r} at {key}"
                )
            if (
                key.endswith("quant_type")
                and key != "model_quant_type"
                and value != "FLOAT"
            ):
                raise NotImplementedError(
                    f"Quantized attention/cache metadata at {key}"
                )
            if key == "kv_cache_type" and value != "FLOAT":
                raise NotImplementedError("Quantized ModelSlim KV cache is unsupported")
            if key == "group_size" and value != 0:
                raise NotImplementedError(
                    "Grouped ModelSlim quantization is unsupported"
                )
        self.quant_description = dict(description)
        if self.hf_to_vllm_mapper is not None:
            self.quant_description = self.hf_to_vllm_mapper.apply_dict(
                self.quant_description
            )

    @classmethod
    def get_name(cls):
        return "ascend"

    @classmethod
    def get_supported_act_dtypes(cls):
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls):
        raise NotImplementedError("Ascend does not have a CUDA compute capability")

    @staticmethod
    def get_config_filenames():
        # The descriptor is loaded explicitly by maybe_update_config.
        return []

    @classmethod
    def from_config(cls, config):
        return cls(config)

    def maybe_update_config(self, model_name, hf_config=None, revision=None):
        if self.quant_description:
            return
        path = Path(model_name) / MODELSLIM_CONFIG_FILENAME
        if not path.is_file():
            raise FileNotFoundError(f"Local ModelSlim descriptor missing: {path}")
        self._set_description(json.loads(path.read_text()))

    def apply_vllm_mapper(self, mapper):
        if self.hf_to_vllm_mapper is mapper:
            return
        if self.hf_to_vllm_mapper is not None:
            raise ValueError("ModelSlim mapper already applied; create a fresh config")
        self.quant_description = mapper.apply_dict(self.quant_description)
        self.hf_to_vllm_mapper = mapper

    def get_layer_quant_type(self, prefix, *, is_moe=False):
        # Prefixes come from runtime modules. Only checkpoint keys are mapped;
        # reapplying a model mapper here can remap an already canonical prefix.
        name = prefix.rsplit(".", 1)[-1]
        base = prefix[: -len(name)]
        # configure_quant_config replaces the model's Linear packing map by
        # reference. Expert-zero aliases describe the ModelSlim checkpoint
        # format, so they must survive that independent framework operation.
        shards = (
            _EXPERT_SHARDS if is_moe else self.packed_modules_mapping.get(name, [name])
        )
        keys = [base + shard + ".weight" for shard in shards]
        missing = [key for key in keys if key not in self.quant_description]
        if missing:
            raise ValueError(f"Missing ModelSlim weight metadata: {missing}")
        types = {self.quant_description[key] for key in keys}
        if len(types) != 1:
            raise ValueError(f"Mixed ModelSlim quantization for packed layer {prefix}")
        return types.pop()

    def get_quant_method(self, layer, prefix):
        from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
        from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner

        is_moe = isinstance(layer, (RoutedExperts, MoERunner))
        if not isinstance(layer, LinearBase) and not is_moe:
            return None
        quant_type = self.get_layer_quant_type(prefix, is_moe=is_moe)
        if quant_type == "FLOAT":
            # The existing FL MoE factory retains its unquantized method path.
            return None if is_moe else UnquantizedLinearMethod()
        if quant_type != "W8A8_DYNAMIC":
            raise NotImplementedError(f"Unsupported ModelSlim type {quant_type!r}")
        if is_moe:
            return AscendModelSlimMoEMethod(layer.moe_config)
        return AscendW8A8DynamicLinearMethod(self.weight_nz_mode)


@cache
def register_modelslim_config():
    """Register only when an Ascend integration explicitly requests ModelSlim."""
    from vllm.model_executor.layers.quantization import register_quantization_config

    register_quantization_config(AscendModelSlimConfig.get_name())(
        AscendModelSlimConfig
    )


_prepare_weight = CachedOp("modelslim_w8a8_prepare_weight")


_linear = CachedOp("modelslim_w8a8_dynamic_linear")


class AscendW8A8DynamicLinearMethod(LinearMethodBase):
    """Load [N,K] int8 weights and channel scales; execute prepared [K,N].

    The caller supplies the resolved weight_nz_mode when config integration is
    added. Modes 1 and 2 both NZ-cast int8; mode 0 retains ND. This component
    supports A2/A3 without FlashComm2 quantized communication or DSA-CP splitting.
    """

    def __init__(self, weight_nz_mode: int = 1):
        if weight_nz_mode not in (0, 1, 2):
            raise ValueError("weight_nz_mode must be 0, 1 or 2")
        self.weight_nz_mode = weight_nz_mode

    def create_weights(
        self,
        layer,
        input_size_per_partition,
        output_partition_sizes,
        input_size,
        output_size,
        params_dtype,
        **extra_weight_attrs,
    ):
        if params_dtype not in (torch.bfloat16, torch.float16):
            raise NotImplementedError("ModelSlim dynamic Linear requires BF16/FP16")
        n = sum(output_partition_sizes)
        weight = torch.nn.Parameter(
            torch.empty(n, input_size_per_partition, dtype=torch.int8),
            requires_grad=False,
        )
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        set_weight_attrs(weight, extra_weight_attrs)
        layer.register_parameter("weight", weight)
        for name in ("weight_scale", "weight_offset"):
            param = torch.nn.Parameter(
                (torch.zeros if name == "weight_offset" else torch.empty)(
                    n, 1, dtype=params_dtype
                ),
                requires_grad=False,
            )
            set_weight_attrs(param, {"output_dim": 0})
            set_weight_attrs(param, extra_weight_attrs)
            layer.register_parameter(name, param)

    def get_computed_params(self) -> set[str]:
        # Symmetric offsets can be computed rather than required weights.
        return {"weight_offset"}

    def process_weights_after_loading(self, layer):
        if layer.weight.dtype != torch.int8 or layer.weight.ndim != 2:
            raise ValueError("Expected ModelSlim [N,K] int8 weight")
        if layer.weight_scale.shape != (layer.weight.shape[0], 1):
            raise ValueError("Expected one ModelSlim scale per output channel")
        if layer.weight_offset.shape != layer.weight_scale.shape:
            raise ValueError("Expected one ModelSlim offset per output channel")
        # Dynamic INT8 arithmetic loads offsets but does not apply them.
        # Restrict this bounded component to the symmetric contract explicitly.
        if torch.count_nonzero(layer.weight_offset).item():
            raise NotImplementedError("Nonzero ModelSlim dynamic weight offsets")
        if layer.weight.shape[0] >= 65536:
            raise NotImplementedError(
                "Large-output DSA-CP Linear splitting is not enabled"
            )
        weight, weight_scale = _prepare_weight(
            layer.weight.data.transpose(0, 1).contiguous(),
            layer.weight_scale.data.flatten(),
            self.weight_nz_mode,
        )
        layer.weight.data = weight
        layer.weight_scale.data = weight_scale
        layer.weight_offset.data = layer.weight_offset.data.flatten()

    def apply(self, layer, x, bias=None):
        if x.dtype not in (torch.bfloat16, torch.float16):
            raise NotImplementedError(
                "ModelSlim dynamic Linear requires BF16/FP16 input"
            )
        if x.ndim < 2 or x.shape[-1] != layer.weight.shape[0]:
            raise ValueError("Expected [..., K] activations matching prepared [K,N]")
        if bias is not None and bias.shape != (layer.weight.shape[1],):
            raise ValueError("Expected bias [N]")
        shape = (*x.shape[:-1], layer.weight.shape[1])
        return _linear(
            x.reshape(-1, x.shape[-1]).contiguous(),
            layer.weight,
            layer.weight_scale,
            bias,
        ).reshape(shape)


_prepare = CachedOp("modelslim_w8a8_prepare_experts")


class AscendModelSlimMoEMethod(FusedMoEMethodBase):
    def __init__(self, moe_config):
        super().__init__(moe_config)
        parallel = moe_config.moe_parallel_config
        if moe_config.moe_backend not in ("auto", "triton"):
            raise NotImplementedError("Unsupported explicit ModelSlim MoE backend")
        if parallel.is_sequence_parallel and not (
            parallel.use_ep
            and parallel.dp_size > 1
            and parallel.all2all_backend == "allgather_reducescatter"
        ):
            raise NotImplementedError(
                "ModelSlim MoE token splitting requires DP EP AgRs"
            )
        if (
            moe_config.is_lora_enabled
            or moe_config.has_bias
            or parallel.enable_eplb
            or parallel.pcp_size != 1
            or parallel.use_batched_activation_format
        ):
            raise NotImplementedError(
                "ModelSlim experts do not support LoRA/bias/EPLB/PCP/batched routing"
            )
        if parallel.dp_size > 1 and (
            not parallel.use_ep or parallel.all2all_backend != "allgather_reducescatter"
        ):
            raise NotImplementedError("ModelSlim distributed experts require AgRs EP")
        if moe_config.activation.value not in ("silu", "swiglustep"):
            raise NotImplementedError("Unsupported ModelSlim expert activation")
        if moe_config.swiglu_alpha not in (None, 1.0) or moe_config.swiglu_beta not in (
            None,
            0.0,
        ):
            raise NotImplementedError("Unsupported ModelSlim SwiGLU alpha/beta")
        if parallel.use_ep:
            from vllm.config import get_current_vllm_config
            from vllm.platforms import current_platform

            config = get_current_vllm_config()
            if (
                current_platform.dist_backend != "hccl"
                or config.parallel_config.expert_placement_strategy != "linear"
                or config.compilation_config.pass_config.enable_sp
                or moe_config.num_experts % parallel.ep_size
            ):
                raise NotImplementedError(
                    "ModelSlim EP requires HCCL and equal linear experts without compiler SP"
                )

    def create_weights(
        self,
        layer,
        num_experts,
        hidden_size,
        intermediate_size_per_partition,
        params_dtype,
        **extra_weight_attrs,
    ):
        if params_dtype not in (torch.bfloat16, torch.float16):
            raise NotImplementedError("ModelSlim dynamic experts require BF16/FP16")
        shapes = {
            "w13": (num_experts, 2 * intermediate_size_per_partition, hidden_size),
            "w2": (num_experts, hidden_size, intermediate_size_per_partition),
        }
        for prefix, shape in shapes.items():
            param = torch.nn.Parameter(
                torch.empty(shape, dtype=torch.int8), requires_grad=False
            )
            set_weight_attrs(param, extra_weight_attrs)
            layer.register_parameter(f"{prefix}_weight", param)
            attrs = {
                **extra_weight_attrs,
                "quant_method": FusedMoeWeightScaleSupported.CHANNEL.value,
            }
            for suffix in ("scale", "offset"):
                factory = torch.zeros if suffix == "offset" else torch.empty
                param = torch.nn.Parameter(
                    factory((*shape[:2], 1), dtype=params_dtype),
                    requires_grad=False,
                )
                set_weight_attrs(param, attrs)
                layer.register_parameter(f"{prefix}_weight_{suffix}", param)
        layer.w13_input_scale = None
        layer.w2_input_scale = None

    def get_computed_params(self) -> set[str]:
        return {"w13_weight_offset", "w2_weight_offset"}

    def process_weights_after_loading(self, layer):
        for prefix in ("w13", "w2"):
            weight = getattr(layer, f"{prefix}_weight")
            scale = getattr(layer, f"{prefix}_weight_scale")
            offset = getattr(layer, f"{prefix}_weight_offset")
            if weight.dtype != torch.int8 or weight.ndim != 3:
                raise ValueError(
                    "Expected ModelSlim INT8 [experts, output, input] weights"
                )
            if scale.shape != (*weight.shape[:2], 1) or offset.shape != scale.shape:
                raise ValueError("Expected ModelSlim per-channel expert scales/offsets")
            if torch.count_nonzero(offset).item():
                raise NotImplementedError("Nonzero ModelSlim dynamic expert offsets")
        prepared = _prepare(
            layer.w13_weight.data,
            layer.w2_weight.data,
            layer.w13_weight_scale.data,
            layer.w2_weight_scale.data,
        )
        for name, tensor in zip(
            ("w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale"), prepared
        ):
            getattr(layer, name).data = tensor
        self.moe_quant_config = self.get_fused_moe_quant_config(layer)
        self.moe_kernel = make_int8_moe_kernel(
            self.moe_quant_config,
            self.moe,
            AscendW8A8DynamicExperts,
            routing_tables=layer._expert_routing_tables(),
        )

    def maybe_make_prepare_finalize(self, routing_tables=None):
        raise ValueError(
            "ModelSlim experts initialize the modular kernel after loading"
        )

    def get_fused_moe_quant_config(self, layer):
        config = int8_w8a8_moe_quant_config(
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            a1_scale=None,
            a2_scale=None,
            per_act_token_quant=True,
        )
        config.gemm1_clamp_limit = self.moe.swiglu_limit
        return config

    def apply(
        self, layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input
    ):
        if self.moe_kernel is None:
            raise RuntimeError("ModelSlim expert weights are not prepared")
        return self.moe_kernel.apply(
            x,
            layer.w13_weight,
            layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=layer.activation,
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            shared_experts=shared_experts,
            shared_experts_input=shared_experts_input,
        )


_experts = CachedOp("modelslim_w8a8_experts")


class AscendW8A8DynamicExperts(TritonExperts):
    @property
    def expects_unquantized_inputs(self) -> bool:
        # Prepare/AgRs must not quantize inputs through a CUDA-only kernel.
        return True

    def workspace_shapes(
        self,
        M,
        N,
        K,
        topk,
        global_num_experts,
        local_num_experts,
        expert_tokens_meta,
        activation,
    ):
        return (0,), (0,), (M, K)

    def apply(
        self,
        output,
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        activation,
        global_num_experts,
        expert_map,
        a1q_scale,
        a2_scale,
        workspace13,
        workspace2,
        expert_tokens_meta,
        apply_router_weight_on_input,
    ):
        if a1q_scale is not None or a2_scale is not None:
            raise ValueError(
                "ModelSlim dynamic experts require floating prepare inputs"
            )
        if (
            apply_router_weight_on_input
            or self._lora_context is not None
            or self.w1_bias is not None
            or self.w2_bias is not None
        ):
            raise NotImplementedError(
                "ModelSlim experts require output routing and no LoRA/bias"
            )
        if hidden_states.dtype not in (torch.bfloat16, torch.float16):
            raise TypeError("ModelSlim dynamic experts require BF16/FP16 input")
        first_expert = 0
        if expert_map is not None:
            from vllm.distributed import get_ep_group

            group = get_ep_group()
            if (
                expert_map.numel() != w1.shape[0] * group.world_size
                or global_num_experts != expert_map.numel()
            ):
                raise NotImplementedError(
                    "ModelSlim EP requires evenly partitioned experts"
                )
            first_expert = group.rank_in_group * w1.shape[0]
        elif global_num_experts not in (-1, w1.shape[0]):
            raise NotImplementedError("Sharded ModelSlim experts require an expert map")
        output.copy_(
            _experts(
                hidden_states,
                w1,
                w2,
                self.w1_scale,
                self.w2_scale,
                topk_weights,
                topk_ids,
                first_expert=first_expert,
                global_num_experts=global_num_experts,
                activation=activation.value,
                limit=self.gemm1_clamp_limit,
            )
        )
