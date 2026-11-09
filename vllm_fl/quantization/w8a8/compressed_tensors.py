# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ascend dynamic W8A8 execution for compressed-tensors checkpoints."""

from functools import cache

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
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)
from vllm.model_executor.utils import set_weight_attrs

from vllm_fl.dispatch import CachedOp


class AscendCompressedTensorsConfig(CompressedTensorsConfig):
    """Keep the upstream parser and checkpoint matching; replace supported math.

    Metadata uses checkpoint names, including regexes. Matching before model
    name translation preserves the exporter's ignore rules and packed shards.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._weight_mapper = None
        self._applied_mapper = None
        self.weight_nz_mode = 1
        self._add_fused_moe_to_target_scheme_map()

    def apply_vllm_mapper(self, mapper):
        if self._applied_mapper is mapper:
            return
        if self._applied_mapper is not None:
            raise ValueError("Quantization mapper already applied")
        self._applied_mapper = mapper
        # Regex substitution cannot in general be reversed. Such models retain
        # the upstream canonical-name contract instead of guessing an inverse.
        if mapper.orig_to_new_regex or mapper.orig_to_new_renamings:
            super().apply_vllm_mapper(mapper)
        else:
            self._weight_mapper = mapper

    def _checkpoint_name(self, name):
        mapper = self._weight_mapper
        if mapper is None:
            return name
        # Invert the literal model-owned translation in reverse application order.
        key = name + ".weight"
        for suffix, new in reversed(tuple(mapper.orig_to_new_suffix.items())):
            if new and key.endswith(new):
                key = key[: -len(new)] + suffix
        for prefix, new in reversed(tuple(mapper.orig_to_new_prefix.items())):
            if new and key.startswith(new):
                key = prefix + key[len(new) :]
        for source, new in reversed(tuple(mapper.orig_to_new_substr.items())):
            if new and new in key:
                key = key.replace(new, source, 1)
        return key.removesuffix(".weight")

    def _scheme(self, layer, prefix):
        from vllm.model_executor.layers.quantization.compressed_tensors.utils import (
            should_ignore_layer,
        )

        source = self._checkpoint_name(prefix)
        if should_ignore_layer(source, self.ignore) or should_ignore_layer(
            prefix, self.ignore
        ):
            return None
        scheme = super().get_scheme_dict(layer, source)
        if scheme is None and source != prefix:
            scheme = super().get_scheme_dict(layer, prefix)
        return scheme

    @staticmethod
    def _dynamic_w8a8(scheme):
        if scheme is None:
            return False
        w, a = scheme.get("weights"), scheme.get("input_activations")
        return (
            scheme.get("format") == "int-quantized"
            and w is not None
            and a is not None
            and w.num_bits == a.num_bits == 8
            and w.type == a.type == "int"
            and w.strategy == "channel"
            and a.strategy == "token"
            and w.symmetric
            and a.symmetric
            and not w.dynamic
            and a.dynamic
            and w.group_size is None
        )

    def get_quant_method(self, layer, prefix):
        from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
        from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner

        is_moe = isinstance(layer, (RoutedExperts, MoERunner))
        if not isinstance(layer, LinearBase) and not is_moe:
            return super().get_quant_method(layer, prefix)
        if is_moe:
            names = [
                f"{prefix}.{i}.{proj}"
                for i in range(layer.moe_config.num_experts)
                for proj in ("gate_proj", "up_proj", "down_proj")
            ]
        else:
            name = prefix.rsplit(".", 1)[-1]
            base = prefix[: -len(name)]
            names = [base + n for n in self.packed_modules_mapping.get(name, [name])]
        schemes = [self._scheme(layer, n) for n in names]
        enabled = [self._dynamic_w8a8(s) for s in schemes]
        if any(enabled):
            if not all(enabled):
                raise ValueError(f"Mixed W8A8 schemes in packed layer {prefix}")
            if self.kv_cache_scheme or self.transform_config:
                raise NotImplementedError(
                    "Ascend W8A8 does not implement KV quantization or transforms"
                )
            if is_moe:
                return AscendW8A8DynamicMoEMethod(layer.moe_config)
            return AscendW8A8DynamicLinearMethod(self.weight_nz_mode)
        if all(s is None for s in schemes):
            return None if is_moe else UnquantizedLinearMethod()
        raise NotImplementedError(
            f"Unsupported Ascend compressed-tensors scheme at {prefix}"
        )


@cache
def register_ascend_compressed_tensors():
    """Override compressed-tensors only in Ascend integration processes."""
    from vllm.model_executor.layers.quantization import register_quantization_config

    register_quantization_config("compressed-tensors")(AscendCompressedTensorsConfig)


_prepare_weight = CachedOp("w8a8_prepare_weight")


_linear = CachedOp("w8a8_dynamic_linear")


class AscendW8A8DynamicLinearMethod(LinearMethodBase):
    """Load [N,K] int8 weights and channel scales; execute prepared [K,N].

    The quantization config supplies weight_nz_mode, defaulting to 1.
    Modes 1 and 2 both NZ-cast int8; mode 0 retains ND. This component
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
            raise NotImplementedError("Dynamic W8A8 Linear requires BF16/FP16")
        n = sum(output_partition_sizes)
        weight = torch.nn.Parameter(
            torch.empty(n, input_size_per_partition, dtype=torch.int8),
            requires_grad=False,
        )
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        set_weight_attrs(weight, extra_weight_attrs)
        layer.register_parameter("weight", weight)
        scale = torch.nn.Parameter(
            torch.empty(n, 1, dtype=torch.float32), requires_grad=False
        )
        set_weight_attrs(scale, {"output_dim": 0})
        set_weight_attrs(scale, extra_weight_attrs)
        layer.register_parameter("weight_scale", scale)

    def process_weights_after_loading(self, layer):
        if layer.weight.dtype != torch.int8 or layer.weight.ndim != 2:
            raise ValueError("Expected W8A8 [N,K] int8 weight")
        if layer.weight_scale.shape != (layer.weight.shape[0], 1):
            raise ValueError("Expected one W8A8 scale per output channel")
        if layer.weight.shape[0] >= 65536:
            raise NotImplementedError(
                "Large-output DSA-CP Linear splitting is not enabled"
            )
        if getattr(self, "output_groups", None) is not None:
            groups = self.output_groups
            if groups < 1 or layer.weight.shape[0] % groups:
                raise ValueError("Grouped W8A8 outputs must divide the weight rows")
            self.group_input_size = layer.weight.shape[1]
            for group, (weight, scale) in enumerate(
                zip(
                    layer.weight.data.chunk(groups, dim=0),
                    layer.weight_scale.data.chunk(groups, dim=0),
                )
            ):
                weight, scale = _prepare_weight(
                    weight.transpose(0, 1).contiguous(),
                    scale.flatten(),
                    self.weight_nz_mode,
                )
                layer.register_buffer(
                    f"_grouped_w8a8_weight_{group}", weight, persistent=False
                )
                layer.register_buffer(
                    f"_grouped_w8a8_scale_{group}", scale, persistent=False
                )
            # Keep parameter ownership while releasing the unused logical bank.
            # Other groups remain module-owned buffers, moved with the layer.
            layer.weight.data = layer._grouped_w8a8_weight_0
            return
        weight, weight_scale = _prepare_weight(
            layer.weight.data.transpose(0, 1).contiguous(),
            layer.weight_scale.data.flatten(),
            self.weight_nz_mode,
        )
        layer.weight.data = weight
        layer.weight_scale.data = weight_scale

    def configure_grouped(self, groups):
        """Set the independent output groups before post-load preparation."""
        if groups < 1:
            raise ValueError("Grouped W8A8 requires positive groups")
        self.output_groups = groups

    def apply_grouped(self, layer, x):
        groups = self.output_groups
        tokens = x.shape[0]
        grouped = x.reshape(tokens, groups, self.group_input_size)
        outputs = [
            _linear(
                grouped[:, group].contiguous(),
                getattr(layer, f"_grouped_w8a8_weight_{group}"),
                getattr(layer, f"_grouped_w8a8_scale_{group}"),
            )
            for group in range(groups)
        ]
        return torch.cat(outputs, dim=-1)

    def apply(self, layer, x, bias=None):
        if x.dtype not in (torch.bfloat16, torch.float16):
            raise NotImplementedError("Dynamic W8A8 Linear requires BF16/FP16 input")
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


_prepare = CachedOp("w8a8_prepare_experts")


class AscendW8A8DynamicMoEMethod(FusedMoEMethodBase):
    def __init__(self, moe_config):
        super().__init__(moe_config)
        parallel = moe_config.moe_parallel_config
        if moe_config.moe_backend not in ("auto", "triton"):
            raise NotImplementedError("Unsupported explicit W8A8 MoE backend")
        if parallel.is_sequence_parallel and not (
            parallel.use_ep
            and parallel.dp_size > 1
            and parallel.all2all_backend == "allgather_reducescatter"
        ):
            raise NotImplementedError("W8A8 MoE token splitting requires DP EP AgRs")
        if (
            moe_config.is_lora_enabled
            or moe_config.has_bias
            or parallel.enable_eplb
            or parallel.pcp_size != 1
            or parallel.use_batched_activation_format
        ):
            raise NotImplementedError(
                "W8A8 experts do not support LoRA/bias/EPLB/PCP/batched routing"
            )
        if parallel.dp_size > 1 and (
            not parallel.use_ep or parallel.all2all_backend != "allgather_reducescatter"
        ):
            raise NotImplementedError("W8A8 distributed experts require AgRs EP")
        if moe_config.activation.value not in ("silu", "swiglustep"):
            raise NotImplementedError("Unsupported W8A8 expert activation")
        if moe_config.swiglu_alpha not in (None, 1.0) or moe_config.swiglu_beta not in (
            None,
            0.0,
        ):
            raise NotImplementedError("Unsupported W8A8 SwiGLU alpha/beta")
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
                    "W8A8 EP requires HCCL and equal linear experts without compiler SP"
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
            raise NotImplementedError("Dynamic W8A8 experts require BF16/FP16")
        self._activation_dtype = params_dtype
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
            scale = torch.nn.Parameter(
                torch.empty((*shape[:2], 1), dtype=torch.float32), requires_grad=False
            )
            set_weight_attrs(scale, attrs)
            layer.register_parameter(f"{prefix}_weight_scale", scale)
        layer.w13_input_scale = None
        layer.w2_input_scale = None

    def process_weights_after_loading(self, layer):
        for prefix in ("w13", "w2"):
            weight = getattr(layer, f"{prefix}_weight")
            scale = getattr(layer, f"{prefix}_weight_scale")
            if weight.dtype != torch.int8 or weight.ndim != 3:
                raise ValueError("Expected W8A8 INT8 [experts, output, input] weights")
            if scale.shape != (*weight.shape[:2], 1):
                raise ValueError("Expected W8A8 per-channel expert scales")
        prepared = _prepare(
            layer.w13_weight.data,
            layer.w2_weight.data,
            layer.w13_weight_scale.data,
            layer.w2_weight_scale.data,
            output_dtype=self._activation_dtype,
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
        raise ValueError("W8A8 experts initialize the modular kernel after loading")

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
            raise RuntimeError("W8A8 expert weights are not prepared")
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


_experts = CachedOp("w8a8_experts")


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
            raise ValueError("Dynamic W8A8 experts require floating prepare inputs")
        if (
            apply_router_weight_on_input
            or self._lora_context is not None
            or self.w1_bias is not None
            or self.w2_bias is not None
        ):
            raise NotImplementedError(
                "W8A8 experts require output routing and no LoRA/bias"
            )
        if hidden_states.dtype not in (torch.bfloat16, torch.float16):
            raise TypeError("Dynamic W8A8 experts require BF16/FP16 input")
        first_expert = 0
        if expert_map is not None:
            from vllm.distributed import get_ep_group

            group = get_ep_group()
            if (
                expert_map.numel() != w1.shape[0] * group.world_size
                or global_num_experts != expert_map.numel()
            ):
                raise NotImplementedError("W8A8 EP requires evenly partitioned experts")
            first_expert = group.rank_in_group * w1.shape[0]
        elif global_num_experts not in (-1, w1.shape[0]):
            raise NotImplementedError("Sharded W8A8 experts require an expert map")
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
