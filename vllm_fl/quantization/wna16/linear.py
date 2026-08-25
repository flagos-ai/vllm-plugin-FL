# Copyright (c) 2026 BAAI. All rights reserved.

"""Portable OOT adapter for vLLM's Triton W4A16 Linear kernel."""

from __future__ import annotations

import torch

from vllm.model_executor.kernels.linear import (
    MPLinearLayerConfig,
)
from vllm.model_executor.kernels.linear.mixed_precision.triton_w4a16 import (
    TRITON_W4A16_SUPPORTED_GROUP_SIZES,
    TritonW4A16LinearKernel,
)
from vllm.model_executor.layers.quantization.utils import (
    replace_parameter,
)
from vllm.model_executor.parameter import (
    BasevLLMParameter,
    permute_param_layout_,
)

from .repack import (
    repack_uint4_kpacked_to_npacked,
)


class FLTritonWNA16LinearKernel(
    TritonW4A16LinearKernel
):
    """
    Reuse vLLM's Triton W4A16 GEMM on CUDA-shaped OOT devices.

    Selection is capability-based rather than vendor-based. Weight conversion
    is implemented locally to avoid the FlagGems sum.dim_IntList reduction.
    """

    @classmethod
    def get_min_capability(cls) -> int:
        return -1

    @classmethod
    def can_implement(
        cls,
        config: MPLinearLayerConfig,
    ) -> tuple[bool, str | None]:
        if config.weight_type not in cls.SUPPORTED_QUANT_TYPES:
            return (
                False,
                f"Quant type {config.weight_type} is unsupported; "
                f"supported: {cls.SUPPORTED_QUANT_TYPES}",
            )

        if config.act_type not in (
            torch.float16,
            torch.bfloat16,
        ):
            return (
                False,
                "FLTritonWNA16 requires FP16 or BF16 activations",
            )

        if config.has_g_idx:
            return (
                False,
                "FLTritonWNA16 does not support activation ordering",
            )

        input_size, output_size = (
            config.partition_weight_shape
        )

        if input_size % 8 != 0:
            return (
                False,
                f"Input size {input_size} must be divisible by 8",
            )

        if output_size % 8 != 0:
            return (
                False,
                f"Output size {output_size} must be divisible by 8",
            )

        group_size = config.group_size
        full_input_size = config.full_weight_shape[0]

        if (
            group_size != -1
            and group_size
            not in TRITON_W4A16_SUPPORTED_GROUP_SIZES
            and group_size != full_input_size
        ):
            return (
                False,
                f"Group size {group_size} is unsupported; "
                f"supported: "
                f"{TRITON_W4A16_SUPPORTED_GROUP_SIZES} "
                f"or full K ({full_input_size})",
            )

        effective_group_size = (
            input_size
            if group_size == -1
            else group_size
        )

        if effective_group_size <= 0:
            return (
                False,
                f"Invalid group size {effective_group_size}",
            )

        if input_size % effective_group_size != 0:
            return (
                False,
                f"Input size {input_size} is not divisible by "
                f"group size {effective_group_size}",
            )

        return True, None

    def process_weights_after_loading(
        self,
        layer: torch.nn.Module,
    ) -> None:
        def repack_weight(
            parameter: BasevLLMParameter,
        ) -> BasevLLMParameter:
            # Restore canonical checkpoint layout [N, K // 8].
            permute_param_layout_(
                parameter,
                input_dim=1,
                output_dim=0,
                packed_dim=1,
            )

            parameter.data = (
                repack_uint4_kpacked_to_npacked(
                    parameter.data,
                )
            )
            return parameter

        def transpose_scale(
            parameter: BasevLLMParameter,
        ) -> BasevLLMParameter:
            # Checkpoint: [N, K // G]
            # Kernel:     [K // G, N]
            permute_param_layout_(
                parameter,
                input_dim=1,
                output_dim=0,
            )
            parameter.data = (
                parameter.data
                .transpose(-2, -1)
                .contiguous()
            )
            return parameter

        self._transform_param(
            layer,
            self.w_q_name,
            repack_weight,
        )
        self._transform_param(
            layer,
            self.w_s_name,
            transpose_scale,
        )

        if self.w_zp_name is not None:
            zero_point = getattr(
                layer,
                self.w_zp_name,
                None,
            )

            if zero_point is not None:
                replace_parameter(
                    layer,
                    self.w_zp_name,
                    torch.nn.Parameter(
                        zero_point.data
                        .transpose(-2, -1)
                        .contiguous(),
                        requires_grad=False,
                    ),
                )


def register_fl_triton_wna16_linear_kernel(
    registry: dict,
) -> bool:
    """
    Register the portable Triton WNA16 adapter for CUDA-shaped OOT devices.
    """
    from vllm.platforms import (
        PlatformEnum,
        current_platform,
    )

    if current_platform._enum != PlatformEnum.OOT:
        return False

    if getattr(
        current_platform,
        "device_type",
        None,
    ) != "cuda":
        return False

    candidates = registry.setdefault(
        PlatformEnum.OOT,
        [],
    )

    if FLTritonWNA16LinearKernel not in candidates:
        candidates.insert(
            0,
            FLTritonWNA16LinearKernel,
        )

    return True


__all__ = [
    "FLTritonWNA16LinearKernel",
    "register_fl_triton_wna16_linear_kernel",
]