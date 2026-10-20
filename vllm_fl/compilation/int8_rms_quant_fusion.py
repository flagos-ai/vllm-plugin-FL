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
"""Add INT8 RMSNorm-quant patterns missing from vLLM 0.24.

vLLM's fused CUDA kernel already accepts an INT8 output tensor, but the
0.24 fusion pass only registers FP8 patterns.  Keep the implementation in
the plugin so the base vLLM installation remains untouched.
"""

from __future__ import annotations

import logging

import torch
import torch._inductor.pattern_matcher as pm
from torch._higher_order_ops.auto_functionalize import auto_functionalized
from torch._inductor.pattern_matcher import PatternMatcherPass

import vllm.ir.ops
from vllm.compilation.passes.fusion.rms_quant_fusion import (
    _rms_input_weight_dtype_match,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op

logger = logging.getLogger(__name__)


def _empty_bf16(*shape: int) -> torch.Tensor:
    return torch.empty(
        *shape,
        dtype=torch.bfloat16,
        device=current_platform.device_type,
    )


def _dynamic_int8_quant(
    value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    result = torch.empty_like(value, dtype=torch.int8)
    scale = torch.empty(
        (value.numel() // value.shape[-1], 1),
        dtype=torch.float32,
        device=value.device,
    )
    at = auto_functionalized(
        torch.ops._C.dynamic_scaled_int8_quant.default,
        result=result,
        input=value.contiguous(),
        scale=scale,
        azp=None,
    )
    return at[1], at[2]


def _deepseek_v4_qr_int8(
    qr: torch.Tensor,
    kv: torch.Tensor,
    q_weight: torch.Tensor,
    kv_weight: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Eager fallback for the semantic op when compilation is disabled."""
    from vllm_fl.ops.deepseek_v4 import fused_q_kv_rmsnorm_int8

    return fused_q_kv_rmsnorm_int8(
        qr,
        kv,
        q_weight,
        kv_weight,
        epsilon,
    )


def _deepseek_v4_qr_int8_fusion(
    qr: torch.Tensor,
    kv: torch.Tensor,
    q_weight: torch.Tensor,
    kv_weight: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Execute the OpManager-selected fused DSV4 QR/KV implementation."""
    from vllm_fl.ops.deepseek_v4 import fused_q_kv_rmsnorm_int8

    return fused_q_kv_rmsnorm_int8(
        qr,
        kv,
        q_weight,
        kv_weight,
        epsilon,
    )


def _deepseek_v4_qr_int8_fusion_fake(
    qr: torch.Tensor,
    kv: torch.Tensor,
    q_weight: torch.Tensor,
    kv_weight: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    del q_weight, kv_weight, epsilon
    qr_out = torch.empty_like(qr)
    kv_out = torch.empty_like(kv)
    qr_quant = torch.empty_like(qr, dtype=torch.int8)
    qr_scale = torch.empty(
        (qr.numel() // qr.shape[-1], 1),
        dtype=torch.float32,
        device=qr.device,
    )
    return qr_out, kv_out, qr_quant, qr_scale


direct_register_custom_op(
    op_name="deepseek_v4_qr_int8",
    op_func=_deepseek_v4_qr_int8,
    mutates_args=[],
    fake_impl=_deepseek_v4_qr_int8_fusion_fake,
    dispatch_key=current_platform.dispatch_key,
)

direct_register_custom_op(
    op_name="deepseek_v4_qr_int8_fusion",
    op_func=_deepseek_v4_qr_int8_fusion,
    mutates_args=[],
    fake_impl=_deepseek_v4_qr_int8_fusion_fake,
    dispatch_key=current_platform.dispatch_key,
)


class _DeepseekV4QRDynamicInt8Pattern:
    """Replace the semantic DSV4 QR INT8 op with its fused implementation."""

    def __init__(self, epsilon: float) -> None:
        self.epsilon = epsilon

    def register(self, patterns: PatternMatcherPass) -> None:
        epsilon = self.epsilon

        def pattern(
            qr: torch.Tensor,
            kv: torch.Tensor,
            q_weight: torch.Tensor,
            kv_weight: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            return torch.ops.vllm.deepseek_v4_qr_int8.default(
                qr,
                kv,
                q_weight,
                kv_weight,
                epsilon,
            )

        def replacement(
            qr: torch.Tensor,
            kv: torch.Tensor,
            q_weight: torch.Tensor,
            kv_weight: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            return torch.ops.vllm.deepseek_v4_qr_int8_fusion.default(
                qr,
                kv,
                q_weight,
                kv_weight,
                epsilon,
            )

        pm.register_replacement(
            pattern,
            replacement,
            [
                _empty_bf16(5, 16),
                _empty_bf16(5, 8),
                _empty_bf16(16),
                _empty_bf16(8),
            ],
            pm.fwd_only,
            patterns,
        )


class _RMSNormDynamicInt8Pattern:
    def __init__(self, epsilon: float) -> None:
        self.epsilon = epsilon
        self.fused_op = torch.ops._C.rms_norm_dynamic_per_token_quant.default

    def register(self, patterns: PatternMatcherPass) -> None:
        epsilon = self.epsilon
        fused_op = self.fused_op

        def pattern(
            input: torch.Tensor,
            weight: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            normalized = vllm.ir.ops.rms_norm(input, weight, epsilon)
            return _dynamic_int8_quant(normalized)

        def replacement(
            input: torch.Tensor,
            weight: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            result = torch.empty_like(input, dtype=torch.int8)
            scale = torch.empty(
                (input.numel() // input.shape[-1], 1),
                dtype=torch.float32,
                device=input.device,
            )
            at = auto_functionalized(
                fused_op,
                result=result,
                input=input,
                weight=weight,
                scale=scale,
                epsilon=epsilon,
                scale_ub=None,
                residual=None,
            )
            return at[1], at[2]

        pm.register_replacement(
            pattern,
            replacement,
            [_empty_bf16(5, 16), _empty_bf16(16)],
            pm.fwd_only,
            patterns,
            extra_check=_rms_input_weight_dtype_match,
        )


class _FusedAddRMSNormDynamicInt8Pattern:
    def __init__(self, epsilon: float) -> None:
        self.epsilon = epsilon
        self.fused_op = torch.ops._C.rms_norm_dynamic_per_token_quant.default

    def register(self, patterns: PatternMatcherPass) -> None:
        epsilon = self.epsilon
        fused_op = self.fused_op

        def pattern(
            input: torch.Tensor,
            weight: torch.Tensor,
            residual: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            normalized, residual_out = vllm.ir.ops.fused_add_rms_norm(
                input,
                residual,
                weight,
                epsilon,
            )
            result, scale = _dynamic_int8_quant(normalized)
            return result, residual_out, scale

        def replacement(
            input: torch.Tensor,
            weight: torch.Tensor,
            residual: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            result = torch.empty_like(input, dtype=torch.int8)
            scale = torch.empty(
                (input.numel() // input.shape[-1], 1),
                dtype=torch.float32,
                device=input.device,
            )
            at = auto_functionalized(
                fused_op,
                result=result,
                input=input,
                weight=weight,
                scale=scale,
                epsilon=epsilon,
                scale_ub=None,
                residual=residual,
            )
            return at[1], at[3], at[2]

        inputs = [
            _empty_bf16(5, 16),
            _empty_bf16(16),
            _empty_bf16(5, 16),
        ]
        pm.register_replacement(
            pattern,
            replacement,
            inputs,
            pm.fwd_only,
            patterns,
            extra_check=_rms_input_weight_dtype_match,
        )


def patch_vllm_int8_rms_quant_fusion() -> bool:
    """Register INT8 patterns in vLLM's existing norm-quant pass."""
    from vllm.compilation.passes.fusion.rms_quant_fusion import (
        RMSNormQuantFusionPass,
    )

    if getattr(RMSNormQuantFusionPass, "_fl_int8_patched", False):
        return False

    original_init = RMSNormQuantFusionPass.__init__
    original_uuid = RMSNormQuantFusionPass.uuid

    def patched_init(self, config) -> None:
        original_init(self, config)
        # Register the DSV4 multi-output pattern before the single-RMS
        # patterns, otherwise the Q-side subset would be consumed first.
        for epsilon in (1e-5, 1e-6):
            _DeepseekV4QRDynamicInt8Pattern(epsilon).register(self.patterns)
            _FusedAddRMSNormDynamicInt8Pattern(epsilon).register(self.patterns)
            _RMSNormDynamicInt8Pattern(epsilon).register(self.patterns)

    def patched_uuid(self) -> str:
        # Keep Inductor from reusing graphs compiled before these patterns.
        return original_uuid(self) + "-fl-int8-rms-quant-v2"

    RMSNormQuantFusionPass.__init__ = patched_init
    RMSNormQuantFusionPass.uuid = patched_uuid
    RMSNormQuantFusionPass._fl_int8_patched = True
    logger.info("Enabled vLLM RMSNorm + dynamic INT8 quant fusion patterns")
    return True
