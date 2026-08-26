# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
"""Current vLLM-Ascend RMSNorm implementations adapted to FL ownership."""

from __future__ import annotations

import torch
import torch.nn.functional as F
import torch_npu
from torch import nn
from vllm.config import get_current_vllm_config
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.layernorm import GemmaRMSNorm, RMSNorm, RMSNormGated
from vllm.utils.torch_utils import direct_register_custom_op

from vllm_fl.ascend_custom_ops import enable_custom_op
from vllm_fl.ascend_forward_context import _EXTRA_CTX
from vllm_fl.dispatch.backends.vendor.ascend.impl.triton.layernorm_gated import (
    layer_norm_fwd_npu,
)


def _maybe_chunk_residual_impl(
    x: torch.Tensor,
    residual: torch.Tensor,
) -> torch.Tensor:
    """Current vLLM-Ascend residual contract for TP/SP execution."""
    try:
        get_forward_context()
    except AssertionError:
        return residual

    if x.size(0) != residual.size(0):
        pad_size = _EXTRA_CTX.pad_size
        if pad_size > 0:
            residual = F.pad(residual, (0, 0, 0, pad_size))
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        residual = torch.chunk(residual, tp_size, dim=0)[tp_rank]
    return residual


def _maybe_chunk_residual_fake(
    x: torch.Tensor,
    residual: torch.Tensor,
) -> torch.Tensor:
    del residual
    return torch.empty_like(x)


if not hasattr(torch.ops.vllm, "maybe_chunk_residual"):
    direct_register_custom_op(
        op_name="maybe_chunk_residual",
        op_func=_maybe_chunk_residual_impl,
        fake_impl=_maybe_chunk_residual_fake,
        mutates_args=[],
        dispatch_key="PrivateUse1",
    )


class AscendRMSNorm(RMSNorm):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        var_hidden_size: int | None = None,
        has_weight: bool = True,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(hidden_size, eps, var_hidden_size, has_weight, dtype)
        vllm_config = get_current_vllm_config()
        self.bias = None
        self.bias_loaded = False
        if vllm_config.quant_config is not None and any(
            "norm.bias" in name
            for name in vllm_config.quant_config.quant_description
        ):
            self.bias = torch.nn.Parameter(
                torch.zeros(hidden_size), requires_grad=False
            )
            self.bias.weight_loader = self._bias_weight_loader

    def _bias_weight_loader(
        self,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
    ) -> None:
        if param.numel() == 1 and loaded_weight.numel() == 1:
            param.data.fill_(loaded_weight.item())
        else:
            assert param.size() == loaded_weight.size(), (
                f"Attempted to load weight ({loaded_weight.size()}) into "
                f"parameter ({param.size()})"
            )
            param.data.copy_(loaded_weight)
        self.bias_loaded = True

    def forward_oot(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is not None:
            residual = torch.ops.vllm.maybe_chunk_residual(x, residual)
            if enable_custom_op() and hasattr(
                torch.ops._C_ascend, "npu_add_rms_norm_bias"
            ):
                x, _, residual = torch.ops._C_ascend.npu_add_rms_norm_bias(
                    x,
                    residual,
                    self.weight,
                    self.bias,
                    self.variance_epsilon,
                )
            else:
                x, _, residual = torch_npu.npu_add_rms_norm(
                    x, residual, self.weight, self.variance_epsilon
                )
                if self.bias is not None:
                    x.add_(self.bias)
            return x, residual
        x, _ = torch_npu.npu_rms_norm(
            x, self.weight, self.variance_epsilon
        )
        if self.bias_loaded:
            x.add_(self.bias)
        return x


class AscendGemmaRMSNorm(GemmaRMSNorm):
    def forward_oot(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is not None:
            residual = torch.ops.vllm.maybe_chunk_residual(x, residual)
            weight = 1.0 + self.weight
            if enable_custom_op() and hasattr(
                torch.ops._C_ascend, "npu_add_rms_norm_bias"
            ):
                x, _, residual = torch.ops._C_ascend.npu_add_rms_norm_bias(
                    x, residual, weight, None, self.variance_epsilon
                )
            else:
                x, _, residual = torch_npu.npu_add_rms_norm(
                    x, residual, weight, self.variance_epsilon
                )
            return x, residual

        if not enable_custom_op() or not hasattr(
            torch.ops._C_ascend, "npu_gemma_rms_norm"
        ):
            # Gemma stores delta weights and therefore normalizes with 1+w.
            x, _ = torch_npu.npu_rms_norm(
                x, 1.0 + self.weight, self.variance_epsilon
            )
            return x
        x, _ = torch.ops._C_ascend.npu_gemma_rms_norm(
            x, self.weight, self.variance_epsilon
        )
        return x


class LayerNormFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        weight,
        bias,
        z=None,
        eps=1e-6,
        group_size=None,
        norm_before_gate=True,
        is_rms_norm=False,
        activation: str = "swish",
    ):
        x_shape = x.shape
        x = x.reshape(-1, x.shape[-1]).contiguous()
        if z is not None:
            z = z.reshape(-1, z.shape[-1]).contiguous()
        y, _, _ = layer_norm_fwd_npu(
            x,
            weight.contiguous(),
            bias.contiguous() if bias is not None else None,
            eps,
            z=z,
            group_size=group_size,
            norm_before_gate=norm_before_gate,
            is_rms_norm=is_rms_norm,
        )
        return y.reshape(x_shape)


def _rms_norm_gated_impl(
    x: torch.Tensor,
    z: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    group_size: int,
    norm_before_gate: bool,
) -> torch.Tensor:
    """Execute gated RMSNorm with shapes resolved at runtime.

    Keeping the Triton launch behind a custom-op boundary is important for
    FULL_DECODE_ONLY: vLLM traces the model with a one-token decode input but
    executes chunked prefill through the resulting eager FX graph.  If the
    Triton helper is inlined into that trace, both its launch grid and final
    reshape are frozen to the one-token shape.
    """
    original_x = x
    x = x.reshape(-1, x.shape[-1]).contiguous()
    z = z.reshape(-1, z.shape[-1]).contiguous()
    y, _, _ = layer_norm_fwd_npu(
        x,
        weight.contiguous(),
        None,
        eps,
        z=z,
        group_size=None if group_size < 0 else group_size,
        norm_before_gate=norm_before_gate,
        is_rms_norm=True,
    )
    return y.reshape_as(original_x)


def _rms_norm_gated_fake(
    x: torch.Tensor,
    z: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    group_size: int,
    norm_before_gate: bool,
) -> torch.Tensor:
    del z, weight, eps, group_size, norm_before_gate
    return torch.empty_like(x)


if not hasattr(torch.ops.vllm, "ascend_rms_norm_gated"):
    direct_register_custom_op(
        op_name="ascend_rms_norm_gated",
        op_func=_rms_norm_gated_impl,
        fake_impl=_rms_norm_gated_fake,
        mutates_args=[],
        dispatch_key="PrivateUse1",
    )


class AscendRMSNormGated(RMSNormGated):
    def __init__(
        self,
        hidden_size,
        eps: float = 1e-5,
        group_size: int | None = None,
        norm_before_gate: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        activation: str = "swish",
    ):
        super().__init__(
            hidden_size,
            eps,
            group_size,
            norm_before_gate,
            device,
            dtype,
            activation=activation,
        )
        factory_kwargs = {"device": device, "dtype": dtype}
        self.eps = eps
        self.weight = nn.Parameter(torch.empty(hidden_size, **factory_kwargs))
        self.register_parameter("bias", None)
        self.group_size = group_size
        self.norm_before_gate = norm_before_gate
        self.activation = activation
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)

    def forward_oot(self, x, z=None):
        if z is None:
            raise RuntimeError("Ascend RMSNormGated requires a gate tensor")
        return torch.ops.vllm.ascend_rms_norm_gated(
            x,
            z,
            self.weight,
            self.eps,
            -1 if self.group_size is None else self.group_size,
            self.norm_before_gate,
        )
