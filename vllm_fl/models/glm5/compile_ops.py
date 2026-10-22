# SPDX-License-Identifier: Apache-2.0
"""Opaque GLM numerical boundaries around the common policy binding.

The selected numerical APIs allocate fresh outputs. Fake implementations
describe their dense GLM ABI without resolving policy or optional backends.
"""

import torch

from vllm.utils.torch_utils import direct_register_custom_op


def _binding(name):
    from vllm_fl.models.glm5.runtime import _portable_binding

    return _portable_binding(name)


def _mhc_pre(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 1,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _binding("mhc_pre")(
        None,
        residual,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        n_splits=n_splits,
        norm_weight=norm_weight,
        norm_eps=norm_eps,
    )


def _mhc_post(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    return _binding("mhc_post")(None, x, residual, post_layer_mix, comb_res_mix)


def _mhc_fused_post_pre(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 1,
    tile_n: int = 1,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return _binding("mhc_fused_post_pre")(
        None,
        x,
        residual,
        post_layer_mix,
        comb_res_mix,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        n_splits=n_splits,
        tile_n=tile_n,
        norm_weight=norm_weight,
        norm_eps=norm_eps,
    )


def _clamp(
    x: torch.Tensor,
    swiglu_limit: float,
    alpha: float,
    beta: float,
) -> torch.Tensor:
    from types import SimpleNamespace

    state = SimpleNamespace(swiglu_limit=swiglu_limit, alpha=alpha, beta=beta)
    return _binding("silu_and_mul_with_clamp")(state, x)


def _fwht128_quant_fp8(q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    from vllm_fl.models.glm5.indexer_backend import INDEXER_BACKEND

    return INDEXER_BACKEND.fwht128_quant_fp8(q)


def _pre_outputs(residual):
    outer = residual.shape[:-2]
    streams, hidden = residual.shape[-2:]
    return (
        residual.new_empty((*outer, streams, 1), dtype=torch.float32),
        residual.new_empty((*outer, streams, streams), dtype=torch.float32),
        residual.new_empty((*outer, hidden), dtype=torch.bfloat16),
    )


def _mhc_pre_fake(
    residual,
    fn,
    hc_scale,
    hc_base,
    rms_eps,
    hc_pre_eps,
    hc_sinkhorn_eps,
    hc_post_mult_value,
    sinkhorn_repeat,
    n_splits=1,
    norm_weight=None,
    norm_eps=0.0,
):
    return _pre_outputs(residual)


def _mhc_post_fake(x, residual, post_layer_mix, comb_res_mix):
    return torch.empty_like(residual, dtype=torch.bfloat16)


def _mhc_fused_post_pre_fake(
    x,
    residual,
    post_layer_mix,
    comb_res_mix,
    fn,
    hc_scale,
    hc_base,
    rms_eps,
    hc_pre_eps,
    hc_sinkhorn_eps,
    hc_post_mult_value,
    sinkhorn_repeat,
    n_splits=1,
    tile_n=1,
    norm_weight=None,
    norm_eps=0.0,
):
    return (torch.empty_like(residual, dtype=torch.bfloat16), *_pre_outputs(residual))


def _clamp_fake(x, swiglu_limit, alpha, beta):
    return x.new_empty((*x.shape[:-1], x.shape[-1] // 2))


def _fwht128_quant_fp8_fake(q):
    from vllm.platforms import current_platform

    dtype = torch.float8_e4m3fn if current_platform.supports_fp8() else torch.bfloat16
    return (
        q.new_empty(q.shape, dtype=dtype),
        q.new_empty((q.shape[0], 1), dtype=torch.float32),
    )


for _name, _host, _fake in (
    ("glm5_mhc_pre", _mhc_pre, _mhc_pre_fake),
    ("glm5_mhc_post", _mhc_post, _mhc_post_fake),
    ("glm5_mhc_fused_post_pre", _mhc_fused_post_pre, _mhc_fused_post_pre_fake),
    ("glm5_silu_and_mul_with_clamp", _clamp, _clamp_fake),
    ("glm5_fwht128_quant_fp8", _fwht128_quant_fp8, _fwht128_quant_fp8_fake),
):
    direct_register_custom_op(
        op_name=_name,
        op_func=_host,
        mutates_args=[],
        fake_impl=_fake,
        dispatch_key="CompositeExplicitAutograd",
    )


def forward_mhc_pre(self, *args, **kwargs):
    return torch.ops.vllm.glm5_mhc_pre(*args, **kwargs)


def forward_mhc_post(self, *args, **kwargs):
    return torch.ops.vllm.glm5_mhc_post(*args, **kwargs)


def forward_mhc_fused_post_pre(self, *args, **kwargs):
    return torch.ops.vllm.glm5_mhc_fused_post_pre(*args, **kwargs)


def forward_clamp(self, x):
    return torch.ops.vllm.glm5_silu_and_mul_with_clamp(
        x, self.swiglu_limit, self.alpha, self.beta
    )


def fwht128_quant_fp8(q):
    return torch.ops.vllm.glm5_fwht128_quant_fp8(q)
