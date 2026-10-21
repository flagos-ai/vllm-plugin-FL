# SPDX-License-Identifier: Apache-2.0
"""DSV4 producer + dynamic per-token INT8 quantization fusions."""

from __future__ import annotations

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _round_int8(x):
    return tl.extra.cuda.libdevice.round(x).to(tl.int8)


@triton.jit
def _fused_q_kv_rmsnorm_int8_kernel(
    q_ptr,
    q_out_ptr,
    q_int8_ptr,
    q_scale_ptr,
    q_weight_ptr,
    q_in_stride,
    q_out_stride,
    kv_ptr,
    kv_out_ptr,
    kv_weight_ptr,
    kv_in_stride,
    kv_out_stride,
    eps,
    Q_SIZE: tl.constexpr,
    KV_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(0).to(tl.int64)
    pid_task = tl.program_id(1)
    if pid_task == 0:
        size = Q_SIZE
        row_in = q_ptr + token_idx * q_in_stride
        weight_ptr = q_weight_ptr
        row_out = q_out_ptr + token_idx * q_out_stride
    else:
        size = KV_SIZE
        row_in = kv_ptr + token_idx * kv_in_stride
        weight_ptr = kv_weight_ptr
        row_out = kv_out_ptr + token_idx * kv_out_stride

    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < size
    x = tl.load(row_in + offs, mask=mask, other=0.0).to(tl.float32)
    variance = tl.sum(x * x, axis=0) / size
    rrms = tl.rsqrt(variance + eps)
    w = tl.load(weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y_bf16 = (x * rrms * w).to(tl.bfloat16)
    tl.store(row_out + offs, y_bf16, mask=mask)

    if pid_task == 0:
        values = tl.where(mask, y_bf16.to(tl.float32), 0.0)
        absmax = tl.maximum(tl.max(tl.abs(values), axis=0), 1.0e-10)
        scale = absmax / 127.0
        quant = _round_int8(values / scale)
        tl.store(q_int8_ptr + token_idx * Q_SIZE + offs, quant, mask=mask)
        tl.store(q_scale_ptr + token_idx, scale)


def fused_q_kv_rmsnorm_int8_triton(
    qr: torch.Tensor,
    kv: torch.Tensor,
    q_weight: torch.Tensor,
    kv_weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    assert qr.ndim == 2 and kv.ndim == 2
    assert qr.shape[0] == kv.shape[0]
    q_size, kv_size = qr.shape[1], kv.shape[1]
    num_tokens = qr.shape[0]
    qr_out = torch.empty_like(qr)
    kv_out = torch.empty_like(kv)
    qr_int8 = torch.empty_like(qr, dtype=torch.int8)
    qr_scale = torch.empty((num_tokens, 1), dtype=torch.float32, device=qr.device)
    if num_tokens:
        block = triton.next_power_of_2(max(q_size, kv_size))
        _fused_q_kv_rmsnorm_int8_kernel[(num_tokens, 2)](
            qr,
            qr_out,
            qr_int8,
            qr_scale,
            q_weight,
            qr.stride(0),
            qr_out.stride(0),
            kv,
            kv_out,
            kv_weight,
            kv.stride(0),
            kv_out.stride(0),
            eps,
            Q_SIZE=q_size,
            KV_SIZE=kv_size,
            BLOCK_SIZE=block,
        )
    return qr_out, kv_out, qr_int8, qr_scale
