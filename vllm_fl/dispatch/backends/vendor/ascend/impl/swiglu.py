# Copyright © 2025 Huawei Technologies Co., Ltd.
# Copyright 2024 LinkedIn Corporation
import functools
import torch
import triton

import triton.language as tl
import torch.nn as nn


def ensure_contiguous(fn):
    @functools.wraps(fn)
    def wrapper(ctx, *args, **kwargs):
        def maybe_to_contiguous(x):
            return x.contiguous() if isinstance(x, torch.Tensor) else x

        args = [maybe_to_contiguous(arg) for arg in args]
        kwargs = {k: maybe_to_contiguous(v) for k, v in kwargs.items()}
        return fn(ctx, *args, **kwargs)

    return wrapper


@triton.jit
def silu(x):
    return x * tl.sigmoid(x)


@triton.jit
def _swiglu_forward_kernel(
    a_ptr, b_ptr, c_ptr, stride, n_cols: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    program_id = tl.program_id(0).to(tl.int64)

    # locate start index
    a_ptr += program_id * stride
    b_ptr += program_id * stride
    c_ptr += program_id * stride

    base_offsets = tl.arange(0, BLOCK_SIZE)
    for i in range(0, n_cols, BLOCK_SIZE):
        col_offsets = base_offsets + i
        mask = col_offsets < n_cols

        # sigmoid requires type float32
        a_row = tl.load(a_ptr + col_offsets, mask=mask, other=0).to(tl.float32)
        b_row = tl.load(b_ptr + col_offsets, mask=mask, other=0)
        c_row = silu(a_row).cast(b_row.dtype) * b_row
        tl.store(c_ptr + col_offsets, c_row, mask=mask)


@triton.jit
def _swiglu_backward_kernel(
    dc_ptr, a_ptr, b_ptr, stride, n_cols: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    program_id = tl.program_id(0).to(tl.int64)

    # locate start index
    dc_ptr += program_id * stride
    a_ptr += program_id * stride
    b_ptr += program_id * stride

    base_offsets = tl.arange(0, BLOCK_SIZE)
    for i in range(0, n_cols, BLOCK_SIZE):
        col_offsets = base_offsets + i
        mask = col_offsets < n_cols

        dc_row = tl.load(dc_ptr + col_offsets, mask=mask, other=0)
        # sigmoid requires type float32
        a_row = tl.load(a_ptr + col_offsets, mask=mask, other=0).to(tl.float32)
        b_row = tl.load(b_ptr + col_offsets, mask=mask, other=0)

        # recomputation to save memory
        sig_a = tl.sigmoid(a_row)
        silu_a = a_row * sig_a
        db_row = dc_row * silu_a
        da_row = dc_row * (silu_a * (1 - sig_a) + sig_a) * b_row

        tl.store(a_ptr + col_offsets, da_row, mask=mask)
        tl.store(b_ptr + col_offsets, db_row, mask=mask)


def swiglu_forward(a, b):
    ori_shape = a.shape

    n_cols = ori_shape[-1]
    a = a.view(-1, n_cols)
    b = b.view(-1, n_cols)
    c = torch.empty_like(a)
    n_rows = a.shape[0]

    BLOCK_SIZE = min(5120, triton.next_power_of_2(n_cols))

    _swiglu_forward_kernel[(n_rows,)](
        a,
        b,
        c,
        c.stride(-2),
        n_cols=n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return a, b, c.view(*ori_shape)


def swiglu_backward(a, b, dc):
    ori_shape = dc.shape
    n_cols = ori_shape[-1]
    dc = dc.view(-1, n_cols)
    n_rows = dc.shape[0]

    BLOCK_SIZE = min(4096, triton.next_power_of_2(n_cols))

    _swiglu_backward_kernel[(n_rows,)](
        dc,
        a,
        b,
        dc.stride(-2),
        n_cols=n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return a.view(*ori_shape), b.view(*ori_shape)


class SiLUMulFunction(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(ctx, a, b):
        a, b, c = swiglu_forward(a, b)
        ctx.save_for_backward(a, b)
        return c

    @staticmethod
    @ensure_contiguous
    def backward(ctx, dc):
        a, b = ctx.saved_tensors
        a, b = swiglu_backward(a, b, dc)
        return a, b


class SwiGLUMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        if config.hidden_act not in ["silu", "swish"]:
            raise ValueError(f"Activation function {config.hidden_act} not supported.")

    def forward(self, x):
        return self.down_proj(SiLUMulFunction.apply(self.gate_proj(x), self.up_proj(x)))


class BlockSparseTop2MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ffn_dim = config.intermediate_size
        self.hidden_dim = config.hidden_size

        self.w1 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)
        self.w2 = nn.Linear(self.ffn_dim, self.hidden_dim, bias=False)
        self.w3 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)

        if config.hidden_act not in ["silu", "swish"]:
            raise ValueError(f"Activation function {config.hidden_act} not supported.")

    def forward(self, x):
        return self.w2(SiLUMulFunction.apply(self.w1(x), self.w3(x)))


class Phi3SwiGLUMLP(nn.Module):
    """
    Patch Phi3MLP to use SiLUMulFunction
    https://github.com/huggingface/transformers/blob/v4.41.0/src/transformers/models/phi3/modeling_phi3.py#L241
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_up_proj = nn.Linear(
            self.hidden_size, 2 * self.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        if config.hidden_act not in ["silu", "swish"]:
            raise ValueError(f"Activation function {config.hidden_act} not supported.")

    def forward(self, x):
        up_states = self.gate_up_proj(x)
        gate, up_states = up_states.chunk(2, dim=-1)
        return self.down_proj(SiLUMulFunction.apply(gate, up_states))


class Qwen3MoeSwiGLUMLP(nn.Module):
    """
    Patch Qwen3MoeMLP to use SiLUMulFunction.
    https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/qwen3_moe/modular_qwen3_moe.py#L57
    """

    def __init__(self, config, intermediate_size=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = (
            intermediate_size
            if intermediate_size is not None
            else config.intermediate_size
        )
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        if config.hidden_act not in ["silu", "swish"]:
            raise ValueError(f"Activation function {config.hidden_act} not supported.")

    def forward(self, x):
        return self.down_proj(SiLUMulFunction.apply(self.gate_proj(x), self.up_proj(x)))
