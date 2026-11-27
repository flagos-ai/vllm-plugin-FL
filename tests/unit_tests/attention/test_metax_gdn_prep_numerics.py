# Copyright (c) 2026 BAAI. All rights reserved.
"""Qwen3.5-4B's strided GDN prefill preparation versus CPU formulas."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from tests.utils.device_utils import get_backend

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        get_backend() != "metax" or not torch.cuda.is_available(),
        reason="MetaX GPU and MACA runtime required",
    ),
]

HK, HV, K, V = 16, 32, 128, 128
QKV_DIM = 2 * HK * K + HV * V


def _projection_inputs(length):
    rng = torch.Generator().manual_seed(32035)
    # Qwen3.5 slices qkv from [q,k,v,z], and b/a from [b,a]. Keep both
    # token strides and nonzero storage offsets instead of packing the views.
    qkvz = (torch.randn(length + 4, QKV_DIM + HV * V, generator=rng) * 0.4).bfloat16()
    ba = torch.randn(length + 4, 2 * HV, generator=rng).bfloat16()
    ba[2, HV : HV + 8] = torch.tensor([-60, -22, -5, 0, 5, 19, 22, 60])
    a_log = torch.linspace(-2, 1, HV)
    dt_bias = torch.linspace(-1.5, 1.5, HV)
    return qkvz, ba, a_log, dt_bias


def _views(qkvz, ba):
    return qkvz[2:-2, :QKV_DIM], ba[2:-2, HV:], ba[2:-2, :HV]


def _prep_reference(conv, a, b, a_log, dt_bias):
    q, k, v = conv.float().split([HK * K, HK * K, HV * V], dim=-1)
    q, k, v = q.view(-1, HK, K), k.view(-1, HK, K), v.view(-1, HV, V)
    q *= torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6)
    k *= torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
    g = -a_log.float().exp() * F.softplus(a.float() + dt_bias.float(), threshold=20)
    beta = b.float().sigmoid()
    return q.to(conv.dtype), k.to(conv.dtype), v.to(conv.dtype), g, beta


def _assert_prepared(actual, reference):
    for name, value, expected in zip(("q", "k", "v", "g", "beta"), actual, reference):
        # Gates stay float32; q/k/v are written back in the model dtype.
        atol, rtol = (3e-5, 3e-5) if name in ("g", "beta") else (0.001, 0.008)
        torch.testing.assert_close(
            value.cpu().float(), expected.float(), rtol=rtol, atol=atol, msg=name
        )
        assert value.is_contiguous()


@torch.inference_mode()
def test_real_gdn_post_conv_prep_handles_projection_slices_and_gates():
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
        fused_post_conv_prep,
    )

    owners = _projection_inputs(37)  # Tail of the 16-token Triton tile.
    conv, a, b = _views(*owners[:2])
    expected = _prep_reference(conv, a, b, *owners[2:])
    qkvz_gpu, ba_gpu, a_log_gpu, bias_gpu = (x.cuda().contiguous() for x in owners)
    conv_gpu, a_gpu, b_gpu = _views(qkvz_gpu, ba_gpu)
    assert conv_gpu.stride(0) == QKV_DIM + HV * V and conv_gpu.stride(1) == 1
    assert (
        a_gpu.stride(0) == 2 * HV and a_gpu.storage_offset() != b_gpu.storage_offset()
    )
    actual = fused_post_conv_prep(
        conv_output=conv_gpu,
        a=a_gpu,
        b=b_gpu,
        A_log=a_log_gpu,
        dt_bias=bias_gpu,
        num_k_heads=HK,
        head_k_dim=K,
        head_v_dim=V,
        apply_l2norm=True,
        output_g_exp=False,
    )
    torch.cuda.synchronize()
    _assert_prepared(actual, expected)


def _conv_reference(x, weight, states, boundaries, indices, has_initial):
    output = torch.empty_like(x)
    expected_states = states.clone()
    width = weight.shape[-1]
    for start, end, index, use_initial in zip(
        boundaries, boundaries[1:], indices, has_initial
    ):
        history = (
            states[index].float().clone()
            if use_initial
            else torch.zeros(QKV_DIM, width - 1)
        )
        for token in range(start, end):
            window = torch.cat((history, x[token].float().unsqueeze(-1)), dim=-1)
            output[token] = F.silu((window * weight.float()).sum(-1)).to(x.dtype)
            history = window[:, 1:]
        expected_states[index] = history.to(states.dtype)
    return output, expected_states


@torch.inference_mode()
def test_real_gdn_varlen_conv_and_post_prep_preserve_indexed_state_and_strides():
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
        causal_conv1d_fn,
        fused_post_conv_prep,
    )
    from vllm.v1.attention.backends.utils import compute_causal_conv1d_metadata

    boundaries, indices, has_initial = (0, 2, 13, 73), (3, 1, 4), (True, False, True)
    owners = _projection_inputs(boundaries[-1])
    x, _, _ = _views(*owners[:2])
    rng = torch.Generator().manual_seed(3532)
    weight = (torch.randn(QKV_DIM, 4, generator=rng) * 0.3).bfloat16()
    # Real allocation places conv and SSM states in one page. This gap between
    # conv states also catches kernels that flatten the indexed cache view.
    state_owner = (torch.randn(6, QKV_DIM * 3 + 257, generator=rng) * 0.2).bfloat16()
    states = state_owner[:, : QKV_DIM * 3].view(6, QKV_DIM, 3)
    expected_conv, expected_states = _conv_reference(
        x, weight, states, boundaries, indices, has_initial
    )
    expected_owner = state_owner.clone()
    expected_owner[:, : QKV_DIM * 3] = expected_states.flatten(1)

    qkvz_gpu, ba_gpu, a_log_gpu, bias_gpu = (x.cuda().contiguous() for x in owners)
    x_gpu, a_gpu, b_gpu = _views(qkvz_gpu, ba_gpu)
    state_owner_gpu = state_owner.cuda().contiguous()
    states_gpu = state_owner_gpu[:, : QKV_DIM * 3].view(6, QKV_DIM, 3)
    starts_cpu = torch.tensor(boundaries, dtype=torch.int32)
    nums, batch_ptr, chunk_ptr = compute_causal_conv1d_metadata(
        starts_cpu, device=torch.device("cuda")
    )
    conv_gpu = causal_conv1d_fn(
        x_gpu.transpose(0, 1),
        weight.cuda().contiguous(),
        None,
        conv_states=states_gpu,
        query_start_loc=starts_cpu.cuda(),
        cache_indices=torch.tensor(indices, device="cuda", dtype=torch.int32),
        has_initial_state=torch.tensor(has_initial, device="cuda"),
        activation="silu",
        metadata=SimpleNamespace(
            nums_dict=nums, batch_ptr=batch_ptr, token_chunk_offset_ptr=chunk_ptr
        ),
    ).transpose(0, 1)
    prepared = fused_post_conv_prep(
        conv_gpu,
        a_gpu,
        b_gpu,
        a_log_gpu,
        bias_gpu,
        HK,
        K,
        V,
        apply_l2norm=True,
        output_g_exp=False,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(
        conv_gpu.cpu().float(), expected_conv.float(), rtol=0.015, atol=0.003
    )
    # Cache writes are copies of pre-convolution inputs, and should be exact;
    # reserved/unused state blocks and the SSM padding must remain untouched.
    torch.testing.assert_close(state_owner_gpu.cpu(), expected_owner, rtol=0, atol=0)
    conv_cpu = conv_gpu.cpu()
    _, a, b = _views(*owners[:2])
    _assert_prepared(prepared, _prep_reference(conv_cpu, a, b, *owners[2:]))
