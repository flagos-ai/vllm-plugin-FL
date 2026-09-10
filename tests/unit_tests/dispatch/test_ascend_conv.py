# Copyright (c) 2026 BAAI. All rights reserved.

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("torch_npu")

from vllm.model_executor.warmup.qwen_triton_warmup import (
    _warm_causal_conv1d_fwd_kernel,
)
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID, PAD_SLOT_ID

from vllm_fl.dispatch.backends.vendor.ascend.impl.causal_conv1d import causal_conv1d_fn
from vllm_fl.dispatch.backends.vendor.ascend.patch import patch_causal_conv1d


@pytest.mark.gpu
def test_conv_prefill_preserves_padded_slots_and_upstream_warmup():
    generator = torch.Generator().manual_seed(41)
    x = torch.randn(8, 7, generator=generator).to(torch.bfloat16)
    weight = torch.randn(8, 4, generator=generator).to(torch.bfloat16)
    bias = torch.randn(8, generator=generator).to(torch.bfloat16)
    storage = torch.randn(4, 8 * 3 + 8, generator=generator).to(torch.bfloat16)
    expected_storage = storage.clone()
    expected_state = expected_storage[:, :-8].view(4, 8, 3)
    expected = torch.zeros_like(x)
    for start, end, index, has_initial in [(0, 2, 1, False), (2, 5, 3, True)]:
        state = expected_state[index].float() if has_initial else torch.zeros(8, 3)
        sequence = torch.cat([state, x[:, start:end].float()], dim=-1)
        output = F.conv1d(
            sequence.unsqueeze(0), weight.float().unsqueeze(1), bias.float(), groups=8
        )
        expected[:, start:end] = F.silu(output).squeeze(0).to(torch.bfloat16)
        expected_state[index].copy_(sequence[:, -3:])
    npu_storage = storage.npu()
    states = npu_storage[:, :-8].view(4, 8, 3)
    output = causal_conv1d_fn(
        x.npu(),
        weight.npu(),
        bias.npu(),
        states,
        torch.tensor([0, 2, 5, 6, 7], dtype=torch.int32, device="npu"),
        cache_indices=torch.tensor(
            [1, 3, NULL_BLOCK_ID, PAD_SLOT_ID], dtype=torch.int32, device="npu"
        ),
        has_initial_state=torch.tensor([False, True, True, False], device="npu"),
        activation="silu",
        null_block_id=NULL_BLOCK_ID,
        validate_data=False,
    )
    torch.testing.assert_close(output.cpu(), expected, rtol=0.02, atol=0.04)
    torch.testing.assert_close(npu_storage.cpu(), expected_storage)

    patch_causal_conv1d()
    _warm_causal_conv1d_fwd_kernel(
        torch.device("npu"),
        SimpleNamespace(
            conv_dim=8,
            conv_kernel_size=4,
            conv_dtype=torch.bfloat16,
            conv_state=states,
        ),
    )
    torch.testing.assert_close(npu_storage.cpu(), expected_storage)


@pytest.mark.gpu
def test_conv_prefill_layout_matches_gdn_fused_post_conv():
    from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as gdn

    generator = torch.Generator().manual_seed(47)
    tokens, key_heads, value_heads, head_dim = 25, 8, 24, 128
    channels = (2 * key_heads + value_heads) * head_dim
    x = torch.randn(tokens, channels, generator=generator).to(torch.bfloat16).npu()
    weight = (
        (torch.randn(channels, 4, generator=generator) * 0.1).to(torch.bfloat16).npu()
    )
    states = torch.zeros(2, 3, channels, device="npu", dtype=torch.bfloat16).transpose(
        1, 2
    )
    conv = causal_conv1d_fn(
        x.T,
        weight,
        None,
        states,
        torch.tensor([0, tokens], device="npu", dtype=torch.int32),
        cache_indices=torch.tensor([1], device="npu", dtype=torch.int32),
        has_initial_state=torch.tensor([False], device="npu"),
    ).T
    assert conv.stride(-1) == 1
    a, b = [
        torch.randn(tokens, value_heads, generator=generator).to(torch.bfloat16).npu()
        for _ in range(2)
    ]
    a_log = torch.randn(value_heads, generator=generator).npu()
    bias = torch.randn(value_heads, generator=generator).to(torch.bfloat16).npu()
    q, k, v, g, beta = gdn.fused_post_conv_prep(
        conv, a, b, a_log, bias, key_heads, head_dim, head_dim, True, False
    )
    expected_q, expected_k, expected_v = (
        conv.cpu()
        .float()
        .split(
            [key_heads * head_dim, key_heads * head_dim, value_heads * head_dim], dim=-1
        )
    )
    for actual, expected in [(q, expected_q), (k, expected_k)]:
        expected = F.normalize(expected.view(tokens, key_heads, head_dim), dim=-1)
        torch.testing.assert_close(
            actual.cpu().float(), expected, rtol=0.01, atol=0.002
        )
    torch.testing.assert_close(v.cpu().float(), expected_v.view_as(v))
    torch.testing.assert_close(
        g.cpu(), -a_log.cpu().exp() * F.softplus(a.cpu().float() + bias.cpu().float())
    )
    torch.testing.assert_close(beta.cpu(), b.cpu().float().sigmoid())
