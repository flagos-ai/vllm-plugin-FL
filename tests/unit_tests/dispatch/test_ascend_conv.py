# Copyright (c) 2026 BAAI. All rights reserved.

import inspect
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("torch_npu")

from vllm.model_executor.warmup.qwen_triton_warmup import (
    _warm_causal_conv1d_fwd_kernel,
)
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID, PAD_SLOT_ID

from vllm_fl.dispatch.backends.vendor.ascend.impl.causal_conv1d import (
    causal_conv1d_fn,
    causal_conv1d_update_npu,
)
from vllm_fl.dispatch.backends.vendor.ascend.patch import patch_causal_conv1d


def test_conv_update_matches_vllm_028_call_contract():
    assert tuple(inspect.signature(causal_conv1d_update_npu).parameters) == (
        "x",
        "conv_state",
        "weight",
        "bias",
        "activation",
        "conv_state_indices",
        "num_accepted_tokens",
        "query_start_loc",
        "max_query_len",
        "null_block_id",
        "block_idx_last_scheduled_token",
        "initial_state_idx",
        "validate_data",
        "out",
    )
    assert (
        inspect.signature(causal_conv1d_update_npu).parameters["null_block_id"].default
        == NULL_BLOCK_ID
    )


@pytest.mark.gpu
def test_conv_update_honors_out_buffer_and_null_block():
    generator = torch.Generator().manual_seed(53)
    batch, dim, width, num_cache_lines = 3, 8, 4, 4
    state_len = width - 1
    null_block_id = NULL_BLOCK_ID

    x_cpu = (torch.randn(batch, dim, generator=generator) * 0.25).to(torch.bfloat16)
    weight_cpu = (torch.randn(dim, width, generator=generator) * 0.25).to(
        torch.bfloat16
    )
    bias_cpu = (torch.randn(dim, generator=generator) * 0.1).to(torch.bfloat16)
    state_storage_cpu = (
        torch.randn(num_cache_lines, state_len, dim, generator=generator) * 0.25
    ).to(torch.bfloat16)
    initial_states = state_storage_cpu.transpose(1, 2).clone()
    expected_states = initial_states.clone()
    expected_out = torch.full_like(x_cpu, -7)
    state_indices = [1, null_block_id, 3]

    for batch_idx, state_idx in enumerate(state_indices):
        if state_idx == null_block_id:
            continue
        history = initial_states[state_idx].to(torch.float16).float()
        current = x_cpu[batch_idx].to(torch.float16).float().unsqueeze(-1)
        window = torch.cat((history, current), dim=-1)
        expected_out[batch_idx] = (
            (window * weight_cpu.float()).sum(dim=-1) + bias_cpu.float()
        ).to(torch.bfloat16)
        expected_states[state_idx] = torch.cat(
            (initial_states[state_idx, :, 1:], x_cpu[batch_idx].unsqueeze(-1)),
            dim=-1,
        )

    x = x_cpu.npu()
    x_before = x.clone()
    states = state_storage_cpu.npu().transpose(1, 2)
    out = torch.full_like(x, -7)
    result = causal_conv1d_update_npu(
        x,
        states,
        weight_cpu.npu(),
        bias_cpu.npu(),
        activation=None,
        conv_state_indices=torch.tensor(state_indices, dtype=torch.int32, device="npu"),
        null_block_id=null_block_id,
        validate_data=True,
        out=out,
    )

    assert result.data_ptr() == out.data_ptr()
    torch.testing.assert_close(x.cpu(), x_before.cpu())
    torch.testing.assert_close(result.cpu(), expected_out, rtol=0.02, atol=0.02)
    torch.testing.assert_close(states.cpu(), expected_states, rtol=0, atol=0)


@pytest.mark.gpu
def test_conv_update_varlen_spec_honors_out_and_accepted_state():
    generator = torch.Generator().manual_seed(59)
    dim, width, max_query_len, num_cache_lines = 8, 4, 3, 5
    state_len = width - 1 + max_query_len - 1
    sequence_lengths = [3, 2]
    state_indices = [1, 3]
    num_accepted_tokens = [1, 2]

    x_cpu = (torch.randn(sum(sequence_lengths), dim, generator=generator) * 0.25).to(
        torch.bfloat16
    )
    weight_cpu = (torch.randn(dim, width, generator=generator) * 0.25).to(
        torch.bfloat16
    )
    bias_cpu = (torch.randn(dim, generator=generator) * 0.1).to(torch.bfloat16)
    state_storage_cpu = (
        torch.randn(num_cache_lines, state_len, dim, generator=generator) * 0.25
    ).to(torch.bfloat16)
    initial_states = state_storage_cpu.transpose(1, 2).clone()
    expected_states = initial_states.clone()
    expected_out = torch.full_like(x_cpu, -9)

    token_start = 0
    for sequence_len, state_idx, num_accepted in zip(
        sequence_lengths, state_indices, num_accepted_tokens
    ):
        token_end = token_start + sequence_len
        state_offset = num_accepted - 1
        history = (
            initial_states[state_idx, :, state_offset : state_offset + width - 1]
            .to(torch.float16)
            .float()
        )

        for token_idx in range(token_start, token_end):
            current = x_cpu[token_idx].to(torch.float16).float()
            window = torch.cat((history, current.unsqueeze(-1)), dim=-1)
            expected_out[token_idx] = (
                (window * weight_cpu.float()).sum(dim=-1) + bias_cpu.float()
            ).to(torch.bfloat16)
            history = torch.cat((history[:, 1:], current.unsqueeze(-1)), dim=-1)

        # Spec decode slides one accepted-token position, keeps width - 2
        # historical entries, then appends this request's draft tokens.
        keep = width - 2
        expected_states[state_idx, :, :keep] = initial_states[
            state_idx, :, state_offset + 1 : state_offset + 1 + keep
        ]
        expected_states[state_idx, :, keep : keep + sequence_len] = x_cpu[
            token_start:token_end
        ].T
        token_start = token_end

    x = x_cpu.npu()
    x_before = x.clone()
    states = state_storage_cpu.npu().transpose(1, 2)
    out = torch.full_like(x, -9)
    result = causal_conv1d_update_npu(
        x,
        states,
        weight_cpu.npu(),
        bias_cpu.npu(),
        activation=None,
        conv_state_indices=torch.tensor(state_indices, dtype=torch.int32, device="npu"),
        num_accepted_tokens=torch.tensor(
            num_accepted_tokens, dtype=torch.int32, device="npu"
        ),
        query_start_loc=torch.tensor([0, 3, 5], dtype=torch.int32, device="npu"),
        max_query_len=max_query_len,
        validate_data=True,
        out=out,
    )

    assert result.data_ptr() == out.data_ptr()
    torch.testing.assert_close(x.cpu(), x_before.cpu(), rtol=0, atol=0)
    torch.testing.assert_close(result.cpu(), expected_out, rtol=0.02, atol=0.02)
    torch.testing.assert_close(states.cpu(), expected_states, rtol=0, atol=0)


@pytest.mark.gpu
def test_conv_update_apc_routes_state_and_honors_out_buffer():
    generator = torch.Generator().manual_seed(61)
    batch, dim, width, num_cache_lines = 3, 8, 4, 6
    state_len = width - 1
    # Each row is a state block table. The second column is read as the
    # initial state and the third column receives the updated state.
    state_block_table = [[0, 1, 3], [0, 2, 4], [0, 0, 5]]
    initial_state_idx = [1, 1, 1]
    last_scheduled_idx = [2, 2, 2]

    x_cpu = (torch.randn(batch, dim, generator=generator) * 0.25).to(torch.bfloat16)
    weight_cpu = (torch.randn(dim, width, generator=generator) * 0.25).to(
        torch.bfloat16
    )
    bias_cpu = (torch.randn(dim, generator=generator) * 0.1).to(torch.bfloat16)
    state_storage_cpu = (
        torch.randn(num_cache_lines, state_len, dim, generator=generator) * 0.25
    ).to(torch.bfloat16)
    initial_states = state_storage_cpu.transpose(1, 2).clone()
    expected_states = initial_states.clone()
    expected_out = torch.full_like(x_cpu, -11)

    for batch_idx, block_table in enumerate(state_block_table):
        source_idx = block_table[initial_state_idx[batch_idx]]
        destination_idx = block_table[last_scheduled_idx[batch_idx]]
        if source_idx == NULL_BLOCK_ID:
            continue

        history = initial_states[source_idx].to(torch.float16).float()
        current = x_cpu[batch_idx].to(torch.float16).float()
        window = torch.cat((history, current.unsqueeze(-1)), dim=-1)
        conv = (window * weight_cpu.float()).sum(dim=-1) + bias_cpu.float()
        expected_out[batch_idx] = F.silu(conv).to(torch.bfloat16)
        expected_states[destination_idx] = torch.cat(
            (initial_states[source_idx, :, 1:], x_cpu[batch_idx].unsqueeze(-1)),
            dim=-1,
        )

    x = x_cpu.npu()
    x_before = x.clone()
    states = state_storage_cpu.npu().transpose(1, 2)
    out = torch.full_like(x, -11)
    result = causal_conv1d_update_npu(
        x,
        states,
        weight_cpu.npu(),
        bias_cpu.npu(),
        activation="silu",
        conv_state_indices=torch.tensor(
            state_block_table, dtype=torch.int32, device="npu"
        ),
        block_idx_last_scheduled_token=torch.tensor(
            last_scheduled_idx, dtype=torch.int32, device="npu"
        ),
        initial_state_idx=torch.tensor(
            initial_state_idx, dtype=torch.int32, device="npu"
        ),
        null_block_id=NULL_BLOCK_ID,
        validate_data=True,
        out=out,
    )

    assert result.data_ptr() == out.data_ptr()
    torch.testing.assert_close(x.cpu(), x_before.cpu(), rtol=0, atol=0)
    torch.testing.assert_close(result.cpu(), expected_out, rtol=0.02, atol=0.02)
    torch.testing.assert_close(states.cpu(), expected_states, rtol=0, atol=0)


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
