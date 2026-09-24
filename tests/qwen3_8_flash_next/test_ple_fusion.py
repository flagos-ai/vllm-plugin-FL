# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from vllm_fl.models.qwen3_8_flash_next.gpu.ops.ple_fusion import (
    can_use_ple_gate_norm_triton,
    ple_gate_norm,
    ple_prefill_short_conv_,
)

pytestmark = pytest.mark.gpu

NULL_BLOCK_ID = -1


def _grouped_norm_reference(
    x: torch.Tensor, weight: torch.Tensor, eps: float
) -> torch.Tensor:
    x_float = x.float()
    variance = x_float.square().mean(dim=-1, keepdim=True)
    return (
        x_float
        * torch.rsqrt(variance + eps)
        * (1.0 + weight.float().view(1, *x.shape[1:]))
    ).to(x.dtype)


def _gate_norm_reference(
    key: torch.Tensor,
    query: torch.Tensor,
    value: torch.Tensor,
    key_weight: torch.Tensor,
    query_weight: torch.Tensor,
    conv_weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    hidden_size = key.shape[-1]
    norm_key = _grouped_norm_reference(key, key_weight, eps)
    norm_query = _grouped_norm_reference(query, query_weight, eps)
    gate = (norm_key * norm_query).sum(dim=-1, keepdim=True) / math.sqrt(hidden_size)
    gate = torch.sigmoid(gate.sign() * gate.abs().clamp_min(1.0e-6).sqrt())
    gated = gate * value.unsqueeze(1)
    normalized = _grouped_norm_reference(gated, conv_weight, eps)
    return gated, normalized


def _prefill_conv_reference(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    conv_weight: torch.Tensor,
    query_start_loc: torch.Tensor,
    state_indices: torch.Tensor,
    has_initial_state: torch.Tensor,
    *,
    max_len: int,
    dilation: int,
) -> torch.Tensor:
    num_reqs = state_indices.numel()
    channels, kernel_width = conv_weight.shape
    state_len = (kernel_width - 1) * dilation
    lengths = query_start_loc[1:].long() - query_start_loc[:-1].long()
    positions = torch.arange(x.shape[0], device=x.device, dtype=torch.int64)
    req_indices = torch.searchsorted(query_start_loc[1:].long(), positions, right=True)
    col_indices = positions - query_start_loc.long()[req_indices]

    packed = x.new_zeros((num_reqs, max_len, channels))
    packed[req_indices, col_indices] = x
    packed = packed.transpose(1, 2).contiguous()
    valid_state = state_indices != NULL_BLOCK_ID
    safe_indices = torch.where(
        valid_state, state_indices.long(), torch.zeros_like(state_indices.long())
    )
    state = conv_state.index_select(0, safe_indices)[..., :state_len].to(x.dtype)
    initial = torch.where(
        (valid_state & has_initial_state).view(num_reqs, 1, 1),
        state,
        torch.zeros_like(state),
    )
    history = torch.cat((initial, packed), dim=-1)
    conv = F.conv1d(
        history,
        conv_weight.unsqueeze(1),
        groups=channels,
        dilation=dilation,
    )
    conv = F.silu(conv).transpose(1, 2).contiguous()
    valid_tokens = torch.arange(max_len, device=x.device).view(
        1, max_len
    ) < lengths.view(num_reqs, 1)
    conv.masked_fill_(~(valid_tokens & valid_state.view(num_reqs, 1)).unsqueeze(-1), 0)
    output = conv[req_indices, col_indices].contiguous()

    state_offsets = torch.arange(state_len, device=x.device).view(1, 1, state_len)
    next_state = history.gather(
        2,
        (lengths.view(num_reqs, 1, 1) + state_offsets).expand(-1, channels, -1),
    )
    update_mask = valid_state & (lengths > 0)
    selected = conv_state.index_select(0, safe_indices)
    selected[..., :state_len] = torch.where(
        update_mask.view(num_reqs, 1, 1),
        next_state.to(conv_state.dtype),
        selected[..., :state_len],
    )
    conv_state.index_copy_(0, safe_indices[update_mask], selected[update_mask])
    return output


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("hidden_size", [31, 32, 33, 127, 128, 129, 2560])
def test_ple_gate_norm_matches_reference_and_is_deterministic(
    dtype: torch.dtype, hidden_size: int
) -> None:
    torch.manual_seed(123 + hidden_size)
    device = torch.device("cuda")
    tokens = 3
    hc_count = 4
    shape = (tokens, hc_count, hidden_size)
    key = torch.randn(shape, device=device, dtype=dtype)
    query = torch.randn_like(key)
    value = torch.randn((tokens, hidden_size), device=device, dtype=dtype)
    weights = [
        torch.randn((hc_count * hidden_size,), device=device, dtype=dtype) * 0.01
        for _ in range(3)
    ]
    assert can_use_ple_gate_norm_triton(
        key, query, value, *weights, hc_count, min_tokens=0
    )
    assert not can_use_ple_gate_norm_triton(
        key, query, value, *weights, hc_count, min_tokens=tokens + 1
    )
    expected = _gate_norm_reference(key, query, value, *weights, 1.0e-6)
    actual = ple_gate_norm(key, query, value, *weights, hc_count, 1.0e-6)
    tolerance = 2.0e-5 if dtype == torch.float32 else 4.0e-2
    torch.testing.assert_close(actual[0], expected[0], atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(actual[1], expected[1], atol=tolerance, rtol=tolerance)
    for _ in range(10):
        repeated = ple_gate_norm(key, query, value, *weights, hc_count, 1.0e-6)
        assert torch.equal(repeated[0], actual[0])
        assert torch.equal(repeated[1], actual[1])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("lengths", [[1], [15, 16, 17], [0, 32, 33]])
def test_ple_prefill_conv_boundaries_state_and_determinism(
    dtype: torch.dtype, lengths: list[int]
) -> None:
    torch.manual_seed(456 + sum(lengths))
    device = torch.device("cuda")
    channels = 65
    kernel_width = 4
    dilation = 3
    state_len = (kernel_width - 1) * dilation
    query_start_loc = torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist()],
        device=device,
        dtype=torch.int32,
    )
    total_tokens = sum(lengths)
    x = torch.randn((total_tokens, channels), device=device, dtype=dtype)
    weight = torch.randn((channels, kernel_width), device=device, dtype=dtype)
    slots = max(len(lengths), 1) + 1
    initial_state = torch.randn(
        (slots, state_len + 2, channels), device=device, dtype=dtype
    ).transpose(1, 2)
    assert not initial_state.is_contiguous()
    state_indices = torch.arange(len(lengths), device=device, dtype=torch.int32)
    if len(lengths) > 1:
        state_indices[-1] = NULL_BLOCK_ID
    has_initial = torch.tensor(
        [(index % 2) == 0 for index in range(len(lengths))],
        device=device,
        dtype=torch.bool,
    )

    reference_state = initial_state.clone()
    expected = _prefill_conv_reference(
        x,
        reference_state,
        weight,
        query_start_loc,
        state_indices,
        has_initial,
        max_len=max(lengths),
        dilation=dilation,
    )
    actual_state = initial_state.clone()
    actual = torch.empty_like(x)
    threshold_output = torch.full_like(x, 7.0)
    threshold_state = initial_state.clone()
    assert not ple_prefill_short_conv_(
        x,
        threshold_output,
        threshold_state,
        weight,
        query_start_loc,
        state_indices,
        has_initial,
        num_prefills=len(lengths),
        max_len=max(lengths),
        state_len=state_len,
        kernel_width=kernel_width,
        dilation=dilation,
        null_block_id=NULL_BLOCK_ID,
        min_tokens=total_tokens + 1,
    )
    assert torch.equal(threshold_output, torch.full_like(x, 7.0))
    assert torch.equal(threshold_state, initial_state)
    assert ple_prefill_short_conv_(
        x,
        actual,
        actual_state,
        weight,
        query_start_loc,
        state_indices,
        has_initial,
        num_prefills=len(lengths),
        max_len=max(lengths),
        state_len=state_len,
        kernel_width=kernel_width,
        dilation=dilation,
        null_block_id=NULL_BLOCK_ID,
        min_tokens=0,
    )
    tolerance = 2.0e-5 if dtype == torch.float32 else 4.0e-2
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(
        actual_state, reference_state, atol=tolerance, rtol=tolerance
    )

    first_output = actual.clone()
    first_state = actual_state.clone()
    for _ in range(10):
        actual_state.copy_(initial_state)
        assert ple_prefill_short_conv_(
            x,
            actual,
            actual_state,
            weight,
            query_start_loc,
            state_indices,
            has_initial,
            num_prefills=len(lengths),
            max_len=max(lengths),
            state_len=state_len,
            kernel_width=kernel_width,
            dilation=dilation,
            null_block_id=NULL_BLOCK_ID,
            min_tokens=0,
        )
        assert torch.equal(actual, first_output)
        assert torch.equal(actual_state, first_state)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_ple_fusions_capture_and_replay() -> None:
    torch.manual_seed(789)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    tokens, hc_count, hidden_size = 65, 4, 129
    key = torch.randn((tokens, hc_count, hidden_size), device=device, dtype=dtype)
    query = torch.randn_like(key)
    value = torch.randn((tokens, hidden_size), device=device, dtype=dtype)
    weights = [
        torch.randn((hc_count * hidden_size,), device=device, dtype=dtype) * 0.01
        for _ in range(3)
    ]
    gated = torch.empty_like(key)
    normalized = torch.empty_like(key)

    channels, kernel_width, dilation = 129, 4, 3
    state_len = (kernel_width - 1) * dilation
    conv_x = torch.randn((tokens, channels), device=device, dtype=dtype)
    conv_weight = torch.randn((channels, kernel_width), device=device, dtype=dtype)
    query_start = torch.tensor([0, tokens], device=device, dtype=torch.int32)
    state_indices = torch.tensor([0], device=device, dtype=torch.int32)
    has_initial = torch.tensor([True], device=device, dtype=torch.bool)
    initial_state = torch.randn((2, channels, state_len), device=device, dtype=dtype)
    conv_state = initial_state.clone()
    conv_output = torch.empty_like(conv_x)

    # Eager warmup primes Triton compilation before capture.
    torch.ops.vllm.qwen3_8_flash_next_ple_gate_norm_(
        key, query, value, *weights, gated, normalized, hc_count, 1.0e-6
    )
    assert ple_prefill_short_conv_(
        conv_x,
        conv_output,
        conv_state,
        conv_weight,
        query_start,
        state_indices,
        has_initial,
        num_prefills=1,
        max_len=tokens,
        state_len=state_len,
        kernel_width=kernel_width,
        dilation=dilation,
        null_block_id=NULL_BLOCK_ID,
        min_tokens=0,
    )
    conv_state.copy_(initial_state)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, capture_error_mode="thread_local"):
        torch.ops.vllm.qwen3_8_flash_next_ple_gate_norm_(
            key, query, value, *weights, gated, normalized, hc_count, 1.0e-6
        )
        assert ple_prefill_short_conv_(
            conv_x,
            conv_output,
            conv_state,
            conv_weight,
            query_start,
            state_indices,
            has_initial,
            num_prefills=1,
            max_len=tokens,
            state_len=state_len,
            kernel_width=kernel_width,
            dilation=dilation,
            null_block_id=NULL_BLOCK_ID,
            min_tokens=0,
        )

    key.copy_(torch.randn_like(key))
    conv_x.copy_(torch.randn_like(conv_x))
    conv_state.copy_(initial_state)
    expected_gate = _gate_norm_reference(key, query, value, *weights, 1.0e-6)
    expected_state = initial_state.clone()
    expected_conv = _prefill_conv_reference(
        conv_x,
        expected_state,
        conv_weight,
        query_start,
        state_indices,
        has_initial,
        max_len=tokens,
        dilation=dilation,
    )
    allocated_before_replay = torch.cuda.memory_allocated()
    graph.replay()
    torch.cuda.synchronize()
    allocated_after_replay = torch.cuda.memory_allocated()
    assert allocated_after_replay == allocated_before_replay
    torch.testing.assert_close(gated, expected_gate[0], atol=4.0e-2, rtol=4.0e-2)
    torch.testing.assert_close(normalized, expected_gate[1], atol=4.0e-2, rtol=4.0e-2)
    torch.testing.assert_close(conv_output, expected_conv, atol=4.0e-2, rtol=4.0e-2)
    torch.testing.assert_close(conv_state, expected_state, atol=4.0e-2, rtol=4.0e-2)
