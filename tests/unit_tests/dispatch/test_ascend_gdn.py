# Copyright (c) 2026 BAAI. All rights reserved.

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("torch_npu")

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as gdn

from vllm_fl.dispatch.backends.vendor.ascend.patch import (
    patch_causal_conv1d,
    patch_fla_ops,
)
from vllm_fl.patches.gdn_packed_decode import patch_vllm_packed_gdn_beta


def test_ascend_patches_current_qwen_gdn_imports():
    from vllm_fl.dispatch.backends.vendor.ascend.impl.causal_conv1d import (
        causal_conv1d_fn,
        causal_conv1d_update_npu,
    )
    from vllm_fl.dispatch.backends.vendor.ascend.impl.fla.compat import (
        chunk_gated_delta_rule,
    )

    patch_causal_conv1d()
    patch_fla_ops()
    assert gdn.causal_conv1d_fn is causal_conv1d_fn
    assert gdn.causal_conv1d_update is causal_conv1d_update_npu
    assert gdn.fla_chunk_gated_delta_rule is chunk_gated_delta_rule


def _assert_gdn_chunk_matches_recurrence(
    key_dim, key_heads, value_heads, sequence_lengths
):
    patch_fla_ops()
    generator = torch.Generator().manual_seed(23)
    num_tokens = sum(sequence_lengths)
    q, k = [
        F.normalize(
            torch.randn(1, num_tokens, key_heads, key_dim, generator=generator),
            dim=-1,
        ).to(torch.bfloat16)
        for _ in range(2)
    ]
    v = torch.randn(1, num_tokens, value_heads, 128, generator=generator).to(
        torch.bfloat16
    )
    g = -torch.rand(1, num_tokens, value_heads, generator=generator)
    beta = torch.rand(1, num_tokens, value_heads, generator=generator).to(
        torch.bfloat16
    )
    initial = (
        torch.randn(
            len(sequence_lengths),
            value_heads,
            128,
            key_dim,
            generator=generator,
        )
        * 0.1
    )
    expected_state = initial.clone()
    expected = torch.empty_like(v, dtype=torch.float32)
    scale = key_dim**-0.5
    boundaries = [0]
    for length in sequence_lengths:
        boundaries.append(boundaries[-1] + length)
    for sequence, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
        state = expected_state[sequence]
        for token in range(start, end):
            state *= g[0, token].exp()[:, None, None]
            key = k[0, token].float().repeat_interleave(value_heads // key_heads, dim=0)
            residual = v[0, token].float() - torch.einsum("hvk,hk->hv", state, key)
            residual *= beta[0, token].float()[:, None]
            state += torch.einsum("hv,hk->hvk", residual, key)
            expected[0, token] = (
                torch.einsum(
                    "hvk,hk->hv",
                    state,
                    q[0, token]
                    .float()
                    .repeat_interleave(value_heads // key_heads, dim=0),
                )
                * scale
            )

    buffer = torch.full((v.numel() + 8,), 7, device="npu", dtype=torch.bfloat16)
    config = VllmConfig()
    config.model_config = SimpleNamespace(
        hf_text_config=SimpleNamespace(linear_key_head_dim=key_dim)
    )
    with torch.inference_mode(), set_current_vllm_config(config):
        op = gdn.ChunkGatedDeltaRule()
        output, state = op(
            q=q.npu(),
            k=k.npu(),
            v=v.npu(),
            g=g.npu(),
            beta=beta.npu(),
            initial_state=initial.npu(),
            output_final_state=True,
            cu_seqlens=torch.tensor(boundaries, dtype=torch.int32, device="npu"),
            chunk_indices=torch.empty((0, 2), dtype=torch.int32, device="npu"),
            chunk_offsets=torch.empty(0, dtype=torch.int32, device="npu"),
            use_qk_l2norm_in_kernel=False,
            core_attn_out=buffer,
        )
    assert output.data_ptr() == buffer.data_ptr()
    torch.testing.assert_close(output.cpu().float(), expected, rtol=0.03, atol=0.01)
    torch.testing.assert_close(state.cpu(), expected_state, rtol=0.03, atol=0.03)
    assert torch.all(buffer[-8:].cpu() == 7)


@pytest.mark.gpu
@pytest.mark.parametrize("key_dim", [64, 128])
@pytest.mark.parametrize("key_heads,value_heads", [(2, 2), (8, 16), (8, 24)])
def test_gdn_chunk_matches_recurrence_with_v_first_state(
    key_dim, key_heads, value_heads
):
    _assert_gdn_chunk_matches_recurrence(
        key_dim, key_heads, value_heads, sequence_lengths=(5, 4)
    )


@pytest.mark.gpu
@pytest.mark.parametrize("sequence_length", [65, 128])
def test_gdn_single_sequence_crosses_chunk_boundary(sequence_length):
    # Qwen3.6-35B-A3B has H=16/HV=32 globally and uses TP=2, so each worker
    # executes the Ascend chunk kernel with the H=8/HV=16 geometry below.
    _assert_gdn_chunk_matches_recurrence(
        key_dim=128,
        key_heads=8,
        value_heads=16,
        sequence_lengths=(sequence_length,),
    )


@pytest.mark.gpu
@pytest.mark.parametrize("state_indices", [[1], [1, 3]])
@pytest.mark.parametrize("key_heads,value_heads", [(2, 2), (8, 16), (16, 32)])
def test_packed_decode_updates_only_selected_padded_states(
    state_indices, key_heads, value_heads
):
    patch_vllm_packed_gdn_beta()
    generator = torch.Generator().manual_seed(31)
    batch_size = len(state_indices)
    q, k = [
        torch.randn(batch_size, key_heads, 128, generator=generator).to(torch.bfloat16)
        for _ in range(2)
    ]
    v = torch.randn(batch_size, value_heads, 128, generator=generator).to(
        torch.bfloat16
    )
    a, b = [
        torch.randn(batch_size, value_heads, generator=generator).to(torch.bfloat16)
        for _ in range(2)
    ]
    a_log = torch.linspace(-1.0, -0.5, value_heads)
    bias = torch.linspace(0.1, -0.2, value_heads)
    storage = torch.randn(4, value_heads * 128 * 128 + 16, generator=generator) * 0.1
    expected_storage = storage.clone()
    expected_state = expected_storage[:, :-16].view(4, value_heads, 128, 128)
    expected_output = torch.empty(batch_size, 1, value_heads, 128)
    for row, state_index in enumerate(state_indices):
        state = expected_state[state_index]
        key = F.normalize(k[row].float(), dim=-1)
        query = F.normalize(q[row].float(), dim=-1)
        decay = (-a_log.exp() * F.softplus(a[row].float() + bias)).exp()
        state *= decay[:, None, None]
        expanded_key = key.repeat_interleave(value_heads // key_heads, dim=0)
        expanded_query = query.repeat_interleave(value_heads // key_heads, dim=0)
        residual = v[row].float() - torch.einsum("hvk,hk->hv", state, expanded_key)
        residual *= b[row].float().sigmoid()[:, None]
        state += torch.einsum("hv,hk->hvk", residual, expanded_key)
        expected_output[row, 0] = (
            torch.einsum("hvk,hk->hv", state, expanded_query) * 128**-0.5
        )

    npu_storage = storage.npu()
    state = npu_storage[:, :-16].view(4, value_heads, 128, 128)
    out = torch.empty(
        batch_size,
        1,
        value_heads,
        128,
        device="npu",
        dtype=torch.bfloat16,
    )
    gdn.fused_recurrent_gated_delta_rule_packed_decode(
        mixed_qkv=torch.cat([x.flatten(1) for x in (q, k, v)], dim=1).npu(),
        a=a.npu(),
        b=b.npu(),
        A_log=a_log.npu(),
        dt_bias=bias.npu(),
        scale=128**-0.5,
        initial_state=state,
        out=out,
        ssm_state_indices=torch.tensor(state_indices, device="npu", dtype=torch.int32),
        use_qk_l2norm_in_kernel=True,
    )
    torch.testing.assert_close(out.cpu().float(), expected_output, rtol=0.03, atol=0.02)
    actual_storage = npu_storage.cpu()
    untouched = [row for row in range(4) if row not in state_indices]
    torch.testing.assert_close(
        actual_storage[untouched], storage[untouched], rtol=0, atol=0
    )
    torch.testing.assert_close(
        actual_storage[:, -16:], storage[:, -16:], rtol=0, atol=0
    )

    # Ascend and CPU use different FP32 reduction trees for the 128-wide
    # state update. A few near-zero elements have a relatively large error,
    # while the mean error remains small. Check both bounds instead of using
    # one loose elementwise absolute tolerance for the whole state tensor.
    selected_error = (
        actual_storage[state_indices, :-16] - expected_storage[state_indices, :-16]
    ).abs()
    assert selected_error.mean().item() < 5e-4
    assert selected_error.max().item() < 0.25
