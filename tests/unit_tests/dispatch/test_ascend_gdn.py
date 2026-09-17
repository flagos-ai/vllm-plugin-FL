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


@pytest.mark.gpu
@pytest.mark.parametrize("key_dim", [64, 128])
@pytest.mark.parametrize("key_heads,value_heads", [(2, 2), (8, 24)])
def test_gdn_chunk_matches_recurrence_with_v_first_state(
    key_dim, key_heads, value_heads
):
    patch_fla_ops()
    generator = torch.Generator().manual_seed(23)
    q, k = [
        F.normalize(
            torch.randn(1, 9, key_heads, key_dim, generator=generator), dim=-1
        ).to(torch.bfloat16)
        for _ in range(2)
    ]
    v = torch.randn(1, 9, value_heads, 128, generator=generator).to(torch.bfloat16)
    g = -torch.rand(1, 9, value_heads, generator=generator)
    beta = torch.rand(1, 9, value_heads, generator=generator).to(torch.bfloat16)
    initial = torch.randn(2, value_heads, 128, key_dim, generator=generator) * 0.1
    expected_state = initial.clone()
    expected = torch.empty_like(v, dtype=torch.float32)
    scale = key_dim**-0.5
    for sequence, (start, end) in enumerate([(0, 5), (5, 9)]):
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
            cu_seqlens=torch.tensor([0, 5, 9], dtype=torch.int32, device="npu"),
            chunk_indices=torch.tensor(
                [[0, 0], [1, 0]], dtype=torch.int32, device="npu"
            ),
            chunk_offsets=torch.tensor([0, 1, 2], dtype=torch.int32, device="npu"),
            use_qk_l2norm_in_kernel=False,
            core_attn_out=buffer,
        )
    assert output.data_ptr() == buffer.data_ptr()
    torch.testing.assert_close(output.cpu().float(), expected, rtol=0.03, atol=0.01)
    torch.testing.assert_close(state.cpu(), expected_state, rtol=0.03, atol=0.03)
    assert torch.all(buffer[-8:].cpu() == 7)


@pytest.mark.gpu
def test_packed_decode_updates_only_selected_padded_states():
    generator = torch.Generator().manual_seed(31)
    q, k, v = [
        torch.randn(2, 2, 128, generator=generator).to(torch.bfloat16) for _ in range(3)
    ]
    a, b = [torch.randn(2, 2, generator=generator).to(torch.bfloat16) for _ in range(2)]
    a_log = torch.tensor([-1.0, -0.5])
    bias = torch.tensor([0.1, -0.2])
    storage = torch.randn(4, 2 * 128 * 128 + 16, generator=generator) * 0.1
    expected_storage = storage.clone()
    expected_state = expected_storage[:, :-16].view(4, 2, 128, 128)
    expected_output = torch.empty(2, 1, 2, 128)
    for row, state_index in enumerate([1, 3]):
        state = expected_state[state_index]
        key = F.normalize(k[row].float(), dim=-1)
        query = F.normalize(q[row].float(), dim=-1)
        decay = (-a_log.exp() * F.softplus(a[row].float() + bias)).exp()
        state *= decay[:, None, None]
        residual = v[row].float() - torch.einsum("hvk,hk->hv", state, key)
        residual *= b[row].float().sigmoid()[:, None]
        state += torch.einsum("hv,hk->hvk", residual, key)
        expected_output[row, 0] = torch.einsum("hvk,hk->hv", state, query) * 128**-0.5

    npu_storage = storage.npu()
    state = npu_storage[:, :-16].view(4, 2, 128, 128)
    out = torch.empty(2, 1, 2, 128, device="npu", dtype=torch.bfloat16)
    gdn.fused_recurrent_gated_delta_rule_packed_decode(
        mixed_qkv=torch.cat([x.flatten(1) for x in (q, k, v)], dim=1).npu(),
        a=a.npu(),
        b=b.npu(),
        A_log=a_log.npu(),
        dt_bias=bias.npu(),
        scale=128**-0.5,
        initial_state=state,
        out=out,
        ssm_state_indices=torch.tensor([1, 3], device="npu", dtype=torch.int32),
        use_qk_l2norm_in_kernel=True,
    )
    torch.testing.assert_close(out.cpu().float(), expected_output, rtol=0.03, atol=0.01)
    torch.testing.assert_close(
        npu_storage.cpu(), expected_storage, rtol=0.03, atol=0.01
    )
