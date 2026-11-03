# Copyright (c) 2026 BAAI. All rights reserved.
"""Check the complete GDN prefill/decode chain against a CPU recurrence."""

import pytest
import torch

from tests.utils.device_utils import get_backend

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        get_backend() != "metax" or not torch.cuda.is_available(),
        reason="MetaX GPU and MACA runtime required",
    ),
]


def _inputs(length, key_dim, value_dim, sequences, key_heads=2, value_heads=4):
    rng = torch.Generator().manual_seed(320)
    q = torch.randn(1, length, key_heads, key_dim, generator=rng).bfloat16()
    k = torch.randn(1, length, key_heads, key_dim, generator=rng).bfloat16()
    v = torch.randn(1, length, value_heads, value_dim, generator=rng).bfloat16()
    g = -torch.rand(1, length, value_heads, generator=rng) * 0.3
    beta = torch.rand(1, length, value_heads, generator=rng).bfloat16()
    state = torch.randn(sequences, value_heads, value_dim, key_dim, generator=rng) * 0.1
    return q, k, v, g, beta, state


def _reference(q, k, v, g, beta, initial, boundaries, round_normalized):
    # The state stores value rows and key columns. This token-by-token delta
    # rule is independent of the chunk implementation's WY decomposition.
    dtype = q.dtype
    q, k = q.float(), k.float()
    q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6)
    k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
    if round_normalized:
        q, k = q.to(dtype).float(), k.to(dtype).float()
    q = q.repeat_interleave(v.shape[2] // q.shape[2], dim=2)
    k = k.repeat_interleave(v.shape[2] // k.shape[2], dim=2)
    output = torch.empty_like(v, dtype=torch.float32)
    states = []
    for seq, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
        state = initial[seq].float().clone()
        for token in range(start, end):
            state *= g[0, token].float().exp()[:, None, None]
            key = k[0, token]
            delta = v[0, token].float() - torch.einsum("hvk,hk->hv", state, key)
            delta *= beta[0, token].float()[:, None]
            state += delta[:, :, None] * key[:, None, :]
            output[0, token] = (
                torch.einsum("hvk,hk->hv", state, q[0, token]) * key.shape[-1] ** -0.5
            )
        states.append(state)
    return output, torch.stack(states)


@pytest.mark.parametrize(
    "key_dim,value_dim,key_heads,value_heads", [(32, 20, 2, 4), (128, 128, 16, 32)]
)
@torch.inference_mode()
def test_gdn_chunk_prefill_matches_token_recurrence(
    key_dim, value_dim, key_heads, value_heads
):
    from vllm.third_party.flash_linear_attention.ops.chunk import chunk_gated_delta_rule

    # Install the same single-stage chunk-state patch used by a real worker.
    from vllm_fl.dispatch.backends.vendor.metax.patches import (
        chunk_delta_h,  # noqa: F401
    )

    boundaries = (0, 5, 73)
    values = _inputs(73, key_dim, value_dim, 2, key_heads, value_heads)
    expected = _reference(*values, boundaries, round_normalized=True)
    q, k, v, g, beta, initial = (x.cuda().contiguous() for x in values)
    output_buffer = torch.full_like(v, float("nan"))
    actual = chunk_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        initial_state=initial,
        output_final_state=True,
        cu_seqlens=torch.tensor(boundaries, device="cuda", dtype=torch.int32),
        use_qk_l2norm_in_kernel=True,
        core_attn_out=output_buffer,
    )
    torch.cuda.synchronize()
    for result, reference in zip(actual, expected):
        torch.testing.assert_close(
            result.cpu().float(), reference, rtol=0.04, atol=0.003
        )
    torch.testing.assert_close(output_buffer, actual[0])


@torch.inference_mode()
def test_gdn_decode_uses_indexed_v_first_state():
    from vllm.third_party.flash_linear_attention.ops.fused_recurrent import (
        fused_recurrent_gated_delta_rule,
    )

    values = _inputs(2, 128, 128, 4, 16, 32)
    q, k, v, g, beta, initial = values
    # Block zero is the reserved NULL_BLOCK_ID in vLLM 0.32. Use two live
    # non-identity indices and also verify that the null block is untouched.
    indices = torch.tensor([3, 1], dtype=torch.int32)
    expected_out, expected_state = _reference(
        q,
        k,
        v,
        g,
        beta,
        initial[indices.long()],
        (0, 1, 2),
        round_normalized=False,
    )
    q, k, v, g, beta, state = (x.cuda().contiguous() for x in values)
    actual, final = fused_recurrent_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        initial_state=state,
        inplace_final_state=True,
        cu_seqlens=torch.tensor([0, 1, 2], device="cuda", dtype=torch.int32),
        ssm_state_indices=indices.cuda(),
        use_qk_l2norm_in_kernel=True,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(
        actual.cpu().float(), expected_out, rtol=0.02, atol=0.002
    )
    final_cpu = final.cpu()
    torch.testing.assert_close(
        final_cpu[indices.long()], expected_state, rtol=0.02, atol=0.002
    )
    torch.testing.assert_close(final_cpu[[0, 2]], initial[[0, 2]], rtol=0, atol=0)
