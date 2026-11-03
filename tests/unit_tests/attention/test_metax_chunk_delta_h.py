# Copyright (c) 2026 BAAI. All rights reserved.
"""Numerical regressions for MACA's single-stage vLLM 0.32 chunk-state kernel."""

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

BOUNDARIES = (0, 5, 73)
CHUNK_SIZE = 64


def _block_recurrence(k, w, u, initial_state, g, gk, use_exp2):
    """Torch reference for the WY block recurrence, with V-first state.

    h is the state at each chunk's start. v_new is the residual u - w @ h,
    saved before decay; the final state includes the gated k.T @ residual.
    Low-precision casts match the two kernel matmul operands, while state
    accumulation stays in float32.
    """
    exp = torch.exp2 if use_exp2 else torch.exp
    heads = u.shape[2]
    k = k.repeat_interleave(heads // k.shape[2], dim=2)
    h = []
    v_new = torch.empty_like(u)
    final_states = []
    for sequence, (bos, eos) in enumerate(zip(BOUNDARIES, BOUNDARIES[1:])):
        state = initial_state[sequence].float().clone()
        for start in range(bos, eos, CHUNK_SIZE):
            end = min(start + CHUNK_SIZE, eos)
            h.append(state.to(k.dtype))
            residual = u[0, start:end].float() - torch.einsum(
                "thk,hvk->thv",
                w[0, start:end].float(),
                state.to(w.dtype).float(),
            )
            v_new[0, start:end] = residual.to(u.dtype)
            if g is not None:
                gates = g[0, start:end].float()
                residual *= exp(gates[-1:] - gates).unsqueeze(-1)
                state *= exp(gates[-1]).view(heads, 1, 1)
            if gk is not None:
                state *= exp(gk[0, end - 1].float()).unsqueeze(1)
            state += torch.einsum(
                "thk,thv->hvk",
                k[0, start:end].float(),
                residual.to(k.dtype).float(),
            )
        final_states.append(state)
    return torch.stack(h).unsqueeze(0), v_new, torch.stack(final_states)


@pytest.mark.parametrize("use_exp2", [False, True])
@pytest.mark.parametrize("gate_kind", ["g", "gk"])
def test_chunk_state_matches_block_recurrence_with_varlen_tails(gate_kind, use_exp2):
    from vllm.third_party.flash_linear_attention.ops import chunk_delta_h

    from vllm_fl.dispatch.backends.vendor.metax.patches import (
        chunk_delta_h as patch,
    )

    kernel = patch.chunk_gated_delta_rule_fwd_kernel_h_blockdim64
    assert chunk_delta_h.chunk_gated_delta_rule_fwd_kernel_h_blockdim64 is kernel
    autotuner = kernel
    while not hasattr(autotuner, "configs"):
        autotuner = autotuner.fn
    assert {config.num_stages for config in autotuner.configs} == {1}

    generator = torch.Generator().manual_seed(32)
    # Unequal K/V, GQA, two sequence tails and one full chunk exercise every
    # pointer mask. cu_seqlens also ensures sequence 2 cannot read sequence 1.
    k = (torch.randn(1, 73, 1, 32, generator=generator) * 0.04).to(torch.bfloat16)
    w = (torch.randn(1, 73, 2, 32, generator=generator) * 0.04).to(torch.bfloat16)
    u = (torch.randn(1, 73, 2, 20, generator=generator) * 0.1).to(torch.bfloat16)
    initial = torch.randn(2, 2, 20, 32, generator=generator) * 0.2
    g = torch.empty(1, 73, 2)
    gk = torch.empty(1, 73, 2, 32)
    for bos, eos in zip(BOUNDARIES, BOUNDARIES[1:]):
        for start in range(bos, eos, CHUNK_SIZE):
            end = min(start + CHUNK_SIZE, eos)
            progress = torch.linspace(0.05, 1.0, end - start)
            g[0, start:end] = -progress[:, None] * torch.tensor([0.7, 1.2])
            gk[0, start:end] = (
                -progress[:, None, None] * torch.linspace(0.1, 1.1, 32)[None, None, :]
            )
    g = g if gate_kind == "g" else None
    gk = gk if gate_kind == "gk" else None
    expected = _block_recurrence(k, w, u, initial, g, gk, use_exp2)
    # Keep this fixture sensitive to the old bug that accepted USE_EXP2 but
    # always used exp(): the selected data must distinguish the two results.
    other_base = _block_recurrence(k, w, u, initial, g, gk, not use_exp2)
    assert (expected[2] - other_base[2]).abs().max() > 0.02

    with torch.inference_mode():
        actual = chunk_delta_h.chunk_gated_delta_rule_fwd_h(
            # MACA may transfer 4D tensors in channels-last format. The raw
            # pointer launcher requires contiguous [B, T, H, D] / [N, H, V, K].
            k=k.cuda().contiguous(),
            w=w.cuda().contiguous(),
            u=u.cuda().contiguous(),
            g=g.cuda().contiguous() if g is not None else None,
            gk=gk.cuda().contiguous() if gk is not None else None,
            initial_state=initial.cuda().contiguous(),
            output_final_state=True,
            chunk_size=CHUNK_SIZE,
            save_new_value=True,
            cu_seqlens=torch.tensor(BOUNDARIES, device="cuda", dtype=torch.int32),
            chunk_indices=torch.tensor(
                [[0, 0], [1, 0], [1, 1]], device="cuda", dtype=torch.int32
            ),
            chunk_offsets=torch.tensor([0, 1, 3], device="cuda", dtype=torch.int32),
            use_exp2=use_exp2,
        )
        # Also surfaces asynchronous masked-load/store failures before comparing.
        torch.cuda.synchronize()
    for result, reference in zip(actual, expected):
        assert result.shape == reference.shape
        torch.testing.assert_close(
            result.cpu().float(), reference.float(), rtol=0.02, atol=0.002
        )
