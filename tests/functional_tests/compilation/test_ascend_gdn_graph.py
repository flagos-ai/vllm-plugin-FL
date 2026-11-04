# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(current_platform.device_type != "npu", reason="Requires Ascend"),
]


def _inputs(tokens):
    torch.npu.set_device(0)
    torch.manual_seed(7)
    kwargs = dict(
        conv_weights=torch.randn(16, 4, device="npu", dtype=torch.bfloat16) * 0.1,
        conv_bias=None,
        activation="silu",
        A_log=torch.zeros(2, device="npu"),
        dt_bias=torch.zeros(2, device="npu"),
        num_key_heads=1,
        num_value_heads=2,
        head_k_dim=4,
        head_v_dim=4,
    )
    x = torch.randn(tokens, 16, device="npu", dtype=torch.bfloat16)
    a, b = [
        torch.randn(tokens, 2, device="npu", dtype=torch.bfloat16) for _ in range(2)
    ]
    conv = torch.zeros(4, 16, 3, device="npu", dtype=torch.bfloat16)
    state = torch.zeros(4, 2, 4, 4, device="npu", dtype=torch.float32)
    return x, a, b, conv, state, kwargs


@torch.inference_mode()
def test_gdn_prefill_matches_token_decode():
    from vllm_fl.dispatch.backends.vendor.ascend.impl.gdn_reference import (
        gdn_decode,
        gdn_prefill,
    )

    x, a, b, conv, state, kwargs = _inputs(5)
    conv_decode, state_decode = conv.clone(), state.clone()
    index = torch.tensor([1], device="npu", dtype=torch.int32)
    output = torch.empty(5, 2, 4, device="npu", dtype=torch.bfloat16)
    expected = torch.empty_like(output)
    gdn_prefill(
        x,
        a,
        b,
        index,
        torch.tensor([False], device="npu"),
        conv,
        state[1],
        output,
        **kwargs,
    )
    for token in range(5):
        gdn_decode(
            x[token : token + 1],
            a[token : token + 1],
            b[token : token + 1],
            index,
            conv_decode,
            state_decode,
            expected[token : token + 1],
            packed_beta=False,
            **kwargs,
        )
    torch.testing.assert_close(output, expected, atol=0.01, rtol=0.05)
    torch.testing.assert_close(conv, conv_decode)
    torch.testing.assert_close(state, state_decode, atol=0.01, rtol=0.05)


def _decode_reference(x, a, b, indices, conv, state, kwargs):
    """Independent CPU recurrence; no production GDN or convolution helpers."""
    result = torch.zeros(len(indices), 2, 4, dtype=x.dtype)
    weight = kwargs["conv_weights"].cpu().float()
    for lane, slot in enumerate(indices.tolist()):
        if slot <= 0:
            continue
        window = torch.cat((conv[slot], x[lane, :, None]), dim=-1)
        conv[slot] = window[:, -3:]
        projected = (window.float() * weight).sum(-1).to(x.dtype)
        projected = torch.nn.functional.silu(projected).float()
        query, key, value = projected.split((4, 4, 8))
        query /= torch.sqrt(query.square().sum() + 1e-6)
        key /= torch.sqrt(key.square().sum() + 1e-6)
        value = value.reshape(2, 4)
        decay = torch.exp(
            -kwargs["A_log"].cpu().exp()
            * torch.log1p(torch.exp(a[lane].float() + kwargs["dt_bias"].cpu()))
        )
        beta = 1 / (1 + torch.exp(-b[lane].float()))
        for head in range(2):
            updated = state[slot, head] * decay[head]
            residual = (value[head] - updated @ key) * beta[head]
            state[slot, head] = updated + torch.outer(residual, key)
            result[lane, head] = state[slot, head] @ (query * 0.5)
    return result


@pytest.mark.parametrize("strided", [False, True])
@torch.inference_mode()
def test_gdn_decode_capture_updates_live_slots_and_preserves_padding(strided):
    from vllm_fl.dispatch import CachedOp

    decode_op = CachedOp("gdn_decode")
    x, a, b, conv, state, kwargs = _inputs(2)
    if strided:
        # Include a nonzero storage offset and transposed convolution channels.
        conv_backing = torch.full((4, 2, 3, 16), 7, device="npu", dtype=conv.dtype)
        state_backing = torch.full((4, 2, 2, 4, 4), 7, device="npu", dtype=state.dtype)
        conv = conv_backing[:, 1].transpose(-1, -2)
        state = state_backing[:, 1]
    conv.fill_(0.125)
    state.fill_(0.25)
    initial_conv, initial_state = conv.cpu(), state.cpu()
    indices = torch.tensor([1, -1], device="npu", dtype=torch.int32)
    output = torch.empty(2, 2, 4, device="npu", dtype=torch.bfloat16)

    def decode():
        decode_op(x, a, b, indices, conv, state, output, packed_beta=False, **kwargs)

    decode()
    assert decode_op._impl.impl_id == "vendor.ascend"
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        decode()
    for slot in (3, 1, 2):
        x.normal_()
        a.normal_()
        b.normal_()
        live_indices = torch.tensor([slot, 0 if slot != 1 else -1], dtype=torch.int32)
        indices.copy_(live_indices)
        expected_conv, expected_state = initial_conv.clone(), initial_state.clone()
        expected_output = _decode_reference(
            x.cpu(),
            a.cpu(),
            b.cpu(),
            live_indices,
            expected_conv,
            expected_state,
            kwargs,
        )
        # Also retain exact eager/replay consistency checks: CPU reductions
        # and NPU BF16 convolution can use different accumulation orders.
        conv.copy_(initial_conv)
        state.copy_(initial_state)
        decode()
        eager_output, eager_conv, eager_state = (
            output.clone(),
            conv.clone(),
            state.clone(),
        )
        conv.copy_(initial_conv)
        state.copy_(initial_state)
        output.fill_(float("nan"))
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(output.cpu(), expected_output, atol=0.01, rtol=0.05)
        torch.testing.assert_close(conv.cpu(), expected_conv)
        torch.testing.assert_close(state.cpu(), expected_state, atol=0.01, rtol=0.05)
        torch.testing.assert_close(output, eager_output)
        torch.testing.assert_close(conv, eager_conv)
        torch.testing.assert_close(state, eager_state)
        assert torch.count_nonzero(output[1]) == 0
        torch.testing.assert_close(conv[0].cpu(), initial_conv[0], atol=0, rtol=0)
        torch.testing.assert_close(state[0].cpu(), initial_state[0], atol=0, rtol=0)
        if strided:
            torch.testing.assert_close(
                conv_backing[:, 0], torch.full_like(conv_backing[:, 0], 7)
            )
            torch.testing.assert_close(
                state_backing[:, 0], torch.full_like(state_backing[:, 0], 7)
            )


@torch.inference_mode()
def test_gdn_strided_cache_copy_memory_is_bounded_by_active_rows():
    from vllm_fl.dispatch.backends.vendor.ascend.impl.gdn_reference import (
        _gather_rows as gather_state_rows,
        _scatter_rows as scatter_state_rows,
    )

    torch.npu.set_device(0)
    backing = torch.zeros(1024, 2, 2, 64, 64, device="npu")
    state = backing[:, 0]
    indices = torch.tensor([1, 127, 511, 1023], device="npu", dtype=torch.int64)
    rows = gather_state_rows(state, indices)
    scatter_state_rows(state, indices, rows)
    del rows
    torch.npu.synchronize()
    before = torch.npu.memory_allocated()
    torch.npu.reset_peak_memory_stats()
    rows = gather_state_rows(state, indices)
    rows.fill_(7)
    scatter_state_rows(state, indices, rows)
    torch.npu.synchronize()
    # Four active rows need 128 KiB; a full logical cache copy needs 32 MiB.
    assert torch.npu.max_memory_allocated() - before < 4 * 1024**2
    for slot in (1, 127, 511, 1023):
        torch.testing.assert_close(state[slot], torch.full_like(state[slot], 7))
    assert torch.count_nonzero(state[0]) == 0
    assert torch.count_nonzero(state[2]) == 0
    assert torch.count_nonzero(backing[:, 1]) == 0
