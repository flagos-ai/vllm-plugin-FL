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


@torch.inference_mode()
def test_gdn_decode_capture_updates_live_slots_and_preserves_padding():
    from vllm_fl.dispatch.backends.vendor.ascend.impl.gdn_reference import gdn_decode

    x, a, b, conv, state, kwargs = _inputs(2)
    indices = torch.tensor([1, -1], device="npu", dtype=torch.int32)
    output = torch.empty(2, 2, 4, device="npu", dtype=torch.bfloat16)

    def decode(conv, state, output):
        gdn_decode(
            x,
            a,
            b,
            indices,
            conv,
            state,
            output,
            packed_beta=False,
            **kwargs,
        )

    decode(conv, state, output)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        decode(conv, state, output)
    for slot in (3, 1, 2):
        x.normal_()
        indices.copy_(torch.tensor([slot, 0], device="npu", dtype=torch.int32))
        expected_conv, expected_state = conv.clone(), state.clone()
        expected_output = torch.empty_like(output)
        decode(expected_conv, expected_state, expected_output)
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(output, expected_output)
        torch.testing.assert_close(conv, expected_conv)
        torch.testing.assert_close(state, expected_state)
        assert torch.count_nonzero(output[1]) == 0
        assert torch.count_nonzero(conv[0]) == 0
        assert torch.count_nonzero(state[0]) == 0
