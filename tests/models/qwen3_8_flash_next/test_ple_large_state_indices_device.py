# SPDX-License-Identifier: Apache-2.0
"""Actual hybrid-page strides above 2^31 elements, without a full checkpoint."""

import os
from types import SimpleNamespace as NS

import pytest
import torch


@pytest.mark.skipif(
    os.environ.get("QWEN38_RUN_DEVICE_TESTS") != "1",
    reason="opt-in assigned accelerator",
)
@pytest.mark.parametrize("speculative", [False, True])
@torch.inference_mode()
def test_large_ple_state_rows_eager_and_changed_graph(speculative):
    import vllm_fl

    vllm_fl.register_model()
    import flag_gems

    from vllm_fl.models.qwen3_8_flash_next.gpu.ple_layer import Qwen3_8FlashNextPLELayer

    rows = [10485, 10486, 11393]
    channels, stride = 10240, 204800
    state = torch.empty_strided(
        (11394, channels, 10), (stride, 1, 10240), device="cuda", dtype=torch.bfloat16
    )
    indices = torch.tensor(rows, dtype=torch.int32, device="cuda")
    length = 2 if speculative else 16
    x = (
        torch.arange(3 * length * channels, device="cuda").reshape(3 * length, channels)
        % 17
        - 8
    ).bfloat16() / 8
    weight = torch.ones(channels, 4, device="cuda", dtype=torch.bfloat16)
    starts = torch.arange(4, device="cuda", dtype=torch.int32) * length
    accepted = torch.tensor([1, 2, 1], device="cuda", dtype=torch.int32)
    old = [
        ((torch.arange(10).float() / 8 + i).repeat(channels, 1)).bfloat16()
        for i in range(3)
    ]

    def reset():
        for row, value in zip(rows, old):
            state[row].copy_(value)

    meta = NS(
        num_actual_tokens=x.shape[0],
        spec_sequence_masks=torch.ones(3, device="cuda", dtype=torch.bool)
        if speculative
        else None,
        num_decodes=0,
        num_prefills=0 if speculative else 3,
        spec_state_indices_tensor=indices,
        spec_query_start_loc=starts,
        num_accepted_tokens=accepted,
        spec_query_len=2,
        state_indices_tensor=indices,
        non_spec_query_start_loc=starts,
        has_initial_states_p=torch.zeros(3, device="cuda", dtype=torch.bool),
        has_initial_states_d=None,
    )

    from vllm_fl.dispatch import resolve_op

    ple_provider = resolve_op("ple_short_conv_")

    def run(inp):
        return Qwen3_8FlashNextPLELayer._short_conv_dispatch_metax(
            NS(short_conv_dilation=3, _ple_short_conv=ple_provider),
            inp,
            meta,
            state,
            weight,
        )

    def oracle(inp):
        cpu = inp.cpu()
        result = torch.zeros_like(cpu)
        states = []
        for i in range(3):
            segment = cpu[i * length : (i + 1) * length]
            rollback = [0, 1, 0][i]
            history = (
                old[i][:, rollback : rollback + 9].T
                if speculative
                else torch.zeros(9, channels, dtype=torch.bfloat16)
            )
            full = torch.cat([history, segment]).float()
            for t in range(length):
                value = torch.zeros(channels)
                for tap in range(4):
                    value += full[9 + t - (3 - tap) * 3]
                rounded = value.bfloat16().float()
                result[i * length + t] = (rounded * rounded.sigmoid()).bfloat16()
            if speculative:
                states.append(
                    torch.cat(
                        [old[i][:, rollback + 1 : rollback + 9], segment.T], dim=1
                    )
                )
            else:
                expected = old[i].clone()
                expected[:, :9] = segment[-9:].T
                states.append(expected)
        return result, states

    def check(result, expected, expected_states):
        torch.testing.assert_close(result.cpu(), expected, atol=0.0078125, rtol=0)
        for row, value in zip(rows, expected_states):
            torch.testing.assert_close(state[row].cpu(), value, atol=0, rtol=0)

    expected, states = oracle(x)
    reset()
    with flag_gems.use_gems():
        result = run(x)
        torch.cuda.synchronize()
    check(result, expected, states)
    changed = x + 0.25
    expected, states = oracle(changed)
    reset()
    with flag_gems.use_gems():
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = run(x)
        reset()
        x.copy_(changed)
        graph.replay()
        torch.cuda.synchronize()
    check(captured, expected, states)
    print(
        {
            "speculative": speculative,
            "channels": channels,
            "rows": rows,
            "stride": state.stride(),
            "max_element_offset": rows[-1] * stride,
            "passed": True,
        }
    )
