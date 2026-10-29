# SPDX-License-Identifier: Apache-2.0
"""K1/K2/K3 recurrent and convolution state contracts, eager and changed graph."""

import pytest
import torch
import torch.nn.functional as F

_SPEC_CASES = [(k, length) for k in (1, 2, 3) for length in range(1, k + 2)]
_SPEC_IDS = [f"k{k}-q{length}" for k, length in _SPEC_CASES]


def _recurrent_reference(q, k, v, a, b, alog, bias, initial, indices, accepted, starts):
    q, k, v, a, b, alog, bias = [x.cpu().float() for x in (q, k, v, a, b, alog, bias)]
    indices, accepted, starts = [x.cpu() for x in (indices, accepted, starts)]
    state = initial.cpu().clone()
    result = torch.zeros_like(v)
    valid = torch.zeros(q.shape[1], dtype=torch.bool)
    group = v.shape[2] // q.shape[2]
    for req in range(accepted.numel()):
        begin, end = int(starts[req]), int(starts[req + 1])
        slot = int(indices[req, int(accepted[req]) - 1])
        if slot <= 0:
            continue
        h = state[slot].clone()
        for offset, token in enumerate(range(begin, end)):
            qr, kr = q[0, token], k[0, token]
            qr = qr * torch.rsqrt(qr.square().sum(-1, keepdim=True) + 1e-6)
            kr = kr * torch.rsqrt(kr.square().sum(-1, keepdim=True) + 1e-6)
            qr = qr.repeat_interleave(group, 0) * q.shape[-1] ** -0.5
            kr = kr.repeat_interleave(group, 0)
            x = a[token] + bias
            g = -alog.exp() * torch.where(x <= 20, torch.log(1 + x.exp()), x)
            beta = b[token].sigmoid()
            h = h * g.exp()[:, None, None]
            delta = (v[0, token] - (h * kr[:, None, :]).sum(-1)) * beta[:, None]
            h += delta[:, :, None] * kr[:, None, :]
            result[0, token] = (h * qr[:, None, :]).sum(-1)
            destination = int(indices[req, offset])
            if destination > 0:
                state[destination] = h
            valid[token] = True
    return result.bfloat16(), state, valid


@pytest.mark.parametrize("num_spec,query_len", _SPEC_CASES, ids=_SPEC_IDS)
def test_native_gdn_acceptance_null_eager_graph(num_spec, query_len):
    from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as native

    torch.manual_seed(146)
    requests, hk, hv, d = num_spec + 2, 2, 6, 128
    t = requests * query_len
    q = torch.randn(1, t, hk, d, dtype=torch.bfloat16, device="cuda")
    k = torch.randn_like(q)
    v = torch.randn(1, t, hv, d, dtype=torch.bfloat16, device="cuda")
    a = torch.randn(t, hv, dtype=torch.bfloat16, device="cuda")
    b = torch.randn_like(a)
    alog = torch.randn(hv, device="cuda") * 0.1 - 1
    bias = torch.randn(hv, device="cuda") * 0.1
    state_rows = 1 + (num_spec + 1) ** 2
    initial = torch.randn(state_rows, hv, d, d, device="cuda") * 0.03
    state = initial.clone()
    indices = torch.zeros((requests, num_spec + 1), device="cuda", dtype=torch.int32)
    indices[:-1] = torch.arange(1, state_rows, device="cuda").reshape(-1, num_spec + 1)
    accepted = torch.tensor(
        [*range(1, num_spec + 2), 1], device="cuda", dtype=torch.int32
    )
    starts = torch.arange(requests + 1, device="cuda", dtype=torch.int32) * query_len

    def reference(previous):
        return _recurrent_reference(
            q, k, v, a, b, alog, bias, previous, indices, accepted, starts
        )

    def run():
        return native.fused_sigmoid_gating_delta_rule_update(
            A_log=alog,
            a=a,
            b=b,
            dt_bias=bias,
            q=q,
            k=k,
            v=v,
            initial_state=state,
            inplace_final_state=True,
            cu_seqlens=starts,
            ssm_state_indices=indices,
            num_accepted_tokens=accepted,
            use_qk_l2norm_in_kernel=True,
        )[0]

    def check(output, expected):
        exp_out, exp_state, valid = expected
        # Native null rows intentionally do not write their output allocation.
        torch.testing.assert_close(
            output.cpu()[:, valid], exp_out[:, valid], atol=2e-3, rtol=1e-2
        )
        torch.testing.assert_close(state.cpu(), exp_state, atol=5e-5, rtol=5e-4)
        torch.testing.assert_close(state[0].cpu(), initial[0].cpu(), atol=0, rtol=0)

    expected = reference(initial)
    check(run(), expected)
    state.copy_(initial)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_output = run()
    q.normal_()
    v.normal_()
    b.normal_()
    accepted[:-1].copy_(accepted[:-1].roll(1))
    indices[:-1].copy_(indices[:-1].roll(1, 0))
    state.copy_(initial)
    expected = reference(initial)
    graph.replay()
    check(captured_output, expected)
    # Continue from the actual preceding candidate windows. The selected prefix
    # cannot exceed that preceding query length; do not reseed between rounds.
    for step in range(3):
        q.normal_()
        v.normal_()
        b.normal_()
        accepted[:-1].copy_(
            1 + (torch.arange(requests - 1, device="cuda") + step) % query_len
        )
        expected = reference(expected[1])
        graph.replay()
        check(captured_output, expected)
    print(
        {
            "provider": native.fused_sigmoid_gating_delta_rule_update.__module__,
            "k": num_spec,
            "query_len": query_len,
        }
    )


def _conv_reference(
    x,
    weight,
    initial,
    indices,
    accepted,
    starts,
    *,
    dilation=1,
    round_before_activation=False,
    zero_null=False,
):
    """Independent history-sequence oracle, using CPU grouped convolution."""
    x, weight = x.cpu().float(), weight.cpu().float()
    indices, accepted, starts = [x.cpu() for x in (indices, accepted, starts)]
    state = initial.cpu().clone()
    output = torch.zeros_like(x) if zero_null else x.clone()
    history_width = (weight.shape[1] - 1) * dilation
    for req, slot in enumerate(indices.tolist()):
        start, end = int(starts[req]), int(starts[req + 1])
        if slot == 0 or start == end:
            continue
        rollback = int(accepted[req]) - 1
        history = state[slot, :, rollback : rollback + history_width].float()
        sequence = torch.cat((history, x[start:end].T), dim=-1)
        conv = F.conv1d(
            sequence[None], weight[:, None], groups=x.shape[1], dilation=dilation
        )[0].T
        if round_before_activation:
            conv = conv.bfloat16().float()
        output[start:end] = F.silu(conv)
        # Preserve every candidate prefix, plus untouched capacity after it.
        state[slot, :, : history_width + end - start - 1] = sequence[:, 1:]
    return output.bfloat16(), state


def _check_conv(output, state, expected):
    exp_out, exp_state = expected
    torch.testing.assert_close(output.cpu(), exp_out, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(state.cpu(), exp_state, atol=0, rtol=0)


def _conv_inputs(num_spec, query_len, dilation):
    requests, channels = num_spec + 2, 1280
    x = torch.randn(requests * query_len, channels, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(channels, 4, device="cuda", dtype=torch.bfloat16) * 0.1
    initial = torch.randn(
        requests,
        channels,
        3 * dilation + num_spec,
        device="cuda",
        dtype=torch.bfloat16,
    )
    indices = torch.tensor([*range(1, requests), 0], device="cuda", dtype=torch.int32)
    accepted = torch.tensor(
        [*range(1, num_spec + 2), 1], device="cuda", dtype=torch.int32
    )
    starts = torch.arange(requests + 1, device="cuda", dtype=torch.int32) * query_len
    return x, weight, initial, indices, accepted, starts


def _exercise_conv_graph(
    x, state, initial, indices, accepted, query_len, run, reference
):
    expected = reference(initial)
    _check_conv(run(), state, expected)
    x.normal_()
    state.copy_(initial)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_output = run()
    x.normal_()
    indices[:-1].copy_(indices[:-1].roll(1))
    accepted[:-1].copy_(accepted[:-1].roll(1))
    state.copy_(initial)
    expected = reference(initial)
    graph.replay()
    _check_conv(captured_output, state, expected)
    for step in range(3):
        # Native convolution overwrites x. Supply new inputs every round, and
        # retain the candidate history produced by the preceding graph replay.
        x.normal_()
        accepted[:-1].copy_(
            1 + (torch.arange(indices.numel() - 1, device="cuda") + step) % query_len
        )
        expected = reference(expected[1])
        graph.replay()
        _check_conv(captured_output, state, expected)


@pytest.mark.parametrize("num_spec,query_len", _SPEC_CASES, ids=_SPEC_IDS)
def test_native_causal_conv_acceptance_null_eager_graph(num_spec, query_len):
    from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as native

    torch.manual_seed(147)
    x, weight, initial, indices, accepted, starts = _conv_inputs(num_spec, query_len, 1)
    state = initial.clone()

    def reference(previous):
        return _conv_reference(x, weight, previous, indices, accepted, starts)

    def run():
        return native.causal_conv1d_update(
            x,
            state,
            weight,
            activation="silu",
            conv_state_indices=indices,
            num_accepted_tokens=accepted,
            query_start_loc=starts,
            max_query_len=num_spec + 1,
            null_block_id=0,
            validate_data=False,
        )

    _exercise_conv_graph(
        x, state, initial, indices, accepted, query_len, run, reference
    )
    print(
        {
            "provider": native.causal_conv1d_update.__module__,
            "k": num_spec,
            "query_len": query_len,
        }
    )


@pytest.mark.parametrize("num_spec,query_len", _SPEC_CASES, ids=_SPEC_IDS)
def test_fgv_ple_acceptance_null_eager_graph(num_spec, query_len):
    from flaggems_vllm.ops.qwen4.ple_conv import ple_short_conv_

    torch.manual_seed(148)
    x, weight, initial, indices, accepted, starts = _conv_inputs(num_spec, query_len, 3)
    state = initial.clone()
    output = torch.empty_like(x)

    def reference(previous):
        return _conv_reference(
            x,
            weight,
            previous,
            indices,
            accepted,
            starts,
            dilation=3,
            round_before_activation=True,
            zero_null=True,
        )

    def run():
        ple_short_conv_(
            x,
            output,
            state,
            weight,
            indices.to(torch.int64),
            dilation=3,
            query_start_loc=starts,
            accepted_tokens=accepted,
            max_query_len=num_spec + 1,
            null_block_id=0,
        )
        return output

    _exercise_conv_graph(
        x, state, initial, indices, accepted, query_len, run, reference
    )
    print(
        {"provider": ple_short_conv_.__module__, "k": num_spec, "query_len": query_len}
    )
