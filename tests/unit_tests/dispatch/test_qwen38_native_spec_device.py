# SPDX-License-Identifier: Apache-2.0
"""K1 correctness checks for the native paths retained by the Qwen adapter."""

import torch

from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as native


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


def test_native_gdn_k1_acceptance_null_eager_graph():
    torch.manual_seed(146)
    t, hk, hv, d = 6, 2, 6, 128
    q = torch.randn(1, t, hk, d, dtype=torch.bfloat16, device="cuda")
    k = torch.randn_like(q)
    v = torch.randn(1, t, hv, d, dtype=torch.bfloat16, device="cuda")
    a = torch.randn(t, hv, dtype=torch.bfloat16, device="cuda")
    b = torch.randn_like(a)
    alog = torch.randn(hv, device="cuda") * 0.1 - 1
    bias = torch.randn(hv, device="cuda") * 0.1
    initial = torch.randn(7, hv, d, d, device="cuda") * 0.03
    state = initial.clone()
    indices = torch.tensor([[1, 2], [3, 4], [0, 0]], device="cuda", dtype=torch.int32)
    accepted = torch.tensor([1, 2, 1], device="cuda", dtype=torch.int32)
    starts = torch.tensor([0, 2, 4, 6], device="cuda", dtype=torch.int32)

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

    def check(output):
        expected, expected_state, valid = _recurrent_reference(
            q, k, v, a, b, alog, bias, initial, indices, accepted, starts
        )
        # Native null rows intentionally do not write their output allocation.
        torch.testing.assert_close(
            output.cpu()[:, valid], expected[:, valid], atol=2e-3, rtol=1e-2
        )
        torch.testing.assert_close(state.cpu(), expected_state, atol=5e-5, rtol=5e-4)
        torch.testing.assert_close(state[0].cpu(), initial[0].cpu(), atol=0, rtol=0)
        print(
            {
                "native_gdn_state_max_abs": float(
                    (state.cpu() - expected_state).abs().max()
                )
            }
        )

    output = run()
    check(output)
    state.copy_(initial)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_output = run()
    q.normal_()
    v.normal_()
    b.normal_()
    accepted.copy_(torch.tensor([2, 1, 1], device="cuda", dtype=torch.int32))
    indices.copy_(
        torch.tensor([[3, 4], [1, 2], [0, 0]], device="cuda", dtype=torch.int32)
    )
    state.copy_(initial)
    graph.replay()
    check(captured_output)
    print({"provider": native.fused_sigmoid_gating_delta_rule_update.__module__})


def _conv_reference(x, weight, initial, indices, accepted, starts):
    x, weight = x.cpu().float(), weight.cpu().float()
    indices, accepted, starts = [x.cpu() for x in (indices, accepted, starts)]
    state = initial.cpu().clone()
    output = x.clone()
    history_width = weight.shape[1] - 1
    for req in range(indices.numel()):
        slot = int(indices[req])
        if slot == 0:
            continue
        start, end = int(starts[req]), int(starts[req + 1])
        rollback = int(accepted[req]) - 1
        history = state[slot, :, rollback : rollback + history_width].float()
        sequence = torch.cat((history, x[start:end].T), dim=-1)
        for local, token in enumerate(range(start, end)):
            acc = (sequence[:, local : local + weight.shape[1]] * weight).sum(-1)
            output[token] = acc * acc.sigmoid()
        state[slot, :, : history_width + end - start - 1] = sequence[:, 1:]
    return output.bfloat16(), state


def test_native_causal_conv_k1_acceptance_null_eager_graph():
    torch.manual_seed(147)
    x = torch.randn(6, 1280, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(1280, 4, device="cuda", dtype=torch.bfloat16) * 0.1
    initial = torch.randn(4, 1280, 4, device="cuda", dtype=torch.bfloat16)
    state = initial.clone()
    indices = torch.tensor([1, 2, 0], device="cuda", dtype=torch.int32)
    accepted = torch.tensor([1, 2, 1], device="cuda", dtype=torch.int32)
    starts = torch.tensor([0, 2, 4, 6], device="cuda", dtype=torch.int32)

    def reference():
        return _conv_reference(x, weight, initial, indices, accepted, starts)

    def run():
        return native.causal_conv1d_update(
            x,
            state,
            weight,
            activation="silu",
            conv_state_indices=indices,
            num_accepted_tokens=accepted,
            query_start_loc=starts,
            max_query_len=2,
            null_block_id=0,
            validate_data=False,
        )

    def check(output, expected):
        exp_out, exp_state = expected
        torch.testing.assert_close(output.cpu(), exp_out, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(state.cpu(), exp_state, atol=0, rtol=0)
        print(
            {
                "native_causal_conv_max_abs": float(
                    (output.cpu().float() - exp_out.float()).abs().max()
                )
            }
        )

    expected = reference()
    check(run(), expected)
    x.normal_()
    state.copy_(initial)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_output = run()
    x.normal_()
    indices.copy_(torch.tensor([2, 1, 0], device="cuda", dtype=torch.int32))
    accepted.copy_(torch.tensor([2, 1, 1], device="cuda", dtype=torch.int32))
    state.copy_(initial)
    expected = reference()
    graph.replay()
    check(captured_output, expected)
    print({"provider": native.causal_conv1d_update.__module__})
