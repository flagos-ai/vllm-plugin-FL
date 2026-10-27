# SPDX-License-Identifier: Apache-2.0
"""Small correctness-only MetaX checks; run on an explicitly assigned GPU."""

from __future__ import annotations

import math

import pytest
import torch

from vllm_fl.dispatch.backends.flaggems.impl import qwen3_8_flash_next as ops
from vllm_fl.dispatch.backends.vendor.metax.impl import qwen3_8_flash_next as metax

pytestmark = pytest.mark.gpu


def _packed_oracle(mixed, a, b, A_log, dt_bias, state, indices):
    mixed, a, b = mixed.cpu().float(), a.cpu().float(), b.cpu().float()
    A_log, dt_bias = A_log.cpu().float(), dt_bias.cpu().float()
    state, indices = state.cpu().clone(), indices.cpu()
    rows = mixed.shape[0]
    hv, vd, kd = state.shape[-3:]
    h = (mixed.shape[1] - hv * vd) // (2 * kd)
    q, k, v = mixed.split((h * kd, h * kd, hv * vd), dim=-1)
    q, k, v = q.view(rows, h, kd), k.view(rows, h, kd), v.view(rows, hv, vd)
    out = torch.zeros(rows, 1, hv, vd)
    for row in range(rows):
        slot = int(indices[row])
        if slot <= 0:
            continue
        qr = q[row] / torch.sqrt(q[row].square().sum(-1, keepdim=True) + 1e-6)
        kr = k[row] / torch.sqrt(k[row].square().sum(-1, keepdim=True) + 1e-6)
        qr = qr.repeat_interleave(hv // h, 0) * kd**-0.5
        kr = kr.repeat_interleave(hv // h, 0)
        x = a[row] + dt_bias
        softplus = torch.where(x <= 20, torch.log(1 + torch.exp(x)), x)
        g = -torch.exp(A_log) * softplus
        beta = torch.sigmoid(b[row])
        updated = state[slot] * torch.exp(g)[:, None, None]
        delta = (v[row] - (updated * kr[:, None, :]).sum(-1)) * beta[:, None]
        updated += delta[:, :, None] * kr[:, None, :]
        out[row, 0] = (updated * qr[:, None, :]).sum(-1)
        state[slot] = updated
    return out.to(torch.bfloat16), state


def _assert_packed(mixed, a, b, alog, bias, state, indices, out, before):
    expected_out, expected_state = _packed_oracle(
        mixed, a, b, alog, bias, before, indices
    )
    actual_state = state.cpu()
    state_error = float((actual_state - expected_state).abs().max())
    torch.testing.assert_close(out.cpu(), expected_out, atol=2e-3, rtol=1e-2)
    torch.testing.assert_close(actual_state, expected_state, atol=3e-5, rtol=3e-4)
    torch.testing.assert_close(actual_state[0], before.cpu()[0], atol=0, rtol=0)
    return state_error


def test_packed_gdn_fp32_beta_eager_and_changed_input_graph():
    import flaggems_vllm

    torch.manual_seed(721)
    rows, h, hv, d = 3, 2, 6, 128
    mixed = torch.randn(rows, (2 * h + hv) * d, device="cuda", dtype=torch.bfloat16)
    a = torch.randn(rows, hv, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(rows, hv, device="cuda", dtype=torch.bfloat16)
    alog = torch.randn(hv, device="cuda") * 0.1 - 1.0
    bias = torch.randn(hv, device="cuda") * 0.1
    initial = torch.randn(5, hv, d, d, device="cuda") * 0.03
    state = initial.clone()
    indices = torch.tensor([1, 2, 0], device="cuda", dtype=torch.int32)
    out = torch.empty(rows, 1, hv, d, device="cuda", dtype=torch.bfloat16)

    def run():
        return ops.gdn_packed_decode(
            mixed, a, b, alog, bias, d**-0.5, state, out, indices, True
        )

    run()
    eager_error = _assert_packed(mixed, a, b, alog, bias, state, indices, out, initial)

    # The independent oracle must distinguish the uncorrected BF16-beta ABI.
    old_state, old_out = initial.clone(), torch.empty_like(out)
    flaggems_vllm.fused_recurrent_gated_delta_rule_packed_decode(
        mixed, a, b, alog, bias, d**-0.5, old_state, old_out, indices, True
    )
    _, reference_state = _packed_oracle(mixed, a, b, alog, bias, initial, indices)
    old_error = float((old_state.cpu() - reference_state).abs().max())
    assert old_error > max(1e-4, eager_error * 3), (old_error, eager_error)

    state.copy_(initial)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    mixed.normal_()
    a.normal_()
    b.normal_()
    indices.copy_(torch.tensor([2, 1, -1], device="cuda", dtype=torch.int32))
    state.copy_(initial)
    graph.replay()
    graph_error = _assert_packed(mixed, a, b, alog, bias, state, indices, out, initial)
    print(
        {
            "packed_eager_state_max_abs": eager_error,
            "packed_graph_state_max_abs": graph_error,
            "uncorrected_bf16_beta_state_max_abs": old_error,
        }
    )


def _score_oracle(q, cache, table, requests, positions, lengths, ratio):
    q, cache, table = q.cpu().float(), cache.cpu().float(), table.cpu()
    requests, positions, lengths = requests.cpu(), positions.cpu(), lengths.cpu()
    cols = table.shape[1] * cache.shape[1]
    scores = torch.full((q.shape[0], cols), -float("inf"))
    visible = torch.zeros(q.shape[0], dtype=torch.int32)
    for row in range(q.shape[0]):
        req = int(requests[row])
        if not 0 <= req < table.shape[0]:
            continue
        count = min((int(positions[row]) + 1) // ratio, int(lengths[req]) // ratio)
        visible[row] = max(0, count)
        physical = cache[table[req].long(), :, 0].reshape(cols, -1)
        values = (q[row] @ physical.T).clamp_min(0).sum(0) / math.sqrt(q.shape[-1])
        scores[row, :count] = values[:count]
    return scores, visible


def _selection_oracle(scores, positions, lengths, requests, budget=2048, ratio=4):
    selected = torch.full((scores.shape[0], budget + ratio - 1), -1, dtype=torch.int32)
    positions, lengths, requests = positions.cpu(), lengths.cpu(), requests.cpu()
    for row in range(scores.shape[0]):
        req, pos = int(requests[row]), int(positions[row])
        if not 0 <= req < lengths.numel():
            continue
        count = min((pos + 1) // ratio, int(lengths[req]) // ratio, budget // ratio)
        block_count = min((pos + 1) // ratio, int(lengths[req]) // ratio)
        blocks = scores[row, :block_count].topk(count).indices.tolist()
        tokens = [block * ratio + offset for block in blocks for offset in range(ratio)]
        tokens += list(range(((pos + 1) // ratio) * ratio, pos + 1))
        tokens = [x for x in tokens if 0 <= x < int(lengths[req])]
        selected[row, : len(tokens)] = torch.tensor(tokens, dtype=torch.int32)
    return selected


@pytest.mark.parametrize("pages", [64, 1100])
def test_qsa_selection_eager_and_changed_input_graph(pages):
    # Binary keys and power-of-two queries avoid ambiguous top-k ties.
    rows, page_size, dim = 2, 16, 128
    logical = torch.arange(pages * page_size, dtype=torch.int64)
    num_bits = (pages * page_size - 1).bit_length()
    bits = ((logical[:, None] >> torch.arange(num_bits)) & 1).to(torch.bfloat16)
    cpu_cache = torch.zeros(pages * page_size, dim, dtype=torch.bfloat16)
    cpu_cache[:, :num_bits] = bits
    cache = cpu_cache.reshape(pages, page_size, 1, dim).cuda()
    q = torch.zeros(rows, 4, dim, dtype=torch.bfloat16, device="cuda")
    q[:, :, :num_bits] = (2.0 ** torch.arange(num_bits, device="cuda")) / 32
    table = (
        torch.stack((torch.arange(pages), torch.arange(pages - 1, -1, -1))).cuda().int()
    )
    requests = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    positions = torch.tensor(
        [2401, 3202] if pages == 64 else [69001, 69630],
        device="cuda",
        dtype=torch.int64,
    )
    lengths = torch.tensor(
        [4096, 4096] if pages == 64 else [69632, 69632],
        device="cuda",
        dtype=torch.int64,
    )
    out = torch.empty(rows, 2051, device="cuda", dtype=torch.int32)

    def check():
        expected_scores, expected_visible = _score_oracle(
            q, cache, table, requests, positions, lengths, 4
        )
        scores, visible = ops.qsa_mqa_paged(
            q, cache, table, requests, positions, lengths, 4
        )
        torch.testing.assert_close(scores.cpu(), expected_scores, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(visible.cpu(), expected_visible, atol=0, rtol=0)
        expected = _selection_oracle(expected_scores, positions, lengths, requests)
        # Token order within the selected set does not change sparse attention.
        torch.testing.assert_close(
            out.cpu().sort(-1).values, expected.sort(-1).values, atol=0, rtol=0
        )

    def run():
        return metax.qsa_select_paged_tokens(
            q, cache, table, requests, positions, lengths, 2048, 4, out
        )

    run()
    check()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    requests.copy_(torch.tensor([1, 0], device="cuda", dtype=torch.int32))
    positions.copy_(
        torch.tensor(
            [3003, 2100] if pages == 64 else [69003, 69628],
            device="cuda",
            dtype=torch.int64,
        )
    )
    q[1].mul_(0.5)
    graph.replay()
    check()


def _sparse_oracle(q, key, value, indices, table, requests):
    q, key, value = q.cpu().float(), key.cpu().float(), value.cpu().float()
    indices, table, requests = indices.cpu(), table.cpu(), requests.cpu()
    result = torch.zeros_like(q)
    group = q.shape[1] // key.shape[2]
    for row in range(q.shape[0]):
        req = int(requests[row])
        if not 0 <= req < table.shape[0]:
            continue
        positions = indices[row][indices[row] >= 0].long()
        physical = table[req, positions // key.shape[1]].long()
        offsets = positions % key.shape[1]
        for head in range(q.shape[1]):
            k = key[physical, offsets, head // group]
            v = value[physical, offsets, head // group]
            logits = (k @ q[row, head]) * q.shape[-1] ** -0.5
            result[row, head] = logits.softmax(-1) @ v
    return result.to(torch.bfloat16)


def test_qsa_sparse_attention_eager_and_changed_input_graph():
    torch.manual_seed(812)
    q = torch.randn(2, 3, 256, device="cuda", dtype=torch.bfloat16)
    key = torch.randn(8, 16, 1, 256, device="cuda", dtype=torch.bfloat16)
    value = torch.randn_like(key)
    table = torch.stack((torch.arange(8), torch.arange(7, -1, -1))).cuda().int()
    requests = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    indices = torch.arange(20, device="cuda", dtype=torch.int32).repeat(2, 1)
    indices[:, -2:] = -1
    out = torch.empty_like(q)

    def run():
        return ops.qsa_sparse_paged_attention(
            q, key, value, indices, table, requests, out=out
        )

    run()
    torch.testing.assert_close(
        out.cpu(),
        _sparse_oracle(q, key, value, indices, table, requests),
        atol=2e-2,
        rtol=2e-2,
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    q.normal_()
    value.normal_()
    requests.copy_(torch.tensor([1, 0], device="cuda", dtype=torch.int32))
    indices[:, :18].add_(32)
    graph.replay()
    torch.testing.assert_close(
        out.cpu(),
        _sparse_oracle(q, key, value, indices, table, requests),
        atol=2e-2,
        rtol=2e-2,
    )


def test_gdn_prefill_preserves_fp32_vmajor_state():
    torch.manual_seed(913)
    t, h, hv, d = 8, 2, 6, 128
    q = torch.randn(1, t, h, d)
    k = torch.randn_like(q)
    q = (q / torch.sqrt(q.square().sum(-1, keepdim=True) + 1e-6)).bfloat16()
    k = (k / torch.sqrt(k.square().sum(-1, keepdim=True) + 1e-6)).bfloat16()
    v = (torch.randn(1, t, hv, d) * 0.3).bfloat16()
    g = -torch.rand(1, t, hv) * 0.5
    beta = torch.rand(1, t, hv)
    initial = torch.randn(1, hv, d, d) * 0.01
    reference_state = initial.clone()
    reference_output = torch.empty_like(v)
    for token in range(t):
        qr = q[0, token].float().repeat_interleave(hv // h, 0) * d**-0.5
        kr = k[0, token].float().repeat_interleave(hv // h, 0)
        state = reference_state[0] * g[0, token].exp()[:, None, None]
        delta = v[0, token].float() - (state * kr[:, None, :]).sum(-1)
        delta *= beta[0, token, :, None]
        state += delta[:, :, None] * kr[:, None, :]
        reference_output[0, token] = (state * qr[:, None, :]).sum(-1)
        reference_state[0] = state
    output, final = metax.gdn_prefill(
        q.cuda(),
        k.cuda(),
        v.cuda(),
        g.cuda(),
        beta.cuda(),
        initial.cuda(),
        True,
        cu_seqlens=torch.tensor([0, t], dtype=torch.int32, device="cuda"),
        chunk_indices=torch.tensor([[0, 0]], dtype=torch.int32, device="cuda"),
        chunk_offsets=torch.tensor([0, 1], dtype=torch.int32, device="cuda"),
        use_qk_l2norm_in_kernel=False,
    )
    assert final.dtype == torch.float32
    torch.testing.assert_close(output.cpu(), reference_output, atol=3e-3, rtol=2e-2)
    torch.testing.assert_close(final.cpu(), reference_state, atol=3e-3, rtol=2e-2)
    print({"prefill_state_max_abs": float((final.cpu() - reference_state).abs().max())})
