"""Packed FP32-beta GDN: independent oracle, strides and dynamic graph replay."""

import pytest
import torch
import torch.nn.functional as F

from vllm_fl.dispatch import resolve_op


def _reference(qkv, a, b, alog, bias, state, ids, h, k, v):
    hv = state.shape[1]
    q, key, value = qkv.float().split((h * k, h * k, hv * v), dim=-1)
    q = q.reshape(-1, h, k).repeat_interleave(hv // h, dim=1)
    key = key.reshape(-1, h, k).repeat_interleave(hv // h, dim=1)
    q = q / (q.square().sum(-1, keepdim=True) + 1e-6).sqrt() * k**-0.5
    key = key / (key.square().sum(-1, keepdim=True) + 1e-6).sqrt()
    value = value.reshape(-1, hv, v)
    result = torch.zeros((qkv.shape[0], 1, hv, v), dtype=qkv.dtype)
    for row, index in enumerate(ids.tolist()):
        if index <= 0 or index >= state.shape[0]:
            continue
        decay = (-alog.float().exp() * F.softplus(a[row].float() + bias.float())).exp()
        updated = state[index] * decay[:, None, None]
        delta = (value[row] - (updated * key[row, :, None, :]).sum(-1)) * b[
            row
        ].float().sigmoid()[:, None]
        updated += delta[:, :, None] * key[row, :, None, :]
        state[index] = updated
        result[row, 0] = (updated * q[row, :, None, :]).sum(-1).to(qkv.dtype)
    return result, state


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("graph_mode", [False, True])
def test_gdn_decode_oracle_eager_graph(dtype, graph_mode):
    if not torch.cuda.is_available():
        pytest.skip("CUDA-compatible accelerator runtime required by this graph test")
    torch.manual_seed(42)
    batch, h, hv, k, v = 3, 2, 4, 16, 17
    # Non-contiguous token strides are supported; inner dimensions are dense.
    qkv = torch.randn(batch * 2, 2 * h * k + hv * v, device="cuda", dtype=dtype)[::2]
    a = torch.randn(batch * 2, hv, device="cuda", dtype=dtype)[::2]
    b = torch.randn_like(a)
    alog = torch.randn(hv, device="cuda")
    bias = torch.randn(hv, device="cuda")
    state = torch.randn(4, hv, v, k, device="cuda", dtype=torch.float32)
    original = state.clone()
    ids = torch.tensor([1, 2, 0], device="cuda", dtype=torch.int64)
    output = torch.empty(batch, 1, hv, v, device="cuda", dtype=dtype)
    op = resolve_op("gdn_packed_decode")

    def run():
        return op(qkv, a, b, alog, bias, k**-0.5, state, output, ids, True)

    # Warm kernels on a side stream before capture.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
    torch.cuda.current_stream().wait_stream(stream)
    if graph_mode:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
    state.copy_(original)
    reference_state = original.cpu()
    for step in range(12):
        b.add_(0.03125)
        # New mapping must be read by replay, including null and invalid rows.
        ids.copy_(torch.tensor([2, 1, 99 if step % 2 else 0], device="cuda"))
        expected, reference_state = _reference(
            qkv.cpu(),
            a.cpu(),
            b.cpu(),
            alog.cpu(),
            bias.cpu(),
            reference_state,
            ids.cpu(),
            h,
            k,
            v,
        )
        graph.replay() if graph_mode else run()
        torch.cuda.synchronize()
        torch.testing.assert_close(state.cpu(), reference_state, rtol=3e-4, atol=3e-5)
        torch.testing.assert_close(output.cpu(), expected, rtol=2e-2, atol=2e-3)


def test_gdn_rejects_invalid_dtype_before_launch():
    from flag_gems.fused import gdn_packed_decode

    x = torch.zeros(1, 8, dtype=torch.int32)
    with pytest.raises(TypeError, match="QKV"):
        gdn_packed_decode(x, x, x, x, x, 1.0, x, x, x)
