# Copyright (c) 2026 BAAI. All rights reserved.
"""Exact LightOp quantization contract and actual wo_a model dispatch."""
from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='HIP GPU required')


def table(dtype):
    angle = torch.randn(131072, 32, device='cuda')
    return torch.cat((angle.cos(), angle.sin()), -1).to(dtype)


@pytest.mark.parametrize('m', [257, 1024, 8131, 8192])
@pytest.mark.parametrize('strided', [False, True])
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_quantized_values_and_scales_exact_graph(m, strided, dtype):
    from lightop.quant import per_token_quant_int8
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import _fused_inverse_rope_gptj
    from vllm_fl.dispatch.backends.vendor.hygon.impl.attention.prefill_inverse_rope_quant import prefill_inverse_rope_quant
    torch.manual_seed(920)
    storage = torch.empty(m*2 if strided else m, 16 if strided else 8, 512,
                          device='cuda', dtype=torch.bfloat16)
    o = storage[::2, ::2] if strided else storage
    positions = torch.zeros(m, device='cuda', dtype=torch.int32 if strided else torch.int64)
    cs = table(dtype)
    def run():return prefill_inverse_rope_quant(o, positions, cs)
    o.zero_();run();torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = run()
    for std, start in [(0., 0), (1e-12, 16384), (.25, 65536-m), (10., 131072-m)]:
        o.normal_(std=std)
        positions.copy_(torch.arange(start, start+m, device='cuda'))
        before = o.clone();graph.replay()
        expected = per_token_quant_int8(_fused_inverse_rope_gptj(o, positions, cs, 64).reshape(m, 4096))
        assert all(torch.equal(a, b) for a, b in zip(actual, expected))
        assert torch.equal(o, before)


@pytest.mark.parametrize('case', ['off', 'm16', 'm24', 'm32', 'm48', 'm64', 'm256',
                                 'groups', 'backend', 'dtype', 'position_stride', 'table_stride'])
def test_rollback_and_decode_isolation(case, monkeypatch):
    from vllm_fl.dispatch.backends.vendor.hygon.impl.attention import mla, prefill_inverse_rope_quant as leaf
    def forbidden(*args, **kwargs):
        pytest.fail('Unsupported/disabled/decode input reached the prefill kernel')
    monkeypatch.setattr(leaf, 'prefill_inverse_rope_quant', forbidden)
    monkeypatch.setenv('VLLM_FL_HYGON_PREFILL_INVERSE_ROPE_QUANT', '0' if case == 'off' else '1')
    m = int(case[1:]) if case.startswith('m') else 257
    o = torch.empty((m, 8, 512), device='cuda', dtype=torch.float32 if case == 'dtype' else torch.bfloat16)
    positions = torch.zeros(m*2 if case == 'position_stride' else m, device='cuda', dtype=torch.int64)
    if case == 'position_stride':positions = positions[::2]
    cs = torch.empty((8, 128 if case == 'table_stride' else 64), device='cuda')
    if case == 'table_stride':cs = cs[:, ::2]
    model = SimpleNamespace(rotary_emb=SimpleNamespace(cos_sin_cache=cs),
                            n_local_groups=2 if case == 'groups' else 1,
                            o_lora_rank=1024, rope_head_dim=64)
    backend = SimpleNamespace(name='LMSlim' if case == 'backend' else 'LightOp')
    assert mla._try_prefill_inverse_rope_wo_a(model, o, positions, backend) is None


@pytest.mark.parametrize('m', [257, 1024, 8131, 8192])
def test_actual_o_proj_dispatch_and_graph(m, monkeypatch):
    from vllm_fl.dispatch.backends.vendor.hygon.impl.attention import mla
    from vllm.models.deepseek_v4.amd.rocm import DeepseekV4ROCMAiterMLAAttention
    torch.manual_seed(921)
    mla._patch_rocm_wo_a_group_gemm()
    model = SimpleNamespace(
        wo_a=SimpleNamespace(weight=torch.randint(-30, 31, (1024, 4096), device='cuda', dtype=torch.int8),
                             weight_scale=torch.rand((1024, 1), device='cuda')*.01),
        rotary_emb=SimpleNamespace(cos_sin_cache=table(torch.float32)),
        n_local_groups=1, o_lora_rank=1024, rope_head_dim=64,
        wo_b=lambda x: x,
    )
    o = torch.randn((m, 8, 512), device='cuda', dtype=torch.bfloat16)
    positions = torch.arange(65536-m, 65536, device='cuda')
    def run():return DeepseekV4ROCMAiterMLAAttention._o_proj(model, o, positions)
    monkeypatch.setenv('VLLM_FL_HYGON_PREFILL_INVERSE_ROPE_QUANT', '0')
    run();old_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(old_graph):expected = run()
    monkeypatch.setenv('VLLM_FL_HYGON_PREFILL_INVERSE_ROPE_QUANT', '1')
    run();new_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(new_graph):actual = run()
    for std in (0., .25, 10.):
        o.normal_(std=std);positions.copy_(torch.randint(0, 131072, (m,), device='cuda'))
        old_graph.replay();new_graph.replay()
        assert torch.equal(actual, expected)


def test_mixed_lengths_reuse_compilation():
    from vllm_fl.dispatch.backends.vendor.hygon.impl.attention.prefill_inverse_rope_quant import prefill_inverse_rope_quant, _inverse_rope_quant
    o = torch.randn((8192, 8, 512), device='cuda', dtype=torch.bfloat16)
    positions = torch.arange(8192, device='cuda');cs = table(torch.float32)
    prefill_inverse_rope_quant(o, positions, cs);torch.cuda.synchronize()
    cache = _inverse_rope_quant.device_caches[torch.cuda.current_device()][0]
    before = len(cache)
    for m in range(8192, 8128, -1):prefill_inverse_rope_quant(o[:m], positions[:m], cs)
    torch.cuda.synchronize()
    assert len(cache) == before
