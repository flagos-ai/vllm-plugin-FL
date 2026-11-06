# Copyright (c) 2026 BAAI. All rights reserved.
"""Prefill integration regressions, including the fixed LightOp K512 failures."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='HIP GPU required')


def _oracle(x, starts, ends, indices):
    k = indices.shape[1]
    for row, (s, e) in enumerate(zip(starts, ends)):
        s = min(x.shape[1], max(0, int(s)))
        e = min(x.shape[1], max(s, int(e)))
        values = x[row, s:e]
        values = values[values != -np.inf]
        need = min(k, len(values))
        chosen = indices[row][indices[row] != -1]
        assert len(chosen) == need
        assert (chosen >= 0).all() and (chosen < e-s).all()
        assert len(np.unique(chosen)) == len(chosen)
        if need:
            threshold = np.partition(values, -need)[-need]
            selected = x[row, s+chosen]
            assert (selected >= threshold).all()
            assert (selected > threshold).sum() == (values > threshold).sum()


@pytest.mark.parametrize('case', ['saved', 'equal', 'plateau', 'narrow', 'short', 'wide'])
@pytest.mark.parametrize('strided', [False, True])
def test_exact_topk_graph(case, strided):
    from vllm_fl.dispatch.backends.vendor.hygon.impl.attention.sparse_attn_indexer import top_k_per_row_prefill_hygon_out
    torch.manual_seed(945)
    if case == 'saved':
        d = torch.load(Path(__file__).parent/'data/hygon_prefill_topk512.pt', weights_only=True)
        x, s, e = d['logits'], d['starts'], d['ends']
    else:
        n = 16389 if case == 'wide' else (127 if case == 'short' else 4096)
        x = torch.randn(8, n)
        s = torch.tensor([0, 1, 17, 63, -2, 1, 0, 63], dtype=torch.int32)
        e = torch.tensor([0, 2, 513, n, n+7, 512, n-1, 128], dtype=torch.int32)
        if case == 'equal':x.zero_()
        if case == 'plateau':x = (x*4).round()
        if case == 'narrow':x = 1+x*1e-6
    m, n = x.shape
    backing = torch.empty((m, n*(2 if strided else 1)), device='cuda')
    inp = backing[:, ::2] if strided else backing
    ob = torch.full((m, 1024), -777, device='cuda', dtype=torch.int32)
    out = ob[:, ::2] if strided else ob[:, :512]
    ss, ee = s.cuda(), e.cuda()
    def run():
        top_k_per_row_prefill_hygon_out(inp, ss, ee, out, m, *inp.stride(), 512)
    inp.copy_(x);run();torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):run()
    for stage in range(3):
        xx = x.clone() if stage != 2 else torch.zeros_like(x)
        if stage == 1:xx *= .5
        for row in range(m):
            lo = min(n, max(0, int(s[row])))
            hi = min(n, max(lo, int(e[row])))
            xx[row, :lo] = torch.nan;xx[row, hi:] = torch.nan
        inp.copy_(xx);ob.fill_(-777);graph.replay()
        _oracle(xx.numpy(), s.numpy(), e.numpy(), out.cpu().numpy())
        unused = ob[:, 1::2] if strided else ob[:, 512:]
        assert bool((unused == -777).all())


@pytest.mark.parametrize('mode', ['auto', 'baseline', 'reference'])
@pytest.mark.parametrize('spec', [
    ([4], [4], None), ([127, 129], [1024, 1536], None),
    ([1024, 1024], [8192, 8192], None),
    ([1024, 1024], [16383, 16369], (3, 1555)),
    ([3, 1024], [3, 16384], None),
    ([129], [65531], (1, 128)),
])
def test_chunk_metadata_and_workspace_graph(monkeypatch, mode, spec):
    from lightop import attention
    from vllm.v1.attention.backends.mla import indexer
    from vllm_fl.dispatch.backends.vendor.hygon.impl.attention import bf16_indexer as impl
    from vllm_fl.dispatch.backends.vendor.hygon.impl.attention.prefill_indexer_metadata import patch_prefill_indexer_metadata
    monkeypatch.setenv('VLLM_FL_HYGON_PREFILL_INDEXER', mode)
    patch_prefill_indexer_metadata()
    torch.manual_seed(946)
    qlens, slens, qs = spec
    # Leave three leading decode tokens untouched by the prefill invocation.
    qcpu = torch.tensor([0, 1, 3]+(3+torch.tensor(qlens).cumsum(0)).tolist(), dtype=torch.int32)
    seqcpu = torch.tensor([128, 128]+slens, dtype=torch.int32)
    seq = seqcpu.cuda();compressed = seq//4
    table = torch.zeros((len(slens)+2, 1024), device='cuda', dtype=torch.int32)
    chunk = indexer.build_prefill_chunk_metadata(2, len(slens)+2, qcpu.cuda(), qcpu,
                seq, compressed, seqcpu//4, table, 4,
                query_slice=None if qs is None else slice(*qs))
    m, n = chunk.token_end-chunk.token_start, chunk.total_seq_lens
    assert sum(qe-qb for qb,qe,_,_ in chunk.hygon_request_spans) == m
    q = torch.randn(int(qcpu[-1]), 64, 128, device='cuda', dtype=torch.bfloat16)*.125
    k = torch.randn(n, 128, device='cuda', dtype=torch.bfloat16)*.125
    w = torch.randn(q.shape[0], 64, device='cuda')*.02
    out = torch.full((q.shape[0], 512), -777, device='cuda', dtype=torch.int32)
    # Force an additional forward-time workspace split beyond metadata slicing.
    monkeypatch.setattr(impl, '_LOGITS_BUFFER_ELEMENTS', 128*((n+127)//128*128))
    monkeypatch.setattr(impl, '_logits_buffers', {})
    def run():impl._prefill_bf16_mqa_chunked(chunk,q,k,w,out,512)
    run();torch.cuda.synchronize();graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):run()
    orig = chunk.cu_seqlen_ke.clone()
    for stage in range(3):
        q.normal_(std=.125);k.normal_(std=.125);w.normal_(std=.02)
        if stage == 2:q.zero_()
        chunk.cu_seqlen_ke.copy_(chunk.cu_seqlen_ks+(orig-chunk.cu_seqlen_ks)//(stage+1))
        if hasattr(chunk, 'hygon_local_ends'):
            chunk.hygon_local_ends.copy_(chunk.cu_seqlen_ke-chunk.cu_seqlen_ks)
        out.fill_(-777);graph.replay()
        ref = attention.mqa_logits(q[chunk.token_start:chunk.token_end],k,
            w[chunk.token_start:chunk.token_end],chunk.cu_seqlen_ks,chunk.cu_seqlen_ke)
        _oracle(ref.cpu().numpy(),chunk.cu_seqlen_ks.cpu().numpy(),chunk.cu_seqlen_ke.cpu().numpy(),
                out[chunk.token_start:chunk.token_end].cpu().numpy())
        assert bool((out[:chunk.token_start] == -777).all())
        assert bool((out[chunk.token_end:] == -777).all())


def test_empty_topk():
    from vllm_fl.dispatch.backends.vendor.hygon.impl.attention.prefill_topk import prefill_topk
    x = torch.empty((4, 0), device='cuda');s = torch.zeros(4, device='cuda', dtype=torch.int32)
    out = torch.empty((4,512), device='cuda', dtype=torch.int32)
    prefill_topk(x,s,s,out)
    assert bool((out == -1).all())


def test_varying_mixed_lengths_reuse_compiled_kernels():
    """Changing exact lengths/compact strides must not create a kernel per step."""
    from vllm_fl.dispatch.backends.vendor.hygon.impl.attention.prefill_mqa import prefill_mqa, mqa_headloop
    from vllm_fl.dispatch.backends.vendor.hygon.impl.attention.prefill_topk import prefill_topk, _radix
    torch.manual_seed(956)
    device = torch.cuda.current_device()
    q = torch.randn(8192, 64, 128, device='cuda', dtype=torch.bfloat16) * .125
    k = torch.randn(4096, 128, device='cuda', dtype=torch.bfloat16) * .125
    w = torch.randn(8192, 64, device='cuda') * .02
    s = torch.zeros(8192, device='cuda', dtype=torch.int32)
    e = torch.empty_like(s)
    storage = torch.empty(8192 * 4096, device='cuda')
    output = torch.empty(8192, 512, device='cuda', dtype=torch.int32)
    baseline = None
    for delta in range(64):
        m, n = 8192 - delta, 4096 - delta
        e.fill_(n)
        logits = storage[:m*n].view(m,n)
        prefill_mqa(q[:m], k[:n], w[:m], s[:m], e[:m], logits)
        prefill_topk(logits, s[:m], e[:m], output[:m], already_masked=True)
        torch.cuda.synchronize()
        sizes = (len(mqa_headloop.device_caches[device][0]), len(_radix.device_caches[device][0]))
        if baseline is None:
            baseline = sizes
        assert sizes == baseline, (delta, baseline, sizes)
        # Check boundary rows against an independent GPU TopK threshold.
        rows = torch.tensor([0,m-1],device='cuda')
        scores = logits[rows]
        chosen = output[rows].long()
        threshold = scores.topk(512,dim=-1).values[:,-1:]
        assert bool((scores.gather(1,chosen) >= threshold).all())
        assert bool((chosen.sort(dim=1).values[:,1:] != chosen.sort(dim=1).values[:,:-1]).all())


@pytest.mark.parametrize('reference', [False, True])
@pytest.mark.parametrize('n', [127, 2047, 4096, 4097, 16389, 32768])
@pytest.mark.parametrize('already_masked', [False, True])
def test_topk_under_flaggems_dispatch(reference, n, already_masked):
    """Service dispatch must preserve padded/strided input and graph replay."""
    import flag_gems
    from vllm_fl.dispatch.backends.vendor.hygon.impl.attention.prefill_topk import prefill_topk
    m = 8
    rng = np.random.default_rng(947)
    x = rng.normal(size=(m, n)).astype(np.float32)
    s = np.array([0, 1, 17, 63, -2, 1, 0, 63], dtype=np.int32)
    e = np.array([0, 2, 513, n, n+7, 512, n-1, 128], dtype=np.int32)
    # Both padded rows and a nonunit column stride, with invalid-region NaNs.
    backing = torch.empty(m, 2*n+128, device='cuda')
    inp = backing[:, :2*n:2]
    ob = torch.full((m, 1024), -777, device='cuda', dtype=torch.int32)
    out = ob[:, ::2]
    ss, ee = torch.from_numpy(s).cuda(), torch.from_numpy(e).cuda()
    with flag_gems.use_gems(include=['topk']):
        assert 'topk' in flag_gems.current_work_registrar.get_all_keys()
        assert 'topk.values' not in flag_gems.current_work_registrar.get_all_keys()
        assert '/pytorch/' in torch._C._dispatch_dump_table('aten::topk.values')
        def run():
            prefill_topk(inp, ss, ee, out, reference=reference,
                         already_masked=already_masked)
        inp.copy_(torch.from_numpy(x));run();torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):run()
        for stage in range(3):
            xx = x.copy() if stage == 0 else (np.round(x) if stage == 1 else np.zeros_like(x))
            for row in range(m):
                lo = min(n, max(0, int(s[row])))
                hi = min(n, max(lo, int(e[row])))
                invalid = -np.inf if already_masked else np.nan
                xx[row, :lo] = invalid;xx[row, hi:] = invalid
            inp.copy_(torch.from_numpy(xx));ob.fill_(-777);graph.replay()
            _oracle(xx, s, e, out.cpu().numpy())
            assert bool((ob[:, 1::2] == -777).all())


def test_baseline_uses_lightop_mqa_only(monkeypatch):
    from vllm.v1.attention.backends.mla import indexer
    from vllm_fl.dispatch.backends.vendor.hygon.impl.attention import bf16_indexer as impl, prefill_mqa
    from vllm_fl.dispatch.backends.vendor.hygon.impl.attention.prefill_indexer_metadata import patch_prefill_indexer_metadata
    import flag_gems
    monkeypatch.setenv('VLLM_FL_HYGON_PREFILL_INDEXER', 'baseline')
    patch_prefill_indexer_metadata()
    def forbidden(*args, **kwargs):
        raise AssertionError('baseline must not use Triton MQA')
    monkeypatch.setattr(prefill_mqa, 'prefill_mqa', forbidden)
    with flag_gems.use_gems(include=['topk']):
        test_chunk_metadata_and_workspace_graph(monkeypatch, 'baseline',
                                                ([1024,1024], [8192,8192], None))


def test_bf16_gather_dynamic_lengths_and_layout():
    from vllm_fl.dispatch.backends.vendor.hygon.impl.attention.bf16_indexer import (
        gather_bf16_indexer_cache, _gather_bf16_indexer_cache_kernel)
    rng = np.random.default_rng(948)
    device = torch.cuda.current_device()
    compiled = None
    for step in range(16):
        batch = 1 + step % 4
        lens = [257-step-i*3 for i in range(batch)]
        cu = np.array([0]+list(np.cumsum(lens)), dtype=np.int32)
        width = 8 + step
        # Padded physical cache pages and changing block-table strides.
        cpu_cache = rng.normal(size=(64, 64, 128)).astype(np.float32)
        storage = torch.empty(64,64*128+16*step,device='cuda',dtype=torch.bfloat16)
        cache = storage[:, :64*128].view(64,64,128)
        cache.copy_(torch.from_numpy(cpu_cache))
        ids = rng.integers(0,64,size=(batch,width),dtype=np.int32)
        table = torch.from_numpy(ids).cuda()
        out = torch.empty(int(cu[-1]),128,device='cuda',dtype=torch.bfloat16)
        gather_bf16_indexer_cache(cache,out,table,torch.from_numpy(cu).cuda())
        torch.cuda.synchronize()
        count = len(_gather_bf16_indexer_cache_kernel.device_caches[device][0])
        if compiled is None:compiled=count
        assert count == compiled, (step,compiled,count)
        rounded = torch.from_numpy(cpu_cache).to(torch.bfloat16).float().numpy()
        expected = np.concatenate([rounded[ids[b,np.arange(n)//64], np.arange(n)%64]
                                   for b,n in enumerate(lens)])
        np.testing.assert_array_equal(out.float().cpu().numpy(), expected)
