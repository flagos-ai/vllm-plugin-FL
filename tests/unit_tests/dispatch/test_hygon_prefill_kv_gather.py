# Copyright (c) 2026 BAAI. All rights reserved.
import numpy as np
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='HIP GPU required')


@pytest.mark.parametrize('fnuz', [False])
@pytest.mark.parametrize('layout', ['flat', 'padded_view', 'packed_view'])
@pytest.mark.parametrize('block,requests,length,tail', [
    (2,4,515,137), (64,1,16387,None), (64,4,16387,4095),
    (256,4,65539,8191), (64,2,4097,0),
])
def test_gather_graph_and_cpu(block, requests, length, tail, fnuz, layout, monkeypatch):
    from vllm.models.deepseek_v4.common.ops import cache_utils
    from vllm.models.deepseek_v4.common.ops.cache_utils import dequantize_and_gather_k_cache_triton
    from vllm_fl.dispatch.backends.vendor.hygon.impl.attention.prefill_kv_gather import dequantize_and_gather_k_cache_hygon
    torch.manual_seed(952)
    nb = (length+block-1)//block+3
    stride = ((block*584+575)//576)*576
    physical_stride = stride * 2 + 128 if layout == 'packed_view' else stride
    offset_bytes = 64 if layout == 'packed_view' else 0
    backing_cache = torch.zeros(requests*nb,physical_stride,device='cuda',dtype=torch.uint8)
    pages = backing_cache[:,offset_bytes:offset_bytes+stride]
    cache = pages if layout == 'flat' else torch.as_strided(
        pages, (requests*nb,block,584), (physical_stride,584,1))
    if layout != 'flat':
        assert not cache.is_contiguous()
    # A result-only test could silently pass via the original fallback.
    def forbidden_fallback(*args, **kwargs):
        pytest.fail('Supported physical-page view fell back to the original kernel')
    monkeypatch.setattr(cache_utils, 'dequantize_and_gather_k_cache', forbidden_fallback)
    data = pages[:,:block*576].view(-1,block,576)
    raw = torch.randint(0,256,(requests*nb,block,448),device='cuda',dtype=torch.uint8)
    if fnuz:raw.masked_fill_(raw == 128,0)
    else:raw.masked_fill_((raw&127) == 127,0)
    data[:,:,:448] = raw
    data[:,:,448:] = torch.randn(requests*nb,block,64,device='cuda',dtype=torch.bfloat16).view(torch.uint8)
    pages[:,block*576:block*584] = torch.randint(118,134,(requests*nb,block*8),device='cuda',dtype=torch.uint8)
    seqcpu = torch.tensor([length-r*3 for r in range(requests)],dtype=torch.int32)
    seq = seqcpu.cuda()
    lens = None if tail is None else torch.full((requests,),tail,device='cuda',dtype=torch.int32)
    tb = torch.randperm(requests*nb,device='cuda').view(requests,nb).int()
    table_storage = torch.empty(requests,nb+3,device='cuda',dtype=torch.int32)
    table = table_storage[:,:nb];table.copy_(tb)
    offset = 17
    capacity = length if tail is None else tail
    backing = torch.full((requests,capacity+offset+9,520),-7.,device='cuda',dtype=torch.bfloat16)
    ref_backing = torch.full_like(backing,-7.)
    out, ref = backing[:,:,:512],ref_backing[:,:,:512]
    def run():dequantize_and_gather_k_cache_hygon(out,cache,seq,lens,table,block,offset,fnuz)
    run();torch.cuda.synchronize();graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):run()
    for stage in range(3):
        current = (seqcpu//(stage+1)).clamp_min(0)
        seq.copy_(current)
        gathered = current if tail is None else current.clamp_max(tail//(stage+1))
        if lens is not None:lens.copy_(gathered)
        table.copy_(table.roll(1,1))
        backing.fill_(-7.);ref_backing.fill_(-7.)
        graph.replay()
        dequantize_and_gather_k_cache_triton(ref,cache,seq,lens,table.contiguous(),block,offset,use_fnuz=fnuz)
        assert torch.equal(backing.view(torch.int16),ref_backing.view(torch.int16))
        # Decode selected physical bytes independently on CPU in float64.
        for r in range(requests):
            count = int(gathered[r])
            for i in sorted(set([0,count//2,count-1])) if count else []:
                pos = int(current[r])-count+i
                physical = int(table[r,pos//block])
                page = pages[physical].cpu()
                at = (pos%block)*576
                bits = page[at:at+448].numpy().astype(np.int32)
                exponent,mantissa = (bits>>3)&15,bits&7
                bias = 8 if fnuz else 7
                value = np.where(exponent == 0,np.ldexp(mantissa/8.,1-bias),np.ldexp(1+mantissa/8.,exponent-bias))
                value *= 1-2*(bits>>7)
                scale = page[block*576+(pos%block)*8:block*576+(pos%block)*8+7].numpy().astype(np.int32)
                value *= np.exp2(np.repeat(scale,64)-127.)
                expected = torch.cat((torch.from_numpy(value).bfloat16(),page[at+448:at+576].view(torch.bfloat16)))
                assert torch.equal(out[r,offset+i].cpu().view(torch.int16),expected.view(torch.int16))


def test_hook_and_disabled_fallback(monkeypatch):
    from vllm.models.deepseek_v4.amd import rocm
    from vllm.models.deepseek_v4.common.ops import cache_utils
    from vllm_fl.dispatch.backends.vendor.hygon.impl.attention.prefill_kv_gather import patch_prefill_kv_gather, dequantize_and_gather_k_cache_hygon
    patch_prefill_kv_gather()
    assert rocm.dequantize_and_gather_k_cache is dequantize_and_gather_k_cache_hygon
    calls = []
    monkeypatch.setenv('VLLM_FL_HYGON_PREFILL_KV_GATHER','0')
    monkeypatch.setattr(cache_utils,'dequantize_and_gather_k_cache',lambda *a,**kw:calls.append((a,kw)))
    out = torch.empty(1,1,512,device='cuda',dtype=torch.bfloat16)
    cache = torch.empty(1,64*584,device='cuda',dtype=torch.uint8)
    seq = torch.zeros(1,device='cuda',dtype=torch.int32)
    table = torch.zeros(1,1,device='cuda',dtype=torch.int32)
    rocm.dequantize_and_gather_k_cache(out,cache,seq,None,table,64,0)
    assert len(calls) == 1


def test_fnuz_retains_reference(monkeypatch):
    from vllm.models.deepseek_v4.common.ops import cache_utils
    from vllm_fl.dispatch.backends.vendor.hygon.impl.attention.prefill_kv_gather import dequantize_and_gather_k_cache_hygon
    calls = []
    monkeypatch.setattr(cache_utils, 'dequantize_and_gather_k_cache', lambda *args: calls.append(args))
    out = torch.empty(1, 1, 512, device='cuda', dtype=torch.bfloat16)
    cache = torch.empty(1, 64*584, device='cuda', dtype=torch.uint8)
    seq = torch.zeros(1, device='cuda', dtype=torch.int32)
    table = torch.zeros(1, 1, device='cuda', dtype=torch.int32)
    dequantize_and_gather_k_cache_hygon(out, cache, seq, None, table, 64, 0, True)
    assert len(calls) == 1 and calls[0][-1] is True


def test_reject_unsupported_page_layout():
    from vllm_fl.dispatch.backends.vendor.hygon.impl.attention.prefill_kv_gather import _paged_cache_layout_supported
    raw = torch.empty(4, 64, 1168, device='cuda', dtype=torch.uint8)
    assert not _paged_cache_layout_supported(raw[:, :, ::2], 64)
    assert not _paged_cache_layout_supported(raw[:, :63, :584], 64)
    flat = raw.view(4, -1)
    assert not _paged_cache_layout_supported(flat[:, 1:64*584+1], 64)


@pytest.mark.parametrize('block', [2, 64, 256])
def test_paged_gather_to_sparse_attention(block):
    from tests.unit_tests.dispatch.test_hygon_main_sparse_attention_correctness import (
        _make_ocp_fp8_ds_mla_cache, _sparse_attention_reference,
    )
    from vllm.models.deepseek_v4.amd import rocm
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import rocm_sparse_attn_prefill
    from vllm_fl.dispatch.backends.vendor.hygon.impl.attention.prefill_kv_gather import patch_prefill_kv_gather

    torch.manual_seed(955)
    patch_prefill_kv_gather()
    rows = torch.randn(256, 512, device='cuda', dtype=torch.bfloat16) * .125
    cache, decoded = _make_ocp_fp8_ds_mla_cache(rows, block)
    table = torch.arange(cache.shape[0], device='cuda', dtype=torch.int32)[None]
    seq = torch.tensor([249], device='cuda', dtype=torch.int32)
    lens = torch.tensor([137], device='cuda', dtype=torch.int32)
    gathered = torch.zeros(1, 154, 512, device='cuda', dtype=torch.bfloat16)
    q = torch.randn(6, 16, 512, device='cuda', dtype=torch.bfloat16) * .125
    idx = torch.stack([torch.randperm(137, device='cuda')[:96] + 17 for _ in range(6)]).int()
    lengths = torch.full((6,), 96, device='cuda', dtype=torch.int32)
    sink = torch.randn(16, device='cuda') * .1
    out = torch.empty_like(q)

    def run():
        rocm.dequantize_and_gather_k_cache(gathered, cache, seq, lens, table, block, 17)
        rocm_sparse_attn_prefill(
            q=q, kv=gathered[0, :, None], indices=idx, topk_length=lengths,
            scale=512**-.5, head_dim=512, nope_head_dim=448, rope_head_dim=64,
            attn_sink=sink, output=out,
        )
    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    graph.replay()
    expected = _sparse_attention_reference(
        q, [decoded[112 + idx[r].long() - 17] for r in range(6)], 512**-.5, sink)
    assert torch.equal(gathered[0, 17:], decoded[112:249])
    torch.testing.assert_close(out, expected, atol=.035, rtol=.04)
