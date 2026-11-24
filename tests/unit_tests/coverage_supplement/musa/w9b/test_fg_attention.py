"""Coverage tests for flaggems/impl/attention.py (CPU-only, stubbed kernels)."""
from types import SimpleNamespace
import pytest
import torch
from vllm.v1.attention.backend import AttentionType
import vllm_fl.dispatch.backends.flaggems.impl.attention as attention

def _impl(**kw):
    params = dict(num_heads=4, head_size=64, scale=0.125, num_kv_heads=2, alibi_slopes=None, sliding_window=None, kv_cache_dtype='auto')
    params.update(kw)
    return attention.AttentionFLImpl(**params)

def _layer():
    return SimpleNamespace(_q_scale=torch.ones((), dtype=torch.float32), _k_scale=torch.ones((), dtype=torch.float32), _v_scale=torch.ones((), dtype=torch.float32))

def _metadata(**kw):
    base = dict(num_actual_tokens=3, use_cascade=False, query_start_loc=torch.tensor([0, 3], dtype=torch.int32), seq_lens=torch.tensor([5], dtype=torch.int32), max_query_len=3, max_seq_len=5, block_table=torch.zeros((1, 2), dtype=torch.int32), scheduler_metadata=None, causal=True)
    base.update(kw)
    return SimpleNamespace(**base)

class _FlashRecorder:

    def __init__(self, return_lse=False):
        self.calls = []
        self.return_lse = return_lse

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        q = kwargs['q']
        if kwargs.get('return_softmax_lse'):
            b = q.shape[0]
            return (torch.zeros_like(q), torch.zeros((q.shape[1], b)))
        return kwargs.get('out', torch.zeros_like(q))

def test_backend_metadata_and_capabilities(monkeypatch):
    b = attention.AttentionFLBackend
    assert b.get_name() == 'CUSTOM'
    assert b.get_impl_cls() is attention.AttentionFLImpl
    assert b.get_builder_cls() is attention.AttentionFLMetadataBuilder
    assert b.supports_sink() is False
    assert b.supports_attn_type(AttentionType.DECODER) is True
    assert b.supports_attn_type(AttentionType.ENCODER_DECODER) is True
    assert b.supports_attn_type('nope') is False
    assert b.supports_kv_cache_dtype(None) is True
    assert b.supports_kv_cache_dtype('auto') is True
    assert b.supports_kv_cache_dtype('fp8') is False
    assert b.supports_head_size(64) is True
    assert b.supports_head_size(65) is False
    assert b.supports_head_size(264) is False
    assert b.supports_combination(64, torch.float16, 'auto', 16, False, True, False, False, None) == 'not support sink'
    assert 'multimodal' in b.supports_combination(64, torch.float16, 'auto', 16, False, False, False, True, None)
    assert b.supports_combination(64, torch.float16, 'auto', 16, False, False, False, False, None) is None

def test_supported_block_sizes_variants(monkeypatch):
    monkeypatch.setattr(attention, 'get_current_vllm_config', lambda : SimpleNamespace(model_config=SimpleNamespace(is_hybrid=True), cache_config=SimpleNamespace(mamba_ssm_cache_dtype='float32', mamba_cache_dtype='auto')))
    assert attention.AttentionFLBackend.get_supported_kernel_block_sizes() == [16, 32, 64]
    monkeypatch.setattr(attention, 'get_current_vllm_config', lambda : SimpleNamespace(model_config=SimpleNamespace(is_hybrid=False), cache_config=SimpleNamespace(mamba_ssm_cache_dtype='auto', mamba_cache_dtype='auto')))
    sizes = attention.AttentionFLBackend.get_supported_kernel_block_sizes()
    assert len(sizes) == 1

def test_split_kv_cache_errors():
    with pytest.raises(ValueError, match='equal K/V head sizes'):
        attention._split_kv_cache(torch.zeros((2, 2, 4, 24)), 8)
    with pytest.raises(ValueError, match='Unsupported FlagGems KV cache shape'):
        attention._split_kv_cache(torch.zeros((2, 2, 4)), 8)

def test_impl_init_variants():
    impl = _impl(alibi_slopes=[0.1, 0.2], logits_soft_cap=30.0, kv_sharing_target_layer_name='x')
    assert impl.alibi_slopes.dtype == torch.float32
    assert impl.logits_soft_cap == 30.0
    assert impl.kv_sharing_target_layer_name == 'x'
    assert impl.sliding_window == (-1, -1)
    enc = _impl(attn_type=AttentionType.ENCODER_ONLY, sliding_window=8)
    assert enc.sliding_window == (7, 7)
    dec = _impl(sliding_window=8)
    assert dec.sliding_window == (7, 0)

def test_impl_init_rejects_quantized_kv_cache():
    with pytest.raises(NotImplementedError, match='quantization kv-cache'):
        _impl(kv_cache_dtype='fp8')

def test_do_kv_cache_update_encoder_is_noop(monkeypatch):
    impl = _impl(attn_type=AttentionType.ENCODER_ONLY)
    cache = torch.zeros((2, 2, 16, 128))
    out = impl.do_kv_cache_update(_layer(), torch.zeros((3, 2, 64)), torch.zeros((3, 2, 64)), cache, torch.zeros(3, dtype=torch.int64))
    assert out is None

def test_do_kv_cache_update_writes_via_triton(monkeypatch):
    calls = []
    monkeypatch.setitem(__import__('sys').modules, 'vllm.v1.attention.ops.triton_reshape_and_cache_flash', SimpleNamespace(triton_reshape_and_cache_flash=lambda *a, **k: calls.append((a, k))))
    impl = _impl()
    cache = torch.zeros((2, 2, 16, 128))
    slots = torch.tensor([0, 1, 2], dtype=torch.int64)
    impl.do_kv_cache_update(_layer(), torch.zeros((3, 2, 64)), torch.zeros((3, 2, 64)), cache, slots)
    assert len(calls) == 1
    assert calls[0][0][4] is slots

def test_forward_requires_output():
    with pytest.raises(AssertionError, match='Output tensor must be provided'):
        _impl().forward(_layer(), None, None, None, None, _metadata())

def test_forward_rejects_fused_quant():
    with pytest.raises(NotImplementedError, match='fused output quantization'):
        _impl().forward(_layer(), None, None, None, None, _metadata(), output=torch.zeros((3, 4, 64)), output_scale=torch.ones(1))

def test_forward_none_metadata_fills_zero():
    out = torch.ones((3, 4, 64))
    ret = _impl().forward(_layer(), None, None, None, None, None, out)
    assert ret is out and out.count_nonzero() == 0

def test_forward_decoder_calls_kernel(monkeypatch):
    rec = _FlashRecorder()
    monkeypatch.setattr(attention, 'flash_attn_varlen_func', rec)
    out = torch.empty((3, 4, 64))
    cache = torch.zeros((2, 2, 16, 128))
    impl = _impl()
    impl.dcp_world_size = 1
    ret = impl.forward(_layer(), torch.zeros((3, 4, 64)), torch.zeros((3, 2, 64)), torch.zeros((3, 2, 64)), cache, _metadata(), out)
    assert ret is out
    kw = rec.calls[0]
    assert kw['num_splits'] == 0 and kw['fa_version'] == 2
    assert kw['out'].shape == out.shape

def test_forward_encoder_path(monkeypatch):
    impl = _impl(attn_type=AttentionType.ENCODER_ONLY)
    rec = _FlashRecorder()
    monkeypatch.setattr(attention, 'flash_attn_varlen_func', rec)
    out = torch.empty((3, 4, 64))
    ret = impl.forward(_layer(), torch.zeros((3, 4, 64)), torch.zeros((3, 2, 64)), torch.zeros((3, 2, 64)), None, _metadata(), out)
    assert ret.shape == out.shape and rec.calls[0]['causal'] is False

def test_forward_encoder_rejects_fp8():
    impl = _impl(attn_type=AttentionType.ENCODER_ONLY)
    impl.kv_cache_dtype = 'fp8'
    with pytest.raises(NotImplementedError, match='quantization is not supported'):
        impl._forward_encoder_attention(torch.zeros((3, 4, 64)), torch.zeros((3, 2, 64)), torch.zeros((3, 2, 64)), torch.zeros((3, 4, 64)), _metadata(), _layer())

def test_forward_dcp_path(monkeypatch):
    group = SimpleNamespace(all_gather=lambda q, dim: q)
    monkeypatch.setattr(attention, 'get_dcp_group', lambda : group)
    rec = _FlashRecorder()
    monkeypatch.setattr(attention, 'flash_attn_varlen_func', rec)
    monkeypatch.setattr(attention, 'cp_lse_ag_out_rs', lambda out, lse, g, return_lse: (out, lse))
    merged = []
    monkeypatch.setattr(attention, 'merge_attn_states', lambda *a: merged.append(a))
    impl = _impl()
    impl.dcp_world_size = 2
    md = _metadata(dcp_context_kv_lens=torch.tensor([2]), max_dcp_context_kv_len=4)
    out = torch.empty((3, 4, 64))
    ret = impl.forward(_layer(), torch.zeros((3, 4, 64)), torch.zeros((3, 2, 64)), torch.zeros((3, 2, 64)), torch.zeros((2, 2, 16, 128)), md, out)
    assert ret is out
    assert len(rec.calls) == 2 and len(merged) == 1

def _builder(monkeypatch, **attrs):
    monkeypatch.setattr(attention.AttentionMetadataBuilder, '__init__', lambda self, *a, **k: None)

    def _dcp_raises():
        raise AssertionError('dcp not init')
    monkeypatch.setattr('vllm.distributed.parallel_state.get_dcp_group', _dcp_raises)
    cfg = SimpleNamespace(model_config=SimpleNamespace(get_num_attention_heads=lambda pc: 4, get_num_kv_heads=lambda pc: 2, get_head_size=lambda : 64), parallel_config=SimpleNamespace(cp_kv_cache_interleave_size=1), cache_config=SimpleNamespace(), compilation_config=SimpleNamespace(cudagraph_mode=SimpleNamespace(has_full_cudagraphs=lambda : False), max_cudagraph_capture_size=0))
    spec = SimpleNamespace(dtype=torch.float16, block_size=16)
    builder = attention.AttentionFLMetadataBuilder(spec, ['layer'], cfg, torch.device('cpu'))
    builder.device = torch.device('cpu')
    builder.vllm_config = cfg
    for (key, value) in attrs.items():
        setattr(builder, key, value)
    return builder

def test_builder_init(monkeypatch):
    builder = _builder(monkeypatch)
    assert builder.num_heads_q == 4 and builder.num_heads_kv == 2
    assert builder.headdim == 64 and builder.block_size == 16
    assert builder.dcp_world_size == 1 and builder.dcp_rank == 0
    assert builder.max_num_splits == 0 and builder.aot_sliding_window is None

def _common(num_reqs=1, num_actual_tokens=3, causal=True):
    return SimpleNamespace(num_reqs=num_reqs, num_actual_tokens=num_actual_tokens, max_query_len=3, max_seq_len=5, query_start_loc=torch.tensor([0, 3], dtype=torch.int32), seq_lens=torch.tensor([5], dtype=torch.int32), block_table_tensor=torch.zeros((1, 2), dtype=torch.int32), slot_mapping=torch.zeros(3, dtype=torch.int64), causal=causal)

def test_builder_build_default(monkeypatch):
    builder = _builder(monkeypatch)
    md = builder.build(0, _common())
    assert md.num_actual_tokens == 3 and md.use_cascade is False
    assert md.scheduler_metadata is None and md.dcp_context_kv_lens is None

def test_builder_build_cudagraph_split(monkeypatch):
    builder = _builder(monkeypatch, use_full_cuda_graph=True, max_cudagraph_size=10, max_num_splits=7)
    md = builder.build(0, _common(num_actual_tokens=3))
    assert md.max_num_splits == 7

def test_builder_build_aot_sliding_window_branches(monkeypatch):
    builder = _builder(monkeypatch, aot_schedule=True)
    monkeypatch.setattr(attention, '_get_sliding_window_configs', lambda cfg: {(7, 0)})
    builder.aot_sliding_window = None
    builder.build(0, _common(fast_build=False) if False else _common())
    assert builder.aot_sliding_window == (7, 0)
    builder2 = _builder(monkeypatch, aot_schedule=True)
    monkeypatch.setattr(attention, '_get_sliding_window_configs', lambda cfg: {(7, 0), (-1, -1)})
    builder2.aot_sliding_window = None
    builder2.build(0, _common())
    assert builder2.aot_schedule is False
    builder3 = _builder(monkeypatch, aot_schedule=True)
    monkeypatch.setattr(attention, '_get_sliding_window_configs', lambda cfg: {None})
    builder3.aot_sliding_window = None
    builder3.build(0, _common())
    assert builder3.aot_sliding_window == (-1, -1)

def test_builder_use_cascade_attention_delegates(monkeypatch):
    monkeypatch.setattr(attention, 'use_cascade_attention', lambda *a, **k: 'sentinel')
    builder = _builder(monkeypatch)
    assert builder.use_cascade_attention() == 'sentinel'

def test_forward_with_dcp_merges(monkeypatch):
    group = SimpleNamespace(all_gather=lambda q, dim: q)
    monkeypatch.setattr(attention, 'get_dcp_group', lambda : group)
    rec = _FlashRecorder()
    monkeypatch.setattr(attention, 'flash_attn_varlen_func', rec)
    monkeypatch.setattr(attention, 'cp_lse_ag_out_rs', lambda out, lse, g, return_lse: (out, lse))
    merged = []
    monkeypatch.setattr(attention, 'merge_attn_states', lambda *a: merged.append(a))
    out = torch.empty((3, 4, 64))
    _impl()._forward_with_dcp(torch.zeros((3, 4, 64)), torch.zeros((3, 2, 64)), torch.zeros((3, 2, 64)), torch.zeros((2, 2, 16, 64)), torch.zeros((2, 2, 16, 64)), out, _metadata(dcp_context_kv_lens=torch.tensor([2]), max_dcp_context_kv_len=4))
    assert len(rec.calls) == 2 and merged

def test_cascade_attention_runs(monkeypatch):
    rec = _FlashRecorder()
    monkeypatch.setattr(attention, 'flash_attn_varlen_func', rec)
    merged = []
    monkeypatch.setattr(attention, 'merge_attn_states', lambda *a: merged.append(a))
    out = torch.empty((3, 4, 64))
    attention.cascade_attention(out, torch.zeros((3, 4, 64)), torch.zeros((2, 16, 2, 64)), torch.zeros((2, 16, 2, 64)), cu_query_lens=torch.tensor([0, 3]), max_query_len=3, cu_prefix_query_lens=torch.tensor([0, 3]), prefix_kv_lens=torch.tensor([32]), suffix_kv_lens=torch.tensor([5]), max_kv_len=37, softmax_scale=0.1, alibi_slopes=None, sliding_window=(-1, -1), logits_soft_cap=0.0, block_table=torch.zeros((1, 3), dtype=torch.int32), common_prefix_len=32, fa_version=2, q_descale=torch.ones(()), k_descale=torch.ones(()), v_descale=torch.ones(()))
    assert len(rec.calls) == 2 and merged

def test_cascade_attention_rejects_alibi_and_window():
    with pytest.raises(AssertionError, match='ALiBi'):
        attention.cascade_attention(torch.empty(1), torch.empty(1), torch.zeros((1, 16, 1, 8)), torch.zeros((1, 16, 1, 8)), torch.tensor([0, 1]), 1, torch.tensor([0, 1]), torch.tensor([16]), torch.tensor([1]), 17, 0.1, torch.ones(1), (-1, -1), 0.0, torch.zeros((1, 2)), 16, 2)
    with pytest.raises(AssertionError, match='sliding window'):
        attention.cascade_attention(torch.empty(1), torch.empty(1), torch.zeros((1, 16, 1, 8)), torch.zeros((1, 16, 1, 8)), torch.tensor([0, 1]), 1, torch.tensor([0, 1]), torch.tensor([16]), torch.tensor([1]), 17, 0.1, None, (3, 0), 0.0, torch.zeros((1, 2)), 16, 2)

def test_use_cascade_attention_decisions():
    import numpy as np
    fn = attention.use_cascade_attention
    assert fn(255, np.array([1] * 8), 8, 8, False, False, False, 100, 1) is False
    assert fn(512, np.array([1] * 8), 8, 8, True, False, False, 100, 1) is False
    assert fn(512, np.array([1] * 4), 8, 8, False, False, False, 100, 1) is False
    assert fn(512, np.array([1] * 8), 8, 8, False, False, False, 100, 2) is False
    assert fn(512, np.array([2] * 8), 8, 8, False, False, False, 100, 1) is True
    assert isinstance(fn(512, np.array([1] * 8), 64, 8, False, False, False, 100, 1), bool)
    assert isinstance(fn(512, np.array([1] * 8), 8, 8, False, False, False, 1, 1), bool)