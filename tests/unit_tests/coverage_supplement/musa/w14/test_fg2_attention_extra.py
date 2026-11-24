"""Extra coverage for flaggems/impl/attention.py branches left by w9b."""
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

    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        q = kwargs['q']
        if kwargs.get('return_softmax_lse'):
            return (torch.zeros_like(q), torch.zeros((q.shape[1], q.shape[0])))
        return kwargs.get('out', torch.zeros_like(q))

def _common(num_reqs=1, num_actual_tokens=3, seq_lens=None):
    query = torch.tensor([0, 3], dtype=torch.int32)
    if num_reqs != 1:
        query = torch.tensor([0, 2, 3], dtype=torch.int32)
    return SimpleNamespace(num_reqs=num_reqs, num_actual_tokens=num_actual_tokens, max_query_len=3, max_seq_len=5, query_start_loc=query, seq_lens=torch.tensor(seq_lens if seq_lens else [5], dtype=torch.int32), block_table_tensor=torch.zeros((num_reqs, 2), dtype=torch.int32), slot_mapping=torch.zeros(num_actual_tokens, dtype=torch.int64), causal=True)

def _cfg():
    return SimpleNamespace(model_config=SimpleNamespace(get_num_attention_heads=lambda pc: 4, get_num_kv_heads=lambda pc: 2, get_head_size=lambda : 64), parallel_config=SimpleNamespace(cp_kv_cache_interleave_size=2), cache_config=SimpleNamespace(), compilation_config=SimpleNamespace(cudagraph_mode=SimpleNamespace(has_full_cudagraphs=lambda : False), max_cudagraph_capture_size=0))

def _bare_builder(**attrs):
    builder = object.__new__(attention.AttentionFLMetadataBuilder)
    defaults = dict(aot_schedule=False, aot_sliding_window=None, use_full_cuda_graph=False, max_cudagraph_size=0, max_num_splits=0, dcp_world_size=1, dcp_rank=0, cp_kv_cache_interleave_size=1, device=torch.device('cpu'), vllm_config=None)
    defaults.update(attrs)
    for (key, value) in defaults.items():
        setattr(builder, key, value)
    return builder

def test_supports_attn_type_all_enum_members():
    backend = attention.AttentionFLBackend
    for attn_type in (AttentionType.DECODER, AttentionType.ENCODER, AttentionType.ENCODER_ONLY, AttentionType.ENCODER_DECODER):
        assert backend.supports_attn_type(attn_type) is True
    assert backend.supports_attn_type('nope') is False

def test_supports_head_size_boundaries():
    backend = attention.AttentionFLBackend
    assert backend.supports_head_size(8) is True
    assert backend.supports_head_size(128) is True
    assert backend.supports_head_size(256) is True
    assert backend.supports_head_size(7) is False
    assert backend.supports_head_size(264) is False

def test_supports_combination_ignores_sparse_without_sink_or_prefix():
    backend = attention.AttentionFLBackend
    assert backend.supports_combination(64, torch.float16, 'auto', 16, False, False, True, False, None) is None

def test_block_sizes_without_model_config(monkeypatch):
    monkeypatch.setattr(attention, 'get_current_vllm_config', lambda : SimpleNamespace(model_config=None, cache_config=SimpleNamespace(mamba_ssm_cache_dtype='auto', mamba_cache_dtype='auto')))
    sizes = attention.AttentionFLBackend.get_supported_kernel_block_sizes()
    assert len(sizes) == 1
    assert getattr(sizes[0], 'base', sizes[0]) == 16

def test_block_sizes_hybrid_with_float32_mamba_cache(monkeypatch):
    monkeypatch.setattr(attention, 'get_current_vllm_config', lambda : SimpleNamespace(model_config=SimpleNamespace(is_hybrid=True), cache_config=SimpleNamespace(mamba_ssm_cache_dtype='auto', mamba_cache_dtype='float32')))
    assert attention.AttentionFLBackend.get_supported_kernel_block_sizes() == [16, 32, 64]

def test_get_sliding_window_configs_walks_real_layers(monkeypatch):
    layers = {'a': SimpleNamespace(impl=_impl()), 'b': SimpleNamespace(impl=_impl(sliding_window=5))}
    monkeypatch.setattr(attention, 'get_layers_from_vllm_config', lambda *args: layers)
    assert attention._get_sliding_window_configs(object()) == {(-1, -1), (4, 0)}
    monkeypatch.setattr(attention, 'get_layers_from_vllm_config', lambda *args: {})
    assert attention._get_sliding_window_configs(object()) == set()

def test_builder_init_reads_live_dcp_group(monkeypatch):
    import vllm.distributed.parallel_state as parallel_state
    group = SimpleNamespace(world_size=2, rank_in_group=1)
    monkeypatch.setattr(parallel_state, 'get_dcp_group', lambda : group)
    monkeypatch.setattr(attention.AttentionMetadataBuilder, '__init__', lambda self, *a, **k: None)
    spec = SimpleNamespace(dtype=torch.float16, block_size=16)
    builder = attention.AttentionFLMetadataBuilder(spec, ['layer'], _cfg(), torch.device('cpu'))
    assert builder.dcp_world_size == 2
    assert builder.dcp_rank == 1
    assert builder.cp_kv_cache_interleave_size == 2
    assert builder.use_full_cuda_graph is False
    assert builder.aot_schedule is False
    assert builder.max_num_splits == 0

def test_build_keeps_preexisting_aot_window(monkeypatch):
    builder = _bare_builder(aot_schedule=True, aot_sliding_window=(3, 0))
    calls = []
    monkeypatch.setattr(attention, '_get_sliding_window_configs', lambda cfg: calls.append(cfg))
    metadata = builder.build(0, _common())
    assert calls == []
    assert builder.aot_sliding_window == (3, 0)
    assert metadata.use_cascade is False

def test_build_drops_splits_above_cudagraph_cap():
    builder = _bare_builder(use_full_cuda_graph=True, max_cudagraph_size=2, max_num_splits=7)
    metadata = builder.build(0, _common(num_actual_tokens=3))
    assert metadata.max_num_splits == 0

def test_build_cascade_multiple_requests():
    builder = _bare_builder()
    metadata = builder.build(4, _common(num_reqs=2, seq_lens=[10, 12]))
    assert metadata.use_cascade is True
    assert metadata.cu_prefix_query_lens.tolist() == [0, 3]
    assert metadata.prefix_kv_lens.tolist() == [4]
    assert metadata.suffix_kv_lens.tolist() == [6, 8]
    assert metadata.prefix_scheduler_metadata is None

def test_build_dcp_interleave_upper_bound(monkeypatch):
    builder = _bare_builder(dcp_world_size=2, dcp_rank=1, cp_kv_cache_interleave_size=2)
    monkeypatch.setattr(attention, 'get_dcp_local_seq_lens', lambda lens, world, rank, interleave: lens)
    metadata = builder.build(0, _common())
    assert metadata.max_dcp_context_kv_len == 4
    assert metadata.dcp_context_kv_lens.tolist() == [2]

def test_split_kv_cache_matches_transposed_layout():
    cache = torch.arange(2 * 2 * 16 * 128, dtype=torch.float32).reshape(2, 2, 16, 128)
    (key_cache, value_cache) = attention._split_kv_cache(cache, 64)
    assert key_cache.shape == (2, 16, 2, 64)
    assert value_cache.shape == (2, 16, 2, 64)
    torch.testing.assert_close(key_cache, cache.transpose(1, 2)[..., :64])
    torch.testing.assert_close(value_cache, cache.transpose(1, 2)[..., 64:])

def test_do_kv_cache_update_encoder_type_is_noop():
    impl = _impl(attn_type=AttentionType.ENCODER)
    assert impl.do_kv_cache_update(_layer(), torch.zeros((3, 2, 64)), torch.zeros((3, 2, 64)), torch.zeros((2, 2, 16, 128)), torch.zeros(3, dtype=torch.int64)) is None

def test_forward_rejects_output_block_scale():
    with pytest.raises(NotImplementedError, match='fused output quantization'):
        _impl().forward(_layer(), None, None, None, None, _metadata(), output=torch.zeros((3, 4, 64)), output_block_scale=torch.ones(1))

def test_cascade_attention_allows_none_descales(monkeypatch):
    recorder = _FlashRecorder()
    monkeypatch.setattr(attention, 'flash_attn_varlen_func', recorder)
    merged = []
    monkeypatch.setattr(attention, 'merge_attn_states', lambda *a: merged.append(a))
    attention.cascade_attention(torch.empty((3, 4, 64)), torch.zeros((3, 4, 64)), torch.zeros((2, 16, 2, 64)), torch.zeros((2, 16, 2, 64)), cu_query_lens=torch.tensor([0, 3]), max_query_len=3, cu_prefix_query_lens=torch.tensor([0, 3]), prefix_kv_lens=torch.tensor([32]), suffix_kv_lens=torch.tensor([5]), max_kv_len=37, softmax_scale=0.1, alibi_slopes=None, sliding_window=(-1, -1), logits_soft_cap=0.0, block_table=torch.zeros((1, 3), dtype=torch.int32), common_prefix_len=32, fa_version=2)
    assert len(recorder.calls) == 2
    assert recorder.calls[0]['q_descale'] is None
    assert recorder.calls[0]['k_descale'] is None
    assert recorder.calls[1]['v_descale'] is None
    assert merged

def test_use_cascade_attention_rejects_local_attention():
    import numpy as np
    fn = attention.use_cascade_attention
    assert fn(512, np.array([2] * 8), 8, 8, False, False, True, 100, 1) is False