"""Coverage for vllm_fl.worker.model_runner input-build helpers (w3b, local CPU)."""
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from vllm_fl.worker import model_runner as mr

def make_runner(**attrs):
    runner = object.__new__(mr.ModelRunnerFL)
    runner.__dict__.update(attrs)
    return runner

def fake_buf(size, dtype, value=0):
    np_arr = np.full(size, value, dtype=dtype)
    gpu = torch.full((size,), value, dtype=torch.int32)

    def copy_to_gpu(n=None):
        n = size if n is None else n
        gpu[:n] = torch.as_tensor(np_arr[:n], dtype=torch.int32)
    return SimpleNamespace(np=np_arr, gpu=gpu, copy_to_gpu=copy_to_gpu)

@pytest.fixture(autouse=True)
def _cpu_pinned(monkeypatch):
    monkeypatch.setattr(mr, 'PIN_MEMORY', False, raising=False)

def test_get_positions_int_plain_and_mrope():
    positions = torch.arange(6)
    r = make_runner(uses_mrope=False, positions=positions)
    assert torch.equal(r._get_positions(3), positions[:3])
    mrope = SimpleNamespace(gpu=torch.arange(12).reshape(3, 4))
    r2 = make_runner(uses_mrope=True, mrope_positions=mrope)
    assert torch.equal(r2._get_positions(2), mrope.gpu[:, :2])

def test_get_cumsum_and_arange():
    r = make_runner(arange_np=np.arange(16))
    arange_out = np.empty(16, dtype=np.int64)
    cu = r._get_cumsum_and_arange(np.array([2, 5, 3]), arange_out, cumsum_dtype=np.int64)
    assert cu.tolist() == [2, 7, 10]
    assert arange_out[:10].tolist() == [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]

def test_compute_prev_positions_without_mapping():
    prev = fake_buf(4, np.int64, 99)
    inp = SimpleNamespace(prev_req_id_to_index=None, req_ids=['a', 'b'])
    r = make_runner(input_batch=inp, prev_positions=prev)
    r._compute_prev_positions(2)
    assert prev.np[:2].tolist() == [-1, -1]

def test_compute_prev_positions_maps_req_ids():
    prev = fake_buf(4, np.int64, 99)
    inp = SimpleNamespace(prev_req_id_to_index={'a': 3}, req_ids=['a', 'b'])
    r = make_runner(input_batch=inp, prev_positions=prev)
    r._compute_prev_positions(2)
    assert prev.np[:2].tolist() == [3, -1]

def test_get_encoder_seq_lens_non_cross_returns_none():
    r = make_runner()
    assert r._get_encoder_seq_lens({}, object(), 1) == (None, None)

def test_get_encoder_seq_lens_builds_and_ignores_plain_feature(monkeypatch):
    spec = object.__new__(mr.CrossAttentionSpec)
    lens = fake_buf(4, np.int32)
    mm_feat = SimpleNamespace(mm_position=SimpleNamespace(length=5))
    reqs = {'a': SimpleNamespace(mm_features=[mm_feat]), 'b': SimpleNamespace(mm_features=None)}
    inp = SimpleNamespace(req_id_to_index={'a': 0, 'b': 1})
    r = make_runner(encoder_seq_lens=lens, requests=reqs, input_batch=inp, max_encoder_len=99)
    (gpu, cpu) = r._get_encoder_seq_lens({'a': 1, 'b': 1}, spec, 2)
    assert cpu.tolist()[:2] == [5, 0]
    assert gpu.tolist()[:2] == [5, 0]

def test_get_encoder_seq_lens_cudagraph_capture_uses_max_len():
    spec = object.__new__(mr.CrossAttentionSpec)
    lens = fake_buf(4, np.int32)
    inp = SimpleNamespace(req_id_to_index={})
    r = make_runner(encoder_seq_lens=lens, requests={}, input_batch=inp, max_encoder_len=128, model_config=SimpleNamespace(hf_config=SimpleNamespace()))
    (_, cpu) = r._get_encoder_seq_lens({}, spec, 3, for_cudagraph_capture=True)
    assert cpu.tolist() == [128, 128, 128]

class _AS:

    def __init__(self, block_size=16, num_kv_heads=2):
        self.block_size = block_size
        self.num_kv_heads = num_kv_heads

class _SW(_AS):
    pass

class _CLA(_AS):
    pass

class _FA(_AS):

    def __init__(self, block_size=16, sliding_window=None, attention_chunk_size=None):
        super().__init__(block_size)
        self.sliding_window = sliding_window
        self.attention_chunk_size = attention_chunk_size

class _EO(_AS):
    pass

@pytest.fixture
def cascade_classes(monkeypatch):
    monkeypatch.setattr(mr, 'AttentionSpec', _AS, raising=False)
    monkeypatch.setattr(mr, 'SlidingWindowSpec', _SW, raising=False)
    monkeypatch.setattr(mr, 'ChunkedLocalAttentionSpec', _CLA, raising=False)
    monkeypatch.setattr(mr, 'FullAttentionSpec', _FA, raising=False)
    monkeypatch.setattr(mr, 'EncoderOnlyAttentionSpec', _EO, raising=False)

def test_cascade_prefix_len_zero_common_blocks(cascade_classes):
    r = make_runner(num_query_heads=4, use_alibi=False, num_sms=1, dcp_world_size=1)
    assert r._compute_cascade_attn_prefix_len(np.array([1]), np.array([0]), 0, _AS(), SimpleNamespace()) == 0

def test_cascade_prefix_len_uses_builder_decision(cascade_classes):
    builder = SimpleNamespace(use_cascade_attention=lambda **kw: True)
    r = make_runner(num_query_heads=4, use_alibi=False, num_sms=8, dcp_world_size=2)
    got = r._compute_cascade_attn_prefix_len(np.array([2, 2]), np.array([40, 33]), 3, _AS(block_size=16), builder)
    assert got == 32

def test_cascade_prefix_len_disabled_by_builder(cascade_classes):
    seen = {}

    def use(**kw):
        seen.update(kw)
        return False
    builder = SimpleNamespace(use_cascade_attention=use)
    r = make_runner(num_query_heads=4, use_alibi=True, num_sms=8, dcp_world_size=1)
    got = r._compute_cascade_attn_prefix_len(np.array([2]), np.array([32]), 2, _FA(sliding_window=64), builder)
    assert got == 0
    assert seen['use_sliding_window'] is True
    assert seen['use_local_attention'] is False

def test_cascade_prefix_lens_none_when_no_cascade(cascade_classes, monkeypatch):
    groups = [SimpleNamespace(kv_cache_spec=_AS(), get_metadata_builder=lambda : None)]
    r = make_runner(attn_groups=[groups], kv_cache_config=SimpleNamespace(kv_cache_groups=[1]))
    monkeypatch.setattr(r, '_compute_cascade_attn_prefix_len', lambda *a: 0)
    assert r._compute_cascade_attn_prefix_lens(np.array([1]), np.array([0]), [0]) is None

def test_cascade_prefix_lens_encoder_only_and_cascade(cascade_classes, monkeypatch):
    encoder_only = _EO()
    groups = [[SimpleNamespace(kv_cache_spec=encoder_only, get_metadata_builder=lambda : None)], [SimpleNamespace(kv_cache_spec=_AS(), get_metadata_builder=lambda : None)]]
    r = make_runner(attn_groups=groups, kv_cache_config=SimpleNamespace(kv_cache_groups=[1, 1]))
    monkeypatch.setattr(r, '_compute_cascade_attn_prefix_len', lambda *a: 7)
    got = r._compute_cascade_attn_prefix_lens(np.array([1]), np.array([0]), [0, 1])
    assert got == [[0], [7]]

def test_calc_spec_decode_metadata():
    monkeypatch_async = lambda arr, device=None: torch.as_tensor(np.asarray(arr))
    orig = mr.async_tensor_h2d
    mr.async_tensor_h2d = monkeypatch_async
    try:
        r = make_runner(arange_np=np.arange(64), _arange_scratch=np.empty(64, dtype=np.int32), input_ids=SimpleNamespace(gpu=torch.arange(20, dtype=torch.int32)), device=torch.device('cpu'))
        meta = r._calc_spec_decode_metadata(np.array([1, 0, 2], dtype=np.int32), np.array([2, 5, 8], dtype=np.int32))
    finally:
        mr.async_tensor_h2d = orig
    assert meta.num_draft_tokens == [1, 0, 2]
    assert meta.cu_num_draft_tokens.tolist() == [1, 1, 3]
    assert meta.cu_num_sampled_tokens.tolist() == [2, 3, 6]
    assert meta.logits_indices.tolist() == [0, 1, 4, 5, 6, 7]

def test_batch_mm_inputs_empty_scheduled():
    r = make_runner()
    got = r._batch_mm_inputs_from_scheduler(SimpleNamespace(scheduled_encoder_inputs={}))
    assert got == ([], [], [])

def test_batch_mm_inputs_skips_none_data():
    pos = SimpleNamespace()
    good = SimpleNamespace(identifier='h1', modality='image', data={'x': 1}, mm_position=pos)
    bad = SimpleNamespace(identifier='h2', modality='image', data=None, mm_position=pos)
    r = make_runner(requests={'r': SimpleNamespace(mm_features=[good, bad])})
    (hashes, kwargs, refs) = r._batch_mm_inputs_from_scheduler(SimpleNamespace(scheduled_encoder_inputs={'r': [0, 1]}))
    assert hashes == ['h1']
    assert kwargs == [('image', {'x': 1})]
    assert refs == [('r', pos)]

def test_extract_mm_kwargs_early_returns():
    r = make_runner(is_multimodal_raw_input_only_model=False)
    assert r._extract_mm_kwargs(None) == {}
    assert r._extract_mm_kwargs(SimpleNamespace()) == {}

def test_extract_mm_kwargs_combines_batches(monkeypatch):
    feat = SimpleNamespace(modality='image', data={'a': 1})
    req = SimpleNamespace(mm_features=[feat])
    monkeypatch.setattr(mr, 'group_and_batch_mm_kwargs', lambda items, device, pin_memory: iter([('image', 1, {'k': 'v'})]))
    r = make_runner(is_multimodal_raw_input_only_model=True, device=torch.device('cpu'))
    out = r._extract_mm_kwargs(SimpleNamespace(scheduled_new_reqs=[req]))
    assert out == {'k': 'v'}

def test_dummy_mm_kwargs_early_returns():
    r = make_runner(is_multimodal_raw_input_only_model=False)
    assert r._dummy_mm_kwargs(2) == {}

def test_dummy_mm_kwargs_no_tower_modalities():
    budget = SimpleNamespace(mm_max_toks_per_item=0)
    r = make_runner(is_multimodal_raw_input_only_model=True, mm_budget=budget)
    assert r._dummy_mm_kwargs(2) == {}

def test_dummy_mm_kwargs_uses_max_modality():
    budget = SimpleNamespace(mm_max_toks_per_item=10, get_modality_with_max_tokens=lambda : 'video')
    r = make_runner(is_multimodal_raw_input_only_model=True, mm_budget=budget)
    r._get_mm_dummy_batch = lambda modality, n: {'mod': modality, 'n': n}
    assert r._dummy_mm_kwargs(3) == {'mod': 'video', 'n': 3}

def test_gather_mm_embeddings_empty():
    r = make_runner(input_batch=SimpleNamespace(req_ids=[]))
    (embeds, mask) = r._gather_mm_embeddings(SimpleNamespace(total_num_scheduled_tokens=0))
    assert embeds == []
    assert mask.numel() == 0

def _gather_setup(monkeypatch):
    monkeypatch.setattr(mr, 'get_mm_features_in_window', lambda *a, **k: (0, 1))
    monkeypatch.setattr(mr, 'set_mm_embedding_modality', lambda *a, **k: None)

def test_gather_mm_embeddings_encoder_cache_miss_raises(monkeypatch):
    _gather_setup(monkeypatch)
    pos = SimpleNamespace(offset=0, length=3, is_embed=None, get_embeds_indices_in_range=lambda a, b: (a, b))
    feat = SimpleNamespace(identifier='h', modality='image', mm_position=pos)
    req_state = SimpleNamespace(num_computed_tokens=0, mm_features=[feat])
    r = make_runner(input_batch=SimpleNamespace(req_ids=['r']), requests={'r': req_state}, encoder_cache={}, is_multimodal_pruning_enabled=False)
    sched = SimpleNamespace(total_num_scheduled_tokens=3, num_scheduled_tokens={'r': 3})
    with pytest.raises(RuntimeError, match='Encoder cache miss'):
        r._gather_mm_embeddings(sched)

def test_gather_mm_embeddings_fills_mask(monkeypatch):
    _gather_setup(monkeypatch)
    pos = SimpleNamespace(offset=1, length=3, is_embed=None, get_embeds_indices_in_range=lambda a, b: (a, b))
    feat = SimpleNamespace(identifier='h', modality='image', mm_position=pos)
    req_state = SimpleNamespace(num_computed_tokens=0, mm_features=[feat])
    r = make_runner(input_batch=SimpleNamespace(req_ids=['r']), requests={'r': req_state}, encoder_cache={'h': torch.arange(6).reshape(3, 2)}, is_multimodal_pruning_enabled=False)
    sched = SimpleNamespace(total_num_scheduled_tokens=3, num_scheduled_tokens={'r': 3})
    (embeds, mask) = r._gather_mm_embeddings(sched)
    assert len(embeds) == 1
    assert embeds[0].shape == (2, 2)
    assert mask.tolist() == [False, True, True]

def test_prepare_kv_sharing_fast_prefill_pads_tail():
    buf = torch.zeros(6, dtype=torch.int64)
    dispatcher = SimpleNamespace(dispatch=lambda n, invalid_modes=None: (None, SimpleNamespace(num_tokens=6)))
    r = make_runner(kv_sharing_fast_prefill_logits_indices=buf, cudagraph_dispatcher=dispatcher)
    out = r._prepare_kv_sharing_fast_prefill(torch.tensor([3, 4, 5]))
    assert out.tolist() == [3, 4, 5, 5, 5, 5]

def test_build_attention_metadata_attention_free():
    r = make_runner(kv_cache_config=SimpleNamespace(kv_cache_groups=[]))
    assert r._build_attention_metadata(num_tokens=1, num_reqs=1, max_query_len=1) == ({}, None)