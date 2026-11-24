"""CPU-only contracts for the DeepSeek-V4 Ascend FL adapter.

The real module imports vLLM model packages that are unavailable on MUSA.
We therefore exec only the pure functions/classes from the checked-in source
against stub base classes, exercising the genuine branch logic with torch CPU
tensors and object.__new__-built attention instances.  No weights are loaded.
"""
from __future__ import annotations
import ast
import os
import pathlib
import types
from types import SimpleNamespace
import pytest
import torch

def _find_source(rel: str) -> pathlib.Path:
    override = os.environ.get('VLLM_FL_SRC')
    candidates = []
    if override:
        candidates.append(pathlib.Path(override) / rel)
    candidates.append(pathlib.Path('/tmp/flsrc') / rel)
    here = pathlib.Path(__file__).resolve()
    for parent in here.parents:
        candidates.append(parent / rel)
    candidates.append(pathlib.Path.cwd() / rel)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f'could not locate {rel}; tried {candidates}')
_NAMES = {'_validate_short_context_attention', '_cache_rows', '_store_cache_rows', 'DeepseekV4FLMetadataBuilder', 'DeepseekV4FLSWABackend', 'DeepseekV4FLBackend', 'DeepseekV4FLAttention'}

@pytest.fixture(scope='module')
def dsv4():
    path = _find_source('vllm_fl/models/deepseek_v4.py')
    tree = ast.parse(path.read_text(encoding='utf-8'))
    future = next((node for node in tree.body if isinstance(node, ast.ImportFrom) and node.module == '__future__'))
    nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in _NAMES]

    class _StubMetadataBuilder:
        pass

    class _StubSWABackend:
        pass

    class _StubMLABackend:
        pass

    class _StubAttention:

        def __init__(self, *args, **kwargs):
            pass
    mod = types.ModuleType('w8b_deepseek_v4')
    mod.__dict__.update({'torch': torch, 'cast': __import__('typing').cast, 'get_forward_context': lambda : SimpleNamespace(attn_metadata=None), 'DeepseekSparseSWAMetadataBuilder': _StubMetadataBuilder, 'DeepseekSparseSWABackend': _StubSWABackend, 'DeepseekV4SparseMLABackend': _StubMLABackend, 'DeepseekV4SparseMLAMetadataBuilder': _StubMetadataBuilder, 'DeepseekV4FlashMLAMetadata': object, 'DeepseekSparseSWAMetadata': object, 'DeepseekV4Attention': _StubAttention})
    exec(compile(ast.Module(body=[future, *nodes], type_ignores=[]), str(path), 'exec'), mod.__dict__)
    yield mod

def _cfg(*, ratios=(0, 1, 4, 128), layers=4, sliding=128, max_len=128, topk=512, compressor=None):
    hf = SimpleNamespace(compress_ratios=list(ratios), num_hidden_layers=layers, sliding_window=sliding, index_topk=topk)
    if compressor is not None:
        hf.compressor = compressor
    return SimpleNamespace(model_config=SimpleNamespace(hf_config=hf, max_model_len=max_len))

def _build(cls, **attrs):
    obj = object.__new__(cls)
    for (name, value) in attrs.items():
        setattr(obj, name, value)
    return obj

def test_valid_short_context_is_accepted(dsv4):
    assert dsv4._validate_short_context_attention(_cfg()) is None
    assert dsv4._validate_short_context_attention(_cfg(ratios=(0, 4, 128, 4, 8), layers=4)) is None

def test_unsupported_compress_ratio_reports_layer(dsv4):
    with pytest.raises(NotImplementedError) as excinfo:
        dsv4._validate_short_context_attention(_cfg(ratios=(0, 1, 8, 4), layers=4))
    assert 'layer 2 compress_ratio=8' in str(excinfo.value)

def test_context_over_bound_is_rejected(dsv4):
    with pytest.raises(ValueError) as excinfo:
        dsv4._validate_short_context_attention(_cfg(max_len=256))
    message = str(excinfo.value)
    assert '--max-model-len 128' in message
    assert 'sliding_window' in message

def test_sliding_window_smaller_than_128_sets_bound(dsv4):
    with pytest.raises(ValueError) as excinfo:
        dsv4._validate_short_context_attention(_cfg(sliding=32, max_len=64))
    assert '--max-model-len 32' in str(excinfo.value)

def test_c4a_candidate_overflow_is_rejected(dsv4):
    with pytest.raises(NotImplementedError) as excinfo:
        dsv4._validate_short_context_attention(_cfg(max_len=64, topk=8))
    assert 'all compressed candidates' in str(excinfo.value)

def test_c4a_candidate_fit_is_accepted(dsv4):
    assert dsv4._validate_short_context_attention(_cfg(max_len=64, topk=16)) is None

def test_cache_rows_reads_logical_slots(dsv4):
    cache = torch.arange(2 * 2 * 3, dtype=torch.float32).reshape(2, 2, 3)
    slots = torch.tensor([0, 3, 2, 1])
    out = dsv4._cache_rows(cache, slots, 2)
    torch.testing.assert_close(out, cache[slots // 2, slots % 2])
    assert out.shape == (4, 3)

@pytest.mark.parametrize('shape,block', [((2, 3, 4), 2), ((2, 2), 2), ((4, 2, 3), 4)])
def test_cache_rows_rejects_bad_layout(dsv4, shape, block):
    with pytest.raises(ValueError) as excinfo:
        dsv4._cache_rows(torch.zeros(shape), torch.tensor([0]), block)
    assert 'Unexpected DeepSeek-V4 KV cache shape' in str(excinfo.value)

def test_store_rows_writes_cast_values(dsv4):
    cache = torch.zeros(2, 2, 2, dtype=torch.float16)
    rows = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.float64)
    slots = torch.tensor([0, 2, 3])
    dsv4._store_cache_rows(cache, slots, rows, 2)
    assert cache.dtype == torch.float16
    torch.testing.assert_close(cache[0, 0], rows[0].to(torch.float16))
    torch.testing.assert_close(cache[1, 0], rows[1].to(torch.float16))
    torch.testing.assert_close(cache[1, 1], rows[2].to(torch.float16))

@pytest.mark.parametrize('shape,block', [((2, 2), 2), ((1, 3, 2), 2)])
def test_store_rows_rejects_bad_layout(dsv4, shape, block):
    with pytest.raises(ValueError):
        dsv4._store_cache_rows(torch.zeros(shape), torch.tensor([0]), torch.zeros(1, 2), block)

def test_metadata_builder_and_backend_identity(dsv4):
    builder = dsv4.DeepseekV4FLMetadataBuilder()
    assert builder.build_tile_scheduler(7) == {'swaonly': None, 'c4a': None, 'c128a': None}
    assert dsv4.DeepseekV4FLSWABackend.get_name() == 'FL_ASCEND_V4_BF16_SWA_CACHE'
    assert dsv4.DeepseekV4FLSWABackend.get_builder_cls() is dsv4.DeepseekV4FLMetadataBuilder
    assert dsv4.DeepseekV4FLBackend.get_name() == 'FL_ASCEND_V4_BF16_SWA'
    assert dsv4.DeepseekV4FLBackend.supported_kv_cache_dtypes == ['auto']
    assert dsv4.DeepseekV4FLBackend.supports_compute_capability(99) is True
    assert dsv4.DeepseekV4FLAttention.get_padded_num_q_heads(12) == 12

def test_parallel_projections_without_compressor(dsv4):
    attn = _build(dsv4.DeepseekV4FLAttention)
    attn._fused_wqa_wkv_gemm = lambda hidden: torch.zeros(hidden.shape[0], 6)
    produced = torch.randn(3, 4)
    attn.compressor = None
    (qr_kv, kv_score, iks, iw) = attn._run_parallel_input_projections(produced)
    assert qr_kv.shape == (3, 6)
    assert kv_score is None and iks is None and (iw is None)

def test_parallel_projections_with_compressor_keeps_fp32(dsv4):
    weight = torch.randn(8, 4)
    compressor = SimpleNamespace(fused_wkv_wgate=SimpleNamespace(weight=weight))
    attn = _build(dsv4.DeepseekV4FLAttention, compressor=compressor)
    hidden = torch.randn(3, 4, dtype=torch.float32)
    attn._fused_wqa_wkv_gemm = lambda hs: torch.zeros(hs.shape[0], 6)
    (_, kv_score, iks, iw) = attn._run_parallel_input_projections(hidden)
    assert kv_score.shape == (3, 8)
    assert kv_score.dtype == torch.float32
    torch.testing.assert_close(kv_score, torch.mm(hidden.float(), weight.float().T))
    assert iks is None and iw is None

def test_qnorm_rope_insert_non_dict_returns_q(dsv4):
    attn = _build(dsv4.DeepseekV4FLAttention)
    q = torch.randn(2, 2, 4)
    assert attn._fused_qnorm_rope_kv_insert(q, None, None, object()) is q

def _swa_attn(dsv4, slots, block_size=2):
    cache = torch.zeros(2, block_size, 4)
    rotary = SimpleNamespace(forward_native=lambda positions, q, kv, inverse=False: (q, kv))
    layer = SimpleNamespace(prefix='swa', kv_cache=cache)
    meta = SimpleNamespace(slot_mapping=slots, block_size=block_size)
    attn = _build(dsv4.DeepseekV4FLAttention, swa_cache_layer=layer, rotary_emb=rotary, eps=1e-06)
    return (attn, cache, {'swa': meta})

def test_qnorm_rope_insert_writes_valid_slots(dsv4):
    slots = torch.tensor([0, 2, -1])
    (attn, cache, metadata) = _swa_attn(dsv4, slots)
    q = torch.randn(3, 2, 4)
    kv = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 9.0, 9.0, 9.0]])
    out = attn._fused_qnorm_rope_kv_insert(q, kv, torch.tensor([0, 1, 2]), metadata)
    torch.testing.assert_close(cache[0, 0], kv[0])
    torch.testing.assert_close(cache[1, 0], kv[1])
    assert torch.count_nonzero(cache[0, 1]) == 0
    torch.testing.assert_close(out.pow(2).mean(-1), torch.full((3, 2), 1.0), atol=0.001, rtol=0.001)

def test_qnorm_rope_insert_all_invalid_leaves_cache(dsv4):
    slots = torch.tensor([-1, -1])
    (attn, cache, metadata) = _swa_attn(dsv4, slots)
    out = attn._fused_qnorm_rope_kv_insert(torch.randn(2, 2, 4), torch.randn(2, 4), torch.tensor([0, 1]), metadata)
    assert out.shape == (2, 2, 4)
    assert torch.count_nonzero(cache) == 0

def test_compress_kv_bf16_non_dict_metadata_returns(dsv4, monkeypatch):
    monkeypatch.setattr(dsv4, 'get_forward_context', lambda : SimpleNamespace(attn_metadata=None))
    attn = _build(dsv4.DeepseekV4FLAttention, compressor=SimpleNamespace())
    attn._compress_kv_bf16(torch.randn(2, 8), torch.tensor([0, 1]))

def test_prepare_and_attn_without_compressor(dsv4, monkeypatch):
    attn = _build(dsv4.DeepseekV4FLAttention, n_local_heads=2, head_dim=4, compress_ratio=1, compressor=None, wq_b=lambda x: torch.zeros(x.shape[0], 8))
    captured = {}

    def fake_insert(q, kv, positions, metadata):
        captured['q'] = q
        captured['metadata'] = metadata
        return q

    def fake_forward_mqa(q, kv, positions, out):
        captured['mqa'] = q
    attn._fused_qnorm_rope_kv_insert = fake_insert
    attn.forward_mqa = fake_forward_mqa
    sentinel = object()
    monkeypatch.setattr(dsv4, 'get_forward_context', lambda : SimpleNamespace(attn_metadata=sentinel))
    out = torch.empty(3, 2, 4)
    attn._prepare_and_attn(torch.randn(3, 4), torch.randn(3, 2), torch.randn(3, 4), None, None, None, torch.tensor([0, 1, 2]), out)
    assert captured['q'].shape == (3, 2, 4)
    assert captured['metadata'] is sentinel
    assert captured['mqa'].shape == (3, 2, 4)

def test_prepare_and_attn_c4a_fills_topk_buffer(dsv4, monkeypatch):
    attn = _build(dsv4.DeepseekV4FLAttention, n_local_heads=1, head_dim=2, compress_ratio=4, compressor=None, indexer=SimpleNamespace(topk_tokens=4), topk_indices_buffer=torch.full((3, 4), 99, dtype=torch.int32), wq_b=lambda x: torch.zeros(x.shape[0], 2))
    attn._fused_qnorm_rope_kv_insert = lambda q, kv, positions, meta: q
    attn.forward_mqa = lambda q, kv, positions, out: None
    monkeypatch.setattr(dsv4, 'get_forward_context', lambda : SimpleNamespace(attn_metadata={}))
    attn._prepare_and_attn(torch.randn(3, 4), torch.randn(3, 2), torch.randn(3, 2), None, None, None, torch.tensor([0, 3, 7]), torch.empty(3, 1, 2))
    buffer = attn.topk_indices_buffer
    assert buffer[0].tolist() == [-1, -1, -1, -1]
    assert buffer[1].tolist() == [0, -1, -1, -1]
    assert buffer[2].tolist() == [0, 1, -1, -1]

def test_o_proj_requires_single_group(dsv4):
    attn = _build(dsv4.DeepseekV4FLAttention, n_local_groups=2, rotary_emb=SimpleNamespace(forward_native=lambda positions, o, none, inverse=False: (o, None)))
    with pytest.raises(ValueError) as excinfo:
        attn._o_proj(torch.randn(2, 3), torch.tensor([0, 1]))
    assert 'one group per TP rank' in str(excinfo.value)

def test_o_proj_applies_wo_a_then_wo_b(dsv4):
    attn = _build(dsv4.DeepseekV4FLAttention, n_local_groups=1, wo_a=lambda x: x * 2, wo_b=lambda x: x - 1, rotary_emb=SimpleNamespace(forward_native=lambda positions, o, none, inverse=False: (o, None)))
    o = torch.arange(6, dtype=torch.float32).reshape(2, 3, 1).squeeze(-1)
    out = attn._o_proj(o, torch.tensor([0, 1]))
    torch.testing.assert_close(out, o.flatten(1) * 2 - 1)

def test_forward_mqa_non_dict_zeroes_output(dsv4, monkeypatch):
    monkeypatch.setattr(dsv4, 'get_forward_context', lambda : SimpleNamespace(attn_metadata=[]))
    attn = _build(dsv4.DeepseekV4FLAttention)
    out = torch.ones(2, 2, 3)
    attn.forward_mqa(torch.randn(2, 2, 4), None, torch.tensor([0, 1]), out)
    assert torch.count_nonzero(out) == 0

def _mqa_attn(dsv4, *, compress_ratio=1, num_tokens=3):
    (num_heads, head_dim, window) = (2, 4, 3)
    swa_cache = torch.randn(3, 2, head_dim)
    swa_meta = SimpleNamespace(token_to_req_indices=torch.zeros(num_tokens, dtype=torch.long), block_size=2, block_table=torch.tensor([[0, 1, 2]]))
    layer = SimpleNamespace(prefix='swa', kv_cache=swa_cache)
    metadata = {'swa': swa_meta}
    attrs = dict(swa_cache_layer=layer, window_size=window, n_local_heads=num_heads, head_dim=head_dim, scale=head_dim ** (-0.5), attn_sink=torch.zeros(num_heads), compress_ratio=compress_ratio, max_model_len=16, prefix='comp')
    if compress_ratio > 1:
        attrs['kv_cache'] = torch.randn(2, 1, head_dim)
        attrs['topk_indices_buffer'] = torch.tensor([[0, 1, -1, -1]] * num_tokens)
        metadata['comp'] = SimpleNamespace(block_size=4, block_table=torch.tensor([[0, 1]]))
    return (_build(dsv4.DeepseekV4FLAttention, **attrs), metadata)

def test_forward_mqa_swa_only_is_finite_and_uses_cache(dsv4, monkeypatch):
    (attn, metadata) = _mqa_attn(dsv4, compress_ratio=1)
    monkeypatch.setattr(dsv4, 'get_forward_context', lambda : SimpleNamespace(attn_metadata=metadata))
    q = torch.randn(3, 2, 4)
    out = torch.full((3, 2, 4), -5.0)
    attn.forward_mqa(q, None, torch.tensor([0, 1, 2]), out)
    assert out.shape == (3, 2, 4)
    assert torch.isfinite(out).all()
    changed = metadata['swa'].__class__(token_to_req_indices=torch.zeros(3, dtype=torch.long), block_size=2, block_table=torch.tensor([[0, 1, 2]]))
    attn.swa_cache_layer.kv_cache = torch.randn(3, 2, 4)
    metadata['swa'] = changed
    out2 = torch.full((3, 2, 4), -5.0)
    attn.forward_mqa(q, None, torch.tensor([0, 1, 2]), out2)
    assert torch.isfinite(out2).all()
    assert not torch.allclose(out, out2)

def test_forward_mqa_all_blocks_invalid_zeroes_output(dsv4, monkeypatch):
    (attn, metadata) = _mqa_attn(dsv4, compress_ratio=1)
    metadata['swa'].block_table = torch.tensor([[-1, -1, -1]])
    monkeypatch.setattr(dsv4, 'get_forward_context', lambda : SimpleNamespace(attn_metadata=metadata))
    out = torch.ones(3, 2, 4)
    attn.forward_mqa(torch.randn(3, 2, 4), None, torch.tensor([0, 1, 2]), out)
    assert torch.count_nonzero(out) == 0

def test_forward_mqa_c4a_concatenates_compressed_history(dsv4, monkeypatch):
    (attn, metadata) = _mqa_attn(dsv4, compress_ratio=4)
    monkeypatch.setattr(dsv4, 'get_forward_context', lambda : SimpleNamespace(attn_metadata=metadata))
    out = torch.full((3, 2, 4), -1.0)
    attn.forward_mqa(torch.randn(3, 2, 4), None, torch.tensor([0, 1, 2]), out)
    assert torch.isfinite(out).all()
    assert out.shape == (3, 2, 4)
    assert not torch.allclose(out, torch.full_like(out, -1.0))