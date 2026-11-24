"""Coverage-focused unit tests for ``vllm_fl.models.deepseek_v4``.

Self-contained: only pytest + unittest.mock + stdlib are required at import
time; torch is required (skipped otherwise) because the module is a torch
adapter.  No weights are loaded and no real kernels/backends are executed.

When the real ``vllm`` package (or the 0.28 ``deepseek_v4`` submodules) is not
importable in the current interpreter, minimal stand-in modules are injected
into ``sys.modules`` so the target module can still be imported and its pure
Python control flow exercised.  Real modules are preferred whenever available.
"""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

torch = pytest.importorskip("torch")

TARGET = "vllm_fl.models.deepseek_v4"


# ---------------------------------------------------------------------------
# Minimal stand-ins for the vLLM symbols imported by the target module.
# ---------------------------------------------------------------------------


def _fake_get_forward_context():
    return SimpleNamespace(attn_metadata=None)


class _StandIn:
    """Generic permissive base class used for stubbed vLLM parents."""

    def __init__(self, *args, **kwargs):
        pass


_REQUIRED_MODULES = {
    "vllm": {},
    "vllm.forward_context": {"get_forward_context": _fake_get_forward_context},
    "vllm.models": {},
    "vllm.models.deepseek_v4": {},
    "vllm.models.deepseek_v4.attention": {
        "DeepseekV4Attention": _StandIn,
    },
    "vllm.models.deepseek_v4.sparse_mla": {
        "DeepseekV4FlashMLAMetadata": _StandIn,
        "DeepseekV4SparseMLABackend": _StandIn,
        "DeepseekV4SparseMLAMetadataBuilder": _StandIn,
    },
    "vllm.models.deepseek_v4.xpu": {},
    "vllm.models.deepseek_v4.xpu.model": {
        "DeepseekV4ForCausalLM": _StandIn,
    },
    "vllm.platforms": {
        "current_platform": SimpleNamespace(device_type="cuda"),
    },
    "vllm.v1": {},
    "vllm.v1.attention": {},
    "vllm.v1.attention.backends": {},
    "vllm.v1.attention.backends.mla": {},
    "vllm.v1.attention.backends.mla.sparse_swa": {
        "DeepseekSparseSWABackend": _StandIn,
        "DeepseekSparseSWAMetadata": _StandIn,
        "DeepseekSparseSWAMetadataBuilder": _StandIn,
    },
}

_PACKAGE_NAMES = {
    "vllm",
    "vllm.models",
    "vllm.models.deepseek_v4",
    "vllm.models.deepseek_v4.xpu",
    "vllm.v1",
    "vllm.v1.attention",
    "vllm.v1.attention.backends",
    "vllm.v1.attention.backends.mla",
}


def _ensure_module(name, attrs):
    """Create ``name`` (and any missing parents) and set ``attrs`` on it."""
    parts = name.split(".")
    for i in range(1, len(parts)):
        parent_name = ".".join(parts[:i])
        if parent_name not in sys.modules:
            parent = types.ModuleType(parent_name)
            parent.__path__ = []
            sys.modules[parent_name] = parent
            grand = sys.modules.get(".".join(parts[: i - 1]))
            if grand is not None:
                setattr(grand, parts[i - 1], parent)

    leaf = sys.modules.get(name)
    if leaf is None:
        leaf = types.ModuleType(name)
        if name in _PACKAGE_NAMES:
            leaf.__path__ = []
        sys.modules[name] = leaf
        parent = sys.modules.get(".".join(parts[:-1]))
        if parent is not None:
            setattr(parent, parts[-1], leaf)

    for key, value in attrs.items():
        setattr(leaf, key, value)
    return leaf


def _prepare_dependencies():
    """Prefer real vLLM modules; fall back to stand-ins only when needed."""
    for name, attrs in _REQUIRED_MODULES.items():
        module = sys.modules.get(name)
        if module is None:
            try:
                module = importlib.import_module(name)
            except Exception:
                module = None
        if module is None:
            _ensure_module(name, attrs)
        else:
            for key, value in attrs.items():
                if not hasattr(module, key):
                    setattr(module, key, value)


_prepare_dependencies()

try:
    mod = importlib.import_module(TARGET)
except Exception as exc:  # pragma: no cover - environment without the package
    pytest.skip(f"cannot import {TARGET}: {exc!r}", allow_module_level=True)

TARGET_PATH = Path(mod.__file__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_attention(**attrs):
    """Build a ``DeepseekV4FLAttention`` instance bypassing its __init__."""

    class _Attn(mod.DeepseekV4FLAttention):
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    _Attn.__abstractmethods__ = frozenset()
    return _Attn(**attrs)


def _rotary_stub(positions, x, y, inverse=False):
    return x, y


def _bare_instance(cls):
    """Allocate ``cls`` without running ``__init__``/metaclass checks."""
    try:
        return object.__new__(cls)
    except TypeError:
        return cls.__new__(cls)


def _build_cfg(
    compress_ratios=None,
    num_hidden_layers=4,
    sliding_window=128,
    index_topk=64,
    max_model_len=128,
):
    hf_config = SimpleNamespace(
        compress_ratios=compress_ratios if compress_ratios is not None else [0, 1, 4, 128],
        num_hidden_layers=num_hidden_layers,
        sliding_window=sliding_window,
        index_topk=index_topk,
    )
    model_config = SimpleNamespace(hf_config=hf_config, max_model_len=max_model_len)
    return SimpleNamespace(model_config=model_config)


# ---------------------------------------------------------------------------
# Module surface
# ---------------------------------------------------------------------------


def test_module_exports_and_classes():
    assert set(mod.__all__) == {"DeepseekV4ForCausalLM", "DeepseekV4FLAttention"}
    assert mod.DeepseekV4FLAttention.__name__ == "DeepseekV4FLAttention"
    assert issubclass(mod.DeepseekV4FLAttention, mod.DeepseekV4Attention)
    assert issubclass(
        mod.DeepseekV4ForCausalLM, mod._xpu_model.DeepseekV4ForCausalLM
    )


def test_validate_short_context_attention_accepts_supported_config():
    cfg = _build_cfg(compress_ratios=[0, 1, 4, 128], max_model_len=128)
    assert mod._validate_short_context_attention(cfg) is None


def test_validate_short_context_attention_rejects_unknown_ratio():
    cfg = _build_cfg(compress_ratios=[0, 2, 4, 128], num_hidden_layers=4)
    with pytest.raises(NotImplementedError, match="compress_ratio=2"):
        mod._validate_short_context_attention(cfg)


def test_validate_short_context_attention_rejects_long_context():
    cfg = _build_cfg(sliding_window=64, max_model_len=129)
    with pytest.raises(ValueError, match="max_model_len <= 128"):
        mod._validate_short_context_attention(cfg)


def test_validate_short_context_attention_rejects_large_c4a_index():
    cfg = _build_cfg(compress_ratios=[4], index_topk=8, max_model_len=128)
    with pytest.raises(NotImplementedError, match="all compressed candidates"):
        mod._validate_short_context_attention(cfg)


def test_cache_rows_reads_logical_rows_and_rejects_bad_shape():
    cache = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
    slots = torch.tensor([0, 1, 2, 3, 4, 5])
    rows = mod._cache_rows(cache, slots, 3)
    assert rows.shape == (6, 4)
    assert torch.equal(rows[0], cache[0, 0])
    assert torch.equal(rows[5], cache[1, 2])

    with pytest.raises(ValueError):
        mod._cache_rows(torch.zeros(3, 4), slots, 3)
    with pytest.raises(ValueError):
        mod._cache_rows(torch.zeros(2, 5, 4), slots, 3)


def test_store_cache_rows_writes_cast_rows_and_rejects_bad_shape():
    cache = torch.zeros(2, 3, 4, dtype=torch.float32)
    slots = torch.tensor([0, 3, 5])
    rows = torch.ones(3, 4, dtype=torch.float64)
    mod._store_cache_rows(cache, slots, rows, 3)
    assert cache.dtype == torch.float32
    assert torch.all(cache[0, 0] == 1)
    assert torch.all(cache[1, 0] == 1)
    assert torch.all(cache[1, 2] == 1)

    with pytest.raises(ValueError):
        mod._store_cache_rows(torch.zeros(4), slots, rows, 3)
    with pytest.raises(ValueError):
        mod._store_cache_rows(torch.zeros(2, 2, 4), slots, rows, 3)


# ---------------------------------------------------------------------------
# Metadata builder + backends
# ---------------------------------------------------------------------------


def test_metadata_builder_build_tile_scheduler():
    result = mod.DeepseekV4FLMetadataBuilder.build_tile_scheduler(object(), 32)
    assert result == {"swaonly": None, "c4a": None, "c128a": None}


def test_swa_backend_name_and_builder():
    assert mod.DeepseekV4FLSWABackend.get_name() == "FL_ASCEND_V4_BF16_SWA_CACHE"
    assert (
        mod.DeepseekV4FLSWABackend.get_builder_cls()
        is mod.DeepseekV4FLMetadataBuilder
    )


def test_mla_backend_surface():
    assert mod.DeepseekV4FLBackend.supported_kv_cache_dtypes == ["auto"]
    assert mod.DeepseekV4FLBackend.get_name() == "FL_ASCEND_V4_BF16_SWA"
    assert (
        mod.DeepseekV4FLBackend.get_builder_cls()
        is mod.DeepseekV4SparseMLAMetadataBuilder
    )
    assert mod.DeepseekV4FLBackend.supports_compute_capability("whatever") is True


def test_get_padded_num_q_heads_is_identity():
    assert mod.DeepseekV4FLAttention.get_padded_num_q_heads(7) == 7


# ---------------------------------------------------------------------------
# DeepseekV4FLAttention.__init__ / forward
# ---------------------------------------------------------------------------


def test_attention_init_swaps_event_and_accepts_kwargs(monkeypatch):
    real_event = torch.cuda.Event

    class _FakeEvent:
        pass

    monkeypatch.setattr(torch, "npu", SimpleNamespace(Event=_FakeEvent), raising=False)

    base_init = mock.Mock()
    inst = _bare_instance(mod.DeepseekV4FLAttention)
    with mock.patch.object(mod.DeepseekV4Attention, "__init__", base_init), mock.patch.object(
        mod, "_validate_short_context_attention"
    ) as validator:
        cfg = object()
        mod.DeepseekV4FLAttention.__init__(inst, vllm_config=cfg, some="arg")
    validator.assert_called_once_with(cfg)
    base_init.assert_called_once_with(vllm_config=cfg, some="arg")
    assert torch.cuda.Event is real_event
    assert isinstance(inst, mod.DeepseekV4FLAttention)


def test_attention_init_uses_first_positional_config(monkeypatch):
    monkeypatch.setattr(
        torch, "npu", SimpleNamespace(Event=torch.cuda.Event), raising=False
    )
    base_init = mock.Mock()
    cfg = object()
    inst = _bare_instance(mod.DeepseekV4FLAttention)
    with mock.patch.object(mod.DeepseekV4Attention, "__init__", base_init), mock.patch.object(
        mod, "_validate_short_context_attention"
    ) as validator:
        mod.DeepseekV4FLAttention.__init__(inst, cfg)
    validator.assert_called_once_with(cfg)
    base_init.assert_called_once_with(cfg)


def test_attention_init_requires_config():
    inst = _bare_instance(mod.DeepseekV4FLAttention)
    with pytest.raises(AssertionError):
        mod.DeepseekV4FLAttention.__init__(inst)


def test_attention_forward_normalizes_and_projects():
    num_tokens, n_local_heads, head_dim, q_lora_rank = 3, 2, 4, 3
    hidden_states = torch.randn(num_tokens, 8)
    positions = torch.arange(num_tokens)

    recorded = {}

    def _prepare(hs, qr, kv, kv_score, iks, iw, pos, out):
        recorded["kv"] = kv
        out.zero_()

    attn = _make_attention(
        padded_heads=n_local_heads,
        head_dim=head_dim,
        q_lora_rank=q_lora_rank,
        n_local_heads=n_local_heads,
        eps=1e-6,
        q_norm=SimpleNamespace(weight=torch.ones(q_lora_rank)),
        kv_norm=SimpleNamespace(weight=torch.ones(head_dim)),
        _run_parallel_input_projections=lambda hs: (
            torch.randn(num_tokens, q_lora_rank + head_dim),
            None,
            None,
            None,
        ),
        _prepare_and_attn_fn=_prepare,
        _o_proj=lambda o, p: o,
    )
    out = attn.forward(positions, hidden_states, llama_4_scaling=torch.ones(1))
    assert out.shape == (num_tokens, n_local_heads, head_dim)
    assert recorded["kv"].shape == (num_tokens, head_dim)


def test_run_parallel_input_projections_without_compressor():
    num_tokens, q_lora_rank, head_dim = 2, 3, 4
    attn = _make_attention(
        compressor=None,
        q_lora_rank=q_lora_rank,
        head_dim=head_dim,
        _fused_wqa_wkv_gemm=lambda hs: torch.zeros(num_tokens, q_lora_rank + head_dim),
    )
    qr_kv, kv_score, iks, iw = attn._run_parallel_input_projections(torch.randn(num_tokens, 5))
    assert qr_kv.shape == (num_tokens, q_lora_rank + head_dim)
    assert kv_score is None
    assert iks is None and iw is None


def test_run_parallel_input_projections_with_compressor():
    num_tokens, q_lora_rank, head_dim = 2, 3, 4
    weight = torch.randn(6, 5)
    attn = _make_attention(
        compressor=SimpleNamespace(
            fused_wkv_wgate=SimpleNamespace(weight=weight)
        ),
        q_lora_rank=q_lora_rank,
        head_dim=head_dim,
        _fused_wqa_wkv_gemm=lambda hs: torch.zeros(num_tokens, q_lora_rank + head_dim),
    )
    _, kv_score, _, _ = attn._run_parallel_input_projections(torch.randn(num_tokens, 5))
    assert kv_score.shape == (num_tokens, 6)


def test_fused_qnorm_rope_kv_insert_returns_early_for_non_dict():
    q = torch.randn(2, 2, 4)
    attn = _make_attention(eps=1e-6)
    out = attn._fused_qnorm_rope_kv_insert(q, torch.randn(2, 4), torch.arange(2), None)
    assert out is q


def test_fused_qnorm_rope_kv_insert_stores_valid_slots():
    num_tokens, n_heads, head_dim = 4, 2, 4
    block_size = 3
    q = torch.randn(num_tokens, n_heads, head_dim)
    kv = torch.randn(num_tokens, head_dim)
    slots = torch.tensor([0, 3, -1, 5])
    cache = torch.zeros(2, block_size, head_dim)
    metadata = SimpleNamespace(slot_mapping=slots, block_size=block_size)
    attn = _make_attention(
        eps=1e-6,
        rotary_emb=SimpleNamespace(forward_native=_rotary_stub),
        swa_cache_layer=SimpleNamespace(prefix="swa", kv_cache=cache),
    )
    out = attn._fused_qnorm_rope_kv_insert(
        q, kv, torch.arange(num_tokens), {"swa": metadata}
    )
    assert out.shape == q.shape
    assert torch.allclose(cache[0, 0], kv[0])
    assert torch.allclose(cache[1, 0], kv[1])
    assert torch.allclose(cache[1, 2], kv[3])


def test_fused_qnorm_rope_kv_insert_skips_store_when_no_valid_slots():
    num_tokens, n_heads, head_dim = 3, 1, 3
    slots = torch.tensor([-1, -1, -1])
    cache = torch.zeros(1, 2, head_dim)
    metadata = SimpleNamespace(slot_mapping=slots, block_size=2)
    attn = _make_attention(
        eps=1e-6,
        rotary_emb=SimpleNamespace(forward_native=_rotary_stub),
        swa_cache_layer=SimpleNamespace(prefix="swa", kv_cache=cache),
    )
    out = attn._fused_qnorm_rope_kv_insert(
        torch.randn(num_tokens, n_heads, head_dim),
        torch.randn(num_tokens, head_dim),
        torch.arange(num_tokens),
        {"swa": metadata},
    )
    assert out.shape == (num_tokens, n_heads, head_dim)
    assert torch.all(cache == 0)


def test_o_proj_requires_single_group_and_projects():
    o = torch.randn(3, 8)

    def _wo_a(flat):
        return flat * 2.0

    def _wo_b(z):
        return z + 1.0

    attn = _make_attention(
        rotary_emb=SimpleNamespace(forward_native=_rotary_stub),
        n_local_groups=1,
        wo_a=_wo_a,
        wo_b=_wo_b,
    )
    out = attn._o_proj(o, torch.arange(3))
    assert out.shape == o.shape

    bad = _make_attention(
        rotary_emb=SimpleNamespace(forward_native=_rotary_stub),
        n_local_groups=2,
    )
    with pytest.raises(ValueError, match="one group per TP rank"):
        bad._o_proj(o, torch.arange(3))


def test_prepare_and_attn_without_indexer_branch():
    num_tokens, n_local_heads, head_dim = 2, 1, 2
    recorded = {}

    def _fused(q, kv, positions, metadata):
        return q

    attn = _make_attention(
        wq_b=lambda qr: torch.randn(num_tokens, n_local_heads * head_dim),
        n_local_heads=n_local_heads,
        head_dim=head_dim,
        compressor=None,
        compress_ratio=1,
        _fused_qnorm_rope_kv_insert=_fused,
        forward_mqa=lambda q, kv, pos, out: recorded.setdefault("called", q),
    )
    out = torch.empty(num_tokens, n_local_heads, head_dim)
    with mock.patch.object(
        mod, "get_forward_context", lambda: SimpleNamespace(attn_metadata={})
    ):
        attn._prepare_and_attn(
            torch.randn(num_tokens, head_dim),
            torch.randn(num_tokens, 3),
            torch.randn(num_tokens, head_dim),
            None,
            None,
            None,
            torch.arange(num_tokens),
            out,
        )
    assert "called" in recorded


def test_prepare_and_attn_c4a_fills_topk_buffer():
    num_tokens, n_local_heads, head_dim = 2, 1, 2
    topk = 4
    positions = torch.tensor([0, 3])
    buffer = torch.full((num_tokens, topk), 7, dtype=torch.int32)
    compressed = mock.Mock()

    attn = _make_attention(
        wq_b=lambda qr: torch.randn(num_tokens, n_local_heads * head_dim),
        n_local_heads=n_local_heads,
        head_dim=head_dim,
        compressor=SimpleNamespace(),
        compress_ratio=4,
        indexer=SimpleNamespace(topk_tokens=topk),
        topk_indices_buffer=buffer,
        _fused_qnorm_rope_kv_insert=lambda q, kv, pos, md: q,
        _compress_kv_bf16=compressed,
        forward_mqa=lambda *a: None,
    )
    out = torch.empty(num_tokens, n_local_heads, head_dim)
    with mock.patch.object(
        mod, "get_forward_context", lambda: SimpleNamespace(attn_metadata={})
    ):
        attn._prepare_and_attn(
            torch.randn(num_tokens, head_dim),
            torch.randn(num_tokens, 3),
            torch.randn(num_tokens, head_dim),
            torch.randn(num_tokens, 8),
            None,
            None,
            positions,
            out,
        )
    compressed.assert_called_once()
    assert torch.equal(buffer[0], torch.full((topk,), -1, dtype=torch.int32))
    assert buffer[1, 0].item() == 0
    assert buffer[1, 1].item() == -1


# ---------------------------------------------------------------------------
# DeepseekV4FLAttention._compress_kv_bf16
# ---------------------------------------------------------------------------


def _compress_setup(no_active=False):
    ratio = 4
    head_dim = 2
    state_width = 2 * head_dim  # coff * head_dim with overlap
    state_block_size = 4
    state_cache = torch.zeros(2, state_block_size, 2 * state_width)

    state_metadata = SimpleNamespace(
        slot_mapping=torch.tensor([0, 1, 2, 3]),
        block_size=state_block_size,
        token_to_req_indices=torch.tensor([0, 0, 1, 1]),
        block_table=torch.tensor([[0, 0, 0, 0], [1, 1, 1, 1]]),
    )
    compressed_metadata = SimpleNamespace(
        slot_mapping=torch.tensor([0, 4, 8, 12]),
        block_size=ratio * 2,
    )
    kv_cache = torch.zeros(8, 2, head_dim)
    compressor = SimpleNamespace(
        state_cache=SimpleNamespace(prefix="state", kv_cache=state_cache),
        ape=torch.arange(ratio * state_width, dtype=torch.float32).reshape(
            ratio, state_width
        ),
        norm=SimpleNamespace(weight=torch.ones(head_dim)),
    )
    positions = torch.tensor([0, 1, 2, 5]) if no_active else torch.tensor([0, 3, 4, 7])
    if no_active:
        state_metadata.slot_mapping = torch.tensor([-1, -1, -1, -1])
    attn = _make_attention(
        compress_ratio=ratio,
        head_dim=head_dim,
        rope_head_dim=2,
        eps=1e-6,
        prefix="comp",
        kv_cache=kv_cache,
        compressor=compressor,
        rotary_emb=SimpleNamespace(
            cos_sin_cache=torch.ones(8, 2),
            forward_native=_rotary_stub,
        ),
    )
    kv_score = torch.randn(4, 2 * state_width)
    metadata_dict = {"state": state_metadata, "comp": compressed_metadata}
    return attn, kv_score, positions, metadata_dict


def test_compress_kv_bf16_returns_for_non_dict_metadata():
    attn, kv_score, positions, _ = _compress_setup()
    with mock.patch.object(
        mod, "get_forward_context", lambda: SimpleNamespace(attn_metadata=[1, 2])
    ):
        assert attn._compress_kv_bf16(kv_score, positions) is None


def test_compress_kv_bf16_skips_store_and_returns_when_inactive():
    attn, kv_score, positions, metadata_dict = _compress_setup(no_active=True)
    with mock.patch.object(
        mod,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata=metadata_dict),
    ):
        assert attn._compress_kv_bf16(kv_score, positions) is None
    assert torch.all(attn.compressor.state_cache.kv_cache == 0)
    assert torch.all(attn.kv_cache == 0)


def test_compress_kv_bf16_writes_state_and_compressed_cache():
    attn, kv_score, positions, metadata_dict = _compress_setup()
    state_cache = attn.compressor.state_cache.kv_cache
    with mock.patch.object(
        mod,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata=metadata_dict),
    ):
        assert attn._compress_kv_bf16(kv_score, positions) is None
    # packed_state written into the four valid state rows
    assert state_cache.abs().sum().item() > 0
    # two active tokens (positions 3 and 7) write compressed rows 4 and 12
    assert torch.any(attn.kv_cache[2, 0] != 0)
    assert torch.any(attn.kv_cache[6, 0] != 0)


# ---------------------------------------------------------------------------
# DeepseekV4FLAttention.forward_mqa
# ---------------------------------------------------------------------------


def _forward_mqa_setup(compress_ratio):
    num_tokens = 3
    head_dim = 2
    window = 4
    block_size = 4
    swa_cache = torch.randn(4, block_size, head_dim)
    swa_metadata = SimpleNamespace(
        token_to_req_indices=torch.tensor([0, 0, 0]),
        block_size=block_size,
        block_table=torch.tensor([[0, 1, 2, 3]]),
    )
    metadata_dict = {"swa": swa_metadata}
    kv_cache = torch.randn(4, 2, head_dim)
    attrs = dict(
        eps=1e-6,
        scale=0.5,
        window_size=window,
        compress_ratio=compress_ratio,
        n_local_heads=1,
        head_dim=head_dim,
        max_model_len=8,
        prefix="comp",
        kv_cache=kv_cache,
        attn_sink=torch.zeros(1),
        swa_cache_layer=SimpleNamespace(prefix="swa", kv_cache=swa_cache),
    )
    positions = torch.tensor([0, 1, 2])
    if compress_ratio == 4:
        topk = torch.tensor([[0, 1], [0, 1], [0, -1]], dtype=torch.int32)
        attrs["topk_indices_buffer"] = topk
        attrs["indexer"] = SimpleNamespace(topk_tokens=2)
        metadata_dict["comp"] = SimpleNamespace(
            block_size=8,
            block_table=torch.tensor([[0, 1, 2, 3]]),
        )
    attn = _make_attention(**attrs)
    return attn, positions, metadata_dict, num_tokens, head_dim


def _run_forward_mqa(compress_ratio):
    attn, positions, metadata_dict, num_tokens, head_dim = _forward_mqa_setup(
        compress_ratio
    )
    q = torch.randn(num_tokens, 1, head_dim)
    output = torch.zeros(num_tokens, 1, head_dim)
    with mock.patch.object(
        mod,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata=metadata_dict),
    ):
        attn.forward_mqa(q, torch.randn(num_tokens, head_dim), positions, output)
    return output


def test_forward_mqa_zeroes_output_for_non_dict_metadata():
    attn, positions, _, num_tokens, head_dim = _forward_mqa_setup(1)
    output = torch.ones(num_tokens, 1, head_dim)
    with mock.patch.object(
        mod, "get_forward_context", lambda: SimpleNamespace(attn_metadata=None)
    ):
        attn.forward_mqa(torch.randn(num_tokens, 1, head_dim), torch.randn(num_tokens, 2), positions, output)
    assert torch.all(output == 0)


def test_forward_mqa_swa_only_path():
    output = _run_forward_mqa(1)
    assert output.shape == (3, 1, 2)
    assert torch.isfinite(output).all()


def test_forward_mqa_compressed_path():
    output = _run_forward_mqa(4)
    assert output.shape == (3, 1, 2)
    assert torch.isfinite(output).all()


# ---------------------------------------------------------------------------
# DeepseekV4ForCausalLM + top-level install hooks
# ---------------------------------------------------------------------------


def test_causal_lm_init_validates_then_delegates():
    base = mod._xpu_model.DeepseekV4ForCausalLM
    with mock.patch.object(base, "__init__", mock.Mock()) as base_init, mock.patch.object(
        mod, "_validate_short_context_attention"
    ) as validator:
        cfg = object()
        mod.DeepseekV4ForCausalLM(vllm_config=cfg, prefix="p")
    validator.assert_called_once_with(cfg)
    base_init.assert_called_once_with(vllm_config=cfg, prefix="p")


def test_install_hc_head_fallback_patches_and_is_idempotent():
    mhc_name = "vllm.model_executor.layers.mhc"
    flaggems_name = "flaggems_vllm"
    saved_mhc = sys.modules.get(mhc_name)
    saved_fg = sys.modules.get(flaggems_name)

    fake_mhc = _ensure_module(mhc_name, {})
    saved_hc = getattr(fake_mhc, "HCHeadOp", None)
    had_hc = hasattr(fake_mhc, "HCHeadOp")

    class _HCHeadOp:
        pass

    fake_mhc.HCHeadOp = _HCHeadOp

    calls = {}

    def _kernel(flat, hc_fn, hc_scale, hc_base, out, hidden_size, rms, hc_eps, hc_mult):
        calls.update(
            hidden_size=hidden_size, hc_mult=hc_mult, flat_shape=tuple(flat.shape)
        )
        out.fill_(1.0)

    fake_fg = _ensure_module(flaggems_name, {})
    saved_kernel = getattr(fake_fg, "hc_head_fused_kernel", None)
    had_kernel = hasattr(fake_fg, "hc_head_fused_kernel")
    fake_fg.hc_head_fused_kernel = _kernel

    try:
        mod._install_hc_head_fallback()
        assert getattr(_HCHeadOp, "_vllm_fl_ascend", False) is True

        hidden = torch.zeros(2, 3, 4)
        out = _HCHeadOp().forward_native(
            hidden,
            torch.ones(4),
            torch.ones(1),
            torch.zeros(4),
            1e-6,
            1e-6,
        )
        assert out.shape == (2, 4)
        assert torch.all(out == 1.0)
        assert calls["hidden_size"] == 4
        assert calls["hc_mult"] == 3

        patched = _HCHeadOp.forward_native
        mod._install_hc_head_fallback()
        assert _HCHeadOp.forward_native is patched
    finally:
        if had_hc:
            fake_mhc.HCHeadOp = saved_hc
        elif hasattr(fake_mhc, "HCHeadOp"):
            delattr(fake_mhc, "HCHeadOp")
        if had_kernel:
            fake_fg.hc_head_fused_kernel = saved_kernel
        elif hasattr(fake_fg, "hc_head_fused_kernel"):
            delattr(fake_fg, "hc_head_fused_kernel")
        if saved_mhc is not None:
            sys.modules[mhc_name] = saved_mhc
        else:
            sys.modules.pop(mhc_name, None)
        if saved_fg is not None:
            sys.modules[flaggems_name] = saved_fg
        else:
            sys.modules.pop(flaggems_name, None)


def _exec_source_with_device_type(device_type):
    """Re-execute the target source with a forced platform device_type.

    Returns ``(namespace, hc_head_op_cls)`` where ``hc_head_op_cls`` is the
    stand-in ``HCHeadOp`` the module saw during execution.  Used to cover both
    sides of the module-level ``if current_platform.device_type == "npu"``
    guards regardless of the platform the test suite actually runs on.
    """
    source = TARGET_PATH.read_text(encoding="utf-8")
    code = compile(source, str(TARGET_PATH), "exec")

    platform_name = "vllm.platforms"
    xpu_pkg = "vllm.models.deepseek_v4.xpu"
    xpu_model_name = xpu_pkg + ".model"
    mhc_name = "vllm.model_executor.layers.mhc"

    platform_mod = sys.modules[platform_name]
    saved_cp = getattr(platform_mod, "current_platform", None)
    setattr(platform_mod, "current_platform", SimpleNamespace(device_type=device_type))

    saved_xpu = sys.modules.get(xpu_model_name)
    fake_xpu = types.ModuleType(xpu_model_name)
    fake_xpu.DeepseekV4ForCausalLM = _StandIn
    sys.modules[xpu_model_name] = fake_xpu
    xpu_parent = sys.modules.get(xpu_pkg)
    if xpu_parent is not None:
        setattr(xpu_parent, "model", fake_xpu)

    saved_mhc = sys.modules.get(mhc_name)
    fake_mhc = _ensure_module(mhc_name, {})
    saved_hc = getattr(fake_mhc, "HCHeadOp", None)
    had_hc = hasattr(fake_mhc, "HCHeadOp")

    class _HCHeadOp:
        pass

    fake_mhc.HCHeadOp = _HCHeadOp

    namespace = {
        "__name__": "vllm_fl.models.deepseek_v4",
        "__file__": str(TARGET_PATH),
        "__package__": "vllm_fl.models",
    }
    try:
        exec(code, namespace)
    finally:
        setattr(platform_mod, "current_platform", saved_cp)
        if saved_xpu is not None:
            sys.modules[xpu_model_name] = saved_xpu
            if xpu_parent is not None:
                setattr(xpu_parent, "model", saved_xpu)
        else:
            sys.modules.pop(xpu_model_name, None)
        if had_hc:
            fake_mhc.HCHeadOp = saved_hc
        elif hasattr(fake_mhc, "HCHeadOp"):
            delattr(fake_mhc, "HCHeadOp")
        if saved_mhc is not None:
            sys.modules[mhc_name] = saved_mhc
        else:
            sys.modules.pop(mhc_name, None)
    return namespace, _HCHeadOp


def test_top_level_npu_branch_installs_fallback_and_overrides_attention():
    namespace, hc_head_op = _exec_source_with_device_type("npu")
    assert namespace["DeepseekV4FLAttention"].__name__ == "DeepseekV4FLAttention"
    assert getattr(hc_head_op, "_vllm_fl_ascend", False) is True
    assert hasattr(hc_head_op, "forward_native")


def test_top_level_non_npu_branch_skips_install():
    namespace, hc_head_op = _exec_source_with_device_type("cuda")
    assert namespace["DeepseekV4FLAttention"].__name__ == "DeepseekV4FLAttention"
    assert not hasattr(hc_head_op, "forward_native")
    assert getattr(hc_head_op, "_vllm_fl_ascend", False) is False
