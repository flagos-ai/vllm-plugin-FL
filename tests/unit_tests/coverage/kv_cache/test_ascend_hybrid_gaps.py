# Copyright (c) 2026 BAAI. All rights reserved.

"""CPU tests for vllm_fl.kv_cache.ascend.hybrid logic not covered elsewhere.

``hybrid.py`` implements dense Ascend hybrid cache views over shared byte
pools.  The module imports vllm at module level, so the stub vllm graph
below only mocks the external vllm boundary (interface dataclasses and
helpers); every function under test runs unmodified on plain CPU tensors.
Numeric expectations are computed with independent arithmetic in the tests
(byte offsets, page paddings, copy/zero semantics spelled out by hand).
"""

import math
import sys
import types
from dataclasses import dataclass

import pytest
import torch


def _install_vllm_stubs():
    """Minimal vllm module graph hybrid.py imports at module level."""
    vllm = types.ModuleType("vllm")
    vllm.__path__ = []
    models = types.ModuleType("vllm.model_executor.models")
    models.__path__ = []
    models.ModelRegistry = types.SimpleNamespace(
        resolve_model_cls=lambda *a, **k: (None, None)
    )
    math_utils = types.ModuleType("vllm.utils.math_utils")
    math_utils.cdiv = lambda a, b: -(-a // b)
    torch_utils = types.ModuleType("vllm.utils.torch_utils")

    def get_dtype_size(dtype):
        """Independent dtype-size oracle (matches torch's element_size)."""
        return torch.empty((), dtype=dtype).element_size()

    torch_utils.STR_DTYPE_TO_TORCH_DTYPE = {
        "auto": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    torch_utils.get_dtype_size = get_dtype_size
    utils = types.ModuleType("vllm.utils")
    utils.__path__ = []
    utils.math_utils = math_utils
    utils.torch_utils = torch_utils
    iface = types.ModuleType("vllm.v1.kv_cache_interface")
    iface.__path__ = []

    @dataclass
    class AttentionSpec:
        """Stub of the vllm interface class used for isinstance dispatch."""

        block_size: int = 16
        num_kv_heads: int = 2
        head_size: int = 8
        head_size_v: int = 8
        page_size_bytes: int = 256
        dtype: object = torch.float32
        tokens_per_state: int = 1
        page_size_padded: int | None = None

    @dataclass
    class MambaSpec:
        """Stub of the vllm interface class used for isinstance dispatch."""

        page_size_bytes: int = 256
        shapes: tuple = ()
        dtypes: tuple = ()

    class UniformTypeKVCacheSpecs:
        """Stub of the per-group spec wrapper."""

        def __init__(self, kv_cache_specs):
            self.kv_cache_specs = kv_cache_specs

    iface.AttentionSpec = AttentionSpec
    iface.MambaSpec = MambaSpec
    iface.UniformTypeKVCacheSpecs = UniformTypeKVCacheSpecs
    v1 = types.ModuleType("vllm.v1")
    v1.__path__ = []
    kv_cache_dir = types.ModuleType("vllm.v1.worker")
    kv_cache_dir.__path__ = []
    v1.worker = kv_cache_dir
    vllm.model_executor = types.SimpleNamespace(models=models)
    vllm.utils = utils
    vllm.v1 = v1
    for name, mod in {
        "vllm": vllm,
        "vllm.model_executor": types.SimpleNamespace(models=models),
        "vllm.model_executor.models": models,
        "vllm.utils": utils,
        "vllm.utils.math_utils": math_utils,
        "vllm.utils.torch_utils": torch_utils,
        "vllm.v1": v1,
        "vllm.v1.kv_cache_interface": iface,
        "vllm.v1.worker": kv_cache_dir,
    }.items():
        sys.modules[name] = mod
    return iface


import importlib.util
from pathlib import Path

# hybrid.py's contracts are pinned against the STUB interface classes above
# (``page_size_bytes`` is not an attribute of vLLM 0.32's real AttentionSpec),
# and the module must import identically whether or not a real vllm was
# already imported by an earlier test file in this session. So the stub graph
# is always installed, the module is loaded under a private name through
# spec_from_file_location, and the pre-existing vllm module entries (real or
# absent) are restored afterwards.
_STUB_ENTRIES = [
    "vllm",
    "vllm.model_executor",
    "vllm.model_executor.models",
    "vllm.utils",
    "vllm.utils.math_utils",
    "vllm.utils.torch_utils",
    "vllm.v1",
    "vllm.v1.kv_cache_interface",
    "vllm.v1.worker",
]
_SAVED = {name: sys.modules.get(name) for name in _STUB_ENTRIES}
_iface = _install_vllm_stubs()

_SOURCE = (
    Path(__file__).resolve().parents[4]
    / "vllm_fl"
    / "kv_cache"
    / "ascend"
    / "hybrid.py"
)
_spec = importlib.util.spec_from_file_location("supp_ascend_hybrid_under_test", _SOURCE)
hybrid = importlib.util.module_from_spec(_spec)
sys.modules["supp_ascend_hybrid_under_test"] = hybrid
try:
    _spec.loader.exec_module(hybrid)
finally:
    # Restore whatever was in sys.modules before (real vllm entries included).
    for name in _STUB_ENTRIES:
        if _SAVED[name] is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = _SAVED[name]

AttentionSpec = _iface.AttentionSpec
MambaSpec = _iface.MambaSpec
UniformTypeKVCacheSpecs = _iface.UniformTypeKVCacheSpecs


class TestUsesDenseHybridCache:
    """uses_dense_hybrid_cache model-type gate."""

    def test_qwen3_5_text_with_hybrid(self):
        """A hybrid qwen3_5_text model selects the dense layout."""
        cfg = types.SimpleNamespace(
            is_hybrid=True,
            hf_text_config=types.SimpleNamespace(model_type="qwen3_5_text"),
        )
        assert hybrid.uses_dense_hybrid_cache(cfg) is True

    def test_qwen3_5_moe_text_with_hybrid(self):
        """A hybrid qwen3_5_moe_text model also selects the dense layout."""
        cfg = types.SimpleNamespace(
            is_hybrid=True,
            hf_text_config=types.SimpleNamespace(model_type="qwen3_5_moe_text"),
        )
        assert hybrid.uses_dense_hybrid_cache(cfg) is True

    def test_other_model_type_rejected(self):
        """Non-GDN model types fall back to the non-dense path."""
        cfg = types.SimpleNamespace(
            is_hybrid=True,
            hf_text_config=types.SimpleNamespace(model_type="llama"),
        )
        assert hybrid.uses_dense_hybrid_cache(cfg) is False

    def test_non_hybrid_rejected(self):
        """is_hybrid=False short-circuits regardless of model type."""
        cfg = types.SimpleNamespace(
            is_hybrid=False,
            hf_text_config=types.SimpleNamespace(model_type="qwen3_5_text"),
        )
        assert hybrid.uses_dense_hybrid_cache(cfg) is False

    def test_missing_hf_config_raises_attribute_error(self):
        """A missing hf_text_config surfaces as AttributeError (no default)."""
        cfg = types.SimpleNamespace(is_hybrid=True)
        with pytest.raises(AttributeError):
            hybrid.uses_dense_hybrid_cache(cfg)


class _AlignConfig:
    """Config double for align_hybrid_block_size."""

    def __init__(self, conv_shape, state_shape, head_size=8, num_kv_heads=2):
        cache = types.SimpleNamespace(
            cache_dtype="auto",
            user_specified_block_size=False,
            block_size=0,
            mamba_page_size_padded=0,
            mamba_block_size=0,
            mamba_cache_mode="align",
        )
        conv_bytes = math.prod(conv_shape) * 2
        state_bytes = math.prod(state_shape) * 2
        self.model = types.SimpleNamespace(
            architecture="X",
            use_mla=False,
            dtype=torch.bfloat16,
            get_head_size=lambda: head_size,
            get_num_kv_heads=lambda pc: num_kv_heads,
            max_model_len=4096,
        )
        self.model_cls = types.SimpleNamespace(
            get_mamba_state_shape_from_config=lambda cfg: [conv_shape, state_shape],
            get_mamba_state_dtype_from_config=lambda cfg: [torch.bfloat16] * 2,
        )
        # vllm_config attribute names the function reads.
        self.parallel_config = types.SimpleNamespace()
        self.cache_config = cache
        self.model_config = self.model
        self._cache = cache
        self._conv_bytes = conv_bytes
        self._state_bytes = state_bytes


def _patch_registry(monkeypatch, cfg):
    """Route ModelRegistry.resolve_model_cls to the config's model class."""

    def fake_resolve(architecture, model_config=None):
        assert architecture == "X"
        return cfg.model_cls, None

    monkeypatch.setattr(hybrid.ModelRegistry, "resolve_model_cls", fake_resolve)


class TestAlignHybridBlockSize:
    """align_hybrid_block_size page arithmetic."""

    def test_alignment_computes_block_size(self, monkeypatch):
        """block_size = 128 * cdiv(state_bytes, 128 * key_token_bytes)."""
        # state: 128*2 = 256 bytes; key_token_bytes = 8 * 2 * 2 = 32
        # → block_size = 128 * cdiv(256, 128*32) = 128 * 1 = 128; 128*32=4096
        # needs state_bytes = 4096: shape (2, 32, 32) bf16 = 4096 bytes.
        cfg = _AlignConfig((1, 2), (2, 32, 32))
        _patch_registry(monkeypatch, cfg)
        hybrid.align_hybrid_block_size(cfg)
        key_token_bytes = 8 * 2 * 2
        expected = 128 * math.ceil(cfg._state_bytes / (128 * key_token_bytes))
        assert cfg.cache_config.block_size == expected
        assert (
            cfg.cache_config.mamba_page_size_padded
            == 2 * cfg._state_bytes + cfg._conv_bytes
        )

    def test_mamba_block_size_align_mode(self, monkeypatch):
        """mamba_cache_mode='align' keeps mamba_block_size == block_size."""
        cfg = _AlignConfig((1, 2), (2, 32, 32))
        cfg.cache_config.mamba_cache_mode = "align"
        _patch_registry(monkeypatch, cfg)
        hybrid.align_hybrid_block_size(cfg)
        assert cfg.cache_config.mamba_block_size == cfg.cache_config.block_size

    def test_mamba_block_size_non_align_mode(self, monkeypatch):
        """A non-'align' mamba mode sets mamba_block_size = max_model_len."""
        cfg = _AlignConfig((1, 2), (2, 32, 32))
        cfg.cache_config.mamba_cache_mode = "host"
        _patch_registry(monkeypatch, cfg)
        hybrid.align_hybrid_block_size(cfg)
        assert cfg.cache_config.mamba_block_size == cfg.model_config.max_model_len

    def test_mla_rejected(self, monkeypatch):
        """MLA models are rejected (unsupported state family)."""
        cfg = _AlignConfig((1, 2), (2, 32, 32))
        cfg.model.use_mla = True
        _patch_registry(monkeypatch, cfg)
        with pytest.raises(NotImplementedError):
            hybrid.align_hybrid_block_size(cfg)

    def test_single_state_rejected(self, monkeypatch):
        """Fewer than two state tensors is not conv+GDN."""
        cfg = _AlignConfig((1, 2), (2, 32, 32))
        cfg.model_cls.get_mamba_state_shape_from_config = lambda c: [(2, 32, 32)]
        _patch_registry(monkeypatch, cfg)
        with pytest.raises(NotImplementedError):
            hybrid.align_hybrid_block_size(cfg)

    def test_inverted_state_sizes_rejected(self, monkeypatch):
        """conv bytes >= state bytes violates the conv-then-GDN invariant."""
        cfg = _AlignConfig((2, 64, 64), (1, 2, 2))  # conv much larger
        _patch_registry(monkeypatch, cfg)
        with pytest.raises(NotImplementedError):
            hybrid.align_hybrid_block_size(cfg)

    def test_unalignable_pages_rejected(self, monkeypatch):
        """state_bytes not a multiple of 128*key_token_bytes raises."""
        cfg = _AlignConfig((1, 2), (3, 3, 3))  # 54 bytes: never aligns
        _patch_registry(monkeypatch, cfg)
        with pytest.raises(ValueError, match="align Ascend key"):
            hybrid.align_hybrid_block_size(cfg)

    def test_user_block_size_too_large_rejected(self, monkeypatch):
        """A user block_size above the required one is rejected."""
        cfg = _AlignConfig((1, 2), (2, 32, 32))
        cfg.cache_config.user_specified_block_size = True
        cfg.cache_config.block_size = 10**9  # far above the computed 128
        _patch_registry(monkeypatch, cfg)
        with pytest.raises(ValueError, match="preserve shared K/SSM"):
            hybrid.align_hybrid_block_size(cfg)


class TestPadAttentionSpecs:
    """pad_attention_specs global page-size application."""

    def test_large_attention_spec_untouched(self):
        """Attention specs already at/above the page size are unchanged."""
        spec = AttentionSpec(page_size_bytes=512)
        out = hybrid.pad_attention_specs({"a": spec}, page_size=256)
        # No padding applied: the field stays at its default.
        assert out["a"].page_size_padded is None
        assert out["a"].page_size_bytes == 512

    def test_non_attention_spec_untouched(self):
        """Mamba specs never receive the attention page padding."""
        spec = MambaSpec(page_size_bytes=64)
        out = hybrid.pad_attention_specs({"m": spec}, page_size=256)
        assert not hasattr(out["m"], "page_size_padded")


class TestDenseViews:
    """dense_attention_view / dense_mamba_views byte-slicing arithmetic."""

    def test_attention_view_slices_tail(self):
        """The view consumes the pool tail with exact dtype reinterpretation."""
        dtype = torch.float32
        payload = 2 * 3 * 4 * 4  # shape (2,3,4) float32
        raw = torch.zeros(payload + 24, dtype=torch.uint8)
        # The view starts at offset = raw.nbytes - payload; the LAST 4 bytes
        # of the raw buffer are the final float of the viewed tensor.
        raw[-4:].view(torch.float32).fill_(7.0)
        spec = types.SimpleNamespace(dtype=dtype)
        view = hybrid.dense_attention_view(raw, spec, (2, 3, 4))
        assert view.shape == (2, 3, 4)
        assert view.dtype is dtype
        # Flattened view order is row-major: the last element carries 7.0.
        assert view.flatten()[-1].item() == 7.0
        assert torch.all(view.flatten()[:-1] == 0.0)

    def test_attention_view_rejects_misaligned(self):
        """A payload offset not aligned to the dtype size is rejected."""
        raw = torch.zeros(31, dtype=torch.uint8)  # 31 % 4 != 0
        spec = types.SimpleNamespace(dtype=torch.float32)
        with pytest.raises(ValueError, match="backing"):
            hybrid.dense_attention_view(raw, spec, (2, 3, 2))

    def test_attention_view_rejects_oversized(self):
        """A payload larger than the backing is rejected."""
        raw = torch.zeros(8, dtype=torch.uint8)
        spec = types.SimpleNamespace(dtype=torch.float32)
        with pytest.raises(ValueError, match="backing"):
            hybrid.dense_attention_view(raw, spec, (2, 3, 4))

    def test_mamba_views_grouped_densely(self):
        """Each state type gets its own dense slab of num_blocks rows."""
        shapes = [(2, 4), (2, 8)]
        dtypes = [torch.float32, torch.bfloat16]
        num_blocks = 3
        total = 3 * (2 * 4 * 4 + 2 * 8 * 2)
        raw = torch.zeros(total, dtype=torch.uint8)
        # Mark the second slab's first row with a known bf16 pattern.
        first_slab = 3 * 2 * 4 * 4
        raw[first_slab : first_slab + 2].view(torch.bfloat16).fill_(1.0)
        views = hybrid.dense_mamba_views(raw, shapes, dtypes, num_blocks)
        assert len(views) == 2
        assert views[0].shape == (3, 2, 4)
        assert views[1].shape == (3, 2, 8)
        # Only the first bf16 element of slab 2 was marked with 1.0.
        assert views[1][0, 0, 0].item() == 1.0
        assert torch.all(views[1][0, 0, 1:] == 0.0)

    def test_mamba_views_rejects_short_backing(self):
        """A backing smaller than the slabs is rejected."""
        raw = torch.zeros(8, dtype=torch.uint8)
        with pytest.raises(ValueError, match="state backing"):
            hybrid.dense_mamba_views(raw, [(2, 4)], [torch.float32], 3)


class TestAllocateAscendKvCacheErrors:
    """allocate_ascend_kv_cache guard rails (CPU-drivable branches)."""

    def test_empty_tensors_returns_empty(self):
        """No cache tensors → empty views and planes."""
        cfg = types.SimpleNamespace(kv_cache_tensors=[])
        bind, mgr, planes = hybrid.allocate_ascend_kv_cache(
            cfg, torch.device("cpu"), None, {}, [], use_dense=True
        )
        assert bind == {} and mgr == {} and planes == []

    def test_host_resident_rejected(self):
        """Host-resident pools are rejected with NotImplementedError."""
        tensor = types.SimpleNamespace(host_resident=True)
        cfg = types.SimpleNamespace(kv_cache_tensors=[tensor])
        with pytest.raises(NotImplementedError, match="host-resident"):
            hybrid.allocate_ascend_kv_cache(
                cfg, torch.device("cpu"), None, {}, [], use_dense=True
            )
