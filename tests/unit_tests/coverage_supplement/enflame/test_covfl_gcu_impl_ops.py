# Copyright (c) 2026 BAAI. All rights reserved.

"""Coverage tests for GCU operator wrappers under ``gcu/impl``.

The tests never launch a real GCU/Triton kernel.  Heavy vLLM/Triton imports
are replaced with lightweight stand-ins, and every Triton kernel symbol is
swapped for a call-capturing stub before the Python launcher is exercised.
The tests only assert on control flow, argument plumbing, grid computation and
validation errors, so they are safe on a CPU-only host.
"""

from __future__ import annotations

import builtins
import contextlib
import os
import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

_SRC_ROOT = (
    os.environ.get("VLLM_FL_SRC")
    or os.environ.get("ENFLAME_SRC")
    or r"C:\Users\21649\AppData\Local\Temp\opencode\enflame-src-030b"
)
IMPL_DIR = Path(_SRC_ROOT) / "vllm_fl" / "dispatch" / "backends" / "vendor" / "gcu" / "impl"
_PKG = "_covfl_gcu_impl"
_MISSING = object()
_TL_SENTINEL = object()


def _cdiv(a, b):
    return (a + b - 1) // b


def _next_pow2(n):
    n = int(n)
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


class _StubKernel:
    """Stand-in for a Triton kernel that records every launch."""

    def __init__(self, name=None):
        self.name = name
        self.calls = []

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.calls.append((grid, args, kwargs))
            return None

        return launch

    def __call__(self, *args, **kwargs):
        self.calls.append((None, args, kwargs))
        return None


class _FakeTriton(types.ModuleType):
    def __init__(self):
        super().__init__("triton")
        self.cdiv = _cdiv
        self.next_power_of_2 = _next_pow2
        self.Config = lambda *a, **k: types.SimpleNamespace(args=a, kwargs=k)

    def jit(self, *args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return _StubKernel(getattr(args[0], "__name__", "kernel"))

        def decorator(fn):
            return _StubKernel(getattr(fn, "__name__", "kernel"))

        return decorator

    def heuristics(self, *args, **kwargs):
        return lambda fn: fn

    def autotune(self, *args, **kwargs):
        return lambda fn: fn


def _default_prepare_indices(cu_seqlens, chunk_size):
    return torch.arange(3, dtype=torch.int32)


def _default_prepare_offsets(cu_seqlens, chunk_size):
    return torch.zeros(max(len(cu_seqlens) - 1, 1), dtype=torch.int32)


def _make_base_modules():
    fake_triton = _FakeTriton()

    class _PermissiveTL:
        """Any ``tl.<name>`` resolves to an inert sentinel (annotations only)."""

        def __getattr__(self, _name):
            return _TL_SENTINEL

        def __call__(self, *a, **k):
            return _TL_SENTINEL

    fake_tl = _PermissiveTL()
    exp2 = getattr(torch, "exp2", torch.exp)
    return {
        "triton": {
            "jit": fake_triton.jit,
            "heuristics": fake_triton.heuristics,
            "autotune": fake_triton.autotune,
            "Config": fake_triton.Config,
            "cdiv": fake_triton.cdiv,
            "next_power_of_2": fake_triton.next_power_of_2,
        },
        "triton.language": {},
        "vllm": {},
        "vllm.triton_utils": {"tl": fake_tl, "triton": fake_triton},
        "vllm.third_party": {},
        "vllm.third_party.flash_linear_attention": {},
        "vllm.third_party.flash_linear_attention.ops": {},
        "vllm.third_party.flash_linear_attention.ops.index": {
            "prepare_chunk_indices": _default_prepare_indices,
            "prepare_chunk_offsets": _default_prepare_offsets,
        },
        "vllm.third_party.flash_linear_attention.ops.op": {
            "exp": torch.exp,
            "exp2": exp2,
        },
        "vllm.third_party.flash_linear_attention.ops.utils": {
            "FLA_CHUNK_SIZE": 64,
            "use_cuda_graph": False,
            "check_shared_mem": lambda: False,
            "is_nvidia_hopper": False,
        },
        "vllm_fl": {},
        "vllm_fl.patches": {},
        "vllm_fl.patches.triton_kernel": {"patch_kernel_launch_meta": lambda *a, **k: None},
    }


@contextlib.contextmanager
def fake_vllm_env(extra=None):
    extra = extra or {}
    saved = {}
    created = {}

    def _install(name, attrs):
        if name not in created:
            if name not in saved:
                saved[name] = sys.modules.get(name, _MISSING)
            mod = types.ModuleType(name)
            mod.__path__ = []
            sys.modules[name] = mod
            created[name] = mod
        for key, value in (attrs or {}).items():
            setattr(created[name], key, value)
        return created[name]

    try:
        for full_name, attrs in _make_base_modules().items():
            _install(full_name, attrs)
        for full_name, attrs in extra.items():
            if full_name in created:
                _install(full_name, attrs)
                continue
            parts = full_name.split(".")
            for depth in range(1, len(parts) + 1):
                prefix = ".".join(parts[:depth])
                _install(prefix, attrs if prefix == full_name else None)
        yield created
    finally:
        for name, previous in saved.items():
            if previous is _MISSING:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


def _load_modules():
    package = types.ModuleType(_PKG)
    package.__path__ = [str(IMPL_DIR)]
    package.__package__ = _PKG
    sys.modules[_PKG] = package

    loaded = {}
    order = [
        "fla_grid",
        "rotary",
        "rms_kernel",
        "bilinear_pos_embed",
        "chunk_o_gcu",
        "fused_sigmoid_gating_gcu",
        "chunk_delta_h",
        "fused_recurrent_packed_decode",
        "causal_conv1d",
    ]
    for name in order:
        full = f"{_PKG}.{name}"
        spec = __import__("importlib.util", fromlist=["util"]).spec_from_file_location(
            full, IMPL_DIR / f"{name}.py"
        )
        mod = __import__("importlib.util", fromlist=["util"]).module_from_spec(spec)
        sys.modules[full] = mod
        spec.loader.exec_module(mod)
        loaded[name] = mod
    return loaded


with fake_vllm_env():
    MODS = _load_modules()


def _mod(name):
    return MODS[name]


def _raise_import(name, *args, **kwargs):
    if name == "vllm" or name.startswith("vllm."):
        raise ImportError("blocked for coverage test")
    return _ORIGINAL_IMPORT(name, *args, **kwargs)


_ORIGINAL_IMPORT = builtins.__import__


# ---------------------------------------------------------------------------
# fla_grid
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "programs,expected",
    [(1, (1, 1)), (65535, (65535, 1)), (65536, (65535, 2))],
)
def test_gcu_flat_grid_valid(monkeypatch, programs, expected):
    mod = _mod("fla_grid")
    assert mod.gcu_flat_grid(programs) == expected


@pytest.mark.parametrize("programs", [0, -3, 65535 * 256])
def test_gcu_flat_grid_invalid(monkeypatch, programs):
    mod = _mod("fla_grid")
    with pytest.raises(ValueError):
        mod.gcu_flat_grid(programs)


def test_apply_fla_grid_patch_with_model(monkeypatch):
    mod = _mod("fla_grid")
    extra = {
        "vllm.third_party.flash_linear_attention.ops.chunk": {"chunk_fwd_o": "old"},
        "vllm.third_party.flash_linear_attention.ops.chunk_o": {"chunk_fwd_o": "old"},
        "vllm.third_party.flash_linear_attention.ops.fused_sigmoid_gating": {
            "fused_sigmoid_gating_delta_rule_update": "old"
        },
        "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn": {
            "fused_sigmoid_gating_delta_rule_update": "old"
        },
    }
    with fake_vllm_env(extra) as mods:
        mod.apply_fla_grid_gcu_patch()
        chunk_o = mods["vllm.third_party.flash_linear_attention.ops.chunk_o"]
        chunk = mods["vllm.third_party.flash_linear_attention.ops.chunk"]
        gating = mods["vllm.third_party.flash_linear_attention.ops.fused_sigmoid_gating"]
        ops = mods["vllm.third_party.flash_linear_attention.ops"]
        model = mods["vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn"]

        chunk_mod = _mod("chunk_o_gcu")
        gate_mod = _mod("fused_sigmoid_gating_gcu")
        assert chunk_o.chunk_fwd_o is chunk_mod.chunk_fwd_o
        assert chunk.chunk_fwd_o is chunk_mod.chunk_fwd_o
        assert gating.fused_sigmoid_gating_delta_rule_update is (
            gate_mod.fused_sigmoid_gating_delta_rule_update
        )
        assert ops.fused_sigmoid_gating_delta_rule_update is (
            gate_mod.fused_sigmoid_gating_delta_rule_update
        )
        assert model.fused_sigmoid_gating_delta_rule_update is (
            gate_mod.fused_sigmoid_gating_delta_rule_update
        )


def test_apply_fla_grid_patch_without_model(monkeypatch):
    mod = _mod("fla_grid")
    extra = {
        "vllm.third_party.flash_linear_attention.ops.chunk": {"chunk_fwd_o": "old"},
        "vllm.third_party.flash_linear_attention.ops.chunk_o": {"chunk_fwd_o": "old"},
        "vllm.third_party.flash_linear_attention.ops.fused_sigmoid_gating": {
            "fused_sigmoid_gating_delta_rule_update": "old"
        },
    }
    with fake_vllm_env(extra) as mods:
        monkeypatch.delitem(
            sys.modules,
            "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn",
            raising=False,
        )
        mod.apply_fla_grid_gcu_patch()
        chunk_mod = _mod("chunk_o_gcu")
        assert mods["vllm.third_party.flash_linear_attention.ops.chunk_o"].chunk_fwd_o is (
            chunk_mod.chunk_fwd_o
        )


# ---------------------------------------------------------------------------
# rotary
# ---------------------------------------------------------------------------


def _rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _rotate_interleaved(x):
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def test_rotary_dim4_1d_pos_equal_rotary_dim():
    mod = _mod("rotary")
    q = torch.randn(2, 3, 2, 8)
    k = torch.randn(2, 3, 2, 8)
    cos = torch.randn(5, 8)
    sin = torch.randn(5, 8)
    pos = torch.tensor([0, 1, 2])
    qe, ke = mod.rotary_embedding_gcu(None, q, k, cos, sin, pos)
    cs = cos[pos].view(1, 3, 1, 8)
    ss = sin[pos].view(1, 3, 1, 8)
    assert qe.shape == q.shape
    assert ke.shape == k.shape
    torch.testing.assert_close(qe, q * cs + _rotate_half(q) * ss)
    torch.testing.assert_close(ke, k * cs + _rotate_half(k) * ss)


def test_rotary_dim3_1d_pos():
    mod = _mod("rotary")
    q = torch.randn(2, 4, 8)
    k = torch.randn(2, 4, 8)
    cos = torch.randn(5, 8)
    sin = torch.randn(5, 8)
    pos = torch.tensor([0, 3])
    qe, ke = mod.rotary_embedding_gcu(None, q, k, cos, sin, pos)
    cs = cos[pos].unsqueeze(1)
    ss = sin[pos].unsqueeze(1)
    torch.testing.assert_close(qe, q * cs + _rotate_half(q) * ss)
    torch.testing.assert_close(ke, k * cs + _rotate_half(k) * ss)


def test_rotary_2d_position_ids():
    mod = _mod("rotary")
    q = torch.randn(2, 3, 3, 8)
    k = torch.randn(2, 3, 3, 8)
    cos = torch.randn(5, 8)
    sin = torch.randn(5, 8)
    pos = torch.tensor([[0, 1, 2], [3, 4, 1]])
    qe, ke = mod.rotary_embedding_gcu(None, q, k, cos, sin, pos)
    cs = cos[pos].unsqueeze(1)
    ss = sin[pos].unsqueeze(1)
    torch.testing.assert_close(qe, q * cs + _rotate_half(q) * ss)
    torch.testing.assert_close(ke, k * cs + _rotate_half(k) * ss)


def test_rotary_duplicates_when_rotary_dim_differs():
    mod = _mod("rotary")
    q = torch.randn(2, 3, 2, 8)
    k = torch.randn(2, 3, 2, 8)
    cos = torch.randn(5, 4)
    sin = torch.randn(5, 4)
    pos = torch.tensor([0, 1, 2])
    qe, ke = mod.rotary_embedding_gcu(None, q, k, cos, sin, pos)
    cs = torch.cat([cos[pos], cos[pos]], dim=-1).unsqueeze(1)
    ss = torch.cat([sin[pos], sin[pos]], dim=-1).unsqueeze(1)
    assert qe.shape == q.shape
    torch.testing.assert_close(qe, q * cs + _rotate_half(q) * ss)
    torch.testing.assert_close(ke, k * cs + _rotate_half(k) * ss)


def test_rotary_interleaved_branch():
    mod = _mod("rotary")
    q = torch.randn(1, 2, 2, 8)
    k = torch.randn(1, 2, 2, 8)
    cos = torch.randn(4, 8)
    sin = torch.randn(4, 8)
    pos = torch.tensor([0, 1])
    qe, ke = mod.rotary_embedding_gcu(
        None, q, k, cos, sin, pos, rotary_interleaved=True
    )
    cs = cos[pos].view(1, 2, 1, 8)
    ss = sin[pos].view(1, 2, 1, 8)
    torch.testing.assert_close(qe, q * cs + _rotate_interleaved(q) * ss)
    torch.testing.assert_close(ke, k * cs + _rotate_interleaved(k) * ss)


# ---------------------------------------------------------------------------
# rms_kernel
# ---------------------------------------------------------------------------


def test_rms_launch_without_residual(monkeypatch):
    mod = _mod("rms_kernel")
    stub = _StubKernel("rms")
    monkeypatch.setattr(mod, "_rms_kernel", stub)
    x = torch.arange(16, dtype=torch.float32).reshape(2, 8)
    weight = torch.ones(8)
    output, summed = mod.launch(x, weight, None, 8, 1e-6, True)
    assert output.shape == x.shape
    assert summed.shape == x.shape
    grid, args, kwargs = stub.calls[0]
    assert grid == (1,)
    assert args[0] is x
    assert args[1] is weight
    assert args[2] is x
    assert args[5] == 8
    assert args[6] == 8
    assert args[7] == 1e-6
    assert args[8] is False
    assert args[9] is True
    assert kwargs == {
        "BLOCK": 8,
        "ROWS": 2,
        "num_warps": 4,
        "enable_fp_fusion": False,
    }


def test_rms_launch_with_residual(monkeypatch):
    mod = _mod("rms_kernel")
    stub = _StubKernel("rms")
    monkeypatch.setattr(mod, "_rms_kernel", stub)
    x = torch.zeros(3, 6)
    residual = torch.ones(3, 6)
    weight = torch.ones(6)
    output, _ = mod.launch(x, weight, residual, 6, 1e-5, False)
    grid, args, kwargs = stub.calls[0]
    assert grid == (1,)
    assert args[2] is residual
    assert args[8] is True
    assert args[9] is False
    assert kwargs["BLOCK"] == 8
    assert kwargs["ROWS"] == 3
    assert output.shape == x.shape


def test_rms_launch_rows_branch(monkeypatch):
    mod = _mod("rms_kernel")
    stub = _StubKernel("rms")
    monkeypatch.setattr(mod, "_rms_kernel", stub)
    x = torch.empty(0, 8)
    output, summed = mod.launch(x, torch.ones(8), None, 8, 1e-6, True)
    assert stub.calls == []
    assert output.shape == (0, 8)
    assert summed.shape == (0, 8)


# ---------------------------------------------------------------------------
# bilinear_pos_embed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "t,h,w,m,num_grid,hidden,expected_grid",
    [
        (1, 2, 2, 2, 2, 8, (4, 1)),
        (1, 1, 1, 1, 2, 4, (1, 1)),
        (65536, 1, 1, 1, 2, 4, (65535, 2)),
    ],
)
def test_bilinear_launch_grid_and_args(
    monkeypatch, t, h, w, m, num_grid, hidden, expected_grid
):
    mod = _mod("bilinear_pos_embed")
    stub = _StubKernel("bilinear")
    monkeypatch.setattr(mod, "_bilinear_pos_embed_kernel_gcu", stub)
    embed = torch.zeros(num_grid * num_grid, hidden)
    out = mod.triton_pos_embed_interpolate_gcu(
        embed, t, h, w, num_grid, m, torch.float32
    )
    assert out.shape == (t * h * w, hidden)
    assert out.dtype == torch.float32
    grid, args, _ = stub.calls[0]
    assert grid == expected_grid
    assert args[0] is embed
    assert args[1] is out
    assert args[2] == h
    assert args[3] == w
    assert args[8] == hidden
    assert args[9] == _next_pow2(hidden)
    assert args[10] == t * h * w


def test_bilinear_unit_dim_scales_are_zero(monkeypatch):
    mod = _mod("bilinear_pos_embed")
    stub = _StubKernel("bilinear")
    monkeypatch.setattr(mod, "_bilinear_pos_embed_kernel_gcu", stub)
    embed = torch.zeros(4, 4)
    mod.triton_pos_embed_interpolate_gcu(embed, 2, 1, 1, 2, 1, torch.float32)
    _, args, _ = stub.calls[0]
    assert args[4] == 0.0
    assert args[5] == 0.0


def test_bilinear_requires_divisible_dims(monkeypatch):
    mod = _mod("bilinear_pos_embed")
    stub = _StubKernel("bilinear")
    monkeypatch.setattr(mod, "_bilinear_pos_embed_kernel_gcu", stub)
    with pytest.raises(AssertionError):
        mod.triton_pos_embed_interpolate_gcu(
            torch.zeros(4, 4), 1, 3, 2, 2, 2, torch.float32
        )
    assert stub.calls == []


def test_bilinear_patch_requires_gcu(monkeypatch):
    mod = _mod("bilinear_pos_embed")
    monkeypatch.setattr(mod, "_patched", False)
    monkeypatch.delattr(torch, "gcu", raising=False)
    monkeypatch.setattr(builtins, "__import__", _raise_import)
    mod.apply_bilinear_pos_embed_gcu_patch()
    assert mod._patched is False


def test_bilinear_patch_skips_when_unavailable(monkeypatch):
    mod = _mod("bilinear_pos_embed")
    monkeypatch.setattr(mod, "_patched", False)
    monkeypatch.setattr(
        torch, "gcu", types.SimpleNamespace(is_available=lambda: False), raising=False
    )
    monkeypatch.setattr(builtins, "__import__", _raise_import)
    mod.apply_bilinear_pos_embed_gcu_patch()
    assert mod._patched is False


def test_bilinear_patch_skips_without_triton(monkeypatch):
    mod = _mod("bilinear_pos_embed")
    monkeypatch.setattr(mod, "_patched", False)
    monkeypatch.setattr(
        torch, "gcu", types.SimpleNamespace(is_available=lambda: True), raising=False
    )
    extra = {
        "vllm.model_executor.models.qwen3_vl": {
            "HAS_TRITON": False,
            "triton_pos_embed_interpolate": "orig",
        }
    }
    with fake_vllm_env(extra) as mods:
        mod.apply_bilinear_pos_embed_gcu_patch()
        assert mod._patched is False
        assert mods["vllm.model_executor.models.qwen3_vl"].triton_pos_embed_interpolate == "orig"


def test_bilinear_patch_success(monkeypatch):
    mod = _mod("bilinear_pos_embed")
    monkeypatch.setattr(mod, "_patched", False)
    monkeypatch.setattr(
        torch, "gcu", types.SimpleNamespace(is_available=lambda: True), raising=False
    )
    extra = {
        "vllm.model_executor.models.qwen3_vl": {
            "HAS_TRITON": True,
            "triton_pos_embed_interpolate": "orig",
            "_bilinear_pos_embed_kernel": "orig",
        }
    }
    with fake_vllm_env(extra) as mods:
        mod.apply_bilinear_pos_embed_gcu_patch()
        qwen = mods["vllm.model_executor.models.qwen3_vl"]
        assert mod._patched is True
        assert qwen.triton_pos_embed_interpolate is mod.triton_pos_embed_interpolate_gcu
        assert qwen._bilinear_pos_embed_kernel is mod._bilinear_pos_embed_kernel_gcu


def test_bilinear_patch_swallows_import_error(monkeypatch):
    mod = _mod("bilinear_pos_embed")
    monkeypatch.setattr(mod, "_patched", False)
    monkeypatch.setattr(
        torch, "gcu", types.SimpleNamespace(is_available=lambda: True), raising=False
    )
    with fake_vllm_env():
        mod.apply_bilinear_pos_embed_gcu_patch()
        assert mod._patched is False


# ---------------------------------------------------------------------------
# chunk_o_gcu
# ---------------------------------------------------------------------------


def _chunk_o_inputs(dtype=torch.float32):
    q = torch.zeros(1, 4, 2, 8, dtype=dtype)
    k = torch.zeros(1, 4, 2, 8, dtype=dtype)
    v = torch.zeros(1, 4, 2, 8, dtype=dtype)
    h = torch.zeros(1, 1, 2, 8, 8, dtype=dtype)
    return q, k, v, h


def test_chunk_fwd_o_basic(monkeypatch):
    mod = _mod("chunk_o_gcu")
    stub = _StubKernel("chunk_o")
    monkeypatch.setattr(mod, "chunk_fwd_kernel_o", stub)
    q, k, v, h = _chunk_o_inputs()
    out = mod.chunk_fwd_o(q, k, v, h)
    assert out.shape == v.shape
    grid, args, kwargs = stub.calls[0]
    assert args[8] == pytest.approx(8 ** -0.5)
    assert kwargs["GCU_NT"] == 1
    assert kwargs["GCU_BH"] == 2
    assert kwargs["H"] == 2
    assert kwargs["Hg"] == 2
    assert kwargs["K"] == 8
    assert kwargs["V"] == 8
    assert kwargs["BT"] == 64
    assert grid({"BV": 32}) == (2, 1)


def test_chunk_fwd_o_scale_and_core_attn_out(monkeypatch):
    mod = _mod("chunk_o_gcu")
    stub = _StubKernel("chunk_o")
    monkeypatch.setattr(mod, "chunk_fwd_kernel_o", stub)
    q, k, v, h = _chunk_o_inputs()
    core = torch.empty(v.numel() + 7, dtype=v.dtype)
    out = mod.chunk_fwd_o(q, k, v, h, scale=0.25, core_attn_out=core)
    assert out.shape == v.shape
    assert out.data_ptr() == core.view(-1).data_ptr()
    _, args, _ = stub.calls[0]
    assert args[8] == 0.25


def test_chunk_fwd_o_core_attn_out_too_small(monkeypatch):
    mod = _mod("chunk_o_gcu")
    monkeypatch.setattr(mod, "chunk_fwd_kernel_o", _StubKernel("chunk_o"))
    q, k, v, h = _chunk_o_inputs()
    with pytest.raises(AssertionError):
        mod.chunk_fwd_o(q, k, v, h, core_attn_out=torch.empty(3))


def test_chunk_fwd_o_varlen(monkeypatch):
    mod = _mod("chunk_o_gcu")
    stub = _StubKernel("chunk_o")
    monkeypatch.setattr(mod, "chunk_fwd_kernel_o", stub)
    calls = []

    def fake_indices(cu_seqlens, chunk_size):
        calls.append((cu_seqlens, chunk_size))
        return torch.arange(3, dtype=torch.int32)

    monkeypatch.setattr(mod, "prepare_chunk_indices", fake_indices)
    q, k, v, h = _chunk_o_inputs()
    cu = torch.tensor([0, 2, 4])
    out = mod.chunk_fwd_o(q, k, v, h, cu_seqlens=cu)
    assert out.shape == v.shape
    assert len(calls) == 1
    assert calls[0][0] is cu
    assert calls[0][1] == 64
    _, args, kwargs = stub.calls[0]
    assert args[6] is cu
    assert kwargs["GCU_NT"] == 3


# ---------------------------------------------------------------------------
# fused_sigmoid_gating_gcu
# ---------------------------------------------------------------------------


def _gate_inputs(B=1, T=2, H=2, HV=2, K=4, V=4, dtype=torch.float32):
    k = torch.randn(B, T, H, K, dtype=dtype)
    q = torch.randn(B, T, H, K, dtype=dtype)
    v = torch.randn(B, T, HV, V, dtype=dtype)
    a = torch.randn(B, T, HV, dtype=dtype)
    b = torch.randn(B, T, HV, dtype=dtype)
    A_log = torch.randn(HV, dtype=dtype)
    dt_bias = torch.randn(HV, dtype=dtype)
    init = torch.randn(B, HV, V, K, dtype=dtype)
    return q, k, v, a, b, A_log, dt_bias, init


def test_fused_gating_basic(monkeypatch):
    mod = _mod("fused_sigmoid_gating_gcu")
    stub = _StubKernel("gate")
    monkeypatch.setattr(mod, "fused_sigmoid_gating_delta_rule_update_kernel", stub)
    q, k, v, a, b, A_log, dt_bias, init = _gate_inputs()
    out, final = mod.fused_sigmoid_gating_delta_rule_update(
        A_log, a, b, dt_bias, q, k, v, initial_state=init
    )
    assert out.shape == v.shape
    assert final is init
    grid, _, kwargs = stub.calls[0]
    assert grid == (2, 1)
    assert kwargs["N"] == 1
    assert kwargs["T"] == 2
    assert kwargs["B"] == 1
    assert kwargs["HV"] == 2
    assert kwargs["K"] == 4
    assert kwargs["V"] == 4
    assert kwargs["BK"] == 4
    assert kwargs["BV"] == 4
    assert kwargs["scale"] == pytest.approx(4 ** -0.5)
    assert kwargs["INPLACE_FINAL_STATE"] is True
    assert kwargs["IS_KDA"] is False
    assert kwargs["USE_QK_L2NORM_IN_KERNEL"] is False


def test_fused_gating_scale_and_flags(monkeypatch):
    mod = _mod("fused_sigmoid_gating_gcu")
    stub = _StubKernel("gate")
    monkeypatch.setattr(mod, "fused_sigmoid_gating_delta_rule_update_kernel", stub)
    q, k, v, a, b, A_log, dt_bias, init = _gate_inputs()
    mod.fused_sigmoid_gating_delta_rule_update(
        A_log,
        a,
        b,
        dt_bias,
        q,
        k,
        v,
        beta=0.5,
        threshold=15.0,
        scale=0.5,
        initial_state=init,
        use_qk_l2norm_in_kernel=True,
        is_kda=True,
    )
    _, _, kwargs = stub.calls[0]
    assert kwargs["scale"] == 0.5
    assert kwargs["beta"] == 0.5
    assert kwargs["threshold"] == 15.0
    assert kwargs["USE_QK_L2NORM_IN_KERNEL"] is True
    assert kwargs["IS_KDA"] is True


def test_fused_gating_invalid_scale(monkeypatch):
    mod = _mod("fused_sigmoid_gating_gcu")
    monkeypatch.setattr(mod, "fused_sigmoid_gating_delta_rule_update_kernel", _StubKernel())
    q, k, v, a, b, A_log, dt_bias, init = _gate_inputs()
    with pytest.raises(AssertionError):
        mod.fused_sigmoid_gating_delta_rule_update(
            A_log, a, b, dt_bias, q, k, v, scale=0.0, initial_state=init
        )


def test_fused_gating_not_inplace(monkeypatch):
    mod = _mod("fused_sigmoid_gating_gcu")
    stub = _StubKernel("gate")
    monkeypatch.setattr(mod, "fused_sigmoid_gating_delta_rule_update_kernel", stub)
    q, k, v, a, b, A_log, dt_bias, init = _gate_inputs(T=3)
    out, final = mod.fused_sigmoid_gating_delta_rule_update(
        A_log, a, b, dt_bias, q, k, v, initial_state=init, inplace_final_state=False
    )
    assert final is not init
    assert final.shape == (3, 2, 4, 4)
    assert final.dtype == init.dtype
    _, _, kwargs = stub.calls[0]
    assert kwargs["INPLACE_FINAL_STATE"] is False


def test_fused_gating_varlen_rejects_batch(monkeypatch):
    mod = _mod("fused_sigmoid_gating_gcu")
    monkeypatch.setattr(mod, "fused_sigmoid_gating_delta_rule_update_kernel", _StubKernel())
    q, k, v, a, b, A_log, dt_bias, init = _gate_inputs(B=2)
    with pytest.raises(ValueError):
        mod.fused_sigmoid_gating_delta_rule_update(
            A_log, a, b, dt_bias, q, k, v, initial_state=init,
            cu_seqlens=torch.tensor([0, 2]),
        )


def test_fused_gating_varlen_ok(monkeypatch):
    mod = _mod("fused_sigmoid_gating_gcu")
    stub = _StubKernel("gate")
    monkeypatch.setattr(mod, "fused_sigmoid_gating_delta_rule_update_kernel", stub)
    q, k, v, a, b, A_log, dt_bias, init = _gate_inputs(B=1)
    cu = torch.tensor([0, 1, 2])
    mod.fused_sigmoid_gating_delta_rule_update(
        A_log, a, b, dt_bias, q, k, v, initial_state=init, cu_seqlens=cu
    )
    _, _, kwargs = stub.calls[0]
    assert kwargs["cu_seqlens"] is cu
    assert kwargs["N"] == 2


@pytest.mark.parametrize(
    "ssm,expected",
    [
        (None, (1, 1)),
        ("1d", None),
        ("2d", None),
    ],
)
def test_fused_gating_state_index_strides(monkeypatch, ssm, expected):
    mod = _mod("fused_sigmoid_gating_gcu")
    stub = _StubKernel("gate")
    monkeypatch.setattr(mod, "fused_sigmoid_gating_delta_rule_update_kernel", stub)
    q, k, v, a, b, A_log, dt_bias, init = _gate_inputs(B=2)
    if ssm == "1d":
        ssm = torch.arange(2, dtype=torch.long)
    elif ssm == "2d":
        ssm = torch.arange(4, dtype=torch.long).reshape(2, 2)
    mod.fused_sigmoid_gating_delta_rule_update(
        A_log, a, b, dt_bias, q, k, v, initial_state=init, ssm_state_indices=ssm
    )
    _, _, kwargs = stub.calls[0]
    got = (kwargs["stride_indices_seq"], kwargs["stride_indices_tok"])
    if expected is not None:
        assert got == expected
    elif isinstance(ssm, torch.Tensor) and ssm.ndim == 1:
        assert got == (ssm.stride(0), 1)
    else:
        assert got == ssm.stride()


# ---------------------------------------------------------------------------
# chunk_delta_h
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "V,nh_total,meta,expected",
    [
        (8, 2, {"BV": 32}, (2, 1)),
        (8192, 300, {"BV": 32}, (65535, 2)),
        (8192, 70000, {"BV": 32}, (65535, 255)),
    ],
)
def test_chunk_delta_gcu_grid(V, nh_total, meta, expected):
    mod = _mod("chunk_delta_h")
    assert mod._gcu_grid(V, nh_total, meta) == expected


def test_chunk_delta_basic(monkeypatch):
    mod = _mod("chunk_delta_h")
    stub = _StubKernel("delta_h")
    monkeypatch.setattr(mod, "chunk_gated_delta_rule_fwd_kernel_h_blockdim64_gcu", stub)
    k = torch.zeros(1, 4, 2, 8)
    u = torch.zeros(1, 4, 2, 8)
    h, v_new, final = mod.chunk_gated_delta_rule_fwd_h_gcu(k, k, u)
    assert h.shape == (1, 1, 2, 8, 8)
    assert v_new.shape == u.shape
    assert final is None
    grid, _, kwargs = stub.calls[0]
    assert grid({"BV": 32}) == (2, 1)
    assert kwargs["T"] == 4
    assert kwargs["H"] == 2
    assert kwargs["Hg"] == 2
    assert kwargs["K"] == 8
    assert kwargs["V"] == 8
    assert kwargs["BT"] == 64
    assert kwargs["NH_TOTAL"] == 2
    assert kwargs["USE_EXP2"] is False


def test_chunk_delta_flags(monkeypatch):
    mod = _mod("chunk_delta_h")
    stub = _StubKernel("delta_h")
    monkeypatch.setattr(mod, "chunk_gated_delta_rule_fwd_kernel_h_blockdim64_gcu", stub)
    k = torch.zeros(2, 4, 2, 8)
    u = torch.zeros(2, 4, 2, 8)
    g = torch.zeros(2, 4, 2)
    gk = torch.zeros(2, 4, 2, 8)
    init = torch.zeros(2, 2, 8, 8)
    h, v_new, final = mod.chunk_gated_delta_rule_fwd_h_gcu(
        k, k, u, g=g, gk=gk, initial_state=init,
        output_final_state=True, save_new_value=False, use_exp2=True,
    )
    assert v_new is None
    assert final.shape == (2, 2, 8, 8)
    assert final.dtype == torch.float32
    _, _, kwargs = stub.calls[0]
    assert kwargs["g"] is g
    assert kwargs["gk"] is gk
    assert kwargs["h0"] is init
    assert kwargs["ht"] is final
    assert kwargs["USE_EXP2"] is True
    assert kwargs["NH_TOTAL"] == 4


def test_chunk_delta_rejects_large_head_dim(monkeypatch):
    mod = _mod("chunk_delta_h")
    monkeypatch.setattr(mod, "chunk_gated_delta_rule_fwd_kernel_h_blockdim64_gcu", _StubKernel())
    k = torch.zeros(1, 4, 2, 512)
    u = torch.zeros(1, 4, 2, 8)
    with pytest.raises(AssertionError):
        mod.chunk_gated_delta_rule_fwd_h_gcu(k, k, u)


def test_chunk_delta_varlen_prepares_indices(monkeypatch):
    mod = _mod("chunk_delta_h")
    stub = _StubKernel("delta_h")
    monkeypatch.setattr(mod, "chunk_gated_delta_rule_fwd_kernel_h_blockdim64_gcu", stub)
    idx_calls, off_calls = [], []

    def fake_indices(cu, bt):
        idx_calls.append((cu, bt))
        return torch.arange(3, dtype=torch.int32)

    def fake_offsets(cu, bt):
        off_calls.append((cu, bt))
        return torch.tensor([0, 1], dtype=torch.int32)

    monkeypatch.setattr(mod, "prepare_chunk_indices", fake_indices)
    monkeypatch.setattr(mod, "prepare_chunk_offsets", fake_offsets)
    k = torch.zeros(1, 4, 2, 8)
    u = torch.zeros(1, 4, 2, 8)
    cu = torch.tensor([0, 2, 4])
    h, _, _ = mod.chunk_gated_delta_rule_fwd_h_gcu(k, k, u, cu_seqlens=cu)
    assert h.shape == (1, 3, 2, 8, 8)
    assert len(idx_calls) == 1
    assert idx_calls[0][0] is cu
    assert idx_calls[0][1] == 64
    assert len(off_calls) == 1
    assert off_calls[0][0] is cu
    assert off_calls[0][1] == 64
    _, _, kwargs = stub.calls[0]
    assert kwargs["NH_TOTAL"] == 4


def test_chunk_delta_varlen_precomputed(monkeypatch):
    mod = _mod("chunk_delta_h")
    stub = _StubKernel("delta_h")
    monkeypatch.setattr(mod, "chunk_gated_delta_rule_fwd_kernel_h_blockdim64_gcu", stub)
    monkeypatch.setattr(
        mod, "prepare_chunk_indices", lambda *a, **k: (_ for _ in ()).throw(AssertionError()),
    )
    monkeypatch.setattr(
        mod, "prepare_chunk_offsets", lambda *a, **k: (_ for _ in ()).throw(AssertionError()),
    )
    k = torch.zeros(1, 4, 2, 8)
    u = torch.zeros(1, 4, 2, 8)
    cu = torch.tensor([0, 2, 4])
    chunk_indices = torch.arange(2 * 2, dtype=torch.int32).reshape(2, 2)
    chunk_offsets = torch.tensor([0, 1], dtype=torch.int32)
    h, _, _ = mod.chunk_gated_delta_rule_fwd_h_gcu(
        k, k, u, cu_seqlens=cu,
        chunk_indices=chunk_indices, chunk_offsets=chunk_offsets,
    )
    assert h.shape == (1, 2, 2, 8, 8)


def test_chunk_delta_patch_requires_gcu(monkeypatch):
    mod = _mod("chunk_delta_h")
    monkeypatch.setattr(mod, "_patched", False)
    monkeypatch.delattr(torch, "gcu", raising=False)
    monkeypatch.setattr(builtins, "__import__", _raise_import)
    mod.apply_chunk_delta_h_gcu_patch()
    assert mod._patched is False


def test_chunk_delta_patch_skips_when_unavailable(monkeypatch):
    mod = _mod("chunk_delta_h")
    monkeypatch.setattr(mod, "_patched", False)
    monkeypatch.setattr(
        torch, "gcu", types.SimpleNamespace(is_available=lambda: False), raising=False
    )
    monkeypatch.setattr(builtins, "__import__", _raise_import)
    mod.apply_chunk_delta_h_gcu_patch()
    assert mod._patched is False


def test_chunk_delta_patch_already_patched(monkeypatch):
    mod = _mod("chunk_delta_h")
    monkeypatch.setattr(mod, "_patched", True)
    monkeypatch.setattr(
        torch, "gcu", types.SimpleNamespace(is_available=lambda: True), raising=False
    )
    monkeypatch.setattr(builtins, "__import__", _raise_import)
    mod.apply_chunk_delta_h_gcu_patch()
    assert mod._patched is True


def test_chunk_delta_patch_success(monkeypatch):
    mod = _mod("chunk_delta_h")
    monkeypatch.setattr(mod, "_patched", False)
    monkeypatch.setattr(
        torch, "gcu", types.SimpleNamespace(is_available=lambda: True), raising=False
    )
    extra = {
        "vllm.third_party.flash_linear_attention.ops.chunk": {
            "chunk_gated_delta_rule_fwd_h": "old"
        },
        "vllm.third_party.flash_linear_attention.ops.chunk_delta_h": {
            "chunk_gated_delta_rule_fwd_h": "old"
        },
        "vllm.third_party.flash_linear_attention.ops.kda": {
            "chunk_gated_delta_rule_fwd_h": "old"
        },
    }
    with fake_vllm_env(extra) as mods:
        mod.apply_chunk_delta_h_gcu_patch()
        assert mod._patched is True
        for name in (
            "vllm.third_party.flash_linear_attention.ops.chunk",
            "vllm.third_party.flash_linear_attention.ops.chunk_delta_h",
            "vllm.third_party.flash_linear_attention.ops.kda",
        ):
            assert mods[name].chunk_gated_delta_rule_fwd_h is (
                mod.chunk_gated_delta_rule_fwd_h_gcu
            )


def test_chunk_delta_patch_swallows_import_error(monkeypatch):
    mod = _mod("chunk_delta_h")
    monkeypatch.setattr(mod, "_patched", False)
    monkeypatch.setattr(
        torch, "gcu", types.SimpleNamespace(is_available=lambda: True), raising=False
    )
    with fake_vllm_env():
        mod.apply_chunk_delta_h_gcu_patch()
        assert mod._patched is False


# ---------------------------------------------------------------------------
# fused_recurrent_packed_decode
# ---------------------------------------------------------------------------


def _packed_base():
    B, HV, V, K, H = 2, 2, 4, 4, 2
    qkv = 2 * H * K + HV * V
    return {
        "mixed_qkv": torch.randn(B, qkv),
        "a": torch.randn(B, HV),
        "b": torch.randn(B, HV),
        "A_log": torch.randn(HV),
        "dt_bias": torch.randn(HV),
        "scale": 1.0,
        "initial_state": torch.randn(B, HV, V, K),
        "out": torch.zeros(B, 1, HV, V),
        "ssm_state_indices": torch.arange(B, dtype=torch.long),
        "use_qk_l2norm_in_kernel": False,
    }


def _mut_mixed_ndim(d):
    return {**d, "mixed_qkv": d["mixed_qkv"][0].clone()}


def _mut_mixed_stride(d):
    mq = d["mixed_qkv"]
    return {**d, "mixed_qkv": torch.empty(mq.shape[1], mq.shape[0]).t()}


def _mut_a_ndim(d):
    return {**d, "a": d["a"][0].clone()}


def _mut_a_stride(d):
    a = d["a"]
    return {**d, "a": torch.empty(a.shape[1], a.shape[0]).t()}


def _mut_alog_ndim(d):
    return {**d, "A_log": d["A_log"].reshape(1, -1)}


def _mut_alog_stride(d):
    return {**d, "A_log": torch.empty(2 * d["A_log"].numel())[::2]}


def _mut_ssm_ndim(d):
    return {**d, "ssm_state_indices": d["ssm_state_indices"].reshape(1, -1)}


def _mut_out_noncontig(d):
    out = d["out"]
    big = torch.empty(out.shape[0], 1, out.shape[2], out.shape[3] * 2)
    return {**d, "out": big[..., : out.shape[3]]}


def _mut_device(d):
    return {**d, "dt_bias": torch.empty(d["dt_bias"].numel(), device="meta")}


def _mut_batch(d):
    return {**d, "a": torch.randn(d["mixed_qkv"].shape[0] + 1, d["a"].shape[1])}


def _mut_ssm_len(d):
    return {**d, "ssm_state_indices": torch.arange(d["mixed_qkv"].shape[0] + 1)}


def _mut_init_ndim(d):
    s = d["initial_state"]
    return {**d, "initial_state": torch.randn(*s.shape[-3:])}


def _mut_init_stride(d):
    s = d["initial_state"]
    return {**d, "initial_state": torch.empty(s.shape[0], s.shape[1], s.shape[3], s.shape[2]).transpose(-1, -2)}


def _mut_a_hv(d):
    return {**d, "a": torch.randn(d["a"].shape[0], d["a"].shape[1] + 1)}


def _mut_alog_numel(d):
    return {**d, "A_log": torch.randn(d["A_log"].numel() + 1)}


def _mut_out_shape(d):
    o = d["out"]
    return {**d, "out": torch.zeros(o.shape[0], 1, o.shape[2], o.shape[3] + 1)}


def _mut_qk_nonpositive(d):
    mq = d["mixed_qkv"]
    hv_v = d["initial_state"].shape[1] * d["initial_state"].shape[2]
    return {**d, "mixed_qkv": torch.randn(mq.shape[0], hv_v - 1)}


def _mut_qk_odd(d):
    mq = d["mixed_qkv"]
    hv_v = d["initial_state"].shape[1] * d["initial_state"].shape[2]
    return {**d, "mixed_qkv": torch.randn(mq.shape[0], hv_v + 1)}


def _mut_qdim_not_divisible(d):
    mq = d["mixed_qkv"]
    hv_v = d["initial_state"].shape[1] * d["initial_state"].shape[2]
    return {**d, "mixed_qkv": torch.randn(mq.shape[0], hv_v + 6)}


def _mut_bad_head_ratio(d):
    mq = d["mixed_qkv"]
    hv_v = d["initial_state"].shape[1] * d["initial_state"].shape[2]
    return {**d, "mixed_qkv": torch.randn(mq.shape[0], hv_v + 24)}


@pytest.mark.parametrize(
    "mutator",
    [
        _mut_mixed_ndim,
        _mut_mixed_stride,
        _mut_a_ndim,
        _mut_a_stride,
        _mut_alog_ndim,
        _mut_alog_stride,
        _mut_ssm_ndim,
        _mut_out_noncontig,
        _mut_device,
        _mut_batch,
        _mut_ssm_len,
        _mut_init_ndim,
        _mut_init_stride,
        _mut_a_hv,
        _mut_alog_numel,
        _mut_out_shape,
        _mut_qk_nonpositive,
        _mut_qk_odd,
        _mut_qdim_not_divisible,
        _mut_bad_head_ratio,
    ],
    ids=lambda fn: fn.__name__,
)
def test_packed_decode_validation(monkeypatch, mutator):
    mod = _mod("fused_recurrent_packed_decode")
    stub = _StubKernel("packed")
    monkeypatch.setattr(mod, "fused_recurrent_gated_delta_rule_packed_decode_kernel_gcu", stub)
    params = mutator(_packed_base())
    with pytest.raises(ValueError):
        mod.fused_recurrent_gated_delta_rule_packed_decode_gcu(**params)
    assert stub.calls == []


@pytest.mark.parametrize(
    "n_v,nh_total,expected",
    [(1, 100, (100, 1)), (256, 300, (65535, 2)), (256, 70000, (65535, 255))],
)
def test_packed_decode_gcu_grid(n_v, nh_total, expected):
    mod = _mod("fused_recurrent_packed_decode")
    assert mod._gcu_grid(n_v, nh_total) == expected


@pytest.mark.parametrize("l2norm", [False, True])
def test_packed_decode_success(monkeypatch, l2norm):
    mod = _mod("fused_recurrent_packed_decode")
    stub = _StubKernel("packed")
    monkeypatch.setattr(mod, "fused_recurrent_gated_delta_rule_packed_decode_kernel_gcu", stub)
    params = _packed_base()
    params["use_qk_l2norm_in_kernel"] = l2norm
    out, init = mod.fused_recurrent_gated_delta_rule_packed_decode_gcu(**params)
    assert out is params["out"]
    assert init is params["initial_state"]
    grid, _, kwargs = stub.calls[0]
    assert grid == (4, 1)
    assert kwargs["H"] == 2
    assert kwargs["HV"] == 2
    assert kwargs["K"] == 4
    assert kwargs["V"] == 4
    assert kwargs["BK"] == 4
    assert kwargs["BV"] == 4
    assert kwargs["NH_TOTAL"] == 4
    assert kwargs["SOFTPLUS_THRESHOLD"] == 20.0
    assert kwargs["USE_QK_L2NORM_IN_KERNEL"] is l2norm
    assert kwargs["num_warps"] == 1
    assert kwargs["num_stages"] == 3


def test_packed_decode_patch_requires_gcu(monkeypatch):
    mod = _mod("fused_recurrent_packed_decode")
    monkeypatch.setattr(mod, "_patched", False)
    monkeypatch.delattr(torch, "gcu", raising=False)
    monkeypatch.setattr(builtins, "__import__", _raise_import)
    mod.apply_fused_recurrent_packed_decode_gcu_patch()
    assert mod._patched is False


def test_packed_decode_patch_skips_when_unavailable(monkeypatch):
    mod = _mod("fused_recurrent_packed_decode")
    monkeypatch.setattr(mod, "_patched", False)
    monkeypatch.setattr(
        torch, "gcu", types.SimpleNamespace(is_available=lambda: False), raising=False
    )
    monkeypatch.setattr(builtins, "__import__", _raise_import)
    mod.apply_fused_recurrent_packed_decode_gcu_patch()
    assert mod._patched is False


def test_packed_decode_patch_already_patched(monkeypatch):
    mod = _mod("fused_recurrent_packed_decode")
    monkeypatch.setattr(mod, "_patched", True)
    monkeypatch.setattr(
        torch, "gcu", types.SimpleNamespace(is_available=lambda: True), raising=False
    )
    monkeypatch.setattr(builtins, "__import__", _raise_import)
    mod.apply_fused_recurrent_packed_decode_gcu_patch()
    assert mod._patched is True


def test_packed_decode_patch_success(monkeypatch):
    mod = _mod("fused_recurrent_packed_decode")
    monkeypatch.setattr(mod, "_patched", False)
    monkeypatch.setattr(
        torch, "gcu", types.SimpleNamespace(is_available=lambda: True), raising=False
    )
    extra = {
        "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn": {
            "fused_recurrent_gated_delta_rule_packed_decode": "old"
        },
        "vllm.third_party.flash_linear_attention.ops": {
            "fused_recurrent_gated_delta_rule_packed_decode": "old"
        },
        "vllm.third_party.flash_linear_attention.ops.fused_recurrent": {
            "fused_recurrent_gated_delta_rule_packed_decode": "old"
        },
    }
    with fake_vllm_env(extra) as mods:
        mod.apply_fused_recurrent_packed_decode_gcu_patch()
        assert mod._patched is True
        for name in (
            "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn",
            "vllm.third_party.flash_linear_attention.ops",
            "vllm.third_party.flash_linear_attention.ops.fused_recurrent",
        ):
            assert mods[name].fused_recurrent_gated_delta_rule_packed_decode is (
                mod.fused_recurrent_gated_delta_rule_packed_decode_gcu
            )


def test_packed_decode_patch_swallows_import_error(monkeypatch):
    mod = _mod("fused_recurrent_packed_decode")
    monkeypatch.setattr(mod, "_patched", False)
    monkeypatch.setattr(
        torch, "gcu", types.SimpleNamespace(is_available=lambda: True), raising=False
    )
    with fake_vllm_env():
        mod.apply_fused_recurrent_packed_decode_gcu_patch()
        assert mod._patched is False


# ---------------------------------------------------------------------------
# causal_conv1d
# ---------------------------------------------------------------------------


def test_causal_conv1d_configs():
    mod = _mod("causal_conv1d")
    assert mod.CAUSAL_CONV1D_FWD_CONFIG == {"BLOCK_N": 2048, "num_stages": 2}
    assert mod.CAUSAL_CONV1D_UPDATE_CONFIG == {"BLOCK_N": 1024}


def test_causal_conv1d_apply_patch(monkeypatch):
    mod = _mod("causal_conv1d")
    target = types.SimpleNamespace()
    imported = []
    patched = []

    def fake_import(name):
        imported.append(name)
        return target

    def fake_patch(module, kernel_name, config):
        patched.append((module, kernel_name, config))

    monkeypatch.setattr(mod, "importlib", types.SimpleNamespace(import_module=fake_import))
    monkeypatch.setattr(mod, "patch_kernel_launch_meta", fake_patch)

    mod.apply_causal_conv1d_gcu_patch()

    assert imported == ["vllm.model_executor.layers.mamba.ops.causal_conv1d"]
    assert patched[0] == (target, "_causal_conv1d_fwd_kernel", mod.CAUSAL_CONV1D_FWD_CONFIG)
    assert patched[1] == (target, "_causal_conv1d_update_kernel", mod.CAUSAL_CONV1D_UPDATE_CONFIG)


def test_module_constants():
    assert _mod("chunk_delta_h").GCU_MAX_GRID_X == 65535
    assert _mod("chunk_delta_h").GCU_MAX_GRID_YZ == 255
    assert _mod("fused_recurrent_packed_decode").GCU_MAX_GRID_YZ == 255
    assert _mod("bilinear_pos_embed").GCU_MAX_GRID_X == 65535
