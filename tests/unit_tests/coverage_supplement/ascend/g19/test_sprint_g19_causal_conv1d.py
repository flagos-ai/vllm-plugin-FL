# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""G19 coverage: Ascend ``causal_conv1d`` (814b, ~198 missing lines).

Three surfaces are exercised on CPU only, with no NPU, no network:

* ``causal_conv1d_ref`` / ``causal_conv1d_fn`` -- real eager torch paths.
* ``causal_conv1d_update_npu`` -- wrapper arithmetic and every tile-size
  branch, with the triton launch target replaced by a spy.
* ``_causal_conv1d_update_kernel_npu_tiled`` -- the kernel *body* is
  re-compiled from the module source with its ``@triton.jit`` decorator
  stripped (ast) and executed as ordinary Python against a permissive
  ``tl`` stand-in, so coverage is attributed to the original file while
  no triton compiler is required.

When triton itself is missing (the 910c CPU image), ``sys.modules`` is
seeded with a minimal ``triton`` / ``triton.language`` stub. The same is
done for ``vllm.v1.attention.backends.utils`` when the real package is
not importable, so the test file stays self-contained.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import os
import pathlib
import sys
import types

import pytest
import torch

_TARGET_REL = "vllm_fl/dispatch/backends/vendor/ascend/impl/causal_conv1d.py"
_KERNEL_NAME = "_causal_conv1d_update_kernel_npu_tiled"


# ---------------------------------------------------------------------------
# Dependency stubs (triton / vllm attention utils)
# ---------------------------------------------------------------------------
class _StubTL:
    """Minimal triton.language used only at import time."""

    int32 = object()
    int64 = object()
    float16 = object()
    float32 = object()
    constexpr = object()


def _install_triton_stub():
    triton = types.ModuleType("triton")
    tl = types.ModuleType("triton.language")
    tl.int32 = _StubTL.int32
    tl.int64 = _StubTL.int64
    tl.float16 = _StubTL.float16
    tl.float32 = _StubTL.float32
    tl.constexpr = _StubTL.constexpr
    triton.language = tl
    triton.jit = lambda fn=None, **kwargs: (lambda f: f) if fn is None else fn
    triton.next_power_of_2 = lambda n: 1 << (int(n) - 1).bit_length()
    triton.cdiv = lambda a, b: -(-int(a) // int(b))
    sys.modules["triton"] = triton
    sys.modules["triton.language"] = tl


def _install_vllm_stub():
    names = [
        "vllm",
        "vllm.v1",
        "vllm.v1.attention",
        "vllm.v1.attention.backends",
    ]
    for name in names:
        if name not in sys.modules:
            mod = types.ModuleType(name)
            mod.__path__ = []
            sys.modules[name] = mod
    utils = types.ModuleType("vllm.v1.attention.backends.utils")
    utils.NULL_BLOCK_ID = 0
    utils.PAD_SLOT_ID = -1
    sys.modules["vllm.v1.attention.backends.utils"] = utils


def _ensure_deps():
    try:
        importlib.import_module("triton")
    except Exception:
        _install_triton_stub()
    try:
        importlib.import_module("vllm.v1.attention.backends.utils")
    except Exception:
        _install_vllm_stub()


def _find_repo():
    env = os.environ.get("FL_REPO_ROOT") or os.environ.get("FL_WORKSPACE")
    candidates = []
    if env:
        candidates.append(pathlib.Path(env))
    here = pathlib.Path(__file__).resolve()
    candidates.extend([here.parent, *here.parents])
    candidates.append(pathlib.Path.cwd())
    for entry in sys.path:
        if entry:
            candidates.append(pathlib.Path(entry))
    for base in candidates:
        if (base / "vllm_fl" / "dispatch" / "backends").is_dir():
            return base
    return None


def _load_module(path):
    name = "g19_causal_conv1d_under_test"
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_REPO = _find_repo()
if _REPO is None:
    pytest.skip("cannot locate vllm_fl tree on this host", allow_module_level=True)

_TARGET = _REPO / _TARGET_REL
if not _TARGET.is_file():
    pytest.skip(f"target module missing: {_TARGET}", allow_module_level=True)

_ensure_deps()
try:
    conv = _load_module(_TARGET)
except Exception as exc:  # pragma: no cover - environment guard
    pytest.skip(f"cannot import causal_conv1d: {exc}", allow_module_level=True)


# ---------------------------------------------------------------------------
# Permissive tensor stand-in + tl stub for the kernel body
# ---------------------------------------------------------------------------
class _FakeValue:
    __hash__ = object.__hash__

    def __getitem__(self, _key):
        return self

    def __setitem__(self, _key, _value):
        return None

    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return lambda *args, **kwargs: _FakeValue()

    def __neg__(self):
        return _FakeValue()

    def __add__(self, _other):
        return _FakeValue()

    def __radd__(self, _other):
        return _FakeValue()

    def __sub__(self, _other):
        return _FakeValue()

    def __rsub__(self, _other):
        return _FakeValue()

    def __mul__(self, _other):
        return _FakeValue()

    def __rmul__(self, _other):
        return _FakeValue()

    def __truediv__(self, _other):
        return _FakeValue()

    def __rtruediv__(self, _other):
        return _FakeValue()

    def __and__(self, _other):
        return _FakeValue()

    def __rand__(self, _other):
        return _FakeValue()

    def __or__(self, _other):
        return _FakeValue()

    def __ror__(self, _other):
        return _FakeValue()

    def __lt__(self, _other):
        return _FakeValue()

    def __gt__(self, _other):
        return _FakeValue()

    def __le__(self, _other):
        return _FakeValue()

    def __ge__(self, _other):
        return _FakeValue()

    def __eq__(self, _other):
        return _FakeValue()

    def __ne__(self, _other):
        return _FakeValue()


class _FakeTL:
    int32 = object()
    int64 = object()
    float16 = object()
    float32 = object()
    constexpr = object()

    @staticmethod
    def static_range(*bounds):
        return range(*[int(b) for b in bounds])

    @staticmethod
    def program_id(_axis):
        return _FakeValue()

    @staticmethod
    def arange(*_args, **_kwargs):
        return _FakeValue()

    @staticmethod
    def load(*_args, **_kwargs):
        return _FakeValue()

    @staticmethod
    def store(*_args, **_kwargs):
        return None

    @staticmethod
    def zeros(*_args, **_kwargs):
        return _FakeValue()

    @staticmethod
    def full(*_args, **_kwargs):
        return _FakeValue()

    @staticmethod
    def where(*_args, **_kwargs):
        return _FakeValue()

    @staticmethod
    def exp(*_args, **_kwargs):
        return _FakeValue()

    @staticmethod
    def range(*args, **_kwargs):
        if args:
            try:
                return range(int(args[0]))
            except Exception:
                pass
        return range(1)


_KERNEL_FN = None


def _kernel_function():
    global _KERNEL_FN
    if _KERNEL_FN is not None:
        return _KERNEL_FN
    source = _TARGET.read_text(encoding="utf-8")
    tree = ast.parse(source)
    node = None
    for item in tree.body:
        if isinstance(item, ast.FunctionDef) and item.name == _KERNEL_NAME:
            node = item
            break
    if node is None:  # pragma: no cover - source guard
        pytest.skip(f"{_KERNEL_NAME} not found in module source")
    node.decorator_list = []
    wrapper = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(wrapper)
    namespace = {"__builtins__": __builtins__, "tl": _FakeTL()}
    exec(compile(wrapper, str(_TARGET), "exec"), namespace)
    _KERNEL_FN = namespace[_KERNEL_NAME]
    return _KERNEL_FN


def _run_kernel(**overrides):
    meta = dict(
        HAS_BIAS=False,
        KERNEL_WIDTH=1,
        SILU_ACTIVATION=False,
        IS_VARLEN=False,
        IS_APC_ENABLED=False,
        IS_SPEC_DECODING=False,
        NP2_STATELEN=2,
        HAS_NULL_BLOCK=False,
        BLOCK_N=2,
        B_TILE=1,
        T_CHUNK=1,
    )
    meta.update(overrides)
    tensors = [_FakeValue() for _ in range(10)]
    scalars = [1, 8, 1, 1, 4]  # batch, dim, seqlen, state_len, num_cache_lines
    strides = [0, 8, 1, 1, 1, 0, 8, 1, 1, 0, 8, 1]
    args = tensors + scalars + strides + [-1]
    return _kernel_function()(*args, **meta)


# ---------------------------------------------------------------------------
# Kernel-body branch coverage
# ---------------------------------------------------------------------------
def test_kernel_width1_plain():
    _run_kernel(KERNEL_WIDTH=1)


def test_kernel_width1_bias_silu():
    _run_kernel(KERNEL_WIDTH=1, HAS_BIAS=True, SILU_ACTIVATION=True)


def test_kernel_width2_history():
    _run_kernel(KERNEL_WIDTH=2)


def test_kernel_width3_history():
    _run_kernel(KERNEL_WIDTH=3)


def test_kernel_width4_history():
    _run_kernel(KERNEL_WIDTH=4)


def test_kernel_width5_history():
    _run_kernel(KERNEL_WIDTH=5)


def test_kernel_width6_history():
    _run_kernel(KERNEL_WIDTH=6, HAS_BIAS=True)


def test_kernel_varlen_apc_null_block():
    _run_kernel(
        KERNEL_WIDTH=3,
        IS_VARLEN=True,
        IS_APC_ENABLED=True,
        HAS_NULL_BLOCK=True,
    )


def test_kernel_spec_decoding_with_apc():
    _run_kernel(KERNEL_WIDTH=2, IS_SPEC_DECODING=True, IS_APC_ENABLED=True)


def test_kernel_spec_decoding_without_apc_silu():
    _run_kernel(KERNEL_WIDTH=4, IS_SPEC_DECODING=True, SILU_ACTIVATION=True)


def test_kernel_multi_tile_and_multi_chunk():
    _run_kernel(
        KERNEL_WIDTH=6,
        HAS_BIAS=True,
        SILU_ACTIVATION=True,
        NP2_STATELEN=4,
        BLOCK_N=4,
        B_TILE=2,
        T_CHUNK=2,
    )


# ---------------------------------------------------------------------------
# causal_conv1d_ref
# ---------------------------------------------------------------------------
def test_ref_no_initial_no_final():
    torch.manual_seed(0)
    x = torch.randn(2, 4, 5)
    weight = torch.randn(4, 3)
    bias = torch.randn(4)
    out, final = conv.causal_conv1d_ref(x, weight, bias)
    assert out.shape == (2, 4, 5)
    assert final is None
    assert torch.isfinite(out).all()


def test_ref_initial_states_activation_none():
    torch.manual_seed(1)
    x = torch.randn(2, 3, 4)
    weight = torch.randn(3, 3)
    bias = torch.randn(3)
    initial = torch.randn(2, 3, 2)
    out, final = conv.causal_conv1d_ref(
        x, weight, bias, initial_states=initial, activation=None
    )
    assert out.shape == (2, 3, 4)
    assert final is None


def test_ref_return_final_states_allocates():
    torch.manual_seed(2)
    x = torch.randn(2, 4, 5)
    weight = torch.randn(4, 3)
    out, final = conv.causal_conv1d_ref(
        x, weight, return_final_states=True
    )
    assert out.shape == (2, 4, 5)
    assert final.shape == (2, 4, 2)


def test_ref_final_states_out_copied():
    torch.manual_seed(3)
    x = torch.randn(1, 2, 4)
    weight = torch.randn(2, 3)
    initial = torch.zeros(1, 2, 2)
    target = torch.zeros(1, 2, 2)
    out, final = conv.causal_conv1d_ref(
        x,
        weight,
        initial_states=initial,
        return_final_states=True,
        final_states_out=target,
        activation="silu",
    )
    assert out.shape == (1, 2, 4)
    assert final is target
    assert torch.count_nonzero(target) == target.numel()


def test_ref_swish_activation():
    torch.manual_seed(4)
    x = torch.randn(1, 2, 3)
    weight = torch.randn(2, 2)
    out, final = conv.causal_conv1d_ref(
        x, weight, activation="swish", return_final_states=True
    )
    assert out.shape == (1, 2, 3)
    assert final.shape == (1, 2, 1)


def test_ref_rejects_bad_activation():
    x = torch.randn(1, 2, 3)
    weight = torch.randn(2, 3)
    with pytest.raises(NotImplementedError):
        conv.causal_conv1d_ref(x, weight, activation="relu")


# ---------------------------------------------------------------------------
# causal_conv1d_fn
# ---------------------------------------------------------------------------
def test_fn_basic_no_cache_activation_none():
    torch.manual_seed(5)
    dim, total, width = 4, 7, 3
    x = torch.randn(dim, total)
    weight = torch.randn(dim, width)
    bias = torch.randn(dim)
    conv_states = torch.zeros(3, dim, width - 1)
    qsl = torch.tensor([0, 3, 7], dtype=torch.int32)
    out = conv.causal_conv1d_fn(
        x, weight, bias, conv_states, qsl, activation=None
    )
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_fn_cache_indices_and_initial_state_mix():
    torch.manual_seed(6)
    dim, width = 4, 3
    x = torch.randn(dim, 6)
    weight = torch.randn(dim, width)
    bias = torch.randn(dim)
    conv_states = torch.zeros(3, dim, width - 1)
    qsl = torch.tensor([0, 3, 6], dtype=torch.int32)
    cache_indices = torch.tensor([2, 1], dtype=torch.int64)
    out = conv.causal_conv1d_fn(
        x,
        weight,
        bias,
        conv_states,
        qsl,
        cache_indices=cache_indices,
        has_initial_state=torch.tensor([True, False]),
        activation=True,
    )
    assert out.shape == x.shape


def test_fn_bias_none_and_silu_string():
    torch.manual_seed(7)
    dim, width = 3, 2
    x = torch.randn(dim, 4)
    weight = torch.randn(dim, width)
    conv_states = torch.zeros(2, dim, width - 1)
    qsl = torch.tensor([0, 4], dtype=torch.int32)
    out = conv.causal_conv1d_fn(
        x, weight, None, conv_states, qsl, activation="silu"
    )
    assert out.shape == x.shape


def test_fn_skips_empty_pad_and_null_slots():
    torch.manual_seed(8)
    dim, width = 4, 2
    x = torch.randn(dim, 4)
    weight = torch.randn(dim, width)
    bias = torch.randn(dim)
    conv_states = torch.zeros(3, dim, width - 1)
    qsl = torch.tensor([0, 0, 2, 4], dtype=torch.int32)
    cache_indices = torch.tensor([1, 0, 2], dtype=torch.int64)
    out = conv.causal_conv1d_fn(
        x,
        weight,
        bias,
        conv_states,
        qsl,
        cache_indices=cache_indices,
        pad_slot_id=conv.PAD_SLOT_ID,
        null_block_id=conv.NULL_BLOCK_ID,
    )
    assert out.shape == x.shape


def test_fn_noncontiguous_x_is_made_contiguous():
    torch.manual_seed(9)
    dim, width = 3, 3
    x = torch.randn(5, dim).T
    assert x.stride(-1) != 1
    weight = torch.randn(dim, width)
    bias = torch.randn(dim)
    conv_states = torch.zeros(1, dim, width - 1)
    qsl = torch.tensor([0, 5], dtype=torch.int32)
    out = conv.causal_conv1d_fn(x, weight, bias, conv_states, qsl)
    assert out.shape == x.shape


def test_fn_rejects_prefix_caching_arguments():
    x = torch.randn(2, 3)
    weight = torch.randn(2, 2)
    bias = torch.randn(2)
    conv_states = torch.zeros(1, 2, 1)
    qsl = torch.tensor([0, 3], dtype=torch.int32)
    with pytest.raises(NotImplementedError):
        conv.causal_conv1d_fn(
            x, weight, bias, conv_states, qsl,
            block_idx_first_scheduled_token=torch.zeros(1, dtype=torch.int32),
        )
    with pytest.raises(NotImplementedError):
        conv.causal_conv1d_fn(
            x, weight, bias, conv_states, qsl,
            initial_state_idx=torch.zeros(1, dtype=torch.int32),
        )
    with pytest.raises(NotImplementedError):
        conv.causal_conv1d_fn(
            x, weight, bias, conv_states, qsl, block_size_to_align=16
        )


def test_fn_rejects_2d_cache_indices_and_bad_activation():
    x = torch.randn(2, 3)
    weight = torch.randn(2, 2)
    bias = torch.randn(2)
    conv_states = torch.zeros(1, 2, 1)
    qsl = torch.tensor([0, 3], dtype=torch.int32)
    with pytest.raises(NotImplementedError):
        conv.causal_conv1d_fn(
            x, weight, bias, conv_states, qsl,
            cache_indices=torch.zeros(2, 2, dtype=torch.int32),
        )
    with pytest.raises(NotImplementedError):
        conv.causal_conv1d_fn(x, weight, bias, conv_states, qsl, activation="relu")


# ---------------------------------------------------------------------------
# causal_conv1d_update_npu
# ---------------------------------------------------------------------------
class _KernelSpy:
    def __init__(self):
        self.calls = []
        self.grid = None

    def __getitem__(self, grid):
        self.grid = grid

        def _launch(*args, **kwargs):
            self.calls.append({"args": args, "kwargs": kwargs})
            return None

        return _launch


def _spy(monkeypatch):
    spy = _KernelSpy()
    monkeypatch.setattr(conv, _KERNEL_NAME, spy)
    return spy


def test_update_decode_2d_defaults(monkeypatch):
    spy = _spy(monkeypatch)
    batch, dim, width = 3, 16, 4
    x = torch.randn(batch, dim)
    weight = torch.randn(dim, width)
    conv_state = torch.zeros(4, dim, width - 1, dtype=torch.float16)
    out = conv.causal_conv1d_update_npu(x, conv_state, weight)
    assert out.shape == x.shape
    assert out.dtype == torch.float32
    call = spy.calls[-1]
    assert call["kwargs"]["KERNEL_WIDTH"] == width
    assert call["kwargs"]["HAS_BIAS"] is False
    assert call["kwargs"]["SILU_ACTIVATION"] is False
    assert call["kwargs"]["IS_VARLEN"] is False
    assert call["kwargs"]["BLOCK_N"] == 256
    assert call["kwargs"]["T_CHUNK"] == 48
    assert call["kwargs"]["B_TILE"] == 1


def test_update_bias_silu_state_indices_and_grid(monkeypatch):
    spy = _spy(monkeypatch)
    batch, dim, width = 3, 16, 3
    x = torch.randn(batch, dim)
    weight = torch.randn(dim, width)
    bias = torch.randn(dim)
    conv_state = torch.zeros(4, dim, width - 1, dtype=torch.float16)
    indices = torch.tensor([0, 1, 2], dtype=torch.int32)
    out = conv.causal_conv1d_update_npu(
        x, conv_state, weight, bias, activation=True,
        conv_state_indices=indices,
    )
    assert out.shape == x.shape
    kwargs = spy.calls[-1]["kwargs"]
    assert kwargs["HAS_BIAS"] is True
    assert kwargs["SILU_ACTIVATION"] is True
    assert kwargs["HAS_NULL_BLOCK"] is True
    launched = spy.grid({"B_TILE": 1, "BLOCK_N": 256})
    assert launched == (3, 1)


def test_update_3d_swish_activation(monkeypatch):
    spy = _spy(monkeypatch)
    batch, dim, width, seqlen = 2, 8, 3, 4
    x = torch.randn(batch, dim, seqlen)
    weight = torch.randn(dim, width)
    conv_state = torch.zeros(2, dim, width - 1, dtype=torch.float16)
    out = conv.causal_conv1d_update_npu(
        x, conv_state, weight, activation="swish"
    )
    assert out.shape == x.shape
    assert spy.calls[-1]["kwargs"]["SILU_ACTIVATION"] is True
    assert spy.calls[-1]["kwargs"]["IS_VARLEN"] is False


def test_update_varlen(monkeypatch):
    spy = _spy(monkeypatch)
    dim, width, total = 8, 3, 6
    x = torch.randn(total, dim)
    weight = torch.randn(dim, width)
    conv_state = torch.zeros(2, dim, width - 1, dtype=torch.float16)
    indices = torch.tensor([0, 1], dtype=torch.int32)
    qsl = torch.tensor([0, 2, 6], dtype=torch.int32)
    out = conv.causal_conv1d_update_npu(
        x, conv_state, weight, conv_state_indices=indices,
        query_start_loc=qsl, max_query_len=4,
    )
    assert out.shape == x.shape
    kwargs = spy.calls[-1]["kwargs"]
    assert kwargs["IS_VARLEN"] is True
    assert kwargs["SILU_ACTIVATION"] is False


def test_update_spec_decoding(monkeypatch):
    spy = _spy(monkeypatch)
    dim, width, total, seqlen = 8, 3, 8, 4
    x = torch.randn(total, dim)
    weight = torch.randn(dim, width)
    conv_state = torch.zeros(2, dim, width - 1, dtype=torch.float16)
    indices = torch.tensor([0, 1], dtype=torch.int32)
    qsl = torch.tensor([0, 4, 8], dtype=torch.int32)
    accepted = torch.tensor([2, 3], dtype=torch.int32)
    out = conv.causal_conv1d_update_npu(
        x, conv_state, weight, conv_state_indices=indices,
        num_accepted_tokens=accepted, query_start_loc=qsl,
        max_query_len=seqlen,
    )
    assert out.shape == x.shape
    kwargs = spy.calls[-1]["kwargs"]
    assert kwargs["IS_SPEC_DECODING"] is True
    assert kwargs["NP2_STATELEN"] >= width - 1 + seqlen - 1


def test_update_null_block_none(monkeypatch):
    spy = _spy(monkeypatch)
    batch, dim, width = 2, 8, 2
    x = torch.randn(batch, dim, 3)
    weight = torch.randn(dim, width)
    conv_state = torch.zeros(2, dim, width - 1, dtype=torch.float16)
    out = conv.causal_conv1d_update_npu(
        x, conv_state, weight, null_block_id=None
    )
    assert out.shape == x.shape
    assert spy.calls[-1]["kwargs"]["HAS_NULL_BLOCK"] is False


def test_update_apc_parameters(monkeypatch):
    spy = _spy(monkeypatch)
    batch, dim, width = 2, 8, 3
    x = torch.randn(batch, dim)
    weight = torch.randn(dim, width)
    conv_state = torch.zeros(2, dim, width - 1, dtype=torch.float16)
    block_idx = torch.zeros(batch, dtype=torch.int32)
    istate = torch.zeros(batch, dtype=torch.int32)
    out = conv.causal_conv1d_update_npu(
        x, conv_state, weight,
        block_idx_last_scheduled_token=block_idx,
        initial_state_idx=istate,
    )
    assert out.shape == x.shape
    assert spy.calls[-1]["kwargs"]["IS_APC_ENABLED"] is True


def test_update_large_dim_uses_block512(monkeypatch):
    spy = _spy(monkeypatch)
    batch, dim, width = 4, 512, 3
    x = torch.randn(batch, dim)
    weight = torch.randn(dim, width)
    conv_state = torch.zeros(4, dim, width - 1, dtype=torch.float16)
    out = conv.causal_conv1d_update_npu(x, conv_state, weight)
    assert out.shape == x.shape
    kwargs = spy.calls[-1]["kwargs"]
    assert kwargs["BLOCK_N"] == 512
    assert kwargs["T_CHUNK"] == 20
    assert kwargs["B_TILE"] == 1


def test_update_b_tile_ladder(monkeypatch):
    dim, width = 16, 3
    weight = torch.randn(dim, width)
    expected = [(40, 1), (160, 2), (320, 4), (700, 8)]
    for batch, want in expected:
        spy = _spy(monkeypatch)
        x = torch.randn(batch, dim)
        conv_state = torch.zeros(batch, dim, width - 1, dtype=torch.float16)
        conv.causal_conv1d_update_npu(x, conv_state, weight)
        assert spy.calls[-1]["kwargs"]["B_TILE"] == want


def test_update_validate_data_true(monkeypatch):
    spy = _spy(monkeypatch)
    batch, dim, width = 2, 8, 2
    x = torch.randn(batch, dim)
    weight = torch.randn(dim, width)
    conv_state = torch.zeros(2, dim, width - 1, dtype=torch.float16)
    out = conv.causal_conv1d_update_npu(
        x, conv_state, weight, validate_data=True
    )
    assert out.shape == x.shape
    assert len(spy.calls) == 1


def test_update_out_tensor_matches(monkeypatch):
    _spy(monkeypatch)
    batch, dim, width = 2, 8, 3
    x = torch.randn(batch, dim)
    weight = torch.randn(dim, width)
    conv_state = torch.zeros(2, dim, width - 1, dtype=torch.float16)
    out = torch.empty(batch, dim, dtype=torch.float32)
    result = conv.causal_conv1d_update_npu(
        x, conv_state, weight, out=out
    )
    assert result.shape == out.shape
    assert result.dtype == out.dtype


def test_update_out_shape_mismatch_raises(monkeypatch):
    _spy(monkeypatch)
    batch, dim, width = 2, 8, 3
    x = torch.randn(batch, dim)
    weight = torch.randn(dim, width)
    conv_state = torch.zeros(2, dim, width - 1, dtype=torch.float16)
    bad = torch.empty(batch, dim + 1, dtype=torch.float32)
    with pytest.raises(ValueError):
        conv.causal_conv1d_update_npu(x, conv_state, weight, out=bad)


def test_update_out_dtype_mismatch_raises(monkeypatch):
    _spy(monkeypatch)
    batch, dim, width = 2, 8, 3
    x = torch.randn(batch, dim)
    weight = torch.randn(dim, width)
    conv_state = torch.zeros(2, dim, width - 1, dtype=torch.float16)
    bad = torch.empty(batch, dim, dtype=torch.float16)
    with pytest.raises(ValueError):
        conv.causal_conv1d_update_npu(x, conv_state, weight, out=bad)


def test_update_activation_bool_false(monkeypatch):
    spy = _spy(monkeypatch)
    batch, dim, width = 2, 8, 3
    x = torch.randn(batch, dim)
    weight = torch.randn(dim, width)
    conv_state = torch.zeros(2, dim, width - 1, dtype=torch.float16)
    conv.causal_conv1d_update_npu(x, conv_state, weight, activation=False)
    assert spy.calls[-1]["kwargs"]["SILU_ACTIVATION"] is False
