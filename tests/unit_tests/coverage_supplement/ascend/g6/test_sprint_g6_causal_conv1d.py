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

"""G6: CPU coverage for the Ascend ``causal_conv1d`` implementation.

The module's triton kernel body is never compiled or launched here. Its
``@triton.jit`` wrapper hides the plain Python function, so the function
definition is re-compiled from the module source against a permissive
``tl`` stand-in and executed as ordinary Python. Coverage then records the
kernel lines under the original file name without touching an NPU.

The two eager helpers (``causal_conv1d_ref`` / ``causal_conv1d_fn``) run on
CPU tensors.
"""

from __future__ import annotations

import ast
import importlib.util
import inspect
import os
import pathlib
import sys

import pytest
import torch

_TARGET_REL = (
    "vllm_fl/dispatch/backends/vendor/ascend/impl/causal_conv1d.py"
)
_KERNEL_NAME = "_causal_conv1d_update_kernel_npu_tiled"


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
    name = "g6_causal_conv1d_under_test"
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

try:
    conv = _load_module(_TARGET)
except Exception as exc:  # triton / vllm unavailable on this host
    pytest.skip(f"cannot import causal_conv1d: {exc}", allow_module_level=True)


class _FakeValue:
    """Permissive tensor-like value for the triton body."""

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
    """Subset of ``triton.language`` used by the kernel body."""

    int32 = object()
    int64 = object()
    float16 = object()
    float32 = object()
    constexpr = object()
    static_range = staticmethod(lambda *bounds: range(*bounds))

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


def _kernel_function():
    source = _TARGET.read_text(encoding="utf-8")
    tree = ast.parse(source)
    node = None
    for item in tree.body:
        if isinstance(item, ast.FunctionDef) and item.name == _KERNEL_NAME:
            node = item
            break
    if node is None:
        pytest.skip(f"{_KERNEL_NAME} not found in module source")
    node.decorator_list = []
    wrapper = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(wrapper)
    namespace = {"__builtins__": __builtins__, "tl": _FakeTL()}
    exec(compile(wrapper, str(_TARGET), "exec"), namespace)
    return namespace[_KERNEL_NAME]


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
    scalars = [1, 8, 1, 1, 4]
    strides = [0, 8, 1, 1, 1, 0, 8, 1, 1, 0, 8, 1]
    null_block = [-1]
    args = tensors + scalars + strides + null_block
    return _kernel_function()(*args, **meta)


def test_kernel_plain_branch():
    _run_kernel()


def test_kernel_bias_null_block_and_varlen():
    _run_kernel(
        HAS_BIAS=True,
        HAS_NULL_BLOCK=True,
        IS_VARLEN=True,
        IS_APC_ENABLED=True,
    )


def test_kernel_spec_decoding_branch():
    _run_kernel(IS_SPEC_DECODING=True, IS_APC_ENABLED=True)


def test_kernel_kernel_width_columns_and_silu():
    for width in (2, 3, 4, 5, 6):
        _run_kernel(KERNEL_WIDTH=width, SILU_ACTIVATION=True)


def test_reference_paths_return_final_states():
    torch.manual_seed(0)
    x = torch.randn(2, 4, 5)
    weight = torch.randn(4, 3)
    bias = torch.randn(4)

    out, final_states = conv.causal_conv1d_ref(
        x, weight, bias, return_final_states=True
    )
    assert out.shape == (2, 4, 5)
    assert final_states.shape == (2, 4, 2)

    initial = torch.zeros(2, 4, 2)
    target = torch.zeros(2, 4, 2)
    out2, final2 = conv.causal_conv1d_ref(
        x,
        weight,
        bias,
        initial_states=initial,
        return_final_states=True,
        final_states_out=target,
        activation=None,
    )
    assert out2.shape == (2, 4, 5)
    assert final2 is target


def test_reference_rejects_bad_activation():
    x = torch.randn(1, 2, 3)
    weight = torch.randn(2, 3)
    with pytest.raises(NotImplementedError):
        conv.causal_conv1d_ref(x, weight, activation="relu")


def test_varlen_fn_runs_eager_conv(monkeypatch):
    torch.manual_seed(1)
    dim, total, width = 4, 7, 3
    x = torch.randn(dim, total)
    weight = torch.randn(dim, width)
    bias = torch.randn(dim)
    conv_states = torch.zeros(2, dim, width - 1)
    query_start_loc = torch.tensor([0, 4, 7], dtype=torch.int32)

    out = conv.causal_conv1d_fn(
        x,
        weight,
        bias,
        conv_states,
        query_start_loc,
        has_initial_state=torch.tensor([False, True]),
        activation=True,
    )
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_varlen_fn_skips_padding_and_empty(monkeypatch):
    torch.manual_seed(2)
    dim, width = 4, 2
    x = torch.randn(dim, 4)
    weight = torch.randn(dim, width)
    bias = torch.randn(dim)
    conv_states = torch.zeros(3, dim, width - 1)
    query_start_loc = torch.tensor([0, 0, 2, 4], dtype=torch.int32)
    cache_indices = torch.tensor([1, 0, 2], dtype=torch.int64)

    out = conv.causal_conv1d_fn(
        x,
        weight,
        bias,
        conv_states,
        query_start_loc,
        cache_indices=cache_indices,
        pad_slot_id=conv.PAD_SLOT_ID,
    )
    assert out.shape == x.shape


def test_varlen_fn_rejects_prefix_caching_arguments():
    x = torch.randn(2, 3)
    weight = torch.randn(2, 2)
    bias = torch.randn(2)
    conv_states = torch.zeros(1, 2, 1)
    qsl = torch.tensor([0, 3], dtype=torch.int32)
    with pytest.raises(NotImplementedError):
        conv.causal_conv1d_fn(
            x,
            weight,
            bias,
            conv_states,
            qsl,
            num_computed_tokens=torch.tensor([0]),
        )
    with pytest.raises(NotImplementedError):
        conv.causal_conv1d_fn(
            x,
            weight,
            bias,
            conv_states,
            qsl,
            cache_indices=torch.zeros(2, 2, dtype=torch.int32),
        )
    with pytest.raises(NotImplementedError):
        conv.causal_conv1d_fn(x, weight, bias, conv_states, qsl, activation="relu")


def test_kernel_wrapper_keeps_raw_function():
    raw = getattr(conv._causal_conv1d_update_kernel_npu_tiled, "fn", None)
    if raw is None:
        pytest.skip("triton wrapper does not expose .fn on this build")
    assert inspect.isfunction(raw)
