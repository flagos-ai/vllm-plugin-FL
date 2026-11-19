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

"""CPU tests for the MoE-sum dispatch bridge.

``vllm._custom_ops.moe_sum`` and the plugin's ``CachedOp`` are the external
boundary.  The replacement is exercised on small tensors: the zero-token
guard, the stride-safe fallback (checked against an explicit FP32 sum), and
the contiguous dispatch path's exact positional arguments.
"""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest
import torch

# The shim import is load-bearing: it must precede vllm_fl imports.
# isort: split
from . import _flaggems_shim  # noqa: F401

# isort: split
import vllm_fl.patches.moe_sum as patch_module


class _DispatchRecorder:
    """Stand-in for the module-level CachedOp."""

    def __init__(self, result="DISPATCHED"):
        self.calls = []
        self.result = result

    def __call__(self, input, output):
        self.calls.append((input, output))
        return self.result


def _installed(monkeypatch, result="DISPATCHED"):
    """Install the bridge on a fake ops module and return (module, dispatch)."""
    monkeypatch.setattr(patch_module, "use_flaggems_op", lambda name: True)
    dispatch = _DispatchRecorder(result=result)
    monkeypatch.setattr(patch_module, "_dispatch_moe_sum", dispatch)
    original = lambda input, output: "ORIGINAL"  # noqa: E731
    ops_module = SimpleNamespace(moe_sum=original)
    assert patch_module.patch_vllm_moe_sum(ops_module) is True
    return ops_module, dispatch, original


def test_install_skips_when_flaggems_disabled(monkeypatch):
    """The policy gate returns False before touching the ops module."""
    monkeypatch.setattr(patch_module, "use_flaggems_op", lambda name: False)
    original = lambda input, output: "ORIGINAL"  # noqa: E731
    ops_module = SimpleNamespace(moe_sum=original)

    assert patch_module.patch_vllm_moe_sum(ops_module) is False
    assert ops_module.moe_sum is original


def test_install_wraps_once_and_keeps_original(monkeypatch):
    """The first install wraps; a second call detects the marker."""
    ops_module, _, original = _installed(monkeypatch)
    wrapper = ops_module.moe_sum

    assert wrapper._vllm_fl_moe_sum_patch is True
    assert wrapper._vllm_fl_original is original
    assert patch_module.patch_vllm_moe_sum(ops_module) is False
    assert ops_module.moe_sum is wrapper


def test_install_can_resolve_vllm_custom_ops(monkeypatch):
    """With no module argument the real vllm custom-op module is patched."""
    import vllm

    monkeypatch.setattr(patch_module, "use_flaggems_op", lambda name: True)
    original = lambda input, output: "ORIGINAL"  # noqa: E731
    fake_ops = types.ModuleType("vllm._custom_ops")
    fake_ops.moe_sum = original
    monkeypatch.setitem(sys.modules, "vllm._custom_ops", fake_ops)
    monkeypatch.setattr(vllm, "_custom_ops", fake_ops, raising=False)

    assert patch_module.patch_vllm_moe_sum() is True
    assert fake_ops.moe_sum._vllm_fl_moe_sum_patch is True
    assert fake_ops.moe_sum._vllm_fl_original is original


@pytest.mark.parametrize("empty", ["input", "output"])
def test_zero_sized_tensors_return_none_without_dispatch(monkeypatch, empty):
    """Triton cannot launch a zero grid, so both zero-size cases short-circuit."""
    ops_module, dispatch, _ = _installed(monkeypatch)
    wrapper = ops_module.moe_sum
    input = torch.zeros(0, 3, 4) if empty == "input" else torch.zeros(2, 3, 4)
    output = torch.zeros(0, 4) if empty == "output" else torch.zeros(2, 4)

    assert wrapper(input, output) is None
    assert dispatch.calls == []


def test_noncontiguous_input_uses_reference_sum(monkeypatch):
    """A strided input is reduced in FP32 and cast to the output dtype."""
    ops_module, dispatch, _ = _installed(monkeypatch)
    wrapper = ops_module.moe_sum
    base = torch.arange(24, dtype=torch.float32).reshape(3, 2, 4)
    input = base[:, :, ::2]
    assert input.stride(-1) == 2
    output = torch.zeros(3, 2, dtype=torch.float16)

    assert wrapper(input, output) is None

    assert dispatch.calls == []
    expected = input.float().sum(dim=1).to(torch.float16)
    assert output.dtype == torch.float16
    assert torch.allclose(output, expected)
    assert output.tolist() == expected.tolist()


def test_noncontiguous_output_uses_reference_sum(monkeypatch):
    """A strided output buffer also takes the stride-safe fallback."""
    ops_module, dispatch, _ = _installed(monkeypatch)
    wrapper = ops_module.moe_sum
    input = torch.arange(24, dtype=torch.float32).reshape(3, 2, 4).contiguous()
    storage = torch.zeros(3, 8, dtype=torch.float32)
    output = storage[:, ::2]
    assert output.stride(-1) == 2

    assert wrapper(input, output) is None

    assert dispatch.calls == []
    expected = input.float().sum(dim=1)
    assert torch.allclose(output, expected)
    assert torch.allclose(storage[:, 1::2], torch.zeros(3, 4))


def test_contiguous_tensors_use_dispatch(monkeypatch):
    """Contiguous input and output go to the plugin OpManager unchanged."""
    ops_module, dispatch, _ = _installed(monkeypatch, result="MANAGER-RESULT")
    wrapper = ops_module.moe_sum
    input = torch.zeros(2, 3, 4)
    output = torch.zeros(2, 4)

    assert wrapper(input, output) == "MANAGER-RESULT"
    assert dispatch.calls == [(input, output)]


def test_dispatch_return_value_is_passed_through(monkeypatch):
    """The wrapper returns whatever the dispatch manager returns."""
    sentinel = object()
    ops_module, dispatch, _ = _installed(monkeypatch, result=sentinel)
    assert ops_module.moe_sum(torch.zeros(1, 1, 1), torch.zeros(1, 1)) is sentinel


def test_torch_fallback_matches_explicit_sum(monkeypatch):
    """_torch_moe_sum matches an explicit per-element reference."""
    base = torch.tensor(
        [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[0.5, 1.5, 2.5], [3.5, 4.5, 5.5]]]
    )
    input = base
    output = torch.empty(2, 3, dtype=torch.float64)

    patch_module._torch_moe_sum(input, output)

    expected = torch.tensor([[5.0, 7.0, 9.0], [4.0, 6.0, 8.0]], dtype=torch.float64)
    assert torch.allclose(output, expected)
