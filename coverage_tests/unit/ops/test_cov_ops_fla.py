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

"""CPU tests for the FL FLA gated-delta-rule ops.

The Triton/FlagGems forward kernels and ``l2norm_fwd`` are replaced with
recorders.  The autograd functions' argument marshalling, contiguity
adaptation and shape validation are asserted directly; the native op
wrappers are constructed with ``CustomOp.__init__`` stubbed so no vLLM
compilation config is required.
"""

from __future__ import annotations

import contextlib
import importlib
from types import SimpleNamespace

import pytest
import torch

from . import _imports  # noqa: F401  (installs CPU-only import placeholders)


def _load_fla():
    """Import the FLA package through the upstream (non-FlagGems) branch."""
    from vllm_fl import utils as fl_utils

    original = fl_utils.use_flaggems_op
    fl_utils.use_flaggems_op = lambda op: False
    try:
        package = importlib.import_module("vllm_fl.ops.fla")
        chunk = importlib.import_module("vllm_fl.ops.fla.chunk")
        recurrent = importlib.import_module("vllm_fl.ops.fla.fused_recurrent")
        utils = importlib.import_module("vllm_fl.ops.fla.utils")
    finally:
        fl_utils.use_flaggems_op = original
    return package, chunk, recurrent, utils


FLA, chunk_mod, recurrent_mod, utils_mod = _load_fla()


class _Recorder:
    def __init__(self, result=None):
        self.calls = []
        self.result = result

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.result


class _DeviceRecorder:
    def __init__(self):
        self.calls = []

    def __call__(self, index):
        self.calls.append(index)
        return contextlib.nullcontext()


@pytest.fixture(autouse=True)
def _cpu_device(monkeypatch):
    """Keep ``input_guard`` off the CUDA/NPU device context machinery."""
    device = _DeviceRecorder()

    class _Npu:
        @staticmethod
        def is_available():
            return False

    monkeypatch.setattr(torch, "npu", _Npu(), raising=False)
    monkeypatch.setattr(torch.cuda, "device", device, raising=False)
    return device


def test_package_exports_both_gated_delta_rule_ops():
    assert FLA.ChunkGatedDeltaRuleOp is chunk_mod.ChunkGatedDeltaRuleOp
    assert (
        FLA.FusedRecurrentGatedDeltaRuleOp
        is recurrent_mod.FusedRecurrentGatedDeltaRuleOp
    )
    assert set(FLA.__all__) == {
        "ChunkGatedDeltaRuleOp",
        "FusedRecurrentGatedDeltaRuleOp",
    }


def test_input_guard_makes_positional_and_keyword_tensors_contiguous():
    seen = {}

    @utils_mod.input_guard
    def target(x, y, flag):
        seen.update(x=x, y=y, flag=flag)
        return "ok"

    x = torch.zeros(4, 2).t()  # non-contiguous view
    y = torch.zeros(3, 5).t()
    assert target.__name__ == "target"
    assert target(x, y, flag=7) == "ok"
    assert seen["x"].is_contiguous()
    assert seen["y"].is_contiguous()
    assert torch.equal(seen["x"], x)
    assert seen["flag"] == 7


def test_input_guard_selects_first_keyword_tensor_for_device(_cpu_device):
    seen = {}

    @utils_mod.input_guard
    def target(value):
        seen["value"] = value
        return "ok"

    value = torch.zeros(2, 3).t()
    assert target(value=value) == "ok"
    assert seen["value"].is_contiguous()
    assert _cpu_device.calls == [None]


def test_input_guard_uses_nullcontext_without_tensors(_cpu_device):
    @utils_mod.input_guard
    def target(value):
        return value * 2

    assert target(value=3) == 6
    assert _cpu_device.calls == []


def test_input_guard_prefers_npu_device_when_available(monkeypatch):
    calls = []

    class _Npu:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def device(index):
            calls.append(index)
            return contextlib.nullcontext()

    monkeypatch.setattr(torch, "npu", _Npu(), raising=False)

    @utils_mod.input_guard
    def target(value):
        return value

    tensor = torch.zeros(2)
    assert target(tensor) is not None
    assert calls == [None]


def _chunk_ctx():
    return SimpleNamespace()


def test_chunk_forward_marshals_kernel_with_l2norm(monkeypatch):
    l2norm_calls = []

    def l2norm(value):
        l2norm_calls.append(value)
        return value + 1

    kernel = _Recorder()
    monkeypatch.setattr(chunk_mod, "l2norm_fwd", l2norm)
    monkeypatch.setattr(chunk_mod, "chunk_gated_delta_rule_fwd", kernel)

    q = torch.randn(2, 3, 4, 8, dtype=torch.float16)
    k = torch.randn(2, 3, 4, 8, dtype=torch.float16)
    v = torch.randn(2, 3, 4, 8, dtype=torch.float16)
    g = torch.randn(2, 3, 4)
    beta = torch.randn(2, 3, 4)
    initial_state = torch.zeros(2, 4, 8, 8)
    cu_seqlens = torch.tensor([0, 3, 6])
    kernel.result = (
        "g",
        torch.zeros(2, 3, 4, 8, dtype=torch.float64),
        "A",
        "final-state",
        "w",
        "h",
        "v-new",
    )
    ctx = _chunk_ctx()

    o, final_state = chunk_mod.ChunkGatedDeltaRuleFunction.forward(
        ctx,
        q,
        k,
        v,
        g,
        beta,
        0.25,
        initial_state,
        False,
        cu_seqlens,
        True,
    )

    assert len(l2norm_calls) == 2
    kwargs = kernel.calls[0][1]
    assert torch.equal(kwargs["q"], q + 1)
    assert torch.equal(kwargs["k"], k + 1)
    assert kwargs["v"] is v
    assert kwargs["g"] is g
    assert kwargs["beta"] is beta
    assert kwargs["scale"] == 0.25
    assert kwargs["initial_state"] is initial_state
    assert kwargs["output_final_state"] is False
    assert kwargs["cu_seqlens"] is cu_seqlens
    assert o.dtype == torch.float16
    assert final_state == "final-state"
    assert ctx.scale == 0.25
    assert ctx.use_qk_l2norm_in_kernel is True


def test_chunk_forward_skips_l2norm_when_disabled(monkeypatch):
    l2norm = _Recorder()
    kernel = _Recorder()
    monkeypatch.setattr(chunk_mod, "l2norm_fwd", l2norm)
    monkeypatch.setattr(chunk_mod, "chunk_gated_delta_rule_fwd", kernel)
    q = torch.randn(1, 2, 2, 4, dtype=torch.float16)
    kernel.result = (
        "g",
        torch.zeros_like(q),
        "A",
        None,
        "w",
        "h",
        "v-new",
    )
    ctx = _chunk_ctx()

    chunk_mod.ChunkGatedDeltaRuleFunction.forward(
        ctx, q, q, q, torch.zeros(1, 2, 2), torch.ones(1, 2, 2), 0.5, None, True
    )

    assert l2norm.calls == []
    assert ctx.use_qk_l2norm_in_kernel is False


def _chunk_op(monkeypatch, **kwargs):
    monkeypatch.setattr(
        chunk_mod.CustomOp,
        "__init__",
        lambda self: torch.nn.Module.__init__(self),
    )
    return chunk_mod.ChunkGatedDeltaRuleOp(**kwargs)


def test_chunk_op_constructor_reads_environment(monkeypatch):
    monkeypatch.setenv("GDN_RECOMPUTE_SUPPRESS_LEVEL", "5")
    op = _chunk_op(monkeypatch, output_final_state=True, use_qk_l2norm_in_kernel=True)
    assert op.output_final_state is True
    assert op.use_qk_l2norm_in_kernel is True
    assert op.suppress_level == 5


def _check_dtype_and_beta(op):
    q = torch.zeros(1, 2, 2, 4, dtype=torch.float16)
    with pytest.raises(AssertionError):
        op.forward_native(q, q.float(), q, torch.zeros(1, 2, 2), torch.ones(1, 2, 2))
    with pytest.raises(AssertionError, match="does not support float32"):
        op.forward_native(
            q.float(), q.float(), q.float(), torch.zeros(1, 2, 2), torch.ones(1, 2, 2)
        )
    with pytest.raises(AssertionError, match="beta must be"):
        op.forward_native(q, q, q, torch.zeros(1, 2, 2), torch.ones(1, 2))


def test_chunk_op_forward_validates_dtype_and_beta(monkeypatch):
    op = _chunk_op(monkeypatch)
    _check_dtype_and_beta(op)
    # Independently drive one contract violation from the test body: a beta
    # with the wrong top-k width must be rejected with the documented message.
    q = torch.zeros(1, 2, 2, 4, dtype=torch.float16)
    beta = torch.ones(1, 2)
    with pytest.raises(AssertionError, match="beta must be"):
        op.forward_native(q, q, q, torch.zeros(1, 2, 2), beta)


def test_chunk_op_warns_on_head_first_shapes(monkeypatch):
    apply = _Recorder(result=(torch.zeros(1, 2, 4, 8, dtype=torch.float16), None))
    monkeypatch.setattr(chunk_mod.ChunkGatedDeltaRuleFunction, "apply", apply)
    op = _chunk_op(monkeypatch)
    q = torch.zeros(1, 2, 4, 8, dtype=torch.float16)
    beta = torch.zeros(1, 2, 4)
    with pytest.warns(UserWarning, match="format mismatch"):
        o, final_state = op.forward_native(q, q, q, beta, beta)
    assert o.shape == (1, 2, 4, 8)
    assert final_state is None


def test_chunk_op_scales_and_forwards_options(monkeypatch):
    apply = _Recorder(result=(torch.zeros(1, 2, 2, 4), "state"))
    monkeypatch.setattr(chunk_mod.ChunkGatedDeltaRuleFunction, "apply", apply)
    op = _chunk_op(monkeypatch, output_final_state=True, use_qk_l2norm_in_kernel=True)
    q = torch.zeros(1, 2, 2, 4, dtype=torch.float16)
    beta = torch.zeros(1, 2, 2)
    initial_state = torch.zeros(2, 2, 4, 4)
    cu_seqlens = torch.tensor([0, 2, 4])

    op.forward_native(
        q, q, q, beta, beta, initial_state=initial_state, cu_seqlens=cu_seqlens
    )

    args = apply.calls[0][0]
    assert args[5] == 0.5  # 1 / sqrt(4)
    assert args[6] is initial_state
    assert args[7] is True  # output_final_state
    assert args[8] is cu_seqlens
    assert args[9] is True  # use_qk_l2norm_in_kernel


def test_chunk_op_rejects_cu_seqlens_with_batch(monkeypatch):
    op = _chunk_op(monkeypatch)
    q = torch.zeros(2, 2, 2, 4, dtype=torch.float16)
    beta = torch.zeros(2, 2, 2)
    with pytest.raises(ValueError, match="batch size is expected to be 1"):
        op.forward_native(q, q, q, beta, beta, cu_seqlens=torch.tensor([0, 2, 4]))


def test_chunk_op_rejects_initial_state_count(monkeypatch):
    op = _chunk_op(monkeypatch)
    q = torch.zeros(1, 2, 2, 4, dtype=torch.float16)
    beta = torch.zeros(1, 2, 2)
    initial_state = torch.zeros(3, 2, 4, 4)
    with pytest.raises(ValueError, match="number of initial states"):
        op.forward_native(
            q,
            q,
            q,
            beta,
            beta,
            initial_state=initial_state,
            cu_seqlens=torch.tensor([0, 2, 4]),
        )


def test_chunk_op_uses_explicit_scale(monkeypatch):
    apply = _Recorder(result=(torch.zeros(1, 2, 2, 4), None))
    monkeypatch.setattr(chunk_mod.ChunkGatedDeltaRuleFunction, "apply", apply)
    op = _chunk_op(monkeypatch)
    q = torch.zeros(1, 2, 2, 4, dtype=torch.float16)
    beta = torch.zeros(1, 2, 2)
    op.forward_native(q, q, q, beta, beta, scale=0.125)
    assert apply.calls[0][0][5] == 0.125


# ---------------------------------------------------------------------------
# fused-recurrent variants
# ---------------------------------------------------------------------------


def test_recurrent_function_marshals_contiguous_inputs(monkeypatch):
    kernel = _Recorder(result=("o", "state"))
    monkeypatch.setattr(recurrent_mod, "fused_recurrent_gated_delta_rule_fwd", kernel)
    q = torch.zeros(1, 2, 2, 4)
    base = torch.arange(8.0).reshape(2, 2, 2)
    g = base.transpose(0, 1)  # non-contiguous [B, T, HV] view
    beta = (base + 1.0).transpose(0, 1)
    ctx = SimpleNamespace()

    o, state = recurrent_mod.FusedRecurrentFunction.forward(
        ctx,
        q,
        q,
        q,
        g,
        beta,
        0.5,
        "init",
        False,
        None,
        "indices",
        "accepted",
        True,
    )

    kwargs = kernel.calls[0][1]
    assert kwargs["q"] is q
    assert kwargs["g"].is_contiguous()
    assert kwargs["beta"].is_contiguous()
    assert torch.equal(kwargs["g"], g)
    assert torch.equal(kwargs["beta"], beta)
    assert kwargs["scale"] == 0.5
    assert kwargs["initial_state"] == "init"
    assert kwargs["inplace_final_state"] is False
    assert kwargs["cu_seqlens"] is None
    assert kwargs["ssm_state_indices"] == "indices"
    assert kwargs["num_accepted_tokens"] == "accepted"
    assert kwargs["use_qk_l2norm_in_kernel"] is True
    assert (o, state) == ("o", "state")


def _recurrent_op(monkeypatch, **kwargs):
    monkeypatch.setattr(
        recurrent_mod.CustomOp,
        "__init__",
        lambda self: torch.nn.Module.__init__(self),
    )
    return recurrent_mod.FusedRecurrentGatedDeltaRuleOp(**kwargs)


def test_recurrent_op_constructor_and_default_scale(monkeypatch):
    apply = _Recorder(result=("o", None))
    monkeypatch.setattr(recurrent_mod.FusedRecurrentFunction, "apply", apply)
    op = _recurrent_op(monkeypatch, inplace_final_state=True)
    assert op.inplace_final_state is True
    assert op.use_qk_l2norm_in_kernel is False

    q = torch.zeros(1, 2, 2, 4)
    op.forward_native(q, q, q, torch.zeros(1, 2, 2))
    args = apply.calls[0][0]
    assert args[5] == 0.5  # 1 / sqrt(4)
    assert torch.equal(args[4], torch.ones_like(q[..., 0]))
    assert args[7] is True  # inplace_final_state


def test_recurrent_op_requires_positive_scale(monkeypatch):
    op = _recurrent_op(monkeypatch)
    q = torch.zeros(1, 2, 2, 4)
    with pytest.raises(AssertionError, match="scale must be positive"):
        op.forward_native(q, q, q, torch.zeros(1, 2, 2), scale=-1.0)


def test_recurrent_op_rejects_cu_seqlens_with_batch(monkeypatch):
    op = _recurrent_op(monkeypatch)
    q = torch.zeros(2, 2, 2, 4)
    with pytest.raises(ValueError, match="batch size is expected to be 1"):
        op.forward_native(
            q, q, q, torch.zeros(2, 2, 2), cu_seqlens=torch.tensor([0, 2, 4])
        )


def test_recurrent_op_keeps_explicit_scale(monkeypatch):
    apply = _Recorder(result=("o", None))
    monkeypatch.setattr(recurrent_mod.FusedRecurrentFunction, "apply", apply)
    op = _recurrent_op(monkeypatch)
    q = torch.zeros(1, 2, 2, 4)
    op.forward_native(q, q, q, torch.zeros(1, 2, 2), scale=0.25)
    assert apply.calls[0][0][5] == 0.25
