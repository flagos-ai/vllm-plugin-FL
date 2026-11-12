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

"""CPU tests for the FL fused activation ops.

The ``CachedOp`` kernels are replaced with recorders, so the tests assert the
exact provider object/arguments each branch passes and the output-shape
contract of the registered fake kernels.  ``SiluAndMulWithClampFL.enabled``
is driven through a stand-in compilation config.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from . import _imports  # noqa: F401  (installs CPU-only import placeholders)
from vllm_fl import envs as fl_envs, utils as fl_utils
from vllm_fl.ops import activation as Mod


def _bare(cls, **attrs):
    instance = object.__new__(cls)
    torch.nn.Module.__init__(instance)
    for name, value in attrs.items():
        setattr(instance, name, value)
    return instance


class _Recorder:
    def __init__(self, result=None):
        self.calls = []
        self.result = result

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.result


def _compile_flags(monkeypatch, *, env=True, oot=True):
    monkeypatch.setitem(
        fl_envs.__dict__, "VLLM_FL_USE_FLAGGEMS_VLLM", env
    )
    monkeypatch.setattr(fl_utils, "is_oot_enabled", lambda: oot)


def _policy(monkeypatch, *, whitelist=None, blacklist=None, enabled=None):
    monkeypatch.setattr(fl_utils, "get_oot_whitelist", lambda: whitelist)
    monkeypatch.setattr(fl_utils, "get_oot_blacklist", lambda: blacklist)
    enabled = {} if enabled is None else enabled
    monkeypatch.setattr(
        fl_utils, "use_flaggems_op", lambda op: enabled.get(op, True)
    )


def test_enable_returns_when_flaggems_vllm_disabled(monkeypatch):
    _compile_flags(monkeypatch, env=False)
    config = SimpleNamespace(custom_ops=[])
    Mod.enable_fused_activations_for_compile(config)
    assert config.custom_ops == []


def test_enable_returns_when_oot_disabled(monkeypatch):
    _compile_flags(monkeypatch, oot=False)
    config = SimpleNamespace(custom_ops=[])
    Mod.enable_fused_activations_for_compile(config)
    assert config.custom_ops == []


def test_enable_appends_all_selected_ops(monkeypatch):
    _compile_flags(monkeypatch)
    _policy(monkeypatch)
    config = SimpleNamespace(custom_ops=[])
    Mod.enable_fused_activations_for_compile(config)
    assert config.custom_ops == [
        "+SiluAndMul",
        "+GeluAndMul",
        "+silu_and_mul_with_clamp",
    ]


def test_enable_skips_policy_disabled_op(monkeypatch):
    _compile_flags(monkeypatch)
    _policy(monkeypatch, enabled={"silu_and_mul": False})
    config = SimpleNamespace(custom_ops=[])
    Mod.enable_fused_activations_for_compile(config)
    assert config.custom_ops == ["+GeluAndMul", "+silu_and_mul_with_clamp"]


def test_enable_skips_blacklisted_op(monkeypatch):
    _compile_flags(monkeypatch)
    _policy(monkeypatch, blacklist=["gelu_and_mul"])
    config = SimpleNamespace(custom_ops=[])
    Mod.enable_fused_activations_for_compile(config)
    assert config.custom_ops == ["+SiluAndMul", "+silu_and_mul_with_clamp"]


def test_enable_respects_whitelist(monkeypatch):
    _compile_flags(monkeypatch)
    _policy(monkeypatch, whitelist=["silu_and_mul"])
    config = SimpleNamespace(custom_ops=[])
    Mod.enable_fused_activations_for_compile(config)
    assert config.custom_ops == ["+SiluAndMul"]


@pytest.mark.parametrize("disabled", ["-silu_and_mul", "-SiluAndMul"])
def test_enable_honours_disabled_alias(monkeypatch, disabled):
    _compile_flags(monkeypatch)
    _policy(monkeypatch)
    config = SimpleNamespace(custom_ops=[disabled])
    Mod.enable_fused_activations_for_compile(config)
    assert "+SiluAndMul" not in config.custom_ops
    assert "+GeluAndMul" in config.custom_ops


def test_enable_does_not_duplicate_already_selected(monkeypatch):
    _compile_flags(monkeypatch)
    _policy(monkeypatch)
    config = SimpleNamespace(custom_ops=["+SiluAndMul"])
    Mod.enable_fused_activations_for_compile(config)
    assert config.custom_ops.count("+SiluAndMul") == 1


@pytest.mark.parametrize(
    ("fake", "args", "expected_shape"),
    [
        (Mod._silu_and_mul_graph_fake, (), (2, 4)),
        (Mod._gelu_and_mul_graph_fake, ("tanh",), (2, 4)),
        (Mod._silu_and_mul_with_clamp_graph_fake, (7.0,), (2, 4)),
    ],
)
def test_fake_kernels_allocate_half_width_output(fake, args, expected_shape):
    x = torch.zeros(2, 8, dtype=torch.float16)
    out = fake(x, *args)
    assert out.shape == expected_shape
    assert out.dtype == x.dtype
    assert out.device == x.device


def test_silu_graph_passes_none_compilation_config(monkeypatch):
    expected = torch.full((2, 4), 3.0)
    recorder = _Recorder(expected)
    monkeypatch.setattr(Mod, "_silu_and_mul", recorder)
    x = torch.zeros(2, 8)
    assert torch.equal(Mod._silu_and_mul_graph(x), expected)
    args, kwargs = recorder.calls[0]
    assert args[0] is None
    assert args[1] is x
    assert kwargs == {}


def test_gelu_graph_passes_approximate_namespace(monkeypatch):
    expected = torch.full((2, 4), 4.0)
    recorder = _Recorder(expected)
    monkeypatch.setattr(Mod, "_gelu_and_mul", recorder)
    x = torch.zeros(2, 8)
    assert torch.equal(Mod._gelu_and_mul_graph(x, "tanh"), expected)
    ctx, seen = recorder.calls[0][0]
    assert ctx.approximate == "tanh"
    assert seen is x


def test_clamp_graph_passes_default_swiglu_params(monkeypatch):
    expected = torch.full((2, 4), 5.0)
    recorder = _Recorder(expected)
    monkeypatch.setattr(Mod, "_silu_and_mul_with_clamp", recorder)
    x = torch.zeros(2, 8)
    assert torch.equal(Mod._silu_and_mul_with_clamp_graph(x, 5.0), expected)
    ctx, seen = recorder.calls[0][0]
    assert (ctx.swiglu_limit, ctx.alpha, ctx.beta) == (5.0, 1.0, 0.0)
    assert seen is x


def test_silu_forward_oot_dispatches_instance_and_input(monkeypatch):
    recorder = _Recorder("out")
    monkeypatch.setattr(Mod, "_silu_and_mul", recorder)
    instance = _bare(Mod.SiluAndMulFL)
    x = torch.zeros(2, 8)
    assert instance.forward_oot(x) == "out"
    args, _ = recorder.calls[0]
    assert args[0] is instance
    assert args[1] is x


def test_silu_forward_oot_uses_graph_when_compiling(monkeypatch):
    expected = torch.full((2, 4), 6.0)
    recorder = _Recorder(expected)
    monkeypatch.setattr(Mod, "_silu_and_mul", recorder)
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    instance = _bare(Mod.SiluAndMulFL)
    x = torch.zeros(2, 8)
    assert torch.equal(instance.forward_oot(x), expected)
    args, _ = recorder.calls[0]
    assert args[0] is None
    assert args[1] is x


def test_silu_forward_hip_delegates_to_oot(monkeypatch):
    recorder = _Recorder("hip")
    monkeypatch.setattr(Mod, "_silu_and_mul", recorder)
    instance = _bare(Mod.SiluAndMulFL)
    x = torch.zeros(2, 8)
    assert instance.forward_hip(x) == "hip"
    assert recorder.calls[0][0][0] is instance


def test_silu_and_mul_constructor_calls_upstream(monkeypatch):
    calls = []
    monkeypatch.setattr(
        Mod.SiluAndMul, "__init__", lambda self: calls.append(self)
    )
    instance = Mod.SiluAndMulFL()
    assert calls == [instance]


def test_gelu_forward_oot_and_hip_dispatch_instance(monkeypatch):
    recorder = _Recorder("gelu")
    monkeypatch.setattr(Mod, "_gelu_and_mul", recorder)
    instance = _bare(Mod.GeluAndMulFL, approximate="tanh")
    x = torch.zeros(2, 8)
    assert instance.forward_oot(x) == "gelu"
    assert instance.forward_hip(x) == "gelu"
    assert recorder.calls[0][0][0] is instance
    assert recorder.calls[1][0][0] is instance


def test_gelu_forward_oot_uses_graph_when_compiling(monkeypatch):
    expected = torch.full((2, 4), 7.0)
    recorder = _Recorder(expected)
    monkeypatch.setattr(Mod, "_gelu_and_mul", recorder)
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    instance = _bare(Mod.GeluAndMulFL, approximate="tanh")
    x = torch.zeros(2, 8)
    assert torch.equal(instance.forward_oot(x), expected)
    ctx, seen = recorder.calls[0][0]
    assert ctx.approximate == "tanh"
    assert seen is x


def test_gelu_constructor_forwards_approximate(monkeypatch):
    calls = []

    def fake_init(self, approximate="none"):
        calls.append(approximate)
        self.approximate = approximate

    monkeypatch.setattr(Mod.GeluAndMul, "__init__", fake_init)
    instance = Mod.GeluAndMulFL(approximate="tanh")
    assert calls == ["tanh"]
    assert instance.approximate == "tanh"


def _clamp_controls(monkeypatch, controls, default_on):
    import vllm.config

    monkeypatch.setattr(
        vllm.config,
        "get_cached_compilation_config",
        lambda: SimpleNamespace(custom_ops=list(controls)),
    )
    monkeypatch.setattr(
        Mod.SiluAndMulWithClampFL,
        "default_on",
        classmethod(lambda cls: default_on),
    )


@pytest.mark.parametrize(
    ("controls", "default_on", "expected"),
    [
        ([], False, False),
        ([], True, True),
        (["+silu_and_mul_with_clamp"], False, True),
        (["+SiluAndMulWithClamp"], False, True),
        (["-SiluAndMulWithClamp"], True, False),
        (["-silu_and_mul_with_clamp"], True, False),
    ],
)
def test_clamp_enabled_reads_compilation_controls(
    monkeypatch, controls, default_on, expected
):
    _clamp_controls(monkeypatch, controls, default_on)
    assert Mod.SiluAndMulWithClampFL.enabled() is expected


def test_clamp_enabled_rejects_conflicting_controls(monkeypatch):
    _clamp_controls(
        monkeypatch,
        ["+silu_and_mul_with_clamp", "-silu_and_mul_with_clamp"],
        False,
    )
    with pytest.raises(AssertionError, match="enable and disable"):
        Mod.SiluAndMulWithClampFL.enabled()


@pytest.mark.parametrize("attrs", [{"alpha": 2.0}, {"beta": 1.0}])
def test_clamp_forward_oot_uses_native_for_non_default_params(monkeypatch, attrs):
    kernel = _Recorder("fused")
    monkeypatch.setattr(Mod, "_silu_and_mul_with_clamp", kernel)
    instance = _bare(
        Mod.SiluAndMulWithClampFL,
        swiglu_limit=7.0,
        alpha=attrs.get("alpha", 1.0),
        beta=attrs.get("beta", 0.0),
    )
    native = _Recorder("native")
    monkeypatch.setattr(instance, "forward_native", native)
    x = torch.zeros(2, 8)
    assert instance.forward_oot(x) == "native"
    assert native.calls[0][0][0] is x
    assert kernel.calls == []


def test_clamp_forward_oot_uses_fused_kernel(monkeypatch):
    kernel = _Recorder("fused")
    monkeypatch.setattr(Mod, "_silu_and_mul_with_clamp", kernel)
    instance = _bare(
        Mod.SiluAndMulWithClampFL, swiglu_limit=7.0, alpha=1.0, beta=0.0
    )
    x = torch.zeros(2, 8)
    assert instance.forward_oot(x) == "fused"
    args, _ = kernel.calls[0]
    assert args[0] is instance
    assert args[1] is x


def test_clamp_forward_oot_uses_graph_when_compiling(monkeypatch):
    expected = torch.full((2, 4), 8.0)
    kernel = _Recorder(expected)
    monkeypatch.setattr(Mod, "_silu_and_mul_with_clamp", kernel)
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    instance = _bare(
        Mod.SiluAndMulWithClampFL, swiglu_limit=6.5, alpha=1.0, beta=0.0
    )
    x = torch.zeros(2, 8)
    assert torch.equal(instance.forward_oot(x), expected)
    ctx, seen = kernel.calls[0][0]
    assert ctx.swiglu_limit == 6.5
    assert seen is x
