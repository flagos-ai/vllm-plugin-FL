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

"""CPU tests for the FL fused-MoE layer factory and quant-method wrapper.

The upstream factory, backend oracle, quant-method base and router patcher
are recorders, so the tests assert the factory's delegation order and the
unquantized-only substitution rule without building a real MoE runner.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from . import _imports  # noqa: F401  (installs CPU-only import placeholders)
from vllm_fl.ops.fused_moe import layer as Mod


class _Recorder:
    def __init__(self, result=None):
        self.calls = []
        self.result = result

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.result


def test_fl_quant_method_selects_backend_and_calls_super(monkeypatch):
    base_init_calls = []

    def base_init(self, moe):
        # The adapter reads self.moe after delegating to the base class.
        self.moe = moe
        base_init_calls.append(moe)

    selector = _Recorder(result=("backend", "experts-cls"))
    monkeypatch.setattr(Mod.UnquantizedFusedMoEMethod, "__init__", base_init)
    monkeypatch.setattr(Mod, "select_unquantized_moe_backend_oot", selector)
    moe = SimpleNamespace(name="moe")

    # Avoid CustomOp.__new__, which may substitute a registered OOT class on
    # the image; call the adapter's own __init__ on a bare instance instead.
    instance = object.__new__(Mod.UnquantizedFusedMoEMethodFL)
    torch.nn.Module.__init__(instance)
    instance.__init__(moe)

    assert base_init_calls == [moe]
    assert selector.calls == [
        ((), {"moe_config": moe, "prefer_flaggems_experts": True})
    ]
    assert instance.moe is moe
    assert instance.unquantized_backend == "backend"
    assert instance.experts_cls == "experts-cls"


def _bare(**attrs):
    instance = object.__new__(Mod.UnquantizedFusedMoEMethodFL)
    for name, value in attrs.items():
        object.__setattr__(instance, name, value)
    return instance


def test_is_monolithic_prefers_composed_kernel():
    kernel = SimpleNamespace(is_monolithic=True)
    instance = _bare(moe_kernel=kernel, experts_cls=object())
    assert instance.is_monolithic is True


def test_is_monolithic_true_without_kernel_or_experts():
    assert _bare(moe_kernel=None, experts_cls=None).is_monolithic is True


def test_is_monolithic_uses_experts_class():
    class Experts:
        @staticmethod
        def is_monolithic():
            return False

    assert _bare(moe_kernel=None, experts_cls=Experts).is_monolithic is False


def test_factory_replaces_only_unquantized_method(monkeypatch):
    class FakeUnquantized:
        pass

    class FakeFLMethod:
        def __init__(self, moe):
            self.moe = moe

    runner = SimpleNamespace(
        moe_config="moe-config",
        _quant_method=FakeUnquantized(),
        replaced=[],
    )

    def replace(method):
        runner.replaced.append(method)

    runner._replace_quant_method = replace

    factory = _Recorder(result=runner)
    router = _Recorder()
    infos = []

    monkeypatch.setattr(Mod, "_OrigFusedMoE", factory)
    monkeypatch.setattr(Mod, "UnquantizedFusedMoEMethod", FakeUnquantized)
    monkeypatch.setattr(Mod, "UnquantizedFusedMoEMethodFL", FakeFLMethod)
    monkeypatch.setattr(Mod, "replace_router_with_fl", router)
    monkeypatch.setattr(
        Mod, "logger", SimpleNamespace(info_once=lambda *a, **k: infos.append(a))
    )

    result = Mod.FusedMoEFL(4, hidden_size=8)

    assert result is runner
    assert factory.calls == [((4,), {"hidden_size": 8})]
    assert len(runner.replaced) == 1
    assert isinstance(runner.replaced[0], FakeFLMethod)
    assert runner.replaced[0].moe == "moe-config"
    assert router.calls == [((), {})]
    assert infos == []


def test_factory_preserves_quantized_method(monkeypatch):
    class FakeUnquantized:
        pass

    quant_method = object()
    runner = SimpleNamespace(
        moe_config="moe-config",
        _quant_method=quant_method,
        replaced=[],
    )
    factory = _Recorder(result=runner)
    router = _Recorder()
    infos = []

    monkeypatch.setattr(Mod, "_OrigFusedMoE", factory)
    monkeypatch.setattr(Mod, "UnquantizedFusedMoEMethod", FakeUnquantized)
    monkeypatch.setattr(Mod, "replace_router_with_fl", router)
    monkeypatch.setattr(
        Mod, "logger", SimpleNamespace(info_once=lambda *a, **k: infos.append(a))
    )

    result = Mod.FusedMoEFL()

    assert result is runner
    assert runner.replaced == []
    assert router.calls == [((), {})]
    assert infos and infos[0][0].startswith("Preserving upstream quantized")
