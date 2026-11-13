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

"""CPU tests for OOT operator registration in ``vllm_fl.ops.custom_ops``.

The vLLM registration bases, the upstream oracle/factory modules and the
``vllm_fl.utils`` policy helpers are replaced with recorders.  The tests
assert whitelist/blacklist routing, the fused-MoE factory condition and the
platform-patch fan-out without importing any vendor backend.
"""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest

from . import _imports  # noqa: F401  (installs CPU-only import placeholders)
from vllm_fl import utils as fl_utils
from vllm_fl.dispatch.config import utils as dispatch_config_utils
from vllm_fl.ops import custom_ops as Mod


class _Registrar:
    calls = []


_LOG = []


class _PluggableBase(_Registrar):
    @classmethod
    def register_oot(cls, _decorated_layer_cls=None, name=None):
        cls.calls.append((_decorated_layer_cls, name))


class _CustomBase(_Registrar):
    @classmethod
    def register_oot(cls, _decorated_op_cls=None, name=None):
        cls.calls.append((_decorated_op_cls, name))


@pytest.fixture(autouse=True)
def _clean_registrars():
    _Registrar.calls = []
    _LOG.clear()
    yield
    _Registrar.calls = []
    _LOG.clear()


def _logger():
    return SimpleNamespace(
        warning=lambda *a, **k: _LOG.append(("warning", a)),
        info=lambda *a, **k: _LOG.append(("info", a)),
        debug=lambda *a, **k: _LOG.append(("debug", a)),
        info_once=lambda *a, **k: _LOG.append(("info_once", a)),
    )


def _policy(monkeypatch, *, oot=True, whitelist=None, blacklist=None, ops=None):
    monkeypatch.setattr(fl_utils, "is_oot_enabled", lambda: oot)
    monkeypatch.setattr(fl_utils, "get_oot_whitelist", lambda: whitelist)
    monkeypatch.setattr(fl_utils, "get_oot_blacklist", lambda: blacklist)
    ops = {} if ops is None else ops
    monkeypatch.setattr(fl_utils, "use_flaggems_op", lambda op: ops.get(op, True))


def test_register_returns_early_when_oot_disabled(monkeypatch):
    _policy(monkeypatch, oot=False)
    oracle_calls = []
    monkeypatch.setattr(
        Mod, "_patch_unquantized_moe_oracle", lambda **kw: oracle_calls.append(kw)
    )
    Mod.register_oot_ops()
    assert oracle_calls == [{"prefer_flaggems_experts": False}]


def test_register_applies_whitelist_blacklist_and_unknown_warning(monkeypatch):
    class Silu(_CustomBase):
        pass

    class Unquantized(_CustomBase):
        pass

    _policy(
        monkeypatch,
        whitelist=["silu_and_mul", "unquantized_fused_moe_method", "bogus"],
        blacklist=[],
        ops={"unquantized_fused_moe_method": False},
    )
    monkeypatch.setattr(Mod, "PluggableLayer", _PluggableBase)
    monkeypatch.setattr(Mod, "CustomOp", _CustomBase)
    monkeypatch.setattr(Mod, "logger", _logger())
    monkeypatch.setattr(
        Mod,
        "OOT_OPS",
        {
            "silu_and_mul": (Silu, "SiluAndMul"),
            "unquantized_fused_moe_method": (Unquantized, "Unquantized"),
        },
    )
    oracle_calls = []
    monkeypatch.setattr(
        Mod, "_patch_unquantized_moe_oracle", lambda **kw: oracle_calls.append(kw)
    )
    factory_calls = []
    monkeypatch.setattr(
        Mod, "_patch_fused_moe_factory", lambda: factory_calls.append(True)
    )

    Mod.register_oot_ops()

    assert _Registrar.calls[0] == (Silu, "SiluAndMul")
    warnings = [call for call in _LOG if call[0] == "warning"]
    assert "bogus" in warnings[0][1][0]
    assert oracle_calls == [{"prefer_flaggems_experts": False}]
    assert factory_calls == []


def test_register_registers_all_ops_and_enables_fused_moe(monkeypatch):
    class Pluggable(_PluggableBase):
        pass

    class Clamp(_CustomBase):
        pass

    _policy(monkeypatch, whitelist=None, blacklist=None)
    monkeypatch.setattr(Mod, "PluggableLayer", _PluggableBase)
    monkeypatch.setattr(Mod, "CustomOp", _CustomBase)
    monkeypatch.setattr(Mod, "logger", _logger())
    monkeypatch.setattr(
        Mod,
        "OOT_OPS",
        {
            "silu_and_mul": (Pluggable, "SiluAndMul"),
            "silu_and_mul_with_clamp": (Clamp, "SiluAndMulWithClamp"),
        },
    )
    oracle_calls = []
    monkeypatch.setattr(
        Mod, "_patch_unquantized_moe_oracle", lambda **kw: oracle_calls.append(kw)
    )
    factory_calls = []
    monkeypatch.setattr(
        Mod, "_patch_fused_moe_factory", lambda: factory_calls.append(True)
    )

    Mod.register_oot_ops()

    assert (Pluggable, "SiluAndMul") in _Registrar.calls
    assert (Clamp, "SiluAndMulWithClamp") in _Registrar.calls
    assert Clamp.name == "silu_and_mul_with_clamp"
    assert oracle_calls == [{"prefer_flaggems_experts": True}]
    assert factory_calls == [True]


def test_register_honours_explicit_whitelist_argument(monkeypatch):
    class Silu(_CustomBase):
        pass

    _policy(monkeypatch, whitelist=None, blacklist=None)
    monkeypatch.setattr(Mod, "PluggableLayer", _PluggableBase)
    monkeypatch.setattr(Mod, "CustomOp", _CustomBase)
    monkeypatch.setattr(Mod, "logger", _logger())
    monkeypatch.setattr(Mod, "OOT_OPS", {"silu_and_mul": (Silu, "SiluAndMul")})
    oracle_calls = []
    monkeypatch.setattr(
        Mod, "_patch_unquantized_moe_oracle", lambda **kw: oracle_calls.append(kw)
    )
    factory_calls = []
    monkeypatch.setattr(
        Mod, "_patch_fused_moe_factory", lambda: factory_calls.append(True)
    )

    Mod.register_oot_ops(whitelist=["silu_and_mul"])

    assert _Registrar.calls == [(Silu, "SiluAndMul")]
    assert oracle_calls == [{"prefer_flaggems_experts": False}]
    assert factory_calls == []


def test_register_blacklists_fused_moe_and_skips_factory(monkeypatch):
    _policy(monkeypatch, whitelist=None, blacklist=["fused_moe"])
    monkeypatch.setattr(Mod, "OOT_OPS", {})
    monkeypatch.setattr(Mod, "logger", _logger())
    oracle_calls = []
    monkeypatch.setattr(
        Mod, "_patch_unquantized_moe_oracle", lambda **kw: oracle_calls.append(kw)
    )
    factory_calls = []
    monkeypatch.setattr(
        Mod, "_patch_fused_moe_factory", lambda: factory_calls.append(True)
    )

    Mod.register_oot_ops()

    assert oracle_calls == [{"prefer_flaggems_experts": False}]
    assert factory_calls == []


def test_register_applies_vendor_patches_for_platform(monkeypatch):
    _policy(monkeypatch, whitelist=None, blacklist=None)
    monkeypatch.setattr(Mod, "OOT_OPS", {})
    monkeypatch.setattr(Mod, "logger", _logger())
    monkeypatch.setattr(Mod, "_patch_unquantized_moe_oracle", lambda **kw: None)
    monkeypatch.setattr(Mod, "_patch_fused_moe_factory", lambda: None)

    applied = []
    for name, function in (
        ("vllm_fl.dispatch.backends.vendor.ascend.patch", "apply_ascend_patches"),
        ("vllm_fl.dispatch.backends.vendor.sunrise.patch", "apply_sunrise_patches"),
        ("vllm_fl.dispatch.backends.vendor.gcu.patch", "apply_gcu_patches"),
        (
            "vllm_fl.dispatch.backends.vendor.kunlunxin.patch",
            "apply_kunlunxin_patches",
        ),
    ):
        module = types.ModuleType(name)
        setattr(module, function, lambda n=name: applied.append(n))
        monkeypatch.setitem(sys.modules, name, module)

    infos = []
    monkeypatch.setattr(
        Mod, "logger", SimpleNamespace(info=lambda *a, **k: infos.append(a))
    )

    from vllm import platforms as vllm_platforms

    monkeypatch.setattr(
        vllm_platforms.current_platform, "device_type", "npu", raising=False
    )
    monkeypatch.setattr(
        vllm_platforms.current_platform, "vendor_name", "gcu", raising=False
    )
    monkeypatch.setattr(dispatch_config_utils, "get_platform_name", lambda: "kunlunxin")

    Mod.register_oot_ops()

    assert applied == [
        "vllm_fl.dispatch.backends.vendor.ascend.patch",
        "vllm_fl.dispatch.backends.vendor.gcu.patch",
        "vllm_fl.dispatch.backends.vendor.kunlunxin.patch",
    ]


def test_patch_unquantized_moe_oracle_wraps_both_modules(monkeypatch):
    # The patcher imports these modules by dotted name; on the image the real
    # parents are already loaded, so patch the exact objects it resolves.
    import vllm.model_executor.layers.fused_moe.oracle.unquantized as oracle_module
    import vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method as method_module

    import vllm_fl.ops.fused_moe.fused_moe_utils as utils_module

    calls = []

    def select_oot(moe_config, prefer_flaggems_experts=False):
        calls.append((moe_config, prefer_flaggems_experts))
        return "backend"

    monkeypatch.setattr(utils_module, "select_unquantized_moe_backend_oot", select_oot)
    monkeypatch.setattr(oracle_module, "select_unquantized_moe_backend", object())
    monkeypatch.setattr(method_module, "select_unquantized_moe_backend", object())

    infos = []
    monkeypatch.setattr(
        Mod, "logger", SimpleNamespace(info=lambda *a, **k: infos.append(a))
    )

    Mod._patch_unquantized_moe_oracle(prefer_flaggems_experts=True)

    wrapper = oracle_module.select_unquantized_moe_backend
    assert wrapper is method_module.select_unquantized_moe_backend
    config = object()
    assert wrapper(config) == "backend"
    assert calls == [(config, True)]
    assert infos and "prefer_flaggems_experts" in infos[0][0]


def test_patch_fused_moe_factory_rewrites_bound_model_symbols(monkeypatch):
    # Patch the real defining modules the factory patcher resolves.
    import vllm.model_executor.layers.fused_moe as package
    import vllm.model_executor.layers.fused_moe.layer as layer

    native = object()
    monkeypatch.setattr(layer, "FusedMoE", native)
    monkeypatch.setattr(package, "FusedMoE", native)

    bound = types.ModuleType("vllm.model_executor.models.bound")
    bound.FusedMoE = native
    unrelated = types.ModuleType("vllm.model_executor.models.unrelated")
    unrelated_callable = lambda *a, **k: None
    unrelated.FusedMoE = unrelated_callable
    monkeypatch.setitem(sys.modules, bound.__name__, bound)
    monkeypatch.setitem(sys.modules, unrelated.__name__, unrelated)

    infos = []
    monkeypatch.setattr(
        Mod, "logger", SimpleNamespace(info=lambda *a, **k: infos.append(a))
    )

    Mod._patch_fused_moe_factory()

    assert layer.FusedMoE is Mod.FusedMoEFL
    assert package.FusedMoE is Mod.FusedMoEFL
    assert bound.FusedMoE is Mod.FusedMoEFL
    assert unrelated.FusedMoE is unrelated_callable
    assert infos and infos[0][0].startswith("Monkey-patched FusedMoE factory")

    # Already-patched modules are left untouched on a second call.
    Mod._patch_fused_moe_factory()
    assert layer.FusedMoE is Mod.FusedMoEFL
