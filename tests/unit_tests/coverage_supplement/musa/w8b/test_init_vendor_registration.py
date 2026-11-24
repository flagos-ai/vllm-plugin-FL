# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for vllm_fl top-level registration helpers.

The real ``vllm_fl/__init__.py`` runs vendor detection at import time.  To
exercise its branch logic hermetically we exec only the function definitions
and install fake vLLM/vendor modules in ``sys.modules``; every vendor
environment-variable combination is driven via ``monkeypatch``.
"""

from __future__ import annotations

import ast
import importlib
import logging
import os
import pathlib
import sys
import types
from threading import Lock
from types import SimpleNamespace

import pytest


def _find_source(rel: str) -> pathlib.Path:
    override = os.environ.get("VLLM_FL_SRC")
    candidates = []
    if override:
        candidates.append(pathlib.Path(override) / rel)
    candidates.append(pathlib.Path("/tmp/flsrc") / rel)
    for parent in pathlib.Path(__file__).resolve().parents:
        candidates.append(parent / rel)
    candidates.append(pathlib.Path.cwd() / rel)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"could not locate {rel}")


_FUNCS = {
    "_arm_cpu_platform",
    "__getattr__",
    "_register_flagcx_connector",
    "_patch_flash_attn_import",
    "_is_metax_backend",
    "_patch_custom_ops",
    "_init_vendor_device",
    "_install_metax_compat",
    "_install_kunlunxin_fla_alias",
    "register",
    "register_quant_linear",
    "register_router",
    "_register_gdn_packed_decode_patch",
    "register_model",
}


@pytest.fixture(scope="module")
def initmod():
    path = _find_source("vllm_fl/__init__.py")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in _FUNCS
    ]
    mod = types.ModuleType("w8b_vllm_fl_init")
    mod.__dict__.update(
        {
            "os": os,
            "sys": sys,
            "importlib": importlib,
            "logging": logging,
            "logger": logging.getLogger("w8b.init"),
            "Lock": Lock,
            "_kunlunxin_fla_alias_lock": Lock(),
            "__name__": "w8b_vllm_fl_init",
            "_get_op_config": lambda: None,
        }
    )
    exec(
        compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"),
        mod.__dict__,
    )
    return mod


def _ensure(monkeypatch, name):
    existing = sys.modules.get(name)
    if isinstance(existing, types.ModuleType):
        return existing
    module = types.ModuleType(name)
    module.__path__ = []
    monkeypatch.setitem(sys.modules, name, module)
    if "." in name:
        parent_name, leaf = name.rsplit(".", 1)
        parent = _ensure(monkeypatch, parent_name)
        monkeypatch.setattr(parent, leaf, module, raising=False)
    return module


def _stub(monkeypatch, name, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    if "." in name:
        parent_name, leaf = name.rsplit(".", 1)
        parent = _ensure(monkeypatch, parent_name)
        monkeypatch.setattr(parent, leaf, module, raising=False)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def _pkg(monkeypatch, name):
    module = types.ModuleType(name)
    module.__path__ = []
    monkeypatch.setitem(sys.modules, name, module)
    return module


def _utils(monkeypatch, vendor, oot=False):
    class DeviceInfo:
        def __init__(self):
            self.vendor_name = vendor

    return _stub(
        monkeypatch,
        "vllm_fl.utils",
        DeviceInfo=DeviceInfo,
        get_op_config=lambda: {"vendor": vendor},
        is_oot_enabled=lambda: oot,
    )


def test_arm_cpu_platform_detection(initmod, monkeypatch):
    import platform as platform_mod

    monkeypatch.setattr(platform_mod, "machine", lambda: "x86_64")
    assert initmod._arm_cpu_platform() is None

    sentinel = object()
    _stub(monkeypatch, "vllm.platforms", cpu_platform_plugin=lambda: sentinel)
    monkeypatch.setattr(platform_mod, "machine", lambda: "AARCH64")
    assert initmod._arm_cpu_platform() is sentinel
    monkeypatch.setattr(platform_mod, "machine", lambda: "arm64")
    assert initmod._arm_cpu_platform() is sentinel


def test_getattr_lazily_imports_distributed(initmod, monkeypatch):
    recorded = []
    target = types.ModuleType("w8b.distributed")

    def fake_import(name, package=None):
        recorded.append((name, package))
        return target

    monkeypatch.setattr(importlib, "import_module", fake_import)
    assert initmod.__getattr__("distributed") is target
    assert recorded == [(".distributed", "w8b_vllm_fl_init")]
    assert initmod.distributed is target
    with pytest.raises(AttributeError):
        initmod.__getattr__("missing")


def test_register_flagcx_connector_registers_aliases(initmod, monkeypatch):
    calls = []

    class KVConnectorFactory:
        _registry = {}

        @classmethod
        def register_connector(cls, alias, module, name):
            calls.append((alias, module, name))
            cls._registry[alias] = name

    _stub(
        monkeypatch,
        "vllm.distributed.kv_transfer.kv_connector.factory",
        KVConnectorFactory=KVConnectorFactory,
    )
    initmod._register_flagcx_connector()
    initmod._register_flagcx_connector()
    assert [c[0] for c in calls] == ["FlagCXConnector", "FlagcxConnector"]
    assert all(
        c[1] == "vllm_fl.distributed.kv_transfer.flagcx_connector"
        and c[2] == "FlagCXConnector"
        for c in calls
    )


def test_patch_flash_attn_import_returns_when_present(initmod, monkeypatch):
    existing = types.ModuleType("vllm.vllm_flash_attn")
    existing.marker = object()
    monkeypatch.setitem(sys.modules, "vllm.vllm_flash_attn", existing)
    initmod._patch_flash_attn_import()
    assert sys.modules["vllm.vllm_flash_attn"] is existing


def test_patch_flash_attn_import_builds_fallback(initmod, monkeypatch):
    _pkg(monkeypatch, "vllm")
    monkeypatch.delitem(sys.modules, "vllm.vllm_flash_attn", raising=False)
    monkeypatch.delitem(
        sys.modules, "vllm.vllm_flash_attn.flash_attn_interface", raising=False
    )
    initmod._patch_flash_attn_import()
    stub = sys.modules["vllm.vllm_flash_attn"]
    assert stub.FA2_AVAILABLE is False and stub.FA3_AVAILABLE is False
    assert stub.flash_attn_varlen_func is None
    assert stub.compile_flash_attn_varlen_func_from_specs is None
    assert stub.get_scheduler_metadata is None
    assert "not available" in stub.fa_version_unsupported_reason()
    assert stub.is_fa_version_supported() is False


def test_is_metax_backend_delegates(initmod, monkeypatch):
    sentinel = {"value": True}
    _stub(
        monkeypatch,
        "vllm_fl.ops._C_ops_registry",
        _is_metax_backend=lambda: sentinel["value"],
        register_op_schemas=lambda: None,
    )
    assert initmod._is_metax_backend() is True
    sentinel["value"] = False
    assert initmod._is_metax_backend() is False


def test_patch_custom_ops_skips_on_metax(initmod, monkeypatch):
    calls = []
    _stub(
        monkeypatch,
        "vllm_fl.ops._C_ops_registry",
        _is_metax_backend=lambda: True,
        register_op_schemas=lambda: calls.append("register"),
    )
    initmod._patch_custom_ops()
    assert calls == []


def test_patch_custom_ops_returns_when_vllm_extension_imports(initmod, monkeypatch):
    calls = []
    _stub(
        monkeypatch,
        "vllm_fl.ops._C_ops_registry",
        _is_metax_backend=lambda: False,
        register_op_schemas=lambda: calls.append("register"),
    )
    monkeypatch.setitem(sys.modules, "vllm._C", types.ModuleType("vllm._C"))
    initmod._patch_custom_ops()
    assert calls == []


def test_patch_custom_ops_registers_fallback_schemas(initmod, monkeypatch):
    calls = []
    _stub(
        monkeypatch,
        "vllm_fl.ops._C_ops_registry",
        _is_metax_backend=lambda: False,
        register_op_schemas=lambda: calls.append("register"),
    )
    _pkg(monkeypatch, "vllm")
    _pkg(monkeypatch, "vllm_fl")
    for name in ("vllm._C", "vllm._C_stable_libtorch", "vllm_fl._C"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    initmod._patch_custom_ops()
    assert calls == ["register"]


def test_install_metax_compat_paths(initmod, monkeypatch):
    calls = []
    state = {"active": False, "raise": False}

    def install():
        calls.append("install")
        if state["raise"]:
            raise RuntimeError("boom")

    _stub(
        monkeypatch,
        "vllm_fl.dispatch.backends.vendor.metax.shims",
        install=install,
        metax_active=lambda: state["active"],
    )
    initmod._install_metax_compat()
    assert calls == []
    state["active"] = True
    initmod._install_metax_compat()
    assert calls == ["install"]
    state["raise"] = True
    initmod._install_metax_compat()  # exception swallowed
    assert calls == ["install", "install"]


def test_install_kunlunxin_alias_success_and_swallow(initmod, monkeypatch):
    calls = []
    state = {"raise": False}

    def install():
        calls.append("fla")
        if state["raise"]:
            raise RuntimeError("boom")

    _stub(
        monkeypatch,
        "vllm_fl.dispatch.backends.vendor.kunlunxin.patches.patch_fla_utils",
        _fix_platform_vars=lambda: calls.append("fix"),
        _patch_xpu_get_device=lambda: calls.append("xpu"),
    )
    _stub(monkeypatch, "vllm_fl.patches.fla_module_relocation", install=install)
    initmod._install_kunlunxin_fla_alias()
    assert calls == ["fla", "fix"]
    assert initmod._kunlunxin_fla_alias_lock.locked() is False
    state["raise"] = True
    initmod._install_kunlunxin_fla_alias()
    assert calls[-1] == "fla"
    assert initmod._kunlunxin_fla_alias_lock.locked() is False


def _vendor_env(initmod, monkeypatch, vendor, *, musa_ok=True, metax=False):
    """Install every fake module register() may import; return recorders."""
    rec = {
        "musa": [],
        "kunlunxin": [],
        "amd": [],
        "op_config": 0,
        "metax_probe": [],
        "schemas": [],
        "flagcx": [],
    }
    _utils(monkeypatch, vendor, oot=False)
    _pkg(monkeypatch, "vllm")
    _pkg(monkeypatch, "vllm_fl")

    def op_config():
        rec["op_config"] += 1

    monkeypatch.setattr(initmod, "_get_op_config", op_config)
    _stub(
        monkeypatch,
        "vllm_fl.dispatch.backends.vendor.metax.shims",
        install=lambda: None,
        metax_active=lambda: False,
    )
    _stub(
        monkeypatch,
        "vllm_fl.dispatch.backends.vendor.musa.patch",
        patch_accelerator_missing_attrs=lambda: (
            rec["musa"].append("patch") or musa_ok
        ),
    )
    _stub(
        monkeypatch,
        "vllm_fl.dispatch.backends.vendor.kunlunxin.patches.patch_fla_utils",
        _patch_xpu_get_device=lambda: rec["kunlunxin"].append("xpu"),
        _fix_platform_vars=lambda: rec["kunlunxin"].append("fix"),
    )
    _stub(
        monkeypatch,
        "vllm_fl.patches.flagtree_target_info",
        apply_flagtree_target_info=lambda: rec["amd"].append("apply"),
    )
    _stub(
        monkeypatch,
        "vllm_fl.patches.fla_module_relocation",
        install=lambda: rec["kunlunxin"].append("fla"),
    )

    def is_metax():
        rec["metax_probe"].append("probe")
        return metax

    _stub(
        monkeypatch,
        "vllm_fl.ops._C_ops_registry",
        _is_metax_backend=is_metax,
        register_op_schemas=lambda: rec["schemas"].append("register"),
    )
    monkeypatch.setitem(sys.modules, "vllm._C", types.ModuleType("vllm._C"))
    monkeypatch.setitem(
        sys.modules, "vllm.vllm_flash_attn", types.ModuleType("vllm.vllm_flash_attn")
    )

    class KVConnectorFactory:
        _registry = {}

        @classmethod
        def register_connector(cls, alias, module, name):
            rec["flagcx"].append(alias)
            cls._registry[alias] = name

    # register() defers connector registration, register_model() uses it.
    _stub(
        monkeypatch,
        "vllm.distributed.kv_transfer.kv_connector.factory",
        KVConnectorFactory=KVConnectorFactory,
    )
    return rec


@pytest.mark.parametrize(
    "vendor,expected",
    [
        ("mthreads", "vllm_fl.platform.PlatformFL"),
        ("kunlunxin", "vllm_fl.platform.PlatformFL"),
        ("amd", "vllm_fl.platform.PlatformFL"),
        ("iluvatar", "vllm_fl.platform.PlatformFL"),
        ("enflame", "vllm_fl.platform.PlatformFL"),
        ("gcu", "vllm_fl.platform.PlatformFL"),
        ("tsingmicro", "vllm_fl.platform.PlatformFL"),
        ("nvidia", "vllm_fl.nvidia_platform.NvidiaPlatformFL"),
    ],
)
def test_register_returns_platform_for_each_vendor(
    initmod, monkeypatch, vendor, expected
):
    import platform as platform_mod

    monkeypatch.setattr(platform_mod, "machine", lambda: "x86_64")
    monkeypatch.delenv("VLLM_WORKER_MULTIPROC_METHOD", raising=False)
    monkeypatch.delenv("VLLM_USE_V2_MODEL_RUNNER", raising=False)
    rec = _vendor_env(initmod, monkeypatch, vendor)

    assert initmod.register() == expected
    assert os.environ["VLLM_WORKER_MULTIPROC_METHOD"] == "spawn"
    assert rec["op_config"] == 1

    if vendor == "nvidia":
        assert rec["metax_probe"] == []
    else:
        assert rec["metax_probe"] == ["probe"]

    if vendor == "mthreads":
        assert rec["musa"] == ["patch"]
    if vendor == "kunlunxin":
        assert rec["kunlunxin"] == ["xpu", "fla", "fix"]
    if vendor == "amd":
        assert rec["amd"] == ["apply"]
    if vendor in ("enflame", "gcu", "tsingmicro"):
        assert os.environ["VLLM_USE_V2_MODEL_RUNNER"] == "0"
    else:
        assert "VLLM_USE_V2_MODEL_RUNNER" not in os.environ


def test_register_preserves_existing_multiproc_method(initmod, monkeypatch):
    import platform as platform_mod

    monkeypatch.setattr(platform_mod, "machine", lambda: "x86_64")
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "fork")
    _vendor_env(initmod, monkeypatch, "mthreads")
    assert initmod.register() == "vllm_fl.platform.PlatformFL"
    assert os.environ["VLLM_WORKER_MULTIPROC_METHOD"] == "fork"


def test_register_musa_allocator_failure_raises(initmod, monkeypatch):
    import platform as platform_mod

    monkeypatch.setattr(platform_mod, "machine", lambda: "x86_64")
    monkeypatch.delenv("VLLM_WORKER_MULTIPROC_METHOD", raising=False)
    _vendor_env(initmod, monkeypatch, "mthreads", musa_ok=False)
    with pytest.raises(RuntimeError):
        initmod.register()


def test_register_arm_cpu_shortcut(initmod, monkeypatch):
    import platform as platform_mod

    monkeypatch.setattr(platform_mod, "machine", lambda: "aarch64")
    sentinel = "cpu-platform"
    _stub(monkeypatch, "vllm.platforms", cpu_platform_plugin=lambda: sentinel)
    _utils(monkeypatch, "nvidia")
    _stub(
        monkeypatch,
        "vllm_fl.dispatch.backends.vendor.metax.shims",
        install=lambda: pytest.fail("metax shims must not install"),
        metax_active=lambda: False,
    )
    assert initmod.register() is sentinel


def test_register_quant_linear_skips_unsupported(initmod, monkeypatch):
    calls = []
    _stub(
        monkeypatch,
        "vllm.platforms",
        current_platform=SimpleNamespace(device_type="musa"),
    )
    _stub(
        monkeypatch,
        "vllm_fl.quantization.quant_linear",
        add_oot_quant_kernel=lambda: calls.append("add"),
    )
    initmod.register_quant_linear()
    _stub(
        monkeypatch,
        "vllm.platforms",
        current_platform=SimpleNamespace(device_type="gcu"),
    )
    initmod.register_quant_linear()
    assert calls == []

    _stub(
        monkeypatch,
        "vllm.platforms",
        current_platform=SimpleNamespace(device_type="npu"),
    )
    initmod.register_quant_linear()
    assert calls == ["add"]


def test_register_router_branches(initmod, monkeypatch):
    calls = []

    def replace():
        calls.append("replace")

    _utils(monkeypatch, "npu", oot=False)
    _stub(
        monkeypatch,
        "vllm_fl.ops.fused_moe.router",
        replace_router_with_fl=replace,
    )
    _stub(
        monkeypatch,
        "vllm.platforms",
        current_platform=SimpleNamespace(device_type="musa"),
    )
    initmod.register_router()
    assert calls == []

    _stub(
        monkeypatch,
        "vllm.platforms",
        current_platform=SimpleNamespace(device_type="npu"),
    )
    initmod.register_router()  # is_oot_enabled False
    assert calls == []

    _utils(monkeypatch, "npu", oot=True)
    initmod.register_router()
    assert calls == ["replace"]


def test_gdn_packed_decode_patch_outcomes(initmod, monkeypatch):
    _stub(
        monkeypatch,
        "vllm_fl.patches.gdn_packed_decode",
        patch_vllm_packed_gdn_beta=lambda: True,
    )
    assert initmod._register_gdn_packed_decode_patch() is True
    _stub(
        monkeypatch,
        "vllm_fl.patches.gdn_packed_decode",
        patch_vllm_packed_gdn_beta=lambda: False,
    )
    assert initmod._register_gdn_packed_decode_patch() is False
    _stub(monkeypatch, "vllm_fl.patches.gdn_packed_decode")
    assert initmod._register_gdn_packed_decode_patch() is False
    monkeypatch.setitem(sys.modules, "vllm_fl.patches.gdn_packed_decode", None)
    monkeypatch.delitem(
        sys.modules, "vllm_fl.patches.gdn_packed_decode", raising=False
    )
    _pkg(monkeypatch, "vllm_fl")
    _pkg(monkeypatch, "vllm_fl.patches")
    assert initmod._register_gdn_packed_decode_patch() is False


def _model_env(monkeypatch, device_type):
    _pkg(monkeypatch, "vllm")
    _pkg(monkeypatch, "vllm_fl")
    rec = {"gdn": 0, "w4a8": 0, "shutdown": 0, "minicpmo": 0, "add": 0,
           "replace": 0, "models": [], "flagcx": []}
    _stub(
        monkeypatch,
        "vllm.platforms",
        current_platform=SimpleNamespace(device_type=device_type),
        cpu_platform_plugin=lambda: "cpu-platform",
    )
    _stub(
        monkeypatch,
        "vllm_fl.patches.arm_cpu_gdn",
        apply_arm_cpu_gdn_state_indices_patch=lambda: rec.__setitem__(
            "gdn", rec["gdn"] + 1
        ),
    )
    _stub(
        monkeypatch,
        "vllm_fl.quantization.arm_cpu_w4a8",
        install_arm_cpu_packed_w4a8=lambda: rec.__setitem__(
            "w4a8", rec["w4a8"] + 1
        ),
    )
    _stub(
        monkeypatch,
        "vllm_fl.dispatch.backends.vendor.iluvatar.shutdown",
        install_worker_cleanup=lambda: rec.__setitem__(
            "shutdown", rec["shutdown"] + 1
        ),
    )
    _stub(
        monkeypatch,
        "vllm_fl.patches.minicpmo_audio",
        patch_minicpmo_audio_budget=lambda: rec.__setitem__(
            "minicpmo", rec["minicpmo"] + 1
        ),
    )
    _stub(
        monkeypatch,
        "vllm_fl.quantization.quant_linear",
        add_oot_quant_kernel=lambda: rec.__setitem__("add", rec["add"] + 1),
    )
    _stub(
        monkeypatch,
        "vllm_fl.ops.fused_moe.router",
        replace_router_with_fl=lambda: rec.__setitem__("replace", rec["replace"] + 1),
    )
    _stub(
        monkeypatch,
        "vllm_fl.patches.gdn_packed_decode",
        patch_vllm_packed_gdn_beta=lambda: rec.__setitem__("gdn", rec["gdn"] + 1)
        or True,
    )

    class ModelRegistry:
        @staticmethod
        def register_model(name, target):
            rec["models"].append((name, target))

    _stub(
        monkeypatch,
        "vllm.model_executor.models",
        ModelRegistry=ModelRegistry,
    )
    _stub(monkeypatch, "vllm_fl.quantization.modelslim_w8a8")

    class KVConnectorFactory:
        _registry = {}

        @classmethod
        def register_connector(cls, alias, module, name):
            rec["flagcx"].append(alias)
            cls._registry[alias] = name

    _stub(
        monkeypatch,
        "vllm.distributed.kv_transfer.kv_connector.factory",
        KVConnectorFactory=KVConnectorFactory,
    )
    return rec


def test_register_model_cpu_arm_installs_packed_w4a8(initmod, monkeypatch):
    import platform as platform_mod

    monkeypatch.setattr(platform_mod, "machine", lambda: "aarch64")
    rec = _model_env(monkeypatch, "cpu")
    _pkg(monkeypatch, "flag_gems")
    flag_gems = types.ModuleType("flag_gems")
    flag_gems.vendor_name = "arm"
    monkeypatch.setitem(sys.modules, "flag_gems", flag_gems)
    assert initmod.register_model() is None
    assert rec["gdn"] == 1 and rec["w4a8"] == 1
    assert rec["flagcx"] == []  # returns before connector registration
    assert rec["add"] == 0


def test_register_model_cpu_arm_non_arm_vendor_skips(initmod, monkeypatch):
    import platform as platform_mod

    monkeypatch.setattr(platform_mod, "machine", lambda: "aarch64")
    rec = _model_env(monkeypatch, "cpu")
    flag_gems = types.ModuleType("flag_gems")
    flag_gems.vendor_name = "mthreads"
    monkeypatch.setitem(sys.modules, "flag_gems", flag_gems)
    assert initmod.register_model() is None
    assert rec["gdn"] == 1 and rec["w4a8"] == 0


def test_register_model_ascend_registers_models(initmod, monkeypatch):
    rec = _model_env(monkeypatch, "npu")
    _utils(monkeypatch, "ascend", oot=False)
    assert initmod.register_model() is None
    assert rec["flagcx"] == ["FlagCXConnector", "FlagcxConnector"]
    assert rec["add"] == 1
    assert rec["gdn"] == 1
    assert rec["minicpmo"] == 0
    assert rec["shutdown"] == 0
    assert [name for name, _ in rec["models"]] == [
        "DeepseekV4ForCausalLM",
        "GlmMoeDsaForCausalLM",
    ]
    assert all(
        target.startswith("vllm_fl.models.")
        for _, target in rec["models"]
    )


def test_register_model_nvidia_skips_gdn_installs_audio_patch(initmod, monkeypatch):
    rec = _model_env(monkeypatch, "cuda")
    _utils(monkeypatch, "nvidia", oot=False)
    assert initmod.register_model() is None
    assert rec["gdn"] == 0  # NVIDIA keeps upstream packed GDN precision
    assert rec["minicpmo"] == 1
    assert rec["add"] == 1
    assert rec["flagcx"] == ["FlagCXConnector", "FlagcxConnector"]
    assert rec["models"] == []


def test_register_model_iluvatar_installs_cleanup(initmod, monkeypatch):
    rec = _model_env(monkeypatch, "npu")
    _utils(monkeypatch, "iluvatar", oot=False)
    assert initmod.register_model() is None
    assert rec["shutdown"] == 1
    assert rec["gdn"] == 1
    assert rec["minicpmo"] == 0


def test_register_model_cpu_non_arm_takes_generic_path(initmod, monkeypatch):
    import platform as platform_mod

    monkeypatch.setattr(platform_mod, "machine", lambda: "x86_64")
    rec = _model_env(monkeypatch, "cpu")
    _utils(monkeypatch, "npu", oot=False)
    assert initmod.register_model() is None
    assert rec["gdn"] == 1
    assert rec["w4a8"] == 0
    assert rec["gdn"] == 1 and rec["add"] == 1
