# Copyright (c) 2026 BAAI. All rights reserved.

"""Coverage tests for vllm_fl OOT ops and FLA wrappers (S60 / Enflame).

Targets (read-only source, commit 030b452f):
  - vllm_fl/ops/custom_ops.py
  - vllm_fl/ops/_C_ops_registry.py
  - vllm_fl/ops/activation.py
  - vllm_fl/ops/layernorm.py
  - vllm_fl/ops/rotary_embedding.py
  - vllm_fl/ops/__init__.py
  - vllm_fl/ops/fla/{__init__,chunk,fused_recurrent,utils}.py

CPU only.  No kernel, C extension, FlagGems, Triton or device code is ever run:
FLA kernels and the vendored ``vllm._C`` extension are mocked, and heavy module
imports are stubbed in ``sys.modules``.  Assertions are on executed control
flow, return values and mock call plumbing.
"""

from __future__ import annotations

import importlib
import importlib.machinery
import importlib.util
import os
import sys
import types
import uuid
import warnings
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

_MISSING = object()

_REL_REGISTRY = os.path.join("vllm_fl", "ops", "_C_ops_registry.py")


def _source_root():
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.environ.get("VLLM_FL_SRC"),
        r"C:\Users\21649\AppData\Local\Temp\opencode\enflame-src-030b",
        here,
        os.getcwd(),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        base = os.path.abspath(candidate)
        for probe in (base, os.path.dirname(base)):
            if os.path.isfile(os.path.join(probe, _REL_REGISTRY)):
                return probe
    return None


SRC_ROOT = _source_root()
if SRC_ROOT is None:
    pytest.skip(
        "cannot locate vllm_fl source tree; set VLLM_FL_SRC",
        allow_module_level=True,
    )

_FLA_DIR = os.path.join(SRC_ROOT, "vllm_fl", "ops", "fla")


def _new_module(name, attrs=None, is_package=False, path=None):
    module = types.ModuleType(name)
    if is_package:
        module.__path__ = list(path or [])
    for key, value in (attrs or {}).items():
        setattr(module, key, value)
    return module


def _stub_leaf(monkeypatch, name, attrs=None):
    module = _new_module(name, attrs)
    monkeypatch.setitem(sys.modules, name, module)
    return module


# --------------------------------------------------------------------------- #
# _C_ops_registry helpers
# --------------------------------------------------------------------------- #

def _load_registry(
    monkeypatch,
    schemas,
    *,
    native=False,
    mcoplib_present=False,
    mcoplib_importable=False,
    probes=None,
):
    mod_name = "covfl_registry_" + uuid.uuid4().hex
    monkeypatch.setitem(sys.modules, "vllm", _new_module("vllm", is_package=True))
    monkeypatch.setitem(
        sys.modules,
        "vllm._C",
        _new_module("vllm._C") if native else None,
    )
    if mcoplib_present:
        mcoplib = _new_module("mcoplib", is_package=True)
        mcoplib.__spec__ = importlib.machinery.ModuleSpec("mcoplib", loader=None)
        monkeypatch.setitem(sys.modules, "mcoplib", mcoplib)
        if mcoplib_importable:
            monkeypatch.setitem(
                sys.modules, "mcoplib._C", _new_module("mcoplib._C")
            )
    else:
        monkeypatch.setitem(sys.modules, "mcoplib", None)

    schema_mod = _new_module("vllm_fl.ops._C_ops_schemas", {"SCHEMAS": schemas})
    monkeypatch.setitem(sys.modules, "vllm_fl.ops._C_ops_schemas", schema_mod)

    path = os.path.join(SRC_ROOT, _REL_REGISTRY)
    spec = importlib.util.spec_from_file_location(mod_name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, mod_name, module)
    spec.loader.exec_module(module)

    if probes is not None:
        metax, present = probes
        monkeypatch.setattr(module, "_is_metax_backend", lambda: metax)
        monkeypatch.setattr(module, "_C_ops_present", lambda: present)
    return module


def _load_registry_simple():
    name = "covfl_registry_simple_" + uuid.uuid4().hex
    path = os.path.join(SRC_ROOT, _REL_REGISTRY)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _install_device_info(monkeypatch, vendor=None, raises=False):
    if raises:
        class DeviceInfo:  # noqa: N801
            def __init__(self):
                raise RuntimeError("DeviceInfo probe failed")
    else:
        class DeviceInfo:  # noqa: N801
            def __init__(self):
                self.vendor_name = vendor

    module = _new_module("vllm_fl.utils", {"DeviceInfo": DeviceInfo})
    monkeypatch.setitem(sys.modules, "vllm_fl.utils", module)
    return module


def _destroy_registry_libs(module):
    for lib in getattr(module.register_op_schemas, "_libs", {}).values():
        lib._destroy()


# --------------------------------------------------------------------------- #
# FLA package loader (real files, stubbed dependencies)
# --------------------------------------------------------------------------- #

class _FakeCustomOp:
    registry = {}

    def __init__(self, *args, **kwargs):
        pass

    @classmethod
    def register(cls, name):
        def decorator(klass):
            cls.registry[name] = klass
            klass._fl_registration_name = name
            return klass

        return decorator


def _load_fla_package(pkg_name, use_gems):
    saved = {}

    def put(name, module):
        if name not in saved:
            saved[name] = sys.modules.get(name, _MISSING)
        sys.modules[name] = module
        parent, _, leaf = name.rpartition(".")
        if parent:
            parent_mod = sys.modules.get(parent)
            if parent_mod is not None:
                setattr(parent_mod, leaf, module)

    try:
        put("vllm", _new_module("vllm", is_package=True))
        put("vllm.model_executor",
            _new_module("vllm.model_executor", is_package=True))
        put("vllm.model_executor.custom_op",
            _new_module("vllm.model_executor.custom_op",
                        {"CustomOp": _FakeCustomOp}))

        put("vllm.third_party",
            _new_module("vllm.third_party", is_package=True))
        put("vllm.third_party.flash_linear_attention",
            _new_module("vllm.third_party.flash_linear_attention",
                        is_package=True))
        ops_pkg = _new_module("vllm.third_party.flash_linear_attention.ops",
                              is_package=True)
        put("vllm.third_party.flash_linear_attention.ops", ops_pkg)
        put("vllm.third_party.flash_linear_attention.ops.l2norm",
            _new_module(
                "vllm.third_party.flash_linear_attention.ops.l2norm",
                {"l2norm_fwd": MagicMock(name="l2norm_fwd")},
            ))
        put("vllm.third_party.flash_linear_attention.ops.chunk",
            _new_module(
                "vllm.third_party.flash_linear_attention.ops.chunk",
                {"chunk_gated_delta_rule_fwd":
                    MagicMock(name="upstream_chunk_fwd")},
            ))
        put("vllm.third_party.flash_linear_attention.ops.fused_recurrent",
            _new_module(
                "vllm.third_party.flash_linear_attention.ops.fused_recurrent",
                {"fused_recurrent_gated_delta_rule_fwd":
                    MagicMock(name="upstream_recurrent_fwd")},
            ))

        put("vllm_fl", _new_module("vllm_fl", is_package=True))
        put(
            "vllm_fl.utils",
            _new_module(
                "vllm_fl.utils",
                {"use_flaggems_op": lambda op_name, default=True: use_gems},
            ),
        )

        if use_gems:
            put("flag_gems", _new_module("flag_gems", is_package=True))
            put("flag_gems.fused",
                _new_module("flag_gems.fused", is_package=True))
            put(
                "flag_gems.fused.FLA",
                _new_module(
                    "flag_gems.fused.FLA",
                    {
                        "chunk_gated_delta_rule_fwd":
                            MagicMock(name="gems_chunk_fwd"),
                        "fused_recurrent_gated_delta_rule_fwd":
                            MagicMock(name="gems_recurrent_fwd"),
                    },
                ),
            )

        spec = importlib.util.spec_from_file_location(
            pkg_name,
            os.path.join(_FLA_DIR, "__init__.py"),
            submodule_search_locations=[_FLA_DIR],
        )
        package = importlib.util.module_from_spec(spec)
        sys.modules[pkg_name] = package
        spec.loader.exec_module(package)
        return package
    finally:
        for name in reversed(list(saved)):
            previous = saved[name]
            if previous is _MISSING:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


_FLA_UPSTREAM = _load_fla_package("covfl_fla_upstream", False)
_FLA_GEMS = _load_fla_package("covfl_fla_gems", True)

# =========================================================================== #
# _C_ops_registry
# =========================================================================== #

def test_log_amd_host_view_once_is_cached():
    registry = _load_registry_simple()
    assert registry._log_amd_host_view_once() is None
    assert registry._log_amd_host_view_once() is None


def test_unavailable_cuda_view_raises():
    with pytest.raises(RuntimeError, match="mapped CUDA view"):
        _load_registry_simple()._unavailable_cuda_view(torch.zeros(1))


def test_unavailable_weak_ref_raises():
    with pytest.raises(RuntimeError, match="weak references"):
        _load_registry_simple()._unavailable_weak_ref(torch.zeros(1))


def test_accelerator_view_amd_via_deviceinfo(monkeypatch):
    registry = _load_registry(monkeypatch, [], probes=(False, False))
    _install_device_info(monkeypatch, vendor="amd")
    sentinel = object()
    mapping = MagicMock(return_value=sentinel)
    _stub_leaf(
        monkeypatch,
        "vllm_fl.dispatch.backends.vendor.amd.impl.uva",
        {"get_cuda_view_from_cpu_tensor_amd": mapping},
    )
    tensor = torch.zeros(2)

    assert registry._accelerator_view_from_cpu_tensor(tensor) is sentinel
    mapping.assert_called_once_with(tensor)


def test_accelerator_view_amd_via_platform_fallback(monkeypatch):
    registry = _load_registry(monkeypatch, [], probes=(False, False))
    _install_device_info(monkeypatch, raises=True)
    monkeypatch.setitem(
        sys.modules,
        "vllm.platforms",
        _new_module("vllm.platforms",
                    {"current_platform": SimpleNamespace(vendor_name="amd")}),
    )
    sentinel = object()
    mapping = MagicMock(return_value=sentinel)
    _stub_leaf(
        monkeypatch,
        "vllm_fl.dispatch.backends.vendor.amd.impl.uva",
        {"get_cuda_view_from_cpu_tensor_amd": mapping},
    )

    assert registry._accelerator_view_from_cpu_tensor(torch.zeros(1)) is sentinel
    mapping.assert_called_once()


def test_accelerator_view_unavailable_for_other_vendor(monkeypatch):
    registry = _load_registry(monkeypatch, [], probes=(False, False))
    _install_device_info(monkeypatch, vendor="cuda")
    with pytest.raises(RuntimeError, match="mapped CUDA view"):
        registry._accelerator_view_from_cpu_tensor(torch.zeros(1))


def test_accelerator_view_unavailable_when_vendor_unknown(monkeypatch):
    registry = _load_registry(monkeypatch, [], probes=(False, False))
    _install_device_info(monkeypatch, raises=True)
    monkeypatch.setitem(sys.modules, "vllm.platforms", None)
    with pytest.raises(RuntimeError, match="mapped CUDA view"):
        registry._accelerator_view_from_cpu_tensor(torch.zeros(1))


class _MetaPlatform:
    def __init__(self, verdict):
        self._verdict = verdict

    def is_metax(self):
        if isinstance(self._verdict, BaseException):
            raise self._verdict
        return self._verdict


@pytest.mark.parametrize(
    ("verdict", "expected"),
    [(True, True), (False, False), (RuntimeError("boom"), False)],
)
def test_is_metax_backend_resolved_platform(monkeypatch, verdict, expected):
    registry = _load_registry(monkeypatch, [])
    monkeypatch.delenv("GEMS_VENDOR", raising=False)
    monkeypatch.setitem(
        sys.modules,
        "vllm.platforms",
        _new_module("vllm.platforms",
                    {"_current_platform": _MetaPlatform(verdict)}),
    )
    assert registry._is_metax_backend() is expected


def test_is_metax_backend_env_override(monkeypatch):
    registry = _load_registry(monkeypatch, [])
    monkeypatch.setenv("GEMS_VENDOR", "MeTaX")
    assert registry._is_metax_backend() is True


def test_is_metax_backend_mcoplib_probe(monkeypatch):
    registry = _load_registry(monkeypatch, [], mcoplib_present=True)
    monkeypatch.delenv("GEMS_VENDOR", raising=False)
    assert registry._is_metax_backend() is True


def test_is_metax_backend_absent(monkeypatch):
    registry = _load_registry(monkeypatch, [])
    monkeypatch.delenv("GEMS_VENDOR", raising=False)
    assert registry._is_metax_backend() is False


@pytest.mark.parametrize(
    ("c_value", "expected"),
    [
        (None, False),
        (SimpleNamespace(weak_ref_tensor=object()), True),
        (SimpleNamespace(apply_repetition_penalties_=object()), True),
        (SimpleNamespace(unrelated=object()), False),
    ],
)
def test_C_ops_present(monkeypatch, c_value, expected):
    registry = _load_registry(monkeypatch, [])
    fake_ops = SimpleNamespace() if c_value is None else SimpleNamespace(_C=c_value)
    monkeypatch.setattr(torch, "ops", fake_ops)
    assert registry._C_ops_present() is expected


def test_apply_repetition_penalties_impl():
    registry = _load_registry_simple()
    logits = torch.tensor([[1.0, -2.0, 0.0], [3.0, 4.0, -5.0]])
    prompt_mask = torch.tensor([[True, False, False], [False, True, False]])
    output_mask = torch.tensor([[False, True, False], [False, False, False]])
    penalties = torch.tensor([2.0, 0.5])

    registry._apply_repetition_penalties_impl(
        logits, prompt_mask, output_mask, penalties
    )

    rp = penalties.unsqueeze(1).repeat(1, logits.size(1))
    eff = torch.where(prompt_mask | output_mask, rp, 1.0)
    base = torch.tensor([[1.0, -2.0, 0.0], [3.0, 4.0, -5.0]])
    expected = base * torch.where(base > 0, 1.0 / eff, eff)
    torch.testing.assert_close(logits, expected)


def test_register_schemas_short_circuits(monkeypatch):
    registry = _load_registry(monkeypatch, [], probes=(False, False))
    sentinel = object()
    monkeypatch.setattr(
        registry.register_op_schemas, "_libs", {"sentinel": sentinel},
        raising=False,
    )
    registry.register_op_schemas()
    assert registry.register_op_schemas._libs == {"sentinel": sentinel}


@pytest.mark.parametrize("kind", ["metax", "present", "native"])
def test_register_schemas_skip_paths(monkeypatch, kind):
    if kind == "native":
        registry = _load_registry(
            monkeypatch, [("covfl_ns_native", "bad(")], native=True,
            probes=(False, False),
        )
    else:
        registry = _load_registry(
            monkeypatch, [("covfl_ns_skip", "bad(")],
            probes=(kind == "metax", kind == "present"),
        )
    registry.register_op_schemas()
    assert not hasattr(registry.register_op_schemas, "_libs")


def test_register_schemas_registers_new_and_skips_existing(monkeypatch):
    ns = "covfl_skip_" + uuid.uuid4().hex
    new_ns = "covfl_new_" + uuid.uuid4().hex
    existing = torch.library.Library(ns, "FRAGMENT")
    existing.define("probe(Tensor x) -> Tensor")
    registry = _load_registry(
        monkeypatch,
        [(ns, "probe(Tensor x) -> Tensor"),
         (new_ns, "fresh(Tensor x) -> Tensor")],
        probes=(False, False),
    )
    try:
        registry.register_op_schemas()
        libs = registry.register_op_schemas._libs
        assert set(libs) == {ns, new_ns}
        found = torch._C._dispatch_find_schema_or_throw(
            f"{new_ns}::fresh", ""
        ).schema()
        assert found.name == f"{new_ns}::fresh"
    finally:
        _destroy_registry_libs(registry)
        existing._destroy()


def test_register_schemas_optional_native_only_skipped(monkeypatch):
    registry = _load_registry(
        monkeypatch,
        [("_C", "hadacore_transform(Tensor x) -> Tensor")],
        probes=(False, False),
    )
    registry.register_op_schemas()
    assert registry.register_op_schemas._libs == {}


def test_register_schemas_installs_c_lib_impls(monkeypatch):
    op_name = "covfl_probe_" + uuid.uuid4().hex
    registry = _load_registry(
        monkeypatch, [("_C", f"{op_name}(Tensor x) -> Tensor")],
        probes=(False, False),
    )
    try:
        registry.register_op_schemas()
        libs = registry.register_op_schemas._libs
        assert "_C" in libs
        assert hasattr(libs["_C"], "impl")
    finally:
        _destroy_registry_libs(registry)


def test_register_schemas_mcoplib_import_paths(monkeypatch, caplog):
    registry = _load_registry(
        monkeypatch, [], mcoplib_present=True, mcoplib_importable=False,
        probes=(False, False),
    )
    registry.register_op_schemas()
    assert "Failed to import mcoplib._C" in caplog.text

    caplog.clear()
    registry = _load_registry(
        monkeypatch, [], mcoplib_present=True, mcoplib_importable=True,
        probes=(False, False),
    )
    registry.register_op_schemas()
    assert "Failed to import mcoplib._C" not in caplog.text


def test_register_schemas_empty_and_invalid(monkeypatch, caplog):
    registry = _load_registry(monkeypatch, [], probes=(False, False))
    registry.register_op_schemas()
    assert "No op schemas found" in caplog.text
    assert not hasattr(registry.register_op_schemas, "_libs")

    bad_ns = "covfl_bad_" + uuid.uuid4().hex
    registry = _load_registry(
        monkeypatch, [(bad_ns, "not_a_valid_schema")], probes=(False, False)
    )
    with pytest.raises(RuntimeError, match="Failed to register the vLLM"):
        registry.register_op_schemas()


def test_register_schemas_idempotent_libs(monkeypatch):
    ns = "covfl_idem_" + uuid.uuid4().hex
    registry = _load_registry(
        monkeypatch, [(ns, "probe(Tensor x) -> Tensor")], probes=(False, False)
    )
    try:
        registry.register_op_schemas()
        libs = registry.register_op_schemas._libs
        registry.register_op_schemas()
        assert registry.register_op_schemas._libs is libs
    finally:
        _destroy_registry_libs(registry)


# =========================================================================== #
# custom_ops
# =========================================================================== #

def test_patch_unquantized_moe_oracle(monkeypatch):
    import vllm.model_executor.layers.fused_moe.oracle.unquantized as oracle_mod
    import vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method as method_mod

    import vllm_fl.ops.custom_ops as custom_ops
    import vllm_fl.ops.fused_moe.fused_moe_utils as moe_utils

    sentinel = object()
    monkeypatch.setattr(moe_utils, "select_unquantized_moe_backend_oot", sentinel)
    monkeypatch.setattr(
        oracle_mod, "select_unquantized_moe_backend", object(), raising=False
    )
    monkeypatch.setattr(
        method_mod, "select_unquantized_moe_backend", object(), raising=False
    )

    custom_ops._patch_unquantized_moe_oracle()

    assert oracle_mod.select_unquantized_moe_backend is sentinel
    assert method_mod.select_unquantized_moe_backend is sentinel


def _patch_oot_env(
    monkeypatch, *, enabled=True, whitelist=None, blacklist=None, flaggems=True
):
    import vllm_fl.utils as fl_utils

    monkeypatch.setattr(fl_utils, "is_oot_enabled", lambda: enabled)
    monkeypatch.setattr(fl_utils, "get_oot_whitelist", lambda: whitelist)
    monkeypatch.setattr(fl_utils, "get_oot_blacklist", lambda: blacklist)
    monkeypatch.setattr(
        fl_utils, "use_flaggems_op", lambda op_name, default=True: flaggems
    )


def _patch_register(monkeypatch):
    import vllm.model_executor.custom_op as custom_op_mod
    import vllm_fl.ops.custom_ops as custom_ops

    monkeypatch.setattr(custom_op_mod.CustomOp, "register_oot", MagicMock())
    monkeypatch.setattr(
        custom_ops, "_patch_unquantized_moe_oracle", MagicMock()
    )
    factory = MagicMock()
    monkeypatch.setattr(custom_ops, "_patch_fused_moe_factory", factory)
    import vllm_fl.dispatch.config.utils as cfg_utils
    monkeypatch.setattr(cfg_utils, "get_platform_name", lambda: "cuda")
    return custom_op_mod, custom_ops, factory


def test_register_oot_disabled(monkeypatch):
    import vllm_fl.ops.custom_ops as custom_ops

    _patch_oot_env(monkeypatch, enabled=False)
    oracle = MagicMock()
    monkeypatch.setattr(custom_ops, "_patch_unquantized_moe_oracle", oracle)

    custom_ops.register_oot_ops()

    oracle.assert_called_once_with()


def test_register_oot_whitelist_precedence(monkeypatch):
    custom_op_mod, custom_ops, factory = _patch_register(monkeypatch)
    _patch_oot_env(monkeypatch, whitelist=["rms_norm"])
    register = MagicMock()
    monkeypatch.setattr(custom_op_mod.CustomOp, "register_oot", register)

    custom_ops.register_oot_ops(whitelist=["silu_and_mul"])

    register.assert_called_once_with(
        _decorated_op_cls=custom_ops.RMSNormFL, name="RMSNorm"
    )
    factory.assert_called_once_with()


def test_register_oot_blacklist_filters(monkeypatch):
    custom_op_mod, custom_ops, factory = _patch_register(monkeypatch)
    _patch_oot_env(monkeypatch, blacklist=["silu_and_mul"])
    register = MagicMock()
    monkeypatch.setattr(custom_op_mod.CustomOp, "register_oot", register)

    custom_ops.register_oot_ops()

    registered = {c.kwargs["name"] for c in register.call_args_list}
    assert "SiluAndMul" not in registered
    assert {"GeluAndMul", "RMSNorm", "RotaryEmbedding"} <= registered
    factory.assert_called_once_with()


def test_register_oot_unknown_skipped(monkeypatch, caplog):
    custom_op_mod, custom_ops, _ = _patch_register(monkeypatch)
    _patch_oot_env(monkeypatch, whitelist=["does_not_exist"])
    register = MagicMock()
    monkeypatch.setattr(custom_op_mod.CustomOp, "register_oot", register)
    oracle = MagicMock()
    monkeypatch.setattr(custom_ops, "_patch_unquantized_moe_oracle", oracle)

    with caplog.at_level("WARNING"):
        custom_ops.register_oot_ops()

    register.assert_not_called()
    oracle.assert_called_once_with()
    assert "not found in OOT_OPS" in caplog.text


def test_register_oot_unquantized_flaggems_disabled(monkeypatch):
    custom_op_mod, custom_ops, _ = _patch_register(monkeypatch)
    monkeypatch.setitem(
        custom_ops.OOT_OPS,
        "unquantized_fused_moe_method",
        (object, "UnquantizedFusedMoEMethod"),
    )
    _patch_oot_env(
        monkeypatch,
        whitelist=["unquantized_fused_moe_method"],
        flaggems=False,
    )
    register = MagicMock()
    monkeypatch.setattr(custom_op_mod.CustomOp, "register_oot", register)

    custom_ops.register_oot_ops()

    register.assert_not_called()


def test_register_oot_pluggable_branch(monkeypatch):
    custom_op_mod, custom_ops, _ = _patch_register(monkeypatch)

    class _PluggableProbe:
        pass

    monkeypatch.setitem(
        custom_ops.OOT_OPS, "pluggable_probe", (_PluggableProbe, "PlugProbe")
    )
    _patch_oot_env(monkeypatch, whitelist=["pluggable_probe"])
    pluggable_register = MagicMock()
    custom_register = MagicMock()
    monkeypatch.setattr(
        custom_op_mod.PluggableLayer, "register_oot", pluggable_register
    )
    monkeypatch.setattr(custom_op_mod.CustomOp, "register_oot", custom_register)
    monkeypatch.setattr(
        custom_ops,
        "issubclass",
        lambda cls, base: cls is _PluggableProbe
        and base is custom_op_mod.PluggableLayer,
    )

    custom_ops.register_oot_ops()

    pluggable_register.assert_called_once_with(
        _decorated_layer_cls=_PluggableProbe, name="PlugProbe"
    )
    custom_register.assert_not_called()


@pytest.mark.parametrize(
    ("device_type", "vendor_name", "platform_name", "expected"),
    [
        ("npu", "npu", "cuda", "ascend"),
        ("cuda", "ptpu", "cuda", "sunrise"),
        ("cuda", "gcu", "cuda", "gcu"),
        ("cuda", "enflame", "cuda", "gcu"),
        ("cuda", "cuda", "kunlunxin", "kunlunxin"),
    ],
)
def test_register_oot_platform_patches(
    monkeypatch, device_type, vendor_name, platform_name, expected
):
    custom_op_mod, custom_ops, _ = _patch_register(monkeypatch)
    import vllm.platforms as vllm_platforms

    _patch_oot_env(monkeypatch, whitelist=["silu_and_mul"])
    monkeypatch.setattr(
        vllm_platforms,
        "current_platform",
        SimpleNamespace(device_type=device_type, vendor_name=vendor_name),
    )
    import vllm_fl.dispatch.config.utils as cfg_utils
    monkeypatch.setattr(cfg_utils, "get_platform_name", lambda: platform_name)

    patches = {
        "ascend": MagicMock(),
        "sunrise": MagicMock(),
        "gcu": MagicMock(),
        "kunlunxin": MagicMock(),
    }
    _stub_leaf(
        monkeypatch,
        "vllm_fl.dispatch.backends.vendor.ascend.patch",
        {"apply_ascend_patches": patches["ascend"]},
    )
    _stub_leaf(
        monkeypatch,
        "vllm_fl.dispatch.backends.vendor.sunrise.patch",
        {"apply_sunrise_patches": patches["sunrise"]},
    )
    _stub_leaf(
        monkeypatch,
        "vllm_fl.dispatch.backends.vendor.gcu.patch",
        {"apply_gcu_patches": patches["gcu"]},
    )
    _stub_leaf(
        monkeypatch,
        "vllm_fl.dispatch.backends.vendor.kunlunxin.patch",
        {"apply_kunlunxin_patches": patches["kunlunxin"]},
    )

    custom_ops.register_oot_ops()

    for name, mock in patches.items():
        if name == expected:
            mock.assert_called()
        else:
            mock.assert_not_called()


def test_register_oot_blacklist_fused_moe_skips_factory(monkeypatch):
    _, custom_ops, factory = _patch_register(monkeypatch)
    _patch_oot_env(monkeypatch, whitelist=["rms_norm"], blacklist=["fused_moe"])

    custom_ops.register_oot_ops()

    factory.assert_not_called()


def test_patch_fused_moe_factory_already_patched(monkeypatch):
    import vllm.model_executor.layers.fused_moe as package
    import vllm.model_executor.layers.fused_moe.layer as layer

    import vllm_fl.ops.custom_ops as custom_ops

    monkeypatch.setattr(
        layer, "FusedMoEFactory", custom_ops.FusedMoEFactoryFL, raising=False
    )
    original = object()
    monkeypatch.setattr(package, "FusedMoEFactory", original, raising=False)

    custom_ops._patch_fused_moe_factory()

    assert package.FusedMoEFactory is original


def test_patch_fused_moe_factory_full(monkeypatch):
    import vllm.model_executor.layers.fused_moe as package
    import vllm.model_executor.layers.fused_moe.layer as layer

    import vllm_fl.ops.custom_ops as custom_ops

    original_factory = object()
    original_legacy = object()
    replacement = object()
    monkeypatch.setattr(custom_ops, "FusedMoEFactoryFL", replacement)
    monkeypatch.setattr(custom_ops, "FusedMoEFL", object())
    monkeypatch.setattr(layer, "FusedMoEFactory", original_factory)
    monkeypatch.setattr(layer, "FusedMoE", original_legacy)
    monkeypatch.setattr(package, "FusedMoEFactory", original_factory)
    monkeypatch.setattr(package, "FusedMoE", original_legacy)

    cached = _new_module("covfl_cached_model")
    cached.FusedMoEFactory = original_factory
    cached.FusedMoE = original_legacy
    monkeypatch.setitem(sys.modules, cached.__name__, cached)
    monkeypatch.setitem(sys.modules, "covfl_none_module", None)
    untouched = _new_module("covfl_untouched_model")
    untouched.FusedMoEFactory = object()
    monkeypatch.setitem(sys.modules, untouched.__name__, untouched)

    custom_ops._patch_fused_moe_factory()

    assert layer.FusedMoEFactory is replacement
    assert layer.FusedMoE is replacement
    assert package.FusedMoEFactory is replacement
    assert package.FusedMoE is replacement
    assert cached.FusedMoEFactory is replacement
    assert cached.FusedMoE is replacement
    assert untouched.FusedMoEFactory is not replacement


# =========================================================================== #
# vllm_fl.ops package lazy __getattr__
# =========================================================================== #

def test_ops_package_lazy_getattr(monkeypatch):
    import vllm_fl.ops as ops

    assert ops.__all__ == ["activation", "rotary_embedding"]
    monkeypatch.delattr(ops, "activation", raising=False)
    monkeypatch.delattr(ops, "rotary_embedding", raising=False)

    activation = ops.activation
    rotary = ops.rotary_embedding

    assert activation.__name__ == "vllm_fl.ops.activation"
    assert rotary.__name__ == "vllm_fl.ops.rotary_embedding"
    assert ops.activation is activation
    with pytest.raises(AttributeError, match="has no attribute"):
        getattr(ops, "definitely_missing_attribute")


# =========================================================================== #
# activation
# =========================================================================== #

def _install_compiler(monkeypatch, value):
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: value)


def test_silu_amd_compiling(monkeypatch):
    import vllm_fl.ops.activation as activation

    monkeypatch.setattr(activation.SiluAndMul, "__init__",
                        lambda self, *a, **k: None)
    monkeypatch.setattr(
        activation, "current_platform", SimpleNamespace(vendor_name="amd")
    )
    _install_compiler(monkeypatch, True)
    custom = MagicMock(return_value="silu-out")
    _stub_leaf(
        monkeypatch,
        "vllm_fl.dispatch.backends.vendor.amd.graph_ops",
        {"silu_and_mul_custom_op": custom, "gelu_and_mul_custom_op": MagicMock()},
    )
    monkeypatch.setattr(activation, "_silu_and_mul", MagicMock())

    layer = activation.SiluAndMulFL()
    result = layer.forward_oot(torch.zeros(2, 4))

    assert result == "silu-out"
    custom.assert_called_once()
    activation._silu_and_mul.assert_not_called()


def test_silu_non_compiling_and_cuda(monkeypatch):
    import vllm_fl.ops.activation as activation

    monkeypatch.setattr(activation.SiluAndMul, "__init__",
                        lambda self, *a, **k: None)
    monkeypatch.setattr(
        activation, "current_platform", SimpleNamespace(vendor_name="cuda")
    )
    _install_compiler(monkeypatch, False)
    cached = MagicMock(return_value="cached")
    monkeypatch.setattr(activation, "_silu_and_mul", cached)

    layer = activation.SiluAndMulFL()
    x = torch.zeros(2, 8)

    assert layer.forward_oot(x) == "cached"
    cached.assert_called_once_with(layer, x)

    cached.reset_mock()
    assert layer.forward_cuda(x) == "cached"
    cached.assert_called_once_with(layer, x)


def test_gelu_amd_compiling(monkeypatch):
    import vllm_fl.ops.activation as activation

    monkeypatch.setattr(activation.GeluAndMul, "__init__",
                        lambda self, *a, **k: None)
    monkeypatch.setattr(
        activation, "current_platform", SimpleNamespace(vendor_name="amd")
    )
    _install_compiler(monkeypatch, True)
    custom = MagicMock(return_value="gelu-out")
    _stub_leaf(
        monkeypatch,
        "vllm_fl.dispatch.backends.vendor.amd.graph_ops",
        {"silu_and_mul_custom_op": MagicMock(), "gelu_and_mul_custom_op": custom},
    )

    layer = activation.GeluAndMulFL(approximate="tanh")
    layer.approximate = "tanh"
    result = layer.forward_oot(torch.zeros(2, 4))

    assert result == "gelu-out"
    custom.assert_called_once()
    assert custom.call_args.args[1] == "tanh"


def test_gelu_non_compiling_and_cuda(monkeypatch):
    import vllm_fl.ops.activation as activation

    monkeypatch.setattr(activation.GeluAndMul, "__init__",
                        lambda self, *a, **k: None)
    monkeypatch.setattr(
        activation, "current_platform", SimpleNamespace(vendor_name="cuda")
    )
    _install_compiler(monkeypatch, False)
    cached = MagicMock(return_value="gelu-cached")
    monkeypatch.setattr(activation, "_gelu_and_mul", cached)

    layer = activation.GeluAndMulFL()
    x = torch.zeros(2, 8)

    assert layer.forward_oot(x) == "gelu-cached"
    cached.assert_called_once_with(layer, x)
    cached.reset_mock()
    assert layer.forward_cuda(x) == "gelu-cached"
    cached.assert_called_once_with(layer, x)


# =========================================================================== #
# layernorm
# =========================================================================== #

def _make_rms(monkeypatch, layernorm, vendor="cuda"):
    monkeypatch.setattr(layernorm.RMSNorm, "__init__",
                        lambda self, *a, **k: None)
    monkeypatch.setattr(
        layernorm, "current_platform", SimpleNamespace(vendor_name=vendor)
    )
    layer = layernorm.RMSNormFL(hidden_size=4, eps=1e-6)
    layer.weight = torch.ones(4)
    layer.variance_epsilon = 1e-6
    return layer


def test_rms_amd_residual_none(monkeypatch):
    import vllm_fl.ops.layernorm as layernorm

    layer = _make_rms(monkeypatch, layernorm, vendor="amd")
    _install_compiler(monkeypatch, True)
    out = torch.ones(2, 4)
    custom = MagicMock(return_value=(out, torch.zeros(2, 4)))
    _stub_leaf(
        monkeypatch,
        "vllm_fl.dispatch.backends.vendor.amd.graph_ops",
        {"rms_norm_custom_op": custom},
    )

    result = layer.forward_oot(torch.zeros(2, 4))

    assert result is out
    custom.assert_called_once()
    args = custom.call_args.args
    assert args[2] is layer.weight
    assert args[3] == layer.variance_epsilon


def test_rms_amd_residual_tuple(monkeypatch):
    import vllm_fl.ops.layernorm as layernorm

    layer = _make_rms(monkeypatch, layernorm, vendor="amd")
    _install_compiler(monkeypatch, True)
    out = torch.ones(2, 4)
    new_residual = torch.zeros(2, 4)
    custom = MagicMock(return_value=(out, new_residual))
    _stub_leaf(
        monkeypatch,
        "vllm_fl.dispatch.backends.vendor.amd.graph_ops",
        {"rms_norm_custom_op": custom},
    )

    result = layer.forward_oot(torch.zeros(2, 4), residual=torch.ones(2, 4))

    assert isinstance(result, tuple)
    assert result[0] is out
    assert result[1] is new_residual


def test_rms_non_amd_and_cuda(monkeypatch):
    import vllm_fl.ops.layernorm as layernorm

    layer = _make_rms(monkeypatch, layernorm, vendor="cuda")
    _install_compiler(monkeypatch, False)
    cached = MagicMock(return_value="rms-out")
    monkeypatch.setattr(layernorm, "_rms_norm", cached)
    x = torch.zeros(2, 4)
    residual = torch.zeros(2, 4)

    assert layer.forward_oot(x, residual) == "rms-out"
    cached.assert_called_once_with(layer, x, residual)

    cached.reset_mock()
    assert layer.forward_cuda(x, None) == "rms-out"
    cached.assert_called_once_with(layer, x, None)


# =========================================================================== #
# rotary_embedding
# =========================================================================== #

def _make_rotary(
    monkeypatch, rotary, *, head_size=64, rotary_dim=32, is_neox_style=True,
    vendor="cuda", device_type="cuda",
):
    monkeypatch.setattr(
        rotary.RotaryEmbedding, "__init__", lambda self, *a, **k: None
    )
    monkeypatch.setattr(
        rotary,
        "current_platform",
        SimpleNamespace(device_type=device_type, vendor_name=vendor),
    )
    layer = rotary.RotaryEmbeddingFL(
        head_size=head_size,
        rotary_dim=rotary_dim,
        max_position_embeddings=16,
        base=10000.0,
        is_neox_style=is_neox_style,
        dtype=torch.float32,
        init_cache=False,
    )
    layer.head_size = head_size
    layer.rotary_dim = rotary_dim
    layer.is_neox_style = is_neox_style
    layer.cos_sin_cache = torch.zeros(16, 2 * rotary_dim)
    return layer


def test_rotary_key_none_uses_native(monkeypatch):
    import vllm_fl.ops.rotary_embedding as rotary

    layer = _make_rotary(monkeypatch, rotary)
    sentinel = (object(), None)
    native = MagicMock(return_value=sentinel)
    monkeypatch.setattr(rotary.RotaryEmbeddingFL, "forward_native", native)
    cached = MagicMock()
    monkeypatch.setattr(rotary, "_rotary_embedding", cached)
    positions = torch.arange(4)
    query = torch.zeros(4, 2 * 64)

    assert layer.forward_oot(positions, query, None) is sentinel
    native.assert_called_once_with(positions, query, None)
    cached.assert_not_called()


def test_rotary_partial_dim_cat(monkeypatch):
    import vllm_fl.ops.rotary_embedding as rotary

    layer = _make_rotary(monkeypatch, rotary, rotary_dim=32)
    cache = torch.arange(16 * 64, dtype=torch.float32).reshape(16, 64)
    monkeypatch.setattr(
        rotary.RotaryEmbeddingFL, "_match_cos_sin_cache_dtype",
        lambda self, q: cache,
    )
    positions = torch.arange(3)
    query = torch.zeros(3, 2 * 64)
    key = torch.zeros(3, 2 * 64)
    cached = MagicMock(
        return_value=(torch.ones(3, 2, 32), torch.ones(3, 2, 32))
    )
    monkeypatch.setattr(rotary, "_rotary_embedding", cached)

    out_q, out_k = layer.forward_oot(positions, query, key)

    assert out_q.shape == query.shape
    assert out_k.shape == key.shape
    assert cached.call_args.args[0] is layer
    assert cached.call_args.args[6] is True
    expected = torch.cat(
        (torch.ones(3, 2, 32), torch.zeros(3, 2, 32)), dim=-1
    ).reshape(3, 2 * 64)
    torch.testing.assert_close(out_q, expected)


def test_rotary_full_dim_reshape_and_cuda(monkeypatch):
    import vllm_fl.ops.rotary_embedding as rotary

    layer = _make_rotary(monkeypatch, rotary, rotary_dim=64, is_neox_style=False)
    monkeypatch.setattr(
        rotary.RotaryEmbeddingFL, "_match_cos_sin_cache_dtype",
        lambda self, q: torch.zeros(16, 128),
    )
    positions = torch.arange(2)
    query = torch.zeros(2, 2 * 64)
    key = torch.zeros(2, 2 * 64)
    cached = MagicMock(
        return_value=(torch.ones(2, 2, 64), torch.ones(2, 2, 64))
    )
    monkeypatch.setattr(rotary, "_rotary_embedding", cached)

    out_q, out_k = layer.forward_cuda(positions, query, key)

    assert out_q.shape == query.shape
    assert out_k.shape == key.shape
    assert cached.call_args.args[6] is True

    cached.reset_mock()
    layer.forward_oot(positions, query, key)
    cached.assert_called_once()


def test_rotary_amd_compiling(monkeypatch):
    import vllm_fl.ops.rotary_embedding as rotary

    layer = _make_rotary(monkeypatch, rotary, rotary_dim=64, vendor="amd")
    monkeypatch.setattr(
        rotary.RotaryEmbeddingFL, "_match_cos_sin_cache_dtype",
        lambda self, q: torch.zeros(16, 128),
    )
    _install_compiler(monkeypatch, True)
    custom = MagicMock(
        return_value=(torch.ones(2, 2, 64), torch.ones(2, 2, 64))
    )
    _stub_leaf(
        monkeypatch,
        "vllm_fl.dispatch.backends.vendor.amd.graph_ops",
        {"rotary_embedding_custom_op": custom},
    )
    cached = MagicMock()
    monkeypatch.setattr(rotary, "_rotary_embedding", cached)

    out_q, out_k = layer.forward_oot(
        torch.arange(2), torch.zeros(2, 2 * 64), torch.zeros(2, 2 * 64)
    )

    assert out_q.shape == (2, 2 * 64)
    assert out_k.shape == (2, 2 * 64)
    custom.assert_called_once()
    assert custom.call_args.kwargs["rotary_interleaved"] is False
    cached.assert_not_called()


def test_rotary_npu_init(monkeypatch):
    import vllm_fl.ops.rotary_embedding as rotary

    init = MagicMock()
    _stub_leaf(
        monkeypatch,
        "vllm_fl.dispatch.backends.vendor.ascend.impl.rotary",
        {"initialize_rotary_embedding_ascend": init},
    )
    _make_rotary(
        monkeypatch, rotary, vendor="npu", device_type="npu"
    )

    init.assert_called_once_with()


# =========================================================================== #
# vllm_fl.ops.fla.utils
# =========================================================================== #

def _fla_leaf(package, leaf):
    return sys.modules[package.__name__ + "." + leaf]


def test_fla_package_exports():
    chunk = _fla_leaf(_FLA_UPSTREAM, "chunk")
    recurrent = _fla_leaf(_FLA_UPSTREAM, "fused_recurrent")
    assert _FLA_UPSTREAM.__all__ == [
        "ChunkGatedDeltaRuleOp",
        "FusedRecurrentGatedDeltaRuleOp",
    ]
    assert _FLA_UPSTREAM.ChunkGatedDeltaRuleOp is chunk.ChunkGatedDeltaRuleOp
    assert (
        _FLA_UPSTREAM.FusedRecurrentGatedDeltaRuleOp
        is recurrent.FusedRecurrentGatedDeltaRuleOp
    )


def test_fla_import_branch_selection():
    chunk_up = _fla_leaf(_FLA_UPSTREAM, "chunk")
    rec_up = _fla_leaf(_FLA_UPSTREAM, "fused_recurrent")
    chunk_gems = _fla_leaf(_FLA_GEMS, "chunk")
    rec_gems = _fla_leaf(_FLA_GEMS, "fused_recurrent")
    assert chunk_up._chunk_impl.__name__.endswith("ops.chunk")
    assert chunk_gems._chunk_impl.__name__ == "flag_gems.fused.FLA"
    assert rec_up.fused_recurrent_gated_delta_rule_fwd is not None
    assert rec_gems.fused_recurrent_gated_delta_rule_fwd is not None
    assert rec_gems.fused_recurrent_gated_delta_rule_fwd is not (
        rec_up.fused_recurrent_gated_delta_rule_fwd
    )


class _Ctx:
    def __init__(self, index):
        self.index = index
        self.entered = False

    def __enter__(self):
        self.entered = True
        return self

    def __exit__(self, *exc):
        return False


def test_input_guard_cuda_context_and_contiguous(monkeypatch):
    utils = _fla_leaf(_FLA_UPSTREAM, "utils")
    created = {}

    def fake_device(index):
        ctx = _Ctx(index)
        created["ctx"] = ctx
        return ctx

    monkeypatch.setattr(
        torch, "npu", SimpleNamespace(is_available=lambda: False), raising=False
    )
    monkeypatch.setattr(torch.cuda, "device", fake_device)

    @utils.input_guard
    def fn(x):
        return x

    tensor = torch.arange(12).reshape(3, 4).t()
    assert not tensor.is_contiguous()

    result = fn(tensor)

    assert result.is_contiguous()
    assert created["ctx"].entered is True


def test_input_guard_npu_device(monkeypatch):
    utils = _fla_leaf(_FLA_UPSTREAM, "utils")
    npu_device = MagicMock(return_value=_Ctx(3))
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(is_available=lambda: True, device=npu_device),
        raising=False,
    )

    @utils.input_guard
    def fn(x):
        return x

    tensor = torch.zeros(2)
    fn(tensor)

    npu_device.assert_called_once_with(tensor.device.index)


def test_input_guard_kwargs_tensor(monkeypatch):
    utils = _fla_leaf(_FLA_UPSTREAM, "utils")
    monkeypatch.setattr(
        torch, "npu", SimpleNamespace(is_available=lambda: False), raising=False
    )
    monkeypatch.setattr(torch.cuda, "device", lambda index: _Ctx(index))

    @utils.input_guard
    def fn(*args, **kwargs):
        return args, kwargs

    value = torch.arange(6).reshape(2, 3).t()
    _, kwargs = fn(1, x=value)

    assert kwargs["x"].is_contiguous()


def test_input_guard_no_tensor_and_wraps(monkeypatch):
    utils = _fla_leaf(_FLA_UPSTREAM, "utils")
    device = MagicMock()
    monkeypatch.setattr(
        torch, "npu", SimpleNamespace(is_available=lambda: False), raising=False
    )
    monkeypatch.setattr(torch.cuda, "device", device)

    @utils.input_guard
    def sample(*args):
        return args

    assert sample(1, 2) == (1, 2)
    assert sample.__name__ == "sample"
    device.assert_not_called()


# =========================================================================== #
# vllm_fl.ops.fla.chunk
# =========================================================================== #

def _chunk_env():
    chunk = _fla_leaf(_FLA_UPSTREAM, "chunk")
    return chunk, chunk.ChunkGatedDeltaRuleFunction, chunk.ChunkGatedDeltaRuleOp


def _unwrapped_forward(func):
    return getattr(func, "__wrapped__", func)


def test_chunk_init_reads_suppress_level(monkeypatch):
    _, _, op_cls = _chunk_env()
    monkeypatch.setenv("GDN_RECOMPUTE_SUPPRESS_LEVEL", "2")
    op = op_cls(output_final_state=True, use_qk_l2norm_in_kernel=True)
    assert op.output_final_state is True
    assert op.use_qk_l2norm_in_kernel is True
    assert op.suppress_level == 2


def test_chunk_forward_l2norm_and_dtype(monkeypatch):
    chunk, fwd_cls, _ = _chunk_env()
    q = torch.zeros(2, 3, 2, 8, dtype=torch.bfloat16)
    k = torch.zeros_like(q)
    v = torch.zeros_like(q)
    g = torch.zeros(2, 3, 2, dtype=torch.bfloat16)
    beta = torch.zeros(2, 3, 2, dtype=torch.bfloat16)
    o = torch.zeros(2, 3, 2, 8, dtype=torch.float32)
    final_state = object()
    kernel = MagicMock(return_value=(g, o, None, final_state, None, None, None))
    monkeypatch.setattr(chunk._chunk_impl, "chunk_gated_delta_rule_fwd", kernel)
    l2 = MagicMock(side_effect=lambda t: t)
    monkeypatch.setattr(chunk, "l2norm_fwd", l2)
    ctx = SimpleNamespace()

    out, out_state = _unwrapped_forward(fwd_cls.forward)(
        ctx, q, k, v, g, beta, 0.5, None, True, None, True
    )

    assert out.dtype == torch.bfloat16
    assert out_state is final_state
    assert ctx.scale == 0.5
    assert ctx.use_qk_l2norm_in_kernel is True
    assert l2.call_count == 2
    kernel.assert_called_once()


def test_chunk_forward_without_l2norm(monkeypatch):
    chunk, fwd_cls, _ = _chunk_env()
    q = torch.zeros(1, 2, 2, 8, dtype=torch.bfloat16)
    g = torch.zeros(1, 2, 2, dtype=torch.bfloat16)
    beta = torch.zeros(1, 2, 2, dtype=torch.bfloat16)
    o = torch.zeros(1, 2, 2, 8, dtype=torch.bfloat16)
    kernel = MagicMock(return_value=(g, o, None, None, None, None, None))
    monkeypatch.setattr(chunk._chunk_impl, "chunk_gated_delta_rule_fwd", kernel)
    l2 = MagicMock()
    monkeypatch.setattr(chunk, "l2norm_fwd", l2)

    out, _ = _unwrapped_forward(fwd_cls.forward)(
        SimpleNamespace(), q, q, q, g, beta, 1.0, None, False, None, False
    )

    assert out.dtype == torch.bfloat16
    l2.assert_not_called()


def _patch_chunk_apply(monkeypatch, fwd_cls):
    recorder = MagicMock(return_value=(object(), object()))
    monkeypatch.setattr(fwd_cls, "apply", staticmethod(recorder))
    return recorder


def test_chunk_forward_native_asserts(monkeypatch):
    _, fwd_cls, op_cls = _chunk_env()
    _patch_chunk_apply(monkeypatch, fwd_cls)
    op = op_cls()

    good = torch.zeros(1, 2, 2, 8, dtype=torch.bfloat16)
    beta = torch.zeros(1, 2, 2, dtype=torch.bfloat16)
    with pytest.raises(AssertionError):
        op.forward_native(good, torch.zeros_like(good, dtype=torch.float16),
                          good, beta[..., 0], beta)
    with pytest.raises(AssertionError, match="float32"):
        op.forward_native(good.float(), good.float(), good.float(),
                          beta[..., 0], beta)
    with pytest.raises(AssertionError, match="beta must be"):
        op.forward_native(good, good, good, beta[..., 0],
                          torch.zeros(1, 2, dtype=torch.bfloat16))


def test_chunk_forward_native_shape_warning(monkeypatch):
    _, fwd_cls, op_cls = _chunk_env()
    _patch_chunk_apply(monkeypatch, fwd_cls)
    op = op_cls()
    q = torch.zeros(2, 3, 4, 8, dtype=torch.bfloat16)
    beta = torch.zeros(2, 3, 4, dtype=torch.bfloat16)

    with pytest.warns(UserWarning, match="potential format mismatch"):
        op.forward_native(q, q, q, beta[..., 0], beta)


@pytest.mark.parametrize("case", ["batch", "initial_state"])
def test_chunk_forward_native_cu_seqlens_errors(monkeypatch, case):
    _, fwd_cls, op_cls = _chunk_env()
    _patch_chunk_apply(monkeypatch, fwd_cls)
    op = op_cls()
    if case == "batch":
        q = torch.zeros(2, 4, 2, 8, dtype=torch.bfloat16)
        cu_seqlens = torch.tensor([0, 2, 4])
        initial_state = None
    else:
        q = torch.zeros(1, 4, 2, 8, dtype=torch.bfloat16)
        cu_seqlens = torch.tensor([0, 2, 4])
        initial_state = torch.zeros(3, 2, 8, 8, dtype=torch.bfloat16)
    beta = torch.zeros(q.shape[0], q.shape[1], q.shape[2],
                       dtype=torch.bfloat16)

    with pytest.raises(ValueError):
        op.forward_native(
            q, q, q, beta[..., 0], beta,
            initial_state=initial_state, cu_seqlens=cu_seqlens,
        )


def test_chunk_forward_native_scale_and_args(monkeypatch):
    _, fwd_cls, op_cls = _chunk_env()
    recorder = _patch_chunk_apply(monkeypatch, fwd_cls)
    op = op_cls(output_final_state=True, use_qk_l2norm_in_kernel=True)
    q = torch.zeros(1, 4, 2, 8, dtype=torch.bfloat16)
    beta = torch.zeros(1, 4, 2, dtype=torch.bfloat16)

    result = op.forward_native(q, q, q, beta[..., 0], beta)

    assert result == recorder.return_value
    args = recorder.call_args.args
    assert args[5] == pytest.approx(8 ** -0.5)
    assert args[7] is True
    assert args[9] is True


# =========================================================================== #
# vllm_fl.ops.fla.fused_recurrent
# =========================================================================== #

def _recurrent_env():
    module = _fla_leaf(_FLA_UPSTREAM, "fused_recurrent")
    return module, module.FusedRecurrentFunction, module.FusedRecurrentGatedDeltaRuleOp


def _patch_recurrent_apply(monkeypatch, fwd_cls):
    recorder = MagicMock(return_value=(object(), object()))
    monkeypatch.setattr(fwd_cls, "apply", staticmethod(recorder))
    return recorder


def test_recurrent_init():
    _, _, op_cls = _recurrent_env()
    op = op_cls(inplace_final_state=True, use_qk_l2norm_in_kernel=True)
    assert op.inplace_final_state is True
    assert op.use_qk_l2norm_in_kernel is True


def test_recurrent_forward_plumbing(monkeypatch):
    module, fwd_cls, _ = _recurrent_env()
    o = torch.zeros(2, 3, 2, 8)
    final_state = object()
    kernel = MagicMock(return_value=(o, final_state))
    monkeypatch.setattr(module, "fused_recurrent_gated_delta_rule_fwd", kernel)
    g = torch.arange(12, dtype=torch.float32).reshape(3, 4).t()
    beta = torch.arange(12, dtype=torch.float32).reshape(3, 4).t()
    q = torch.zeros(2, 3, 2, 8)

    out, state = fwd_cls.forward(
        SimpleNamespace(), q, q, q, g, beta, 0.5, None,
        True, None, None, None, False,
    )

    assert out is o and state is final_state
    kwargs = kernel.call_args.kwargs
    assert kwargs["g"].is_contiguous()
    assert kwargs["beta"].is_contiguous()
    assert kwargs["inplace_final_state"] is True
    assert kwargs["use_qk_l2norm_in_kernel"] is False


def test_recurrent_cu_seqlens_error(monkeypatch):
    _, fwd_cls, op_cls = _recurrent_env()
    _patch_recurrent_apply(monkeypatch, fwd_cls)
    op = op_cls()
    q = torch.zeros(2, 4, 2, 8)

    with pytest.raises(ValueError):
        op.forward_native(
            q, q, q, torch.zeros(2, 4, 2), torch.zeros(2, 4, 2),
            cu_seqlens=torch.tensor([0, 2, 4]),
        )


def test_recurrent_scale_handling(monkeypatch):
    _, fwd_cls, op_cls = _recurrent_env()
    recorder = _patch_recurrent_apply(monkeypatch, fwd_cls)
    op = op_cls()
    q = torch.zeros(1, 4, 2, 8)
    beta = torch.zeros(1, 4, 2)

    op.forward_native(q, q, q, beta, beta)
    assert recorder.call_args.args[5] == pytest.approx(8 ** -0.5)

    recorder.reset_mock()
    op.forward_native(q, q, q, beta, beta, scale=2.0)
    assert recorder.call_args.args[5] == 2.0

    recorder.reset_mock()
    with pytest.raises(AssertionError, match="scale must be positive"):
        op.forward_native(q, q, q, beta, beta, scale=0.0)


def test_recurrent_beta_none(monkeypatch):
    _, fwd_cls, op_cls = _recurrent_env()
    recorder = _patch_recurrent_apply(monkeypatch, fwd_cls)
    op = op_cls()
    q = torch.zeros(2, 4, 3, 8)

    op.forward_native(q, q, q, torch.zeros(2, 4, 3), None, scale=1.0)

    beta = recorder.call_args.args[4]
    assert beta.shape == (2, 4, 3)
    torch.testing.assert_close(beta, torch.ones(2, 4, 3))


def test_recurrent_forward_native_args(monkeypatch):
    _, fwd_cls, op_cls = _recurrent_env()
    recorder = _patch_recurrent_apply(monkeypatch, fwd_cls)
    op = op_cls(inplace_final_state=True, use_qk_l2norm_in_kernel=True)
    q = torch.zeros(1, 4, 2, 8)
    beta = torch.zeros(1, 4, 2)
    initial_state = torch.zeros(1, 2, 8, 8)
    ssm = torch.zeros(1, dtype=torch.long)

    result = op.forward_native(
        q, q, q, beta, beta, scale=1.0, initial_state=initial_state,
        ssm_state_indices=ssm,
    )

    assert result == recorder.return_value
    args = recorder.call_args.args
    assert args[6] is initial_state
    assert args[7] is True
    assert args[9] is ssm
    assert args[11] is True

