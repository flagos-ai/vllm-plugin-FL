# Coverage tests: vllm_fl/ops/_C_ops_registry.py and _C_ops_schemas.py.

import importlib.util
import sys
import uuid
from types import ModuleType, SimpleNamespace

import pytest
import torch

from vllm_fl.ops import _C_ops_registry as registry


def test_unavailable_helpers_raise():
    with pytest.raises(RuntimeError, match="mapped CUDA view"):
        registry._unavailable_cuda_view(None)
    with pytest.raises(RuntimeError, match="weak references"):
        registry._unavailable_weak_ref(None)


def test_log_amd_host_view_once_cached():
    registry._log_amd_host_view_once()
    assert registry._log_amd_host_view_once() is None


def test_apply_repetition_penalties_matches_formula():
    logits = torch.tensor([[2.0, -3.0, 0.5], [1.0, 2.0, -1.0]])
    prompt_mask = torch.tensor([[True, False, True], [False, False, False]])
    output_mask = torch.tensor([[False, True, False], [True, False, False]])
    penalties = torch.tensor([2.0, 0.5])

    expected = logits.clone()
    rp = penalties.unsqueeze(1).repeat(1, logits.size(1))
    mask = prompt_mask | output_mask
    scaling = torch.where(
        expected > 0,
        1.0 / torch.where(mask, rp, torch.ones_like(rp)),
        torch.where(mask, rp, torch.ones_like(rp)),
    )
    expected.mul_(scaling)

    registry._apply_repetition_penalties_impl(
        logits, prompt_mask, output_mask, penalties
    )
    assert torch.allclose(logits, expected)


def _stub_amd_uva(monkeypatch, sentinel):
    vendor = ModuleType("vllm_fl.dispatch.backends.vendor.amd")
    vendor.__path__ = []
    impl = ModuleType("vllm_fl.dispatch.backends.vendor.amd.impl")
    impl.__path__ = []
    uva = ModuleType("vllm_fl.dispatch.backends.vendor.amd.impl.uva")
    uva.get_cuda_view_from_cpu_tensor_amd = lambda tensor: sentinel
    monkeypatch.setitem(sys.modules, vendor.__name__, vendor)
    monkeypatch.setitem(sys.modules, impl.__name__, impl)
    monkeypatch.setitem(sys.modules, uva.__name__, uva)


def test_accelerator_view_dispatches_to_amd(monkeypatch):
    import vllm_fl.utils as fl_utils

    counted = []
    monkeypatch.setattr(
        fl_utils, "DeviceInfo", lambda: SimpleNamespace(vendor_name="amd")
    )
    monkeypatch.setattr(
        registry, "_log_amd_host_view_once", lambda: counted.append(True)
    )
    sentinel = torch.zeros(2, 3)
    _stub_amd_uva(monkeypatch, sentinel)

    result = registry._accelerator_view_from_cpu_tensor(torch.zeros(1))

    assert result is sentinel
    assert counted == [True]


def test_accelerator_view_rejects_non_amd_vendor(monkeypatch):
    import vllm_fl.utils as fl_utils

    monkeypatch.setattr(
        fl_utils, "DeviceInfo", lambda: SimpleNamespace(vendor_name="cuda")
    )

    with pytest.raises(RuntimeError):
        registry._accelerator_view_from_cpu_tensor(torch.zeros(1))


def test_accelerator_view_falls_back_to_current_platform(monkeypatch):
    import vllm.platforms as platforms
    import vllm_fl.utils as fl_utils

    def broken():
        raise RuntimeError("no vendor probe")

    monkeypatch.setattr(fl_utils, "DeviceInfo", broken)
    monkeypatch.setattr(platforms, "current_platform", SimpleNamespace(vendor_name=None))

    with pytest.raises(RuntimeError):
        registry._accelerator_view_from_cpu_tensor(torch.zeros(1))


def test_is_metax_backend_from_resolved_platform(monkeypatch):
    class Platform:
        def is_metax(self):
            return True

    monkeypatch.setitem(
        sys.modules,
        "vllm.platforms",
        SimpleNamespace(_current_platform=Platform()),
    )
    assert registry._is_metax_backend() is True


def test_is_metax_backend_swallows_platform_error_and_uses_env(monkeypatch):
    class Platform:
        def is_metax(self):
            raise RuntimeError("bad platform")

    monkeypatch.setitem(
        sys.modules,
        "vllm.platforms",
        SimpleNamespace(_current_platform=Platform()),
    )
    monkeypatch.setenv("GEMS_VENDOR", " MetaX ")
    assert registry._is_metax_backend() is True


def test_is_metax_backend_detects_mcoplib(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "vllm.platforms", SimpleNamespace(_current_platform=None)
    )
    monkeypatch.delenv("GEMS_VENDOR", raising=False)
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name: object() if name == "mcoplib" else None,
    )
    assert registry._is_metax_backend() is True


def test_is_metax_backend_handles_probe_failure(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "vllm.platforms", SimpleNamespace(_current_platform=None)
    )
    monkeypatch.delenv("GEMS_VENDOR", raising=False)

    def boom(name):
        raise RuntimeError("probe failed")

    monkeypatch.setattr(importlib.util, "find_spec", boom)
    assert registry._is_metax_backend() is False


def test_is_metax_backend_non_metax_run(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "vllm.platforms", SimpleNamespace(_current_platform=None)
    )
    monkeypatch.delenv("GEMS_VENDOR", raising=False)
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    assert registry._is_metax_backend() is False


def test_c_ops_present_probe(monkeypatch):
    fake = SimpleNamespace(ops=SimpleNamespace(_C=None))
    monkeypatch.setattr(registry, "torch", fake)
    assert registry._C_ops_present() is False

    fake.ops._C = SimpleNamespace(weak_ref_tensor=object())
    assert registry._C_ops_present() is True

    fake.ops._C = SimpleNamespace(apply_repetition_penalties_=object())
    assert registry._C_ops_present() is True

    fake.ops._C = SimpleNamespace(something_else=object())
    assert registry._C_ops_present() is False


def _patch_registry_guards(monkeypatch, schemas):
    monkeypatch.setattr(registry, "_is_metax_backend", lambda: False)
    monkeypatch.setattr(registry, "_C_ops_present", lambda: False)
    monkeypatch.setitem(sys.modules, "vllm._C", None)
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    schema_module = ModuleType("vllm_fl.ops._C_ops_schemas")
    schema_module.SCHEMAS = schemas
    monkeypatch.setitem(
        sys.modules, "vllm_fl.ops._C_ops_schemas", schema_module
    )
    monkeypatch.delattr(registry.register_op_schemas, "_libs", raising=False)


def _destroy_libs():
    for lib in getattr(registry.register_op_schemas, "_libs", {}).values():
        lib._destroy()
    if hasattr(registry.register_op_schemas, "_libs"):
        del registry.register_op_schemas._libs


def test_register_op_schemas_returns_when_libs_cached(monkeypatch):
    sentinel = {"cached": object()}
    monkeypatch.setattr(
        registry.register_op_schemas, "_libs", sentinel, raising=False
    )
    registry.register_op_schemas()
    assert registry.register_op_schemas._libs is sentinel


def test_register_op_schemas_skips_native_extension(monkeypatch):
    monkeypatch.delattr(registry.register_op_schemas, "_libs", raising=False)
    monkeypatch.setattr(registry, "_is_metax_backend", lambda: True)
    monkeypatch.setattr(registry, "_C_ops_present", lambda: False)
    registry.register_op_schemas()
    assert not hasattr(registry.register_op_schemas, "_libs")


def test_register_op_schemas_accepts_importable_native_extension(monkeypatch):
    monkeypatch.delattr(registry.register_op_schemas, "_libs", raising=False)
    monkeypatch.setattr(registry, "_is_metax_backend", lambda: False)
    monkeypatch.setattr(registry, "_C_ops_present", lambda: False)
    monkeypatch.setitem(sys.modules, "vllm._C", ModuleType("vllm._C"))

    registry.register_op_schemas()
    assert not hasattr(registry.register_op_schemas, "_libs")


def test_register_op_schemas_registers_and_is_idempotent(monkeypatch):
    uid = uuid.uuid4().hex
    regular = f"fl_cov_{uid}"
    cache = f"fl_cov_{uid}_cache"
    schemas = [
        (regular, "copy(Tensor input) -> Tensor"),
        (cache, "copy(Tensor input, Tensor! cache, Tensor! out) -> ()"),
        ("_C", f"fl_cov_op_{uid}(Tensor input) -> Tensor"),
        ("_C", "get_cuda_view_from_cpu_tensor(Tensor cpu_tensor) -> Tensor"),
        ("_C", "cutlass_scaled_mm_supports_fp8(int cuda_device_capability) -> bool"),
        (
            "_C",
            "cutlass_scaled_mm_supports_block_fp8(int cuda_device_capability) -> bool",
        ),
        ("_C", "cutlass_group_gemm_supported(int cuda_device_capability) -> bool"),
        ("_C", "cutlass_scaled_mm_supports_fp4(int cuda_device_capability) -> bool"),
        ("_C", "weak_ref_tensor(Tensor input) -> Tensor"),
        (
            "_C",
            "apply_repetition_penalties_(Tensor! logits, Tensor prompt_mask, "
            "Tensor output_mask, Tensor repetition_penalties) -> ()",
        ),
    ]
    _patch_registry_guards(monkeypatch, schemas)
    try:
        registry.register_op_schemas()
        libraries = registry.register_op_schemas._libs
        assert set(libraries) == {regular, cache, "_C"}
        registered = torch._C._dispatch_find_schema_or_throw(
            f"{regular}::copy", ""
        ).schema()
        assert str(registered).startswith(f"{regular}::copy(")
        first = libraries
        registry.register_op_schemas()
        assert registry.register_op_schemas._libs is first
    finally:
        _destroy_libs()


def test_register_op_schemas_empty_schema_list(monkeypatch):
    _patch_registry_guards(monkeypatch, [])
    try:
        registry.register_op_schemas()
        assert not hasattr(registry.register_op_schemas, "_libs")
    finally:
        _destroy_libs()


def test_register_skips_existing_overload(monkeypatch):
    uid = uuid.uuid4().hex
    namespace = f"fl_cov_existing_{uid}"
    schema = "copy(Tensor input) -> Tensor"
    existing = torch.library.Library(namespace, "FRAGMENT")
    existing.define(schema)
    _patch_registry_guards(monkeypatch, [(namespace, schema)])
    try:
        registry.register_op_schemas()
        assert namespace in registry.register_op_schemas._libs
    finally:
        _destroy_libs()
        existing._destroy()


def test_optional_native_only_exclusion_is_scoped_to_c_namespace(monkeypatch):
    uid = uuid.uuid4().hex
    name = f"optional_{uid}"
    cache = f"fl_cov_optional_{uid}"
    schema = f"{name}(Tensor input) -> Tensor"
    _patch_registry_guards(monkeypatch, [("_C", schema), (cache, schema)])
    monkeypatch.setattr(
        registry,
        "_OPTIONAL_NATIVE_ONLY_OPS",
        frozenset(registry._OPTIONAL_NATIVE_ONLY_OPS | {name}),
    )
    try:
        registry.register_op_schemas()
        with pytest.raises(RuntimeError):
            torch._C._dispatch_find_schema_or_throw(f"_C::{name}", "")
        registered = torch._C._dispatch_find_schema_or_throw(f"{cache}::{name}", "")
        assert str(registered.schema()).startswith(f"{cache}::{name}(")
    finally:
        _destroy_libs()


def test_schema_catalog_is_well_formed():
    from vllm_fl.ops._C_ops_schemas import SCHEMAS

    assert SCHEMAS
    for namespace, schema in SCHEMAS:
        assert isinstance(namespace, str)
        assert "->" in schema
