"""K3 ABI additions reuse the original SDK wrapper methods and signatures."""

from unittest.mock import Mock

import pytest

from vllm_fl.dispatch.backends.vendor.metax.patches import abi
from vllm_fl.dispatch.backends.vendor.metax.patches.cuda_wrapper import (
    CudaRTLibrary as NativeCudaRTLibrary,
)
from vllm_fl.dispatch.backends.vendor.metax.patches.pynccl_wrapper import (
    MCCLLibrary as NativeMCCLLibrary,
)


def test_cuda_calls_and_exports_are_inherited():
    assert (
        abi.CudaRTLibrary.exported_functions is NativeCudaRTLibrary.exported_functions
    )
    for name, fn in NativeCudaRTLibrary.__dict__.items():
        if callable(fn) and name != "__init__":
            assert getattr(abi.CudaRTLibrary, name) is fn


def test_mccl_collectives_are_inherited_without_reordering():
    overrides = {
        "__init__",
        "ncclCommAbort",
        "ncclCommWindowRegister",
        "ncclCommWindowDeregister",
    }
    for name, fn in NativeMCCLLibrary.__dict__.items():
        if callable(fn) and name not in overrides:
            assert getattr(abi.MCCLLibrary, name) is fn
    baseline = [f.name for f in NativeMCCLLibrary.exported_functions]
    selected = [f.name for f in abi.MCCLLibrary.exported_functions]
    expected = list(baseline)
    expected.insert(expected.index("mcclGroupStart"), "mcclCommAbort")
    assert selected == expected
    assert (
        abi.MCCLLibrary.path_to_library_cache
        is not NativeMCCLLibrary.path_to_library_cache
    )


def test_cuda_keeps_verified_explicit_library_resolution(monkeypatch):
    init = Mock()
    monkeypatch.setattr(NativeCudaRTLibrary, "__init__", init)
    monkeypatch.setenv("VLLM_CUDART_SO_PATH", "/verified/libmcruntime.so")
    abi.CudaRTLibrary()
    init.assert_called_once_with("/verified/libmcruntime.so")


def test_missing_cuda_runtime_fails_explicitly(monkeypatch):
    monkeypatch.delenv("VLLM_CUDART_SO_PATH", raising=False)
    monkeypatch.setattr(abi, "find_loaded_library", lambda name: None)
    with pytest.raises(RuntimeError, match="libmcruntime"):
        abi.CudaRTLibrary()


def test_abort_and_unsupported_windows_keep_the_original_contract():
    wrapper = abi.MCCLLibrary.__new__(abi.MCCLLibrary)
    abort = Mock(return_value=0)
    wrapper._funcs = {"mcclCommAbort": abort}
    wrapper.NCCL_CHECK = Mock()
    wrapper.ncclCommAbort("comm")
    abort.assert_called_once_with("comm")
    wrapper.NCCL_CHECK.assert_called_once_with(0)
    with pytest.raises(NotImplementedError, match="symmetric-memory"):
        wrapper.ncclCommWindowRegister("comm", None, 0, 0)
    with pytest.raises(NotImplementedError, match="symmetric-memory"):
        wrapper.ncclCommWindowDeregister("comm", None)


def test_open_runtime_entry_is_idempotent_and_uses_only_sdk_hooks(monkeypatch):
    from vllm.distributed.device_communicators import (
        cuda_wrapper,
        pynccl,
        pynccl_wrapper,
    )
    from vllm.utils import nccl

    from vllm_fl.dispatch.backends.vendor.metax import runtime

    targets = (
        (cuda_wrapper, "CudaRTLibrary"),
        (pynccl, "NCCLLibrary"),
        (pynccl_wrapper, "NCCLLibrary"),
        (nccl, "find_nccl_library"),
    )
    for module, name in targets:
        monkeypatch.setattr(module, name, getattr(module, name))
    monkeypatch.setattr(runtime, "_REGISTERED", False)
    monkeypatch.setattr(runtime, "uses_open_metax_runtime", lambda: True)
    monkeypatch.setattr(runtime, "version", lambda _: "0.28.0+empty")
    monkeypatch.delenv("FLAGCX_PATH", raising=False)
    monkeypatch.setattr(abi.envs, "VLLM_USE_NCCL_SYMM_MEM", False)
    runtime.register_runtime()
    assert cuda_wrapper.CudaRTLibrary is abi.CudaRTLibrary
    assert pynccl.NCCLLibrary is pynccl_wrapper.NCCLLibrary is abi.MCCLLibrary
    assert nccl.find_nccl_library is abi.find_mccl_library
    current = [getattr(m, n) for m, n in targets]
    runtime.register_runtime()
    assert current == [getattr(m, n) for m, n in targets]


def test_legacy_runtime_keeps_the_original_vendor_import_order(monkeypatch):
    import builtins
    from types import SimpleNamespace

    from vllm_fl.dispatch.backends.vendor.metax import runtime

    expected = (
        "mcoplib._C",
        "mcoplib._moe_C",
        "vllm_fl.dispatch.backends.vendor.metax.patches",
    )
    imported = []
    original = builtins.__import__

    def observe(name, *args, **kwargs):
        if name in expected:
            imported.append(name)
            return SimpleNamespace()
        return original(name, *args, **kwargs)

    monkeypatch.setattr(runtime, "uses_open_metax_runtime", lambda: False)
    monkeypatch.setattr(builtins, "__import__", observe)
    runtime.register_runtime()
    assert imported == list(expected)


def test_open_runtime_rejects_an_incompatible_api_before_binding(monkeypatch):
    from vllm.distributed.device_communicators import cuda_wrapper

    from vllm_fl.dispatch.backends.vendor.metax import runtime

    original = cuda_wrapper.CudaRTLibrary
    monkeypatch.setattr(runtime, "_REGISTERED", False)
    monkeypatch.setattr(runtime, "uses_open_metax_runtime", lambda: True)
    monkeypatch.setattr(runtime, "version", lambda _: "0.29.0")
    with pytest.raises(RuntimeError, match="pinned"):
        runtime.register_runtime()
    assert cuda_wrapper.CudaRTLibrary is original
