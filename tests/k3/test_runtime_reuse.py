"""K3 ABI additions reuse the original SDK wrapper methods and signatures."""

from unittest.mock import Mock

import pytest

from vllm_fl.dispatch.backends.vendor.metax.patches.cuda_wrapper import (
    CudaRTLibrary as NativeCudaRTLibrary,
)
from vllm_fl.dispatch.backends.vendor.metax.patches.pynccl_wrapper import (
    MCCLLibrary as NativeMCCLLibrary,
)
from vllm_fl.patches.kimi_k3 import cuda_runtime, mccl_runtime


def test_cuda_calls_and_exports_are_inherited():
    assert (
        cuda_runtime.CudaRTLibrary.exported_functions
        is NativeCudaRTLibrary.exported_functions
    )
    for name, fn in NativeCudaRTLibrary.__dict__.items():
        if callable(fn) and name != "__init__":
            assert getattr(cuda_runtime.CudaRTLibrary, name) is fn


def test_mccl_collectives_are_inherited_without_reordering():
    overrides = {
        "__init__",
        "ncclCommAbort",
        "ncclCommWindowRegister",
        "ncclCommWindowDeregister",
    }
    for name, fn in NativeMCCLLibrary.__dict__.items():
        if callable(fn) and name not in overrides:
            assert getattr(mccl_runtime.MCCLLibrary, name) is fn
    baseline = [f.name for f in NativeMCCLLibrary.exported_functions]
    selected = [f.name for f in mccl_runtime.MCCLLibrary.exported_functions]
    expected = list(baseline)
    expected.insert(expected.index("mcclGroupStart"), "mcclCommAbort")
    assert selected == expected
    assert (
        mccl_runtime.MCCLLibrary.path_to_library_cache
        is not NativeMCCLLibrary.path_to_library_cache
    )


def test_cuda_keeps_verified_explicit_library_resolution(monkeypatch):
    init = Mock()
    monkeypatch.setattr(NativeCudaRTLibrary, "__init__", init)
    monkeypatch.setenv("VLLM_CUDART_SO_PATH", "/verified/libmcruntime.so")
    cuda_runtime.CudaRTLibrary()
    init.assert_called_once_with("/verified/libmcruntime.so")


def test_missing_cuda_runtime_fails_explicitly(monkeypatch):
    monkeypatch.delenv("VLLM_CUDART_SO_PATH", raising=False)
    monkeypatch.setattr(cuda_runtime, "find_loaded_library", lambda name: None)
    with pytest.raises(RuntimeError, match="libmcruntime"):
        cuda_runtime.CudaRTLibrary()


def test_abort_and_unsupported_windows_keep_the_original_contract():
    wrapper = mccl_runtime.MCCLLibrary.__new__(mccl_runtime.MCCLLibrary)
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
