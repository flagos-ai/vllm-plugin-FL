# Copyright (c) 2026 BAAI. All rights reserved.

"""MCCL abort forwards the real communicator ABI and propagates failures."""

import ctypes
import importlib
from unittest.mock import Mock

import pytest

from tests.utils.device_utils import get_backend

pytestmark = pytest.mark.skipif(get_backend() != "metax", reason="MetaX patch")


@pytest.fixture
def module():
    return importlib.import_module(
        "vllm_fl.dispatch.backends.vendor.metax.patches.pynccl_wrapper"
    )


def test_abort_export_signature(module):
    exports = [
        function
        for function in module.MCCLLibrary.exported_functions
        if function.name == "mcclCommAbort"
    ]
    assert len(exports) == 1
    assert exports[0].restype is module.ncclResult_t
    assert exports[0].argtypes == [module.ncclComm_t]


@pytest.mark.parametrize("result", [0, 3])
def test_abort_forwarding_and_error_propagation(module, result):
    library = module.MCCLLibrary.__new__(module.MCCLLibrary)
    abort = Mock(return_value=result)
    destroy = Mock()
    library._funcs = {
        "mcclCommAbort": abort,
        "mcclCommDestroy": destroy,
        "mcclGetErrorString": Mock(return_value=b"invalid communicator"),
    }
    comm = module.ncclComm_t(1234)
    if result:
        with pytest.raises(RuntimeError, match="invalid communicator"):
            library.ncclCommAbort(comm)
    else:
        assert library.ncclCommAbort(comm) is None
    abort.assert_called_once_with(comm)
    destroy.assert_not_called()


def test_library_without_abort_is_rejected(module, monkeypatch):
    class IncompleteLibrary:
        def __getattr__(self, name):
            if name == "mcclCommAbort":
                raise AttributeError(name)
            return Mock(return_value=0)

    monkeypatch.setattr(ctypes, "CDLL", lambda path: IncompleteLibrary())
    monkeypatch.setattr(module.MCCLLibrary, "path_to_library_cache", {})
    monkeypatch.setattr(module.MCCLLibrary, "path_to_dict_mapping", {})
    with pytest.raises(AttributeError, match="mcclCommAbort"):
        module.MCCLLibrary("/unit-test-mccl.so")
