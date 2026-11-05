# Copyright (c) 2026 BAAI. All rights reserved.

"""Abort only the communicator created by this test, through vLLM teardown."""

import importlib
import threading

import pytest
import torch

from tests.utils.device_utils import get_backend

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(get_backend() != "metax", reason="Requires the MetaX SDK"),
]


def test_native_mccl_abort_through_pynccl_destroy(monkeypatch):
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

    module = importlib.import_module(
        "vllm_fl.dispatch.backends.vendor.metax.patches.pynccl_wrapper"
    )
    torch.cuda.set_device(0)
    library = module.MCCLLibrary()
    comm = library.ncclCommInitRank(1, library.ncclGetUniqueId(), 0)
    completed = threading.Event()
    errors = []
    calls = []
    native_abort = library.ncclCommAbort

    def abort(handle):
        try:
            calls.append(handle)
            native_abort(handle)
        except BaseException as error:
            errors.append(error)
            raise
        finally:
            completed.set()

    monkeypatch.setattr(library, "ncclCommAbort", abort)
    owner = PyNcclCommunicator.__new__(PyNcclCommunicator)
    owner.available = True
    owner.disabled = False
    owner.device = torch.device("cuda:0")
    owner.nccl = library
    owner.comm = comm
    owner.destroy()
    assert completed.wait(10), "Native MCCL abort did not finish"
    assert not errors
    assert calls == [comm]
    assert not owner.available and owner.disabled
    owner.destroy()
    assert calls == [comm]
