# SPDX-License-Identifier: Apache-2.0
"""THead reuses NCCL without initializing NVIDIA-only optional packages."""

import builtins
from importlib import util
from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize("world_size", [1, 4])
@pytest.mark.parametrize("stateless", [False, True])
@pytest.mark.parametrize("parent_patched", [False, True])
def test_nccl_transport_without_flashinfer(
    monkeypatch, world_size, stateless, parent_patched
):
    from vllm.distributed.device_communicators import cuda_communicator

    from vllm_fl.distributed import nccl_communicator as module

    if parent_patched:
        monkeypatch.setattr(
            cuda_communicator.CudaCommunicator,
            "all_reduce",
            lambda *a: pytest.fail("THead test invoked another vendor's global patch"),
        )
    parent_method = cuda_communicator.CudaCommunicator.all_reduce
    # This test simulates THead's upstream NCCL method. MetaX patches the
    # shared parent method globally at registration; load the installed source
    # independently and bind only this test's subclass, preserving MetaX's patch.
    spec = util.spec_from_file_location(
        "vllm.distributed.device_communicators._nccl_contract",
        cuda_communicator.__file__,
    )
    upstream = util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    monkeypatch.setattr(
        module.NcclCommunicatorFL, "all_reduce", upstream.CudaCommunicator.all_reduce
    )
    original_import = builtins.__import__

    def checked_import(name, *args, **kwargs):
        if "flashinfer" in name or "custom_all_reduce" in name:
            pytest.fail(f"unsupported NVIDIA communicator imported: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", checked_import)
    group, store = object(), object() if stateless else None

    def init_base(self, cpu_group, device, device_group, unique_name, *args):
        self.cpu_group = cpu_group
        self.device = device
        self.device_group = device_group
        self.world_size = world_size
        self.use_all2all = False
        self.all2all_manager = None

    monkeypatch.setattr(module.DeviceCommunicatorBase, "__init__", init_base)
    calls = []
    transport = SimpleNamespace(
        disabled=False,
        world_size=world_size,
        all_reduce=lambda x: x * world_size,
        destroy=lambda: calls.append("destroy"),
    )

    def init_transport(**kwargs):
        calls.append(kwargs)
        return transport

    monkeypatch.setattr(module, "PyNcclCommunicator", init_transport)
    communicator = module.NcclCommunicatorFL(
        group, device="cuda:0", unique_name="tp", tcp_store_group=store
    )
    if world_size > 1:
        assert calls == [{"group": store or group, "device": "cuda:0"}]
        x = torch.ones(3)
        torch.testing.assert_close(communicator.all_reduce(x), x * world_size)
    else:
        assert not calls
        assert communicator.pynccl_comm is None
    communicator.destroy()
    assert cuda_communicator.CudaCommunicator.all_reduce is parent_method
    assert communicator.pynccl_comm is None
    if world_size > 1:
        assert calls[-1] == "destroy"


@pytest.mark.parametrize(
    "vendor,backend,expected",
    [
        ("thead", "nccl", "nccl_communicator.NcclCommunicatorFL"),
        ("thead", "flagcx", "communicator.CommunicatorFL"),
        ("nvidia", "nccl", "cuda_communicator.CudaCommunicator"),
    ],
)
def test_platform_selects_vendor_transport(monkeypatch, vendor, backend, expected):
    from vllm_fl.platform import PlatformFL

    monkeypatch.setattr(PlatformFL, "vendor_name", vendor)
    monkeypatch.setattr(PlatformFL, "dist_backend", backend)
    assert PlatformFL.get_device_communicator_cls().endswith(expected)
