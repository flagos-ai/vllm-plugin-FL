# Copyright (c) 2026 BAAI. All rights reserved.

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("torch_npu")

import vllm_fl.dispatch.backends.vendor.ascend.impl.batch_memcpy as batch_memcpy_impl
import vllm_fl.dispatch.backends.vendor.ascend.patch as ascend_patch
from vllm_fl.dispatch.backends.vendor.ascend.patches import triton_compat


def test_apply_ascend_patches_installs_mamba_batch_memcpy(monkeypatch):
    from vllm.v1.worker import mamba_utils

    create_func = mamba_utils.MambaCopyBuffers.create.__func__
    if getattr(create_func, "_vllm_fl_ascend_patched", False):
        # Keep the test independent of earlier patch installation.
        monkeypatch.setattr(
            mamba_utils.MambaCopyBuffers,
            "create",
            classmethod(create_func.__wrapped__.__func__),
        )
    sentinel = object()
    monkeypatch.setattr(mamba_utils, "batch_memcpy", sentinel)
    monkeypatch.setattr(mamba_utils, "batch_memcpy_kernel", sentinel)
    monkeypatch.setattr(ascend_patch, "_patches_applied", False)
    monkeypatch.setattr(triton_compat, "patch_triton_compile_hooks", lambda: None)
    for patch_name in (
        "patch_topk_topp_sampler",
        "patch_causal_conv1d",
        "patch_fla_ops",
        "patch_op_cls",
        "patch_fused_moe",
    ):
        monkeypatch.setattr(ascend_patch, patch_name, lambda: None)

    ascend_patch.apply_ascend_patches()

    assert mamba_utils.batch_memcpy is batch_memcpy_impl.batch_memcpy
    assert mamba_utils.batch_memcpy_kernel is batch_memcpy_impl.batch_memcpy_kernel


def test_mamba_patch_rolls_back_after_partial_install(monkeypatch):
    from vllm.v1 import worker

    owner_calls = []

    class Buffers:
        @classmethod
        def create(cls, max_num_reqs, kv_cache_config, copy_funcs, make_buffer):
            owner_calls.append(cls)
            return make_buffer(max_num_reqs, dtype=torch.uint64)

    original_create = vars(Buffers)["create"]
    original_kernel = object()
    original_memcpy = object()

    class RejectLastAssignment:
        def __init__(self):
            self.MambaCopyBuffers = Buffers
            self.batch_memcpy_kernel = original_kernel
            self.batch_memcpy = original_memcpy
            self.fail_once = True

        def __setattr__(self, name, value):
            object.__setattr__(self, name, value)
            if name == "batch_memcpy" and getattr(self, "fail_once", False):
                object.__setattr__(self, "fail_once", False)
                raise RuntimeError("injected installation failure")

    fake_utils = RejectLastAssignment()
    monkeypatch.setattr(worker, "mamba_utils", fake_utils)

    assert ascend_patch.patch_mamba_batch_memcpy() is False
    assert vars(Buffers)["create"] is original_create
    assert fake_utils.batch_memcpy_kernel is original_kernel
    assert fake_utils.batch_memcpy is original_memcpy

    assert ascend_patch.patch_mamba_batch_memcpy() is True
    installed_create = vars(Buffers)["create"]
    assert installed_create.__func__.__wrapped__.__func__ is original_create.__func__
    assert fake_utils.batch_memcpy_kernel is batch_memcpy_impl.batch_memcpy_kernel
    assert fake_utils.batch_memcpy is batch_memcpy_impl.batch_memcpy
    assert Buffers.create(1, None, None, lambda size, *, dtype: dtype) == torch.int64
    assert owner_calls == [Buffers]

    assert ascend_patch.patch_mamba_batch_memcpy() is True
    assert vars(Buffers)["create"] is installed_create
    assert fake_utils.batch_memcpy_kernel is batch_memcpy_impl.batch_memcpy_kernel
    assert fake_utils.batch_memcpy is batch_memcpy_impl.batch_memcpy


def test_mamba_patch_does_not_publish_when_create_lookup_fails(monkeypatch):
    from vllm.v1 import worker

    class BrokenMeta(type):
        @property
        def create(cls):
            raise RuntimeError("injected descriptor failure")

    class Buffers(metaclass=BrokenMeta):
        pass

    fake_utils = SimpleNamespace(
        MambaCopyBuffers=Buffers,
        batch_memcpy_kernel=object(),
        batch_memcpy=object(),
    )
    original_kernel = fake_utils.batch_memcpy_kernel
    original_memcpy = fake_utils.batch_memcpy
    monkeypatch.setattr(worker, "mamba_utils", fake_utils)

    assert ascend_patch.patch_mamba_batch_memcpy() is False
    assert fake_utils.batch_memcpy_kernel is original_kernel
    assert fake_utils.batch_memcpy is original_memcpy


def test_failed_mamba_install_leaves_apply_retryable(monkeypatch):
    monkeypatch.setattr(ascend_patch, "_patches_applied", False)
    monkeypatch.setattr(ascend_patch, "_core_patches_applied", False)
    calls = []
    monkeypatch.setattr(
        "vllm_fl.dispatch.backends.vendor.ascend.patches.triton_compat.patch_triton_compile_hooks",
        lambda: calls.append("triton"),
    )
    for patch_name in (
        "patch_topk_topp_sampler",
        "patch_causal_conv1d",
        "patch_fla_ops",
        "patch_op_cls",
        "patch_fused_moe",
    ):
        monkeypatch.setattr(
            ascend_patch,
            patch_name,
            lambda name=patch_name: calls.append(name),
        )
    monkeypatch.setattr(ascend_patch, "patch_mamba_batch_memcpy", lambda: False)

    ascend_patch.apply_ascend_patches()
    assert ascend_patch._patches_applied is False
    assert ascend_patch._core_patches_applied is True
    assert calls == [
        "triton",
        "patch_topk_topp_sampler",
        "patch_causal_conv1d",
        "patch_fla_ops",
        "patch_op_cls",
        "patch_fused_moe",
    ]

    monkeypatch.setattr(ascend_patch, "patch_mamba_batch_memcpy", lambda: True)
    ascend_patch.apply_ascend_patches()
    assert ascend_patch._patches_applied is True
    assert len(calls) == 6


def test_mamba_copy_buffers_use_ascend_supported_pointer_dtype(monkeypatch):
    from vllm.v1.worker import mamba_utils

    ascend_patch.patch_mamba_batch_memcpy()
    patched_create = mamba_utils.MambaCopyBuffers.create.__func__
    assert patched_create._vllm_fl_ascend_patched

    # Reapplying the patch must not wrap create a second time.
    ascend_patch.patch_mamba_batch_memcpy()
    assert mamba_utils.MambaCopyBuffers.create.__func__ is patched_create

    mamba_spec = object()
    monkeypatch.setattr(
        mamba_utils,
        "get_mamba_groups",
        lambda _config: ([0], mamba_spec),
    )
    kv_cache_config = SimpleNamespace(
        kv_cache_groups=[SimpleNamespace(layer_names=["layer.0", "layer.1"])]
    )
    allocations = []

    def make_buffer(size, *, dtype):
        allocations.append((size, dtype))
        return SimpleNamespace(dtype=dtype)

    buffers = mamba_utils.MambaCopyBuffers.create(
        max_num_reqs=3,
        kv_cache_config=kv_cache_config,
        copy_funcs=(object(), object()),
        make_buffer=make_buffer,
    )

    assert allocations == [
        (12, torch.int64),
        (12, torch.int64),
        (12, torch.int32),
    ]
    assert buffers.src_ptrs.dtype == torch.int64
    assert buffers.dst_ptrs.dtype == torch.int64
    assert buffers.sizes.dtype == torch.int32


@pytest.mark.gpu
def test_mamba_copy_buffers_create_allocates_on_npu(monkeypatch):
    from vllm.v1.utils import CpuGpuBuffer
    from vllm.v1.worker import mamba_utils

    ascend_patch.patch_mamba_batch_memcpy()
    monkeypatch.setattr(
        mamba_utils,
        "get_mamba_groups",
        lambda _config: ([0], object()),
    )
    kv_cache_config = SimpleNamespace(
        kv_cache_groups=[SimpleNamespace(layer_names=["layer.0"])]
    )

    def make_buffer(*size, dtype):
        return CpuGpuBuffer(
            *size,
            dtype=dtype,
            device=torch.device("npu"),
            pin_memory=False,
        )

    buffers = mamba_utils.MambaCopyBuffers.create(
        max_num_reqs=2,
        kv_cache_config=kv_cache_config,
        copy_funcs=(object(),),
        make_buffer=make_buffer,
    )

    assert buffers.src_ptrs.cpu.dtype == torch.int64
    assert buffers.src_ptrs.gpu.dtype == torch.int64
    assert buffers.dst_ptrs.cpu.dtype == torch.int64
    assert buffers.dst_ptrs.gpu.dtype == torch.int64
    assert buffers.sizes.cpu.dtype == torch.int32
    assert buffers.sizes.gpu.dtype == torch.int32


def test_batch_memcpy_launches_with_8192_byte_blocks(monkeypatch):
    captured = {}

    class FakeKernel:
        def __getitem__(self, grid):
            captured["grid"] = grid

            def launch(*args, **kwargs):
                captured["args"] = args
                captured["kwargs"] = kwargs

            return launch

    monkeypatch.setattr(batch_memcpy_impl, "batch_memcpy_kernel", FakeKernel())
    src_ptrs = torch.empty(3)
    dst_ptrs = torch.empty(3)
    sizes = torch.empty(3)

    batch_memcpy_impl.batch_memcpy(src_ptrs, dst_ptrs, sizes)

    assert captured["grid"] == (3,)
    assert captured["args"] == (src_ptrs, dst_ptrs, sizes)
    assert captured["kwargs"] == {"BLOCK_SIZE": 8192}


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_batch_memcpy_copies_multiple_blocks_and_masked_tail(dtype):
    element_size = torch.empty((), dtype=dtype).element_size()
    byte_sizes = [
        element_size,
        8192 - element_size,
        8192,
        8192 + element_size,
        2 * 8192 + 3 * element_size,
    ]
    generator = torch.Generator().manual_seed(71)
    sources = [
        torch.randn(size // element_size, generator=generator, dtype=dtype).npu()
        for size in byte_sizes
    ]
    destinations = [torch.empty_like(source) for source in sources]
    src_ptrs = torch.tensor(
        [source.data_ptr() for source in sources], dtype=torch.int64, device="npu"
    )
    dst_ptrs = torch.tensor(
        [destination.data_ptr() for destination in destinations],
        dtype=torch.int64,
        device="npu",
    )
    sizes = torch.tensor(byte_sizes, dtype=torch.int32, device="npu")

    batch_memcpy_impl.batch_memcpy(src_ptrs, dst_ptrs, sizes)
    torch.npu.synchronize()

    for source, destination in zip(sources, destinations):
        torch.testing.assert_close(destination.cpu(), source.cpu(), rtol=0, atol=0)
