# Copyright (c) 2026 BAAI. All rights reserved.

"""The NVIDIA runtime uses vLLM's native CUDA wheel and kernel registrations."""

import importlib
import importlib.machinery

import pytest
import torch

from tests.utils.device_utils import get_backend

pytestmark = pytest.mark.skipif(
    get_backend() != "nvidia", reason="NVIDIA native CUDA runtime"
)


def test_native_cuda_extension_is_loaded():
    extension = importlib.import_module("vllm._C_stable_libtorch")
    assert extension.__file__.endswith(tuple(importlib.machinery.EXTENSION_SUFFIXES))


@pytest.mark.parametrize(
    "name,key",
    [
        ("_C::silu_and_mul", "CUDA"),
        ("_C::weak_ref_tensor", "CUDA"),
        ("_C::get_cuda_view_from_cpu_tensor", "CPU"),
        ("_C_cache_ops::reshape_and_cache_flash", "CUDA"),
    ],
)
def test_native_dispatch_kernels_are_registered(name, key):
    importlib.import_module("vllm._C_stable_libtorch")
    assert torch._C._dispatch_has_kernel_for_dispatch_key(name, key)


def test_native_flash_attention_import_is_executable():
    from vllm.vllm_flash_attn import flash_attn_varlen_func

    assert callable(flash_attn_varlen_func)
