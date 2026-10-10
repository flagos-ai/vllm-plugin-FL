# SPDX-License-Identifier: Apache-2.0
import pytest

from vllm_fl.kernels.glm5_next import provider


@pytest.fixture(autouse=True)
def clear_provider_cache():
    provider._has_nvidia_reference_kernels.cache_clear()
    yield
    provider._has_nvidia_reference_kernels.cache_clear()


@pytest.mark.parametrize(
    "cuda,abi,deep,expected",
    [
        (True, True, True, True),
        (False, True, True, False),
        (True, False, True, False),
        (True, True, False, False),
    ],
)
def test_native_requires_all_capabilities(monkeypatch, cuda, abi, deep, expected):
    monkeypatch.setattr(provider.current_platform, "is_cuda", lambda: cuda)
    monkeypatch.setattr(provider, "_has_vllm_native_extension", lambda: abi)
    monkeypatch.setattr(provider, "_has_deep_gemm", lambda: deep)
    assert provider.use_nvidia_reference() is expected
