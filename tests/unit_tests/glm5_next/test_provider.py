# SPDX-License-Identifier: Apache-2.0
import pytest

from vllm_fl.kernels.glm5_next import provider


@pytest.fixture(autouse=True)
def clear_provider_cache():
    provider.get_glm5_provider.cache_clear()
    provider._has_nvidia_reference_kernels.cache_clear()
    yield
    provider.get_glm5_provider.cache_clear()
    provider._has_nvidia_reference_kernels.cache_clear()


def test_auto_uses_nvidia_reference_on_cuda(monkeypatch) -> None:
    monkeypatch.delenv(provider.ENV_NAME, raising=False)
    monkeypatch.setattr(provider.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(provider, "_has_vllm_native_extension", lambda: True)
    monkeypatch.setattr(provider, "_has_deep_gemm", lambda: True)

    assert provider.get_glm5_provider() == "auto"
    assert provider.use_nvidia_reference()


def test_auto_falls_back_without_nvidia_reference_prerequisites(monkeypatch) -> None:
    monkeypatch.delenv(provider.ENV_NAME, raising=False)
    monkeypatch.setattr(provider.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(provider, "_has_vllm_native_extension", lambda: False)
    monkeypatch.setattr(provider, "_has_deep_gemm", lambda: True)

    assert provider.get_glm5_provider() == "auto"
    assert not provider.use_nvidia_reference()


def test_flaggems_overrides_cuda_reference(monkeypatch) -> None:
    monkeypatch.setenv(provider.ENV_NAME, "flaggems")
    monkeypatch.setattr(provider.current_platform, "is_cuda", lambda: True)

    assert provider.get_glm5_provider() == "flaggems"
    assert not provider.use_nvidia_reference()


def test_nvidia_is_rejected_on_non_cuda(monkeypatch) -> None:
    monkeypatch.setenv(provider.ENV_NAME, "nvidia")
    monkeypatch.setattr(provider.current_platform, "is_cuda", lambda: False)

    with pytest.raises(RuntimeError, match="requires an NVIDIA CUDA platform"):
        provider.get_glm5_provider()


def test_invalid_provider_is_rejected(monkeypatch) -> None:
    monkeypatch.setenv(provider.ENV_NAME, "portable")

    with pytest.raises(ValueError, match="auto\\|nvidia\\|flaggems"):
        provider.get_glm5_provider()
