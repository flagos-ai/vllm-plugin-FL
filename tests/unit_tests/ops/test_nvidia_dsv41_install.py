# Copyright (c) 2026 BAAI. All rights reserved.

"""DSV4.1's EMPTY kernel selection must preserve native model ownership."""

from types import SimpleNamespace

import pytest
import torch

from tests.utils.device_utils import get_backend

pytestmark = pytest.mark.skipif(
    get_backend() != "nvidia", reason="NVIDIA DSV4.1 EMPTY integration"
)


def _config():
    from vllm.config import AttentionConfig, CacheConfig

    return SimpleNamespace(
        model_config=SimpleNamespace(architectures=["DeepseekV41ForCausalLM"]),
        attention_config=AttentionConfig(),
        cache_config=CacheConfig(),
        speculative_config=None,
    )


@pytest.fixture
def adapter_state(monkeypatch):
    from vllm.model_executor.custom_op import op_registry_oot

    from vllm_fl.ops import nvidia_dsv41

    monkeypatch.delitem(op_registry_oot, "SparseAttnIndexer", raising=False)
    monkeypatch.setattr(nvidia_dsv41, "_installed", False)
    yield nvidia_dsv41
    op_registry_oot.pop("SparseAttnIndexer", None)


@pytest.mark.parametrize("architectures", [[], ["Qwen4ExpForCausalLM"]])
def test_dsv41_install_does_not_modify_other_models(adapter_state, architectures):
    config = _config()
    config.model_config.architectures = architectures
    assert adapter_state.install_dsv41_empty_adapters(config) is False
    assert adapter_state._installed is False


def test_dsv41_install_ignores_missing_model(adapter_state):
    config = _config()
    config.model_config = None
    assert adapter_state.install_dsv41_empty_adapters(config) is False


def test_dsv41_install_preserves_native_wheel(adapter_state, monkeypatch):
    monkeypatch.setattr(adapter_state, "has_native_vllm_kernels", lambda: True)
    # Native kernels remain untouched even for a configuration this EMPTY
    # implementation deliberately does not support.
    config = _config()
    config.speculative_config = object()
    assert adapter_state.install_dsv41_empty_adapters(config) is False
    assert adapter_state._installed is False


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("speculative_config", object(), "DSpark/speculative"),
        ("attention_config.backend", "TRITON_MLA", "sparse MLA backend"),
        ("attention_config.indexer_kv_dtype", "fp4", "FP8 dense indexer"),
        ("attention_config.indexer_sparse_logits", True, "FP8 dense indexer"),
        ("cache_config.cache_dtype", "bfloat16", "584-byte"),
    ],
)
def test_dsv41_install_rejects_unsupported_config(
    adapter_state, monkeypatch, field, value, error
):
    from vllm.platforms import current_platform
    from vllm.platforms.interface import DeviceCapability

    monkeypatch.setattr(adapter_state, "has_native_vllm_kernels", lambda: False)
    monkeypatch.setattr(
        current_platform, "get_device_capability", lambda: DeviceCapability(9, 0)
    )
    config = _config()
    target = config
    *prefix, name = field.split(".")
    for part in prefix:
        target = getattr(target, part)
    setattr(target, name, value)
    with pytest.raises(NotImplementedError, match=error):
        adapter_state.install_dsv41_empty_adapters(config)
    assert adapter_state._installed is False


@pytest.mark.parametrize("major", [8, 10, None])
def test_dsv41_install_rejects_unsupported_hardware(adapter_state, monkeypatch, major):
    from vllm.platforms import current_platform
    from vllm.platforms.interface import DeviceCapability

    monkeypatch.setattr(adapter_state, "has_native_vllm_kernels", lambda: False)
    capability = None if major is None else DeviceCapability(major, 0)
    monkeypatch.setattr(current_platform, "get_device_capability", lambda: capability)
    with pytest.raises(NotImplementedError, match="Hopper SM90"):
        adapter_state.install_dsv41_empty_adapters(_config())
    assert adapter_state._installed is False


@pytest.mark.gpu
def test_dsv41_installs_real_kernels_without_advertising_native_libraries(
    adapter_state, monkeypatch
):
    from vllm.model_executor.layers import sparse_attn_indexer
    from vllm.models.deepseek_v4.nvidia.ops import o_proj
    from vllm.models.deepseek_v41.nvidia import flashmla
    from vllm.platforms import current_platform
    from vllm.utils.import_utils import has_deep_gemm
    from vllm.v1.attention.ops.flashmla import is_flashmla_sparse_supported

    from vllm_fl.dispatch.backends.flaggems.impl import dsv41_mla
    from vllm_fl.ops import nvidia_dsv41_indexer

    if adapter_state.has_native_vllm_kernels():
        pytest.skip("Real EMPTY registration is tested in the EMPTY environment")
    capability = current_platform.get_device_capability()
    if capability is None or capability.major != 9:
        pytest.skip("DSV4.1 sparse MLA currently targets Hopper")

    bindings = [
        (flashmla, "flash_mla_sparse_fwd", dsv41_mla.flash_mla_sparse_fwd),
        (flashmla, "flash_mla_with_kvcache", dsv41_mla.flash_mla_with_kvcache),
        (
            flashmla.DeepseekSparseSWAFlashMLAMetadataBuilder,
            "build_tile_scheduler",
            dsv41_mla.build_tile_scheduler,
        ),
        (
            sparse_attn_indexer,
            "fp8_fp4_mqa_logits",
            nvidia_dsv41_indexer.fp8_mqa_logits,
        ),
        (
            sparse_attn_indexer,
            "fp8_fp4_paged_mqa_logits",
            nvidia_dsv41_indexer.fp8_paged_mqa_logits,
        ),
        (o_proj, "fp8_einsum", nvidia_dsv41_indexer.fp8_einsum),
    ]
    # Record every original module attribute before the real installer mutates
    # it, so pytest restores these bindings even when an assertion fails.
    for module, name, _ in bindings:
        monkeypatch.setattr(module, name, getattr(module, name))

    def native_capabilities():
        return (
            adapter_state.has_native_vllm_kernels(),
            has_deep_gemm(),
            is_flashmla_sparse_supported(),
        )

    before = native_capabilities()
    assert adapter_state.install_dsv41_empty_adapters(_config()) is True
    assert adapter_state.install_dsv41_empty_adapters(_config()) is True
    assert native_capabilities() == before
    for module, name, expected in bindings:
        assert getattr(module, name) is expected
    for op in (
        "_C::top_k_per_row_prefill",
        "_C_cache_ops::cp_gather_indexer_k_quant_cache",
        "_C::fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert",
        "_C::fused_deepseek_v4_kv_rope_insert",
    ):
        assert torch._C._dispatch_has_kernel_for_dispatch_key(op, "CUDA")
