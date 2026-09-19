# SPDX-License-Identifier: Apache-2.0
"""The real runner honors explicit GLM layouts and preserves other backends."""

import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from vllm.v1.kv_cache_interface import MLAAttentionSpec

from vllm_fl.models.glm5_next_kpool import Glm5NextIndexerAttentionBackend
from vllm_fl.runtime.kv_layout import get_physical_cache_layout


def test_tail_backend_preserves_stride_capability_without_changing_deepseek():
    from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerBackend

    from vllm_fl.models.glm5_next_kpool import Glm5NextTailCache

    # The runner replaces the spec's flag with the backend's declaration.
    # Tail pages share padded index-cache allocations, so that declaration
    # must survive removing the old global Deepseek backend patch.
    cache = object.__new__(Glm5NextTailCache)
    torch.nn.Module.__init__(cache)
    cache.index_kpool = 4
    cache.head_dim = 128
    spec = cache.get_kv_cache_spec(None)
    assert spec.indexes_kv_by_block_stride
    assert cache.get_attn_backend().indexes_kv_by_block_stride()
    assert not DeepseekV32IndexerBackend.indexes_kv_by_block_stride()


@pytest.mark.parametrize("preinitialized", [False, True])
def test_real_kv_registry_handles_lazy_and_late_plugin_registration(preinitialized):
    code = f"""
import torch
from vllm.v1 import kv_cache_spec_registry as registry
from vllm.v1.core import single_type_kv_cache_manager as managers
from vllm.v1.kv_cache_interface import FullAttentionSpec
from vllm_fl.models.glm5_next_kpool import KpoolTailSpec, KpoolTailManager
from vllm_fl.patches.glm5_next_kpool_v024 import install_glm5_next_kpool_v024

registry._REGISTRY_KVCACHESPEC_LIST.clear()
if {preinitialized!r}:
    managers.register_all_kvcache_specs(None)
install_glm5_next_kpool_v024()
tail = KpoolTailSpec(block_size=4, num_kv_heads=1, head_size=128,
                    dtype=torch.float32, sliding_window=4)
ordinary = FullAttentionSpec(block_size=16, num_kv_heads=1, head_size=128,
                             dtype=torch.float32)
registry.KVCacheSpecRegistry.check_kv_cache_spec_registry(
    {{"tail": tail, "ordinary": ordinary}})
assert registry.KVCacheSpecRegistry.get_manager_class(tail) is KpoolTailManager
assert registry.KVCacheSpecRegistry.get_uniform_type_base_spec(tail) is KpoolTailSpec
assert registry.KVCacheSpecRegistry.get_manager_class(ordinary) is managers.FullAttentionManager
assert registry.KVCacheSpecRegistry.get_uniform_type_base_spec(ordinary) is FullAttentionSpec
# Re-registration keeps the same native and custom manager contracts.
install_glm5_next_kpool_v024()
managers.register_all_kvcache_specs(None)
assert registry.KVCacheSpecRegistry.get_manager_class(tail) is KpoolTailManager
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=120,
        env=os.environ.copy(),
    )
    assert result.returncode == 0, result.stdout + result.stderr


class OtherCompressedBackend:
    @staticmethod
    def get_kv_cache_shape(blocks, page, heads, width, **kwargs):
        return blocks, page, heads, width


def run_reshape(backend, spec, kernel_size):
    from vllm_fl.worker.model_runner import ModelRunnerFL

    group = SimpleNamespace(
        kv_cache_spec=spec,
        backend=backend,
        kv_cache_group_id=0,
        layer_names=["indexer"],
    )
    runner = SimpleNamespace(
        kv_cache_config=SimpleNamespace(kv_cache_tensors=[], kv_cache_groups=[group]),
        cache_config=SimpleNamespace(cache_dtype="auto"),
        runner_only_attn_layers=set(),
        _kv_cache_spec_attn_group_iterator=lambda: iter([group]),
    )
    raw = torch.zeros(spec.page_size_bytes * 3, dtype=torch.uint8)
    output = ModelRunnerFL._reshape_kv_cache_tensors(
        runner, {"indexer": raw}, [kernel_size]
    )
    return raw, output["indexer"]


@pytest.mark.parametrize("storage", [48, 64, 96])
def test_other_compressed_backend_keeps_own_page_size(storage):
    spec = MLAAttentionSpec(
        block_size=storage * 4,
        compress_ratio=4,
        num_kv_heads=1,
        head_size=132,
        dtype=torch.uint8,
    )
    assert get_physical_cache_layout(OtherCompressedBackend, spec) is None
    raw, cache = run_reshape(OtherCompressedBackend, spec, spec.block_size)
    assert cache.shape == (3, storage, 1, 132)
    assert cache.data_ptr() == raw.data_ptr()


@pytest.mark.parametrize("logical,padding", [(256, 0), (512, 0), (512, 256)])
def test_glm_padded_pages_are_views_with_shared_descriptor(
    logical, padding, monkeypatch
):
    from vllm.platforms import current_platform

    monkeypatch.setattr(
        current_platform, "get_device_capability", lambda: SimpleNamespace(major=9)
    )
    spec = MLAAttentionSpec(
        block_size=logical,
        compress_ratio=4,
        num_kv_heads=1,
        head_size=132,
        dtype=torch.uint8,
        page_size_padded=logical // 4 * 132 + padding,
        indexes_kv_by_block_stride=True,
    )
    layout = get_physical_cache_layout(Glm5NextIndexerAttentionBackend, spec)
    raw, cache = run_reshape(Glm5NextIndexerAttentionBackend, spec, 64)
    assert cache.shape[:2] == (3 * layout.pages_per_block, layout.kernel_block_size)
    assert cache.stride(0) == spec.page_size_bytes // layout.pages_per_block
    assert cache.data_ptr() == raw.data_ptr()
    cache[-1, -1, ...] = 17
    assert raw.eq(17).any()


def test_metadata_builder_delegates_non_glm_and_uses_same_glm_layout(monkeypatch):
    from vllm.platforms import current_platform
    from vllm.v1.worker.utils import AttentionGroup

    from vllm_fl.patches import glm5_next_kpool_v024 as hooks

    monkeypatch.setattr(
        current_platform, "get_device_capability", lambda: SimpleNamespace(major=9)
    )
    delegated = []
    monkeypatch.setitem(
        hooks._RUNTIME_BASELINES,
        hooks._kpool_target(AttentionGroup, "create_metadata_builders"),
        lambda *args: delegated.append(args[0]),
    )
    patched = hooks._create_metadata_builders_patch("test-layout").replacement
    spec = MLAAttentionSpec(
        block_size=512,
        compress_ratio=4,
        num_kv_heads=1,
        head_size=132,
        dtype=torch.uint8,
    )
    group = SimpleNamespace(
        backend=OtherCompressedBackend, kv_cache_spec=spec, layer_names=["indexer"]
    )
    patched(group, None, torch.device("cpu"), 64)
    assert delegated == [group]
    monkeypatch.setattr(
        Glm5NextIndexerAttentionBackend,
        "get_builder_cls",
        lambda: lambda spec, *args: SimpleNamespace(kv_cache_spec=spec),
    )
    group.backend = Glm5NextIndexerAttentionBackend
    patched(group, None, torch.device("cpu"), 64)
    builder = group.metadata_builders[0]
    assert builder.kv_cache_spec.storage_block_size == 64
    assert builder._glm5_physical_layout.pages_per_block == 2
