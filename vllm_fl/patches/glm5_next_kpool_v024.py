# SPDX-License-Identifier: Apache-2.0
"""Plugin-only vLLM 0.24 KV plumbing for GLM5-Next kpool/tail caches."""

from __future__ import annotations

import logging
import os
from dataclasses import replace
from functools import wraps

from vllm.utils.math_utils import cdiv

from vllm_fl.models.glm5_next_kpool import (
    KpoolTailManager,
    KpoolTailSpec,
)

logger = logging.getLogger(__name__)


def _inner_specs(groups):
    from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs

    for group in groups:
        spec = group.kv_cache_spec
        if isinstance(spec, UniformTypeKVCacheSpecs):
            yield from spec.kv_cache_specs.items()
        else:
            yield from ((name, spec) for name in group.layer_names)


def _is_glm5_kpool_groups(groups) -> bool:
    return any(isinstance(spec, KpoolTailSpec) for _, spec in _inner_specs(groups))


def _group_glm5_kpool(vllm_config, kv_cache_spec):
    """Reference grouping semantics without modifying the vLLM package.

    Main MLA and compressed indexer layers share one token-granular block table;
    tail rings have their own 4-token group; KDA state remains a separate group.
    The reference tree slot-shares KDA/MLA storage as a memory optimization. This
    plugin keeps standalone KDA tensors but preserves every allocator/cache
    semantic involved in kpool and tail correctness.
    """
    from vllm.v1.kv_cache_interface import (
        KVCacheGroupSpec,
        MLAAttentionSpec,
        MambaSpec,
        UniformTypeKVCacheSpecs,
    )

    del vllm_config
    mamba_specs = {
        name: spec for name, spec in kv_cache_spec.items() if isinstance(spec, MambaSpec)
    }
    tail_specs = {
        name: spec
        for name, spec in kv_cache_spec.items()
        if isinstance(spec, KpoolTailSpec)
    }
    attn_specs = {
        name: spec
        for name, spec in kv_cache_spec.items()
        if not isinstance(spec, (MambaSpec, KpoolTailSpec))
    }
    assert tail_specs and all(type(spec) is MLAAttentionSpec for spec in attn_specs.values())
    index_pages = {
        spec.page_size_bytes
        for spec in attn_specs.values()
        if spec.compress_ratio > 1
    }
    assert len(index_pages) == 1
    index_page = next(iter(index_pages))

    attn_uniform = UniformTypeKVCacheSpecs.from_specs(attn_specs)
    assert attn_uniform is not None
    padded_tail_specs = {
        name: replace(spec, page_size_padded=index_page)
        for name, spec in tail_specs.items()
    }
    tail_uniform = UniformTypeKVCacheSpecs.from_specs(padded_tail_specs)
    assert tail_uniform is not None

    groups = [
        KVCacheGroupSpec(list(attn_specs), attn_uniform),
        KVCacheGroupSpec(list(padded_tail_specs), tail_uniform),
    ]
    if mamba_specs:
        exemplar = next(iter(mamba_specs.values()))
        assert all(spec == exemplar for spec in mamba_specs.values())
        groups.append(KVCacheGroupSpec(list(mamba_specs), exemplar))
    return groups


def install_glm5_next_kpool_v024() -> None:
    from vllm.platforms.interface import Platform
    from vllm.v1 import kv_cache_spec_registry
    from vllm.v1.attention.backends.mla import indexer as indexer_backend
    from vllm.v1.core import (
        kv_cache_coordinator,
        kv_cache_utils,
        single_type_kv_cache_manager,
    )
    from vllm.v1.kv_cache_interface import (
        KVCacheConfig,
        KVCacheTensor,
    )
    from vllm.v1.kv_cache_interface import AttentionSpec
    from vllm.v1.worker import utils as worker_utils

    # Upstream GLM5-Next rounds the hybrid attention block *after* accounting
    # for the KDA state page.  Stock v0.24 only does the latter, which turns a
    # requested 128-token page into 192 tokens for this model.  The kpool
    # paged-MQA layout requires a multiple of kpool * 32, so reproduce the
    # reference platform hook here without changing the FlagOS vLLM package.
    original_align_descriptor = Platform.__dict__["_align_hybrid_block_size"]
    original_align = original_align_descriptor.__func__
    if not getattr(original_align, "_glm5_kpool", False):

        def align_hybrid(cls, vllm_config, backend_cls):
            original_align(cls, vllm_config, backend_cls)
            text_config = vllm_config.model_config.hf_text_config
            if not getattr(text_config, "index_kpool_compress", False):
                return
            kpool = int(getattr(text_config, "index_kpool", 1) or 1)
            if kpool <= 1:
                return
            cache_config = vllm_config.cache_config
            old_block_size = cache_config.block_size
            capability = cls.get_device_capability()
            # DeepGEMM accepts a 32-entry paged-MQA block only on SM100.
            # H100/SM90 therefore needs kpool*64 alignment; otherwise a
            # 384/4=96 storage block would be split into illegal 32 pages.
            compressed_page = (
                64 if capability is not None and capability.major < 10 else 32
            )
            alignment = kpool * compressed_page
            aligned_block_size = alignment * cdiv(old_block_size, alignment)
            if aligned_block_size == old_block_size:
                return
            cache_config.block_size = aligned_block_size
            if cache_config.mamba_cache_mode == "align":
                cache_config.mamba_block_size = aligned_block_size
            if cache_config.mamba_page_size_padded is not None:
                assert cache_config.mamba_page_size_padded % old_block_size == 0
                cache_config.mamba_page_size_padded = (
                    cache_config.mamba_page_size_padded
                    // old_block_size
                    * aligned_block_size
                )

        align_hybrid._glm5_kpool = True
        Platform._align_hybrid_block_size = classmethod(align_hybrid)

    # Stock v0.24 copies every AttentionSpec to the selected attention-kernel
    # block size before constructing its metadata builder.  For a compressed
    # indexer that would turn logical 256 / pool 4 into an invalid 16-entry
    # DeepGEMM page.  The reference keeps the compressed spec at pool-page
    # granularity and records the co-located MLA kernel size for block-table
    # translation.
    original_create_builders = worker_utils.AttentionGroup.create_metadata_builders
    if not getattr(original_create_builders, "_glm5_kpool", False):

        def create_metadata_builders(
            self,
            vllm_config,
            device,
            kernel_block_size=None,
            num_metadata_builders=1,
        ):
            spec = self.kv_cache_spec
            is_compressed = (
                isinstance(spec, AttentionSpec)
                and spec.storage_block_size != spec.block_size
            )
            if not is_compressed:
                return original_create_builders(
                    self,
                    vllm_config,
                    device,
                    kernel_block_size,
                    num_metadata_builders,
                )

            storage_block_size = spec.storage_block_size
            if storage_block_size <= 64:
                compressed_kernel_size = storage_block_size
            else:
                compressed_kernel_size = (
                    64 if storage_block_size % 64 == 0 else 32
                )
            assert compressed_kernel_size in (32, 64)
            assert storage_block_size % compressed_kernel_size == 0
            compress_ratio = spec.block_size // storage_block_size
            builder_spec = spec.copy_with_new_block_size(
                compressed_kernel_size * compress_ratio
            )
            self.metadata_builders = [
                self.backend.get_builder_cls()(
                    builder_spec,
                    self.layer_names,
                    vllm_config,
                    device,
                )
                for _ in range(num_metadata_builders)
            ]
            if kernel_block_size is not None:
                for builder in self.metadata_builders:
                    builder.kernel_block_size = kernel_block_size

        create_metadata_builders._glm5_kpool = True
        worker_utils.AttentionGroup.create_metadata_builders = (
            create_metadata_builders
        )

    # The scheduler table is shared with the MLA cache and therefore contains
    # one physical id per 64-token kernel sub-page.  The compressed index cache
    # owns one physical page per logical block.  Translate
    #   [4*b+0, 4*b+1, 4*b+2, 4*b+3] -> [b]
    # before *all* stock builder operations, so prefill writes and decode reads
    # address the same page.  A persistent buffer keeps the path graph-safe
    # after its warm-up allocation.
    builder_cls = indexer_backend.DeepseekV32IndexerMetadataBuilder
    original_indexer_build = builder_cls.build
    if not getattr(original_indexer_build, "_glm5_kpool", False):

        def build_indexer_metadata(
            self,
            common_prefix_len,
            common_attn_metadata,
            fast_build=False,
        ):
            spec = self.kv_cache_spec
            kernel_block_size = getattr(self, "kernel_block_size", None)
            if (
                getattr(spec, "compress_ratio", 1) > 1
                and kernel_block_size is not None
                and spec.block_size != kernel_block_size
            ):
                assert spec.block_size % kernel_block_size == 0
                factor = spec.block_size // kernel_block_size
                compressed = (
                    common_attn_metadata.block_table_tensor[:, ::factor]
                    // factor
                )
                buffer = getattr(self, "_glm5_indexer_block_table", None)
                rows, cols = compressed.shape
                if buffer is None:
                    # Keep one base address across B1/B2/... FULL graphs.
                    # Reallocating this tensor when only the request count
                    # changes leaves earlier graphs pointing at freed storage.
                    max_rows = self.vllm_config.scheduler_config.max_num_batched_tokens
                    buffer = compressed.new_empty((max_rows, cols))
                    self._glm5_indexer_block_table = buffer
                elif buffer.shape[1] != cols:
                    raise RuntimeError(
                        "GLM5-Next indexer block-table width changed after "
                        f"initialization: {buffer.shape[1]} -> {cols}"
                    )
                if rows > buffer.shape[0]:
                    raise RuntimeError(
                        "GLM5-Next indexer block-table rows exceed the stable "
                        f"buffer: {rows} > {buffer.shape[0]}"
                    )
                translated_table = buffer[:rows, :cols]
                translated_table.copy_(compressed)
                common_attn_metadata = common_attn_metadata.replace(
                    block_table_tensor=translated_table
                )
            return original_indexer_build(
                self,
                common_prefix_len,
                common_attn_metadata,
                fast_build,
            )

        build_indexer_metadata._glm5_kpool = True
        builder_cls.build = build_indexer_metadata

    # Reference KVBlockZeroer semantics: compressed index pages never share
    # storage with KDA state and are overwritten before their first read, so
    # exclude them from the page-uniform zeroing pass.  Stock 0.24 includes
    # them and asserts because their physical page is intentionally smaller
    # than an MLA page.
    zeroer_cls = worker_utils.KVBlockZeroer
    original_zeroer_init = zeroer_cls.__init__
    if not getattr(original_zeroer_init, "_glm5_kpool", False):

        def init_zeroer(
            self,
            device,
            pin_memory,
            attn_groups_iter,
            kernel_block_sizes,
            cache_dtype,
            static_forward_context,
            runner_only_attn_layers=None,
        ):
            groups = [
                group
                for group in attn_groups_iter
                if not (
                    isinstance(group.kv_cache_spec, AttentionSpec)
                    and group.kv_cache_spec.storage_block_size
                    != group.kv_cache_spec.block_size
                )
            ]
            return original_zeroer_init(
                self,
                device,
                pin_memory,
                groups,
                kernel_block_sizes,
                cache_dtype,
                static_forward_context,
                runner_only_attn_layers,
            )

        init_zeroer._glm5_kpool = True
        zeroer_cls.__init__ = init_zeroer

    # Register only after the built-ins; registering first would make v0.24's
    # lazy registry incorrectly believe initialization was already complete.
    original_register_all = single_type_kv_cache_manager.register_all_kvcache_specs
    if not getattr(original_register_all, "_glm5_kpool", False):

        @wraps(original_register_all)
        def register_all(vllm_config):
            original_register_all(vllm_config)
            kv_cache_spec_registry.KVCacheSpecRegistry.register(
                KpoolTailSpec,
                KpoolTailManager,
                uniform_type_base_spec=KpoolTailSpec,
            )

        register_all._glm5_kpool = True
        single_type_kv_cache_manager.register_all_kvcache_specs = register_all

    # Plugin loading can happen after another component has forced lazy
    # registration. In that case add only our spec immediately.
    if kv_cache_spec_registry._REGISTRY_KVCACHESPEC_LIST:
        kv_cache_spec_registry.KVCacheSpecRegistry.register(
            KpoolTailSpec,
            KpoolTailManager,
            uniform_type_base_spec=KpoolTailSpec,
        )

    # Plugin activation can occur after vLLM has imported the manager factory
    # into kv_cache_coordinator, and on that path the lazy registry may already
    # have resolved KpoolTailSpec through its SlidingWindowSpec base class.
    # Explicitly route the exact tail spec here.  The native sitecustomize path
    # registers before that import; this OOT compatibility hook makes the two
    # activation orders semantically identical without editing vLLM.
    original_get_manager = (
        single_type_kv_cache_manager.get_manager_for_kv_cache_spec
    )
    if not getattr(original_get_manager, "_glm5_kpool", False):

        @wraps(original_get_manager)
        def get_manager_for_kv_cache_spec(
            kv_cache_spec,
            max_num_batched_tokens,
            max_model_len,
            **kwargs,
        ):
            if type(kv_cache_spec) is KpoolTailSpec:
                return KpoolTailManager(kv_cache_spec, **kwargs)
            return original_get_manager(
                kv_cache_spec,
                max_num_batched_tokens,
                max_model_len,
                **kwargs,
            )

        get_manager_for_kv_cache_spec._glm5_kpool = True
        single_type_kv_cache_manager.get_manager_for_kv_cache_spec = (
            get_manager_for_kv_cache_spec
        )
        kv_cache_coordinator.get_manager_for_kv_cache_spec = (
            get_manager_for_kv_cache_spec
        )

    original_groups = kv_cache_utils.get_kv_cache_groups
    if not getattr(original_groups, "_glm5_kpool", False):

        @wraps(original_groups)
        def get_groups(vllm_config, kv_cache_spec):
            if any(isinstance(spec, KpoolTailSpec) for spec in kv_cache_spec.values()):
                return _group_glm5_kpool(vllm_config, kv_cache_spec)
            return original_groups(vllm_config, kv_cache_spec)

        get_groups._glm5_kpool = True
        kv_cache_utils.get_kv_cache_groups = get_groups

    original_pool_bytes = kv_cache_utils._pool_bytes_per_block
    if not getattr(original_pool_bytes, "_glm5_kpool", False):

        @wraps(original_pool_bytes)
        def pool_bytes(vllm_config, groups):
            if _is_glm5_kpool_groups(groups):
                # Tail pages share their sibling indexer's backing tensor.
                total = 0
                for _name, spec in _inner_specs(groups):
                    if isinstance(spec, KpoolTailSpec):
                        continue
                    total += spec.page_size_bytes
                return total
            return original_pool_bytes(vllm_config, groups)

        pool_bytes._glm5_kpool = True
        kv_cache_utils._pool_bytes_per_block = pool_bytes

    original_max_usage = kv_cache_utils._max_memory_usage_bytes_from_groups
    if not getattr(original_max_usage, "_glm5_kpool", False):

        @wraps(original_max_usage)
        def max_usage(vllm_config, groups):
            if _is_glm5_kpool_groups(groups):
                total = 0
                for _name, spec in _inner_specs(groups):
                    if isinstance(spec, KpoolTailSpec):
                        # One tail page per active request; the scheduler's
                        # max_num_seqs is the exact worst-case request count.
                        total += (
                            spec.page_size_bytes
                            * vllm_config.scheduler_config.max_num_seqs
                        )
                    else:
                        total += spec.max_memory_usage_bytes(vllm_config)
                return total
            return original_max_usage(vllm_config, groups)

        max_usage._glm5_kpool = True
        kv_cache_utils._max_memory_usage_bytes_from_groups = max_usage

    original_config = kv_cache_utils.get_kv_cache_config_from_groups
    if not getattr(original_config, "_glm5_kpool", False):

        @wraps(original_config)
        def cache_config(vllm_config, groups, available_memory):
            if not _is_glm5_kpool_groups(groups):
                return original_config(vllm_config, groups, available_memory)

            per_layer = dict(_inner_specs(groups))
            tail_names = {
                name for name, spec in per_layer.items() if isinstance(spec, KpoolTailSpec)
            }
            index_names = {
                name
                for name, spec in per_layer.items()
                if getattr(spec, "compress_ratio", 1) > 1
            }
            bytes_per_block = sum(
                spec.page_size_bytes
                for name, spec in per_layer.items()
                if name not in tail_names
            )
            num_blocks = kv_cache_utils.may_override_num_blocks(
                vllm_config, available_memory // bytes_per_block
            )

            tensors = []
            consumed_tail = set()
            for name, spec in per_layer.items():
                if name in tail_names:
                    continue
                shared_by = [name]
                if name in index_names:
                    tail_name = name.removesuffix(".k_cache") + ".tail_cache"
                    if tail_name in tail_names:
                        shared_by.append(tail_name)
                        consumed_tail.add(tail_name)
                tensors.append(
                    KVCacheTensor(
                        size=spec.page_size_bytes * num_blocks,
                        shared_by=shared_by,
                    )
                )
            assert consumed_tail == tail_names
            return KVCacheConfig(
                num_blocks=num_blocks,
                kv_cache_tensors=tensors,
                kv_cache_groups=groups,
            )

        cache_config._glm5_kpool = True
        kv_cache_utils.get_kv_cache_config_from_groups = cache_config

    original_concurrency = kv_cache_utils.get_max_concurrency_for_kv_cache_config
    if not getattr(original_concurrency, "_glm5_kpool", False):

        @wraps(original_concurrency)
        def concurrency(vllm_config, cache):
            if _is_glm5_kpool_groups(cache.kv_cache_groups):
                max_len = vllm_config.model_config.max_model_len
                main_block = max(
                    spec.block_size
                    for _, spec in _inner_specs(cache.kv_cache_groups)
                    if getattr(spec, "compress_ratio", 1) == 1
                    and not isinstance(spec, KpoolTailSpec)
                )
                return cache.num_blocks / cdiv(max_len, main_block)
            return original_concurrency(vllm_config, cache)

        concurrency._glm5_kpool = True
        kv_cache_utils.get_max_concurrency_for_kv_cache_config = concurrency

    # Opt-in scheduler-capacity diagnostics.  This remains dormant in normal
    # serving and is useful on immutable-vLLM FlagOS images because it reports
    # the plugin-owned cache-manager view without modifying the vLLM package.
    if os.environ.get("VLLM_FL_GLM5_DEBUG_KV_CAPACITY") == "1":
        from vllm.v1.core.kv_cache_manager import KVCacheManager

        original_allocate_slots = KVCacheManager.allocate_slots
        if not getattr(original_allocate_slots, "_glm5_capacity_debug", False):

            @wraps(original_allocate_slots)
            def allocate_slots_with_capacity_debug(
                self, request, num_new_tokens, *args, **kwargs
            ):
                result = original_allocate_slots(
                    self, request, num_new_tokens, *args, **kwargs
                )
                if result is None:
                    count = getattr(self, "_glm5_capacity_debug_count", 0)
                    if count < 8:
                        self._glm5_capacity_debug_count = count + 1
                        probe_tokens = min(
                            request.num_computed_tokens + num_new_tokens,
                            self.max_model_len,
                        )
                        per_manager = []
                        for manager in self.coordinator.single_type_managers:
                            try:
                                required = manager.get_num_blocks_to_allocate(
                                    request.request_id,
                                    probe_tokens,
                                    [],
                                    request.num_computed_tokens,
                                    probe_tokens,
                                )
                            except Exception as exc:  # pragma: no cover - debug only
                                required = f"error:{exc!r}"
                            per_manager.append(
                                {
                                    "manager": type(manager).__name__,
                                    "block_size": manager.block_size,
                                    "required": required,
                                    "held": len(
                                        manager.req_to_blocks.get(
                                            request.request_id, ()
                                        )
                                    ),
                                }
                            )
                        logger.warning(
                            "GLM5 KV capacity rejection: request=%s "
                            "prompt_tokens=%d computed=%d new=%d free=%d/%d "
                            "managers=%s",
                            request.request_id,
                            request.num_tokens,
                            request.num_computed_tokens,
                            num_new_tokens,
                            self.block_pool.get_num_free_blocks(),
                            self.block_pool.num_gpu_blocks,
                            per_manager,
                        )
                return result

            allocate_slots_with_capacity_debug._glm5_capacity_debug = True
            KVCacheManager.allocate_slots = allocate_slots_with_capacity_debug


__all__ = ["install_glm5_next_kpool_v024"]
