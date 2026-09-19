# SPDX-License-Identifier: Apache-2.0
"""Plugin-only vLLM 0.24 KV plumbing for GLM5-Next kpool/tail caches."""

from __future__ import annotations

import inspect
import logging
import os
from dataclasses import replace
from functools import wraps

from vllm.utils.math_utils import cdiv

from vllm_fl.activation import PendingPatch, bind_patches
from vllm_fl.models.glm5_next_kpool import (
    glm5_indexer_page_alignment,
    KpoolTailManager,
    KpoolTailSpec,
)

logger = logging.getLogger(__name__)

# Worker-side kpool patches.  These run when the model runner builds attention
# metadata and the KV-block zeroer, i.e. after model construction, so they are
# bound to the GLM5 activation plan instead of plugin registration.  The
# engine-core / config-time hooks below must stay at registration: vLLM needs
# them while resolving ``VllmConfig``, registering the KV-cache spec and
# computing the scheduler's cache config, which can happen before the worker
# (and therefore before any activation) exists.
_RUNTIME_BASELINES: dict[str, object] = {}
_EARLY_PATCHES: list[PendingPatch] | None = None


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
        name: spec
        for name, spec in kv_cache_spec.items()
        if isinstance(spec, MambaSpec)
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
    assert tail_specs and all(
        type(spec) is MLAAttentionSpec for spec in attn_specs.values()
    )
    index_pages = {
        spec.page_size_bytes for spec in attn_specs.values() if spec.compress_ratio > 1
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


def _kpool_target(owner, attr: str) -> str:
    return f"{owner.__module__}.{owner.__name__}.{attr}"


def _capture_runtime_baseline(owner, attr: str) -> None:
    _RUNTIME_BASELINES.setdefault(_kpool_target(owner, attr), getattr(owner, attr))


def _runtime_pristine(owner, attr: str):
    target = _kpool_target(owner, attr)
    value = _RUNTIME_BASELINES.get(target)
    if value is None:  # pragma: no cover - core install captures these
        value = getattr(owner, attr)
        logger.warning(
            "No import-time kpool runtime baseline for %s; capturing at "
            "activation time",
            target,
        )
    return value


def _runtime_patch(
    owner,
    attr: str,
    value,
    fingerprint: str,
    params: tuple[str, ...],
) -> PendingPatch:
    return PendingPatch(
        target=_kpool_target(owner, attr),
        owner=owner,
        attr=attr,
        replacement=value,
        fingerprint=fingerprint,
        pristine=_runtime_pristine(owner, attr),
        expected_params=params,
    )


def _create_metadata_builders_patch(fingerprint: str) -> PendingPatch:
    """Keep the compressed indexer at pool-page granularity in metadata."""
    from vllm_fl.runtime.kv_layout import get_physical_cache_layout
    from vllm.v1.worker import utils as worker_utils

    owner = worker_utils.AttentionGroup
    pristine = _runtime_pristine(owner, "create_metadata_builders")

    def create_metadata_builders(
        self,
        vllm_config,
        device,
        kernel_block_size=None,
        num_metadata_builders=1,
    ):
        spec = self.kv_cache_spec
        layout = get_physical_cache_layout(self.backend, spec)
        if layout is None:
            return pristine(
                self,
                vllm_config,
                device,
                kernel_block_size,
                num_metadata_builders,
            )

        builder_spec = spec.copy_with_new_block_size(layout.metadata_block_size)
        self.metadata_builders = [
            self.backend.get_builder_cls()(
                builder_spec,
                self.layer_names,
                vllm_config,
                device,
            )
            for _ in range(num_metadata_builders)
        ]
        for builder in self.metadata_builders:
            builder._glm5_physical_layout = layout
            if kernel_block_size is not None:
                builder.kernel_block_size = kernel_block_size

    return _runtime_patch(
        owner,
        "create_metadata_builders",
        create_metadata_builders,
        fingerprint,
        ("self", "vllm_config", "device", "kernel_block_size", "num_metadata_builders"),
    )


def _indexer_build_patch(fingerprint: str) -> PendingPatch:
    """Translate the shared block table before every indexer build."""
    from vllm.v1.attention.backends.mla import indexer as indexer_backend

    owner = indexer_backend.DeepseekV32IndexerMetadataBuilder
    pristine = _runtime_pristine(owner, "build")

    def build(
        self,
        common_prefix_len,
        common_attn_metadata,
        fast_build=False,
    ):
        spec = self.kv_cache_spec
        kernel_block_size = getattr(self, "kernel_block_size", None)
        if (
            getattr(self, "_glm5_physical_layout", None) is not None
            and kernel_block_size is not None
            and spec.block_size != kernel_block_size
        ):
            assert spec.block_size % kernel_block_size == 0
            factor = spec.block_size // kernel_block_size
            compressed = common_attn_metadata.block_table_tensor[:, ::factor] // factor
            buffer = getattr(self, "_glm5_indexer_block_table", None)
            rows, cols = compressed.shape
            if buffer is None:
                # Keep one base address across B1/B2/... FULL graphs.
                # Reallocating this tensor when only the request count changes
                # leaves earlier graphs pointing at freed storage.
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
        return pristine(
            self,
            common_prefix_len,
            common_attn_metadata,
            fast_build,
        )

    return _runtime_patch(
        owner,
        "build",
        build,
        fingerprint,
        ("self", "common_prefix_len", "common_attn_metadata", "fast_build"),
    )


def _zeroer_init_patch(fingerprint: str) -> PendingPatch:
    """Exclude compressed index pages from the page-uniform zeroing pass."""
    from vllm_fl.runtime.kv_layout import get_physical_cache_layout
    from vllm.v1.worker import utils as worker_utils

    owner = worker_utils.KVBlockZeroer
    pristine = _runtime_pristine(owner, "__init__")

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
                get_physical_cache_layout(group.backend, group.kv_cache_spec)
                is not None
            )
        ]
        return pristine(
            self,
            device,
            pin_memory,
            groups,
            kernel_block_sizes,
            cache_dtype,
            static_forward_context,
            runner_only_attn_layers,
        )

    return _runtime_patch(
        owner,
        "__init__",
        init_zeroer,
        fingerprint,
        (
            "self",
            "device",
            "pin_memory",
            "attn_groups_iter",
            "kernel_block_sizes",
            "cache_dtype",
            "static_forward_context",
            "runner_only_attn_layers",
        ),
    )


def glm5_next_kpool_runtime_patches(fingerprint: str) -> list[PendingPatch]:
    """Worker-side kpool patches, installed by the GLM5 activation plan.

    Keeping these out of plugin registration means a plain ``import vllm_fl``
    no longer rewrites ``AttentionGroup``, the indexer metadata builder or
    ``KVBlockZeroer`` for models that never request the kpool layout.
    """
    return [
        _create_metadata_builders_patch(fingerprint),
        _indexer_build_patch(fingerprint),
        _zeroer_init_patch(fingerprint),
    ]


def install_glm5_next_kpool_v024() -> None:
    """Register scoped engine hooks as one owned, rollback-safe transaction."""
    global _EARLY_PATCHES
    if _EARLY_PATCHES is not None:
        bind_patches(_EARLY_PATCHES)
        return
    patches = []

    def stage(owner, attr, replacement):
        name = getattr(owner, "__module__", "")
        target = f"{name}.{owner.__name__}.{attr}".lstrip(".")
        patches.append(
            PendingPatch(
                target=target,
                owner=owner,
                attr=attr,
                replacement=replacement,
                pristine=inspect.getattr_static(owner, attr),
                get_current=lambda: inspect.getattr_static(owner, attr),
                fingerprint="glm5.kpool.v024",
                phase="engine/config",
            )
        )

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
            if getattr(
                text_config, "model_type", None
            ) != "glm5_next_text" or not getattr(
                text_config, "index_kpool_compress", False
            ):
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
            compressed_page = glm5_indexer_page_alignment(capability)
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
        stage(Platform, "_align_hybrid_block_size", classmethod(align_hybrid))

    # Capture the pristine worker-side callables before any activation can
    # rebind them.  The patches themselves are bound by the GLM5 plan, so a
    # plain plugin import never rewrites these classes.
    _capture_runtime_baseline(worker_utils.AttentionGroup, "create_metadata_builders")
    _capture_runtime_baseline(
        indexer_backend.DeepseekV32IndexerMetadataBuilder, "build"
    )
    _capture_runtime_baseline(worker_utils.KVBlockZeroer, "__init__")

    # Register only after the built-ins; registering first would make v0.24's
    # lazy registry incorrectly believe initialization was already complete.
    original_register_all = single_type_kv_cache_manager.register_all_kvcache_specs
    if not getattr(original_register_all, "_glm5_kpool", False):

        @wraps(original_register_all)
        def register_all(vllm_config):
            original_register_all(vllm_config)
            kv_cache_spec_registry.KVCacheSpecRegistry.register(
                kvcache_spec_cls=KpoolTailSpec,
                manager_class=KpoolTailManager,
                uniform_type_base_spec=KpoolTailSpec,
            )

        register_all._glm5_kpool = True
        stage(single_type_kv_cache_manager, "register_all_kvcache_specs", register_all)

    # Plugin activation can occur after vLLM has imported the manager factory
    # into kv_cache_coordinator, and on that path the lazy registry may already
    # have resolved KpoolTailSpec through its SlidingWindowSpec base class.
    # Explicitly route the exact tail spec here.  The native sitecustomize path
    # registers before that import; this OOT compatibility hook makes the two
    # activation orders semantically identical without editing vLLM.
    original_get_manager = single_type_kv_cache_manager.get_manager_for_kv_cache_spec
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
        stage(
            single_type_kv_cache_manager,
            "get_manager_for_kv_cache_spec",
            get_manager_for_kv_cache_spec,
        )
        stage(
            kv_cache_coordinator,
            "get_manager_for_kv_cache_spec",
            get_manager_for_kv_cache_spec,
        )

    original_groups = kv_cache_utils.get_kv_cache_groups
    if not getattr(original_groups, "_glm5_kpool", False):

        @wraps(original_groups)
        def get_groups(vllm_config, kv_cache_spec):
            if any(isinstance(spec, KpoolTailSpec) for spec in kv_cache_spec.values()):
                return _group_glm5_kpool(vllm_config, kv_cache_spec)
            return original_groups(vllm_config, kv_cache_spec)

        get_groups._glm5_kpool = True
        stage(kv_cache_utils, "get_kv_cache_groups", get_groups)

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
        stage(kv_cache_utils, "_pool_bytes_per_block", pool_bytes)

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
        stage(kv_cache_utils, "_max_memory_usage_bytes_from_groups", max_usage)

    original_config = kv_cache_utils.get_kv_cache_config_from_groups
    if not getattr(original_config, "_glm5_kpool", False):

        @wraps(original_config)
        def cache_config(vllm_config, groups, available_memory):
            if not _is_glm5_kpool_groups(groups):
                return original_config(vllm_config, groups, available_memory)

            per_layer = dict(_inner_specs(groups))
            tail_names = {
                name
                for name, spec in per_layer.items()
                if isinstance(spec, KpoolTailSpec)
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
        stage(kv_cache_utils, "get_kv_cache_config_from_groups", cache_config)

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
        stage(kv_cache_utils, "get_max_concurrency_for_kv_cache_config", concurrency)

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
                if result is None and any(
                    isinstance(manager, KpoolTailManager)
                    for manager in self.coordinator.single_type_managers
                ):
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
            stage(KVCacheManager, "allocate_slots", allocate_slots_with_capacity_debug)

    from vllm.config.compilation import CompilationConfig

    kpool_op = "vllm::sparse_attn_indexer_kpool"
    if kpool_op not in CompilationConfig._attention_ops:
        stage(
            CompilationConfig,
            "_attention_ops",
            [*CompilationConfig._attention_ops, kpool_op],
        )

    bind_patches(patches)
    _EARLY_PATCHES = patches

    # Plugin loading can happen after another component has forced lazy
    # registration. In that case add only our spec immediately.
    if kv_cache_spec_registry._REGISTRY_KVCACHESPEC_LIST:
        kv_cache_spec_registry.KVCacheSpecRegistry.register(
            kvcache_spec_cls=KpoolTailSpec,
            manager_class=KpoolTailManager,
            uniform_type_base_spec=KpoolTailSpec,
        )


__all__ = ["install_glm5_next_kpool_v024"]
