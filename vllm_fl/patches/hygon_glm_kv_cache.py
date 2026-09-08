# Copyright (c) 2026 BAAI. All rights reserved.

"""Hygon compatibility patch for GLM/DSA multi-KV-cache binding.

vLLM allows multiple cache-backed attention modules to share the same
decoder-layer index on CUDA-like, XPU and CPU runners.

GLM DSA has this layout on physical Indexer layers:

    decoder layer
      ├── MLA KV cache
      └── Indexer K cache

Hygon uses the FL GPU runner, but PlatformFL is intentionally registered as
an OOT platform and reports ``is_cuda_alike() == False``. Therefore upstream
``bind_kv_cache()`` rejects this otherwise valid multi-cache layout.

This patch is deliberately narrow:

* it does not modify PlatformFL;
* it does not modify the FL model runner;
* it does not affect non-Hygon platforms;
* it does not affect ordinary Hygon models without an Indexer cache;
* it only bypasses the upstream platform guard for the Indexer multi-cache
  layout.

All other calls are delegated to the original vLLM implementation.
"""

from __future__ import annotations

from collections import defaultdict
from functools import wraps

from vllm.logger import init_logger

logger = init_logger(__name__)


def _group_cache_layers(
    kv_caches,
    num_attn_module: int,
):
    """Group KV-cache layer names by decoder-layer index."""

    from vllm.model_executor.models.utils import (
        extract_layer_index,
    )

    index2name = defaultdict(list)

    for layer_name in kv_caches:
        layer_index = extract_layer_index(
            layer_name,
            num_attn_module,
        )
        index2name[layer_index].append(layer_name)

    return index2name


def _is_hygon_indexer_multi_cache(
    kv_caches,
    num_attn_module: int,
) -> bool:
    """Return whether this is the Hygon DSA Indexer multi-cache case.

    The check intentionally errs on the conservative side. Any layout that
    does not clearly contain an Indexer cache falls back to upstream vLLM.
    """

    from vllm.platforms import current_platform

    if (
        getattr(
            current_platform,
            "vendor_name",
            None,
        )
        != "hygon"
    ):
        return False

    # Ordinary models such as Qwen have no Indexer cache and must keep the
    # exact upstream bind_kv_cache path.
    if not any(
        ".indexer" in layer_name
        for layer_name in kv_caches
    ):
        return False

    index2name = _group_cache_layers(
        kv_caches,
        num_attn_module,
    )

    duplicate_groups = [
        layer_names
        for layer_names in index2name.values()
        if len(layer_names) > 1
    ]

    if not duplicate_groups:
        return False

    # Only accept the expected DSA layout:
    # one ordinary MLA cache plus one Indexer cache under the same
    # decoder-layer index.
    #
    # Unknown multi-cache layouts deliberately fall back to upstream vLLM.
    for layer_names in duplicate_groups:
        if len(layer_names) != 2:
            return False

        indexer_count = sum(
            ".indexer" in layer_name
            for layer_name in layer_names
        )

        if indexer_count != 1:
            return False

    return True


def _bind_hygon_indexer_kv_cache(
    kv_caches,
    forward_context,
    runner_kv_caches,
    num_attn_module: int,
) -> None:
    """Bind the known-safe Hygon DSA multi-cache layout.

    The implementation below intentionally mirrors vLLM 0.20.0
    ``bind_kv_cache()``. The only omitted part is its platform guard.
    """

    assert len(runner_kv_caches) == 0

    index2name = _group_cache_layers(
        kv_caches,
        num_attn_module,
    )

    # Preserve vLLM's layer-index ordering.
    for layer_index in sorted(index2name.keys()):
        layer_names = index2name[layer_index]

        for layer_name in layer_names:
            runner_kv_caches.append(
                kv_caches[layer_name]
            )

    # Bind each allocated cache back to its Attention object.
    for layer_name, kv_cache in kv_caches.items():
        forward_context[layer_name].kv_cache = kv_cache


def apply_hygon_glm_kv_cache_patch() -> None:
    """Install the Hygon GLM/DSA KV-cache compatibility patch.

    This function must run before ``vllm_fl.worker.model_runner`` imports
    ``bind_kv_cache`` from ``vllm.v1.worker.utils``.

    The patch is idempotent.
    """

    from vllm.v1.worker import utils as worker_utils

    current_bind = worker_utils.bind_kv_cache

    if getattr(
        current_bind,
        "_fl_hygon_glm_kv_cache_patched",
        False,
    ):
        return

    original_bind = current_bind

    @wraps(original_bind)
    def patched_bind_kv_cache(
        kv_caches,
        forward_context,
        runner_kv_caches,
        num_attn_module=1,
    ):
        # Every non-target case executes the original vLLM function.
        if not _is_hygon_indexer_multi_cache(
            kv_caches,
            num_attn_module,
        ):
            return original_bind(
                kv_caches,
                forward_context,
                runner_kv_caches,
                num_attn_module,
            )

        logger.info_once(
            "Using Hygon DSA multi-KV-cache binding compatibility path."
        )

        return _bind_hygon_indexer_kv_cache(
            kv_caches,
            forward_context,
            runner_kv_caches,
            num_attn_module,
        )

    patched_bind_kv_cache._fl_hygon_glm_kv_cache_patched = True
    patched_bind_kv_cache._fl_original_bind_kv_cache = original_bind

    worker_utils.bind_kv_cache = patched_bind_kv_cache

    logger.info(
        "Applied Hygon GLM/DSA KV-cache binding compatibility patch."
    )