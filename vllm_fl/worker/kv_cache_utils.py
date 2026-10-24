# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import defaultdict
from typing import Any

import torch

from vllm.model_executor.models.utils import extract_layer_index
from vllm.platforms import current_platform
from vllm.v1.worker.utils import bind_kv_cache as upstream_bind_kv_cache


def bind_kv_cache(
    kv_caches: dict[str, torch.Tensor],
    forward_context: dict[str, Any],
    runner_kv_caches: list[torch.Tensor],
    num_attn_module: int = 1,
) -> None:
    """Bind all attention caches on the Iluvatar GPU runner.

    vLLM 0.24 permits multiple caches per decoder layer only for CUDA-like,
    XPU and CPU platforms. Iluvatar uses the same runner cache representation
    but deliberately reports is_cuda_alike() as False. Handle its binding
    here without enabling unrelated CUDA kernels or allocator capabilities.
    """
    if not (
        getattr(current_platform, "vendor_name", None) == "iluvatar"
        and current_platform.device_type == "cuda"
    ):
        upstream_bind_kv_cache(
            kv_caches, forward_context, runner_kv_caches, num_attn_module
        )
        return

    assert len(runner_kv_caches) == 0
    index_to_names: dict[int, list[str]] = defaultdict(list)
    for layer_name in kv_caches:
        index_to_names[extract_layer_index(layer_name, num_attn_module)].append(
            layer_name
        )

    # Match the GPU runner's order, including every cache in a layer. Keep
    # references to the original tensors: graph capture and shared KV layers
    # depend on their identity, storage and layout remaining unchanged.
    for layer_index in sorted(index_to_names):
        for layer_name in index_to_names[layer_index]:
            runner_kv_caches.append(kv_caches[layer_name])

    for layer_name, kv_cache in kv_caches.items():
        forward_context[layer_name].kv_cache = kv_cache
