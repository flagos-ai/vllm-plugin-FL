# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Lightweight stand-ins for the vllm config objects the connector reads.

The connector touches a handful of plain attributes on VllmConfig,
KVCacheConfig, KVTransferConfig and Request. These dataclass-shaped
fakes provide exactly those attributes so scheduler and worker logic
runs on CPU without a model download or a GPU.
"""

from __future__ import annotations

import threading

import torch

from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    MambaSpec,
    SlidingWindowSpec,
)


def event():
    """A fresh threading.Event (kept here to avoid late imports)."""
    return threading.Event()


class ModelInfo:
    """The slice of ModelConfig the connector reads."""

    def __init__(self, use_mla: bool = False, total_num_kv_heads: int = 8):
        self.use_mla = use_mla
        self._total_num_kv_heads = total_num_kv_heads

    def get_total_num_kv_heads(self) -> int:
        return self._total_num_kv_heads


class ParallelInfo:
    """The slice of ParallelConfig the connector reads."""

    def __init__(self, tp_size: int = 1, dp_rank: int = 0):
        self.tensor_parallel_size = tp_size
        self.data_parallel_rank = dp_rank


class TransferInfo:
    """The slice of KVTransferConfig the connector reads."""

    def __init__(
        self,
        kv_role: str = "kv_both",
        engine_id: str = "eng-test",
        extra_config: dict | None = None,
    ):
        self.kv_role = kv_role
        self.engine_id = engine_id
        self.kv_connector_extra_config = extra_config or {}


class CacheInfo:
    """The slice of CacheConfig the connector reads."""

    def __init__(self, block_size: int = 16):
        self.block_size = block_size


class SchedulerInfo:
    """The slice of SchedulerConfig the connector reads."""

    def __init__(self, disable_hybrid_kv_cache_manager: bool = True):
        self.disable_hybrid_kv_cache_manager = disable_hybrid_kv_cache_manager


class CompilationInfo:
    """Holds the static_forward_context used by backend discovery."""

    def __init__(self, forward_context: dict):
        self.static_forward_context = forward_context


class _VllmConfig:
    """Aggregates the per-part config fakes into one object."""

    def __init__(self, parts: dict):
        self.__dict__.update(parts)

    def __repr__(self):
        return f"FakeVllmConfig(parts={sorted(self.__dict__)!r})"


def vllm_config(
    model: ModelInfo | None = "default",
    parallel: ParallelInfo | None = None,
    transfer: TransferInfo | None = None,
    cache: CacheInfo | None = None,
    scheduler: SchedulerInfo | None = None,
    compilation: CompilationInfo | None = None,
) -> _VllmConfig:
    """Build the fake VllmConfig, filling unspecified parts with defaults."""
    return _VllmConfig(
        {
            "model_config": ModelInfo() if model == "default" else model,
            "parallel_config": (parallel if parallel is not None else ParallelInfo()),
            "kv_transfer_config": (
                transfer if transfer is not None else TransferInfo()
            ),
            "cache_config": cache if cache is not None else CacheInfo(),
            "scheduler_config": (
                scheduler if scheduler is not None else SchedulerInfo()
            ),
            "compilation_config": (
                compilation if compilation is not None else CompilationInfo({})
            ),
        }
    )


class FakeKVGroup:
    """One KVCacheGroup: a spec plus its member layer names."""

    def __init__(self, spec, layer_names: list[str]):
        self.kv_cache_spec = spec
        self.layer_names = layer_names


class FakeKVCacheConfig:
    """The slice of KVCacheConfig the connector reads."""

    def __init__(self, groups, has_mamba_layers: bool = False):
        self.kv_cache_groups = list(groups)
        self.has_mamba_layers = has_mamba_layers


def full_attention_group(
    layer: int = 0, block_size: int = 16, num_kv_heads: int = 2
) -> FakeKVGroup:
    """A group of full-attention layers with a real FullAttentionSpec."""
    spec = FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=8,
        dtype=torch.float16,
    )
    return FakeKVGroup(spec, [f"model.layers.{layer}.self_attn"])


def sliding_window_group(
    layer: int = 1,
    block_size: int = 16,
    sliding_window: int = 32,
    num_kv_heads: int = 2,
) -> FakeKVGroup:
    """A group of sliding-window layers with a real SlidingWindowSpec."""
    spec = SlidingWindowSpec(
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=8,
        dtype=torch.float16,
        sliding_window=sliding_window,
    )
    return FakeKVGroup(spec, [f"model.layers.{layer}.self_attn"])


def mamba_group(layer: int = 2, block_size: int = 16) -> FakeKVGroup:
    """A group of mamba/GDN state layers with a real MambaSpec."""
    spec = MambaSpec(block_size=block_size, shapes=((8, 2),), dtypes=(torch.float32,))
    return FakeKVGroup(spec, [f"model.layers.{layer}.mamba"])


class FakeRequest:
    """The slice of vllm Request the scheduler reads/writes."""

    def __init__(
        self,
        request_id: str = "req-1",
        prompt_token_ids: list[int] | None = "default",
        kv_transfer_params: dict | None = None,
        status=None,
        num_prompt_tokens: int | None = None,
        max_tokens: int = 16,
        prompt_embeds=None,
    ):
        from vllm.v1.request import RequestStatus

        self.request_id = request_id
        if prompt_token_ids == "default":
            prompt_token_ids = [1, 2, 3, 4]
        self.prompt_token_ids = prompt_token_ids
        self.prompt_embeds = prompt_embeds
        self.kv_transfer_params = kv_transfer_params
        self.status = status if status is not None else RequestStatus.RUNNING
        self.num_prompt_tokens = (
            num_prompt_tokens
            if num_prompt_tokens is not None
            else len(self.prompt_token_ids)
        )
        self.max_tokens = max_tokens
        if self.prompt_token_ids is None:
            # Mirrors vllm Request: placeholders sized by the prompt embeds.
            self._all_token_ids = [0] * self.num_prompt_tokens
        else:
            self._all_token_ids = list(self.prompt_token_ids)


class FakeBlocks:
    """Stands in for KVCacheBlocks; returns per-group unhashed ids."""

    def __init__(self, unhashed_all_groups):
        self._unhashed = unhashed_all_groups

    def get_unhashed_block_ids_all_groups(self):
        return [list(g) for g in self._unhashed]


class FakeAttnLayer(AttentionLayerBase):
    """A static_forward_context entry pointing at one backend class.

    Subclasses AttentionLayerBase so vllm's layer discovery accepts it
    and never falls back to platform-specific backend selection.
    """

    def __init__(self, backend_cls):
        self._backend_cls = backend_cls

    def get_attn_backend(self):
        return self._backend_cls

    def get_kv_cache_spec(self, vllm_config):
        return None


class FakeAttnBackend(AttentionBackend):
    """A minimal concrete AttentionBackend for CPU runs.

    ``get_supported_kernel_block_sizes`` inherits MultipleOf(1), so any
    logical block size is accepted and stays the kernel block size, and
    ``full_cls_name``/``get_name`` come along from the base class.
    """

    @staticmethod
    def get_name() -> str:
        return "FAKE_ATTN"

    @staticmethod
    def get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size):
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_impl_cls():
        return object

    @staticmethod
    def get_builder_cls():
        return object
