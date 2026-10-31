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

"""Restore worker helpers that vLLM 0.32 removed.

The FL worker and model runner are adapted from vLLM v0.28's GPU worker and
call a few helpers that the 0.32 release dropped during its worker refactor.
Rather than fork those call sites, re-publish the helpers on the upstream
modules where the worker already looks for them, so a single source tree runs
against both targets.

Every patch prefers whatever the installed vLLM provides and only installs a
v0.28-equivalent when the name is genuinely absent, so a build that still ships
the original keeps using it.

Patched names:

``vllm.v1.worker.utils.is_residual_scattered_for_sp``
    Sequence-parallel residual predicate; the logic moved into
    ``forward_context``/model code upstream. Every ``vllm_config`` field it
    reads still exists in 0.32.

``vllm.model_executor.layers.rotary_embedding.XDRotaryEmbedding``
``vllm.model_executor.models.interfaces.SupportsXDRoPE``
``vllm.model_executor.models.interfaces.supports_xdrope``
    The XDRoPE path was folded into MRoPE (``xdrope_section`` is now a legacy
    alias of ``mrope_section``). MRoPE is *not* a drop-in: XDRoPE's position
    helper takes no ``mrope_position_delta``. ``supports_xdrope`` becomes a real
    capability probe, so it correctly reports False for 0.32 models and the
    XDRoPE branch is simply never taken.

``vllm.v1.worker.gpu.attn_utils._reshape_attention_kv_cache``
    Private helper the model runner uses to build KV-cache views. Its
    ``page_aligned_blocks`` branch is what handles KV-first
    ``(2, num_blocks, ...)`` layouts such as ROCm's.

``vllm.v1.outputs.RoutedExpertsLists``
``vllm.v1.outputs.RoutedExpertsTensors``
    Carriers for the opt-in routed-experts output, redesigned upstream around
    ``vllm.distributed.aux_output_connector``. Only the gated
    ``enable_return_routed_experts`` path uses them.
"""

from __future__ import annotations

from math import prod
from typing import NamedTuple, Protocol, runtime_checkable

import numpy as np
import torch

_installed = False


# ---------------------------------------------------------------------------
# Sequence parallelism
# ---------------------------------------------------------------------------
def _is_residual_scattered_for_sp(vllm_config, num_input_tokens: int) -> bool:
    """Return True when the residual is scattered across TP ranks for SP.

    Port of vLLM v0.28's ``is_residual_scattered_for_sp``. The residual is
    scattered only when sequence parallelism is enabled, TP > 1, and full-graph
    compilation is in effect.
    """
    pass_config = vllm_config.compilation_config.pass_config
    if not getattr(pass_config, "enable_sp", False):
        return False

    tp = vllm_config.parallel_config.tensor_parallel_size
    if tp == 1:
        return False

    compilation_config = vllm_config.compilation_config
    assert (
        getattr(compilation_config, "use_inductor_graph_partition", False)
        or not compilation_config.splitting_ops
    ), "Sequence parallelism requires full-graph compilation"

    # SP always pads num_input_tokens to a multiple of tp earlier.
    assert num_input_tokens % tp == 0

    return True


# ---------------------------------------------------------------------------
# XDRoPE
# ---------------------------------------------------------------------------
@runtime_checkable
class _SupportsXDRoPE(Protocol):
    """Structural interface for models providing XDRoPE input positions."""

    def get_xdrope_input_positions(self, *args, **kwargs): ...


def _supports_xdrope(model: object) -> bool:
    """Return True when *model* actually implements the XDRoPE position hook."""
    return callable(getattr(model, "get_xdrope_input_positions", None))


class _XDRotaryEmbedding:
    """Static XDRoPE position helper ported from vLLM v0.28."""

    @staticmethod
    def get_next_input_positions_tensor(
        out: np.ndarray,
        out_offset: int,
        context_len: int,
        num_new_tokens: int,
    ) -> None:
        """Write the next XDRoPE input positions into ``out`` in place.

        XDRoPE positions advance with the context length directly, without the
        ``mrope_position_delta`` shift that MRoPE applies.
        """
        values = np.arange(
            context_len,
            context_len + num_new_tokens,
            dtype=out.dtype,
        )
        out[:, out_offset : out_offset + num_new_tokens] = values


# ---------------------------------------------------------------------------
# KV cache views
# ---------------------------------------------------------------------------
def _dtype_size(dtype: torch.dtype) -> int:
    return torch.tensor([], dtype=dtype).element_size()


def _reshape_attention_kv_cache(
    kv_raw_tensor: torch.Tensor,
    kv_cache_spec,
    kv_cache_shape: tuple[int, ...],
    kv_cache_stride_order: tuple[int, ...],
    num_blocks: int,
    packing: tuple[int, int] | None,
    page_aligned_blocks: bool = False,
) -> torch.Tensor:
    """Build the typed KV-cache view over a raw byte tensor (vLLM v0.28 port)."""
    permuted_kv_cache_shape = tuple(kv_cache_shape[i] for i in kv_cache_stride_order)
    inv_order = [
        kv_cache_stride_order.index(i) for i in range(len(kv_cache_stride_order))
    ]
    dtype = kv_cache_spec.dtype

    if packing is not None:
        offset, block_stride = packing
        assert inv_order[0] == 0
        page_bytes = prod(kv_cache_shape[1:]) * _dtype_size(dtype)
        kv_cache = (
            kv_raw_tensor.view(-1, block_stride)[:, offset : offset + page_bytes]
            .view(dtype)
            .view(permuted_kv_cache_shape)
        )
    elif kv_cache_spec.page_size_padded is not None:
        # Use a strided view to skip the padding between physical pages.
        #
        # Only num-blocks-first layouts are supported (the block dimension is
        # dim 0 of the unpermuted shape). kv-first layouts such as ROCm's
        # ``(2, num_blocks, ...)`` are intentionally not supported here. For a
        # num-blocks-first layout the only stride that must change is the block
        # stride: every other (contiguous) stride already steps within the
        # unpadded region of a page, so no further adjustment is needed.
        assert kv_cache_shape[0] == num_blocks, (
            "Padded KV pages require a num-blocks-first KV cache layout (got "
            f"shape {kv_cache_shape} with num_blocks={num_blocks}); "
            "kv-first layouts are not supported."
        )
        dtype_size = _dtype_size(kv_cache_spec.dtype)
        page_stride = kv_cache_spec.page_size_bytes // dtype_size

        num_blocks_dim = inv_order[0]
        strides = list(torch.empty(permuted_kv_cache_shape, device="meta").stride())
        strides[num_blocks_dim] = page_stride

        kv_cache = torch.as_strided(
            kv_raw_tensor.view(dtype),
            size=permuted_kv_cache_shape,
            stride=tuple(strides),
        )
    elif page_aligned_blocks:
        # A KV-first layout such as ROCm's ``(2, num_blocks, ...)`` puts block
        # ``b``'s K and V in two far-apart halves of the allocation, so block
        # ``b`` does not cover page ``b``. Mamba layers sharing the allocation
        # do address their state by page, so the two would resolve the same
        # bytes. Build the view page-first instead, then swap the dims back.
        assert kv_cache_shape[1] == num_blocks and kv_cache_stride_order == tuple(
            range(len(kv_cache_shape))
        ), (
            "Page-aligned KV blocks expect a default-strided (kv, num_blocks, "
            f"...) layout, got shape {kv_cache_shape} with stride order "
            f"{kv_cache_stride_order} and num_blocks={num_blocks}."
        )
        kv_cache = (
            kv_raw_tensor.view(dtype)
            .view(num_blocks, kv_cache_shape[0], *kv_cache_shape[2:])
            .transpose(0, 1)
        )
    else:
        # No padding — safe to use a contiguous view.
        kv_cache = kv_raw_tensor.view(dtype).view(permuted_kv_cache_shape)

    return kv_cache.permute(*inv_order)


# ---------------------------------------------------------------------------
# Routed experts carriers
# ---------------------------------------------------------------------------
class _RoutedExpertsLists(NamedTuple):
    """CPU-side routed experts, batched per scheduler step."""

    # (num_scheduled_tokens, num_layers, num_experts_per_tok)
    routing_data: np.ndarray
    # (num_scheduled_tokens,)
    slot_mapping: np.ndarray


class _RoutedExpertsTensors(NamedTuple):
    """Device-side snapshot of routed-experts data, pending async D2H."""

    # (num_scheduled_tokens, num_layers, num_experts_per_tok)
    routing_data: torch.Tensor
    # (num_scheduled_tokens,)
    slot_mapping: torch.Tensor

    def to_cpu_nonblocking(self) -> _RoutedExpertsTensors:
        """Issue a non-blocking D2H copy on the current stream."""
        if self.routing_data.device.type == "cpu":
            return self
        return _RoutedExpertsTensors(
            self.routing_data.to("cpu", non_blocking=True),
            self.slot_mapping.to("cpu", non_blocking=True),
        )

    def tolists(self) -> _RoutedExpertsLists:
        """Convert to the numpy-backed form consumed by the scheduler."""
        return _RoutedExpertsLists(
            self.routing_data.cpu().numpy(),
            self.slot_mapping.cpu().numpy(),
        )


def _set_missing(module, name: str, value) -> None:
    """Publish *value* as ``module.name`` only when the name is absent."""
    if not hasattr(module, name):
        setattr(module, name, value)


def apply_worker_symbols_032() -> bool:
    """Re-publish v0.28 worker helpers that vLLM 0.32 removed. Idempotent."""
    global _installed
    if _installed:
        return True

    from vllm.model_executor.layers import rotary_embedding
    from vllm.model_executor.models import interfaces
    from vllm.v1 import outputs
    from vllm.v1.worker import utils as worker_utils
    from vllm.v1.worker.gpu import attn_utils

    _set_missing(
        worker_utils, "is_residual_scattered_for_sp", _is_residual_scattered_for_sp
    )
    _set_missing(rotary_embedding, "XDRotaryEmbedding", _XDRotaryEmbedding)
    _set_missing(interfaces, "SupportsXDRoPE", _SupportsXDRoPE)
    _set_missing(interfaces, "supports_xdrope", _supports_xdrope)
    _set_missing(attn_utils, "_reshape_attention_kv_cache", _reshape_attention_kv_cache)
    _set_missing(outputs, "RoutedExpertsLists", _RoutedExpertsLists)
    _set_missing(outputs, "RoutedExpertsTensors", _RoutedExpertsTensors)

    _installed = True
    return True
