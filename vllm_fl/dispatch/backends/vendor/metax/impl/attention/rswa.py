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

"""Retain Triton KV/prefill contracts and use MACA prefix/window decode."""

from dataclasses import dataclass, fields

import torch
import triton
import triton.language as tl

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends.triton_attn import (
    TritonAttentionBackend,
    TritonAttentionImpl,
    TritonAttentionMetadata,
    TritonAttentionMetadataBuilder,
)
from vllm.v1.kv_cache_interface import KVQuantMode

logger = init_logger(__name__)


def vendor_rswa_available():
    try:
        from flaggems_vllm import is_paged_prefix_window_attention_available
    except ImportError:
        return False
    return is_paged_prefix_window_attention_available()


@triton.jit
def _disjoint_prefix_counts(PREFIX, LENGTHS, OUT, N: tl.constexpr,
                            WINDOW: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    prefix = tl.load(PREFIX + row, row < N, 0)
    length = tl.load(LENGTHS + row, row < N, 0)
    cut = tl.minimum(prefix, tl.maximum(length - WINDOW, 0))
    tl.store(OUT + row, cut, row < N)


@dataclass
class MetaXRSWAMetadata(TritonAttentionMetadata):
    disjoint_prefix_lens: torch.Tensor | None = None


class MetaXRSWAMetadataBuilder(TritonAttentionMetadataBuilder):
    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        # Request metadata is owned and staged by the framework, once per
        # batch. Layers share these stable addresses on the current stream.
        self._prefix_counts = torch.empty(
            vllm_config.scheduler_config.max_num_seqs,
            dtype=torch.int32, device=device,
        )

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        if common_attn_metadata.num_reqs > self._prefix_counts.numel():
            raise ValueError("R-SWA metadata exceeds the configured request capacity")
        base = super().build(common_prefix_len, common_attn_metadata, fast_build)
        cut = None
        if base.max_query_len == 1 and base.rswa_prefix_lens is not None:
            n = base.seq_lens.shape[0]
            if n > self._prefix_counts.numel():
                raise ValueError("R-SWA metadata exceeds the configured request capacity")
            cut = self._prefix_counts[:n]
            if n:
                _disjoint_prefix_counts[(triton.cdiv(n, 128),)](
                    base.rswa_prefix_lens, base.seq_lens, cut, n, base.rswa_window,
                    128, num_warps=4,
                )
        return MetaXRSWAMetadata(
            **{field.name: getattr(base, field.name)
               for field in fields(TritonAttentionMetadata)},
            disjoint_prefix_lens=cut,
        )


class MetaXRSWAImpl(TritonAttentionImpl):
    def forward(self, layer, query, key, value, kv_cache, attn_metadata, output,
                output_scale=None, output_block_scale=None):
        # Eligibility is determined before either numerical implementation
        # starts. Native Triton retains prefill, masks and unsupported layouts.
        meta = attn_metadata
        eligible = (
            meta is not None
            and self.attn_type == AttentionType.DECODER
            and meta.max_query_len == 1
            # Eager B1 is dominated by the two-call SDK wrapper. Keep the
            # native segmented decode path for that measured regime.
            and meta.num_actual_tokens > 1
            and getattr(meta, "disjoint_prefix_lens", None) is not None
            and meta.num_actual_tokens == meta.seq_lens.shape[0]
            and meta.causal is True
            and meta.mm_prefix_range_tensor is None
            and self.head_size == 128
            and query.dtype in (torch.float16, torch.bfloat16)
            and kv_cache.dtype == query.dtype
            and output.dtype == query.dtype
            and self._kv_quant_mode == KVQuantMode.NONE
            and self.alibi_slopes is None
            and not self.use_alibi_sqrt
            and self.logits_soft_cap == 0
            and self.sinks is None
            and self.chunk_lookback == -1
            and not envs.VLLM_BATCH_INVARIANT
            and self.sliding_window in ((-1, -1), (meta.rswa_window - 1, 0))
            and not getattr(layer, "mm_prefix_clamp_sliding_window", False)
            and output_scale is None
            and output_block_scale is None
        )
        if not eligible:
            return super().forward(
                layer, query, key, value, kv_cache, attn_metadata, output,
                output_scale, output_block_scale,
            )

        from flaggems_vllm import paged_prefix_window_attention

        n = meta.num_actual_tokens
        q = query if query.shape[0] == n else query[:n]
        out = output if output.shape[0] == n else output[:n]
        cache = kv_cache.transpose(1, 2)
        k_cache, v_cache = cache.split(self.head_size, dim=-1)
        paged_prefix_window_attention(
            q, k_cache, v_cache, meta.seq_lens, meta.block_table,
            meta.disjoint_prefix_lens, meta.rswa_window, self.scale, out=out,
        )
        logger.info_once("Using MACA prefix/window decode through FlagGems-vllm")
        return output


class MetaXRSWABackend(TritonAttentionBackend):
    # Keep the native name, layout and KV lifecycle. The selected class path
    # and kernel trace distinguish the vendor decode implementation.
    @staticmethod
    def get_impl_cls():
        return MetaXRSWAImpl

    @staticmethod
    def get_builder_cls():
        return MetaXRSWAMetadataBuilder
