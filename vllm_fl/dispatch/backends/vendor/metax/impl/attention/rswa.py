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

"""Keep native KV contracts and route safe causal/prefix-window request runs."""

from dataclasses import dataclass, fields

import torch
import triton
import triton.language as tl

import vllm.envs as envs
from vllm.config import CUDAGraphMode
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


def vendor_causal_available():
    try:
        from flaggems_vllm import is_paged_causal_attention_available
    except ImportError:
        return False
    return is_paged_causal_attention_available()


@triton.jit
def _disjoint_prefix_counts(PREFIX, LENGTHS, OUT, N: tl.constexpr,
                            WINDOW: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    prefix = tl.load(PREFIX + row, row < N, 0)
    length = tl.load(LENGTHS + row, row < N, 0)
    cut = tl.minimum(prefix, tl.maximum(length - WINDOW, 0))
    tl.store(OUT + row, cut, row < N)


@triton.jit
def _normalize_query_starts(STARTS, OUT, ROW: tl.constexpr,
                            TOKEN: tl.constexpr, N: tl.constexpr,
                            BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(STARTS + ROW + index, index < N, 0) - TOKEN
    tl.store(OUT + index, value, index < N)


@dataclass(frozen=True)
class RSWAQueryRun:
    kind: str
    token_start: int
    token_end: int
    metadata: TritonAttentionMetadata
    prefix_counts: torch.Tensor


@dataclass
class MetaXRSWAMetadata(TritonAttentionMetadata):
    disjoint_prefix_lens: torch.Tensor | None = None
    query_runs: tuple[RSWAQueryRun, ...] = ()


class MetaXRSWAMetadataBuilder(TritonAttentionMetadataBuilder):
    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        # Request metadata is owned and staged by the framework, once per
        # batch. Layers share these stable addresses on the current stream.
        self._prefix_counts = torch.empty(
            vllm_config.scheduler_config.max_num_seqs,
            dtype=torch.int32, device=device,
        )
        self._run_starts = torch.empty(
            2 * vllm_config.scheduler_config.max_num_seqs + 1,
            dtype=torch.int32, device=device,
        )
        mode = vllm_config.compilation_config.cudagraph_mode
        parallel = vllm_config.parallel_config
        # CPU boundaries are not exact for adaptive speculative verification.
        # Full mixed graphs also cannot capture Python request classification.
        self._split_requests = (
            vllm_config.speculative_config is None
            and (mode is None or mode.mixed_mode() != CUDAGraphMode.FULL)
            and getattr(parallel, "decode_context_parallel_size", 1) == 1
            and getattr(parallel, "prefill_context_parallel_size", 1) == 1
        )

    def _build_query_runs(self, base, common, cut):
        if not self._split_requests or base.max_query_len <= 1 or cut is None:
            return ()
        starts = common.query_start_loc_cpu
        lengths = common.seq_lens_cpu_upper_bound
        prefixes = common.rswa_prefix_lens
        if any(t is None or t.device.type != "cpu" for t in (starts, lengths, prefixes)):
            return ()
        count = common.num_reqs
        offsets = starts[: count + 1].tolist()
        upper = lengths[:count].tolist()
        protected = prefixes[:count].tolist()
        if len(offsets) != count + 1 or len(upper) != count or len(protected) != count:
            return ()
        if offsets[0] != 0 or offsets[-1] != base.num_actual_tokens:
            return ()
        kinds = []
        for row in range(count):
            size = offsets[row + 1] - offsets[row]
            if size < 0 or upper[row] < size:
                raise ValueError("Invalid R-SWA query/cache boundaries")
            if size == 0:
                kinds.append("skip")
            elif size == 1:
                kinds.append("decode")
            elif upper[row] <= protected[row]:
                # All keys for these queries are inside the protected prompt,
                # so causal R-SWA is exactly ordinary causal attention.
                kinds.append("prefill")
            else:
                kinds.append("native")
        runs = []
        first = cursor = 0
        base_fields = {field.name: getattr(base, field.name)
                       for field in fields(TritonAttentionMetadata)}
        while first < count:
            end = first + 1
            while end < count and kinds[end] == kinds[first]:
                end += 1
            kind = kinds[first]
            if kind != "skip":
                n = end - first + 1
                cu = self._run_starts[cursor:cursor + n]
                cursor += n
                _normalize_query_starts[(triton.cdiv(n, 128),)](
                    base.query_start_loc, cu, first, offsets[first], n,
                    128, num_warps=4,
                )
                run_fields = dict(base_fields)
                run_fields.update(
                    num_actual_tokens=offsets[end] - offsets[first],
                    max_query_len=max(offsets[i + 1] - offsets[i]
                                      for i in range(first, end)),
                    query_start_loc=cu,
                    max_seq_len=max(upper[first:end]),
                    seq_lens=base.seq_lens[first:end],
                    block_table=base.block_table[first:end],
                    slot_mapping=base.slot_mapping[offsets[first]:offsets[end]],
                    rswa_prefix_lens=base.rswa_prefix_lens[first:end],
                )
                runs.append(RSWAQueryRun(
                    kind, offsets[first], offsets[end],
                    TritonAttentionMetadata(**run_fields), cut[first:end],
                ))
            first = end
        return tuple(runs)

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        if common_attn_metadata.num_reqs > self._prefix_counts.numel():
            raise ValueError("R-SWA metadata exceeds the configured request capacity")
        base = super().build(common_prefix_len, common_attn_metadata, fast_build)
        cut = None
        if base.rswa_prefix_lens is not None:
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
            query_runs=self._build_query_runs(base, common_attn_metadata, cut),
        )


class MetaXRSWAImpl(TritonAttentionImpl):
    def forward(self, layer, query, key, value, kv_cache, attn_metadata, output,
                output_scale=None, output_block_scale=None):
        # Eligibility is determined before either numerical implementation
        # starts. Native Triton retains prefill, masks and unsupported layouts.
        meta = attn_metadata
        if meta is None or (meta.max_query_len == 1 and meta.num_actual_tokens <= 1):
            return super().forward(
                layer, query, key, value, kv_cache, meta, output,
                output_scale, output_block_scale,
            )
        eligible = (
            meta is not None
            and self.attn_type == AttentionType.DECODER
            and getattr(meta, "disjoint_prefix_lens", None) is not None
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

        if meta.max_query_len > 1:
            runs = getattr(meta, "query_runs", ())
            if not runs:
                return super().forward(
                    layer, query, key, value, kv_cache, meta, output,
                    output_scale, output_block_scale,
                )
            import flaggems_vllm

            causal_available = vendor_causal_available()
            cache = kv_cache.transpose(1, 2)
            k_cache, v_cache = cache.split(self.head_size, dim=-1)
            # Decide every run's provider before numerical execution. Runtime
            # failures propagate; no partially written batch is retried.
            providers = [
                "prefill" if run.kind == "prefill" and causal_available else
                "decode" if run.kind == "decode" and
                run.metadata.seq_lens.shape[0] > 1 else "native"
                for run in runs
            ]
            for run, provider in zip(runs, providers):
                start, end = run.token_start, run.token_end
                run_q, run_out = query[start:end], output[start:end]
                data = run.metadata
                if provider == "prefill":
                    flaggems_vllm.paged_causal_attention(
                        run_q, k_cache, v_cache, data.query_start_loc,
                        data.seq_lens, data.block_table, data.max_query_len,
                        data.max_seq_len, self.scale, out=run_out,
                    )
                    logger.info_once("Using FGV/MACA paged causal R-SWA prefill")
                elif provider == "decode":
                    flaggems_vllm.paged_prefix_window_attention(
                        run_q, k_cache, v_cache, data.seq_lens, data.block_table,
                        run.prefix_counts, meta.rswa_window, self.scale, out=run_out,
                    )
                else:
                    super().forward(
                        layer, run_q, key[start:end] if key is not None else None,
                        value[start:end] if value is not None else None,
                        kv_cache, data, run_out, output_scale, output_block_scale,
                    )
            return output

        if meta.num_actual_tokens != meta.seq_lens.shape[0]:
            return super().forward(
                layer, query, key, value, kv_cache, meta, output,
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
