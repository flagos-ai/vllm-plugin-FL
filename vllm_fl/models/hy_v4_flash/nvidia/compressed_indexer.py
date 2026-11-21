# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Flash coarse-only gated IndexPool with bounded paged FP8 scoring."""

import torch
import torch.nn.functional as F
from torch import nn

from vllm import envs
from vllm.config import CUDAGraphMode
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import (
    fp8_fp4_paged_mqa_logits,
    get_paged_mqa_logits_metadata,
    has_deep_gemm,
)
from vllm.utils.math_utils import cdiv
from vllm.utils.platform_utils import num_compute_units

from .indexpool_cache import (
    POOL_PAGE_SIZE,
    IndexPoolCache,
    IndexPoolMetadata,
    tail_ring_size,
)
from .indexpool_metadata import expand_groups, prepare_indexpool_metadata
from .indexpool_workspace import profile_peak_bytes, scoring_chunk_rows
from vllm_fl.ops.hy4_flash_indexpool import (
    fwht_quantize,
    gated_pool,
    gather_groups,
    rotate_indexer,
    select_groups,
    write_pool,
    write_tails,
)


class FlashCompressedIndexer(nn.Module):
    def __init__(self, vllm_config, config, topk_indices_buffer, prefix):
        super().__init__()
        if not current_platform.is_device_capability_family(90):
            raise NotImplementedError(
                "This Flash plugin backport supports H100 SM90 FP8 scales only"
            )
        if not has_deep_gemm():
            raise RuntimeError("Flash IndexPool requires DeepGEMM paged MQA")
        self.ratio = config.compressed_indexer_compress_ratio
        self.group_budget = config.compressed_indexer_group_topk - 1
        self.num_heads = config.compressed_indexer_num_attention_heads
        if not 1 <= self.num_heads <= 64:
            raise ValueError("IndexPool requires 1..64 scoring heads")
        self.rope_dim = config.qk_rope_head_dim
        self.rope_parameters = config.rope_parameters
        rotary_inv_freq = None
        if self.rope_dim:
            theta = self.rope_parameters.get("rope_theta", 10000.0)
            steps = torch.arange(
                0,
                self.rope_dim,
                2,
                device=vllm_config.device_config.device,
                dtype=torch.float32,
            )
            # Match the checkpoint indexer's FP32 coefficient initialization.
            rotary_inv_freq = theta ** (-steps / self.rope_dim)
        self.register_buffer("rotary_inv_freq", rotary_inv_freq, persistent=False)
        self.eps = config.rms_norm_eps
        self.topk_indices_buffer = topk_indices_buffer
        self.max_model_len = vllm_config.model_config.max_model_len
        self.logits_budget = envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB * 1024 * 1024 // 4
        for name, input_size, output_size in (
            ("query_projection", config.q_lora_rank, self.num_heads * 128),
            ("head_weight_projection", config.hidden_size, self.num_heads),
            ("token_key_projection", config.hidden_size, 128),
        ):
            setattr(
                self,
                name,
                ReplicatedLinear(
                    input_size,
                    output_size,
                    bias=False,
                    quant_config=vllm_config.quant_config,
                    prefix=f"{prefix}.{name}",
                ),
            )
        self.token_key_norm = nn.RMSNorm(128, eps=self.eps)
        self.key_value_projection = nn.Linear(128, 128, bias=False, dtype=torch.float32)
        self.key_gate_projection = nn.Linear(128, 128, bias=False, dtype=torch.float32)
        self.key_norm = nn.RMSNorm(128, eps=self.eps, dtype=torch.float32)
        self.intra_group_ape = nn.Parameter(
            torch.empty(self.ratio, 128, dtype=torch.float32)
        )
        self.k_cache = IndexPoolCache(
            f"{prefix}.k_cache", vllm_config.cache_config, self.ratio
        )
        self.tail_cache = IndexPoolCache(
            f"{prefix}.tail_cache",
            vllm_config.cache_config,
            self.ratio,
            tail=True,
        )
        self.num_sms = num_compute_units()
        self.max_batch_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        self.max_batch_requests = vllm_config.scheduler_config.max_num_seqs
        self.tail_ring = tail_ring_size(self.ratio, vllm_config.num_speculative_tokens)
        self.max_scoring_width = self._scoring_width(self.max_model_len)
        if not 1 <= self.group_budget <= 2048:
            raise ValueError("IndexPool requires a selection budget of 1..2048")
        # Reject a globally invalid setting before any cache can be written.
        scoring_chunk_rows(self.logits_budget, self.max_scoring_width)
        self.varlen = current_platform.is_device_capability_family(100)
        self.profile_workspace_bytes = profile_peak_bytes(
            tokens=self.max_batch_tokens,
            batch=self.max_batch_requests,
            ratio=self.ratio,
            tail_ring=self.tail_ring,
            heads=self.num_heads,
            budget=self.group_budget,
            max_width=self.max_scoring_width,
            page_size=POOL_PAGE_SIZE,
            logits_elems=self.logits_budget,
            num_sms=self.num_sms,
        )

    def _scoring_width(self, max_seq_len):
        return max(
            POOL_PAGE_SIZE,
            cdiv(max_seq_len, self.ratio * POOL_PAGE_SIZE) * POOL_PAGE_SIZE,
        )

    def _profile(self, device):
        # V2 profiles with skip_attn=True before binding the KV caches.
        # Account for peak transients, not a permanent per-layer buffer.
        torch.empty(self.profile_workspace_bytes, dtype=torch.uint8, device=device)

    def forward(self, hidden_states, q_c, positions, rope=None):
        output = self.topk_indices_buffer[: hidden_states.shape[0]]
        output.fill_(-1)
        context = get_forward_context()
        metadata = context.attn_metadata
        if metadata is None:
            self._profile(hidden_states.device)
            return output
        if not isinstance(metadata, dict):
            raise TypeError("IndexPool requires per-layer metadata")
        km = metadata[self.k_cache.prefix]
        tm = metadata[self.tail_cache.prefix]
        if not isinstance(km, IndexPoolMetadata) or not isinstance(
            tm, IndexPoolMetadata
        ):
            raise TypeError("IndexPool cache metadata has an incompatible type")
        tokens = km.num_actual_tokens
        if tokens == 0 or km.seq_lens.numel() == 0:
            return output
        width = self._scoring_width(
            self.max_model_len
            if context.cudagraph_runtime_mode == CUDAGraphMode.FULL
            else km.max_seq_len
        )
        if width > self.max_scoring_width:
            raise ValueError("IndexPool metadata exceeds the configured context limit")
        chunk = scoring_chunk_rows(self.logits_budget, width)
        keys = self.token_key_norm(
            self.token_key_projection(hidden_states)[0]
        ).bfloat16()
        cache, tail = self.k_cache.kv_cache, self.tail_cache.kv_cache
        plan = prepare_indexpool_metadata(
            positions,
            tail,
            cache,
            km.block_table,
            tm.block_table,
            km.query_start_loc,
            km.seq_lens,
            tokens,
            self.ratio,
        )
        raw = gather_groups(keys, tail, plan.gather_key_rows, plan.gather_tail_slots)
        if raw.shape[0]:
            raw_fp32 = raw.float()
            values = self.key_value_projection(raw_fp32)
            gates = self.key_gate_projection(raw_fp32)
            del raw_fp32
            pooled = gated_pool(
                values, gates, self.intra_group_ape, self.key_norm.weight, self.eps
            ).bfloat16()
            del values, gates
            quant, scales = fwht_quantize(
                rotate_indexer(
                    pooled,
                    plan.group_positions,
                    self.rope_dim,
                    self.rope_parameters,
                    self.rotary_inv_freq,
                )
            )
            write_pool(cache, plan.pool_slots, quant, scales)
            del pooled, quant, scales
        del raw
        write_tails(keys, positions, tail, plan.tail_key_rows, plan.tail_slots)
        owner, logical, upper, live = plan.owner, plan.logical, plan.upper, plan.live
        del plan, keys
        query = self.query_projection(q_c)[0][:tokens].reshape(
            tokens, self.num_heads, 128
        )
        query = rotate_indexer(
            query,
            positions[..., :tokens],
            self.rope_dim,
            self.rope_parameters,
            self.rotary_inv_freq,
        )
        quant_q, scales_q = fwht_quantize(query.reshape(-1, 128).bfloat16())
        heads = 32 if self.num_heads <= 32 else 64
        quant_q = F.pad(
            quant_q.view(torch.uint8).reshape(tokens, self.num_heads, 128),
            (0, 0, 0, heads - self.num_heads),
        ).view(torch.float8_e4m3fn)
        weights = self.head_weight_projection(hidden_states)[0][:tokens].float()
        weights = weights * (self.num_heads**-0.5 * 128**-0.5)
        weights = F.pad(
            weights * scales_q.reshape(tokens, self.num_heads),
            (0, heads - self.num_heads),
        )
        quant_q = quant_q[:, None]
        # Native non-SM100 semantics return (q_scale, weights) unchanged.
        mx_scales = None
        for begin in range(0, tokens, chunk):
            end = min(tokens, begin + chunk)
            bounds = upper[begin:end]
            lengths = bounds.clamp_min(1).int()[:, None].contiguous()
            indices = owner[begin:end].int().contiguous() if self.varlen else None
            schedule = get_paged_mqa_logits_metadata(
                lengths, POOL_PAGE_SIZE, self.num_sms, indices=indices
            )
            table = km.block_table[owner[begin:end], : cdiv(width, POOL_PAGE_SIZE)]
            logits = fp8_fp4_paged_mqa_logits(
                (
                    quant_q[begin:end],
                    mx_scales[begin:end] if mx_scales is not None else None,
                ),
                cache.unsqueeze(2),
                weights[begin:end],
                lengths,
                table,
                schedule,
                max_model_len=width,
                clean_logits=False,
                indices=indices,
            )
            recalled = select_groups(logits, bounds, self.group_budget)
            del logits
            output[begin:end].copy_(
                expand_groups(
                    recalled,
                    logical[begin:end],
                    self.ratio,
                    active=live[begin:end],
                )
            )
            del recalled, table, schedule, lengths, indices, bounds
        return output
