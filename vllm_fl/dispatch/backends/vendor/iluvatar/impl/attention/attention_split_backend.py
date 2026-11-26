# Copyright (c) 2026 BAAI. All rights reserved.
"""Prefill/decode-split attention backend for GQA models on Iluvatar GPUs,
with self-developed I8G16 8-bit group-quantized paged KV cache and native matrix-core decode enabled by default.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import torch

from vllm.v1.attention.backend import AttentionCGSupport, AttentionType
from vllm.v1.attention.backends.triton_attn import (
    TritonAttentionBackend,
    TritonAttentionImpl,
    TritonAttentionMetadata,
    TritonAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.utils import (
    CommonAttentionMetadata,
    split_decodes_and_prefills,
)
from vllm.v1.attention.ops.triton_unified_attention import unified_attention
from vllm.v1.kv_cache_interface import KVQuantMode

from .gqa_prefill_attention import gqa_prefill_attention
from .gqa_decode_attention import (
    gqa_decode_attention,
    pick_num_splits,
    _gqa_decode_combine_kernel,
)
from .kv_i8 import (
    ROW_SLOTS,
    reshape_and_cache_i8,
    dequant_kv_pages_i8,
    dequant_pack_varlen_i8,
)
from .gqa_prefill_i8 import gqa_prefill_attention_i8
from .gqa_decode_i8 import gqa_decode_attention_i8
from .native_decode import gqa_decode_native, get_module as _get_native_decode_module

try:
    from ixformer.contrib.vllm_flash_attn.flash_attn_interface import (
        flash_attn_varlen_func as _ixf_flash_attn_varlen_func,
    )
except Exception:  # noqa: BLE001
    _ixf_flash_attn_varlen_func = None

_DECODE_KERNEL = os.environ.get("VLLM_FL_GQA_DECODE", "packed").lower()
_DECODE_MAX_SPLITS = 8
# Production default: K8V5 Asymmetric Split-Bitplane paged KV cache enabled (2.21x KV capacity + ~2.2x decode bandwidth).
_KV_I8 = os.environ.get("VLLM_FL_KV_I8", "1") == "1"
_I8_HEAD_SIZE = 128
_SCRATCH_MB = int(os.environ.get("VLLM_FL_I8_SCRATCH_MB", "128"))
_scratch = {}


def _get_scratch(device, num_kv_heads, head_dim, block_size):
    """(k_scratch, v_scratch) bf16 [pages, block_size, Hkv, D], allocated once per device."""
    key = (str(device), num_kv_heads, head_dim, block_size)
    if key not in _scratch:
        page_bytes = block_size * num_kv_heads * head_dim * 2
        pages = max(1, (_SCRATCH_MB << 20) // (2 * page_bytes))
        k = torch.empty(
            (pages, block_size, num_kv_heads, head_dim),
            dtype=torch.bfloat16,
            device=device,
        )
        v = torch.empty_like(k)
        _scratch[key] = (k, v)
    return _scratch[key]


def kv_i8_enabled(vllm_config) -> bool:
    if not _KV_I8:
        return False
    try:
        mc = vllm_config.model_config
        cc = vllm_config.cache_config
        if cc.cache_dtype not in ("auto", "bfloat16", "float16"):
            return False
        if mc.get_head_size() != _I8_HEAD_SIZE:
            return False
        if (
            getattr(mc, "get_sliding_window", None) is not None
            and mc.get_sliding_window() is not None
        ):
            return False
        nq = mc.get_num_attention_heads(vllm_config.parallel_config)
        nkv = mc.get_num_kv_heads(vllm_config.parallel_config)
        if nkv <= 0 or nq % nkv != 0 or nq // nkv > 16:
            return False
        if cc.block_size is not None and cc.block_size % 16 != 0:
            return False
    except Exception:  # noqa: BLE001
        return False
    return True


class _DecodeWorkspace:
    """fp32 split-KV partials, allocated once at the maximum size (graph-capture safe)."""

    def __init__(self, max_num_seqs: int, num_heads: int, head_dim: int, device):
        cap = max_num_seqs * num_heads * _DECODE_MAX_SPLITS
        self.acc = torch.zeros((cap, head_dim), dtype=torch.float32, device=device)
        self.m = torch.full((cap,), float("-inf"), dtype=torch.float32, device=device)
        self.l = torch.zeros((cap,), dtype=torch.float32, device=device)

    def get(self, num_seqs, num_heads, num_splits, head_dim, device):
        assert num_seqs * num_heads * num_splits <= self.acc.shape[0]
        return self.acc, self.m, self.l


@dataclass
class SplitAttentionMetadata(TritonAttentionMetadata):
    num_decodes: int = 0
    num_decode_tokens: int = 0
    num_prefills: int = 0
    decode_max_seq_len: int = 0
    decode_num_splits: int = 1
    decode_workspace: object = None
    i8_scratch_sp: torch.Tensor | None = None
    i8_scratch_block_table: torch.Tensor | None = None
    prefill_q_starts: torch.Tensor | None = None
    prefill_kv_starts: torch.Tensor | None = None
    prefill_total_kv: int = 0
    prefill_max_q: int = 0
    prefill_max_kv: int = 0
    prefill_is_initial: bool = False


class SplitAttentionMetadataBuilder(TritonAttentionMetadataBuilder):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH
    reorder_batch_threshold: ClassVar[int] = 1

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        max_num_seqs = max(
            vllm_config.scheduler_config.max_num_seqs,
            vllm_config.compilation_config.max_cudagraph_capture_size or 0,
        )
        head_dim = vllm_config.model_config.get_head_size()
        self.decode_workspace = _DecodeWorkspace(
            max_num_seqs, self.num_heads_q, head_dim, device
        )
        self.kv_i8 = kv_i8_enabled(vllm_config)
        self.i8_block_size = kv_cache_spec.block_size
        self._prefill_ring = 8
        self._prefill_slot = 0
        off_cap = 2 * (max_num_seqs + 1)
        self._prefill_off_cpu = torch.zeros(
            (self._prefill_ring, off_cap), dtype=torch.int32, pin_memory=True
        )
        self._prefill_off_gpu = torch.zeros(
            (self._prefill_ring, off_cap), dtype=torch.int32, device=device
        )
        # Eagerly compile/load the native matrix-core decode extension before graph capture.
        if _DECODE_KERNEL == "packed":
            _get_native_decode_module()
        if self.kv_i8:
            self.i8_scratch = _get_scratch(
                device, self.num_heads_kv, head_dim, self.i8_block_size
            )
            max_scratch_pages = int(self.i8_scratch[0].shape[0])
            max_model_len = vllm_config.model_config.max_model_len
            max_blocks_per_req = (
                max_model_len + self.i8_block_size - 1
            ) // self.i8_block_size
            sbt_cap = max(max_scratch_pages, max_num_seqs * max_blocks_per_req)
            self._i8_slot = 0
            self._i8_ring = 8
            self._i8_sp_cpu = torch.zeros(
                (self._i8_ring, max_scratch_pages, 2), dtype=torch.int32, pin_memory=True
            )
            self._i8_sp_np = [self._i8_sp_cpu[i].numpy() for i in range(self._i8_ring)]
            self._i8_sp_gpu = torch.zeros(
                (self._i8_ring, max_scratch_pages, 2), dtype=torch.int32, device=device
            )
            self._i8_sbt_cpu = torch.zeros(
                (self._i8_ring, sbt_cap), dtype=torch.int32, pin_memory=True
            )
            self._i8_sbt_np = [self._i8_sbt_cpu[i].numpy() for i in range(self._i8_ring)]
            self._i8_sbt_gpu = torch.zeros(
                (self._i8_ring, sbt_cap), dtype=torch.int32, device=device
            )
            self._arange_blocks = np.arange(max_blocks_per_req + 16, dtype=np.int32)
            self._arange_scratch = np.arange(max_scratch_pages + 16, dtype=np.int32)

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> SplitAttentionMetadata:
        base = super().build(common_prefix_len, common_attn_metadata, fast_build)
        num_reqs = common_attn_metadata.num_reqs
        num_decodes, num_prefills, num_decode_tokens, _ = split_decodes_and_prefills(
            common_attn_metadata, decode_threshold=1
        )
        sl_cpu = common_attn_metadata._seq_lens_cpu
        if sl_cpu is None:
            sl_cpu = common_attn_metadata.seq_lens_cpu_upper_bound
        if _DECODE_KERNEL == "packed":
            decode_max_seq_len = common_attn_metadata.max_seq_len if num_decodes > 0 else 0
        else:
            if sl_cpu is None and num_decodes > 0:
                sl_cpu = common_attn_metadata.seq_lens.to("cpu")
            decode_max_seq_len = (
                int(sl_cpu[:num_decodes].max()) if num_decodes > 0 else 0
            )
        md = SplitAttentionMetadata(**base.__dict__)
        md.num_decodes = num_decodes
        md.num_decode_tokens = num_decode_tokens
        md.num_prefills = num_prefills
        md.decode_max_seq_len = decode_max_seq_len
        md.decode_num_splits = pick_num_splits(
            max(num_decodes, 1), self.num_heads_kv, self.device, _DECODE_MAX_SPLITS
        )
        md.decode_workspace = self.decode_workspace
        if num_prefills > 0 and not torch.cuda.is_current_stream_capturing():
            if sl_cpu is None:
                sl_cpu = common_attn_metadata.seq_lens.to("cpu")
            q_start_cpu = getattr(common_attn_metadata, "query_start_loc_cpu", None)
            if q_start_cpu is None:
                q_start_cpu = common_attn_metadata.query_start_loc.to("cpu")
            q_cpu = q_start_cpu[num_decodes : num_reqs + 1]
            pf_q_lens_cpu = q_cpu[1:] - q_cpu[:-1]
            pf_kv_lens_cpu = sl_cpu[num_decodes:num_reqs]
            md.prefill_max_q = int(pf_q_lens_cpu.max())
            md.prefill_max_kv = int(pf_kv_lens_cpu.max())
            pf_initial = bool(torch.equal(pf_q_lens_cpu, pf_kv_lens_cpu))
            md.prefill_is_initial = pf_initial
            if num_decodes == 0:
                q_gpu = common_attn_metadata.query_start_loc[: num_prefills + 1]
                md.prefill_q_starts = q_gpu
                if pf_initial:
                    md.prefill_kv_starts = q_gpu
                    md.prefill_total_kv = int(q_cpu[-1])
                else:
                    pslot = self._prefill_slot
                    self._prefill_slot = (pslot + 1) & 7
                    n_off = num_prefills + 1
                    kv_cpu = self._prefill_off_cpu[pslot, :n_off]
                    kv_cpu[0] = 0
                    torch.cumsum(
                        pf_kv_lens_cpu, dim=0, dtype=torch.int32, out=kv_cpu[1:]
                    )
                    md.prefill_total_kv = int(kv_cpu[-1])
                    kv_gpu = self._prefill_off_gpu[pslot, :n_off]
                    kv_gpu.copy_(kv_cpu, non_blocking=True)
                    md.prefill_kv_starts = kv_gpu
            else:
                pslot = self._prefill_slot
                self._prefill_slot = (pslot + 1) & 7
                n_off = num_prefills + 1
                if pf_initial:
                    q_rel_cpu = self._prefill_off_cpu[pslot, :n_off]
                    torch.sub(q_cpu, int(q_cpu[0]), out=q_rel_cpu)
                    md.prefill_total_kv = int(q_rel_cpu[-1])
                    q_gpu = self._prefill_off_gpu[pslot, :n_off]
                    q_gpu.copy_(q_rel_cpu, non_blocking=True)
                    md.prefill_q_starts = q_gpu
                    md.prefill_kv_starts = q_gpu
                else:
                    both_cpu = self._prefill_off_cpu[pslot, : 2 * n_off]
                    torch.sub(q_cpu, int(q_cpu[0]), out=both_cpu[:n_off])
                    both_cpu[n_off] = 0
                    torch.cumsum(
                        pf_kv_lens_cpu,
                        dim=0,
                        dtype=torch.int32,
                        out=both_cpu[n_off + 1 :],
                    )
                    md.prefill_total_kv = int(both_cpu[-1])
                    both_gpu = self._prefill_off_gpu[pslot, : 2 * n_off]
                    both_gpu.copy_(both_cpu, non_blocking=True)
                    md.prefill_q_starts = both_gpu[:n_off]
                    md.prefill_kv_starts = both_gpu[n_off:]

        if (
            self.kv_i8
            and num_prefills > 0
            and (_ixf_flash_attn_varlen_func is None or md.prefill_q_starts is None)
        ):
            if sl_cpu is None:
                sl_cpu = common_attn_metadata.seq_lens.to("cpu")
            sl_np = sl_cpu[num_decodes:num_reqs].numpy()
            n_pages = (sl_np.astype(np.int64) + self.i8_block_size - 1) // self.i8_block_size
            total = int(n_pages.sum())
            max_scratch_pages = int(self.i8_scratch[0].shape[0])
            if 0 < total <= max_scratch_pages:
                slot = self._i8_slot
                self._i8_slot = (slot + 1) & 7
                S = int(num_prefills)
                max_p = max(int(n_pages.max()), 1)
                sp_np = self._i8_sp_np[slot]
                sbt_flat = self._i8_sbt_np[slot][: S * max_p]
                sbt_flat.fill(0)
                sbt_2d = sbt_flat.reshape(S, max_p)
                cur = 0
                for s in range(S):
                    n = int(n_pages[s])
                    if n > 0:
                        sp_np[cur : cur + n, 0] = s
                        sp_np[cur : cur + n, 1] = self._arange_blocks[:n]
                        sbt_2d[s, :n] = self._arange_scratch[cur : cur + n]
                        cur += n
                sp_gpu = self._i8_sp_gpu[slot, :total]
                sp_gpu.copy_(self._i8_sp_cpu[slot, :total], non_blocking=True)
                sbt_gpu_1d = self._i8_sbt_gpu[slot, : S * max_p]
                sbt_gpu_1d.copy_(self._i8_sbt_cpu[slot, : S * max_p], non_blocking=True)
                md.i8_scratch_sp = sp_gpu
                md.i8_scratch_block_table = sbt_gpu_1d.view(S, max_p)
        return md

    def build_for_cudagraph_capture(self, common_attn_metadata: CommonAttentionMetadata):
        md = self.build(0, common_attn_metadata)
        md.seq_lens.fill_(1)
        return md


class SplitAttentionImpl(TritonAttentionImpl):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kv_i8 = (
            _KV_I8
            and self.attn_type == AttentionType.DECODER
            and self.head_size == _I8_HEAD_SIZE
            and self.sliding_window == (-1, -1)
            and self.alibi_slopes is None
            and self.logits_soft_cap == 0
            and self.sinks is None
            and self._kv_quant_mode == KVQuantMode.NONE
            and self.num_queries_per_kv <= 16
        )

    def _packed_prefill_ok(self, attn_metadata) -> bool:
        return (
            self.attn_type == AttentionType.DECODER
            and attn_metadata.num_prefills > 0
            and not isinstance(attn_metadata.causal, torch.Tensor)
            and attn_metadata.causal
            and self.sliding_window == (-1, -1)
            and self.alibi_slopes is None
            and self.logits_soft_cap == 0
            and self.sinks is None
            and self._kv_quant_mode == KVQuantMode.NONE
            and attn_metadata.mm_prefix_range_tensor is None
            and self.chunk_lookback == -1
            and (self.head_size & (self.head_size - 1)) == 0
        )

    def _packed_decode_ok(self, attn_metadata) -> bool:
        return (
            _DECODE_KERNEL == "packed"
            and self.attn_type == AttentionType.DECODER
            and attn_metadata.num_decodes > 0
            and attn_metadata.num_decode_tokens == attn_metadata.num_decodes
            and not isinstance(attn_metadata.causal, torch.Tensor)
            and attn_metadata.causal
            and self.sliding_window == (-1, -1)
            and self.alibi_slopes is None
            and self.logits_soft_cap == 0
            and self.sinks is None
            and self._kv_quant_mode == KVQuantMode.NONE
            and attn_metadata.mm_prefix_range_tensor is None
            and self.chunk_lookback == -1
            and self.num_queries_per_kv <= 16
            and (self.head_size & (self.head_size - 1)) == 0
        )

    def do_kv_cache_update(self, layer, key, value, kv_cache, slot_mapping):
        if not self.kv_i8 or kv_cache.numel() == 0:
            return super().do_kv_cache_update(layer, key, value, kv_cache, slot_mapping)
        reshape_and_cache_i8(key, value, kv_cache, slot_mapping)

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: SplitAttentionMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if attn_metadata is None:
            return output
        if self.kv_i8:
            return self._forward_i8(query, key, value, kv_cache, attn_metadata, output)
        if output_scale is not None or output_block_scale is not None:
            return super().forward(
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output,
                output_scale,
                output_block_scale,
            )
        packed_prefill = self._packed_prefill_ok(attn_metadata)
        packed_decode = self._packed_decode_ok(attn_metadata)
        if not packed_prefill and not packed_decode:
            return super().forward(
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output,
                output_scale,
                output_block_scale,
            )
        if not packed_prefill and attn_metadata.num_prefills > 0:
            return super().forward(
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output,
                output_scale,
                output_block_scale,
            )

        num_actual_tokens = attn_metadata.num_actual_tokens
        num_decodes = attn_metadata.num_decodes
        num_decode_tokens = attn_metadata.num_decode_tokens
        key_cache, value_cache = kv_cache.unbind(1)

        if num_decodes > 0 and packed_decode:
            done = gqa_decode_native(
                query[:num_decode_tokens],
                key_cache,
                value_cache,
                output[:num_decode_tokens],
                attn_metadata.seq_lens[:num_decodes],
                attn_metadata.block_table[:num_decodes],
                self.scale,
                _gqa_decode_combine_kernel,
                attn_metadata.decode_num_splits,
                attn_metadata.decode_workspace,
                i8=False,
            )
            if not done:
                gqa_decode_attention(
                    query[:num_decode_tokens],
                    key_cache,
                    value_cache,
                    output[:num_decode_tokens],
                    attn_metadata.seq_lens[:num_decodes],
                    attn_metadata.block_table[:num_decodes],
                    self.scale,
                    num_splits=attn_metadata.decode_num_splits,
                    workspace=attn_metadata.decode_workspace,
                )
        elif num_decodes > 0:
            descale_shape = (num_decodes, key_cache.shape[2])
            unified_attention(
                q=query[:num_decode_tokens],
                k=key_cache,
                v=value_cache,
                out=output[:num_decode_tokens],
                cu_seqlens_q=attn_metadata.query_start_loc[: num_decodes + 1],
                max_seqlen_q=1,
                seqused_k=attn_metadata.seq_lens[:num_decodes],
                max_seqlen_k=attn_metadata.decode_max_seq_len,
                softmax_scale=self.scale,
                causal=True,
                window_size=self.sliding_window,
                block_table=attn_metadata.block_table[:num_decodes],
                softcap=self.logits_soft_cap,
                q_descale=None,
                k_descale=layer._k_scale.expand(descale_shape),
                v_descale=layer._v_scale.expand(descale_shape),
                seq_threshold_3D=attn_metadata.seq_threshold_3D,
                num_par_softmax_segments=attn_metadata.num_par_softmax_segments,
                softmax_segm_output=attn_metadata.softmax_segm_output,
                softmax_segm_max=attn_metadata.softmax_segm_max,
                softmax_segm_expsum=attn_metadata.softmax_segm_expsum,
                kv_quant_mode=self._kv_quant_mode,
                use_td=self.use_td,
            )

        if attn_metadata.num_prefills == 0:
            return output
        if (
            _ixf_flash_attn_varlen_func is not None
            and attn_metadata.prefill_is_initial
            and attn_metadata.prefill_q_starts is not None
            and key is not None
            and value is not None
            and query.stride(-1) == 1
            and output.stride(-1) == 1
        ):
            _ixf_flash_attn_varlen_func(
                query[num_decode_tokens:num_actual_tokens],
                key[num_decode_tokens:num_actual_tokens],
                value[num_decode_tokens:num_actual_tokens],
                attn_metadata.prefill_q_starts,
                attn_metadata.prefill_kv_starts,
                attn_metadata.prefill_max_q,
                attn_metadata.prefill_max_kv,
                softmax_scale=self.scale,
                causal=True,
                out=output[num_decode_tokens:num_actual_tokens],
            )
            return output
        gqa_prefill_attention(
            query,
            key_cache,
            value_cache,
            output,
            attn_metadata.query_start_loc[num_decodes:],
            attn_metadata.seq_lens[num_decodes:],
            attn_metadata.block_table[num_decodes:],
            self.scale,
            num_prefill_tokens=num_actual_tokens - num_decode_tokens,
        )
        return output

    def _forward_i8(self, query, key, value, kv_cache, attn_metadata, output):
        if attn_metadata.num_prefills > 0 and not self._packed_prefill_ok(attn_metadata):
            raise RuntimeError("K8V5 KV cache: unsupported prefill attention variant")
        if (
            attn_metadata.num_decodes > 0
            and attn_metadata.num_decode_tokens != attn_metadata.num_decodes
        ):
            raise RuntimeError("K8V5 KV cache: multi-token decode is not supported")
        num_actual_tokens = attn_metadata.num_actual_tokens
        num_decodes = attn_metadata.num_decodes
        num_decode_tokens = attn_metadata.num_decode_tokens
        block_size = kv_cache.shape[2]
        Hkv = kv_cache.shape[3]
        if num_decodes > 0:
            done = gqa_decode_native(
                query[:num_decode_tokens],
                kv_cache,
                kv_cache,
                output[:num_decode_tokens],
                attn_metadata.seq_lens[:num_decodes],
                attn_metadata.block_table[:num_decodes],
                self.scale,
                _gqa_decode_combine_kernel,
                attn_metadata.decode_num_splits,
                attn_metadata.decode_workspace,
                i8=True,
                mode=2,
            )
            if not done:
                gqa_decode_attention_i8(
                    query[:num_decode_tokens],
                    kv_cache,
                    output[:num_decode_tokens],
                    attn_metadata.seq_lens[:num_decodes],
                    attn_metadata.block_table[:num_decodes],
                    self.scale,
                    _gqa_decode_combine_kernel,
                    num_splits=attn_metadata.decode_num_splits,
                    workspace=attn_metadata.decode_workspace,
                )
        if attn_metadata.num_prefills == 0:
            return output
        if (
            _ixf_flash_attn_varlen_func is not None
            and attn_metadata.prefill_q_starts is not None
            and query.stride(-1) == 1
            and output.stride(-1) == 1
        ):
            q_pf = query[num_decode_tokens:num_actual_tokens]
            out_pf = output[num_decode_tokens:num_actual_tokens]
            if (
                attn_metadata.prefill_is_initial
                and key is not None
                and value is not None
                and key.stride(-1) == 1
                and value.stride(-1) == 1
            ):
                _ixf_flash_attn_varlen_func(
                    q_pf,
                    key[num_decode_tokens:num_actual_tokens],
                    value[num_decode_tokens:num_actual_tokens],
                    attn_metadata.prefill_q_starts,
                    attn_metadata.prefill_kv_starts,
                    attn_metadata.prefill_max_q,
                    attn_metadata.prefill_max_kv,
                    softmax_scale=self.scale,
                    causal=True,
                    out=out_pf,
                )
                return output
            tot_kv = attn_metadata.prefill_total_kv
            sk, sv = _get_scratch(query.device, Hkv, self.head_size, block_size)
            max_tokens = sk.shape[0] * sk.shape[1]
            if 0 < tot_kv <= max_tokens:
                sk_flat = sk.view(-1, Hkv, self.head_size)[:tot_kv]
                sv_flat = sv.view(-1, Hkv, self.head_size)[:tot_kv]
                dequant_pack_varlen_i8(
                    kv_cache,
                    sk_flat,
                    sv_flat,
                    attn_metadata.block_table[num_decodes:],
                    attn_metadata.prefill_kv_starts,
                    attn_metadata.prefill_max_kv,
                )
                _ixf_flash_attn_varlen_func(
                    q_pf,
                    sk_flat,
                    sv_flat,
                    attn_metadata.prefill_q_starts,
                    attn_metadata.prefill_kv_starts,
                    attn_metadata.prefill_max_q,
                    attn_metadata.prefill_max_kv,
                    softmax_scale=self.scale,
                    causal=True,
                    out=out_pf,
                )
                return output
        if attn_metadata.i8_scratch_sp is not None:
            sk, sv = _get_scratch(query.device, Hkv, self.head_size, block_size)
            dequant_kv_pages_i8(
                kv_cache,
                sk,
                sv,
                attn_metadata.block_table[num_decodes:],
                attn_metadata.i8_scratch_sp,
            )
            gqa_prefill_attention(
                query,
                sk,
                sv,
                output,
                attn_metadata.query_start_loc[num_decodes:],
                attn_metadata.seq_lens[num_decodes:],
                attn_metadata.i8_scratch_block_table,
                self.scale,
                num_prefill_tokens=num_actual_tokens - num_decode_tokens,
            )
            return output
        gqa_prefill_attention_i8(
            query,
            kv_cache,
            output,
            attn_metadata.query_start_loc[num_decodes:],
            attn_metadata.seq_lens[num_decodes:],
            attn_metadata.block_table[num_decodes:],
            self.scale,
            num_prefill_tokens=num_actual_tokens - num_decode_tokens,
        )
        return output


class SplitAttentionBackend(TritonAttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "TRITON_ATTN"

    @staticmethod
    def get_impl_cls() -> type[SplitAttentionImpl]:
        return SplitAttentionImpl

    @staticmethod
    def get_builder_cls() -> type[SplitAttentionMetadataBuilder]:
        return SplitAttentionMetadataBuilder

    @staticmethod
    def kv_cache_row_slots(vllm_config) -> int | None:
        return ROW_SLOTS if kv_i8_enabled(vllm_config) else None
