"""Integrate hybrid prefill and split decode with the Triton attention backend.

The INT8 cache is quantized in ``do_kv_cache_update``. Prefill and mixed
batches use hybrid plans, while single-token decode uses the INT8 decode
kernel in eager execution and CUDA graphs. BF16 caches use upstream attention.
"""

import os

import torch

from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.attention.backends.triton_attn import (
    TritonAttentionBackend,
    TritonAttentionImpl,
    TritonAttentionMetadataBuilder,
)
from vllm.v1.attention.ops import triton_unified_attention as upstream

from vllm_fl import envs as fl_envs
from vllm_fl.attention import int8_decode_config, kv_int8, mixed_select
from vllm_fl.attention.gqa_hybrid import HybridPlan, HybridWorkspace


class HybridMetadataBuilder(TritonAttentionMetadataBuilder):
    _cudagraph_support = AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hybrid_workspace = HybridWorkspace(self.device)
        self.hybrid_split_prefix_mask = os.environ.get(
            "VLLM_FL_GQA_HYBRID_PREFIX_MASK", "0"
        ).lower() in ("1", "true", "yes")
        self.hybrid_prefill_order = (
            "lpt"
            if fl_envs.switch("VLLM_FL_GQA_HYBRID_PREFILL_LPT", auto=True)
            else "natural"
        )
        self.hybrid_prefill_reblock = fl_envs.switch(
            "VLLM_FL_GQA_HYBRID_PREFILL_REBLOCK", auto=True
        )
        self.hybrid_layer_ids = {name: i for i, name in enumerate(self.layer_names)}
        self.kv_int8 = self.kv_cache_spec.dtype == torch.int8
        if self.kv_int8:
            if not kv_int8.kv_int8_enabled():
                raise RuntimeError("INT8 KV spec without VLLM_FL_KV_INT8=1")
            self.kv_int8_decode_cfg = int8_decode_config.resolve_decode(self.device)
            self.kv_int8_decode_ws = kv_int8.DecodeWorkspace(self.device)
            self.kv_int8_scratch_tokens = kv_int8.prefill_scratch_cap_tokens()
            kv_int8.logger.info(
                "VLLM_FL_KV_INT8: hybrid builder for %d layers, INT8 decode kernel, "
                "prefill scratch cap %d tokens",
                len(self.layer_names),
                self.kv_int8_scratch_tokens,
            )
        self.mixed = mixed_select.MixedDecision(
            reblock_allowed=self.hybrid_prefill_reblock
        )
        if (
            self.kv_int8
            and self.kv_cache_spec.block_size == 16
            and (self.num_heads_q, self.num_heads_kv, self.headdim) == (16, 2, 128)
            and torch.version.cuda is not None
            and torch.version.hip is None
            and not upstream.is_batch_invariant
        ):
            self.mixed = mixed_select.resolve(
                self.device,
                page=16,
                lpt=self.hybrid_prefill_order == "lpt",
                reblock_ok=self.hybrid_prefill_reblock,
                decode=(self.kv_int8_decode_ws, self.kv_int8_decode_cfg),
                scratch_tokens=self.kv_int8_scratch_tokens,
            )

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        result = super().build(common_prefix_len, common_attn_metadata, fast_build)
        c = common_attn_metadata
        if self.kv_int8:
            result.kv_int8_decode_ws = self.kv_int8_decode_ws
            result.kv_int8_decode_cfg = self.kv_int8_decode_cfg
        if (
            not self.kv_int8
            or upstream.is_batch_invariant
            or torch.version.cuda is None
            or torch.version.hip is not None
            or self.kv_cache_spec.dtype != torch.int8
            or (self.num_heads_q, self.num_heads_kv, self.headdim) != (16, 2, 128)
        ):
            return result
        if (
            common_prefix_len
            or c.max_query_len <= 1
            or c.causal is not True
            or c.mm_req_doc_ranges is not None
        ):
            return result
        # Upper bounds are scheduler-owned CPU tensors; never invoke the
        # deprecated seq_lens_cpu property (it can synchronize the GPU).
        lengths = c._seq_lens_cpu
        if lengths is None:
            lengths = c.seq_lens_cpu_upper_bound
        if (lengths is None or lengths.device.type != "cpu") and self.kv_int8:
            # INT8 pages have no upstream fallback; a rare synchronizing copy
            # (warm-up/dummy batches) is preferable to failing.
            lengths = c.seq_lens[: c.num_reqs].cpu()
        if lengths is None or lengths.device.type != "cpu":
            return result
        cu = c.query_start_loc_cpu[: c.num_reqs + 1].tolist()
        qlens = [b - a for a, b in zip(cu, cu[1:])]
        max_query = max((q for q in qlens if q > 1), default=0)
        lengths_list = lengths[: c.num_reqs].tolist()
        num_decode = qlens.count(1)
        choice = (
            self.mixed.lookup(num_decode, max_query, max(lengths_list, default=0))
            if self.kv_int8
            else mixed_select.Choice(
                tile=mixed_select.rule_tile(max_query, self.hybrid_prefill_reblock)
            )
        )
        pm, pn, prefill_reblock = mixed_select.TILES[choice.tile]
        prefill_wide = bool(
            self.kv_int8
            and self.mixed.prefill_wide
            and max_query >= 1024
            and choice.tile == "reblock"
            and (choice.mode == "separate" or num_decode == 0)
        )
        if self.kv_int8:
            result.hybrid_mixed_mode = choice.mode
            if choice.mode == "separate":
                result.hybrid_decode_segments = mixed_select.decode_segments(
                    self.kv_int8_decode_cfg, num_decode
                )
            from vllm_fl.ops import int8_sparse

            sparse_bundle = None
            if int8_sparse.is_enabled():
                sparse_bundle = int8_sparse.make_prefill_bundle(
                    c,
                    qlens,
                    lengths_list,
                    choice=choice,
                    prefill_wide=prefill_wide,
                    fixed_anchor=(prefill_wide and self.mixed.prefill_fixed_anchor),
                    lpt=self.hybrid_prefill_order == "lpt",
                    split_prefix_mask=self.hybrid_split_prefix_mask,
                    decode_external=choice.mode == "separate",
                )
            if sparse_bundle is not None:
                result.hybrid_plan = None
                result.int8_sparse_prefill_bundle = sparse_bundle
                result.hybrid_layer_ids = self.hybrid_layer_ids
                return result
        plan = HybridPlan.build(
            qlens,
            lengths_list,
            device=self.device,
            pm=pm,
            pn=pn,
            layers=len(self.hybrid_layer_ids),
            workspace=self.hybrid_workspace,
            split_prefix_mask=self.hybrid_split_prefix_mask,
            prefill_order=self.hybrid_prefill_order,
            prefill_reblock=prefill_reblock,
            prefill_wide=prefill_wide,
            prefill_fixed_anchor=(prefill_wide and self.mixed.prefill_fixed_anchor),
            kv_int8_scratch_tokens=self.kv_int8_scratch_tokens if self.kv_int8 else 0,
            decode_external=choice.mode == "separate",
        )
        plan.reset()
        result.hybrid_plan = plan
        result.hybrid_layer_ids = self.hybrid_layer_ids
        return result


class HybridAttentionImpl(TritonAttentionImpl):
    def _int8_cache(self, kv_cache):
        return (
            self.kv_cache_dtype == "auto"
            and kv_cache.dtype == torch.int8
            and self.attn_type == "decoder"
        )

    def do_kv_cache_update(self, layer, key, value, kv_cache, slot_mapping):
        if self._int8_cache(kv_cache):
            kv_int8.kv_int8_write(key, value, kv_cache, slot_mapping)
            return
        return super().do_kv_cache_update(layer, key, value, kv_cache, slot_mapping)

    def _forward_int8(
        self,
        layer,
        query,
        kv_cache,
        attn_metadata,
        output,
        output_scale,
        output_block_scale,
    ):
        if attn_metadata is None:
            return output.fill_(0)  # profiling run
        if not (
            not upstream.is_batch_invariant
            and query.dtype == torch.bfloat16
            and (self.num_heads, self.num_kv_heads, self.head_size) == (16, 2, 128)
            and self.sliding_window == (-1, -1)
            and not self.logits_soft_cap
            and self.alibi_slopes is None
            and self.sinks is None
            and output_scale is None
            and output_block_scale is None
            and not self.use_td
            and self.chunk_lookback == -1
            and attn_metadata.mm_prefix_range_tensor is None
            and attn_metadata.causal is True
        ):
            raise NotImplementedError("VLLM_FL_KV_INT8: unsupported attention features")
        scales = getattr(layer, "kv_int8_scales", None)
        if scales is None or scales.data_ptr() != kv_cache.data_ptr():
            scales = kv_int8.half_base(kv_cache)  # view only; no device work
        plan = getattr(attn_metadata, "hybrid_plan", None)
        sparse_bundle = getattr(attn_metadata, "int8_sparse_prefill_bundle", None)
        if sparse_bundle is not None:
            from vllm_fl.ops import int8_sparse

            ids = attn_metadata.hybrid_layer_ids
            if layer.layer_name not in ids:
                raise RuntimeError("hybrid layer has no private scheduling state")
            int8_sparse.run_prefill(
                sparse_bundle,
                query,
                kv_cache,
                attn_metadata.block_table,
                attn_metadata.seq_lens,
                output,
                self.scale,
                layer_id=ids[layer.layer_name],
            )
            if sparse_bundle.decode_external:
                mixed_select.standalone_decode(
                    dict(
                        q=query,
                        kv_cache=kv_cache,
                        kv_scales=scales,
                        out=output,
                        cu_seqlens_q=attn_metadata.query_start_loc,
                        seqused_k=attn_metadata.seq_lens,
                        block_table=attn_metadata.block_table,
                        softmax_scale=self.scale,
                    ),
                    attn_metadata.kv_int8_decode_ws,
                    attn_metadata.kv_int8_decode_cfg,
                    attn_metadata.hybrid_decode_segments,
                )
            return output
        if plan is not None:
            ids = attn_metadata.hybrid_layer_ids
            if layer.layer_name not in ids:
                raise RuntimeError("hybrid layer has no private scheduling state")
            k, v = kv_cache.unbind(1)
            mode = attn_metadata.hybrid_mixed_mode
            decode = None
            if mode == "separate":
                decode = lambda kw: mixed_select.standalone_decode(
                    kw,
                    attn_metadata.kv_int8_decode_ws,
                    attn_metadata.kv_int8_decode_cfg,
                    attn_metadata.hybrid_decode_segments,
                )
            mixed_select.run_mixed(
                plan,
                dict(
                    q=query,
                    k=k,
                    v=v,
                    out=output,
                    cu_seqlens_q=attn_metadata.query_start_loc,
                    seqused_k=attn_metadata.seq_lens,
                    block_table=attn_metadata.block_table,
                    softmax_scale=self.scale,
                    kv_scales=scales,
                    kv_cache=kv_cache,
                ),
                mode,
                decode,
                layer=ids[layer.layer_name],
            )
            return output
        if attn_metadata.max_query_len != 1:
            raise RuntimeError("VLLM_FL_KV_INT8: batch has no hybrid plan")
        decode_table = attn_metadata.block_table
        decode_lengths = attn_metadata.seq_lens
        from vllm_fl.ops import int8_sparse

        if int8_sparse.is_enabled() and int8_sparse.should_compact_decode(
            self, attn_metadata
        ):
            compact = int8_sparse.compact_decode(
                self,
                query[: decode_lengths.shape[0]],
                decode_table,
                decode_lengths,
            )
            if compact is not None:
                decode_table, decode_lengths = compact
        kv_int8.kv_int8_decode(
            query,
            kv_cache,
            scales,
            output,
            attn_metadata.query_start_loc,
            decode_lengths,
            decode_table,
            self.scale,
            attn_metadata.kv_int8_decode_ws,
            config=attn_metadata.kv_int8_decode_cfg,
        )
        return output

    def forward(
        self,
        layer,
        query,
        key,
        value,
        kv_cache,
        attn_metadata,
        output,
        output_scale=None,
        output_block_scale=None,
    ):
        if attn_metadata is None and kv_int8.kv_int8_enabled():
            # Profiling run: reserve the prefill scratch now so the KV cache
            # budget excludes it.
            cap = kv_int8.prefill_scratch_cap_tokens()
            if cap > 0:
                kv_int8.prefill_scratch(query.device, cap)
        if self._int8_cache(kv_cache):
            return self._forward_int8(
                layer,
                query,
                kv_cache,
                attn_metadata,
                output,
                output_scale,
                output_block_scale,
            )
        plan = getattr(attn_metadata, "hybrid_plan", None)
        supported = (
            plan is not None
            and not upstream.is_batch_invariant
            and query.dtype == torch.bfloat16
            and kv_cache.dtype == torch.bfloat16
            and (self.num_heads, self.num_kv_heads, self.head_size) == (16, 2, 128)
            and self.sliding_window == (-1, -1)
            and not self.logits_soft_cap
            and self.alibi_slopes is None
            and self.sinks is None
            and self.attn_type == "decoder"
            and output_scale is None
            and output_block_scale is None
            and not self.use_td
            and self.chunk_lookback == -1
        )
        if not supported:
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
        ids = attn_metadata.hybrid_layer_ids
        if layer.layer_name not in ids:
            raise RuntimeError("hybrid layer has no private scheduling state")
        k, v = kv_cache.unbind(1)
        plan.run(
            dict(
                q=query,
                k=k,
                v=v,
                out=output,
                cu_seqlens_q=attn_metadata.query_start_loc,
                seqused_k=attn_metadata.seq_lens,
                block_table=attn_metadata.block_table,
                softmax_scale=self.scale,
            ),
            layer=ids[layer.layer_name],
            mode="fused",
        )
        return output


class HybridAttentionBackend(TritonAttentionBackend):
    @staticmethod
    def get_impl_cls():
        return HybridAttentionImpl

    @staticmethod
    def get_builder_cls():
        return HybridMetadataBuilder
