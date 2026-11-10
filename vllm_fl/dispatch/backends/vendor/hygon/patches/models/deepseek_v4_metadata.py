# Copyright (c) 2026 BAAI. All rights reserved.
"""Hygon decode metadata launch reduction; numerical attention is unchanged."""

import functools
import logging
import os


def patch_deepseek_v4_ragged_metadata():
    if os.getenv("VLLM_FL_HYGON_FUSED_RAGGED_METADATA", "1") == "0":
        return
    from vllm.models.deepseek_v4.amd import rocm
    from ...impl.attention.ragged_metadata import build_decode_ragged_into
    from ...impl.attention.swa_decode_metadata import try_build_swa_decode
    fuse_swa = os.getenv("VLLM_FL_HYGON_FUSED_SWA_METADATA", "1") != "0"

    swa_cls = rocm.DeepseekV4ROCMAiterSparseSWAMetadataBuilder
    mla_cls = rocm.DeepseekV4ROCMAiterMLASparseMetadataBuilder
    if getattr(swa_cls.build, "_hygon_fused_ragged", False):
        return

    def pack(dense, lengths, flat_buffer, indptr_buffer, width):
        result = build_decode_ragged_into(dense, lengths, flat_buffer, indptr_buffer)
        if result is not None:
            return result
        flat, indptr = rocm.build_ragged_indices_from_dense(
            dense.reshape(dense.shape[0], -1), lengths)
        return rocm._copy_ragged_to_graph_buffers(
            flat, indptr, flat_buffer, indptr_buffer, dense.shape[0], width)

    @functools.wraps(swa_cls.build)
    def build_swa(self, common_prefix_len, common_attn_metadata, fast_build=False):
        if fuse_swa:
            result = try_build_swa_decode(
                self, common_attn_metadata, rocm.DeepseekV4ROCMAiterSparseSWAMetadata)
            if result is not None:
                return result
        base = rocm.DeepseekSparseSWAMetadataBuilder.build(
            self, common_prefix_len, common_attn_metadata, fast_build)
        flat = indptr = None
        if (base.num_decode_tokens > 0 and base.decode_swa_indices is not None
                and base.decode_swa_lens is not None):
            flat, indptr = pack(
                base.decode_swa_indices, base.decode_swa_lens,
                self.decode_swa_ragged_indices_buffer,
                self.decode_swa_ragged_indptr_buffer, self.window_size)
        return rocm.DeepseekV4ROCMAiterSparseSWAMetadata(
            **vars(base), decode_swa_ragged_indices=flat,
            decode_swa_ragged_indptr=indptr)

    @functools.wraps(mla_cls.build)
    def build_mla(self, common_prefix_len, common_attn_metadata, fast_build=False):
        base = rocm.DeepseekV4FlashMLAMetadataBuilder.build(
            self, common_prefix_len, common_attn_metadata, fast_build)
        flat = indptr = None
        dense = base.c128a_global_decode_topk_indices
        lengths = base.c128a_decode_topk_lens
        if dense is not None and lengths is not None:
            flat, indptr = pack(
                dense, lengths, self.c128a_decode_topk_ragged_indices_buffer,
                self.c128a_decode_topk_ragged_indptr_buffer,
                self.c128a_max_compressed)
        return rocm.DeepseekV4ROCMAiterMLASparseMetadata(
            **vars(base), c128a_decode_topk_ragged_indices=flat,
            c128a_decode_topk_ragged_indptr=indptr)

    build_swa._hygon_fused_ragged = True
    build_mla._hygon_fused_ragged = True
    swa_cls.build = build_swa
    mla_cls.build = build_mla
    logging.getLogger(__name__).info("Enabled fused Hygon DV4 decode ragged metadata")
