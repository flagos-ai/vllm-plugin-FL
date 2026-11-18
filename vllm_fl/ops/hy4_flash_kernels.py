# SPDX-License-Identifier: Apache-2.0
"""Kernel dispatch used by the NVIDIA Hy4 Flash backend."""

from vllm_fl.dispatch import CachedOp

ihc_pre = CachedOp("hy4_flash_ihc_pre")
ihc_post = CachedOp("hy4_flash_ihc_post")
sparse_mla = CachedOp("hy4_flash_sparse_mla")
fp32_router_gemm = CachedOp("hy4_flash_fp32_router_gemm")
paged_mqa = CachedOp("hy4_flash_paged_mqa")
int8_scaled_mm = CachedOp("hy4_flash_int8_scaled_mm")
concat_mla_q = CachedOp("hy4_flash_concat_mla_q")
concat_and_cache_mla = CachedOp("hy4_flash_concat_and_cache_mla")
moe_align = CachedOp("hy4_flash_moe_align")
moe_sum = CachedOp("hy4_flash_moe_sum")
expert_quant_int8 = CachedOp("hy4_flash_expert_quant_int8")
mrope = CachedOp("hy4_flash_mrope")
