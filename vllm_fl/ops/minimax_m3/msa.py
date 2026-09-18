# SPDX-License-Identifier: Apache-2.0
from vllm_fl.dispatch import CachedOp

minimax_m3_index_score = CachedOp("m3_index_score")
minimax_m3_index_topk = CachedOp("m3_index_topk")
minimax_m3_index_decode = CachedOp("m3_index_decode")
minimax_m3_sparse_attn = CachedOp("m3_sparse_attn")
minimax_m3_sparse_attn_decode = CachedOp("m3_sparse_decode")
