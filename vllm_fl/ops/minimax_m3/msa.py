# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors

from vllm_fl.ops.minimax_m3 import torch_ops

minimax_m3_index_score = torch_ops.index_score
minimax_m3_index_topk = torch_ops.index_topk
minimax_m3_index_decode = torch_ops.index_decode
minimax_m3_sparse_attn = torch_ops.sparse_attn
minimax_m3_sparse_attn_decode = torch_ops.sparse_decode
