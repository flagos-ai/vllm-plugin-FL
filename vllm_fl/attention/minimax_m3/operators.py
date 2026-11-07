# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors

"""Common-resolver bindings for numerical operations selected by M3 metadata."""

from vllm_fl.dispatch import CachedOp

index_score = CachedOp("m3_index_score", retry_on_error=False)
prefill = CachedOp("m3_sparse_attn", retry_on_error=False)
decode = CachedOp("m3_sparse_decode", retry_on_error=False)
index_topk = CachedOp("m3_index_topk", retry_on_error=False)
native_index_score = CachedOp("m3_index_score_native", retry_on_error=False)
native_index_decode = CachedOp("m3_index_decode_native", retry_on_error=False)
native_prefill = CachedOp("m3_sparse_attn_native", retry_on_error=False)
native_decode = CachedOp("m3_sparse_decode_native", retry_on_error=False)
