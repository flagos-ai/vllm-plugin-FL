# SPDX-License-Identifier: Apache-2.0
"""M3 custom operators exposed as torch.ops.vllm_fl, with honest mutation schemas."""

import torch

from . import ops as _raw, qkv as _qkv

_LIB = torch.library.Library("vllm_fl", "FRAGMENT")


def _register(schema, implementation, fake):
    name = schema.split("(", 1)[0]
    _LIB.define(schema)
    _LIB.impl(name, implementation, "CUDA")
    torch.library.register_fake("vllm_fl::" + name)(fake)


def _empty(x, shape=None, dtype=None):
    return torch.empty(
        tuple(x.shape) if shape is None else shape,
        device=x.device,
        dtype=x.dtype if dtype is None else dtype,
    )


_register(
    "m3_gemma(Tensor x, Tensor weight, float eps) -> Tensor",
    lambda x, weight, eps: _raw.gemma(x, weight, eps),
    lambda x, weight, eps: _empty(x),
)


def _gemma_add(x, residual, weight, eps):
    _raw.gemma(x, weight, eps, residual)


_register(
    "m3_gemma_add_(Tensor(a!) x, Tensor(b!) residual, Tensor weight, float eps) -> ()",
    _gemma_add,
    lambda *a: None,
)

_register(
    "m3_swiglu(Tensor x, float limit, float alpha, float beta) -> Tensor",
    lambda x, limit, alpha, beta: _raw.swiglu(x, limit, alpha, beta),
    lambda x, limit, alpha, beta: _empty(x, (*x.shape[:-1], x.shape[-1] // 2)),
)


def _swiglu_out(x, limit, alpha, beta, out):
    _raw.swiglu(x, limit, alpha, beta, out=out)


_register(
    "m3_swiglu_out_(Tensor x, float limit, float alpha, float beta, Tensor(a!) out) -> ()",
    _swiglu_out,
    lambda *a: None,
)

_register(
    'm3_qknorm_rope_insert_(Tensor(a!) qkv, Tensor q_norm_weight, Tensor k_norm_weight, Tensor cos_sin_cache, Tensor positions, int num_heads, int num_kv_heads, int rotary_dim, float eps, Tensor? index_q_norm_weight=None, Tensor? index_k_norm_weight=None, int num_index_heads=0, Tensor? slot_mapping=None, Tensor? index_slot_mapping=None, Tensor(b!)? kv_cache=None, Tensor(c!)? index_cache=None, int block_size=0, Tensor(d!)? q_out=None, Tensor(e!)? index_q_out=None, str kv_cache_dtype="auto") -> ()',
    _qkv.qknorm_rope_insert,
    lambda *a, **kw: None,
)


def _fake_score(
    idx_q,
    index_kv_cache,
    block_table,
    cu_seqlens_q,
    seq_lens,
    prefix_lens,
    max_query_len,
    max_seq_len,
    num_kv_heads,
):
    blocks = (max_seq_len + 127) // 128
    return _empty(
        idx_q,
        (idx_q.shape[1], idx_q.shape[0], ((blocks + 15) // 16) * 16),
        torch.float32,
    )


_register(
    "m3_index_score(Tensor idx_q, Tensor index_kv_cache, Tensor block_table, Tensor cu_seqlens_q, Tensor seq_lens, Tensor prefix_lens, int max_query_len, int max_seq_len, int num_kv_heads) -> Tensor",
    _raw.index_score,
    _fake_score,
)


def _index_topk_out(
    score,
    cu_seqlens_q,
    prefix_lens,
    max_query_len,
    topk,
    init_blocks,
    local_blocks,
    out,
):
    _raw.index_topk(
        score,
        cu_seqlens_q,
        prefix_lens,
        max_query_len,
        topk,
        init_blocks,
        local_blocks,
        out=out,
    )


_register(
    "m3_index_topk_out_(Tensor score, Tensor cu_seqlens_q, Tensor prefix_lens, int max_query_len, int topk, int init_blocks, int local_blocks, Tensor(a!) out) -> ()",
    _index_topk_out,
    lambda *a: None,
)


def _index_decode_out(
    idx_q,
    index_kv_cache,
    block_table,
    seq_lens,
    max_seq_len,
    topk,
    init_blocks,
    local_blocks,
    num_kv_heads,
    decode_query_len,
    max_decode_query_len,
    out,
):
    _raw.index_decode(
        idx_q,
        index_kv_cache,
        block_table,
        seq_lens,
        max_seq_len,
        topk,
        init_blocks,
        local_blocks,
        num_kv_heads,
        decode_query_len,
        max_decode_query_len,
        out=out,
    )


_register(
    "m3_index_decode_out_(Tensor idx_q, Tensor index_kv_cache, Tensor block_table, Tensor seq_lens, int max_seq_len, int topk, int init_blocks, int local_blocks, int num_kv_heads, int decode_query_len, int max_decode_query_len, Tensor(a!) out) -> ()",
    _index_decode_out,
    lambda *a: None,
)

_register(
    "m3_sparse_attn_out_(Tensor q, Tensor kv_cache, Tensor topk_idx, Tensor block_table, Tensor cu_seqlens_q, Tensor seq_lens, Tensor prefix_lens, int max_query_len, int num_kv_heads, float sm_scale, Tensor(a!) output) -> ()",
    _raw.sparse_attn,
    lambda *a: None,
)
_register(
    "m3_sparse_decode_out_(Tensor q, Tensor kv_cache, Tensor topk_idx, Tensor block_table, Tensor seq_lens, int num_kv_heads, float sm_scale, Tensor(a!) output, int decode_query_len) -> ()",
    _raw.sparse_decode,
    lambda *a: None,
)


def _vision_rope(x, cos, sin):
    from .vision import vision_rope as implementation

    return implementation(x, cos, sin)


def _vision_attention(q, k, v, cu, maxlen, scale):
    from .vision import vision_attention as implementation

    return implementation(q, k, v, cu, maxlen, scale)


_register(
    "m3_vision_rope(Tensor x, Tensor cos, Tensor sin) -> Tensor",
    _vision_rope,
    lambda x, cos, sin: _empty(x),
)
_register(
    "m3_vision_attention(Tensor query, Tensor key, Tensor value, Tensor? cu_seqlens, int max_seqlen, float scale) -> Tensor",
    _vision_attention,
    lambda query, key, value, cu_seqlens, max_seqlen, scale: _empty(query),
)


# FL implementation entrypoints keep the existing caller API and select a torch op.
def gemma(x, weight, eps, residual=None):
    if residual is None:
        return torch.ops.vllm_fl.m3_gemma(x, weight, eps)
    torch.ops.vllm_fl.m3_gemma_add_(x, residual, weight, eps)
    return x, residual


def swiglu(x, limit=7.0, alpha=1.702, beta=1.0, out=None):
    if out is None:
        return torch.ops.vllm_fl.m3_swiglu(x, limit, alpha, beta)
    torch.ops.vllm_fl.m3_swiglu_out_(x, limit, alpha, beta, out)
    return out


def qknorm_rope_insert(*args, **kwargs):
    return torch.ops.vllm_fl.m3_qknorm_rope_insert_(*args, **kwargs)


def index_score(*args, **kwargs):
    return torch.ops.vllm_fl.m3_index_score(*args, **kwargs)


def index_topk(
    score,
    cu_seqlens_q,
    prefix_lens,
    max_query_len,
    topk,
    init_blocks,
    local_blocks,
    out=None,
):
    if out is None:
        out = _empty(score, (score.shape[0], score.shape[1], topk), torch.int32)
    torch.ops.vllm_fl.m3_index_topk_out_(
        score,
        cu_seqlens_q,
        prefix_lens,
        max_query_len,
        topk,
        init_blocks,
        local_blocks,
        out,
    )
    return out[:, : score.shape[1], :]


def index_decode(
    idx_q,
    index_kv_cache,
    block_table,
    seq_lens,
    max_seq_len,
    topk,
    init_blocks,
    local_blocks,
    num_kv_heads,
    decode_query_len,
    max_decode_query_len,
    out=None,
):
    if out is None:
        out = _empty(idx_q, (num_kv_heads, idx_q.shape[0], topk), torch.int32)
    torch.ops.vllm_fl.m3_index_decode_out_(
        idx_q,
        index_kv_cache,
        block_table,
        seq_lens,
        max_seq_len,
        topk,
        init_blocks,
        local_blocks,
        num_kv_heads,
        decode_query_len,
        max_decode_query_len,
        out,
    )
    return out[:, : idx_q.shape[0], :]


def sparse_attn(*args, **kwargs):
    return torch.ops.vllm_fl.m3_sparse_attn_out_(*args, **kwargs)


def sparse_decode(*args, **kwargs):
    return torch.ops.vllm_fl.m3_sparse_decode_out_(*args, **kwargs)


def vision_rope(x, cos, sin):
    return torch.ops.vllm_fl.m3_vision_rope(x, cos, sin)


def vision_attention(query, key, value, cu_seqlens, max_seqlen, scale):
    maxlen = (
        int(max_seqlen.item())
        if isinstance(max_seqlen, torch.Tensor)
        else int(max_seqlen or query.shape[1])
    )
    return torch.ops.vllm_fl.m3_vision_attention(
        query, key, value, cu_seqlens, maxlen, scale
    )


# A full non-overlapping Conv3d patch is exactly a row-wise matrix projection.
def _patch_embed(pixel_values, weight):
    import flag_gems

    assert pixel_values.ndim == 2 and weight.ndim == 5
    assert pixel_values.shape[1] == weight[0].numel()
    assert pixel_values.dtype == weight.dtype
    return flag_gems.mm(pixel_values, weight.flatten(1).t())


_register(
    "m3_patch_embed(Tensor pixel_values, Tensor weight) -> Tensor",
    _patch_embed,
    lambda pixel_values, weight: _empty(
        pixel_values, (pixel_values.shape[0], weight.shape[0])
    ),
)


def patch_embed(pixel_values, weight):
    return torch.ops.vllm_fl.m3_patch_embed(pixel_values, weight)


SCHEMAS = sorted(
    n for n in torch._C._dispatch_get_all_op_names() if n.startswith("vllm_fl::m3_")
)
