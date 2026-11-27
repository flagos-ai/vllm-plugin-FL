# Copyright (c) 2026 BAAI. All rights reserved.
"""Public numerical contracts for V4.1 through the existing uniform dispatcher.

These forwarders never catch execution errors or choose another backend.
Request mapping, cache allocation and buffer lifetimes stay in the model.
"""

import functools
import inspect

# Required public arguments establish the exact installed dependency contract.
V41_PUBLIC_OPERATORS = {
    "deepseek_v41_compress_norm": {
        "kv_score",
        "positions",
        "state_cache",
        "slot_mapping",
        "query_start_loc",
        "token_to_req_indices",
        "rms_norm_weight",
        "rms_norm_eps",
        "compress_ratio",
        "latent_out",
    },
    "gather_rows": {"x", "indices"},
    "sqrtsoftplus_topk": {
        "logits",
        "correction_bias",
        "topk",
        "input_ids",
        "bias_vl",
        "image_sentinel_lo",
    },
    "fused_q_kv_rmsnorm": {"qr", "kv", "q_weight", "kv_weight", "eps"},
    "fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert": {
        "q",
        "kv",
        "k_cache",
        "slot_mapping",
        "position_ids",
        "cos_sin_cache",
        "eps",
        "cache_block_size",
        "apply_q_norm",
    },
    "fused_indexer_q_rope_quant": {
        "positions",
        "index_q",
        "index_q_cos_sin_cache",
        "index_weights",
        "index_weights_softmax_scale",
        "index_weights_head_scale",
        "use_fp4",
    },
    "deepseek_v41_indexer_k_store": {
        "k_pre",
        "group_positions",
        "cos_sin_cache",
        "rms_norm_weight",
        "rms_norm_eps",
        "k_cache",
        "slots",
    },
    "paged_mqa_logits": {
        "q",
        "kv_cache",
        "weights",
        "context_lens",
        "block_tables",
        "schedule_metadata",
        "max_model_len",
        "clean_logits",
    },
    "deepseek_v41_select_candidate_blocks": {
        "logits",
        "row_ks",
        "row_ke",
        "topk_blocks",
        "block_size",
        "out",
        "row_repeat",
    },
    "deepseek_v41_apply_candidate_mask": {
        "logits",
        "row_ks",
        "row_ke",
        "candidate_blocks",
        "block_size",
        "row_repeat",
    },
    "persistent_topk": {
        "logits",
        "lengths",
        "output",
        "workspace",
        "k",
        "max_seq_len",
        "stable",
    },
    "apply_rotary_pos_emb": {
        "q",
        "k",
        "cos",
        "sin",
        "position_ids",
        "rotary_interleaved",
        "inplace",
    },
    "mhc_pre_delayed": {
        "residual",
        "fn",
        "hc_scale",
        "hc_base",
        "rms_eps",
        "hc_pre_eps",
        "hc_sinkhorn_eps",
        "hc_post_mult_value",
        "sinkhorn_repeat",
        "pre_mix",
        "norm_weight",
        "norm_eps",
    },
    "mhc_post": {"x", "residual", "post_layer_mix", "comb_res_mix"},
    "mhc_collapse": {"residual", "pre_mix"},
    "mhc_rms_norm": {"layer_input", "norm_weight", "norm_eps"},
    "v41_sparse_attention": {
        "q",
        "k_cache",
        "block_table",
        "cache_seqlens",
        "head_dim_v",
        "tile_scheduler_metadata",
        "indices",
        "attn_sink",
        "extra_k_cache",
        "extra_indices_in_kvcache",
    },
}


def is_v41_public_available(op_name):
    try:
        import flaggems_vllm
    except ModuleNotFoundError as error:
        if error.name != "flaggems_vllm":
            raise
        return False
    public_name = (
        "flash_mla_with_kvcache" if op_name == "v41_sparse_attention" else op_name
    )
    implementation = getattr(flaggems_vllm, public_name, None)
    if not callable(implementation):
        return False
    try:
        matches = V41_PUBLIC_OPERATORS[op_name].issubset(
            inspect.signature(implementation).parameters
        )
    except (TypeError, ValueError):
        return False
    if op_name == "v41_sparse_attention":
        from flaggems_vllm.ops.flash_mla_with_kvcache import get_mla_metadata

        matches = matches and callable(get_mla_metadata)
    return matches


def _forward_public(op_name, *args, **kwargs):
    import flaggems_vllm

    return getattr(flaggems_vllm, op_name)(*args, **kwargs)


def _sparse_attention(
    q, cache, indices, attn_sink, *, extra_cache=None, extra_indices=None
):
    """BF16 [tokens,1,64,512], packed FP8 pages and physical INT32 indices.

    Inputs encode causality; negative indices are skipped. Main SWA and
    optional compressed cache retain their independent physical block sizes.
    Sink-aware BF16 attention output is fresh. Metadata is call-local, so
    independent graph shapes and concurrent calls share no mutable scheduler.
    """
    from flaggems_vllm import flash_mla_with_kvcache
    from flaggems_vllm.ops.flash_mla_with_kvcache import get_mla_metadata

    metadata, _ = get_mla_metadata()
    output, _ = flash_mla_with_kvcache(
        q,
        cache,
        None,
        None,
        512,
        metadata,
        softmax_scale=512**-0.5,
        causal=False,
        is_fp8_kvcache=True,
        indices=indices,
        attn_sink=attn_sink,
        extra_k_cache=extra_cache,
        extra_indices_in_kvcache=extra_indices,
    )
    return output


def get_public_forwarder(op_name):
    if op_name not in V41_PUBLIC_OPERATORS:
        raise KeyError(op_name)
    if op_name == "v41_sparse_attention":
        return _sparse_attention
    return functools.partial(_forward_public, op_name)
