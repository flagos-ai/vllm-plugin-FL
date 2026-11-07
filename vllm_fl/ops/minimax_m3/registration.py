# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors

"""Profile-scoped numerical providers and explicit native compatibility paths."""

from vllm_fl.dispatch import BackendImplKind, BackendPriority, OpImpl
from vllm_fl.minimax_m3_config import is_enabled


def vllm_fl_register(registry):
    if not is_enabled():
        return
    from vllm_fl.dispatch.backends.flaggems.impl.minimax_m3 import (
        moe,
        numerics,
        vision,
    )
    from vllm_fl.dispatch.backends.vendor.metax.impl.minimax_m3 import (
        attention_launch,
        moe_gemv,
        native_index,
        native_sparse,
    )

    library = {
        "gemma": ("m3.flagos.gemma", numerics.gemma),
        "swiglu": ("m3.flaggems_vllm.swiglu", numerics.swiglu),
        "qknorm_rope_insert": (
            "m3.flaggems_vllm.qknorm_rope_insert",
            numerics.qknorm_rope_insert,
        ),
        "patch_embed": ("m3.flag_gems.patch_embed", vision._patch_embed),
        "vision_attention": ("m3.flag_gems.vision_attention", vision.vision_attention),
        "router_bf16_fp32": (
            "m3.flaggems_vllm.router_bf16_fp32",
            numerics.router_bf16_load_fp32_dot,
        ),
        "moe_align_block_size": (
            "m3.flaggems_vllm.moe_align_block_size",
            moe.moe_align_block_size,
        ),
        "invoke_fused_moe_triton_kernel": (
            "m3.flaggems_vllm.invoke_fused_moe_triton_kernel",
            moe.invoke_fused_moe_triton_kernel,
        ),
        "moe_sum": ("m3.flaggems_vllm.moe_sum", moe.moe_sum),
        "grouped_topk": ("m3.flagos.grouped_topk", moe.grouped_topk),
    }
    native = {
        "index_score": attention_launch.index_score,
        "sparse_attn": attention_launch.prefill,
        "sparse_decode": attention_launch.decode,
        "expert_gemv": moe_gemv.expert_gemv,
        "index_score_native": native_index.minimax_m3_index_score,
        "index_topk": native_index.minimax_m3_index_topk,
        "index_decode_native": native_index.minimax_m3_index_decode,
        "sparse_attn_native": native_sparse.minimax_m3_sparse_attn,
        "sparse_decode_native": native_sparse.minimax_m3_sparse_attn_decode,
    }
    for name, (impl_id, fn) in library.items():
        registry.register_impl(
            OpImpl(
                op_name="m3_" + name,
                impl_id=impl_id,
                kind=BackendImplKind.DEFAULT,
                fn=fn,
                priority=BackendPriority.DEFAULT,
            )
        )
    for name, fn in native.items():
        registry.register_impl(
            OpImpl(
                op_name="m3_" + name,
                impl_id="vendor.metax.m3." + name,
                kind=BackendImplKind.VENDOR,
                vendor="metax",
                fn=fn,
                priority=BackendPriority.VENDOR,
            )
        )
