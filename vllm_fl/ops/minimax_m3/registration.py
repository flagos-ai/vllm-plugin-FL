# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors

"""Profile-scoped dispatch: library adapters and MetaX MSA have distinct owners."""

from vllm_fl.dispatch import BackendImplKind, BackendPriority, OpImpl
from vllm_fl.minimax_m3_config import is_enabled

LIBRARY_OPS = {
    "gemma": "m3.flaggems_vllm.",
    "swiglu": "m3.flaggems_vllm.",
    "qknorm_rope_insert": "m3.flaggems_vllm.",
    "patch_embed": "m3.flag_gems.",
    "vision_attention": "m3.flag_gems.",
}
METAX_ATTENTION_OPS = (
    "index_score",
    "index_topk",
    "index_decode",
    "sparse_attn",
    "sparse_decode",
)
MOE_OPS = (
    "moe_align_block_size",
    "invoke_fused_moe_triton_kernel",
    "moe_sum",
    "grouped_topk",
)


def vllm_fl_register(registry):
    if not is_enabled():
        return
    from . import torch_ops
    from vllm_fl.dispatch.backends.flaggems.impl.minimax_m3 import moe

    for name, prefix in LIBRARY_OPS.items():
        registry.register_impl(
            OpImpl(
                op_name="m3_" + name,
                impl_id=prefix + name,
                kind=BackendImplKind.DEFAULT,
                fn=getattr(torch_ops, name),
                priority=BackendPriority.DEFAULT,
            )
        )
    for name in METAX_ATTENTION_OPS:
        registry.register_impl(
            OpImpl(
                op_name="m3_" + name,
                impl_id="vendor.metax.m3." + name,
                kind=BackendImplKind.VENDOR,
                vendor="metax",
                fn=getattr(torch_ops, name),
                priority=BackendPriority.VENDOR,
            )
        )
    for name in MOE_OPS:
        registry.register_impl(
            OpImpl(
                op_name="m3_" + name,
                impl_id="m3.flaggems_vllm." + name,
                kind=BackendImplKind.DEFAULT,
                fn=getattr(moe, name),
                priority=BackendPriority.DEFAULT,
            )
        )
