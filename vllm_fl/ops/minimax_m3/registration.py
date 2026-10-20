# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors

"""FL operator registrations. Computation sources are explicit and independently importable."""

from . import torch_ops
from vllm_fl.dispatch import BackendImplKind, OpImpl

NAMES = (
    "gemma",
    "swiglu",
    "index_score",
    "index_topk",
    "index_decode",
    "sparse_attn",
    "sparse_decode",
    "vision_attention",
    "qknorm_rope_insert",
    "patch_embed",
)


def vllm_fl_register(registry):
    for name in NAMES:
        registry.register_impl(
            OpImpl(
                op_name="m3_" + name,
                impl_id=(
                    "m3.flag_gems."
                    if name in {"patch_embed", "vision_attention"}
                    else (
                        "m3.fl."
                        if name
                        in {
                            "index_score",
                            "index_topk",
                            "index_decode",
                            "sparse_attn",
                            "sparse_decode",
                        }
                        else "m3.flaggems_vllm."
                    )
                )
                + name,
                kind=BackendImplKind.DEFAULT,
                fn=getattr(torch_ops, name),
                priority=150,
            )
        )

    from . import library_bindings

    for name in (
        "moe_align_block_size",
        "invoke_fused_moe_triton_kernel",
        "moe_sum",
        "grouped_topk",
    ):
        registry.register_impl(
            OpImpl(
                op_name="m3_" + name,
                impl_id="m3.flaggems_vllm." + name,
                kind=BackendImplKind.DEFAULT,
                fn=getattr(library_bindings, name),
                priority=150,
            )
        )
