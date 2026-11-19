# SPDX-License-Identifier: Apache-2.0
"""Shape-aware tiling for FLA's GDN state recurrence (chunk fwd_h) in CP prefill.

vLLM autotunes chunk_gated_delta_rule_fwd_kernel_h_blockdim64 over BV in {32, 64}
keyed only on (H, K, V, BT), so one long sequence with 8 v-heads per TP4 rank
launches ceil(V/32) * N * H = 32 blocks on 132 SMs. BV=16 doubles the grid and
is 23% faster (bit-identical output) while it still fits in one wave; once it
would exceed the SM count (>= 3 sequences here) BV=32 is faster. Only CP
prefill calls this kernel (TP uses FlashInfer GDN, decode uses the recurrent
kernel), and the override is installed only by patch_model().
"""

import os

import torch

ENABLED = os.environ.get("VLLM_FL_FWD_H_TUNE", "1") == "1"


def pick_small_bv(num_seqs: int, heads: int, v_dim: int, num_sms: int) -> bool:
    """True when the BV=16 grid still fits in one wave of the GPU."""
    return -(-v_dim // 16) * num_seqs * heads <= num_sms


def install():
    if not ENABLED:
        return
    import triton

    from vllm.model_executor.layers.fla.ops import chunk, chunk_delta_h

    if getattr(chunk.chunk_gated_delta_rule_fwd_h, "_fl_tuned", False):
        return
    original = chunk_delta_h.chunk_gated_delta_rule_fwd_h
    autotuner = chunk_delta_h.chunk_gated_delta_rule_fwd_kernel_h_blockdim64.fn
    small = [triton.Config({"BV": 16}, num_warps=4, num_stages=4)]
    large = [triton.Config({"BV": 32}, num_warps=4, num_stages=3)]
    sms = {}

    def tuned(k, w, u, *args, cu_seqlens=None, **kwargs):
        device = u.device
        if device not in sms:
            sms[device] = torch.cuda.get_device_properties(device).multi_processor_count
        num_seqs = u.shape[0] if cu_seqlens is None else cu_seqlens.numel() - 1
        saved = autotuner.configs
        # A single config skips autotuning and caching in triton's Autotuner.
        autotuner.configs = (small if pick_small_bv(num_seqs, u.shape[-2], u.shape[-1],
                                                    sms[device]) else large)
        try:
            return original(k, w, u, *args, cu_seqlens=cu_seqlens, **kwargs)
        finally:
            autotuner.configs = saved

    tuned._fl_tuned = True
    chunk.chunk_gated_delta_rule_fwd_h = tuned
