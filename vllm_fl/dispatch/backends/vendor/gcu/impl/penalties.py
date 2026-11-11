# Copyright (c) 2026 BAAI. All rights reserved.

"""Build V2 penalty statistics without unsupported GCU atomic OR.

This runs only when requests are staged, outside model graph replay. Keep
penalty application on device; transfer only the newly staged token rows.
"""

import numpy as np
import torch


def bincount_gcu(
    expanded_idx_mapping,
    all_token_ids,
    prompt_len,
    prefill_len,
    prompt_bin_mask,
    output_bin_counts,
    max_prefill_len,
):
    indices = expanded_idx_mapping.detach().cpu().tolist()
    for req in indices:
        n = int(prefill_len[req].item())
        prompt_n = int(prompt_len[req].item())
        tokens = all_token_ids[req, :n].detach().cpu().numpy().astype(np.int64)
        words = np.zeros(prompt_bin_mask.shape[1], dtype=np.uint32)
        prompt = tokens[:prompt_n]
        np.bitwise_or.at(
            words,
            prompt // 32,
            np.left_shift(np.uint32(1), (prompt % 32).astype(np.uint32)),
        )
        counts = np.bincount(
            tokens[prompt_n:], minlength=output_bin_counts.shape[1]
        ).astype(np.int32)
        prompt_bin_mask[req].copy_(torch.from_numpy(words.view(np.int32)))
        output_bin_counts[req].copy_(torch.from_numpy(counts))


def apply_penalties_gcu_patch() -> bool:
    from vllm.v1.worker.gpu.sample import penalties

    original = penalties.bincount
    if getattr(original, "_vllm_fl_gcu_bincount", False):
        return False

    def bincount(*args, **kwargs):
        mapping = args[0] if args else kwargs["expanded_idx_mapping"]
        if mapping.device.type != "gcu":
            return original(*args, **kwargs)
        return bincount_gcu(*args, **kwargs)

    bincount._vllm_fl_gcu_bincount = True
    penalties.bincount = bincount
    return True
