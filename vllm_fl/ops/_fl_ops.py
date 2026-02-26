# Copyright (c) 2025 BAAI. All rights reserved.

import torch
import flag_gems

class FLOps:

    ### activation
    @staticmethod
    def silu_and_mul(x):
        d = x.shape[-1] // 2
        x1, x2 = x[..., :d], x[..., d:]
        return flag_gems.modules.activation.gems_silu_and_mul(x1, x2)
    
    @staticmethod
    def gelu_and_mul(x, approximate="none"):
        d = x.shape[-1] // 2
        x1, x2 = x[..., :d], x[..., d:]
        return flag_gems.fused.gelu_and_mul(x1, x2, approximate)

    ### moe 
    @staticmethod
    def topk_softmax(topk_weights, topk_indices, token_expert_indices, gating_output, renormalize=False):
        flag_gems.topk_softmax(
            topk_weights,
            topk_indices,
            token_expert_indices,
            gating_output,
            renormalize,
        )
        return topk_weights, topk_indices
    
    @staticmethod
    def moe_sum(input, output):
        flag_gems.moe_sum(input, output)

    @staticmethod
    def moe_align_block_size(topk_ids, num_experts, block_size, sorted_ids,
                             expert_ids, num_tokens_post_pad,):
        flag_gems.moe_align_block_size_triton(topk_ids, num_experts, block_size, sorted_ids,
                             expert_ids, num_tokens_post_pad,)