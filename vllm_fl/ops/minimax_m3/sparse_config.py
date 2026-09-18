# SPDX-License-Identifier: Apache-2.0
"""MetaX launch constraints for vLLM's M3 sparse kernels.

The QK/softmax/PV and index-score kernel bodies remain upstream vLLM code.
Only Triton launch configuration changes. Decode indexing uses the exact
full-score/top-k path in ops.py for query length one.
"""

_INSTALLED = False


def install_safe_launches() -> bool:
    global _INSTALLED
    if _INSTALLED:
        return False

    from vllm.models.minimax_m3.common.ops import index_topk, sparse_attn
    from vllm.triton_utils import triton

    fwd = sparse_attn._gqa_sparse_fwd_kernel
    if not hasattr(fwd, "values") or not hasattr(fwd, "fn"):
        raise RuntimeError("Unsupported vLLM M3 sparse prefill kernel wrapper")
    for name in ("_topk_index_kernel", "_topk_index_partial_kernel"):
        kernel = getattr(index_topk, name)
        if not hasattr(kernel, "fn") or not hasattr(kernel.fn, "configs"):
            raise RuntimeError(f"Unsupported vLLM M3 top-k wrapper: {name}")

    # Preserve every other upstream heuristic. The MetaX matrix lowering
    # requires a minimum 16-row head tile even when TP16 has only four heads.
    values = dict(fwd.values)
    values["BLOCK_SIZE_H"] = lambda args: max(
        16, triton.next_power_of_2(args["gqa_group_size"])
    )
    values["BLOCK_SIZE_QH"] = lambda args: args["BLOCK_SIZE_Q"] * max(
        16, triton.next_power_of_2(args["gqa_group_size"])
    )
    sparse_attn._gqa_sparse_fwd_kernel = triton.heuristics(values)(fwd.fn)
    sparse_attn._SPARSE_ATTN_NUM_STAGES_KWARG = {"num_stages": 1}

    # K64 is a processing tile, not a truncation of the candidate page list.
    for name in ("_topk_index_kernel", "_topk_index_partial_kernel"):
        tuner = getattr(index_topk, name).fn
        tuner.configs = [triton.Config({"BLOCK_SIZE_K": 64}, num_warps=2, num_stages=1)]
        tuner.cache.clear()
    index_topk._index_block_score_kernel = triton.heuristics(
        {"num_stages": lambda args: 1}
    )(index_topk._index_block_score_kernel)
    index_topk._decode_index_score_kernel = triton.heuristics(
        {
            "num_stages": lambda args: 1,
            "BLOCK_SIZE_Q": lambda args: max(
                16 // args["num_idx_heads"], args["BLOCK_SIZE_Q"]
            ),
        }
    )(index_topk._decode_index_score_kernel)
    _INSTALLED = True
    return True
