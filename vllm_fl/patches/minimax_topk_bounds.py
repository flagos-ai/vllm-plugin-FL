# SPDX-License-Identifier: Apache-2.0
"""Remove one invalid FlagTree size inference for MiniMax's native sorter.

The pinned AABS analyzer maps BLOCK_SIZE_K to init_blocks, the count of forced
initial blocks. It is not a tensor dimension; zero shrinks the sorting tile to
one and violates K > T. Correct only this kernel's cached dependency metadata.
Native kernel math, launch wrappers and the full autotuning family are retained.
"""
import importlib
import logging

logger = logging.getLogger("vllm.fl.minimax_topk_bounds")


def patch_minimax_topk_bounds(model_config):
    if getattr(model_config, "architecture", None) != "MiniMaxM3SparseForConditionalGeneration":
        return False
    import triton.knobs

    if not getattr(triton.knobs.autotuning, "adjust_block_size", False):
        return False
    native = importlib.import_module("vllm.models.minimax_m3.common.ops.index_topk")
    kernel = native._topk_index_kernel
    if getattr(kernel, "_fl_topk_bounds", False):
        return False
    from triton.runtime.adjust_kernel_param import analyze_kernel_dependencies

    jit = kernel.fn.fn
    load_map = analyze_kernel_dependencies(jit)[0]
    if not isinstance(load_map, dict) or load_map.get("BLOCK_SIZE_K") != "init_blocks":
        return False
    # The returned map is cached by JIT identity and pre-hook. This kernel's
    # native autotuning configs have no pre-hook, so all launches use this map.
    if any(c.pre_hook is not None for c in kernel.fn.configs):
        raise RuntimeError("MiniMax top-k AABS guard requires native configs without pre-hooks")
    del load_map["BLOCK_SIZE_K"]
    kernel._fl_topk_bounds = True
    logger.info("FL_MINIMAX_TOPK_BOUNDS removed invalid BLOCK_SIZE_K -> init_blocks inference")
    return True
