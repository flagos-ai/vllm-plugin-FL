# Copyright (c) 2026 Kunlunxin, Inc. All rights reserved.

# --------------------------------------------------------------------------------
# Hotfix: FlagTree's XPU Triton backend cannot lower an ``scf.if`` that yields a
# tensor once the yielded tensor has 256 or more elements.  The V2 model runner's
# M-RoPE position kernel, ``vllm.v1.worker.gpu.mm.rope.
# _prepare_rope_positions_kernel``, selects its position vector with exactly that
# construct and runs with BLOCK_SIZE=1024::
#
#     if is_prefill:
#         pos = tl.load(prefill_positions_ptr + ... + orig_pos, mask=mask)
#     else:
#         pos = orig_pos + delta
#     tl.store(positions_ptr + ..., pos, mask=mask)
#
# so every V2 run of an M-RoPE model dies during ``profile_run`` with::
#
#     loc(".../vllm/v1/worker/gpu/mm/rope.py":180:20): error: LLVM Translation
#     failed for operation: builtin.unrealized_conversion_cast
#     RuntimeError: failed to translate module to LLVM IR
#
# Measured on P800 (FlagTree triton, /opt/flagtree/triton/backends/xpu), holding
# everything else fixed and varying only BLOCK_SIZE and the select construct:
#
#     construct              BLOCK=64  128   256    1024
#     scf.if yields tensor     OK      OK   FAIL    FAIL
#     tl.where                 OK      OK    OK      OK
#
# The dtype is irrelevant (int32 and int64 stores fail alike), so this is the
# tensor-yielding branch itself, not the widening at the store.
#
# Replace the kernel with a ``tl.where`` formulation.  It is branch-free, so no
# tensor crosses an ``scf.if`` boundary and the module lowers at every block
# size.  Semantics are preserved exactly: the same value is stored for both the
# prefill and the decode case.
#
# The one subtlety is that ``tl.where`` evaluates both operands, so the prefill
# load now also executes on decode steps, where ``orig_pos`` can run past
# ``max_model_len`` and read into the next request's row.  Its mask is therefore
# tightened with ``is_prefill``, which keeps the load inside the bounds the
# original only ever read under the branch.
# --------------------------------------------------------------------------------

import logging

logger = logging.getLogger(__name__)

_PATCH_FLAG = "_vllm_fl_kunlunxin_rope_where"


def _build_kernel():
    """Build the branch-free M-RoPE position kernel."""
    from vllm.triton_utils import tl, triton

    @triton.jit
    def _prepare_rope_positions_kernel(
        positions_ptr,
        positions_stride,
        prefill_positions_ptr,
        prefill_positions_stride0,
        prefill_positions_stride1,
        prefill_delta_ptr,
        idx_mapping_ptr,
        query_start_loc_ptr,
        prefill_lens_ptr,
        num_computed_tokens_ptr,
        BLOCK_SIZE: tl.constexpr,
        NUM_DIMS: tl.constexpr,
    ):
        batch_idx = tl.program_id(0)
        req_state_idx = tl.load(idx_mapping_ptr + batch_idx)

        prefill_len = tl.load(prefill_lens_ptr + req_state_idx)
        num_computed = tl.load(num_computed_tokens_ptr + req_state_idx)
        is_prefill = num_computed < prefill_len

        query_start = tl.load(query_start_loc_ptr + batch_idx)
        query_end = tl.load(query_start_loc_ptr + batch_idx + 1)
        query_len = query_end - query_start

        delta = tl.load(prefill_delta_ptr + req_state_idx)

        for i in range(0, query_len, BLOCK_SIZE):
            block = i + tl.arange(0, BLOCK_SIZE)
            mask = block < query_len
            orig_pos = num_computed + block

            for j in tl.static_range(NUM_DIMS):
                # tl.where evaluates both operands, so gate the load on
                # is_prefill too -- on decode steps orig_pos can exceed
                # prefill_positions_stride1 and spill into the next row.
                prefill_pos = tl.load(
                    prefill_positions_ptr
                    + req_state_idx * prefill_positions_stride0
                    + j * prefill_positions_stride1
                    + orig_pos,
                    mask=mask & is_prefill,
                    other=0,
                )
                pos = tl.where(is_prefill, prefill_pos, orig_pos + delta)
                tl.store(
                    positions_ptr + j * positions_stride + query_start + block,
                    pos,
                    mask=mask,
                )

    return _prepare_rope_positions_kernel


def apply_rope_compat() -> None:
    """Swap in a branch-free M-RoPE position kernel that FlagTree can lower."""
    try:
        from vllm.v1.worker.gpu.mm import rope
    except ImportError:
        # Builds without the V2 runner have nothing to patch.
        return

    current = getattr(rope, "_prepare_rope_positions_kernel", None)
    if current is None or getattr(current, _PATCH_FLAG, False):
        return

    kernel = _build_kernel()
    setattr(kernel, _PATCH_FLAG, True)
    rope._prepare_rope_positions_kernel = kernel
    logger.info(
        "kunlunxin rope_compat: replaced _prepare_rope_positions_kernel with a "
        "tl.where formulation; FlagTree's XPU backend cannot lower an scf.if "
        "that yields a tensor of 256 or more elements"
    )
