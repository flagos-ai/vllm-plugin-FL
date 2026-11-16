# Copyright (c) 2026 BAAI. All rights reserved.

"""Native tensor copies for TXDA's mamba state-cache maintenance.

Upstream's ``do_mamba_copy_block`` feeds *raw device pointers* (collected on
the host) to the ``batch_memcpy`` Triton kernel.  On txda that raw-pointer
kernel path is what crashes during warmup/decode with
``HPGR: deviceErrno 0x6000002 (KCORE internal error)`` — once the device
context is poisoned every subsequent op fails with
``TXDA error: (GetDevice) = Device failed``.

This patch keeps upstream's metadata collection (the np pointer buffers stay
filled exactly as upstream wrote them, so any other reader is unaffected) and
instead:

1. Wraps ``collect_mamba_copy_meta`` to also record, per entry, the owning
   state tensor, so the copy can later be performed with native
   ``Tensor.copy_`` (a real TXDA kernel) instead of the raw-pointer Triton
   kernel.  Each recorded pointer range is validated up front — element
   alignment, non-negative element-aligned size, and containment inside the
   state's backing storage — so an out-of-range pointer raises on the host
   instead of faulting the device.  On rejection the buffer offset is
   restored so the batch stays all-or-nothing.
2. Replaces ``do_mamba_copy_block`` with a loop over the recorded plan that
   executes each copy natively and clears both plan and offset.

Overlapping copies clone the source first, so the copy is correct regardless
of the direction of the overlap.

Regressions: ``tests/unit_tests/attention/test_txda_mamba_memcpy.py``.
"""

from __future__ import annotations

import functools

import torch

# Attribute set on MambaCopyBuffers to carry the native copy plan between
# collect (preprocess) and execute (do_mamba_copy_block) phases.
_PLAN_ATTRIBUTE = "_fl_native_copy_plan"


def _flat_storage_view(state: torch.Tensor) -> torch.Tensor:
    """1-D tensor over the whole backing storage of ``state``.

    ``state`` may be a strided view (mamba cache slices are), so
    ``state.view(-1)`` is not generally legal; rebuilding from the storage
    keeps the resolved pointers valid for any layout.
    """
    storage = state.untyped_storage()
    element_size = state.element_size()
    numel = storage.nbytes() // element_size
    return torch.empty(0, dtype=state.dtype, device=state.device).set_(
        storage, 0, (numel,), (1,)
    )


def _state_view(state: torch.Tensor, start_addr: int, num_bytes: int):
    """Resolve a raw pointer + byte length into a validated tensor view.

    The pointer must be element-aligned and the whole range must lie inside
    the state's backing storage; anything else would make the native copy
    read/write out of bounds exactly like the raw-pointer kernel it replaces.
    """
    storage = state.untyped_storage()
    base_addr = storage.data_ptr()
    storage_nbytes = storage.nbytes()
    element_size = state.element_size()
    if num_bytes < 0 or num_bytes % element_size != 0:
        raise ValueError(
            f"copy size is not element-aligned: {num_bytes} bytes with "
            f"element size {element_size}"
        )
    if start_addr % element_size != 0:
        raise ValueError(f"copy start address is not element-aligned: 0x{start_addr:x}")
    if start_addr < base_addr or start_addr + num_bytes > base_addr + storage_nbytes:
        raise ValueError(
            f"copy range [0x{start_addr:x}, 0x{start_addr + num_bytes:x}) is "
            f"outside state storage [0x{base_addr:x}, "
            f"0x{base_addr + storage_nbytes:x})"
        )
    flat = _flat_storage_view(state)
    offset = (start_addr - base_addr) // element_size
    return flat[offset : offset + num_bytes // element_size]


def _make_collect_and_plan(wrapped_collect):
    def _collect_and_plan(*args, **kwargs):
        """Run upstream metadata collection, then record owning tensors.

        Maintains the invariant ``len(plan) == copy_bufs.offset``: a fresh
        batch (offset reset to 0 by the caller) clears the plan; entries are
        appended in exactly the order upstream wrote the pointer buffers.
        """
        copy_bufs = args[0]
        offset_before = copy_bufs.offset

        plan = getattr(copy_bufs, _PLAN_ATTRIBUTE, None)
        if plan is None:
            plan = []
            setattr(copy_bufs, _PLAN_ATTRIBUTE, plan)
        if offset_before == 0:
            # A new batch started (preprocess resets offset directly); drop
            # any plan left over from a batch that was never executed.
            del plan[:]
        plan_baseline = len(plan)
        if plan_baseline != offset_before:
            # Should not happen — offset and plan drifted apart. Refuse
            # rather than copy the wrong tensors.
            raise RuntimeError(
                f"txda native mamba copy: plan length {plan_baseline} != "
                f"buffer offset {offset_before}"
            )

        (
            _bufs,
            kv_cache_config,
            _mamba_state_copy_funcs,
            mamba_group_ids,
            _src_block_idx,
            dest_block_idx,
            accept_token_bias,
            req_state,
            forward_context,
        ) = args

        # Mirror upstream's iteration order so plan[i] owns pointer entry i.
        if not (_src_block_idx == dest_block_idx and accept_token_bias == 0):
            for mamba_group_id in mamba_group_ids:
                block_ids = req_state.block_ids[mamba_group_id]
                dest_block_id = block_ids[dest_block_idx]
                kv_cache_group = kv_cache_config.kv_cache_groups[mamba_group_id]
                for layer_name in kv_cache_group.layer_names:
                    for state in forward_context[layer_name].kv_cache:
                        plan.append(state)

        wrapped_collect(*args, **kwargs)

        # Validate the newly recorded entries before any copy can run,
        # restoring the offset on rejection so a bad batch is all-or-nothing.
        src_ptrs_np = copy_bufs.src_ptrs.np
        dst_ptrs_np = copy_bufs.dst_ptrs.np
        sizes_np = copy_bufs.sizes.np
        try:
            for i in range(offset_before, copy_bufs.offset):
                num_bytes = int(sizes_np[i])
                if num_bytes == 0:
                    continue
                state = plan[i]
                _state_view(state, int(src_ptrs_np[i]), num_bytes)
                _state_view(state, int(dst_ptrs_np[i]), num_bytes)
        except ValueError:
            del plan[plan_baseline:]
            copy_bufs.offset = offset_before
            raise

    return _collect_and_plan


def _do_mamba_copy_block_native(copy_bufs) -> None:
    """Execute the recorded plan with native tensor copies."""
    n = copy_bufs.offset
    if n == 0:
        return

    plan = getattr(copy_bufs, _PLAN_ATTRIBUTE, None)
    if plan is None:
        raise RuntimeError(
            "txda native mamba copy: copy_bufs does not carry an owning "
            "tensor plan; collect_mamba_copy_meta was not patched in this "
            "process"
        )

    src_ptrs_np = copy_bufs.src_ptrs.np
    dst_ptrs_np = copy_bufs.dst_ptrs.np
    sizes_np = copy_bufs.sizes.np

    for i in range(n):
        src_addr = int(src_ptrs_np[i])
        dst_addr = int(dst_ptrs_np[i])
        num_bytes = int(sizes_np[i])
        if num_bytes == 0 or src_addr == dst_addr:
            continue

        state = plan[i]
        src_view = _state_view(state, src_addr, num_bytes)
        dst_view = _state_view(state, dst_addr, num_bytes)
        if src_addr < dst_addr + num_bytes and dst_addr < src_addr + num_bytes:
            src_view = src_view.clone()
        dst_view.copy_(src_view)

    # Upstream's caller resets ``copy_bufs.offset = 0`` at the start of the
    # next batch; keep the same contract (the plan tracks it via the
    # offset==0 branch in _collect_and_plan, so a stale plan is dropped then).
    del plan[:]


def patch_mamba_state_copy() -> None:
    """Apply the native-copy patch to vllm.v1.worker.mamba_utils.

    Idempotent: a second call keeps the already-patched functions and the
    upstream ``MambaCopyBuffers.create`` descriptor untouched.

    The import is deferred so unit tests can substitute a stub module into
    ``sys.modules`` before loading this file.
    """
    import sys

    mamba = sys.modules["vllm.v1.worker.mamba_utils"]

    if getattr(mamba.collect_mamba_copy_meta, "_fl_txda_native", False):
        return

    mamba._fl_upstream_collect = mamba.collect_mamba_copy_meta
    mamba.collect_mamba_copy_meta = _make_collect_and_plan(
        mamba.collect_mamba_copy_meta
    )
    mamba.collect_mamba_copy_meta._fl_txda_native = True
    # functools.wraps-style marker so callers (and tests) can reach the
    # upstream collector through the wrapper.
    mamba.collect_mamba_copy_meta.__wrapped__ = mamba._fl_upstream_collect
    mamba.do_mamba_copy_block = _do_mamba_copy_block_native


patch_mamba_state_copy()
