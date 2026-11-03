# Copyright (c) 2026 BAAI. All rights reserved.
"""Native tensor copies for TXDA's vLLM 0.28 Mamba state boundaries."""

from functools import wraps

_PLAN_ATTRIBUTE = "_vllm_fl_txda_copy_plan"


def _state_view(state, address: int, size_bytes: int):
    """Resolve a copy spec against its owning tensor, including storage padding."""
    element_size = state.element_size()
    storage_byte_offset = (
        state.storage_offset() * element_size + address - state.data_ptr()
    )
    if size_bytes < 0 or size_bytes % element_size:
        raise ValueError("TXDA Mamba copy size is not element-aligned")
    if storage_byte_offset % element_size:
        raise ValueError("TXDA Mamba copy address is not element-aligned")
    if (
        storage_byte_offset < 0
        or storage_byte_offset + size_bytes > state.untyped_storage().nbytes()
    ):
        raise ValueError("TXDA Mamba copy range is outside its tensor storage")
    return state.as_strided(
        (size_bytes // element_size,),
        (1,),
        storage_offset=storage_byte_offset // element_size,
    )


def patch_mamba_state_copy() -> None:
    from vllm.v1.worker import mamba_utils

    original_collect = mamba_utils.collect_mamba_copy_meta
    if getattr(original_collect, "_vllm_fl_txda_native_copy", False):
        return

    @wraps(original_collect)
    def collect_mamba_copy_meta(
        copy_bufs,
        kv_cache_config,
        mamba_state_copy_funcs,
        mamba_group_ids,
        src_block_idx,
        dest_block_idx,
        accept_token_bias,
        req_state,
        forward_context,
    ) -> None:
        offset = copy_bufs.offset
        plan = getattr(copy_bufs, _PLAN_ATTRIBUTE, [])
        if offset == 0:
            plan = []
            setattr(copy_bufs, _PLAN_ATTRIBUTE, plan)
        if offset > len(plan):
            raise RuntimeError("TXDA Mamba copy metadata has no owning tensor plan")

        # Keep upstream callbacks, byte counts, pointer metadata and offsets.
        # Resolve their ranges while the owning tensors are available instead
        # of casting raw GPU addresses inside the unsupported Triton kernel.
        original_collect(
            copy_bufs,
            kv_cache_config,
            mamba_state_copy_funcs,
            mamba_group_ids,
            src_block_idx,
            dest_block_idx,
            accept_token_bias,
            req_state,
            forward_context,
        )
        if copy_bufs.offset == offset:
            return

        entries = []
        try:
            for group_id in mamba_group_ids:
                for layer_name in kv_cache_config.kv_cache_groups[group_id].layer_names:
                    for state, _ in zip(
                        forward_context[layer_name].kv_cache, mamba_state_copy_funcs
                    ):
                        index = offset + len(entries)
                        size = int(copy_bufs.sizes.np[index])
                        if size == 0:
                            entries.append(None)
                            continue
                        src_addr = int(copy_bufs.src_ptrs.np[index])
                        dst_addr = int(copy_bufs.dst_ptrs.np[index])
                        source = _state_view(state, src_addr, size)
                        target = _state_view(state, dst_addr, size)
                        overlap = max(src_addr, dst_addr) < min(
                            src_addr + size, dst_addr + size
                        )
                        entries.append((source, target, overlap))
            if offset + len(entries) != copy_bufs.offset:
                raise RuntimeError("TXDA Mamba copy metadata and tensor plan disagree")
        except Exception:
            copy_bufs.offset = offset
            raise
        setattr(copy_bufs, _PLAN_ATTRIBUTE, plan[:offset] + entries)

    @wraps(mamba_utils.do_mamba_copy_block)
    def do_mamba_copy_block(copy_bufs) -> None:
        count = copy_bufs.offset
        if count == 0:
            setattr(copy_bufs, _PLAN_ATTRIBUTE, [])
            return
        plan = getattr(copy_bufs, _PLAN_ATTRIBUTE, [])
        if count > len(plan):
            raise RuntimeError("TXDA Mamba copy metadata has no owning tensor plan")
        for entry in plan[:count]:
            if entry is not None:
                source, target, overlap = entry
                target.copy_(source.clone() if overlap else source)

    collect_mamba_copy_meta._vllm_fl_txda_native_copy = True
    # ModelRunnerFL imports the module. Upstream preprocess_mamba resolves
    # these names via its module globals, including existing callable aliases.
    mamba_utils.collect_mamba_copy_meta = collect_mamba_copy_meta
    mamba_utils.do_mamba_copy_block = do_mamba_copy_block


patch_mamba_state_copy()
