# SPDX-License-Identifier: Apache-2.0
"""Reserve decode growth when a new request would take a full KV pool above 90%."""


from .horizon import projected_headroom


def configure_scheduler(config):
    # The locked VllmConfig resolves async scheduling before platform config hooks.
    if config.scheduler_cls is None and config.async_scheduling is not None:
        name = "IluvatarAsyncScheduler" if config.async_scheduling else "IluvatarScheduler"
        config.scheduler_cls = "vllm_fl.dispatch.backends.vendor.iluvatar.kv_scheduler." + name


def headroom_blocks(scheduler):
    manager = scheduler.kv_cache_manager
    page = scheduler.block_size
    total = 0
    for active in scheduler.running:
        # Count the physical blocks actually owned, including unused room in the last page.
        allocated = len(manager.get_blocks(active.request_id).blocks[0])
        growth = max(0, min(512, active.max_tokens - active.num_output_tokens,
                            manager.max_model_len - active.num_tokens))
        future_blocks = (active.num_tokens + growth + page - 1) // page
        # A chunked prompt can still own only its first few blocks. Its unscheduled
        # input is distinct from the decode growth allowance.
        total += min((growth + page - 1) // page, max(0, future_blocks - allocated))
    return total


def should_defer(scheduler, request, options):
    if not scheduler.running:
        return False
    manager = scheduler.kv_cache_manager
    pool = manager.block_pool
    capacity = pool.num_gpu_blocks - 1  # The block pool retains one null block.
    available = pool.get_num_free_blocks() - options.get("reserved_blocks", 0)
    full = min(request.num_tokens, manager.max_model_len)
    page = scheduler.block_size
    upper = (full + page - 1) // page
    if (capacity - available + upper) * 10 <= capacity * 9:
        return False
    # Credit shared prefixes and already-owned recomputation blocks exactly as vLLM does.
    cached = options.get("new_computed_blocks")
    blocks = cached.blocks if cached is not None else manager.empty_kv_cache_blocks.blocks
    computed = min(manager.max_model_len, request.num_computed_tokens
                   + options.get("num_new_computed_tokens", 0)
                   + options.get("num_external_computed_tokens", 0))
    needed = manager.coordinator.get_num_blocks_to_allocate(
        request_id=request.request_id, num_tokens=full, new_computed_blocks=blocks,
        num_encoder_tokens=options.get("num_encoder_tokens", 0), total_computed_tokens=computed,
        num_tokens_main_model=full, apply_admission_cap=True)
    if (capacity - available + needed) * 10 <= capacity * 9:
        return False
    return available - needed < projected_headroom(scheduler, request)
