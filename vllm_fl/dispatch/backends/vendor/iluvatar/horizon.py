# SPDX-License-Identifier: Apache-2.0
"""Account for outstanding prefill pages and decode growth to the next release."""


def projected_headroom(scheduler, incoming):
    pool = scheduler.kv_cache_manager
    page = scheduler.block_size
    # Estimate the remaining decode steps before a resident can release pages.
    # A resident still processing its prompt cannot give an earlier deadline.
    active = []
    ready = []
    for request in scheduler.running:
        remaining = max(0, min(request.max_tokens - request.num_output_tokens,
                               pool.max_model_len - request.num_tokens))
        active.append((request, remaining))
        if request.num_computed_tokens >= request.num_prompt_tokens:
            ready.append(remaining)
    short_horizon = 512
    # Cover two ordinary reservation windows at most, including one partial page
    # of uncertainty from async scheduling. The horizon is independent of prompt shape.
    horizon = min(2 * short_horizon, max(short_horizon, min(ready, default=2*short_horizon) + page))
    required = 0
    for request, remaining in active:
        target = request.num_tokens + min(horizon, remaining)
        owned = len(pool.get_blocks(request.request_id).blocks[0])
        required += max(0, (target + page - 1) // page - owned)
    full = min(incoming.num_tokens, pool.max_model_len)
    growth = max(0, min(horizon, incoming.max_tokens - incoming.num_output_tokens,
                        pool.max_model_len - full))
    # The caller already charged the incoming prompt, net of its shared/owned pages.
    required += (full + growth + page - 1) // page - (full + page - 1) // page
    return required
