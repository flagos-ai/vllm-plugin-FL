"""Batch-level adaptive prefill policy; no benchmark-dependent inputs."""
BASE_CHUNK = 2048
SHORT_BUDGET = 2 * BASE_CHUNK
LONG_BUDGET = 16 * BASE_CHUNK
LONG_CHUNK = LONG_BUDGET // 2
SHORT_BATCH_BUDGET = LONG_BUDGET // 2

def select_mode(running, waiting, skipped):
    def is_long_prefill(request):
        return request.num_computed_tokens < request.num_prompt_tokens and request.num_prompt_tokens > SHORT_BUDGET
    requests = [r for queue in (running, waiting, skipped) for r in queue
                if r.num_computed_tokens < r.num_prompt_tokens]
    def needs_baseline(request):
        params = getattr(request, 'sampling_params', None)
        return params is None or getattr(params, 'prompt_logprobs', None) is not None
    if any(needs_baseline(r) for r in requests):
        return 'baseline'
    return 'long' if any(is_long_prefill(r) for r in requests) else 'short' 

_MODE_LIMITS = {
    'baseline': (BASE_CHUNK, BASE_CHUNK),
    'long': (LONG_BUDGET, LONG_CHUNK),
    'short': (SHORT_BATCH_BUDGET, 0),
}


def schedule_with_policy(scheduler, original, *args, mode=None, **kwargs):
    if mode is None:
        mode = select_mode(scheduler.running, scheduler.waiting, scheduler.skipped_waiting)
    budget, limit = _MODE_LIMITS[mode]
    old_budget, old_limit = scheduler.max_num_scheduled_tokens, scheduler.scheduler_config.long_prefill_token_threshold
    try:
        scheduler.max_num_scheduled_tokens = budget
        scheduler.scheduler_config.long_prefill_token_threshold = limit
        return original(scheduler, *args, **kwargs)
    finally:
        scheduler.max_num_scheduled_tokens = old_budget
        scheduler.scheduler_config.long_prefill_token_threshold = old_limit


def install():
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.logger import init_logger
    logger = init_logger(__name__)
    if getattr(Scheduler.schedule, "_e388_adaptive", False):
        return
    original = Scheduler.schedule
    def adaptive(self, *args, **kwargs):
        mode = select_mode(self.running, self.waiting, self.skipped_waiting)
        seen = getattr(self, "_e388_seen", set())
        if mode not in seen:
            seen.add(mode); self._e388_seen = seen
            budget, limit = _MODE_LIMITS[mode]
            logger.warning("E388 %s prefill active budget=%d request_cap=%d", mode, budget, limit)
        return schedule_with_policy(self, original, *args, mode=mode, **kwargs)
    adaptive._e388_adaptive = True
    Scheduler.schedule = adaptive

    from vllm.v1.core.block_pool import BlockPool
    if not getattr(BlockPool.cache_full_blocks, "_e466_pyramid_safe", False):
        from flag_gems.fused.minicpm import pyramid_prefill as PP
        orig_cache_full = BlockPool.cache_full_blocks
        _pc_logged = [False]

        def safe_cache_full_blocks(
            self,
            request,
            blocks,
            num_cached_blocks,
            num_full_blocks,
            block_size,
            kv_cache_group_id,
            block_mask=None,
        ):
            params = getattr(request, "sampling_params", None)
            no_prompt_lp = params is not None and getattr(params, "prompt_logprobs", None) is None
            if (
                no_prompt_lp
                and request.num_prompt_tokens >= PP.MIN_PROMPT_TOKENS
                and (num_cached_blocks == 0 or getattr(request, "_pyramid_uncached", False))
            ):
                request._pyramid_uncached = True
                if not _pc_logged[0]:
                    _pc_logged[0] = True
                    logger.warning(
                        "E466 prefix-cache skip for compacted long prompt tokens=%d min_tokens=%d",
                        request.num_prompt_tokens,
                        PP.MIN_PROMPT_TOKENS,
                    )
                return
            return orig_cache_full(
                self,
                request,
                blocks,
                num_cached_blocks,
                num_full_blocks,
                block_size,
                kv_cache_group_id,
                block_mask,
            )

        safe_cache_full_blocks._e466_pyramid_safe = True
        BlockPool.cache_full_blocks = safe_cache_full_blocks
