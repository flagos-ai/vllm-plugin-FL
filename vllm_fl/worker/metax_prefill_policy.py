"""Prefill-first dynamic batching and compacted-prompt prefix-cache guard for MetaX C500."""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

BASE_CHUNK = 2048
SHORT_BATCH_BUDGET = 16384
LONG_BATCH_BUDGET = 32768
SHORT_PROMPT_LIMIT = 8192


def install() -> None:
    from vllm.v1.core.sched.scheduler import Scheduler

    if getattr(Scheduler.schedule, "_e469_metax_policy", False):
        return

    orig_schedule = Scheduler.schedule
    _logged = [False]
    _pf_logged = [False]

    def metax_adaptive_schedule(self, *args, **kwargs):
        waiting_prefills = [
            r for r in self.waiting if r.num_computed_tokens < r.num_prompt_tokens
        ]
        running_prefills = [
            r for r in self.running if r.num_computed_tokens < r.num_prompt_tokens
        ]
        prefills = waiting_prefills + running_prefills
        if not prefills:
            return orig_schedule(self, *args, **kwargs)

        def needs_baseline(r) -> bool:
            p = getattr(r, "sampling_params", None)
            return p is None or getattr(p, "prompt_logprobs", None) is not None

        if any(needs_baseline(r) for r in prefills):
            budget, limit = BASE_CHUNK, BASE_CHUNK
            do_pf_first = False
        elif any(r.num_prompt_tokens > SHORT_PROMPT_LIMIT for r in prefills):
            budget, limit = LONG_BATCH_BUDGET, 0
            do_pf_first = True
        else:
            budget, limit = SHORT_BATCH_BUDGET, 0
            do_pf_first = True

        if not _logged[0]:
            _logged[0] = True
            logger.warning(
                "E469 metax dynamic batching active (short=%d, long=%d, prefill_first=True)",
                SHORT_BATCH_BUDGET,
                LONG_BATCH_BUDGET,
            )

        old_b = self.max_num_scheduled_tokens
        old_l = self.scheduler_config.long_prefill_token_threshold
        # Prefill-first: when waiting prefills exist and there is seq capacity, pause
        # running decode requests for this step so prefills form pure prefill batches
        # (n_dec == 0 -> triggers 3-stage Pyramid Prefill + aligned mm_kernel_nt shapes).
        can_admit = (
            do_pf_first
            and len(waiting_prefills) > 0
            and len(running_prefills) == 0
            and len(self.running) > 0
            and len(self.running) < self.max_num_running_reqs
        )
        try:
            self.max_num_scheduled_tokens = min(old_b, budget) if old_b > 0 else budget
            self.scheduler_config.long_prefill_token_threshold = limit
            if can_admit:
                saved_running = list(self.running)
                self.running.clear()
                try:
                    out = orig_schedule(self, *args, **kwargs)
                    newly_running = list(self.running)
                finally:
                    self.running[:] = saved_running + newly_running if 'newly_running' in locals() else saved_running
                if getattr(out, "total_num_scheduled_tokens", 0) > 0:
                    if not _pf_logged[0]:
                        _pf_logged[0] = True
                        logger.warning(
                            "E469 prefill-first pure batch scheduled tokens=%d paused_decode=%d",
                            out.total_num_scheduled_tokens,
                            len(saved_running),
                        )
                    return out
                # Fallback if KV cache full and no waiting req could be admitted
            return orig_schedule(self, *args, **kwargs)
        finally:
            self.max_num_scheduled_tokens = old_b
            self.scheduler_config.long_prefill_token_threshold = old_l

    metax_adaptive_schedule._e469_metax_policy = True
    Scheduler.schedule = metax_adaptive_schedule

    # Guard BlockPool.cache_full_blocks so compacted pyramid prompts (>= 1024 tokens)
    # are never hashed/published into the prefix cache when enable_prefix_caching=True.
    from vllm.v1.core.block_pool import BlockPool

    if not getattr(BlockPool.cache_full_blocks, "_e469_pyramid_safe", False):
        from vllm_fl.dispatch.backends.vendor.metax.impl.attention import pyramid_prefill as PP

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
                        "E469 prefix-cache skip for compacted long prompt tokens=%d min_tokens=%d",
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

        safe_cache_full_blocks._e469_pyramid_safe = True
        BlockPool.cache_full_blocks = safe_cache_full_blocks
