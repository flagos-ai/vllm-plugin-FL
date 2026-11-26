# SPDX-License-Identifier: Apache-2.0
"""Single-device full-attention scheduling for Iluvatar."""
from functools import wraps
from time import monotonic, time

from vllm.logger import init_logger
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import FullAttentionSpec, UniformTypeKVCacheSpecs
from vllm.v1.request import RequestStatus

from .admission import should_defer

# Each experiment binds this constant when staging its independent source tree.
PHASE_STEPS = False
logger = init_logger(__name__)


def full_attention_pool(scheduler):
    groups = scheduler.kv_cache_config.kv_cache_groups
    if not (1 <= len(groups) <= 2):
        return False
    for grp in groups:
        spec = grp.kv_cache_spec
        specs = spec.kv_cache_specs.values() if isinstance(spec, UniformTypeKVCacheSpecs) else (spec,)
        if not all(isinstance(entry, FullAttentionSpec) and entry.block_size == scheduler.block_size
                   and not getattr(entry, "sliding_window", None) and not getattr(entry, "attention_chunk_size", None)
                   for entry in specs):
            return False
    return True


class _IluvatarScheduling:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._iluvatar_policy = (
            full_attention_pool(self) and self.connector is None and self.ec_connector is None
            and self.vllm_config.speculative_config is None and not self.use_pp
            and self.pcp_world_size == self.dcp_world_size == 1
            and self.parallel_config.data_parallel_size == 1)
        self._phase = None
        self._decode_until = 0.
        self._prefill_steps = 0
        self._first_wait = {}
        self._admission_holds = 0
        self._phase_counts = {"prefill": 0, "full_prefill": 0, "decode": 0, "nonuniform_decode": 0, "empty": 0}
        if self._iluvatar_policy:
            logger.info("Iluvatar decode growth reservation active, phase steps=%s, block size=%d, token budget=%d",
                        PHASE_STEPS, self.block_size, self.max_num_scheduled_tokens)
            allocate = self.kv_cache_manager.allocate_slots

            @wraps(allocate)
            def guarded(request, num_new_tokens, *extra, **options):
                # RUNNING allocations have no full_sequence_must_fit argument. A rejected
                # WAITING allocation follows vLLM's ordinary queue/encoder unwind path.
                if "full_sequence_must_fit" in options and request.status in (
                        RequestStatus.WAITING, RequestStatus.PREEMPTED) and self.running:
                    if self._phase == "decode":
                        return None
                    if should_defer(self, request, options):
                        self._admission_holds += 1
                        self._decode_until = monotonic() + .8
                        return None
                return allocate(request, num_new_tokens, *extra, **options)
            self.kv_cache_manager.allocate_slots = guarded

    def _preempt_request(self, request, timestamp):
        if getattr(self, "_iluvatar_policy", False):
            # The base scheduler has already removed the victim from self.running.
            pool = self.kv_cache_manager
            active = [(int(r.num_prompt_tokens), int(r.num_computed_tokens),
                       int(r.num_output_tokens), int(r.max_tokens-r.num_output_tokens),
                       len(pool.get_blocks(r.request_id).blocks[0])) for r in [*self.running,request]]
            record = dict(step=self.current_step, free=pool.block_pool.get_num_free_blocks(),
                          phase=self._phase, victim_computed=request.num_computed_tokens,
                          victim_output=request.num_output_tokens,
                          fields=("prompt","computed","output","remaining","blocks"), active=active)
            logger.info("Iluvatar KV preemption %s", record)
        return super()._preempt_request(request, timestamp)

    def _get_phase(self, now):
        if not self.running:
            self._first_wait.clear()
            self._prefill_steps = 0
            return None
        prefilling = [r for r in self.running if r.num_computed_tokens < r.num_prompt_tokens]
        decode = len(self.running) - len(prefilling)
        if not decode:
            return None
        queue = self._select_waiting_queue_for_scheduling()
        head = queue.peek_request() if queue is not None else None
        if head is None:
            self._first_wait.clear()
        elif head.request_id not in self._first_wait:
            self._first_wait = {head.request_id: now}
        # Short prompts can complete together with decodes using the ordinary scheduler.
        long_inflight = any(r.num_prompt_tokens >= self.max_num_scheduled_tokens for r in prefilling)
        long_waiting = head is not None and head.num_prompt_tokens >= self.max_num_scheduled_tokens
        if not long_inflight and not long_waiting:
            return None
        if now < self._decode_until or self._prefill_steps >= 8:
            return "decode"
        if long_inflight:
            return "prefill"
        pending = len(self.waiting) + len(self.skipped_waiting)
        # Use the configured capture sizes as read-only efficiency boundaries.
        sizes = self.vllm_config.compilation_config.cudagraph_capture_sizes or ()
        below_capture = any(size == decode + 1 for size in sizes)
        waited = max(now - self._first_wait[head.request_id], time() - head.arrival_time)
        if pending >= 4 or waited >= 1.0 or below_capture:
            return "prefill"
        return "decode"

    def schedule(self, throttle_prefills=False):
        if not self._iluvatar_policy or throttle_prefills:
            return super().schedule(throttle_prefills)
        phase = self._get_phase(monotonic()) if PHASE_STEPS else None
        self._phase = phase
        held = []
        try:
            if phase is not None:
                eligible_after = self.current_step + 2
                for request in self.running:
                    prefill = request.num_computed_tokens < request.num_prompt_tokens
                    if prefill != (phase == "prefill") and request.next_decode_eligible_step < eligible_after:
                        held.append((request, request.next_decode_eligible_step))
                        request.next_decode_eligible_step = eligible_after
            result = super().schedule(throttle_prefills)
        finally:
            for request, original in held:
                request.next_decode_eligible_step = original
            self._phase = None
        if phase == "prefill":
            self._prefill_steps += 1
            if result.total_num_scheduled_tokens == 0:
                self._decode_until = monotonic() + .8
        else:
            self._prefill_steps = 0
        if phase is not None:
            self._phase_counts[phase] += 1
            if result.total_num_scheduled_tokens == 0:
                self._phase_counts["empty"] += 1
            elif phase == "prefill" and result.total_num_scheduled_tokens == self.max_num_scheduled_tokens:
                self._phase_counts["full_prefill"] += 1
            elif phase == "decode" and result.total_num_scheduled_tokens != len(result.num_scheduled_tokens):
                self._phase_counts["nonuniform_decode"] += 1
        if self.current_step % 1024 == 0:
            logger.info("Iluvatar scheduler step %d, admission holds=%d, phases=%s",
                        self.current_step, self._admission_holds, self._phase_counts)
        return result


class IluvatarScheduler(_IluvatarScheduling, Scheduler):
    pass


class IluvatarAsyncScheduler(_IluvatarScheduling, AsyncScheduler):
    pass
