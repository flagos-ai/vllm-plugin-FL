"""Request-generation ownership, independent of persistent batch row indices.

The caller supplies the actual SchedulerOutput and post-update runner request
states, prepares metadata before forward, and commits only after successful
completion of all model layers. This module does not emulate model execution.
"""
from dataclasses import dataclass

import triton
import triton.language as tl


@dataclass
class RequestLease:
    request_id: str
    slot: int
    generation: int
    committed_length: int
    physical_blocks: tuple


@dataclass(frozen=True)
class OwnershipPlan:
    slots: tuple
    generations: tuple
    starts: tuple
    counts: tuple
    reset_slots: tuple
    active_requests: tuple
    pending_commits: tuple


class RequestOwnership:
    def __init__(self, capacity, block_size=16):
        if type(capacity) is not int or capacity < 1 or block_size != 16:
            raise ValueError('positive slot capacity and block16 required')
        self.capacity = capacity
        self.block_size = block_size
        self.leases = {}
        self.free = list(range(capacity))
        self.slot_generations = [0]*capacity
        self.pending = None

    def _release(self, request_id):
        lease = self.leases.pop(request_id, None)
        if lease is not None:
            self.free.append(lease.slot)
            self.free.sort()

    def prepare(self, scheduler_output, requests, batch_request_ids, *, padded_rows):
        # Preparation validates the whole batch before the caller can launch
        # GPU mutation. Validation failures must not consume a slot/generation.
        before = self.leases.copy(), self.free.copy(), self.slot_generations.copy(), self.pending
        try:
            return self._prepare(scheduler_output, requests, batch_request_ids, padded_rows=padded_rows)
        except Exception:
            self.leases, self.free, self.slot_generations, self.pending = before
            raise

    def _prepare(self, scheduler_output, requests, batch_request_ids, *, padded_rows):
        if self.pending is not None:
            raise RuntimeError('prior model execution has not committed ownership state')
        if any(scheduler_output.scheduled_spec_decode_tokens.values()):
            raise ValueError('speculative rollback requires a separate committed-tail contract')
        if scheduler_output.kv_connector_metadata is not None:
            raise ValueError('external KV transfer requires original-tail transfer metadata')
        ids = list(batch_request_ids)
        if len(ids) != len(set(ids)) or padded_rows < len(ids) or padded_rows > self.capacity:
            raise ValueError('batch coverage/unique identity/padding capacity mismatch')
        counts_by_id = scheduler_output.num_scheduled_tokens
        if set(ids) != {rid for rid, count in counts_by_id.items() if count > 0}:
            raise ValueError('post-update batch must cover all and only scheduled requests')
        if sum(counts_by_id.values()) != scheduler_output.total_num_scheduled_tokens:
            raise ValueError('scheduler total does not match per-request tokens')
        fresh = {x.req_id for x in scheduler_output.scheduled_new_reqs}
        resumed = set(scheduler_output.scheduled_cached_reqs.resumed_req_ids)
        finished = set(scheduler_output.finished_req_ids)
        preempted = set(scheduler_output.preempted_req_ids or ())
        if preempted & set(ids):
            raise ValueError('preempted request is still scheduled in same step')
        # Finish must precede admitting a same-ID replacement. Ordinary pause
        # has neither event and retains its slot and exact per-layer tail.
        for rid in finished | preempted | resumed:
            self._release(rid)
        slots, generations, starts, counts, resets, commits = [], [], [], [], [], []
        writable = set()
        for rid in ids:
            state = requests[rid]
            start, count = int(state.num_computed_tokens), int(counts_by_id[rid])
            if start < 0 or count <= 0 or len(state.block_ids) != 1:
                raise ValueError('single KV group/nonnegative start/positive count required')
            blocks = tuple(state.block_ids[0])
            end = start+count
            needed = (end+15)//16
            if len(blocks) < needed or any(type(x) is not int or x <= 0 for x in blocks[:needed]):
                raise ValueError('physical block coverage is invalid')
            if len(set(blocks[:needed])) != needed:
                raise ValueError('one request aliases its logical pages')
            lease = self.leases.get(rid)
            if lease is None:
                if rid not in fresh | resumed:
                    raise ValueError('missing lease without a new or resumed request event')
                if start % 16:
                    raise ValueError('new/resumed partial prefix lacks exact BF16 tail; no INT8 reconstruction')
                if not self.free:
                    raise RuntimeError('request tail slot capacity exhausted; paused leases cannot be evicted')
                slot = self.free.pop(0)
                self.slot_generations[slot] += 1
                lease = RequestLease(rid, slot, self.slot_generations[slot], start, blocks[:(start+15)//16])
                self.leases[rid] = lease
                resets.append(slot)
            else:
                if start != lease.committed_length:
                    raise ValueError('continuation length differs from committed tail generation')
                prior_pages = (start+15)//16
                if blocks[:prior_pages] != lease.physical_blocks[:prior_pages]:
                    raise ValueError('existing physical prefix changed without resumed/rebuild event')
            for page in blocks[start//16:needed]:
                if page in writable:
                    raise ValueError('two active writers target the same physical page')
                writable.add(page)
            slots.append(lease.slot)
            generations.append(lease.generation)
            starts.append(start)
            counts.append(count)
            commits.append((rid, lease.slot, lease.generation, end, blocks[:needed]))
        # A writer may not modify a block referenced as history by another
        # active request, even when that other request writes a different tail.
        for rid in ids:
            start = int(requests[rid].num_computed_tokens)
            full_history = set(requests[rid].block_ids[0][:start//16])
            own_writes = set(requests[rid].block_ids[0][start//16:(start+counts_by_id[rid]+15)//16])
            if (writable-own_writes) & full_history:
                raise ValueError('writer would overwrite another active request shared prefix')
        padding = padded_rows-len(ids)
        plan = OwnershipPlan(tuple(slots+[-1]*padding), tuple(generations+[0]*padding),
                             tuple(starts+[0]*padding), tuple(counts+[0]*padding),
                             tuple(resets), tuple(ids), tuple(commits))
        self.pending = plan
        return plan

    def commit(self, plan):
        if self.pending is not plan:
            raise RuntimeError('commit does not match the pending model execution')
        for rid, slot, generation, end, blocks in plan.pending_commits:
            lease = self.leases[rid]
            if (lease.slot, lease.generation) != (slot, generation):
                raise RuntimeError('stale request generation at commit')
            lease.committed_length = end
            lease.physical_blocks = blocks
        self.pending = None


@triton.jit
def _reset_layer_slot_state(Stage, Banks, ResetSlots, SLOTS: tl.constexpr,
                            WORDS: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0)
    layer = tl.program_id(1)
    chunk = tl.program_id(2)
    slot = tl.load(ResetSlots+index)
    if slot >= 0 and slot < SLOTS:
        offsets = chunk*BLOCK+tl.arange(0, BLOCK)
        tl.store(Stage+(layer*SLOTS+slot)*WORDS+offsets, 0, mask=offsets < WORDS)
        if chunk == 0:
            tl.store(Banks+layer*SLOTS+slot, 0)


def reset_layer_slot_state(stages, banks, reset_slots):
    if stages.ndim != 6 or stages.shape[2:] != (2, 16, 2, 128):
        raise ValueError('stages must be [layer,slot,2,16,2,128]')
    if banks.shape != stages.shape[:2] or not stages.is_contiguous() or not banks.is_contiguous():
        raise ValueError('bank/stage shape or contiguity mismatch')
    if reset_slots.numel():
        _reset_layer_slot_state[(reset_slots.numel(), stages.shape[0], triton.cdiv(8192, 256))](
            stages, banks, reset_slots, stages.shape[1], 8192, 256, num_warps=4, num_stages=1)
