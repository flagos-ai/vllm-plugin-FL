"""K3 host-packed sampled-token transport with vLLM's unchanged PP FIFO."""

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass

import torch
from vllm.distributed.parallel_state import get_pp_group
from vllm.distributed.utils import get_distributed_timeout_or_none
from vllm.v1.worker.gpu.pp_utils import (
    PPHandler,
    PendingRecv,
    compute_need_sampled_mask,
)


@dataclass
class HostPendingRecv(PendingRecv):
    completion: Future | None = None
    host_packet: torch.Tensor | None = None


class HostPackedPPHandler(PPHandler):
    def _init_broadcast_transport(self) -> None:
        # Run at the original constructor position, without an unused GPU group.
        self.feedback_mode = "host_packed"
        self.feedback_cpu_group = _make_feedback_cpu_group()
        self.feedback_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="k3-pp-feedback"
        )
        self.feedback_send_futures = deque()
        self.feedback_recv_futures = deque()

    def _wait_for_recv(self, slot: HostPendingRecv) -> None:
        if slot.completion is not None:
            slot.completion.result()
        super()._wait_for_recv(slot)

    def receive(self, input_batch):
        assert not self.is_last_rank
        return self._receive_packed(input_batch)

    def broadcast(self, sampled_token_ids, num_sampled, num_rejected, input_batch):
        assert self.is_last_rank
        return self._broadcast_packed(
            sampled_token_ids, num_sampled, num_rejected, input_batch
        )

    @torch.inference_mode()
    def _host_receive(self, packet, host_packet, ready, done):
        torch.cuda.set_device(self.device)
        torch.distributed.broadcast(
            host_packet, src=self.last_rank, group=self.feedback_cpu_group
        )
        with torch.cuda.stream(self.broadcast_stream):
            # ready was recorded by the caller before future graph submissions.
            # Querying main_stream here instead could create a dependency cycle.
            self.broadcast_stream.wait_event(ready)
            packet.copy_(host_packet, non_blocking=True)
            packet.record_stream(self.broadcast_stream)
            done.record(self.broadcast_stream)

    @torch.inference_mode()
    def _host_send(self, host_packet, copied):
        torch.cuda.set_device(self.device)
        copied.synchronize()
        torch.distributed.broadcast(
            host_packet, src=self.last_rank, group=self.feedback_cpu_group
        )

    def _receive_packed(self, input_batch):
        need = compute_need_sampled_mask(input_batch)
        if need is None:
            return False
        n = input_batch.num_reqs
        generation = self.req_idx_gen_np[input_batch.idx_mapping_np].copy()
        packet = torch.empty(
            n * (self.max_sample_len * 8 + 8), dtype=torch.uint8, device=self.device
        )
        tokens, sampled, rejected = _feedback_views(packet, n, self.max_sample_len)
        host_packet = torch.empty(packet.shape, dtype=torch.uint8, pin_memory=True)
        ready = self.main_stream.record_event()
        event = torch.cuda.Event()
        completion = self.feedback_executor.submit(
            self._host_receive, packet, host_packet, ready, event
        )
        self.feedback_recv_futures.append(completion)
        while self.feedback_recv_futures and self.feedback_recv_futures[0].done():
            self.feedback_recv_futures.popleft().result()
        self.queue[-1] = HostPendingRecv(
            event,
            tokens,
            sampled,
            rejected,
            input_batch.idx_mapping,
            input_batch.idx_mapping_np,
            need,
            generation,
            completion,
            host_packet,
        )
        return bool(need.all())

    def _broadcast_packed(self, token_ids, sampled, rejected, input_batch):
        if compute_need_sampled_mask(input_batch) is None:
            return
        packet = _feedback_pack(token_ids, sampled, rejected)
        ready = self.main_stream.record_event()
        with torch.cuda.stream(self.broadcast_stream):
            self.broadcast_stream.wait_event(ready)
            host_packet = torch.empty(packet.shape, dtype=torch.uint8, pin_memory=True)
            host_packet.copy_(packet, non_blocking=True)
            copied = self.broadcast_stream.record_event()
        packet.record_stream(self.broadcast_stream)
        future = self.feedback_executor.submit(self._host_send, host_packet, copied)
        self.feedback_send_futures.append(future)
        while self.feedback_send_futures and self.feedback_send_futures[0].done():
            self.feedback_send_futures.popleft().result()

    def drain_feedback(self):
        """Complete transport work without destroying resources or threads."""
        if self.feedback_executor is not None:
            for future in self.feedback_send_futures:
                future.result()
            # Include transfers whose request slots were invalidated meanwhile.
            for future in self.feedback_recv_futures:
                future.result()
        self.broadcast_stream.synchronize()

    def destroy_feedback(self):
        """Only after the caller has globally drained peers; never time teardown."""
        if self.feedback_executor is not None:
            self.feedback_executor.shutdown(wait=True)
        if self.feedback_cpu_group is not None:
            torch.distributed.destroy_process_group(self.feedback_cpu_group)


def _make_feedback_cpu_group():
    pp = get_pp_group()
    own = None
    for ranks in pp.group_ranks:
        pg = torch.distributed.new_group(
            ranks,
            backend="gloo",
            timeout=get_distributed_timeout_or_none(),
            group_desc="k3_pp_sample_feedback_cpu",
        )
        if pp.rank in ranks:
            own = pg
    assert own is not None
    return own


def _feedback_pack(tokens, sampled, rejected):
    # Byte packing preserves int64 token IDs and int32 counters exactly.
    assert tokens.dtype == torch.int64
    assert sampled.dtype == rejected.dtype == torch.int32
    return torch.cat(
        (
            tokens.contiguous().view(torch.uint8).reshape(-1),
            sampled.contiguous().view(torch.uint8).reshape(-1),
            rejected.contiguous().view(torch.uint8).reshape(-1),
        )
    )


def _feedback_views(packet, num_reqs, max_sample_len):
    token_bytes = num_reqs * max_sample_len * 8
    tokens = packet[:token_bytes].view(torch.int64).view(num_reqs, max_sample_len)
    counters = packet[token_bytes:].view(torch.int32).view(2, num_reqs)
    return tokens, counters[0], counters[1]
