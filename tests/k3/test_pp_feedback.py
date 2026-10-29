"""CPU checks for PP lifetime and packet boundaries; no GPU arithmetic."""

from collections import deque
from concurrent.futures import Future
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from vllm.v1.worker.gpu import pp_utils

import vllm_fl.distributed.kimi_k3_pp as pp_feedback


def test_host_constructor_does_not_create_unused_gpu_communicator():
    group = Mock(is_last_rank=False, last_rank=3, world_size=4)
    with (
        patch.object(pp_utils, "get_pp_group", return_value=group),
        patch.object(torch.cuda, "current_stream", return_value=Mock()),
        patch.object(torch.cuda, "Stream", return_value=Mock()),
        patch.object(pp_feedback, "_make_feedback_cpu_group", return_value=Mock()),
    ):
        handler = pp_feedback.HostPackedPPHandler(64, 0, torch.device("cpu"))
    try:
        group.make_sibling_device_group.assert_not_called()
        assert list(handler.queue) == [None] * 4
        assert handler.max_sample_len == 1
        assert not handler.req_idx_gen_np.any()
    finally:
        handler.feedback_executor.shutdown(wait=True)


def test_future_failure_propagates_before_waiting_on_unrecorded_event():
    handler = pp_feedback.HostPackedPPHandler.__new__(pp_feedback.HostPackedPPHandler)
    handler.main_stream = Mock()
    failed = Future()
    failed.set_exception(RuntimeError("transport failed"))
    slot = Mock(completion=failed)
    with pytest.raises(RuntimeError, match="transport failed"):
        handler._wait_for_recv(slot)
    handler.main_stream.wait_event.assert_not_called()


def test_completed_receive_waits_on_the_recorded_event():
    handler = pp_feedback.HostPackedPPHandler.__new__(pp_feedback.HostPackedPPHandler)
    handler.main_stream = Mock()
    future = Future()
    future.set_result(None)
    slot = Mock(completion=future)
    handler._wait_for_recv(slot)
    handler.main_stream.wait_event.assert_called_once_with(slot.event)


def test_invalidated_slot_is_skipped_but_transport_still_drains():
    handler = pp_feedback.HostPackedPPHandler.__new__(pp_feedback.HostPackedPPHandler)
    future = Future()
    future.set_exception(RuntimeError("invalidated transfer failed"))
    slot = Mock(
        idx_mapping_np=np.array([0, 1]),
        gen_at_receive_np=np.array([0, 0]),
        need_sampled_mask=np.array([True, True]),
        completion=future,
    )
    handler.queue = deque([slot, None, None, None])
    handler.req_idx_gen_np = np.array([1, 1])
    handler.feedback_executor = Mock()
    handler.feedback_send_futures = deque()
    handler.feedback_recv_futures = deque([future])
    handler.broadcast_stream = Mock()
    assert handler.get_prev_sampled_outputs() is None
    assert len(handler.queue) == 4
    with pytest.raises(RuntimeError, match="invalidated transfer failed"):
        handler.drain_feedback()


@pytest.mark.parametrize("sample_len", [1, 4])
def test_packet_roundtrip_preserves_int64_tokens_and_int32_counters(sample_len):
    tokens = torch.tensor([-(2**63), 2**63 - 1], dtype=torch.int64)
    tokens = tokens[:, None].expand(2, sample_len)
    sampled = torch.tensor([0, sample_len], dtype=torch.int32)
    rejected = torch.tensor([1, 0], dtype=torch.int32)
    packet = pp_feedback._feedback_pack(tokens, sampled, rejected)
    restored = pp_feedback._feedback_views(packet, 2, sample_len)
    for original, actual in zip((tokens, sampled, rejected), restored):
        assert torch.equal(original, actual)
