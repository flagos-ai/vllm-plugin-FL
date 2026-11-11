"""Contract coverage for instance-local Iluvatar PP transport."""

from types import SimpleNamespace

import pytest
import torch

from vllm_fl.dispatch.backends.vendor.iluvatar import pipeline


@pytest.mark.parametrize("split", [False, True])
def test_cpu_metadata_empty_and_gather(monkeypatch, split):
    mailbox = {}
    source = torch.arange(8).reshape(2, 4)
    group = SimpleNamespace(
        world_size=2,
        rank_in_group=0,
        ranks=[4, 7],
        cpu_group=object(),
        _should_use_all_gather=lambda *args: split,
        send_object=lambda value, dst: mailbox.update(metadata=value),
        recv_object=lambda src: mailbox["metadata"],
    )
    gather = SimpleNamespace(
        world_size=2,
        rank_in_group=0,
        all_gather=lambda x, dim: torch.cat([x, x], dim=dim),
    )

    def send(tensor, dst, group):
        assert dst == 7
        mailbox["tensor"] = tensor.clone()

    def recv(tensor, src, group):
        assert src == 4
        tensor.copy_(mailbox["tensor"])

    monkeypatch.setattr(torch.distributed, "send", send)
    monkeypatch.setattr(torch.distributed, "recv", recv)
    values = {"x": source, "empty": torch.empty(0), "label": "payload"}
    handles = pipeline._send(group, values, all_gather_group=gather)
    assert handles[0].wait() and handles[0].is_completed()
    group.rank_in_group = 1
    result, handles, callbacks = pipeline._receive(group, all_gather_group=gather)
    assert handles == []
    for callback in callbacks:
        callback()
    expected = torch.cat([source[:1], source[:1]]) if split else source
    assert torch.equal(result["x"], expected)
    assert result["empty"].numel() == 0 and result["label"] == "payload"


def test_single_rank_noop():
    group = SimpleNamespace(world_size=1)
    assert pipeline._send(group, {}) == []
    assert pipeline._receive(group) == (None, [], [])


def test_installer_leaves_other_communicators_unchanged():
    group = SimpleNamespace(world_size=2, device_communicator=object())
    pipeline.install_pipeline_transport(group)
    assert not hasattr(group, "isend_tensor_dict")


def test_installer_is_instance_local_and_idempotent():
    from vllm_fl.dispatch.backends.vendor.iluvatar.communicator import (
        IluvatarCommunicatorFL,
    )

    class Group:
        world_size = 2
        device_communicator = object.__new__(IluvatarCommunicatorFL)

        def isend_tensor_dict(self):
            return "original"

    selected = Group()
    untouched = Group()
    original = Group.isend_tensor_dict
    pipeline.install_pipeline_transport(selected)
    installed = selected.isend_tensor_dict
    assert installed.__self__ is selected
    assert installed.__func__ is pipeline._send
    assert selected.irecv_tensor_dict.__func__ is pipeline._receive
    pipeline.install_pipeline_transport(selected)
    assert selected.isend_tensor_dict is installed
    assert Group.isend_tensor_dict is original
    assert untouched.isend_tensor_dict() == "original"


def test_installer_single_rank_keeps_original():
    from vllm_fl.dispatch.backends.vendor.iluvatar.communicator import (
        IluvatarCommunicatorFL,
    )

    group = SimpleNamespace(
        world_size=1, device_communicator=object.__new__(IluvatarCommunicatorFL)
    )
    pipeline.install_pipeline_transport(group)
    assert not hasattr(group, "isend_tensor_dict")


def test_device_payload_reuses_group_send_recv(monkeypatch):
    from unittest.mock import Mock

    tensor = SimpleNamespace(
        numel=lambda: 4, is_cpu=False, device="cuda", dtype=torch.float32,
        size=lambda: (4,), copy_=Mock(),
    )
    metadata = pipeline.TensorMetadata("cuda", torch.float32, (4,))
    group = SimpleNamespace(
        world_size=2, rank_in_group=0, ranks=[4, 7],
        _should_use_all_gather=lambda *args: False,
        send_object=Mock(), recv_object=lambda src: [("x", metadata)],
        send=Mock(), recv=Mock(return_value=tensor),
    )
    monkeypatch.setattr(pipeline, "_split_tensor_dict", lambda values: ([("x", metadata)], [tensor]))
    monkeypatch.setattr(pipeline, "torch", SimpleNamespace(
        Tensor=SimpleNamespace, empty=lambda *a, **kw: tensor,
        cuda=SimpleNamespace(synchronize=Mock()),
    ))
    pipeline._send(group, {"x": tensor})
    group.send.assert_called_once_with(tensor, dst=1)
    group.rank_in_group = 1
    result, _, _ = pipeline._receive(group)
    group.recv.assert_called_once_with((4,), torch.float32, 0)
    assert result["x"] is tensor
    tensor.copy_.assert_called_once_with(tensor)
