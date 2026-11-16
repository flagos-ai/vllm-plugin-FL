# Copyright (c) 2026 BAAI. All rights reserved.

"""MetaX FlagCX handoff and native CUDA graph bridge contracts."""

from unittest.mock import MagicMock

import pytest

pytest.importorskip("vllm")

from vllm.distributed.device_communicators.base_device_communicator import (  # noqa: E402
    DeviceCommunicatorBase,
)
from vllm.distributed.device_communicators.cuda_communicator import (  # noqa: E402
    CudaCommunicator,
)

import vllm_fl.distributed.communicator as communicator_module  # noqa: E402
import vllm_fl.distributed.metax_communicator as metax_communicator_module  # noqa: E402

CommunicatorFL = communicator_module.CommunicatorFL
MetaxCommunicatorFL = metax_communicator_module.MetaxCommunicatorFL


@pytest.fixture
def mocked_base_init(monkeypatch):
    def fake_base_init(self, cpu_group, device=None, *args, **kwargs):
        self.cpu_group = cpu_group
        self.device = device
        self.world_size = 1
        self.use_all2all = False

    monkeypatch.setattr(DeviceCommunicatorBase, "__init__", fake_base_init)


def test_metax_bridge_satisfies_graph_contract(mocked_base_init):
    communicator = MetaxCommunicatorFL(cpu_group=object())

    assert isinstance(communicator, CudaCommunicator)
    assert communicator.ca_comm is None
    assert communicator.pynccl_comm is None
    assert communicator.pyflagcx_comm is None
    assert MetaxCommunicatorFL.broadcast is CommunicatorFL.broadcast


def test_metax_bridge_enters_native_graph_capture(monkeypatch, mocked_base_init):
    from contextlib import nullcontext
    from types import SimpleNamespace

    import torch

    from vllm.distributed.parallel_state import GroupCoordinator

    communicator = MetaxCommunicatorFL(cpu_group=object())
    coordinator = SimpleNamespace(device_communicator=communicator)
    stream = object()
    capture_context = SimpleNamespace(stream=stream)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: stream)
    monkeypatch.setattr(torch.cuda, "stream", lambda _: nullcontext())

    # Exercise upstream's attribute reads instead of copying its expected
    # field list. vLLM 0.32 also consults fi_pcie_ipc_ar_comm during capture.
    with GroupCoordinator.graph_capture(coordinator, capture_context) as context:
        assert context is capture_context
    assert communicator.pyflagcx_comm is None


def test_metax_bridge_keeps_flagcx_as_only_device_communicator(
    monkeypatch, mocked_base_init
):
    from types import SimpleNamespace

    base_init = DeviceCommunicatorBase.__init__

    def two_rank_base(self, *args, **kwargs):
        base_init(self, *args, **kwargs)
        self.world_size = 2

    native = MagicMock(disabled=False)
    native.all_reduce.return_value = expected = object()
    monkeypatch.setattr(DeviceCommunicatorBase, "__init__", two_rank_base)
    monkeypatch.setattr(
        communicator_module, "PyFlagcxCommunicator", lambda **kwargs: native
    )
    monkeypatch.setattr(
        CudaCommunicator,
        "__init__",
        lambda *args, **kwargs: pytest.fail("second CUDA communicator initialized"),
    )
    communicator = MetaxCommunicatorFL(
        cpu_group=object(), device=SimpleNamespace(type="cuda")
    )
    value = object()
    assert communicator.all_reduce(value) is expected
    native.all_reduce.assert_called_once_with(value)
    assert communicator.pyflagcx_comm is native
    assert communicator.pynccl_comm is None
    assert communicator.ca_comm is None


@pytest.fixture
def metax_handoff(monkeypatch):
    from types import SimpleNamespace

    import torch

    events = []
    calls = []
    stream = SimpleNamespace(synchronize=lambda: events.append("sync"))
    native = SimpleNamespace(
        disabled=False, _current_stream=MagicMock(return_value=stream)
    )
    communicator = MetaxCommunicatorFL.__new__(MetaxCommunicatorFL)
    communicator.device = torch.device("cuda:0")
    communicator.world_size = 2
    communicator.pyflagcx_comm = native
    value = MagicMock(spec=torch.Tensor)
    value.device = communicator.device
    output = object()
    capture = MagicMock(return_value=False)
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: False)
    monkeypatch.setattr(
        metax_communicator_module,
        "current_platform",
        SimpleNamespace(
            torch_device_fn=SimpleNamespace(is_current_stream_capturing=capture)
        ),
    )

    def all_reduce(self, input_):
        assert self is communicator
        events.append("ar")
        calls.append(("ar", input_))
        return output

    def all_gather(self, input_, dim=-1):
        assert self is communicator
        events.append(("ag", dim))
        calls.append(("ag", input_, dim))
        return output

    monkeypatch.setattr(CommunicatorFL, "all_reduce", all_reduce)
    monkeypatch.setattr(CommunicatorFL, "all_gather", all_gather)
    return SimpleNamespace(
        comm=communicator,
        value=value,
        output=output,
        native=native,
        stream=stream,
        capture=capture,
        events=events,
        calls=calls,
        module=metax_communicator_module,
    )


def test_metax_capture_history_marks_only_successful_capture(metax_handoff):
    h = metax_handoff
    assert h.comm.all_reduce(h.value) is h.output
    assert not h.comm._flagcx_capture_seen
    assert h.comm.all_gather(h.value, 0) is h.output
    assert h.events == ["ar", ("ag", 0)]
    assert h.calls[0][1] is h.value and h.calls[1][1] is h.value
    h.native._current_stream.assert_not_called()

    h.capture.return_value = True
    assert h.comm.all_reduce(h.value) is h.output
    assert h.comm._flagcx_capture_seen
    other = MetaxCommunicatorFL.__new__(MetaxCommunicatorFL)
    assert not other._flagcx_capture_seen
    assert h.comm.all_gather(h.value) is h.output
    h.native._current_stream.assert_not_called()


def test_metax_handoff_uses_actual_flagcx_stream_before_ag(metax_handoff):
    h = metax_handoff
    h.capture.return_value = True
    h.comm.all_reduce(h.value)
    h.capture.return_value = False
    h.events.clear()

    # History remains conservative across calls without another Python AR.
    # Native graph replay itself never enters this Python communicator method.
    for dim in (-1, 0):
        assert h.comm.all_gather(h.value, dim) is h.output
    assert h.events == ["sync", ("ag", -1), "sync", ("ag", 0)]
    assert h.native._current_stream.call_count == 2
    assert h.comm._flagcx_capture_seen


@pytest.mark.parametrize(
    "excluded",
    [
        "compiler",
        "cpu",
        "meta",
        "fake",
        "not_tensor",
        "world1",
        "disabled",
        "missing_native",
        "wrong_device",
    ],
)
def test_metax_handoff_exclusions_do_not_query_capture(
    monkeypatch, metax_handoff, excluded
):
    import torch
    from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

    h = metax_handoff
    h.comm._flagcx_capture_seen = True
    if excluded == "compiler":
        monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    elif excluded in ("cpu", "meta"):
        h.value = torch.empty(2, device=excluded)
    elif excluded == "fake":
        # CUDA FakeTensor construction may initialize a real GPU context.
        # Its local fake-device metadata is sufficient to test this exclusion.
        h.value = FakeTensor(
            FakeTensorMode(), torch.empty(2, device="meta"), torch.device("cpu")
        )
        h.value.fake_device = h.comm.device
        assert h.value.device.type == "cuda"
    elif excluded == "not_tensor":
        h.value = object()
    elif excluded == "world1":
        h.comm.world_size = 1
    elif excluded == "disabled":
        h.native.disabled = True
    elif excluded == "missing_native":
        h.comm.pyflagcx_comm = None
    elif excluded == "wrong_device":
        h.value.device = torch.device("cuda:1")
    h.capture.side_effect = lambda: pytest.fail("excluded path queried CUDA capture")

    assert h.comm.all_reduce(h.value) is h.output
    assert h.comm.all_gather(h.value, 0) is h.output
    assert h.events == ["ar", ("ag", 0)]
    h.capture.assert_not_called()
    h.native._current_stream.assert_not_called()


def test_metax_failed_captured_ar_does_not_record_history(monkeypatch, metax_handoff):
    h = metax_handoff
    h.capture.return_value = True

    def fail_ar(self, value):
        raise RuntimeError("native AR failed")

    monkeypatch.setattr(CommunicatorFL, "all_reduce", fail_ar)
    with pytest.raises(RuntimeError, match="native AR failed"):
        h.comm.all_reduce(h.value)
    assert not h.comm._flagcx_capture_seen
    h.native._current_stream.assert_not_called()


@pytest.mark.parametrize("failure", ["capture_query", "sync", "ag"])
def test_metax_handoff_errors_propagate_without_false_success(
    monkeypatch, metax_handoff, failure
):
    h = metax_handoff
    h.comm._flagcx_capture_seen = True

    def fail():
        raise RuntimeError(failure)

    if failure == "capture_query":
        h.capture.side_effect = fail
    elif failure == "sync":
        h.stream.synchronize = fail
    else:

        def fail_ag(self, value, dim=-1):
            h.events.append("ag_failed")
            raise RuntimeError(failure)

        monkeypatch.setattr(CommunicatorFL, "all_gather", fail_ag)
    with pytest.raises(RuntimeError, match=failure):
        h.comm.all_gather(h.value)
    assert h.events == (["sync", "ag_failed"] if failure == "ag" else [])
    assert h.comm._flagcx_capture_seen


def test_metax_all_gatherv_keeps_grouped_submissions_without_fence(metax_handoff):
    import torch

    h = metax_handoff
    h.comm._flagcx_capture_seen = True
    h.native.group_start = lambda: h.events.append("group_start")
    h.native.group_end = lambda: h.events.append("group_end")

    def gather(output, value):
        h.events.append("native_ag")
        output.copy_(torch.cat((value, value)))

    h.native.all_gather = gather
    h.capture.side_effect = lambda: pytest.fail("grouped AG queried capture")
    values = [torch.tensor([[1.0], [2.0]]), torch.tensor([[3.0], [4.0]])]
    outputs = h.comm.all_gatherv(values)
    assert h.events == ["group_start", "native_ag", "native_ag", "group_end"]
    for output, value in zip(outputs, values):
        torch.testing.assert_close(output, torch.cat((value, value)))
    h.capture.assert_not_called()
    h.native._current_stream.assert_not_called()


def test_metax_handoff_cpu_fullgraph_does_not_query_cuda(monkeypatch):
    from types import SimpleNamespace

    import torch

    def no_cuda():
        pytest.fail("CPU fullgraph path queried CUDA")

    communicator = MetaxCommunicatorFL.__new__(MetaxCommunicatorFL)
    communicator.device = torch.device("cuda:0")
    communicator.world_size = 2
    communicator.pyflagcx_comm = SimpleNamespace(
        disabled=False, _current_stream=no_cuda
    )
    communicator._flagcx_capture_seen = True
    monkeypatch.setattr(
        metax_communicator_module,
        "current_platform",
        SimpleNamespace(
            torch_device_fn=SimpleNamespace(is_current_stream_capturing=no_cuda)
        ),
    )
    monkeypatch.setattr(CommunicatorFL, "all_reduce", lambda self, x: x + 1)
    monkeypatch.setattr(CommunicatorFL, "all_gather", lambda self, x, dim=-1: x * 2)

    @torch.compile(backend="eager", fullgraph=True)
    def compute(x):
        return communicator.all_gather(communicator.all_reduce(x))

    for offset in (0, 1, 3):
        values = torch.arange(8, dtype=torch.float32) + offset
        torch.testing.assert_close(compute(values), (values + 1) * 2)
