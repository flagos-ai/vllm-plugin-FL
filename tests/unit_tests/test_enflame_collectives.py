import weakref
from types import SimpleNamespace

import pytest
import torch

from vllm.distributed import parallel_state

from vllm_fl.platform import PlatformFL


@pytest.mark.parametrize(
    "vendor, expected",
    [
        ("enflame", True),
        ("gcu", True),
        ("nvidia", True),
        ("thead", True),
        ("iluvatar", True),
        ("ascend", False),
    ],
)
def test_collective_custom_op_policy(monkeypatch, vendor, expected):
    monkeypatch.setattr(PlatformFL, "vendor_name", vendor)
    assert PlatformFL.use_custom_op_collectives() is expected


@pytest.mark.parametrize("vendor", ["enflame", "gcu"])
def test_compiled_collective_keeps_python_stream_outside_graph(monkeypatch, vendor):
    monkeypatch.setattr(PlatformFL, "vendor_name", vendor)
    group = object.__new__(parallel_state.GroupCoordinator)
    group.world_size = 2
    group.unique_name = "test_enflame_compile_boundary"
    group.use_custom_op_call = PlatformFL.use_custom_op_collectives()
    calls = []

    def reduce_tensor(tensor):
        stream = SimpleNamespace(cuda_stream=17)
        calls.append(stream.cuda_stream)
        return tensor + 3

    group.device_communicator = SimpleNamespace(all_reduce=reduce_tensor)
    monkeypatch.setitem(parallel_state._groups, group.unique_name, weakref.ref(group))
    library = torch.library.Library("vllm", "IMPL", "CPU")
    try:
        library.impl("all_reduce", parallel_state.all_reduce)
        compiled = torch.compile(group.all_reduce, backend="eager", fullgraph=True)
        for length in [2, 5]:
            tensor = torch.arange(length, dtype=torch.float32)
            torch.testing.assert_close(compiled(tensor), tensor + 3)
    finally:
        library._destroy()
    assert calls == [17, 17]
