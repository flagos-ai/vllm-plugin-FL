"""Workload guards and public policy must share the same resolver."""

import pytest

from vllm_fl.dispatch.binding import OperatorBinding
from vllm_fl.dispatch.manager import OpManager
from vllm_fl.dispatch.policy import SelectionPolicy, policy_context
from vllm_fl.dispatch.types import BackendImplKind, OpImpl


def make_binding(native, flaggems):
    manager = OpManager(register_builtins=False)
    manager.registry.register_many(
        [
            OpImpl("test_mm", "native", BackendImplKind.VENDOR, native, vendor="cuda"),
            OpImpl("test_mm", "flaggems", BackendImplKind.DEFAULT, flaggems),
        ]
    )
    return OperatorBinding(
        manager,
        "test_mm",
        supports={"native": lambda rows: rows <= 2},
        default_order=("native", "flaggems"),
    )


def test_shape_guard_does_not_disable_later_supported_shapes():
    binding = make_binding(lambda rows: "native", lambda rows: "flaggems")
    assert [binding(rows) for rows in (1, 3, 2)] == ["native", "flaggems", "native"]
    assert binding.manager.get_failed_impls() == {}


def test_public_selection_and_vendor_exclusion_override_shape_default():
    binding = make_binding(lambda rows: "native", lambda rows: "flaggems")
    with policy_context(
        SelectionPolicy.from_dict(per_op_order={"test_mm": ["flagos"]})
    ):
        assert binding(1) == "flaggems"
    with policy_context(SelectionPolicy.from_dict(deny_vendors={"cuda"})):
        assert binding(1) == "flaggems"
    assert binding(1) == "native"


def test_launch_errors_propagate_without_retrying_another_backend():
    calls = []

    def broken(rows):
        calls.append("native")
        raise RuntimeError("device launch failed")

    binding = make_binding(broken, lambda rows: calls.append("flaggems"))
    with pytest.raises(RuntimeError, match="device launch failed"):
        binding(1)
    assert calls == ["native"]
