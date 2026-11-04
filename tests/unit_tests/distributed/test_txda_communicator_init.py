# Copyright (c) 2026 BAAI. All rights reserved.
"""CPU initialization contract for mandatory TXDA FlagCX communication."""

import ast
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture
def loaded():
    path = Path(__file__).resolve().parents[3] / "vllm_fl/distributed/communicator.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "CommunicatorFL"
    )

    class Base:
        def __init__(self, cpu_group, device, *args, use_all2all=False):
            self.cpu_group = cpu_group
            self.device = device
            self.world_size = cpu_group.world_size
            self.use_all2all = use_all2all

    factory = Mock()
    namespace = {
        "DeviceCommunicatorBase": Base,
        "PyFlagcxCommunicator": factory,
        "logger": logging.getLogger("test.txda.communicator.init"),
    }
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias("annotations")], level=0
            ),
            cls,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["CommunicatorFL"], factory


def flagcx_comm(*, disabled=False):
    return SimpleNamespace(
        disabled=disabled,
        init_error=RuntimeError("FlagCX SDK could not load"),
        rank=1,
        world_size=2,
        flagcx=SimpleNamespace(lib=SimpleNamespace(_name="/sdk/lib/libflagcx.so")),
        _sync_collectives=True,
    )


@pytest.mark.parametrize("device_type", ["txda", "npu"])
def test_required_flagcx_initialization_failure_raises_with_original_cause(
    loaded, device_type, caplog
):
    cls, factory = loaded
    native = flagcx_comm(disabled=True)
    factory.return_value = native
    group = SimpleNamespace(world_size=2)
    device = SimpleNamespace(type=device_type)

    with (
        caplog.at_level(logging.INFO),
        pytest.raises(
            RuntimeError, match=f"FlagCX requested on {device_type.upper()}"
        ) as error,
    ):
        cls(group, device)

    assert error.value.__cause__ is native.init_error
    factory.assert_called_once_with(group=group, device=device)
    assert "TXDA FlagCX enabled" not in caplog.text


@pytest.mark.parametrize("device_type", ["txda", "npu", "cuda", "musa"])
def test_available_flagcx_is_retained_and_txda_activation_is_auditable(
    loaded, device_type, caplog
):
    cls, factory = loaded
    native = flagcx_comm()
    factory.return_value = native
    group = SimpleNamespace(world_size=2)
    device = SimpleNamespace(type=device_type)

    with caplog.at_level(logging.INFO):
        comm = cls(group, device)

    assert comm.pyflagcx_comm is native
    factory.assert_called_once_with(group=group, device=device)
    if device_type == "txda":
        for text in (
            "TXDA FlagCX enabled",
            "rank=1 world_size=2",
            "library=/sdk/lib/libflagcx.so",
            "sync_collectives=True",
        ):
            assert text in caplog.text
    else:
        assert "TXDA FlagCX enabled" not in caplog.text


@pytest.mark.parametrize("device_type", ["cuda", "musa", "ptpu"])
def test_other_backends_keep_their_disabled_communicator_behavior(
    loaded, device_type, caplog
):
    cls, factory = loaded
    native = flagcx_comm(disabled=True)
    factory.return_value = native

    with caplog.at_level(logging.INFO):
        comm = cls(SimpleNamespace(world_size=2), SimpleNamespace(type=device_type))

    assert comm.pyflagcx_comm is native
    assert "TXDA FlagCX enabled" not in caplog.text


def test_single_rank_txda_does_not_require_or_claim_a_native_communicator(
    loaded, caplog
):
    cls, factory = loaded

    with caplog.at_level(logging.INFO):
        comm = cls(SimpleNamespace(world_size=1), SimpleNamespace(type="txda"))

    assert comm.pyflagcx_comm is None
    factory.assert_not_called()
    assert "TXDA FlagCX enabled" not in caplog.text
