# Copyright (c) 2025 BAAI. All rights reserved.

"""CPU contracts for loading FlagCX source and wheel layouts."""

import importlib.util
import logging
import os
import sys
from contextlib import nullcontext
from pathlib import Path
from threading import Lock
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

_SOURCE = (
    Path(__file__).resolve().parents[3]
    / "vllm_fl/distributed/device_communicators/flagcx.py"
)
_SUBJECT = "_test_flagcx_communicator"
_WHEEL_API = f"{_SUBJECT}._wheel_api"
_API_SOURCE = """\
from __future__ import annotations
from dataclasses import dataclass

@dataclass
class flagcxUniqueId:
    internal: bytes = b""

class FLAGCXLibrary:
    pass

buffer_type = flagcxComm_t = flagcxDataTypeEnum = flagcxRedOpTypeEnum = object()
"""


class _Group:
    rank = 0
    world_size = 2


@pytest.fixture
def load_flagcx(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "vllm.distributed.utils",
        SimpleNamespace(StatelessProcessGroup=_Group),
    )
    monkeypatch.setitem(
        sys.modules, "vllm.logger", SimpleNamespace(init_logger=logging.getLogger)
    )
    monkeypatch.setitem(
        sys.modules, "vllm.utils.torch_utils", SimpleNamespace(current_stream=Mock())
    )
    monkeypatch.setattr(sys, "path", sys.path.copy())
    monkeypatch.delitem(sys.modules, _WHEEL_API, raising=False)

    def load():
        spec = importlib.util.spec_from_file_location(_SUBJECT, _SOURCE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    yield load
    sys.modules.pop(_WHEEL_API, None)


def _wheel_layout(tmp_path, api_source=_API_SOURCE):
    root = tmp_path / "flagcx"
    (root / "lib").mkdir(parents=True)
    (root / "lib" / "libflagcx.so").touch()
    (root / "api.py").write_text(api_source, encoding="utf-8")
    (root / "__init__.py").write_text(
        'raise AssertionError("FlagCX backend initializer must stay unloaded")\n',
        encoding="utf-8",
    )
    return root


@pytest.mark.parametrize("library_path", [None, "explicit/libflagcx.so"])
def test_wheel_layout_uses_direct_api_and_paired_library(
    load_flagcx, monkeypatch, tmp_path, library_path
):
    root = _wheel_layout(tmp_path)
    monkeypatch.setenv("FLAGCX_PATH", str(root))
    monkeypatch.syspath_prepend(str(tmp_path))
    previous_package = sys.modules.get("flagcx")
    module = load_flagcx()
    assert module._flagcx_available
    assert module.flagcxUniqueId().__class__.__module__ == _WHEEL_API
    assert sys.modules.get("flagcx") is previous_package
    assert str(root) not in sys.path

    monkeypatch.setenv("FLAGCX_PATH", str(tmp_path / "other"))
    library = Mock(side_effect=OSError("stub native library"))
    monkeypatch.setattr(module, "FLAGCXLibrary", library)
    communicator = module.PyFlagcxCommunicator(
        _Group(), "npu:0", library_path=library_path
    )
    library.assert_called_once_with(
        library_path if library_path is not None else str(root / "lib" / "libflagcx.so")
    )
    assert communicator.disabled
    assert isinstance(communicator.init_error, OSError)


def test_source_layout_reads_default_library_path_at_construction(
    load_flagcx, monkeypatch, tmp_path
):
    source_root = tmp_path / "source"
    source_root.mkdir()
    monkeypatch.setenv("FLAGCX_PATH", str(source_root))
    library = Mock(side_effect=OSError("stub native library"))
    wrapper = SimpleNamespace(
        FLAGCXLibrary=library,
        buffer_type=object(),
        flagcxComm_t=object(),
        flagcxDataTypeEnum=object(),
        flagcxUniqueId=object(),
        flagcxRedOpTypeEnum=object(),
    )
    monkeypatch.setitem(sys.modules, "plugin.interservice.flagcx_wrapper", wrapper)

    module = load_flagcx()
    assert module._flagcx_available
    assert str(source_root) in sys.path
    assert _WHEEL_API not in sys.modules

    late_root = tmp_path / "late-source"
    monkeypatch.setenv("FLAGCX_PATH", str(late_root))
    communicator = module.PyFlagcxCommunicator(_Group(), "npu:0")
    library.assert_called_once_with(
        os.path.join(str(late_root), "build/lib/libflagcx.so")
    )
    assert communicator.disabled


@pytest.mark.parametrize("api_source", ["raise ImportError('unavailable')", ""])
def test_invalid_wheel_api_is_unavailable_and_cleans_module(
    load_flagcx, monkeypatch, tmp_path, api_source
):
    root = _wheel_layout(tmp_path, api_source)
    monkeypatch.setenv("FLAGCX_PATH", str(root))
    module = load_flagcx()
    assert not module._flagcx_available
    assert module.FLAGCXLibrary is None
    assert _WHEEL_API not in sys.modules


def test_npu_all_reduce_joins_streams_under_submit_lock(load_flagcx, monkeypatch):
    monkeypatch.delenv("FLAGCX_PATH", raising=False)
    module = load_flagcx()
    comm = module.PyFlagcxCommunicator.__new__(module.PyFlagcxCommunicator)
    comm.disabled = False
    comm.device = "npu:0"
    comm._npu_submit_lock = Lock()
    ops = Mock()

    def check_lock(*args):
        assert comm._npu_submit_lock.locked()

    for operation in (
        ops.record_input,
        ops.wait_input,
        ops.reduce,
        ops.record_output,
        ops.wait_output,
    ):
        operation.side_effect = check_lock

    caller = SimpleNamespace(wait_event=ops.wait_output)
    side = SimpleNamespace(wait_event=ops.wait_input)
    comm._npu_comm_stream = side
    comm._npu_input_ready = SimpleNamespace(record=ops.record_input)
    comm._npu_output_ready = SimpleNamespace(record=ops.record_output)
    comm._all_reduce_on_stream = ops.reduce
    stream_context = Mock(return_value=nullcontext())
    monkeypatch.setattr(
        module.torch, "npu", SimpleNamespace(stream=stream_context), raising=False
    )
    input_tensor = SimpleNamespace(device=comm.device)
    output_tensor = object()

    assert comm.all_reduce(input_tensor, output_tensor, stream=caller) is output_tensor
    assert ops.mock_calls == [
        call.record_input(caller),
        call.wait_input(comm._npu_input_ready),
        call.reduce(input_tensor, output_tensor, module.ReduceOp.SUM, side),
        call.record_output(side),
        call.wait_output(comm._npu_output_ready),
    ]
    stream_context.assert_called_once_with(side)
    assert not comm._npu_submit_lock.locked()
