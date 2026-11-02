# Copyright (c) 2026 BAAI. All rights reserved.
"""GCU stream adaptation does not depend on CUDA aliases."""
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from vllm_fl.distributed.device_communicators.flagcx import PyFlagcxCommunicator


def test_gcu_stream_bridge(monkeypatch):
    stream = SimpleNamespace(gcu_stream=123)
    gcu = SimpleNamespace(current_stream=Mock(return_value=stream))
    monkeypatch.setattr(torch, 'gcu', gcu, raising=False)
    comm = PyFlagcxCommunicator.__new__(PyFlagcxCommunicator)
    comm.device = SimpleNamespace(type='gcu')
    comm.flagcx = SimpleNamespace(adaptor_stream_copy=Mock(return_value='copied'))
    assert comm._current_stream() is stream
    gcu.current_stream.assert_called_once_with(comm.device)
    assert comm._copy_flagcx_stream(stream) == 'copied'
    adapted = comm.flagcx.adaptor_stream_copy.call_args.args[0]
    assert adapted.cuda_stream == 123


def test_other_device_stream_unchanged():
    stream = object()
    comm = PyFlagcxCommunicator.__new__(PyFlagcxCommunicator)
    comm.device = SimpleNamespace(type='cuda')
    comm.flagcx = SimpleNamespace(adaptor_stream_copy=Mock())
    comm._copy_flagcx_stream(stream)
    comm.flagcx.adaptor_stream_copy.assert_called_once_with(stream)
