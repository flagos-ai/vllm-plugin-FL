"""Opt-in routing must not alter other vendors or the default communicator."""
import pytest

from vllm_fl.platform import PlatformFL


@pytest.mark.parametrize('vendor,enabled,expected', [
    ('hygon', '0', 'vllm.distributed.device_communicators.cuda_communicator.CudaCommunicator'),
    ('hygon', '1', 'vllm_fl.distributed.hygon_split_graph_comm.HygonSplitGraphCommunicator'),
    ('nvidia', '1', 'vllm.distributed.device_communicators.cuda_communicator.CudaCommunicator'),
])
def test_split_graph_communicator_is_hygon_opt_in(monkeypatch, vendor, enabled, expected):
    monkeypatch.setattr(PlatformFL, 'vendor_name', vendor)
    monkeypatch.setattr(PlatformFL, 'dist_backend', 'nccl')
    monkeypatch.setenv('VLLM_FL_HYGON_SPLIT_GRAPH_COMM', enabled)
    assert PlatformFL.get_device_communicator_cls() == expected


def test_flagcx_remains_selected_when_split_requested(monkeypatch):
    monkeypatch.setattr(PlatformFL, 'vendor_name', 'hygon')
    monkeypatch.setattr(PlatformFL, 'dist_backend', 'flagcx')
    monkeypatch.setenv('VLLM_FL_HYGON_SPLIT_GRAPH_COMM', '1')
    assert PlatformFL.get_device_communicator_cls() == 'vllm_fl.distributed.communicator.CommunicatorFL'


def test_symmetric_memory_is_rejected_before_initializing_communicators(monkeypatch):
    from vllm_fl.distributed.hygon_split_graph_comm import HygonSplitGraphCommunicator
    monkeypatch.setenv('VLLM_USE_NCCL_SYMM_MEM', '1')
    with pytest.raises(RuntimeError, match='does not support NCCL symmetric memory'):
        HygonSplitGraphCommunicator(None)
