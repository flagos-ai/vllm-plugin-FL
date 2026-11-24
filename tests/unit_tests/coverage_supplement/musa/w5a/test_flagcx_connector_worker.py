"""Worker-side coverage for the FlagCX connector.

Construction runs the production ``__init__`` with CPU doubles for the
collective/topology boundary; registration, the sender thread/worker, the
write-plan builder, receive coroutines, expiry sweeps and shutdown are
driven directly.  The native library is faked and, in one test, loaded
through a monkeypatched ``ctypes.CDLL`` for both the success and failure
paths as required for the no-shared-object container.
"""
from __future__ import annotations
import asyncio
import ctypes
import os
import sys
import threading
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pytest
import torch
import zmq
import test_flagcx_connector_support as support

@pytest.fixture()
def worker_env(monkeypatch):
    (module, recorder) = support.load_connector(monkeypatch)
    support.reset_topology_state()
    monkeypatch.delenv('FLAGCX_LIB_PATH', raising=False)
    monkeypatch.delenv('FLAGCX_PATH', raising=False)
    workers = []

    def build(role='kv_consumer', **kwargs):
        worker = support.make_worker(module, role=role, **kwargs)
        workers.append(worker)
        return worker
    try:
        yield (module, recorder, build)
    finally:
        for worker in workers:
            try:
                worker.shutdown()
            except Exception:
                pass
        sys.modules.pop('vllm_fl.distributed.kv_transfer.flagcx_connector', None)

def _cfg(**kwargs):
    return support.vllm_config(model=support.ModelInfo(use_mla=kwargs.get('use_mla', False), total_num_kv_heads=kwargs.get('total_kv_heads', 8)), parallel=support.ParallelInfo(tp_size=kwargs.get('tp_size', 1), dp_rank=kwargs.get('dp_rank', 0)), transfer=support.TransferInfo(kv_role=kwargs.get('role', 'kv_both'), extra_config=kwargs.get('extra_config')), cache=support.CacheInfo(block_size=kwargs.get('block_size', 16)))

def _kv_cache(num_blocks=4, block_size=16, heads=2, head_size=8):
    return torch.zeros(2, num_blocks, block_size, heads, head_size, dtype=torch.float16)

def _region(module, base_addr, *, block_len=512, kv_block_len=512, name='l0', layer_index=0, group_index=0):
    return module.TransferRegion(layer_name=name, layer_index=layer_index, base_addr=base_addr, block_len=block_len, kv_block_len=kv_block_len, group_index=group_index)

def _send_meta(module, transfer_id='t1', ids=None, p_req_id='p1', ready=True):
    event = threading.Event()
    if ready:
        event.set()
    return module.SendBlockMeta(p_req_id=p_req_id, transfer_id=transfer_id, local_block_ids=[[0, 1]] if ids is None else [list(g) for g in ids], ready=event)

def _agent_meta(module, req_blocks, remote_tp_rank=0, remote_tp_size=1):
    return module.FlagCXAgentMetadata(remote_hostname='10.0.0.2', remote_port=4242, remote_tp_size=remote_tp_size, remote_tp_rank=remote_tp_rank, req_blocks=req_blocks, kv_caches_base_addr=[], block_lens=[], kv_block_lens=[], registered_layer_names=[], registered_layer_indices=[], registered_group_indices=[])

def _seed_send_state(module, worker, *, local_bases=(4096, 6144), remote_bases=(32768, 34816), block_lens=(512, 512), kv_block_lens=(512, 512), remote_kv_block_lens=None, local_ids=((0, 1),), remote_ids=((4, 5),), remote_names=('l0', 'l0'), remote_tp_rank=0, remote_tp_size=1, p_req_id='p1', transfer_id='t1', d_req_id='d1', ready=True):
    if remote_kv_block_lens is None:
        remote_kv_block_lens = kv_block_lens
    worker.kv_caches_base_addr = list(local_bases)
    worker.block_len_per_layer = list(block_lens)
    worker.kv_block_len_per_layer = list(kv_block_lens)
    worker.registered_layer_names = ['l0', 'l0']
    worker.registered_layer_indices = [0, 0]
    worker.registered_group_indices = [0, 0]
    send_meta = _send_meta(module, transfer_id=transfer_id, ids=local_ids, p_req_id=p_req_id, ready=ready)
    worker.reqs_need_send.reqs[transfer_id] = send_meta
    agent = module.FlagCXAgentMetadata(remote_hostname='10.0.0.2', remote_port=4242, remote_tp_size=remote_tp_size, remote_tp_rank=remote_tp_rank, req_blocks={d_req_id: (transfer_id, [list(g) for g in remote_ids])}, kv_caches_base_addr=list(remote_bases), block_lens=list(block_lens), kv_block_lens=list(remote_kv_block_lens), registered_layer_names=list(remote_names), registered_layer_indices=[0, 0], registered_group_indices=[0, 0])
    return (agent, send_meta)

def _writes(recorder):
    return [call for call in recorder if call[0] == 'flagcxP2pBatchWriteSync']

def test_worker_init_consumer(worker_env):
    (module, recorder, build) = worker_env
    worker = build('kv_consumer', extra_config={'num_workers': 3})
    assert worker.engine_id == 'eng-w'
    assert worker.kv_role == 'kv_consumer'
    assert worker.rpc_port == 0
    assert worker.backend_name == 'FAKE_ATTN'
    assert worker.num_workers == 3
    assert worker.session_conns == {}
    assert worker.reqs_need_send.reqs == {}
    assert worker.receiver_loop.is_running()
    assert not hasattr(worker, '_sender_executor')
    assert not hasattr(worker, '_sender_t')
    assert ('FLAGCXLibrary', 'build/lib/libflagcx.so') in recorder
    assert ('flagcxP2pEngineCreate',) in recorder

def test_worker_init_producer(worker_env):
    (module, recorder, build) = worker_env
    worker = build('kv_producer', extra_config={'num_workers': 4})
    assert worker.kv_role == 'kv_producer'
    assert not hasattr(worker, 'receiver_loop')
    assert worker._sender_executor._max_workers == 4
    assert worker._sender_t is None

def test_worker_init_num_workers_default_and_layers(worker_env):
    (module, _, build) = worker_env
    groups = [support.full_group(0), support.sliding_group(1)]
    worker = build('kv_producer', groups=groups)
    assert worker.num_workers == 10
    assert worker._layer_group_indices == {'model.layers.0.self_attn': 0, 'model.layers.1.self_attn': 1}
    assert worker._layer_specs['model.layers.0.self_attn'] is groups[0].kv_cache_spec
    assert worker._layer_specs['model.layers.1.self_attn'] is groups[1].kv_cache_spec
    assert worker.registered_layer_names == []

def test_worker_syncs_block_size_with_kernel(worker_env, monkeypatch):
    (module, _, build) = worker_env

    class _BlockTable:

        @staticmethod
        def map_to_kernel_blocks(group, factor, arange):
            out = []
            for block in group.tolist():
                out.extend((block * factor + i for i in range(factor)))
            return np.array(out)
    monkeypatch.setattr(module, 'select_common_block_size', lambda size, b: 8)
    monkeypatch.setattr(module, 'BlockTable', _BlockTable)
    groups = [support.full_group(0), support.mamba_group(2)]
    worker = build('kv_consumer', groups=groups, block_size=16)
    assert worker.block_size == 8
    assert worker.kv_topo.block_size == 8
    assert worker._physical_blocks_per_logical_kv_block == 2
    assert worker._logical_to_kernel_block_ids([[1, 2], [7]]) == [[2, 3, 4, 5], [7]]
    assert worker._logical_to_kernel_block_ids([[], []]) == [[], []]

def test_logical_to_kernel_passthrough(worker_env):
    (module, _, build) = worker_env
    worker = build('kv_consumer')
    assert worker._physical_blocks_per_logical_kv_block == 1
    assert worker._logical_to_kernel_block_ids([[3, 4], [7]]) == [[3, 4], [7]]
    assert worker._logical_to_kernel_block_ids([]) == []

def test_worker_get_conn_memoizes(worker_env):
    (module, recorder, build) = worker_env
    worker = build('kv_consumer')
    first = worker._get_conn('10.0.0.5:4242')
    second = worker._get_conn('10.0.0.5:4242')
    third = worker._get_conn('10.0.0.6:4242')
    assert first is second
    assert third is not first
    assert list(worker.session_conns) == ['10.0.0.5:4242', '10.0.0.6:4242']
    assert [c for c in recorder if c[0] == 'flagcxP2pGetConn'] == [('flagcxP2pGetConn', '10.0.0.5:4242'), ('flagcxP2pGetConn', '10.0.0.6:4242')]

def test_producer_cache_is_replicated(worker_env):
    (module, _, build) = worker_env
    sharded = build('kv_producer', total_kv_heads=2)
    assert sharded._producer_cache_is_replicated() is False
    mla = build('kv_producer', total_kv_heads=2, use_mla=True)
    assert mla._producer_cache_is_replicated() is True

def test_register_kv_caches_consumer_kv_first(worker_env):
    (module, recorder, build) = worker_env
    worker = build('kv_consumer')
    cache = _kv_cache()
    worker.register_kv_caches({'model.layers.0.self_attn': cache})
    block_len = cache[0].stride(0) * cache[0].element_size()
    assert worker.kv_caches_base_addr == [cache[0].data_ptr(), cache[1].data_ptr()]
    assert worker.block_len_per_layer == [block_len, block_len]
    assert worker.kv_block_len_per_layer == [block_len, block_len]
    assert worker.registered_layer_names == ['model.layers.0.self_attn'] * 2
    assert worker.registered_layer_indices == [0, 0]
    assert worker.registered_group_indices == [0, 0]
    assert worker.device_kv_caches['model.layers.0.self_attn'] is cache
    storage = cache.untyped_storage()
    registers = [c for c in recorder if c[0] == 'flagcxP2pRegister']
    assert registers == [('flagcxP2pRegister', storage.data_ptr(), storage.nbytes())]
    assert worker.rpc_port == 4242
    assert ('flagcxP2pStartRpcServer',) in recorder
    assert not hasattr(worker, '_sender_executor')

def test_register_kv_caches_skips_unknown_and_requires_tensor(worker_env):
    (module, _, build) = worker_env
    worker = build('kv_consumer')
    known = _kv_cache()
    worker.register_kv_caches({'model.layers.0.self_attn': known, 'model.layers.9.self_attn': _kv_cache()})
    assert worker.registered_layer_names == ['model.layers.0.self_attn'] * 2
    assert worker.kv_caches_base_addr == [known[0].data_ptr(), known[1].data_ptr()]
    empty = build('kv_consumer')
    with pytest.raises(RuntimeError, match='No KV cache tensors'):
        empty.register_kv_caches({'model.layers.9.self_attn': _kv_cache()})
    assert empty.kv_caches_base_addr == []

def test_register_kv_caches_mla_uses_page_size(worker_env):
    (module, _, build) = worker_env
    spec = support.FakeMLASpec(page_size_bytes=256)
    group = support.FakeKVGroup(spec, ['model.layers.5.self_attn'])
    worker = build('kv_consumer', groups=[group], use_mla=True)
    cache = torch.zeros(4, 16, 8, dtype=torch.float32)
    worker.register_kv_caches({'model.layers.5.self_attn': cache})
    block_len = cache.stride(0) * cache.element_size()
    assert worker.block_len_per_layer == [block_len]
    assert worker.kv_block_len_per_layer == [256]
    assert worker.registered_layer_indices == [5]

def test_register_kv_caches_mamba_conv_only(worker_env):
    (module, _, build) = worker_env
    worker = build('kv_consumer', groups=[support.mamba_group(2)])
    conv = torch.zeros(4, 8, 2, dtype=torch.float32)
    ssm = torch.zeros(4, 16, dtype=torch.float32)
    worker.register_kv_caches({'model.layers.2.mamba': [conv, ssm]})
    block_len = conv.stride(0) * conv.element_size()
    assert worker.kv_caches_base_addr == [conv.data_ptr()]
    assert worker.block_len_per_layer == [block_len]
    assert worker.kv_block_len_per_layer == [block_len]
    assert worker.registered_layer_indices == [2]

def test_register_kv_caches_blocks_first_halves(worker_env):
    (module, _, build) = worker_env
    support.TOPO_STATE['blocks_first'] = True
    worker = build('kv_consumer', groups=[support.full_group(0)], has_mamba=True)
    cache = _kv_cache()
    worker.register_kv_caches({'model.layers.0.self_attn': cache})
    assert worker.kv_topo.is_kv_layout_blocks_first is True
    assert worker.kv_caches_base_addr == [cache.data_ptr()]
    block_len = worker.block_len_per_layer[0]
    assert worker.kv_block_len_per_layer == [block_len // 2]

def test_get_transfer_regions_whole(worker_env):
    (module, _, build) = worker_env
    worker = build('kv_consumer')
    regions = worker._get_transfer_regions([4096, 6144], [512, 512], [512, 512], ['l0', 'l0'], [3, 3], [0, 0])
    assert [(r.layer_name, r.layer_index, r.base_addr, r.block_len, r.kv_block_len, r.group_index) for r in regions] == [('l0', 3, 4096, 512, 512, 0), ('l0', 3, 6144, 512, 512, 0)]

def test_get_transfer_regions_blocks_first_splits_attention(worker_env):
    (module, _, build) = worker_env
    support.TOPO_STATE['blocks_first'] = True
    groups = [support.full_group(0), support.mamba_group(2), support.FakeKVGroup(support.FakeMLASpec(page_size_bytes=256), ['model.layers.5.self_attn'])]
    worker = build('kv_consumer', groups=groups, has_mamba=True)
    regions = worker._get_transfer_regions([4096, 8192, 12288], [512, 512, 512], [256, 256, 256], ['model.layers.0.self_attn', 'model.layers.2.mamba', 'model.layers.5.self_attn'], [0, 2, 5], [0, 1, 2])
    assert [(r.layer_name, r.layer_index, r.base_addr, r.group_index) for r in regions] == [('model.layers.0.self_attn', 0, 4096, 0), ('model.layers.0.self_attn', 0, 4352, 0), ('model.layers.2.mamba', 2, 8192, 1), ('model.layers.5.self_attn', 5, 12288, 2)]

def test_build_transfer_params_coalesces(worker_env):
    (module, _, build) = worker_env
    worker = build('kv_consumer')
    local = [_region(module, 4096), _region(module, 6144)]
    remote = [_region(module, 32768), _region(module, 34816)]
    send_meta = _send_meta(module, ids=[[0, 1, 2]])
    agent = _agent_meta(module, {'d1': ('t1', [[5, 6, 7]])})
    (src, dst, sizes) = worker._build_transfer_params([('d1', send_meta)], agent, local, remote)
    assert src == [4096, 6144]
    assert dst == [32768 + 5 * 512, 34816 + 5 * 512]
    assert sizes == [3 * 512, 3 * 512]

def test_build_transfer_params_per_block(worker_env):
    (module, _, build) = worker_env
    worker = build('kv_consumer')
    local = [_region(module, 4096, block_len=512, kv_block_len=256), _region(module, 6144, block_len=512, kv_block_len=256)]
    remote = [_region(module, 32768, block_len=512, kv_block_len=256), _region(module, 34816, block_len=512, kv_block_len=256)]
    send_meta = _send_meta(module, ids=[[0, 1]])
    agent = _agent_meta(module, {'d1': ('t1', [[2, 3]])})
    (src, dst, sizes) = worker._build_transfer_params([('d1', send_meta)], agent, local, remote)
    assert src == [4096, 4608, 6144, 6656]
    assert dst == [32768 + 2 * 512, 32768 + 3 * 512, 34816 + 2 * 512, 34816 + 3 * 512]
    assert sizes == [256, 256, 256, 256]

def test_build_transfer_params_trims_local_tail(worker_env):
    (module, _, build) = worker_env
    worker = build('kv_consumer')
    local = [_region(module, 4096), _region(module, 6144)]
    remote = [_region(module, 32768), _region(module, 34816)]
    send_meta = _send_meta(module, ids=[[0, 1, 2, 3]])
    agent = _agent_meta(module, {'d1': ('t1', [[5]])})
    (src, dst, sizes) = worker._build_transfer_params([('d1', send_meta)], agent, local, remote)
    assert src == [4096 + 3 * 512, 6144 + 3 * 512]
    assert dst == [32768 + 5 * 512, 34816 + 5 * 512]
    assert sizes == [512, 512]

def test_build_transfer_params_rejects_malformed(worker_env):
    (module, _, build) = worker_env
    worker = build('kv_consumer')
    local = [_region(module, 4096)]
    cases = [_agent_meta(module, {'d1': ('t1', [[1], [2]])}), _agent_meta(module, {'d1': ('t1', [[1, 2]])}), _agent_meta(module, {'d1': ('t1', [[]])})]
    for agent in cases:
        send_meta = _send_meta(module, ids=[[0]])
        assert worker._build_transfer_params([('d1', send_meta)], agent, local, local) == ([], [], [])
    assert worker._build_transfer_params([], _agent_meta(module, {}), [], []) == ([], [], [])

def test_build_transfer_params_strips_mamba_nulls(worker_env):
    (module, _, build) = worker_env
    worker = build('kv_consumer', groups=[support.mamba_group(2)])
    local = [_region(module, 4096)]
    remote = [_region(module, 32768)]
    send_meta = _send_meta(module, ids=[[0, 3]])
    agent = _agent_meta(module, {'d1': ('t1', [[0, 4]])})
    (src, dst, sizes) = worker._build_transfer_params([('d1', send_meta)], agent, local, remote)
    assert src == [4096 + 3 * 512]
    assert dst == [32768 + 4 * 512]
    assert sizes == [512]

def test_build_transfer_params_skips_empty_and_unknown_groups(worker_env):
    (module, _, build) = worker_env
    worker = build('kv_consumer', groups=[support.full_group(0), support.full_group(1)])
    local = [_region(module, 4096, group_index=0), _region(module, 6144, group_index=1), _region(module, 8192, group_index=5)]
    remote = [_region(module, 32768, group_index=0), _region(module, 34816, group_index=1), _region(module, 36864, group_index=5)]
    send_meta = _send_meta(module, ids=[[], [5]])
    agent = _agent_meta(module, {'d1': ('t1', [[], [6]])})
    (src, dst, sizes) = worker._build_transfer_params([('d1', send_meta)], agent, local, remote)
    assert src == [6144 + 5 * 512]
    assert dst == [34816 + 6 * 512]
    assert sizes == [512]

def test_send_kv_to_decode_writes_and_finishes(worker_env):
    (module, recorder, build) = worker_env
    worker = build('kv_producer')
    (agent, _) = _seed_send_state(module, worker)
    worker._send_kv_to_decode(agent)
    assert _writes(recorder) == [('flagcxP2pBatchWriteSync', '10.0.0.2:4242', [4096, 6144], [2 * 512, 2 * 512])]
    assert worker.reqs_need_send.reqs == {}
    assert worker.finished_sending_reqs.set == {'p1'}
    stats = worker.get_kv_connector_stats()
    assert stats.data['bytes_transferred'] == [4 * 512]
    assert stats.data['num_descriptors'] == [2]

def test_send_kv_to_decode_warns_unpaired_rank(worker_env):
    (module, recorder, build) = worker_env
    worker = build('kv_producer')
    (agent, _) = _seed_send_state(module, worker, remote_tp_rank=1)
    worker._send_kv_to_decode(agent)
    assert len(_writes(recorder)) == 1
    assert worker.finished_sending_reqs.set == {'p1'}

def test_send_kv_to_decode_timeout(worker_env):
    (module, recorder, build) = worker_env
    worker = build('kv_producer')
    (agent, _) = _seed_send_state(module, worker, ready=False)
    worker._abort_request_timeout = 0
    with pytest.raises(RuntimeError, match='Timed out waiting for prefill blocks'):
        worker._send_kv_to_decode(agent)
    assert _writes(recorder) == []
    assert 't1' in worker.reqs_need_send.reqs

def test_send_kv_to_decode_unalignable(worker_env):
    (module, recorder, build) = worker_env
    worker = build('kv_producer')
    (agent, _) = _seed_send_state(module, worker, remote_names=('mismatched.layer', 'mismatched.layer'))
    with pytest.raises(RuntimeError, match='no matching consumer occurrence'):
        worker._send_kv_to_decode(agent)
    assert _writes(recorder) == []

def test_send_kv_to_decode_asymmetric_length(worker_env):
    (module, recorder, build) = worker_env
    worker = build('kv_producer')
    (agent, _) = _seed_send_state(module, worker, remote_kv_block_lens=(1024, 1024))
    with pytest.raises(RuntimeError, match='KV region length mismatch'):
        worker._send_kv_to_decode(agent)
    assert _writes(recorder) == []

def test_send_kv_to_decode_records_write_failure(worker_env):
    (module, _, build) = worker_env
    worker = build('kv_producer')
    (agent, _) = _seed_send_state(module, worker)
    worker.flagcx.write_should_raise = RuntimeError('rdma-eio')
    with pytest.raises(RuntimeError, match='rdma-eio'):
        worker._send_kv_to_decode(agent)
    assert worker.xfer_stats.data['num_failed_transfers'] == [1]
    assert 't1' in worker.reqs_need_send.reqs

def test_send_kv_to_decode_requires_prefill_id(worker_env):
    (module, _, build) = worker_env
    worker = build('kv_producer')
    (agent, _) = _seed_send_state(module, worker, p_req_id='')
    with pytest.raises(RuntimeError, match='Missing Prefill request ID'):
        worker._send_kv_to_decode(agent)
    assert 't1' not in worker.reqs_need_send.reqs

def test_start_load_kv_producer_arms(worker_env):
    (module, _, build) = worker_env
    worker = build('kv_producer')
    meta = module.FlagCXConnectorMetadata()
    meta.reqs_to_send['pr1'] = ('tr1', [[1, 2], [3]])
    meta.reqs_to_send['pr2'] = ('tr2', [])
    worker.start_load_kv(meta)
    armed = worker.reqs_need_send.reqs['tr1']
    assert armed.p_req_id == 'pr1'
    assert armed.ready.is_set()
    assert armed.expire_time != float('inf')
    unarmed = worker.reqs_need_send.reqs['tr2']
    assert not unarmed.ready.is_set()
    assert unarmed.expire_time == float('inf')
    again = module.FlagCXConnectorMetadata()
    again.reqs_to_send['pr1'] = ('tr1', [[7]])
    worker.start_load_kv(again)
    assert worker.reqs_need_send.reqs['tr1'] is armed
    assert armed.local_block_ids == [[7]]

def test_start_load_kv_consumer_ignores_send(worker_env):
    (module, _, build) = worker_env
    worker = build('kv_consumer')
    meta = module.FlagCXConnectorMetadata()
    meta.reqs_to_send['pr9'] = ('tr9', [[9]])
    worker.start_load_kv(meta)
    assert worker.reqs_need_send.reqs == {}

def test_get_finished_expires_stale_sends(worker_env):
    (module, _, build) = worker_env
    worker = build('kv_producer')
    meta = module.FlagCXConnectorMetadata()
    meta.reqs_to_send['pr1'] = ('tr1', [[1]])
    meta.reqs_to_send['pr2'] = ('tr2', [[2]])
    worker.start_load_kv(meta)
    fresh = worker.reqs_need_send.reqs['tr1']
    stale = worker.reqs_need_send.reqs['tr2']
    stale.expire_time = 0.0
    (finished_sending, finished_recving) = worker.get_finished()
    assert finished_sending == {'pr2'}
    assert finished_recving is None
    assert 'tr2' not in worker.reqs_need_send.reqs
    assert 'tr1' in worker.reqs_need_send.reqs
    assert worker.xfer_stats.data['num_kv_expired_reqs'] == [1]
    assert fresh.expire_time > 0.0
    assert worker.get_finished() == (None, None)

def test_get_finished_consumer(worker_env):
    (module, _, build) = worker_env
    worker = build('kv_consumer')
    assert worker.wait_for_layer_load() is None
    assert worker.get_finished() == (None, None)

def test_get_kv_connector_stats_snapshot(worker_env):
    (module, _, build) = worker_env
    worker = build('kv_producer')
    assert worker.get_kv_connector_stats() is None
    worker.xfer_stats.record_transfer(0.5, 2 ** 20, 3)
    snapshot = worker.get_kv_connector_stats()
    assert snapshot.data['transfer_duration'] == [0.5]
    assert snapshot.data['bytes_transferred'] == [2 ** 20]
    assert snapshot.data['num_descriptors'] == [3]
    assert worker.get_kv_connector_stats() is None

class _FakeSock:

    def __init__(self, reply=None, raise_on_send=None):
        self.reply = reply
        self.raise_on_send = raise_on_send
        self.sent = None
        self.closed = False
        self.options = []

    def setsockopt(self, *args):
        self.options.append(args)

    async def send(self, payload):
        if self.raise_on_send is not None:
            raise self.raise_on_send
        self.sent = payload

    async def recv(self):
        return self.reply

    def close(self):
        self.closed = True

def test_receive_kv_error_reply_counts_failure(worker_env, monkeypatch):
    (module, _, build) = worker_env
    worker = build('kv_consumer')
    sock = _FakeSock(reply=module.TRANS_ERROR)
    monkeypatch.setattr(module, 'make_zmq_socket', lambda *a, **k: sock)
    worker._pull_pending['cr1'] = 2
    asyncio.run(worker._receive_kv('tcp://127.0.0.1:1', {'cr1': ('t1', [[1]])}))
    assert worker.xfer_stats.data['num_failed_recvs'] == [1]
    assert worker.finished_recving_reqs.set == set()
    assert worker._pull_pending == {'cr1': 2}

def test_receive_kv_exception_counts_failure(worker_env, monkeypatch):
    (module, _, build) = worker_env
    worker = build('kv_consumer')
    sock = _FakeSock(raise_on_send=RuntimeError('side-channel down'))
    monkeypatch.setattr(module, 'make_zmq_socket', lambda *a, **k: sock)
    asyncio.run(worker._receive_kv('tcp://127.0.0.1:1', {'cx': ('t1', [])}))
    assert sock.closed is True
    assert worker.xfer_stats.data['num_failed_recvs'] == [1]
    assert worker.finished_recving_reqs.set == set()

def test_receive_kv_context_terminated(worker_env, monkeypatch):
    (module, _, build) = worker_env
    worker = build('kv_consumer')
    sock = _FakeSock(raise_on_send=module.zmq.ContextTerminated())
    monkeypatch.setattr(module, 'make_zmq_socket', lambda *a, **k: sock)
    worker._pull_pending['ct'] = 1
    asyncio.run(worker._receive_kv('tcp://127.0.0.1:1', {'ct': ('t1', [])}))
    assert sock.closed is True
    assert worker.xfer_stats.data['num_failed_recvs'] == []
    assert 'ct' in worker.finished_recving_reqs.set

def test_group_kv_pull_fans_out(worker_env):
    (module, _, build) = worker_env
    support.TOPO_STATE['targets'] = lambda rank, size, remote: [0] if remote <= size else list(range(remote))
    worker = build('kv_consumer', tp_size=1)
    calls = []

    async def fake_receive(path, req_blocks):
        calls.append((path, req_blocks))
    worker._receive_kv = fake_receive
    reqs = {'r1': module.RecvReqMeta(local_block_ids=[[1, 2]], remote_host='10.0.0.7', remote_port=9000, transfer_id='t1', remote_tp_size=2), 'r2': module.RecvReqMeta(local_block_ids=[], remote_host='10.0.0.8', remote_port=9100, transfer_id='t2', remote_tp_size=0)}

    async def run():
        await worker._group_kv_pull(reqs)
        await asyncio.sleep(0.05)
    asyncio.run(run())
    assert worker._pull_pending == {'r1': 2, 'r2': 1}
    paths = sorted((path for (path, _) in calls))
    assert paths == sorted(['tcp://10.0.0.7:9000', 'tcp://10.0.0.7:9001', 'tcp://10.0.0.8:9100'])
    by_path = dict(calls)
    assert by_path['tcp://10.0.0.7:9000'] == {'r1': ('t1', [[1, 2]])}
    assert by_path['tcp://10.0.0.8:9100'] == {'r2': ('t2', [])}

class _CapturePusher:

    def __init__(self):
        self.frames = []
        self.closed = False

    def send_multipart(self, frames):
        self.frames.append(frames)

    def close(self):
        self.closed = True

def test_sender_worker_success(worker_env, monkeypatch):
    (module, recorder, build) = worker_env
    worker = build('kv_producer')
    (agent, _) = _seed_send_state(module, worker)
    pusher = _CapturePusher()
    monkeypatch.setattr(module, 'make_zmq_socket', lambda *a, **k: pusher)
    worker._sender_worker(b'client', worker._encoder.encode(agent), 'inproc://x')
    assert pusher.frames == [(b'client', module.TRANS_DONE)]
    assert pusher.closed is True
    assert len(_writes(recorder)) == 1

def test_sender_worker_setup_error(worker_env, monkeypatch):
    (module, recorder, build) = worker_env
    worker = build('kv_producer')
    (agent, _) = _seed_send_state(module, worker, remote_names=('mismatched.layer', 'mismatched.layer'))
    pusher = _CapturePusher()
    monkeypatch.setattr(module, 'make_zmq_socket', lambda *a, **k: pusher)
    worker._sender_worker(b'client', worker._encoder.encode(agent), 'inproc://x')
    assert pusher.frames == [(b'client', module.TRANS_ERROR)]
    assert _writes(recorder) == []

def test_sender_worker_push_failure_is_swallowed(worker_env, monkeypatch):
    (module, _, build) = worker_env
    worker = build('kv_producer')
    (agent, _) = _seed_send_state(module, worker)

    class _BadPusher:

        def send_multipart(self, frames):
            raise module.zmq.ZMQError('shutting down')

        def close(self):
            pass
    monkeypatch.setattr(module, 'make_zmq_socket', lambda *a, **k: _BadPusher())
    worker._sender_worker(b'client', worker._encoder.encode(agent), 'inproc://x')

def test_shutdown_destroys_engine_once(worker_env):
    (module, recorder, build) = worker_env
    worker = build('kv_consumer')
    assert worker.engine is not None
    loop = worker.receiver_loop
    worker.shutdown()
    assert worker.engine is None
    assert not loop.is_running()
    destroys = [c for c in recorder if c[0] == 'flagcxP2pEngineDestroy']
    assert len(destroys) == 1
    worker.shutdown()
    assert len([c for c in recorder if c[0] == 'flagcxP2pEngineDestroy']) == 1

def test_shutdown_swallows_destroy_failure(worker_env, monkeypatch):
    (module, _, build) = worker_env
    worker = build('kv_consumer')

    def explode(engine):
        raise RuntimeError('destroy failed')
    monkeypatch.setattr(worker.flagcx, 'flagcxP2pEngineDestroy', explode)
    worker.shutdown()
    assert worker.engine is None

def test_native_library_ctypes_success_and_failure(worker_env, monkeypatch, tmp_path):
    (module, _, _) = worker_env

    class _CtypesLibrary:

        def __init__(self, path):
            self.dll = ctypes.CDLL(path)
            self.rpc_port = 4242

        def flagcxP2pEngineCreate(self):
            return object()

        def flagcxP2pEngineDestroy(self, engine):
            pass
    monkeypatch.setattr(module, 'FLAGCXLibrary', _CtypesLibrary)
    library_path = str(tmp_path / 'libflagcx.so')
    monkeypatch.setenv('FLAGCX_LIB_PATH', library_path)
    loaded = []
    monkeypatch.setattr(ctypes, 'CDLL', lambda path: loaded.append(path) or object())
    support.TP_STATE['size'] = 1
    support.TP_STATE['rank'] = 0
    worker = module.FlagCXConnectorWorker(_cfg(role='kv_producer'), 'eng-ctypes', support.FakeKVCacheConfig([support.full_group(0)]))
    assert loaded == [library_path]
    worker.shutdown()

    def fail(path):
        raise OSError('no such shared object')
    monkeypatch.setattr(ctypes, 'CDLL', fail)
    with pytest.raises(OSError, match='no such shared object'):
        module.FlagCXConnectorWorker(_cfg(role='kv_producer'), 'eng-ctypes', support.FakeKVCacheConfig([support.full_group(0)]))