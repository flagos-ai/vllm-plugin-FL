"""CPU branch tests for vllm_fl.distributed.device_communicators.flagcx.

torch.distributed / FlagCX native handles are replaced with stubs. No real
collective, device or wheel is required.
"""
import ctypes
import sys
import threading
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
import torch
import vllm_fl.distributed.device_communicators.flagcx as fx

class FakeDevice:
    """torch.device double that keeps only the attributes the module reads."""

    def __init__(self, name):
        if isinstance(name, FakeDevice):
            self.type = name.type
            self.index = name.index
            return
        text = str(name)
        if ':' in text:
            (self.type, idx) = text.split(':', 1)
            self.index = int(idx)
        else:
            (self.type, self.index) = (text, None)

    def __str__(self):
        return self.type if self.index is None else f'{self.type}:{self.index}'
    __repr__ = __str__

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

class FakeTensor:

    def __init__(self, device, shape=(4,), dtype=torch.float32):
        self.device = device
        self.shape = tuple(shape) if not callable(shape) else shape
        self.dtype = dtype
        self.copies = []

    def data_ptr(self):
        return 4242

    def numel(self):
        total = 1
        for size in self.shape:
            total *= size
        return total

    def __getitem__(self, _sl):
        return self

    def copy_(self, other):
        self.copies.append(other)

def buffer_type(ptr=0):
    return ('buf', ptr)

class _DType:

    @staticmethod
    def from_torch(dtype):
        return f'dtype:{dtype}'

class _RedOp:

    @staticmethod
    def from_torch(op):
        return f'op:{op}'

@pytest.fixture
def bindings(monkeypatch):
    monkeypatch.setattr(fx, 'buffer_type', buffer_type)
    monkeypatch.setattr(fx, 'flagcxDataTypeEnum', _DType)
    monkeypatch.setattr(fx, 'flagcxRedOpTypeEnum', _RedOp)
    monkeypatch.setattr(fx, 'flagcxComm_t', object)

def _flagcx(**extra):
    ns = SimpleNamespace(flagcxAllReduce=Mock(), flagcxAllGather=Mock(), flagcxReduce=Mock(), flagcxReduceScatter=Mock(), flagcxSend=Mock(), flagcxRecv=Mock(), flagcxBroadcast=Mock(), flagcxGroupStart=Mock(), flagcxGroupEnd=Mock(), adaptor_stream_copy=Mock(return_value='flagcx_stream'), adaptor_stream_free=Mock())
    ns.__dict__.update(extra)
    return ns

def make_comm(device=None, rank=0, world_size=2, disabled=False, comm='COMM'):
    obj = fx.PyFlagcxCommunicator.__new__(fx.PyFlagcxCommunicator)
    obj.comm = comm
    obj.device = torch.device('cpu') if device is None else device
    obj.rank = rank
    obj.world_size = world_size
    obj.disabled = disabled
    obj.available = True
    obj._all_reduce_stream = None
    obj._destroy_lock = threading.Lock()
    obj._destroy_started = False
    obj._destroy_thread = None
    obj.flagcx = _flagcx()
    return obj

def test_is_trace_tensor_classification(monkeypatch):
    assert fx.is_trace_tensor(None) is False
    assert fx.is_trace_tensor(5) is False
    assert fx.is_trace_tensor(torch.ones(2)) is False
    assert fx.is_trace_tensor(torch.empty(2, device='meta')) is True

def test_all_reduce_disabled_returns_none(bindings):
    comm = make_comm(disabled=True)
    assert comm.all_reduce(torch.ones(2)) is None

def test_all_reduce_trace_tensor_returns_empty_like(bindings):
    comm = make_comm()
    meta = torch.empty(3, device='meta')
    out = comm.all_reduce(meta)
    assert out.device.type == 'meta'
    assert out.shape == meta.shape

def test_all_reduce_rejects_foreign_device(bindings):
    comm = make_comm(device=torch.device('cpu'))
    bad = FakeTensor(SimpleNamespace(type='cuda'))
    with pytest.raises(AssertionError):
        comm.all_reduce(bad, stream=object())

def test_all_reduce_allocates_output_and_dispatches(bindings):
    comm = make_comm()
    src = torch.arange(4, dtype=torch.float32)
    stream = object()
    out = comm.all_reduce(src, stream=stream)
    assert out.shape == src.shape and out is not src
    call = comm.flagcx.flagcxAllReduce.call_args
    args = call.args
    assert args[0] == ('buf', src.data_ptr())
    assert args[1] == ('buf', out.data_ptr())
    assert args[2] == src.numel()
    assert args[3] == f'dtype:{src.dtype}'
    assert args[4] == f'op:{fx.ReduceOp.SUM}'
    assert args[5] == 'COMM'
    assert args[6] == 'flagcx_stream'
    comm.flagcx.adaptor_stream_copy.assert_called_once_with(stream)
    comm.flagcx.adaptor_stream_free.assert_called_once_with('flagcx_stream')

def test_all_reduce_uses_submit_stream_when_present(bindings):
    comm = make_comm()
    comm._all_reduce_stream = SimpleNamespace(submit=Mock())
    src = torch.ones(2)
    out = torch.zeros(2)
    stream = object()
    assert comm.all_reduce(src, out, stream=stream) is out
    comm._all_reduce_stream.submit.assert_called_once_with(comm._all_reduce_on_stream, src, out, fx.ReduceOp.SUM, stream)
    comm.flagcx.flagcxAllReduce.assert_not_called()

def test_all_reduce_on_stream_frees_even_on_failure(bindings):
    comm = make_comm()
    comm.flagcx.flagcxAllReduce.side_effect = RuntimeError('boom')
    with pytest.raises(RuntimeError):
        comm._all_reduce_on_stream(torch.ones(1), torch.ones(1), fx.ReduceOp.SUM, object())
    comm.flagcx.adaptor_stream_free.assert_called_once()

def test_all_gather_disabled_and_trace(bindings):
    assert make_comm(disabled=True).all_gather(torch.ones(2), torch.ones(2)) is None
    comm = make_comm()
    meta = torch.empty(2, device='meta')
    assert comm.all_gather(meta, meta) is None

def test_all_gather_rejects_foreign_device(bindings):
    comm = make_comm()
    with pytest.raises(AssertionError):
        comm.all_gather(FakeTensor(SimpleNamespace(type='cuda')), FakeTensor(SimpleNamespace(type='cuda')))

def test_all_gather_dispatches(bindings):
    comm = make_comm()
    inp = torch.ones(2)
    out = torch.empty(4)
    comm.all_gather(out, inp, stream=object())
    args = comm.flagcx.flagcxAllGather.call_args.args
    assert args[0] == ('buf', inp.data_ptr())
    assert args[1] == ('buf', out.data_ptr())
    assert args[2] == inp.numel()
    assert args[3] == f'dtype:{inp.dtype}'
    assert args[4] == 'COMM'
    comm.flagcx.adaptor_stream_free.assert_called_once()

def test_all_gatherv_disabled_and_trace(bindings):
    assert make_comm(disabled=True).all_gatherv(torch.ones(4), torch.ones(2), [2, 2]) is None
    comm = make_comm(device=torch.device('meta'))
    meta = torch.empty(2, device='meta')
    assert comm.all_gatherv(torch.empty(4, device='meta'), meta, [2, 2]) is None

def test_all_gatherv_validates_sizes_and_output(bindings):
    comm = make_comm()
    fake_dev = torch.device('cpu')
    inp = FakeTensor(fake_dev, (2,))
    out = FakeTensor(fake_dev, (3,))
    with pytest.raises(AssertionError):
        comm.all_gatherv(out, inp, [2, 2], stream=object())

def test_all_gatherv_broadcasts_each_slice(bindings):
    comm = make_comm(rank=1)
    dev = torch.device('cpu')
    inp = FakeTensor(dev, (2,))
    out = FakeTensor(dev, (4,))
    comm.all_gatherv(out, inp, [2, 2], stream=object())
    assert comm.flagcx.flagcxBroadcast.call_count == 2
    roots = [c.args[4] for c in comm.flagcx.flagcxBroadcast.call_args_list]
    assert roots == [0, 1]
    comm.flagcx.flagcxGroupStart.assert_called_once()
    comm.flagcx.flagcxGroupEnd.assert_called_once()
    comm.flagcx.adaptor_stream_free.assert_called_once()

def test_all_gatherv_npu_rank_size_and_own_slice_copy(bindings, monkeypatch):
    monkeypatch.setattr(fx.torch, 'npu', SimpleNamespace(stream=lambda _s: nullcontext()), raising=False)
    dev = SimpleNamespace(type='npu')
    comm = make_comm(device=dev, rank=0)
    inp = FakeTensor(dev, (2,))
    out = FakeTensor(dev, (4,))
    comm.all_gatherv(out, inp, [2, 2], stream=SimpleNamespace(npu_stream='S'))
    assert out.copies == [inp]
    bad = FakeTensor(dev, (3,))
    with pytest.raises(AssertionError):
        comm.all_gatherv(FakeTensor(dev, (4,)), bad, [2, 2], stream=object())

def test_reduce_scatter_paths(bindings):
    assert make_comm(disabled=True).reduce_scatter(torch.ones(2), torch.ones(4)) is None
    meta_comm = make_comm(device=torch.device('meta'))
    meta = torch.empty(2, device='meta')
    assert meta_comm.reduce_scatter(meta, torch.empty(4, device='meta')) is None
    comm = make_comm()
    with pytest.raises(AssertionError):
        comm.reduce_scatter(FakeTensor(SimpleNamespace(type='cuda')), FakeTensor(SimpleNamespace(type='cuda')))
    out = torch.empty(2)
    inp = torch.ones(4)
    comm.reduce_scatter(out, inp, stream=object())
    args = comm.flagcx.flagcxReduceScatter.call_args.args
    assert args[2] == out.numel()
    assert args[3] == f'dtype:{inp.dtype}'
    assert args[4] == f'op:{fx.ReduceOp.SUM}'
    comm.flagcx.adaptor_stream_free.assert_called_once()

def test_reduce_scatterv_paths(bindings):
    assert make_comm(disabled=True).reduce_scatterv(torch.ones(2), torch.ones(4), [2, 2]) is None
    meta_comm = make_comm(device=torch.device('meta'))
    meta = torch.empty(2, device='meta')
    assert meta_comm.reduce_scatterv(meta, torch.empty(4, device='meta'), [2, 2]) is None
    comm = make_comm()
    with pytest.raises(AssertionError):
        comm.reduce_scatterv(FakeTensor(SimpleNamespace(type='cuda')), FakeTensor(SimpleNamespace(type='cuda')), [2, 2])
    dev = torch.device('cpu')
    out = FakeTensor(dev, (2,))
    inp = FakeTensor(dev, (4,))
    comm.reduce_scatterv(out, inp, [2, 2], stream=object())
    assert comm.flagcx.flagcxReduce.call_count == 2
    roots = [c.args[5] for c in comm.flagcx.flagcxReduce.call_args_list]
    assert roots == [0, 1]
    comm.flagcx.flagcxGroupStart.assert_called_once()
    comm.flagcx.flagcxGroupEnd.assert_called_once()

@pytest.mark.parametrize('method', ['send', 'recv'])
def test_point_to_point_paths(bindings, method):
    assert getattr(make_comm(disabled=True), method)(torch.ones(2), 1) is None
    meta_comm = make_comm(device=torch.device('meta'))
    assert getattr(meta_comm, method)(torch.empty(2, device='meta'), 1) is None
    comm = make_comm()
    with pytest.raises(AssertionError):
        getattr(comm, method)(FakeTensor(SimpleNamespace(type='cuda')), 1)
    tensor = torch.ones(2)
    if method == 'send':
        comm.send(tensor, 1, stream=object())
        called = comm.flagcx.flagcxSend
    else:
        comm.recv(tensor, 1, stream=object())
        called = comm.flagcx.flagcxRecv
    args = called.call_args.args
    assert args[0] == ('buf', tensor.data_ptr())
    assert args[1] == tensor.numel()
    comm.flagcx.adaptor_stream_free.assert_called_once()

def test_broadcast_paths(bindings):
    assert make_comm(disabled=True).broadcast(torch.ones(2), 0) is None
    assert make_comm(device=torch.device('meta')).broadcast(torch.empty(2, device='meta'), 0) is None
    comm = make_comm()
    with pytest.raises(AssertionError):
        comm.broadcast(FakeTensor(SimpleNamespace(type='cuda')), 0)
    tensor = torch.ones(2)
    comm.broadcast(tensor, 0, stream=object())
    args = comm.flagcx.flagcxBroadcast.call_args.args
    assert args[0] == ('buf', tensor.data_ptr())
    assert args[1] == ('buf', tensor.data_ptr())

def test_broadcast_non_root_sends_empty_buffer(bindings):
    comm = make_comm(rank=1)
    tensor = torch.ones(2)
    comm.broadcast(tensor, 0, stream=object())
    args = comm.flagcx.flagcxBroadcast.call_args.args
    assert args[0] == ('buf', 0)
    assert args[1] == ('buf', tensor.data_ptr())

def test_copy_flagcx_stream_wraps_npu_stream(bindings):
    npu_stream = SimpleNamespace(npu_stream='npu_stream')
    comm = make_comm(device=SimpleNamespace(type='npu'))
    assert comm._copy_flagcx_stream(npu_stream) == 'flagcx_stream'
    passed = comm.flagcx.adaptor_stream_copy.call_args.args[0]
    assert passed.cuda_stream == 'npu_stream'

def test_copy_flagcx_stream_passthrough(bindings):
    comm = make_comm()
    marker = object()
    comm._copy_flagcx_stream(marker)
    comm.flagcx.adaptor_stream_copy.assert_called_once_with(marker)

def test_current_stream_dispatch(bindings, monkeypatch):
    calls = []
    monkeypatch.setattr(fx.torch, 'npu', SimpleNamespace(current_stream=lambda d: calls.append(('npu', d)) or 'N'), raising=False)
    monkeypatch.setattr(fx.torch, 'txda', SimpleNamespace(current_stream=lambda d: calls.append(('txda', d)) or 'T'), raising=False)
    monkeypatch.setattr(fx.torch, 'cuda', SimpleNamespace(current_stream=lambda d: calls.append(('cuda', d)) or 'C'))
    monkeypatch.setattr(fx, 'current_stream', lambda : calls.append(('default', None)) or 'D')
    dev = SimpleNamespace(type='npu')
    assert make_comm(device=dev)._current_stream() == 'N'
    dev = SimpleNamespace(type='txda')
    assert make_comm(device=dev)._current_stream() == 'T'
    dev = SimpleNamespace(type='cuda')
    monkeypatch.setattr(fx, '_flagcx_amd_source', True)
    assert make_comm(device=dev)._current_stream() == 'C'
    monkeypatch.setattr(fx, '_flagcx_amd_source', False)
    assert make_comm(device=dev)._current_stream() == 'D'

@pytest.mark.parametrize('npu,amd', [(True, False), (False, True), (False, False)])
def test_group_start_end_variants(bindings, monkeypatch, npu, amd):
    monkeypatch.setattr(fx, '_flagcx_amd_source', amd)
    comm = make_comm(device=SimpleNamespace(type='npu' if npu else 'cpu'))
    comm.group_start()
    comm.group_end()
    if npu or amd:
        comm.flagcx.flagcxGroupStart.assert_called_once_with('COMM')
        comm.flagcx.flagcxGroupEnd.assert_called_once_with('COMM')
    else:
        comm.flagcx.flagcxGroupStart.assert_called_once_with()
        comm.flagcx.flagcxGroupEnd.assert_called_once_with()

def test_destroy_noop_for_non_txda(bindings):
    comm = make_comm(device=torch.device('cpu'))
    comm.destroy()
    assert comm.comm == 'COMM'
    assert not comm._destroy_started

def test_destroy_txda_without_comm_is_noop(bindings):
    comm = make_comm(device=SimpleNamespace(type='txda'), comm=None)
    comm.destroy()
    assert not comm._destroy_started

def test_destroy_txda_runs_native_abort(bindings, monkeypatch):
    started = []

    class ThreadDouble:

        def __init__(self, target, name, daemon):
            self.target = target
            self.name = name
            self.daemon = daemon
            started.append(self)

        def start(self):
            self.target()

        def is_alive(self):
            return False

        def join(self, timeout=None):
            self.joined = timeout
    monkeypatch.setattr(fx.threading, 'Thread', ThreadDouble)
    monkeypatch.setattr(fx.torch, 'txda', SimpleNamespace(device=lambda _d: nullcontext()), raising=False)
    native = Mock(return_value=0)
    library = _flagcx()
    library.lib = {'flagcxCommAbort': native}
    library.FLAGCX_CHECK = Mock()
    comm = make_comm(device=SimpleNamespace(type='txda'))
    comm.flagcx = library
    comm.destroy()
    assert comm._destroy_started
    assert comm.comm is None
    assert comm.disabled and (not comm.available)
    native.assert_called_once_with('COMM')
    native.argtypes and library.FLAGCX_CHECK.assert_called_once_with(0)
    assert started and started[0].joined == fx._TXDA_ABORT_TIMEOUT_SECONDS
    comm.destroy()
    native.assert_called_once()

def test_destroy_txda_swallows_abort_failure(bindings, monkeypatch):

    class ThreadDouble:

        def __init__(self, target, name, daemon):
            self.target = target

        def start(self):
            self.target()

        def is_alive(self):
            return False

        def join(self, timeout=None):
            pass
    monkeypatch.setattr(fx.threading, 'Thread', ThreadDouble)
    monkeypatch.setattr(fx.torch, 'txda', SimpleNamespace(device=lambda _d: nullcontext()), raising=False)
    library = _flagcx()
    library.lib = {'flagcxCommAbort': Mock(side_effect=RuntimeError('native'))}
    comm = make_comm(device=SimpleNamespace(type='txda'))
    comm.flagcx = library
    comm.destroy()
    assert comm._destroy_started

def test_destroy_txda_thread_start_failure_logs(bindings, monkeypatch):

    class ThreadDouble:

        def __init__(self, target, name, daemon):
            pass

        def start(self):
            raise RuntimeError('cannot start')

        def is_alive(self):
            return False

        def join(self, timeout=None):
            pass
    monkeypatch.setattr(fx.threading, 'Thread', ThreadDouble)
    comm = make_comm(device=SimpleNamespace(type='txda'))
    comm.destroy()
    assert comm._destroy_started
    assert comm.comm is None

def test_destroy_txda_timeout_warns(bindings, monkeypatch):
    warnings = []
    monkeypatch.setattr(fx, 'logger', SimpleNamespace(warning=lambda *a: warnings.append(a), exception=Mock()))

    class ThreadDouble:

        def __init__(self, target, name, daemon):
            self.target = target

        def start(self):
            pass

        def is_alive(self):
            return True

        def join(self, timeout=None):
            self.joined = timeout
    monkeypatch.setattr(fx.threading, 'Thread', ThreadDouble)
    comm = make_comm(device=SimpleNamespace(type='txda'))
    comm.destroy()
    assert warnings

class StatelessGroup:

    def __init__(self, rank=0, world_size=2, uid=None):
        self.rank = rank
        self.world_size = world_size
        self.uid = uid
        self.calls = []

    def broadcast_obj(self, obj, src=0):
        self.calls.append((obj, src))
        return self.uid if self.uid is not None else obj

@pytest.fixture
def patch_stateless(monkeypatch):
    monkeypatch.setattr(fx, 'StatelessProcessGroup', StatelessGroup)
    return StatelessGroup

def _init_env(monkeypatch, library_cls, device_ctx=nullcontext(), stream='stream'):
    monkeypatch.setattr(fx, 'FLAGCXLibrary', library_cls)
    monkeypatch.setattr(fx, 'flagcxUniqueId', lambda : SimpleNamespace(internal=[0, 0, 0]))
    monkeypatch.setattr(fx, '_flagcx_wheel_library', None)
    monkeypatch.setattr(fx, 'PyFlagcxCommunicator', fx.PyFlagcxCommunicator)
    monkeypatch.setattr(fx.PyFlagcxCommunicator, '_current_stream', lambda self: SimpleNamespace(synchronize=lambda : None))
    monkeypatch.setattr(fx.PyFlagcxCommunicator, 'all_reduce', lambda self, data: None)
    monkeypatch.setattr(fx.torch, 'zeros', lambda *a, **k: object())
    monkeypatch.setattr(fx.torch, 'device', FakeDevice)
    for name in ('musa', 'npu', 'txda', 'cuda'):
        monkeypatch.setattr(fx.torch, name, SimpleNamespace(device=lambda _d: device_ctx), raising=False)
    return device_ctx

def test_init_world_size_one_disables(monkeypatch, patch_stateless):
    monkeypatch.setattr(fx, 'FLAGCXLibrary', Mock())
    comm = fx.PyFlagcxCommunicator(StatelessGroup(world_size=1), 'cpu')
    assert comm.available is False
    assert comm.disabled is True

def test_init_missing_library_disables(monkeypatch, patch_stateless):
    monkeypatch.setattr(fx, 'FLAGCXLibrary', None)
    monkeypatch.setattr(fx, '_flagcx_import_error', ImportError('missing'))
    comm = fx.PyFlagcxCommunicator(StatelessGroup(), 'cpu')
    assert comm.disabled is True
    assert isinstance(comm.init_error, ImportError)

@pytest.mark.parametrize('device_type', ['musa', 'npu', 'ptpu', 'txda', 'cuda'])
def test_init_selects_device_context_and_warms_up(monkeypatch, device_type):
    monkeypatch.setattr(fx, 'StatelessProcessGroup', StatelessGroup)
    created = []

    class Library:

        def __init__(self, path):
            created.append(path)
            self.comm = 'NATIVE'
            self.flagcxGetUniqueId = Mock(return_value='UID')

        def flagcxCommInitRank(self, world, uid, rank):
            self.init_args = (world, uid, rank)
            return 'NATIVE'
    stream = SimpleNamespace(synchronize=Mock())
    monkeypatch.setattr(fx.PyFlagcxCommunicator, '_current_stream', lambda self: stream)
    warm = []
    monkeypatch.setattr(fx.PyFlagcxCommunicator, 'all_reduce', lambda self, data, **k: warm.append(data))
    monkeypatch.setattr(fx.torch, 'zeros', lambda *a, **k: object())
    monkeypatch.setattr(fx.torch, 'device', FakeDevice)
    monkeypatch.setattr(fx, 'StatelessProcessGroup', StatelessGroup)
    monkeypatch.setattr(fx, 'flagcxUniqueId', lambda : SimpleNamespace(internal=[0, 0, 0]))
    monkeypatch.setattr(fx, 'FLAGCXLibrary', Library)
    monkeypatch.setattr(fx, '_flagcx_wheel_library', None)
    for name in ('musa', 'npu', 'txda', 'cuda'):
        monkeypatch.setattr(fx.torch, name, SimpleNamespace(device=lambda _d: nullcontext()), raising=False)
    dev = FakeDevice(device_type)
    comm = fx.PyFlagcxCommunicator(StatelessGroup(rank=0), dev, library_path='/lib.so')
    assert created == ['/lib.so']
    assert comm.device is dev
    assert comm.comm == 'NATIVE'
    assert comm.available and (not comm.disabled)
    assert warm
    stream.synchronize.assert_called_once()
    assert comm.flagcx.init_args == (2, 'UID', 0)

def test_init_npu_installs_all_reduce_stream(monkeypatch, patch_stateless):

    class Library:

        def __init__(self, path):
            self.flagcxGetUniqueId = Mock(return_value='UID')

        def flagcxCommInitRank(self, world, uid, rank):
            return 'NATIVE'

    class StreamWrapper:

        def __init__(self, device):
            self.device = device
    monkeypatch.setitem(sys.modules, 'vllm_fl.dispatch.backends.vendor.ascend.patches.flagcx', SimpleNamespace(FlagCXAllReduceStream=StreamWrapper))
    monkeypatch.setattr(fx, 'FLAGCXLibrary', Library)
    monkeypatch.setattr(fx, '_flagcx_wheel_library', '/wheel/libflagcx.so')
    monkeypatch.setattr(fx, 'flagcxUniqueId', lambda : SimpleNamespace(internal=[0, 0, 0]))
    monkeypatch.setattr(fx.PyFlagcxCommunicator, '_current_stream', lambda self: SimpleNamespace(synchronize=lambda : None))
    monkeypatch.setattr(fx.PyFlagcxCommunicator, 'all_reduce', lambda self, data: None)
    monkeypatch.setattr(fx.torch, 'zeros', lambda *a, **k: object())
    monkeypatch.setattr(fx.torch, 'device', FakeDevice)
    monkeypatch.setattr(fx.torch, 'npu', SimpleNamespace(device=lambda _d: nullcontext()), raising=False)
    dev = FakeDevice('npu')
    comm = fx.PyFlagcxCommunicator(StatelessGroup(rank=0), dev)
    assert isinstance(comm._all_reduce_stream, StreamWrapper)

def test_init_default_library_path_from_env(monkeypatch, patch_stateless):
    seen = []

    class Library:

        def __init__(self, path):
            seen.append(path)
            self.flagcxGetUniqueId = Mock(return_value='UID')

        def flagcxCommInitRank(self, world, uid, rank):
            return 'NATIVE'
    monkeypatch.setattr(fx, 'FLAGCXLibrary', Library)
    monkeypatch.setattr(fx, '_flagcx_wheel_library', None)
    monkeypatch.setenv('FLAGCX_PATH', '/opt/flagcx')
    monkeypatch.setattr(fx, 'flagcxUniqueId', lambda : SimpleNamespace(internal=[0, 0, 0]))
    monkeypatch.setattr(fx.PyFlagcxCommunicator, '_current_stream', lambda self: SimpleNamespace(synchronize=lambda : None))
    monkeypatch.setattr(fx.PyFlagcxCommunicator, 'all_reduce', lambda self, data: None)
    monkeypatch.setattr(fx.torch, 'zeros', lambda *a, **k: object())
    monkeypatch.setattr(fx.torch, 'device', FakeDevice)
    monkeypatch.setattr(fx.torch, 'ptpu', SimpleNamespace(device=lambda _d: nullcontext()), raising=False)
    fx.PyFlagcxCommunicator(StatelessGroup(rank=0), FakeDevice('ptpu'))
    assert seen == ['/opt/flagcx\\build/lib/libflagcx.so'] or seen == ['/opt/flagcx/build/lib/libflagcx.so']