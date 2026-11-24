"""Coverage tests for the vllm_fl.distributed communicator stack (Enflame S60).

The real sources are loaded from the read-only checkout while torch/vllm/FlagCX
are replaced by in-memory stubs, so no GCU, FlagCX C library, process group,
collective, thread or network call is ever executed for real.
"""

from __future__ import annotations

import contextlib
import importlib.util
import logging
import os
import sys
import threading
import types
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

_HERE = Path(__file__).resolve().parent
_SOURCE_ROOT = Path(
    os.environ.get("ENFLAME_SRC") or (_HERE.parent / "enflame-src-030b")
)
_FLAGCX_SRC = (
    _SOURCE_ROOT / "vllm_fl" / "distributed" / "device_communicators" / "flagcx.py"
)
_COMM_SRC = _SOURCE_ROOT / "vllm_fl" / "distributed" / "communicator.py"
_INIT_SRC = _SOURCE_ROOT / "vllm_fl" / "distributed" / "__init__.py"
for _path in (_FLAGCX_SRC, _COMM_SRC, _INIT_SRC):
    if not _path.is_file():
        pytest.skip(
            f"enflame source checkout not found at {_SOURCE_ROOT}",
            allow_module_level=True,
        )


# ---------------------------------------------------------------------------
# Minimal stand-ins for torch.Tensor / torch.device
# ---------------------------------------------------------------------------
class _Dev:
    def __init__(self, spec="cpu:0"):
        if isinstance(spec, _Dev):
            self._spec = spec._spec
        else:
            self._spec = str(spec)
        if ":" in self._spec:
            self.type, idx = self._spec.split(":", 1)
            self.index = int(idx) if idx.isdigit() else idx
        else:
            self.type = self._spec
            self.index = None

    def __repr__(self):
        return f"device(type={self.type!r})"

    def __eq__(self, other):
        return isinstance(other, _Dev) and other._spec == self._spec

    def __hash__(self):
        return hash(self._spec)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _T:
    def __init__(self, shape=(), dtype="float32", device=None, data=None,
                 is_meta=False):
        if isinstance(shape, int):
            shape = (shape,)
        self._shape = tuple(shape)
        self.dtype = dtype or "float32"
        self.device = device if isinstance(device, _Dev) else _Dev(device or "cpu:0")
        self._data = list(data) if data is not None else None
        self.is_meta = bool(is_meta)

    def size(self):
        return self._shape

    @property
    def shape(self):
        return self._shape

    def dim(self):
        return len(self._shape)

    def numel(self):
        total = 1
        for dim in self._shape:
            total *= int(dim)
        return total

    def data_ptr(self):
        return 4096

    def clone(self):
        return _T(self._shape, self.dtype, self.device, self._data)

    def contiguous(self):
        return self

    def movedim(self, src, dst):
        dims = list(self._shape)
        dims.insert(dst, dims.pop(src))
        return _T(tuple(dims), self.dtype, self.device)

    def reshape(self, shape):
        if isinstance(shape, int):
            shape = (shape,)
        flat = []
        for item in shape:
            if isinstance(item, (tuple, list)):
                flat.extend(item)
            else:
                flat.append(item)
        return _T(tuple(flat), self.dtype, self.device)

    def tolist(self):
        return list(self._data) if self._data is not None else []

    def copy_(self, other):
        return self

    def synchronize(self):
        return None

    def __getitem__(self, item):
        if not isinstance(item, tuple):
            item = (item,)
        first = item[0]
        if isinstance(first, slice):
            start = 0 if first.start is None else first.start
            stop = self._shape[0] if first.stop is None else first.stop
            length = max(0, stop - start)
            return _T((length,) + self._shape[1:], self.dtype, self.device,
                      self._data)
        return _T(self._shape[:-1], self.dtype, self.device, self._data)


class _FakeStream:
    def __init__(self):
        self.synchronize = mock.Mock()
        self.npu_stream = 11
        self.gcu_stream = 22
        self.cuda_stream = 33


def _ctx(*args, **kwargs):
    return contextlib.nullcontext()


def _build_fake_torch():
    torch_mod = types.ModuleType("torch")
    torch_mod.Tensor = _T
    torch_mod.device = _Dev
    torch_mod.Size = tuple
    torch_mod.dtype = type("dtype", (), {})
    torch_mod.float32 = "float32"
    torch_mod.uint8 = "uint8"
    torch_mod.version = SimpleNamespace(hip=None, cuda="12.1")
    torch_mod.nn = types.ModuleType("torch.nn")
    torch_mod._C = SimpleNamespace()
    torch_mod.empty = lambda size=(), dtype=None, device=None, **_: _T(
        size, dtype or "float32", device
    )
    torch_mod.empty_like = lambda tensor, **_: _T(
        tensor.shape, tensor.dtype, tensor.device
    )
    torch_mod.zeros = lambda size=(), dtype=None, device=None, **_: _T(
        size, dtype or "float32", device
    )
    torch_mod.ByteTensor = lambda values: _T(
        (len(list(values)),), "uint8", _Dev("cpu:0"), data=list(values)
    )

    stream = _FakeStream()
    for backend in ("cuda", "npu", "musa", "gcu", "txda"):
        setattr(
            torch_mod,
            backend,
            SimpleNamespace(
                device=_ctx,
                stream=_ctx,
                current_stream=lambda *a, **k: stream,
            ),
        )

    dist = types.ModuleType("torch.distributed")
    dist.Backend = SimpleNamespace(NCCL="nccl")
    dist.ProcessGroup = type("ProcessGroup", (), {})
    dist.ReduceOp = type("ReduceOp", (), {"SUM": "sum", "REDUCE": "reduce"})
    dist.is_initialized = mock.Mock(return_value=True)
    dist.get_backend = mock.Mock(return_value="gloo")
    dist.get_rank = mock.Mock(return_value=0)
    dist.get_world_size = mock.Mock(return_value=2)
    dist.get_process_group_ranks = mock.Mock(return_value=[0, 1])
    dist.broadcast = mock.Mock()
    dist.all_reduce = mock.Mock()
    dist.send = mock.Mock()
    dist.recv = mock.Mock()
    torch_mod.distributed = dist
    return torch_mod, stream


_TORCH, _STREAM = _build_fake_torch()


class _UniqueId:
    def __init__(self, size=4):
        self.internal = [0] * size


class _StatelessPG:
    def __init__(self, rank=0, world_size=1):
        self.rank = rank
        self.world_size = world_size
        self.broadcast_obj = mock.Mock(side_effect=lambda obj, src=0: obj)


class _Wrapper:  # mirrors plugin.interservice.flagcx_wrapper
    FLAGCXLibrary = type("FLAGCXLibrary", (), {})
    buffer_type = staticmethod(lambda value=None: value)
    flagcxComm_t = object
    flagcxDataTypeEnum = SimpleNamespace(from_torch=mock.Mock())
    flagcxUniqueId = _UniqueId
    flagcxRedOpTypeEnum = SimpleNamespace(from_torch=mock.Mock())


class _BaseCommunicator:
    default_all2all_backend = "naive"

    def __init__(self, cpu_group, device=None, device_group=None,
                 unique_name="", global_ranks=None, global_world_size=None,
                 use_all2all=False):
        self.cpu_group = cpu_group
        self.device_group = device_group
        self.device = device if isinstance(device, _Dev) else _Dev(
            device or "cpu:0"
        )
        self.unique_name = unique_name
        self.rank = int(getattr(cpu_group, "rank", 0))
        self.world_size = int(getattr(cpu_group, "world_size", 1))
        self.rank_in_group = self.rank
        self.local_rank = self.rank
        self.global_ranks = list(global_ranks or [0])
        self.global_world_size = global_world_size or 1
        self.ranks = list(getattr(cpu_group, "ranks", [0]))
        self.use_all2all = use_all2all
        self.all2all_backend = type(self).default_all2all_backend
        self.all2all_manager = None
        self._base_all_gather = mock.Mock()
        self._base_broadcast = mock.Mock()

    def all_gather(self, input_, dim=-1):
        self._base_all_gather(input_, dim)
        return "base-all-gather"

    def broadcast(self, tensor, src=0):
        self._base_broadcast(tensor, src)
        return tensor


def _make_module(name, path=False, **attrs):
    module = types.ModuleType(name)
    if path:
        module.__path__ = []
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def _load_file(name, path, package=False):
    kwargs = {"submodule_search_locations": []} if package else {}
    spec = importlib.util.spec_from_file_location(name, str(path), **kwargs)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def _load_targets():
    saved_modules = dict(sys.modules)
    saved_flagcx_path = os.environ.pop("FLAGCX_PATH", None)
    try:
        try:
            import torch  # noqa: F401
        except Exception:
            sys.modules["torch"] = _TORCH
            sys.modules["torch.distributed"] = _TORCH.distributed

        vllm = _make_module("vllm", path=True)
        vllm_dist = _make_module("vllm.distributed", path=True)
        vllm_dev_comm = _make_module(
            "vllm.distributed.device_communicators", path=True
        )
        vllm_utils_dist = _make_module(
            "vllm.distributed.utils", StatelessProcessGroup=_StatelessPG
        )
        vllm_utils = _make_module("vllm.utils", path=True)
        vllm_torch_utils = _make_module(
            "vllm.utils.torch_utils",
            current_stream=mock.Mock(return_value=_STREAM),
        )
        vllm_logger = _make_module(
            "vllm.logger", init_logger=lambda name: logging.getLogger(name)
        )
        vllm_platforms = _make_module(
            "vllm.platforms",
            current_platform=SimpleNamespace(vendor_name="gcu"),
        )
        vllm_base = _make_module(
            "vllm.distributed.device_communicators.base_device_communicator",
            DeviceCommunicatorBase=_BaseCommunicator,
        )

        vllm_fl = _make_module("vllm_fl", path=True)
        vllm_fl_utils = _make_module(
            "vllm_fl.utils",
            _env_vendor_name=lambda: os.environ.get("VLLM_FL_VENDOR", ""),
        )
        vllm_fl_dist = _make_module("vllm_fl.distributed", path=True)
        vllm_fl_dev_comm = _make_module(
            "vllm_fl.distributed.device_communicators", path=True
        )

        plugin = _make_module("plugin", path=True)
        plugin_interservice = _make_module("plugin.interservice", path=True)
        flagcx_wrapper = _make_module(
            "plugin.interservice.flagcx_wrapper",
            FLAGCXLibrary=_Wrapper.FLAGCXLibrary,
            buffer_type=_Wrapper.buffer_type,
            flagcxComm_t=_Wrapper.flagcxComm_t,
            flagcxDataTypeEnum=_Wrapper.flagcxDataTypeEnum,
            flagcxUniqueId=_Wrapper.flagcxUniqueId,
            flagcxRedOpTypeEnum=_Wrapper.flagcxRedOpTypeEnum,
        )

        stubs = [
            vllm,
            vllm_dist,
            vllm_dev_comm,
            vllm_utils_dist,
            vllm_utils,
            vllm_torch_utils,
            vllm_logger,
            vllm_platforms,
            vllm_base,
            vllm_fl,
            vllm_fl_utils,
            vllm_fl_dist,
            vllm_fl_dev_comm,
            plugin,
            plugin_interservice,
            flagcx_wrapper,
        ]
        for module in stubs:
            sys.modules[module.__name__] = module

        vllm.distributed = vllm_dist
        vllm_dist.device_communicators = vllm_dev_comm
        vllm_dist.utils = vllm_utils_dist
        vllm_dev_comm.base_device_communicator = vllm_base
        vllm.utils = vllm_utils
        vllm_utils.torch_utils = vllm_torch_utils
        vllm.platforms = vllm_platforms
        vllm_fl.utils = vllm_fl_utils
        vllm_fl.distributed = vllm_fl_dist
        vllm_fl_dist.device_communicators = vllm_fl_dev_comm
        plugin.interservice = plugin_interservice
        plugin_interservice.flagcx_wrapper = flagcx_wrapper

        flagcx_mod = _load_file(
            "vllm_fl.distributed.device_communicators.flagcx", _FLAGCX_SRC
        )
        vllm_fl_dev_comm.flagcx = flagcx_mod

        comm_mod = _load_file("vllm_fl.distributed.communicator", _COMM_SRC)
        vllm_fl_dist.communicator = comm_mod

        init_mod = _load_file("vllm_fl.distributed", _INIT_SRC, package=True)
        vllm_fl.distributed = init_mod
        sys.modules["vllm_fl.distributed"] = init_mod

        return flagcx_mod, comm_mod, init_mod
    finally:
        for name in list(sys.modules):
            if name not in saved_modules:
                del sys.modules[name]
        for name, module in saved_modules.items():
            sys.modules[name] = module
        if saved_flagcx_path is not None:
            os.environ["FLAGCX_PATH"] = saved_flagcx_path


_IMPORT_ERROR = None
try:
    flagcx_mod, comm_mod, init_mod = _load_targets()
except Exception as exc:  # pragma: no cover - environment guard
    flagcx_mod = comm_mod = init_mod = None
    _IMPORT_ERROR = exc

pytestmark = pytest.mark.skipif(
    flagcx_mod is None,
    reason=f"cannot load vllm_fl.distributed sources: {_IMPORT_ERROR!r}",
)


@pytest.fixture(autouse=True)
def _stub_torch(monkeypatch):
    if flagcx_mod is None:
        return
    monkeypatch.setattr(flagcx_mod, "torch", _TORCH)
    monkeypatch.setattr(flagcx_mod, "dist", _TORCH.distributed)
    monkeypatch.setattr(comm_mod, "torch", _TORCH)


def _install_module(monkeypatch, name, module):
    parts = name.split(".")
    for index in range(1, len(parts)):
        parent_name = ".".join(parts[:index])
        parent = sys.modules.get(parent_name)
        if parent is None:
            parent = types.ModuleType(parent_name)
            parent.__path__ = []
            monkeypatch.setitem(sys.modules, parent_name, parent)
            grand = sys.modules.get(".".join(parts[: index - 1]))
            if grand is not None:
                monkeypatch.setattr(grand, parts[index - 1], parent,
                                    raising=False)
    monkeypatch.setitem(sys.modules, name, module)
    parent = sys.modules.get(".".join(parts[:-1]))
    if parent is not None:
        monkeypatch.setattr(parent, parts[-1], module, raising=False)


def _fake_platform(monkeypatch, vendor="gcu"):
    module = _make_module(
        "vllm.platforms", current_platform=SimpleNamespace(vendor_name=vendor)
    )
    _install_module(monkeypatch, "vllm.platforms", module)
    return module


# ---------------------------------------------------------------------------
# FlagCX communicator helpers
# ---------------------------------------------------------------------------
def _fake_library(legacy=False, **overrides):
    uid = _UniqueId()
    lib = SimpleNamespace(
        flagcxGetUniqueId=mock.Mock(
            return_value=SimpleNamespace(contents=uid) if legacy else uid
        ),
        flagcxCommInitRank=mock.Mock(return_value="NATIVE"),
        adaptor_stream_copy=mock.Mock(return_value="FS"),
        adaptor_stream_free=mock.Mock(),
        flagcxAllReduce=mock.Mock(),
        flagcxAllGather=mock.Mock(),
        flagcxBroadcast=mock.Mock(),
        flagcxSend=mock.Mock(),
        flagcxRecv=mock.Mock(),
        flagcxReduce=mock.Mock(),
        flagcxReduceScatter=mock.Mock(),
        flagcxGroupStart=mock.Mock(),
        flagcxGroupEnd=mock.Mock(),
    )
    if legacy:
        lib.handler = object()
    else:
        lib.devHandle = object()
    for key, value in overrides.items():
        setattr(lib, key, value)
    return lib


def _new_flagcx(**attrs):
    inst = flagcx_mod.PyFlagcxCommunicator.__new__(flagcx_mod.PyFlagcxCommunicator)
    inst.comm = "COMM"
    inst.available = True
    inst.disabled = False
    inst.device = _Dev("cpu:0")
    inst.rank = 0
    inst.world_size = 2
    inst.group = object()
    inst.flagcx = _fake_library()
    inst._destroy_lock = threading.Lock()
    inst._destroy_started = False
    inst._destroy_thread = None
    for key, value in attrs.items():
        setattr(inst, key, value)
    return inst


def _tensor(shape=(2,), device=None, is_meta=False, data=None):
    return _T(shape, "float32", device or _Dev("cpu:0"), data=data,
              is_meta=is_meta)


def _symbols(monkeypatch, lib=None):
    monkeypatch.setattr(
        flagcx_mod, "buffer_type", mock.Mock(side_effect=lambda *a: ("buf", a))
    )
    monkeypatch.setattr(
        flagcx_mod,
        "flagcxDataTypeEnum",
        SimpleNamespace(from_torch=mock.Mock(return_value="dtype")),
    )
    monkeypatch.setattr(
        flagcx_mod,
        "flagcxRedOpTypeEnum",
        SimpleNamespace(from_torch=mock.Mock(return_value="op")),
    )
    monkeypatch.setattr(flagcx_mod, "flagcxUniqueId", _UniqueId)
    monkeypatch.setattr(flagcx_mod, "flagcxComm_t", object)
    if lib is not None:
        monkeypatch.setattr(flagcx_mod, "FLAGCXLibrary", lib)


def _fresh_dist(**overrides):
    values = {
        "is_initialized": mock.Mock(return_value=True),
        "get_backend": mock.Mock(return_value="gloo"),
        "get_rank": mock.Mock(return_value=0),
        "get_world_size": mock.Mock(return_value=2),
        "get_process_group_ranks": mock.Mock(return_value=[0, 1]),
        "broadcast": mock.Mock(),
        "all_reduce": mock.Mock(),
        "send": mock.Mock(),
        "recv": mock.Mock(),
    }
    values.update(overrides)
    ns = SimpleNamespace(Backend=SimpleNamespace(NCCL="nccl"), **values)
    return ns


# ---------------------------------------------------------------------------
# flagcx: module level loaders / helpers
# ---------------------------------------------------------------------------
def test_is_trace_tensor_variants(monkeypatch):
    assert flagcx_mod.is_trace_tensor(None) is False
    assert flagcx_mod.is_trace_tensor(object()) is False
    assert flagcx_mod.is_trace_tensor(_tensor(is_meta=True)) is True
    assert flagcx_mod.is_trace_tensor(_tensor()) is False

    class _Trace(_T):
        pass

    monkeypatch.setattr(flagcx_mod, "_FAKE_TENSOR_CLS", (_Trace,))
    assert flagcx_mod.is_trace_tensor(_Trace()) is True


_WHEEL_API = (
    "class FLAGCXLibrary: pass\n"
    "buffer_type = int\n"
    "flagcxComm_t = object\n"
    "flagcxDataTypeEnum = object\n"
    "flagcxUniqueId = object\n"
    "flagcxRedOpTypeEnum = object\n"
)


def test_load_wheel_bindings_success(tmp_path):
    api = tmp_path / "api.py"
    api.write_text(_WHEEL_API)
    result = flagcx_mod._load_flagcx_wheel_bindings(api)
    assert len(result) == 6
    assert result[0].__name__ == "FLAGCXLibrary"
    sys.modules.pop(f"{flagcx_mod.__name__}._wheel_api", None)


def test_load_wheel_bindings_bad_spec(monkeypatch, tmp_path):
    monkeypatch.setattr(
        importlib.util, "spec_from_file_location", lambda *a, **k: None
    )
    with pytest.raises(ImportError):
        flagcx_mod._load_flagcx_wheel_bindings(tmp_path / "api.py")


def test_load_wheel_bindings_incomplete(tmp_path):
    api = tmp_path / "api.py"
    api.write_text("class FLAGCXLibrary: pass\n")
    with pytest.raises(ImportError):
        flagcx_mod._load_flagcx_wheel_bindings(api)
    sys.modules.pop(f"{flagcx_mod.__name__}._wheel_api", None)


def test_load_wheel_bindings_exec_error_restores_module(tmp_path, monkeypatch):
    api = tmp_path / "api.py"
    api.write_text("raise RuntimeError('boom')\n")
    name = f"{flagcx_mod.__name__}._wheel_api"
    monkeypatch.setitem(sys.modules, name, "previous")
    with pytest.raises(RuntimeError):
        flagcx_mod._load_flagcx_wheel_bindings(api)
    assert sys.modules[name] == "previous"


def test_load_amd_api_success(tmp_path):
    api = tmp_path / "api.py"
    api.write_text("VALUE = 42\n")
    module = flagcx_mod._load_amd_flagcx_api(api)
    assert module.VALUE == 42
    sys.modules.pop(f"{flagcx_mod.__name__}._amd_api", None)


def test_load_amd_api_bad_spec(monkeypatch, tmp_path):
    monkeypatch.setattr(
        importlib.util, "spec_from_file_location", lambda *a, **k: None
    )
    with pytest.raises(ImportError):
        flagcx_mod._load_amd_flagcx_api(tmp_path / "api.py")


def test_load_amd_api_exec_error(tmp_path):
    api = tmp_path / "api.py"
    api.write_text("raise RuntimeError('boom')\n")
    name = f"{flagcx_mod.__name__}._amd_api"
    with pytest.raises(RuntimeError):
        flagcx_mod._load_amd_flagcx_api(api)
    assert name not in sys.modules


@pytest.mark.parametrize(
    "vendor, expected",
    [("kunlunxin", True), ("gcu", False)],
)
def test_is_kunlunxin_from_env(monkeypatch, vendor, expected):
    monkeypatch.setattr(flagcx_mod, "_env_vendor_name", lambda: vendor)
    assert flagcx_mod._is_kunlunxin() is expected


def test_is_kunlunxin_from_platform(monkeypatch):
    monkeypatch.setattr(flagcx_mod, "_env_vendor_name", lambda: "")
    _fake_platform(monkeypatch, vendor="kunlunxin")
    assert flagcx_mod._is_kunlunxin() is True


def test_is_kunlunxin_import_error(monkeypatch):
    monkeypatch.setattr(flagcx_mod, "_env_vendor_name", lambda: "")
    monkeypatch.setitem(sys.modules, "vllm.platforms", None)
    assert flagcx_mod._is_kunlunxin() is False


def test_load_kunlunxin_api_plugin(tmp_path, monkeypatch):
    plugin = _make_module("plugin", path=True)
    interservice = _make_module("plugin.interservice", path=True)
    wrapper = _make_module("plugin.interservice.flagcx_wrapper", MARK=1)
    plugin.interservice = interservice
    interservice.flagcx_wrapper = wrapper
    monkeypatch.setitem(sys.modules, "plugin", plugin)
    monkeypatch.setitem(sys.modules, "plugin.interservice", interservice)
    monkeypatch.setitem(
        sys.modules, "plugin.interservice.flagcx_wrapper", wrapper
    )
    saved_path = list(sys.path)
    try:
        assert flagcx_mod._load_kunlunxin_flagcx_api(str(tmp_path)) is wrapper
        assert str(tmp_path) in sys.path
    finally:
        sys.path[:] = saved_path


def test_load_kunlunxin_api_flagcx_fallback(monkeypatch):
    api = SimpleNamespace(FLAGCXLibrary=object)
    monkeypatch.setitem(sys.modules, "flagcx", SimpleNamespace(api=api))
    monkeypatch.delitem(sys.modules, "plugin", raising=False)
    monkeypatch.delitem(sys.modules, "plugin.interservice", raising=False)
    monkeypatch.delitem(
        sys.modules, "plugin.interservice.flagcx_wrapper", raising=False
    )
    assert flagcx_mod._load_kunlunxin_flagcx_api(None) is api


# ---------------------------------------------------------------------------
# flagcx: PyFlagcxCommunicator.__init__
# ---------------------------------------------------------------------------
def test_init_single_world_size(monkeypatch):
    library = mock.Mock()
    monkeypatch.setattr(flagcx_mod, "FLAGCXLibrary", library)
    inst = flagcx_mod.PyFlagcxCommunicator(_StatelessPG(0, 1), "ptpu:0")
    assert inst.available is False and inst.disabled is True
    library.assert_not_called()


def test_init_without_bindings(monkeypatch):
    monkeypatch.setattr(flagcx_mod, "FLAGCXLibrary", None)
    inst = flagcx_mod.PyFlagcxCommunicator(_StatelessPG(1, 2), "ptpu:0")
    assert inst.disabled is True
    assert isinstance(inst.init_error, ImportError)


def test_init_library_load_failure(monkeypatch):
    library = mock.Mock(side_effect=OSError("no libflagcx"))
    monkeypatch.setattr(flagcx_mod, "FLAGCXLibrary", library)
    monkeypatch.setattr(flagcx_mod, "_flagcx_wheel_library", "/tmp/lib.so")
    inst = flagcx_mod.PyFlagcxCommunicator(_StatelessPG(1, 2), "ptpu:0")
    assert inst.disabled is True
    assert isinstance(inst.init_error, OSError)


def test_init_missing_library_path(monkeypatch):
    _symbols(monkeypatch, lib=mock.Mock())
    monkeypatch.setattr(flagcx_mod, "_flagcx_wheel_library", None)
    monkeypatch.setattr(flagcx_mod, "_flagcx_packaged_library", False)
    monkeypatch.delenv("FLAGCX_PATH", raising=False)
    inst = flagcx_mod.PyFlagcxCommunicator(_StatelessPG(1, 2), "ptpu:0")
    assert inst.disabled is True
    assert isinstance(inst.init_error, TypeError)


def test_init_library_path_from_env(monkeypatch):
    library = mock.Mock(return_value=_fake_library())
    _symbols(monkeypatch, lib=library)
    monkeypatch.setattr(flagcx_mod, "_flagcx_wheel_library", None)
    monkeypatch.setattr(flagcx_mod, "_flagcx_packaged_library", False)
    monkeypatch.setenv("FLAGCX_PATH", "/opt/flagcx")
    monkeypatch.setattr(flagcx_mod, "current_stream",
                        mock.Mock(return_value=_STREAM))
    inst = flagcx_mod.PyFlagcxCommunicator(_StatelessPG(1, 2), "ptpu:0")
    assert inst.available is True
    library.assert_called_once_with("/opt/flagcx/build/lib/libflagcx.so")


def test_init_packaged_library_no_path(monkeypatch):
    library = mock.Mock(return_value=_fake_library())
    _symbols(monkeypatch, lib=library)
    monkeypatch.setattr(flagcx_mod, "_flagcx_packaged_library", True)
    monkeypatch.setattr(flagcx_mod, "current_stream",
                        mock.Mock(return_value=_STREAM))
    flagcx_mod.PyFlagcxCommunicator(_StatelessPG(1, 2), "ptpu:0")
    library.assert_called_once_with(None)


def test_init_explicit_library_path(monkeypatch):
    library = mock.Mock(return_value=_fake_library())
    _symbols(monkeypatch, lib=library)
    monkeypatch.setattr(flagcx_mod, "current_stream",
                        mock.Mock(return_value=_STREAM))
    flagcx_mod.PyFlagcxCommunicator(
        _StatelessPG(1, 2), "ptpu:0", library_path="/custom/lib.so"
    )
    library.assert_called_once_with("/custom/lib.so")


def test_init_stateless_modern_rank1(monkeypatch):
    lib = _fake_library()
    _symbols(monkeypatch, lib=mock.Mock(return_value=lib))
    monkeypatch.setattr(flagcx_mod, "_flagcx_wheel_library", "/tmp/lib.so")
    monkeypatch.setattr(flagcx_mod, "current_stream",
                        mock.Mock(return_value=_STREAM))
    group = _StatelessPG(rank=1, world_size=2)
    inst = flagcx_mod.PyFlagcxCommunicator(group, "ptpu:0")
    assert inst.available is True and inst.disabled is False
    assert inst._legacy_unique_id_api is False
    assert isinstance(inst.device, _Dev)
    assert group.broadcast_obj.call_count == 1
    lib.flagcxCommInitRank.assert_called_once_with(
        2, inst.unique_id, 1
    )
    assert inst.comm == "NATIVE"
    assert lib.flagcxAllReduce.call_count == 1
    assert _STREAM.synchronize.call_count >= 1


def test_init_normal_group_legacy_rank0(monkeypatch):
    lib = _fake_library(legacy=True)
    _symbols(monkeypatch, lib=mock.Mock(return_value=lib))
    monkeypatch.setattr(flagcx_mod, "_flagcx_wheel_library", "/tmp/lib.so")
    monkeypatch.setattr(flagcx_mod, "current_stream",
                        mock.Mock(return_value=_STREAM))
    monkeypatch.setattr(
        flagcx_mod, "ctypes", SimpleNamespace(pointer=mock.Mock(return_value="PTR"))
    )
    dist = _fresh_dist()
    monkeypatch.setattr(flagcx_mod, "dist", dist)
    inst = flagcx_mod.PyFlagcxCommunicator(object(), "ptpu:0")
    assert inst._legacy_unique_id_api is True
    dist.broadcast.assert_called_once()
    lib.flagcxCommInitRank.assert_called_once_with(2, "PTR", 0)
    assert inst.unique_id.internal == [0, 0, 0, 0]


def test_init_normal_group_requires_nccl_free_backend(monkeypatch):
    _symbols(monkeypatch, lib=mock.Mock())
    monkeypatch.setattr(flagcx_mod, "dist",
                        _fresh_dist(get_backend=mock.Mock(return_value="nccl")))
    with pytest.raises(AssertionError):
        flagcx_mod.PyFlagcxCommunicator(object(), "ptpu:0")


def test_init_requires_initialized_group(monkeypatch):
    _symbols(monkeypatch, lib=mock.Mock())
    monkeypatch.setattr(
        flagcx_mod, "dist",
        _fresh_dist(is_initialized=mock.Mock(return_value=False)),
    )
    with pytest.raises(AssertionError):
        flagcx_mod.PyFlagcxCommunicator(object(), "ptpu:0")


def test_init_npu_creates_all_reduce_stream(monkeypatch):
    lib = _fake_library()
    _symbols(monkeypatch, lib=mock.Mock(return_value=lib))
    monkeypatch.setattr(flagcx_mod, "_flagcx_wheel_library", "/tmp/lib.so")
    monkeypatch.setattr(flagcx_mod, "current_stream",
                        mock.Mock(return_value=_STREAM))
    submit = mock.Mock()
    stream_cls = mock.Mock(return_value=SimpleNamespace(submit=submit))
    ascend = _make_module(
        "vllm_fl.dispatch.backends.vendor.ascend.patches.flagcx",
        FlagCXAllReduceStream=stream_cls,
    )
    _install_module(
        monkeypatch,
        "vllm_fl.dispatch.backends.vendor.ascend.patches.flagcx",
        ascend,
    )
    inst = flagcx_mod.PyFlagcxCommunicator(_StatelessPG(1, 2), "npu:0")
    assert inst.device.type == "npu"
    stream_cls.assert_called_once()
    assert submit.call_count == 1


def test_init_warmup_failure_txda_destroys(monkeypatch):
    lib = _fake_library()
    _symbols(monkeypatch, lib=mock.Mock(return_value=lib))
    monkeypatch.setattr(flagcx_mod, "_flagcx_wheel_library", "/tmp/lib.so")
    monkeypatch.setattr(flagcx_mod, "current_stream",
                        mock.Mock(return_value=_STREAM))
    destroy = mock.Mock()
    monkeypatch.setattr(flagcx_mod.PyFlagcxCommunicator, "all_reduce",
                        mock.Mock(side_effect=RuntimeError("warmup")))
    monkeypatch.setattr(flagcx_mod.PyFlagcxCommunicator, "destroy", destroy)
    with pytest.raises(RuntimeError):
        flagcx_mod.PyFlagcxCommunicator(_StatelessPG(1, 2), "txda:0")
    destroy.assert_called_once()


def test_init_warmup_failure_other_reraises(monkeypatch):
    lib = _fake_library()
    _symbols(monkeypatch, lib=mock.Mock(return_value=lib))
    monkeypatch.setattr(flagcx_mod, "_flagcx_wheel_library", "/tmp/lib.so")
    monkeypatch.setattr(flagcx_mod, "current_stream",
                        mock.Mock(return_value=_STREAM))
    destroy = mock.Mock()
    monkeypatch.setattr(flagcx_mod.PyFlagcxCommunicator, "all_reduce",
                        mock.Mock(side_effect=RuntimeError("warmup")))
    monkeypatch.setattr(flagcx_mod.PyFlagcxCommunicator, "destroy", destroy)
    with pytest.raises(RuntimeError):
        flagcx_mod.PyFlagcxCommunicator(_StatelessPG(1, 2), "ptpu:0")
    destroy.assert_not_called()


# ---------------------------------------------------------------------------
# flagcx: destroy / streams
# ---------------------------------------------------------------------------
class _FakeThread:
    instances = []

    def __init__(self, target=None, name=None, daemon=None):
        self.target = target
        self.name = name
        self.daemon = daemon
        self.started = False
        self.alive = False
        self.joins = []
        _FakeThread.instances.append(self)

    def start(self):
        self.started = True

    def join(self, timeout=None):
        self.joins.append(timeout)

    def is_alive(self):
        return self.alive


class _StartFailThread(_FakeThread):
    def start(self):
        raise RuntimeError("cannot start")


def _txda_library():
    abort = mock.Mock()
    return SimpleNamespace(
        lib={"flagcxCommAbort": abort}, FLAGCX_CHECK=mock.Mock()
    ), abort


@pytest.fixture
def fake_threads(monkeypatch):
    _FakeThread.instances = []
    monkeypatch.setattr(
        flagcx_mod, "threading",
        SimpleNamespace(Thread=_FakeThread, Lock=threading.Lock),
    )
    return _FakeThread


def test_destroy_noop_for_non_txda(monkeypatch, fake_threads):
    inst = _new_flagcx(device=_Dev("gcu:0"))
    inst.flagcx = mock.Mock()
    inst.destroy()
    assert inst.comm == "COMM"
    assert fake_threads.instances == []


def test_destroy_txda_aborts(monkeypatch, fake_threads):
    library, abort = _txda_library()
    logger = mock.Mock()
    monkeypatch.setattr(flagcx_mod, "logger", logger)
    inst = _new_flagcx(device=_Dev("txda:0"), flagcx=library)
    inst.destroy()
    assert inst.available is False and inst.disabled is True
    assert inst.comm is None
    thread = fake_threads.instances[0]
    assert thread.started is True
    assert thread.joins == [5.0]
    thread.target()
    assert abort.call_count == 1
    logger.warning.assert_not_called()


def test_destroy_txda_abort_error_logged(monkeypatch, fake_threads):
    library, abort = _txda_library()
    library.FLAGCX_CHECK.side_effect = RuntimeError("abort failed")
    logger = mock.Mock()
    monkeypatch.setattr(flagcx_mod, "logger", logger)
    inst = _new_flagcx(device=_Dev("txda:0"), flagcx=library)
    inst.destroy()
    fake_threads.instances[0].target()
    logger.exception.assert_called_once()


def test_destroy_txda_thread_start_failure(monkeypatch, fake_threads):
    monkeypatch.setattr(
        flagcx_mod, "threading",
        SimpleNamespace(Thread=_StartFailThread, Lock=threading.Lock),
    )
    library, _ = _txda_library()
    logger = mock.Mock()
    monkeypatch.setattr(flagcx_mod, "logger", logger)
    inst = _new_flagcx(device=_Dev("txda:0"), flagcx=library)
    inst.destroy()
    logger.exception.assert_called_once()


def test_destroy_txda_alive_warns(monkeypatch, fake_threads):
    library, _ = _txda_library()
    logger = mock.Mock()
    monkeypatch.setattr(flagcx_mod, "logger", logger)
    inst = _new_flagcx(device=_Dev("txda:0"), flagcx=library)
    original_join = _FakeThread.join

    def join_then_alive(self, timeout=None):
        original_join(self, timeout)
        self.alive = True

    monkeypatch.setattr(_FakeThread, "join", join_then_alive)
    inst.destroy()
    logger.warning.assert_called_once()


def test_destroy_txda_idempotent(monkeypatch, fake_threads):
    library, _ = _txda_library()
    monkeypatch.setattr(flagcx_mod, "logger", mock.Mock())
    inst = _new_flagcx(device=_Dev("txda:0"), flagcx=library)
    inst.destroy()
    inst.destroy()
    assert len(fake_threads.instances) == 1


def test_destroy_txda_without_comm(monkeypatch, fake_threads):
    monkeypatch.setattr(flagcx_mod, "logger", mock.Mock())
    inst = _new_flagcx(device=_Dev("txda:0"), comm=None)
    inst.destroy()
    assert fake_threads.instances == []


@pytest.mark.parametrize("dev", ["npu:0", "gcu:0", "txda:0"])
def test_current_stream_vendor(dev):
    inst = _new_flagcx(device=_Dev(dev))
    assert inst._current_stream() is _STREAM


def test_current_stream_amd_and_default(monkeypatch):
    inst = _new_flagcx(device=_Dev("cpu:0"))
    monkeypatch.setattr(flagcx_mod, "_flagcx_amd_source", True)
    assert inst._current_stream() is _STREAM
    monkeypatch.setattr(flagcx_mod, "current_stream",
                        mock.Mock(return_value="DEFAULT"))
    monkeypatch.setattr(flagcx_mod, "_flagcx_amd_source", False)
    assert inst._current_stream() == "DEFAULT"


@pytest.mark.parametrize(
    "dev, expected", [("npu:0", 11), ("gcu:0", 22), ("cpu:0", None)]
)
def test_copy_flagcx_stream(dev, expected):
    inst = _new_flagcx(device=_Dev(dev))
    inst._copy_flagcx_stream(_STREAM)
    stream_arg = inst.flagcx.adaptor_stream_copy.call_args.args[0]
    if expected is None:
        assert stream_arg is _STREAM
    else:
        assert stream_arg.cuda_stream == expected


# ---------------------------------------------------------------------------
# flagcx: collectives
# ---------------------------------------------------------------------------
def test_all_reduce_disabled():
    inst = _new_flagcx(disabled=True)
    assert inst.all_reduce(_tensor()) is None
    inst.flagcx.flagcxAllReduce.assert_not_called()


def test_all_reduce_trace_tensor(monkeypatch):
    monkeypatch.setattr(flagcx_mod, "is_trace_tensor", lambda t: True)
    inst = _new_flagcx()
    out = inst.all_reduce(_tensor())
    assert out.shape == (2,)
    inst.flagcx.flagcxAllReduce.assert_not_called()


def test_all_reduce_device_mismatch():
    inst = _new_flagcx()
    with pytest.raises(AssertionError):
        inst.all_reduce(_tensor(device=_Dev("cpu:1")))


def test_all_reduce_default_path(monkeypatch):
    _symbols(monkeypatch)
    inst = _new_flagcx()
    inst._current_stream = mock.Mock(return_value=_STREAM)
    out = inst.all_reduce(_tensor((3,)))
    assert isinstance(out, _T)
    inst.flagcx.flagcxAllReduce.assert_called_once()
    inst.flagcx.adaptor_stream_free.assert_called_once_with("FS")
    inst.flagcx.flagcxAllReduce.assert_called_once_with(
        ("buf", (4096,)), ("buf", (4096,)), 3, "dtype", "op", "COMM", "FS"
    )


def test_all_reduce_explicit_out_and_stream(monkeypatch):
    _symbols(monkeypatch)
    inst = _new_flagcx()
    out = _tensor((2,))
    result = inst.all_reduce(_tensor(), out_tensor=out, stream="S")
    assert result is out
    assert inst.flagcx.adaptor_stream_copy.call_args.args[0] == "S"


def test_all_reduce_uses_side_stream(monkeypatch):
    _symbols(monkeypatch)
    submit = mock.Mock()
    inst = _new_flagcx(_all_reduce_stream=SimpleNamespace(submit=submit))
    out = inst.all_reduce(_tensor(), stream="S")
    assert isinstance(out, _T)
    submit.assert_called_once()
    assert submit.call_args.args[0] == inst._all_reduce_on_stream


def test_all_reduce_on_stream_frees_on_error(monkeypatch):
    _symbols(monkeypatch)
    inst = _new_flagcx()
    inst._copy_flagcx_stream = mock.Mock(return_value="FS")
    inst.flagcx.flagcxAllReduce.side_effect = RuntimeError("boom")
    with pytest.raises(RuntimeError):
        inst._all_reduce_on_stream(_tensor(), _tensor(), "op", "S")
    inst.flagcx.adaptor_stream_free.assert_called_once_with("FS")


def test_all_gather_disabled_and_trace(monkeypatch):
    inst = _new_flagcx(disabled=True)
    assert inst.all_gather(_tensor(), _tensor()) is None
    monkeypatch.setattr(flagcx_mod, "is_trace_tensor", lambda t: True)
    inst = _new_flagcx()
    assert inst.all_gather(_tensor(), _tensor()) is None
    inst.flagcx.flagcxAllGather.assert_not_called()


def test_all_gather_device_mismatch():
    inst = _new_flagcx()
    with pytest.raises(AssertionError):
        inst.all_gather(_tensor(), _tensor(device=_Dev("npu:0")))


def test_all_gather_path(monkeypatch):
    _symbols(monkeypatch)
    inst = _new_flagcx()
    inst._current_stream = mock.Mock(return_value=_STREAM)
    input_t = _tensor((3, 4))
    output_t = _tensor((6, 4))
    inst.all_gather(output_t, input_t)
    inst.flagcx.flagcxAllGather.assert_called_once()
    assert inst.flagcx.flagcxAllGather.call_args.args[2] == 12
    inst.flagcx.adaptor_stream_free.assert_called_once_with("FS")


def test_all_gatherv_disabled_and_trace(monkeypatch):
    inst = _new_flagcx(disabled=True)
    assert inst.all_gatherv(_tensor((2,)), _tensor((2,)), sizes=[2]) is None
    monkeypatch.setattr(flagcx_mod, "is_trace_tensor", lambda t: True)
    inst = _new_flagcx()
    assert inst.all_gatherv(_tensor((2,)), _tensor((2,)), sizes=[2]) is None
    inst.flagcx.flagcxBroadcast.assert_not_called()


def test_all_gatherv_device_mismatch():
    inst = _new_flagcx()
    with pytest.raises(AssertionError):
        inst.all_gatherv(
            _tensor((2,)), _tensor((2,), device=_Dev("npu:0")), sizes=[2]
        )


def test_all_gatherv_size_sum_assert():
    inst = _new_flagcx()
    with pytest.raises(AssertionError):
        inst.all_gatherv(_tensor((3,)), _tensor((2,)), sizes=[2])


def test_all_gatherv_broadcast_loop(monkeypatch):
    _symbols(monkeypatch)
    inst = _new_flagcx()
    inst._current_stream = mock.Mock(return_value=_STREAM)
    inst.flagcx.flagcxBroadcast.reset_mock()
    inst.all_gatherv(_tensor((4, 4)), _tensor((2, 4)), sizes=[2, 2])
    assert inst.flagcx.flagcxBroadcast.call_count == 2
    assert inst.flagcx.flagcxGroupStart.call_count == 1
    assert inst.flagcx.flagcxGroupEnd.call_count == 1
    inst.flagcx.adaptor_stream_free.assert_called_once_with("FS")


def test_all_gatherv_npu_copies_own_slice(monkeypatch):
    _symbols(monkeypatch)
    inst = _new_flagcx(device=_Dev("npu:0"), rank=0)
    inst._current_stream = mock.Mock(return_value=_STREAM)
    output = _tensor((4, 4), device=_Dev("npu:0"))
    copy = mock.Mock(return_value=output.__getitem__(slice(0, 2)))
    orig_getitem = _T.__getitem__

    def tracking_getitem(self, item):
        result = orig_getitem(self, item)
        result.copy_ = copy
        return result

    monkeypatch.setattr(_T, "__getitem__", tracking_getitem)
    inst.all_gatherv(output, _tensor((2, 4), device=_Dev("npu:0")),
                     sizes=[2, 2])
    assert copy.call_count == 1


def test_reduce_scatter_disabled_and_trace(monkeypatch):
    inst = _new_flagcx(disabled=True)
    assert inst.reduce_scatter(_tensor((2,)), _tensor((4,))) is None
    monkeypatch.setattr(flagcx_mod, "is_trace_tensor", lambda t: True)
    inst = _new_flagcx()
    assert inst.reduce_scatter(_tensor((2,)), _tensor((4,))) is None


def test_reduce_scatter_path(monkeypatch):
    _symbols(monkeypatch)
    inst = _new_flagcx()
    inst._current_stream = mock.Mock(return_value=_STREAM)
    inst.reduce_scatter(_tensor((2, 4)), _tensor((4, 4)))
    inst.flagcx.flagcxReduceScatter.assert_called_once()
    assert inst.flagcx.flagcxReduceScatter.call_args.args[2] == 8


def test_reduce_scatterv_loop(monkeypatch):
    _symbols(monkeypatch)
    inst = _new_flagcx()
    inst._current_stream = mock.Mock(return_value=_STREAM)
    inst.reduce_scatterv(
        _tensor((2, 4)), _tensor((4, 4)), sizes=[2, 2]
    )
    assert inst.flagcx.flagcxReduce.call_count == 2
    assert inst.flagcx.flagcxGroupStart.call_count == 1
    assert inst.flagcx.flagcxGroupEnd.call_count == 1


def test_reduce_scatterv_disabled(monkeypatch):
    inst = _new_flagcx(disabled=True)
    assert inst.reduce_scatterv(_tensor((2,)), _tensor((4,)), sizes=[2]) is None


@pytest.mark.parametrize("method", ["send", "recv", "broadcast"])
def test_point_to_point_disabled(monkeypatch, method):
    inst = _new_flagcx(disabled=True)
    if method == "recv":
        inst.flagcx.flagcxRecv = mock.Mock()
    getattr(inst, method)(*(_tensor(), 1))
    assert True


def test_send_device_mismatch():
    inst = _new_flagcx()
    with pytest.raises(AssertionError):
        inst.send(_tensor(device=_Dev("npu:0")), 1)


def test_send_path(monkeypatch):
    _symbols(monkeypatch)
    inst = _new_flagcx()
    inst._current_stream = mock.Mock(return_value=_STREAM)
    inst.send(_tensor((3,)), 1)
    inst.flagcx.flagcxSend.assert_called_once()
    assert inst.flagcx.flagcxSend.call_args.args[3] == 1


@pytest.mark.parametrize("is_trace", [False, True])
def test_recv_path(monkeypatch, is_trace):
    _symbols(monkeypatch)
    monkeypatch.setattr(flagcx_mod, "is_trace_tensor", lambda t: is_trace)
    inst = _new_flagcx()
    inst._current_stream = mock.Mock(return_value=_STREAM)
    out = inst.recv((2,), "float32", 0)
    assert out.shape == (2,)
    assert inst.flagcx.flagcxRecv.call_count == (0 if is_trace else 1)


@pytest.mark.parametrize("src", [0, 1])
def test_broadcast_path(monkeypatch, src):
    _symbols(monkeypatch)
    inst = _new_flagcx(rank=0)
    inst._current_stream = mock.Mock(return_value=_STREAM)
    inst.broadcast(_tensor((2,)), src)
    sendbuff = inst.flagcx.flagcxBroadcast.call_args.args[0]
    if src == inst.rank:
        assert sendbuff == ("buf", (4096,))
    else:
        assert sendbuff == ("buf", ())


def test_broadcast_trace(monkeypatch):
    _symbols(monkeypatch)
    monkeypatch.setattr(flagcx_mod, "is_trace_tensor", lambda t: True)
    inst = _new_flagcx()
    inst.broadcast(_tensor(), 0)
    inst.flagcx.flagcxBroadcast.assert_not_called()


@pytest.mark.parametrize(
    "dev, amd, comm_arg",
    [("npu:0", False, True), ("gcu:0", True, True), ("cpu:0", False, False)],
)
def test_group_start_end(monkeypatch, dev, amd, comm_arg):
    inst = _new_flagcx(device=_Dev(dev))
    monkeypatch.setattr(flagcx_mod, "_flagcx_amd_source", amd)
    inst.group_start()
    inst.group_end()
    start_args = inst.flagcx.flagcxGroupStart.call_args.args
    end_args = inst.flagcx.flagcxGroupEnd.call_args.args
    assert (start_args == ("COMM",)) is comm_arg
    assert (end_args == ("COMM",)) is comm_arg


# ---------------------------------------------------------------------------
# CommunicatorFL helpers
# ---------------------------------------------------------------------------
def _new_comm(device="cpu:0", world_size=2, rank=0, py=None):
    inst = comm_mod.CommunicatorFL.__new__(comm_mod.CommunicatorFL)
    inst.cpu_group = object()
    inst.device_group = "DEVGROUP"
    inst.device = _Dev(device)
    inst.rank = rank
    inst.rank_in_group = rank
    inst.world_size = world_size
    inst.ranks = list(range(world_size))
    inst.use_all2all = False
    inst.all2all_backend = "naive"
    inst.all2all_manager = None
    inst.pyflagcx_comm = py
    inst._base_all_gather = mock.Mock()
    inst._base_broadcast = mock.Mock()
    return inst


class _RecordingManager:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.destroy = mock.Mock()
        self.checkpoint_prepare = mock.Mock()
        self.checkpoint_restore = mock.Mock()


_ALL2ALL_NAMES = [
    "AgRsAll2AllManager",
    "DeepEPHTAll2AllManager",
    "DeepEPLLAll2AllManager",
    "MoriAll2AllManager",
    "DeepEPV2All2AllManager",
    "NixlEPAll2AllManager",
    "FlashInferNVLinkTwoSidedManager",
    "FlashInferNVLinkOneSidedManager",
]


def _all2all_stub():
    return _make_module(
        "vllm.distributed.device_communicators.all2all",
        **{name: type(name, (_RecordingManager,), {}) for name in _ALL2ALL_NAMES},
    )


def _group(world_size=2, rank=0):
    return SimpleNamespace(
        rank=rank, world_size=world_size, ranks=list(range(world_size))
    )


# ---------------------------------------------------------------------------
# CommunicatorFL.__init__
# ---------------------------------------------------------------------------
def test_comm_init_world_size_one(monkeypatch):
    factory = mock.Mock()
    monkeypatch.setattr(comm_mod, "PyFlagcxCommunicator", factory)
    inst = comm_mod.CommunicatorFL(_group(world_size=1))
    assert inst.pyflagcx_comm is None
    factory.assert_not_called()


def test_comm_init_creates_pyflagcx(monkeypatch):
    created = SimpleNamespace(disabled=False)
    factory = mock.Mock(return_value=created)
    monkeypatch.setattr(comm_mod, "PyFlagcxCommunicator", factory)
    cpu_group = _group()
    inst = comm_mod.CommunicatorFL(cpu_group, device=_Dev("gcu:0"))
    assert inst.pyflagcx_comm is created
    factory.assert_called_once_with(group=cpu_group, device=inst.device)


def test_comm_init_disabled_npu_raises(monkeypatch):
    _fake_platform(monkeypatch, vendor="gcu")
    factory = mock.Mock(
        return_value=SimpleNamespace(disabled=True, init_error=ValueError("x"))
    )
    monkeypatch.setattr(comm_mod, "PyFlagcxCommunicator", factory)
    with pytest.raises(RuntimeError, match="FlagCX requested on NPU"):
        comm_mod.CommunicatorFL(_group(), device=_Dev("npu:0"))


def test_comm_init_disabled_amd_raises(monkeypatch):
    _fake_platform(monkeypatch, vendor="amd")
    factory = mock.Mock(
        return_value=SimpleNamespace(disabled=True, init_error=ValueError("x"))
    )
    monkeypatch.setattr(comm_mod, "PyFlagcxCommunicator", factory)
    with pytest.raises(RuntimeError, match="FlagCX requested on AMD"):
        comm_mod.CommunicatorFL(_group(), device=_Dev("gcu:0"))


def test_comm_init_disabled_other_vendor_ok(monkeypatch):
    _fake_platform(monkeypatch, vendor="gcu")
    created = SimpleNamespace(disabled=True, init_error=ValueError("x"))
    monkeypatch.setattr(
        comm_mod, "PyFlagcxCommunicator", mock.Mock(return_value=created)
    )
    inst = comm_mod.CommunicatorFL(_group(), device=_Dev("gcu:0"))
    assert inst.pyflagcx_comm is created


def test_comm_init_use_all2all(monkeypatch):
    monkeypatch.setattr(
        comm_mod,
        "PyFlagcxCommunicator",
        mock.Mock(return_value=SimpleNamespace(disabled=False)),
    )
    _install_module(
        monkeypatch, "vllm.distributed.device_communicators.all2all",
        _all2all_stub(),
    )
    logger = mock.Mock()
    monkeypatch.setattr(comm_mod, "logger", logger)
    inst = comm_mod.CommunicatorFL(
        _group(), device=_Dev("gcu:0"), tcp_store_group="TCP", use_all2all=True
    )
    assert type(inst.all2all_manager).__name__ == "AgRsAll2AllManager"
    assert inst.all2all_manager.args[1] == "TCP"
    logger.info_once.assert_called_once()


# ---------------------------------------------------------------------------
# CommunicatorFL._init_all2all_manager
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "backend, cls_name, shape",
    [
        ("naive", "AgRsAll2AllManager", "tcp"),
        ("allgather_reducescatter", "AgRsAll2AllManager", "tcp"),
        ("deepep_high_throughput", "DeepEPHTAll2AllManager", "tcp"),
        ("deepep_low_latency", "DeepEPLLAll2AllManager", "tcp"),
        ("mori_high_throughput", "MoriAll2AllManager", "backend"),
        ("mori_low_latency", "MoriAll2AllManager", "backend"),
        ("deepep_v2", "DeepEPV2All2AllManager", "device_group"),
        ("nixl_ep", "NixlEPAll2AllManager", "tcp"),
        ("flashinfer_all2allv", "FlashInferNVLinkTwoSidedManager", "tcp_warn"),
        ("flashinfer_nvlink_two_sided", "FlashInferNVLinkTwoSidedManager",
         "tcp"),
        ("flashinfer_nvlink_one_sided", "FlashInferNVLinkOneSidedManager",
         "single"),
    ],
)
def test_all2all_manager_selection(monkeypatch, backend, cls_name, shape):
    _install_module(
        monkeypatch, "vllm.distributed.device_communicators.all2all",
        _all2all_stub(),
    )
    logger = mock.Mock()
    monkeypatch.setattr(comm_mod, "logger", logger)
    inst = _new_comm()
    inst.all2all_backend = backend
    comm_mod.CommunicatorFL._init_all2all_manager(inst, "TCP")
    manager = inst.all2all_manager
    assert type(manager).__name__ == cls_name
    assert manager.args[0] is inst.cpu_group
    if shape == "tcp":
        assert manager.args == (inst.cpu_group, "TCP")
    elif shape == "tcp_warn":
        assert manager.args == (inst.cpu_group, "TCP")
        logger.warning_once.assert_called_once()
    elif shape == "backend":
        assert manager.args == (inst.cpu_group, backend)
    elif shape == "device_group":
        assert manager.kwargs == {"device_group": inst.device_group}
    else:
        assert manager.args == (inst.cpu_group,)
    logger.info_once.assert_called_once_with(
        "Using %s all2all manager.", cls_name, scope="global"
    )


def test_all2all_manager_unknown_backend(monkeypatch):
    inst = _new_comm()
    inst.all2all_backend = "nope"
    with pytest.raises(ValueError, match="Unknown all2all backend: nope"):
        comm_mod.CommunicatorFL._init_all2all_manager(inst, None)


# ---------------------------------------------------------------------------
# CommunicatorFL collectives
# ---------------------------------------------------------------------------
def test_comm_all_reduce_fallback():
    _TORCH.distributed.all_reduce.reset_mock()
    py = mock.Mock(all_reduce=mock.Mock(return_value=None))
    inst = _new_comm(py=py)
    out = inst.all_reduce(_tensor((2,)))
    assert isinstance(out, _T)
    _TORCH.distributed.all_reduce.assert_called_once_with(
        out, group=inst.device_group
    )


def test_comm_all_reduce_pyflagcx_result():
    backend_out = _tensor((2,))
    py = mock.Mock(all_reduce=mock.Mock(return_value=backend_out))
    inst = _new_comm(py=py)
    assert inst.all_reduce(_tensor((2,))) is backend_out


def test_comm_all_reduce_without_comm():
    inst = _new_comm(py=None)
    with pytest.raises(AssertionError):
        inst.all_reduce(_tensor((2,)))


def test_comm_all_gather_single_world_size():
    inst = _new_comm(world_size=1)
    tensor = _tensor((3,))
    assert inst.all_gather(tensor) is tensor


@pytest.mark.parametrize("dim, expected", [(0, (6, 2)), (-1, (3, 4)), (1, (3, 4))])
def test_comm_all_gather_backend(dim, expected):
    py = mock.Mock(disabled=False)
    inst = _new_comm(py=py)
    result = inst.all_gather(_tensor((3, 2)), dim)
    assert result.shape == expected
    assert py.all_gather.call_count == 1


@pytest.mark.parametrize("py", [None, mock.Mock(disabled=True)])
def test_comm_all_gather_base_fallback(py):
    inst = _new_comm(py=py)
    assert inst.all_gather(_tensor((2,))) == "base-all-gather"
    inst._base_all_gather.assert_called_once()


@pytest.mark.parametrize("dim", [0, -2])
def test_comm_reduce_scatter(dim):
    py = mock.Mock()
    inst = _new_comm(py=py)
    result = inst.reduce_scatter(_tensor((4, 2)), dim)
    assert result.shape == (2, 2)
    py.reduce_scatter.assert_called_once()


def test_comm_reduce_scatter_divisibility_assert():
    inst = _new_comm(py=mock.Mock())
    with pytest.raises(AssertionError):
        inst.reduce_scatter(_tensor((3,)))


def test_comm_reduce_scatterv_with_sizes():
    py = mock.Mock()
    inst = _new_comm(py=py)
    result = inst.reduce_scatterv(_tensor((4, 2)), sizes=[2, 2])
    assert result.shape == (2, 2)
    py.reduce_scatterv.assert_called_once()
    assert py.reduce_scatterv.call_args.kwargs["sizes"] == [2, 2]


def test_comm_reduce_scatterv_without_sizes():
    py = mock.Mock()
    inst = _new_comm(py=py)
    inst.reduce_scatterv(_tensor((4, 2)))
    py.reduce_scatter.assert_called_once()


def test_comm_reduce_scatterv_bad_sizes():
    inst = _new_comm(py=mock.Mock())
    with pytest.raises(AssertionError):
        inst.reduce_scatterv(_tensor((4, 2)), sizes=[1, 1])


def test_comm_send_default_dst():
    py = mock.Mock(disabled=False)
    inst = _new_comm(py=py, rank=0)
    tensor = _tensor((2,))
    inst.send(tensor)
    py.send.assert_called_once_with(tensor, 1)


def test_comm_send_explicit_and_fallback():
    py = mock.Mock(disabled=False)
    inst = _new_comm(py=py)
    tensor = _tensor((2,))
    inst.send(tensor, dst=0)
    py.send.assert_called_once_with(tensor, 0)

    _TORCH.distributed.send.reset_mock()
    fallback = _new_comm(py=mock.Mock(disabled=True))
    fallback.send(tensor, dst=1)
    _TORCH.distributed.send.assert_called_once_with(
        tensor, fallback.ranks[1], fallback.device_group
    )


def test_comm_recv_default_src():
    py = mock.Mock(disabled=False)
    inst = _new_comm(py=py, rank=1)
    tensor = inst.recv((2,), "float32")
    assert tensor.shape == (2,)
    py.recv.assert_called_once_with(tensor, 0)


def test_comm_recv_fallback():
    _TORCH.distributed.recv.reset_mock()
    inst = _new_comm(py=mock.Mock(disabled=True), rank=0)
    inst.recv((2,), "float32", src=1)
    _TORCH.distributed.recv.assert_called_once()
    assert _TORCH.distributed.recv.call_args.args[1] == inst.ranks[1]


def test_comm_broadcast_single_rank():
    inst = _new_comm(world_size=1)
    tensor = _tensor((2,))
    assert inst.broadcast(tensor) is tensor


def test_comm_broadcast_backend():
    py = mock.Mock(disabled=False)
    inst = _new_comm(py=py)
    tensor = _tensor((2,))
    assert inst.broadcast(tensor, 1) is tensor
    py.broadcast.assert_called_once_with(tensor, 1)


def test_comm_broadcast_base_fallback():
    inst = _new_comm(py=None)
    tensor = _tensor((2,))
    assert inst.broadcast(tensor, 1) is tensor
    inst._base_broadcast.assert_called_once_with(tensor, 1)


# ---------------------------------------------------------------------------
# CommunicatorFL teardown / all2all dispatch
# ---------------------------------------------------------------------------
def test_comm_destroy_txda():
    py = mock.Mock()
    inst = _new_comm(device="txda:0", py=py)
    manager = mock.Mock()
    inst.all2all_manager = manager
    inst.destroy()
    py.destroy.assert_called_once()
    manager.destroy.assert_called_once()
    assert inst.pyflagcx_comm is None
    assert inst.all2all_manager is None


def test_comm_destroy_other_device():
    py = mock.Mock()
    inst = _new_comm(device="gcu:0", py=py)
    inst.destroy()
    py.destroy.assert_not_called()
    assert inst.pyflagcx_comm is None


def test_comm_destroy_all_none():
    inst = _new_comm(py=None)
    inst.destroy()
    assert inst.pyflagcx_comm is None


@pytest.mark.parametrize("method", ["checkpoint_prepare", "checkpoint_restore"])
def test_comm_checkpoint_with_manager(method):
    inst = _new_comm()
    manager = mock.Mock()
    inst.all2all_manager = manager
    getattr(inst, method)()
    getattr(manager, method).assert_called_once()


@pytest.mark.parametrize("method", ["checkpoint_prepare", "checkpoint_restore"])
def test_comm_checkpoint_without_manager(method):
    inst = _new_comm()
    assert getattr(inst, method)() is None


def test_comm_all_gatherv_bad_dim():
    inst = _new_comm(py=mock.Mock(disabled=False))
    with pytest.raises(NotImplementedError):
        inst.all_gatherv(_tensor((2,)), dim=1)


def test_comm_all_gatherv_requires_comm():
    inst = _new_comm(py=None)
    with pytest.raises(AssertionError):
        inst.all_gatherv(_tensor((2,)))


def test_comm_all_gatherv_equal_sizes_collapse():
    py = mock.Mock(disabled=False)
    inst = _new_comm(py=py, rank=0)
    result = inst.all_gatherv(_tensor((2, 4)), sizes=[2, 2])
    assert result.shape == (4, 4)
    py.all_gather.assert_called_once()
    py.all_gatherv.assert_not_called()


def test_comm_all_gatherv_with_sizes():
    py = mock.Mock(disabled=False)
    inst = _new_comm(py=py, rank=0)
    result = inst.all_gatherv(_tensor((1, 4)), sizes=[1, 3])
    assert result.shape == (4, 4)
    py.all_gatherv.assert_called_once()
    assert py.all_gatherv.call_args.kwargs["sizes"] == [1, 3]


def test_comm_all_gatherv_list_input():
    py = mock.Mock(disabled=False)
    inst = _new_comm(py=py, rank=0)
    outputs = inst.all_gatherv([_tensor((2, 4)), _tensor((2, 4))])
    assert len(outputs) == 2
    py.group_start.assert_called_once()
    py.group_end.assert_called_once()
    assert py.all_gather.call_count == 2


def test_comm_dispatch_router_logits():
    manager = mock.Mock()
    inst = _new_comm()
    inst.all2all_manager = manager
    hidden, logits, extra = _tensor(), _tensor(), [_tensor()]
    inst.dispatch_router_logits(hidden, logits, True, extra)
    manager.dispatch_router_logits.assert_called_once_with(
        hidden, logits, True, extra
    )


def test_comm_dispatch_and_combine():
    manager = mock.Mock()
    inst = _new_comm()
    inst.all2all_manager = manager
    hidden, weights, ids = _tensor(), _tensor(), _tensor()
    inst.dispatch(hidden, weights, ids, False, None)
    manager.dispatch.assert_called_once_with(hidden, weights, ids, False, None)
    inst.combine(hidden, True)
    manager.combine.assert_called_once_with(hidden, True)


@pytest.mark.parametrize(
    "call",
    [
        lambda inst, t: inst.dispatch_router_logits(t, t),
        lambda inst, t: inst.dispatch(t, t, t),
        lambda inst, t: inst.combine(t),
    ],
)
def test_comm_all2all_requires_manager(call):
    inst = _new_comm()
    with pytest.raises(AssertionError):
        call(inst, _tensor())


# ---------------------------------------------------------------------------
# package __init__ lazy attribute
# ---------------------------------------------------------------------------
def test_package_lazy_attr(monkeypatch):
    monkeypatch.setitem(sys.modules, "vllm_fl.distributed", init_mod)
    monkeypatch.setitem(
        sys.modules, "vllm_fl.distributed.communicator", comm_mod
    )
    assert init_mod.__all__ == ["communicator"]
    init_mod.__dict__.pop("communicator", None)
    try:
        assert getattr(init_mod, "communicator") is comm_mod
        assert init_mod.communicator is comm_mod
    finally:
        init_mod.__dict__.pop("communicator", None)


def test_package_unknown_attr(monkeypatch):
    monkeypatch.setitem(sys.modules, "vllm_fl.distributed", init_mod)
    with pytest.raises(AttributeError, match="has no attribute 'missing'"):
        getattr(init_mod, "missing")

