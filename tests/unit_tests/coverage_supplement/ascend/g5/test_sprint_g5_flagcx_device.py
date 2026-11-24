# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""814b (Ascend vLLM 0.32) coverage for ``device_communicators/flagcx.py``.

This file is self-contained on purpose: it does not import the shared
``coverage_tests`` helpers, so it can be dropped into any test directory of
the 814b tree.  The 814b module differs from ``main``:

* it has no ``_flagcx_module`` / ``_FLAGCX_NAMES`` / ``_resolve_flagcx_library_path``
  (the import ladder uses ``FLAGCX_PATH`` + ``plugin.interservice.flagcx_wrapper``
  and a wheel ``api.py`` side load);
* ``PyFlagcxCommunicator`` may never receive a ``.flagcx`` attribute when the
  bindings are missing (construction disables itself before binding), so every
  access below goes through objects we control.

The FlagCX C facade is never imported for real: the module globals
(``FLAGCXLibrary``, ``buffer_type``, ``flagcxDataTypeEnum``, ``flagcxUniqueId``,
``flagcxRedOpTypeEnum``) are monkeypatched to recorders.  NPU-only branches are
guarded by an availability probe and skipped on hosts without ``torch.npu``.
"""

from __future__ import annotations

import ctypes
import importlib
import sys
import types
from contextlib import nullcontext

import pytest
import torch

MODULE_NAME = "vllm_fl.distributed.device_communicators.flagcx"


def _load():
    try:
        return importlib.import_module(MODULE_NAME)
    except Exception as exc:  # pragma: no cover - environment guard
        pytest.skip(f"flagcx device module not importable: {exc!r}")


try:
    _MODULE = importlib.import_module(MODULE_NAME)
except Exception:  # pragma: no cover - environment guard
    _MODULE = None


def _mod():
    if _MODULE is None:  # pragma: no cover - environment guard
        pytest.skip("flagcx device module not importable")
    return _MODULE


def _has_npu() -> bool:
    try:
        probe = torch.zeros(1, device="npu")
        del probe
        return True
    except Exception:
        return False


def _has_txda() -> bool:
    return getattr(getattr(torch, "txda", None), "device", None) is not None


# ---------------------------------------------------------------------------
# Recording fakes for the ctypes facade
# ---------------------------------------------------------------------------


class _Buffer:
    def __init__(self, ptr=None):
        self.ptr = ptr


class _DataTypeEnum:
    @staticmethod
    def from_torch(dtype):
        return f"dtype:{dtype}"


class _RedOpEnum:
    @staticmethod
    def from_torch(op):
        return f"op:{op}"


class _UniqueId(ctypes.Structure):
    _fields_ = [("internal", ctypes.c_ubyte * 8)]


class _Stream:
    def __init__(self):
        self.synchronized = False

    def synchronize(self):
        self.synchronized = True


class _FakeCuda:
    def device(self, device):
        return nullcontext()


class _FakeLibrary:
    """Recording stand-in for ``FLAGCXLibrary`` covering both id APIs."""

    def __init__(self, recorder, *, legacy=False, unique_id=None, handler_value=None):
        self.recorder = recorder
        self.calls = recorder
        self.comm = "COMM"
        self.unique_id = unique_id or _UniqueId()
        self.legacy_contents = None
        if legacy:
            self.handler = handler_value if handler_value is not None else object()
            self.legacy_contents = types.SimpleNamespace(contents=_UniqueId())
        else:
            self.devHandle = object()

    def _rec(self, name, *args):
        self.recorder.append((name,) + args)

    def flagcxGetUniqueId(self):
        self._rec("flagcxGetUniqueId")
        if self.legacy_contents is not None:
            return self.legacy_contents
        return self.unique_id

    def flagcxCommInitRank(self, world, unique_id, rank):
        self._rec("flagcxCommInitRank", world, unique_id, rank)
        return self.comm

    def adaptor_stream_copy(self, stream):
        self._rec("adaptor_stream_copy", stream)
        return ("flagcx-stream", id(stream))

    def adaptor_stream_free(self, stream):
        self._rec("adaptor_stream_free", stream)

    def flagcxAllReduce(self, *args):
        self._rec("flagcxAllReduce", *args)

    def flagcxAllGather(self, *args):
        self._rec("flagcxAllGather", *args)

    def flagcxBroadcast(self, *args):
        self._rec("flagcxBroadcast", *args)

    def flagcxReduce(self, *args):
        self._rec("flagcxReduce", *args)

    def flagcxReduceScatter(self, *args):
        self._rec("flagcxReduceScatter", *args)

    def flagcxSend(self, *args):
        self._rec("flagcxSend", *args)

    def flagcxRecv(self, *args):
        self._rec("flagcxRecv", *args)

    def flagcxGroupStart(self, *args):
        self._rec("flagcxGroupStart", *args)

    def flagcxGroupEnd(self, *args):
        self._rec("flagcxGroupEnd", *args)


def _patch_symbols(monkeypatch, module, stream_factory=_Stream):
    monkeypatch.setattr(module, "buffer_type", _Buffer, raising=False)
    monkeypatch.setattr(module, "flagcxDataTypeEnum", _DataTypeEnum, raising=False)
    monkeypatch.setattr(module, "flagcxRedOpTypeEnum", _RedOpEnum, raising=False)
    monkeypatch.setattr(module, "flagcxUniqueId", _UniqueId, raising=False)
    monkeypatch.setattr(module, "flagcxComm_t", object, raising=False)
    monkeypatch.setattr(module, "current_stream", lambda: stream_factory(), raising=False)
    monkeypatch.setattr(module, "_flagcx_amd_source", False, raising=False)


def _make_comm(module, library, *, device="cpu", rank=0, world_size=2, disabled=False):
    comm = object.__new__(module.PyFlagcxCommunicator)
    comm.rank = rank
    comm.world_size = world_size
    comm.device = torch.device(device) if isinstance(device, str) else device
    comm.comm = "COMM"
    comm.available = True
    comm.disabled = disabled
    comm.flagcx = library
    return comm


def _calls(recorder, name):
    return [call for call in recorder if call[0] == name]


# ---------------------------------------------------------------------------
# Import-ladder helpers (target 32, 54, 60-72, 80, 82, 87-94, 123)
# ---------------------------------------------------------------------------


_API_BODY = (
    "class FLAGCXLibrary:\n    pass\n"
    "class buffer_type:\n    pass\n"
    "class flagcxComm_t:\n    pass\n"
    "class flagcxDataTypeEnum:\n    pass\n"
    "class flagcxUniqueId:\n    pass\n"
    "class flagcxRedOpTypeEnum:\n    pass\n"
)


def test_load_flagcx_wheel_bindings_returns_bound_names(tmp_path):
    """A complete wheel ``api.py`` side load yields the six bound names."""
    module = _mod()
    api_path = tmp_path / "api.py"
    api_path.write_text(_API_BODY, encoding="utf-8")
    internal_name = f"{module.__name__}._wheel_api"

    try:
        loaded = module._load_flagcx_wheel_bindings(str(api_path))
    finally:
        sys.modules.pop(internal_name, None)

    assert [item.__name__ for item in loaded] == [
        "FLAGCXLibrary",
        "buffer_type",
        "flagcxComm_t",
        "flagcxDataTypeEnum",
        "flagcxUniqueId",
        "flagcxRedOpTypeEnum",
    ]


def test_load_flagcx_wheel_bindings_spec_without_loader(tmp_path):
    """A file the import machinery cannot map to a loader is rejected early."""
    module = _mod()
    unknown = tmp_path / "api.ini"
    unknown.write_text("", encoding="utf-8")
    with pytest.raises(ImportError, match="Cannot load FlagCX API"):
        module._load_flagcx_wheel_bindings(str(unknown))


def test_load_flagcx_wheel_bindings_restores_previous_module(tmp_path, monkeypatch):
    """A failing side load restores the previously cached internal module."""
    module = _mod()
    internal_name = f"{module.__name__}._wheel_api"
    sentinel = types.ModuleType(internal_name)
    monkeypatch.setitem(sys.modules, internal_name, sentinel)

    api_path = tmp_path / "api.py"
    api_path.write_text("raise RuntimeError('wheel-exec-failed')\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="wheel-exec-failed"):
        module._load_flagcx_wheel_bindings(str(api_path))

    assert sys.modules[internal_name] is sentinel


def test_load_flagcx_wheel_bindings_incomplete_api(tmp_path):
    """An api.py missing a required name is reported as incomplete."""
    module = _mod()
    api_path = tmp_path / "api.py"
    api_path.write_text("class FLAGCXLibrary:\n    pass\n", encoding="utf-8")
    internal_name = f"{module.__name__}._wheel_api"
    try:
        with pytest.raises(ImportError, match="Incomplete FlagCX API"):
            module._load_flagcx_wheel_bindings(str(api_path))
    finally:
        sys.modules.pop(internal_name, None)


def test_load_amd_flagcx_api_success_and_failure(tmp_path):
    """The explicit AMD source loader execs the file or cleans up and raises."""
    module = _mod()
    good = tmp_path / "good_api.py"
    good.write_text("MARK = 7\n", encoding="utf-8")
    name = f"{module.__name__}._amd_api"
    try:
        loaded = module._load_amd_flagcx_api(good)
        assert loaded.MARK == 7
    finally:
        sys.modules.pop(name, None)

    bad = tmp_path / "bad_api.py"
    bad.write_text("raise ValueError('amd-load-failed')\n", encoding="utf-8")
    with pytest.raises(ValueError, match="amd-load-failed"):
        module._load_amd_flagcx_api(bad)
    assert name not in sys.modules


def test_amd_source_reload_binds_amd_api(tmp_path, monkeypatch):
    """FLAGCX_PATH + ROCm hip + AMD vendor selects the source ``api.py`` path."""
    try:
        import vllm.platforms as platforms_mod
    except Exception as exc:  # pragma: no cover - environment guard
        pytest.skip(f"vllm.platforms unavailable: {exc!r}")

    module = _mod()
    api_dir = tmp_path / "src" / "flagcx"
    api_dir.mkdir(parents=True)
    (api_dir / "api.py").write_text(_API_BODY, encoding="utf-8")

    monkeypatch.setenv("FLAGCX_PATH", str(tmp_path))
    monkeypatch.setattr(module.torch.version, "hip", "6.0", raising=False)
    monkeypatch.setattr(
        platforms_mod,
        "current_platform",
        types.SimpleNamespace(vendor_name="amd"),
        raising=False,
    )
    try:
        reloaded = importlib.reload(module)
        assert reloaded._flagcx_amd_source is True
        assert reloaded._flagcx_available is True
        assert reloaded.FLAGCXLibrary is not None
    finally:
        monkeypatch.delenv("FLAGCX_PATH", raising=False)
        monkeypatch.setattr(module.torch.version, "hip", None, raising=False)
        importlib.reload(module)


def test_non_import_error_binding_failure_is_reraised(tmp_path, monkeypatch):
    """A wheel api.py that raises OSError on a non-AMD host is re-raised."""
    module = _mod()
    (tmp_path / "api.py").write_text("raise OSError('wheel-boom')\n", encoding="utf-8")
    (tmp_path / "lib").mkdir()
    (tmp_path / "lib" / "libflagcx.so").write_bytes(b"")

    monkeypatch.setenv("FLAGCX_PATH", str(tmp_path))
    monkeypatch.setattr(module.torch.version, "hip", None, raising=False)
    try:
        with pytest.raises(OSError, match="wheel-boom"):
            importlib.reload(module)
    finally:
        monkeypatch.delenv("FLAGCX_PATH", raising=False)
        importlib.reload(module)


# ---------------------------------------------------------------------------
# is_trace_tensor (target 155)
# ---------------------------------------------------------------------------


def test_is_trace_tensor_flags_meta_only():
    module = _mod()
    assert module.is_trace_tensor(None) is False
    assert module.is_trace_tensor(torch.zeros(2)) is False
    assert module.is_trace_tensor(torch.zeros(2, device="meta")) is True


# ---------------------------------------------------------------------------
# Constructor branches (target 204, 232, 237, 240-246, 274, 281, 285, 293)
# ---------------------------------------------------------------------------


class _FakeDist:
    NCCL = "nccl"
    Backend = types.SimpleNamespace(NCCL="nccl")

    def __init__(self, rank=0, world_size=2, backend="gloo"):
        self.rank = rank
        self.world_size = world_size
        self.backend = backend
        self.broadcasts = []

    def is_initialized(self):
        return True

    def get_backend(self, group):
        return self.backend

    def get_rank(self, group):
        return self.rank

    def get_world_size(self, group):
        return self.world_size

    def get_process_group_ranks(self, group):
        return [10, 11, 12, 13]

    def broadcast(self, tensor, src=None, group=None):
        self.broadcasts.append((tensor, src, group))
        if self.rank == 0:
            tensor.fill_(0)
        else:
            tensor.copy_(torch.arange(1, tensor.numel() + 1, dtype=tensor.dtype))


def test_constructor_disables_when_bindings_missing(monkeypatch):
    """``FLAGCXLibrary is None`` disables the communicator and records why."""
    module = _mod()
    _patch_symbols(monkeypatch, module)
    monkeypatch.setattr(module, "FLAGCXLibrary", None, raising=False)
    monkeypatch.setattr(module, "_flagcx_import_error", ImportError("absent"), raising=False)
    monkeypatch.setattr(module, "dist", _FakeDist(), raising=False)

    comm = module.PyFlagcxCommunicator(object(), "cpu")
    assert comm.available is False
    assert comm.disabled is True
    assert isinstance(comm.init_error, ImportError)
    assert not hasattr(comm, "flagcx")


@pytest.mark.parametrize("rank", [0, 1])




def _install_fake_leaf(monkeypatch, dotted, leaf):
    parts = dotted.split(".")
    for i in range(1, len(parts)):
        prefix = ".".join(parts[:i])
        if prefix not in sys.modules:
            pkg = types.ModuleType(prefix)
            pkg.__path__ = []
            monkeypatch.setitem(sys.modules, prefix, pkg)
    monkeypatch.setitem(sys.modules, dotted, leaf)


def test_constructor_npu_builds_all_reduce_stream(monkeypatch):
    """NPU + wheel library installs the vendor FlagCXAllReduceStream."""
    if not _has_npu():  # pragma: no cover - hardware guard
        pytest.skip("torch.npu is unavailable on this host")
    module = _mod()
    recorder = []
    library = _FakeLibrary(recorder)
    _patch_symbols(monkeypatch, module)
    monkeypatch.setattr(module, "FLAGCXLibrary", lambda path: library, raising=False)
    monkeypatch.setattr(module, "_flagcx_wheel_library", "/fake/libflagcx.so", raising=False)
    monkeypatch.setattr(module, "dist", _FakeDist(rank=0), raising=False)

    created = []

    class _FakeReduceStream:
        def __init__(self, device):
            self.device = device
            self.submitted = []
            created.append(self)

        def submit(self, fn, *args):
            self.submitted.append((fn, args))
            return fn(*args)

    leaf_name = "vllm_fl.dispatch.backends.vendor.ascend.patches.flagcx"
    try:
        leaf = importlib.import_module(leaf_name)
        monkeypatch.setattr(leaf, "FlagCXAllReduceStream", _FakeReduceStream, raising=False)
    except Exception:  # pragma: no cover - fallback when the vendor tree is absent
        _install_fake_leaf(monkeypatch, leaf_name, types.ModuleType(leaf_name))
        sys.modules[leaf_name].FlagCXAllReduceStream = _FakeReduceStream

    monkeypatch.setattr(
        module.PyFlagcxCommunicator,
        "_copy_flagcx_stream",
        lambda self, stream: ("flagcx-stream", id(stream)),
    )

    comm = module.PyFlagcxCommunicator(object(), "npu")

    assert comm.available is True
    assert isinstance(comm._all_reduce_stream, _FakeReduceStream)
    assert created and created[0].submitted
    assert _calls(recorder, "flagcxAllReduce")


def test_copy_flagcx_stream_wraps_npu_stream(monkeypatch):
    """The NPU stream is wrapped so the facade can read ``cuda_stream``."""
    if not _has_npu():  # pragma: no cover - hardware guard
        pytest.skip("torch.npu is unavailable on this host")
    module = _mod()
    recorder = []
    library = _FakeLibrary(recorder)
    _patch_symbols(monkeypatch, module)
    comm = _make_comm(module, library, device="npu")

    copied = module.PyFlagcxCommunicator._copy_flagcx_stream(
        comm, types.SimpleNamespace(npu_stream=1234)
    )

    assert copied[0] == "flagcx-stream"
    copies = _calls(recorder, "adaptor_stream_copy")
    assert len(copies) == 1
    assert copies[0][1].cuda_stream == 1234


# ---------------------------------------------------------------------------
# Collectives (target 380, 382, 383, 394, 404, 407, 410, 418-420, 430,
# 438, 440, 445, 448-451, 456, 470, 473-486, 495-497, 500-504)
# ---------------------------------------------------------------------------


def test_all_reduce_allocates_output_and_uses_current_stream(monkeypatch):
    """Default output + default stream go through ``_all_reduce_on_stream``."""
    module = _mod()
    recorder = []
    library = _FakeLibrary(recorder)
    _patch_symbols(monkeypatch, module)
    comm = _make_comm(module, library)
    tensor = torch.arange(4, dtype=torch.float32)

    out = module.PyFlagcxCommunicator.all_reduce(comm, tensor)

    assert out.shape == tensor.shape and out is not tensor
    reduce_call = _calls(recorder, "flagcxAllReduce")[0]
    assert reduce_call[1].ptr == tensor.data_ptr()
    assert reduce_call[2].ptr == out.data_ptr()
    assert reduce_call[3] == 4
    assert reduce_call[-1][0] == "flagcx-stream"
    assert recorder[-1][0] == "adaptor_stream_free"


def test_all_reduce_trace_tensor_returns_empty_like(monkeypatch):
    """A meta (trace) tensor short-circuits before touching the C facade."""
    module = _mod()
    recorder = []
    library = _FakeLibrary(recorder)
    _patch_symbols(monkeypatch, module)
    comm = _make_comm(module, library)

    out = module.PyFlagcxCommunicator.all_reduce(comm, torch.zeros(3, device="meta"))

    assert out.device.type == "meta"
    assert recorder == []


def test_all_reduce_stream_submission_path(monkeypatch):
    """A configured ``_all_reduce_stream`` receives the submission closure."""
    module = _mod()
    recorder = []
    library = _FakeLibrary(recorder)
    _patch_symbols(monkeypatch, module)
    comm = _make_comm(module, library)
    comm._all_reduce_stream = types.SimpleNamespace(
        submit=lambda fn, *a: fn(*a)
    )

    out = module.PyFlagcxCommunicator.all_reduce(comm, torch.ones(2))

    assert out.shape == (2,)
    assert _calls(recorder, "flagcxAllReduce")


def test_all_gather_records_and_rejects_foreign_device(monkeypatch):
    """all_gather forwards buffers; a tensor on another device asserts."""
    module = _mod()
    recorder = []
    library = _FakeLibrary(recorder)
    _patch_symbols(monkeypatch, module)
    comm = _make_comm(module, library)
    inp = torch.ones(3)
    out = torch.zeros(6)

    module.PyFlagcxCommunicator.all_gather(comm, out, inp)
    call = _calls(recorder, "flagcxAllGather")[0]
    assert call[1].ptr == inp.data_ptr()
    assert call[2].ptr == out.data_ptr()
    assert call[3] == 3

    if _has_npu():  # pragma: no cover - hardware guard
        foreign = torch.zeros(2, device="npu")
        with pytest.raises(AssertionError, match="created to work on"):
            module.PyFlagcxCommunicator.all_gather(comm, torch.zeros(4, device="npu"), foreign)


def test_all_gather_trace_tensor_is_a_noop(monkeypatch):
    """A meta input/output returns before dispatch."""
    module = _mod()
    recorder = []
    library = _FakeLibrary(recorder)
    _patch_symbols(monkeypatch, module)
    comm = _make_comm(module, library)

    assert (
        module.PyFlagcxCommunicator.all_gather(
            comm, torch.zeros(2, device="meta"), torch.zeros(1, device="meta")
        )
        is None
    )
    assert recorder == []


def test_all_gatherv_cpu_groups_and_broadcasts(monkeypatch):
    """CPU all_gatherv brackets a group and broadcasts each split."""
    module = _mod()
    recorder = []
    library = _FakeLibrary(recorder)
    _patch_symbols(monkeypatch, module)
    comm = _make_comm(module, library, rank=0)
    inp = torch.arange(3.0)
    out = torch.zeros(5)

    module.PyFlagcxCommunicator.all_gatherv(comm, out, inp, [2, 3])

    names = [call[0] for call in recorder]
    assert names.count("flagcxBroadcast") == 2
    assert names.index("flagcxGroupStart") < names.index("flagcxBroadcast")
    assert names.index("flagcxGroupEnd") > names.index("flagcxBroadcast")
    broadcasts = _calls(recorder, "flagcxBroadcast")
    assert broadcasts[0][1].ptr == inp.data_ptr()
    assert broadcasts[0][2].ptr == out[0:2].data_ptr()
    assert broadcasts[0][3] == 2
    assert broadcasts[1][2].ptr == out[2:5].data_ptr()
    assert broadcasts[1][3] == 3
    with pytest.raises(AssertionError):
        module.PyFlagcxCommunicator.all_gatherv(comm, out, inp, [4, 3])


def test_all_gatherv_npu_copies_own_slice(monkeypatch):
    """The NPU path fills the caller's own slice after the broadcast loop."""
    if not _has_npu():  # pragma: no cover - hardware guard
        pytest.skip("torch.npu is unavailable on this host")
    module = _mod()
    recorder = []
    library = _FakeLibrary(recorder)
    _patch_symbols(monkeypatch, module)
    comm = _make_comm(module, library, device="npu", rank=1)
    monkeypatch.setattr(
        module.PyFlagcxCommunicator,
        "_copy_flagcx_stream",
        lambda self, stream: ("flagcx-stream", id(stream)),
    )
    inp = torch.arange(2, dtype=torch.float32, device="npu")
    out = torch.zeros(3, dtype=torch.float32, device="npu")

    module.PyFlagcxCommunicator.all_gatherv(comm, out, inp, [1, 2])

    assert _calls(recorder, "flagcxBroadcast")
    assert _calls(recorder, "flagcxGroupStart")
    assert _calls(recorder, "flagcxGroupEnd")
    torch.npu.synchronize()
    assert torch.equal(out[1:3].cpu(), inp.cpu())


def test_reduce_scatter_stream_wrapper_records(monkeypatch):
    """reduce_scatter uses ``_copy_flagcx_stream`` and frees it (NPU wrap)."""
    module = _mod()
    recorder = []
    library = _FakeLibrary(recorder)
    _patch_symbols(monkeypatch, module)
    comm = _make_comm(module, library, device="cpu", rank=0)
    out = torch.zeros(2)
    inp = torch.ones(4)

    module.PyFlagcxCommunicator.reduce_scatter(comm, out, inp)

    call = _calls(recorder, "flagcxReduceScatter")[0]
    assert call[1].ptr == inp.data_ptr()
    assert call[2].ptr == out.data_ptr()
    assert call[3] == 2
    assert recorder[-1][0] == "adaptor_stream_free"
