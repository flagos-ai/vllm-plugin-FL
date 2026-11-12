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
"""CPU-collection helpers for the FlagCX kv_transfer coverage tests.

The NVIDIA CI image (v0.24.0-cuda-ci) ships no flagcx package, so the
sanctioned boundary for testing the FlagCX connector is a fake flagcx
wheel facade installed into ``sys.modules`` (the same external-resource
mock style as tests/unit_tests/distributed/test_flagcx.py). This module
provides that facade plus the lightweight config scaffolding the
connector reads, so scheduler and worker logic runs on CPU.

On hosts where the real packages exist every placeholder below is a
no-op: the flag_gems shim mirrors the dispatch area's _flaggems_shim,
and the flagcx facade is only installed when a test explicitly asks for
it via ``install_fake_flagcx_wheel``.
"""

from __future__ import annotations

import sys
import types


def _importable(name: str) -> bool:
    """Return True when ``name`` is already imported or imports cleanly."""
    if name in sys.modules:
        return True
    try:
        __import__(name)
        return True
    except Exception:
        return False


def _install_placeholder_flag_gems() -> None:
    """Install an inert flag_gems package when the real one is absent.

    ``vllm_fl.utils`` reads exactly three attributes at import time; on
    the CI image the real FlagGems provides them and this is a no-op.
    """
    if _importable("flag_gems"):
        return

    flag_gems = types.ModuleType("flag_gems")
    flag_gems._FULL_CONFIG = ()

    runtime = types.ModuleType("flag_gems.runtime")
    backend = types.ModuleType("flag_gems.runtime.backend")

    def _set_torch_backend_device_fn(vendor_name: str) -> None:
        return None

    def _gen_torch_device_object():
        import torch

        return torch.cuda

    def _get_torch_backend_device_fn():
        return None

    backend.set_torch_backend_device_fn = _set_torch_backend_device_fn
    backend.gen_torch_device_object = _gen_torch_device_object
    backend.get_torch_backend_device_fn = _get_torch_backend_device_fn

    device = types.ModuleType("flag_gems.runtime.backend.device")

    class DeviceDetector:
        """Minimal detector matching the flag_gems <= 5.0.2 shape."""

        def __init__(self, *args, **kwargs):
            self.vendor_name = "nvidia"
            self.device_type = "cuda"
            self.name = "cuda"  # vllm_fl DeviceInfo.device_type reads .name
            self.dispatch_key = "CUDA"  # vllm_fl DeviceInfo.dispatch_key

    device.DeviceDetector = DeviceDetector

    flag_gems.runtime = runtime
    runtime.backend = backend
    backend.device = device

    sys.modules["flag_gems"] = flag_gems
    sys.modules["flag_gems.runtime"] = runtime
    sys.modules["flag_gems.runtime.backend"] = backend
    sys.modules["flag_gems.runtime.backend.device"] = device


def _install_placeholder_uvloop() -> None:
    """Install an inert uvloop when absent (vllm.v1.utils imports it)."""
    if _importable("uvloop"):
        return
    sys.modules["uvloop"] = types.ModuleType("uvloop")


def _install_placeholder_xgrammar() -> None:
    """Install an inert xgrammar when absent (vllm parser imports it)."""
    if _importable("xgrammar"):
        return
    xgrammar = types.ModuleType("xgrammar")
    xgrammar.Grammar = object
    xgrammar.StructuralTag = object
    xgrammar.normalize_tool_choice = lambda *a, **k: None
    xgrammar.get_model_structural_tag = lambda *a, **k: None
    xgrammar.__path__ = []
    sys.modules["xgrammar"] = xgrammar

    structural_tag = types.ModuleType("xgrammar.structural_tag")
    for attr in (
        "AnyTextFormat",
        "ConstStringFormat",
        "Format",
        "GrammarFormat",
        "JSONSchemaFormat",
        "OptionalFormat",
        "OrFormat",
        "PlusFormat",
        "RegexFormat",
        "SequenceFormat",
        "StarFormat",
        "StructuralTagItems",
        "TagFormat",
        "TagsWithSeparatorFormat",
        "TokenTriggeredTagsFormat",
        "TriggerSpaces",
        "TriggeredTagsFormat",
        "buildStructuralTag",
    ):
        setattr(structural_tag, attr, object)
    sys.modules["xgrammar.structural_tag"] = structural_tag

    tool_call_schema = types.ModuleType("xgrammar.openai_tool_call_schema")
    tool_call_schema.BuiltinToolParam = object
    tool_call_schema.FunctionToolParam = object
    sys.modules["xgrammar.openai_tool_call_schema"] = tool_call_schema


def ensure_importable_env() -> None:
    """Make the vllm_fl import chain work on hosts lacking optional wheels.

    Idempotent; on the CI image (real flag_gems/uvloop/xgrammar present)
    every step is a no-op.
    """
    _install_placeholder_flag_gems()
    _install_placeholder_uvloop()
    _install_placeholder_xgrammar()


def ensure_vllm_api_compat() -> None:
    """Backfill vllm APIs the connector expects on dev-only vllm builds.

    The connector imports ``get_kv_cache_layout`` from
    ``vllm.v1.attention.backends.utils``; it exists in the CI image's
    vllm (0.24.0). When running against a vllm build that lacks it, the
    value is only used to populate ``worker.kv_cache_layout`` — no test
    asserts on it — so a constant placeholder keeps the module
    importable without changing any behavior under test.
    """
    import vllm.v1.attention.backends.utils as attn_utils

    if not hasattr(attn_utils, "get_kv_cache_layout"):
        attn_utils.get_kv_cache_layout = lambda: "HND"


class FlagCXCallRecorder(list):
    """Call log shared by the fake library/engine/connection objects."""

    def record(self, name, *args, **kwargs):
        self.append((name,) + args)


class FakeP2PEngine:
    """Opaque engine handle handed out by the fake library."""

    def __init__(self, recorder: FlagCXCallRecorder, engine_id: int):
        self.recorder = recorder
        self.engine_id = engine_id

    def __repr__(self):  # pragma: no cover - debug aid only
        return f"FakeP2PEngine({self.engine_id})"


class FakeP2PConn:
    """Opaque connection handle; equality is identity, like the real one."""

    def __init__(self, session: str):
        self.session = session


class FakeFLAGCXLibrary:
    """Recording stand-in for the flagcx wheel's FLAGCXLibrary facade.

    Mirrors the method names the connector calls. Every method records
    its arguments on the shared recorder so tests can assert call
    parameters (R4).
    """

    def __init__(self, recorder: FlagCXCallRecorder):
        self.recorder = recorder
        self.rpc_port = 4242
        self.write_should_raise: Exception | None = None

    def flagcxP2pEngineCreate(self):
        self.recorder.record("flagcxP2pEngineCreate")
        return FakeP2PEngine(self.recorder, engine_id=len(self.recorder))

    def flagcxP2pEngineDestroy(self, engine):
        self.recorder.record("flagcxP2pEngineDestroy", engine.engine_id)

    def flagcxP2pGetConn(self, engine, session):
        self.recorder.record("flagcxP2pGetConn", session)
        return FakeP2PConn(session)

    def flagcxP2pGetRpcPort(self, engine):
        self.recorder.record("flagcxP2pGetRpcPort")
        return self.rpc_port

    def flagcxP2pStartRpcServer(self, engine):
        self.recorder.record("flagcxP2pStartRpcServer")

    def flagcxP2pRegister(self, engine, base_addr, size):
        self.recorder.record("flagcxP2pRegister", base_addr, size)

    def flagcxP2pBatchWriteSync(self, conn, src_vas, dst_vas, sizes):
        self.recorder.record(
            "flagcxP2pBatchWriteSync", conn.session, list(src_vas), list(sizes)
        )
        if self.write_should_raise is not None:
            raise self.write_should_raise


def install_fake_flagcx_wheel(monkeypatch) -> FlagCXCallRecorder:
    """Install the fake flagcx wheel facade into sys.modules.

    Uses monkeypatch.setitem so pytest restores the real state (usually
    "absent") on teardown. Returns the call recorder the library writes
    to. The fake ``flagcx.api`` module name makes the connector take the
    wheel path: ``_flagcx_is_wheel`` is True and FLAGCXLibrary gets
    ``library_path=None``.
    """
    recorder = FlagCXCallRecorder()

    def library_factory(library_path=None):
        recorder.record("FLAGCXLibrary", library_path)
        return FakeFLAGCXLibrary(recorder)

    fake_api = types.ModuleType("flagcx.api")
    fake_api.FLAGCXLibrary = library_factory
    fake_api.__name__ = "flagcx.api"

    fake_flagcx = types.ModuleType("flagcx")
    fake_flagcx.__path__ = []
    fake_flagcx.api = fake_api

    monkeypatch.setitem(sys.modules, "flagcx", fake_flagcx)
    monkeypatch.setitem(sys.modules, "flagcx.api", fake_api)
    return recorder


def drop_connector_module() -> None:
    """Remove the cached connector so the import ladder re-runs.

    The connector binds ``FLAGCXLibrary`` / ``_flagcx_is_wheel`` at module
    import time; dropping it from sys.modules lets a test that installed
    a different facade re-import against it. Contract-sanctioned
    "resetting module caches" for an external-resource boundary.
    """
    sys.modules.pop("vllm_fl.distributed.kv_transfer.flagcx_connector", None)
