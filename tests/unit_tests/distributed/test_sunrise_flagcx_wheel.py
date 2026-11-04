# Copyright (c) 2026 BAAI. All rights reserved.
"""Sunrise contracts for legacy FlagCX checkouts and the 0.14 wheel API."""

import ctypes
import importlib.util
import logging
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm_fl.dispatch.backends.vendor.sunrise import patch

ROOT = Path(__file__).resolve().parents[3]


def install_module(monkeypatch, name, **attrs):
    module = ModuleType(name)
    module.__dict__.update(attrs)
    monkeypatch.setitem(sys.modules, name, module)
    return module


@pytest.mark.parametrize("api", ["legacy", "wheel"])
@pytest.mark.parametrize("explicit_library", [None, "/custom/libflagcx.so"])
def test_communicator_loads_matched_api_and_library(monkeypatch, api, explicit_library):
    class Group:
        rank = 0
        world_size = 2

        def broadcast_obj(self, value, src):
            return value

    class UniqueId(ctypes.Structure):
        _fields_ = [("internal", ctypes.c_byte * 256)]

    class Library:
        paths = []

        def __init__(self, path):
            self.paths.append(path)
            if api == "legacy":
                self.handler = object()
            else:
                self.devHandle = object()

        def flagcxGetUniqueId(self):
            value = UniqueId()
            return ctypes.pointer(value) if api == "legacy" else value

        def flagcxCommInitRank(self, size, unique_id, rank):
            return ctypes.c_void_p(0x1234)

    attrs = dict(
        FLAGCXLibrary=Library,
        flagcxUniqueId=UniqueId,
        flagcxComm_t=ctypes.c_void_p,
        buffer_type=ctypes.c_void_p,
        flagcxDataTypeEnum=object,
        flagcxRedOpTypeEnum=SimpleNamespace(from_torch=lambda x: x),
    )
    install_module(monkeypatch, "vllm.distributed.utils", StatelessProcessGroup=Group)
    install_module(monkeypatch, "vllm.logger", init_logger=logging.getLogger)
    install_module(monkeypatch, "vllm.utils.torch_utils", current_stream=Mock())
    install_module(
        monkeypatch,
        "vllm.platforms",
        current_platform=SimpleNamespace(device_type="ptpu"),
    )
    monkeypatch.setitem(sys.modules, "plugin.interservice.flagcx_wrapper", None)
    if api == "legacy":
        install_module(monkeypatch, "plugin.interservice.flagcx_wrapper", **attrs)
        monkeypatch.setenv("FLAGCX_PATH", "/vendor/FlagCX")
    else:
        install_module(monkeypatch, "flagcx.api", **attrs)
        monkeypatch.delenv("FLAGCX_PATH", raising=False)
    path = ROOT / "vllm_fl/distributed/device_communicators/flagcx.py"
    spec = importlib.util.spec_from_file_location("_sunrise_flagcx_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class Device:
        type = "ptpu"

        def __init__(self, value):
            self.value = value

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

    module.torch = SimpleNamespace(device=Device, zeros=Mock())
    monkeypatch.setattr(
        module.PyFlagcxCommunicator, "_current_stream", Mock(return_value=Mock())
    )
    monkeypatch.setattr(module.PyFlagcxCommunicator, "all_reduce", Mock())
    communicator = module.PyFlagcxCommunicator(
        Group(), "ptpu:0", library_path=explicit_library
    )
    assert module._flagcx_available
    assert module._flagcx_wheel_api is (api == "wheel")
    assert not communicator.disabled
    expected_path = explicit_library
    if expected_path is None and api == "legacy":
        expected_path = os.path.join("/vendor/FlagCX", "build/lib/libflagcx.so")
    assert Library.paths == [expected_path]
    assert communicator._legacy_unique_id_api is (api == "legacy")


def test_wheel_fallback_is_not_enabled_on_other_platforms(monkeypatch):
    install_module(monkeypatch, "vllm.distributed.utils", StatelessProcessGroup=object)
    install_module(monkeypatch, "vllm.logger", init_logger=logging.getLogger)
    install_module(monkeypatch, "vllm.utils.torch_utils", current_stream=Mock())
    install_module(
        monkeypatch,
        "vllm.platforms",
        current_platform=SimpleNamespace(device_type="cuda"),
    )
    monkeypatch.setitem(sys.modules, "plugin.interservice.flagcx_wrapper", None)
    install_module(monkeypatch, "flagcx.api", FLAGCXLibrary=Mock())
    path = ROOT / "vllm_fl/distributed/device_communicators/flagcx.py"
    spec = importlib.util.spec_from_file_location("_sunrise_flagcx_other_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert not module._flagcx_available
    assert not module._flagcx_wheel_api


@pytest.fixture
def stream_library(monkeypatch):
    def prepare(api):
        install_module(
            monkeypatch,
            "vllm.platforms",
            current_platform=SimpleNamespace(device_type="ptpu"),
        )
        calls = []

        class Library:
            def __init__(self):
                # The stream patch must leave each wrapper's native ABI alone.
                if api == "legacy":
                    self.handler = SimpleNamespace(
                        contents=SimpleNamespace(devHandle="legacy-handle")
                    )
                else:
                    self.devHandle = "wheel-handle"

            def adaptor_stream_copy(self, stream):
                handle = (
                    self.handler.contents.devHandle
                    if api == "legacy"
                    else self.devHandle
                )
                raw = stream.ptpu_stream if api == "native_ptpu" else stream.cuda_stream
                calls.append((handle, raw))
                return 0xBEEF

        original = Library.adaptor_stream_copy
        wrapper = SimpleNamespace(FLAGCXLibrary=Library)
        native_import = patch.importlib.import_module

        def import_wrapper(name):
            if name == "plugin.interservice.flagcx_wrapper":
                if api == "legacy":
                    return wrapper
                raise ModuleNotFoundError(name)
            if name == "flagcx.api":
                return wrapper
            return native_import(name)

        monkeypatch.setattr(patch.importlib, "import_module", import_wrapper)
        patch.patch_flagcx_stream_adapter()
        return Library, calls, original

    return prepare


@pytest.mark.parametrize("api", ["legacy", "wheel", "native_ptpu"])
@pytest.mark.parametrize(
    "stream",
    [
        123,
        ctypes.c_void_p(123),
        SimpleNamespace(ptpu_stream=123),
        SimpleNamespace(ptpu_stream=ctypes.c_void_p(123)),
        SimpleNamespace(cuda_stream=123),
        SimpleNamespace(stream=lambda: 123),
    ],
)
def test_stream_pointer_reaches_native_adapter(stream_library, api, stream):
    library_cls, calls, original = stream_library(api)
    library = library_cls()
    assert library.adaptor_stream_copy(stream) == 0xBEEF
    assert calls == [("legacy-handle" if api == "legacy" else "wheel-handle", 123)]
    assert library_cls._sunrise_original_stream_copy is original
    adapted = library_cls.adaptor_stream_copy
    patch.patch_flagcx_stream_adapter()
    assert library_cls.adaptor_stream_copy is adapted


@pytest.mark.parametrize("stream", [object(), ctypes.c_void_p(None)])
def test_unknown_stream_cannot_silently_use_default_pointer(stream_library, stream):
    library_cls, calls, _ = stream_library("wheel")
    with pytest.raises((AttributeError, ValueError)):
        library_cls().adaptor_stream_copy(stream)
    assert not calls
